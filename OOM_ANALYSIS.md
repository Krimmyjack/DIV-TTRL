# CUDA Out of Memory 问题详细分析

## 问题现象
- 错误: `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.29 GiB`
- 触发条件: **当response回答长度有提高趋势时，OOM发生**
- GPU状态: 
  - 总容量: 31.47 GiB
  - 可用: 2.02 GiB
  - PyTorch已分配: 33.04 GiB（**超出GPU容量！**)
  - PyTorch保留未分配: 5.75 GiB

## 根本原因分析

### 1. **PRIMARY CAUSE: 批量auto_verify全样本导致内存爆炸** ⚠️ 最严重
**文件**: `verl/workers/reward_manager/diversity_reward.py:373`

```python
# 一次性对所有 total_samples (~1024-2048) 个样本调用auto_verify
all_true_rewards_list, _ = auto_verify(
    task, all_response_strs,
    [all_ground_truths[i] for i in range(total_samples)],
    extra_info=all_extra_infos
)
```

**问题**:
- 在train模式下，所有2048个response都被解码成字符串（all_response_strs）
- auto_verify对所有字符串进行复杂的验证逻辑（答案提取、对比等）
- 如果response长度从4096增加到8192，内存占用近似翻倍

**内存计算**:
```
response_strs占用: ~1024 samples × 4096 tokens × 平均2-4 bytes/token ≈ 8-16 GB（取决于tokenizer）
auto_verify中间结果: 3-5 GB
其他缓存: 2-3 GB
总计: 13-24 GB（接近GPU容量）
```

### 2. **SECONDARY CAUSE: Advantage张量GPU内存积累**
**文件**: `verl/trainer/ppo/core_algos.py:602-603`

```python
# response_mask形状: (batch_size, response_length) = (32, 4096) = 131K floats
advantages_raw_tensor = torch.tensor(advantages_raw_np, dtype=dtype, device=device)
advantages = advantages_raw_tensor.unsqueeze(-1) * response_mask
```

**问题**:
- 虽然advantages计算在CPU上执行（numpy），但最后要转**GPU张量**
- advantages和returns各占: 32 × 4096 × 4 bytes = 512 KB
- 中间过程中存在多个副本（unsqueeze, 乘法操作的临时张量）
- 响应长度增加时，这个成本按比例增长

### 3. **TERTIARY CAUSE: 长序列时训练图的积累**
**文件**: `verl/workers/actor/dp_actor.py:393` (backward调用)

**问题**:
- 当response_length=4096时，计算图节点数为batch_size × response_length = 32 × 4096 = 131K
- backward时保存中间activation用于梯度计算
- 虽然开启了gradient_checkpointing，但长序列仍会导致较大的激活内存

### 4. **QUATERNARY CAUSE: response_mask重复建立（低优先级）**
**文件**: `verl/trainer/ppo/ray_trainer.py:226-227`

```python
if "response_mask" not in data.batch.keys():
    data.batch["response_mask"] = compute_response_mask(data)
```

**问题**: 虽然是条件式，但如果response稍长，这个additional mask也会占用空间

---

## 详细修复方案

### 🔴 FIX #1: 优化auto_verify调用 (最关键，预期释放 5-10 GB)

**当前代码** (diversity_reward.py:373):
```python
# 一次性all_true_rewards_list验证所有2048个样本
all_true_rewards_list, _ = auto_verify(
    task, all_response_strs,
    [all_ground_truths[i] for i in range(total_samples)],
    extra_info=all_extra_infos
)
```

**修复方案**: 采用**分组验证** + **及时释放**

```python
# 分批验证: 每次只验证n_votes_per_prompt个样本
BATCH_SIZE_VERIFY = self.n_votes_per_prompt  # 64
all_true_rewards_list = []

for batch_start in range(0, total_samples, BATCH_SIZE_VERIFY):
    batch_end = min(batch_start + BATCH_SIZE_VERIFY, total_samples)
    
    batch_responses = all_response_strs[batch_start:batch_end]
    batch_labels = [all_ground_truths[i] for i in range(batch_start, batch_end)]
    batch_extra = [all_extra_infos[i] for i in range(batch_start, batch_end)]
    
    batch_rewards, _ = auto_verify(task, batch_responses, batch_labels, extra_info=batch_extra)
    all_true_rewards_list.extend(batch_rewards)
    
    # 显式释放GPU缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# 使用完all_response_strs后立即释放
del all_response_strs
del all_prompt_strs
del all_valid_resp_lengths
if torch.cuda.is_available():
    torch.cuda.empty_cache()
```

**预期效果**: 
- 从一次性加载2048个response降至64个
- 内存占用从16GB+ 降至 ~2-3GB
- 内存峰值大幅下降

---

### 🟡 FIX #2: GPU张量及时释放 (预期释放 1-2 GB)

**当前代码** (core_algos.py:602-609):
```python
advantages_raw_tensor = torch.tensor(advantages_raw_np, dtype=dtype, device=device)
advantages = advantages_raw_tensor.unsqueeze(-1) * response_mask
returns = advantages

# del可能不会立即释放GPU内存
del actual_lengths_cpu
del advantages_raw_np
```

**修复方案**: 显式GPU缓存清理

```python
# 将numpy转GPU
advantages_raw_tensor = torch.tensor(advantages_raw_np, dtype=dtype, device=device)
advantages = advantages_raw_tensor.unsqueeze(-1) * response_mask
returns = advantages.clone()  # 确保是独立的张量

# 立即释放中间tensor和numpy
del actual_lengths_cpu
del advantages_raw_np
del advantages_raw_tensor  # 显式删除中间tensor

# 强制GPU缓存清理
if torch.cuda.is_available():
    torch.cuda.empty_cache()

return advantages, returns, metrics
```

---

### 🟠 FIX #3: 分批处理advantage计算 (预期释放 1-2 GB，需要更大改动)

如果上述两个fix仍然不足，考虑：

**在compute_advantage中分批处理**:
```python
elif adv_estimator == AdvantageEstimator.PASS_GRPO_PENALIZED:
    # 分批处理大batch
    max_sub_batch_size = 8  # 降低单次advantage计算的batch size
    bs = data.batch["token_level_rewards"].shape[0]
    
    all_advantages = []
    all_returns = []
    
    for sub_start in range(0, bs, max_sub_batch_size):
        sub_end = min(sub_start + max_sub_batch_size, bs)
        sub_bs = sub_end - sub_start
        
        # 提取子batch
        sub_data = create_sub_batch(data, range(sub_start, sub_end))
        
        # 计算advantage
        sub_adv, sub_ret, _ = core_algos.compute_pass_grpo_penalized_advantage(...)
        
        all_advantages.append(sub_adv)
        all_returns.append(sub_ret)
        
        # 及时释放
        del sub_data
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # 拼接结果
    data.batch["advantages"] = torch.cat(all_advantages, dim=0)
    data.batch["returns"] = torch.cat(all_returns, dim=0)
```

---

### 🟢 FIX #4: 降低batch_size (临时/应急方案)

如果无法立即应用上述修复，临时解决方案：

**配置更改**:
```yaml
data.train_batch_size: 16    # 从 32 → 16 (降低50%)
actor_rollout_ref.actor.ppo_mini_batch_size: 1
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu: 1
```

**权衡**: 训练速度降低50%，但可完成训练

---

## 修复优先级

| 优先级 | 修复项 | 预期收益 | 实现难度 | 建议 |
|-------|------|--------|--------|------|
| 🔴 **最高** | FIX #1: 分批auto_verify | 5-10 GB | ⭐ 低 | **立即实施** |
| 🟡 **高** | FIX #2: GPU缓存清理 | 1-2 GB | ⭐ 低 | **立即实施** |
| 🟠 **中** | FIX #3: 分批advantage计算 | 1-2 GB | ⭐⭐ 中 | 如果#1#2不足 |
| 🟢 **低** | FIX #4: 降低batch size | 50% 内存 | ⭐ 低 | 应急方案 |

---

## 测试方案

### 验证修复效果

1. **修复FIX #1后测试**:
   ```bash
   # 监控GPU内存使用
   watch -n 1 nvidia-smi
   
   # 训练首个epoch（会触发reward计算）
   python train.py ... 2>&1 | tee train.log
   ```

2. **预期结果**:
   - GPU内存峰值从 29-30 GB 降至 15-18 GB
   - 第一个batch的reward计算不再OOM

3. **逐步增加response长度验证**:
   - max_response_length: 4096 → 5120 → 6144
   - 观察OOM是否仍然发生

4. **内存监控指标**:
   ```python
   # 在_compute_ttrl_reward开始和结束时打印
   import torch
   if torch.cuda.is_available():
       print(f"GPU Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB / {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
   ```

---

## 配置建议

为了在应用fix的同时确保稳定性，建议的配置：

```yaml
# data config
data.train_batch_size: 32        # 保持原样（如果apply FIX#1#2）
data.max_response_length: 4096   # 或更高，取决于fix效果

# actor config
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu: 2  # 保持
actor_rollout_ref.actor.ppo_mini_batch_size: 1           # 保持

# reward config (可选调整)
reward_model.reward_kwargs.n_samples_per_prompt: 32       # 保持
reward_model.reward_kwargs.n_votes_per_prompt: 64         # 保持

# trainer config
trainer.save_freq: 15
trainer.test_freq: 5
```

---

## 检查清单

- [ ] 理解OOM的根本原因：批量auto_verify加上长序列
- [ ] 应用FIX #1（最关键）：分批auto_verify
- [ ] 应用FIX #2：GPU缓存显式清理
- [ ] 测试训练的第一个epoch
- [ ] 逐步增加response_length，验证稳定性
- [ ] 监控GPU内存使用（峰值应降低40-50%）
- [ ] 验证training loss曲线与之前相同（确保修复没有改变训练逻辑）
