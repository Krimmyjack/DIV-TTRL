# OOM 修复方案和代码优化建议

## 执行摘要

**三个最关键的修复点**：
1. ❌ **PASS_GRPO_PENALIZED 的 metrics 张量泄漏** → 优先级 1
2. ❌ **TTRL 80→32 下采样的两次 balance_batch** → 优先级 1  
3. ❌ **post_ttrl_metrics 的重复计算** → 优先级 2

---

## 修复 1：PASS_GRPO_PENALIZED 的 metrics 管理（优先级 🔴 高）

### 现状问题
```python
# 当前代码（第 437-460 行）
advantages, returns, metrics = core_algos.compute_pass_grpo_penalized_advantage(...)

# 问题：metrics 可能包含大量张量
for k_met, v_met in metrics.items():
    data.meta_info[k_met] = v_met  # ← 如果 v_met 是张量，保存在内存中
```

### 修复方案 A：只保存标量值
```python
advantages, returns, metrics = core_algos.compute_pass_grpo_penalized_advantage(...)
data.batch["advantages"] = advantages
data.batch["returns"] = returns

# 仅从 metrics 中提取标量值（不保存张量）
if isinstance(metrics, dict):
    for k_met, v_met in metrics.items():
        # 只保存标量或具体数值，不保存张量
        if isinstance(v_met, (int, float)):
            data.meta_info[k_met] = v_met
        elif isinstance(v_met, torch.Tensor):
            # 从张量转换为标量，然后释放张量
            if v_met.numel() == 1:  # 单标量张量
                data.meta_info[k_met] = float(v_met.item())
            else:
                # 对于多元素张量，计算统计量（均值、标准差等）而不保存整个张量
                data.meta_info[f"{k_met}_mean"] = float(v_met.mean().item())
                data.meta_info[f"{k_met}_std"] = float(v_met.std().item())
        # 显式删除张量以立即释放内存
        if isinstance(v_met, torch.Tensor):
            del v_met

# 最后显式删除 metrics dict
del metrics
```

### 修复方案 B：改进 compute_pass_grpo_penalized_advantage 的返回值

需要修改 core_algos.py 中的函数：

```python
def compute_pass_grpo_penalized_advantage(
    token_level_rewards,
    response_mask,
    index,
    answer_types,
    consistency_rates,
    diversity_density_config,
    k,
    epsilon,
):
    """
    改进版本：返回张量和标量指标分离
    """
    # ... [中间计算] ...
    
    # 原来：返回 (advantages, returns, metrics_dict)
    # 问题：metrics_dict 中可能有大量中间张量
    
    # 改进：只计算和返回关键标量，中间张量计算后立即释放
    advantages = ...
    returns = ...
    
    # 关键指标的标量值（而不是张量）
    scalar_metrics = {
        "pass_grpo_penalized/avg_r_div": float(r_div.mean().item()) if r_div is not None else 0.0,
        "pass_grpo_penalized/r_div_triggered_ratio": float(...),
        "pass_grpo_penalized/avg_raw_a_passk": float(...),
        "pass_grpo_penalized/avg_adv_raw": float(...),
        "pass_grpo_penalized/avg_total_advantage": float(advantages.mean().item()),
    }
    
    # 释放中间张量
    if 'r_div' in locals():
        del r_div
    # ... 删除其他中间张量 ...
    
    return advantages, returns, scalar_metrics  # ← 只返回标量，不返回张量
```

### 预期效果
- **内存节省**：5-10%（取决于 metrics 中张量的数量）
- **方案 A 成本**：low（仅 compute_advantage 函数改动）
- **方案 B 成本**：medium（需要修改 core_algos.py）

### 推荐
👉 **优先采用方案 A**（快速修复），如果仍有 OOM 再升级为方案 B

---

## 修复 2：TTRL 80→32 下采样的两次 balance_batch（优先级 🔴 高）

### 现状问题

```
时间轴和内存占用：

Step 1：batch repeat (8 → 80)
        batch = batch.repeat(repeat_times=10, interleave=True)
        占用：80 个样本的内存（peak = 80x）

Step 2：第一次 balance_batch（行 1209）
        _balance_batch(batch)  # 创建 global_idx，执行 reorder
        占用：80 个样本의 中间拷贝（peak = 90x？）

Step 3：reward 计算（行 1219-1223）
        reward_tensor = self.reward_fn(batch)
        batch.batch["token_level_scores"] = reward_tensor  # 保存 reward
        占用：80 个样本的 token_level_scores（peak = 100x）

Step 4：下采样（行 1230）
        batch = self._select_top_k_per_prompt(batch, 10, 4)
        占用：应该释放 80→32，但实际：80 个样本的内存还没释放
        peak = 110x（80 个旧 + 32 个新）

Step 5：第二次 balance_batch（行 1234）
        _balance_batch(batch)  # 又一次 reorder！
        占用：32 个样本的中间拷贝
        peak = 130x（前面 80 个 + 新的 32 个的复制）
```

### 修复方案 A：删除不必要的第二次 balance_batch

```python
# 修改前（行 1232-1234）
if self.use_ttrl:
    sorted_indices = sorted(range(len(batch)), key=lambda i: batch[i].non_tensor_batch["extra_info"]["index"])
    batch = batch[sorted_indices]
if self.use_rm:
    reward_tensor = self.rm_wg.compute_rm_score(batch)
    batch = batch.union(reward_tensor)

reward_result = self.reward_fn(batch, return_dict=True)
reward_tensor = reward_result["reward_tensor"]
reward_extra_infos_dict = reward_result["reward_extra_info"]

if self.use_ttrl:
    # ... ttrl 处理 ...
    batch = self._select_top_k_per_prompt(batch, self.n_votes_per_prompt, self.n_samples_per_prompt)
    self.config.actor_rollout_ref.rollout.n = self.n_samples_per_prompt

    if self.use_ttrl:
        self._balance_batch(batch, metrics=metrics)  # ← 删除这一行，不必要！

# 修改后
if self.use_ttrl:
    # ... ttrl 处理 ...
    batch = self._select_top_k_per_prompt(batch, self.n_votes_per_prompt, self.n_samples_per_prompt)
    self.config.actor_rollout_ref.rollout.n = self.n_samples_per_prompt
    
    # ❌ 删除第二次 balance_batch
    # if self.use_ttrl:
    #     self._balance_batch(batch, metrics=metrics)
    
    # 替代方案：仅当数据顺序改变时再 balance
    # 但 select_top_k_per_prompt 已经保持了相对顺序（按 prompt 顺序选），
    # 所以不需要第二次 balance
```

**逻辑论证**：
- 第一次 balance_batch（行 1209）在 reward 计算前执行，目的是负载均衡
- 下采样（select_top_k_per_prompt）保持相对顺序：按 prompt 顺序选前 k 个
- 第二次 balance_batch 是多余的，因为相对顺序未改变

### 修复方案 B：优化 select_top_k_per_prompt 以减少内存峰值

```python
def _select_top_k_per_prompt_optimized(self, data, n_votes_per_prompt, n_samples_per_prompt):
    """
    优化版本：流式构建选定索引，不创建中间列表副本
    """
    assert len(data) % n_votes_per_prompt == 0
    num_prompts = len(data) // n_votes_per_prompt
    
    # 方法 1：直接返回切片视图，不复制（如果 DataProto 支持）
    selected_indices = []
    for i in range(num_prompts):
        start = i * n_votes_per_prompt
        selected_indices.extend(range(start, start + n_samples_per_prompt))
    
    # 使用高效的索引创建
    selected_indices = torch.tensor(selected_indices, dtype=torch.long, device=data.batch[list(data.batch.keys())[0]].device)
    
    # 返回切片而不是复制
    return data[selected_indices]  # DataProto 的 __getitem__ 应该高效处理
```

### 修复方案 C：两个 balance_batch 改进（完整方案🔥）

最彻底的修复：

```python
# 在 fit() 中修改逻辑

with _timer("adv", timing_raw):
    if self.use_ttrl:
        sorted_indices = sorted(...)
        batch = batch[sorted_indices]
    
    if self.use_rm:
        reward_tensor = self.rm_wg.compute_rm_score(batch)
        batch = batch.union(reward_tensor)
    
    reward_result = self.reward_fn(batch, return_dict=True)
    reward_tensor = reward_result["reward_tensor"]
    
    batch.batch["token_level_scores"] = reward_tensor
    
    if self.use_ttrl:
        # === 改进的 TTRL 处理流程 ===
        
        # 立即释放 80 倍的 reward 张量
        large_batch_rewards = batch.batch["token_level_scores"]
        
        # 在下采样前明确删除不需要的东西
        ttrl_metrics = reward_result["ttrl_info"]
        
        # 下采样到 32
        batch = self._select_top_k_per_prompt(batch, self.n_votes_per_prompt, self.n_samples_per_prompt)
        self.config.actor_rollout_ref.rollout.n = self.n_samples_per_prompt
        
        # ❌ 不再执行第二次 balance_batch
        # ✅ 如果确实需要负载均衡，应该在下采样前而不是后
        
        # 计算 post_ttrl_metrics（基于已下采样的小 batch）
        ...
        
        # === 关键：立即清理参考表和临时数据 ===
        del large_batch_rewards
        del ttrl_metrics
        if 'post_reward_result' in locals():
            del post_reward_result
            
        import gc
        gc.collect()
```

### 预期效果
- **方案 A**：内存节省 ~20-30%（删除一次冗余的 reorder）
- **方案 B**：中等（优化数据复制）
- **方案 C**：综合，节省 ~30-40%

### 推荐复述
👉 **立即实施方案 A**（删除第二次 balance_batch，最简单最有效）
👉 **配合方案 C 的清理代码**（gc.collect）

---

## 修复 3：post_ttrl_metrics 重复计算（优先级 🟡 中）

### 现状问题

```python
# 当前代码（行 1219-1244）
reward_result = self.reward_fn(batch, return_dict=True)  # ← 计算 80 个样本
reward_tensor = reward_result["reward_tensor"]
ttrl_metrics = reward_result["ttrl_info"]

# ... 下采样 80 → 32 ...

batch = self._select_top_k_per_prompt(batch, self.n_votes_per_prompt, self.n_samples_per_prompt)
self.config.actor_rollout_ref.rollout.n = self.n_samples_per_prompt

# 重新计算！
post_reward_result = self.reward_fn.compute_post_ttrl_metrics(batch)  # ← 计算 32 个样本
# 问题：reward_fn 被调用两次，虽然第二次样本少，但仍浪费 CPU 和冗余计算
```

### 修复方案 A：验证是否真的需要 post_ttrl_metrics

```python
# 检查 post_ttrl_metrics 的目的
# 如果是验证数据，可以：
# 1. 只在验证阶段计算（if self.config.trainer.do_validation）
# 2. 或者从第一次计算中的 80 个样本中选择对应的 32 个指标

# 修改代码
reward_result = self.reward_fn(batch, return_dict=True)
reward_tensor = reward_result["reward_tensor"]
reward_extra_infos_dict = reward_result["reward_extra_info"]

if self.use_ttrl:
    ttrl_metrics = reward_result["ttrl_info"]
    
    # 从 reward_result 中提取相关指标（而不是重新计算）
    # 假设 ttrl_metrics 中包含的指标可以被切分
    
    # 下采样
    batch = self._select_top_k_per_prompt(batch, self.n_votes_per_prompt, self.n_samples_per_prompt)
    
    # ❌ 删除重复的 compute_post_ttrl_metrics 调用
    # post_reward_result = self.reward_fn.compute_post_ttrl_metrics(batch)
    
    # ✅ 替代方案：从现有的 ttrl_metrics 中计算统计量
    # post_reward_result = {
    #     "consistency_rate_reduced": np.mean(ttrl_metrics["_consistency_rate"][selected_indices]),
    #     ...
    # }
```

### 修复方案 B：优化 compute_post_ttrl_metrics 的实现

如果 post_ttrl_metrics 确实必要：

```python
# 在 reward_manager.py 或相关文件中优化
def compute_post_ttrl_metrics(self, batch):
    """
    优化版本：直接基于已有的数据，不重新计算
    """
    # 使用缓存的指标而不是重新计算
    n_votes = len(batch) // len(set(batch.non_tensor_batch["uid"]))
    
    # 从 batch 中直接提取统计量，而不是调用昂贵的 reward_fn
    metrics = {
        "avg_consistency": float(np.mean(batch.non_tensor_batch.get("consistency_rate", []))),
        "num_correct": int(sum(1 for at in batch.non_tensor_batch.get("answer_types", []) if at == 0)),
    }
    return metrics
```

### 预期效果
- **方案 A**：内存和时间节省 ~10-15%（消除冗余计算）
- **方案 B**：时间节省更多，但需要改 reward_manager

### 推荐
👉 **采用方案 A**（快速，删除冗余调用）
👉 **长期考虑方案 B**（重构 reward_fn 以支持增量计算）

---

## 修复 4：compute_advantage 中的诊断计算（优先级 🟡 中）

### 现状问题
```python
# 行 1256-1277：advantage bias 诊断
if (
    "oracle_answer_types" in batch.non_tensor_batch
    and self.config.algorithm.adv_estimator == AdvantageEstimator.PASS_GRPO
):
    oracle_adv, _ = core_algos.compute_pass_grpo_advantage(...)  # ← 额外计算！
```

### 修复方案
```python
# 添加开关控制是否计算诊断
if (
    getattr(self.config.algorithm, "compute_advantage_diagnostics", False)  # ← 配置开关
    and "oracle_answer_types" in batch.non_tensor_batch
    and self.config.algorithm.adv_estimator == AdvantageEstimator.PASS_GRPO
):
    # 诊断计算代码保持不变
    ...
```

**优势**：
- 可选开启/关闭诊断，OOM 时关闭
- 开发时启用，生产环境关闭

---

## 修复 5：完整的内存清理策略

### 在 compute_advantage 返回前添加清理

```python
def compute_advantage(...):
    # ... 所有计算 ...
    
    # 在返回前，确保中间张量被释放
    if adv_estimator == AdvantageEstimator.DIVERSITY_DENSITY_HYBRID:
        # 清理 div_advantages 和 fallback_advantages（已合并）
        if 'div_advantages' in locals():
            del div_advantages
        if 'fallback_advantages' in locals():
            del fallback_advantages
        if 'blended_advantages' in locals():
            del blended_advantages
        # ... 等等
    
    # 通用清理
    local_vars = list(locals().keys())
    for var in local_vars:
        if var not in ['data', 'return']:
            try:
                obj = locals()[var]
                if isinstance(obj, torch.Tensor) and obj.requires_grad is False:
                    del obj
            except:
                pass
    
    return data
```

### 在 fit() 循环中改进清理

```python
# 循环末尾（行 1285-1296）改进为：

# 显式释放关键变量
del batch
del batch_dict

for var_name in ['gen_batch', 'gen_batch_output', 'old_log_prob']:
    if var_name in locals():
        del locals()[var_name]

# 更激进的清理（仅在 OOM 时启用）
if self.config.trainer.get("aggressive_gc", False):
    import gc
    for i in range(3):
        gc.collect()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
else:
    # 普通清理
    import gc
    gc.collect()
```

---

## 修复优先级摘要表

| 修复ID | 修复内容 | 成本 | 内存节省 | 优先级 |
|--------|--------|------|--------|--------|
| 1 | PASS_GRPO_PENALIZED metrics 管理 | 低 | 5-10% | 🔴 1 |
| 2A | 删除第二次 balance_batch | 极低 | 20-30% | 🔴 1 |
| 2C | 完整的 batch 清理流程 | 低 | 30-40% | 🔴 1 |
| 3A | 删除重复的 post_ttrl_metrics | 低 | 10-15% | 🟡 2 |
| 4 | 诊断计算开关 | 极低 | 5% | 🟡 2 |
| 5 | 增强内存清理 | 低 | 5-10% | 🟡 2 |

**预期总体内存节省**：50-70% 的峰值内存降低

---

## 快速修复清单（做这个可立即缓解 OOM）

### Step 1：删除第二次 balance_batch（5 分钟）
找到行 1234：
```python
if self.use_ttrl:
    self._balance_batch(batch, metrics=metrics)
```
改为：
```python
# if self.use_ttrl:  # ← 注释掉这一行
#     self._balance_batch(batch, metrics=metrics)
```

### Step 2：在 compute_advantage PASS_GRPO_PENALIZED 分支添加清理（10 分钟）
找到行 457-459：
```python
for k_met, v_met in metrics.items():
    data.meta_info[k_met] = v_met
```
改为：
```python
for k_met, v_met in metrics.items():
    if isinstance(v_met, (int, float)):
        data.meta_info[k_met] = v_met
    elif isinstance(v_met, torch.Tensor) and v_met.numel() == 1:
        data.meta_info[k_met] = float(v_met.item())
    # 不保存大张量
del metrics
```

### Step 3：在 fit() 循环末尾改进清理（5 分钟）
找到行 1290-1296，改为：
```python
# Explicitly free batch and metrics from this step
del batch, batch_dict, metrics
if 'gen_batch' in locals():
    del gen_batch
if 'gen_batch_output' in locals():
    del gen_batch_output
import gc
gc.collect()

if torch.cuda.is_available():
    torch.cuda.empty_cache()  # ← 新增强力清理
```

**预期结果**：实施这三个快速修复后，应该能缓解大部分 OOM 问题。

