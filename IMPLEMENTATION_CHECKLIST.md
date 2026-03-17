# OOM修复 - 实施检查清单

## ✅ 已完成

- [x] **分析根本原因**：确认是批量auto_verify导致内存爆炸（5-10GB）+ 长序列张量积累
  - 原因1: 一次性加载2048个decoded strings进行auto_verify
  - 原因2: response长度不足时，tensors占用33GB超过31.47GB GPU容量
  - 原因3: GPU缓存碎片化，未分配内存5.75GB无法释放

- [x] **应用FIX #1**：分批auto_verify  
  文件: `verl/workers/reward_manager/diversity_reward.py` (行373-405)
  ```
  改动: 
  - from all_true_rewards_list = auto_verify(...all 2048 samples...)
  - to   for chunk in chunks of 64: auto_verify(chunk)
  - add  torch.cuda.empty_cache() 在每个chunk之后
  ```

- [x] **应用FIX #2**：GPU缓存显式清理  
  文件: `verl/trainer/ppo/core_algos.py` (行602-613)
  ```
  改动:
  - 添加 returns.clone() 确保独立张量
  - 添加 del advantages_raw_tensor
  - 添加 torch.cuda.empty_cache()
  ```

- [x] **创建详细分析文档**  
  - OOM_ANALYSIS.md - 完整根因分析和修复指南
  - OOM_FIX_SUMMARY.md - 修复方案总结和验证指南

---

## 🔄 需要执行的测试

### Test 1: 快速验证 (10分钟)

```bash
# 运行部分epoch验证修复
python verl/trainer/main_ppo.py \
  reward_model.reward_manager=diversity_ttrl \
  data.train_files=[data/MATH-TTT/train-simplerl.parquet] \
  data.val_files=[data/MATH-TTT/test-simplerl.parquet] \
  data.train_batch_size=32 \
  data.max_response_length=4096 \
  trainer.total_epochs=1 \
  trainer.n_gpus_per_node=1 \
  2>&1 | tee validation_test_1.log

# 成功标志: 完成epoch 1 without OOM
grep -E "OOM|TrainingComplete|Exception" validation_test_1.log
```

### Test 2: 中等压力 (30分钟)

```bash
# 更接近原配置
python verl/trainer/main_ppo.py \
  reward_model.reward_manager=diversity_ttrl \
  data.train_files=[data/MATH-TTT/train-simplerl.parquet] \
  data.val_files=[data/MATH-TTT/test-simplerl.parquet] \
  data.train_batch_size=32 \
  data.max_response_length=4096 \
  trainer.total_epochs=2 \
  trainer.n_gpus_per_node=8 \
  trainer.nnodes=1 \
  2>&1 | tee validation_test_2.log

# 监控GPU内存使用
watch -n 2 nvidia-smi
# 预期: 单GPU占用 <18 GB
```

### Test 3: 全配置测试 (需要多小时)

```bash
# 运行完整配置
python verl/trainer/main_ppo.py \
  reward_model.reward_manager=diversity_ttrl \
  reward_model.reward_kwargs.n_samples_per_prompt=32 \
  reward_model.reward_kwargs.n_votes_per_prompt=64 \
  reward_model.reward_kwargs.mode=train \
  data.train_files=[data/MATH-TTT/train-simplerl.parquet] \
  data.val_files=[data/MATH-TTT/test-simplerl.parquet] \
  data.max_prompt_length=1024 \
  data.max_response_length=4096 \
  data.train_batch_size=32 \
  data.truncation=error \
  trainer.total_epochs=4 \
  trainer.n_gpus_per_node=8 \
  trainer.nnodes=1 \
  2>&1 | tee validation_test_full.log

# 成功标志: 完成所有4个epoch，无OOM
tail -100 validation_test_full.log | grep -E "epoch|Training|OOM"
```

---

## 📊 验证指标

### 内存指标

计算前：
```
torch.OutOfMemoryError
GPU allocated: 33.04 GiB (超容)
GPU reserved: 38.79 GiB
```

计算后期望：
- GPU allocated: **12-18 GiB** (↓ 40-60%)
- GPU reserved: 18-25 GiB
- GPU free: >10 GiB

### 日志检查

```bash
# 宏观检查
grep -i "ERROR\|FATAL\|OOM" validation_test_*.log
# 期望: 无匹配

# 微观检查
grep -i "starting TTRL\|processing chunk\|GPU Memory" validation_test_*.log
# 期望: 看到分批处理日志（如果有debug打印）

# 训练进度
grep "epoch\|step" validation_test_*.log | head -20
# 期望: 看到epoch 1, 2, 3, 4完成日志
```

### 性能指标

```bash
# 提取训练速度
grep "tokens/sec\|samples/sec" validation_test_*.log

# 预期:
# 修复前: N/A (OOM halt)
# 修复后: 25-35 samples/s (取决于GPU配置)

# 提取Loss曲线
python << 'EOF'
import re
with open('validation_test_*.log', 'r') as f:
    losses = re.findall(r'loss[:\s=]+([0-9.]+)', f.read())
    if losses:
        print(f"Loss start: {losses[0]}")
        print(f"Loss end: {losses[-1]}")
        print(f"Loss is decreasing: {float(losses[0]) > float(losses[-1])}")
EOF

# 期望: Loss单调递减（表示训练正常）
```

---

## ⚠️ 如果测试失败

### 仍然OOM: 应用FIX #3

如果修复后仍然OOM，需要应用 FIX #3（分批advantage计算）。

实现步骤:
```python
# 在 verl/trainer/ppo/ray_trainer.py compute_advantage() 中
# 针对 PASS_GRPO_PENALIZED 分批处理

max_sub_batch = 8
for sub_start in range(0, bs, max_sub_batch):
    # 计算子batch advantages
    # del并清缓存
    torch.cuda.empty_cache()
```

详见 OOM_ANALYSIS.md FIX #3 章节

### Loss异常: 验证advantage值

```python
# 在 ray_trainer.compute_advantage() 末尾添加检查

import math
advantages = data.batch["advantages"]
returns = data.batch["returns"]

assert torch.isfinite(advantages).all(), "Advantages contain NaN/Inf"
assert torch.isfinite(returns).all(), "Returns contain NaN/Inf"

avg_adv = advantages.mean().item()
std_adv = advantages.std().item()
max_adv = advantages.abs().max().item()

print(f"Advantage stats: mean={avg_adv:.4f}, std={std_adv:.4f}, max={max_adv:.4f}")

# 期望:
# mean 在 0 附近（-1 到 1）
# std ~0.5-1.0
# max <10（异常高为 danger sign）
```

### GPU缓存警告: 可忽略

```
UserWarning: CUDA out of memory. Forcing to empty_cache() caused empty_cache() to not work properly.
```

这表示GPU已满但我们仍在尝试清理，此warning不影响功能。

---

## 📝 变更记录

### 修改的文件

1. **verl/workers/reward_manager/diversity_reward.py**
   - Line 373-405: 替换批量auto_verify为分批版本
   - 新增: chunk处理循环 + torch.cuda.empty_cache()
   - 新增: 显式del large lists

2. **verl/trainer/ppo/core_algos.py**
   - Line 602-613: 增强内存清理
   - 改动: returns.clone() 替换 returns = advantages
   - 新增: del advantages_raw_tensor
   - 新增: torch.cuda.empty_cache()

### 未修改的文件

- ray_trainer.py: 暂不需要（FIX #1#2充分）
- dp_actor.py: backward操作保持不变
- fsdp_workers.py: 无需改动

---

## 🎯 关键要点

1. **根本原因是批量auto_verify**
   - 决定因素: n_votes_per_prompt=64, n_samples_per_prompt=32
   - 2048个样本 × 4096 token length = 巨大内存占用

2. **修复优先级**
   - FIX #1 (分批verify): 释放5-10GB，必须做
   - FIX #2 (GPU缓存清理): 释放1-2GB，必须做
   - FIX #3 (分批advantage): 可选，如果#1#2不足

3. **验证策略**
   - Test 1验证基础修复
   - Test 2验证多卡场景
   - Test 3验证完整配置
   - 逐步增加负载找到稳定点

4. **性能预期**
   - 训练速度: -5-10%（可接受）
   - Loss曲线: 不变（修复是内存优化）
   - 收敛速度: 不变

---

## 后续优化方向（可选）

如果OOM问题完全解决且有时间，可考虑：

1. **流式处理**: 
   - 而不是一次性解码所有responses，逐个处理

2. **增量计算**:
   - reward只计算训练中使用的样本，不是所有

3. **异步验证**:
   - auto_verify在后台CPU线程中执行，同时准备下一batch

4. **模型并行**:
   - 使用tensor并行降低单卡内存占用

---

## 联系与报告

如遇问题：

1. 检查OOM_ANALYSIS.md了解详细原理
2. 查看日志确认修复是否生效
3. 逐步应用FIX #3和更激进的措施
4. 保存日志用于后续分析

