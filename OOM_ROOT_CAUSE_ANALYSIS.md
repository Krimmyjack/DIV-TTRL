# OOM 问题根本原因分析

## 一、新增的 AdvantageEstimator 类型导致的 OOM

### 1.1 PASS_GRPO_PENALIZED（新增，第 89 行）
**新版本新增**：
```python
PASS_GRPO_PENALIZED = "pass_grpo_penalized"
```

**在 compute_advantage 中的处理**（第 437-460 行）：
```python
elif adv_estimator == AdvantageEstimator.PASS_GRPO_PENALIZED:
    # 调用新的 core_algos.compute_pass_grpo_penalized_advantage
    advantages, returns, metrics = core_algos.compute_pass_grpo_penalized_advantage(
        token_level_rewards=data.batch["token_level_rewards"],
        response_mask=data.batch["response_mask"],
        index=data.non_tensor_batch["uid"],
        answer_types=data.non_tensor_batch["answer_types"],
        consistency_rates=consistency_rates,
        diversity_density_config=diversity_density_config,
        k=k,
        epsilon=epsilon
    )
    data.batch["advantages"] = advantages
    data.batch["returns"] = returns
    
    # 将所有计算结果存储到 meta_info
    for k_met, v_met in metrics.items():
        data.meta_info[k_met] = v_met
    
    data.meta_info["pass_grpo_penalized/avg_total_advantage"] = advantages.mean().item()
```

**OOM 问题**：
- `compute_pass_grpo_penalized_advantage` 计算返回三个值（advantages, returns, metrics）
- metrics 字典包含多个中间计算结果，可能保存了大量的中间张量
- 这些 metrics 被逐个存储到 `data.meta_info` 中，可能未及时释放

---

## 二、fit() 方法中的内存累积问题

### 2.1 diversity_density_config 参数膨胀（第 1130-1150 行）

**新版本对比 Pre 版本的新增参数**：

```python
# 新版本增加的参数（共 13 个参数）
diversity_density_config = {
    "k": getattr(self.config.algorithm, "diversity_density_k", 4),  # 改为默认 4，之前是 8
    "fallback_estimator": getattr(..., "grpo"),
    "use_metric": getattr(..., "consistency_rate"),
    "consistency_threshold": getattr(..., 0.0),
    "selective_passk_threshold": getattr(..., 0.5),
    # === 以下是新增参数 ===
    "lam_div": getattr(self.config.algorithm, "lam_div", 0.05),
    "c_max": getattr(self.config.algorithm, "c_max", 2.0),
    "tau_rep": getattr(self.config.algorithm, "tau_rep", 0.2),
    "gamma": getattr(self.config.algorithm, "gamma", 1.0),
    "p_max": getattr(self.config.algorithm, "p_max", 0.15),
    "n_gram_size": getattr(self.config.algorithm, "n_gram_size", 3),
    "use_rep_penalty": getattr(self.config.algorithm, "use_rep_penalty", False),
    "div_sc_threshold": getattr(self.config.algorithm, "div_sc_threshold", 0.5),
}
```

**OOM 原因**：
- 这些配置参数被传入 `compute_advantage` 和进一步的 `core_algos` 函数
- 新增的 8 个参数（lam_div, c_max, tau_rep, gamma, p_max, n_gram_size, use_rep_penalty, div_sc_threshold）
- 这些参数可能导致 `compute_pass_grpo_penalized_advantage` 进行额外的中间计算，未及时释放张量

### 2.2 新增的 PASS_GRPO_PENALIZED 处理（第 1283-1289 行）

```python
# 新增：pass_grpo_penalized 指标记录
for pp_key in [
    "pass_grpo_penalized/avg_r_div",
    "pass_grpo_penalized/r_div_triggered_ratio",
    "pass_grpo_penalized/avg_raw_a_passk",
    "pass_grpo_penalized/avg_adv_raw",
    "pass_grpo_penalized/avg_total_advantage",
]:
    if pp_key in batch.meta_info:
        metrics[f"train/{pp_key.replace('/', '_')}"] = float(batch.meta_info[pp_key])
```

**OOM 原因**：
- 这些指标记录操作本身占内存不大
- 但反映了计算过程中产生了大量中间结果

---

## 三、数据重复与排序导致的内存膨胀

### 3.1 batch repeat（第 1186 行）

**代码**：
```python
batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
batch = batch.union(gen_batch_output)
```

**OOM 问题**：
- 当 `self.n_votes_per_prompt` 很大时（例如 10 或更多），batch 会被重复 10 倍
- 对于 8 个 sample 的 batch，重复后变成 80
- 而后又执行了 `_select_top_k_per_prompt` 过滤（第 1227-1230 行）：
  ```python
  # Down Sampling
  batch = self._select_top_k_per_prompt(batch, self.n_votes_per_prompt, self.n_samples_per_prompt)
  self.config.actor_rollout_ref.rollout.n = self.n_samples_per_prompt
  ```

**关键问题**：
1. 先扩大 batch （×10），然后再缩小（÷10）
2. 这两步操作之间，中间的 batch 占用大量内存版本 2：
   - 生成了 n_votes_per_prompt（通常 10）倍的数据
   - 计算了所有这些数据的 reward
   - 然后只保留其中的 n_samples_per_prompt（通常 4）
   - 白白浪费了 6/10 的内存

### 3.2 _balance_batch 被调用两次（第 1210 和 1234 行）

```python
# 第一次：生成 reward 前
if self.config.trainer.balance_batch:
    self._balance_batch(batch, metrics=metrics)

# 第二次：下采样后
if self.use_ttrl:
    self._balance_batch(batch, metrics=metrics)  # 第 1234 行
```

**OOM 问题**：
- 每次 balance_batch 都重新计算 sequence length 并重新排序
- 这会导致 batch 的多次拷贝和排列

---

## 四、advantage 计算中的中间张量逻辑

### 4.1 DIVERSITY_DENSITY_HYBRID 中的双计算（第 305-390 行）

```python
elif adv_estimator == AdvantageEstimator.DIVERSITY_DENSITY_HYBRID:
    # 1. Compute diversity density advantage
    div_advantages, div_returns = core_algos.compute_diversity_density_advantage_from_prompts(...)
    
    # 2. Fallback advantage - 三选一，导致多次计算
    if fallback == "grpo":
        fallback_advantages, fallback_returns = core_algos.compute_grpo_outcome_advantage(...)
    elif fallback == "pass_grpo":
        fallback_advantages, fallback_returns = core_algos.compute_pass_grpo_advantage(...)
    elif fallback == "rloo":
        fallback_advantages, fallback_returns = core_algos.compute_rloo_outcome_advantage(...)
    else:
        fallback_advantages, fallback_returns = core_algos.compute_grpo_outcome_advantage(...)
    
    # 3. 随机选择混合
    use_diversity = (random_vals < (1 - p)).float().unsqueeze(-1)
    blended_advantages = use_diversity * div_advantages + (1 - use_diversity) * fallback_advantages
    blended_returns = use_diversity * div_returns + (1 - use_diversity) * fallback_returns
    
    advantages = train_mask * blended_advantages
    returns = train_mask * blended_returns
```

**OOM 问题**：
- 同时计算两种 advantage（diversity 和 fallback）并保存
- 混合计算产生新的 blended_advantages 和 blended_returns
- 这些中间张量同时占用内存，直到最后赋值

### 4.2 ADAPTIVE_PASSK 中的分组计算（第 395-425 行）

```python
elif adv_estimator == AdvantageEstimator.ADAPTIVE_PASSK:
    # 创建三个独立的 advantages 和 returns 张量
    advantages = torch.zeros(bs, response_length, dtype=dtype, device=device)
    returns = torch.zeros(bs, response_length, dtype=dtype, device=device)
    
    for k_val, indices in k_to_indices.items():
        # 对每个 k 值，创建子批次并计算
        sub_rewards = data.batch["token_level_rewards"][indices_tensor]
        sub_mask = data.batch["response_mask"][indices_tensor]
        # ...计算 sub_adv, sub_ret（新张量）
        # 然后 scatter 回到主张量
        advantages[indices_tensor] = sub_adv
        returns[indices_tensor] = sub_ret
```

**OOM 问题**：
- 每次循环迭代都创建新的 sub_adv 和 sub_ret 张量
- 这些张量都在 GPU 上，尽管最后被 scatter，但中间仍占用大量内存
- 如果有 3-6 个不同的 k 值分组，每个都占用独立的内存空间

---

## 五、ttrl_metrics 处理中的内存问题

### 5.1 答案类型数据的保存（第 1247-1258 行）

```python
if "_answer_types" in ttrl_metrics:
    batch.non_tensor_batch["answer_types"] = ttrl_metrics["_answer_types"]
if "_oracle_answer_types" in ttrl_metrics:
    batch.non_tensor_batch["oracle_answer_types"] = ttrl_metrics["_oracle_answer_types"]
if "_consistency_rate" in ttrl_metrics:
    batch.non_tensor_batch["consistency_rate"] = ttrl_metrics["_consistency_rate"]
if "_accuracy_rate" in ttrl_metrics:
    batch.non_tensor_batch["accuracy_rate"] = ttrl_metrics["_accuracy_rate"]
if "_label_accuracy" in ttrl_metrics:
    batch.non_tensor_batch["label_accuracy"] = ttrl_metrics["_label_accuracy"]
if "_zero_advantage_mask" in ttrl_metrics:
    batch.non_tensor_batch["zero_advantage_mask"] = ttrl_metrics["_zero_advantage_mask"]
```

**OOM 问题**：
- ttrl_metrics 包含多个数组（answer_types, oracle_answer_types, consistency_rate 等）
- 这些都被复制保存到 batch.non_tensor_batch
- 对于大 batch 大小，这些数组占用额外的内存
- 当用于 DIVERSITY_DENSITY_HYBRID 和 PASS_GRPO_PENALIZED 计算时，会被多次引用

---

## 六、缺失的内存释放

### 6.1 新版本增加的内存清理（Pre 版本没有）

**_validate() 方法末尾**（第 948-955 行）：
```python
# Clean up memory explicitly to prevent OOM in subsequent generation
import gc
del test_batch, test_gen_batch, test_gen_batch_padded, test_output_gen_batch
del input_ids, output_ids
gc.collect()
import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()
```

**fit() 方法循环末尾**（第 1290-1296 行）：
```python
# Explicitly free batch and metrics from this step to prevent memory leaks from python's delayed GC
del batch, batch_dict
if 'gen_batch' in locals():
    del gen_batch
if 'gen_batch_output' in locals():
    del gen_batch_output
import gc
gc.collect()
```

**注意**：这些清理代码是 **新版本才有的**，说明新版本已经意识到内存问题！

但这些清理仍然不够，因为：
1. 没有清理 advantages, returns 等计算结果
2. 没有清理 diversity_density_config 和其他临时结构
3. 没有在关键的中间步骤（如 compute_advantage 中）进行清理

---

## 七、关键 OOM 发生点总结

### 优先级 1（最可能导致 OOM）

1. **PASS_GRPO_PENALIZED 的 metrics 计算**
   - 位置：compute_advantage 第 437-460 行
   - 问题：计算返回 metrics dict，其中可能包含大量中间张量
   - 影响：每个 batch 都会执行，累积释放慢

2. **TTRL 数据重复→下采样的内存峰值**
   - 位置：fit() 第 1184-1235 行
   - 步骤：
     - 行 1186-1187：batch 扩大 10 倍
     - 行 1209：_balance_batch（重新排序，中间拷贝）
     - 行～1220：reward 计算（保存 10 倍的 token_level_scores）
     - 行 1227-1230：下采样到 4 倍
   - 问题：峰值内存在 10 倍时，总内存需求达到最大
   - 影响：若 batch_size=8, n_votes=10，则中间产生 80 个样本的数据

3. **DIVERSITY_DENSITY_HYBRID 的双期望计算**
   - 位置：compute_advantage 第 305-390 行
   - 问题：同时保持 div_advantages + fallback_advantages 在内存中，再做混合
   - 影响：内存占用 >2x 的单一 advantage 大小

### 优先级 2（中等 OOM 风险）

4. **diversity_density_config 新增 8 个参数**
   - 可能导致 compute_pass_grpo_penalized_advantage 进行复杂计算

5. **ADAPTIVE_PASSK 的分组循环**
   - 每个 k 值都产生独立的计算中间结果

### 优先级 3（已有缓解，但仍需改进）

6. **缺少显式的张量释放**
   - 新版本增加了 gc.collect() 但没有显式 del
   - 应该在 compute_advantage 返回前清理中间张量

---

## 八、建议的 OOM 修复方案

### 8.1 立即修复（最有效）

**修复 1：在 compute_advantage 返回前清理中间张量**
```python
# 在 compute_advantage 函数末尾
del advantages, returns  # 至少保存到 data.batch
# 清理所有临时的或中间阶段创建的张量
```

**修复 2：DIVERSITY_DENSITY_HYBRID 中分离计算**
```python
# 不要同时保存 div_advantages 和 fallback_advantages
# 方案：条件化计算，根据 use_diversity mask 决定计算哪个
```

**修复 3：TTRL 重复→下采样的优化**
```python
# 不先扩大再缩小，而是直接采样
# 改为：生成 n_votes_per_prompt 时，直接只计算前 n_samples_per_prompt 个
# 或使用 torch.randperm 进行随机选择，避免存储所有 n_votes
```

### 8.2 深层修复（需要改动 core_algos）

**修复 4：compute_pass_grpo_penalized_advantage 返回值优化**
- 不返回 metrics dict，改为返回关键标量值
- 或使用流式计算，边算边释放中间结果

**修复 5：advantage 计算流程优化**
- 使用 inplace 操作减少张量拷贝
- 对于大 batch，考虑分块计算（chunk-based computation）

---

## 九、 Pre 版本为何更稳定？

Pre 版本没有以下内容：
1. PASS_GRPO_PENALIZED 估计器
2. 10 倍的 batch 重复（仅 n_samples_per_prompt 倍）
3. diversity_density_config 的 8 个新参数
4. TTRL 下采样前的 double balance_batch

因此内存占用较低。

---

## 十、最关键的 OOM 源头

**单个最大的 OOM 原因**：

### TTRL 下采样瓶颈（第 1184-1235 行）

```
原始 batch：8 个样本
↓ (repeat 10倍)
扩大后：80 个样本（occupancy = 10x）
↓ (balance_batch - 重新排序)
中间状态：仍然 80 个数据，但多个拷贝在内存中
↓ (reward_fn 计算)
占用峰值：80 个样本的 token_level_scores 保存在内存（10x peak）
↓ (select_top_k_per_prompt)
缩小后：32 个样本（仅保留 4/10）
```

这导致：
- **10 倍的内存峰值**
- 6/10 的计算在下采样时被浪费
- 中间没有及时释放

加上新增的：
- PASS_GRPO_PENALIZED 的 metrics
- diversity_density_config 的复杂计算

导致 OOM 严重化。

