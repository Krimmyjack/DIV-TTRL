# 逐一对比导致 OOM 的具体代码改动

## 改动 1：新增 PASS_GRPO_PENALIZED AdvantageEstimator

### 位置
- **Pre 版本**：lines 56-66（共 11 个 enum 值）
- **新版本**：lines 56-67（共 12 个 enum 值）

### 代码对比
```python
# PRE 版本（第 56-66 行）
class AdvantageEstimator(str, Enum):
    GAE = "gae"
    GRPO = "grpo"
    REINFORCE_PLUS_PLUS = "reinforce_plus_plus"
    REINFORCE_PLUS_PLUS_BASELINE = "reinforce_plus_plus_baseline"
    REMAX = "remax"
    RLOO = "rloo"
    DIVERSITY_DENSITY = "diversity_density"
    DIVERSITY_DENSITY_HYBRID = "diversity_density_hybrid"
    PASS_GRPO = "pass_grpo"
    SELECTIVE_PASSK = "selective_passk"
    ADAPTIVE_PASSK = "adaptive_passk"
    # ← 缺少 PASS_GRPO_PENALIZED

# 新版本（第 56-68 行）
class AdvantageEstimator(str, Enum):
    GAE = "gae"
    GRPO = "grpo"
    REINFORCE_PLUS_PLUS = "reinforce_plus_plus"
    REINFORCE_PLUS_PLUS_BASELINE = "reinforce_plus_plus_baseline"
    REMAX = "remax"
    RLOO = "rloo"
    DIVERSITY_DENSITY = "diversity_density"
    DIVERSITY_DENSITY_HYBRID = "diversity_density_hybrid"
    PASS_GRPO = "pass_grpo"
    SELECTIVE_PASSK = "selective_passk"
    ADAPTIVE_PASSK = "adaptive_passk"
    PASS_GRPO_PENALIZED = "pass_grpo_penalized"  # ← 新增
```

### OOM 概率分析
🟢 **低**：enum 定义本身不占用额外内存
但这个新 enum 会被用于 compute_advantage 中的复杂计算（见改动 5）

---

## 改动 2：RayPPOTrainer.__init__ 中的 use_critic 检查

### 位置
- **Pre 版本**：lines 305-316
- **新版本**：lines 305-320

### 代码对比
```python
# PRE 版本（第 316 行）
elif self.config.algorithm.adv_estimator in [
    AdvantageEstimator.GRPO,
    AdvantageEstimator.REINFORCE_PLUS_PLUS,
    AdvantageEstimator.REMAX,
    AdvantageEstimator.RLOO,
    AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE,
    AdvantageEstimator.DIVERSITY_DENSITY,
    AdvantageEstimator.DIVERSITY_DENSITY_HYBRID,
    AdvantageEstimator.PASS_GRPO,
    AdvantageEstimator.SELECTIVE_PASSK,
    AdvantageEstimator.ADAPTIVE_PASSK,
]:  # ← 缺少 PASS_GRPO_PENALIZED
    self.use_critic = False

# 新版本（第 320 行）
elif self.config.algorithm.adv_estimator in [
    AdvantageEstimator.GRPO,
    AdvantageEstimator.REINFORCE_PLUS_PLUS,
    AdvantageEstimator.REMAX,
    AdvantageEstimator.RLOO,
    AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE,
    AdvantageEstimator.DIVERSITY_DENSITY,
    AdvantageEstimator.DIVERSITY_DENSITY_HYBRID,
    AdvantageEstimator.PASS_GRPO,
    AdvantageEstimator.SELECTIVE_PASSK,
    AdvantageEstimator.ADAPTIVE_PASSK,
    AdvantageEstimator.PASS_GRPO_PENALIZED,  # ← 新增
]:
    self.use_critic = False
```

### OOM 概率分析
🟢 **低**：这只是配置逻辑，不直接影响内存
但设置 use_critic=False 意味着不使用 critic，内存上节省了（矛盾?）

---

## 改动 3：compute_advantage 函数新增 PASS_GRPO_PENALIZED 处理

### 位置
- **Pre 版本**：无此分支（compute_advantage 在第 193-428 行，最后是 ADAPTIVE_PASSK）
- **新版本**：lines 437-460

### 代码对比
```python
# 新版本新增（lines 437-460）
elif adv_estimator == AdvantageEstimator.PASS_GRPO_PENALIZED:
    if diversity_density_config is None:
        diversity_density_config = {}
    
    k = diversity_density_config.get("k", 4)
    epsilon = diversity_density_config.get("epsilon", 1e-6)
    
    if "answer_types" not in data.non_tensor_batch:
        raise ValueError("PASS_GRPO_PENALIZED requires 'answer_types' in data.non_tensor_batch")
    
    consistency_rates = data.non_tensor_batch.get("consistency_rate", None)
    if consistency_rates is None:
        raise ValueError("PASS_GRPO_PENALIZED requires 'consistency_rate' in data.non_tensor_batch")
        
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
    
    # Log metrics to meta_info
    for k_met, v_met in metrics.items():
        data.meta_info[k_met] = v_met
        
    data.meta_info["pass_grpo_penalized/avg_total_advantage"] = advantages.mean().item()
```

### OOM 风险详解
🔴 **高**

1. **三返回值**：`compute_pass_grpo_penalized_advantage` 返回 (advantages, returns, metrics)
   - advantages 和 returns：正常张量，必要的
   - **metrics：可能包含大量中间计算张量！**

2. **metrics 循环存储**（第 457-459 行）：
   ```python
   for k_met, v_met in metrics.items():
       data.meta_info[k_met] = v_met  # ← 如果 v_met 是张量，这会保存在内存中
   ```
   - 如果 metrics 包含中间的张量（如 "penalty_matrix", "diversity_scores" 等）
   - 这些张量会被保存在 meta_info 中，直到 batch 被处理完
   - 在循环中，未使用的旧 metrics 无法及时释放

3. **可能的 metrics 内容**（推测）：
   - r_div（多年次的diversity reward）：shape (batch_size, response_length)
   - raw_a_passk（原始 pass_grpo 优势）：shape (batch_size, response_length)
   - 其他中间计算张量

4. **内存峰值**：
   - 保存 advantages：(bs, response_length)
   - 保存 returns：(bs, response_length)
   - 保存 metrics 中的所有张量（可能 2-4 个额外的同样大小的张量）
   - **总占用：~5-6倍单个张量的大小**

---

## 改动 4：_validate() 方法增加内存清理

### 位置
- **Pre 版本**：无内存清理（第 948 行直接 return metric_dict）
- **新版本**：lines 948-955

### 代码对比
```python
# 新版本新增（lines 948-955）
# Clean up memory explicitly to prevent OOM in subsequent generation
import gc
del test_batch, test_gen_batch, test_gen_batch_padded, test_output_gen_batch
del input_ids, output_ids
gc.collect()
import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()

return metric_dict
```

### OOM 意义分析
🟡 **中等**：这是新增的清理代码，说明新版本已经意识到内存问题！

**但清理不彻底**：
- 只清理了 _validate() 方法内的变量
- 没有清理 fit() 主循环中的关键变量
- torch.cuda.empty_cache() 有性能成本

**为什么仍然 OOM**：
- _validate() 通常间隔执行（test_freq），不是每个 step
- 主要的 OOM 在 fit() 的训练循环中（每个 step 都执行）

---

## 改动 5：fit() 方法中的 diversity_density_config 膨胀

### 位置
- **Pre 版本**：lines 1124-1136（仅 5 个参数）
- **新版本**：lines 1131-1154（共 13 个参数，新增 8 个）

### 代码对比
```python
# PRE 版本（lines 1124-1136）
diversity_density_config = {
    "k": getattr(self.config.algorithm, "diversity_density_k", 8),
    "fallback_estimator": getattr(
        self.config.algorithm, "diversity_density_fallback", "grpo"
    ),
    "use_metric": getattr(
        self.config.algorithm, "diversity_density_use_metric", "consistency_rate"
    ),
    "consistency_threshold": getattr(
        self.config.algorithm, "consistency_threshold", 0.0
    ),
    "selective_passk_threshold": getattr(
        self.config.algorithm, "selective_passk_threshold", 0.5
    ),
}

# 新版本（lines 1131-1154）
diversity_density_config = {
    "k": getattr(self.config.algorithm, "diversity_density_k", 4),  # ← 改为 4
    "fallback_estimator": getattr(..., "grpo"),
    "use_metric": getattr(..., "consistency_rate"),
    "consistency_threshold": getattr(..., 0.0),
    "selective_passk_threshold": getattr(..., 0.5),
    # === 新增 8 个参数 ===
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

### OOM 风险分析
🟡 **中等**

1. **Direct dict overhead**：新增 8 个 key-value 对，占内存微小
   
2. **Indirect overhead**：这些参数被传入 compute_advantage
   ```python
   batch = compute_advantage(
       batch,
       adv_estimator=self.config.algorithm.adv_estimator,
       ...
       diversity_density_config=diversity_density_config,  # ← 8 个新参数在此传递
   )
   ```

3. **在 PASS_GRPO_PENALIZED 中的使用**（第 449 行）：
   ```python
   advantages, returns, metrics = core_algos.compute_pass_grpo_penalized_advantage(
       ...
       diversity_density_config=diversity_density_config,  # ← 这里使用
       ...
   )
   ```

4. **推断的计算复杂性增加**：
   - lam_div, c_max, tau_rep 等参数暗示有额外的多步计算（diversity multiplication, penalty computation）
   - n_gram_size=3 暗示可能有 n-gram 重复检测（需要额外空间）
   - use_rep_penalty=False （默认关闭，但代表可以打开）

---

## 改动 6：fit() 方法的 advantage 计算指标增加

### 位置
- **Pre 版本**：lines 1252-1286（各类指标记录）
- **新版本**：lines 1260-1310（增加了 bootstrap_passk 和 pass_grpo_penalized 指标）

### 代码对比
```python
# 新版本新增（lines 1278-1310）

# Log bootstrap_passk metrics if available  ← 新增
for bp_key in [
    "bootstrap_passk/num_low_prompts",
    "bootstrap_passk/num_high_prompts",
    "bootstrap_passk/low_ratio",
    "bootstrap_passk/avg_low_advantage",
    "bootstrap_passk/avg_high_advantage",
    "bootstrap_passk/avg_total_advantage",
]:
    if bp_key in batch.meta_info:
        metrics[f"train/{bp_key.replace('/', '_')}"] = float(batch.meta_info[bp_key])
            
# Log pass_grpo_penalized metrics if available  ← 新增
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

### OOM 风险分析
🟢 **低**：指标记录本身不占用内存
但反映了这些指标是从中间计算结果提取的，这些中间结果 **必须存在**

---

## 改动 7：fit() 循环末尾增加显式内存清理

### 位置
- **Pre 版本**：无（循环末尾直接 progress_bar.update(1) 和 global_steps+=1）
- **新版本**：lines 1287-1296（循环末尾新增清理代码）

### 代码对比
```python
# 新版本新增（lines 1287-1296）
# Explicitly free batch and metrics from this step to prevent memory leaks from python's delayed GC
del batch, batch_dict
if 'gen_batch' in locals():
    del gen_batch
if 'gen_batch_output' in locals():
    del gen_batch_output
import gc
gc.collect()

if is_last_step:
    pprint(f"Final validation metrics: {last_val_metrics}")
    progress_bar.close()
    return

progress_bar.update(1)
self.global_steps += 1
```

### OOM 意义分析
🟡 **中等**：这是另一个说明新版本已意识到内存问题的证据！

**清理内容**：
- batch：主数据结构
- batch_dict：原始数据字典
- gen_batch：生成前的批次
- gen_batch_output：生成输出

**未清理的关键变量**：
- metrics（可能很大）
- batch 内部的 tensor（advantages, returns 等）还在内存中

**gc.collect() 的局限**：
- Python GC 是延迟的，不保证立即释放
- 对于 GPU 张量，还需要 torch.cuda.empty_cache()

---

## 改动 8：diversity_density_config 默认的 "k" 值从 8 改为 4

### 位置
- **Pre**：`"k": getattr(self.config.algorithm, "diversity_density_k", 8)`
- **新版本**：`"k": getattr(self.config.algorithm, "diversity_density_k", 4)`

### 改变意义
📊 **这是一个重要的调参变化！**

```
k=8 意味着：每个 prompt 的 8 个回答一起处理（分组大小）
↓
k=4 意味着：每个 prompt 的 4 个回答一起处理

- 分组计算复杂度与 k 成正比（通常 O(k^2) 或 O(k) 的某个幂）
- k 变小本应减少内存，但...
```

**可能的负面效应**：
- k 变小意味着精度降低（分组太小，难以准确评估多样性）
- 为了补偿精度损失，可能增加了额外的计算步骤
- 这导致新增的 8 个参数（lam_div, tau_rep 等）用于"更精细的调整"

**Net effect**: 想降低内存反而引入了复杂度，未必节省内存

---

## 改动 9：ttrl_metrics 数据保存和重复计算

### 位置
- **Pre 版本**：lines 1223-1240
- **新版本**：lines 1232-1251（增加了 recompute_post_ttrl_metrics）

### 代码对比
```python
# PRE 版本（lines 1223-1240） - 仅保存数据
if self.use_ttrl:
    from copy import deepcopy
    ttrl_metrics = reward_result["ttrl_info"]
    for k, v in ttrl_metrics.items():
        if not k.startswith("_"):  # Skip per-sample arrays
            metrics.update({f"train/{k}": v})
    
    # Down Sampling
    batch = self._select_top_k_per_prompt(batch, self.n_votes_per_prompt, self.n_samples_per_prompt)
    self.config.actor_rollout_ref.rollout.n = self.n_samples_per_prompt
    # 没有 post_ttrl_metrics，直接进入 advantage 计算

# 新版本（lines 1232-1251）- 增加 post_ttrl_metrics
if self.use_ttrl:
    from copy import deepcopy
    ttrl_metrics = reward_result["ttrl_info"]
    for k, v in ttrl_metrics.items():
        if not k.startswith("_"):  # Skip per-sample arrays
            metrics.update({f"train/{k}": v})
    
    # Down Sampling
    batch = self._select_top_k_per_prompt(batch, self.n_votes_per_prompt, self.n_samples_per_prompt)
    self.config.actor_rollout_ref.rollout.n = self.n_samples_per_prompt

    # === 新增：Recompute ttrl metrics ===
    post_reward_result = self.reward_fn.compute_post_ttrl_metrics(batch)
    for k, v in post_reward_result.items():
        metrics.update({f"train/{k}": v})

    # === 新增：Recompute Entropy ===
    post_entropy_loss = agg_loss(
        loss_mat=batch.batch["entropys"], loss_mask=batch.batch["response_mask"], loss_agg_mode=loss_agg_mode
    )
    metrics.update({"train/post_entropy": post_entropy_loss.detach().item()})
```

### OOM 风险分析
🟡 **中等**

1. **新增的 compute_post_ttrl_metrics() 调用**：
   - 这是在下采样**之后**重新计算
   - 虽然 batch 已from 80→32，但仍需重新计算一遍指标

2. **重复计算成本**：
   - 先算一遍 reward_result["ttrl_info"]（80 个样本）
   - 下采样到 32 个
   - 再算一遍 post_ttrl_metrics（32 个样本）
   - **效率低：浪费了 6/10 的计算且重复了计算**

3. **可能的内存跳跃**：
   ```
   时间轴：
   t1: reward_fn(batch) 计算 [80 个样本] → ttrl_metrics
   t2: select_top_k_per_prompt [消耗 80, 输出 32]
   t3: compute_post_ttrl_metrics(batch) 计算 [32 个样本] → post_reward_result
   t4: agg_loss 计算 entropy
   
   峰值内存：t1 时的 80 个样本数据可能未及时释放，导致 t2 时内存压力
   ```

---

## 改动 10：batch 重复和 balance_batch 的两次调用

### 位置
- **Pre 版本**：行 1174-1182（一次 balance_batch）
- **新版本**：行 1183-1234（两次 balance_batch，加上新的下采样流程）

### 代码对比
```python
# PRE 版本（行 1174-1182）
batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
batch = batch.union(gen_batch_output)

batch.batch["response_mask"] = compute_response_mask(batch)
if self.config.trainer.balance_batch:
    self._balance_batch(batch, metrics=metrics)  # ← 仅一次

# 新版本（行 1183-1234）
batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
batch = batch.union(gen_batch_output)

batch.batch["response_mask"] = compute_response_mask(batch)
if self.config.trainer.balance_batch:
    self._balance_batch(batch, metrics=metrics)  # ← 第一次

# ... [reward 计算，batch 从 80→32] ...

if self.use_ttrl:
    self._balance_batch(batch, metrics=metrics)  # ← 第二次（新增）

# ... [advantage 计算] ...
```

### OOM 风险分析
🔴 **高**：这可能是最关键的 OOM 源头！

1. **_balance_batch 的含义**：
   ```python
   def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen"):
       # ... 创建 global_idx 张量，用于重新排序
       global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
       batch.reorder(global_idx)  # ← 这个操作会创建中间拷贝！
   ```

2. **第一次 balance_batch（行 1209）**：
   - batch 大小：80 个样本（×n_votes_per_prompt）
   - 创建 reorder
   - batch 内部多个张量被拷贝

3. **第二次 balance_batch（行 1234）**：
   - batch 大小：32 个样本（×n_samples_per_prompt）
   - 再次创建 reorder
   - **内存还未从第一次释放，累积！**

4. **总体内存压力**：
   ```
   第一次 balance_batch：
   - 输入：80 个样本的 batch
   - 过程：计算分区，创建 global_idx，执行 reorder（中间拷贝）
   - 占用峰值：80 个样本的多个副本
   
   下采样到 32：
   - select_top_k_per_prompt 创建新的 index list
   - batch 被切片
   
   第二次 balance_batch：
   - 输入：32 个样本的 batch
   - 过程：再次计算分区和 reorder
   - 占用峰值：32 个样本的多个副本
   
   如果两次的内存未及时释放：总峰值 = 80 + 32 = 112 个样本的内存
   ```

---

## 改动 11：compute_advantage 中的诊断指标计算

### 位置
- **Pre 版本**：无这些诊断代码
- **新版本**：lines 1256-1277（advantage bias diagnostics）

### 新增代码
```python
# === Advantage Bias Diagnostics ===
# Compare TTA advantage (from pseudo-labels) with Oracle advantage (from true labels)
if (
    "oracle_answer_types" in batch.non_tensor_batch
    and self.config.algorithm.adv_estimator == AdvantageEstimator.PASS_GRPO
    and diversity_density_config is not None
):
    try:
        oracle_adv, _ = core_algos.compute_pass_grpo_advantage(
            token_level_rewards=batch.batch["token_level_rewards"],
            response_mask=batch.batch["response_mask"],
            index=batch.non_tensor_batch["uid"],
            answer_types=batch.non_tensor_batch["oracle_answer_types"],  # ← 额外的 advantage 计算！
            k=diversity_density_config["k"],
        )
        tta_adv = batch.batch["advantages"]
        
        # Per-sample scalar advantages
        tta_scalar = tta_adv.sum(-1)
        oracle_scalar = oracle_adv.sum(-1)
        valid = batch.batch["response_mask"].sum(-1) > 0
        
        # 创建了额外的张量：sign_match, 计算了 MSE 等
        ...
    except Exception as e:
        print(f"Warning: Advantage bias diagnostics failed: {e}")
```

### OOM 风险分析
🟡 **中等**

1. **额外的 advantage 计算**：
   - compute_pass_grpo_advantage 被调用两次（一次用 answer_types，一次用 oracle_answer_types）
   - 这产生了额外的 oracle_adv 张量

2. **中间张量**：
   - tta_scalar：sum(tta_adv)
   - oracle_scalar：sum(oracle_adv)
   - sign_match：bool 张量
   - MSE 计算的中间结果

3. **这在主循环中每个 step 都执行**（if self.use_ttrl）：
   - 每个 training step 都额外计算一遍 oracle advantage
   - 虽然是 diagnostic，但在 OOM 环境下这是奢侈的

---

## 改动总结表

| 改动编号 | 改动内容 | OOM风险 | 位置 | 影响范围 |
|---------|--------|--------|------|--------|
| 1 | 新增 PASS_GRPO_PENALIZED enum | 🟢 低 | lines 67 | enum def |
| 2 | use_critic 检查中新增 enum | 🟢 低 | line 320 | config check |
| 3 | compute_advantage 中的 PASS_GRPO_PENALIZED 处理 | 🔴 **高** | lines 437-460 | every batch |
| 4 | _validate() 内存清理 | 🟡 中 | lines 948-955 | validation only |
| 5 | diversity_density_config 中新增 8 参数 | 🟡 中 | lines 1131-1154 | 传入 compute_advantage |
| 6 | 新增指标记录（bootstrap, pass_grpo_penalized） | 🟢 低 | lines 1278-1310 | logging only |
| 7 | fit() 循环末尾内存清理 | 🟡 中 | lines 1287-1296 | every step |
| 8 | k 默认值从 8→4 | 🟡 中 | line 1131 | config change |
| 9 | 增加 post_ttrl_metrics 重计算 | 🟡 中 | lines 1240-1244 | when use_ttrl |
| 10 | 两次 balance_batch 调用 | 🔴 **高** | lines 1209, 1234 | every batch with ttrl |
| 11 | advantage bias 诊断计算 | 🟡 中 | lines 1256-1277 | when use_ttrl |

---

## 最终 OOM 排序（按优先级）

### 🔴 最严重（必须修复）
1. **改动 3**：PASS_GRPO_PENALIZED 的 metrics 管理
   - 每个 batch 执行
   - metrics dict 可能包含大量张量
   
2. **改动 10**：两次 balance_batch
   - 80→32 的过程中重复排序
   - 80 倍膨胀的峰值内存未及时释放

### 🟡 中等（应该修复）
3. **改动 5**：complexity 增加（8 个新参数）
   - 导致 PASS_GRPO_PENALIZED 计算更复杂
   
4. **改动 9**：post_ttrl_metrics 重计算
   - 下采样后再算一遍，效率低
   
5. **改动 11**：oracle advantage bias 诊断
   - 每个 step 额外计算

### 🟢 轻微（监测）
6. **改动 1, 2, 6**：定义和日志
   - 直接内存影响小

---

