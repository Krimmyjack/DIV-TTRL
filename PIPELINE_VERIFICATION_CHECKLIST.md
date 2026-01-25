# Pipeline 完整性检查清单

## 文件修改列表及验证

### 1. `verl/workers/reward_manager/diversity_reward.py`

#### 修改1: `_apply_diversity_adjustment` 的返回类型
- **行号**: ~186
- **验证**: `def _apply_diversity_adjustment(...) -> tuple[list[float], float]:`
- [ ] 检查返回类型注解已更新

#### 修改2: `_compute_ttrl_reward` 中添加答案类型收集
- **行号**: ~308-438
- **验证内容**:
  - [ ] `all_answer_types = []` 被初始化
  - [ ] `all_consistency_rates = []` 被初始化
  - [ ] 在循环中收集 `answer_types` 和 `consistency_rate`
  - [ ] `answer_to_id = {ans: hash(ans) for ans in set(final_answers)}` 被执行
  - [ ] 最后返回的 `ttrl_info` 包含:
    ```python
    ttrl_info["_answer_types"] = np.array(training_answer_types)
    ttrl_info["_consistency_rate"] = np.array(training_consistency_rates)
    ```

**检查命令**:
```bash
grep -n "_answer_types\|_consistency_rate" verl/verl/workers/reward_manager/diversity_reward.py
```

---

### 2. `verl/trainer/ppo/core_algos.py`

#### 修改1: 导入必要的模块
- **行号**: ~23
- **验证**: 包含 `from scipy.special import gammaln`
- [ ] scipy.special.gammaln 已导入
- [ ] Tuple, List, Optional 已导入

#### 修改2: 多样性密度优势函数组
- **行号**: ~32-355
- **验证的函数**:
  - [ ] `_log_comb(n: np.ndarray, k: np.ndarray) -> np.ndarray`
  - [ ] `_prob_not_in_group(N: int, s_i: int, k: int) -> float`
  - [ ] `_prob_not_in_group_vectorized(N: int, s_arr: np.ndarray, k: int) -> np.ndarray`
  - [ ] `_prob_not_in_group_excluding_one(N: int, s_j: int, k: int) -> float`
  - [ ] `compute_diversity_density_advantage(...) -> Tuple[torch.Tensor, torch.Tensor]`
  - [ ] `compute_diversity_density_advantage_from_prompts(...) -> Tuple[torch.Tensor, torch.Tensor]`

**检查返回语句**:
```bash
grep -A 2 "def compute_diversity_density_advantage" verl/verl/trainer/ppo/core_algos.py | grep "return"
```

---

### 3. `verl/trainer/ppo/ray_trainer.py`

#### 修改1: AdvantageEstimator 枚举
- **行号**: ~73-86
- **验证**: 包含两个新的枚举值:
  - [ ] `DIVERSITY_DENSITY = "diversity_density"`
  - [ ] `DIVERSITY_DENSITY_HYBRID = "diversity_density_hybrid"`

**检查命令**:
```bash
grep -A 8 "class AdvantageEstimator" verl/verl/trainer/ppo/ray_trainer.py
```

#### 修改2: `compute_advantage` 函数
- **行号**: ~191-375
- **验证内容**:
  - [ ] 函数签名包含 `diversity_density_config: dict = None`
  - [ ] 包含处理 `AdvantageEstimator.DIVERSITY_DENSITY` 的代码块
  - [ ] 包含处理 `AdvantageEstimator.DIVERSITY_DENSITY_HYBRID` 的代码块
  - [ ] HYBRID模式中实现概率混合:
    ```python
    p = torch.tensor(consistency_rates, dtype=dtype, device=device)
    random_vals = torch.rand(bs, device=device, dtype=dtype)
    use_diversity = (random_vals < p).float().unsqueeze(-1)
    advantages = use_diversity * div_advantages + (1 - use_diversity) * fallback_advantages
    ```

**检查命令**:
```bash
grep -n "DIVERSITY_DENSITY\|use_diversity\|consistency_rates" verl/verl/trainer/ppo/ray_trainer.py
```

#### 修改3: `_balance_batch` 方法
- **行号**: ~985-1010
- **验证内容**:
  - [ ] 添加了非张量数据同步重新排序:
    ```python
    if "answer_types" in batch.non_tensor_batch:
        batch.non_tensor_batch["answer_types"] = \
            batch.non_tensor_batch["answer_types"][global_idx.cpu().numpy()]
    ```
  - [ ] `consistency_rate` 也被同步处理

**检查命令**:
```bash
grep -A 20 "def _balance_batch" verl/verl/trainer/ppo/ray_trainer.py | grep -E "answer_types|consistency_rate"
```

#### 修改4: `_select_top_k_per_prompt` 方法
- **行号**: ~1013-1032
- **验证内容**:
  - [ ] 返回 `data_selected` 而不是 `data[selected_indices]`
  - [ ] 添加了非张量数据的选择:
    ```python
    if "answer_types" in data.non_tensor_batch:
        selected_answer_types = data.non_tensor_batch["answer_types"][selected_indices]
        data_selected.non_tensor_batch["answer_types"] = selected_answer_types
    ```
  - [ ] `consistency_rate` 也被同步选择

**检查命令**:
```bash
grep -A 20 "def _select_top_k_per_prompt" verl/verl/trainer/ppo/ray_trainer.py | tail -15
```

#### 修改5: fit() 函数中的 TTRL 处理
- **行号**: ~1161-1210
- **验证内容**:
  - [ ] 在 `_select_top_k_per_prompt` 之前添加答案类型数据:
    ```python
    if "_answer_types" in ttrl_metrics:
        batch.non_tensor_batch["answer_types"] = ttrl_metrics["_answer_types"]
    if "_consistency_rate" in ttrl_metrics:
        batch.non_tensor_batch["consistency_rate"] = ttrl_metrics["_consistency_rate"]
    ```
  - [ ] 然后调用 `_select_top_k_per_prompt`

**检查命令**:
```bash
grep -B 5 -A 15 "_select_top_k_per_prompt" verl/verl/trainer/ppo/ray_trainer.py | head -30
```

#### 修改6: compute_advantage 的调用
- **行号**: ~1240-1270
- **验证内容**:
  - [ ] 构建 `diversity_density_config`:
    ```python
    diversity_density_config = {
        "k": getattr(self, "n_votes_per_prompt", 64),
        "fallback_estimator": getattr(self.config.algorithm, "diversity_density_fallback", "grpo"),
    }
    ```
  - [ ] 调用 `compute_advantage` 时传递 `diversity_density_config`
  - [ ] 记录 `diversity_density_usage_ratio` 指标

**检查命令**:
```bash
grep -B 5 -A 10 "compute_advantage(" verl/verl/trainer/ppo/ray_trainer.py | tail -20
```

---

## 数据流验证

### Pipeline 顺序检查表

```
Step 1: diversity_reward.py 计算答案类型
  └─ 检查点: _answer_types, _consistency_rate 在 ttrl_info 中

Step 2: ray_trainer fit() 获取 ttrl_metrics
  └─ 检查点: reward_result["ttrl_info"] 包含这些键

Step 3: 存储到 batch.non_tensor_batch【CRITICAL: 必须在 _select_top_k 前】
  └─ 检查点: 代码顺序正确

Step 4: _select_top_k_per_prompt 同时调整大小
  └─ 检查点: selected_answer_types 和 selected_consistency_rates 被正确赋值

Step 5: _balance_batch 同步重新排序
  └─ 检查点: 非张量数据使用相同的 global_idx 重新排序

Step 6: compute_advantage 使用数据
  └─ 检查点: compute_diversity_density_advantage_from_prompts 能收到数据

Step 7: 概率混合
  └─ 检查点: use_diversity 采样和混合计算执行
```

---

## 快速测试命令

### 检查所有关键函数是否存在
```bash
# 检查新增的函数
python3 -c "from verl.trainer.ppo.core_algos import compute_diversity_density_advantage; print('✓ compute_diversity_density_advantage exists')"
python3 -c "from verl.trainer.ppo.core_algos import compute_diversity_density_advantage_from_prompts; print('✓ compute_diversity_density_advantage_from_prompts exists')"
```

### 检查枚举值
```bash
python3 -c "from verl.trainer.ppo.ray_trainer import AdvantageEstimator; print(f'✓ DIVERSITY_DENSITY = {AdvantageEstimator.DIVERSITY_DENSITY.value}'); print(f'✓ DIVERSITY_DENSITY_HYBRID = {AdvantageEstimator.DIVERSITY_DENSITY_HYBRID.value}')"
```

### 检查配置示例
```bash
# 在你的配置文件中应该有：
grep -r "adv_estimator.*diversity" .

# 输出示例：
# algorithm:
#   adv_estimator: "diversity_density_hybrid"
```

---

## 预期行为

### 配置为 `diversity_density_hybrid` 时的执行流程

```
1. 前向传播和奖励计算
   ├─ 计算 n_votes_per_prompt 个投票
   ├─ 提取答案类型和计算一致性率
   └─ 保存 _answer_types 和 _consistency_rate 到 ttrl_info

2. 批处理
   ├─ 存储答案类型到 batch.non_tensor_batch
   ├─ 下采样到 n_samples_per_prompt
   ├─ 重新排序批次
   └─ 所有非张量数据同步变化

3. 优势计算
   ├─ 以概率 p 使用多样性密度优势（鼓励探索）
   ├─ 以概率 (1-p) 使用 GRPO/RLOO（鼓励利用）
   └─ 记录使用比率作为指标

4. 损失计算
   └─ 使用混合优势进行PPO更新
```

### 预期输出日志

```
train/diversity_ratio: 0.XX        # 多样性比率
train/diversity_density_usage_ratio: 0.YY  # 使用多样性优势的比率
```

---

## 常见错误及修复

| 错误 | 原因 | 修复 |
|------|------|------|
| `KeyError: 'answer_types'` | 数据在 `_select_top_k` 前没被添加 | 检查fit()函数中的代码顺序 |
| `IndexError: index out of bounds` | 数据大小不匹配 | 检查 `_select_top_k_per_prompt` 是否同时调整了非张量数据 |
| `AttributeError: 'dict' object has no attribute` | consistency_rate 是列表而不是numpy数组 | 检查 `diversity_reward.py` 中的数组转换 |
| `ValueError: Expected 2D tensor` | 优势形状错误 | 检查 `compute_diversity_density_advantage` 的返回形状 |

---

## 最终验收标准

- [ ] 所有函数签名和返回类型正确
- [ ] 数据流在所有修改batch的操作中保持同步
- [ ] 概率混合策略被正确实现
- [ ] 配置文件中正确设置了 `adv_estimator`
- [ ] 没有缺失的导入或未定义的变量
- [ ] 单元测试通过（参考 test_diversity_pipeline.py）

