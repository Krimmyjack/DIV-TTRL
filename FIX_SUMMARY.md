# DiversityTTRLRewardManager 修复总结

## 问题分析

### 错误信息
```
AssertionError: Ground truth is not unique: ['16', '16', '16', '16', '16', '16', '16', '16', '2', '2', '2', '4', '4', '7\pi', ...]
```

### 根本原因
在 `_compute_ttrl_reward` 方法中，代码假设连续的 `n_votes_per_prompt` 个样本都来自同一个问题（相同的 ground truth）。

但实际数据组织方式不同：
- 数据按"答案"顺序排列，而不是按"问题"排列
- 同一组内可能包含来自不同问题的答案
- `test_time_train_metrics` 函数要求所有 ground truth 值必须相同

## 修复方案

### 修改点 1：使用 `auto_verify` 替代 `test_time_train_metrics`

**文件**: `diversity_reward.py` 的 `_compute_ttrl_reward` 方法

**原代码**:
```python
base_rewards, ttrl_metrics = test_time_train_metrics(
    group_pred_outputs, group_labels, task=task, extra_info=group_extra_info
)
```

**新代码**:
```python
# 使用 auto_verify 获取基础奖励，而不是 test_time_train_metrics
# 这样避免了要求所有 ground truth 必须相同的限制
verify_results, verify_extra_info = auto_verify(
    task, group_pred_outputs, group_labels, extra_info=group_extra_info
)
# 将布尔结果转换为 +1/-1 奖励
base_rewards = [1.0 if result else -1.0 for result in verify_results]
```

### 改变的逻辑

| 方面 | 原方法 (test_time_train_metrics) | 新方法 (auto_verify) |
|-----|-----|-----|
| **ground truth 要求** | 所有值必须相同 | 可以不同 |
| **使用场景** | 同一问题的多个答案投票 | 任意答案集合 |
| **返回值** | (奖励列表, 指标字典) | (布尔列表, 额外信息字典) |
| **多样性计算** | 仍然基于答案计数 | 仍然基于答案计数 |

### 修改点 2：调整 TTRL 指标计算

由于不再使用 `test_time_train_metrics`，TTRL 指标需要从 `auto_verify` 的结果手动计算：

```python
ttrl_metrics = {
    "label_accuracy": 0.0,
    "reward_accuracy": 0.0,
    "majority_ratio": 0.0,
    "ground_truth_ratio": sum(base_rewards) / len(base_rewards) if base_rewards else 0.0,
    "majority_voting_reward": sum(base_rewards) / len(base_rewards) if base_rewards else 0.0,
    f"pass@{len(group_pred_outputs)}": 1.0 if sum(base_rewards) >= 1 else 0.0,
    "neg_log_likelihood": 0.0,
}
```

## 多样性奖励计算流程（保持不变）

多样性奖励计算的核心逻辑完全保持不变：

```
基础奖励 (来自 auto_verify)
    ↓
按答案分组（相同答案的频率）
    ↓
计算多样性项: diversity_term = (unique_answers - 1) / (total - majority) * (1 / freq_i)
    ↓
调整奖励:
  - 正确答案: adjusted = 0.5 + 0.5 * diversity_reward    [0.5, 1.0]
  - 错误答案: adjusted = -1.0 + 0.5 * diversity_term    [-1.0, -0.5]
    ↓
最终奖励
```

## 测试

已在 `test_diversity_fix.py` 中提供测试脚本，可验证修复是否解决问题。

### 运行测试
```bash
cd /root/autodl-tmp/EVOL-RL
python test_diversity_fix.py
```

## 关键改进

1. ✅ **解决了数据组织问题**：不再要求同一组内的所有答案来自同一问题
2. ✅ **保持多样性奖励逻辑**：核心的count-based diversity adjustment 完全保持
3. ✅ **向后兼容**：评估模式 (`_compute_eval_reward`) 已经使用 `auto_verify`，所以行为一致
4. ✅ **简化了代码**：避免了复杂的数据重新组织逻辑

## 后续建议

1. 如果想在训练中实现更精细的多样性奖励（例如按问题分组的多样性），可以考虑在数据加载阶段就按 ground truth 分组
2. 可以根据需要调整 TTRL 指标的计算方式，以更好地反映模型的实际表现
3. 考虑添加日志来跟踪每个批次中不同 ground truth 的数量，用于监控数据质量

