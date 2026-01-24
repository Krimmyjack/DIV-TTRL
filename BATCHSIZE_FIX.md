# DiversityTTRLRewardManager Batch Size Mismatch Fix

## 问题

运行 `DiversityTTRLRewardManager` 时出现以下错误：

```
RuntimeError: batch dimension mismatch, got self.batch_size=torch.Size([1024]) and value.shape=torch.Size([512, 3072]).
```

## 原因

在 `_compute_ttrl_reward` 方法中，reward_tensor 的大小被错误地设置为 `n_samples_per_prompt` 的大小：

```python
# ❌ 错误：只创建 [512, 3072] 的 tensor
reward_tensor = torch.zeros_like(
    data.batch["responses"][: prompt_num * self.n_samples_per_prompt], dtype=torch.float32
)
```

但实际上需要覆盖**所有**样本（包括不用于训练的 vote 样本）：
- 总样本数：1024
- n_votes_per_prompt：64
- n_samples_per_prompt：32
- 期望 reward_tensor 大小：[1024, 3072]

## 修复

**文件**: `verl/verl/workers/reward_manager/diversity_reward.py`

**改动前**:
```python
def _compute_ttrl_reward(self, data: DataProto):
    # ...
    prompt_num = len(data) // self.n_votes_per_prompt
    reward_tensor = torch.zeros_like(
        data.batch["responses"][: prompt_num * self.n_samples_per_prompt], dtype=torch.float32
    )
```

**改动后**:
```python
def _compute_ttrl_reward(self, data: DataProto):
    # ...
    prompt_num = len(data) // self.n_votes_per_prompt
    # Create reward tensor with full data size (all samples, not just n_samples_per_prompt)
    reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
```

## 为什么这样修复

1. **与评估模式一致**：`_compute_eval_reward` 本来就使用 `torch.zeros_like(data.batch["responses"])`
2. **与 TTRL 原始实现一致**：`ttrl.py` 也是这样做的
3. **正确的语义**：虽然只有 `n_samples_per_prompt` 个样本用于训练，但所有 `n_votes_per_prompt` 个样本都需要奖励值

## 奖励值填充逻辑

代码已经正确处理了这一点：
- 只有前 `n_samples_per_prompt` 个样本的奖励会填充到 reward_tensor 中
- 其余 vote 样本的奖励仍然会存储在 `scores` 列表中供后续使用
- 系统会从 `data.batch["acc"]` 中读取完整的奖励值

```python
for i in range(self.n_votes_per_prompt):
    current_reward = final_rewards[i]
    vlen = group_resp_lengths[i]

    # 只填充前 n_samples_per_prompt 个样本的奖励到 tensor
    if i < self.n_samples_per_prompt and vlen > 0:
        reward_tensor[prompt_i * self.n_samples_per_prompt + i, vlen - 1] = current_reward

    # 但是所有样本的奖励都会存储在 scores 中
    scores[prompt_i * self.n_votes_per_prompt + i] = current_reward
```

## 验证

修复后，训练应该能够正常运行，不会再出现 batch size mismatch 的错误。

