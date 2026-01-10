# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Core functions to implement PPO algorithms.
The function implemented in this file should be used by trainer with different distributed strategies to
implement PPO
"""

from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from scipy.special import gammaln

import verl.utils.torch_functional as verl_F


# ==============================================================================
# Hypergeometric Distribution Based Diversity Density Advantage
# ==============================================================================

def _log_comb(n: np.ndarray, k: np.ndarray) -> np.ndarray:
    """
    Compute log(C(n, k)) using log-gamma for numerical stability.
    
    Uses: log(C(n,k)) = log(n!) - log(k!) - log((n-k)!)
                      = gammaln(n+1) - gammaln(k+1) - gammaln(n-k+1)
    
    Args:
        n: Array of n values (can be negative, will return -inf)
        k: Array of k values
    
    Returns:
        Array of log(C(n,k)) values
    """
    n = np.asarray(n, dtype=np.float64)
    k = np.asarray(k, dtype=np.float64)
    
    # Handle invalid cases: n < k or n < 0 or k < 0
    valid = (n >= k) & (n >= 0) & (k >= 0)
    result = np.full_like(n, -np.inf, dtype=np.float64)
    
    # Only compute for valid cases
    valid_n = n[valid]
    valid_k = k[valid]
    result[valid] = (gammaln(valid_n + 1) - gammaln(valid_k + 1) - 
                     gammaln(valid_n - valid_k + 1))
    
    return result


def _prob_not_in_group(N: int, s_i: int, k: int) -> float:
    """
    Compute P(u_i not in G) = C(N - s_i, k) / C(N, k).
    
    Uses product form for numerical stability(stable way for k):
    P = prod_{j=0}^{k-1} (N - s_i - j) / (N - j)
    
    Args:
        N: Total number of samples
        s_i: Count of answer type i
        k: Group size
    
    Returns:
        Probability that answer type i is not in a random group of size k
    """
    if s_i >= N or k > N - s_i:
        return 0.0
    
    log_prob = 0.0
    for j in range(k):
        if N - j <= 0:
            return 0.0
        log_prob += np.log(max(N - s_i - j, 1e-10)) - np.log(N - j)
    
    return np.exp(log_prob)


def _prob_not_in_group_vectorized(N: int, s_arr: np.ndarray, k: int) -> np.ndarray:
    """
    Vectorized version: compute P(u_i not in G) for all answer types.
    
    P(u_i not in G) = C(N - s_i, k) / C(N, k)
    
    Args:
        N: Total number of samples
        s_arr: Array of counts for each answer type [s_1, s_2, ..., s_D]
        k: Group size
    
    Returns:
        Array of probabilities for each answer type
    """
    # Use log-space for numerical stability
    log_comb_N_k = _log_comb(np.array([N]), np.array([k]))[0]
    log_comb_N_minus_s_k = _log_comb(N - s_arr, np.full_like(s_arr, k))
    
    log_probs = log_comb_N_minus_s_k - log_comb_N_k
    
    # Convert back to probability, handle -inf
    probs = np.where(np.isfinite(log_probs), np.exp(log_probs), 0.0)
    
    return probs


def _prob_not_in_group_excluding_one(N: int, s_j: int, k: int) -> float:
    """
    Compute P(u_j not in G | x in G where x belongs to type t).
    
    This is C(N - 1 - s_j, k - 1) / C(N - 1, k - 1) for j != t.
    
    Essentially, we remove one sample (x) from the pool, so N -> N-1, k -> k-1.
    
    Args:
        N: Original total number of samples
        s_j: Count of answer type j (j != t)
        k: Original group size
    
    Returns:
        Conditional probability
    """
    new_N = N - 1
    new_k = k - 1
    
    if new_k <= 0:
        return 1.0  # No other samples in group
    
    if s_j >= new_N or new_k > new_N - s_j:
        return 0.0
    
    log_prob = 0.0
    for j in range(new_k):
        if new_N - j <= 0:
            return 0.0
        log_prob += np.log(max(new_N - s_j - j, 1e-10)) - np.log(new_N - j)
    
    return np.exp(log_prob)


def compute_diversity_density_advantage(
    answer_counts: Dict[int, int],
    sample_answer_types: List[int],
    k: int,
    response_mask: torch.Tensor,
    epsilon: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute diversity density advantage based on hypergeometric distribution.
    
    Mathematical formulation:
    - R = m/k where m is the number of unique answers in a random group of size k
    - μ = (1/k) * Σᵢ (1 - C(N-sᵢ,k)/C(N,k))  [global mean reward]
    - R(x) = (1/k) * [1 + Σⱼ≠ₜ (1 - C(N-1-sⱼ,k-1)/C(N-1,k-1))]  [conditional expected reward]
    - σ² = Var[R]  [computed analytically]
    - Â(x) = (R(x) - μ) / σ  [normalized advantage]
    
    Args:
        answer_counts: Dict mapping answer_id -> count {a₁: s₁, a₂: s₂, ...}
        sample_answer_types: List of answer type for each sample [t₁, t₂, ...]
        k: Group size (typically n_votes_per_prompt)
        response_mask: Shape (bs, response_length)
        epsilon: Small value for numerical stability
    
    Returns:
        advantages: (torch.Tensor) shape (bs, response_length)
        returns: (torch.Tensor) shape (bs, response_length)
    """
    # Convert answer_counts to arrays
    unique_answers = sorted(answer_counts.keys())
    D = len(unique_answers)  # Number of unique answer types
    
    if D == 0:
        # No valid answers, return zero advantages
        bs, response_length = response_mask.shape
        return torch.zeros_like(response_mask, dtype=torch.float32), torch.zeros_like(response_mask, dtype=torch.float32)
    
    answer_to_idx = {a: i for i, a in enumerate(unique_answers)}
    s_arr = np.array([answer_counts[a] for a in unique_answers], dtype=np.float64)
    N = int(s_arr.sum())  # Total number of samples
    
    # Clamp k to be valid
    k = min(k, N)
    
    if k <= 0 or N <= 0:
        bs, response_length = response_mask.shape
        return torch.zeros_like(response_mask, dtype=torch.float32), torch.zeros_like(response_mask, dtype=torch.float32)
    
    # ==== Step 1: Compute global mean μ ====
    # μ = (1/k) * Σᵢ (1 - P(uᵢ not in G))
    # P(uᵢ not in G) = C(N - sᵢ, k) / C(N, k)
    
    p_not_in_group = _prob_not_in_group_vectorized(N, s_arr, k)  # Shape: (D,)
    p_in_group = 1.0 - p_not_in_group
    global_mean = np.sum(p_in_group) / k  # μ
    
    # ==== Step 2: Compute conditional expected reward R(x) for each sample ====
    # For a sample x belonging to answer type t:
    # R(x) = (1/k) * [1 + Σⱼ≠ₜ (1 - P(uⱼ not in G | x in G))]
    # P(uⱼ not in G | x in G) = C(N-1-sⱼ, k-1) / C(N-1, k-1)
    
    bs = len(sample_answer_types)
    conditional_rewards = np.zeros(bs, dtype=np.float64)
    
    new_N = N - 1
    new_k = k - 1
    
    # Precompute P(uⱼ not in G | x in G) for all answer types j (assuming x is from type t)
    # Note: For type t itself, we need special handling
    
    if new_k > 0:
        # For j != t: P = C(N-1-sⱼ, k-1) / C(N-1, k-1)
        p_not_in_group_cond = _prob_not_in_group_vectorized(new_N, s_arr, new_k)
    else:
        # k=1, only x in group, so only type t contributes
        p_not_in_group_cond = np.ones(D, dtype=np.float64)
    
    for i, t in enumerate(sample_answer_types):
        if t not in answer_to_idx:
            # Unknown answer type, use global mean
            conditional_rewards[i] = global_mean
            continue
        
        t_idx = answer_to_idx[t]
        
        # Type t is guaranteed in group (count 1 from x)
        # For j != t: use precomputed probabilities
        # For j == t: we need to adjust since one sample of type t is already taken
        
        # The contribution from type t is always 1/k (x itself)
        # For other types j != t: contribute (1 - P(uⱼ not in G | x in G)) / k
        
        if new_k > 0:
            # For type t: P(another sample of type t in remaining k-1) 
            # = C(N-1-(sₜ-1), k-1) / C(N-1, k-1)
            # Note: s_arr[t_idx] - 1 because one sample is already x
            s_t_remaining = s_arr[t_idx] - 1
            if s_t_remaining >= 0:
                p_t_not_in_remaining = _prob_not_in_group_vectorized(
                    new_N, np.array([s_t_remaining]), new_k
                )[0]
            else:
                p_t_not_in_remaining = 1.0  # No other samples of type t
            
            # Sum contributions from all types
            # Type t: 1/k (guaranteed) + (1 - p_t_not_in_remaining)/k * 0 (already counted)
            # Actually, the formula is:
            # R(x) = (1/k) * [1 + Σⱼ≠ₜ (1 - P(uⱼ not in G | x in G))]
            
            sum_contrib = 1.0  # Type t is guaranteed to contribute 1
            for j_idx in range(D):
                if j_idx != t_idx:
                    sum_contrib += (1.0 - p_not_in_group_cond[j_idx])
            
            conditional_rewards[i] = sum_contrib / k
        else:
            # k=1, only x in group
            conditional_rewards[i] = 1.0  # R(x) = 1/1 = 1
    
    # ==== Step 3: Compute variance for normalization ====
    # Approximate variance using the formula for sampling without replacement
    # For simplicity, use empirical variance of conditional rewards
    variance = np.var(conditional_rewards)
    std = np.sqrt(variance + epsilon)
    
    # ==== Step 4: Compute normalized advantage ====
    # Â(x) = (R(x) - μ) / σ
    advantages_np = (conditional_rewards - global_mean) / std
    
    # ==== Step 5: Expand to token level ====
    response_length = response_mask.shape[1]
    
    # Convert to torch and expand
    advantages = torch.tensor(advantages_np, dtype=response_mask.dtype, device=response_mask.device)
    advantages = advantages.unsqueeze(-1) * response_mask  # (bs, response_length)
    
    returns = advantages.clone()  # For outcome-based, returns = advantages
    
    return advantages, returns


def compute_diversity_density_advantage_from_prompts(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    answer_types: np.ndarray,
    k: int,
    epsilon: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute diversity density advantage per prompt group.
    
    This is the interface function that matches other compute_*_outcome_advantage functions.
    
    Args:
        token_level_rewards: (torch.Tensor) shape (bs, response_length) - used for device/dtype
        response_mask: (torch.Tensor) shape (bs, response_length)
        index: (np.ndarray) prompt indices for each sample
        answer_types: (np.ndarray) answer type/id for each sample
        k: Group size (typically n_votes_per_prompt)
        epsilon: Small value for numerical stability
    
    Returns:
        advantages: (torch.Tensor) shape (bs, response_length)
        returns: (torch.Tensor) shape (bs, response_length)
    """
    bs, response_length = token_level_rewards.shape
    device = token_level_rewards.device
    dtype = token_level_rewards.dtype
    
    # Group samples by prompt
    prompt_to_samples = defaultdict(list)
    for i in range(bs):
        prompt_to_samples[index[i]].append(i)
    
    # Initialize advantages
    advantages = torch.zeros(bs, response_length, dtype=dtype, device=device)
    # split by prompt and compute advantage per prompt
    for prompt_idx, sample_indices in prompt_to_samples.items():
        # Collect answer types for this prompt
        prompt_answer_types = [answer_types[i] for i in sample_indices]
        
        # Count answer occurrences
        answer_counts = defaultdict(int)
        for at in prompt_answer_types:
            answer_counts[at] += 1
        
        # Compute advantages for this prompt group
        prompt_response_mask = response_mask[sample_indices]
        prompt_adv, _ = compute_diversity_density_advantage(
            answer_counts=dict(answer_counts),
            sample_answer_types=prompt_answer_types,
            k=k,
            response_mask=prompt_response_mask,
            epsilon=epsilon
        )
        
        # Assign back to global advantages
        for local_i, global_i in enumerate(sample_indices):
            advantages[global_i] = prompt_adv[local_i]
    
    return advantages, advantages.clone()


def _compute_pass_at_k_weight(N: int, c_maj: int, k: int) -> float:
    """
    Compute pass@k weight using the formula: ρ_K = 1 - C(N - c_maj, k) / C(N, k)
    
    This represents the probability of having at least one correct answer
    in a random group of k samples from N total samples.
    
    Args:
        N: Total number of samples
        c_maj: Count of majority (correct) answers
        k: Group size
    
    Returns:
        Weight ρ_K in [0, 1]
    """
    if N <= 0 or k <= 0 or k > N:
        return 0.0
    
    # Handle edge cases
    if c_maj >= N:  # All are correct
        return 1.0
    
    if c_maj <= 0:  # None are correct
        return 0.0
    
    # Use log-space for numerical stability
    log_comb_N_k = _log_comb(np.array([N]), np.array([k]))[0]
    log_comb_N_minus_c_k = _log_comb(np.array([N - c_maj]), np.array([k]))[0]
    
    if not np.isfinite(log_comb_N_k) or not np.isfinite(log_comb_N_minus_c_k):
        # Fallback: use product form
        log_prob = 0.0
        for j in range(k):
            log_prob += np.log(max(N - c_maj - j, 1e-10)) - np.log(N - j)
        prob_not_in_group = np.exp(log_prob)
    else:
        prob_not_in_group = np.exp(log_comb_N_minus_c_k - log_comb_N_k)
    
    # Clamp to [0, 1]
    prob_not_in_group = np.clip(prob_not_in_group, 0.0, 1.0)
    weight = 1.0 - prob_not_in_group
    
    return weight


def _compute_pass_at_k_weight_vectorized(N: int, c_maj_arr: np.ndarray, k: int) -> np.ndarray:
    """
    Vectorized version: compute pass@k weights for multiple groups.
    
    Args:
        N: Total number of samples
        c_maj_arr: Array of majority answer counts for each group
        k: Group size
    
    Returns:
        Array of weights ρ_K for each group
    """
    c_maj_arr = np.asarray(c_maj_arr, dtype=np.float64)
    weights = np.zeros_like(c_maj_arr, dtype=np.float64)
    
    for i, c_maj in enumerate(c_maj_arr):
        weights[i] = _compute_pass_at_k_weight(N, int(c_maj), k)
    
    return weights


def compute_pass_grpo_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    answer_types: np.ndarray,
    k: int,
    epsilon: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute pass@k reweighted GRPO advantage.
    
    This implements the pass@k reweighting scheme:
    1. Compute standard GRPO advantages
    2. Per prompt group, compute pass@k weight: ρ_K = 1 - C(N - c_maj, k) / C(N, k)
    3. Reweight advantages by ρ_K within each prompt group
    
    where:
    - N: total samples per prompt
    - c_maj: count of majority (correct) answers per prompt  
    - k: group size (typically n_votes_per_prompt)
    
    Args:
        token_level_rewards: (torch.Tensor) shape (bs, response_length)
        response_mask: (torch.Tensor) shape (bs, response_length)
        index: (np.ndarray) prompt indices for each sample
        answer_types: (np.ndarray) answer type/id for each sample
        k: Group size
        epsilon: Small value for numerical stability
    
    Returns:
        advantages: (torch.Tensor) shape (bs, response_length) - reweighted advantages
        returns: (torch.Tensor) shape (bs, response_length) - same as advantages
    """
    bs, response_length = token_level_rewards.shape
    device = token_level_rewards.device
    dtype = token_level_rewards.dtype
    
    # Step 1: Compute base GRPO advantage
    grpo_advantages, _ = compute_grpo_outcome_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=index,
    )
    
    # Step 2: Group samples by prompt and compute pass@k weights
    prompt_to_samples = defaultdict(list)
    prompt_to_answers = defaultdict(list)
    
    for i in range(bs):
        prompt_idx = index[i]
        prompt_to_samples[prompt_idx].append(i)
        prompt_to_answers[prompt_idx].append(answer_types[i])
    
    # Initialize reweight factors
    reweight_factors = torch.ones(bs, 1, dtype=dtype, device=device)
    
    # Compute pass@k weight per prompt
    for prompt_idx, sample_indices in prompt_to_samples.items():
        N = len(sample_indices)  # Total samples for this prompt
        
        # Count majority (correct) answers - assume 0 is correct/majority
        # and other values are incorrect
        answers = prompt_to_answers[prompt_idx]
        c_maj = sum(1 for a in answers if a == 0)  # Count of correct answers
        
        # Compute pass@k weight
        pass_at_k_weight = _compute_pass_at_k_weight(N, c_maj, k)
        
        # Assign reweight factor to all samples in this prompt
        weight_tensor = torch.tensor(pass_at_k_weight, dtype=dtype, device=device)
        for sample_idx in sample_indices:
            reweight_factors[sample_idx] = weight_tensor
    
    # Step 3: Reweight advantages
    advantages = grpo_advantages * reweight_factors
    returns = advantages.clone()
    
    return advantages, returns


class AdaptiveKLController:
    """
    Adaptive KL controller described in the paper:
    https://arxiv.org/pdf/1909.08593.pdf
    """

    def __init__(self, init_kl_coef, target_kl, horizon):
        self.value = init_kl_coef
        self.target = target_kl
        self.horizon = horizon

    def update(self, current_kl, n_steps):
        target = self.target
        proportional_error = np.clip(current_kl / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult


class FixedKLController:
    """Fixed KL controller."""

    def __init__(self, kl_coef):
        self.value = kl_coef

    def update(self, current_kl, n_steps):
        pass


def get_kl_controller(kl_ctrl):
    if kl_ctrl.type == "fixed":
        return FixedKLController(kl_coef=kl_ctrl.kl_coef)
    elif kl_ctrl.type == "adaptive":
        assert kl_ctrl.horizon > 0, f"horizon must be larger than 0. Got {kl_ctrl.horizon}"
        return AdaptiveKLController(init_kl_coef=kl_ctrl.kl_coef, target_kl=kl_ctrl.target_kl, horizon=kl_ctrl.horizon)
    else:
        raise NotImplementedError


def compute_gae_advantage_return(
    token_level_rewards: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    gamma: torch.Tensor,
    lam: torch.Tensor,
):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        values: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length). [EOS] mask. The token after [EOS] have mask zero.
        gamma: `(float)`
            discounted factor used in RL
        lam: `(float)`
            lambda value when computing Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    with torch.no_grad():
        lastgaelam = 0
        advantages_reversed = []
        gen_len = token_level_rewards.shape[-1]

        for t in reversed(range(gen_len)):
            nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
            delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam = delta + gamma * lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)

        returns = advantages + values
        advantages = verl_F.masked_whiten(advantages, response_mask)
    return advantages, returns


# NOTE(sgm): this implementation only consider outcome supervision, where the reward is a scalar.
def compute_grpo_outcome_advantage(
    token_level_rewards: torch.Tensor, response_mask: torch.Tensor, index: np.ndarray, epsilon: float = 1e-6
):
    """
    Compute advantage for GRPO, operating only on Outcome reward
    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores


def compute_reinforce_plus_plus_baseline_outcome_advantage(
    token_level_rewards: torch.Tensor, response_mask: torch.Tensor, index: torch.Tensor, epsilon: float = 1e-6
):
    """
    Compute advantage for RF++-baseline (https://arxiv.org/abs/2501.03262), operating only on Outcome reward
    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = scores[i] - id2mean[index[i]]

        scores = scores.unsqueeze(-1).tile([1, response_length]) * response_mask
        scores = verl_F.masked_whiten(scores, response_mask)

    return scores, scores


def compute_rloo_outcome_advantage(
    token_level_rewards: torch.Tensor, response_mask: torch.Tensor, index: np.ndarray, epsilon: float = 1e-6
):
    """
    Compute advantage for RLOO based on https://arxiv.org/abs/2402.14740
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            response_num = len(id2score[index[i]])
            if response_num > 1:
                scores[i] = scores[i] * response_num / (response_num - 1) - id2mean[index[i]] * response_num / (
                    response_num - 1
                )
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores


def compute_reinforce_plus_plus_outcome_advantage(
    token_level_rewards: torch.Tensor, response_mask: torch.Tensor, gamma: torch.Tensor
):
    """
    Compute advantage for REINFORCE++.
    This implementation is based on the paper: https://arxiv.org/abs/2501.03262
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """

    with torch.no_grad():
        returns = torch.zeros_like(token_level_rewards)
        running_return = 0

        for t in reversed(range(token_level_rewards.shape[1])):
            running_return = token_level_rewards[:, t] + gamma * running_return
            returns[:, t] = running_return
            # Reset after EOS
            running_return = running_return * response_mask[:, t]

        advantages = verl_F.masked_whiten(returns, response_mask)
        advantages = advantages * response_mask

    return advantages, returns


def compute_remax_outcome_advantage(
    token_level_rewards: torch.Tensor, reward_baselines: torch.Tensor, response_mask: torch.Tensor
):
    """
    Compute advantage for ReMax, operating only on Outcome reward
    This implementation is based on the paper: https://arxiv.org/abs/2310.10505

    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        reward_baselines: `(torch.Tensor)`
            shape: (bs,)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """

    with torch.no_grad():
        returns = (token_level_rewards * response_mask).flip(dims=[-1]).cumsum(dim=-1).flip(dims=[-1])
        advantages = returns - reward_baselines.unsqueeze(-1) * response_mask

    return advantages, returns


def compute_rewards(token_level_scores, old_log_prob, ref_log_prob, kl_ratio):
    kl = old_log_prob - ref_log_prob
    return token_level_scores - kl * kl_ratio


def agg_loss(loss_mat: torch.Tensor, loss_mask: torch.Tensor, loss_agg_mode: str):
    """
    Aggregate the loss matrix into a scalar.
    Args:
        loss_mat: `(torch.Tensor)`
            shape: (bs, response_length)
        loss_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        loss_agg_mode: (str) choices: "token-mean" / "seq-mean-token-sum" / "seq-mean-token-mean"
            "token-mean" is the default behavior
    Returns:
        loss: `a scalar torch.Tensor`
            aggregated loss
    """
    if loss_agg_mode == "token-mean":
        loss = verl_F.masked_mean(loss_mat, loss_mask)
    elif loss_agg_mode == "seq-mean-token-sum":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)  # token-sum
        loss = torch.mean(seq_losses)  # seq-mean
    elif loss_agg_mode == "seq-mean-token-mean":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1) / torch.sum(loss_mask, dim=-1)  # token-mean
        loss = torch.mean(seq_losses)  # seq-mean
    else:
        raise ValueError(f"Invalid loss_agg_mode: {loss_agg_mode}")

    return loss


def compute_policy_loss(
    old_log_prob,
    log_prob,
    advantages,
    response_mask,
    cliprange=None,
    cliprange_low=None,
    cliprange_high=None,
    clip_ratio_c=3.0,
    loss_agg_mode="token-mean",
):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1122
    Args:
        old_log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        cliprange: (float)
            The clip range used in PPO. See https://arxiv.org/abs/1707.06347
        cliprange_low: (float)
            The lower clip range used in PPO.
        cliprange_high: (float)
            The higher clip range used in PPO.
        clip_ratio_c: (float) default: 3.0
            The lower bound of the ratio for dual-clip PPO, See https://arxiv.org/pdf/1912.09729
        loss_agg_mode: (str) choices: "token-mean" / "seq-mean-token-sum" / "seq-mean-token-mean"
            "token-mean" is the default behavior

    Returns:
        pg_loss: `a scalar torch.Tensor`
            policy gradient loss computed via PPO
        pg_clipfrac: (float)
            the fraction of policy gradient loss being clipped
        ppo_kl: (float)
            the estimated KL divergence between the latest updating policy and the old sampling policy
        pg_clipfrac_lower: (float)
            the fraction of policy gradient loss being clipped when the advantage is negative
    """
    assert clip_ratio_c > 1.0, (
        f"The lower bound of the clip_ratio_c for dual-clip PPO should be greater than 1.0, but get the value: {clip_ratio_c}."
    )

    negative_approx_kl = log_prob - old_log_prob
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    pg_losses1 = -advantages * ratio
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange
    pg_losses2 = -advantages * torch.clamp(
        ratio, 1 - cliprange_low, 1 + cliprange_high
    )  # - clip(ratio, 1-cliprange, 1+cliprange) * A
    clip_pg_losses1 = torch.maximum(
        pg_losses1, pg_losses2
    )  # max(-ratio * A, -clip(ratio, 1-cliprange, 1+cliprange) * A)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)

    pg_losses3 = -advantages * clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    pg_clipfrac_lower = verl_F.masked_mean(
        torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(), response_mask
    )

    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

    return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower


def compute_entropy_loss(logits, response_mask):
    """Compute Categorical entropy loss

    Args:
        logits: `(torch.Tensor)`
            shape: (bs, response_length, vocab_size)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        entropy: a scalar torch.Tensor

    """
    # compute entropy
    entropy = verl_F.entropy_from_logits(logits)  # (bs, response_len)
    entropy_loss = verl_F.masked_mean(entropy, mask=response_mask)
    return entropy_loss


def compute_value_loss(vpreds, returns, values, response_mask, cliprange_value):
    """Compute the value loss. Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1151

    Args:
        vpreds (`torch.FloatTensor`):
            Predicted values of the value head, shape (`batch_size`, `response_length`)
        values (`torch.FloatTensor`):
            Old values of value head, shape (`batch_size`, `response_length`)
        returns: (`torch.FloatTensor`):
            Ground truth returns, shape (`batch_size`, `response_length`)

    Returns:
        vf_loss: a scalar (`torch.FloatTensor`):
            value function loss
        vf_clipfrac: a float
            The ratio of vf being clipped

    """
    vpredclipped = verl_F.clip_by_value(vpreds, values - cliprange_value, values + cliprange_value)
    vf_losses1 = (vpreds - returns) ** 2
    vf_losses2 = (vpredclipped - returns) ** 2
    vf_loss = 0.5 * verl_F.masked_mean(torch.max(vf_losses1, vf_losses2), response_mask)
    vf_clipfrac = verl_F.masked_mean(torch.gt(vf_losses2, vf_losses1).float(), response_mask)
    return vf_loss, vf_clipfrac


def kl_penalty(logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, kl_penalty) -> torch.FloatTensor:
    """Compute KL divergence given logprob and ref_logprob.
    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1104

    Args:
        logprob:
        ref_logprob:

    Returns:

    """
    if kl_penalty == "kl":
        return logprob - ref_logprob

    if kl_penalty == "abs":
        return (logprob - ref_logprob).abs()

    if kl_penalty == "mse":
        return 0.5 * (logprob - ref_logprob).square()

    # J. Schulman. Approximating kl divergence, 2020.
    # # URL http://joschu.net/blog/kl-approx.html.
    if kl_penalty == "low_var_kl":
        kl = ref_logprob - logprob
        ratio = torch.exp(kl)
        kld = (ratio - kl - 1).contiguous()
        return torch.clamp(kld, min=-10, max=10)

    if kl_penalty == "full":
        # so, here logprob and ref_logprob should contain the logits for every token in vocabulary
        raise NotImplementedError

    raise NotImplementedError
