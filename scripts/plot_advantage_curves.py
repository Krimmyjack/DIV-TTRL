#!/usr/bin/env python
"""
Advantage Function Visualization Script for DIV-TTRL

This script visualizes TWO types of advantage functions from core_algos.py:

1. Pass@k Advantage (compute_pass_grpo_advantage):
   - Based on probability of at least one correct answer in k samples
   - R_mean = 1 - C(N_neg, k) / C(N, k)
   - Â_pos = P_fail / σ  (Eq. 14)
   - Â_neg = (P_fail - P_cond) / σ  (Eq. 15)

2. Diversity Density Advantage (compute_diversity_density_advantage):
   - Based on hypergeometric distribution
   - μ = (1/k) * Σᵢ (1 - C(N-sᵢ,k)/C(N,k))  [global mean reward]
   - R(x) = Expected diversity reward given sample x is in group
   - Â(x) = (R(x) - μ) / σ  [normalized advantage]

Sum of Absolute Advantage (Eq. 16): η = N_pos * |Â_pos| + N_neg * |Â_neg|
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gammaln
from typing import Tuple, Dict, List
from collections import defaultdict


# ==============================================================================
# Global Constants
# ==============================================================================
N_ROLLOUT = 32  # Number of rollouts per prompt


# ==============================================================================
# Mathematical Helper Functions (from core_algos.py)
# ==============================================================================
def _log_comb(n: np.ndarray, k: np.ndarray) -> np.ndarray:
    """
    Compute log(C(n, k)) using log-gamma for numerical stability.
    
    Uses: log(C(n,k)) = log(n!) - log(k!) - log((n-k)!)
                      = gammaln(n+1) - gammaln(k+1) - gammaln(n-k+1)
    """
    n = np.asarray(n, dtype=np.float64)
    k = np.asarray(k, dtype=np.float64)
    
    # Handle invalid cases: n < k or n < 0 or k < 0
    valid = (n >= k) & (n >= 0) & (k >= 0)
    result = np.full_like(n, -np.inf, dtype=np.float64)
    
    valid_n = n[valid]
    valid_k = k[valid]
    result[valid] = (gammaln(valid_n + 1) - gammaln(valid_k + 1) - 
                     gammaln(valid_n - valid_k + 1))
    
    return result


def compute_pass_k_advantage(N: int, accuracy: float, k: int, epsilon: float = 1e-6) -> Tuple[float, float]:
    """
    Compute Pass@k advantage values for a given accuracy level.
    
    Based on compute_pass_grpo_advantage in core_algos.py:
    - R_mean = 1 - C(N_neg, k) / C(N, k)  [Prob of at least one correct in k samples]
    - σ = √(R_mean * (1 - R_mean))
    - Â_pos = (1 - R_mean) / σ = P_fail / σ  (Eq. 14)
    - Â_neg = (P_fail - P_cond) / σ  (Eq. 15)
    
    Args:
        N: Total number of samples (N_rollout)
        accuracy: Accuracy level (0.0 to 1.0)
        k: Group size for Pass@k
        epsilon: Small value for numerical stability
    
    Returns:
        Tuple of (A_pos, A_neg)
    """
    # Derived counts
    N_pos = int(round(N * accuracy))
    N_neg = N - N_pos
    
    # Handle edge cases
    if N_pos == N:  # 100% accuracy - all correct
        return 0.0, 0.0
    if N_pos == 0:  # 0% accuracy - all wrong
        return 0.0, 0.0
    
    # 1. Compute Group Mean Reward: R_mean = 1 - C(N_neg, k) / C(N, k)
    log_prob_fail = _log_comb(np.array([N_neg]), np.array([k]))[0] - \
                    _log_comb(np.array([N]), np.array([k]))[0]
    
    prob_fail = np.exp(log_prob_fail) if np.isfinite(log_prob_fail) else 0.0
    prob_fail = np.clip(prob_fail, 0.0, 1.0)
    r_mean = 1.0 - prob_fail
    
    # 2. Compute Group Standard Deviation
    variance = r_mean * (1.0 - r_mean)
    std = np.sqrt(variance)
    
    if std < epsilon:
        return 0.0, 0.0
    
    # 3. Positive Advantage (Eq. 14): Â_pos = P_fail / σ
    a_pos = prob_fail / std
    
    # 4. Negative Advantage (Eq. 15)
    # P_cond = C(N_neg - 1, k - 1) / C(N - 1, k - 1)
    if N_neg >= 1 and k >= 1:
        log_p_cond_fail = _log_comb(np.array([N_neg - 1]), np.array([k - 1]))[0] - \
                          _log_comb(np.array([N - 1]), np.array([k - 1]))[0]
        p_cond_fail = np.exp(log_p_cond_fail) if np.isfinite(log_p_cond_fail) else 0.0
        p_cond_fail = np.clip(p_cond_fail, 0.0, 1.0)
    else:
        p_cond_fail = 0.0
    
    # Â_neg = (P_fail - P_cond) / σ
    a_neg = (prob_fail - p_cond_fail) / std
    
    return a_pos, a_neg


def get_advantage_values(accuracy: float, mode: str = 'pass_at_1', N: int = N_ROLLOUT) -> Tuple[float, float]:
    """
    Modular function to get advantage values based on mode.
    
    This is a placeholder function designed for easy experimentation.
    
    Args:
        accuracy: Rollout accuracy (0.0 to 1.0)
        mode: 'pass_at_1' or 'pass_at_k'
        N: Number of rollouts
    
    Returns:
        Tuple of (A_pos, A_neg)
    """
    if mode == 'pass_at_1':
        k = 1
    elif mode == 'pass_at_k':
        # For Pass@k, use k = N/2 or a reasonable k value
        # Can adjust this for different experiments
        k = max(1, N // 4)  # Use k=8 for N=32
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    return compute_pass_k_advantage(N, accuracy, k)


def compute_eta(accuracy: float, A_pos: float, A_neg: float, N: int = N_ROLLOUT) -> float:
    """
    Compute the Sum of Absolute Advantage (Equation 16).
    
    η = N_pos * |Â_pos| + N_neg * |Â_neg|
    
    Args:
        accuracy: Rollout accuracy (0.0 to 1.0)
        A_pos: Positive advantage value
        A_neg: Negative advantage value
        N: Number of rollouts
    
    Returns:
        η value
    """
    N_pos = N * accuracy
    N_neg = N * (1 - accuracy)
    
    return N_pos * np.abs(A_pos) + N_neg * np.abs(A_neg)


def plot_advantage_curves():
    """
    Create the advantage curve visualization (Figure 9).
    """
    # Accuracy range - use 200 points for smooth curves
    accuracies = np.linspace(0.01, 0.99, 200)  # Avoid exact 0 and 1
    
    # Colors (from paper specification)
    COLOR_A_POS = '#6A3D9A'   # Purple
    COLOR_A_NEG = '#A6CEE3'   # Light Blue
    COLOR_ETA = '#9E3645'     # Dark Red
    
    # Create figure with 1 row, 2 columns
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    modes = ['pass_at_1', 'pass_at_k']
    titles = ['(a) Standard GRPO (Pass@1) Training.', '(b) Pass@k Training.']
    
    for ax, mode, title in zip(axes, modes, titles):
        A_pos_values = []
        A_neg_values = []
        eta_values = []
        
        for acc in accuracies:
            A_pos, A_neg = get_advantage_values(acc, mode=mode)
            A_pos_values.append(A_pos)
            A_neg_values.append(A_neg)
            eta_values.append(compute_eta(acc, A_pos, A_neg))
        
        # Convert to numpy arrays
        A_pos_values = np.array(A_pos_values)
        A_neg_values = np.array(A_neg_values)
        eta_values = np.array(eta_values)
        
        # Convert accuracy to percentage
        acc_percent = accuracies * 100
        
        # Plot curves
        ax.plot(acc_percent, A_pos_values, color=COLOR_A_POS, linewidth=2, 
                label=r'$\hat{A}_{pos}$')
        ax.plot(acc_percent, A_neg_values, color=COLOR_A_NEG, linewidth=2, 
                label=r'$\hat{A}_{neg}$')
        ax.plot(acc_percent, eta_values, color=COLOR_ETA, linewidth=2, 
                label=r'$\eta$')
        
        # Styling
        ax.set_xlabel('Rollout Accuracy (%)', fontsize=12)
        ax.set_ylabel('Advantage', fontsize=12)
        ax.set_xlim(0, 100)
        ax.legend(loc='best', fontsize=10)
        
        # Y-axis grid only (dashed)
        ax.yaxis.grid(True, linestyle='--', alpha=0.7)
        ax.xaxis.grid(False)
        
        # Title at bottom center
        ax.text(0.5, -0.12, title, transform=ax.transAxes, fontsize=12,
                ha='center', va='top')
    
    # Main figure title at bottom
    fig.text(0.5, 0.01, 
             f'Figure 9: The curves of advantage function with setting of $N_{{rollout}} = {N_ROLLOUT}$.',
             ha='center', fontsize=12, style='italic')
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.18)
    
    # Save the figure
    output_path = './advantage_curves.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved figure to: {output_path}")
    
    plt.show()


def experiment_different_k_values():
    """
    Additional experiment: Compare different k values for Pass@k.
    """
    accuracies = np.linspace(0.01, 0.99, 200)  # Smoother curves
    k_values = [1, 2, 4, 8, 16]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left plot: A_pos for different k
    ax = axes[0]
    for k in k_values:
        A_pos_values = []
        for acc in accuracies:
            A_pos, _ = compute_pass_k_advantage(N_ROLLOUT, acc, k)
            A_pos_values.append(A_pos)
        ax.plot(accuracies * 100, A_pos_values, linewidth=2, label=f'k={k}')
    
    ax.set_xlabel('Rollout Accuracy (%)', fontsize=12)
    ax.set_ylabel(r'$\hat{A}_{pos}$', fontsize=12)
    ax.set_title(r'$\hat{A}_{pos}$ for different k values', fontsize=12)
    ax.legend()
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    
    # Right plot: η for different k
    ax = axes[1]
    for k in k_values:
        eta_values = []
        for acc in accuracies:
            A_pos, A_neg = compute_pass_k_advantage(N_ROLLOUT, acc, k)
            eta = compute_eta(acc, A_pos, A_neg)
            eta_values.append(eta)
        ax.plot(accuracies * 100, eta_values, linewidth=2, label=f'k={k}')
    
    ax.set_xlabel('Rollout Accuracy (%)', fontsize=12)
    ax.set_ylabel(r'$\eta$', fontsize=12)
    ax.set_title(r'Sum of Absolute Advantage $\eta$ for different k values', fontsize=12)
    ax.legend()
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    output_path = './advantage_curves_k_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved comparison figure to: {output_path}")
    
    plt.show()


# ==============================================================================
# Diversity Density Advantage (from compute_diversity_density_advantage)
# ==============================================================================

def _prob_not_in_group_vectorized(N: int, s_arr: np.ndarray, k: int) -> np.ndarray:
    """
    Vectorized: compute P(u_i not in G) for all answer types.
    
    P(u_i not in G) = C(N - s_i, k) / C(N, k)
    
    Args:
        N: Total number of samples
        s_arr: Array of counts for each answer type [s_1, s_2, ..., s_D]
        k: Group size
    
    Returns:
        Array of probabilities for each answer type
    """
    log_comb_N_k = _log_comb(np.array([N]), np.array([k]))[0]
    log_comb_N_minus_s_k = _log_comb(N - s_arr, np.full_like(s_arr, k))
    
    log_probs = log_comb_N_minus_s_k - log_comb_N_k
    probs = np.where(np.isfinite(log_probs), np.exp(log_probs), 0.0)
    
    return probs


def compute_diversity_density_advantage_single(
    N: int, 
    D: int, 
    s_arr: np.ndarray,
    k: int, 
    epsilon: float = 1e-6
) -> Tuple[np.ndarray, float, float]:
    """
    Compute diversity density advantage for a single prompt group.
    
    Based on compute_diversity_density_advantage in core_algos.py:
    - R = m/k where m is the number of unique answers in a random group of size k
    - μ = (1/k) * Σᵢ (1 - C(N-sᵢ,k)/C(N,k))  [global mean reward]
    - R(x) = (1/k) * [1 + Σⱼ≠ₜ (1 - C(N-1-sⱼ,k-1)/C(N-1,k-1))]  [conditional expected reward]
    - Â(x) = (R(x) - μ) / σ  [normalized advantage]
    
    Args:
        N: Total number of samples
        D: Number of unique answer types
        s_arr: Array of counts for each answer type
        k: Group size
        epsilon: Small value for numerical stability
    
    Returns:
        Tuple of (advantages array per type, global_mean, std)
    """
    k = min(k, N)
    
    if k <= 0 or N <= 0 or D == 0:
        return np.zeros(D), 0.0, 0.0
    
    # Step 1: Compute global mean μ
    # μ = (1/k) * Σᵢ (1 - P(uᵢ not in G))
    p_not_in_group = _prob_not_in_group_vectorized(N, s_arr, k)
    p_in_group = 1.0 - p_not_in_group
    global_mean = np.sum(p_in_group) / k
    
    # Step 2: Compute conditional expected reward R(x) for each answer type
    new_N = N - 1
    new_k = k - 1
    
    if new_k > 0:
        p_not_in_group_cond = _prob_not_in_group_vectorized(new_N, s_arr, new_k)
    else:
        p_not_in_group_cond = np.ones(D, dtype=np.float64)
    
    conditional_rewards = np.zeros(D, dtype=np.float64)
    
    for t_idx in range(D):
        if new_k > 0:
            sum_contrib = 1.0  # Type t is guaranteed to contribute 1
            for j_idx in range(D):
                if j_idx != t_idx:
                    sum_contrib += (1.0 - p_not_in_group_cond[j_idx])
            conditional_rewards[t_idx] = sum_contrib / k
        else:
            conditional_rewards[t_idx] = 1.0
    
    # Step 3: Compute variance for normalization
    variance = np.var(conditional_rewards)
    std = np.sqrt(variance + epsilon)
    
    # Step 4: Compute normalized advantage
    advantages = (conditional_rewards - global_mean) / std
    
    return advantages, global_mean, std


def simulate_diversity_scenario(accuracy: float, N: int, num_unique_answers: int = None, k: int = None) -> Tuple[float, float]:
    """
    Simulate diversity density advantage for a given accuracy.
    
    In diversity density, we care about the diversity of answers, not just correctness.
    For simulation, we model different diversity scenarios:
    - accuracy controls the dominance of the most common answer
    - Higher accuracy = more dominated by one answer = less diverse
    
    Args:
        accuracy: Accuracy/dominance level (0.0 to 1.0)
        N: Total number of samples
        num_unique_answers: Number of unique answer types (if None, derived from accuracy)
        k: Group size
    
    Returns:
        Tuple of (mean_advantage, std_advantage) over all samples
    """
    if k is None:
        k = max(1, N // 4)
    
    # Model diversity scenario:
    # At accuracy=1.0: all answers are the same (1 unique type)
    # At accuracy=0.0: all answers are different (N unique types)
    
    if accuracy >= 0.99:
        # All same answer
        D = 1
        s_arr = np.array([N], dtype=np.float64)
    elif accuracy <= 0.01:
        # All different answers (maximum diversity)
        D = N
        s_arr = np.ones(D, dtype=np.float64)
    else:
        # Mix: dominant answer + others
        # Dominant answer count
        dominant_count = int(N * accuracy)
        remaining = N - dominant_count
        
        if remaining > 0:
            # Spread remaining across unique answers
            # More diversity = more unique answers
            num_other_types = max(1, int(remaining * (1 - accuracy) * 2))
            num_other_types = min(num_other_types, remaining)
            
            s_arr = [dominant_count]
            # Distribute remaining among other types
            per_type = remaining // num_other_types
            remainder = remaining % num_other_types
            for i in range(num_other_types):
                count = per_type + (1 if i < remainder else 0)
                if count > 0:
                    s_arr.append(count)
            s_arr = np.array(s_arr, dtype=np.float64)
            D = len(s_arr)
        else:
            D = 1
            s_arr = np.array([N], dtype=np.float64)
    
    # Compute advantages
    advantages, global_mean, std = compute_diversity_density_advantage_single(N, D, s_arr, k)
    
    # Weight by counts to get overall statistics
    # Positive advantage: for dominant type
    # Negative advantage: for minority types
    a_dominant = advantages[0] if D > 0 else 0.0
    a_minority_mean = np.mean(advantages[1:]) if D > 1 else 0.0
    
    return a_dominant, a_minority_mean


def plot_diversity_density_advantage():
    """
    Visualize how diversity density advantage changes with answer diversity.
    
    This simulates the behavior of compute_diversity_density_advantage_from_prompts.
    """
    accuracies = np.linspace(0.05, 0.95, 200)  # More points for smooth curves
    
    # Colors
    COLOR_DOMINANT = '#6A3D9A'   # Purple - for High Frequency (High Count)
    COLOR_MINORITY = '#A6CEE3'   # Light Blue - for Low Frequency (Low Count)
    COLOR_ETA = '#9E3645'        # Dark Red - for eta (sum of absolute advantage)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    k_values = [1, 8]
    titles = [
        '(a) Diversity Density Advantage (k=1)',
        '(b) Diversity Density Advantage (k=8)'
    ]
    
    for ax, k, title in zip(axes, k_values, titles):
        dominant_adv = []
        minority_adv = []
        eta_values = []
        
        for acc in accuracies:
            a_dom, a_min = simulate_diversity_scenario(acc, N_ROLLOUT, k=k)
            dominant_adv.append(a_dom)
            minority_adv.append(a_min)
            
            # Compute η = N_pos * |Â_pos| + N_neg * |Â_neg| (Eq. 16)
            # For diversity: dominant count vs minority count
            N_pos = N_ROLLOUT * acc
            N_neg = N_ROLLOUT * (1 - acc)
            eta = N_pos * np.abs(a_dom) + N_neg * np.abs(a_min)
            eta_values.append(eta)
        
        dominant_adv = np.array(dominant_adv)
        minority_adv = np.array(minority_adv)
        eta_values = np.array(eta_values)
        
        # X-axis: "Answer Count" (Quantity)
        x_count = accuracies * N_ROLLOUT
        
        ax.plot(x_count, dominant_adv, color=COLOR_DOMINANT, linewidth=2,
                label=r'$\hat{A}$ (High Count $s_i$)', linestyle='-')
        ax.plot(x_count, minority_adv, color=COLOR_MINORITY, linewidth=2,
                label=r'$\hat{A}$ (Low Count $s_i \approx 1$)', linestyle='--')
        ax.plot(x_count, eta_values, color=COLOR_ETA, linewidth=2,
                label=r'$\eta$')
        
        # Add a reference line at y=0
        ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
        
        ax.set_xlabel(f'Count of Dominant Answer ($s_i$) / {N_ROLLOUT}', fontsize=12)
        ax.set_ylabel('Advantage', fontsize=12)
        ax.set_xlim(0, N_ROLLOUT)
        ax.legend(loc='best', fontsize=9)
        ax.yaxis.grid(True, linestyle='--', alpha=0.7)
        ax.xaxis.grid(True, alpha=0.3)
        
        ax.text(0.5, -0.12, title, transform=ax.transAxes, fontsize=12,
                ha='center', va='top')
    
    fig.text(0.5, 0.01,
             f'Diversity Density Advantage (compute_diversity_density_advantage_from_prompts), $N_{{rollout}} = {N_ROLLOUT}$',
             ha='center', fontsize=11, style='italic')
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.18)
    
    output_path = './diversity_density_advantage.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved diversity density figure to: {output_path}")
    
    plt.show()


def compare_pass_k_vs_diversity():
    """
    Compare Pass@k advantage vs Diversity Density advantage.
    """
    accuracies = np.linspace(0.05, 0.95, 200)  # Smoother curves
    k = 8
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Colors
    COLOR_PASS_K = '#E41A1C'    # Red
    COLOR_DIVERSITY = '#377EB8' # Blue
    
    # Left plot: Positive/Dominant advantage comparison
    ax = axes[0]
    
    pass_k_pos = []
    diversity_dom = []
    
    for acc in accuracies:
        a_pos, _ = compute_pass_k_advantage(N_ROLLOUT, acc, k)
        a_dom, _ = simulate_diversity_scenario(acc, N_ROLLOUT, k=k)
        pass_k_pos.append(a_pos)
        diversity_dom.append(a_dom)
    
    ax.plot(accuracies * 100, pass_k_pos, color=COLOR_PASS_K, linewidth=2,
            label=r'Pass@k $\hat{A}_{pos}$')
    ax.plot(accuracies * 100, diversity_dom, color=COLOR_DIVERSITY, linewidth=2,
            label=r'Diversity $\hat{A}_{dominant}$')
    
    ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Accuracy / Dominance (%)', fontsize=12)
    ax.set_ylabel('Advantage', fontsize=12)
    ax.set_title('Positive/Dominant Advantage Comparison', fontsize=12)
    ax.legend()
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    
    # Right plot: Negative/Minority advantage comparison
    ax = axes[1]
    
    pass_k_neg = []
    diversity_min = []
    
    for acc in accuracies:
        _, a_neg = compute_pass_k_advantage(N_ROLLOUT, acc, k)
        _, a_min = simulate_diversity_scenario(acc, N_ROLLOUT, k=k)
        pass_k_neg.append(a_neg)
        diversity_min.append(a_min)
    
    ax.plot(accuracies * 100, pass_k_neg, color=COLOR_PASS_K, linewidth=2,
            label=r'Pass@k $\hat{A}_{neg}$')
    ax.plot(accuracies * 100, diversity_min, color=COLOR_DIVERSITY, linewidth=2,
            label=r'Diversity $\hat{A}_{minority}$')
    
    ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Accuracy / Dominance (%)', fontsize=12)
    ax.set_ylabel('Advantage', fontsize=12)
    ax.set_title('Negative/Minority Advantage Comparison', fontsize=12)
    ax.legend()
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    output_path = './pass_k_vs_diversity_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved comparison figure to: {output_path}")
    
    plt.show()


if __name__ == '__main__':
    print("=" * 60)
    print("Advantage Function Visualization for DIV-TTRL")
    print(f"N_rollout = {N_ROLLOUT}")
    print("=" * 60)
    
    # Main visualization (Figure 9) - Pass@k only
    print("\n[1] Pass@k Advantage Visualization (Figure 9)")
    print("    Note: Pass@1 is mathematically equivalent to Standard GRPO for binary rewards.")
    plot_advantage_curves()
    
    # Experiment with different k values
    print("\n" + "=" * 60)
    print("[2] Experiment: Comparing different k values for Pass@k")
    print("=" * 60)
    experiment_different_k_values()
    
    # NEW: Diversity Density Advantage visualization
    print("\n" + "=" * 60)
    print("[3] Diversity Density Advantage Visualization")
    print("    (from compute_diversity_density_advantage_from_prompts)")
    print("=" * 60)
    plot_diversity_density_advantage()
    
    # Compare Pass@k vs Diversity Density
    print("\n" + "=" * 60)
    print("[4] Comparison: Pass@k vs Diversity Density")
    print("=" * 60)
    compare_pass_k_vs_diversity()

