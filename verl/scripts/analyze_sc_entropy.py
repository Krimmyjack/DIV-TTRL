#!/usr/bin/env python3
"""
Analyze rollout data by sc_score bins: response length histograms & length-vs-entropy scatter plots.

For each of 10 sc_score bins ([0, 0.1), [0.1, 0.2), ..., [0.9, 1.0]):
  1. Histogram of response_length (token count from response_metrics)
  2. Scatter plot: X = response_length, Y = token_entropy_approx, colored by correctness

Usage:
    python scripts/analyze_sc_entropy.py --input qwen4b.jsonl --output_dir plots_qwen4b
"""
import argparse
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── Data loading ────────────────────────────────────────────────────────────

def load_jsonl(path):
    """Load all records from a JSONL file."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def prepare_response_data(records):
    """
    Flatten records into a list of per-response dicts with keys:
        sc_score, response_length, token_entropy_approx, is_correct
    """
    data = []
    for rec in records:
        sc_score = rec.get("sc_score", None)
        if sc_score is None:
            continue

        answer = rec.get("answer", "")
        extracted = rec.get("extracted_answers", [])
        metrics = rec.get("response_metrics", [])

    # Also aggregate per problem average entropy, lengths, and correct ratio
    per_problem = []
    
    # We will also add problem_idx to data entries for the new per-problem scatter plot
    problem_idx = 0
    
    for rec in records:
        sc_score = rec.get("sc_score", None)
        if sc_score is None:
            continue

        answer = rec.get("answer", "")
        extracted = rec.get("extracted_answers", [])
        metrics = rec.get("response_metrics", [])
        
        avg_entropies = [m.get("token_entropy_approx") for m in metrics if m.get("token_entropy_approx") is not None]
        lengths = [m.get("response_length") for m in metrics if m.get("response_length") is not None]
        
        # Calculate correct ratio for this problem
        correct_count = 0
        total_count = len(metrics)
        
        correct_entropies = []
        incorrect_entropies = []
        
        for i, m in enumerate(metrics):
            resp_len = m.get("response_length")
            entropy = m.get("token_entropy_approx")
            
            ea = extracted[i] if i < len(extracted) else "[NO_ANSWER]"
            is_correct = (
                ea not in ("[NO_ANSWER]", "")
                and answer
                and ea.strip() == str(answer).strip()
            )
            
            if resp_len is None:
                continue  # skip if no length info

            data.append({
                "problem_idx": problem_idx,
                "sc_score": sc_score,
                "response_length": resp_len,
                "token_entropy_approx": entropy,  # may be None
                "is_correct": is_correct,
            })
            
            if entropy is not None:
                if is_correct:
                    correct_entropies.append(entropy)
                else:
                    incorrect_entropies.append(entropy)
            
            if is_correct:
                correct_count += 1
                
        correct_ratio = correct_count / total_count if total_count > 0 else 0.0

        if avg_entropies or lengths:
            per_problem.append({
                "problem_idx": problem_idx,
                "sc_score": sc_score,
                "mean_token_entropy": np.mean(avg_entropies) if avg_entropies else None,
                "mean_correct_entropy": np.mean(correct_entropies) if correct_entropies else None,
                "mean_incorrect_entropy": np.mean(incorrect_entropies) if incorrect_entropies else None,
                "mean_token_length": np.mean(lengths) if lengths else None,
                "correct_ratio": correct_ratio,
                "is_majority_correct": correct_ratio > 0.5  # Just an extra metric
            })
            
        problem_idx += 1

    return data, per_problem

# ── Binning helper ──────────────────────────────────────────────────────────

def _bin_index(sc_score, n_bins=10):
    """Return bin index (0..n_bins-1) for a given sc_score in [0, 1]."""
    idx = int(sc_score / 0.1)
    return min(idx, n_bins - 1)


def _bin_label(i, n_bins=10):
    lo = i * 0.1
    hi = (i + 1) * 0.1
    right = "]" if i == n_bins - 1 else ")"
    return f"[{lo:.1f}, {hi:.1f}{right}"


# ── Plot 1: length histograms per sc_score bin ─────────────────────────────

def plot_length_histograms_by_bin(data, output_dir):
    """2×5 sub-plots, one histogram of response_length per sc_score bin."""
    n_bins = 10
    binned_lens = [[] for _ in range(n_bins)]
    for d in data:
        binned_lens[_bin_index(d["sc_score"])].append(d["response_length"])

    fig, axes = plt.subplots(2, 5, figsize=(24, 8))
    axes = axes.flatten()
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, n_bins))

    # Shared x-axis upper bound
    global_max = max((max(b) if b else 0) for b in binned_lens)
    x_upper = min(global_max + 100, max(global_max, 5000))

    for i in range(n_bins):
        ax = axes[i]
        lens = binned_lens[i]
        label = _bin_label(i)

        if lens:
            ax.hist(lens, bins=40, color=colors[i], edgecolor="white", alpha=0.85)
            mean_v = np.mean(lens)
            med_v = np.median(lens)
            ax.axvline(med_v, color="red", linestyle="--", linewidth=1,
                       label=f"med={med_v:.0f}")
            ax.axvline(mean_v, color="orange", linestyle="--", linewidth=1,
                       label=f"mean={mean_v:.0f}")
            ax.legend(fontsize=8)
        else:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=11, color="gray")

        ax.set_title(f"SC {label}  (n={len(lens)})", fontsize=10)
        ax.set_xlabel("Token Length", fontsize=9)
        ax.set_ylabel("Count", fontsize=9)
        ax.set_xlim(0, x_upper)

    fig.suptitle("Response Token-Length Distribution by SC Score Bins",
                 fontsize=14, y=1.01)
    plt.tight_layout()
    path = os.path.join(output_dir, "length_hist_by_sc_bin.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Saved] {path}")


# ── Plot 2: scatter (length vs entropy) per sc_score bin ───────────────────

def plot_scatter_by_bin(data, output_dir):
    """
    2×5 sub-plots scatter: X = response_length, Y = token_entropy_approx.
    Green = correct, Red = incorrect.  Responses with entropy=None are skipped.
    """
    n_bins = 10
    # Separate correct / incorrect per bin
    correct_per_bin = [[] for _ in range(n_bins)]
    incorrect_per_bin = [[] for _ in range(n_bins)]

    for d in data:
        if d["token_entropy_approx"] is None:
            continue
        idx = _bin_index(d["sc_score"])
        point = (d["response_length"], d["token_entropy_approx"])
        if d["is_correct"]:
            correct_per_bin[idx].append(point)
        else:
            incorrect_per_bin[idx].append(point)

    fig, axes = plt.subplots(2, 5, figsize=(24, 8))
    axes = axes.flatten()

    for i in range(n_bins):
        ax = axes[i]
        label = _bin_label(i)

        cor = correct_per_bin[i]
        inc = incorrect_per_bin[i]

        if inc:
            xs, ys = zip(*inc)
            ax.scatter(xs, ys, alpha=0.35, s=12, color="#C44E52",
                       label=f"Wrong ({len(inc)})", zorder=2)
        if cor:
            xs, ys = zip(*cor)
            ax.scatter(xs, ys, alpha=0.35, s=12, color="#55A868",
                       label=f"Correct ({len(cor)})", zorder=3)

        if not cor and not inc:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=11, color="gray")

        ax.set_title(f"SC {label}  (n={len(cor)+len(inc)})", fontsize=10)
        ax.set_xlabel("Token Length", fontsize=9)
        ax.set_ylabel("Entropy (≈ −mean logprob)", fontsize=9)
        ax.legend(fontsize=7, loc="upper right")

    fig.suptitle("Response Length vs. Entropy by SC Score Bins (Green=Correct, Red=Wrong)",
                 fontsize=14, y=1.01)
    plt.tight_layout()
    path = os.path.join(output_dir, "scatter_length_entropy_by_sc_bin.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Saved] {path}")


# ── Plot 3: scatter (mean entropy vs sc_score) ──────────────────────────────

def plot_entropy_vs_sc_score(per_problem, output_dir):
    """
    Scatter plot: X = mean token entropy per problem, Y = sc_score.
    """
    xs = []
    ys = []
    for p in per_problem:
        xs.append(p["mean_token_entropy"])
        ys.append(p["sc_score"])

    if not xs:
        print("[Skip] No data for entropy vs sc_score.")
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(xs, ys, alpha=0.4, s=20, color="#8172B3")
    
    # Optional: Calculate correlation
    if len(xs) > 1:
        corr = np.corrcoef(xs, ys)[0, 1]
        ax.set_title(f"Mean Token Entropy vs. SC Score (r={corr:.2f})", fontsize=13)
    else:
        ax.set_title("Mean Token Entropy vs. SC Score", fontsize=13)
        
    ax.set_xlabel("Mean Token Entropy (per problem)")
    ax.set_ylabel("SC Score (consistency)")
    plt.tight_layout()
    path = os.path.join(output_dir, "entropy_vs_sc_score.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[Saved] {path}")



# ── Plot 4: sc_score vs accuracy ────────────────────────────────────────────

def plot_sc_score_vs_correctness(per_problem, output_dir):
    """
    Scatter plot: X = sc_score, Y = correct_ratio per problem.
    """
    xs = []
    ys = []
    for p in per_problem:
        xs.append(p["sc_score"])
        ys.append(p["correct_ratio"])

    if not xs:
        print("[Skip] No data for sc_score vs correctness.")
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(xs, ys, alpha=0.4, s=20, color="#4C72B0")
    
    if len(xs) > 1:
        corr = np.corrcoef(xs, ys)[0, 1]
        ax.set_title(f"SC Score vs. Correct Ratio (r={corr:.2f})", fontsize=13)
    else:
        ax.set_title("SC Score vs. Correct Ratio", fontsize=13)
        
    ax.set_xlabel("SC Score (consistency)")
    ax.set_ylabel("Correct Ratio per Problem")
    plt.tight_layout()
    path = os.path.join(output_dir, "sc_score_vs_correctness.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[Saved] {path}")


# ── Plot 5: mean entropy vs accuracy ────────────────────────────────────────

def plot_entropy_vs_correctness(per_problem, output_dir):
    """
    Scatter plot: X = mean token entropy, Y = correct_ratio per problem.
    """
    xs = []
    ys = []
    for p in per_problem:
        xs.append(p["mean_token_entropy"])
        ys.append(p["correct_ratio"])

    if not xs:
        print("[Skip] No data for entropy vs correctness.")
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(xs, ys, alpha=0.4, s=20, color="#C44E52")
    
    if len(xs) > 1:
        corr = np.corrcoef(xs, ys)[0, 1]
        ax.set_title(f"Mean Token Entropy vs. Correct Ratio (r={corr:.2f})", fontsize=13)
    else:
        ax.set_title("Mean Token Entropy vs. Correct Ratio", fontsize=13)
        
    ax.set_xlabel("Mean Token Entropy (per problem)")
    ax.set_ylabel("Correct Ratio per Problem")
    plt.tight_layout()
    path = os.path.join(output_dir, "entropy_vs_correctness.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[Saved] {path}")


# ── Plot 6: mean length vs sc_score ────────────────────────────────────────

def plot_length_vs_sc_score(per_problem, output_dir):
    """
    Scatter plot: X = mean token length per problem, Y = sc_score.
    """
    xs = []
    ys = []
    for p in per_problem:
        if p["mean_token_length"] is not None:
            xs.append(p["mean_token_length"])
            ys.append(p["sc_score"])

    if not xs:
        print("[Skip] No data for length vs sc_score.")
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(xs, ys, alpha=0.4, s=20, color="#55A868")
    
    if len(xs) > 1:
        corr = np.corrcoef(xs, ys)[0, 1]
        ax.set_title(f"Mean Token Length vs. SC Score (r={corr:.2f})", fontsize=13)
    else:
        ax.set_title("Mean Token Length vs. SC Score", fontsize=13)
        
    ax.set_xlabel("Mean Token Length (per problem)")
    ax.set_ylabel("SC Score (consistency)")
    plt.tight_layout()
    path = os.path.join(output_dir, "length_vs_sc_score.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[Saved] {path}")



# ── Plot 7: mean length vs accuracy ──────────────────────────────────────────

def plot_length_vs_correctness(per_problem, output_dir):
    """
    Scatter plot: X = mean token length per problem, Y = correct_ratio per problem.
    """
    xs = []
    ys = []
    for p in per_problem:
        if p["mean_token_length"] is not None:
            xs.append(p["mean_token_length"])
            ys.append(p["correct_ratio"])

    if not xs:
        print("[Skip] No data for length vs correctness.")
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(xs, ys, alpha=0.4, s=20, color="#8172B3")
    
    if len(xs) > 1:
        corr = np.corrcoef(xs, ys)[0, 1]
        ax.set_title(f"Mean Token Length vs. Correct Ratio (r={corr:.2f})", fontsize=13)
    else:
        ax.set_title("Mean Token Length vs. Correct Ratio", fontsize=13)
        
    ax.set_xlabel("Mean Token Length (per problem)")
    ax.set_ylabel("Correct Ratio per Problem")
    plt.tight_layout()
    path = os.path.join(output_dir, "length_vs_correctness.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[Saved] {path}")




# ── Plot 8: Token Entropy Distributions (Correct vs Incorrect) ─────────────

def plot_entropy_distributions(data, output_dir):
    """
    Overlapping histograms (or density) to show the overall distribution of 
    token entropy for Correct vs Incorrect answers globally.
    """
    correct_y = [d["token_entropy_approx"] for d in data if d["is_correct"] and d["token_entropy_approx"] is not None]
    incorrect_y = [d["token_entropy_approx"] for d in data if not d["is_correct"] and d["token_entropy_approx"] is not None]

    if not correct_y and not incorrect_y:
        print("[Skip] No data for entropy histograms.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Histogram overlap
    ax = axes[0]
    if incorrect_y:
        ax.hist(incorrect_y, bins=50, alpha=0.5, color="#C44E52", label=f"Wrong (n={len(incorrect_y)})", density=True)
    if correct_y:
        ax.hist(correct_y, bins=50, alpha=0.5, color="#55A868", label=f"Correct (n={len(correct_y)})", density=True)
        
    ax.set_title("Global Token Entropy Distribution", fontsize=13)
    ax.set_xlabel("Token Entropy (≈ −mean logprob)")
    ax.set_ylabel("Density")
    ax.legend()
    
    # Right: Boxplot comparing the two
    ax = axes[1]
    box_data = []
    box_labels = []
    colors = []
    
    if correct_y:
        box_data.append(correct_y)
        box_labels.append(f"Correct\n(n={len(correct_y)})")
        colors.append("#55A868")
    if incorrect_y:
        box_data.append(incorrect_y)
        box_labels.append(f"Wrong\n(n={len(incorrect_y)})")
        colors.append("#C44E52")
        
    if box_data:
        bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
            
    ax.set_title("Boxplot: Token Entropy (Correct vs Wrong)", fontsize=13)
    ax.set_ylabel("Token Entropy (approx)")

    plt.tight_layout()
    path = os.path.join(output_dir, "entropy_distribution_global.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[Saved] {path}")


# ── Plot 9: Per-problem Mean Entropy Sorted ─────────────────────────────────

def plot_entropy_per_problem_sorted(per_problem, output_dir):
    """
    Instead of plotting 32,000 points across 500 problems arbitrarily,
    we sort the problems by accuracy, and plot the MEAN correct entropy 
    vs MEAN incorrect entropy for each problem.
    """
    # Sort by correct_ratio ascending (difficult to easy)
    sorted_probs = sorted(per_problem, key=lambda x: x["correct_ratio"])
    
    xs = list(range(len(sorted_probs)))
    cor_means = [p["mean_correct_entropy"] for p in sorted_probs]
    inc_means = [p["mean_incorrect_entropy"] for p in sorted_probs]
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Plot incorrect means
    ax.scatter(xs, inc_means, color="#C44E52", alpha=0.6, s=15, label="Mean Incorrect Entropy")
    # Plot correct means
    ax.scatter(xs, cor_means, color="#55A868", alpha=0.6, s=15, label="Mean Correct Entropy")
    
    # Add a moving average to see the trend clearer
    window = 10
    if len(xs) > window:
        cor_valid = [(x, y) for x, y in zip(xs, cor_means) if y is not None]
        inc_valid = [(x, y) for x, y in zip(xs, inc_means) if y is not None]
        
        if cor_valid:
            cx, cy = zip(*cor_valid)
            smooth_cor = np.convolve(cy, np.ones(window)/window, mode='valid')
            ax.plot(cx[window-1:], smooth_cor, color="darkgreen", linewidth=2)
            
        if inc_valid:
            ix, iy = zip(*inc_valid)
            smooth_inc = np.convolve(iy, np.ones(window)/window, mode='valid')
            ax.plot(ix[window-1:], smooth_inc, color="darkred", linewidth=2)
            
    ax.set_title("Mean Token Entropy per Problem (Sorted by Problem Accuracy)", fontsize=14)
    ax.set_xlabel("Problem Index (Sorted correctly: Hard → Easy)")
    ax.set_ylabel("Mean Token Entropy")
    ax.legend()
    
    plt.tight_layout()
    path = os.path.join(output_dir, "entropy_per_problem_sorted.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[Saved] {path}")



# ── Plot 10: Zero-Accuracy Problems Analysis (SC Score & Length) ────────────

def plot_zero_accuracy_analysis(per_problem, output_dir):
    """
    Isolate problems with 0% accuracy (model never got it right),
    and analyze their SC Ratio (consistency) and Response Lengths.
    """
    zero_acc = [p for p in per_problem if p["correct_ratio"] == 0]
    
    if not zero_acc:
        print("[Skip] No problems with 0% accuracy found.")
        return
        
    sc_scores = [p["sc_score"] for p in zero_acc]
    lengths = [p["mean_token_length"] for p in zero_acc if p["mean_token_length"] is not None]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: SC Score Histogram for 0-accuracy problems
    ax = axes[0]
    ax.hist(sc_scores, bins=20, color="#C44E52", edgecolor="white", alpha=0.85)
    ax.set_title(f"SC Score Dist. for 0% Accuracy Problems (n={len(zero_acc)})", fontsize=12)
    ax.set_xlabel("SC Score (consistency)")
    ax.set_ylabel("Number of Problems")
    if sc_scores:
        ax.axvline(np.mean(sc_scores), color="orange", linestyle="--", label=f"mean={np.mean(sc_scores):.2f}")
        ax.legend()
        
    # Right: Length Histogram for 0-accuracy problems
    ax = axes[1]
    if lengths:
        ax.hist(lengths, bins=20, color="#8172B3", edgecolor="white", alpha=0.85)
        ax.set_title("Mean Token Length Dist. for 0% Accuracy Problems", fontsize=12)
        ax.set_xlabel("Mean Token Length")
        ax.set_ylabel("Number of Problems")
        ax.axvline(np.mean(lengths), color="orange", linestyle="--", label=f"mean={np.mean(lengths):.0f}")
        ax.legend()
        
    plt.tight_layout()
    path = os.path.join(output_dir, "zero_accuracy_analysis.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[Saved] {path}")


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Analyze rollout data: per-sc-bin length histograms & length-vs-entropy scatter plots."
    )
    parser.add_argument("--input", type=str, required=True,
                        help="Path to the JSONL file (e.g., qwen4b.jsonl)")
    parser.add_argument("--output_dir", type=str, default="plots",
                        help="Directory to save plots (default: plots)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading {args.input} ...")
    records = load_jsonl(args.input)
    print(f"Loaded {len(records)} problems.")

    data, per_problem = prepare_response_data(records)
    print(f"Prepared {len(data)} individual response entries.")

    plot_length_histograms_by_bin(data, args.output_dir)
    plot_scatter_by_bin(data, args.output_dir)
    plot_entropy_vs_sc_score(per_problem, args.output_dir)
    plot_sc_score_vs_correctness(per_problem, args.output_dir)
    plot_entropy_vs_correctness(per_problem, args.output_dir)
    plot_length_vs_sc_score(per_problem, args.output_dir)
    plot_length_vs_correctness(per_problem, args.output_dir)
    plot_entropy_distributions(data, args.output_dir)
    plot_entropy_per_problem_sorted(per_problem, args.output_dir)
    plot_zero_accuracy_analysis(per_problem, args.output_dir)

    print(f"\nAll plots saved to: {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()
