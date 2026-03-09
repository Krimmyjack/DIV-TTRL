#!/usr/bin/env python3
"""
Analyze the response length distribution from a rollout JSONL file (e.g., qwen32.jsonl).

Generates:
  1. Histogram of per-response character lengths
  2. Histogram of per-response token lengths (whitespace-split approximation)
  3. Box plot comparing correct vs. incorrect response lengths
  4. Scatter: response length vs. sc_score per problem
  5. Summary statistics printed to console

Usage:
    python scripts/analyze_response_length.py --input verl/qwen32.jsonl
    python scripts/analyze_response_length.py --input verl/qwen32.jsonl --output_dir plots
"""
import argparse
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_jsonl(path):
    """Load all records from a JSONL file."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def compute_lengths(records):
    """
    Compute per-response lengths in characters and (whitespace-split) tokens.

    Returns:
        all_char_lens: list[int]  – character length of every response
        all_token_lens: list[int] – approximate token count (split on whitespace)
        per_problem: list[dict]   – per-problem aggregated info
    """
    all_char_lens = []
    all_token_lens = []
    per_problem = []

    for rec in records:
        responses = rec.get("responses", [])
        extracted = rec.get("extracted_answers", [])
        answer = rec.get("answer", "")
        sc_score = rec.get("sc_score", None)

        char_lens = [len(r) for r in responses]
        token_lens = [len(r.split()) for r in responses]

        # Simple correctness check: extracted answer matches ground truth literally
        correct_lens = []
        incorrect_lens = []
        for i, r in enumerate(responses):
            cl = char_lens[i]
            ea = extracted[i] if i < len(extracted) else "[NO_ANSWER]"
            if ea not in ("[NO_ANSWER]", "") and answer and ea.strip() == answer.strip():
                correct_lens.append(cl)
            else:
                incorrect_lens.append(cl)

        all_char_lens.extend(char_lens)
        all_token_lens.extend(token_lens)
        per_problem.append({
            "problem_idx": len(per_problem),
            "n_responses": len(responses),
            "char_lens": char_lens,
            "mean_char_len": np.mean(char_lens) if char_lens else 0,
            "median_char_len": np.median(char_lens) if char_lens else 0,
            "correct_char_lens": correct_lens,
            "incorrect_char_lens": incorrect_lens,
            "sc_score": sc_score,
        })

    return all_char_lens, all_token_lens, per_problem


def print_summary(all_char_lens, all_token_lens, per_problem):
    """Print summary statistics to console."""
    arr = np.array(all_char_lens)
    tarr = np.array(all_token_lens)
    print("=" * 60)
    print(f"Total problems       : {len(per_problem)}")
    print(f"Total responses      : {len(all_char_lens)}")
    print(f"Responses per problem: {len(all_char_lens) / max(len(per_problem), 1):.1f}")
    print("-" * 60)
    print(f"Character length  – mean: {arr.mean():.1f}  median: {np.median(arr):.1f}  "
          f"std: {arr.std():.1f}  min: {arr.min()}  max: {arr.max()}")
    print(f"Token length (ws) – mean: {tarr.mean():.1f}  median: {np.median(tarr):.1f}  "
          f"std: {tarr.std():.1f}  min: {tarr.min()}  max: {tarr.max()}")

    # Correct vs incorrect
    correct_all = []
    incorrect_all = []
    for p in per_problem:
        correct_all.extend(p["correct_char_lens"])
        incorrect_all.extend(p["incorrect_char_lens"])
    print("-" * 60)
    if correct_all:
        ca = np.array(correct_all)
        print(f"Correct responses   ({len(ca):>5d}) – mean: {ca.mean():.1f}  median: {np.median(ca):.1f}")
    if incorrect_all:
        ia = np.array(incorrect_all)
        print(f"Incorrect responses ({len(ia):>5d}) – mean: {ia.mean():.1f}  median: {np.median(ia):.1f}")
    print("=" * 60)


def plot_histograms(all_char_lens, all_token_lens, output_dir):
    """Plot character-length and token-length histograms."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(all_char_lens, bins=80, color="#4C72B0", edgecolor="white", alpha=0.85)
    axes[0].set_title("Response Length Distribution (Characters)", fontsize=13)
    axes[0].set_xlabel("Character Length")
    axes[0].set_ylabel("Count")
    axes[0].axvline(np.median(all_char_lens), color="red", linestyle="--", label=f"median={np.median(all_char_lens):.0f}")
    axes[0].axvline(np.mean(all_char_lens), color="orange", linestyle="--", label=f"mean={np.mean(all_char_lens):.0f}")
    axes[0].legend()

    axes[1].hist(all_token_lens, bins=80, color="#55A868", edgecolor="white", alpha=0.85)
    axes[1].set_title("Response Length Distribution (Whitespace Tokens)", fontsize=13)
    axes[1].set_xlabel("Token Count (whitespace split)")
    axes[1].set_ylabel("Count")
    axes[1].axvline(np.median(all_token_lens), color="red", linestyle="--", label=f"median={np.median(all_token_lens):.0f}")
    axes[1].axvline(np.mean(all_token_lens), color="orange", linestyle="--", label=f"mean={np.mean(all_token_lens):.0f}")
    axes[1].legend()

    plt.tight_layout()
    path = os.path.join(output_dir, "length_histogram.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[Saved] {path}")


def plot_correct_vs_incorrect(per_problem, output_dir):
    """Box plot comparing character lengths of correct vs. incorrect responses."""
    correct_all = []
    incorrect_all = []
    for p in per_problem:
        correct_all.extend(p["correct_char_lens"])
        incorrect_all.extend(p["incorrect_char_lens"])

    if not correct_all and not incorrect_all:
        print("[Skip] No correct/incorrect data for box plot.")
        return

    data = []
    labels = []
    if correct_all:
        data.append(correct_all)
        labels.append(f"Correct\n(n={len(correct_all)})")
    if incorrect_all:
        data.append(incorrect_all)
        labels.append(f"Incorrect\n(n={len(incorrect_all)})")

    fig, ax = plt.subplots(figsize=(7, 5))
    bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.5)
    colors = ["#55A868", "#C44E52"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_title("Response Length: Correct vs. Incorrect", fontsize=13)
    ax.set_ylabel("Character Length")
    plt.tight_layout()
    path = os.path.join(output_dir, "correct_vs_incorrect_boxplot.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[Saved] {path}")


def plot_length_vs_score(per_problem, output_dir):
    """Scatter plot: mean response length per problem vs. sc_score."""
    xs, ys = [], []
    for p in per_problem:
        if p["sc_score"] is not None:
            xs.append(p["mean_char_len"])
            ys.append(p["sc_score"])

    if not xs:
        print("[Skip] No sc_score data for scatter plot.")
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(xs, ys, alpha=0.4, s=20, color="#4C72B0")
    ax.set_xlabel("Mean Response Length (chars) per Problem")
    ax.set_ylabel("SC Score (consistency)")
    ax.set_title("Mean Response Length vs. Self-Consistency Score", fontsize=13)
    plt.tight_layout()
    path = os.path.join(output_dir, "length_vs_sc_score.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[Saved] {path}")


def plot_per_problem_lengths(per_problem, output_dir):
    """Plot per-problem mean and median response lengths."""
    indices = [p["problem_idx"] for p in per_problem]
    means = [p["mean_char_len"] for p in per_problem]
    medians = [p["median_char_len"] for p in per_problem]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(indices, means, color="#4C72B0", alpha=0.6, label="Mean")
    ax.plot(indices, medians, color="red", linewidth=1, alpha=0.8, label="Median")
    ax.set_xlabel("Problem Index")
    ax.set_ylabel("Character Length")
    ax.set_title("Per-Problem Response Length (Mean & Median)", fontsize=13)
    ax.legend()
    plt.tight_layout()
    path = os.path.join(output_dir, "per_problem_length.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[Saved] {path}")


def plot_length_by_consistency_bins(per_problem, output_dir):
    """
    Bin problems by sc_score into 10 intervals of width 0.1:
        [0, 0.1), [0.1, 0.2), ..., [0.9, 1.0]
    For each bin, collect ALL individual response char lengths from the
    problems that fall into that bin and plot a histogram.
    """
    bin_edges = np.arange(0, 1.1, 0.1)  # 0.0, 0.1, ..., 1.0
    n_bins = len(bin_edges) - 1          # 10 bins

    # Collect response lengths per bin
    binned_lens = [[] for _ in range(n_bins)]
    for p in per_problem:
        sc = p["sc_score"]
        if sc is None:
            continue
        # Determine which bin this problem falls into
        idx = int(sc / 0.1)
        # sc == 1.0 should go into the last bin [0.9, 1.0]
        if idx >= n_bins:
            idx = n_bins - 1
        binned_lens[idx].extend(p["char_lens"])

    # --- Plot: 2 rows x 5 cols ---
    fig, axes = plt.subplots(2, 5, figsize=(24, 8), sharey=False)
    axes = axes.flatten()

    colors = plt.cm.viridis(np.linspace(0.15, 0.85, n_bins))

    global_max_len = max((max(b) if b else 0) for b in binned_lens)
    x_upper = min(global_max_len + 100, max(global_max_len, 5000))

    for i in range(n_bins):
        ax = axes[i]
        lo = bin_edges[i]
        hi = bin_edges[i + 1]
        data = binned_lens[i]
        label = f"[{lo:.1f}, {hi:.1f}{']' if i == n_bins - 1 else ')'}"

        if data:
            ax.hist(data, bins=40, color=colors[i], edgecolor="white", alpha=0.85)
            mean_v = np.mean(data)
            med_v = np.median(data)
            ax.axvline(med_v, color="red", linestyle="--", linewidth=1,
                       label=f"med={med_v:.0f}")
            ax.axvline(mean_v, color="orange", linestyle="--", linewidth=1,
                       label=f"mean={mean_v:.0f}")
            ax.legend(fontsize=8)
        else:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=11, color="gray")

        ax.set_title(f"SC {label}  (n={len(data)})", fontsize=10)
        ax.set_xlabel("Char Length", fontsize=9)
        ax.set_ylabel("Count", fontsize=9)
        ax.set_xlim(0, x_upper)

    fig.suptitle("Response Length Distribution by Consistency Ratio Bins",
                 fontsize=14, y=1.01)
    plt.tight_layout()
    path = os.path.join(output_dir, "length_by_consistency_bins.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Saved] {path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze response length distribution from rollout JSONL.")
    parser.add_argument("--input", type=str, required=True, help="Path to the JSONL file (e.g., qwen32.jsonl)")
    parser.add_argument("--output_dir", type=str, default="plots", help="Directory to save plots (default: plots)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading {args.input} ...")
    records = load_jsonl(args.input)
    print(f"Loaded {len(records)} problems.")

    all_char_lens, all_token_lens, per_problem = compute_lengths(records)

    print_summary(all_char_lens, all_token_lens, per_problem)

    plot_histograms(all_char_lens, all_token_lens, args.output_dir)
    plot_correct_vs_incorrect(per_problem, args.output_dir)
    plot_length_vs_score(per_problem, args.output_dir)
    plot_per_problem_lengths(per_problem, args.output_dir)
    plot_length_by_consistency_bins(per_problem, args.output_dir)

    print(f"\nAll plots saved to: {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()
