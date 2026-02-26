"""
Bootstrap Pseudo-label Experiments 1-4 (Enhanced Aesthetics & Pass@k Analytics)

Usage:
    python bootstrap_experiments.py --input_file qwen64.jsonl --output_dir bootstrap_results --num_bootstrap 64
"""

import json
import argparse
import random
import os
import math
import numpy as np
from collections import Counter

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[WARN] matplotlib not found, skipping plots")

try:
    from sklearn.metrics import roc_auc_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# =====================================================
# Plot Style & Color Palette
# =====================================================
def set_plot_style():
    if not HAS_MPL:
        return
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams['grid.linestyle'] = '--'
    plt.rcParams['axes.axisbelow'] = True
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['figure.dpi'] = 150

COLORS = {
    "blue": "#4C72B0", "orange": "#DD8452", "green": "#55A868",
    "red": "#C44E52", "purple": "#8172B3", "gray": "#8C8C8C",
    "light_blue": "#93B5C6", "dark_blue": "#2C3E50",
}

def add_value_labels(ax, format_str="{:.2f}", space=0.01):
    for rect in ax.patches:
        height = rect.get_height()
        if height == 0:
            continue
        y_pos = height + space if height > 0 else height - space - 0.05
        va = 'bottom' if height > 0 else 'top'
        ax.text(rect.get_x() + rect.get_width() / 2., y_pos,
                format_str.format(height),
                ha='center', va=va, fontsize=8, color='#333333')


# =====================================================
# Helper Functions
# =====================================================
def strip_string(string):
    if string is None:
        return ""
    string = string.replace("\n", "").replace("\\!", "").replace("\\\\", "\\")
    string = string.replace("tfrac", "frac").replace("dfrac", "frac")
    string = string.replace("\\left", "").replace("\\right", "")
    string = string.replace("^{\\circ}", "").replace("^\\circ", "")
    string = string.replace("\\$", "").replace(" ", "")
    if string == "0.5":
        string = "\\frac{1}{2}"
    return string


def nCr(n, r):
    if r < 0 or r > n:
        return 0
    return math.comb(n, r)


def grpo_advantage(rewards, epsilon=1e-6):
    r = np.array(rewards, dtype=np.float64)
    std = r.std()
    if std < epsilon:
        return np.zeros_like(r)
    return (r - r.mean()) / (std + epsilon)


def passk_advantage(binary_rewards, k=4, epsilon=1e-6):
    r = np.array(binary_rewards, dtype=np.float64)
    N = len(r)
    n_pos = int(r.sum())
    n_neg = N - n_pos

    if n_pos == 0 or n_pos == N:
        return np.zeros_like(r)

    prob_fail_k = nCr(n_neg, k) / nCr(N, k)
    mean_r = 1.0 - prob_fail_k

    if mean_r <= 0 or mean_r >= 1:
        return np.zeros_like(r)

    std_r = math.sqrt(mean_r * (1.0 - mean_r))
    a_pos = (1.0 - mean_r) / (std_r + epsilon)
    term = nCr(n_neg - 1, k - 1) / nCr(N - 1, k - 1) if n_neg > 0 else 0.0
    a_neg = (1.0 - mean_r - term) / (std_r + epsilon)

    adv = np.zeros_like(r)
    adv[r > 0.5] = a_pos
    adv[r < 0.5] = a_neg
    return adv


def cosine_sim(v1, v2):
    v1, v2 = np.array(v1), np.array(v2)
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    return float(np.dot(v1, v2) / norm) if norm > 1e-8 else 0.0


def sign_align(v1, v2):
    v1, v2 = np.array(v1), np.array(v2)
    mask = np.abs(v2) > 1e-8
    if mask.sum() == 0:
        return 0.0
    return float(np.mean(np.sign(v1[mask]) == np.sign(v2[mask])))


def bootstrap_analysis(answers, gt, B=1000, sample_size=None):
    valid = [a for a in answers if a != "[NO_ANSWER]"]
    N_valid = len(valid)
    if N_valid == 0:
        return None
    if sample_size is None:
        sample_size = N_valid

    freq = Counter(valid)
    original_majority = freq.most_common(1)[0][0]
    gt_norm = strip_string(str(gt))

    boot_majorities = [
        Counter(random.choices(valid, k=sample_size)).most_common(1)[0][0]
        for _ in range(B)
    ]
    boot_counter = Counter(boot_majorities)
    answer_conf = {ans: c / B for ans, c in boot_counter.items()}

    return {
        "N": N_valid,
        "answers": valid,
        "freq": freq,
        "n_unique_answers": len(freq),
        "original_majority": original_majority,
        "consistency": freq[original_majority] / N_valid,
        "gt": gt_norm,
        "maj_correct": (original_majority == gt_norm),
        "answer_conf": answer_conf,
        "boot_stability": answer_conf.get(original_majority, 0.0),
        "gt_boot_conf": answer_conf.get(gt_norm, 0.0),
        "gt_covered": gt_norm in answer_conf,
        "n_distinct_maj": len(boot_counter),
        "soft_rewards": [answer_conf.get(a, 0.0) for a in valid],
        "binary_rewards": [1.0 if a == original_majority else 0.0 for a in valid],
        "gt_rewards": [1.0 if a == gt_norm else 0.0 for a in valid],
    }


# =====================================================
# Consistency bucket definition (shared)
# =====================================================
CONS_BUCKETS = [
    ("Low(<=0.3)", lambda r: r["consistency"] <= 0.3),
    ("Mid(0.3-0.7)", lambda r: 0.3 < r["consistency"] <= 0.7),
    ("High(>0.7)", lambda r: r["consistency"] > 0.7),
]
CONS_BUCKETS_ALL = CONS_BUCKETS + [("All", lambda r: True)]


# =====================================================
# Experiment 1: Motivation Visualization
# =====================================================
def experiment_1(results, output_dir, B):
    print("\n" + "=" * 80)
    print("Exp 1: Distribution Robustness (Motivation)")
    print("=" * 80)

    low_cons = [r for r in results if r["consistency"] <= 0.3]
    print(f"  Low consistency problems: {len(low_cons)}")

    # Select 6 representative problems
    selected = []
    cat1 = sorted([r for r in low_cons if not r["maj_correct"] and r["gt_boot_conf"] > 0.1],
                  key=lambda x: -x["gt_boot_conf"])[:2]
    cat2 = sorted([r for r in low_cons if not r["maj_correct"] and r["gt_boot_conf"] <= 0.1],
                  key=lambda x: x["boot_stability"])[:2]
    cat3 = sorted([r for r in low_cons if r["maj_correct"] and r["boot_stability"] < 0.7],
                  key=lambda x: x["boot_stability"])[:2]
    selected = cat1 + cat2 + cat3

    labels = ["MAJ_wrong+GT_high"] * 2 + ["MAJ_wrong+GT_low"] * 2 + ["MAJ_right+unstable"] * 2
    for i, r in enumerate(selected):
        tag = labels[i] if i < len(labels) else "?"
        print(f"\n  [{tag}] consistency={r['consistency']:.2f}, "
              f"maj_correct={r['maj_correct']}, stability={r['boot_stability']:.3f}")
        print(f"    Original majority: '{r['original_majority']}'")
        print(f"    Ground truth:      '{r['gt']}'")
        for ans, conf in sorted(r["answer_conf"].items(), key=lambda x: -x[1])[:5]:
            marks = []
            if ans == r["gt"]:
                marks.append("GT")
            if ans == r["original_majority"]:
                marks.append("MAJ")
            mark = f" <- {'+'.join(marks)}" if marks else ""
            print(f"      P_boot('{ans}') = {conf:.3f}{mark}")

    # Plot
    if HAS_MPL and selected:
        fig, axes = plt.subplots(2, 3, figsize=(16, 9))
        fig.suptitle("Bootstrap Majority Distribution vs Original Hard Voting",
                     fontsize=16, fontweight='bold', y=0.98)

        for idx, (ax, r) in enumerate(zip(axes.flat, selected[:6])):
            sorted_ans = sorted(r["answer_conf"].items(), key=lambda x: -x[1])[:6]
            ans_labels, colors, confs = [], [], []
            for ans, conf in sorted_ans:
                ans_labels.append((ans[:12] + "..") if len(ans) > 12 else ans)
                confs.append(conf)
                if ans == r["gt"] and ans == r["original_majority"]:
                    colors.append(COLORS["green"])
                elif ans == r["gt"]:
                    colors.append(COLORS["blue"])
                elif ans == r["original_majority"]:
                    colors.append(COLORS["red"])
                else:
                    colors.append(COLORS["gray"])

            ax.barh(range(len(ans_labels)), confs, color=colors, height=0.6, edgecolor='white')
            ax.set_yticks(range(len(ans_labels)))
            ax.set_yticklabels(ans_labels, fontsize=9)
            ax.set_xlim(0, max(0.6, max(confs) + 0.1))
            ax.invert_yaxis()
            ax.set_title(f"Consistency: {r['consistency']:.2f}", fontsize=11, color='#555555')
            for j, c in enumerate(confs):
                ax.text(c + 0.02, j, f"{c:.2f}", va='center', fontsize=8, color='#333')

        handles = [
            Patch(facecolor=COLORS["green"], label="GT + MAJ"),
            Patch(facecolor=COLORS["blue"], label="Ground Truth (GT)"),
            Patch(facecolor=COLORS["red"], label="Majority (Wrong)"),
            Patch(facecolor=COLORS["gray"], label="Other Noise"),
        ]
        fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False)
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        path = os.path.join(output_dir, "exp1_motivation.png")
        plt.savefig(path)
        plt.close()
        print(f"\n  Plot saved: {path}")


# =====================================================
# Experiment 2: Confidence vs Quality
# =====================================================
def experiment_2(results, output_dir):
    print("\n" + "=" * 80)
    print("Exp 2: Bootstrap Confidence vs Pseudo-label Quality")
    print("=" * 80)

    buckets = [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]
    bucket_names = ["0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"]

    print(f"\n  {'P_boot bucket':<15} {'Count':<8} {'Maj Accuracy':<16} {'Avg consistency':<18}")
    print("  " + "-" * 58)

    bucket_data = []
    for (lo, hi), name in zip(buckets, bucket_names):
        b = [r for r in results if lo <= r["boot_stability"] < hi]
        if not b:
            bucket_data.append((name, 0, 0, 0))
            continue
        n = len(b)
        acc = sum(1 for r in b if r["maj_correct"]) / n
        avg_cons = np.mean([r["consistency"] for r in b])
        bucket_data.append((name, n, acc, avg_cons))
        print(f"  {name:<15} {n:<8} {acc:<16.1%} {avg_cons:<18.3f}")

    # AUC
    if HAS_SKLEARN:
        labels = [1 if r["maj_correct"] else 0 for r in results]
        scores = [r["boot_stability"] for r in results]
        cons_scores = [r["consistency"] for r in results]
        if len(set(labels)) > 1:
            auc_boot = roc_auc_score(labels, scores)
            auc_cons = roc_auc_score(labels, cons_scores)
            print(f"\n  AUC (P_boot):          {auc_boot:.4f}")
            print(f"  AUC (consistency rate): {auc_cons:.4f}")
            print(f"  -> P_boot {'better' if auc_boot > auc_cons else 'NOT better'} than consistency rate")

    # Plot
    if HAS_MPL:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        valid = [d for d in bucket_data if d[1] > 0]
        names = [d[0] for d in valid]
        accs = [d[2] for d in valid]
        counts = [d[1] for d in valid]

        bars = axes[0].bar(names, accs, color=COLORS["blue"], width=0.6, alpha=0.9)
        add_value_labels(axes[0], "{:.1%}")
        for bar, c in zip(bars, counts):
            axes[0].text(bar.get_x() + bar.get_width() / 2, -0.06,
                         f"n={c}", ha="center", fontsize=8, color='#888')
        axes[0].set_xlabel("Bootstrap Confidence ($P_{boot}$)")
        axes[0].set_ylabel("Majority Accuracy")
        axes[0].set_ylim(0, 1.15)
        axes[0].set_title("Pseudo-label Quality by Confidence")

        confs = [r["boot_stability"] for r in results]
        corr = [1 if r["maj_correct"] else 0 for r in results]
        jitter = [c + np.random.uniform(-0.05, 0.05) for c in corr]
        axes[1].scatter(confs, jitter, alpha=0.25, color=COLORS["dark_blue"], s=15)
        axes[1].set_xlabel("Bootstrap Confidence")
        axes[1].set_yticks([0, 1])
        axes[1].set_yticklabels(["Wrong MAJ", "Correct MAJ"])
        axes[1].set_title("Per-Problem Distribution")

        plt.tight_layout()
        path = os.path.join(output_dir, "exp2_confidence_quality.png")
        plt.savefig(path)
        plt.close()
        print(f"\n  Plot saved: {path}")


# =====================================================
# Experiment 3: Advantage Pos/Neg Analysis (5 Methods)
# =====================================================
def experiment_3(results, output_dir):
    print("\n" + "=" * 80)
    print("Exp 3: Average Advantage for Positive/Negative Samples (5 Methods)")
    print("=" * 80)

    METHOD_NAMES = ["Binary GRPO", "Boot GRPO", "Pass@k TTA", "Oracle GRPO", "Oracle Pass@k"]
    METHOD_KEYS = ["bin", "boot", "pk_tta", "gt_bin", "gt_pk"]

    # --- Table header ---
    header = f"  {'Bucket':<14}"
    for m in METHOD_NAMES:
        header += f" | {m:<16}"
    print(f"\n{header}")
    sub = f"  {'':14s}"
    for _ in METHOD_NAMES:
        sub += f" | {'A_pos':>7} {'A_neg':>7}"
    print(sub)
    print("  " + "-" * (14 + len(METHOD_NAMES) * 19))

    plot_data = []

    for b_name, b_fn in CONS_BUCKETS_ALL:
        bucket = [r for r in results if b_fn(r)]
        if not bucket:
            continue

        accum = {k: {"pos": [], "neg": []} for k in METHOD_KEYS}
        align_cos = {k: [] for k in ["bin", "boot", "pk"]}
        align_sign = {k: [] for k in ["bin", "boot", "pk"]}

        for r in bucket:
            a_bin = grpo_advantage(r["binary_rewards"])
            a_boot = grpo_advantage(r["soft_rewards"])
            a_pk_tta = passk_advantage(r["binary_rewards"], k=4)
            a_gt_bin = grpo_advantage(r["gt_rewards"])
            a_gt_pk = passk_advantage(r["gt_rewards"], k=4)

            advs = {"bin": a_bin, "boot": a_boot, "pk_tta": a_pk_tta,
                    "gt_bin": a_gt_bin, "gt_pk": a_gt_pk}

            gt = np.array(r["gt_rewards"])
            pos_mask = gt > 0.5
            neg_mask = gt < 0.5

            for key, a in advs.items():
                if pos_mask.sum() > 0:
                    accum[key]["pos"].append(a[pos_mask].mean())
                if neg_mask.sum() > 0:
                    accum[key]["neg"].append(a[neg_mask].mean())

            # Alignment metrics
            align_cos["bin"].append(cosine_sim(a_bin, a_gt_bin))
            align_cos["boot"].append(cosine_sim(a_boot, a_gt_bin))
            align_cos["pk"].append(cosine_sim(a_pk_tta, a_gt_pk))
            align_sign["bin"].append(sign_align(a_bin, a_gt_bin))
            align_sign["boot"].append(sign_align(a_boot, a_gt_bin))
            align_sign["pk"].append(sign_align(a_pk_tta, a_gt_pk))

        # Build table row
        row = f"  {b_name:<14}"
        pos_vals, neg_vals = [], []
        for key in METHOD_KEYS:
            p = np.mean(accum[key]["pos"]) if accum[key]["pos"] else 0
            n = np.mean(accum[key]["neg"]) if accum[key]["neg"] else 0
            row += f" | {p:+7.3f} {n:+7.3f}"
            pos_vals.append(p)
            neg_vals.append(n)
        print(row)

        plot_data.append({"name": b_name, "pos": pos_vals, "neg": neg_vals})

    # --- Alignment summary ---
    print(f"\n  Alignment with Oracle (cosine similarity):")
    print(f"  {'Bucket':<14} {'Binary->Oracle':<18} {'Boot->Oracle':<18} {'Pass@k->Oracle':<18}")
    print("  " + "-" * 68)
    for b_name, b_fn in CONS_BUCKETS_ALL:
        bucket = [r for r in results if b_fn(r)]
        if not bucket:
            continue
        cos_bin, cos_boot, cos_pk = [], [], []
        for r in bucket:
            a_bin = grpo_advantage(r["binary_rewards"])
            a_boot = grpo_advantage(r["soft_rewards"])
            a_pk_tta = passk_advantage(r["binary_rewards"], k=4)
            a_gt_bin = grpo_advantage(r["gt_rewards"])
            a_gt_pk = passk_advantage(r["gt_rewards"], k=4)
            cos_bin.append(cosine_sim(a_bin, a_gt_bin))
            cos_boot.append(cosine_sim(a_boot, a_gt_bin))
            cos_pk.append(cosine_sim(a_pk_tta, a_gt_pk))
        print(f"  {b_name:<14} {np.mean(cos_bin):<18.3f} {np.mean(cos_boot):<18.3f} {np.mean(cos_pk):<18.3f}")

    # --- Plot ---
    if HAS_MPL and plot_data:
        fig, axes = plt.subplots(1, 2, figsize=(18, 6))
        names = [d["name"] for d in plot_data]
        x = np.arange(len(names))
        w = 0.15
        method_colors = [COLORS["red"], COLORS["blue"], COLORS["purple"],
                         COLORS["orange"], COLORS["green"]]

        for panel_idx, (ax, target) in enumerate(zip(axes, ["pos", "neg"])):
            for j in range(5):
                vals = [d[target][j] for d in plot_data]
                ax.bar(x + (j - 2) * w, vals, w, label=METHOD_NAMES[j],
                       color=method_colors[j], edgecolor='white', alpha=0.9)
            ax.set_xticks(x)
            ax.set_xticklabels(names)
            ax.set_ylabel("Average Advantage")
            title_type = "GT-Correct (A_pos)" if target == "pos" else "GT-Incorrect (A_neg)"
            ax.set_title(f"Avg Advantage on {title_type} Samples")
            ax.axhline(0, color='black', linewidth=0.8)
            if panel_idx == 0:
                ax.legend(loc='upper right', framealpha=0.9, fontsize=9)

        plt.tight_layout()
        path = os.path.join(output_dir, "exp3_advantage_pos_neg.png")
        plt.savefig(path)
        plt.close()
        print(f"\n  Plot saved: {path}")


# =====================================================
# Experiment 4: Coverage + Answer Count + Thresholding
# =====================================================
def experiment_4(results, output_dir):
    print("\n" + "=" * 80)
    print("Exp 4: Bootstrap Pseudo-label Coverage Analysis")
    print("=" * 80)

    # --- Coverage table ---
    print(f"\n  {'Consistency':<14} {'Count':<8} {'Maj Accuracy':<14} {'Boot Coverage':<16} "
          f"{'Gain':<10} {'GT Avg Conf':<14}")
    print("  " + "-" * 76)

    plot_data = []
    for b_name, b_fn in CONS_BUCKETS_ALL:
        bucket = [r for r in results if b_fn(r)]
        if not bucket:
            continue
        n = len(bucket)
        maj_acc = sum(1 for r in bucket if r["maj_correct"]) / n
        boot_cov = sum(1 for r in bucket if r["gt_covered"]) / n
        gt_conf = np.mean([r["gt_boot_conf"] for r in bucket])
        gain = boot_cov - maj_acc
        print(f"  {b_name:<14} {n:<8} {maj_acc:<14.1%} {boot_cov:<16.1%} "
              f"{gain:+<10.1%} {gt_conf:<14.3f}")
        plot_data.append((b_name, n, maj_acc, boot_cov, gt_conf))

    # --- Low consistency rescue detail ---
    low_cons = [r for r in results if r["consistency"] <= 0.3]
    maj_wrong = [r for r in low_cons if not r["maj_correct"]]
    rescued = [r for r in maj_wrong if r["gt_covered"]]
    if maj_wrong:
        print(f"\n  Low consistency rescue:")
        print(f"    Majority wrong: {len(maj_wrong)} problems")
        print(f"    Bootstrap covers GT: {len(rescued)} ({100 * len(rescued) / len(maj_wrong):.1f}%)")
        print(f"    -> {len(rescued)} correct answers 'rescued' by bootstrap")

    # --- Answer count analysis ---
    print(f"\n  {'':=<80}")
    print(f"  Answer Count Analysis (Original vs Bootstrap)")
    print(f"  {'':=<80}")
    print(f"\n  {'Consistency':<14} {'Count':<8} {'Orig Unique':<14} {'Boot Distinct':<16} "
          f"{'Reduction':<12} {'Top-1 Share':<14} {'Top-3 Share':<14}")
    print(f"  {'-' * 92}")

    for b_name, b_fn in CONS_BUCKETS_ALL:
        bucket = [r for r in results if b_fn(r)]
        if not bucket:
            continue
        n = len(bucket)
        avg_orig = np.mean([r["n_unique_answers"] for r in bucket])
        avg_boot = np.mean([r["n_distinct_maj"] for r in bucket])
        reduction = avg_orig - avg_boot

        top1_shares, top3_shares = [], []
        for r in bucket:
            sorted_conf = sorted(r["answer_conf"].values(), reverse=True)
            top1_shares.append(sorted_conf[0] if sorted_conf else 0)
            top3_shares.append(sum(sorted_conf[:3]))

        print(f"  {b_name:<14} {n:<8} {avg_orig:<14.1f} {avg_boot:<16.1f} "
              f"{reduction:+<12.1f} {np.mean(top1_shares):<14.3f} {np.mean(top3_shares):<14.3f}")

    # --- Threshold filtering ---
    print(f"\n  {'':=<80}")
    print(f"  Threshold Filtering: Answers After Removing P_boot < threshold")
    print(f"  {'':=<80}")

    thresholds = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20]

    for b_name, b_fn in CONS_BUCKETS_ALL:
        bucket = [r for r in results if b_fn(r)]
        if not bucket:
            continue

        print(f"\n    [{b_name}] ({len(bucket)} problems)")
        print(f"    {'Threshold':<12} {'Avg Remaining':<16} {'GT Kept Rate':<14} {'Top-1=GT Rate':<16}")
        print(f"    {'-' * 58}")

        # Baseline (no filter)
        avg_orig = np.mean([len(r["answer_conf"]) for r in bucket])
        gt_base = sum(1 for r in bucket if r["gt_boot_conf"] > 0) / len(bucket)
        print(f"    {'No filter':<12} {avg_orig:<16.1f} {gt_base:<14.1%} {'-':<16}")

        for thresh in thresholds:
            remaining, gt_kept, top1_gt = [], 0, 0
            for r in bucket:
                filtered = {a: c for a, c in r["answer_conf"].items() if c >= thresh}
                remaining.append(len(filtered))
                if r["gt"] in filtered:
                    gt_kept += 1
                if filtered and max(filtered, key=filtered.get) == r["gt"]:
                    top1_gt += 1
            print(f"    {thresh:<12.2f} {np.mean(remaining):<16.1f} "
                  f"{gt_kept / len(bucket):<14.1%} {top1_gt / len(bucket):<16.1%}")

    # --- Plot ---
    if HAS_MPL and plot_data:
        fig, ax = plt.subplots(figsize=(10, 6))
        cov_data = [d for d in plot_data if d[0] != "All"]
        x = np.arange(len(cov_data))
        w = 0.25

        ax.bar(x - w, [d[2] for d in cov_data], w,
               label="Hard Maj Accuracy", color=COLORS["red"], alpha=0.85)
        ax.bar(x, [d[3] for d in cov_data], w,
               label="Bootstrap Coverage", color=COLORS["blue"], alpha=0.85)
        ax.bar(x + w, [d[4] for d in cov_data], w,
               label="GT Avg Conf in Boot", color=COLORS["green"], alpha=0.85)

        add_value_labels(ax, "{:.1%}")

        for i, d in enumerate(cov_data):
            gain = d[3] - d[2]
            if gain > 0.05:
                ax.annotate(f"+{gain:.1%}",
                            xy=(i, d[3] + 0.08), xytext=(i - w, d[2] + 0.12),
                            arrowprops=dict(facecolor='#333', arrowstyle="->",
                                            connectionstyle="arc3,rad=-0.2"),
                            fontsize=10, color=COLORS["red"], fontweight='bold')

        ax.set_xticks(x)
        ax.set_xticklabels([d[0] for d in cov_data], fontsize=11)
        ax.set_ylabel("Rate")
        ax.set_ylim(0, 1.15)
        ax.set_title("Bootstrap Rescue Effect on Low Consistency")
        ax.legend(loc='upper left', framealpha=0.9)

        plt.tight_layout()
        path = os.path.join(output_dir, "exp4_coverage.png")
        plt.savefig(path)
        plt.close()
        print(f"\n  Plot saved: {path}")


# =====================================================
# Main
# =====================================================
def main():
    parser = argparse.ArgumentParser(description="Bootstrap Pseudo-label Experiments 1-4")
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="bootstrap_results")
    parser.add_argument("--num_bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    set_plot_style()

    # Load data
    print(f"Loading data from {args.input_file}")
    data = []
    with open(args.input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    print(f"Loaded {len(data)} problems")

    # Analyze
    print(f"Running bootstrap analysis (B={args.num_bootstrap})...")
    results = []
    for i, item in enumerate(data):
        r = bootstrap_analysis(
            item.get("extracted_answers", []),
            item.get("answer", ""),
            B=args.num_bootstrap
        )
        if r is not None:
            r["problem_idx"] = i
            results.append(r)
    print(f"Analyzed {len(results)} problems")

    # Summary stats
    n_correct = sum(1 for r in results if r["maj_correct"])
    print(f"\n  Majority accuracy: {n_correct}/{len(results)} ({100 * n_correct / len(results):.1f}%)")
    print(f"  Avg consistency:   {np.mean([r['consistency'] for r in results]):.3f}")
    print(f"  Avg bootstrap stability: {np.mean([r['boot_stability'] for r in results]):.3f}")

    # Run experiments
    experiment_1(results, args.output_dir, args.num_bootstrap)
    experiment_2(results, args.output_dir)
    experiment_3(results, args.output_dir)
    experiment_4(results, args.output_dir)

    print("\n" + "=" * 80)
    print(f"ALL EXPERIMENTS COMPLETE — Results in: {args.output_dir}/")
    print("=" * 80)


if __name__ == "__main__":
    main()
