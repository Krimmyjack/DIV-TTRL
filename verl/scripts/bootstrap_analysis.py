"""
Bootstrap Top-K Pseudo-label Set Analysis

Usage:
    python scripts/bootstrap_analysis.py \
        --input_file math500_candidates.jsonl \
        --num_bootstrap 1000
"""

import json
import argparse
import random
import numpy as np
from collections import Counter


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


def analyze_problem(extracted_answers, ground_truth, B=1000):
    valid = [a for a in extracted_answers if a != "[NO_ANSWER]"]
    N = len(valid)
    if N == 0:
        return None

    freq = Counter(valid)
    original_majority = freq.most_common(1)[0][0]
    consistency = freq[original_majority] / N
    gt_norm = strip_string(str(ground_truth))

    # Bootstrap
    boot_majorities = []
    for _ in range(B):
        subset = random.choices(valid, k=N)
        boot_maj = Counter(subset).most_common(1)[0][0]
        boot_majorities.append(boot_maj)

    boot_counter = Counter(boot_majorities)
    answer_conf = {ans: cnt / B for ans, cnt in boot_counter.items()}

    # Sorted candidates after threshold filter
    THRESHOLD = 0.02
    candidates = sorted(
        [(ans, c) for ans, c in answer_conf.items() if c >= THRESHOLD],
        key=lambda x: -x[1]
    )

    # Build top-K sets for K=1,2,3
    topk_info = {}
    for K in [1, 2, 3]:
        topk_set = {ans for ans, _ in candidates[:K]}
        matching = sum(1 for a in valid if a in topk_set)
        topk_info[K] = {
            "set": topk_set,
            "size": len(topk_set),
            "gt_in_set": gt_norm in topk_set,
            "rollouts": matching,
            "coverage": matching / N,
        }

    return {
        "N": N,
        "answers": valid,
        "freq": freq,
        "n_unique": len(freq),
        "consistency": consistency,
        "original_majority": original_majority,
        "maj_correct": original_majority == gt_norm,
        "gt": gt_norm,
        "answer_conf": answer_conf,
        "candidates": candidates,
        "topk": topk_info,
    }


def print_results(results, B):
    total = len(results)
    print(f"\nTotal problems: {total}, Bootstrap iterations: {B}")

    buckets = [
        ("Low(<=0.3)", lambda r: r["consistency"] <= 0.3),
        ("Mid(0.3-0.7)", lambda r: 0.3 < r["consistency"] <= 0.7),
        ("High(>0.7)", lambda r: r["consistency"] > 0.7),
        ("All", lambda r: True),
    ]

    # ============================================================
    # Section 1: Top-1 / Top-2 / Top-3 Comparison
    # ============================================================
    for K in [1, 2, 3]:
        print(f"\n{'=' * 95}")
        print(f"Top-{K} Pseudo-label Set (threshold=0.02, cap={K})")
        print(f"{'=' * 95}")

        print(f"\n  {'Bucket':<16} {'Count':<7} {'MAJ Acc':<10} {'Top{} Acc'.format(K):<10} "
              f"{'Gain':<9} {'Avg Size':<10} {'Avg Rollouts':<14} {'Avg Coverage':<12}")
        print("  " + "-" * 88)

        for name, fn in buckets:
            bucket = [r for r in results if fn(r)]
            if not bucket:
                continue
            n = len(bucket)
            maj_acc = sum(1 for r in bucket if r["maj_correct"]) / n
            topk_acc = sum(1 for r in bucket if r["topk"][K]["gt_in_set"]) / n
            avg_size = np.mean([r["topk"][K]["size"] for r in bucket])
            avg_rollouts = np.mean([r["topk"][K]["rollouts"] for r in bucket])
            avg_coverage = np.mean([r["topk"][K]["coverage"] for r in bucket])
            gain = topk_acc - maj_acc

            print(f"  {name:<16} {n:<7} {maj_acc:<10.1%} {topk_acc:<10.1%} "
                  f"{gain:<+9.1%} {avg_size:<10.1f} {avg_rollouts:<14.1f} {avg_coverage:<12.1%}")

    # ============================================================
    # Section 2: Side-by-side Summary (Low consistency only)
    # ============================================================
    low = [r for r in results if r["consistency"] <= 0.3]
    if low:
        print(f"\n{'=' * 95}")
        print("Side-by-side: Low Consistency (<=0.3)")
        print(f"{'=' * 95}")

        print(f"\n  {'Metric':<30} {'MAJ-only':<15} {'Top-1':<15} {'Top-2':<15} {'Top-3':<15}")
        print("  " + "-" * 85)

        n = len(low)
        maj_acc = sum(1 for r in low if r["maj_correct"]) / n

        metrics = {
            "Accuracy (GT in set)": [maj_acc],
            "Avg rollouts (pos)": [np.mean([r["freq"][r["original_majority"]] for r in low])],
            "Avg coverage": [np.mean([r["freq"][r["original_majority"]] / r["N"] for r in low])],
        }

        for K in [1, 2, 3]:
            metrics["Accuracy (GT in set)"].append(
                sum(1 for r in low if r["topk"][K]["gt_in_set"]) / n)
            metrics["Avg rollouts (pos)"].append(
                np.mean([r["topk"][K]["rollouts"] for r in low]))
            metrics["Avg coverage"].append(
                np.mean([r["topk"][K]["coverage"] for r in low]))

        for label, vals in metrics.items():
            if "rollouts" in label.lower():
                print(f"  {label:<30} {vals[0]:<15.1f} {vals[1]:<15.1f} {vals[2]:<15.1f} {vals[3]:<15.1f}")
            else:
                print(f"  {label:<30} {vals[0]:<15.1%} {vals[1]:<15.1%} {vals[2]:<15.1%} {vals[3]:<15.1%}")

        # Rescued problems
        maj_wrong = [r for r in low if not r["maj_correct"]]
        print(f"\n  Majority wrong: {len(maj_wrong)} problems")
        for K in [1, 2, 3]:
            rescued = sum(1 for r in maj_wrong if r["topk"][K]["gt_in_set"])
            print(f"    Top-{K} rescued: {rescued}/{len(maj_wrong)} ({100*rescued/len(maj_wrong):.1f}%)")

    # ============================================================
    # Section 3: Set Size Distribution
    # ============================================================
    print(f"\n{'=' * 95}")
    print("Set Size Distribution (Low consistency only)")
    print(f"{'=' * 95}")

    if low:
        for K in [1, 2, 3]:
            sizes = [r["topk"][K]["size"] for r in low]
            dist = Counter(sizes)
            parts = [f"size={s}: {c} ({100*c/len(low):.0f}%)" for s, c in sorted(dist.items())]
            print(f"  Top-{K}: {', '.join(parts)}")

    # ============================================================
    # Section 4: FP Rate Comparison
    # ============================================================
    print(f"\n{'=' * 95}")
    print("False Positive Rate: MAJ vs Top-K")
    print(f"{'=' * 95}")

    print(f"\n  {'Bucket':<16} {'MAJ FP%':<12} {'Top1 FP%':<12} {'Top2 FP%':<12} {'Top3 FP%':<12}")
    print("  " + "-" * 60)

    for name, fn in buckets:
        bucket = [r for r in results if fn(r)]
        if not bucket:
            continue

        fp_rates = []

        # MAJ FP
        maj_pos = sum(r["freq"][r["original_majority"]] for r in bucket)
        maj_fp = sum(r["freq"][r["original_majority"]] for r in bucket if not r["maj_correct"])
        fp_rates.append(maj_fp / maj_pos if maj_pos else 0)

        for K in [1, 2, 3]:
            total_pos = sum(r["topk"][K]["rollouts"] for r in bucket)
            total_fp = 0
            for r in bucket:
                gt = r["gt"]
                for ans in r["answers"]:
                    if ans in r["topk"][K]["set"] and ans != gt:
                        total_fp += 1
            fp_rates.append(total_fp / total_pos if total_pos else 0)

        print(f"  {name:<16} {fp_rates[0]:<12.1%} {fp_rates[1]:<12.1%} "
              f"{fp_rates[2]:<12.1%} {fp_rates[3]:<12.1%}")

    # ============================================================
    # Section 5: Advantage Accuracy Analysis
    # ============================================================
    print(f"\n{'=' * 95}")
    print("Advantage Accuracy: MAJ / Top-K / Oracle (GRPO advantage)")
    print(f"{'=' * 95}")
    print("  A_i = (r_i - mean(r)) / std(r), then compare sign & direction with Oracle A")

    def compute_grpo_advantage(rewards):
        """Compute GRPO advantage from reward vector."""
        r = np.array(rewards, dtype=np.float64)
        mean = r.mean()
        std = r.std()
        if std < 1e-8:
            return np.zeros_like(r)
        return (r - mean) / std

    def sign_alignment(a, b):
        """Fraction of samples where sign(a_i) == sign(b_i), ignoring zeros."""
        mask = (a != 0) & (b != 0)
        if mask.sum() == 0:
            return float('nan')
        return (np.sign(a[mask]) == np.sign(b[mask])).mean()

    def cosine_sim(a, b):
        """Cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a < 1e-8 or norm_b < 1e-8:
            return float('nan')
        return np.dot(a, b) / (norm_a * norm_b)

    # Methods: MAJ, Top-1, Top-2, Top-3, Freq, Oracle
    method_names = ["MAJ", "Top-1", "Top-2", "Top-3", "Freq", "Oracle"]

    print(f"\n  {'Bucket':<16} {'Method':<10} {'Sign Align':<12} {'Cos Sim':<10} "
          f"{'Avg A+(gt)':<12} {'Avg A-(gt)':<12} {'Mean Reward':<12}")
    print("  " + "-" * 84)

    for bname, bfn in buckets:
        bucket = [r for r in results if bfn(r)]
        if not bucket:
            continue

        # Accumulate per-method stats
        method_stats = {m: {"sign_aligns": [], "cos_sims": [],
                            "avg_pos": [], "avg_neg": [], "mean_r": []}
                        for m in method_names}

        for r in bucket:
            answers = r["answers"]
            gt = r["gt"]
            N = len(answers)

            # Oracle rewards & advantage
            gt_rewards = np.array([1.0 if a == gt else 0.0 for a in answers])
            gt_adv = compute_grpo_advantage(gt_rewards)

            # Skip if oracle has zero std (all correct or all wrong)
            if np.std(gt_rewards) < 1e-8:
                continue

            # MAJ rewards
            maj_rewards = np.array([1.0 if a == r["original_majority"] else 0.0 for a in answers])

            # Top-K rewards
            topk_rewards = {}
            for K in [1, 2, 3]:
                topk_rewards[K] = np.array(
                    [1.0 if a in r["topk"][K]["set"] else 0.0 for a in answers])

            # Freq rewards: use P_boot as reward, 0 if below threshold
            THRESHOLD = 0.02
            ans_conf = r["answer_conf"]
            freq_rewards = np.array([
                ans_conf.get(a, 0.0) if ans_conf.get(a, 0.0) >= THRESHOLD else 0.0
                for a in answers
            ])

            all_rewards = {
                "MAJ": maj_rewards,
                "Top-1": topk_rewards[1],
                "Top-2": topk_rewards[2],
                "Top-3": topk_rewards[3],
                "Freq": freq_rewards,
                "Oracle": gt_rewards,
            }

            for m in method_names:
                rw = all_rewards[m]
                adv = compute_grpo_advantage(rw)

                if np.std(rw) < 1e-8:
                    continue

                if m == "Oracle":
                    # Oracle vs itself: perfect alignment
                    method_stats[m]["sign_aligns"].append(1.0)
                    method_stats[m]["cos_sims"].append(1.0)
                else:
                    method_stats[m]["sign_aligns"].append(sign_alignment(adv, gt_adv))
                    method_stats[m]["cos_sims"].append(cosine_sim(adv, gt_adv))
                method_stats[m]["mean_r"].append(rw.mean())

                # Avg advantage for GT-positive and GT-negative samples
                gt_pos_mask = gt_rewards > 0
                gt_neg_mask = gt_rewards == 0
                if gt_pos_mask.any():
                    method_stats[m]["avg_pos"].append(adv[gt_pos_mask].mean())
                if gt_neg_mask.any():
                    method_stats[m]["avg_neg"].append(adv[gt_neg_mask].mean())

        for m in method_names:
            s = method_stats[m]
            sa = np.nanmean(s["sign_aligns"]) if s["sign_aligns"] else float('nan')
            cs = np.nanmean(s["cos_sims"]) if s["cos_sims"] else float('nan')
            ap = np.mean(s["avg_pos"]) if s["avg_pos"] else float('nan')
            an = np.mean(s["avg_neg"]) if s["avg_neg"] else float('nan')
            mr = np.mean(s["mean_r"]) if s["mean_r"] else float('nan')

            print(f"  {bname if m == method_names[0] else '':<16} {m:<10} "
                  f"{sa:<12.3f} {cs:<10.3f} {ap:<+12.3f} {an:<+12.3f} {mr:<12.3f}")
        print("  " + "-" * 84)

    # ============================================================
    # Section 6: Pass@k (k=4) Advantage Accuracy Analysis
    # ============================================================
    print(f"\n{'=' * 95}")
    print("Advantage Accuracy: Pass@k TTA (k=4)")
    print(f"{'=' * 95}")
    print("  Using the same formula as compute_pass_grpo_advantage in core_algos.py")

    from scipy.special import gammaln

    def log_comb(n, k):
        """log(C(n, k)) using log-gamma."""
        if n < k or n < 0 or k < 0:
            return -np.inf
        return gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)

    def compute_passk_advantage(answer_types, k=4):
        """
        Compute Pass@k advantage for a single prompt's samples.
        answer_types: list where 0 = correct (matches pseudo-label), others = incorrect.
        Returns: np.array of advantages, same length as answer_types.
        """
        N = len(answer_types)
        c_pos = sum(1 for a in answer_types if a == 0)
        c_neg = N - c_pos

        # R_mean = 1 - C(c_neg, k) / C(N, k)
        log_prob_fail = log_comb(c_neg, k) - log_comb(N, k)
        prob_fail = np.exp(log_prob_fail) if np.isfinite(log_prob_fail) else 0.0
        prob_fail = np.clip(prob_fail, 0.0, 1.0)
        r_mean = 1.0 - prob_fail

        # Variance and std
        variance = r_mean * (1.0 - r_mean)
        std = np.sqrt(variance)

        if std < 1e-6:
            return np.zeros(N)

        # A_pos = P_fail / std
        a_pos = prob_fail / std

        # A_neg = (P_fail - P_cond_fail) / std
        if c_neg >= 1:
            log_p_cond = log_comb(c_neg - 1, k - 1) - log_comb(N - 1, k - 1)
            p_cond_fail = np.exp(log_p_cond) if np.isfinite(log_p_cond) else 0.0
            p_cond_fail = np.clip(p_cond_fail, 0.0, 1.0)
        else:
            p_cond_fail = 0.0

        a_neg = (prob_fail - p_cond_fail) / std

        return np.array([a_pos if a == 0 else a_neg for a in answer_types])

    pk_method_names = ["MAJ", "Top-1", "Top-2", "Top-3", "Oracle"]

    print(f"\n  {'Bucket':<16} {'Method':<10} {'Sign Align':<12} {'Cos Sim':<10} "
          f"{'Avg A+(gt)':<12} {'Avg A-(gt)':<12} {'A_pos':<10} {'A_neg':<10}")
    print("  " + "-" * 92)

    for bname, bfn in buckets:
        bucket = [r for r in results if bfn(r)]
        if not bucket:
            continue

        method_stats = {m: {"sign_aligns": [], "cos_sims": [],
                            "avg_pos": [], "avg_neg": [],
                            "a_pos_vals": [], "a_neg_vals": []}
                        for m in pk_method_names}

        for r in bucket:
            answers = r["answers"]
            gt = r["gt"]
            N = len(answers)

            # Oracle: answer_type = 0 if matches GT
            oracle_types = [0 if a == gt else hash(a) % 1000000 + 1 for a in answers]
            oracle_adv = compute_passk_advantage(oracle_types, k=4)

            # Skip if oracle has no signal
            if np.std(oracle_adv) < 1e-8:
                continue

            gt_pos_mask = np.array([a == gt for a in answers])
            gt_neg_mask = ~gt_pos_mask

            # MAJ: answer_type = 0 if matches majority
            maj_types = [0 if a == r["original_majority"] else hash(a) % 1000000 + 1
                         for a in answers]

            # Top-K: answer_type = 0 if in top-K set
            topk_types = {}
            for K in [1, 2, 3]:
                topk_types[K] = [0 if a in r["topk"][K]["set"] else hash(a) % 1000000 + 1
                                 for a in answers]

            all_types = {
                "MAJ": maj_types,
                "Top-1": topk_types[1],
                "Top-2": topk_types[2],
                "Top-3": topk_types[3],
                "Oracle": oracle_types,
            }

            for m in pk_method_names:
                types = all_types[m]
                adv = compute_passk_advantage(types, k=4)

                if np.std(adv) < 1e-8:
                    continue

                if m == "Oracle":
                    method_stats[m]["sign_aligns"].append(1.0)
                    method_stats[m]["cos_sims"].append(1.0)
                else:
                    method_stats[m]["sign_aligns"].append(sign_alignment(adv, oracle_adv))
                    method_stats[m]["cos_sims"].append(cosine_sim(adv, oracle_adv))

                if gt_pos_mask.any():
                    method_stats[m]["avg_pos"].append(adv[gt_pos_mask].mean())
                if gt_neg_mask.any():
                    method_stats[m]["avg_neg"].append(adv[gt_neg_mask].mean())

                # Extract A_pos and A_neg values
                c_pos = sum(1 for t in types if t == 0)
                c_neg = N - c_pos
                if c_pos > 0 and c_neg > 0:
                    pos_vals = adv[[i for i, t in enumerate(types) if t == 0]]
                    neg_vals = adv[[i for i, t in enumerate(types) if t != 0]]
                    method_stats[m]["a_pos_vals"].append(pos_vals[0])
                    method_stats[m]["a_neg_vals"].append(neg_vals[0])

        for m in pk_method_names:
            s = method_stats[m]
            sa = np.nanmean(s["sign_aligns"]) if s["sign_aligns"] else float('nan')
            cs = np.nanmean(s["cos_sims"]) if s["cos_sims"] else float('nan')
            ap = np.mean(s["avg_pos"]) if s["avg_pos"] else float('nan')
            an = np.mean(s["avg_neg"]) if s["avg_neg"] else float('nan')
            a_pos_v = np.mean(s["a_pos_vals"]) if s["a_pos_vals"] else float('nan')
            a_neg_v = np.mean(s["a_neg_vals"]) if s["a_neg_vals"] else float('nan')

            print(f"  {bname if m == pk_method_names[0] else '':<16} {m:<10} "
                  f"{sa:<12.3f} {cs:<10.3f} {ap:<+12.3f} {an:<+12.3f} "
                  f"{a_pos_v:<+10.3f} {a_neg_v:<+10.3f}")
        print("  " + "-" * 92)
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--num_bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    print(f"Loading {args.input_file}")
    data = []
    with open(args.input_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    print(f"Loaded {len(data)} problems")

    print(f"Bootstrap B={args.num_bootstrap}...")
    results = []
    for item in data:
        r = analyze_problem(item.get("extracted_answers", []),
                            item.get("answer", ""), B=args.num_bootstrap)
        if r is not None:
            results.append(r)
    print(f"Analyzed {len(results)} problems")

    print_results(results, args.num_bootstrap)


if __name__ == "__main__":
    main()
