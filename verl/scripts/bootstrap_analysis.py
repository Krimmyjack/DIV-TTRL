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
    # Section 0: Bootstrap Top-1 vs Raw Majority Comparison
    # ============================================================
    print(f"\n{'=' * 95}")
    print("Bootstrap Top-1 vs Raw Majority")
    print(f"{'=' * 95}")
    print("  Bootstrap top-1 = most frequent answer among 1000 bootstrap iterations")
    print("  Raw MAJ = most frequent answer in original 64 samples")

    print(f"\n  {'Bucket':<16} {'N':<6} {'MAJ Acc':<10} {'Boot1 Acc':<10} "
          f"{'Agree%':<9} {'Disagree':<10} {'MAJ wins':<10} {'Boot wins':<10}")
    print("  " + "-" * 81)

    for name, fn in buckets:
        bucket = [r for r in results if fn(r)]
        if not bucket:
            continue
        n = len(bucket)

        maj_correct = 0
        boot1_correct = 0
        agree = 0
        disagree = 0
        maj_win = 0
        boot_win = 0

        for r in bucket:
            boot_top1 = r["candidates"][0][0] if r["candidates"] else r["original_majority"]
            maj = r["original_majority"]
            gt = r["gt"]

            if maj == gt:
                maj_correct += 1
            if boot_top1 == gt:
                boot1_correct += 1

            if boot_top1 == maj:
                agree += 1
            else:
                disagree += 1
                if boot_top1 == gt and maj != gt:
                    boot_win += 1
                elif maj == gt and boot_top1 != gt:
                    maj_win += 1

        maj_acc = maj_correct / n
        boot1_acc = boot1_correct / n
        agree_pct = agree / n

        print(f"  {name:<16} {n:<6} {maj_acc:<10.1%} {boot1_acc:<10.1%} "
              f"{agree_pct:<9.1%} {disagree:<10} {maj_win:<10} {boot_win:<10}")

    # Per-bucket disagreement detail
    for name, fn in buckets:
        bucket = [r for r in results if fn(r)]
        if not bucket:
            continue

        disagree_cases = []
        for r in bucket:
            boot_top1 = r["candidates"][0][0] if r["candidates"] else r["original_majority"]
            maj = r["original_majority"]
            if boot_top1 != maj:
                gt = r["gt"]
                freq = r["freq"]
                maj_count = freq[maj]
                boot1_count = freq.get(boot_top1, 0)
                margin = maj_count - boot1_count

                disagree_cases.append({
                    "maj_correct": maj == gt, "boot1_correct": boot_top1 == gt,
                    "margin": margin,
                })

        if not disagree_cases:
            print(f"\n  {name}: No disagreements")
            continue

        n_dis = len(disagree_cases)
        boot_win = sum(1 for d in disagree_cases if d["boot1_correct"] and not d["maj_correct"])
        maj_win = sum(1 for d in disagree_cases if d["maj_correct"] and not d["boot1_correct"])
        both_wrong = sum(1 for d in disagree_cases if not d["maj_correct"] and not d["boot1_correct"])
        both_right = sum(1 for d in disagree_cases if d["maj_correct"] and d["boot1_correct"])
        margins = [d["margin"] for d in disagree_cases]
        avg_margin = np.mean(margins)

        print(f"\n  {name} ({n_dis} disagreements, {100*n_dis/len(bucket):.1f}% of bucket):")
        print(f"    Boot wins: {boot_win} ({100*boot_win/n_dis:.1f}%)  |  "
              f"MAJ wins: {maj_win} ({100*maj_win/n_dis:.1f}%)  |  "
              f"Both wrong: {both_wrong} ({100*both_wrong/n_dis:.1f}%)")
        print(f"    Margin (maj_count - boot1_count): "
              f"avg={avg_margin:.1f}, min={min(margins)}, median={np.median(margins):.0f}, max={max(margins)}")

        # Small margin analysis
        for threshold in [1, 2, 3]:
            sm = [d for d in disagree_cases if d["margin"] <= threshold]
            if sm:
                sm_boot = sum(1 for d in sm if d["boot1_correct"])
                sm_maj = sum(1 for d in sm if d["maj_correct"])
                print(f"    Margin<={threshold} ({len(sm)} cases): "
                      f"Boot correct {sm_boot} ({100*sm_boot/len(sm):.0f}%), "
                      f"MAJ correct {sm_maj} ({100*sm_maj/len(sm):.0f}%)")

    # ============================================================
    # Section 2: Coverage Comparison (Freq Top-K vs Bootstrap λ)
    # ============================================================
    print(f"\n{'=' * 95}")
    print("GT Coverage: Raw Frequency Top-K vs Bootstrap (threshold=λ)")
    print(f"{'=' * 95}")

    ks = [1, 2, 3, 4, 5]
    lambdas = [0.005, 0.01, 0.02, 0.03, 0.05, 0.1]

    for bname, bfn in buckets:
        bucket = [r for r in results if bfn(r)]
        if not bucket:
            continue
        n = len(bucket)

        print(f"\n  {bname} (N={n}):")

        # --- Freq Top-K ---
        print(f"    {'Method':<20} {'GT Cover%':<12} {'Avg #Cands':<12}")
        print(f"    " + "-" * 44)
        for k in ks:
            gt_hit = 0
            avg_cands = 0
            for r in bucket:
                freq = r["freq"]
                topk_set = {ans for ans, _ in freq.most_common(k)}
                if r["gt"] in topk_set:
                    gt_hit += 1
                avg_cands += len(topk_set)
            gt_cover = gt_hit / n
            avg_c = avg_cands / n
            print(f"    {'Freq Top-'+str(k):<20} {gt_cover:<12.1%} {avg_c:<12.1f}")

        print(f"    " + "-" * 44)
        for lam in lambdas:
            gt_hit = 0
            total_cands = 0
            for r in bucket:
                boot_cands = {ans for ans, p in r["answer_conf"].items() if p >= lam}
                if r["gt"] in boot_cands:
                    gt_hit += 1
                total_cands += len(boot_cands)
            gt_cover = gt_hit / n
            avg_c = total_cands / n
            print(f"    {'Boot λ='+str(lam):<20} {gt_cover:<12.1%} {avg_c:<12.1f}")


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
