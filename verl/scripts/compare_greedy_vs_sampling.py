#!/usr/bin/env python3
"""
Compare Greedy (temp=0, n=1) vs Sampling (temp=1, n=64) Rollouts

Analyzes:
1. Overall correctness rates for both settings
2. Agreement: when both give the same answer, what's the accuracy?
3. Disagreement: which setting is right more often?
4. Response length comparison
5. Breakdown by sampling consistency (sc_score)

Usage:
    python scripts/compare_greedy_vs_sampling.py \
        --greedy_file qwen4b_greedy.jsonl \
        --sampling_file qwen4b.jsonl
"""

import json
import argparse
import numpy as np
from collections import Counter


def strip_string(string):
    """Normalize answer strings for comparison."""
    if string is None:
        return ""
    string = str(string).replace("\n", "")
    string = string.replace("\\!", "").replace("\\\\", "\\")
    string = string.replace("tfrac", "frac").replace("dfrac", "frac")
    string = string.replace("\\left", "").replace("\\right", "")
    string = string.replace("^{\\circ}", "").replace("^\\circ", "")
    string = string.replace("\\$", "").replace(" ", "")
    if string == "0.5":
        string = "\\frac{1}{2}"
    return string


def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def main():
    parser = argparse.ArgumentParser(description="Compare Greedy vs Sampling rollouts")
    parser.add_argument("--greedy_file", type=str, required=True,
                        help="JSONL from temp=0, n=1 rollout")
    parser.add_argument("--sampling_file", type=str, required=True,
                        help="JSONL from temp=1, n=64 rollout")
    args = parser.parse_args()

    print(f"Loading greedy rollouts from: {args.greedy_file}")
    greedy_data = load_jsonl(args.greedy_file)
    print(f"Loading sampling rollouts from: {args.sampling_file}")
    sampling_data = load_jsonl(args.sampling_file)

    assert len(greedy_data) == len(sampling_data), \
        f"Mismatch: greedy has {len(greedy_data)} problems, sampling has {len(sampling_data)}"

    N = len(greedy_data)
    print(f"Total problems: {N}")

    # ================================================================
    # Per-problem analysis
    # ================================================================
    results = []
    for i in range(N):
        g = greedy_data[i]
        s = sampling_data[i]

        gt = strip_string(str(g.get("answer", "")))

        # Greedy: single response
        g_answer = g.get("sc_answer", "")
        if g_answer is None:
            g_answer = ""
        g_answer = strip_string(str(g_answer))
        g_correct = (g_answer == gt) if gt else False

        # Greedy response length
        g_metrics = g.get("response_metrics", [])
        g_length = g_metrics[0].get("response_length", 0) if g_metrics else 0

        # Sampling: majority vote
        s_answer = s.get("sc_answer", "")
        if s_answer is None:
            s_answer = ""
        s_answer = strip_string(str(s_answer))
        s_correct = (s_answer == gt) if gt else False
        s_sc_score = s.get("sc_score", 0.0)

        # Sampling response lengths (mean across all 64)
        s_metrics = s.get("response_metrics", [])
        s_lengths = [m.get("response_length", 0) for m in s_metrics if m.get("response_length") is not None]
        s_mean_length = np.mean(s_lengths) if s_lengths else 0
        s_median_length = np.median(s_lengths) if s_lengths else 0

        # Pass@N for sampling: is gt in any extracted answer?
        s_extracted = s.get("extracted_answers", [])
        s_pass_at_n = gt in [strip_string(str(a)) for a in s_extracted]

        # Agreement
        agree = (g_answer == s_answer)

        results.append({
            "gt": gt,
            "g_answer": g_answer,
            "g_correct": g_correct,
            "g_length": g_length,
            "s_answer": s_answer,
            "s_correct": s_correct,
            "s_sc_score": s_sc_score,
            "s_mean_length": s_mean_length,
            "s_median_length": s_median_length,
            "s_pass_at_n": s_pass_at_n,
            "agree": agree,
        })

    # ================================================================
    # 1. Overall Statistics
    # ================================================================
    g_total_correct = sum(1 for r in results if r["g_correct"])
    s_total_correct = sum(1 for r in results if r["s_correct"])
    s_pass_n_total = sum(1 for r in results if r["s_pass_at_n"])
    agree_total = sum(1 for r in results if r["agree"])

    print("\n" + "=" * 80)
    print("OVERALL COMPARISON: Greedy (temp=0, n=1) vs Sampling (temp=1, n=64)")
    print("=" * 80)

    print(f"\n  Greedy Accuracy:            {g_total_correct}/{N} ({100*g_total_correct/N:.1f}%)")
    print(f"  Sampling Majority Accuracy: {s_total_correct}/{N} ({100*s_total_correct/N:.1f}%)")
    print(f"  Sampling Pass@N:            {s_pass_n_total}/{N} ({100*s_pass_n_total/N:.1f}%)")
    print(f"  Answer Agreement Rate:      {agree_total}/{N} ({100*agree_total/N:.1f}%)")

    # ================================================================
    # 2. Agreement / Disagreement Analysis
    # ================================================================
    print("\n" + "-" * 80)
    print("AGREEMENT / DISAGREEMENT ANALYSIS")
    print("-" * 80)

    agreed = [r for r in results if r["agree"]]
    disagreed = [r for r in results if not r["agree"]]

    if agreed:
        a_correct = sum(1 for r in agreed if r["g_correct"])
        print(f"\n  When AGREED ({len(agreed)} problems):")
        print(f"    Both Correct:  {a_correct}/{len(agreed)} ({100*a_correct/len(agreed):.1f}%)")
        print(f"    Both Wrong:    {len(agreed)-a_correct}/{len(agreed)} ({100*(len(agreed)-a_correct)/len(agreed):.1f}%)")

    if disagreed:
        g_right_s_wrong = sum(1 for r in disagreed if r["g_correct"] and not r["s_correct"])
        g_wrong_s_right = sum(1 for r in disagreed if not r["g_correct"] and r["s_correct"])
        both_wrong = sum(1 for r in disagreed if not r["g_correct"] and not r["s_correct"])

        print(f"\n  When DISAGREED ({len(disagreed)} problems):")
        print(f"    Greedy Right, Sampling Wrong:  {g_right_s_wrong} ({100*g_right_s_wrong/len(disagreed):.1f}%)")
        print(f"    Greedy Wrong, Sampling Right:   {g_wrong_s_right} ({100*g_wrong_s_right/len(disagreed):.1f}%)")
        print(f"    Both Wrong:                     {both_wrong} ({100*both_wrong/len(disagreed):.1f}%)")

    # ================================================================
    # 3. Response Length Comparison
    # ================================================================
    print("\n" + "-" * 80)
    print("RESPONSE LENGTH COMPARISON")
    print("-" * 80)

    g_lengths_all = [r["g_length"] for r in results if r["g_length"] > 0]
    s_mean_lengths_all = [r["s_mean_length"] for r in results if r["s_mean_length"] > 0]

    if g_lengths_all:
        print(f"\n  Greedy (temp=0, n=1):")
        print(f"    Mean:   {np.mean(g_lengths_all):.0f} tokens")
        print(f"    Median: {np.median(g_lengths_all):.0f} tokens")
        print(f"    Std:    {np.std(g_lengths_all):.0f} tokens")

    if s_mean_lengths_all:
        print(f"\n  Sampling (temp=1, n=64) [mean of 64 per problem]:")
        print(f"    Mean:   {np.mean(s_mean_lengths_all):.0f} tokens")
        print(f"    Median: {np.median(s_mean_lengths_all):.0f} tokens")
        print(f"    Std:    {np.std(s_mean_lengths_all):.0f} tokens")

    if g_lengths_all and s_mean_lengths_all:
        ratio = np.mean(g_lengths_all) / np.mean(s_mean_lengths_all)
        print(f"\n  Length Ratio (Greedy / Sampling Mean): {ratio:.2f}x")

    # Length by correctness
    g_correct_lens = [r["g_length"] for r in results if r["g_correct"] and r["g_length"] > 0]
    g_wrong_lens = [r["g_length"] for r in results if not r["g_correct"] and r["g_length"] > 0]
    s_correct_lens = [r["s_mean_length"] for r in results if r["s_correct"] and r["s_mean_length"] > 0]
    s_wrong_lens = [r["s_mean_length"] for r in results if not r["s_correct"] and r["s_mean_length"] > 0]

    print(f"\n  Length by Correctness:")
    print(f"    {'Setting':<25} {'Correct Mean':<15} {'Wrong Mean':<15} {'Diff':<10}")
    print(f"    {'-'*65}")
    if g_correct_lens and g_wrong_lens:
        gc_m = np.mean(g_correct_lens)
        gw_m = np.mean(g_wrong_lens)
        print(f"    {'Greedy':<25} {gc_m:<15.0f} {gw_m:<15.0f} {gw_m-gc_m:<+10.0f}")
    if s_correct_lens and s_wrong_lens:
        sc_m = np.mean(s_correct_lens)
        sw_m = np.mean(s_wrong_lens)
        print(f"    {'Sampling (mean)':<25} {sc_m:<15.0f} {sw_m:<15.0f} {sw_m-sc_m:<+10.0f}")

    # ================================================================
    # 4. Breakdown by Sampling Consistency (sc_score)
    # ================================================================
    print("\n" + "-" * 80)
    print("BREAKDOWN BY SAMPLING CONSISTENCY (sc_score)")
    print("-" * 80)

    buckets = {
        "Low (0.0 - 0.3)":      lambda r: r["s_sc_score"] <= 0.3,
        "Mid (0.3 - 0.7)":      lambda r: 0.3 < r["s_sc_score"] <= 0.7,
        "High (0.7 - 1.0)":     lambda r: r["s_sc_score"] > 0.7,
    }

    print(f"\n  {'Bucket':<20} {'N':<5} {'Greedy':<10} {'Sampling':<10} {'Pass@N':<10} "
          f"{'Agree%':<8} {'G_len':<8} {'S_len':<8}")
    print(f"  {'-'*79}")

    for bname, bfn in buckets.items():
        bucket = [r for r in results if bfn(r)]
        if not bucket:
            continue
        bn = len(bucket)
        g_c = sum(1 for r in bucket if r["g_correct"])
        s_c = sum(1 for r in bucket if r["s_correct"])
        p_n = sum(1 for r in bucket if r["s_pass_at_n"])
        ag = sum(1 for r in bucket if r["agree"])
        g_l = np.mean([r["g_length"] for r in bucket if r["g_length"] > 0]) if any(r["g_length"] > 0 for r in bucket) else 0
        s_l = np.mean([r["s_mean_length"] for r in bucket if r["s_mean_length"] > 0]) if any(r["s_mean_length"] > 0 for r in bucket) else 0

        print(f"  {bname:<20} {bn:<5} {100*g_c/bn:<9.1f}% {100*s_c/bn:<9.1f}% {100*p_n/bn:<9.1f}% "
              f"{100*ag/bn:<7.1f}% {g_l:<8.0f} {s_l:<8.0f}")

    # ================================================================
    # 5. Greedy-unique insights
    # ================================================================
    print("\n" + "-" * 80)
    print("UNIQUE INSIGHT: Problems Only Greedy Solved (Sampling Failed)")
    print("-" * 80)

    greedy_only = [r for r in results if r["g_correct"] and not r["s_correct"]]
    sampling_only = [r for r in results if not r["g_correct"] and r["s_correct"]]

    print(f"\n  Only Greedy correct:   {len(greedy_only)} problems")
    print(f"  Only Sampling correct: {len(sampling_only)} problems")

    if greedy_only:
        go_sc = [r["s_sc_score"] for r in greedy_only]
        print(f"\n  Greedy-only correct (sc_score of sampling):")
        print(f"    Mean sc_score: {np.mean(go_sc):.3f}")
        print(f"    Median sc_score: {np.median(go_sc):.3f}")
        print(f"    → These are problems where sampling leads to wrong majority despite greedy being right")

    if sampling_only:
        so_sc = [r["s_sc_score"] for r in sampling_only]
        print(f"\n  Sampling-only correct (sc_score):")
        print(f"    Mean sc_score: {np.mean(so_sc):.3f}")
        print(f"    Median sc_score: {np.median(so_sc):.3f}")

    print(f"\n{'='*80}")
    print("Analysis Complete")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
