#!/usr/bin/env python3
"""
Pseudo-Label Selection Strategy Analysis

Compares the accuracy of two different pseudo-label selection strategies:
1. Majority Voting (highest sc_score)
2. Confidence-based Selection (response with lowest token_entropy_approx)

Usage:
    python scripts/analyze_pseudo_labels.py --input_file qwen4b.jsonl
"""

import json
import argparse
import numpy as np
from collections import Counter

def strip_string(string):
    """Normalize answer string to match ground truth."""
    if string is None:
        return ""
    string = str(string).replace("\n", "")
    string = string.replace("\\!", "")
    string = string.replace("\\\\", "\\")
    string = string.replace("tfrac", "frac").replace("dfrac", "frac")
    string = string.replace("\\left", "").replace("\\right", "")
    string = string.replace("^{\\circ}", "").replace("^\\circ", "")
    string = string.replace("\\$", "")
    string = string.replace(" ", "")
    if string == "0.5":
        string = "\\frac{1}{2}"
    return string


def analyze_problem(item):
    """Analyze a single problem's pseudo-label selections."""
    extracted = item.get("extracted_answers", [])
    metrics = item.get("response_metrics", [])
    ground_truth = strip_string(item.get("answer", ""))
    
    # Filter valid answers and track matching metrics
    valid_answers = []
    valid_metrics = []
    
    for i, ans in enumerate(extracted):
        if ans != "[NO_ANSWER]":
            valid_answers.append(strip_string(ans))
            valid_metrics.append(metrics[i] if i < len(metrics) else {})
            
    N = len(valid_answers)
    if N == 0:
        return None
        
    # 1. Majority Voting Selection
    freq = Counter(valid_answers)
    majority_answer = freq.most_common(1)[0][0]
    sc_score = freq[majority_answer] / N
    majority_is_correct = (majority_answer == ground_truth)
    
    # 2. Confidence Selection (Lowest Token Entropy)
    # Filter out responses that don't have an entropy value
    entropy_pairs = [
        (ans, m.get("token_entropy_approx")) 
        for ans, m in zip(valid_answers, valid_metrics) 
        if m.get("token_entropy_approx") is not None
    ]
    
    if not entropy_pairs:
        # Fallback to majority if no entropy data exists
        conf_answer = majority_answer
        min_entropy = None
    else:
        # Select the single response with the lowest entropy
        best_pair = min(entropy_pairs, key=lambda x: x[1])
        conf_answer = best_pair[0]
        min_entropy = best_pair[1]
        
    conf_is_correct = (conf_answer == ground_truth)
    pass_at_n = (ground_truth in valid_answers)
    
    return {
        "N": N,
        "sc_score": sc_score,
        "majority_answer": majority_answer,
        "majority_is_correct": majority_is_correct,
        "conf_answer": conf_answer,
        "min_entropy": min_entropy,
        "conf_is_correct": conf_is_correct,
        "ground_truth": ground_truth,
        "match": (majority_answer == conf_answer),
        "any_is_correct": (majority_is_correct or conf_is_correct),
        "pass_at_n": pass_at_n
    }


def print_summary(results):
    print("\n" + "=" * 80)
    print("PSEUDO-LABEL SELECTION STRATEGY ANALYSIS")
    print("=" * 80)
    
    total = len(results)
    
    # ---- 1. Overall Statistics ----
    maj_correct = sum(1 for r in results if r["majority_is_correct"])
    conf_correct = sum(1 for r in results if r["conf_is_correct"])
    any_correct = sum(1 for r in results if r["any_is_correct"])
    match_count = sum(1 for r in results if r["match"])
    pass_n_count = sum(1 for r in results if r["pass_at_n"])
    
    print(f"\nTotal Problems Analyzed: {total}")
    print(f"Majority Voting Acc:          {maj_correct:>4}/{total} ({100*maj_correct/total:>5.1f}%)")
    print(f"Confidence (Min Entropy) Acc: {conf_correct:>4}/{total} ({100*conf_correct/total:>5.1f}%)")
    print(f"Combined Coverage (Either):   {any_correct:>4}/{total} ({100*any_correct/total:>5.1f}%)")
    print(f"Pass@N (GT in any response):  {pass_n_count:>4}/{total} ({100*pass_n_count/total:>5.1f}%)")
    print(f"Strategies Agreed:            {match_count:>4}/{total} ({100*match_count/total:>5.1f}%)")
    
    # ---- 2. Binning by Consistency Rate (sc_score) ----
    buckets = {
        "Low (0.0 - 0.1)": lambda r: r["sc_score"] <= 0.1,
        "Low-Mid (0.1 - 0.2)": lambda r: 0.1 < r["sc_score"] <= 0.2,
        "mid (0.2 - 0.3)":lambda r: 0.2 < r["sc_score"] <= 0.3,
        "high (0.3 - 0.4)":lambda r: 0.3 < r["sc_score"] <= 0.4,
        "High-Mid (0.4 - 0.7)": lambda r: 0.4 < r["sc_score"] <= 0.7,
        "High (0.7 - 1.0)": lambda r: r["sc_score"] > 0.7,
    }
    
    print("\n" + "-" * 80)
    print("Accuracy by Consistency Rate (sc_score) Bins")
    print("-" * 80)
    
    for bucket_name, bucket_fn in buckets.items():
        bucket_results = [r for r in results if bucket_fn(r)]
        if not bucket_results:
            continue
            
        bn = len(bucket_results)
        b_maj = sum(1 for r in bucket_results if r["majority_is_correct"])
        b_conf = sum(1 for r in bucket_results if r["conf_is_correct"])
        b_any = sum(1 for r in bucket_results if r["any_is_correct"])
        b_match = sum(1 for r in bucket_results if r["match"])
        b_pass_n = sum(1 for r in bucket_results if r["pass_at_n"])
        
        diff = b_conf - b_maj
        diff_str = f"{diff:+d}" if diff != 0 else "0"
        
        print(f"\n  [{bucket_name}] ({bn} problems)")
        print(f"    Majority Acc:   {b_maj:>4}/{bn} ({100*b_maj/bn:>5.1f}%)")
        print(f"    Confidence Acc: {b_conf:>4}/{bn} ({100*b_conf/bn:>5.1f}%)  [Diff: {diff_str}]")
        print(f"    Combined (Any): {b_any:>4}/{bn} ({100*b_any/bn:>5.1f}%)  [Lift: {b_any - b_maj:+d}]")
        print(f"    Pass@N (Upper): {b_pass_n:>4}/{bn} ({100*b_pass_n/bn:>5.1f}%)")
        print(f"    Agreement Rate: {100*b_match/bn:>5.1f}%")

    
    # ---- 3. Hard Problem Analysis (Where Majority Fails) ----
    print("\n" + "-" * 80)
    print("Analysis on Hard Problems (Where Majority Voting is WRONG)")
    print("-" * 80)
    
    maj_wrong_results = [r for r in results if not r["majority_is_correct"]]
    mw_total = len(maj_wrong_results)
    
    if mw_total > 0:
        conf_rescued = sum(1 for r in maj_wrong_results if r["conf_is_correct"])
        print(f"\n  Total Problems where Majority failed: {mw_total}")
        print(f"  Rescued by Confidence Selection:      {conf_rescued}/{mw_total} ({100*conf_rescued/mw_total:.1f}%)")
        
        if conf_rescued > 0:
            print("\n  Examples of Rescued Problems:")
            rescued_examples = [r for r in maj_wrong_results if r["conf_is_correct"]][:3]
            for i, r in enumerate(rescued_examples):
                print(f"    Example {i+1}: sc_score={r['sc_score']:.2f}")
                print(f"      Majority:   '{r['majority_answer']}'")
                print(f"      Confidence: '{r['conf_answer']}' (entropy: {r['min_entropy']:.3f})")
                print(f"      GT:         '{r['ground_truth']}'")
    else:
        print("\n  Majority Voting never failed.")


def main():
    parser = argparse.ArgumentParser(description="Pseudo-Label Selection Analysis")
    parser.add_argument("--input_file", type=str, required=True, help="JSONL file with rollout metrics")
    args = parser.parse_args()
    
    # Load Data
    print(f"Loading data from {args.input_file}")
    data = []
    with open(args.input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
                
    print(f"Loaded {len(data)} problems")
    
    # Analyze
    results = []
    for item in data:
        res = analyze_problem(item)
        if res is not None:
            results.append(res)
            
    print(f"Successfully analyzed {len(results)} valid problems")
    
    # Show Summary
    print_summary(results)

if __name__ == "__main__":
    main()
