"""
Bootstrap Top-K Pseudo-label Set Analysis

Usage:
    python scripts/bootstrap_analysis.py \
        --input_file qwen64.jsonl \
        --num_bootstrap 64

输入: 生成脚本输出的 JSONL，每行包含 extracted_answers (list) 和 answer (ground truth)
输出: 各种统计指标
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
def bootstrap_majority(answers, B=200, sample_size=None):
    """
    对 answers 做 B 次 bootstrap 采样，每次抽 sample_size 个(有放回)，取 majority。
    
    Returns:
        bootstrap_majorities: list of B majority answers
        answer_confidence: dict {answer: P(answer is bootstrap majority)}
        sample_agreement: dict {answer: P(sample with this answer matches bootstrap majority)}
    """
    N = len(answers)
    if sample_size is None:
        sample_size = N - 1  # 默认抽 N-1 个
    
    bootstrap_majorities = []
    majority_ratios = []  # 每次 bootstrap 中 majority 票数占 sample_size 的比例
    for _ in range(B):
        subset = random.choices(answers, k=sample_size)
        cnt = Counter(subset)
        maj, maj_count = cnt.most_common(1)[0]
        bootstrap_majorities.append(maj)
        majority_ratios.append(maj_count / sample_size)
    
    # 每个答案成为 bootstrap majority 的概率
    maj_counter = Counter(bootstrap_majorities)
    answer_confidence = {ans: count / B for ans, count in maj_counter.items()}
    
    # 每个样本的答案与 bootstrap majority 一致的比例
    unique_answers = set(answers)
    sample_agreement = {}
    for ans in unique_answers:
        sample_agreement[ans] = answer_confidence.get(ans, 0.0)
    
    return bootstrap_majorities, answer_confidence, sample_agreement, majority_ratios


def analyze_problem(extracted_answers, ground_truth, B=200, K=3):
    """分析单个问题的 bootstrap 稳定性"""
    
    # 过滤掉 [NO_ANSWER]
    valid_answers = [a for a in extracted_answers if a != "[NO_ANSWER]"]
    N = len(valid_answers)
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

    bootstrap_majorities, answer_confidence, sample_agreement, majority_ratios = bootstrap_majority(
        valid_answers, B=B, sample_size= 32
    )
    
    # Bootstrap majority 票数占 sample_size 的平均比例
    avg_bootstrap_majority_ratio = np.mean(majority_ratios)
    
    # Bootstrap 稳定性: majority 与原始 majority 一致的比例
    bootstrap_stability = sum(1 for m in bootstrap_majorities if m == original_majority) / B
    
    # Bootstrap 中出现了多少种不同的 majority answer
    n_distinct_majorities = len(set(bootstrap_majorities))
    
    # 原始 majority 的 bootstrap confidence
    majority_bootstrap_conf = answer_confidence.get(original_majority, 0.0)
    
    # Ground truth 的 bootstrap confidence
    gt_bootstrap_conf = answer_confidence.get(gt_norm, 0.0)
    
    # 唯一答案数
    n_unique_answers = len(freq)
    
    # 每个样本的 bootstrap-based soft reward
    soft_rewards = [sample_agreement.get(a, 0.0) for a in valid_answers]
    
    # 对比: 原始二元 reward vs soft reward
    binary_rewards = [1.0 if a == original_majority else 0.0 for a in valid_answers]
    
    # ---- Bootstrap pass@n / avg@n ----
    # pass@n: 在 B 次 bootstrap 中，GT 至少出现一次作为 majority → 1，否则 → 0
    # avg@n:  在 B 次 bootstrap 中，GT 成为 majority 的比例
    gt_hits = sum(1 for m in bootstrap_majorities if m == gt_norm)
    bootstrap_pass_at_n = 1.0 if gt_hits > 0 else 0.0
    bootstrap_avg_at_n = gt_hits / B
    
    # ---- Top-K pass@k / avg@k ----
    # 取原始投票计数的前 K 个答案作为伪标签候选集
    top_k_with_counts = freq.most_common(K)  # [(answer, count), ...]
    top_k_answers = [ans for ans, _ in top_k_with_counts]
    top_k_total_votes = sum(cnt for _, cnt in top_k_with_counts)
    # pass@k: GT 是否在 top-K 候选集中
    topk_pass_at_k = 1.0 if gt_norm in top_k_answers else 0.0
    # avg@k: GT 的投票计数 / top-K 答案的总投票计数
    gt_votes_in_topk = freq[gt_norm] if gt_norm in top_k_answers else 0
    topk_avg_at_k = gt_votes_in_topk / top_k_total_votes if top_k_total_votes > 0 else 0.0
    
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
        "bootstrap_pass_at_n": bootstrap_pass_at_n,
        "bootstrap_avg_at_n": bootstrap_avg_at_n,
        "topk_pass_at_k": topk_pass_at_k,
        "topk_avg_at_k": topk_avg_at_k,
        "top_k_answers": top_k_answers,
        "avg_bootstrap_majority_ratio": avg_bootstrap_majority_ratio,
    }


def print_results(results, B, K):
    total = len(results)
    
    # ---- 1. 总体统计 ----
    n_correct = sum(1 for r in results if r["majority_is_correct"])
    avg_consistency = np.mean([r["original_consistency"] for r in results])
    avg_stability = np.mean([r["bootstrap_stability"] for r in results])
    avg_distinct = np.mean([r["n_distinct_majorities"] for r in results])
    avg_bs_maj_ratio = np.mean([r["avg_bootstrap_majority_ratio"] for r in results])
    
    print(f"\n总问题数: {total}")
    print(f"Majority 正确率: {n_correct}/{total} ({100*n_correct/total:.1f}%)")
    print(f"平均 consistency rate: {avg_consistency:.3f}")
    print(f"平均 bootstrap 稳定性: {avg_stability:.3f}")
    print(f"平均 bootstrap 中不同 majority 数: {avg_distinct:.1f}")
    print(f"平均 bootstrap majority 样本占比: {avg_bs_maj_ratio:.3f}")
    
    # ---- 2. 按 consistency 分桶 ----
    buckets = {
        "Low (0.0 - 0.3)": lambda r: r["original_consistency"] <= 0.3,
        # "Low-Mid (0.1 - 0.2)": lambda r: 0.1 < r["original_consistency"] <= 0.2,
        # "mid (0.2 - 0.3)":lambda r: 0.2 < r["original_consistency"] <= 0.3,
        "high (0.3 - 0.4)":lambda r: 0.3 < r["original_consistency"] <= 0.4,
        "High-Mid (0.4 - 0.7)": lambda r: 0.4 < r["original_consistency"] <= 0.7,
        "High (0.7 - 1.0)": lambda r: r["original_consistency"] > 0.7,
    }
    
    print("\n" + "-" * 80)
    print("按 Consistency Rate 分桶")
    print("-" * 80)
    
    for bucket_name, bucket_fn in buckets.items():
        bucket_results = [r for r in results if bucket_fn(r)]
        if not bucket_results:
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
        
        bn = len(bucket_results)
        b_correct = sum(1 for r in bucket_results if r["majority_is_correct"])
        b_stability = np.mean([r["bootstrap_stability"] for r in bucket_results])
        b_distinct = np.mean([r["n_distinct_majorities"] for r in bucket_results])
        b_maj_conf = np.mean([r["majority_bootstrap_conf"] for r in bucket_results])
        b_gt_conf = np.mean([r["gt_bootstrap_conf"] for r in bucket_results])
        b_bs_pass = np.mean([r["bootstrap_pass_at_n"] for r in bucket_results])
        b_bs_avg = np.mean([r["bootstrap_avg_at_n"] for r in bucket_results])
        b_topk_pass = np.mean([r["topk_pass_at_k"] for r in bucket_results])
        b_topk_avg = np.mean([r["topk_avg_at_k"] for r in bucket_results])
        b_bs_maj_ratio = np.mean([r["avg_bootstrap_majority_ratio"] for r in bucket_results])
        
        print(f"\n  [{bucket_name}] ({bn} 题)")
        print(f"    Majority 正确率:        {b_correct}/{bn} ({100*b_correct/bn:.1f}%)")
        print(f"    Bootstrap 稳定性:       {b_stability:.3f}")
        print(f"    不同 majority 数 (avg): {b_distinct:.1f}")
        print(f"    Majority bootstrap 置信度: {b_maj_conf:.3f}")
        print(f"    Ground truth bootstrap 置信度: {b_gt_conf:.3f}")
        print(f"    Bootstrap majority 样本占比: {b_bs_maj_ratio:.3f}")
        print(f"    Bootstrap pass@n:       {b_bs_pass:.3f}")
        print(f"    Bootstrap avg@n:        {b_bs_avg:.3f}")
        print(f"    Top-{K} pass@{K}:            {b_topk_pass:.3f}")
        print(f"    Top-{K} avg@{K}:             {b_topk_avg:.3f}")
    
    # ---- 3. Bootstrap 能否识别出错误的 majority? ----
    print("\n" + "-" * 80)
    print("Bootstrap 置信度 vs Majority 正确性")
    print("-" * 80)
    
    correct_confs = [r["majority_bootstrap_conf"] for r in results if r["majority_is_correct"]]
    wrong_confs = [r["majority_bootstrap_conf"] for r in results if not r["majority_is_correct"]]
    
    if correct_confs:
        print(f"\n  Majority 正确时, bootstrap 置信度: {np.mean(correct_confs):.3f} (n={len(correct_confs)})")
    if wrong_confs:
        print(f"  Majority 错误时, bootstrap 置信度: {np.mean(wrong_confs):.3f} (n={len(wrong_confs)})")
    if correct_confs and wrong_confs:
        print(f"  差值: {np.mean(correct_confs) - np.mean(wrong_confs):+.3f}")
        print(f"  → Bootstrap 置信度{'能' if np.mean(correct_confs) > np.mean(wrong_confs) else '不能'}区分正确/错误的 majority")
    
    # ---- 4. 二元 reward vs Soft reward 对比 ----
    print("\n" + "-" * 80)
    print("二元 Reward vs Bootstrap Soft Reward 对比")
    print("-" * 80)
    
    for bucket_name, bucket_fn in buckets.items():
        bucket_results = [r for r in results if bucket_fn(r)]
        if not bucket_results:
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
    
    # ---- 6. Bootstrap pass@n / avg@n 与 Top-K pass@k / avg@k ----
    print("\n" + "-" * 80)
    print(f"Bootstrap pass@n / avg@n  vs  Top-{K} pass@{K} / avg@{K}")
    print("-" * 80)
    
    # 总体
    overall_bs_pass = np.mean([r["bootstrap_pass_at_n"] for r in results])
    overall_bs_avg = np.mean([r["bootstrap_avg_at_n"] for r in results])
    overall_topk_pass = np.mean([r["topk_pass_at_k"] for r in results])
    overall_topk_avg = np.mean([r["topk_avg_at_k"] for r in results])
    # 对照: 原始 majority (Top-1) 正确率
    overall_top1 = np.mean([r["majority_is_correct"] for r in results])
    
    print(f"\n  {'指标':<35} {'总体':>10}")
    print(f"  {'─' * 45}")
    print(f"  {'Top-1 majority 正确率':<35} {overall_top1:>10.3f}")
    print(f"  {f'Top-{K} pass@{K} (GT in Top-{K})':<35} {overall_topk_pass:>10.3f}")
    print(f"  {f'Top-{K} avg@{K} (GT / K)':<35} {overall_topk_avg:>10.3f}")
    print(f"  {'Bootstrap pass@n (GT 至少出现一次)':<35} {overall_bs_pass:>10.3f}")
    print(f"  {'Bootstrap avg@n (GT 出现比例)':<35} {overall_bs_avg:>10.3f}")
    
    # 分桶
    buckets_topk = {
        "低 (0-0.3)": lambda r: r["original_consistency"] <= 0.3,
        "中 (0.3-0.7)": lambda r: 0.3 < r["original_consistency"] <= 0.7,
        "高 (0.7-1.0)": lambda r: r["original_consistency"] > 0.7,
    }
    
    print(f"\n  按 Consistency Rate 分桶:")
    print(f"  {'桶':<15} {'n':>5} {'Top-1':>8} {f'pass@{K}':>8} {f'avg@{K}':>8} {'BS pass':>8} {'BS avg':>8}")
    print(f"  {'─' * 60}")
    for bname, bfn in buckets_topk.items():
        br = [r for r in results if bfn(r)]
        if not br:
            continue
        bn = len(br)
        t1 = np.mean([r["majority_is_correct"] for r in br])
        tp = np.mean([r["topk_pass_at_k"] for r in br])
        ta = np.mean([r["topk_avg_at_k"] for r in br])
        bp = np.mean([r["bootstrap_pass_at_n"] for r in br])
        ba = np.mean([r["bootstrap_avg_at_n"] for r in br])
        print(f"  {bname:<15} {bn:>5} {t1:>8.3f} {tp:>8.3f} {ta:>8.3f} {bp:>8.3f} {ba:>8.3f}")
    
    # Top-K 提升分析
    topk_lift = overall_topk_pass - overall_top1
    bs_lift = overall_bs_pass - overall_top1
    print(f"\n  Top-{K} 相比 Top-1 的 pass 提升: {topk_lift:+.3f} ({100*topk_lift:.1f}pp)")
    print(f"  Bootstrap 相比 Top-1 的 pass 提升: {bs_lift:+.3f} ({100*bs_lift:.1f}pp)")


def main():
    parser = argparse.ArgumentParser(description="Bootstrap 伪标签分析")
    parser.add_argument("--input_file", type=str, required=True, help="生成脚本输出的 JSONL 文件")
    parser.add_argument("--num_bootstrap", type=int, default=200, help="Bootstrap 迭代次数")
    parser.add_argument("--K", type=int, default=5, help="取投票前 K 个答案作为伪标签候选集")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
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
        extracted = item.get("extracted_answers", [])
        gt = item.get("answer", "")
        
        result = analyze_problem(extracted, gt, B=args.num_bootstrap, K=args.K)
        if result is not None:
            results.append(result)
    
    print(f"Successfully analyzed {len(results)} problems")
    
    # 打印结果
    print_summary(results, args.num_bootstrap, args.K)
        r = analyze_problem(item.get("extracted_answers", []),
                            item.get("answer", ""), B=args.num_bootstrap)
        if r is not None:
            results.append(r)
    print(f"Analyzed {len(results)} problems")

    print_results(results, args.num_bootstrap)


if __name__ == "__main__":
    main()
