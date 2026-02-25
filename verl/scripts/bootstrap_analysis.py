"""
Bootstrap 伪标签稳定性分析

用法:
    python scripts/bootstrap_analysis.py \
        --input_file math500_candidates.jsonl \
        --num_bootstrap 200

输入: 生成脚本输出的 JSONL，每行包含 extracted_answers (list) 和 answer (ground truth)
输出: 各种统计指标
"""

import json
import argparse
import random
import numpy as np
from collections import Counter


def strip_string(string):
    """与生成脚本中保持一致的标准化"""
    if string is None:
        return ""
    string = string.replace("\n", "")
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
    for _ in range(B):
        subset = random.choices(answers, k=sample_size)
        maj = Counter(subset).most_common(1)[0][0]
        bootstrap_majorities.append(maj)
    
    # 每个答案成为 bootstrap majority 的概率
    maj_counter = Counter(bootstrap_majorities)
    answer_confidence = {ans: count / B for ans, count in maj_counter.items()}
    
    # 每个样本的答案与 bootstrap majority 一致的比例
    unique_answers = set(answers)
    sample_agreement = {}
    for ans in unique_answers:
        sample_agreement[ans] = answer_confidence.get(ans, 0.0)
    
    return bootstrap_majorities, answer_confidence, sample_agreement


def analyze_problem(extracted_answers, ground_truth, B=200):
    """分析单个问题的 bootstrap 稳定性"""
    
    # 过滤掉 [NO_ANSWER]
    valid_answers = [a for a in extracted_answers if a != "[NO_ANSWER]"]
    N = len(valid_answers)
    if N == 0:
        return None
    
    # 原始 majority vote
    freq = Counter(valid_answers)
    original_majority = freq.most_common(1)[0][0]
    original_consistency = freq[original_majority] / N
    
    # Ground truth 标准化
    gt_norm = strip_string(str(ground_truth))
    majority_is_correct = (original_majority == gt_norm)
    
    # Bootstrap
    bootstrap_majorities, answer_confidence, sample_agreement = bootstrap_majority(
        valid_answers, B=B
    )
    
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
    
    return {
        "N": N,
        "n_unique_answers": n_unique_answers,
        "original_majority": original_majority,
        "original_consistency": original_consistency,
        "majority_is_correct": majority_is_correct,
        "ground_truth": gt_norm,
        "bootstrap_stability": bootstrap_stability,
        "n_distinct_majorities": n_distinct_majorities,
        "majority_bootstrap_conf": majority_bootstrap_conf,
        "gt_bootstrap_conf": gt_bootstrap_conf,
        "answer_confidence": answer_confidence,
        "soft_rewards": soft_rewards,
        "binary_rewards": binary_rewards,
    }


def print_summary(results, B):
    print("\n" + "=" * 80)
    print(f"BOOTSTRAP 伪标签分析 (B={B})")
    print("=" * 80)
    
    total = len(results)
    
    # ---- 1. 总体统计 ----
    n_correct = sum(1 for r in results if r["majority_is_correct"])
    avg_consistency = np.mean([r["original_consistency"] for r in results])
    avg_stability = np.mean([r["bootstrap_stability"] for r in results])
    avg_distinct = np.mean([r["n_distinct_majorities"] for r in results])
    
    print(f"\n总问题数: {total}")
    print(f"Majority 正确率: {n_correct}/{total} ({100*n_correct/total:.1f}%)")
    print(f"平均 consistency rate: {avg_consistency:.3f}")
    print(f"平均 bootstrap 稳定性: {avg_stability:.3f}")
    print(f"平均 bootstrap 中不同 majority 数: {avg_distinct:.1f}")
    
    # ---- 2. 按 consistency 分桶 ----
    buckets = {
        "低 (0-0.3)": lambda r: r["original_consistency"] <= 0.3,
        "中 (0.3-0.7)": lambda r: 0.3 < r["original_consistency"] <= 0.7,
        "高 (0.7-1.0)": lambda r: r["original_consistency"] > 0.7,
    }
    
    print("\n" + "-" * 80)
    print("按 Consistency Rate 分桶")
    print("-" * 80)
    
    for bucket_name, bucket_fn in buckets.items():
        bucket_results = [r for r in results if bucket_fn(r)]
        if not bucket_results:
            continue
        
        bn = len(bucket_results)
        b_correct = sum(1 for r in bucket_results if r["majority_is_correct"])
        b_stability = np.mean([r["bootstrap_stability"] for r in bucket_results])
        b_distinct = np.mean([r["n_distinct_majorities"] for r in bucket_results])
        b_maj_conf = np.mean([r["majority_bootstrap_conf"] for r in bucket_results])
        b_gt_conf = np.mean([r["gt_bootstrap_conf"] for r in bucket_results])
        
        print(f"\n  [{bucket_name}] ({bn} 题)")
        print(f"    Majority 正确率:        {b_correct}/{bn} ({100*b_correct/bn:.1f}%)")
        print(f"    Bootstrap 稳定性:       {b_stability:.3f}")
        print(f"    不同 majority 数 (avg): {b_distinct:.1f}")
        print(f"    Majority bootstrap 置信度: {b_maj_conf:.3f}")
        print(f"    Ground truth bootstrap 置信度: {b_gt_conf:.3f}")
    
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
            continue
        
        all_binary = []
        all_soft = []
        for r in bucket_results:
            all_binary.extend(r["binary_rewards"])
            all_soft.extend(r["soft_rewards"])
        
        print(f"\n  [{bucket_name}]")
        print(f"    二元 reward — 均值: {np.mean(all_binary):.3f}, 非零比例: {np.mean([b > 0 for b in all_binary]):.3f}")
        print(f"    Soft reward — 均值: {np.mean(all_soft):.3f}, 范围: [{np.min(all_soft):.3f}, {np.max(all_soft):.3f}]")
        print(f"    Soft reward 标准差: {np.std(all_soft):.3f}")
    
    # ---- 5. 低一致性问题的详细分析 ----
    print("\n" + "-" * 80)
    print("低一致性问题详细分析 (consistency ≤ 0.3)")
    print("-" * 80)
    
    low_cons = [r for r in results if r["original_consistency"] <= 0.3]
    if low_cons:
        # 有多少题的 bootstrap majority 和原始 majority 不同？
        unstable = [r for r in low_cons if r["bootstrap_stability"] < 0.5]
        print(f"\n  总计: {len(low_cons)} 题")
        print(f"  Bootstrap 不稳定 (stability < 0.5): {len(unstable)} 题 ({100*len(unstable)/len(low_cons):.1f}%)")
        
        # 在不稳定题中，ground truth 是否出现在 bootstrap majority 中？
        gt_found = sum(1 for r in unstable if r["gt_bootstrap_conf"] > 0)
        print(f"  其中 ground truth 出现在 bootstrap majority 中: {gt_found}/{len(unstable)}")
        
        # 展示几个例子
        print(f"\n  前 5 个低一致性问题的 bootstrap 分布:")
        for i, r in enumerate(low_cons[:5]):
            print(f"\n    问题 {i+1}: consistency={r['original_consistency']:.2f}, "
                  f"majority正确={r['majority_is_correct']}")
            print(f"    原始 majority: '{r['original_majority']}', "
                  f"ground truth: '{r['ground_truth']}'")
            # 展示 top-3 bootstrap answer confidence
            sorted_conf = sorted(r["answer_confidence"].items(), key=lambda x: -x[1])[:3]
            for ans, conf in sorted_conf:
                is_gt = " ← GT" if ans == r["ground_truth"] else ""
                is_maj = " ← MAJ" if ans == r["original_majority"] else ""
                print(f"      '{ans}': {conf:.3f}{is_gt}{is_maj}")


def main():
    parser = argparse.ArgumentParser(description="Bootstrap 伪标签分析")
    parser.add_argument("--input_file", type=str, required=True, help="生成脚本输出的 JSONL 文件")
    parser.add_argument("--num_bootstrap", type=int, default=200, help="Bootstrap 迭代次数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # 加载数据
    print(f"Loading data from {args.input_file}")
    data = []
    with open(args.input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    
    print(f"Loaded {len(data)} problems")
    
    # 分析每个问题
    results = []
    for item in data:
        extracted = item.get("extracted_answers", [])
        gt = item.get("answer", "")
        
        result = analyze_problem(extracted, gt, B=args.num_bootstrap)
        if result is not None:
            results.append(result)
    
    print(f"Successfully analyzed {len(results)} problems")
    
    # 打印结果
    print_summary(results, args.num_bootstrap)


if __name__ == "__main__":
    main()
