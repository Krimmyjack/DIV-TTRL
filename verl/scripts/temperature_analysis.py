#!/usr/bin/env python3
"""
温度采样分析脚本

分析不同温度(0.1-0.7)下的采样结果:
1. 每个问题在不同温度下的 voting (majority vote) 正确情况
2. 每个温度的整体正确率
3. 综合所有温度的 pass@k 情况

用法:
    python scripts/temperature_analysis.py --data_dir . --temps 0.1 0.2 0.3 0.4 0.5 0.6 0.7
"""

import json
import argparse
import numpy as np
from collections import Counter
from math import comb
from pathlib import Path


def strip_string(string):
    """标准化答案字符串"""
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


def is_correct(extracted, ground_truth):
    """判断提取的答案是否正确"""
    return strip_string(extracted) == strip_string(ground_truth)


def majority_vote(answers):
    """对一组答案做 majority voting, 返回最多的答案"""
    normalized = [strip_string(a) for a in answers]
    counter = Counter(normalized)
    if not counter:
        return "[NO_ANSWER]"
    return counter.most_common(1)[0][0]


def pass_at_k_estimator(n, c, k):
    """
    无偏 pass@k 估计器
    n: 总采样数
    c: 正确采样数
    k: pass@k 的 k
    """
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def load_jsonl(path):
    """加载 JSONL 文件"""
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def main():
    parser = argparse.ArgumentParser(description="温度采样分析")
    parser.add_argument("--data_dir", type=str, default=".", help="数据文件所在目录")
    parser.add_argument("--temps", type=float, nargs="+",
                        default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0],
                        help="要分析的温度列表")
    parser.add_argument("--samples_per_temp", type=int, default=32,
                        help="每个温度的采样次数")
    parser.add_argument("--pass_k_values", type=int, nargs="+",
                        default=[1, 2, 4, 8, 16, 32, 64, 128, 224],
                        help="要计算的 pass@k 的 k 值")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    temps = sorted(args.temps)

    # ========== 1. 加载所有温度的数据 ==========
    print("=" * 80)
    print("温度采样综合分析")
    print("=" * 80)

    temp_data = {}
    for t in temps:
        fpath = data_dir / f"temp_{t}.jsonl"
        if not fpath.exists():
            print(f"[WARNING] 文件不存在: {fpath}, 跳过温度 {t}")
            continue
        temp_data[t] = load_jsonl(fpath)
        print(f"  加载 temp_{t}.jsonl: {len(temp_data[t])} 个问题")

    if not temp_data:
        print("没有数据文件，退出。")
        return

    # 验证所有文件的问题数一致
    n_problems_list = [len(v) for v in temp_data.values()]
    assert len(set(n_problems_list)) == 1, f"不同温度文件的问题数不一致: {n_problems_list}"
    n_problems = n_problems_list[0]
    active_temps = sorted(temp_data.keys())
    print(f"\n共 {n_problems} 个问题, {len(active_temps)} 个温度: {active_temps}")

    # ========== 2. 每个温度的 Voting 正确率 ==========
    print("\n" + "=" * 80)
    print("Part 1: 每个温度的 Majority Voting 正确率")
    print("=" * 80)

    # voting_correct[t][i] = True/False 表示温度t下第i个问题voting是否正确
    voting_correct = {}
    # pass_counts[t][i] = 在温度t下第i个问题有多少个正确回答
    pass_counts = {}

    for t in active_temps:
        voting_correct[t] = []
        pass_counts[t] = []
        for i, item in enumerate(temp_data[t]):
            gt = item["answer"]
            extracted = item.get("extracted_answers", [])

            # Majority voting
            mv_answer = majority_vote(extracted)
            mv_is_correct = is_correct(mv_answer, gt)
            voting_correct[t].append(mv_is_correct)

            # 统计正确回答数
            n_correct = sum(1 for a in extracted if is_correct(a, gt))
            pass_counts[t].append(n_correct)

    # 打印每个温度的准确率
    print(f"\n{'温度':>8s}  {'Voting正确数':>12s}  {'Voting正确率':>12s}  "
          f"{'平均正确数/32':>14s}  {'Pass@1':>8s}")
    print("-" * 70)
    for t in active_temps:
        n_correct_voting = sum(voting_correct[t])
        acc = n_correct_voting / n_problems
        avg_correct = np.mean(pass_counts[t])
        # pass@1: 从32个中随机选1个正确的概率
        pass1 = np.mean([pass_at_k_estimator(len(temp_data[t][i].get("extracted_answers", [])),
                                             pass_counts[t][i], 1)
                         for i in range(n_problems)])
        print(f"  {t:>6.1f}  {n_correct_voting:>12d}  {acc:>12.4f}  "
              f"{avg_correct:>14.2f}  {pass1:>8.4f}")

    # ========== 3. 每个问题在不同温度下的 Voting 情况 ==========
    print("\n" + "=" * 80)
    print("Part 2: 每个问题在不同温度下的 Voting 正确情况统计")
    print("=" * 80)

    # 统计每个问题在多少个温度上voting正确
    per_question_correct_temps = []
    for i in range(n_problems):
        n_temps_correct = sum(1 for t in active_temps if voting_correct[t][i])
        per_question_correct_temps.append(n_temps_correct)

    n_temps = len(active_temps)
    print(f"\n{'正确温度数':>10s}  {'问题数':>8s}  {'占比':>8s}")
    print("-" * 35)
    for k in range(n_temps + 1):
        count = sum(1 for x in per_question_correct_temps if x == k)
        print(f"  {k:>8d}  {count:>8d}  {count/n_problems:>8.4f}")

    # 所有温度都正确 vs 所有温度都错
    all_correct = sum(1 for x in per_question_correct_temps if x == n_temps)
    all_wrong = sum(1 for x in per_question_correct_temps if x == 0)
    some_correct = n_problems - all_correct - all_wrong
    print(f"\n  所有温度都正确: {all_correct} ({all_correct/n_problems:.4f})")
    print(f"  所有温度都错误: {all_wrong} ({all_wrong/n_problems:.4f})")
    print(f"  部分温度正确:   {some_correct} ({some_correct/n_problems:.4f})")

    # ========== 4. 温度间的 Voting 一致性 ==========
    print("\n" + "=" * 80)
    print("Part 3: 温度间 Voting 答案一致性")
    print("=" * 80)

    # 对于每个问题，统计不同温度的 voting 答案是否一致
    consistent_count = 0
    for i in range(n_problems):
        mv_answers = set()
        for t in active_temps:
            extracted = temp_data[t][i].get("extracted_answers", [])
            mv = majority_vote(extracted)
            mv_answers.add(mv)
        if len(mv_answers) == 1:
            consistent_count += 1

    print(f"\n  所有温度 voting 答案完全一致的问题数: {consistent_count} / {n_problems} "
          f"({consistent_count/n_problems:.4f})")

    # 温度两两一致性矩阵
    if len(active_temps) > 1:
        print(f"\n  温度两两 Voting 一致率矩阵:")
        header = "        " + "  ".join(f"{t:>6.1f}" for t in active_temps)
        print(header)
        for t1 in active_temps:
            row = f"  {t1:>4.1f}  "
            for t2 in active_temps:
                if t1 == t2:
                    row += f"{'---':>8s}"
                else:
                    agree = 0
                    for i in range(n_problems):
                        mv1 = majority_vote(temp_data[t1][i].get("extracted_answers", []))
                        mv2 = majority_vote(temp_data[t2][i].get("extracted_answers", []))
                        if strip_string(mv1) == strip_string(mv2):
                            agree += 1
                    row += f"{agree/n_problems:>8.4f}"
            print(row)

    # ========== 5. 综合温度的 Pass@K ==========
    print("\n" + "=" * 80)
    print("Part 4: 综合所有温度的 Pass@K")
    print("=" * 80)

    # 方法: 将所有温度的采样合并为一个大 pool
    total_samples_per_question = args.samples_per_temp * len(active_temps)
    print(f"\n  每个问题总采样数: {args.samples_per_temp} × {len(active_temps)} = "
          f"{total_samples_per_question}")

    combined_pass_counts = []
    for i in range(n_problems):
        total_correct = sum(pass_counts[t][i] for t in active_temps)
        combined_pass_counts.append(total_correct)

    # 过滤掉超出总采样数的 k 值
    valid_k_values = [k for k in args.pass_k_values if k <= total_samples_per_question]

    print(f"\n  {'k':>6s}  {'Pass@k':>10s}  {'覆盖问题数':>10s}")
    print("  " + "-" * 35)
    for k in valid_k_values:
        pass_k_values = [pass_at_k_estimator(total_samples_per_question,
                                             combined_pass_counts[i], k)
                         for i in range(n_problems)]
        avg_pass_k = np.mean(pass_k_values)
        # 覆盖问题数: pass@k > 0 的问题数 (即至少有1个正确的)
        covered = sum(1 for c in combined_pass_counts if c > 0)
        print(f"  {k:>6d}  {avg_pass_k:>10.4f}  {covered:>10d}")

    # 同时显示单温度的 pass@k 用于对比
    print(f"\n  各温度单独的 Pass@K 对比:")
    header = f"  {'k':>6s}" + "".join(f"  {'T='+str(t):>10s}" for t in active_temps) + f"  {'综合':>10s}"
    print(header)
    print("  " + "-" * (8 + 12 * (len(active_temps) + 1)))
    for k in valid_k_values:
        if k > args.samples_per_temp:
            # 单温度无法计算超出其采样数的 k
            row = f"  {k:>6d}" + "".join(f"  {'N/A':>10s}" for _ in active_temps)
        else:
            row = f"  {k:>6d}"
            for t in active_temps:
                pk = np.mean([pass_at_k_estimator(args.samples_per_temp,
                                                  pass_counts[t][i], k)
                              for i in range(n_problems)])
                row += f"  {pk:>10.4f}"
        # 综合
        if k <= total_samples_per_question:
            combined_pk = np.mean([pass_at_k_estimator(total_samples_per_question,
                                                       combined_pass_counts[i], k)
                                   for i in range(n_problems)])
            row += f"  {combined_pk:>10.4f}"
        print(row)

    # ========== 6. 综合温度的 Voting (跨温度 majority voting) ==========
    print("\n" + "=" * 80)
    print("Part 5: 跨温度综合 Majority Voting")
    print("=" * 80)

    # 策略1: 将所有温度的 extracted_answers 合并后再 majority vote
    combined_voting_correct = 0
    for i in range(n_problems):
        gt = temp_data[active_temps[0]][i]["answer"]
        all_answers = []
        for t in active_temps:
            all_answers.extend(temp_data[t][i].get("extracted_answers", []))
        combined_mv = majority_vote(all_answers)
        if is_correct(combined_mv, gt):
            combined_voting_correct += 1

    print(f"\n  合并所有温度后 Majority Voting 正确率: "
          f"{combined_voting_correct}/{n_problems} = "
          f"{combined_voting_correct/n_problems:.4f}")

    # 策略2: 对各温度的 voting 结果再做 voting (二级 voting)
    meta_voting_correct = 0
    for i in range(n_problems):
        gt = temp_data[active_temps[0]][i]["answer"]
        mv_answers = []
        for t in active_temps:
            extracted = temp_data[t][i].get("extracted_answers", [])
            mv = majority_vote(extracted)
            mv_answers.append(mv)
        meta_mv = majority_vote(mv_answers)
        if is_correct(meta_mv, gt):
            meta_voting_correct += 1

    print(f"  二级 Voting (对各温度voting结果再voting) 正确率: "
          f"{meta_voting_correct}/{n_problems} = "
          f"{meta_voting_correct/n_problems:.4f}")

    # 策略3: 任一温度 voting 正确即算正确 (oracle)
    any_correct = sum(1 for i in range(n_problems)
                      if any(voting_correct[t][i] for t in active_temps))
    print(f"  任一温度 Voting 正确 (Oracle): "
          f"{any_correct}/{n_problems} = "
          f"{any_correct/n_problems:.4f}")

    # ========== 7. 详细问题级别分析 (可选输出) ==========
    print("\n" + "=" * 80)
    print("Part 6: 按难度分组分析")
    print("=" * 80)

    # 按综合正确率分组
    bins = [(0, 0, "全错 (0%)"),
            (0.001, 0.25, "极难 (0-25%)"),
            (0.25, 0.5, "较难 (25-50%)"),
            (0.5, 0.75, "中等 (50-75%)"),
            (0.75, 0.999, "较易 (75-99%)"),
            (1.0, 1.0, "全对 (100%)")]

    combined_accuracies = [c / total_samples_per_question for c in combined_pass_counts]

    print(f"\n  {'难度分组':>18s}  {'问题数':>8s}  {'占比':>8s}  "
          + "".join(f"{'T='+str(t)+' voting':>14s}" for t in active_temps))
    print("  " + "-" * (38 + 14 * len(active_temps)))

    for lo, hi, label in bins:
        indices = [i for i, acc in enumerate(combined_accuracies)
                   if (lo <= acc <= hi if lo == hi else lo <= acc < hi)]
        if not indices:
            continue
        count = len(indices)
        row = f"  {label:>18s}  {count:>8d}  {count/n_problems:>8.4f}"
        for t in active_temps:
            t_acc = sum(1 for i in indices if voting_correct[t][i]) / count if count > 0 else 0
            row += f"  {t_acc:>14.4f}"
        print(row)

    print("\n" + "=" * 80)
    print("分析完成!")
    print("=" * 80)


if __name__ == "__main__":
    main()
