"""
验证伪标签（多数投票）是否偏向较短的回答。
使用项目现有的 test_time_train_metrics / auto_verify / auto_extract 函数。

Usage:
    python scripts/verify_length_label_bias.py \
        --model /path/to/model \
        --data data/MATH-TTT/train.json \
        --n_samples 64 --temperature 1.0 --max_tokens 4096
"""

import argparse
import json
import sys
import os
import numpy as np
from collections import Counter, defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from verl.utils.reward_score.ttrl.ttt_metrics import test_time_train_metrics
from verl.utils.reward_score.ttrl.auto_verify import auto_verify
from verl.utils.reward_score.ttrl.auto_extract import auto_extract


def load_data(data_path):
    with open(data_path, "r", encoding="utf-8") as f:
        return json.load(f)


def generate_responses(model_path, prompts, n_samples, temperature, max_tokens, tp_size=1):
    """使用 vLLM 生成 N 个回答 per prompt."""
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    llm = LLM(model=model_path, tensor_parallel_size=tp_size,
              trust_remote_code=True, max_model_len=max_tokens + 1024)

    sampling_params = SamplingParams(
        n=n_samples, temperature=temperature, max_tokens=max_tokens,
        top_p=0.95 if temperature > 0 else 1.0,
    )

    formatted = [tokenizer.apply_chat_template(p, tokenize=False, add_generation_prompt=True)
                 if isinstance(p, list) else p for p in prompts]

    print(f"Generating {n_samples} responses x {len(formatted)} prompts ...")
    outputs = llm.generate(formatted, sampling_params)
    return [[o.text for o in out.outputs] for out in outputs]


def analyze(all_responses, ground_truths, task="math"):
    """
    利用 test_time_train_metrics 做多数投票 + ground truth 对比，
    然后按 4 个象限统计回答长度。
    """
    categories = defaultdict(list)   # category -> [lengths]
    bucket_data = defaultdict(lambda: defaultdict(list))  # bucket -> metric -> [values]

    for responses, gt in zip(all_responses, ground_truths):
        n = len(responses)
        gt_list = [gt] * n

        # ---------- 复用已有函数 ----------
        # rewards: 与多数投票匹配为 1, 否则为 0
        # ttrl_metrics: label_accuracy, majority_ratio, ground_truth_ratio 等
        rewards, ttrl_metrics = test_time_train_metrics(responses, gt_list, task=task)

        # 真实标签对每个回答的判断
        true_rewards, _ = auto_verify(task, responses, gt_list)

        consistency = ttrl_metrics["majority_ratio"]
        label_acc   = ttrl_metrics["label_accuracy"]
        lengths     = [len(r) for r in responses]

        # ---------- 4 象限分类 ----------
        for i in range(n):
            is_majority = bool(rewards[i])
            is_correct  = bool(true_rewards[i])
            key = ("Maj" if is_majority else "Min") + ("+" if is_correct else "-") + ("Corr" if is_correct else "Incorr")
            if is_majority and is_correct:
                categories["Majority+Correct"].append(lengths[i])
            elif is_majority and not is_correct:
                categories["Majority+Incorrect"].append(lengths[i])
            elif not is_majority and is_correct:
                categories["Minority+Correct"].append(lengths[i])
            else:
                categories["Minority+Incorrect"].append(lengths[i])

        # ---------- 按一致性率分桶 ----------
        bucket = (
            "[0.0,0.3)" if consistency < 0.3 else
            "[0.3,0.5)" if consistency < 0.5 else
            "[0.5,0.7)" if consistency < 0.7 else
            "[0.7,0.9)" if consistency < 0.9 else
            "[0.9,1.0]"
        )
        maj_lens  = [lengths[i] for i in range(n) if rewards[i]]
        corr_lens = [lengths[i] for i in range(n) if true_rewards[i]]
        bucket_data[bucket]["maj_len"].extend(maj_lens)
        bucket_data[bucket]["corr_len"].extend(corr_lens)
        bucket_data[bucket]["label_acc"].append(label_acc)
        bucket_data[bucket]["count"].append(1)

    # ========== 输出 ==========
    print(f"\n{'='*65}")
    print("1. FOUR-CATEGORY LENGTH ANALYSIS")
    print(f"{'='*65}")
    print(f"{'Category':<25} {'Count':>8} {'Avg Len':>10} {'Median':>10}")
    print("-" * 55)
    for cat in ["Majority+Correct", "Majority+Incorrect", "Minority+Correct", "Minority+Incorrect"]:
        lens = categories.get(cat, [])
        if lens:
            print(f"{cat:<25} {len(lens):>8} {np.mean(lens):>10.1f} {np.median(lens):>10.1f}")
        else:
            print(f"{cat:<25} {'N/A':>8}")

    pseudo_pos = categories.get("Majority+Correct", []) + categories.get("Majority+Incorrect", [])
    true_pos   = categories.get("Majority+Correct", []) + categories.get("Minority+Correct", [])

    print(f"\n{'='*65}")
    print("2. PSEUDO-LABEL vs TRUE-LABEL POSITIVE COMPARISON")
    print(f"{'='*65}")
    if pseudo_pos and true_pos:
        pp, tp = np.mean(pseudo_pos), np.mean(true_pos)
        print(f"  Pseudo-positive (majority) avg len : {pp:.1f}")
        print(f"  True-positive  (correct)   avg len : {tp:.1f}")
        print(f"  Diff (True - Pseudo)               : {tp - pp:+.1f}  ({(tp-pp)/pp*100:+.1f}%)")

    print(f"\n{'='*65}")
    print("3. BUCKETED BY CONSISTENCY RATE")
    print(f"{'='*65}")
    print(f"{'Bucket':>12} {'#Prompts':>10} {'LabelAcc':>10} {'MajLen':>10} {'CorrLen':>10} {'Diff':>10}")
    print("-" * 65)
    for b in ["[0.0,0.3)", "[0.3,0.5)", "[0.5,0.7)", "[0.7,0.9)", "[0.9,1.0]"]:
        if b in bucket_data:
            d = bucket_data[b]
            n_p = sum(d["count"])
            la  = np.mean(d["label_acc"])
            ml  = np.mean(d["maj_len"])  if d["maj_len"]  else float("nan")
            cl  = np.mean(d["corr_len"]) if d["corr_len"] else float("nan")
            print(f"{b:>12} {n_p:>10} {la:>10.2f} {ml:>10.1f} {cl:>10.1f} {cl-ml:>+10.1f}")

    # 结论
    print(f"\n{'='*65}")
    print("CONCLUSION")
    print(f"{'='*65}")
    if pseudo_pos and true_pos:
        diff_pct = (np.mean(true_pos) - np.mean(pseudo_pos)) / np.mean(pseudo_pos) * 100
        if diff_pct > 5:
            print(f"  ✅ 假设成立: 真正正确的回答比伪标签正样本长 {diff_pct:.1f}%")
        elif diff_pct < -5:
            print(f"  ❌ 假设不成立: 伪标签正样本反而更长 {-diff_pct:.1f}%")
        else:
            print(f"  ⚠️ 差异不显著 ({diff_pct:+.1f}%)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",       type=str, required=True)
    parser.add_argument("--data",        type=str, default="data/MATH-TTT/train.json")
    parser.add_argument("--n_samples",   type=int, default=64)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max_tokens",  type=int, default=4096)
    parser.add_argument("--tp_size",     type=int, default=1)
    parser.add_argument("--task",        type=str, default="math")
    parser.add_argument("--max_prompts", type=int, default=None)
    args = parser.parse_args()

    data = load_data(args.data)
    if args.max_prompts:
        data = data[:args.max_prompts]

    # 构建 prompts 和 ground_truths
    sys_prompt = "Please reason step by step, and put your final answer within \\boxed{}."
    prompts, gts = [], []
    for item in data:
        q = item.get("prompt", item.get("question", item.get("problem", "")))
        if isinstance(q, list):
            prompts.append(q)
        else:
            prompts.append([{"role": "system", "content": sys_prompt},
                            {"role": "user",   "content": q}])
        gts.append(item.get("answer", item.get("reward_model", {}).get("ground_truth", "")))

    print(f"Loaded {len(prompts)} prompts | Model: {args.model} | N={args.n_samples} | T={args.temperature}")

    all_responses = generate_responses(args.model, prompts, args.n_samples,
                                       args.temperature, args.max_tokens, args.tp_size)
    analyze(all_responses, gts, task=args.task)


if __name__ == "__main__":
    main()
