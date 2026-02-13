#!/usr/bin/env python3
"""
Analyze the hypothesis: "Do pseudo-labels (majority vote) bias toward shorter responses?"

This script loads a model, generates N responses per problem, and compares:
1. Ground-truth correct vs. incorrect response lengths
2. Majority-matching vs. non-majority response lengths
3. Per-problem breakdown by consistency rate

Usage:
    python scripts/analyze_length_hypothesis.py \
        --model_path /path/to/model \
        --data_path data/MATH-TTT/train.json \
        --n_samples 64 \
        --temperature 1.0 \
        --max_tokens 4096 \
        --output_path results/length_analysis.json
"""

import argparse
import json
import multiprocessing
import os
import sys
from collections import Counter, defaultdict
from functools import partial

# Fix: CUDA cannot re-initialize in forked subprocess
multiprocessing.set_start_method("spawn", force=True)

import numpy as np

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from verl.utils.reward_score.ttrl.auto_verify import auto_verify
from verl.utils.reward_score.ttrl.auto_extract import auto_extract
from verl.utils.reward_score.ttrl.latex_clean import normalize_latex
from verl.utils.reward_score.ttrl.qwen.qwen_math_parser import extract_answer


def load_data(data_path):
    """Load problems from JSON file."""
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    problems = []
    for item in data:
        prompt = item.get("prompt", None)
        if prompt is None:
            # Try other common formats
            prompt = item.get("question", item.get("problem", ""))
        
        # Handle list-of-dicts format (chat template)
        if isinstance(prompt, list):
            # Extract the user message content
            for msg in prompt:
                if msg.get("role") == "user":
                    prompt = msg["content"]
                    break
            else:
                prompt = str(prompt)
        
        ground_truth = item.get("reward_model", {}).get("ground_truth", 
                       item.get("answer", item.get("ground_truth", "")))
        
        problems.append({
            "prompt": prompt,
            "ground_truth": str(ground_truth),
            "data_source": item.get("data_source", "MATH-TTT"),
        })
    
    print(f"Loaded {len(problems)} problems from {data_path}")
    return problems


def generate_responses(model_path, problems, n_samples, temperature, max_tokens, tp_size=1):
    """Generate N responses per problem using vLLM."""
    from vllm import LLM, SamplingParams
    
    print(f"Loading model from {model_path}...")
    llm = LLM(
        model=model_path,
        tensor_parallel_size=tp_size,
        trust_remote_code=True,
        max_model_len=max_tokens + 1024,  # prompt + response
    )
    
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        n=n_samples,
    )
    
    prompts = [p["prompt"] for p in problems]
    print(f"Generating {n_samples} responses for {len(prompts)} problems...")
    
    outputs = llm.generate(prompts, sampling_params)
    
    # Organize results
    all_responses = []
    for i, output in enumerate(outputs):
        responses = [o.text for o in output.outputs]
        all_responses.append(responses)
    
    return all_responses


def analyze_prompt_group(responses, ground_truth, task="math"):
    """
    Analyze a single prompt's responses.
    
    Returns dict with per-response info and group-level statistics.
    """
    n = len(responses)
    
    # Extract final answers
    extract_fn = partial(extract_answer, data_name=task)
    normalized = [normalize_latex(r) for r in responses]
    final_answers = [extract_fn(text) or "<empty>" for text in normalized]
    
    # Majority vote
    freq = Counter(final_answers)
    majority_answer, majority_count = freq.most_common(1)[0]
    consistency_rate = majority_count / n
    
    # Ground truth verification
    true_rewards, _ = auto_verify(task, responses, [ground_truth] * n)
    
    # Majority matching
    majority_rewards, _ = auto_verify(task, responses, [majority_answer] * n)
    
    # Check if majority vote is correct
    majority_correct_check, _ = auto_verify(task, [majority_answer], [ground_truth])
    majority_is_correct = majority_correct_check[0] > 0
    
    # Compute length of each response (in characters, as we don't have tokenizer here)
    response_lengths = [len(r) for r in responses]
    
    # Categorize each response
    per_response = []
    for i in range(n):
        per_response.append({
            "length": response_lengths[i],
            "is_correct": bool(true_rewards[i] > 0),  # matches ground truth
            "matches_majority": bool(majority_rewards[i] > 0),  # matches majority vote
            "final_answer": final_answers[i],
        })
    
    # Group statistics
    correct_lengths = [r["length"] for r in per_response if r["is_correct"]]
    incorrect_lengths = [r["length"] for r in per_response if not r["is_correct"]]
    majority_lengths = [r["length"] for r in per_response if r["matches_majority"]]
    non_majority_lengths = [r["length"] for r in per_response if not r["matches_majority"]]
    
    return {
        "consistency_rate": consistency_rate,
        "majority_is_correct": majority_is_correct,
        "ground_truth_ratio": sum(1 for r in per_response if r["is_correct"]) / n,
        "unique_answers": len(freq),
        "n_correct": len(correct_lengths),
        "n_incorrect": len(incorrect_lengths),
        "n_majority": len(majority_lengths),
        "n_non_majority": len(non_majority_lengths),
        "avg_len_correct": np.mean(correct_lengths) if correct_lengths else 0,
        "avg_len_incorrect": np.mean(incorrect_lengths) if incorrect_lengths else 0,
        "avg_len_majority": np.mean(majority_lengths) if majority_lengths else 0,
        "avg_len_non_majority": np.mean(non_majority_lengths) if non_majority_lengths else 0,
        "avg_len_all": np.mean(response_lengths),
        "per_response": per_response,
    }


def print_summary(all_results):
    """Print a comprehensive summary of the analysis."""
    
    print("\n" + "=" * 80)
    print("LENGTH HYPOTHESIS ANALYSIS RESULTS")
    print("=" * 80)
    
    # Collect per-response data split by majority correctness
    # "positive" = matches majority vote (what Pass@k TTA treats as correct)
    # "negative" = does not match majority vote
    maj_correct = {"pos": [], "neg": [], "all": [], "count": 0}
    maj_wrong   = {"pos": [], "neg": [], "all": [], "count": 0}
    overall     = {"pos": [], "neg": [], "all": [], "count": 0}
    
    for r in all_results:
        bucket = maj_correct if r["majority_is_correct"] else maj_wrong
        bucket["count"] += 1
        overall["count"] += 1
        
        for pr in r["per_response"]:
            length = pr["length"]
            bucket["all"].append(length)
            overall["all"].append(length)
            if pr["matches_majority"]:
                bucket["pos"].append(length)
                overall["pos"].append(length)
            else:
                bucket["neg"].append(length)
                overall["neg"].append(length)
    
    def _fmt(lens):
        if not lens:
            return "N/A"
        return f"{np.mean(lens):.0f} (n={len(lens)})"
    
    def _print_block(label, data):
        print(f"\n  {label} ({data['count']} problems)")
        print(f"    正样本（匹配多数）平均长度:   {_fmt(data['pos'])}")
        print(f"    负样本（不匹配多数）平均长度: {_fmt(data['neg'])}")
        print(f"    总平均长度:                   {_fmt(data['all'])}")
        if data["pos"] and data["neg"]:
            d = np.mean(data["pos"]) - np.mean(data["neg"])
            print(f"    Δ (正 - 负):                  {d:+.0f} chars ({'正样本更长' if d > 0 else '负样本更长'})")
    
    # Summary
    total = len(all_results)
    print(f"\nTotal problems: {total}")
    print(f"Majority correct: {maj_correct['count']}/{total} ({100*maj_correct['count']/total:.1f}%)")
    print(f"Majority wrong:   {maj_wrong['count']}/{total} ({100*maj_wrong['count']/total:.1f}%)")
    
    print("\n" + "-" * 80)
    print("正负样本平均回答长度对比（字符数）")
    print("-" * 80)
    
    _print_block("【整体】", overall)
    _print_block("【Majority 正确时】", maj_correct)
    _print_block("【Majority 错误时】", maj_wrong)


def main():
    parser = argparse.ArgumentParser(description="Analyze response length vs correctness hypothesis")
    parser.add_argument("--model_path", required=True, help="Path to model checkpoint")
    parser.add_argument("--data_path", default="data/MATH-TTT/train.json", help="Path to dataset JSON")
    parser.add_argument("--n_samples", type=int, default=64, help="Number of responses per problem")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--max_tokens", type=int, default=4096, help="Max response tokens")
    parser.add_argument("--tp_size", type=int, default=1, help="Tensor parallel size for vLLM")
    parser.add_argument("--task", default="math", help="Task type for answer extraction")
    parser.add_argument("--output_path", default=None, help="Optional: save raw results to JSON")
    parser.add_argument("--max_problems", type=int, default=None, help="Limit number of problems (for quick test)")
    args = parser.parse_args()
    
    # Load data
    problems = load_data(args.data_path)
    if args.max_problems:
        problems = problems[:args.max_problems]
        print(f"Limited to {len(problems)} problems")
    
    # Generate responses
    all_responses = generate_responses(
        args.model_path, problems, args.n_samples,
        args.temperature, args.max_tokens, args.tp_size
    )
    
    # Analyze each problem
    all_results = []
    for i, (problem, responses) in enumerate(zip(problems, all_responses)):
        if i % 50 == 0:
            print(f"  Analyzing problem {i+1}/{len(problems)}...")
        result = analyze_prompt_group(responses, problem["ground_truth"], task=args.task)
        result["problem_idx"] = i
        all_results.append(result)
    
    # Print summary
    print_summary(all_results)
    
    # Save raw results (without per_response to keep file small)
    if args.output_path:
        save_data = []
        for r in all_results:
            r_copy = {k: v for k, v in r.items() if k != "per_response"}
            save_data.append(r_copy)
        
        os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
        with open(args.output_path, "w") as f:
            json.dump(save_data, f, indent=2, default=str)
        print(f"\nRaw results saved to {args.output_path}")


if __name__ == "__main__":
    main()
