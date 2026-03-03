"""
Bootstrap Self-Verification Offline Experiment

Two experiments comparing bootstrap vs raw-frequency candidate selection:
  Exp A: Bootstrap candidates (P_boot >= 0.02) with frequency shown in prompt
  Exp B: Raw top-5 frequency candidates with frequency shown in prompt

Usage:
    python scripts/self_verify_experiment.py \
        --model_path Qwen/Qwen2.5-7B \
        --input_file qwen64.jsonl \
        --output_file self_verify_results.jsonl \
        --num_bootstrap 1000
"""

import json
import re
import random
import argparse
import numpy as np
from collections import Counter

import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


# =====================================================
# Utility functions
# =====================================================

def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None
    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1
    return None if right_brace_idx is None else string[idx : right_brace_idx + 1]


def remove_boxed(s):
    if s is None:
        return None
    if "\\boxed " in s:
        return s.replace("\\boxed ", "")
    if s.startswith("\\boxed{") and s.endswith("}"):
        return s[len("\\boxed{") : -1]
    return s


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


def extract_answer(text):
    boxed = last_boxed_only_string(text)
    if boxed is None:
        return None
    return remove_boxed(boxed)


# =====================================================
# Verification Prompt (two versions: with/without freq)
# =====================================================

SYSTEM_PROMPT = """You are an expert mathematical verifier. Your task is to rigorously evaluate a few candidate answers to a math problem. You must think independently, solve the problem step-by-step from scratch, and then determine if any of the provided candidates are completely correct."""

USER_TEMPLATE_NO_FREQ = """[Problem]
{problem}

[Candidate Answers]
Several distinct conclusions were proposed derived from rough calculations. They are:
{options}

[Verification Task]
Please act as an impartial judge and follow these strict steps:

1. Independent Derivation: Do NOT trust any of the candidate answers. Solve the problem yourself step-by-step. Show your complete logical reasoning and calculations.

2. Comparison: Compare your final derived result with the Candidate Answers.

3. Final Verdict: Conclude your response by strictly enclosing the correct option in a \\boxed{{}} environment.
   - If your result exactly matches Option 1, output \\boxed{{Option 1}}.
   - If your result exactly matches Option 2, output \\boxed{{Option 2}}.{extra_options}
   - If NONE of the candidate answers are correct, output \\boxed{{None}}."""

USER_TEMPLATE_WITH_FREQ = """[Problem]
{problem}

[Candidate Answers]
Several distinct conclusions were proposed derived from rough calculations. They are:
{options}

[Verification Task]
Please act as an impartial judge and follow these strict steps:

1. Independent Derivation: Do NOT trust any of the candidate answers. The confidence scores above are merely statistical frequencies and may be misleading. Solve the problem yourself step-by-step. Show your complete logical reasoning and calculations.

2. Comparison: Compare your final derived result with the Candidate Answers.

3. Final Verdict: Conclude your response by strictly enclosing the correct option in a \\boxed{{}} environment.
   - If your result exactly matches Option 1, output \\boxed{{Option 1}}.
   - If your result exactly matches Option 2, output \\boxed{{Option 2}}.{extra_options}
   - If NONE of the candidate answers are correct, output \\boxed{{None}}."""


def build_verify_prompt(problem, candidates, tokenizer, model_path, show_freq=False):
    """
    Build verification prompt.
    candidates: list of (answer, frequency) tuples.
    show_freq: if True, show frequency in the options.
    Returns: (prompt_text, option_map)
    """
    options_lines = []
    option_map = {}
    for i, (ans, freq) in enumerate(candidates):
        label = f"Option {i+1}"
        if show_freq:
            options_lines.append(f"{label}: {ans} (confidence: {freq:.1%})")
        else:
            options_lines.append(f"{label}: {ans}")
        option_map[label] = ans
    options_text = "\n".join(options_lines)

    extra_lines = ""
    for i in range(2, len(candidates)):
        extra_lines += f"\n   - If your result exactly matches Option {i+1}, output \\boxed{{Option {i+1}}}."

    template = USER_TEMPLATE_WITH_FREQ if show_freq else USER_TEMPLATE_NO_FREQ
    content = template.format(
        problem=problem, options=options_text, extra_options=extra_lines
    )

    if tokenizer:
        try:
            prompt = tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": content},
                ],
                tokenize=False,
                add_generation_prompt=True
            )
            return prompt, option_map
        except Exception:
            pass

    return f"{SYSTEM_PROMPT}\n\n{content}", option_map


def parse_verify_response(response_text, option_map):
    """Parse model output to extract selected answer."""
    raw = extract_answer(response_text)
    if raw is None:
        return None
    raw_stripped = raw.strip()

    if raw_stripped in option_map:
        return option_map[raw_stripped]
    if raw_stripped.lower() == "none":
        return None

    norm = strip_string(raw_stripped)
    for _label, ans in option_map.items():
        if strip_string(ans) == norm:
            return ans
    return None


# =====================================================
# Candidate Extraction
# =====================================================

def get_bootstrap_candidates(answers, B=1000, threshold=0.01):
    """Bootstrap: P_boot >= threshold."""
    valid = [a for a in answers if a != "[NO_ANSWER]"]
    if not valid:
        return [], 0.0

    freq = Counter(valid)
    majority = freq.most_common(1)[0][0]
    consistency = freq[majority] / len(valid)

    boot_majorities = []
    for _ in range(B):
        subset = random.choices(valid, k=len(valid))
        maj = Counter(subset).most_common(1)[0][0]
        boot_majorities.append(maj)

    boot_counter = Counter(boot_majorities)
    candidates = sorted(
        [(ans, cnt / B) for ans, cnt in boot_counter.items() if cnt / B >= threshold],
        key=lambda x: -x[1]
    )
    return candidates, consistency


def get_frequency_candidates(answers, top_k=5):
    """Raw top-K by frequency."""
    valid = [a for a in answers if a != "[NO_ANSWER]"]
    if not valid:
        return [], 0.0

    freq = Counter(valid)
    N = len(valid)
    majority = freq.most_common(1)[0][0]
    consistency = freq[majority] / N

    candidates = [(ans, cnt / N) for ans, cnt in freq.most_common(top_k)]
    return candidates, consistency


# =====================================================
# Single Experiment Runner
# =====================================================

def run_experiment(data, llm, tokenizer, model_path, sampling_params,
                   candidate_fn, show_freq, exp_name):
    """
    Run one self-verification experiment.
    candidate_fn: function(answers) -> (candidates, consistency)
    show_freq: whether to show frequency in prompt
    Returns: list of result dicts per problem
    """
    print(f"\n{'='*80}")
    print(f"Experiment: {exp_name}")
    print(f"{'='*80}")

    results = []
    verify_indices = []
    verify_prompts = []
    verify_option_maps = []

    for idx, item in enumerate(data):
        answers = item["extracted_answers"]
        candidates, consistency = candidate_fn(answers)
        gt_norm = strip_string(str(item["answer"]))

        r = {
            "consistency": consistency,
            "gt_norm": gt_norm,
            "maj_answer": strip_string(str(item.get("sc_answer", ""))),
            "candidates": candidates,
        }
        r["maj_correct"] = (r["maj_answer"] == gt_norm)
        results.append(r)

        # Only verify low-consistency with >= 2 candidates
        if consistency <= 0.3 and len(candidates) >= 2:
            prompt, option_map = build_verify_prompt(
                item["problem"], candidates[:5],
                tokenizer=tokenizer, model_path=model_path,
                show_freq=show_freq
            )
            verify_indices.append(idx)
            verify_prompts.append(prompt)
            verify_option_maps.append(option_map)

    print(f"  Low-consistency problems to verify: {len(verify_indices)}")

    if verify_prompts:
        outputs = llm.generate(verify_prompts, sampling_params)

        for i, output in enumerate(outputs):
            idx = verify_indices[i]
            response = output.outputs[0].text
            verified = parse_verify_response(response, verify_option_maps[i])

            # Also extract the raw boxed answer (even if not in candidate set)
            raw_boxed = extract_answer(response)
            raw_answer = strip_string(raw_boxed) if raw_boxed else None
            results[idx]["raw_model_answer"] = raw_answer

            if verified is not None:
                results[idx]["verified_answer"] = strip_string(verified)
                results[idx]["verify_source"] = "model"
            else:
                results[idx]["verified_answer"] = results[idx]["candidates"][0][0]
                results[idx]["verify_source"] = "fallback_top1"

    # For non-verified problems, verified = MAJ
    for r in results:
        if "verified_answer" not in r:
            r["verified_answer"] = r["maj_answer"]
            r["verify_source"] = "unchanged"

    return results


# =====================================================
# Analysis
# =====================================================

def print_analysis(results, exp_name):
    """Print analysis for one experiment."""
    print(f"\n--- {exp_name} ---")

    buckets = [
        ("Low(<=0.3)", lambda r: r["consistency"] <= 0.3),
        ("Mid(0.3-0.7)", lambda r: 0.3 < r["consistency"] <= 0.7),
        ("High(>0.7)", lambda r: r["consistency"] > 0.7),
        ("All", lambda r: True),
    ]

    print(f"  {'Bucket':<16} {'N':<6} {'MAJ':<8} {'Verify':<8} {'Gain':<8} "
          f"{'Model':<7} {'FB':<5} {'Net':<6}")
    print("  " + "-" * 64)

    for bname, bfn in buckets:
        bucket = [r for r in results if bfn(r)]
        if not bucket:
            continue
        n = len(bucket)
        maj_c = sum(1 for r in bucket if r["maj_correct"])
        ver_c = sum(1 for r in bucket if r["verified_answer"] == r["gt_norm"])
        model_n = sum(1 for r in bucket if r["verify_source"] == "model")
        fb_n = sum(1 for r in bucket if r["verify_source"] == "fallback_top1")

        rescued = sum(1 for r in bucket
                      if not r["maj_correct"] and r["verified_answer"] == r["gt_norm"])
        harmed = sum(1 for r in bucket
                     if r["maj_correct"] and r["verified_answer"] != r["gt_norm"])

        maj_acc = maj_c / n
        ver_acc = ver_c / n
        gain = ver_acc - maj_acc

        print(f"  {bname:<16} {n:<6} {maj_acc:<8.1%} {ver_acc:<8.1%} {gain:<+8.1%} "
              f"{model_n:<7} {fb_n:<5} {rescued-harmed:<+6d}")

    # Low-consistency detail
    low = [r for r in results if r["consistency"] <= 0.3 and r["verify_source"] != "unchanged"]
    if low:
        model_items = [r for r in low if r["verify_source"] == "model"]
        fb_items = [r for r in low if r["verify_source"] == "fallback_top1"]

        gt_in_cands = sum(1 for r in low
                          if r["gt_norm"] in {a for a, _ in r["candidates"]})

        print(f"\n  Low detail:")
        if model_items:
            mc = sum(1 for r in model_items if r["verified_answer"] == r["gt_norm"])
            print(f"    Model pick: {mc}/{len(model_items)} ({100*mc/len(model_items):.1f}%)")
        if fb_items:
            fc = sum(1 for r in fb_items if r["verified_answer"] == r["gt_norm"])
            print(f"    Fallback (top1):   {fc}/{len(fb_items)} ({100*fc/len(fb_items):.1f}%)")

            # Out-of-set raw answer accuracy
            raw_correct = sum(1 for r in fb_items
                              if r.get("raw_model_answer") and r["raw_model_answer"] == r["gt_norm"])
            raw_is_none = sum(1 for r in fb_items
                              if not r.get("raw_model_answer") or r["raw_model_answer"].lower() == "none")
            raw_new_ans = len(fb_items) - raw_is_none
            print(f"    ├─ Raw out-of-set correct: {raw_correct}/{len(fb_items)} ({100*raw_correct/len(fb_items):.1f}%)")
            print(f"    ├─ Model said None:        {raw_is_none}")
            print(f"    └─ Model gave new answer:  {raw_new_ans}")

        print(f"    GT in candidates: {gt_in_cands}/{len(low)} "
              f"({100*gt_in_cands/len(low):.1f}%) ← upper bound")


# =====================================================
# Main
# =====================================================

def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Load data
    print(f"Loading {args.input_file}")
    data = []
    with open(args.input_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    print(f"Loaded {len(data)} problems")

    # Load tokenizer and model
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    print("Initializing vLLM...")
    llm = LLM(
        model=args.model_path,
        tokenizer=args.model_path,
        tensor_parallel_size=torch.cuda.device_count() or 1,
        trust_remote_code=True,
        dtype="auto",
    )

    sampling_params = SamplingParams(
        n=1,
        temperature=0.0,
        max_tokens=args.max_verify_tokens,
        stop=["<|eot_id|>", "</s>", "<|im_end|>"],
    )

    # ========================================
    # Exp A: Bootstrap candidates + frequency
    # ========================================
    results_boot_freq = run_experiment(
        data, llm, tokenizer, args.model_path, sampling_params,
        candidate_fn=lambda ans: get_bootstrap_candidates(ans, B=args.num_bootstrap),
        show_freq=False,
        exp_name="A: Bootstrap Candidates"
    )

    # ========================================
    # Exp B: Raw frequency candidates
    # ========================================
    results_raw_freq = run_experiment(
        data, llm, tokenizer, args.model_path, sampling_params,
        candidate_fn=lambda ans: get_frequency_candidates(ans, top_k=5),
        show_freq=False,
        exp_name="B: Top-5 Frequency Candidates"
    )

    # ========================================
    # Exp C: Bootstrap candidates + MAJ fallback
    # ========================================
    # Same as A but fallback uses raw majority instead of bootstrap top-1
    from copy import deepcopy
    results_hybrid = deepcopy(results_boot_freq)
    for i, r in enumerate(results_hybrid):
        if r["verify_source"] == "fallback_top1":
            r["verified_answer"] = r["maj_answer"]
            r["verify_source"] = "fallback_maj"

    print(f"\n{'='*80}")
    print("Experiment: C: Bootstrap + MAJ Fallback")
    print(f"{'='*80}")
    print("  (Bootstrap candidates for verification, majority for fallback)")

    # ========================================
    # Exp D: Freq candidates + Bootstrap fallback
    # ========================================
    # Same as B but fallback uses bootstrap top-1 instead of freq top-1
    results_d = deepcopy(results_raw_freq)
    # Build bootstrap top-1 lookup
    boot_top1 = {}
    for i, r in enumerate(results_boot_freq):
        if r["candidates"]:
            boot_top1[i] = r["candidates"][0][0]
        else:
            boot_top1[i] = r["maj_answer"]

    for i, r in enumerate(results_d):
        if r["verify_source"] == "fallback_top1":
            r["verified_answer"] = boot_top1[i]
            r["verify_source"] = "fallback_boot1"

    print(f"\n{'='*80}")
    print("Experiment: D: Freq Candidates + Bootstrap Fallback")
    print(f"{'='*80}")
    print("  (Top-5 freq candidates for verification, bootstrap top-1 for fallback)")

    # ========================================
    # Comparison
    # ========================================
    print(f"\n{'='*80}")
    print("Results Comparison")
    print(f"{'='*80}")

    print_analysis(results_boot_freq, "A: Bootstrap")
    print_analysis(results_raw_freq, "B: Top-5 Freq")
    print_analysis(results_hybrid, "C: Boot cand + MAJ FB")
    print_analysis(results_d, "D: Freq cand + Boot FB")

    # Side-by-side low-consistency comparison
    low_boot = [r for r in results_boot_freq if r["consistency"] <= 0.3]
    low_raw = [r for r in results_raw_freq if r["consistency"] <= 0.3]
    low_hybrid = [r for r in results_hybrid if r["consistency"] <= 0.3]
    low_d = [r for r in results_d if r["consistency"] <= 0.3]
    if low_boot and low_raw and low_hybrid and low_d:
        n = len(low_boot)
        boot_acc = sum(1 for r in low_boot if r["verified_answer"] == r["gt_norm"]) / n
        raw_acc = sum(1 for r in low_raw if r["verified_answer"] == r["gt_norm"]) / n
        hybrid_acc = sum(1 for r in low_hybrid if r["verified_answer"] == r["gt_norm"]) / n
        d_acc = sum(1 for r in low_d if r["verified_answer"] == r["gt_norm"]) / n
        maj_acc = sum(1 for r in low_boot if r["maj_correct"]) / n

        print(f"\n  === Low-Consistency Head-to-Head ===")
        print(f"  MAJ baseline:              {maj_acc:.1%}")
        print(f"  A (Boot cand + Boot FB):   {boot_acc:.1%}  ({boot_acc - maj_acc:+.1%})")
        print(f"  B (Freq cand + Freq FB):   {raw_acc:.1%}  ({raw_acc - maj_acc:+.1%})")
        print(f"  C (Boot cand + MAJ FB):    {hybrid_acc:.1%}  ({hybrid_acc - maj_acc:+.1%})")
        print(f"  D (Freq cand + Boot FB):   {d_acc:.1%}  ({d_acc - maj_acc:+.1%})")
        print(f"  Best vs MAJ:               {max(boot_acc, raw_acc, hybrid_acc, d_acc) - maj_acc:+.1%}")

    # Save results
    if args.output_file:
        print(f"\nSaving results to {args.output_file}")
        with open(args.output_file, "w", encoding="utf-8") as f:
            for i, item in enumerate(data):
                record = {
                    "problem": item["problem"],
                    "answer": item["answer"],
                    "consistency": results_boot_freq[i]["consistency"],
                    "gt_norm": results_boot_freq[i]["gt_norm"],
                    "maj_answer": item.get("sc_answer"),
                    "A_verified": results_boot_freq[i]["verified_answer"],
                    "A_source": results_boot_freq[i]["verify_source"],
                    "B_verified": results_raw_freq[i]["verified_answer"],
                    "B_source": results_raw_freq[i]["verify_source"],
                    "C_verified": results_hybrid[i]["verified_answer"],
                    "C_source": results_hybrid[i]["verify_source"],
                    "D_verified": results_d[i]["verified_answer"],
                    "D_source": results_d[i]["verify_source"],
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bootstrap vs Frequency Self-Verification")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, default="self_verify_results.jsonl")
    parser.add_argument("--num_bootstrap", type=int, default=1000)
    parser.add_argument("--max_verify_tokens", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)
