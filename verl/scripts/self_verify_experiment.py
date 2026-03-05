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

SYSTEM_PROMPT = """You are an expert mathematical reasoning judge. Your task is to rigorously evaluate candidate solutions to a math problem, assess the quality of their reasoning, and select the most reliable one."""

USER_TEMPLATE_WITH_ROLLOUT = """[Test Time Problem]
{problem}

[Model Sampled Candidate Solutions]
Below are {num_candidates} candidate solutions sampled from the model at test time. Each includes a complete reasoning process and a final answer.
Important Context: This is a low-consistency sample with an estimated pass@{num_candidates} = 0.6-0.7. The correct answer MAY OR MAY NOT be among these candidates. Do NOT force a selection if no candidate is reliable.
{options}

[Selection Task]
Please act as a cautious and rigorous reasoning judge. Follow these steps to either select the most reliable candidate OR conclude that no reliable candidate exists:

1. **Preliminary Screening: Fatal Error Check**
   First, quickly eliminate any candidate with OBVIOUS fatal errors:
   - Directly misinterprets the problem statement
   - Contains clear logical contradictions or circular reasoning
   - Has irreversible computational errors in core steps
   List the eliminated candidates and their fatal errors (e.g., "Candidate 5: Misread the problem, used 'area' instead of 'perimeter'"). If no candidates are eliminated, state "All candidates pass preliminary screening."

2. **Individual Reasoning Review & Rigor Scoring**
   For the remaining candidates, carefully examine their reasoning and assign a Rigor Score (0-10, 10 = perfect rigor) based on:
   - Logical consistency (no contradictions, no circular reasoning)
   - Computational accuracy (no arithmetic/algebraic errors; minor typos that don't affect logic are allowed)
   - Justification completeness (all non-trivial steps are explained; no unexplained "leaps of faith")
   - Problem alignment (directly addresses the problem, no off-topic detours)
   - Premise validity (all implicit/explicit assumptions are reasonable and consistent with the problem)
   For each candidate, output:
   - Candidate X: Rigor Score = [score], Reasoning Assessment = [brief summary of strengths/weaknesses; "Perfectly rigorous" if score=10]

3. **Independent Derivation & Self-Confidence Check**
   Solve the problem yourself step-by-step, showing complete logical reasoning and calculations.
   After solving, assign a Self-Confidence Score (0-10, 10 = 100% confident) to your own derivation, based on:
   - Clarity of the problem statement (no ambiguity)
   - Complexity of the reasoning (no highly error-prone steps)
   - Consistency of your own logic (no second-guessing)
   Output:
   - My Independent Derivation: [your step-by-step solution]
   - My Self-Confidence Score: [score]

4. **Cross-Validation & Reliability Decision**
   Compare your independent derivation with the remaining candidates (both reasoning logic and final answer). Then make one of the following decisions:
   a) **Reliable Candidate Exists**: If at least one candidate has a Rigor Score >= 7 AND its reasoning/answer aligns with your independent derivation (or has a different but equally rigorous reasoning path to the same answer), select the candidate with the highest Rigor Score.
   b) **No Reliable Candidate**: If NO candidate meets the above criteria (e.g., all Rigor Scores <7, or no candidate aligns with your high-confidence independent derivation), conclude that no reliable candidate exists in the pool.

5. **Final Verdict**
   - If you selected a reliable candidate, strictly enclose it in a \\boxed{{}} environment (e.g., \\boxed{{Candidate 3}}), followed by a 1-sentence justification.
   - If you concluded no reliable candidate exists, output \\boxed{{No Reliable Candidate}}, followed by a 1-sentence explanation."""

USER_TEMPLATE_NO_ROLLOUT = """[Test Time Problem]
{problem}

[Model Sampled Candidate Answers]
Below are {num_candidates} candidate answers sampled from the model at test time. The correct answer is highly likely to be among these candidates, but majority voting may be unreliable. Please prioritize reasoning rigor over result frequency.
{options}

[Selection Task]
Please act as a rigorous reasoning judge and follow these steps:

1. **Independent Derivation**: Solve the problem yourself step-by-step. Show your complete logical reasoning and calculations.

2. **Comparison**: Compare your final derived result with each Candidate Answer.

3. **Final Selection**: Choose the candidate that matches your independently derived result. If none match, select the one with the most plausible answer.

Conclude your response by strictly enclosing the selected candidate in a \\boxed{{}} environment (e.g., \\boxed{{Candidate 3}}).
If NONE of the candidates are correct, output \\boxed{{None}}."""


def build_verify_prompt(problem, candidates, tokenizer, model_path,
                        show_freq=False, rollout_map=None):
    """
    Build verification prompt.
    candidates: list of (answer, frequency) tuples.
    rollout_map: if provided, dict {answer: rollout_text} to include full solutions.
    Returns: (prompt_text, option_map)
    """
    options_lines = []
    option_map = {}
    for i, (ans, freq) in enumerate(candidates):
        label = f"Candidate {i+1}"
        if rollout_map and ans in rollout_map:
            rollout = rollout_map[ans]
            options_lines.append(
                f"--- {label} ---\n"
                f"{rollout}\n"
                f"**Final Answer: {ans}**"
            )
        else:
            options_lines.append(f"{label}: {ans}")
        option_map[label] = ans

    if rollout_map:
        options_text = "\n\n".join(options_lines)
        template = USER_TEMPLATE_WITH_ROLLOUT
    else:
        options_text = "\n".join(options_lines)
        template = USER_TEMPLATE_NO_ROLLOUT

    content = template.format(
        problem=problem, options=options_text, num_candidates=len(candidates)
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

    # Clean LaTeX artifacts: \text{...}, backslash-space, \mathrm{...}
    cleaned = raw_stripped
    cleaned = re.sub(r'\\text\{([^}]*)\}', r'\1', cleaned)
    cleaned = re.sub(r'\\mathrm\{([^}]*)\}', r'\1', cleaned)
    cleaned = cleaned.replace('\\ ', ' ').replace('\\,', ' ')
    cleaned = cleaned.strip()

    # "No Reliable Candidate" / "None" → fallback
    if "no reliable" in cleaned.lower() or cleaned.lower() == "none":
        return None

    # Match "Candidate X" exactly
    if cleaned in option_map:
        return option_map[cleaned]

    # Try partial match "Candidate N"
    m = re.match(r'[Cc]andidate\s*(\d+)', cleaned)
    if m:
        label = f"Candidate {m.group(1)}"
        if label in option_map:
            return option_map[label]

    # Try direct answer match
    norm = strip_string(cleaned)
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
                   candidate_fn, show_freq, exp_name, use_rollout=False):
    """
    Run one self-verification experiment.
    candidate_fn: function(answers) -> (candidates, consistency)
    show_freq: whether to show frequency in prompt
    use_rollout: if True, include one full rollout per candidate in the prompt
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
            # Build rollout map: pick one representative rollout per candidate answer
            rollout_map = None
            if use_rollout and "responses" in item:
                rollout_map = {}
                responses = item["responses"]
                MAX_ROLLOUT_LEN = 1500
                for ans, _ in candidates[:5]:
                    # Collect all rollouts that produced this answer
                    matching = []
                    for j, ext_ans in enumerate(answers):
                        if ext_ans == ans and j < len(responses):
                            matching.append(responses[j].strip())
                    if not matching:
                        continue
                    # Prefer rollouts <= MAX_ROLLOUT_LEN, pick shortest among those
                    short = [r for r in matching if len(r) <= MAX_ROLLOUT_LEN]
                    if short:
                        rollout_map[ans] = max(short, key=len)  # longest under limit
                    else:
                        rollout_map[ans] = min(matching, key=len)  # shortest overall

            prompt, option_map = build_verify_prompt(
                item["problem"], candidates[:5],
                tokenizer=tokenizer, model_path=model_path,
                show_freq=show_freq,
                rollout_map=rollout_map
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

            # Store prompt and response
            results[idx]["verify_prompt"] = verify_prompts[i]
            results[idx]["verify_response"] = response

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
    # Sweep over K: Freq Top-K + Rollout + MAJ Fallback
    # ========================================
    ks = [5]
    all_results = {}

    for k in ks:
        results_k = run_experiment(
            data, llm, tokenizer, args.model_path, sampling_params,
            candidate_fn=lambda ans, _k=k: get_frequency_candidates(ans, top_k=_k),
            show_freq=False,
            exp_name=f"K={k}: Freq Top-{k}",
            use_rollout=False
        )
        # Fallback = MAJ (freq top-1), which is already the default fallback_top1
        all_results[k] = results_k

    # ========================================
    # Comparison Table
    # ========================================
    print(f"\n{'='*90}")
    print("K-Sweep Results: Freq Top-K + Rollout + MAJ Fallback")
    print(f"{'='*90}")

    buckets = [
        ("Low(<=0.3)", lambda r: r["consistency"] <= 0.3),
        ("Mid(0.3-0.7)", lambda r: 0.3 < r["consistency"] <= 0.7),
        ("High(>0.7)", lambda r: r["consistency"] > 0.7),
        ("All", lambda r: True),
    ]

    for bname, bfn in buckets:
        print(f"\n  {bname}:")
        print(f"    {'K':<6} {'N':<6} {'MAJ':<8} {'Verify':<8} {'Gain':<8} "
              f"{'Model':<7} {'FB':<5} {'Net':<6} {'GT Cover':<10}")
        print(f"    " + "-" * 66)

        for k in ks:
            bucket = [r for r in all_results[k] if bfn(r)]
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

            # GT coverage in candidate set
            verified_items = [r for r in bucket if r["verify_source"] != "unchanged"]
            if verified_items:
                gt_cover = sum(1 for r in verified_items
                               if r["gt_norm"] in {a for a, _ in r["candidates"]}) / len(verified_items)
            else:
                gt_cover = float('nan')

            maj_acc = maj_c / n
            ver_acc = ver_c / n
            gain = ver_acc - maj_acc

            print(f"    {k:<6} {n:<6} {maj_acc:<8.1%} {ver_acc:<8.1%} {gain:<+8.1%} "
                  f"{model_n:<7} {fb_n:<5} {rescued-harmed:<+6d} {gt_cover:<10.1%}")

    # Low-consistency detail per K
    print(f"\n{'='*90}")
    print("Low-Consistency Model Pick Accuracy by K")
    print(f"{'='*90}")
    print(f"  {'K':<6} {'Model Pick Acc':<18} {'Fallback Acc':<15} {'GT in Cands':<15}")
    print(f"  " + "-" * 54)

    for k in ks:
        low = [r for r in all_results[k] if r["consistency"] <= 0.3
               and r["verify_source"] != "unchanged"]
        if not low:
            continue
        model_items = [r for r in low if r["verify_source"] == "model"]
        fb_items = [r for r in low if r["verify_source"] == "fallback_top1"]
        gt_in = sum(1 for r in low if r["gt_norm"] in {a for a, _ in r["candidates"]})

        mc = sum(1 for r in model_items if r["verified_answer"] == r["gt_norm"]) if model_items else 0
        fc = sum(1 for r in fb_items if r["verified_answer"] == r["gt_norm"]) if fb_items else 0

        mc_str = f"{mc}/{len(model_items)} ({100*mc/len(model_items):.1f}%)" if model_items else "N/A"
        fc_str = f"{fc}/{len(fb_items)} ({100*fc/len(fb_items):.1f}%)" if fb_items else "N/A"

        print(f"  {k:<6} {mc_str:<18} {fc_str:<15} {gt_in}/{len(low)} ({100*gt_in/len(low):.1f}%)")

    # Save results
    if args.output_file:
        print(f"\nSaving results to {args.output_file}")
        with open(args.output_file, "w", encoding="utf-8") as f:
            for i, item in enumerate(data):
                record = {
                    "problem": item["problem"],
                    "answer": item["answer"],
                    "maj_answer": item.get("sc_answer"),
                }
                for k in ks:
                    r = all_results[k][i]
                    record[f"K{k}_consistency"] = r["consistency"]
                    record[f"K{k}_candidates"] = [(a, round(f, 4)) for a, f in r["candidates"]]
                    record[f"K{k}_verified"] = r["verified_answer"]
                    record[f"K{k}_source"] = r["verify_source"]
                    record[f"K{k}_raw_model_answer"] = r.get("raw_model_answer")
                    record[f"K{k}_verify_prompt"] = r.get("verify_prompt", "")
                    record[f"K{k}_verify_response"] = r.get("verify_response", "")
                f.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="K-Sweep Self-Verification with Rollout")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, default="self_verify_results.jsonl")
    parser.add_argument("--num_bootstrap", type=int, default=1000)
    parser.add_argument("--max_verify_tokens", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)

