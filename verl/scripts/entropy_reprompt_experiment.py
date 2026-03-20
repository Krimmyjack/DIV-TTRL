"""
Entropy-Based Re-Prompting Experiment ("Local Optimum Escape")

When self-consistency < 0.3, select the lowest-entropy (most confident) rollout,
then re-prompt the model: "this solution may be stuck in a local optimum,
verify it or re-solve from a different angle."

Compares: MAJ voting  vs  Lowest-Entropy raw answer  vs  Re-prompted answer.

Usage:
    python scripts/entropy_reprompt_experiment.py \
        --model_path Qwen/Qwen2.5-7B \
        --input_file scripts/base.jsonl \
        --output_file scripts/entropy_reprompt_results.jsonl
"""

import json
import re
import argparse
from collections import Counter

import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


# =====================================================
# 1. Utility Functions
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
# 2. Core: Select Lowest-Entropy Rollout
# =====================================================

def select_lowest_entropy_response(responses, response_metrics, extracted_answers):
    """
    From 64 rollouts, select the one with the lowest token_entropy_approx
    (= -mean_logprob, i.e. the most confident response per-token).

    Returns: (response_text, answer, entropy_value, index)
    """
    best_idx = None
    best_entropy = float("inf")

    for i, metric in enumerate(response_metrics):
        entropy = metric.get("token_entropy_approx")
        if entropy is not None and entropy < best_entropy:
            # Also check the response actually produced a valid answer
            if extracted_answers[i] != "[NO_ANSWER]":
                best_entropy = entropy
                best_idx = i

    if best_idx is None:
        # Fallback: pick first valid response
        for i, ans in enumerate(extracted_answers):
            if ans != "[NO_ANSWER]":
                return responses[i], ans, None, i
        return responses[0], extracted_answers[0], None, 0

    return responses[best_idx], extracted_answers[best_idx], best_entropy, best_idx


# =====================================================
# 3. Re-Prompting Template ("Local Optimum Escape")
# =====================================================

REPROMPT_TEMPLATE = """Below is a math problem and a solution that was generated with the highest model confidence (lowest per-token entropy). However, high confidence does NOT guarantee correctness — the model may be stuck in a local optimum or a repetitive reasoning pattern.

**[Problem]**
{problem}

**[Most Confident Solution (entropy = {entropy:.4f})]**
{solution}

**Final Answer from this solution: {answer}**

---

**[Your Task]**
This high-confidence solution path may represent a local optimum. Please:

1. **Verify**: Carefully check each reasoning step above. Look for:
   - Logical fallacies or circular reasoning
   - Computational errors (arithmetic, algebra, calculus mistakes)
   - Unjustified assumptions or missing edge cases
   - Incorrect application of formulas or theorems

2. **Diagnose**: If you find errors, explain exactly where the reasoning goes wrong and why.

3. **Re-solve**: Whether or not you found errors, solve the problem again from scratch using a DIFFERENT approach or perspective. Do NOT simply repeat the same reasoning chain — try an alternative method (e.g., if the original used algebra, try geometric reasoning; if it used direct computation, try a symmetry argument, etc.)

4. **Conclude**: Give your final answer inside \\boxed{{}}.

Let's think step by step."""


def build_reprompt(problem, solution_text, answer_text, entropy_value, tokenizer, model_path):
    """Build the re-prompting prompt."""
    content = REPROMPT_TEMPLATE.format(
        problem=problem,
        solution=solution_text.strip(),
        answer=answer_text,
        entropy=entropy_value if entropy_value is not None else 0.0,
    )

    name = model_path.lower()
    # Use chat template for instruct/llama models
    if "instruct" in name or "llama" in name:
        try:
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": content}],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            pass

    # For base models (Qwen, DeepSeek, etc.) — plain text prompt
    return content


# =====================================================
# 4. Analysis & Reporting
# =====================================================

def print_full_comparison(results):
    """Print comprehensive before/after accuracy comparison."""

    buckets = [
        ("Low (SC<0.3)",  lambda r: r["sc_score"] < 0.3),
        ("Mid (0.3-0.7)", lambda r: 0.3 <= r["sc_score"] < 0.7),
        ("High (≥0.7)",   lambda r: r["sc_score"] >= 0.7),
        ("ALL",           lambda r: True),
    ]

    # ────────────── Summary Table ──────────────
    print("\n" + "=" * 100)
    print("ENTROPY-BASED RE-PROMPTING EXPERIMENT RESULTS")
    print("=" * 100)

    header = (
        f"  {'Bucket':<16} {'N':>5}  "
        f"{'MAJ Acc':>8}  {'LoEnt Acc':>9}  {'Reprompt Acc':>12}  "
        f"{'Δ(Re-MAJ)':>10}  {'Δ(Re-LoEnt)':>11}  "
        f"{'Rescued':>7}  {'Harmed':>6}"
    )
    print(header)
    print("  " + "-" * 96)

    for bname, bfn in buckets:
        bucket = [r for r in results if bfn(r)]
        if not bucket:
            continue
        n = len(bucket)

        # Accuracy counts
        maj_correct = sum(1 for r in bucket if r["maj_correct"])
        loent_correct = sum(1 for r in bucket if r["loent_correct"])
        reprompt_correct = sum(1 for r in bucket if r["reprompt_correct"])

        # Rescued: MAJ wrong → reprompt right
        rescued = sum(1 for r in bucket
                      if not r["maj_correct"] and r["reprompt_correct"])
        # Harmed: MAJ right → reprompt wrong
        harmed = sum(1 for r in bucket
                     if r["maj_correct"] and not r["reprompt_correct"])

        maj_acc = maj_correct / n
        loent_acc = loent_correct / n
        reprompt_acc = reprompt_correct / n

        delta_maj = reprompt_acc - maj_acc
        delta_loent = reprompt_acc - loent_acc

        print(
            f"  {bname:<16} {n:>5}  "
            f"{maj_acc:>7.1%}  {loent_acc:>9.1%}  {reprompt_acc:>12.1%}  "
            f"{delta_maj:>+10.1%}  {delta_loent:>+11.1%}  "
            f"{rescued:>7}  {harmed:>6}"
        )

    # ────────────── Low-Consistency Detailed Analysis ──────────────
    low = [r for r in results if r["sc_score"] < 0.3]
    if not low:
        print("\n  No low-consistency problems found.")
        return

    print("\n" + "=" * 100)
    print("LOW-CONSISTENCY (SC < 0.3) DETAILED ANALYSIS")
    print("=" * 100)

    n_low = len(low)
    n_reprompted = sum(1 for r in low if r.get("was_reprompted", False))
    print(f"  Total low-SC problems: {n_low}")
    print(f"  Problems re-prompted:  {n_reprompted}")

    # Method accuracy
    maj_c = sum(1 for r in low if r["maj_correct"])
    loent_c = sum(1 for r in low if r["loent_correct"])
    repr_c = sum(1 for r in low if r["reprompt_correct"])

    print(f"\n  Method Accuracy:")
    print(f"    Majority Voting (MAJ):      {maj_c}/{n_low} ({100*maj_c/n_low:.1f}%)")
    print(f"    Lowest-Entropy Raw Answer:   {loent_c}/{n_low} ({100*loent_c/n_low:.1f}%)")
    print(f"    Re-Prompted Answer:          {repr_c}/{n_low} ({100*repr_c/n_low:.1f}%)")

    # Transition analysis
    rescued = sum(1 for r in low if not r["maj_correct"] and r["reprompt_correct"])
    harmed = sum(1 for r in low if r["maj_correct"] and not r["reprompt_correct"])
    both_correct = sum(1 for r in low if r["maj_correct"] and r["reprompt_correct"])
    both_wrong = sum(1 for r in low if not r["maj_correct"] and not r["reprompt_correct"])

    print(f"\n  Transition Matrix (MAJ → Re-prompted):")
    print(f"    MAJ ✓ → Reprompt ✓ (kept):    {both_correct}")
    print(f"    MAJ ✗ → Reprompt ✓ (rescued):  {rescued}")
    print(f"    MAJ ✓ → Reprompt ✗ (harmed):   {harmed}")
    print(f"    MAJ ✗ → Reprompt ✗ (still wrong): {both_wrong}")
    print(f"    Net gain: {rescued - harmed:+d}")

    # LoEnt → Reprompt transition
    rescued_loent = sum(1 for r in low if not r["loent_correct"] and r["reprompt_correct"])
    harmed_loent = sum(1 for r in low if r["loent_correct"] and not r["reprompt_correct"])
    print(f"\n  Transition Matrix (LoEnt → Re-prompted):")
    print(f"    LoEnt ✗ → Reprompt ✓ (rescued):  {rescued_loent}")
    print(f"    LoEnt ✓ → Reprompt ✗ (harmed):   {harmed_loent}")
    print(f"    Net gain: {rescued_loent - harmed_loent:+d}")

    # Entropy statistics
    entropies = [r["loent_entropy"] for r in low if r.get("loent_entropy") is not None]
    if entropies:
        print(f"\n  Lowest-Entropy Statistics:")
        print(f"    Mean: {sum(entropies)/len(entropies):.4f}")
        print(f"    Min:  {min(entropies):.4f}")
        print(f"    Max:  {max(entropies):.4f}")

    # Per-problem detail for first 20 (for debugging)
    print(f"\n  {'─'*80}")
    print(f"  Sample Detail (first 20 low-SC problems):")
    print(f"  {'Idx':<5} {'SC':>5} {'Entropy':>8} {'MAJ':>5} {'LoEnt':>6} {'Reprmt':>7} {'Transition':<15}")
    print(f"  {'─'*80}")
    for r in low[:20]:
        sc = f"{r['sc_score']:.2f}"
        ent = f"{r['loent_entropy']:.3f}" if r.get('loent_entropy') else "N/A"
        maj_mark = "✓" if r["maj_correct"] else "✗"
        loent_mark = "✓" if r["loent_correct"] else "✗"
        repr_mark = "✓" if r["reprompt_correct"] else "✗"
        if not r["maj_correct"] and r["reprompt_correct"]:
            trans = "RESCUED"
        elif r["maj_correct"] and not r["reprompt_correct"]:
            trans = "HARMED"
        elif r["maj_correct"] and r["reprompt_correct"]:
            trans = "kept ✓"
        else:
            trans = "still ✗"
        print(f"  {r['idx']:<5} {sc:>5} {ent:>8} {maj_mark:>5} {loent_mark:>6} {repr_mark:>7} {trans:<15}")


# =====================================================
# 5. Main
# =====================================================

def main(args):
    # ─── Load data ───
    print(f"Loading {args.input_file}")
    data = []
    with open(args.input_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    print(f"Loaded {len(data)} problems")

    # ─── Count low-SC problems ───
    low_sc_count = sum(1 for item in data if item.get("sc_score", 1.0) < 0.3)
    print(f"Low-consistency (SC < 0.3) problems: {low_sc_count}")

    if low_sc_count == 0:
        print("No low-consistency problems found. Nothing to re-prompt.")
        return

    # ─── Load tokenizer ───
    print(f"Loading tokenizer: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    # ─── Build re-prompting prompts for low-SC problems ───
    reprompt_indices = []
    reprompt_prompts = []
    all_results = []

    for idx, item in enumerate(data):
        sc_score = item.get("sc_score", 1.0)
        gt_norm = strip_string(str(item.get("answer", "")))
        maj_answer = strip_string(str(item.get("sc_answer", "")))

        # Select lowest-entropy response
        responses = item.get("responses", [])
        response_metrics = item.get("response_metrics", [])
        extracted_answers = item.get("extracted_answers", [])

        loent_text, loent_ans, loent_entropy, loent_idx = select_lowest_entropy_response(
            responses, response_metrics, extracted_answers
        )
        loent_ans_norm = strip_string(loent_ans)

        result = {
            "idx": idx,
            "problem": item["problem"],
            "gt": item.get("answer", ""),
            "gt_norm": gt_norm,
            "sc_score": sc_score,
            "maj_answer": maj_answer,
            "maj_correct": (maj_answer == gt_norm),
            "loent_answer": loent_ans_norm,
            "loent_correct": (loent_ans_norm == gt_norm),
            "loent_entropy": loent_entropy,
            "loent_response_idx": loent_idx,
            "was_reprompted": False,
            "reprompt_answer": maj_answer,  # default: same as MAJ
            "reprompt_correct": (maj_answer == gt_norm),  # default
        }
        all_results.append(result)

        # Only re-prompt low-consistency problems
        if sc_score < 0.3:
            prompt = build_reprompt(
                item["problem"], loent_text, loent_ans, loent_entropy,
                tokenizer, args.model_path
            )
            reprompt_indices.append(idx)
            reprompt_prompts.append(prompt)

    print(f"Built {len(reprompt_prompts)} re-prompting prompts")

    # ─── Initialize vLLM and generate ───
    print(f"Initializing vLLM engine for {args.model_path}")
    llm = LLM(
        model=args.model_path,
        tokenizer=args.model_path,
        tensor_parallel_size=torch.cuda.device_count() or 1,
        trust_remote_code=True,
        dtype="auto",
    )

    sampling_params = SamplingParams(
        n=1,
        temperature=0.0,  # greedy — deterministic re-derivation
        max_tokens=args.max_tokens,
        stop=["<|eot_id|>", "</s>", "<|im_end|>", "Q:"],
    )

    print("Generating re-prompted responses...")
    outputs = llm.generate(reprompt_prompts, sampling_params)

    # ─── Parse results ───
    for i, output in enumerate(outputs):
        idx = reprompt_indices[i]
        response = output.outputs[0].text

        raw_answer = extract_answer(response)
        reprompt_ans_norm = strip_string(raw_answer) if raw_answer else ""

        all_results[idx]["was_reprompted"] = True
        all_results[idx]["reprompt_response"] = response
        all_results[idx]["reprompt_raw_answer"] = raw_answer
        all_results[idx]["reprompt_answer"] = reprompt_ans_norm
        all_results[idx]["reprompt_correct"] = (reprompt_ans_norm == all_results[idx]["gt_norm"])

    # ─── Print full comparison ───
    print_full_comparison(all_results)

    # ─── Save detailed results ───
    if args.output_file:
        print(f"\nSaving detailed results to {args.output_file}")
        with open(args.output_file, "w", encoding="utf-8") as f:
            for r in all_results:
                # Don't save the full reprompt_response for non-reprompted items
                record = {
                    "idx": r["idx"],
                    "problem": r["problem"],
                    "gt": r["gt"],
                    "sc_score": r["sc_score"],
                    "maj_answer": r["maj_answer"],
                    "maj_correct": r["maj_correct"],
                    "loent_answer": r["loent_answer"],
                    "loent_correct": r["loent_correct"],
                    "loent_entropy": r["loent_entropy"],
                    "was_reprompted": r["was_reprompted"],
                    "reprompt_answer": r["reprompt_answer"],
                    "reprompt_correct": r["reprompt_correct"],
                }
                if r.get("reprompt_response"):
                    record["reprompt_response"] = r["reprompt_response"]
                    record["reprompt_raw_answer"] = r.get("reprompt_raw_answer")
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print("\nDone!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Entropy-Based Re-Prompting Experiment (Local Optimum Escape)"
    )
    parser.add_argument("--model_path", type=str, required=True,
                        help="HuggingFace model path (e.g. Qwen/Qwen2.5-7B)")
    parser.add_argument("--input_file", type=str, default="scripts/base.jsonl",
                        help="Input JSONL from rollouts.py")
    parser.add_argument("--output_file", type=str, default="scripts/entropy_reprompt_results.jsonl",
                        help="Output JSONL with all comparison results")
    parser.add_argument("--max_tokens", type=int, default=4096,
                        help="Max tokens for re-prompting generation")
    args = parser.parse_args()
    main(args)
