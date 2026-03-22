"""
Top-5 Candidate Verification & Voting Experiment

For each problem with SC < 0.3, extract the top-5 most frequent answers 
from the baseline 64 rollouts. Present these to the model in a single prompt 
and ask it to independently derive the answer and select from the candidates.

Generates 32 verification paths (temperature=0.6) for this single prompt 
and performs a new Majority Vote to see if accuracy improves.

Usage:
    python scripts/top5_candidate_voting_experiment.py \
        --model_path Qwen/Qwen2.5-7B \
        --input_file scripts/base.jsonl \
        --output_file scripts/top5_voting_results.jsonl
"""

import json
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
# 2. Prompting Logic
# =====================================================

TOP5_VOTING_PROMPT = """[Test Time Problem]
{problem}

[Model Sampled Candidate Answers]
Below are the top candidate answers sampled from the model at test time, ordered by frequency. The correct answer is highly likely to be among these candidates, but majority voting may be unreliable.

{options_str}

[Selection Task]
Please act as a rigorous reasoning judge and follow these steps:

1. **Independent Derivation**: Solve the problem yourself step-by-step. Show your complete logical reasoning and calculations.
2. **Comparison**: Compare your final derived result with the Candidate Answers.
3. **Final Selection**: Choose the candidate that matches your independently derived result. If none match, give your own independently derived result.

Conclude your response by strictly enclosing your selected final answer in a \\boxed{{}} environment.
Let's think step by step."""

def build_top5_prompt(problem, answers_list, tokenizer, model_path):
    """
    answers_list: list of top N answers (e.g. 5) 
    """
    options = []
    for i, ans in enumerate(answers_list):
        options.append(f"Option {chr(65+i)}: {ans}")
    options_str = "\n".join(options)

    content = TOP5_VOTING_PROMPT.format(problem=problem, options_str=options_str)

    name = model_path.lower()
    if "instruct" in name or "llama" in name:
        try:
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": content}],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            pass
            
    return content

def get_top_k_answers(extracted_answers, k=5):
    """Extract top-K normalized valid answers."""
    valid_answers = [strip_string(a) for a in extracted_answers if a and a != "[NO_ANSWER]"]
    if not valid_answers:
        return []
    counter = Counter(valid_answers)
    return [ans for ans, _ in counter.most_common(k)]


# =====================================================
# 3. Analysis & Reporting
# =====================================================

def print_full_comparison(results):
    buckets = [
        ("Low (SC<0.3)",  lambda r: r["sc_score"] < 0.3),
        ("Mid (0.3-0.7)", lambda r: 0.3 <= r["sc_score"] < 0.7),
        ("High (≥0.7)",   lambda r: r["sc_score"] >= 0.7),
        ("ALL",           lambda r: True),
    ]

    print("\n" + "=" * 105)
    print("TOP-5 CANDIDATE VERIFICATION & VOTING RESULTS")
    print("=" * 105)

    header = (
        f"  {'Bucket':<16} {'N':>5}  "
        f"{'Orig MAJ Acc':>14}  {'Verify MAJ Acc':>16}  {'Δ(Accuracy)':>13}  "
        f"{'Rescued':>7}  {'Harmed':>6}"
    )
    print(header)
    print("  " + "-" * 101)

    for bname, bfn in buckets:
        bucket = [r for r in results if bfn(r)]
        if not bucket:
            continue
        n = len(bucket)

        maj_correct = sum(1 for r in bucket if r["maj_correct"])
        verify_correct = sum(1 for r in bucket if r["verify_maj_correct"])

        rescued = sum(1 for r in bucket if not r["maj_correct"] and r["verify_maj_correct"])
        harmed = sum(1 for r in bucket if r["maj_correct"] and not r["verify_maj_correct"])

        maj_acc = maj_correct / n
        verify_acc = verify_correct / n
        delta = verify_acc - maj_acc

        print(
            f"  {bname:<16} {n:>5}  "
            f"{maj_acc:>13.1%}    {verify_acc:>14.1%}  {delta:>12.1%}  "
            f"{rescued:>7}  {harmed:>6}"
        )

    low = [r for r in results if r["sc_score"] < 0.3]
    if not low:
        return

    print(f"\n  Transition Matrix (Orig MAJ → Verify MAJ) [Low SC]:")
    both_correct = sum(1 for r in low if r["maj_correct"] and r["verify_maj_correct"])
    rescued = sum(1 for r in low if not r["maj_correct"] and r["verify_maj_correct"])
    harmed = sum(1 for r in low if r["maj_correct"] and not r["verify_maj_correct"])
    both_wrong = sum(1 for r in low if not r["maj_correct"] and not r["verify_maj_correct"])

    print(f"    Orig MAJ ✓ → Verify MAJ ✓ (kept):      {both_correct}")
    print(f"    Orig MAJ ✗ → Verify MAJ ✓ (rescued):   {rescued}")
    print(f"    Orig MAJ ✓ → Verify MAJ ✗ (harmed):    {harmed}")
    print(f"    Orig MAJ ✗ → Verify MAJ ✗ (still ✗):   {both_wrong}")
    print(f"    Net gain: {rescued - harmed:+d}")

    print(f"\n  {'─'*110}")
    print(f"  Sample Detail (first 20 low-SC problems):")
    print(f"  {'Idx':<5} {'Base SC':>8} {'OptN':>5} {'Orig MAJ':>10} {'Verify MAJ':>11} {'Verif SC':>9} {'Transition':<15}")
    print(f"  {'─'*110}")
    
    for r in low[:20]:
        sc = f"{r['sc_score']:.2f}"
        maj_c = "✓" if r["maj_correct"] else "✗"
        verify_c = "✓" if r["verify_maj_correct"] else "✗"
        v_sc = f"{r.get('verify_sc_score', 0.0):.2f}"
        opt_len = len(r.get("top_candidates", []))
        
        if not r["maj_correct"] and r["verify_maj_correct"]:
            trans = "RESCUED"
        elif r["maj_correct"] and not r["verify_maj_correct"]:
            trans = "HARMED"
        elif r["maj_correct"] and r["verify_maj_correct"]:
            trans = "kept ✓"
        else:
            trans = "still ✗"
            
        print(f"  {r['idx']:<5} {sc:>8} {opt_len:>5} {maj_c:>10} {verify_c:>11} {v_sc:>9} {trans:<15}")


# =====================================================
# 4. Main
# =====================================================

def main(args):
    print(f"Loading {args.input_file}")
    data = []
    with open(args.input_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    print(f"Loaded {len(data)} problems")

    low_sc_count = sum(1 for d in data if d.get("sc_score", 1.0) < 0.3)
    print(f"Low-consistency (SC < 0.3) problems: {low_sc_count}")
    if low_sc_count == 0:
        return

    print(f"Loading tokenizer & vLLM engine for {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    
    llm = LLM(
        model=args.model_path,
        tokenizer=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
        trust_remote_code=True,
        dtype="auto",
    )
    
    # We generate multiple verification paths
    sampling_params = SamplingParams(
        n=args.num_return_sequences,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        stop=["<|eot_id|>", "</s>", "<|im_end|>", "Q:"],
    )

    all_prompts = []
    metadata_map = [] # problem_idx list
    all_results = []

    for idx, item in enumerate(data):
        sc_score = item.get("sc_score", 1.0)
        gt_norm = strip_string(str(item.get("answer", "")))
        maj_answer = strip_string(str(item.get("sc_answer", "")))
        
        result = {
            "idx": idx,
            "problem": item["problem"],
            "gt": item.get("answer", ""),
            "gt_norm": gt_norm,
            "sc_score": sc_score,
            "maj_answer": maj_answer,
            "maj_correct": (maj_answer == gt_norm),
            "was_verified": False,
            "verify_maj_answer": maj_answer,
            "verify_maj_correct": (maj_answer == gt_norm),
            "verify_sc_score": 0.0,
            "top_candidates": []
        }
        all_results.append(result)

        if sc_score < 0.3:
            extracted = item.get("extracted_answers", [])
            top_k_answers = get_top_k_answers(extracted, k=5)
            
            # Save candidates
            all_results[idx]["top_candidates"] = top_k_answers
            
            prompt = build_top5_prompt(
                item["problem"], top_k_answers, tokenizer, args.model_path
            )
            
            all_prompts.append(prompt)
            metadata_map.append(idx)

    print(f"Total Top-5 verification prompts built: {len(all_prompts)}")
    print(f"Generating {args.num_return_sequences} responses per prompt (overall roughly {len(all_prompts) * args.num_return_sequences} generations)...")

    outputs = llm.generate(all_prompts, sampling_params)

    # Dictionary mapping p_idx -> list of norm_answers
    verify_answers_per_prob = {idx: [] for idx in metadata_map}

    for i, output in enumerate(outputs):
        p_idx = metadata_map[i]
        
        for out in output.outputs:
            response_text = out.text
            raw_ans = extract_answer(response_text)
            norm_ans = strip_string(raw_ans) if raw_ans else "[NO_ANSWER]"
            
            verify_answers_per_prob[p_idx].append(norm_ans)

    # Calculate new Majority Vote
    for p_idx, answers in verify_answers_per_prob.items():
        all_results[p_idx]["was_verified"] = True
        
        valid_answers = [a for a in answers if a != "[NO_ANSWER]"]
        counter = Counter(valid_answers)
        most_common = counter.most_common(1)
        
        if most_common:
            best_ans, count = most_common[0]
            verify_sc_score = count / len(valid_answers)
        else:
            best_ans = "[NO_ANSWER]"
            verify_sc_score = 0.0
            
        all_results[p_idx]["verify_maj_answer"] = best_ans
        all_results[p_idx]["verify_maj_correct"] = (best_ans == all_results[p_idx]["gt_norm"])
        all_results[p_idx]["verify_sc_score"] = verify_sc_score

    print_full_comparison(all_results)
    
    if args.output_file:
        print(f"\nSaving detailed results to {args.output_file}")
        with open(args.output_file, "w", encoding="utf-8") as f:
            for r in all_results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("\nDone!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Top-5 Candidate Verification & Voting Experiment")
    parser.add_argument("--model_path", type=str, required=True, help="HuggingFace model path")
    parser.add_argument("--input_file", type=str, default="scripts/base.jsonl", help="Input JSONL")
    parser.add_argument("--output_file", type=str, default="scripts/top5_voting_results.jsonl", help="Output JSONL")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="vLLM tensor parallel size")
    parser.add_argument("--max_model_len", type=int, default=4096, help="Max context length to restrict KV cache size")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.7, help="Fraction of GPU memory for vLLM")
    parser.add_argument("--enforce_eager", action="store_true", help="Disable CUDA graphs (saves some memory)")
    parser.add_argument("--max_tokens", type=int, default=4096, help="Max tokens for verification generation")
    parser.add_argument("--num_return_sequences", type=int, default=32, help="Rollouts to generate per problem")
    parser.add_argument("--temperature", type=float, default=0.6, help="Temperature")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p sampling")
    args = parser.parse_args()
    main(args)
