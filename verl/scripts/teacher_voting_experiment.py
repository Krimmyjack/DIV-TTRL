"""
Teacher-Prompt Generation and Voting Experiment

For each problem with SC < 0.3, takes all 64 original rollouts.
Constructs a Teacher Verification prompt depending on whether the rollout 
is "Mainstream" (equals MAJ answer) or "Non-Mainstream".
Uses vLLM to generate a corrected response for each rollout.
Extracts the \boxed{} answers from the 64 new responses and performs 
Majority Voting (MAJ) to see if accuracy improves over the original SC MAJ.

Usage:
python scripts/teacher_voting_experiment.py \
    --model_path Qwen/Qwen2.5-7B \
    --input_file scripts/base.jsonl \
    --output_file scripts/teacher_voting_results.jsonl
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
# 2. Teacher Prompts
# =====================================================

TEACHER_PROMPT_TEMPLATE = """Below is a math problem and a candidate solution.

**[Problem]**
{problem}

**[Candidate Solution ({solution_type})]**
{rollout}

---

**[Your Task]**
{instruction}

Please provide your verification process and then give your final answer inside \\boxed{{}}.
Let's think step by step."""

MAINSTREAM_INSTRUCTION = """This solution represents the mainstream (majority) approach, but it might be stuck in a local optimum or contain errors. Verify it carefully step-by-step, fix any logical or computational errors you find, and provide the correct final answer."""

NON_MAINSTREAM_INSTRUCTION = """This solution represents an unconventional perspective or alternative method. Verify its logic carefully, assess its validity, and attempt to continue deriving along this direction to find the correct final answer."""


def build_verification_prompt(problem, rollout_text, is_mainstream, tokenizer, model_path):
    """Build the ChatML prompt for Teacher Verification."""
    sol_type = "Mainstream Approach" if is_mainstream else "Non-Mainstream Approach"
    instruction = MAINSTREAM_INSTRUCTION if is_mainstream else NON_MAINSTREAM_INSTRUCTION

    content = TEACHER_PROMPT_TEMPLATE.format(
        problem=problem,
        solution_type=sol_type,
        rollout=rollout_text.strip(),
        instruction=instruction
    )

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

    print("\n" + "=" * 100)
    print("TEACHER-PROMPT GENERATION & VOTING RESULTS")
    print("=" * 100)

    header = (
        f"  {'Bucket':<16} {'N':>5}  "
        f"{'Orig MAJ Acc':>14}  {'Teacher MAJ Acc':>17}  {'Δ(Accuracy)':>13}  "
        f"{'Rescued':>7}  {'Harmed':>6}"
    )
    print(header)
    print("  " + "-" * 96)

    for bname, bfn in buckets:
        bucket = [r for r in results if bfn(r)]
        if not bucket:
            continue
        n = len(bucket)

        maj_correct = sum(1 for r in bucket if r["maj_correct"])
        teacher_correct = sum(1 for r in bucket if r["teacher_maj_correct"])

        rescued = sum(1 for r in bucket if not r["maj_correct"] and r["teacher_maj_correct"])
        harmed = sum(1 for r in bucket if r["maj_correct"] and not r["teacher_maj_correct"])

        maj_acc = maj_correct / n
        teacher_acc = teacher_correct / n
        delta = teacher_acc - maj_acc

        print(
            f"  {bname:<16} {n:>5}  "
            f"{maj_acc:>13.1%}    {teacher_acc:>15.1%}  {delta:>12.1%}  "
            f"{rescued:>7}  {harmed:>6}"
        )

    low = [r for r in results if r["sc_score"] < 0.3]
    if not low:
        return

    n_low = len(low)
    print(f"\n  Transition Matrix (Orig MAJ → Teacher MAJ) [Low SC]:")
    both_correct = sum(1 for r in low if r["maj_correct"] and r["teacher_maj_correct"])
    rescued = sum(1 for r in low if not r["maj_correct"] and r["teacher_maj_correct"])
    harmed = sum(1 for r in low if r["maj_correct"] and not r["teacher_maj_correct"])
    both_wrong = sum(1 for r in low if not r["maj_correct"] and not r["teacher_maj_correct"])

    print(f"    Orig MAJ ✓ → Teacher MAJ ✓ (kept):      {both_correct}")
    print(f"    Orig MAJ ✗ → Teacher MAJ ✓ (rescued):   {rescued}")
    print(f"    Orig MAJ ✓ → Teacher MAJ ✗ (harmed):    {harmed}")
    print(f"    Orig MAJ ✗ → Teacher MAJ ✗ (still ✗):   {both_wrong}")
    print(f"    Net gain: {rescued - harmed:+d}")

    print(f"\n  {'─'*110}")
    print(f"  Sample Detail (first 20 low-SC problems):")
    print(f"  {'Idx':<5} {'Base SC':>8} {'Orig MAJ':>10} {'Teach MAJ':>10} {'Teach SC':>10} {'Transition':<15}")
    print(f"  {'─'*110}")
    for r in low[:20]:
        sc = f"{r['sc_score']:.2f}"
        maj_c = "✓" if r["maj_correct"] else "✗"
        teach_c = "✓" if r["teacher_maj_correct"] else "✗"
        teach_sc = f"{r.get('teacher_sc_score', 0.0):.2f}"
        
        if not r["maj_correct"] and r["teacher_maj_correct"]:
            trans = "RESCUED"
        elif r["maj_correct"] and not r["teacher_maj_correct"]:
            trans = "HARMED"
        elif r["maj_correct"] and r["teacher_maj_correct"]:
            trans = "kept ✓"
        else:
            trans = "still ✗"
            
        print(f"  {r['idx']:<5} {sc:>8} {maj_c:>10} {teach_c:>10} {teach_sc:>10} {trans:<15}")


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
    
    # We use greedy decoding for the teacher verification 
    # since we already have 64 unique contexts
    sampling_params = SamplingParams(
        n=1,
        temperature=0.0,
        max_tokens=args.max_tokens,
        stop=["<|eot_id|>", "</s>", "<|im_end|>", "Q:"],
    )

    all_prompts = []
    metadata_map = [] # To map prompt index back to (problem_idx, rollout_idx)
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
            # Default fallbacks
            "was_reprompted": False,
            "teacher_maj_answer": maj_answer,
            "teacher_maj_correct": (maj_answer == gt_norm),
            "teacher_sc_score": 0.0,
            "teacher_derived_answers": []
        }
        all_results.append(result)

        if sc_score < 0.3:
            responses = item.get("responses", [])
            extracted = item.get("extracted_answers", [])
            
            for r_idx, (resp, ans) in enumerate(zip(responses, extracted)):
                if not resp or ans == "[NO_ANSWER]":
                    continue
                
                norm_ans = strip_string(ans)
                is_mainstream = (norm_ans == maj_answer)
                
                prompt = build_verification_prompt(
                    item["problem"], resp, is_mainstream, tokenizer, args.model_path
                )
                
                all_prompts.append(prompt)
                metadata_map.append({
                    "problem_idx": idx,
                    "rollout_idx": r_idx
                })

    print(f"Total Teacher generation prompts built: {len(all_prompts)}")

    print("Generating Teacher Verification responses...")
    outputs = llm.generate(all_prompts, sampling_params)

    # Dictionary to collect new answers per problem mapping p_idx -> list of norm_answers
    teacher_answers_per_prob = {idx: [] for idx, item in enumerate(data) if item.get("sc_score", 1.0) < 0.3}

    for i, output in enumerate(outputs):
        p_idx = metadata_map[i]["problem_idx"]
        response_text = output.outputs[0].text
        
        raw_ans = extract_answer(response_text)
        norm_ans = strip_string(raw_ans) if raw_ans else "[NO_ANSWER]"
        
        teacher_answers_per_prob[p_idx].append(norm_ans)

    # Calculate new Majority Vote
    for p_idx, answers in teacher_answers_per_prob.items():
        all_results[p_idx]["was_reprompted"] = True
        all_results[p_idx]["teacher_derived_answers"] = answers
        
        valid_answers = [a for a in answers if a != "[NO_ANSWER]"]
        counter = Counter(valid_answers)
        most_common = counter.most_common(1)
        
        if most_common:
            best_ans, count = most_common[0]
            teacher_sc_score = count / len(valid_answers)
        else:
            best_ans = "[NO_ANSWER]"
            teacher_sc_score = 0.0
            
        all_results[p_idx]["teacher_maj_answer"] = best_ans
        all_results[p_idx]["teacher_maj_correct"] = (best_ans == all_results[p_idx]["gt_norm"])
        all_results[p_idx]["teacher_sc_score"] = teacher_sc_score

    print_full_comparison(all_results)
    
    if args.output_file:
        print(f"\nSaving detailed results to {args.output_file}")
        with open(args.output_file, "w", encoding="utf-8") as f:
            for r in all_results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("\nDone!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Teacher-Prompt Generation and Voting Experiment")
    parser.add_argument("--model_path", type=str, required=True, help="HuggingFace model path")
    parser.add_argument("--input_file", type=str, default="scripts/base.jsonl", help="Input JSONL")
    parser.add_argument("--output_file", type=str, default="scripts/teacher_voting_results.jsonl", help="Output JSONL")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="vLLM tensor parallel size")
    parser.add_argument("--max_model_len", type=int, default=4096, help="Max context length to restrict KV cache size")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.7, help="Fraction of GPU memory for vLLM")
    parser.add_argument("--enforce_eager", action="store_true", help="Disable CUDA graphs (saves some memory)")
    parser.add_argument("--max_tokens", type=int, default=4096, help="Max tokens for Teacher generation")
    args = parser.parse_args()
    main(args)
