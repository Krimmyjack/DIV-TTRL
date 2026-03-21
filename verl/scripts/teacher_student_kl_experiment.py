"""
Teacher-Student KL Penalty Pseudo-Label Experiment

For each problem with SC < 0.3, evaluate all 64 original rollouts (Student).
Construct a prompt for the Teacher model (same base model, different perspective)
and compute the KL Penalty:
    KL_Penalty_Sum = Student_Sum_Logprob - Teacher_Sum_Logprob
    KL_Penalty_Mean = Student_Mean_Logprob - Teacher_Mean_Logprob

Selects the rollout with the lowest penalty as the pseudo-label, and compares
its accuracy against Majority Voting (MAJ-64).

Usage:
python scripts/teacher_student_kl_experiment.py \
    --model_path Qwen/Qwen2.5-7B \
    --input_file scripts/base.jsonl \
    --output_file scripts/kl_penalty_results.jsonl
"""

import json
import argparse
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

MAINSTREAM_PROMPT = """Below is a math problem and a candidate solution. 
This solution represents the mainstream (majority) approach, but it might be stuck in a local optimum or contain persistent logical flaws. 
Please verify the correctness of this logic, and if there are errors, locate them and try to fix them.

**[Problem]**
{problem}

**[Mainstream Solution]**
"""

NON_MAINSTREAM_PROMPT = """Below is a math problem and a candidate solution. 
This solution represents a non-mainstream (minority) approach, offering an unconventional perspective or alternative method. 
Please verify the logic of this perspective, assess its validity, and attempt to continue deriving along this direction.

**[Problem]**
{problem}

**[Non-Mainstream Solution]**
"""


def format_teacher_prompt(problem, rollout_text, is_mainstream, tokenizer, model_path):
    """
    Returns (full_prompt_string, prefix_string)
    We need prefix_string because we want to isolate the logprobs of `rollout_text`.
    """
    # 1. Build the prefix (Teacher instructions)
    template = MAINSTREAM_PROMPT if is_mainstream else NON_MAINSTREAM_PROMPT
    content_prefix = template.format(problem=problem)
    
    name = model_path.lower()
    full_str = None
    prefix_str = None
    
    if "instruct" in name or "llama" in name:
        try:
            # We want the prompt to be roughly what the model would see if it generated it.
            # But here we are forcing the context. 
            # We can format the 'user' part and then append the rollout as the 'assistant' target.
            messages_prefix = [{"role": "user", "content": content_prefix}]
            prefix_str = tokenizer.apply_chat_template(messages_prefix, tokenize=False, add_generation_prompt=True)
            full_str = prefix_str + rollout_text
        except Exception:
            pass

    if full_str is None:
        # Fallback to plain text 
        prefix_str = content_prefix
        full_str = prefix_str + rollout_text
        
    return full_str, prefix_str


# =====================================================
# 3. Main Logic
# =====================================================

def print_full_comparison(results):
    buckets = [
        ("Low (SC<0.3)",  lambda r: r["sc_score"] < 0.3),
        ("Mid (0.3-0.7)", lambda r: 0.3 <= r["sc_score"] < 0.7),
        ("High (≥0.7)",   lambda r: r["sc_score"] >= 0.7),
        ("ALL",           lambda r: True),
    ]

    print("\n" + "=" * 110)
    print("TEACHER-STUDENT KL PENALTY EXPERIMENT RESULTS")
    print("=" * 110)

    header = (
        f"  {'Bucket':<16} {'N':>5}  "
        f"{'MAJ Acc':>8}  {'KL-Sum Acc':>11}  {'KL-Mean Acc':>12}  "
        f"{'Δ(Sum-MAJ)':>11}  {'Δ(Mean-MAJ)':>12}"
    )
    print(header)
    print("  " + "-" * 106)

    for bname, bfn in buckets:
        bucket = [r for r in results if bfn(r)]
        if not bucket:
            continue
        n = len(bucket)

        maj_correct = sum(1 for r in bucket if r["maj_correct"])
        kl_sum_correct = sum(1 for r in bucket if r["kl_sum_correct"])
        kl_mean_correct = sum(1 for r in bucket if r["kl_mean_correct"])

        maj_acc = maj_correct / n
        kl_sum_acc = kl_sum_correct / n
        kl_mean_acc = kl_mean_correct / n

        print(
            f"  {bname:<16} {n:>5}  "
            f"{maj_acc:>7.1%}  {kl_sum_acc:>10.1%}   {kl_mean_acc:>11.1%}   "
            f"{kl_sum_acc - maj_acc:>+10.1%}  {kl_mean_acc - maj_acc:>+11.1%}"
        )

    low = [r for r in results if r["sc_score"] < 0.3]
    if not low:
        return

    n_low = len(low)
    print(f"\n  Transition Matrix (MAJ → KL-Mean):")
    rescued = sum(1 for r in low if not r["maj_correct"] and r["kl_mean_correct"])
    harmed = sum(1 for r in low if r["maj_correct"] and not r["kl_mean_correct"])
    print(f"    MAJ ✗ → KL-Mean ✓ (rescued): {rescued}")
    print(f"    MAJ ✓ → KL-Mean ✗ (harmed):  {harmed}")
    print(f"    Net gain: {rescued - harmed:+d}")
    
    print(f"\n  Transition Matrix (MAJ → KL-Sum):")
    rescued_s = sum(1 for r in low if not r["maj_correct"] and r["kl_sum_correct"])
    harmed_s = sum(1 for r in low if r["maj_correct"] and not r["kl_sum_correct"])
    print(f"    MAJ ✗ → KL-Sum ✓ (rescued):  {rescued_s}")
    print(f"    MAJ ✓ → KL-Sum ✗ (harmed):   {harmed_s}")
    print(f"    Net gain: {rescued_s - harmed_s:+d}")

    # Details
    print(f"\n  {'─'*110}")
    print(f"  Sample Detail (first 20 low-SC problems):")
    print(f"  {'Idx':<5} {'SC':>6} {'MAJ':>5} {'KL-Mean Correct':>16} {'KL-Sum Correct':>15} {'Mean Pen':>10} {'Sum Pen':>10}")
    print(f"  {'─'*110}")
    for r in low[:20]:
        sc = f"{r['sc_score']:.2f}"
        maj_c = "✓" if r["maj_correct"] else "✗"
        klm_c = "✓" if r["kl_mean_correct"] else "✗"
        kls_c = "✓" if r["kl_sum_correct"] else "✗"
        mean_pen = f"{r.get('kl_mean_best_penalty', 0):.4f}" if "kl_mean_best_penalty" in r else "N/A"
        sum_pen = f"{r.get('kl_sum_best_penalty', 0):.2f}" if "kl_sum_best_penalty" in r else "N/A"
        print(f"  {r['idx']:<5} {sc:>6} {maj_c:>5} {klm_c:>16} {kls_c:>15} {mean_pen:>10} {sum_pen:>10}")


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
    
    # We use vLLM just for scoring (logprobs) without generating new text.
    # We set max_tokens=1 and ask for prompt_logprobs.
    llm = LLM(
        model=args.model_path,
        tokenizer=args.model_path,
        tensor_parallel_size=torch.cuda.device_count() or 1,
        trust_remote_code=True,
        dtype="auto",
    )
    sampling_params = SamplingParams(max_tokens=1, prompt_logprobs=1, temperature=0.0)

    # We will batch all rollouts that need scoring
    all_prompts_to_score = []
    metadata_map = [] # To map prompt index back to (problem_idx, rollout_idx)

    # Process all data
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
            "kl_sum_answer": maj_answer,
            "kl_sum_correct": (maj_answer == gt_norm),
            "kl_mean_answer": maj_answer,
            "kl_mean_correct": (maj_answer == gt_norm),
            "was_scored": False
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
                
                full_str, prefix_str = format_teacher_prompt(
                    item["problem"], resp, is_mainstream, tokenizer, args.model_path
                )
                
                prefix_tokens = tokenizer.encode(prefix_str)
                full_tokens = tokenizer.encode(full_str)
                # Count how many tokens belong to the Teacher prefix
                prefix_len = len(prefix_tokens)
                
                all_prompts_to_score.append(full_tokens)
                metadata_map.append({
                    "problem_idx": idx,
                    "rollout_idx": r_idx,
                    "prefix_len": prefix_len,
                    "response_text": resp,
                    "response_ans_norm": norm_ans
                })

    print(f"Total rollouts to score: {len(all_prompts_to_score)}")

    # Score with vLLM
    print("Scoring with vLLM prompt_logprobs...")
    # prompt_token_ids is a cleaner way to avoid tokenization differences
    outputs = llm.generate(prompt_token_ids=all_prompts_to_score, sampling_params=sampling_params)

    # Aggregate results
    # problem_idx -> list of dicts: {rollout_idx, kl_sum, kl_mean, ans_norm}
    scored_data = {item["idx"]: [] for item in data if item.get("sc_score", 1.0) < 0.3}

    for i, output in enumerate(outputs):
        meta = metadata_map[i]
        p_idx = meta["problem_idx"]
        r_idx = meta["rollout_idx"]
        ans_norm = meta["response_ans_norm"]
        prefix_len = meta["prefix_len"]
        
        # Original Student metrics
        metric = data[p_idx]["response_metrics"][r_idx]
        student_sum = metric.get("sum_logprob", 0.0)
        student_mean = metric.get("mean_logprob", 0.0)
        
        # Teacher metrics
        prompt_logprobs = output.prompt_logprobs
        
        # vLLM returns prompt_logprobs as a list of dictionaries.
        # Index 0 is typically None (no logprob for first token).
        teacher_sum = 0.0
        teacher_tokens_counted = 0
        
        if prompt_logprobs is not None:
            # We want the logprobs for the tokens AFTER the prefix_len
            # In token-by-token generation, prompt_logprobs[j] predicts token at index j
            # So the logprob of the token at index j is at prompt_logprobs[j]
            # Thus, we sum from j = prefix_len to len(prompt_logprobs)-1
            for j in range(prefix_len, len(prompt_logprobs)):
                lp_dict = prompt_logprobs[j]
                if lp_dict is not None and len(lp_dict) > 0:
                    # vLLM prompt_logprobs[j] is {token_id: Logprob(...)}
                    # Get the most likely or the specifically chosen token's logprob
                    # Actually, we want the logprob of the *actual* token in the sequence.
                    # Which token is it? output.prompt_token_ids[j]
                    token_id = output.prompt_token_ids[j]
                    if token_id in lp_dict:
                        teacher_sum += lp_dict[token_id].logprob
                        teacher_tokens_counted += 1
        
        teacher_mean = teacher_sum / teacher_tokens_counted if teacher_tokens_counted > 0 else 0.0
        
        # Penalties
        # min Penalty = most agreement
        kl_sum = student_sum - teacher_sum
        kl_mean = student_mean - teacher_mean
        
        scored_data[p_idx].append({
            "rollout_idx": r_idx,
            "ans_norm": ans_norm,
            "kl_sum": kl_sum,
            "kl_mean": kl_mean,
            "teacher_sum": teacher_sum,
            "teacher_mean": teacher_mean,
            "student_sum": student_sum,
            "student_mean": student_mean,
        })

    # Pick best
    for p_idx, rollouts in scored_data.items():
        if not rollouts:
            continue
            
        all_results[p_idx]["was_scored"] = True
        all_results[p_idx]["rollouts_details"] = rollouts
        
        # Sort by KL Sum (ascending)
        best_sum_rollout = min(rollouts, key=lambda x: x["kl_sum"])
        ans_sum = best_sum_rollout["ans_norm"]
        all_results[p_idx]["kl_sum_answer"] = ans_sum
        all_results[p_idx]["kl_sum_correct"] = (ans_sum == all_results[p_idx]["gt_norm"])
        all_results[p_idx]["kl_sum_best_penalty"] = best_sum_rollout["kl_sum"]
        
        # Sort by KL Mean (ascending)
        best_mean_rollout = min(rollouts, key=lambda x: x["kl_mean"])
        ans_mean = best_mean_rollout["ans_norm"]
        all_results[p_idx]["kl_mean_answer"] = ans_mean
        all_results[p_idx]["kl_mean_correct"] = (ans_mean == all_results[p_idx]["gt_norm"])
        all_results[p_idx]["kl_mean_best_penalty"] = best_mean_rollout["kl_mean"]

    print_full_comparison(all_results)
    
    if args.output_file:
        print(f"\nSaving detailed results to {args.output_file}")
        with open(args.output_file, "w", encoding="utf-8") as f:
            for r in all_results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("\nDone!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Teacher-Student KL Penalty Pseudo-Label Experiment")
    parser.add_argument("--model_path", type=str, required=True, help="HuggingFace model path")
    parser.add_argument("--input_file", type=str, default="scripts/base.jsonl", help="Input JSONL")
    parser.add_argument("--output_file", type=str, default="scripts/kl_penalty_results.jsonl", help="Output JSONL")
    args = parser.parse_args()
    main(args)
