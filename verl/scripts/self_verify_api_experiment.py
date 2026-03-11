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
from openai import OpenAI
import httpx
import concurrent.futures
from tqdm import tqdm


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


# =====================================================
# New Prompt Templates for Improved Verification
# =====================================================

PAIRWISE_TEMPLATE = """[Math Problem]
{problem}

[Pairwise Comparison Task]
Two candidate solutions were sampled from a model. Your job is to determine which one (if either) is correct by carefully checking the reasoning.

--- Solution A ---
{solution_a}
**Final Answer A: {answer_a}**

--- Solution B ---
{solution_b}
**Final Answer B: {answer_b}**

[Instructions]
1. Carefully read both solutions. Check each step for logical errors, computational mistakes, or unjustified leaps.
2. If one solution has a clear error that the other avoids, select the error-free one.
3. If both seem correct but give different answers, solve the problem yourself to break the tie.
4. If both are equally flawed or you cannot determine which is better, output "Tie".

Conclude with exactly one of: \\boxed{{A}}, \\boxed{{B}}, or \\boxed{{Tie}}."""

PAIRWISE_ANSWER_ONLY_TEMPLATE = """[Math Problem]
{problem}

[Pairwise Comparison Task]
Two candidate answers were sampled from a model. Determine which one is correct.

Answer A: {answer_a}
Answer B: {answer_b}

[Instructions]
1. Solve the problem yourself step by step.
2. Compare your result with Answer A and Answer B.
3. Select the answer that matches your derivation.

Conclude with exactly one of: \\boxed{{A}}, \\boxed{{B}}, or \\boxed{{Tie}} if neither matches."""

REVERSE_CHECK_TEMPLATE = """[Math Problem]
{problem}

[Reverse Verification Task]
Below are {num_candidates} candidate answers. Your job is to verify each answer by substituting it back into the problem conditions and checking whether it satisfies all constraints.
{options}

[Instructions]
1. For each candidate answer, substitute it into the original problem and check:
   - Does it satisfy all equations/inequalities/conditions in the problem?
   - Is it consistent with all given constraints?
   - Is the numerical value reasonable given the problem context?
2. Mark each candidate as VALID or INVALID with a brief justification.
3. If exactly one candidate is VALID, select it. If multiple are VALID, select the one with the most rigorous verification. If none are VALID, output "None".

Conclude by enclosing the selected candidate in \\boxed{{}} (e.g., \\boxed{{Candidate 2}}).
If no candidate passes verification, output \\boxed{{None}}."""


def build_verify_prompt(problem, candidates, show_freq=False, rollout_map=None):
    """
    Build verification prompt formatted as Qwen ChatML for the base model API.
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

    # Hardcoded Qwen ChatML structure to avoid requiring a local tokenizer
    prompt = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{content}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    
    return prompt, option_map


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

def run_experiment(data, max_tokens, candidate_fn, show_freq, exp_name, use_rollout=False):
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
                show_freq=show_freq,
                rollout_map=rollout_map
            )
            verify_indices.append(idx)
            verify_prompts.append(prompt)
            verify_option_maps.append(option_map)

    print(f"  Low-consistency problems to verify: {len(verify_indices)}")

    if verify_prompts:
        def call_api(prompt_text):
            # Temporarily remove proxies
            old_http = os.environ.pop("http_proxy", None)
            old_https = os.environ.pop("https_proxy", None)
            try:
                custom_http_client = httpx.Client(verify=False)
                client = OpenAI(
                    api_key=os.environ.get("AUTODL_API_KEY", "EMPTY"), 
                    base_url=os.environ.get("AUTODL_BASE_URL", "https://u630113-8ba4-8da84932.westc.seetacloud.com:8443/v1"),
                    http_client=custom_http_client
                )
                response = client.completions.create(
                    model=os.environ.get("AUTODL_MODEL", "qwen3-4b-base"),
                    prompt=prompt_text,
                    max_tokens=max_tokens,
                    temperature=0.0,
                )
                return response.choices[0].text
            except Exception as e:
                print(f"API Error: {e}")
                return ""
            finally:
                if old_http: os.environ["http_proxy"] = old_http
                if old_https: os.environ["https_proxy"] = old_https

        model_responses = []
        import os
        max_workers = int(os.environ.get("API_VERIFY_MAX_WORKERS", "32"))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(call_api, p) for p in verify_prompts]
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="API Verification"):
                pass
            model_responses = [f.result() for f in futures]

        for i, response in enumerate(model_responses):
            idx = verify_indices[i]
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
# Pairwise Tournament Experiment
# =====================================================

def build_pairwise_prompt(problem, ans_a, ans_b, rollout_a=None, rollout_b=None):
    """Build a pairwise comparison prompt. Optionally include rollouts."""
    if rollout_a and rollout_b:
        template = PAIRWISE_TEMPLATE
        content = template.format(
            problem=problem,
            solution_a=rollout_a, answer_a=ans_a,
            solution_b=rollout_b, answer_b=ans_b
        )
    else:
        template = PAIRWISE_ANSWER_ONLY_TEMPLATE
        content = template.format(
            problem=problem, answer_a=ans_a, answer_b=ans_b
        )

    prompt = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{content}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    return prompt


def parse_pairwise_response(response_text):
    """Parse pairwise response: returns 'A', 'B', or 'Tie'."""
    raw = extract_answer(response_text)
    if raw is None:
        return "Tie"
    cleaned = raw.strip().upper()
    # Remove LaTeX wrappers
    cleaned = re.sub(r'\\text\{([^}]*)\}', r'\1', cleaned)
    cleaned = re.sub(r'\\mathrm\{([^}]*)\}', r'\1', cleaned)
    cleaned = cleaned.strip().upper()
    if cleaned in ("A", "B", "TIE"):
        return cleaned if cleaned != "TIE" else "Tie"
    if "TIE" in cleaned:
        return "Tie"
    return "Tie"


def run_pairwise_tournament_experiment(data, max_tokens, candidate_fn, exp_name, use_rollout=False):
    """
    Pairwise tournament: compare Top-K candidates pairwise.
    For each pair, shuffle order to counteract position bias.
    The candidate winning the most comparisons is selected.
    """
    print(f"\n{'='*80}")
    print(f"Experiment: {exp_name}")
    print(f"{'='*80}")

    results = []
    # Collect all pairwise prompts across all problems
    all_api_calls = []  # list of (problem_idx, pair_idx, prompt, swap_flag, ans_i, ans_j)

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

        if consistency <= 0.3 and len(candidates) >= 2:
            top_k = min(3, len(candidates))
            top_cands = candidates[:top_k]

            # Build rollout map if needed
            rollout_map = {}
            if use_rollout and "responses" in item:
                responses = item["responses"]
                MAX_ROLLOUT_LEN = 1500
                for ans, _ in top_cands:
                    matching = []
                    for j, ext_ans in enumerate(answers):
                        if ext_ans == ans and j < len(responses):
                            matching.append(responses[j].strip())
                    if not matching:
                        continue
                    short = [r for r in matching if len(r) <= MAX_ROLLOUT_LEN]
                    if short:
                        rollout_map[ans] = max(short, key=len)
                    else:
                        rollout_map[ans] = min(matching, key=len)

            # Generate all pairs
            for i in range(top_k):
                for j in range(i + 1, top_k):
                    ans_i, _ = top_cands[i]
                    ans_j, _ = top_cands[j]
                    # Randomly swap order to counteract position bias
                    swap = random.random() < 0.5
                    if swap:
                        a_ans, b_ans = ans_j, ans_i
                        a_roll = rollout_map.get(ans_j)
                        b_roll = rollout_map.get(ans_i)
                    else:
                        a_ans, b_ans = ans_i, ans_j
                        a_roll = rollout_map.get(ans_i)
                        b_roll = rollout_map.get(ans_j)

                    prompt = build_pairwise_prompt(
                        item["problem"], a_ans, b_ans,
                        rollout_a=a_roll if use_rollout else None,
                        rollout_b=b_roll if use_rollout else None
                    )
                    all_api_calls.append((idx, i, j, prompt, swap, ans_i, ans_j))

    print(f"  Low-consistency problems: {sum(1 for r in results if r['consistency'] <= 0.3)}")
    print(f"  Total pairwise API calls: {len(all_api_calls)}")

    if all_api_calls:
        def call_api(prompt_text):
            old_http = os.environ.pop("http_proxy", None)
            old_https = os.environ.pop("https_proxy", None)
            try:
                custom_http_client = httpx.Client(verify=False)
                client = OpenAI(
                    api_key=os.environ.get("AUTODL_API_KEY", "EMPTY"),
                    base_url=os.environ.get("AUTODL_BASE_URL", "https://u630113-8ba4-8da84932.westc.seetacloud.com:8443/v1"),
                    http_client=custom_http_client
                )
                response = client.completions.create(
                    model=os.environ.get("AUTODL_MODEL", "qwen3-4b-base"),
                    prompt=prompt_text,
                    max_tokens=max_tokens,
                    temperature=0.0,
                )
                return response.choices[0].text
            except Exception as e:
                print(f"API Error: {e}")
                return ""
            finally:
                if old_http: os.environ["http_proxy"] = old_http
                if old_https: os.environ["https_proxy"] = old_https

        import os
        max_workers = int(os.environ.get("API_VERIFY_MAX_WORKERS", "32"))
        prompts_only = [c[3] for c in all_api_calls]
        api_responses = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(call_api, p) for p in prompts_only]
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Pairwise API"):
                pass
            api_responses = [f.result() for f in futures]

        # Aggregate pairwise wins per problem
        from collections import defaultdict
        problem_wins = defaultdict(lambda: defaultdict(int))  # {prob_idx: {answer: win_count}}

        for call_info, response in zip(all_api_calls, api_responses):
            prob_idx, i, j, prompt, swap, ans_i, ans_j = call_info
            verdict = parse_pairwise_response(response)

            if verdict == "A":
                winner = ans_j if swap else ans_i
            elif verdict == "B":
                winner = ans_i if swap else ans_j
            else:
                # Tie: both get half a win
                problem_wins[prob_idx][ans_i] += 0.5
                problem_wins[prob_idx][ans_j] += 0.5
                continue
            problem_wins[prob_idx][winner] += 1

        # Select the candidate with the most wins for each verified problem
        for prob_idx, wins in problem_wins.items():
            if not wins:
                continue
            best_ans = max(wins.keys(), key=lambda a: wins[a])
            results[prob_idx]["verified_answer"] = strip_string(best_ans)
            results[prob_idx]["verify_source"] = "pairwise"
            results[prob_idx]["pairwise_wins"] = dict(wins)

    # Fallback for non-verified
    for r in results:
        if "verified_answer" not in r:
            r["verified_answer"] = r["maj_answer"]
            r["verify_source"] = "unchanged"

    return results


# =====================================================
# Multi-Sample Verification Voting Experiment
# =====================================================

def run_multi_sample_experiment(data, max_tokens, candidate_fn, exp_name,
                                use_rollout=False, num_samples=5, temperature=0.6):
    """
    Run verification N times with temperature > 0, then vote on the selected candidate.
    """
    print(f"\n{'='*80}")
    print(f"Experiment: {exp_name} (N={num_samples}, T={temperature})")
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

        if consistency <= 0.3 and len(candidates) >= 2:
            rollout_map = None
            if use_rollout and "responses" in item:
                rollout_map = {}
                responses_list = item["responses"]
                MAX_ROLLOUT_LEN = 1500
                for ans, _ in candidates[:5]:
                    matching = []
                    for j, ext_ans in enumerate(answers):
                        if ext_ans == ans and j < len(responses_list):
                            matching.append(responses_list[j].strip())
                    if not matching:
                        continue
                    short = [r for r in matching if len(r) <= MAX_ROLLOUT_LEN]
                    if short:
                        rollout_map[ans] = max(short, key=len)
                    else:
                        rollout_map[ans] = min(matching, key=len)

            prompt, option_map = build_verify_prompt(
                item["problem"], candidates[:5],
                show_freq=False,
                rollout_map=rollout_map
            )
            verify_indices.append(idx)
            verify_prompts.append(prompt)
            verify_option_maps.append(option_map)

    print(f"  Low-consistency problems to verify: {len(verify_indices)}")
    print(f"  Total API calls: {len(verify_indices) * num_samples}")

    if verify_prompts:
        def call_api(prompt_text):
            old_http = os.environ.pop("http_proxy", None)
            old_https = os.environ.pop("https_proxy", None)
            try:
                custom_http_client = httpx.Client(verify=False)
                client = OpenAI(
                    api_key=os.environ.get("AUTODL_API_KEY", "EMPTY"),
                    base_url=os.environ.get("AUTODL_BASE_URL", "https://u630113-8ba4-8da84932.westc.seetacloud.com:8443/v1"),
                    http_client=custom_http_client
                )
                response = client.completions.create(
                    model=os.environ.get("AUTODL_MODEL", "qwen3-4b-base"),
                    prompt=prompt_text,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                return response.choices[0].text
            except Exception as e:
                print(f"API Error: {e}")
                return ""
            finally:
                if old_http: os.environ["http_proxy"] = old_http
                if old_https: os.environ["https_proxy"] = old_https

        # Expand prompts: each prompt repeated num_samples times
        expanded_prompts = []
        expanded_meta = []  # (original_index_in_verify, sample_id)
        for vi, prompt in enumerate(verify_prompts):
            for s in range(num_samples):
                expanded_prompts.append(prompt)
                expanded_meta.append((vi, s))

        import os
        max_workers = int(os.environ.get("API_VERIFY_MAX_WORKERS", "32"))
        api_responses = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(call_api, p) for p in expanded_prompts]
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Multi-Sample API"):
                pass
            api_responses = [f.result() for f in futures]

        # Group responses by problem and vote
        from collections import defaultdict
        problem_votes = defaultdict(list)  # vi -> list of parsed answers

        for (vi, s), response in zip(expanded_meta, api_responses):
            verified = parse_verify_response(response, verify_option_maps[vi])
            if verified is not None:
                problem_votes[vi].append(strip_string(verified))

        for vi in range(len(verify_indices)):
            idx = verify_indices[vi]
            votes = problem_votes.get(vi, [])

            if votes:
                vote_counter = Counter(votes)
                best_answer = vote_counter.most_common(1)[0][0]
                results[idx]["verified_answer"] = best_answer
                results[idx]["verify_source"] = "multi_vote"
                results[idx]["vote_distribution"] = dict(vote_counter)
            else:
                # All samples failed to parse → fallback
                results[idx]["verified_answer"] = results[idx]["candidates"][0][0]
                results[idx]["verify_source"] = "fallback_top1"

    for r in results:
        if "verified_answer" not in r:
            r["verified_answer"] = r["maj_answer"]
            r["verify_source"] = "unchanged"

    return results


# =====================================================
# Reverse Substitution Check Experiment
# =====================================================

def build_reverse_check_prompt(problem, candidates):
    """Build reverse-substitution verification prompt."""
    options_lines = []
    option_map = {}
    for i, (ans, freq) in enumerate(candidates):
        label = f"Candidate {i+1}"
        options_lines.append(f"{label}: {ans}")
        option_map[label] = ans

    options_text = "\n".join(options_lines)
    content = REVERSE_CHECK_TEMPLATE.format(
        problem=problem, options=options_text, num_candidates=len(candidates)
    )

    prompt = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{content}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    return prompt, option_map


def run_reverse_check_experiment(data, max_tokens, candidate_fn, exp_name):
    """
    Reverse substitution: ask model to substitute each candidate back
    into the problem and check which satisfies all constraints.
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

        if consistency <= 0.3 and len(candidates) >= 2:
            prompt, option_map = build_reverse_check_prompt(
                item["problem"], candidates[:5]
            )
            verify_indices.append(idx)
            verify_prompts.append(prompt)
            verify_option_maps.append(option_map)

    print(f"  Low-consistency problems to verify: {len(verify_indices)}")

    if verify_prompts:
        def call_api(prompt_text):
            old_http = os.environ.pop("http_proxy", None)
            old_https = os.environ.pop("https_proxy", None)
            try:
                custom_http_client = httpx.Client(verify=False)
                client = OpenAI(
                    api_key=os.environ.get("AUTODL_API_KEY", "EMPTY"),
                    base_url=os.environ.get("AUTODL_BASE_URL", "https://u630113-8ba4-8da84932.westc.seetacloud.com:8443/v1"),
                    http_client=custom_http_client
                )
                response = client.completions.create(
                    model=os.environ.get("AUTODL_MODEL", "qwen3-4b-base"),
                    prompt=prompt_text,
                    max_tokens=max_tokens,
                    temperature=0.0,
                )
                return response.choices[0].text
            except Exception as e:
                print(f"API Error: {e}")
                return ""
            finally:
                if old_http: os.environ["http_proxy"] = old_http
                if old_https: os.environ["https_proxy"] = old_https

        import os
        max_workers = int(os.environ.get("API_VERIFY_MAX_WORKERS", "32"))
        api_responses = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(call_api, p) for p in verify_prompts]
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Reverse Check API"):
                pass
            api_responses = [f.result() for f in futures]

        for i, response in enumerate(api_responses):
            idx = verify_indices[i]
            verified = parse_verify_response(response, verify_option_maps[i])

            results[idx]["verify_prompt"] = verify_prompts[i]
            results[idx]["verify_response"] = response

            if verified is not None:
                results[idx]["verified_answer"] = strip_string(verified)
                results[idx]["verify_source"] = "reverse_check"
            else:
                results[idx]["verified_answer"] = results[idx]["candidates"][0][0]
                results[idx]["verify_source"] = "fallback_top1"

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

    # ========================================
    # Experiment 1 (Baseline): Freq Top-K answer-only
    # ========================================
    candidate_fn_5 = lambda ans: get_frequency_candidates(ans, top_k=5)

    print("\n" + "#"*80)
    print("# Running Experiment 1: Baseline (answer-only, single-shot, T=0)")
    print("#"*80)
    results_baseline = run_experiment(
        data, args.max_verify_tokens,
        candidate_fn=candidate_fn_5,
        show_freq=False,
        exp_name="Baseline: Answer-Only Single-Shot",
        use_rollout=False
    )

    # ========================================
    # Experiment 2: Pairwise Tournament (answer-only)
    # ========================================
    print("\n" + "#"*80)
    print("# Running Experiment 2: Pairwise Tournament (answer-only)")
    print("#"*80)
    results_pairwise = run_pairwise_tournament_experiment(
        data, args.max_verify_tokens,
        candidate_fn=candidate_fn_5,
        exp_name="Pairwise Tournament (answer-only)",
        use_rollout=False
    )

    # ========================================
    # Experiment 3: Multi-Sample Voting (N=5, T=0.6)
    # ========================================
    print("\n" + "#"*80)
    print("# Running Experiment 3: Multi-Sample Voting (N=5, T=0.6)")
    print("#"*80)
    results_multi = run_multi_sample_experiment(
        data, args.max_verify_tokens,
        candidate_fn=candidate_fn_5,
        exp_name="Multi-Sample Voting",
        use_rollout=False,
        num_samples=5,
        temperature=0.6
    )

    # ========================================
    # Experiment 4: Reverse Substitution Check
    # ========================================
    print("\n" + "#"*80)
    print("# Running Experiment 4: Reverse Substitution Check")
    print("#"*80)
    results_reverse = run_reverse_check_experiment(
        data, args.max_verify_tokens,
        candidate_fn=candidate_fn_5,
        exp_name="Reverse Substitution Check"
    )

    # Collect all experiments for comparison
    all_experiments = [
        ("Baseline", results_baseline),
        ("Pairwise", results_pairwise),
        ("MultiVote", results_multi),
        ("ReverseCheck", results_reverse),
    ]

    # ========================================
    # Side-by-Side Comparison Table
    # ========================================
    print(f"\n{'='*100}")
    print("SIDE-BY-SIDE COMPARISON: All Experiments")
    print(f"{'='*100}")

    buckets = [
        ("Low(<=0.3)", lambda r: r["consistency"] <= 0.3),
        ("Mid(0.3-0.7)", lambda r: 0.3 < r["consistency"] <= 0.7),
        ("High(>0.7)", lambda r: r["consistency"] > 0.7),
        ("All", lambda r: True),
    ]

    for bname, bfn in buckets:
        print(f"\n  {bname}:")
        print(f"    {'Experiment':<20} {'N':<6} {'MAJ':<8} {'Verify':<8} {'Gain':<8} "
              f"{'Rescued':<9} {'Harmed':<8} {'Net':<6}")
        print(f"    " + "-" * 73)

        for exp_name, exp_results in all_experiments:
            bucket = [r for r in exp_results if bfn(r)]
            if not bucket:
                continue
            n = len(bucket)
            maj_c = sum(1 for r in bucket if r["maj_correct"])
            ver_c = sum(1 for r in bucket if r["verified_answer"] == r["gt_norm"])
            rescued = sum(1 for r in bucket
                          if not r["maj_correct"] and r["verified_answer"] == r["gt_norm"])
            harmed = sum(1 for r in bucket
                         if r["maj_correct"] and r["verified_answer"] != r["gt_norm"])

            maj_acc = maj_c / n
            ver_acc = ver_c / n
            gain = ver_acc - maj_acc

            print(f"    {exp_name:<20} {n:<6} {maj_acc:<8.1%} {ver_acc:<8.1%} {gain:<+8.1%} "
                  f"{rescued:<9d} {harmed:<8d} {rescued-harmed:<+6d}")

    # Low-consistency detail per experiment
    print(f"\n{'='*100}")
    print("Low-Consistency Detail by Experiment")
    print(f"{'='*100}")
    print(f"  {'Experiment':<20} {'Verify Acc':<18} {'Fallback Acc':<15} {'GT in Cands':<15}")
    print(f"  " + "-" * 68)

    for exp_name, exp_results in all_experiments:
        low = [r for r in exp_results if r["consistency"] <= 0.3
               and r["verify_source"] != "unchanged"]
        if not low:
            print(f"  {exp_name:<20} N/A")
            continue

        # Active verifications (not fallback)
        active = [r for r in low if r["verify_source"] not in ("fallback_top1",)]
        fb_items = [r for r in low if r["verify_source"] == "fallback_top1"]
        gt_in = sum(1 for r in low if r["gt_norm"] in {a for a, _ in r["candidates"]})

        ac = sum(1 for r in active if r["verified_answer"] == r["gt_norm"]) if active else 0
        fc = sum(1 for r in fb_items if r["verified_answer"] == r["gt_norm"]) if fb_items else 0

        ac_str = f"{ac}/{len(active)} ({100*ac/len(active):.1f}%)" if active else "N/A"
        fc_str = f"{fc}/{len(fb_items)} ({100*fc/len(fb_items):.1f}%)" if fb_items else "N/A"

        print(f"  {exp_name:<20} {ac_str:<18} {fc_str:<15} {gt_in}/{len(low)} ({100*gt_in/len(low):.1f}%)")

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
                for exp_name, exp_results in all_experiments:
                    r = exp_results[i]
                    prefix = exp_name.replace(" ", "_")
                    record[f"{prefix}_consistency"] = r["consistency"]
                    record[f"{prefix}_candidates"] = [(a, round(freq, 4)) for a, freq in r["candidates"]]
                    record[f"{prefix}_verified"] = r["verified_answer"]
                    record[f"{prefix}_source"] = r["verify_source"]
                    record[f"{prefix}_raw_model_answer"] = r.get("raw_model_answer")
                f.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="K-Sweep Self-Verification with Rollout")
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, default="self_verify_results.jsonl")
    parser.add_argument("--num_bootstrap", type=int, default=1000)
    parser.add_argument("--max_verify_tokens", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)

