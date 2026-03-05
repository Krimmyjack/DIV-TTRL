"""
Self-Verification Test Script using ModelScope API (Qwen/Qwen3-4B).
Tests the verification prompt on a few low-consistency samples.
"""

import json
import re
import time
from collections import Counter
from openai import OpenAI

# =====================================================
# Config
# =====================================================

API_KEY = "ms-06ae5daf-b4e6-4451-83ad-c6a272397f65"
BASE_URL = "https://api-inference.modelscope.cn/v1"
MODEL = "Qwen/Qwen3-4B"
INPUT_FILE = r"D:\学习\科研\DIV-TTRL-PR\verl\scripts\qwen64.jsonl"
NUM_SAMPLES = 10   # Number of low-consistency samples to test

# =====================================================
# Math answer normalization (simplified)
# =====================================================

def strip_string(s):
    """Basic normalization for math answers."""
    if s is None:
        return ""
    s = str(s).strip()
    # Remove \boxed{}
    m = re.search(r'\\boxed\{(.+)\}', s)
    if m:
        s = m.group(1)
    # Remove $...$
    s = s.strip('$').strip()
    # Remove trailing period
    if s.endswith('.'):
        s = s[:-1]
    return s

def extract_answer(text):
    """Extract answer from \\boxed{...}."""
    if not text:
        return None
    # Find last \boxed{...}
    matches = re.findall(r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}', text)
    if matches:
        return matches[-1].strip()
    return None


# =====================================================
# Prompt Templates
# =====================================================

SYSTEM_PROMPT = """You are an expert mathematical reasoning judge. Your task is to rigorously evaluate candidate solutions to a math problem, assess the quality of their reasoning, and select the most reliable one."""

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
# Candidate Extraction
# =====================================================

def get_frequency_candidates(answers, top_k=5):
    """Get top-K most frequent answers as candidates."""
    valid = [a for a in answers if a and a != "[NO_ANSWER]"]
    if not valid:
        return [], 0.0
    freq = Counter(valid)
    N = len(valid)
    majority = freq.most_common(1)[0][0]
    consistency = freq[majority] / N
    candidates = [(ans, cnt / N) for ans, cnt in freq.most_common(top_k)]
    return candidates, consistency


# =====================================================
# Build Prompt
# =====================================================

def build_verify_prompt(problem, candidates):
    """Build verification prompt for API call."""
    options_lines = []
    option_map = {}
    for i, (ans, freq) in enumerate(candidates):
        label = f"Candidate {i+1}"
        options_lines.append(f"{label}: {ans}")
        option_map[label] = ans

    options_text = "\n".join(options_lines)

    content = USER_TEMPLATE_NO_ROLLOUT.format(
        problem=problem,
        options=options_text,
        num_candidates=len(candidates)
    )

    return content, option_map


# =====================================================
# Parse Response
# =====================================================

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
# API Call
# =====================================================

def call_api(client, system_prompt, user_content, max_tokens=4096):
    """Call ModelScope API with thinking mode and return (thinking_text, answer_text)."""
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=0.0,
            max_tokens=max_tokens,
            stream=True,
            extra_body={"enable_thinking": True},
        )
        # Collect streaming response: thinking + answer
        thinking_text = ""
        answer_text = ""
        for chunk in response:
            if chunk.choices:
                delta = chunk.choices[0].delta
                # Thinking content (reasoning process)
                thinking_chunk = getattr(delta, 'reasoning_content', '') or ''
                if thinking_chunk:
                    thinking_text += thinking_chunk
                # Answer content (final response)
                answer_chunk = delta.content or ''
                if answer_chunk:
                    answer_text += answer_chunk
        return thinking_text, answer_text
    except Exception as e:
        print(f"  API Error: {e}")
        return None, None


# =====================================================
# Main
# =====================================================

def main():
    # Load data
    print(f"Loading {INPUT_FILE}")
    data = []
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    print(f"Loaded {len(data)} problems")

    # Initialize API client
    print(f"Initializing API client: {MODEL}")
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    # Find low-consistency samples
    low_sc_samples = []
    for i, item in enumerate(data):
        answers = item["extracted_answers"]
        candidates, consistency = get_frequency_candidates(answers, top_k=5)
        gt_norm = strip_string(str(item["answer"]))
        maj_answer = strip_string(str(item.get("sc_answer", "")))

        if consistency <= 0.3 and len(candidates) >= 2:
            low_sc_samples.append({
                "index": i,
                "item": item,
                "candidates": candidates,
                "consistency": consistency,
                "gt_norm": gt_norm,
                "maj_answer": maj_answer,
            })

    print(f"Found {len(low_sc_samples)} low-consistency samples (sc <= 0.3)")
    print(f"Testing on first {NUM_SAMPLES} samples\n")

    # Test on a few samples
    results = []
    for sample_idx, sample in enumerate(low_sc_samples[:NUM_SAMPLES]):
        item = sample["item"]
        candidates = sample["candidates"]
        gt = sample["gt_norm"]
        maj = sample["maj_answer"]

        print(f"{'='*80}")
        print(f"Sample {sample_idx + 1} / {NUM_SAMPLES} (idx={sample['index']})")
        print(f"{'='*80}")
        print(f"  Problem: {item['problem'][:150]}...")
        print(f"  GT Answer: {gt}")
        print(f"  MAJ Answer: {maj}  (correct: {maj == gt})")
        print(f"  Consistency: {sample['consistency']:.3f}")
        print(f"  Candidates:")
        for ans, freq in candidates:
            marker = " ← GT" if strip_string(ans) == gt else ""
            print(f"    {ans} ({freq:.1%}){marker}")

        # Build prompt
        prompt_content, option_map = build_verify_prompt(item["problem"], candidates)

        print(f"\n  [Calling API with thinking mode...]")
        start_time = time.time()
        thinking_text, answer_text = call_api(client, SYSTEM_PROMPT, prompt_content)
        elapsed = time.time() - start_time

        if answer_text is None:
            print(f"  API call failed!")
            continue

        print(f"  [API responded in {elapsed:.1f}s]")
        print(f"  [Thinking: {len(thinking_text)} chars, Answer: {len(answer_text)} chars]")

        # Parse from answer_text (where \boxed{} should be)
        verified = parse_verify_response(answer_text, option_map)
        verified_norm = strip_string(verified) if verified else None

        if verified_norm:
            source = "model"
        else:
            verified_norm = candidates[0][0]
            source = "fallback_top1"

        print(f"\n  --- Result ---")
        print(f"  Model selected: {verified_norm} (source: {source})")
        print(f"  Correct: {verified_norm == gt}")
        print(f"  MAJ was: {maj} (correct: {maj == gt})")

        # Show raw boxed answer
        raw = extract_answer(answer_text)
        print(f"  Raw \\boxed{{}}: {raw}")

        # Show thinking (truncated)
        if thinking_text:
            print(f"\n  --- Thinking (first 500 chars) ---")
            print(f"  {thinking_text[:500]}")
            if len(thinking_text) > 500:
                print(f"  ... ({len(thinking_text) - 500} more chars)")

        # Show answer (truncated)
        print(f"\n  --- Answer (first 500 chars) ---")
        print(f"  {answer_text[:500]}")
        if len(answer_text) > 500:
            print(f"  ... ({len(answer_text) - 500} more chars)")

        results.append({
            "index": sample["index"],
            "gt": gt,
            "maj": maj,
            "maj_correct": maj == gt,
            "verified": verified_norm,
            "verified_correct": verified_norm == gt,
            "source": source,
        })
        print()

    # Summary
    if results:
        print(f"\n{'='*80}")
        print("Summary")
        print(f"{'='*80}")
        n = len(results)
        maj_correct = sum(1 for r in results if r["maj_correct"])
        ver_correct = sum(1 for r in results if r["verified_correct"])
        print(f"  Samples tested: {n}")
        print(f"  MAJ correct:    {maj_correct}/{n}")
        print(f"  Verify correct: {ver_correct}/{n}")


if __name__ == "__main__":
    main()
