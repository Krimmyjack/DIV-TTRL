import os
import httpx
import json
import re
import time
import concurrent.futures
from collections import Counter
from openai import OpenAI
from tqdm import tqdm

# =====================================================
# Config
# =====================================================

API_KEY = "EMPTY"
BASE_URL = "https://u630113-8ba4-8da84932.westc.gpuhub.com:8443/v1"
MODEL = "qwen3-4b-base"
INPUT_FILE = r"D:\学习\科研\DIV-TTRL-PR\verl\scripts\qwen64.jsonl"
NUM_SAMPLES = None   # Set to None to test all low-consistency samples

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
    """Call AutoDL VLLM Base Model API."""
    try:
        # Combine system prompt and user problem into a single base model prompt
        prompt = f"{system_prompt}\n\n{user_content}\n\nReasoning:\n"
        
        response = client.completions.create(
            model=MODEL,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=0.0,
            stop=["[Test Time Problem]", "Problem:"], 
        )
        answer_text = response.choices[0].text
        
        # Base model usually doesn't separate thinking and answer blocks clearly in its API.
        # We will treat the whole output as answer_text for our parser to parse \boxed{}.
        return "", answer_text
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
    os.environ.pop("http_proxy", None)
    os.environ.pop("https_proxy", None)
    os.environ.pop("HTTP_PROXY", None)
    os.environ.pop("HTTPS_PROXY", None)

    custom_http_client = httpx.Client(verify=False)
    client = OpenAI(
        api_key=API_KEY, 
        base_url=BASE_URL,
        http_client=custom_http_client
    )

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
    
    samples_to_test = low_sc_samples if NUM_SAMPLES is None else low_sc_samples[:NUM_SAMPLES]
    print(f"Testing on {len(samples_to_test)} samples\n")
    print(f"Starting concurrent API verification with ThreadPoolExecutor...")

    def process_sample(sample):
        sample_idx = sample["index"]
        item = sample["item"]
        candidates = sample["candidates"]
        gt = sample["gt_norm"]
        maj = sample["maj_answer"]

        # Build prompt
        prompt_content, option_map = build_verify_prompt(item["problem"], candidates)

        thinking_text, answer_text = call_api(client, SYSTEM_PROMPT, prompt_content)

        if answer_text is None:
            return None

        # Parse from answer_text
        verified = parse_verify_response(answer_text, option_map)
        verified_norm = strip_string(verified) if verified else None

        if verified_norm:
            source = "model"
        else:
            verified_norm = candidates[0][0]
            source = "fallback_top1"

        return {
            "index": sample_idx,
            "gt": gt,
            "maj": maj,
            "maj_correct": maj == gt,
            "verified": verified_norm,
            "verified_correct": verified_norm == gt,
            "source": source,
        }

    results = []
    failed_count = 0
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
        futures = {executor.submit(process_sample, sample): sample for sample in samples_to_test}
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Verifying"):
            res = future.result()
            if res is not None:
                results.append(res)
            else:
                failed_count += 1

    # Summary
    if results:
        print(f"\n{'='*80}")
        print("Summary")
        print(f"{'='*80}")
        n = len(results)
        maj_correct = sum(1 for r in results if r["maj_correct"])
        ver_correct = sum(1 for r in results if r["verified_correct"])
        print(f"  Total samples selected (sc<=0.3): {len(samples_to_test)}")
        print(f"  Successfully processed:           {n}")
        print(f"  Failed API calls:                 {failed_count}")
        print(f"--------------------------------------------------")
        print(f"  BEFORE (Majority Vote) Accuracy:  {maj_correct}/{n} ({maj_correct/n:.2%})")
        print(f"  AFTER  (API Verify)    Accuracy:  {ver_correct}/{n} ({ver_correct/n:.2%})")
        
        # Calculate net improvement
        improvement = ver_correct - maj_correct
        print(f"  Net Change:                       {improvement:+d} correct answers ({improvement/n:+.2%})")


if __name__ == "__main__":
    main()
