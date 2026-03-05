import re
import os
import concurrent.futures
from collections import Counter
from openai import OpenAI
from verl.utils.reward_score.ttrl.auto_extract import auto_extract


# =====================================================
# Config
# =====================================================

# You can override these via environment variables if needed
API_KEY = os.environ.get("MODELSCOPE_API_KEY", "ms-06ae5daf-b4e6-4451-83ad-c6a272397f65")
BASE_URL = os.environ.get("MODELSCOPE_BASE_URL", "https://api-inference.modelscope.cn/v1")
MODEL = os.environ.get("MODELSCOPE_MODEL", "Qwen/Qwen3-4B")

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
# Helpers
# =====================================================

def strip_string(s):
    """Basic normalization for math answers."""
    if s is None:
        return ""
    s = str(s).strip()
    m = re.search(r'\\boxed\{(.+)\}', s)
    if m:
        s = m.group(1)
    s = s.strip('$').strip()
    if s.endswith('.'):
        s = s[:-1]
    return s

def extract_answer(text):
    """Extract answer from \\boxed{...}."""
    if not text:
        return None
    matches = re.findall(r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}', text)
    if matches:
        return matches[-1].strip()
    return None

def parse_verify_response(response_text, option_map):
    """Parse model output to extract selected answer."""
    raw = extract_answer(response_text)
    if raw is None:
        return None
    raw_stripped = raw.strip()

    # Clean LaTeX artifacts
    cleaned = raw_stripped
    cleaned = re.sub(r'\\text\{([^}]*)\}', r'\1', cleaned)
    cleaned = re.sub(r'\\mathrm\{([^}]*)\}', r'\1', cleaned)
    cleaned = cleaned.replace('\\ ', ' ').replace('\\,', ' ')
    cleaned = cleaned.strip()

    if "no reliable" in cleaned.lower() or cleaned.lower() == "none":
        return None

    if cleaned in option_map:
        return option_map[cleaned]

    m = re.match(r'[Cc]andidate\s*(\d+)', cleaned)
    if m:
        label = f"Candidate {m.group(1)}"
        if label in option_map:
            return option_map[label]

    norm = strip_string(cleaned)
    for _label, ans in option_map.items():
        if strip_string(ans) == norm:
            return ans
    return None


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

def build_verify_prompt(problem, candidates):
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
# API Call
# =====================================================

def call_api(system_prompt, user_content, max_tokens=4096):
    """Call ModelScope API with thinking mode."""
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
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
    thinking_text = ""
    answer_text = ""
    for chunk in response:
        if chunk.choices:
            delta = chunk.choices[0].delta
            thinking_chunk = getattr(delta, 'reasoning_content', '') or ''
            if thinking_chunk:
                thinking_text += thinking_chunk
            answer_chunk = delta.content or ''
            if answer_chunk:
                answer_text += answer_chunk
    return thinking_text, answer_text

def verify_single_problem(problem_text, candidate_answers, top_k=5, max_retries=3):
    candidates, _ = get_frequency_candidates(candidate_answers, top_k=top_k)
    # Filter valid candidates up to top_k, at least 2
    if len(candidates) < 2:
        return None, 0, False # verified_answer, retries_used, is_fallback
        
    prompt_content, option_map = build_verify_prompt(problem_text, candidates)
    
    import openai
    import time
    import colorama
    colorama.init(autoreset=True)
    
    retries = 0
    verified = None
    
    while retries < max_retries:
        try:
            _, answer_text = call_api(SYSTEM_PROMPT, prompt_content)
            if answer_text is not None:
                verified = parse_verify_response(answer_text, option_map)
                if verified is not None:
                    break
        except openai.RateLimitError as e:
            print(colorama.Fore.RED + colorama.Style.BRIGHT + f"\n[CRITICAL WARNING] API 限额已满/超出请求速率！Rate Limit Exceeded: {e}\n")
            time.sleep(2)  # basic backoff
        except Exception as e:
            print(f"  API Error during verify: {e}")
            
        retries += 1
        
    if verified is None:
        # Fallback to majority top 1
        fallback_ans = candidates[0][0]
        return strip_string(fallback_ans), retries, True
    else:
        return strip_string(verified), retries, False

def auto_self_verify_batch(tasks, problem_texts, solutions_batch, extra_infos, top_k=5, sc_threshold=0.3, max_workers=8):
    """
    Run self-verification for a batch of problems.
    Skips items where consistency > sc_threshold or candidates < 2.
    Returns:
        List of (verified_answer, consistency). If verification fails or is skipped, verified_answer is None.
    """
    assert len(problem_texts) == len(solutions_batch)
    
    # Extract answers first using rule-based extraction
    all_extracted_answers = []
    for task, solutions, extra_info in zip(tasks, solutions_batch, extra_infos):
        answers = auto_extract(task, solutions, extra_info=extra_info)
        all_extracted_answers.append(answers)

    results = [{"ans": None, "consistency": 1.0, "retries": 0, "is_fallback": False} for _ in range(len(problem_texts))]
    
    # Check consistency and prepare indices to verify
    verify_indices = []
    for i in range(len(problem_texts)):
        candidates, consistency = get_frequency_candidates(all_extracted_answers[i], top_k=top_k)
        if consistency <= sc_threshold and len(candidates) >= 2:
            verify_indices.append(i)
        results[i]["consistency"] = consistency
    
    # We use ThreadPoolExecutor to make concurrent API calls
    if verify_indices:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(verify_single_problem, problem_texts[i], all_extracted_answers[i], top_k): i 
                for i in verify_indices
            }
            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    ans, retries, is_fallback = future.result()
                    results[idx]["ans"] = ans
                    results[idx]["retries"] = retries
                    results[idx]["is_fallback"] = is_fallback
                except Exception as e:
                    print(f"Error during self-verification for index {idx}: {e}")
                
    return results
