import json
import sys
import os
import collections

math_namespace = {}
math_file = os.path.join(os.path.dirname(__file__), '..', 'verl', 'utils', 'reward_score', 'math.py')
with open(math_file, 'r', encoding='utf-8') as f:
    exec(f.read(), math_namespace)
is_equiv = math_namespace['is_equiv']

def is_correct(pred, truth):
    if pred is None: return False
    return is_equiv(pred, truth)
print("Using math is_equiv for evaluation.")

def main():
    with open('qwen64.jsonl', 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    
    results = []
    
    for item in data:
        extracted = item.get('extracted_answers', [])
        # Filter out invalid answers before majority voting
        valid_extracted = [ans for ans in extracted if ans not in (None, "", "[NO_ANSWER]")]
        
        if not valid_extracted:
            continue
            
        counts = collections.Counter(valid_extracted)
        sorted_counts = counts.most_common()
        
        top1_ans, top1_count = sorted_counts[0] if len(sorted_counts) > 0 else (None, 0)
        top2_ans, top2_count = sorted_counts[1] if len(sorted_counts) > 1 else (None, 0)
        
        truth = item.get('answer', '')
        
        top1_correct = is_correct(top1_ans, truth) if top1_ans is not None else False
        top2_correct = is_correct(top2_ans, truth) if top2_ans is not None else False
        
        topk_correct_index = -1
        top5_correct = False
        for i, (ans, c) in enumerate(sorted_counts):
            if is_correct(ans, truth):
                if topk_correct_index == -1:
                    topk_correct_index = i + 1
                if i < 5:
                    top5_correct = True
                
        gap = top1_count - top2_count
        consistency = top1_count / len(valid_extracted) if len(valid_extracted) > 0 else 0
        
        results.append({
            'problem': item.get('problem', ''),
            'truth': truth,
            'top1_ans': top1_ans,
            'top1_count': top1_count,
            'top1_correct': top1_correct,
            'top2_ans': top2_ans,
            'top2_count': top2_count,
            'top2_correct': top2_correct,
            'gap': gap,
            'consistency': consistency,
            'topk_correct_index': topk_correct_index,
            'top5_correct': top5_correct,
            'total_valid': len(valid_extracted),
            'sorted_counts': sorted_counts
        })

    total = len(results)
    avg_gap = sum(r['gap'] for r in results) / total if total > 0 else 0
    top5_coverage = sum(1 for r in results if r['top5_correct'])
    
    print("=== ANALYSIS V2 (Filtered NONE) ===")
    print(f"Total problems with at least 1 valid answer: {total}")
    print(f"Average gap between Top 1 and Top 2: {avg_gap:.2f}")
    if total > 0:
        print(f"Top 1 correct rate: {sum(r['top1_correct'] for r in results) / total * 100:.2f}%")
        print(f"Top 5 Coverage (Is truth in top 5 majority?): {top5_coverage} cases ({top5_coverage / total * 100:.2f}%)")
    
    print("\n--- Gap vs Top-1 Error Rate ---")
    buckets = [0, 5, 10, 20, 30, 40, 50]
    bucket_stats = {b: {'total': 0, 'errors': 0} for b in buckets}
    def get_bucket(gap):
        for b in reversed(buckets):
            if gap >= b: return b
        return 0
        
    for r in results:
        b = get_bucket(r['gap'])
        bucket_stats[b]['total'] += 1
        if not r['top1_correct']:
            bucket_stats[b]['errors'] += 1
            
    for b in buckets:
        st = bucket_stats[b]
        if st['total'] > 0:
            print(f"Gap >= {b}: Error Rate = {st['errors']/st['total']*100:.2f}% ({st['errors']}/{st['total']})")
            
    print("\n--- Consistency vs Error Rate ---")
    cons_buckets = [
        {'name': '[0, 0.3)', 'min': 0.0, 'max': 0.3, 'total': 0, 'errors': 0},
        {'name': '[0.3, 0.5)', 'min': 0.3, 'max': 0.5, 'total': 0, 'errors': 0},
        {'name': '[0.5, 1.0]', 'min': 0.5, 'max': 1.01, 'total': 0, 'errors': 0}
    ]
    for r in results:
        for b in cons_buckets:
            if b['min'] <= r['consistency'] < b['max']:
                b['total'] += 1
                if not r['top1_correct']:
                    b['errors'] += 1
                break
                
    for b in cons_buckets:
        if b['total'] > 0:
            err_rate = b['errors'] / b['total'] * 100
            print(f"Consistency {b['name']}: Error Rate = {err_rate:.2f}% ({b['errors']}/{b['total']})")
        else:
            print(f"Consistency {b['name']}: N/A (0 cases)")
    
    print("\n--- Accuracy of Top-K Ranks under Low Consistency (< 0.3) ---")
    low_cons_results = [r for r in results if r['consistency'] < 0.3]
    total_low_cons = len(low_cons_results)
    
    rank_correct_counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    
    for r in low_cons_results:
        sc = r['sorted_counts']
        truth = r['truth']
        for i in range(min(5, len(sc))):
            ans, _ = sc[i]
            if is_correct(ans, truth):
                rank_correct_counts[i+1] += 1
                
    if total_low_cons > 0:
        for rank in range(1, 6):
            count = rank_correct_counts[rank]
            print(f"Top {rank} individually correct rate: {count/total_low_cons*100:.2f}% ({count}/{total_low_cons})")
        
        cumulative = 0
        for r in low_cons_results:
            if r['topk_correct_index'] != -1 and r['topk_correct_index'] <= 5:
                cumulative += 1
        print(f"Top 1-5 Cumulative Coverage: {cumulative/total_low_cons*100:.2f}% ({cumulative}/{total_low_cons})")

    high_consistency_errors = [r for r in results if r['top1_count'] >= 32 and not r['top1_correct']]
    high_consistency_total = sum(1 for r in results if r['top1_count'] >= 32)
    print(f"High Consistency (>=32) problems: {high_consistency_total}")
    print(f"High Consistency but Top-1 WRONG: {len(high_consistency_errors)} cases (Error rate: {len(high_consistency_errors)/high_consistency_total*100:.2f}% if total>0 else 0)")
    
    print("\nSample cases where high consistency top-1 is wrong:")
    for r in high_consistency_errors[:5]:
        print(f"Prob: {r['problem'][:150]}...")
        print(f"Truth: {r['truth']}")
        print(f"Top 1 ({r['top1_count']}): {r['top1_ans']}")
        if r['topk_correct_index'] > 1:
            ans, c = r['sorted_counts'][r['topk_correct_index']-1]
            print(f"Correct Top K: rank {r['topk_correct_index']} with count {c}, ans: {ans}")
        elif r['topk_correct_index'] == -1:
            print("No top K correct.")
        print("-")
        
    # Write details to findings.md for easier analysis
    with open('findings.md', 'w', encoding='utf-8') as f:
        f.write("# Qualitative Analysis of High Consistency Errors\n\n")
        f.write("Below are the problematic examples where Top 1 is highly consistent but evaluated as incorrect:\n\n")
        for i, r in enumerate(high_consistency_errors):
            f.write(f"## Example {i+1}\n")
            f.write(f"**Problem:** {r['problem']}\n\n")
            f.write(f"**Ground Truth:** `{r['truth']}`\n\n")
            f.write(f"**Top 1 Predict ({r['top1_count']} votes):** `{r['top1_ans']}`\n\n")
            if r['topk_correct_index'] > 1:
                ans, c = r['sorted_counts'][r['topk_correct_index']-1]
                f.write(f"**Correct Prediction in Top-K (Rank {r['topk_correct_index']}, Votes {c}):** `{ans}`\n\n")
            else:
                f.write("**No correct answer among candidates.**\n\n")

if __name__ == '__main__':
    main()
