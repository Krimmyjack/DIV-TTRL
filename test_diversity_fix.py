#!/usr/bin/env python3
"""
简单测试脚本，验证 DiversityTTRLRewardManager 的修复
"""

import torch
import sys
sys.path.insert(0, '/root/autodl-tmp/EVOL-RL')

from verl import DataProto

# 模拟数据
def create_mock_data():
    """创建模拟数据，包含不同的 ground truth"""
    batch_size = 64  # n_votes_per_prompt=64
    prompt_len = 512
    response_len = 1024
    
    # 创建批次数据
    prompts = torch.randint(0, 1000, (batch_size, prompt_len))
    responses = torch.randint(0, 1000, (batch_size, response_len))
    attention_mask = torch.ones(batch_size, prompt_len + response_len)
    old_log_probs = torch.randn(batch_size, response_len) * 0.1
    
    batch = {
        "prompts": prompts,
        "responses": responses,
        "attention_mask": attention_mask,
        "old_log_probs": old_log_probs,
    }
    
    # 创建非张量批次，模拟多个不同的ground truth
    non_tensor_batch = []
    answers = ["16", "2", "4", "7\\pi", "1+274i"]
    
    for i in range(batch_size):
        non_tensor_batch.append({
            "reward_model": {
                "ground_truth": answers[i % len(answers)]  # 不同的 ground truth
            },
            "data_source": "MATH-TTT",
            "extra_info": {},
        })
    
    data = DataProto(batch=batch, non_tensor_batch=non_tensor_batch)
    return data

def test_diversity_reward_manager():
    """测试修复后的 DiversityTTRLRewardManager"""
    from transformers import AutoTokenizer
    from verl.workers.reward_manager import DiversityTTRLRewardManager
    
    print("=" * 60)
    print("测试 DiversityTTRLRewardManager 修复")
    print("=" * 60)
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-1.5B")
    
    # 创建reward manager
    reward_manager = DiversityTTRLRewardManager(
        tokenizer=tokenizer,
        num_examine=1,
        n_votes_per_prompt=64,
        n_samples_per_prompt=32,
        mode="train",
    )
    
    # 创建测试数据
    data = create_mock_data()
    
    print(f"\n数据信息:")
    print(f"  总样本数: {len(data)}")
    print(f"  n_votes_per_prompt: 64")
    print(f"  n_samples_per_prompt: 32")
    
    # 运行reward manager
    try:
        print(f"\n运行 reward manager...")
        result = reward_manager(data, return_dict=True)
        
        print(f"\n✓ 成功!")
        print(f"  奖励张量形状: {result['reward_tensor'].shape}")
        print(f"  奖励范围: [{result['reward_tensor'].min():.4f}, {result['reward_tensor'].max():.4f}]")
        print(f"  TTRL指标: {list(result['ttrl_info'].keys())}")
        
        return True
    except AssertionError as e:
        if "Ground truth is not unique" in str(e):
            print(f"\n✗ 失败: 仍然存在 ground truth 不唯一的问题")
            print(f"  错误: {e}")
            return False
        else:
            raise
    except Exception as e:
        print(f"\n✗ 失败: 发生未预期的错误")
        print(f"  错误: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_diversity_reward_manager()
    sys.exit(0 if success else 1)
