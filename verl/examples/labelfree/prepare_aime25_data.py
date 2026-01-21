"""
Download and prepare AIME25 training data.

Usage:
    python prepare_aime25_data.py --local_dir ./data/AIME25-NEW

This script downloads:
    - AI-MO/aimo-validation-aime as training set
    - math-ai/aime25 as test set

Then converts them to the JSON format required by preprocess_simplerl.py
"""

import argparse
import json
import os
import re
from datasets import load_dataset


def extract_boxed_answer(solution: str) -> str:
    """Extract answer from \\boxed{...} in solution string."""
    # Handle nested braces
    pattern = r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}'
    matches = re.findall(pattern, solution)
    if matches:
        return matches[-1].strip()  # Return the last boxed answer
    
    # Fallback: try simpler pattern
    simple_pattern = r'\\boxed\{([^}]+)\}'
    simple_matches = re.findall(simple_pattern, solution)
    if simple_matches:
        return simple_matches[-1].strip()
    
    return ""


def build_aimo_validation_aime_dataset():
    """Load AI-MO/aimo-validation-aime dataset for training."""
    data_source = "AI-MO/aimo-validation-aime"
    print(f"Loading {data_source} from HuggingFace...")
    
    dataset = load_dataset(data_source, split="train")
    
    result = []
    for idx, example in enumerate(dataset):
        problem = example["problem"]
        solution = example["solution"]
        answer = extract_boxed_answer(solution)
        
        if not answer:
            print(f"Warning: Could not extract answer from example {idx}, using empty string")
        
        result.append({
            "prompt": problem,
            "answer": answer,
            "id": str(idx),
            "source": "aimo-validation-aime"
        })
    
    print(f"Loaded {len(result)} examples from {data_source}")
    return result


def build_aime25_dataset():
    """Load math-ai/aime25 dataset for testing."""
    data_source = "math-ai/aime25"
    print(f"Loading {data_source} from HuggingFace...")
    
    dataset = load_dataset(data_source, split="train")
    
    result = []
    for idx, example in enumerate(dataset):
        problem = example["problem"]
        answer = str(example["answer"])
        
        result.append({
            "prompt": problem,
            "answer": answer,
            "id": example.get("id", str(idx)),
            "source": "aime25"
        })
    
    print(f"Loaded {len(result)} examples from {data_source}")
    return result


def main():
    parser = argparse.ArgumentParser(description="Prepare AIME25 training data")
    parser.add_argument(
        "--local_dir", 
        default="./data/AIME25-TTT",
        help="Output directory for JSON files"
    )
    parser.add_argument(
        "--train_source",
        default="aimo-validation-aime",
        choices=["aimo-validation-aime"],
        help="Training data source"
    )
    parser.add_argument(
        "--test_source", 
        default="aime25",
        choices=["aime25"],
        help="Test data source"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.local_dir, exist_ok=True)
    
    # Download and process datasets
    print("=" * 50)
    print("Preparing AIME25 training data...")
    print("=" * 50)
    
    # Training set: AI-MO/aimo-validation-aime
    train_data = build_aimo_validation_aime_dataset()
    train_file = os.path.join(args.local_dir, "train.json")
    with open(train_file, "w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    print(f"Saved training data to {train_file}")
    
    # Test set: math-ai/aime25
    test_data = build_aime25_dataset()
    test_file = os.path.join(args.local_dir, "test.json")
    with open(test_file, "w", encoding="utf-8") as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)
    print(f"Saved test data to {test_file}")
    
    print("=" * 50)
    print("Data preparation complete!")
    print(f"Training set: {len(train_data)} examples")
    print(f"Test set: {len(test_data)} examples")
    print("=" * 50)
    print()
    print("Next step: Run preprocess_simplerl.py to convert to parquet format:")
    print(f"  cd {os.path.dirname(args.local_dir)}")
    print("  python preprocess_simplerl.py")


if __name__ == "__main__":
    main()
