# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
True Label TTRL Reward Manager

This manager uses ground truth labels directly for reward computation,
instead of pseudo labels from majority voting. It is designed for
supervised/oracle baseline experiments.

Reward computation:
    reward_i = 1.0 if model_answer_i matches ground_truth else 0.0

This manager also computes and passes answer_types for PASS_GRPO
advantage computation.
"""

from collections import Counter, defaultdict
from functools import partial
import random
import numpy as np
import torch

from verl import DataProto
from verl.utils.reward_score.ttrl.auto_verify import auto_verify
from verl.utils.reward_score.ttrl.latex_clean import normalize_latex
from verl.utils.reward_score.ttrl.qwen.qwen_math_parser import extract_answer


class TrueLabelTTRLRewardManager:
    """TTRL Reward Manager using ground truth labels (oracle baseline)."""

    def __init__(
        self,
        tokenizer,
        num_examine,
        reward_fn_key: str = "data_source",
        compute_score=None,
        n_votes_per_prompt: int = 1,
        n_samples_per_prompt: int = 1,
        mode: str = "eval",
        eval_n_samples: int = 1,
        noise_rate: float = 0.0,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.reward_fn_key = reward_fn_key
        self.n_votes_per_prompt = n_votes_per_prompt
        self.n_samples_per_prompt = n_samples_per_prompt
        self.mode = mode
        self.eval_n_samples = eval_n_samples
        self.noise_rate = noise_rate
        self.debug_mode = num_examine > 0

        assert n_votes_per_prompt >= n_samples_per_prompt, (
            f"TTRL requirement: n_votes_per_prompt({n_votes_per_prompt}) >= "
            f"n_samples_per_prompt({n_samples_per_prompt})"
        )

        print(
            "TrueLabelTTRLRewardManager initialized with "
            f"n_votes_per_prompt {n_votes_per_prompt}, "
            f"n_samples_per_prompt {n_samples_per_prompt}, "
            f"eval_n_samples {eval_n_samples}, "
            f"noise_rate {noise_rate}"
        )

    def _data_source_to_task(self, data_source):
        ds = str(data_source)
        if ds in ["MATH-TTT", "AIME-TTT", "AMC-TTT", "AIME25"]:
            return "math"
        if ds in ["GPQA-TTT"]:
            return "gpqa"
        if ds in ["BBEH", "bbeh", "BigBench-Extra-Hard"]:
            return "bbeh"

        dsl = ds.lower()
        if any(key in dsl for key in ["gpqa"]):
            return "gpqa"
        if any(key in dsl for key in ["aime", "math", "amc", "aime25"]):
            return "math"
        if "bbeh" in dsl or "bigbench" in dsl:
            return "bbeh"

        raise NotImplementedError(
            f"Data source {data_source} is not supported for TrueLabelTTRLRewardManager"
        )

    def _extract_final_answers(self, task: str, outputs: list[str]) -> list[str]:
        """Extract final answers from outputs for answer_type computation."""
        extract_fn = partial(extract_answer, data_name=task)
        normalized_outputs = [normalize_latex(x) for x in outputs]
        final_answers = [extract_fn(text) or "<empty>" for text in normalized_outputs]
        return final_answers

    def _decode_data_item(self, data_item):
        prompt_idx = data_item.batch["prompts"]
        prompt_length = prompt_idx.shape[-1]
        valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
        valid_prompt_idx = prompt_idx[-valid_prompt_length:]

        response_idx = data_item.batch["responses"]
        valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
        valid_response_idx = response_idx[:valid_response_length]

        prompt_str = self.tokenizer.decode(valid_prompt_idx, skip_special_tokens=False)
        response_str = self.tokenizer.decode(valid_response_idx, skip_special_tokens=False)

        return prompt_str, response_str, valid_response_length

    def _compute_strategy_entropy(self, data_items):
        """Compute normalized negative log-likelihood (strategy entropy) for a group."""
        if not data_items:
            return 0.0

        log_probs_list = []
        response_lengths = []

        for data_item in data_items:
            if not hasattr(data_item, "batch") or "old_log_probs" not in data_item.batch:
                continue

            batch = data_item.batch
            attention_mask = batch.get("attention_mask", None)
            if attention_mask is None or len(attention_mask) <= batch["prompts"].shape[-1]:
                continue

            prompt_length = batch["prompts"].shape[-1]
            response_length_val = attention_mask[prompt_length:].sum().item()
            if response_length_val <= 0:
                continue

            old_log_probs = batch["old_log_probs"]
            if not isinstance(old_log_probs, torch.Tensor) or old_log_probs.numel() == 0:
                continue

            log_probs_length = old_log_probs.shape[-1]
            if log_probs_length == response_length_val:
                response_log_probs = old_log_probs
            elif log_probs_length == prompt_length + response_length_val:
                response_log_probs = old_log_probs[prompt_length:prompt_length + response_length_val]
            elif log_probs_length > response_length_val:
                response_log_probs = old_log_probs[-response_length_val:]
            else:
                continue

            log_probs_list.append(response_log_probs.sum().item())
            response_lengths.append(response_length_val)

        if not log_probs_list:
            return 0.0

        log_probs_array = np.array(log_probs_list)
        response_lengths_array = np.array(response_lengths)
        neg_log_likelihoods = -log_probs_array / response_lengths_array

        return float(np.mean(neg_log_likelihoods))

    def compute_post_ttrl_metrics(self, data: DataProto):
        """Compute post-TTRL training evaluation metrics."""
        assert len(data) % self.n_samples_per_prompt == 0
        prompt_num = len(data) // self.n_samples_per_prompt
        print(f"Computing post-TTRL metrics, {prompt_num} prompts in total...")

        post_ttrl_info = {}
        post_ttrl_metrics_list = defaultdict(list)

        for prompt_i in range(prompt_num):
            group_vote_rewards = []
            group_pred_outputs = []
            group_labels = []
            group_extra_info = []
            task = None

            for i in range(self.n_samples_per_prompt):
                data_item = data[prompt_i * self.n_samples_per_prompt + i]
                _, response_str, _ = self._decode_data_item(data_item)
                ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
                data_source = data_item.non_tensor_batch[self.reward_fn_key]
                vote_reward = data_item.batch["acc"]
                extra_info = data_item.non_tensor_batch["extra_info"]

                if task is None:
                    task = self._data_source_to_task(data_source)

                group_labels.append(ground_truth)
                group_pred_outputs.append(response_str)
                group_vote_rewards.append(vote_reward)
                group_extra_info.append(extra_info)

            # Compute true rewards against ground truth
            true_rewards, _ = auto_verify(
                task, group_pred_outputs, [group_labels[0]] * len(group_pred_outputs),
                extra_info=group_extra_info
            )
            
            post_ttrl_metrics = {
                "post_ground_truth_ratio": sum(true_rewards) / len(true_rewards),
                f"post_pass@{len(group_pred_outputs)}": 1.0 if sum(true_rewards) > 0 else 0.0,
            }
            
            for k, v in post_ttrl_metrics.items():
                post_ttrl_metrics_list[k].append(v)

        for k, v in post_ttrl_metrics_list.items():
            if isinstance(v, list):
                v = np.mean(v)
                print(f"[{k}]", v)
                post_ttrl_info[k] = v
        return post_ttrl_info

    def _compute_ttrl_reward(self, data: DataProto, noised_prompt_indices: set = None):
        """Compute rewards using ground truth labels (not pseudo labels).
        
        Args:
            noised_prompt_indices: If provided, for these prompt indices, use a random
                wrong answer as the label instead of true ground truth (test_noise mode).
        """
        if noised_prompt_indices is None:
            noised_prompt_indices = set()
        print("Starting TRUE LABEL reward calculation...")
        if noised_prompt_indices:
            print(f"[test_noise] {len(noised_prompt_indices)} prompts will use noised (wrong) labels")

        reward_extra_info = defaultdict(list)
        ttrl_info = {}

        if len(data) % self.n_votes_per_prompt != 0:
            print(f"WARNING: Data size {len(data)} is not divisible by n_votes_per_prompt {self.n_votes_per_prompt}")
            reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
            ttrl_info["_answer_types"] = np.arange(len(data), dtype=np.int64)
            return reward_tensor, reward_extra_info, ttrl_info

        prompt_num = len(data) // self.n_votes_per_prompt
        reward_tensor = torch.zeros_like(
            data.batch["responses"][: prompt_num * self.n_samples_per_prompt], dtype=torch.float32
        )

        already_print_data_sources = {}
        all_ttrl_metrics = defaultdict(list)
        scores = [0.0 for _ in range(len(data))]
        
        # Collect answer_types and per-sample arrays for PASS_GRPO advantage computation
        all_answer_types = []
        all_consistency_rates = []
        all_accuracy_rates = []
        all_label_accuracies = []

        for prompt_i in range(prompt_num):
            group_pred_outputs = []
            group_labels = []
            group_extra_info = []
            group_resp_lengths = []
            group_prompts = []
            task = None

            for i in range(self.n_votes_per_prompt):
                data_item = data[prompt_i * self.n_votes_per_prompt + i]
                prompt_str, response_str, valid_response_length = self._decode_data_item(data_item)
                ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
                data_source = data_item.non_tensor_batch[self.reward_fn_key]
                extra_info = data_item.non_tensor_batch["extra_info"]

                if task is None:
                    task = self._data_source_to_task(data_source)

                group_labels.append(ground_truth)
                group_pred_outputs.append(response_str)
                group_extra_info.append(extra_info)
                group_resp_lengths.append(int(valid_response_length))
                group_prompts.append(prompt_str)

            # ========== KEY CHANGE: Use ground truth for rewards ==========
            # Compute rewards by comparing each output to the ground truth
            ground_truth = group_labels[0]  # All labels should be the same for one prompt
            
            # Verify all labels in this group match (data may be shuffled)
            if len(set(group_labels)) != 1:
                print(f"WARNING: Ground truth not unique in prompt group {prompt_i}, using first label")
            
            # For test_noise mode: replace ground truth with a random wrong answer
            effective_label = ground_truth
            is_noised = prompt_i in noised_prompt_indices
            if is_noised:
                # Extract final answers to pick a wrong one
                noise_answers = self._extract_final_answers(task, group_pred_outputs)
                wrong_answers = [a for a in set(noise_answers) if a != ground_truth and a is not None and a != ""]
                if wrong_answers:
                    effective_label = random.choice(wrong_answers)
                else:
                    # All answers match ground truth or are empty; generate a clearly wrong label
                    effective_label = "NOISE_WRONG_ANSWER_" + str(random.randint(0, 99999))
                print(f"[test_noise] Prompt {prompt_i}: true='{ground_truth}' -> noised='{effective_label}'")
            
            true_rewards, _ = auto_verify(
                task, group_pred_outputs, [effective_label] * len(group_pred_outputs),
                extra_info=group_extra_info
            )
            
            # Compute strategy entropy for this group
            current_group_data = data[prompt_i * self.n_votes_per_prompt : (prompt_i + 1) * self.n_votes_per_prompt]
            strategy_entropy = self._compute_strategy_entropy(current_group_data)
            if self.debug_mode and strategy_entropy > 0:
                print(f"    Strategy entropy: H_ttrl={strategy_entropy:.3f} (normalized negative log-likelihood)")

            # Compute metrics directly (no need for test_time_train_metrics which uses majority voting)
            ground_truth_ratio = sum(true_rewards) / len(true_rewards)

            # Compute answer types for PASS_GRPO
            # answer_type = 0 means correct (matches ground truth), else unique hash
            final_answers = self._extract_final_answers(task, group_pred_outputs)
            for i in range(self.n_votes_per_prompt):
                if true_rewards[i] == 1.0:
                    all_answer_types.append(0)  # Correct answer type = 0
                else:
                    all_answer_types.append(hash(final_answers[i]))  # Unique type for incorrect

            # Compute majority vote accuracy (label_accuracy)
            counter = Counter(final_answers)
            majority_answer, majority_count = counter.most_common(1)[0]
            majority_ratio = majority_count / len(final_answers)
            label_accuracy = 1.0 if auto_verify(
                task, [majority_answer], [ground_truth], extra_info=[group_extra_info[0]]
            )[0][0] else 0.0

            # Compute diversity_ratio: unique answers / total answers
            unique_answers = len(set(final_answers))
            diversity_ratio = unique_answers / len(final_answers) if len(final_answers) > 0 else 0.0

            # Compute majority_rewards: which samples match majority vote (pseudo-label perspective)
            majority_rewards, _ = auto_verify(
                task, group_pred_outputs, [majority_answer] * len(group_pred_outputs),
                extra_info=group_extra_info
            )

            # false_positive_rate: fraction of majority-matching samples that are actually wrong
            n_pseudo_pos = sum(1 for m in majority_rewards if m > 0)
            n_false_pos = sum(1 for m, t in zip(majority_rewards, true_rewards) if m > 0 and t == 0)
            fp_rate = n_false_pos / n_pseudo_pos if n_pseudo_pos > 0 else 0.0

            # false_negative_rate: fraction of majority-non-matching samples that are actually correct
            n_pseudo_neg = sum(1 for m in majority_rewards if m == 0)
            n_false_neg = sum(1 for m, t in zip(majority_rewards, true_rewards) if m == 0 and t > 0)
            fn_rate = n_false_neg / n_pseudo_neg if n_pseudo_neg > 0 else 0.0

            # Store per-sample arrays for downstream advantage computation
            for i in range(self.n_votes_per_prompt):
                all_consistency_rates.append(majority_ratio)
                all_accuracy_rates.append(ground_truth_ratio)
                all_label_accuracies.append(label_accuracy)

            ttrl_metrics = {
                "ground_truth_ratio": ground_truth_ratio,
                f"pass@{len(group_pred_outputs)}": 1.0 if sum(true_rewards) >= 1 else 0.0,
                "label_accuracy": label_accuracy,
                "majority_ratio": majority_ratio,
                "diversity_ratio": diversity_ratio,
                "false_positive_rate": fp_rate,
                "false_negative_rate": fn_rate,
                "neg_log_likelihood": strategy_entropy,
            }

            for k, v in ttrl_metrics.items():
                all_ttrl_metrics[k].append(v)

            # Assign rewards (using true rewards, not pseudo label rewards)
            for i in range(self.n_votes_per_prompt):
                current_reward = float(true_rewards[i])  # 1.0 or 0.0
                vlen = group_resp_lengths[i]

                if i < self.n_samples_per_prompt and vlen > 0:
                    reward_tensor[prompt_i * self.n_samples_per_prompt + i, vlen - 1] = current_reward

                scores[prompt_i * self.n_votes_per_prompt + i] = current_reward

                data_item = data[prompt_i * self.n_votes_per_prompt + i]
                data_source = data_item.non_tensor_batch[self.reward_fn_key]
                if data_source not in already_print_data_sources:
                    already_print_data_sources[data_source] = 0
                if already_print_data_sources[data_source] < self.num_examine:
                    already_print_data_sources[data_source] += 1
                    print("\n    === Sample Debug Output ===")
                    print("    [prompt]", group_prompts[i])
                    print("    [response]", group_pred_outputs[i])
                    print(f"    [true_reward] {current_reward:.1f}")
                    print(f"    [ground_truth] {ground_truth}")

        data.batch["acc"] = torch.tensor(scores, dtype=torch.float32, device=data.batch["prompts"].device)
        
        # Store per-sample arrays for PASS_GRPO and diagnostics (only for training samples)
        training_answer_types = []
        training_consistency_rates = []
        training_accuracy_rates = []
        training_label_accuracies = []
        for prompt_i in range(prompt_num):
            for i in range(self.n_samples_per_prompt):
                global_idx = prompt_i * self.n_votes_per_prompt + i
                training_answer_types.append(all_answer_types[global_idx])
                training_consistency_rates.append(all_consistency_rates[global_idx])
                training_accuracy_rates.append(all_accuracy_rates[global_idx])
                training_label_accuracies.append(all_label_accuracies[global_idx])
        
        ttrl_info["_answer_types"] = np.array(training_answer_types)
        ttrl_info["_consistency_rate"] = np.array(training_consistency_rates)
        ttrl_info["_accuracy_rate"] = np.array(training_accuracy_rates)
        ttrl_info["_label_accuracy"] = np.array(training_label_accuracies)
        
        # Store per-prompt label_accuracy for test_minority mode
        ttrl_info["_per_prompt_label_accuracy"] = all_ttrl_metrics.get("label_accuracy", [])

        print("\n=== True Label TTRL Training Metrics Summary ===")
        for k, v in all_ttrl_metrics.items():
            if isinstance(v, list):
                avg_v = np.mean(v)
                print(f"[{k}] {avg_v:.4f}")
                ttrl_info[k] = avg_v

        return reward_tensor, reward_extra_info, ttrl_info

    def _compute_eval_reward(self, data: DataProto):
        """Compute evaluation rewards using ground truth labels."""
        print("Starting TRUE LABEL evaluation reward calculation...")

        reward_extra_info = defaultdict(list)
        ttrl_info = {}
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        already_print_data_sources = {}

        assert len(data) % self.eval_n_samples == 0
        prompt_num = len(data) // self.eval_n_samples

        # Group by task
        task_groups = {}
        sample_valid_resp_len = {}

        for i in range(len(data)):
            data_item = data[i]
            prompt_str, response_str, valid_response_length = self._decode_data_item(data_item)
            sample_valid_resp_len[i] = int(valid_response_length)

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            extra_info = data_item.non_tensor_batch["extra_info"]

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0
            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)

            task_key = self._data_source_to_task(data_source)
            if task_key not in task_groups:
                task_groups[task_key] = {"indices": [], "outputs": [], "labels": [], "extra": []}
            task_groups[task_key]["indices"].append(i)
            task_groups[task_key]["outputs"].append(response_str)
            task_groups[task_key]["labels"].append(ground_truth)
            task_groups[task_key]["extra"].append(extra_info)

        # Compute rewards per task group using ground truth
        for task_key, group in task_groups.items():
            rewards, verify_extra_info = auto_verify(
                task_key, group["outputs"], group["labels"], extra_info=group["extra"]
            )
            for k, v in verify_extra_info.items():
                if isinstance(v, list):
                    reward_extra_info[k] += v
            for idx_in_group, sample_idx in enumerate(group["indices"]):
                vlen = sample_valid_resp_len[sample_idx]
                if vlen > 0:
                    reward_tensor[sample_idx, vlen - 1] = float(rewards[idx_in_group])

        # Compute TTRL metrics for logging
        all_ttrl_metrics = defaultdict(list)
        for prompt_i in range(prompt_num):
            group_pred_outputs = []
            group_labels = []
            group_extra_info = []
            task = None

            for i in range(self.eval_n_samples):
                idx = prompt_i * self.eval_n_samples + i
                data_item = data[idx]
                _, response_str, _ = self._decode_data_item(data_item)
                ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
                data_source = data_item.non_tensor_batch[self.reward_fn_key]
                extra_info = data_item.non_tensor_batch["extra_info"]

                if task is None:
                    task = self._data_source_to_task(data_source)

                group_labels.append(ground_truth)
                group_pred_outputs.append(response_str)
                group_extra_info.append(extra_info)

            # Compute true rewards for metrics
            ground_truth = group_labels[0]
            true_rewards, _ = auto_verify(
                task, group_pred_outputs, [ground_truth] * len(group_pred_outputs),
                extra_info=group_extra_info
            )
            
            # Compute strategy entropy for eval group
            current_group_data = data[prompt_i * self.eval_n_samples : (prompt_i + 1) * self.eval_n_samples]
            strategy_entropy = self._compute_strategy_entropy(current_group_data)

            ttrl_metrics = {
                "ground_truth_ratio": sum(true_rewards) / len(true_rewards),
                f"pass@{len(group_pred_outputs)}": 1.0 if sum(true_rewards) >= 1 else 0.0,
                "neg_log_likelihood": strategy_entropy,
            }

            for k, v in ttrl_metrics.items():
                all_ttrl_metrics[k].append(v)

        for k, v in all_ttrl_metrics.items():
            if isinstance(v, list):
                avg_v = np.mean(v)
                print(f"[{k}] {avg_v:.4f}")
                ttrl_info[k] = avg_v

        return reward_tensor, reward_extra_info, ttrl_info

    def __call__(self, data: DataProto, return_dict: bool = False):
        print("\n" + "=" * 50)
        print("TrueLabelTTRLRewardManager execution started")
        print(f"Mode: {self.mode}")
        print(f"Data size: {len(data)}")
        print("=" * 50)

        if self.mode in ("train", "test_minority"):
            reward_tensor, reward_extra_info, ttrl_info = self._compute_ttrl_reward(data)
            if self.mode == "test_minority":
                # Build per-sample zero_advantage_mask:
                # For each prompt, if true label != majority answer (label_accuracy == 0),
                # set mask=1 for all responses of that prompt (advantage will be zeroed).
                prompt_num = len(data) // self.n_votes_per_prompt
                zero_mask = np.zeros(prompt_num * self.n_samples_per_prompt, dtype=np.float32)
                # Retrieve per-prompt label_accuracy from ttrl_info
                per_prompt_label_acc = ttrl_info.get("_per_prompt_label_accuracy", [])
                for prompt_i in range(prompt_num):
                    if prompt_i < len(per_prompt_label_acc) and per_prompt_label_acc[prompt_i] == 0.0:
                        for j in range(self.n_samples_per_prompt):
                            zero_mask[prompt_i * self.n_samples_per_prompt + j] = 1.0
                ttrl_info["_zero_advantage_mask"] = zero_mask
                n_zeroed_prompts = int(sum(1 for acc in per_prompt_label_acc if acc == 0.0))
                n_zeroed_samples = int(zero_mask.sum())
                print(f"[test_minority] Zeroing advantage for {n_zeroed_prompts}/{prompt_num} prompts "
                      f"({n_zeroed_samples}/{len(zero_mask)} samples) where true label != majority")
        elif self.mode == "test_noise":
            # Randomly select noise_rate fraction of prompts and use wrong labels
            prompt_num = len(data) // self.n_votes_per_prompt
            n_noised = max(1, int(prompt_num * self.noise_rate))
            noised_indices = set(random.sample(range(prompt_num), min(n_noised, prompt_num)))
            print(f"[test_noise] Noising {len(noised_indices)}/{prompt_num} prompts "
                  f"(noise_rate={self.noise_rate:.2f})")
            reward_tensor, reward_extra_info, ttrl_info = self._compute_ttrl_reward(
                data, noised_prompt_indices=noised_indices
            )
            ttrl_info["noise_rate_actual"] = len(noised_indices) / prompt_num
        elif self.mode == "eval":
            reward_tensor, reward_extra_info, ttrl_info = self._compute_eval_reward(data)
        else:
            raise NotImplementedError(f"Mode {self.mode} is not supported")

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
                "ttrl_info": ttrl_info,
            }
        return reward_tensor
