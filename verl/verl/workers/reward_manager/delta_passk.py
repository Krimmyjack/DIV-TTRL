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
Delta Pass@k Reward Manager: Label-Free Expected Marginal Contribution

This reward manager computes rewards based on the expected marginal contribution
of each sample to the Pass@k probability, without requiring ground-truth labels.

Core formula:
    r(y_i) = E[ΔPass@k] = P(correct) * Δ(c)
           ≈ (c/N) * [C(N-c, k-1) / C(N, k)]

where c is the cluster size (how many rollouts produced the same answer).

The resulting reward has an inverted-U shape:
    - c=1 (hallucination): near-zero reward (filters noise)
    - c≈N (majority lock): near-zero reward (penalizes over-exploitation)
    - c moderate: peak reward (encourages diverse exploration)
"""

from collections import Counter, defaultdict

import numpy as np
import torch

from verl import DataProto
from verl.trainer.ppo.core_algos import compute_expected_marginal_passk_rewards
from verl.utils.reward_score.ttrl.auto_extract import auto_extract
from verl.utils.reward_score.ttrl.auto_verify import auto_verify
from verl.utils.reward_score.ttrl.ttt_metrics import (
    post_test_time_train_metrics,
)


class DeltaPasskRewardManager:
    """
    Label-free reward manager based on expected marginal Pass@k contribution.
    
    Instead of majority voting to assign binary rewards, this manager computes
    a continuous reward based on the "bandpass" formula g(c) = (c/N) * Δ(c),
    which naturally:
        1. Filters out hallucinations (c=1 → reward ≈ 0)
        2. Prevents majority lock-in (large c → reward ≈ 0)
        3. Maximally rewards "promising minority" answers
    """

    def __init__(
        self,
        tokenizer,
        num_examine,
        reward_fn_key="data_source",
        compute_score=None,
        n_samples_per_prompt=16,
        n_votes_per_prompt=None,
        k=4,
        mode="train",
        eval_n_samples=1,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.reward_fn_key = reward_fn_key
        self.n_samples_per_prompt = n_samples_per_prompt
        # For Delta Pass@k, n_votes = n_samples (no separate voting pool)
        self.n_votes_per_prompt = n_votes_per_prompt or n_samples_per_prompt
        self.k = k
        self.mode = mode
        self.eval_n_samples = eval_n_samples

        print(
            f"DeltaPasskRewardManager initialized: "
            f"n_samples_per_prompt={n_samples_per_prompt}, "
            f"k={k}, mode={mode}, eval_n_samples={eval_n_samples}"
        )

    def _data_source_to_task(self, data_source):
        """Standardize data source to task name."""
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
            f"Data source {data_source} is not supported for DeltaPasskRewardManager"
        )

    def _cluster_responses(self, extracted_answers, task, extra_info=None):
        """
        Cluster extracted answers by equivalence and return cluster sizes.
        
        Uses O(N²) pairwise comparison via auto_verify to handle mathematical
        equivalences (e.g., "1/2" == "0.5").
        
        Args:
            extracted_answers: List of extracted answer strings
            task: Task type for equivalence comparison
            extra_info: Optional extra info for verification
            
        Returns:
            List[int]: cluster size c_i for each sample i
        """
        N = len(extracted_answers)
        if N == 0:
            return []

        # Use Counter for initial grouping by exact string match (fast path)
        answer_counter = Counter(extracted_answers)
        
        # For math tasks, we need mathematical equivalence checking
        # Group answers that are mathematically equivalent
        if task == "math":
            # Build equivalence classes using pairwise comparison
            unique_answers = list(answer_counter.keys())
            # Map from each unique answer to its canonical representative
            canonical = {}
            equivalence_groups = []
            
            for ans in unique_answers:
                found = False
                for group_idx, group in enumerate(equivalence_groups):
                    representative = group[0]
                    # Check if ans is equivalent to the representative
                    rewards, _ = auto_verify(
                        task, [ans], [representative],
                        extra_info=extra_info, num_workers=0
                    )
                    if rewards[0] > 0:
                        group.append(ans)
                        canonical[ans] = group_idx
                        found = True
                        break
                if not found:
                    canonical[ans] = len(equivalence_groups)
                    equivalence_groups.append([ans])
            
            # Compute cluster sizes based on equivalence groups
            group_sizes = {}
            for group_idx, group in enumerate(equivalence_groups):
                total = sum(answer_counter[ans] for ans in group)
                group_sizes[group_idx] = total
            
            # Assign cluster sizes to each sample
            cluster_sizes = []
            for ans in extracted_answers:
                group_idx = canonical[ans]
                cluster_sizes.append(group_sizes[group_idx])
        else:
            # For non-math tasks, use exact string matching
            cluster_sizes = [answer_counter[ans] for ans in extracted_answers]
        
        return cluster_sizes

    def _compute_delta_passk_reward(self, data: DataProto):
        """
        Compute Delta Pass@k rewards for training.
        
        For each prompt group of N samples:
        1. Decode responses and extract answers
        2. Cluster answers by equivalence
        3. Compute g(c) = (c/N) * Δ(c) reward for each sample
        4. Fill reward_tensor at the last valid token position
        
        Returns:
            Tuple of (reward_tensor, reward_extra_info, delta_passk_info)
        """
        reward_extra_info = defaultdict(list)
        delta_passk_info = {}

        N = self.n_samples_per_prompt
        assert len(data) % N == 0, (
            f"Length of data {len(data)} should be divisible by "
            f"n_samples_per_prompt {N}"
        )
        prompt_num = len(data) // N

        reward_tensor = torch.zeros_like(
            data.batch["responses"], dtype=torch.float32
        )

        already_print_data_sources = {}
        all_metrics = defaultdict(list)
        scores = [0.0 for _ in range(len(data))]

        for prompt_i in range(prompt_num):
            group_responses = []
            group_extra_info = []
            group_valid_response_lengths = []
            task = None

            # Step 1: Decode all responses in this prompt group
            for i in range(N):
                idx = prompt_i * N + i
                data_item = data[idx]

                prompt_ids = data_item.batch["prompts"]
                prompt_length = prompt_ids.shape[-1]
                valid_prompt_length = data_item.batch["attention_mask"][
                    :prompt_length
                ].sum()
                valid_prompt_ids = prompt_ids[-valid_prompt_length:]

                response_ids = data_item.batch["responses"]
                valid_response_length = data_item.batch["attention_mask"][
                    prompt_length:
                ].sum()
                valid_response_ids = response_ids[:valid_response_length]

                response_str = self.tokenizer.decode(
                    valid_response_ids, skip_special_tokens=False
                )
                data_source = data_item.non_tensor_batch[self.reward_fn_key]
                extra_info = data_item.non_tensor_batch.get("extra_info", None)

                if task is None:
                    task = self._data_source_to_task(data_source)
                else:
                    assert task == self._data_source_to_task(data_source), (
                        f"Inconsistent task within prompt group: "
                        f"{task} vs {self._data_source_to_task(data_source)}"
                    )

                group_responses.append(response_str)
                group_extra_info.append(extra_info)
                group_valid_response_lengths.append(int(valid_response_length))

            # Step 2: Extract answers and cluster
            extracted_answers = auto_extract(
                task, group_responses, extra_info=group_extra_info
            )
            cluster_sizes = self._cluster_responses(
                extracted_answers, task, extra_info=group_extra_info
            )

            # Step 3: Compute bandpass rewards g(c) = (c/N) * Δ(c)
            rewards = compute_expected_marginal_passk_rewards(
                cluster_sizes, N, self.k
            )

            # Step 4: Fill reward_tensor and record scores
            for i in range(N):
                idx = prompt_i * N + i
                valid_len = group_valid_response_lengths[i]
                reward_tensor[idx, valid_len - 1] = rewards[i]
                scores[idx] = rewards[i]

            # Step 5: Collect diagnostic metrics
            unique_answers = len(set(extracted_answers))
            max_cluster = max(cluster_sizes) if cluster_sizes else 0
            avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
            # Find the cluster size with highest reward (bandpass peak)
            if rewards:
                peak_idx = rewards.index(max(rewards))
                peak_c = cluster_sizes[peak_idx]
            else:
                peak_c = 0

            all_metrics["num_unique_answers"].append(unique_answers)
            all_metrics["max_cluster_freq"].append(max_cluster)
            all_metrics["avg_delta_reward"].append(avg_reward)
            all_metrics["bandpass_peak_c"].append(peak_c)
            all_metrics["majority_ratio"].append(max_cluster / N if N > 0 else 0)

            # Also compute ground-truth accuracy metrics if available
            ground_truth = data[prompt_i * N].non_tensor_batch["reward_model"].get(
                "ground_truth", None
            )
            if ground_truth is not None:
                true_rewards, _ = auto_verify(
                    task, group_responses,
                    [ground_truth] * N,
                    extra_info=group_extra_info
                )
                gt_ratio = sum(true_rewards) / len(true_rewards)
                pass_n = 1.0 if sum(true_rewards) >= 1 else 0.0
                all_metrics["ground_truth_ratio"].append(gt_ratio)
                all_metrics[f"pass@{N}"].append(pass_n)

            # Print samples
            data_source = data[prompt_i * N].non_tensor_batch[self.reward_fn_key]
            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0
            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                prompt_str = self.tokenizer.decode(
                    data[prompt_i * N].batch["prompts"][
                        -data[prompt_i * N].batch["attention_mask"][
                            :data[prompt_i * N].batch["prompts"].shape[-1]
                        ].sum():
                    ],
                    skip_special_tokens=False,
                )
                print(f"[prompt] {prompt_str[:200]}...")
                print(f"[cluster_sizes] {cluster_sizes}")
                print(f"[rewards] {[f'{r:.6f}' for r in rewards]}")
                print(f"[unique_answers] {unique_answers}, [max_cluster] {max_cluster}")

        # Store scores for downstream access
        data.batch["acc"] = torch.tensor(
            scores, dtype=torch.float32, device=data.batch["prompts"].device
        )

        # Aggregate metrics
        for k_name, v in all_metrics.items():
            if isinstance(v, list) and len(v) > 0:
                mean_v = np.mean(v)
                print(f"[delta_passk/{k_name}] {mean_v:.4f}")
                delta_passk_info[f"delta_passk/{k_name}"] = mean_v

        return reward_tensor, reward_extra_info, delta_passk_info

    def _compute_eval_reward(self, data: DataProto):
        """
        Compute evaluation rewards using ground-truth labels (standard auto_verify).
        Mirrors TTRLRewardManager._compute_eval_reward.
        """
        reward_extra_info = defaultdict(list)
        eval_info = {}

        reward_tensor = torch.zeros_like(
            data.batch["responses"], dtype=torch.float32
        )
        already_print_data_sources = {}

        # Group by task
        task_groups = {}
        sample_valid_resp_len = {}

        for i in range(len(data)):
            data_item = data[i]
            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data_item.batch["attention_mask"][
                :prompt_length
            ].sum()
            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][
                prompt_length:
            ].sum()
            valid_response_ids = response_ids[:valid_response_length]
            sample_valid_resp_len[i] = int(valid_response_length)

            response_str = self.tokenizer.decode(
                valid_response_ids, skip_special_tokens=False
            )
            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            extra_info = data_item.non_tensor_batch.get("extra_info", None)

            # Print samples
            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0
            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                valid_prompt_ids = prompt_ids[-valid_prompt_length:]
                prompt_str = self.tokenizer.decode(
                    valid_prompt_ids, skip_special_tokens=False
                )
                print("[prompt]", prompt_str)
                print("[response]", response_str)

            task_key = self._data_source_to_task(data_source)
            if task_key not in task_groups:
                task_groups[task_key] = {
                    "indices": [], "outputs": [], "labels": [], "extra": []
                }
            task_groups[task_key]["indices"].append(i)
            task_groups[task_key]["outputs"].append(response_str)
            task_groups[task_key]["labels"].append(ground_truth)
            task_groups[task_key]["extra"].append(extra_info)

        # Verify by task
        for task_key, group in task_groups.items():
            rewards, verify_extra_info = auto_verify(
                task_key, group["outputs"], group["labels"],
                extra_info=group["extra"]
            )
            for k_name, v in verify_extra_info.items():
                if isinstance(v, list):
                    reward_extra_info[k_name] += v
            for idx_in_group, sample_idx in enumerate(group["indices"]):
                valid_len = sample_valid_resp_len[sample_idx]
                reward_tensor[sample_idx, valid_len - 1] = rewards[idx_in_group]

        return reward_tensor, reward_extra_info, eval_info

    def compute_post_ttrl_metrics(self, data: DataProto):
        """Compute post-training metrics (accuracy with GT)."""
        assert len(data) % self.n_samples_per_prompt == 0
        prompt_num = len(data) // self.n_samples_per_prompt

        post_info = {}
        post_metrics_list = defaultdict(list)

        for prompt_i in range(prompt_num):
            group_pred_outputs = []
            group_labels = []
            group_vote_rewards = []
            group_extra_info = []
            task = None

            for i in range(self.n_samples_per_prompt):
                data_item = data[prompt_i * self.n_samples_per_prompt + i]
                prompt_ids = data_item.batch["prompts"]
                prompt_length = prompt_ids.shape[-1]
                response_ids = data_item.batch["responses"]
                valid_response_length = data_item.batch["attention_mask"][
                    prompt_length:
                ].sum()
                valid_response_ids = response_ids[:valid_response_length]

                response_str = self.tokenizer.decode(
                    valid_response_ids, skip_special_tokens=False
                )
                ground_truth = data_item.non_tensor_batch["reward_model"][
                    "ground_truth"
                ]
                data_source = data_item.non_tensor_batch[self.reward_fn_key]
                vote_reward = data_item.batch["acc"]
                extra_info = data_item.non_tensor_batch.get("extra_info", None)

                if task is None:
                    task = self._data_source_to_task(data_source)

                group_labels.append(ground_truth)
                group_pred_outputs.append(response_str)
                group_vote_rewards.append(vote_reward)
                group_extra_info.append(extra_info)

            post_metrics = post_test_time_train_metrics(
                group_pred_outputs, group_labels, group_vote_rewards,
                task=task, extra_info=group_extra_info
            )
            for k_name, v in post_metrics.items():
                post_metrics_list[k_name].append(v)

        for k_name, v in post_metrics_list.items():
            if isinstance(v, list):
                v = np.mean(v)
                print(f"[{k_name}]", v)
                post_info[k_name] = v

        return post_info

    def __call__(self, data: DataProto, return_dict=False):
        if self.mode == "train":
            reward_tensor, reward_extra_info, info = self._compute_delta_passk_reward(
                data
            )
        elif self.mode == "eval":
            reward_tensor, reward_extra_info, info = self._compute_eval_reward(data)
        else:
            raise NotImplementedError(
                f"Mode {self.mode} is not supported for DeltaPasskRewardManager"
            )

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
                "ttrl_info": info,
            }
        else:
            return reward_tensor
