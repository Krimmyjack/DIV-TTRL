#!/bin/bash
"""
Launch script for Delta Pass@k Label-Free RL
Usage: bash examples/labelfree/math_delta_passk.sh --task MATH-TTT --backbone Qwen3-4B-Base
"""
export WANDB_ENTITY=2691454060-ucla
unset VLLM_ATTENTION_BACKEND
export VLLM_USE_V1=1

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --task)
            TASK="$2"
            shift 2
            ;;
        --backbone)
            BACKBONE="$2"
            shift 2
            ;;
        --temp)
            TEMP="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

TASK=${TASK:-"MATH"}
BACKBONE=${BACKBONE:-"Qwen3-4B-Base"}
TEMP=${TEMP:-"1.0"}

RAW_TASK="$TASK"
if [ "$RAW_TASK" = "math_train" ]; then
  TASK="MATH-TTT"
else
  TASK="$TASK-TTT"
fi

pkill -f "python.*main_ppo" || true
pkill -f "python.*main_dapo" || true
ray stop --force 2>/dev/null || true
sleep 2

DATE=$(date +%m%d)
TIME_TAG=$(date +%H%M%S)

# ==== DELTA PASS@K CONFIGURATION ====
ADVANTAGE="delta_passk"
REWARD_MANAGER="delta_passk_ttrl"
# ====================================

# Set K value for Pass@k
K=4
MAX_PROMPT_LENGTH=1024
MAX_RESPONSE_LENGTH=$((1024 * $K))
MAX_TOKEN_LEN=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))
MAX_TOKEN_LEN2=$((MAX_TOKEN_LEN * 2))

EPISODE=4
DATA_TRAIN_BATCH_SIZE=32

# In Delta Pass@k, there is no separate voting pool.
# We set n_votes = n_samples to ensure we use exactly the universe size N for the margin formulation.
N_SAMPLES_PER_PROMPT=32
N_VOTES_PER_PROMPT=$N_SAMPLES_PER_PROMPT 

MINI_BATCH_SIZE=1 
MICRO_BATCH_SIZE=2 

DATA_LOCAL_DIR="data"

if [[ "$BACKBONE" == *"/"* ]]; then
  BACKBONE_PATH="$BACKBONE"
  BACKBONE_NAME="${BACKBONE##*/}"
else
  BACKBONE_PATH="/root/autodl-tmp/model/${BACKBONE}"
  BACKBONE_NAME="$BACKBONE"
fi

MODEL="${TASK}-${BACKBONE_NAME}"
EXPERIMENT="DeltaPassK-RL"
LOG_NAME="${EXPERIMENT}-${MODEL}"
WANDB_PROJECT="TTRL-MATH500-DeltaPassK"
OUTPUT_DIR="/root/autodl-tmp/model/${WANDB_PROJECT}/${MODEL}/${EXPERIMENT}/${TIME_TAG}"

if [ "$RAW_TASK" = "math_train" ]; then
  TRAIN_FILES="math_train_ttrl.parquet"
else
  if [[ "$TASK" == *"AIME"* ]]; then
    TRAIN_FILES="train-simplerl-16.parquet"
  else
    TRAIN_FILES="train-simplerl.parquet"
  fi
fi

echo "=== Delta Pass@k TTRL Configuration ==="
echo "Task: $TASK"
echo "Backbone model: $BACKBONE_PATH"
echo "Reward Manager: $REWARD_MANAGER"
echo "Advantage: $ADVANTAGE"
echo "N samples (universe size): $N_SAMPLES_PER_PROMPT"
echo "Target k: $K"
echo "Output directory: $OUTPUT_DIR"
echo "======================================="

python -m verl.trainer.main_ppo \
  reward_model.reward_manager=$REWARD_MANAGER \
  reward_model.reward_kwargs.n_samples_per_prompt=$N_SAMPLES_PER_PROMPT \
  reward_model.reward_kwargs.n_votes_per_prompt=$N_VOTES_PER_PROMPT \
  reward_model.reward_kwargs.k=$K \
  reward_model.reward_kwargs.mode="train" \
  data.train_files=["$DATA_LOCAL_DIR/$TASK/train-simplerl.parquet"] \
  data.val_files=["$DATA_LOCAL_DIR/$TASK/test-simplerl.parquet"] \
  data.max_prompt_length=$MAX_PROMPT_LENGTH \
  data.max_response_length=$MAX_RESPONSE_LENGTH \
  data.train_batch_size=$DATA_TRAIN_BATCH_SIZE \
  data.filter_overlong_prompts=True \
  data.truncation='error' \
  actor_rollout_ref.model.path=$BACKBONE_PATH \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.ppo_mini_batch_size=$MINI_BATCH_SIZE \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.optim.lr=5e-7 \
  actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.03 \
  actor_rollout_ref.actor.optim.warmup_style='cosine' \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$((MAX_TOKEN_LEN2)) \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.temperature=$TEMP \
  actor_rollout_ref.rollout.enforce_eager=False \
  actor_rollout_ref.rollout.free_cache_engine=False \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
  actor_rollout_ref.rollout.n=$N_SAMPLES_PER_PROMPT \
  actor_rollout_ref.rollout.val_kwargs.do_sample=False \
  actor_rollout_ref.rollout.val_kwargs.top_p=0 \
  actor_rollout_ref.rollout.val_kwargs.temperature=0 \
  actor_rollout_ref.rollout.max_model_len=$((MAX_TOKEN_LEN)) \
  actor_rollout_ref.rollout.max_num_batched_tokens=$((MAX_TOKEN_LEN2)) \
  critic.optim.lr=9e-6 \
  critic.model.use_remove_padding=True \
  critic.model.path=$BACKBONE_PATH \
  critic.model.enable_gradient_checkpointing=True \
  critic.ppo_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
  critic.model.fsdp_config.param_offload=False \
  critic.model.fsdp_config.optimizer_offload=False \
  algorithm.kl_ctrl.kl_coef=0.00 \
  algorithm.adv_estimator=$ADVANTAGE \
  trainer.logger=['console','wandb'] \
  trainer.project_name=$WANDB_PROJECT \
  trainer.experiment_name=$LOG_NAME \
  trainer.n_gpus_per_node=8 \
  trainer.nnodes=1 \
  trainer.save_freq=15 \
  trainer.test_freq=5 \
  trainer.max_actor_ckpt_to_keep=0 \
  trainer.max_critic_ckpt_to_keep=0 \
  trainer.default_local_dir=$OUTPUT_DIR \
  trainer.total_epochs=$EPISODE "$@"

echo "=== Training Completed ==="
