set -x

cd /opt/verl

# preprocess dataset if needed
if [[ -d "processed_data/nvidia_math_dataset" ]] && [[ "$(ls -A processed_data/nvidia_math_dataset)" ]]; then
    echo "processed_data/nvidia_math_dataset already exists and is not empty"
else
    python3 examples/data_preprocess/nvidia_math_dataset.py --local_dir processed_data/nvidia_math_dataset
fi

math_train_path=/opt/verl/processed_data/nvidia_math_dataset/train.parquet
math_test_path=/opt/verl/processed_data/nvidia_math_dataset/test.parquet

train_files="['$math_train_path']"
test_files="['$math_test_path']"

LLM="/lustre/fsw/portfolios/coreai/users/zhiyul/hf/hub/models--deepseek-ai--DeepSeek-V3-medium"

DIST_CKPT_PATH="/lustre/fsw/portfolios/coreai/users/zhiyul/benchmark-rl/modified_checkpoints/tron/deepseek-ai/DeepSeek-V3-medium/iter_0000000/"
NODES=16
PP=8
TP=1
EP=16
ETP=1
INFER_TP=64

additional_params="
+actor_rollout_ref.actor.megatron.override_transformer_config.num_layers=24 \
"
ALL_OFFLOAD=${ALL_OFFLOAD:-True}
COMMON_PARAM_OFFLOAD=${COMMON_PARAM_OFFLOAD:-$ALL_OFFLOAD}
COMMON_GRAD_OFFLOAD=${COMMON_GRAD_OFFLOAD:-$ALL_OFFLOAD}
COMMON_OPTIMIZER_OFFLOAD=${COMMON_OPTIMIZER_OFFLOAD:-$ALL_OFFLOAD}

ACTOR_PARAM_OFFLOAD=${ACTOR_PARAM_OFFLOAD:-$COMMON_PARAM_OFFLOAD}
ACTOR_GRAD_OFFLOAD=${ACTOR_GRAD_OFFLOAD:-$COMMON_GRAD_OFFLOAD}
ACTOR_OPTIMIZER_OFFLOAD=${ACTOR_OPTIMIZER_OFFLOAD:-$COMMON_OPTIMIZER_OFFLOAD}
REF_PARAM_OFFLOAD=${REF_PARAM_OFFLOAD:-$COMMON_PARAM_OFFLOAD}
CRITIC_PARAM_OFFLOAD=${CRITIC_PARAM_OFFLOAD:-$COMMON_PARAM_OFFLOAD}
CRITIC_GRAD_OFFLOAD=${CRITIC_GRAD_OFFLOAD:-$COMMON_GRAD_OFFLOAD}
CRITIC_OPTIMIZER_OFFLOAD=${CRITIC_OPTIMIZER_OFFLOAD:-$COMMON_OPTIMIZER_OFFLOAD}
RM_PARAM_OFFLOAD=${RM_PARAM_OFFLOAD:-$COMMON_PARAM_OFFLOAD}


# full recompute
# +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform \
# +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full \
# +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1 \


# mapping strategy
# data.train_batch_size: policy.train_global_batch_size
# actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu: policy.train_micro_batch_size
# actor_rollout_ref.actor.ppo_mini_batch_size: grpo.num_prompts_per_step
# actor_rollout_ref.rollout.n: grpo.num_generations_per_prompt
# actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu: grpo.logprob_batch_size
# actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu: grpo.logprob_batch_size
# data.max_response_length: generation.vllm_cfg.max_model_len

n_resp_per_prompt=16

# RAY_ADDRESS='auto' ray job submit --working-dir . --
python3 -m verl.trainer.main_ppo --config-path=./config --config-name='ppo_megatron_trainer'\
    algorithm.adv_estimator=grpo \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=256 \
    data.max_prompt_length=128 \
    data.max_response_length=128 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.trust_remote_code=True \
    actor_rollout_ref.model.path=$LLM \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$INFER_TP \
    algorithm.use_kl_in_reward=False \
    trainer.logger=['console','wandb'] \
    trainer.project_name='verl_grpo_example_nvidia_math' \
    trainer.experiment_name='deepseek_v2_chat_math_megatron' \
    trainer.experiment_name=$LLM \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=$NODES \
    trainer.save_freq=-1 \
    trainer.test_freq=5 \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_router_bias_update_rate=0.0 \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_aux_loss_coeff=0.0001 \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=$PP \
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=$PP \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=$TP \
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=$TP \
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=$EP \
    actor_rollout_ref.ref.megatron.expert_model_parallel_size=$EP \
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=$ETP \
    actor_rollout_ref.ref.megatron.expert_tensor_parallel_size=$ETP \
    actor_rollout_ref.actor.megatron.param_offload=${ACTOR_PARAM_OFFLOAD} \
    actor_rollout_ref.actor.megatron.optimizer_offload=${ACTOR_OPTIMIZER_OFFLOAD} \
    actor_rollout_ref.actor.megatron.grad_offload=${ACTOR_GRAD_OFFLOAD} \
    actor_rollout_ref.ref.megatron.param_offload=${REF_PARAM_OFFLOAD} \
    actor_rollout_ref.actor.megatron.use_dist_checkpointing=True \
    actor_rollout_ref.ref.megatron.use_dist_checkpointing=True \
    actor_rollout_ref.actor.megatron.dist_checkpointing_path=$DIST_CKPT_PATH \
    actor_rollout_ref.ref.megatron.dist_checkpointing_path=$DIST_CKPT_PATH \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir=$CKPT_DIR \
    trainer.val_before_train=False \
    trainer.total_epochs=100 $additional_params $@