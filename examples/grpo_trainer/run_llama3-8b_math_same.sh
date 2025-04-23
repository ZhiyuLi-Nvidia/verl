set -x

cd /opt/verl

# preprocess dataset if needed
if [[ -d "processed_data/nvidia_math_dataset" ]] && [[ "$(ls -A processed_data/nvidia_math_dataset)" ]]; then
    echo "processed_data/nvidia_math_dataset already exists and is not empty"
else
    python3 examples/data_preprocess/nvidia_math_dataset.py --local_dir processed_data/nvidia_math_dataset
fi

export VLLM_ATTENTION_BACKEND=XFORMERS

math_train_path=/opt/verl/processed_data/nvidia_math_dataset/train.parquet
math_test_path=/opt/verl/processed_data/nvidia_math_dataset/test.parquet

train_files="['$math_train_path']"
test_files="['$math_test_path']"

# mapping strategy
# data.train_batch_size: policy.train_global_batch_size
# actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu: policy.train_micro_batch_size
# actor_rollout_ref.actor.ppo_mini_batch_size: grpo.num_prompts_per_step
# actor_rollout_ref.rollout.n: grpo.num_generations_per_prompt
# actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu: grpo.logprob_batch_size
# actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu: grpo.logprob_batch_size
# data.max_response_length: generation.vllm_cfg.max_model_len

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=64 \
    data.max_prompt_length=1024 \
    data.max_response_length=4096 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=meta-llama/Llama-3.1-8B-Instruct \
    actor_rollout_ref.actor.optim.lr=3e-7 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=32 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='verl_grpo_example_nvidia_math' \
    trainer.experiment_name='llama3.1_8b_same_param' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=$SLURM_NNODES \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.val_before_train=False \
    trainer.total_epochs=1 $@