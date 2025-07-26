set -x

cd /opt/verl

processed_data_dir="processed_data/nvidia_math_dataset"
# processed_data_dir="processed_data_old_cot_template/nvidia_math_dataset"
# preprocess dataset if needed
if [[ -d "$processed_data_dir" ]] && [[ "$(ls -A $processed_data_dir)" ]]; then
    echo "$processed_data_dir already exists and is not empty"
else
    mkdir -p $processed_data_dir
    python3 examples/data_preprocess/nvidia_math_dataset.py --local_dir $processed_data_dir
fi

# export VLLM_ATTENTION_BACKEND=XFORMERS

math_train_path=/opt/verl/$processed_data_dir/train.parquet
math_test_path=/opt/verl/$processed_data_dir/test.parquet

train_files="['$math_train_path']"
test_files="['$math_test_path']"

# reinforcer args
num_prompts_per_step=64
num_generations_per_prompt=32
train_global_batch_size=64

# mapping strategy
# data.train_batch_size: policy.num_prompts_per_step
# actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu: policy.train_micro_batch_size
# actor_rollout_ref.actor.ppo_mini_batch_size: train_global_batch_size / num_generations_per_prompt
# actor_rollout_ref.rollout.n: grpo.num_generations_per_prompt
# actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu: grpo.logprob_batch_size
# actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu: grpo.logprob_batch_size
# data.max_response_length: generation.vllm_cfg.max_model_len


python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=$num_prompts_per_step \
    data.max_prompt_length=1024 \
    data.max_response_length=4096 \
    data.filter_overlong_prompts=False \
    data.truncation='error' \
    data.shuffle=False \
    actor_rollout_ref.model.path=meta-llama/Llama-3.1-8B-Instruct \
    actor_rollout_ref.actor.optim.lr=0. \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.ppo_mini_batch_size=$((train_global_batch_size / num_generations_per_prompt)) \
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
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.max_model_len=4096 \
    actor_rollout_ref.rollout.n=$num_generations_per_prompt \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.total_training_steps=1 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='verl_grpo_example_nvidia_math' \
    trainer.experiment_name='llama3.1_8b_same_param' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=${NNODES:-1} \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.val_before_train=False \
    trainer.total_epochs=1 $@