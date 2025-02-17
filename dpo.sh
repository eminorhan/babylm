#!/bin/bash

#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=62GB
#SBATCH --time=01:00:00
#SBATCH --job-name=dpo
#SBATCH --output=dpo_%A_%a.out
#SBATCH --array=0

export HF_HOME="/vast/eo41/huggingface"
export HF_DATASETS_CACHE="/vast/eo41/huggingface"

# root model directory
MODEL_ROOT_DIR="/scratch/eo41/babylm/models"

# TODO: arguments are not quite correct yet! FIX
accelerate launch --config_file accelerate_1gpu_config.yaml --num_cpu_threads_per_process 16 /scratch/eo41/babylm/dpo.py \
    --model_name_or_path "${MODEL_ROOT_DIR}/babylm_100M/step_9000" \
    --dataset_name babylm_100m_9000.json\
    --per_device_train_batch_size 6 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 8 \
    --learning_rate 0.0001 \
    --output_dir dpo \
    --num_train_epochs 1 \
    --eval_strategy steps \
    --eval_steps 300 \
    --logging_steps 100 \
    --save_steps 300 \
    --max_length 1024 \
    --max_prompt_length 512 \
    --logging_first_step \
    --no_remove_unused_columns \
    --bf16 \
    --warmup_steps 50

echo "Done"