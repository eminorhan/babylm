#!/bin/bash

#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=62GB
#SBATCH --time=00:55:00
#SBATCH --job-name=train_babylm_random_wikipedia_100M_gpt
#SBATCH --output=train_babylm_random_wikipedia_100M_gpt_%A_%a.out
#SBATCH --array=0

export HF_HOME="/vast/eo41/huggingface"
export HF_DATASETS_CACHE="/vast/eo41/huggingface"

# root model directory
MODEL_ROOT_DIR="/vast/eo41/babylm/models"
SP="babylm_random_wikipedia_10M_1_gpt"

accelerate launch --config_file accelerate_1gpu_config.yaml --num_cpu_threads_per_process 16 /scratch/eo41/babylm/train.py \
    --model_name_or_path "meta-llama/Llama-3.2-1B" \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 4 \
    --learning_rate 0.0003 \
    --output_dir "${MODEL_ROOT_DIR}/${SP}" \
    --save_prefix ${SP} \
    --block_size 1024 \
    --num_train_epochs 25 \
    --checkpointing_steps 100 \
    --overwrite_cache

echo "Done"