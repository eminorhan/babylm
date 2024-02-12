#!/bin/bash

#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=240GB
#SBATCH --time=1:00:00
#SBATCH --job-name=train_gpt2_tinystories
#SBATCH --output=train_gpt2_tinystories_%A_%a.out
#SBATCH --array=0

export TRANSFORMERS_CACHE="/vast/eo41/huggingface"
export HF_DATASETS_CACHE="/vast/eo41/huggingface"

# root model directory
MODEL_ROOT_DIR="/vast/eo41/babylm/models"
SP="gpt2-tinystories"

# gpt2
accelerate launch --config_file accelerate_1gpu_config.yaml --num_cpu_threads_per_process 16 /scratch/eo41/babylm/train.py \
    --model_name_or_path "gpt2" \
    --train_files "data/tinystories/TinyStoriesV2-GPT4-train.txt" \
    --val_files "data/tinystories/TinyStoriesV2-GPT4-valid.txt" \
    --tokenizer_file "tinystories_tokenizer.json" \
    --per_device_train_batch_size 64 \
    --gradient_accumulation_steps 1 \
    --learning_rate 0.0001 \
    --output_dir "${MODEL_ROOT_DIR}/${SP}" \
    --save_prefix ${SP} \
    --block_size 512 \
    --num_train_epochs 1000 \
    --checkpointing_steps 1000 \
    --overwrite_cache

echo "Done"