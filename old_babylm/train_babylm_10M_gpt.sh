#!/bin/bash

#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=240GB
#SBATCH --time=12:00:00
#SBATCH --job-name=train_babylm_10M_gpt
#SBATCH --output=train_babylm_10M_gpt_%A_%a.out
#SBATCH --array=0

export HF_HOME="/vast/eo41/huggingface"
export HF_DATASETS_CACHE="/vast/eo41/huggingface"

# root model directory
MODEL_ROOT_DIR="/vast/eo41/babylm/models"
SP="babylm_10M_gpt"

accelerate launch --config_file accelerate_1gpu_config.yaml --num_cpu_threads_per_process 16 /scratch/eo41/babylm/train.py \
    --model_name_or_path "gpt2-large" \
    --train_files "data/train_10M/childes.txt" \
                  "data/train_10M/bnc_spoken.txt" \
                  "data/train_10M/gutenberg.txt" \
                  "data/train_10M/open_subtitles.txt" \
                  "data/train_10M/simple_wiki.txt" \
                  "data/train_10M/switchboard.txt" \
    --val_files "data/dev/childes.txt" \
                "data/dev/bnc_spoken.txt" \
                "data/dev/gutenberg.txt" \
                "data/dev/open_subtitles.txt" \
                "data/dev/simple_wiki.txt" \
                "data/dev/switchboard.txt" \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 8 \
    --learning_rate 0.0001 \
    --output_dir "${MODEL_ROOT_DIR}/${SP}" \
    --save_prefix ${SP} \
    --block_size 1024 \
    --num_train_epochs 20 \
    --checkpointing_steps 100 \
    --overwrite_cache

echo "Done"