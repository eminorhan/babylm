#!/bin/bash

#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=480GB
#SBATCH --time=48:00:00
#SBATCH --job-name=train_babylm_10M_llama
#SBATCH --output=train_babylm_10M_llama_%A_%a.out
#SBATCH --array=0

export HF_HOME="/vast/eo41/huggingface"
export HF_DATASETS_CACHE="/vast/eo41/huggingface"

# root model directory
MODEL_ROOT_DIR="/vast/eo41/babylm/models"
SP="babylm_10M_llama"

accelerate launch --config_file accelerate_4gpu_config.yaml --num_cpu_threads_per_process 16 /scratch/eo41/babylm/train.py \
    --model_name_or_path "meta-llama/Meta-Llama-3.1-8B" \
    --train_files "data/text_data/train_10M/childes.txt" \
                  "data/text_data/train_10M/bnc_spoken.txt" \
                  "data/text_data/train_10M/gutenberg.txt" \
                  "data/text_data/train_10M/open_subtitles.txt" \
                  "data/text_data/train_10M/simple_wiki.txt" \
                  "data/text_data/train_10M/switchboard.txt" \
    --val_files "data/text_data/dev/childes.txt" \
                "data/text_data/dev/bnc_spoken.txt" \
                "data/text_data/dev/gutenberg.txt" \
                "data/text_data/dev/open_subtitles.txt" \
                "data/text_data/dev/simple_wiki.txt" \
                "data/text_data/dev/switchboard.txt" \
    --per_device_train_batch_size 6 \
    --gradient_accumulation_steps 16 \
    --learning_rate 0.0001 \
    --output_dir "${MODEL_ROOT_DIR}/${SP}" \
    --save_prefix ${SP} \
    --block_size 1024 \
    --num_train_epochs 250 \
    --checkpointing_steps 1000 \
    --overwrite_cache

echo "Done"