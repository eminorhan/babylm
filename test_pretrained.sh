#!/bin/bash

#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=240GB
#SBATCH --time=01:00:00
#SBATCH --job-name=test_pretrained
#SBATCH --output=test_pretrained_%A_%a.out
#SBATCH --array=0

export HF_HOME="/vast/eo41/huggingface"
export HF_DATASETS_CACHE="/vast/eo41/huggingface"

# root model directory
MODEL_ROOT_DIR="/vast/eo41/babylm/models/babylm_10M_llama"

accelerate launch --config_file accelerate_1gpu_config.yaml --num_cpu_threads_per_process 16 /scratch/eo41/babylm/test_pretrained.py \
    --model_name_or_path "${MODEL_ROOT_DIR}/step_1200" \
    --data_files "data/text_data/train_10M/childes.txt" \
                  "data/text_data/train_10M/bnc_spoken.txt" \
                  "data/text_data/train_10M/gutenberg.txt" \
                  "data/text_data/train_10M/open_subtitles.txt" \
                  "data/text_data/train_10M/simple_wiki.txt" \
                  "data/text_data/train_10M/switchboard.txt" \
    --per_device_eval_batch_size 2 \
    --block_size 1024 \
    --overwrite_cache \
    --use_pretrained_weights

echo "Done"