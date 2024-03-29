#!/bin/bash

#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=480GB
#SBATCH --time=48:00:00
#SBATCH --job-name=train
#SBATCH --output=train_%A_%a.out
#SBATCH --array=0

export TRANSFORMERS_CACHE="/vast/eo41/huggingface"
export HF_DATASETS_CACHE="/vast/eo41/huggingface"

# root model directory
MODEL_ROOT_DIR="/vast/eo41/babylm/models"
SP="gpt2-xl-100M"

# gpt2-xl
accelerate launch --config_file accelerate_4gpu_config.yaml --num_cpu_threads_per_process 16 /scratch/eo41/babylm/train.py \
    --model_name_or_path "gpt2-xl" \
    --train_files "data/babylm_100M/aochildes.txt" \
                  "data/babylm_100M/bnc_spoken.txt" \
                  "data/babylm_100M/cbt.txt" \
                  "data/babylm_100M/children_stories.txt" \
                  "data/babylm_100M/gutenberg.txt" \
                  "data/babylm_100M/open_subtitles.txt" \
                  "data/babylm_100M/qed.txt" \
                  "data/babylm_100M/simple_wikipedia.txt" \
                  "data/babylm_100M/switchboard.txt" \
                  "data/babylm_100M/wikipedia.txt" \
    --val_files "data/babylm_dev/aochildes.txt" \
                "data/babylm_dev/bnc_spoken.txt" \
                "data/babylm_dev/cbt.txt" \
                "data/babylm_dev/children_stories.txt" \
                "data/babylm_dev/gutenberg.txt" \
                "data/babylm_dev/open_subtitles.txt" \
                "data/babylm_dev/qed.txt" \
                "data/babylm_dev/simple_wikipedia.txt" \
                "data/babylm_dev/switchboard.txt" \
                "data/babylm_dev/wikipedia.txt" \
    --tokenizer_file "babylm_100M_tokenizer.json" \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --learning_rate 0.0001 \
    --output_dir "${MODEL_ROOT_DIR}/${SP}" \
    --save_prefix ${SP} \
    --block_size 512 \
    --num_train_epochs 1000 \
    --checkpointing_steps 1000 \
    --overwrite_cache

echo "Done"