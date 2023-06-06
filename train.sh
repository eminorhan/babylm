#!/bin/bash

#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=240GB
#SBATCH --time=2:00:00
#SBATCH --job-name=train
#SBATCH --output=train_%A_%a.out
#SBATCH --array=0

export TRANSFORMERS_CACHE="/vast/eo41/huggingface"
export HF_DATASETS_CACHE="/vast/eo41/huggingface"

# root model directory
MODEL_ROOT_DIR="/vast/eo41/babylm/models"
SP="gpt2-xl-10M"

# gpt2
python -u /scratch/eo41/babylm/train.py \
    --model_name_or_path "gpt2-xl" \
    --train_files "data/babylm_10M/aochildes.txt" \
                  "data/babylm_10M/bnc_spoken.txt" \
                  "data/babylm_10M/cbt.txt" \
                  "data/babylm_10M/children_stories.txt" \
                  "data/babylm_10M/gutenberg.txt" \
                  "data/babylm_10M/open_subtitles.txt" \
                  "data/babylm_10M/qed.txt" \
                  "data/babylm_10M/simple_wikipedia.txt" \
                  "data/babylm_10M/switchboard.txt" \
                  "data/babylm_10M/wikipedia.txt" \
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
    --tokenizer_file "babylm_10M_tokenizer.json" \
    --per_device_train_batch_size 6 \
    --learning_rate 0.0001 \
    --output_dir "${MODEL_ROOT_DIR}/${SP}" \
    --save_prefix ${SP} \
    --block_size 512 \
    --num_train_epochs 2 \
    --checkpointing_steps 1000 \
    --overwrite_cache

echo "Done"