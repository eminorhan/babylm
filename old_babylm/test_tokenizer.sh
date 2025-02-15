#!/bin/bash

#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=240GB
#SBATCH --time=1:00:00
#SBATCH --job-name=test_tokenizer
#SBATCH --output=test_tokenizer_%A_%a.out
#SBATCH --array=0

export TRANSFORMERS_CACHE="/vast/eo41/huggingface"
export HF_DATASETS_CACHE="/vast/eo41/huggingface"

python -u /scratch/eo41/babylm/test_tokenizer.py \
    --train_files "data/tinystories/TinyStoriesV2-GPT4-train-all.json" \
    --val_files "data/tinystories/TinyStoriesV2-GPT4-valid-all.json" \
    --tokenizer_file "tinystories_tokenizer.json" \
    --block_size 512 \
    --overwrite_cache

echo "Done"