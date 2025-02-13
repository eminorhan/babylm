#!/bin/bash

#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=240GB
#SBATCH --time=0:10:00
#SBATCH --job-name=sample
#SBATCH --output=sample_%A_%a.out
#SBATCH --array=0

export TRANSFORMERS_CACHE="/vast/eo41/huggingface"

# root model directory
MODEL_ROOT_DIR="/vast/eo41/babylm/models"
SP="phi-3-tinystories"

python -u /scratch/eo41/babylm/sample.py \
    --model_name_or_path "microsoft/Phi-3-mini-4k-instruct" \
    --test_file "data/tinystories/TinyStoriesV2-GPT4-valid-all.json" \
    --output_dir "samples/${SP}" \
    --save_prefix ${SP} \
    --block_size 1024 \
    --per_device_eval_batch_size 1 \
    --overwrite_cache

echo "Done"