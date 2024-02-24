#!/bin/bash

#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=240GB
#SBATCH --time=2:00:00
#SBATCH --job-name=sample
#SBATCH --output=sample_%A_%a.out
#SBATCH --array=0

export TRANSFORMERS_CACHE="/vast/eo41/huggingface"

# root model directory
MODEL_ROOT_DIR="/vast/eo41/babylm/models"
SP="gpt2-tinystories-3000"

python -u /scratch/eo41/babylm/sample.py \
    --model_name_or_path "${MODEL_ROOT_DIR}/${SP}/step_50000" \
    --test_file "data/tinystories/TinyStoriesV2-GPT4-valid-all.json" \
    --output_dir "samples/${SP}" \
    --save_prefix ${SP} \
    --block_size 1024 \
    --per_device_eval_batch_size 1 \
    --overwrite_cache

echo "Done"