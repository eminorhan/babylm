#!/bin/bash

#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=240GB
#SBATCH --time=2:00:00
#SBATCH --job-name=sample
#SBATCH --output=sample_%A_%a.out
#SBATCH --array=0

export TRANSFORMERS_CACHE="/vast/eo41/huggingface"

# root model directory
MODEL_ROOT_DIR="/vast/eo41/babylm/models"
SP="gpt2-10M-all"

# gpt2
python -u /scratch/eo41/babylm/sample.py \
    --model_name_or_path "${MODEL_ROOT_DIR}/${SP}" \
    --output_dir "samples/${SP}" \
    --save_prefix ${SP} \
    --block_size 512 \
    --per_device_eval_batch_size 1 \
    --overwrite_cache

echo "Done"