#!/bin/bash

#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=240GB
#SBATCH --time=1:00:00
#SBATCH --job-name=train
#SBATCH --output=train_%A_%a.out
#SBATCH --array=0

export TRANSFORMERS_CACHE="/vast/eo41/huggingface"

# root model directory
MODEL_ROOT_DIR="/vast/eo41/babylm/models"
SP="gpt2-aochildes"

# gpt2
python -u /scratch/eo41/babylm/train.py \
    --model_name_or_path "gpt2" \
    --train_file "data/babylm_10M/aochildes.txt" \
    --per_device_train_batch_size 256 \
    --learning_rate 0.0005 \
    --output_dir "${MODEL_ROOT_DIR}/gpt2-aochildes" \
    --save_prefix ${SP} \
    --block_size 128 \
    --num_train_epochs 10 \
    --overwrite_cache

echo "Done"