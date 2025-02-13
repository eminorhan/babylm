#!/bin/bash

#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=62GB
#SBATCH --time=2:00:00
#SBATCH --job-name=sample
#SBATCH --output=sample_%A_%a.out
#SBATCH --array=0

export HF_HOME="/vast/eo41/huggingface"
export HF_DATASETS_CACHE="/vast/eo41/huggingface"

# root model directory
MODEL_ROOT_DIR="/scratch/eo41/babylm/models"
SP="babylm_100m_9000"

# dataset names (matching the dictionary keys in train.py)
DATASET_NAMES=(
    "babylm_10M"
    "wikipedia_10M_1" "wikipedia_10M_2" "wikipedia_10M_3"
    "gutenberg_10M_1" "gutenberg_10M_2" "gutenberg_10M_3"
    "tinystories_10M_1" "tinystories_10M_2" "tinystories_10M_3"
    "pythonedu_10M_1" "pythonedu_10M_2" "pythonedu_10M_3"
    "babylm_100M"
    "wikipedia_100M_1" "wikipedia_100M_2" "wikipedia_100M_3"
    "gutenberg_100M_1" "gutenberg_100M_2" "gutenberg_100M_3"
    "tinystories_100M_1" "tinystories_100M_2" "tinystories_100M_3"
    "pythonedu_100M_1" "pythonedu_100M_2" "pythonedu_100M_3"
)

# get dataset name for current array index
DATASET_NAME=${DATASET_NAMES[$SLURM_ARRAY_TASK_ID]}

python -u /scratch/eo41/babylm/sample.py \
    --model_name_or_path "${MODEL_ROOT_DIR}/babylm_100M/step_9000" \
    --dataset_name "$DATASET_NAME" \
    --output_dir "samples/${SP}" \
    --save_prefix ${SP} \
    --block_size 1024 \
    --per_device_batch_size 1 \
    --overwrite_cache

echo "Done"