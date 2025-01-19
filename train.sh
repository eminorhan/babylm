#!/bin/bash

#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=62GB
#SBATCH --time=2:00:00
#SBATCH --job-name=train
#SBATCH --output=train_%A_%a.out
#SBATCH --array=0-12

export HF_HOME="/vast/eo41/huggingface"
export HF_DATASETS_CACHE="/vast/eo41/huggingface"

# root model directory
MODEL_ROOT_DIR="/scratch/eo41/babylm/models"

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

accelerate launch --config_file accelerate_1gpu_config.yaml --num_cpu_threads_per_process 16 /scratch/eo41/babylm/train.py \
    --model_name_or_path "meta-llama/Llama-3.2-1B" \
    --dataset_name "$DATASET_NAME" \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 8 \
    --learning_rate 0.0003 \
    --output_dir "${MODEL_ROOT_DIR}/${DATASET_NAME}" \
    --save_prefix ${DATASET_NAME} \
    --block_size 1024 \
    --num_train_epochs 15 \
    --checkpointing_steps 100 \
    --overwrite_cache

echo "Done"