#!/bin/bash

#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=480GB
#SBATCH --time=48:00:00
#SBATCH --job-name=dpo_babylm_100M_llama
#SBATCH --output=dpo_babylm_100M_llama_%A_%a.out
#SBATCH --array=0

export HF_HOME="/vast/eo41/huggingface"
export HF_DATASETS_CACHE="/vast/eo41/huggingface"

# Define the base path for models
BASE_MODEL=babylm_100M_llama_hq
BASE_PATH=/vast/eo41/babylm/models/${BASE_MODEL}

# Compute the step number based on the array index
STEP=$((1000 * SLURM_ARRAY_TASK_ID))

# Construct the model path
MODEL_PATH=${BASE_PATH}/step_${STEP}
MODEL_BASENAME=$(basename $MODEL_PATH)

# TODO: arguments are not quite correct yet! FIX
accelerate launch --config_file accelerate_4gpu_config.yaml --num_cpu_threads_per_process 16 /scratch/eo41/babylm/dpo.py \
    --model_name_or_path $MODEL_PATH \
    --dataset_name trl-internal-testing/hh-rlhf-helpful-base-trl-style \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 16 \
    --learning_rate 0.0001 \
    --output_dir dpo_hh/${BASE_MODEL}/${MODEL_BASENAME} \
    --num_train_epochs 20 \
    --checkpointing_steps 1000 \
    --overwrite_cache \
    --logging_steps 10 \
    --eval_steps 500 \
    --report_to wandb \
    --logging_first_step \
    --no_remove_unused_columns

echo "Done"