#!/bin/bash

#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=240GB
#SBATCH --time=1:00:00
#SBATCH --job-name=dpo
#SBATCH --output=dpo_%A_%a.out
#SBATCH --array=10

export HF_HOME="/vast/eo41/huggingface"
export HF_DATASETS_CACHE="/vast/eo41/huggingface"

# Define the base path for models
BASE_MODEL=babylm_100M_gpt
BASE_PATH=/vast/eo41/babylm/models/${BASE_MODEL}

# Compute the step number based on the array index
STEP=$((1000 * SLURM_ARRAY_TASK_ID))

# Construct the model path
MODEL_PATH=${BASE_PATH}/step_${STEP}
MODEL_BASENAME=$(basename $MODEL_PATH)

# TODO: arguments are not quite correct yet! FIX
accelerate launch --config_file accelerate_1gpu_config.yaml --num_cpu_threads_per_process 16 /scratch/eo41/babylm/dpo.py \
    --model_name_or_path $MODEL_PATH \
    --dataset_name trl-internal-testing/hh-rlhf-helpful-base-trl-style \
    --per_device_train_batch_size 6 \
    --gradient_accumulation_steps 16 \
    --learning_rate 0.0001 \
    --output_dir /scratch/projects/lakelab/dpo_hh/${BASE_MODEL}/${MODEL_BASENAME} \
    --num_train_epochs 1 \
    --logging_steps 10 \
    --save_steps 10 \
    --eval_steps 10 \
    --max_length 1024 \
    --max_prompt_length 512 \
    --logging_first_step \
    --no_remove_unused_columns \
    --bf16 \
    --warmup_steps 50

echo "Done"