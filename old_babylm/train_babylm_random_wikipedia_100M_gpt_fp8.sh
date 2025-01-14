#!/bin/bash

#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=62GB
#SBATCH --time=00:10:00
#SBATCH --job-name=train_babylm_random_wikipedia_100M_gpt_fp8
#SBATCH --output=train_babylm_random_wikipedia_100M_gpt_fp8_%A_%a.out
#SBATCH --array=0

# path to ms-amp singularity image
SINGULARITY_IMAGE="/home/eo41/.singularity/cache/oci/msamp_main-cuda12.2.sif"

# bind paths
BIND_PATHS=(
    "/vast/eo41:/vast/eo41"
    "/scratch/eo41:/scratch/eo41"
    "/scratch/eo41/babylm:/scratch/eo41/babylm"
    "/vast/eo41/babylm/models:/vast/eo41/babylm/models"
)

export HF_HOME="/vast/eo41/huggingface"
export HF_DATASETS_CACHE="/vast/eo41/huggingface"

# root model directory
MODEL_ROOT_DIR="/vast/eo41/babylm/models"
SP="babylm_random_wikipedia_100M_1_gpt"

# construct the bind string for singularity exec
BIND_STRING=""
for bind_path in "${BIND_PATHS[@]}"; do
    BIND_STRING+=" --bind $bind_path"
done

# source /opt/conda/etc/profile.d/conda.sh; \
# use singularity exec to run the command inside the container
singularity exec \
    --nv \
    $BIND_STRING \ # Add the bind paths to the command
    "$SINGULARITY_IMAGE" \
    bash -c " \
        export HF_HOME=\"/vast/eo41/huggingface\"; \
        export HF_DATASETS_CACHE=\"/vast/eo41/huggingface\"; \
        accelerate launch --config_file /scratch/eo41/babylm/accelerate_1gpu_config_fp8.yaml --num_cpu_threads_per_process 16 /scratch/eo41/babylm/train.py \
            --model_name_or_path \"meta-llama/Llama-3.2-1B\" \
            --per_device_train_batch_size 8 \
            --gradient_accumulation_steps 4 \
            --learning_rate 0.0001 \
            --output_dir \"${MODEL_ROOT_DIR}/${SP}\" \
            --save_prefix ${SP} \
            --block_size 1024 \
            --num_train_epochs 20 \
            --checkpointing_steps 1000 \
            --overwrite_cache \
    "

echo "Done"