#!/bin/bash

#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=240GB
#SBATCH --time=02:00:00
#SBATCH --job-name=qfilter_gpt
#SBATCH --output=qfilter_gpt_%A_%a.out
#SBATCH --array=0

export HF_HOME="/vast/eo41/huggingface"
export HF_DATASETS_CACHE="/vast/eo41/huggingface"

# root model directory
MODEL_ROOT_DIR="/vast/eo41/babylm/data"
SP="wiki"

accelerate launch --config_file accelerate_1gpu_config.yaml --num_cpu_threads_per_process 16 /scratch/eo41/babylm/qfilter.py \
    --model_name_or_path "gpt2-large" \
    --data_files "wikimedia/wikipedia" "20231101.en" \
    --per_device_eval_batch_size 32 \
    --output_dir "${MODEL_ROOT_DIR}/${SP}" \
    --save_prefix ${SP} \
    --block_size 1024 \
    --overwrite_cache \
    --use_pretrained_weights

echo "Done"