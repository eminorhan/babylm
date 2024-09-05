#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB
#SBATCH --time=3:00:00
#SBATCH --job-name=create_babylm_dataset
#SBATCH --output=create_babylm_dataset_%A_%a.out

export HF_HOME="/vast/eo41/huggingface"
export HF_DATASETS_CACHE="/vast/eo41/huggingface"

srun python -u create_babylm_dataset.py 

echo "Done"