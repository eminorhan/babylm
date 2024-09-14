#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB
#SBATCH --time=2:00:00
#SBATCH --job-name=create_random_wikipedia
#SBATCH --output=create_random_wikipedia_%A_%a.out

export HF_HOME="/vast/eo41/huggingface"
export HF_DATASETS_CACHE="/vast/eo41/huggingface"

srun python -u create_random_wikipedia.py 

echo "Done"