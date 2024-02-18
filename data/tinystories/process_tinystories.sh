#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32GB
#SBATCH --time=1:00:00
#SBATCH --job-name=process_tinystories
#SBATCH --output=process_tinystories_videos_%A_%a.out

srun python -u process_tinystories.py 

echo "Done"