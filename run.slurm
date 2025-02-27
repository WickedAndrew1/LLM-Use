#!/bin/bash

#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --job-name=python-gpu
#SBATCH --output=/scratch/%u/%j.out
#SBATCH --error=/scratch/%u/%j.err
#SBATCH --mail-user=aalam24@gmu.edu
#SBATCH --mail-type=END
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --mem-per-cpu=64G
#SBATCH --export=ALL
#SBATCH --time=0-08:00:00


module purge
module load ohpc
module load gnu10
# module load python/3.9.9-jh
# module load miniconda3/22.11.1-gy

# Activate Conda environment
source /home/aalam24/miniforge/bin/activate
conda activate testingenv

# Check to make sure environment and GPU are working
conda info --envs
python --version
nvidia-smi

# Navigate to the directory containing your script
cd /scratch/aalam24

# Run your Python script
python3 testfile3.py