#!/bin/bash
#SBATCH --job-name="Geoguessr"
#SBATCH --partition=gpu-a100
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus-per-task=1
#SBATCH --account=education-eemcs-msc-cs

# Load modules:
module load 2023r1
module load openmpi
module load miniconda3
module load openssh

unset CONDA_SHLVL
source "$(conda info --base)/etc/profile.d/conda.sh"

conda create -n "DL-Geoguessr" python=3.11
conda activate DL-Geoguessr

conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
conda install -c conda-forge transformers
conda install pandas
conda install numpy
conda install -c conda-forge tqdm

export HF_HOME="/scratch/eaydoslu/.cache"

srun python geoguessr.py --annotation_path data/annotations_unglued.csv --img_dir data/images