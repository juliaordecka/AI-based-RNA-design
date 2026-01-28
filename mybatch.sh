#!/bin/bash
#SBATCH -p hgx
#SBATCH -n1
#SBATCH --gres=gpu:1
#SBATCH --time=60:00:00
#SBATCH --output=slurm-%j.out

source ~/miniconda3/etc/profile.d/conda.sh
conda activate PYTORCH

export PYTHONUNBUFFERED=1

python -u run_wgan.py --data data/archiveii_fixed_train_more_than_50.fasta --epochs 400
