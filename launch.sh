#!/bin/bash
#SBATCH --job-name=lofgan
#SBATCH --output=logs/slurm-%j.txt
#SBATCH --open-mode=append
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --partition=rtx6000,t4v2
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=25GB
#SBATCH --exclude=gpu109

dataset=$1

python train.py --conf configs/${dataset}_lofgan.yaml \
--output_dir results/${dataset}_lofgan \
--gpu 0