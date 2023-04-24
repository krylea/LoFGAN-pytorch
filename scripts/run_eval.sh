#!/usr/bin/env bash
#SBATCH --job-name=lofgan-eval
#SBATCH --output=logs/slurm-%j.txt
#SBATCH --open-mode=append
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --partition=rtx6000,t4v2
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=25GB
#SBATCH --exclude=gpu109

N_EXPS=3

dataset=$1
n=$2

python main_metric.py --gpu 0 --dataset $dataset \
--name results/${dataset}_lofgan \
--real_dir datasets/for_fid/$dataset --ckpt gen_00100000.pt \
--fake_dir test_for_fid --n_cond $n --n_exps $N_EXPS