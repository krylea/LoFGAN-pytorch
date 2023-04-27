#!/bin/bash

NREF=(1 5 10 30)

dataset=$1
use_modified_datasets=${2:-0}

for nref in "${NREF[@]}"
do
    sbatch scripts/run_eval.sh $dataset $nref $use_modified_datasets
done