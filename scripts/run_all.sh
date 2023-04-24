#!/bin/bash

NREF=(1 5 10 30)

dataset=$1

for nref in "${NREF[@]}"
do
    sbatch scripts/run_eval.sh $dataset $nref
done