#!/bin/bash
#SBATCH -J Monte-Carlo
#SBATCH -A cs475-575
#SBATCH -p class
#SBATCH --gres=gpu:1
#SBATCH -e montecarlo.csv
nvcc --std=c++11 -o montecarlo montecarlo.cu
for block in 16 32 64 128; do
    for trials in 2048 4096 8192 16384 32768 65536 131072 262144 524288 1048576 2097152 4194304 8388608 16777216 33554432 67108864; do
        ./montecarlo $block $trials
    done
done
