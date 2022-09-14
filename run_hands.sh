#!/bin/bash -eux

#SBATCH --job-name=hands-two_sample_test

#SBATCH --mail-type=END,FAIL

#SBATCH --mail-user=juliana.schneider@hpi.de

#SBATCH --partition=gpu

#SBATCH --cpus-per-task=8 # TODO here

#SBATCH --mem=12gb # TODO here

#SBATCH --time=2:00:00

#SBATCH --gpus=1

#SBATCH --array=0-11%4 # TODO here --> this means 12 jobs are started, at most 4 in parallel

#SBATCH -o logging_results.log

#eval "$(conda shell.bash hook)"
#conda activate conda_hands2
source venv.sh

python -m experiments

