#!/bin/bash
#SBATCH --job-name=LRU_PQNBSM
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16        #maximum cpu limit for each v100 GPU is 6 , each a100 GPU is 8
#SBATCH --gres=gpu:1
#SBATCH --mem=40G      #maximum memory limit for each v100 GPU is 90G , each a100 GPU is 40G
#SBATCH --output=./output/output_PQNLRUBattleShipMedium.txt
#SBATCH --partition=a100_batch    #default batch is a100 

source ~/.bashrc
source activate jaxenv           #need change 'myenv' to your environment
