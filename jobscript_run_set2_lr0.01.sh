#!/bin/bash
#SBATCH -A p31796
#SBATCH -p normal
#SBATCH -t 2:00:00
#SBATCH -N 1
#SBATCH --mem=30G
#SBATCH --output=training_lr0.01_out.txt
#SBATCH --error=training_lr0.01_err.txt

source quest_decoding_venv/bin/activate
module load python/3.9.16
cd neural_decoding/


python3 run/run.py configs/temp_configs/configs_cage_t100_set2_temp50.0_lr0.01.yaml