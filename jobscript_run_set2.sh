#!/bin/bash
#SBATCH -A p31796
#SBATCH -p normal
#SBATCH -t 12:00:00
#SBATCH -N 1
#SBATCH --mem=30G
#SBATCH --output=training_out.txt
#SBATCH --error=training_err.txt

source quest_decoding_venv/bin/activate
module load python/3.9.16
cd neural_decoding/

python3 run/run.py configs/temp_configs/configs_cage_t100_set2_temp0.01.yaml
python3 run/run.py configs/temp_configs/configs_cage_t100_set2_temp0.25.yaml
python3 run/run.py configs/temp_configs/configs_cage_t100_set2_temp0.5.yaml
python3 run/run.py configs/temp_configs/configs_cage_t100_set2_temp0.75.yaml
python3 run/run.py configs/temp_configs/configs_cage_t100_set2_temp1.0.yaml
