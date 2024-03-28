#!/bin/bash
#SBATCH -A p31796
#SBATCH -p normal
#SBATCH -t 2-00:00
#SBATCH -N 1
#SBATCH --mem=30G
#SBATCH --output=training_out.txt
#SBATCH --error=training_err.txt

source quest_decoding_venv/bin/activate
module load python/3.9.16
cd neural_decoding/

python3 run/run.py configs/h_configs/configs_cage_t100_h95.yaml
python3 run/run.py configs/h_configs/configs_cage_t100_h50.yaml
python3 run/run.py configs/h_configs/configs_cage_t100_h20.yaml
python3 run/run.py configs/h_configs/configs_cage_t100_b10_h950.yaml
python3 run/run.py configs/h_configs/configs_cage_t100_b10_h500.yaml
python3 run/run.py configs/h_configs/configs_cage_t100_b10_h200.yaml
