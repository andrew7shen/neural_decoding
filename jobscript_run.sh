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
python3 run/run.py configs/d_configs/configs_cage_t100_d3.yaml
python3 run/run.py configs/d_configs/configs_cage_t100_d4.yaml
python3 run/run.py configs/d_configs/configs_cage_t100_d5.yaml
python3 run/run.py configs/d_configs/configs_cage_t100_d6.yaml
python3 run/run.py configs/d_configs/configs_cage_t100_d7.yaml
python3 run/run.py configs/d_configs/configs_cage_t100_b10_d3.yaml
python3 run/run.py configs/d_configs/configs_cage_t100_b10_d4.yaml
python3 run/run.py configs/d_configs/configs_cage_t100_b10_d5.yaml
python3 run/run.py configs/d_configs/configs_cage_t100_b10_d6.yaml
python3 run/run.py configs/d_configs/configs_cage_t100_b10_d7.yaml
