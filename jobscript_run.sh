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

python3 run/run.py configs/b_configs/configs_cage_t100_b10_w0.5.yaml
python3 run/run.py configs/b_configs/configs_cage_t100_b10_w0.1.yaml
python3 run/run.py configs/b_configs/configs_cage_t100_b10_w0.05.yaml
python3 run/run.py configs/b_configs/configs_cage_t100_b10_w0.01.yaml
python3 run/run.py configs/b_configs/configs_cage_t100_b10_w0.5_2.yaml
python3 run/run.py configs/b_configs/configs_cage_t100_b10_w0.1_2.yaml
python3 run/run.py configs/b_configs/configs_cage_t100_b10_w0.05_2.yaml
python3 run/run.py configs/b_configs/configs_cage_t100_b10_w0.01_2.yaml