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

# python3 run/run.py configs/t100_configs/configs_cage_t100.yaml
# python3 run/run.py configs/t100_configs/configs_cage_t100_2.yaml
# python3 run/run.py configs/b_configs/configs_cage_t100_b10.yaml
# python3 run/run.py configs/b_configs/configs_cage_t100_b10_2.yaml

python3 run/run.py configs/b_configs/configs_cage_t100_b10_w0.05.yaml
python3 run/run.py configs/b_configs/configs_cage_t100_b10_w0.04.yaml
python3 run/run.py configs/b_configs/configs_cage_t100_b10_w0.03.yaml
python3 run/run.py configs/b_configs/configs_cage_t100_b10_w0.05_2.yaml
python3 run/run.py configs/b_configs/configs_cage_t100_b10_w0.04_2.yaml
python3 run/run.py configs/b_configs/configs_cage_t100_b10_w0.03_2.yaml