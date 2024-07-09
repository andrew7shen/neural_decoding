#!/bin/bash
#SBATCH -A p31796
#SBATCH -p normal
#SBATCH -t 3:00:00
#SBATCH -N 1
#SBATCH --mem=30G
#SBATCH --output=training_out.txt
#SBATCH --error=training_err.txt

source quest_decoding_venv/bin/activate
module load python/3.9.16
cd neural_decoding/

python3 run/run.py configs/t100_configs/configs_cage_t100_set1.yaml

# python3 run/run.py configs/t100_configs/configs_cage_t100.yaml
# python3 run/run.py configs/t100_configs/configs_cage_t100_2.yaml
# python3 run/run.py configs/b_configs/configs_cage_t100_b10.yaml
# python3 run/run.py configs/b_configs/configs_cage_t100_b10_2.yaml

# python3 run/run.py configs/w_configs/configs_cage_t100_w0.1.yaml
# python3 run/run.py configs/w_configs/configs_cage_t100_w0.05.yaml
# python3 run/run.py configs/w_configs/configs_cage_t100_w0.025.yaml
# python3 run/run.py configs/w_configs/configs_cage_t100_w0.01.yaml
# python3 run/run.py configs/w_configs/configs_cage_t100_w0.1_2.yaml
# python3 run/run.py configs/w_configs/configs_cage_t100_w0.05_2.yaml
# python3 run/run.py configs/w_configs/configs_cage_t100_w0.025_2.yaml
# python3 run/run.py configs/w_configs/configs_cage_t100_w0.01_2.yaml
# python3 run/run.py configs/b_configs/configs_cage_t100_b10_w0.1.yaml
# python3 run/run.py configs/b_configs/configs_cage_t100_b10_w0.05.yaml
# python3 run/run.py configs/b_configs/configs_cage_t100_b10_w0.025.yaml
# python3 run/run.py configs/b_configs/configs_cage_t100_b10_w0.01.yaml
# python3 run/run.py configs/b_configs/configs_cage_t100_b10_w0.1_2.yaml
# python3 run/run.py configs/b_configs/configs_cage_t100_b10_w0.05_2.yaml
# python3 run/run.py configs/b_configs/configs_cage_t100_b10_w0.025_2.yaml
# python3 run/run.py configs/b_configs/configs_cage_t100_b10_w0.01_2.yaml

# python3 run/run.py configs/b_configs/configs_cage_t100_b10_w0.6.yaml
# python3 run/run.py configs/b_configs/configs_cage_t100_b10_w0.4.yaml
# python3 run/run.py configs/b_configs/configs_cage_t100_b10_w0.2.yaml
# python3 run/run.py configs/b_configs/configs_cage_t100_b10_w0.15.yaml
# python3 run/run.py configs/b_configs/configs_cage_t100_b10_w0.6_2.yaml
# python3 run/run.py configs/b_configs/configs_cage_t100_b10_w0.4_2.yaml
# python3 run/run.py configs/b_configs/configs_cage_t100_b10_w0.2_2.yaml
# python3 run/run.py configs/b_configs/configs_cage_t100_b10_w0.15_2.yaml