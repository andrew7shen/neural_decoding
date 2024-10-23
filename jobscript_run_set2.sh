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


python3 run/run.py configs/temp_configs/configs_cage_t100_set2_temp50.0.yaml
# python3 run/run.py configs/t100_configs/configs_cage_t100_set2_d6.yaml
# python3 run/run.py configs/temp_configs/configs_cage_t100_set2_temp50.0_750epochs_flatten_d6.yaml

# python3 run/run.py configs/t100_configs/configs_cage_t100_set2.yaml
# python3 run/run.py configs/temp_configs/configs_cage_t100_set2_temp1.0.yaml
# python3 run/run.py configs/temp_configs/configs_cage_t100_set2_temp5.0.yaml
# python3 run/run.py configs/temp_configs/configs_cage_t100_set2_temp10.0.yaml
# python3 run/run.py configs/temp_configs/configs_cage_t100_set2_temp50.0.yaml
# python3 run/run.py configs/temp_configs/configs_cage_t100_set2_temp100.0.yaml

# python3 run/run.py configs/temp_configs/configs_cage_t100_set2_temp50.0_850epochs_flatten.yaml
# python3 run/run.py configs/temp_configs/configs_cage_t100_set2_temp50.0_1000epochs_flatten.yaml
# python3 run/run.py configs/temp_configs/configs_cage_t100_set2_temp50.0_1250epochs_flatten.yaml
# python3 run/run.py configs/temp_configs/configs_cage_t100_set2_temp50.0_1500epochs_flatten.yaml