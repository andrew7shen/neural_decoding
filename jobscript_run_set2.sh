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

# python3 run/run.py configs/t100_configs/configs_cage_t100_set2.yaml
# python3 run/run.py configs/t100_configs/configs_cage_t100_set2_l1.yaml
# python3 run/run.py configs/t100_configs/configs_cage_t100_set2_l1_0.05.yaml
# python3 run/run.py configs/t100_configs/configs_cage_t100_set2_l1_0.1.yaml
# python3 run/run.py configs/t100_configs/configs_cage_t100_set2_l1_0.5.yaml

python3 run/run.py configs/t100_configs/configs_cage_t100_set2_l1_0.075.yaml
python3 run/run.py configs/t100_configs/configs_cage_t100_set2_l1_0.125.yaml
python3 run/run.py configs/t100_configs/configs_cage_t100_set2_l1_0.15.yaml
python3 run/run.py configs/t100_configs/configs_cage_t100_set2_l1_0.175.yaml
python3 run/run.py configs/t100_configs/configs_cage_t100_set2_l1_0.2.yaml

python3 run/run.py configs/t100_configs/configs_cage_t100_set2_gb_l1_0.05.yaml
python3 run/run.py configs/t100_configs/configs_cage_t100_set2_gb_l1_0.1.yaml
python3 run/run.py configs/t100_configs/configs_cage_t100_set2_gb_l1_0.15.yaml

# python3 run/run.py configs/t100_configs/configs_cage_t100_set2_globalbias.yaml

# python3 run/run.py configs/temp_configs/configs_cage_t100_set2_temp50.0_lambda0.yaml
# python3 run/run.py configs/temp_configs/configs_cage_t100_set2_temp50.0_lambda0.001.yaml
# python3 run/run.py configs/temp_configs/configs_cage_t100_set2_temp50.0_lambda0.005.yaml
# python3 run/run.py configs/temp_configs/configs_cage_t100_set2_temp50.0_lambda0.01.yaml
# python3 run/run.py configs/temp_configs/configs_cage_t100_set2_temp50.0_lambda0.025.yaml
# python3 run/run.py configs/temp_configs/configs_cage_t100_set2_temp50.0_lambda0.05.yaml
# python3 run/run.py configs/temp_configs/configs_cage_t100_set2_temp50.0_lambda0.1.yaml

# python3 run/run.py configs/temp_configs/configs_cage_t100_set2_temp50.0_500epochs_flatten.yaml
# python3 run/run.py configs/temp_configs/configs_cage_t100_set2_temp50.0_1250epochs_flatten.yaml

# python3 run/run.py configs/temp_configs/configs_cage_t100_set2_temp50.0_endtemp0.01.yaml
# python3 run/run.py configs/temp_configs/configs_cage_t100_set2_temp50.0_endtemp0.05.yaml
# python3 run/run.py configs/temp_configs/configs_cage_t100_set2_temp50.0_endtemp0.1.yaml
# python3 run/run.py configs/temp_configs/configs_cage_t100_set2_temp50.0_endtemp0.2.yaml
# python3 run/run.py configs/temp_configs/configs_cage_t100_set2_temp50.0_endtemp0.3.yaml
# python3 run/run.py configs/temp_configs/configs_cage_t100_set2_temp50.0_endtemp0.4.yaml
# python3 run/run.py configs/temp_configs/configs_cage_t100_set2_temp50.0_endtemp0.5.yaml
# python3 run/run.py configs/temp_configs/configs_cage_t100_set2_temp50.0_endtemp0.75.yaml
# python3 run/run.py configs/temp_configs/configs_cage_t100_set2_temp50.0_endtemp1.yaml
# python3 run/run.py configs/temp_configs/configs_cage_t100_set2_temp50.0_endtemp1.5.yaml


# python3 run/run.py configs/temp_configs/configs_cage_t100_set2_temp50.0_lr0.00025.yaml
# python3 run/run.py configs/temp_configs/configs_cage_t100_set2_temp50.0_lr0.0005.yaml
# python3 run/run.py configs/temp_configs/configs_cage_t100_set2_temp50.0_lr0.001.yaml
# python3 run/run.py configs/temp_configs/configs_cage_t100_set2_temp50.0_lr0.0025.yaml
# python3 run/run.py configs/temp_configs/configs_cage_t100_set2_temp50.0_lr0.005.yaml
# python3 run/run.py configs/temp_configs/configs_cage_t100_set2_temp50.0_lr0.01.yaml

# python3 run/run.py configs/temp_configs/configs_cage_t100_set2_temp50.0_lr0.001_sv.yaml
# python3 run/run.py configs/temp_configs/configs_cage_t100_set2_temp50.0_lr0.001_svb.yaml
# python3 run/run.py configs/temp_configs/configs_cage_t100_set2_temp50.0_lr0.001_svi.yaml
# python3 run/run.py configs/temp_configs/configs_cage_t100_set2_temp50.0_lr0.001_svbi.yaml

# python3 run/run.py configs/temp_configs/configs_cage_t100_set2_temp50.0.yaml
# python3 run/run.py configs/temp_configs/configs_cage_t100_set2_temp60.0.yaml
# python3 run/run.py configs/temp_configs/configs_cage_t100_set2_temp70.0.yaml
# python3 run/run.py configs/temp_configs/configs_cage_t100_set2_temp80.0.yaml
# python3 run/run.py configs/temp_configs/configs_cage_t100_set2_temp90.0.yaml
# python3 run/run.py configs/temp_configs/configs_cage_t100_set2_temp100.0.yaml

# python3 run/run.py configs/temp_configs/configs_cage_t100_set2_temp50.0_endtemp0.01.yaml
# python3 run/run.py configs/temp_configs/configs_cage_t100_set2_temp60.0_endtemp0.01.yaml
# python3 run/run.py configs/temp_configs/configs_cage_t100_set2_temp70.0_endtemp0.01.yaml
# python3 run/run.py configs/temp_configs/configs_cage_t100_set2_temp80.0_endtemp0.01.yaml
# python3 run/run.py configs/temp_configs/configs_cage_t100_set2_temp90.0_endtemp0.01.yaml
# python3 run/run.py configs/temp_configs/configs_cage_t100_set2_temp100.0_endtemp0.01.yaml


# python3 run/run.py configs/temp_configs/configs_cage_t100_set2_temp50.0_d6.yaml
# python3 run/run.py configs/temp_configs/configs_cage_t100_set2_temp60.0_d6.yaml
# python3 run/run.py configs/temp_configs/configs_cage_t100_set2_temp70.0_d6.yaml
# python3 run/run.py configs/temp_configs/configs_cage_t100_set2_temp80.0_d6.yaml
# python3 run/run.py configs/temp_configs/configs_cage_t100_set2_temp90.0_d6.yaml
# python3 run/run.py configs/temp_configs/configs_cage_t100_set2_temp100.0_d6.yaml