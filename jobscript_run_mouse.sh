#!/bin/bash
#SBATCH -A p31796
#SBATCH -p normal
#SBATCH -t 12:00:00
#SBATCH -N 1
#SBATCH --mem=30G
#SBATCH --output=training_mouse_out.txt
#SBATCH --error=training_mouse_err.txt

source quest_decoding_venv/bin/activate
module load python/3.9.16
cd neural_decoding/

python3 run/run.py configs/mouse_configs/configs_mouse_gb_l1_0.yaml
python3 run/run.py configs/mouse_configs/configs_mouse_gb_l1_0.01.yaml
python3 run/run.py configs/mouse_configs/configs_mouse_gb_l1_0.02.yaml
python3 run/run.py configs/mouse_configs/configs_mouse_gb_l1_0.03.yaml
python3 run/run.py configs/mouse_configs/configs_mouse_gb_l1_0.04.yaml
python3 run/run.py configs/mouse_configs/configs_mouse_gb_l1_0.05.yaml

# python3 run/run.py configs/mouse_configs/configs_mouse_baseline.yaml
# python3 run/run.py configs/mouse_configs/configs_mouse_temp1.0.yaml
# python3 run/run.py configs/mouse_configs/configs_mouse_temp50.0.yaml

# python3 run/run.py configs/mouse_configs/configs_mouse_temp50.0.yaml
# python3 run/run.py configs/mouse_configs/configs_mouse_temp40.0.yaml
# python3 run/run.py configs/mouse_configs/configs_mouse_temp30.0.yaml
# python3 run/run.py configs/mouse_configs/configs_mouse_temp20.0.yaml
# python3 run/run.py configs/mouse_configs/configs_mouse_temp10.0.yaml
# python3 run/run.py configs/mouse_configs/configs_mouse_temp5.0.yaml

# python3 run/run.py configs/mouse_configs/configs_mouse_temp50.0.yaml
# python3 run/run.py configs/mouse_configs/configs_mouse_temp60.0.yaml
# python3 run/run.py configs/mouse_configs/configs_mouse_temp70.0.yaml
# python3 run/run.py configs/mouse_configs/configs_mouse_temp80.0.yaml
# python3 run/run.py configs/mouse_configs/configs_mouse_temp90.0.yaml
# python3 run/run.py configs/mouse_configs/configs_mouse_temp100.0.yaml