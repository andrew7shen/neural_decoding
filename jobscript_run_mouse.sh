#!/bin/bash
#SBATCH -A p31796
#SBATCH -p normal
#SBATCH -t 1-00:00
#SBATCH -N 1
#SBATCH --mem=30G
#SBATCH --output=training_mouse_out.txt
#SBATCH --error=training_mouse_err.txt

source quest_decoding_venv/bin/activate
module load python/3.9.16
cd neural_decoding/

python3 run/run.py configs/mouse_configs/configs_mouse_lr0.025.yaml
python3 run/run.py configs/mouse_configs/configs_mouse_lr0.05.yaml
python3 run/run.py configs/mouse_configs/configs_mouse_lr0.1.yaml