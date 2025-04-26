#!/bin/bash
#SBATCH -A p31796
#SBATCH -p normal
#SBATCH -t 12:00:00
#SBATCH -N 1
#SBATCH --mem=30G
#SBATCH --output=evaluate_mouse_out.txt
#SBATCH --error=evaluate_mouse_err.txt

source quest_decoding_venv/bin/activate
module load python/3.9.16
cd neural_decoding/

python3 eval/evaluate.py configs/mouse_configs/configs_mouse_gb_l1_0.yaml