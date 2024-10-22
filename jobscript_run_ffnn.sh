#!/bin/bash
#SBATCH -A p31796
#SBATCH -p normal
#SBATCH -t 3:00:00
#SBATCH -N 1
#SBATCH --mem=30G
#SBATCH --output=ffnn_out.txt
#SBATCH --error=ffnn_err.txt

source quest_decoding_venv/bin/activate
module load python/3.9.16
cd neural_decoding/

python3 eval/evaluate.py configs/mouse_configs/configs_mouse_baseline.yaml
