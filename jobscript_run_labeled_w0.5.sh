#!/bin/bash
#SBATCH -A p31796
#SBATCH -p normal
#SBATCH -t 6:00:00
#SBATCH -N 1
#SBATCH --mem=30G
#SBATCH --output=training_out.txt
#SBATCH --error=training_err.txt

source quest_decoding_venv/bin/activate
module load python/3.9.16
cd neural_decoding/

python3 run/run.py configs/generalizability_configs/regularization/labeled_w0.5.yaml
