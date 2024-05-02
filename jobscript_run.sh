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

python3 run/run.py configs/kmeans_split_configs/configs_cage_t100_kmeans0.yaml
python3 run/run.py configs/kmeans_split_configs/configs_cage_t100_kmeans1.yaml
python3 run/run.py configs/kmeans_split_configs/configs_cage_t100_kmeans2.yaml

# python3 run/run.py configs/t100_configs/configs_cage_t100.yaml

# python3 run/run.py configs/h_configs/configs_cage_t100_h20_method2.yaml
# python3 run/run.py configs/h_configs/configs_cage_t100_h20_method3.yaml
# python3 run/run.py configs/t100_configs/configs_cage_t100.yaml
# python3 run/run.py configs/h_configs/configs_cage_t100_h20_w1.5.yaml
# python3 run/run.py configs/h_configs/configs_cage_t100_h20_w1.yaml
# python3 run/run.py configs/h_configs/configs_cage_t100_h20_bt16.yaml
# python3 run/run.py configs/h_configs/configs_cage_t100_h20_bt32.yaml
# python3 run/run.py configs/h_configs/configs_cage_t100_b10_h200_w1.5.yaml
# python3 run/run.py configs/h_configs/configs_cage_t100_b10_h200_w1.yaml
# python3 run/run.py configs/h_configs/configs_cage_t100_b10_h200_bt16.yaml
# python3 run/run.py configs/h_configs/configs_cage_t100_b10_h200_bt32.yaml
