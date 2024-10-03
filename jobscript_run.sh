#!/bin/bash
#SBATCH -A p31796
#SBATCH -p normal
#SBATCH -t 2:00:00
#SBATCH -N 1
#SBATCH --mem=30G
#SBATCH --output=training_out.txt
#SBATCH --error=training_err.txt

source quest_decoding_venv/bin/activate
module load python/3.9.16
cd neural_decoding/

# python3 run/run.py configs/generalizability_configs/regularization/unlabeled_w0.05.yaml
# python3 run/run.py configs/generalizability_configs/regularization/unlabeled_w0.1.yaml
# python3 run/run.py configs/generalizability_configs/regularization/unlabeled_w0.5.yaml
# python3 run/run.py configs/generalizability_configs/regularization/labeled_w0.01.yaml
# python3 run/run.py configs/generalizability_configs/regularization/labeled_w0.05.yaml
# python3 run/run.py configs/generalizability_configs/regularization/labeled_w0.1.yaml
# python3 run/run.py configs/generalizability_configs/regularization/labeled_w0.5.yaml

# python3 run/run.py configs/generalizability_configs/regularization/unlabeled_w0.02.yaml
# python3 run/run.py configs/generalizability_configs/regularization/unlabeled_w0.03.yaml
# python3 run/run.py configs/generalizability_configs/regularization/unlabeled_w0.04.yaml
# python3 run/run.py configs/generalizability_configs/regularization/unlabeled_w0.05.yaml
# python3 run/run.py configs/generalizability_configs/regularization/unlabeled_w0.06.yaml
# python3 run/run.py configs/generalizability_configs/regularization/unlabeled_w0.07.yaml
# python3 run/run.py configs/generalizability_configs/regularization/unlabeled_w0.08.yaml
# python3 run/run.py configs/generalizability_configs/regularization/unlabeled_w0.09.yaml



# python3 run/run.py configs/generalizability_configs/regularization/unlabeled_d6_w0.01.yaml
# python3 run/run.py configs/generalizability_configs/regularization/unlabeled_d6_w0.05.yaml
# python3 run/run.py configs/generalizability_configs/regularization/unlabeled_d6_w0.1.yaml
# python3 run/run.py configs/generalizability_configs/regularization/unlabeled_d6_w0.5.yaml

# python3 run/run.py configs/generalizability_configs/regularization/labeled_d6_w0.01.yaml
# python3 run/run.py configs/generalizability_configs/regularization/labeled_d6_w0.05.yaml
# python3 run/run.py configs/generalizability_configs/regularization/labeled_d6_w0.1.yaml
# python3 run/run.py configs/generalizability_configs/regularization/labeled_d6_w0.5.yaml




# python3 run/run.py configs/t100_configs/configs_cage_t100_none.yaml
# python3 run/run.py configs/t100_configs/configs_cage_t100_none_scaled.yaml

# python3 run/run.py configs/s_configs/configs_cage_t100_set1_s42.yaml
# python3 run/run.py configs/s_configs/configs_cage_t100_set1_s43.yaml
# python3 run/run.py configs/s_configs/configs_cage_t100_set1_s44.yaml
# python3 run/run.py configs/s_configs/configs_cage_t100_set1_s45.yaml
# python3 run/run.py configs/s_configs/configs_cage_t100_set1_s46.yaml

# python3 run/run.py configs/t100_configs/configs_cage_t100_set1_kmeansweights.yaml
# python3 run/run.py configs/t100_configs/configs_cage_t100_set1_randomweights.yaml

python3 run/run.py configs/generalizability_configs/configs_cage_t100_set1_generalizability_grooming.yaml
python3 run/run.py configs/generalizability_configs/configs_cage_t100_set1_generalizability_grooming_sitting_still.yaml
# python3 run/run.py configs/generalizability_configs/configs_cage_t100_set1_generalizability_unlabeled.yaml
# python3 run/run.py configs/generalizability_configs/configs_cage_t100_set1_generalizability_labeled.yaml

# python3 run/run.py configs/generalizability_configs/configs_cage_t100_set1_generalizability_grooming_d6.yaml
# python3 run/run.py configs/generalizability_configs/configs_cage_t100_set1_generalizability_grooming_sitting_still_d6.yaml
# python3 run/run.py configs/generalizability_configs/configs_cage_t100_set1_generalizability_unlabeled_d6.yaml
# python3 run/run.py configs/generalizability_configs/configs_cage_t100_set1_generalizability_labeled_d6.yaml

python3 run/run.py configs/generalizability_configs/configs_cage_t100_set1_generalizability_grooming_robust.yaml
python3 run/run.py configs/generalizability_configs/configs_cage_t100_set1_generalizability_grooming_sitting_still_robust.yaml
# python3 run/run.py configs/generalizability_configs/configs_cage_t100_set1_generalizability_unlabeled_robust.yaml
# python3 run/run.py configs/generalizability_configs/configs_cage_t100_set1_generalizability_labeled_robust.yaml

# python3 run/run.py configs/generalizability_configs/regularization/grooming_w0.005.yaml
# python3 run/run.py configs/generalizability_configs/regularization/grooming_w0.01.yaml
# python3 run/run.py configs/generalizability_configs/regularization/grooming_w0.025.yaml
# python3 run/run.py configs/generalizability_configs/regularization/grooming_w0.05.yaml
# python3 run/run.py configs/generalizability_configs/regularization/grooming_w0.1.yaml
# python3 run/run.py configs/generalizability_configs/regularization/grooming_w0.25.yaml
# python3 run/run.py configs/generalizability_configs/regularization/grooming_w0.5.yaml

# python3 run/run.py configs/generalizability_configs/regularization/grooming_sitting_still_w0.005.yaml
# python3 run/run.py configs/generalizability_configs/regularization/grooming_sitting_still_w0.01.yaml
# python3 run/run.py configs/generalizability_configs/regularization/grooming_sitting_still_w0.025.yaml
# python3 run/run.py configs/generalizability_configs/regularization/grooming_sitting_still_w0.05.yaml
# python3 run/run.py configs/generalizability_configs/regularization/grooming_sitting_still_w0.1.yaml
# python3 run/run.py configs/generalizability_configs/regularization/grooming_sitting_still_w0.25.yaml
# python3 run/run.py configs/generalizability_configs/regularization/grooming_sitting_still_w0.5.yaml


# python3 run/run.py configs/mouse_configs/configs_mouse.yaml









# python3 run/run.py configs/robust_configs/configs_cage_t100_relu0.1.yaml
# python3 run/run.py configs/robust_configs/configs_cage_t100_relu0.01.yaml
# python3 run/run.py configs/robust_configs/configs_cage_t100_onlyrelu0.01.yaml
# python3 run/run.py configs/robust_configs/configs_cage_t100_onlytanh.yaml

# python3 run/run.py configs/d_configs/configs_cage_t100_set1_d3.yaml
# python3 run/run.py configs/d_configs/configs_cage_t100_set1_d4.yaml
# python3 run/run.py configs/d_configs/configs_cage_t100_set1_d5.yaml
# python3 run/run.py configs/d_configs/configs_cage_t100_set1_d6.yaml
# python3 run/run.py configs/d_configs/configs_cage_t100_set1_d7.yaml

# python3 run/run.py configs/t100_configs/configs_cage_t100_set1.yaml
# python3 run/run.py configs/t100_configs/configs_cage_t100_none.yaml
# python3 run/run.py configs/d_configs/configs_cage_t100_none_d6.yaml

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