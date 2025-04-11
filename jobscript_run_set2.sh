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

# python3 run/run.py configs/t100_configs/configs_cage_t100_set2_gb_l1_0.yaml
# python3 run/run.py configs/t100_configs/configs_cage_t100_set2_gb_l1_tanhscalingoffset_0.yaml

# Temporarily use this file to submit mouse runs also
# python3 run/run.py configs/mouse_configs/configs_mouse_gb_l1_0.yaml
# python3 run/run.py configs/mouse_configs/configs_mouse_gb_l1_tanhscalingoffset_0.yaml
python3 run/run.py configs/mouse_configs/configs_mouse_gb_l1_tanhscalingoffset_w0.003_0.yaml
python3 run/run.py configs/mouse_configs/configs_mouse_gb_l1_tanhscalingoffset_w0.005_0.yaml
python3 run/run.py configs/mouse_configs/configs_mouse_gb_l1_tanhscalingoffset_w0.01_0.yaml
python3 run/run.py configs/mouse_configs/configs_mouse_gb_l1_tanhscalingoffset_w0.05_0.yaml

# python3 run/run.py configs/t100_configs/configs_cage_t100_set2_gb_l1_0.yaml
# python3 run/run.py configs/t100_configs/configs_cage_t100_set2_gb_l1_relu0.0001_0.yaml
# python3 run/run.py configs/t100_configs/configs_cage_t100_set2_gb_l1_relu0.001_0.yaml
# python3 run/run.py configs/t100_configs/configs_cage_t100_set2_gb_l1_relu0.01_0.yaml
# python3 run/run.py configs/t100_configs/configs_cage_t100_set2_gb_l1_relu0.05_0.yaml
# python3 run/run.py configs/t100_configs/configs_cage_t100_set2_gb_l1_relu0.1_0.yaml
# python3 run/run.py configs/t100_configs/configs_cage_t100_set2_gb_l1_relu0.5_0.yaml
# python3 run/run.py configs/t100_configs/configs_cage_t100_set2_gb_l1_relu1_0.yaml

# python3 run/run.py configs/t100_configs/configs_cage_t100_set2_gb_l1_relu_0.yaml
# python3 run/run.py configs/t100_configs/configs_cage_t100_set2_gb_l1_sigmoid_0.yaml
# python3 run/run.py configs/t100_configs/configs_cage_t100_set2_gb_l1_tanh_0.yaml
# python3 run/run.py configs/t100_configs/configs_cage_t100_set2_gb_l1_selu_0.yaml
# python3 run/run.py configs/t100_configs/configs_cage_t100_set2_gb_l1_celu_0.yaml
# python3 run/run.py configs/t100_configs/configs_cage_t100_set2_gb_l1_gelu_0.yaml
# python3 run/run.py configs/t100_configs/configs_cage_t100_set2_gb_l1_elu_0.yaml


# python3 run/run.py configs/t100_configs/configs_cage_t100_set2.yaml
# python3 run/run.py configs/t100_configs/configs_cage_t100_set2_l1_0.01.yaml
# python3 run/run.py configs/t100_configs/configs_cage_t100_set2_l1_0.013.yaml
# python3 run/run.py configs/t100_configs/configs_cage_t100_set2_l1_0.016.yaml
# python3 run/run.py configs/t100_configs/configs_cage_t100_set2_l1_0.019.yaml
# python3 run/run.py configs/t100_configs/configs_cage_t100_set2_l1_0.022.yaml
# python3 run/run.py configs/t100_configs/configs_cage_t100_set2_l1_0.025.yaml
# python3 run/run.py configs/t100_configs/configs_cage_t100_set2_l1_0.05.yaml
# python3 run/run.py configs/t100_configs/configs_cage_t100_set2_l1_0.1.yaml
# python3 run/run.py configs/t100_configs/configs_cage_t100_set2_l1_0.25.yaml
# python3 run/run.py configs/t100_configs/configs_cage_t100_set2_l1_0.5.yaml


# python3 run/run.py configs/t100_configs/configs_cage_t100_set2_gb_l1_0.yaml
# python3 run/run.py configs/t100_configs/configs_cage_t100_set2_gb_l1_0.0025.yaml
# python3 run/run.py configs/t100_configs/configs_cage_t100_set2_gb_l1_0.005.yaml
# python3 run/run.py configs/t100_configs/configs_cage_t100_set2_gb_l1_0.0075.yaml
# python3 run/run.py configs/t100_configs/configs_cage_t100_set2_gb_l1_0.01.yaml
# python3 run/run.py configs/t100_configs/configs_cage_t100_set2_gb_l1_0.013.yaml
# python3 run/run.py configs/t100_configs/configs_cage_t100_set2_gb_l1_0.016_s42.yaml
# python3 run/run.py configs/t100_configs/configs_cage_t100_set2_gb_l1_0.019.yaml
# python3 run/run.py configs/t100_configs/configs_cage_t100_set2_gb_l1_0.02.yaml
# python3 run/run.py configs/t100_configs/configs_cage_t100_set2_gb_l1_0.022.yaml
# python3 run/run.py configs/t100_configs/configs_cage_t100_set2_gb_l1_0.025_s42.yaml
# python3 run/run.py configs/t100_configs/configs_cage_t100_set2_gb_l1_0.03.yaml
# python3 run/run.py configs/t100_configs/configs_cage_t100_set2_gb_l1_0.05.yaml
# python3 run/run.py configs/t100_configs/configs_cage_t100_set2_gb_l1_0.05_s42.yaml
# python3 run/run.py configs/t100_configs/configs_cage_t100_set2_gb_l1_0.1.yaml
# python3 run/run.py configs/t100_configs/configs_cage_t100_set2_gb_l1_0.15.yaml
# python3 run/run.py configs/t100_configs/configs_cage_t100_set2_globalbias.yaml
# python3 run/run.py configs/t100_configs/configs_cage_t100_set2_gb_l1_0.5.yaml

# python3 run/run.py configs/t100_configs/configs_cage_t100_set2_gb_overlap_0.yaml
# python3 run/run.py configs/t100_configs/configs_cage_t100_set2_gb_overlap_0.0005.yaml
# python3 run/run.py configs/t100_configs/configs_cage_t100_set2_gb_overlap_0.001.yaml
# python3 run/run.py configs/t100_configs/configs_cage_t100_set2_gb_overlap_0.002.yaml
# python3 run/run.py configs/t100_configs/configs_cage_t100_set2_gb_overlap_0.003.yaml
# python3 run/run.py configs/t100_configs/configs_cage_t100_set2_gb_overlap_0.04.yaml
# python3 run/run.py configs/t100_configs/configs_cage_t100_set2_gb_overlap_0.05.yaml

# python3 run/run.py configs/t100_configs/configs_cage_t100_set2_gb_l1_0.016_s42.yaml
# python3 run/run.py configs/t100_configs/configs_cage_t100_set2_gb_l1_0.016_600epochs.yaml
# python3 run/run.py configs/t100_configs/configs_cage_t100_set2_gb_l1_0.016_750epochs.yaml
# python3 run/run.py configs/t100_configs/configs_cage_t100_set2_gb_l1_0.016_1000epochs.yaml

# python3 run/run.py configs/t100_configs/configs_cage_t100_set2_gb_l1_0.016_s42.yaml
# python3 run/run.py configs/t100_configs/configs_cage_t100_set2_gb_l1_0.016_s43.yaml
# python3 run/run.py configs/t100_configs/configs_cage_t100_set2_gb_l1_0.016_s44.yaml
# python3 run/run.py configs/t100_configs/configs_cage_t100_set2_gb_l1_0.025_s42.yaml
# python3 run/run.py configs/t100_configs/configs_cage_t100_set2_gb_l1_0.025_s43.yaml
# python3 run/run.py configs/t100_configs/configs_cage_t100_set2_gb_l1_0.025_s44.yaml
# python3 run/run.py configs/t100_configs/configs_cage_t100_set2_gb_l1_0.05_s42.yaml
# python3 run/run.py configs/t100_configs/configs_cage_t100_set2_gb_l1_0.05_s43.yaml
# python3 run/run.py configs/t100_configs/configs_cage_t100_set2_gb_l1_0.05_s44.yaml

# python3 run/run.py configs/t100_configs/configs_cage_t100_set2_logistic_gb_l1_0_s42.yaml
# python3 run/run.py configs/t100_configs/configs_cage_t100_set2_logistic_gb_l1_0.yaml
# python3 run/run.py configs/t100_configs/configs_cage_t100_set2_logistic_gb_l1_0.016.yaml
# python3 run/run.py configs/t100_configs/configs_cage_t100_set2_logistic_gb_l1_0.025.yaml

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
