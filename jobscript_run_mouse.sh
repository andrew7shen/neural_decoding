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
#python3 run/run.py configs/mouse_configs/configs_mouse_gb_l1_val80_w0.0_0.yaml
#python3 run/run.py configs/mouse_configs/configs_mouse_gb_l1_val80_w0.0025_0.yaml
#python3 run/run.py configs/mouse_configs/configs_mouse_gb_l1_val80_w0.005_0.yaml
#python3 run/run.py configs/mouse_configs/configs_mouse_gb_l1_val80_w0.01_0.yaml
#python3 run/run.py configs/mouse_configs/configs_mouse_gb_l1_val80_w0.02_0.yaml
#python3 run/run.py configs/mouse_configs/configs_mouse_gb_l1_val80_w0.05_0.yaml
#python3 run/run.py configs/mouse_configs/configs_mouse_gb_l1_val70_w0.0_0.yaml
#python3 run/run.py configs/mouse_configs/configs_mouse_gb_l1_val70_w0.0025_0.yaml
#python3 run/run.py configs/mouse_configs/configs_mouse_gb_l1_val70_w0.005_0.yaml
#python3 run/run.py configs/mouse_configs/configs_mouse_gb_l1_val70_w0.01_0.yaml
#python3 run/run.py configs/mouse_configs/configs_mouse_gb_l1_val70_w0.02_0.yaml
#python3 run/run.py configs/mouse_configs/configs_mouse_gb_l1_val70_w0.05_0.yaml
#python3 run/run.py configs/mouse_configs/configs_mouse_gb_l1_val60_w0.0_0.yaml
#python3 run/run.py configs/mouse_configs/configs_mouse_gb_l1_val60_w0.0025_0.yaml
#python3 run/run.py configs/mouse_configs/configs_mouse_gb_l1_val60_w0.005_0.yaml
#python3 run/run.py configs/mouse_configs/configs_mouse_gb_l1_val60_w0.01_0.yaml
#python3 run/run.py configs/mouse_configs/configs_mouse_gb_l1_val60_w0.02_0.yaml
#python3 run/run.py configs/mouse_configs/configs_mouse_gb_l1_val60_w0.05_0.yaml

# python3 run/run.py configs/mouse_configs/configs_mouse_gb_l1_0.yaml
# python3 run/run.py configs/mouse_configs/configs_mouse_gb_l1_relu0.01_0.yaml
# python3 run/run.py configs/mouse_configs/configs_mouse_gb_l1_relu0.1_0.yaml

# python3 run/run.py configs/mouse_configs/configs_mouse_gb_l1_method1_0.yaml
# python3 run/run.py configs/mouse_configs/configs_mouse_gb_l1_method2_0.yaml
# python3 run/run.py configs/mouse_configs/configs_mouse_gb_l1_method4_0.yaml

# python3 run/run.py configs/mouse_configs/configs_mouse_gb_l1_method2_0.yaml
# python3 run/run.py configs/mouse_configs/configs_mouse_gb_l1_method2_w0.003_0.yaml
# python3 run/run.py configs/mouse_configs/configs_mouse_gb_l1_method2_w0.005_0.yaml
# python3 run/run.py configs/mouse_configs/configs_mouse_gb_l1_method2_w0.01_0.yaml
# python3 run/run.py configs/mouse_configs/configs_mouse_gb_l1_method2_w0.012_0.yaml
# python3 run/run.py configs/mouse_configs/configs_mouse_gb_l1_method2_w0.015_0.yaml
# python3 run/run.py configs/mouse_configs/configs_mouse_gb_l1_method2_w0.017_0.yaml
# python3 run/run.py configs/mouse_configs/configs_mouse_gb_l1_method2_w0.02_0.yaml
# python3 run/run.py configs/mouse_configs/configs_mouse_gb_l1_method2_w0.022_0.yaml
# python3 run/run.py configs/mouse_configs/configs_mouse_gb_l1_method2_w0.025_0.yaml
# python3 run/run.py configs/mouse_configs/configs_mouse_gb_l1_method2_w0.05_0.yaml

# python3 run/run.py configs/mouse_configs/configs_mouse_gb_l1_w0.003_0.yaml
# python3 run/run.py configs/mouse_configs/configs_mouse_gb_l1_w0.005_0.yaml
# python3 run/run.py configs/mouse_configs/configs_mouse_gb_l1_w0.01_0.yaml
# python3 run/run.py configs/mouse_configs/configs_mouse_gb_l1_w0.05_0.yaml
# python3 run/run.py configs/mouse_configs/configs_mouse_gb_l1_method4_w0.003_0.yaml
# python3 run/run.py configs/mouse_configs/configs_mouse_gb_l1_method4_w0.005_0.yaml
# python3 run/run.py configs/mouse_configs/configs_mouse_gb_l1_method4_w0.01_0.yaml
# python3 run/run.py configs/mouse_configs/configs_mouse_gb_l1_method4_w0.012_0.yaml
# python3 run/run.py configs/mouse_configs/configs_mouse_gb_l1_method4_w0.015_0.yaml
# python3 run/run.py configs/mouse_configs/configs_mouse_gb_l1_method4_w0.05_0.yaml

# python3 run/run.py configs/mouse_configs/configs_mouse_gb_l1_w0.012_0.yaml
# python3 run/run.py configs/mouse_configs/configs_mouse_gb_l1_w0.015_0.yaml
# python3 run/run.py configs/mouse_configs/configs_mouse_gb_l1_w0.02_0.yaml
# python3 run/run.py configs/mouse_configs/configs_mouse_gb_l1_w0.025_0.yaml
# python3 run/run.py configs/mouse_configs/configs_mouse_gb_l1_method4_w0.012_0.yaml
# python3 run/run.py configs/mouse_configs/configs_mouse_gb_l1_method4_w0.015_0.yaml
# python3 run/run.py configs/mouse_configs/configs_mouse_gb_l1_method4_w0.02_0.yaml
# python3 run/run.py configs/mouse_configs/configs_mouse_gb_l1_method4_w0.025_0.yaml

# Nonlinear decoder runs
# python3 run/run.py configs/mouse_configs/configs_mouse_gb_l1_0.yaml
# python3 run/run.py configs/mouse_configs/configs_mouse_gb_l1_tanhscalingoffset_0.yaml
# python3 run/run.py configs/mouse_configs/configs_mouse_gb_l1_tanhscalingoffset_w0.003_0.yaml
# python3 run/run.py configs/mouse_configs/configs_mouse_gb_l1_tanhscalingoffset_w0.005_0.yaml
# python3 run/run.py configs/mouse_configs/configs_mouse_gb_l1_tanhscalingoffset_w0.01_0.yaml
# python3 run/run.py configs/mouse_configs/configs_mouse_gb_l1_tanhscalingoffset_w0.05_0.yaml

# python3 run/run.py configs/mouse_configs/configs_mouse_gb_l1_relu0.05_0.yaml
# python3 run/run.py configs/mouse_configs/configs_mouse_gb_l1_relu0.5_0.yaml
# python3 run/run.py configs/mouse_configs/configs_mouse_gb_l1_selu_0.yaml
# python3 run/run.py configs/mouse_configs/configs_mouse_gb_l1_celu_0.yaml
# python3 run/run.py configs/mouse_configs/configs_mouse_gb_l1_elu_0.yaml

# python3 run/run.py configs/mouse_configs/configs_mouse_gb_l1_0.yaml
# python3 run/run.py configs/mouse_configs/configs_mouse_gb_l1_0.0001.yaml
# python3 run/run.py configs/mouse_configs/configs_mouse_gb_l1_0.0002.yaml
# python3 run/run.py configs/mouse_configs/configs_mouse_gb_l1_0.0005.yaml
# python3 run/run.py configs/mouse_configs/configs_mouse_gb_l1_0.001.yaml
# python3 run/run.py configs/mouse_configs/configs_mouse_gb_l1_0.0025.yaml
# python3 run/run.py configs/mouse_configs/configs_mouse_gb_l1_0.005.yaml
# python3 run/run.py configs/mouse_configs/configs_mouse_gb_l1_0.01.yaml
# python3 run/run.py configs/mouse_configs/configs_mouse_gb_l1_0.02.yaml
# python3 run/run.py configs/mouse_configs/configs_mouse_gb_l1_0.03.yaml
# python3 run/run.py configs/mouse_configs/configs_mouse_gb_l1_0.04.yaml
# python3 run/run.py configs/mouse_configs/configs_mouse_gb_l1_0.05.yaml

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