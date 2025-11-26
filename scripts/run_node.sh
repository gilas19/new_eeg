#!/bin/bash
#SBATCH --job-name=run_eegpt_node
#SBATCH --output=outputs/run_node/%j.out
#SBATCH --error=outputs/run_node/%j.err
#SBATCH --gres=gpu:1

export CUDA_VISIBLE_DEVICES=1 # Specify which GPU to use

source /ems/elsc-labs/kadmon-j/gilad.ticher/pytorch_env/bin/activate

python train.py task=free_instructed model=eegpt preprocessing=eegpt_frontal_central wandb.enabled=false
python train.py task=congruent_incongruent model=eegpt preprocessing=eegpt_frontal_central wandb.enabled=false