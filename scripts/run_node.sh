#!/bin/bash

export CUDA_VISIBLE_DEVICES=1

source /ems/elsc-labs/kadmon-j/gilad.ticher/pytorch_env/bin/activate
python train.py task=free_instructed model=eegpt preprocessing=eegpt_frontal_central wandb.enabled=false

python train.py task=congruent_incongruent model=eegpt preprocessing=eegpt_frontal_central wandb.enabled=false