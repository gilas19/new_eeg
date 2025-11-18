#!/bin/bash
#SBATCH --job-name=optuna_optimization
#SBATCH --output=outputs/optuna_optimization/%j.out
#SBATCH --error=outputs/optuna_optimization/%j.err

# Hyperparameter Optimization Runner Script
# This script provides easy ways to run different optimization configurations

export CUDA_VISIBLE_DEVICES=1

# Default values
CONFIG="configs/optuna/cnn_optimization.yaml"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --config)
            CONFIG="$2"
            shift 2
            ;;
    esac
done

# Set config based on model if specified
if [ ! -z "$MODEL" ]; then
    case $MODEL in
        cnn)
            CONFIG="configs/optuna/cnn_optimization.yaml"
            ;;
        eegpt)
            CONFIG="configs/optuna/eegpt_optimization.yaml"
            ;;
        *)
            echo "Unknown model type: $MODEL"
            echo "Choose 'cnn' or 'eegpt'"
            exit 1
            ;;
    esac
fi

python optuna_optimization.py --config "$CONFIG"