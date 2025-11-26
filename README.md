# EEG Classification Project

A PyTorch-based deep learning framework for EEG signal classification, supporting multiple neural network architectures including CNN and EEGPT (EEG Transformer) models. The project includes comprehensive preprocessing, hyperparameter optimization, and cross-validation capabilities.

## Features

- **Multiple Model Architectures**
  - Custom CNN architecture for temporal EEG signal processing
  - EEGPT: Transformer-based architecture with rotary positional embeddings (RoPE)
  - Support for pretrained EEGPT weights

- **Flexible Task Support**
  - Right/Left hand movement classification
  - Free vs Instructed trial classification
  - Congruent vs Incongruent trial classification

- **Advanced Training Features**
  - Cross-validation support with k-fold splitting
  - Learning rate scheduling (ReduceLROnPlateau, StepLR, CosineAnnealingLR)
  - Early stopping with patience
  - Gradient clipping for stable training
  - Weights & Biases integration for experiment tracking

- **Hyperparameter Optimization**
  - Optuna integration for automated hyperparameter search
  - Support for trial pruning
  - Configurable search spaces for both CNN and EEGPT models
  - TPE sampler with MedianPruner

- **Comprehensive Preprocessing**
  - Response-locked analysis
  - Trial filtering (simple trials, incongruent trials, length-based)
  - Channel selection
  - Trial truncation
  - Per-trial normalization
  - NaN handling

## Project Structure

```
.
├── configs/                  # Hydra configuration files
│   ├── README.md            # Comprehensive configuration documentation
│   ├── config.yaml          # Main configuration file
│   ├── data/                # Data loading configurations (default, cv5, cv10)
│   ├── model/               # Model architectures (CNN, EEGPT)
│   ├── task/                # Task definitions (5 classification tasks)
│   ├── preprocessing/       # Preprocessing pipelines
│   ├── training/            # Training hyperparameters and schedulers
│   ├── wandb/               # Weights & Biases configurations
│   ├── experiment/          # Complete experiment setups (8 experiments)
│   └── optuna/              # Hyperparameter optimization configs
├── src/                     # Source code
│   ├── data/                # Dataset and data loading
│   │   ├── dataset.py       # EEGDataset class
│   │   └── datamodule.py    # Cross-validation data module
│   ├── models/              # Model architectures
│   │   ├── cnn.py           # CNN architecture
│   │   └── EEGPT.py         # EEGPT transformer model
│   ├── training/            # Training and metrics
│   │   ├── trainer.py       # Trainer class
│   │   └── metrics.py       # Evaluation metrics
│   └── utils/               # Preprocessing utilities
│       └── preprocessing.py # Preprocessing functions
├── notebooks/               # Jupyter notebooks for analysis
├── scripts/                 # Shell scripts for SLURM
├── tests/                   # Unit tests
├── checkpoints/             # Model checkpoints and results
├── train.py                 # Main training script
└── optuna_optimization.py   # Hyperparameter optimization script
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd new_eeg
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Training

Train a model using the default configuration:

```bash
python train.py
```

### Experiment Configurations

Use pre-configured experiments for common setups:

```bash
# Run baseline CNN experiment
python train.py experiment=baseline_cnn

# Run EEGPT with pretrained weights
python train.py experiment=eegpt_transfer

# Run 5-fold cross-validation with CNN
python train.py experiment=cv5_cnn

# Run 5-fold cross-validation with EEGPT
python train.py experiment=cv5_eegpt

# Run EEGPT with frontal-central channels only
python train.py experiment=eegpt_frontal

# Run response-locked analysis
python train.py experiment=response_locked

# Quick debug run (3 epochs, no W&B)
python train.py experiment=debug
```

### Custom Configuration

Override configuration parameters:

```bash
# Train with EEGPT model
python train.py model=eegpt

# Train on specific task
python train.py task=congruent_incongruent

# Use offline W&B logging
python train.py wandb=offline

# Change learning rate and batch size
python train.py training.learning_rate=0.001 data.batch_size=128

# Combine experiment with overrides
python train.py experiment=baseline_cnn training.epochs=20
```

### Cross-Validation

Use cross-validation data configurations:

```bash
# 5-fold cross-validation
python train.py data=cv5

# 10-fold cross-validation
python train.py data=cv10

# Combine with specific model
python train.py experiment=baseline_cnn data=cv10
```

### Multi-run Experiments

Run parameter sweeps using Hydra's multirun feature:

```bash
# Sweep over multiple models
python train.py -m model=cnn,eegpt

# Sweep over tasks
python train.py -m task=free_instructed,congruent_incongruent,right_left

# Complex sweep
python train.py -m experiment=baseline_cnn,eegpt_transfer data=default,cv5
```

### Hyperparameter Optimization

Run Optuna-based hyperparameter search:

```bash
# Optimize CNN hyperparameters (50 trials)
python optuna_optimization.py --config configs/optuna/cnn_optimization.yaml

# Optimize EEGPT hyperparameters (30 trials)
python optuna_optimization.py --config configs/optuna/eegpt_optimization.yaml

# Quick optimization for testing (10 trials)
python optuna_optimization.py --config configs/optuna/quick_optimization.yaml
```

The optimization results are stored in SQLite databases under `checkpoints/optimization_results/` and can be analyzed using Optuna's visualization tools.

## Configuration

The project uses [Hydra](https://hydra.cc/) for hierarchical configuration management. All configurations are organized into composable groups.

**📖 See [configs/README.md](configs/README.md) for comprehensive configuration documentation.**

### Configuration Groups

| Group | Files | Description |
|-------|-------|-------------|
| **data** | default, cv5, cv10 | Data loading and cross-validation |
| **model** | cnn, eegpt | Model architectures |
| **task** | 5 task types | Classification tasks |
| **preprocessing** | default, eegpt_frontal_central | Preprocessing pipelines |
| **training** | default | Training hyperparameters |
| **wandb** | default, offline, disabled | Experiment tracking |
| **experiment** | 8 experiments | Complete experiment setups |
| **optuna** | 3 configs | Hyperparameter optimization |

### Quick Configuration Reference

**Available Models:**
- `cnn`: Convolutional neural network
- `eegpt`: Transformer-based model with pretrained weights

**Available Tasks:**
- `free_instructed`: Free vs instructed actions
- `congruent_incongruent`: Congruent vs incongruent trials
- `right_left`: Right vs left hand movements
- `response_locked`: Response-locked analysis
- `simple_trials`: Simple trials only

**Available Experiments:**
- `baseline_cnn`: Basic CNN baseline
- `eegpt_transfer`: EEGPT with pretrained weights
- `cv5_cnn` / `cv5_eegpt`: 5-fold cross-validation
- `eegpt_frontal`: EEGPT with frontal-central channels
- `response_locked`: Response-locked analysis
- `simple_trials`: Simple trials filtering
- `debug`: Quick debugging (3 epochs)

**Key Parameters:**
- `data.batch_size`: Batch size (default: 64)
- `data.n_folds`: Cross-validation folds (1, 5, or 10)
- `training.learning_rate`: Learning rate (default: 0.0001)
- `training.epochs`: Training epochs (default: 15)
- `model.cnn.dropout`: CNN dropout rate (default: 0.3)
- `wandb.enabled`: Enable W&B logging (default: true)

## Data Format

The project expects EEG data in the following NumPy array format:

- `trials_dataset.npy`: (n_channels, n_trials, n_timepoints)
- `electrodes_names.npy`: (n_channels,)
- `true_labels.npy`: (n_trials,)
- `primes.npy`: (n_trials,)
- `cues.npy`: (n_trials,)
- `subjects_through_trials.npy`: (n_trials,)

## Models

### CNN Architecture

A temporal convolutional neural network with:
- 3 convolutional blocks with increasing channels (64 -> 128 -> 256)
- Batch normalization and GELU activation
- Max pooling and dropout
- Fully connected classifier head

See [src/models/cnn.py](src/models/cnn.py) for implementation.

### EEGPT Architecture

A transformer-based model with:
- Patch embedding for EEG signals
- Rotary positional embeddings (RoPE)
- Channel embeddings for spatial information
- Multi-head self-attention with Flash Attention
- Encoder-reconstructor/predictor architecture
- Support for pretrained weights

See [src/models/EEGPT.py](src/models/EEGPT.py) for implementation.

## Experiment Tracking

The project integrates with Weights & Biases for experiment tracking:

- Training/validation metrics (loss, accuracy, precision, recall, F1)
- Learning rate and patience tracking
- Dataset split information
- Cross-validation summary statistics
- Hyperparameter configurations

Configure W&B using the wandb configuration group:

```bash
# Use default online logging
python train.py wandb=default

# Use offline mode (sync later)
python train.py wandb=offline

# Disable W&B completely
python train.py wandb=disabled

# Override W&B entity
python train.py wandb.entity=your-team-name
```

## Results

The training script logs:
- Best validation accuracy
- Test set metrics (accuracy, precision, recall, F1)
- Cross-validation statistics (mean and std across folds)

Results are logged to:
- Console output
- Weights & Biases (if enabled)
- Checkpoints directory (model weights)

## Running on SLURM Clusters

For running on HPC clusters with SLURM:

```bash
# Submit training job
sbatch scripts/run_node.sh

# Submit optimization job
sbatch scripts/run_optimization.sh
```

See [scripts/SLURM_GUIDE.md](scripts/SLURM_GUIDE.md) for detailed instructions.

## Notebooks

- `explore_eegpt.ipynb`: EEGPT model exploration and analysis
- `fc_noise_analysis.ipynb`: Frequency-component noise analysis
- `spectogram_cnn.ipynb`: Spectrogram-based CNN experiments
- `analyze_ckpt.ipynb`: Checkpoint analysis utilities

## Configuration Best Practices

1. **Use Experiment Configs**: Create experiment configs in `configs/experiment/` for reproducible setups
2. **Validate First**: Test with `experiment=debug` before running long experiments
3. **Documentation**: See [configs/README.md](configs/README.md) for detailed configuration guide
4. **Version Control**: All configuration changes are tracked in git

## Testing

Run tests to validate preprocessing and data loading:

```bash
# Run all tests
pytest tests/

# Test preprocessing
pytest tests/test_preprocessing.py

# Test simple preprocessing
pytest tests/test_preprocessing_simple.py
```

## Contributing

When adding new features:

1. Add configuration files to `configs/` following the existing structure
2. Implement models in `src/models/`
3. Update preprocessing in `src/utils/preprocessing.py`
4. Add tests for new functionality in `tests/`
5. Update relevant README files

## Citation

If you use this code, please cite the relevant papers:

- EEGPT architecture is based on transformer models for EEG analysis
- CNN architecture follows best practices for temporal signal processing

## License

[Add your license here]

## Acknowledgments

- EEGPT implementation based on transformer architectures for EEG
- Preprocessing utilities inspired by standard EEG analysis practices
