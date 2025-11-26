# Configuration Files

This directory contains all configuration files for the EEG classification project. The configurations use [Hydra](https://hydra.cc/) for hierarchical composition and management.

## Directory Structure

```
configs/
├── config.yaml                    # Main configuration file
├── data/                          # Data loading configurations
├── model/                         # Model architecture configurations
├── preprocessing/                 # Preprocessing configurations
├── task/                          # Task-specific configurations
├── training/                      # Training hyperparameters
├── wandb/                         # Weights & Biases logging configurations
├── experiment/                    # Complete experiment configurations
└── optuna/                        # Hyperparameter optimization configurations
```

## Configuration Groups

### Main Configuration (`config.yaml`)

The main configuration file that sets defaults for all configuration groups.

**Defaults:**
- `data`: default
- `wandb`: default
- `training`: default
- `task`: free_instructed
- `model`: cnn
- `preprocessing`: default

**Parameters:**
- `task`: Task type (free_instructed, congruent_incongruent, right_left)
- `seed`: Random seed (default: 42)
- `device`: Computing device (cuda/cpu)

### Data Configurations (`data/`)

Control data loading, splitting, and cross-validation.

| File | Description | Parameters |
|------|-------------|------------|
| `default.yaml` | Single fold training | `n_folds: 1` |
| `cv5.yaml` | 5-fold cross-validation | `n_folds: 5` |
| `cv10.yaml` | 10-fold cross-validation | `n_folds: 10` |

**Common Parameters:**
- `data_dir`: Path to raw EEG data
- `batch_size`: Batch size for training (default: 64)
- `val_split`: Validation set ratio (default: 0.1)
- `test_split`: Test set ratio (default: 0.1)
- `n_folds`: Number of cross-validation folds

### Model Configurations (`model/`)

Define model architectures and their hyperparameters.

#### CNN Model (`cnn.yaml`)

Simple convolutional neural network for EEG classification.

**Parameters:**
- `type`: cnn
- `dropout`: Dropout rate (default: 0.3)

#### EEGPT Model (`eegpt.yaml`)

Transformer-based model with pretrained weights for EEG analysis.

**Architecture Parameters:**
- `patch_stride`: Patch stride for tokenization (default: 64)
- `use_chan_conv`: Enable channel convolution (default: true)
- `interpolate_factor`: Interpolation factor (default: 2.0)
- `desired_time_len`: Target time sequence length (default: 256)

**Regularization Parameters:**
- `max_norm_chan_conv`: Max norm for channel convolution (default: 1)
- `max_norm_head`: Max norm for classification head (default: 0.5)
- `qkv_bias`: Use bias in QKV projection (default: true)
- `enc_drop_rate`: Encoder dropout rate (default: 0.1)
- `enc_attn_drop_rate`: Encoder attention dropout (default: 0.1)
- `enc_drop_path_rate`: Encoder drop path rate (default: 0.1)
- `rec_drop_rate`: Reconstructor dropout rate (default: 0.1)
- `rec_attn_drop_rate`: Reconstructor attention dropout (default: 0.1)
- `rec_drop_path_rate`: Reconstructor drop path rate (default: 0.1)
- `use_freeze_encoder`: Freeze encoder weights (default: true)

**Pretrained Weights:**
- `load_pretrained`: Load pretrained checkpoint (default: true)
- `pretrained_checkpoint`: Path to checkpoint file

### Preprocessing Configurations (`preprocessing/`)

Configure data preprocessing pipelines.

#### Default Preprocessing (`default.yaml`)

Standard preprocessing without channel selection.

**Parameters:**
- `enabled`: Enable preprocessing (default: true)
- `filter_simple_trials`: Filter simple trials (default: false)
- `filter_incongruent`: Filter incongruent trials (default: false)
- `response_locking`: Use response-locked analysis (default: false)
- `handle_nans`: Handle NaN values (default: true)
- `normalize_per_trial`: Normalize each trial (default: true)

**Filter by Length:**
- `enabled`: Enable length filtering (default: false)
- `min_length`: Minimum trial length
- `max_length`: Maximum trial length

**Channel Selection:**
- `enabled`: Enable channel selection (default: false)
- `channels`: List of channel names to keep

**Truncation:**
- `enabled`: Enable truncation (default: false)
- `start_timepoint`: Start timepoint
- `end_timepoint`: End timepoint

#### EEGPT Frontal-Central (`eegpt_frontal_central.yaml`)

Preprocessing with 39 frontal and central channel selection optimized for EEGPT model.

**Selected Channels:**
- **Frontal:** Fp1, Fpz, Fp2, AF7, AF3, AF4, AF8, F7, F5, F3, F1, Fz, F2, F4, F6, F8
- **Fronto-Central:** FT7, FC5, FC3, FC1, FCz, FC2, FC4, FC6, FT8
- **Central:** T7, C5, C3, C1, Cz, C2, C4, C6, T8

### Task Configurations (`task/`)

Define classification tasks and their labels.

| File | Task | Description |
|------|------|-------------|
| `free_instructed.yaml` | free_instructed | Free vs instructed actions |
| `congruent_incongruent.yaml` | congruent_incongruent | Congruent vs incongruent trials |
| `right_left.yaml` | right_left | Right vs left hand movements |
| `response_locked.yaml` | right_left | Response-locked right/left analysis |
| `simple_trials.yaml` | right_left | Simple trials only (right/left) |

### Training Configurations (`training/`)

Control training hyperparameters and learning rate scheduling.

**Default Training Parameters:**
- `epochs`: Number of training epochs (default: 15)
- `patience`: Early stopping patience (default: 3)
- `learning_rate`: Initial learning rate (default: 0.0001)
- `weight_decay`: L2 regularization (default: 0.00001)

**Learning Rate Scheduler:**
- `enabled`: Enable scheduler (default: true)
- `type`: Scheduler type (CosineAnnealingLR, ReduceLROnPlateau, StepLR)

**CosineAnnealingLR Parameters:**
- `T_max`: Maximum iterations (default: 15)
- `eta_min`: Minimum learning rate (default: 1e-6)

**ReduceLROnPlateau Parameters:**
- `factor`: LR reduction factor (default: 0.5)
- `patience`: Patience epochs (default: 5)

**StepLR Parameters:**
- `step_size`: Step size (default: 30)
- `gamma`: Multiplicative factor (default: 0.1)

### Wandb Configurations (`wandb/`)

Configure Weights & Biases experiment tracking.

| File | Mode | Description |
|------|------|-------------|
| `default.yaml` | online | Full online logging |
| `offline.yaml` | offline | Offline logging (sync later) |
| `disabled.yaml` | disabled | No W&B logging |

**Parameters:**
- `enabled`: Enable W&B (true/false)
- `project`: Project name (default: eeg-classification)
- `entity`: W&B entity/team name
- `mode`: Logging mode (online/offline/disabled)

### Experiment Configurations (`experiment/`)

Complete experiment setups combining all configuration groups.

| File | Model | Task | Data | Special Features |
|------|-------|------|------|-----------------|
| `baseline_cnn.yaml` | CNN | free_instructed | default | Basic CNN baseline |
| `eegpt_transfer.yaml` | EEGPT | free_instructed | default | EEGPT with pretrained weights |
| `cv5_cnn.yaml` | CNN | free_instructed | cv5 | 5-fold cross-validation |
| `cv5_eegpt.yaml` | EEGPT | free_instructed | cv5 | 5-fold CV with EEGPT |
| `eegpt_frontal.yaml` | EEGPT | free_instructed | default | Frontal-central channels only |
| `response_locked.yaml` | CNN | right_left | default | Response-locked analysis |
| `simple_trials.yaml` | CNN | right_left | default | Simple trials only |
| `debug.yaml` | CNN | simple_trials | default | Quick debugging (3 epochs) |

### Optuna Configurations (`optuna/`)

Hyperparameter optimization using [Optuna](https://optuna.org/).

#### CNN Optimization (`cnn_optimization.yaml`)

Optimize CNN hyperparameters.

**Study Configuration:**
- `study_name`: cnn_eeg_optimization
- `n_trials`: 50
- `storage`: SQLite database for results
- `output_dir`: checkpoints/optimization_results

**Search Space:**
- `learning_rate`: [1e-5, 1e-2] (log scale)
- `weight_decay`: [1e-6, 1e-3] (log scale)
- `batch_size`: [16, 32, 64, 128]
- `dropout`: [0.1, 0.6]
- `eta_min`: [1e-7, 1e-5] (log scale)

#### EEGPT Optimization (`eegpt_optimization.yaml`)

Optimize EEGPT model hyperparameters.

**Study Configuration:**
- `study_name`: eegpt_eeg_optimization
- `n_trials`: 30
- `storage`: SQLite database for results

**Search Space:**
- `learning_rate`: [1e-5, 5e-3] (log scale)
- `weight_decay`: [1e-7, 1e-4] (log scale)
- `batch_size`: [32, 64, 96]
- `enc_drop_rate`: [0.0, 0.3]
- `enc_attn_drop_rate`: [0.0, 0.3]
- `enc_drop_path_rate`: [0.0, 0.3]
- `rec_drop_rate`: [0.0, 0.3]
- `rec_attn_drop_rate`: [0.0, 0.3]
- `rec_drop_path_rate`: [0.0, 0.3]
- `eta_min`: [1e-7, 1e-5] (log scale)

#### Quick Optimization (`quick_optimization.yaml`)

Fast optimization for testing (10 trials).

**Study Configuration:**
- `study_name`: quick_eeg_optimization
- `n_trials`: 10
- `wandb.enabled`: false

**Search Space:**
- `learning_rate`: [1e-4, 1e-3] (log scale)
- `weight_decay`: [1e-5, 1e-4] (log scale)
- `batch_size`: [32, 64]
- `dropout`: [0.2, 0.5]

## Usage

### Running an Experiment

Use Hydra's command-line interface to run experiments:

```bash
# Run default configuration
python train.py

# Run specific experiment
python train.py experiment=baseline_cnn

# Override specific parameters
python train.py experiment=baseline_cnn training.epochs=20 model.cnn.dropout=0.5

# Run with 5-fold cross-validation
python train.py experiment=cv5_cnn

# Run EEGPT with custom task
python train.py experiment=eegpt_transfer task=congruent_incongruent
```

### Hyperparameter Optimization

```bash
# Optimize CNN hyperparameters
python optuna_optimization.py --config configs/optuna/cnn_optimization.yaml

# Optimize EEGPT hyperparameters
python optuna_optimization.py --config configs/optuna/eegpt_optimization.yaml

# Quick test optimization
python optuna_optimization.py --config configs/optuna/quick_optimization.yaml
```

### Configuration Composition

Hydra allows flexible configuration composition:

```bash
# Mix different configuration groups
python train.py model=eegpt task=congruent_incongruent data=cv10 wandb=offline

# Override nested parameters
python train.py model.eegpt.enc_drop_rate=0.2 training.learning_rate=0.001
```

### Multi-run Experiments

Run parameter sweeps using Hydra's multirun feature:

```bash
# Sweep over multiple models
python train.py -m model=cnn,eegpt

# Sweep over dropout values
python train.py -m model.cnn.dropout=0.2,0.3,0.4,0.5

# Complex sweep
python train.py -m model=cnn,eegpt task=free_instructed,congruent_incongruent
```

## Best Practices

1. **Use Experiment Configs**: Create experiment configs for reproducible setups
2. **Version Control**: Track all configuration changes in git
3. **Documentation**: Add comments to custom configurations (will be preserved in experiments)
4. **Naming**: Use descriptive names for custom experiment configs
5. **Validation**: Always validate configs before long runs using `debug.yaml`

## Configuration Validation

All configuration files have been validated for:
- ✅ Valid YAML syntax
- ✅ Proper Hydra structure
- ✅ Consistent parameter naming
- ✅ Complete experiment definitions

## Notes

- All paths in `data/` configs are relative to the project root
- EEGPT model requires pretrained checkpoint (update path in `model/eegpt.yaml`)
- Cross-validation configs override `n_folds` parameter
- Debug config disables W&B and runs only 3 epochs for quick testing
