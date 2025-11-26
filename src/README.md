# Source Code Documentation

This directory contains the core implementation of the EEG classification framework. The code is organized into modular components for data handling, model architectures, training, and utilities.

## Directory Structure

```
src/
├── data/                # Data loading and preprocessing
│   ├── dataset.py      # EEG dataset class
│   └── datamodule.py   # Data module for train/val/test splits
├── models/             # Neural network architectures
│   ├── cnn.py         # CNN model for temporal EEG signals
│   └── EEGPT.py       # Transformer-based EEGPT model
├── training/           # Training infrastructure
│   ├── trainer.py     # Main training loop and validation
│   └── metrics.py     # Evaluation metrics
└── utils/              # Utility functions
    └── preprocessing.py # Preprocessing pipelines
```

## Components

### Data (`data/`)

#### `dataset.py` - EEGDataset

The main dataset class for loading and preprocessing EEG data.

**Key Features:**
- Loads EEG trials with associated metadata (electrodes, labels, primes, cues, reactions)
- Calculates response times from NaN patterns in trials
- Supports multiple classification tasks:
  - `right_left`: Left vs right hand movement
  - `free_instructed`: Free vs instructed trials
  - `congruent_incongruent`: Congruent vs incongruent trials
- Applies preprocessing pipeline (filtering, normalization, response-locking)
- Handles trial validation and indexing

**Usage:**
```python
from src.data.dataset import EEGDataset

dataset = EEGDataset(
    data_dir='path/to/data',
    task='right_left',
    preprocessing_config={
        'enabled': True,
        'response_locking': True,
        'normalize_per_trial': True
    }
)
```

**Methods:**
- `__init__(data_dir, task, preprocessing_config)`: Initialize dataset
- `__len__()`: Get number of valid trials
- `__getitem__(idx)`: Get trial and label tensors
- `get_subject_indices()`: Get trial indices grouped by subject
- `_calculate_response_times()`: Extract response times from NaN patterns
- `_generate_labels()`: Generate labels based on task
- `_get_valid_indices()`: Get valid trial indices for task

**Data Format:**
Expected files in `data_dir`:
- `trials_dataset.npy`: (n_channels, n_trials, n_timepoints) - EEG signals
- `electrodes_names.npy`: (n_channels,) - Channel names
- `true_labels.npy`: (n_trials,) - Reaction labels (1=right, 2=left)
- `primes.npy`: (n_trials,) - Prime stimuli
- `cues.npy`: (n_trials,) - Cue stimuli
- `subjects_through_trials.npy`: (n_trials,) - Subject IDs

#### `datamodule.py` - EEGDataModule

Manages data splitting and DataLoader creation with subject-aware or trial-based splitting.

**Key Features:**
- Subject-based splitting (prevents data leakage across subjects)
- Falls back to trial-based splitting if insufficient subjects
- Support for k-fold cross-validation
- Configurable batch size and validation/test splits

**Usage:**
```python
from src.data.datamodule import EEGDataModule

datamodule = EEGDataModule(
    dataset=dataset,
    batch_size=64,
    val_split=0.1,
    test_split=0.1,
    seed=42,
    n_folds=5,      # For cross-validation
    current_fold=0  # Current fold index
)

train_loader = datamodule.train_dataloader()
val_loader = datamodule.val_dataloader()
test_loader = datamodule.test_dataloader()
```

**Methods:**
- `train_dataloader(shuffle=True)`: Get training DataLoader
- `val_dataloader()`: Get validation DataLoader
- `test_dataloader()`: Get test DataLoader
- `num_batches_per_epoch()`: Get number of training batches
- `get_split_info()`: Get dataset split statistics

---

### Models (`models/`)

#### `cnn.py` - EEG_CNN

A convolutional neural network designed for temporal EEG signal processing.

**Architecture:**
```
Input: (batch_size, n_channels, n_timepoints)

Temporal Convolution Blocks:
├── Conv1D(n_channels → 64, kernel=25)
├── BatchNorm + GELU + MaxPool + Dropout
├── Conv1D(64 → 128, kernel=15)
├── BatchNorm + GELU + MaxPool + Dropout
└── Conv1D(128 → 256, kernel=10)
    └── BatchNorm + GELU + MaxPool + Dropout

Classifier:
├── Linear(flattened → 256)
├── GELU + Dropout
├── Linear(256 → 64)
├── GELU + Dropout
└── Linear(64 → n_classes)
```

**Parameters:**
- `n_channels`: Number of EEG electrodes
- `n_timepoints`: Length of time series
- `n_classes`: Number of output classes (default: 2)
- `dropout`: Dropout probability (default: 0.3)

**Features:**
- Dynamic feature size calculation via dummy forward pass
- Kaiming initialization for Conv layers
- Xavier initialization for Linear layers
- BatchNorm for stable training

**Usage:**
```python
from src.models.cnn import EEG_CNN

model = EEG_CNN(
    n_channels=64,
    n_timepoints=1024,
    n_classes=2,
    dropout=0.3
)

output = model(eeg_tensor)  # (batch_size, n_classes)
```

#### `EEGPT.py` - EEGPTClassifier

A transformer-based architecture for EEG classification with rotary positional embeddings.

**Architecture Components:**

1. **Patch Embedding**: Converts EEG signals to patch tokens
2. **EEGTransformer (Encoder)**:
   - Channel embeddings for spatial information
   - Summary tokens for aggregation
   - Multi-head self-attention blocks
   - No positional encoding (uses channel structure)

3. **EEGTransformerReconstructor/Predictor**:
   - Rotary positional embeddings (RoPE)
   - Channel embeddings
   - Causal or non-causal attention
   - Optional output projection

4. **Classification Head**:
   - Mean pooling or CLS token
   - Layer normalization
   - Linear classifier with optional max-norm constraint

**Key Features:**
- Support for variable-length EEG sequences via temporal interpolation
- NaN handling with attention masking
- Pretrained weight loading
- Channel-wise convolution option
- Flash Attention for efficiency
- Configurable dropout rates for encoder/reconstructor

**Parameters:**
- `num_classes`: Number of output classes
- `in_channels`: Number of EEG channels
- `img_size`: [n_channels, n_timepoints]
- `patch_stride`: Stride for patch embedding
- `use_channels_names`: List of channel names for ordering
- `enc_drop_rate`, `enc_attn_drop_rate`, `enc_drop_path_rate`: Encoder dropout
- `rec_drop_rate`, `rec_attn_drop_rate`, `rec_drop_path_rate`: Reconstructor dropout
- `use_freeze_encoder`: Freeze encoder weights
- `use_freeze_reconstructor`: Freeze reconstructor weights
- `load_pretrained`: Load pretrained weights
- `pretrained_checkpoint`: Path to pretrained checkpoint

**Usage:**
```python
from src.models.EEGPT import EEGPTClassifier

model = EEGPTClassifier(
    num_classes=2,
    in_channels=64,
    img_size=[64, 1024],
    patch_stride=64,
    use_channels_names=channel_names,
    enc_drop_rate=0.1,
    rec_drop_rate=0.1,
    load_pretrained=True,
    pretrained_checkpoint='path/to/checkpoint.pth'
)

output = model(eeg_tensor)  # (batch_size, num_classes)
```

**Channel Dictionary:**
The model includes a standard 10-20 EEG channel mapping in `CHANNEL_DICT` for consistent channel ordering.

---

### Training (`training/`)

#### `trainer.py` - Trainer

Manages the complete training loop with validation, learning rate scheduling, and early stopping.

**Key Features:**
- Cross-entropy loss for classification
- Adam optimizer with configurable weight decay
- Gradient clipping (max_norm=1.0)
- Multiple learning rate schedulers:
  - ReduceLROnPlateau (validation-based)
  - StepLR (epoch-based)
  - CosineAnnealingLR (batch-based)
- Early stopping with patience
- Weights & Biases logging
- Progress bars with tqdm

**Usage:**
```python
from src.training.trainer import Trainer

trainer = Trainer(
    model=model,
    datamodule=datamodule,
    config=training_config,  # OmegaConf DictConfig
    device='cuda'
)

best_val_acc = trainer.run()
test_loss, test_metrics = trainer.test()
```

**Methods:**
- `train_epoch()`: Execute one training epoch
- `validate()`: Run validation
- `test()`: Evaluate on test set
- `train()`: Full training loop with early stopping
- `run()`: Main entry point for training

**Logged Metrics:**
- Training: loss, accuracy, precision, recall, F1
- Validation: loss, accuracy, precision, recall, F1
- Learning rate, patience counter, best validation accuracy
- Early stopping events

#### `metrics.py` - Evaluation Metrics

Computes classification metrics using scikit-learn.

**Metrics:**
- Accuracy
- Precision (binary and weighted)
- Recall (binary and weighted)
- F1 Score (binary and weighted)

**Usage:**
```python
from src.training.metrics import compute_metrics

metrics = compute_metrics(predictions, targets)
# Returns: {'accuracy': ..., 'precision': ..., 'recall': ..., 'f1_score': ...}
```

---

### Utils (`utils/`)

#### `preprocessing.py` - Preprocessing Pipeline

Comprehensive preprocessing utilities for EEG data.

**Components:**

1. **TrialFilter**: Filter trials based on various criteria
   - `filter_simple_trials()`: Keep only simple trial patterns (RRR, LLL, NRR, NLL)
   - `filter_incongruent()`: Keep only incongruent trials (prime ≠ cue)
   - `filter_by_trial_length()`: Filter by response time range

2. **ChannelSelector**: Select specific EEG channels
   - `select_channels()`: Select channels by name (case-insensitive)

3. **TrialTransformer**: Transform trial data
   - `response_locking()`: Align trials to response time
   - `truncate_trials()`: Cut trials to specific timepoints
   - `handle_nans()`: Replace NaN values
   - `normalize_per_trial()`: Per-trial z-score normalization

**Usage:**
```python
from src.utils.preprocessing import apply_preprocessing

config = {
    'enabled': True,
    'filter_simple_trials': True,
    'response_locking': True,
    'truncate': {
        'enabled': True,
        'start_timepoint': 100,
        'end_timepoint': 900
    },
    'select_channels': {
        'enabled': True,
        'channels': ['FZ', 'CZ', 'PZ']
    },
    'normalize_per_trial': True
}

result = apply_preprocessing(
    trials, electrode_names, primes, cues,
    reactions, response_times, subjects, config
)
```

**Pipeline Order:**
1. Trial filtering (simple trials, incongruent, length)
2. Channel selection
3. Response locking
4. Trial truncation
5. NaN handling
6. Per-trial normalization

---

## Design Principles

1. **Modularity**: Each component is self-contained and reusable
2. **Configurability**: All parameters exposed via configuration
3. **Type Safety**: Proper tensor shapes and dtypes
4. **Logging**: Informative logging at each preprocessing step
5. **Subject Awareness**: Prevents data leakage in splits
6. **NaN Handling**: Explicit handling of missing data
7. **Device Agnostic**: Works with CPU and CUDA

## Best Practices

### Adding a New Model

1. Create model file in `src/models/`
2. Inherit from `nn.Module`
3. Implement `__init__()` and `forward()`
4. Add model config to `configs/model/`
5. Update model factory in `train.py`

### Adding a New Task

1. Add label generation logic to `dataset.py:_generate_labels()`
2. Add valid indices logic to `dataset.py:_get_valid_indices()`
3. Create task config in `configs/task/`
4. Update documentation

### Adding Preprocessing Steps

1. Add transformer method to `preprocessing.py`
2. Update `apply_preprocessing()` pipeline
3. Add config parameters to `configs/preprocessing/`
4. Add logging for transparency

## Common Patterns

### Loading Data
```python
from src.data.dataset import EEGDataset
from src.data.datamodule import EEGDataModule

dataset = EEGDataset(data_dir='data/', task='right_left')
datamodule = EEGDataModule(dataset, batch_size=64)
```

### Creating Models
```python
from src.models.cnn import EEG_CNN
from src.models.EEGPT import EEGPTClassifier

# CNN
model = EEG_CNN(n_channels=64, n_timepoints=1024, n_classes=2)

# EEGPT
model = EEGPTClassifier(num_classes=2, img_size=[64, 1024], ...)
```

### Training
```python
from src.training.trainer import Trainer

trainer = Trainer(model, datamodule, config, device='cuda')
best_val_acc = trainer.run()
test_loss, test_metrics = trainer.test()
```

## Performance Tips

1. **Batch Size**: Increase for faster training (if GPU memory allows)
2. **Num Workers**: Set in DataLoader for parallel data loading
3. **Mixed Precision**: Enable for faster training with compatible GPUs
4. **Gradient Accumulation**: For larger effective batch sizes
5. **Flash Attention**: Used by EEGPT for efficient attention

## Troubleshooting

**Shape Mismatches:**
- Check `img_size` matches actual data shape for EEGPT
- Verify channel ordering in `use_channels_names`

**NaN Issues:**
- Enable `handle_nans` in preprocessing
- Check attention masking in EEGPT

**Memory Issues:**
- Reduce batch size
- Use gradient checkpointing
- Reduce model size (depth, embed_dim)

**Poor Performance:**
- Check class balance in dataset
- Verify preprocessing is appropriate for task
- Tune learning rate and dropout
- Try different model architectures

## Further Reading

- [Main README](../README.md): Project overview and usage
- Configuration files in `configs/`: Parameter documentation
- [train.py](../train.py): Training script entry point
- [optuna_optimization.py](../optuna_optimization.py): Hyperparameter tuning
