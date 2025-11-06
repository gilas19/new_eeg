import numpy as np
import matplotlib.pyplot as plt
from src.data.dataset import EEGDataset

data_dir = "./data/09_25_7"

print("=" * 60)
print("Testing Response Locking")
print("=" * 60)

# Load dataset without preprocessing
print("\n1. Loading dataset without preprocessing...")
dataset_no_preproc = EEGDataset(
    data_dir=data_dir,
    task='right_left',
    preprocessing_config={'enabled': False}
)

print(f"Dataset size: {len(dataset_no_preproc)}")
print(f"Trials shape: {dataset_no_preproc.trials.shape}")

# Calculate response times
print("\n2. Response times statistics:")
rt = dataset_no_preproc.response_times
print(f"  Min RT: {rt.min()}")
print(f"  Max RT: {rt.max()}")
print(f"  Mean RT: {rt.mean():.2f}")
print(f"  Median RT: {np.median(rt):.2f}")

# Load dataset WITH response locking
print("\n3. Loading dataset with response locking...")
dataset_with_lock = EEGDataset(
    data_dir=data_dir,
    task='right_left',
    preprocessing_config={
        'enabled': True,
        'response_locking': True,
        'handle_nans': True
    }
)

print(f"Dataset size after preprocessing: {len(dataset_with_lock)}")
print(f"Trials shape: {dataset_with_lock.trials.shape}")

# Compare a specific trial
trial_idx = 0
print(f"\n4. Comparing trial {trial_idx}:")

trial_original = dataset_no_preproc.trials[:, trial_idx, :]
trial_locked = dataset_with_lock.trials[:, trial_idx, :]
rt_value = dataset_no_preproc.response_times[trial_idx]

print(f"  Response time: {rt_value}")
print(f"  Original trial shape: {trial_original.shape}")
print(f"  Locked trial shape: {trial_locked.shape}")

# Check that data is aligned to the end
channel_idx = 0
print(f"\n5. Checking channel {channel_idx}:")
print(f"  Original last 10 values: {trial_original[channel_idx, -10:]}")
print(f"  Locked last 10 values: {trial_locked[channel_idx, -10:]}")

# Visualize a few trials
print("\n6. Creating visualization...")
fig, axes = plt.subplots(2, 3, figsize=(15, 8))

for i in range(3):
    trial_orig = dataset_no_preproc.trials[:, i, :]
    trial_lock = dataset_with_lock.trials[:, i, :]
    rt = dataset_no_preproc.response_times[i]

    # Plot original
    ax = axes[0, i]
    ax.plot(trial_orig[0, :], label='Channel 0', alpha=0.7)
    ax.axvline(rt, color='red', linestyle='--', label=f'RT={rt}')
    ax.set_title(f'Trial {i} - Original')
    ax.legend()
    ax.grid(True)

    # Plot locked
    ax = axes[1, i]
    ax.plot(trial_lock[0, :], label='Channel 0', alpha=0.7)
    ax.axvline(len(trial_lock[0]) - rt, color='red', linestyle='--', label=f'Locked at end')
    ax.set_title(f'Trial {i} - Response Locked')
    ax.legend()
    ax.grid(True)

plt.tight_layout()
plt.savefig('response_locking_test.png', dpi=150)
print("  Saved visualization to: response_locking_test.png")

print("\n" + "=" * 60)
print("Test completed successfully!")
print("=" * 60)
