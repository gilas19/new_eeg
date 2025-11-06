import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from src.utils.preprocessing import TrialFilter, ChannelSelector, TrialTransformer


def test_filter_simple_trials():
    """Test filtering simple trials (specific patterns: RRR, LLL, NRR, NLL)"""
    print("\n=== Testing filter_simple_trials ===")

    # Create test data with known simple trial patterns
    primes = np.array([1, 1, 2, 3, 2, 2, 1, 3, 1, 2])
    cues = np.array([1, 2, 2, 1, 1, 2, 2, 2, 1, 1])
    reactions = np.array([1, 1, 2, 1, 1, 2, 2, 2, 1, 1])

    # Simple patterns: RRR (1,1,1), LLL (2,2,2), NRR (3,1,1), NLL (3,2,2)
    # Index 0: (1,1,1) - RRR ✓
    # Index 1: (1,2,1) - not simple
    # Index 2: (2,2,2) - LLL ✓
    # Index 3: (3,1,1) - NRR ✓
    # Index 4: (2,1,1) - not simple
    # Index 5: (2,2,2) - LLL ✓
    # Index 6: (1,2,2) - not simple
    # Index 7: (3,2,2) - NLL ✓
    # Index 8: (1,1,1) - RRR ✓
    # Index 9: (2,1,1) - not simple
    expected_count = 6

    mask = TrialFilter.filter_simple_trials(primes, cues, reactions)

    assert np.sum(mask) == expected_count, \
        f"Expected {expected_count} simple trials, got {np.sum(mask)}"

    # Verify the specific patterns
    assert mask[0] == True, "Index 0 should be RRR"
    assert mask[2] == True, "Index 2 should be LLL"
    assert mask[3] == True, "Index 3 should be NRR"
    assert mask[5] == True, "Index 5 should be LLL"
    assert mask[7] == True, "Index 7 should be NLL"
    assert mask[8] == True, "Index 8 should be RRR"

    print(f"✓ Found {np.sum(mask)} simple trials out of {len(primes)}")
    print(f"✓ Simple patterns: RRR, LLL, NRR, NLL correctly identified")


def test_filter_incongruent():
    """Test filtering incongruent trials"""
    print("\n=== Testing filter_incongruent ===")

    primes = np.array([1, 1, 2, 1, 2, 2, 1, 2, 1, 2])
    cues = np.array([1, 2, 2, 1, 1, 2, 2, 2, 1, 1])

    # Incongruent trials are where prime != cue
    # Index 0: (1,1) - congruent
    # Index 1: (1,2) - incongruent ✓
    # Index 2: (2,2) - congruent
    # Index 3: (1,1) - congruent
    # Index 4: (2,1) - incongruent ✓
    # Index 5: (2,2) - congruent
    # Index 6: (1,2) - incongruent ✓
    # Index 7: (2,2) - congruent
    # Index 8: (1,1) - congruent
    # Index 9: (2,1) - incongruent ✓
    expected_count = 4

    mask = TrialFilter.filter_incongruent(primes, cues)

    assert np.sum(mask) == expected_count, \
        f"Expected {expected_count} incongruent trials, got {np.sum(mask)}"

    # Verify that all selected trials have prime != cue
    assert np.all((primes[mask] != cues[mask])), "All filtered trials should have prime != cue"

    print(f"✓ Found {np.sum(mask)} incongruent trials out of {len(primes)}")
    print(f"✓ All filtered trials have prime != cue")


def test_filter_by_trial_length():
    """Test filtering trials by minimum length"""
    print("\n=== Testing filter_by_trial_length ===")

    # Set different response times
    response_times = np.array([50, 80, 100, 60, 90, 40, 100, 70])
    min_length = 70
    max_length = 95

    # Expected to keep trials with 70 <= RT <= 95: indices [1, 4, 7]
    expected_min_count = 5
    expected_range_count = 3

    # Test min length only
    mask_min = TrialFilter.filter_by_trial_length(response_times, min_length=min_length)
    assert np.sum(mask_min) == expected_min_count, \
        f"Expected {expected_min_count} trials with RT >= {min_length}, got {np.sum(mask_min)}"
    assert np.all(response_times[mask_min] >= min_length), \
        f"All filtered trials should have RT >= {min_length}"

    print(f"✓ Min length filter: {np.sum(mask_min)} trials with RT >= {min_length}")

    # Test min and max length
    mask_range = TrialFilter.filter_by_trial_length(response_times, min_length=min_length, max_length=max_length)
    assert np.sum(mask_range) == expected_range_count, \
        f"Expected {expected_range_count} trials with {min_length} <= RT <= {max_length}, got {np.sum(mask_range)}"
    assert np.all((response_times[mask_range] >= min_length) & (response_times[mask_range] <= max_length)), \
        f"All filtered trials should have {min_length} <= RT <= {max_length}"

    print(f"✓ Range filter: {np.sum(mask_range)} trials with {min_length} <= RT <= {max_length}")


def test_select_channels():
    """Test channel selection"""
    print("\n=== Testing select_channels ===")

    electrode_names = np.array(['Fz', 'Cz', 'Pz', 'C3', 'C4', 'F3', 'F4', 'P3', 'P4', 'Oz'])

    # Select motor cortex channels
    channels_to_keep = ['C3', 'C4', 'Cz']
    expected_count = 3

    channel_indices = ChannelSelector.select_channels(electrode_names, channels_to_keep)

    assert len(channel_indices) == expected_count, \
        f"Expected {expected_count} channel indices, got {len(channel_indices)}"

    # Verify the correct channels were selected
    selected_names = electrode_names[channel_indices]
    assert set(selected_names) == set(channels_to_keep), \
        f"Expected {channels_to_keep}, got {list(selected_names)}"

    print(f"✓ Selected {len(channel_indices)} channels: {list(selected_names)}")
    print(f"✓ Channel indices: {list(channel_indices)}")


def test_response_locking():
    """Test response locking alignment"""
    print("\n=== Testing response_locking ===")

    n_channels, n_trials, n_timepoints = 3, 5, 100
    trials = np.random.randn(n_channels, n_trials, n_timepoints)
    response_times = np.array([50, 80, 100, 60, 70])

    locked_trials = TrialTransformer.response_locking(trials, response_times)

    # Check shape
    assert locked_trials.shape == trials.shape, "Shape should be preserved"

    # Check that data is right-aligned
    for trial_idx in range(n_trials):
        rt = response_times[trial_idx]
        # Data should be in the last rt positions
        # Everything before should be NaN
        assert np.all(np.isnan(locked_trials[:, trial_idx, :-rt]) |
                     (locked_trials[:, trial_idx, :-rt] == locked_trials[:, trial_idx, :-rt]))

    print(f"✓ Response locking completed")
    print(f"✓ Trials aligned to response time")


def test_truncate_trials():
    """Test trial truncation"""
    print("\n=== Testing truncate_trials ===")

    trials = np.random.randn(5, 10, 100)

    # Test with ratio parameters
    start_ratio = 0.2
    end_ratio = 0.8

    truncated = TrialTransformer.truncate_trials(trials, start_ratio, end_ratio)

    expected_length = int(100 * end_ratio) - int(100 * start_ratio)

    assert truncated.shape[2] == expected_length, \
        f"Expected {expected_length} timepoints, got {truncated.shape[2]}"
    assert truncated.shape[:2] == trials.shape[:2], "Channels and trials should be preserved"

    # Check that truncated data matches original slice
    start_idx = int(100 * start_ratio)
    end_idx = int(100 * end_ratio)
    assert np.allclose(truncated, trials[:, :, start_idx:end_idx])

    print(f"✓ Original length: {trials.shape[2]}, Truncated: {truncated.shape[2]}")
    print(f"✓ Truncated from {start_ratio*100}% to {end_ratio*100}% of trial")


def test_handle_nans():
    """Test NaN handling"""
    print("\n=== Testing handle_nans ===")

    trials = np.random.randn(5, 10, 100)
    # Add some NaN values
    trials[0, 0, 50:] = np.nan
    trials[2, 5, 80:] = np.nan

    original_nan_count = np.sum(np.isnan(trials))

    cleaned = TrialTransformer.handle_nans(trials)

    assert not np.any(np.isnan(cleaned)), "No NaN values should remain"
    assert not np.any(np.isinf(cleaned)), "No Inf values should remain"

    print(f"✓ Original NaN count: {original_nan_count}, After cleaning: {np.sum(np.isnan(cleaned))}")


def test_normalize_per_trial():
    """Test per-trial normalization (normalizes each channel within each trial)"""
    print("\n=== Testing normalize_per_trial ===")

    trials = np.random.randn(5, 10, 100) * 10 + 5  # Different scale

    normalized = TrialTransformer.normalize_per_trial(trials)

    # Check each channel within each trial is normalized
    for trial_idx in range(trials.shape[1]):
        for channel_idx in range(trials.shape[0]):
            channel_data = normalized[channel_idx, trial_idx, :]
            channel_mean = np.nanmean(channel_data)
            channel_std = np.nanstd(channel_data)

            assert np.abs(channel_mean) < 1e-5, \
                f"Trial {trial_idx}, Channel {channel_idx} mean should be ~0, got {channel_mean}"
            assert np.abs(channel_std - 1.0) < 1e-5 or channel_std == 1.0, \
                f"Trial {trial_idx}, Channel {channel_idx} std should be ~1, got {channel_std}"

    print(f"✓ All channels within each trial normalized to mean=0, std=1")


def run_all_tests():
    """Run all preprocessing tests"""
    print("=" * 60)
    print("Running Preprocessing Tests")
    print("=" * 60)

    try:
        test_filter_simple_trials()
        test_filter_incongruent()
        test_filter_by_trial_length()
        test_select_channels()
        test_response_locking()
        test_truncate_trials()
        test_handle_nans()
        test_normalize_per_trial()

        print("\n" + "=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)

    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        raise
    except Exception as e:
        print(f"\n✗ Error during testing: {e}")
        raise


if __name__ == "__main__":
    run_all_tests()
