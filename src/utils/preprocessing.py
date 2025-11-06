import numpy as np
import logging

logger = logging.getLogger(__name__)


class TrialFilter:
    @staticmethod
    def filter_simple_trials(primes, cues, reactions):
        simple_patterns = [
            (1, 1, 1),  # RRR
            (2, 2, 2),  # LLL
            (3, 1, 1),  # NRR
            (3, 2, 2),  # NLL
        ]
        mask = np.zeros(len(primes), dtype=bool)
        for p, c, r in simple_patterns:
            mask |= (primes == p) & (cues == c) & (reactions == r)

        n_kept = np.sum(mask)
        n_total = len(primes)
        logger.info(f"Filter simple trials: kept {n_kept}/{n_total} trials ({100*n_kept/n_total:.1f}%)")
        return mask

    @staticmethod
    def filter_incongruent(primes, cues):
        mask = primes != cues
        n_kept = np.sum(mask)
        n_total = len(primes)
        logger.info(f"Filter incongruent: kept {n_kept}/{n_total} trials ({100*n_kept/n_total:.1f}%)")
        return mask

    @staticmethod
    def filter_by_trial_length(trial_lengths, min_length=None, max_length=None):
        mask = np.ones(len(trial_lengths), dtype=bool)
        if min_length is not None:
            mask &= trial_lengths >= min_length
        if max_length is not None:
            mask &= trial_lengths <= max_length

        n_kept = np.sum(mask)
        n_total = len(trial_lengths)
        logger.info(f"Filter by length (min={min_length}, max={max_length}): kept {n_kept}/{n_total} trials ({100*n_kept/n_total:.1f}%)")
        return mask


class ChannelSelector:
    @staticmethod
    def select_channels(electrode_names, channel_names):
        channel_indices = []
        missing_channels = []
        # Convert all names to uppercase for case-insensitive matching
        electrode_names_upper = np.array([e.upper().strip('.') for e in electrode_names])

        for name in channel_names:
            name_upper = name.upper().strip('.')
            matches = np.where(electrode_names_upper == name_upper)[0]
            if len(matches) > 0:
                channel_indices.append(matches[0])
            else:
                missing_channels.append(name)

        n_selected = len(channel_indices)
        n_requested = len(channel_names)
        logger.info(f"Channel selection: selected {n_selected}/{n_requested} channels")
        if missing_channels:
            logger.warning(f"Missing channels not found in data: {missing_channels}")

        return np.array(channel_indices)


class TrialTransformer:
    @staticmethod
    def response_locking(trials, response_times):
        locked_trials = np.full_like(trials, np.nan)
        n_channels, n_trials, n_timepoints = trials.shape

        valid_trials = 0
        for trial_idx in range(n_trials):
            rt = int(response_times[trial_idx])
            if rt <= 0 or rt >= n_timepoints:
                continue

            trial_data = trials[:, trial_idx, :rt]
            locked_trials[:, trial_idx, -rt:] = trial_data
            valid_trials += 1

        logger.info(f"Response locking: locked {valid_trials}/{n_trials} trials with valid response times")
        return locked_trials

    @staticmethod
    def truncate_trials(trials, start_timepoint=None, end_timepoint=None):
        n_channels, n_trials, n_timepoints = trials.shape
        start_idx = start_timepoint if start_timepoint is not None else 0
        end_idx = end_timepoint if end_timepoint is not None else n_timepoints

        truncated = trials[:, :, start_idx:end_idx]
        new_length = truncated.shape[2]
        logger.info(f"Truncate trials: {n_timepoints} -> {new_length} timepoints (start={start_idx}, end={end_idx})")
        return truncated

    @staticmethod
    def handle_nans(trials, fill_value=0.0):
        return np.nan_to_num(trials, nan=fill_value, posinf=fill_value, neginf=fill_value)

    @staticmethod
    def normalize_per_trial(trials):
        normalized = np.copy(trials)
        n_channels, n_trials, n_timepoints = trials.shape

        for trial_idx in range(n_trials):
            trial_data = trials[:, trial_idx, :]
            mean = np.nanmean(trial_data, axis=1, keepdims=True)
            std = np.nanstd(trial_data, axis=1, keepdims=True)
            std[std == 0] = 1.0
            std[np.isnan(std)] = 1.0
            mean[np.isnan(mean)] = 0.0
            normalized[:, trial_idx, :] = (trial_data - mean) / std

        logger.info(f"Normalize per trial: normalized {n_trials} trials (per-channel z-score)")
        return normalized


def apply_preprocessing(trials, electrode_names, primes, cues, reactions,
                       response_times, subjects, config):
    valid_mask = np.ones(trials.shape[1], dtype=bool)

    if config.get('filter_simple_trials', False):
        valid_mask &= TrialFilter.filter_simple_trials(primes, cues, reactions)

    if config.get('filter_incongruent', False):
        valid_mask &= TrialFilter.filter_incongruent(primes, cues)

    if config.get('filter_by_length', {}).get('enabled', False):
        min_len = config['filter_by_length'].get('min_length')
        max_len = config['filter_by_length'].get('max_length')
        valid_mask &= TrialFilter.filter_by_trial_length(response_times, min_len, max_len)

    trials = trials[:, valid_mask, :]
    primes = primes[valid_mask]
    cues = cues[valid_mask]
    reactions = reactions[valid_mask]
    response_times = response_times[valid_mask]
    subjects = subjects[valid_mask]

    if config.get('select_channels', {}).get('enabled', False):
        channel_names = config['select_channels'].get('channels', [])
        if channel_names:
            channel_indices = ChannelSelector.select_channels(electrode_names, channel_names)
            trials = trials[channel_indices, :, :]
            electrode_names = electrode_names[channel_indices]

    if config.get('response_locking', False):
        trials = TrialTransformer.response_locking(trials, response_times)

    if config.get('truncate', {}).get('enabled', False):
        start_timepoint = config['truncate'].get('start_timepoint', None)
        end_timepoint = config['truncate'].get('end_timepoint', None)
        trials = TrialTransformer.truncate_trials(trials, start_timepoint, end_timepoint)

    if config.get('handle_nans', False):
        trials = TrialTransformer.handle_nans(trials, fill_value=0.0)

    if config.get('normalize_per_trial', False):
        trials = TrialTransformer.normalize_per_trial(trials)

    return trials, electrode_names, primes, cues, reactions, response_times, subjects, valid_mask
