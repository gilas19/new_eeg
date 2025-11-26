import numpy as np
import torch
from torch.utils.data import Dataset
from src.utils.preprocessing import apply_preprocessing


class EEGDataset(Dataset):
    def __init__(self, data_dir, task='right_left', preprocessing_config=None):
        self.task = task
        self.preprocessing_config = preprocessing_config or {}

        self.trials = np.load(f"{data_dir}/trials_dataset.npy")
        self.electrode_names = np.load(f"{data_dir}/electrodes_names.npy")
        self.reactions = np.load(f"{data_dir}/true_labels.npy")
        self.primes = np.load(f"{data_dir}/primes.npy")
        self.cues = np.load(f"{data_dir}/cues.npy")
        self.subjects = np.load(f"{data_dir}/subjects_through_trials.npy")

        self.response_times = self._calculate_response_times()

        if self.preprocessing_config.get('enabled', False):
            self._apply_preprocessing()
        else:
            self._default_normalization()

        self.labels = self._generate_labels()
        self.valid_indices = self._get_valid_indices()

    def _calculate_response_times(self):
        _, n_trials, n_timepoints = self.trials.shape
        response_times = np.zeros(n_trials, dtype=np.int32)

        for trial_idx in range(n_trials):
            trial_data = self.trials[:, trial_idx, :]
            nan_mask = np.isnan(trial_data)

            if np.any(nan_mask):
                first_nan_per_channel = np.argmax(nan_mask, axis=1)
                first_nan_per_channel[~nan_mask.any(axis=1)] = n_timepoints
                response_times[trial_idx] = np.min(first_nan_per_channel)
            else:
                response_times[trial_idx] = n_timepoints

        return response_times

    def _apply_preprocessing(self):
        result = apply_preprocessing(
            self.trials,
            self.electrode_names,
            self.primes,
            self.cues,
            self.reactions,
            self.response_times,
            self.subjects,
            self.preprocessing_config
        )

        (self.trials, self.electrode_names, self.primes, self.cues,
         self.reactions, self.response_times, self.subjects, _) = result

    def _default_normalization(self):
        mean = np.nanmean(self.trials, axis=(1, 2), keepdims=True)
        std = np.nanstd(self.trials, axis=(1, 2), keepdims=True)
        std[std == 0] = 1.0
        std[np.isnan(std)] = 1.0
        mean[np.isnan(mean)] = 0.0
        self.trials = (self.trials - mean) / std
        self.trials = np.nan_to_num(self.trials, nan=0.0, posinf=0.0, neginf=0.0)

    def _generate_labels(self):
        if self.task == 'right_left':
            return (self.reactions == 1).astype(np.int64)

        elif self.task == 'free_instructed':
            return (self.cues == 4).astype(np.int64)

        elif self.task == 'congruent_incongruent':
            congruent = (self.primes == self.cues) & (self.cues == self.reactions)
            incongruent = (self.primes != self.cues) & (self.cues == self.reactions)
            valid = congruent | incongruent
            labels = np.zeros(len(self.reactions), dtype=np.int64)
            labels[congruent] = 1
            labels[incongruent] = 0
            return labels

        else:
            raise ValueError(f"Unknown task: {self.task}")

    def _get_valid_indices(self):
        if self.task == 'congruent_incongruent':
            congruent = (self.primes == self.cues) & (self.cues == self.reactions)
            incongruent = (self.primes != self.cues) & (self.cues == self.reactions)
            return np.where(congruent | incongruent)[0]
        else:
            return np.arange(len(self.reactions))

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        trial = self.trials[:, real_idx, :]
        label = self.labels[real_idx]

        trial_tensor = torch.from_numpy(trial).float()
        label_tensor = torch.tensor(label, dtype=torch.long)

        return trial_tensor, label_tensor

    def get_subject_indices(self):
        subject_indices = {}
        for idx in range(len(self)):
            real_idx = self.valid_indices[idx]
            subject = self.subjects[real_idx]
            if subject not in subject_indices:
                subject_indices[subject] = []
            subject_indices[subject].append(idx)
        return subject_indices
