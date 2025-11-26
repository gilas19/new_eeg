import numpy as np
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split


class EEGDataModule:
    def __init__(self, dataset, batch_size=32, val_split=0.2, seed=42):
        self.dataset = dataset
        self.batch_size = batch_size
        self.val_split = val_split
        self.seed = seed

        self._split_data()

    def _split_data(self):
        subject_indices = self.dataset.get_subject_indices()
        subjects = list(subject_indices.keys())
        n_subjects = len(subjects)

        if n_subjects < 2:
            print(f"Warning: Only {n_subjects} subject(s) found. Using trial-based split instead of subject-based split.")
            all_indices = list(range(len(self.dataset)))

            train_indices, val_indices = train_test_split(
                all_indices, test_size=self.val_split, random_state=self.seed
            )

            self.train_indices = train_indices
            self.val_indices = val_indices
        else:
            train_subjects, val_subjects = train_test_split(
                subjects, test_size=self.val_split, random_state=self.seed
            )

            self.train_indices = []
            for subj in train_subjects:
                self.train_indices.extend(subject_indices[subj])

            self.val_indices = []
            for subj in val_subjects:
                self.val_indices.extend(subject_indices[subj])

    def train_dataloader(self, shuffle=True):
        train_dataset = Subset(self.dataset, self.train_indices)
        return DataLoader(train_dataset, batch_size=self.batch_size, shuffle=shuffle)

    def val_dataloader(self):
        val_dataset = Subset(self.dataset, self.val_indices)
        return DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
