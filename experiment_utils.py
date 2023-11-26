"""Utility functions for experiments."""

from torch.utils.data import Dataset


class UtilDataset(Dataset):
    """A simple Dataset class to store (fake) images."""

    def __init__(self, images, labels) -> None:
        super().__init__()
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]
