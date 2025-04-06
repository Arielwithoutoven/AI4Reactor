import random

import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset

from config import *

# use complex functions
# 1. 1->1
#    y = 5 * exp(cot(x)) + 1
# 2. 2->1
#    y = x1 ** 2 - x2 ** 3 - 6
# 3. 2->2 test


class MyDataset(Dataset):
    def __init__(self, type=1, device=device):
        if type == 1:
            self.features = torch.normal(0, 3, (num_examples, 1))
            self.labels = (
                torch.exp(torch.arctan(self.features)) * 5
                + torch.normal(0, 0.01, (num_examples, 1))
                + 1
            )
        elif type == 2:
            x1 = torch.FloatTensor(num_examples).uniform_(-10, 10)
            x2 = torch.FloatTensor(num_examples).uniform_(-9, 9)
            self.features = torch.stack([x1, x2], dim=1)
            self.labels = (
                1.5 * x1**2 - 2 * x2**2 + 1  # + torch.randn(num_examples) * 0.01
            ).reshape(num_examples, 1)
        else:
            pass
        self.features = self.features.to(device)
        self.labels = self.labels.to(device)
        self.data_iter = self.load_array(
            (self.features[:train_size], self.labels[:train_size]), batch_size
        )
        self.test_iter = self.load_array(
            (self.features[-test_size:], self.labels[-test_size:]),
            batch_size,
            is_train=False,
        )

    def load_array(self, data_arrays, batch_size, is_train=True):
        """return data-sub-set with size of batch_size"""
        dataset = TensorDataset(*data_arrays)
        return DataLoader(dataset, batch_size, shuffle=is_train)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
