import random
import os
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset

from config import *

# use complex functions
# 1. 1->1
#    y = 5 * exp(cot(x)) + 1
# 2. 2->1
#    y = x1 ** 2 - x2 ** 2 + 1
# 3. 2->2 test
#    (y1, y2)

DATASET_PATH = "./dataset/"
DATASET_NAMES = ["一元函数", "二元函数", "矩阵函数"]
EXTENSION = ".data"


def generate_dataset(type):
    if type == 1:
        features = torch.normal(0, 3, (num_examples, 1))
        labels = torch.exp(torch.arctan(features)) * 5 + torch.normal(0, 0.01, (num_examples, 1)) + 1
    elif type == 2:
        x1 = torch.FloatTensor(num_examples).uniform_(-20, 20)
        x2 = torch.FloatTensor(num_examples).uniform_(-15, 15)
        features = torch.stack([x1, x2], dim=1)
        labels = (1.5 * x1**2 - 2 * x2**2 + 1).reshape(num_examples, 1)  # + torch.randn(num_examples) * 0.01
    else:
        pass
    torch.save(features, DATASET_PATH + DATASET_NAMES[type - 1] + "features" + EXTENSION)
    torch.save(labels, DATASET_PATH + DATASET_NAMES[type - 1] + "labels" + EXTENSION)


def load_dataset(type, device, generate=False):
    if (
        not os.path.exists(DATASET_PATH + DATASET_NAMES[type - 1] + "features" + EXTENSION)
        or not os.path.exists(DATASET_PATH + DATASET_NAMES[type - 1] + "labels" + EXTENSION)
        or generate
    ):
        generate_dataset(type)

    features = torch.load(DATASET_PATH + DATASET_NAMES[type - 1] + "features" + EXTENSION).to(device)
    labels = torch.load(DATASET_PATH + DATASET_NAMES[type - 1] + "labels" + EXTENSION).to(device)
    train_iter = load_array((features[:train_size], labels[:train_size]), batch_size, is_train=True)
    test_iter = load_array(
        (features[-test_size:], labels[-test_size:]),
        batch_size,
        is_train=False,
    )
    return train_iter, test_iter


def load_array(data_arrays, batch_size, is_train):
    """return data-sub-set with size of batch_size"""
    dataset = TensorDataset(*data_arrays)
    return DataLoader(dataset, batch_size, shuffle=is_train)
