import random
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split

from config import batch_size, n_samples

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.manual_seed(seed)

vocab = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N"]


def generate_synthetic_data(num_samples: int, vocab: List[str], max_len: int = 15) -> List[List[Tuple[str, float]]]:
    """Generate synthetic dataset of variable-length (name, value) pairs."""
    data = []
    for _ in range(num_samples):
        L = random.randint(1, max_len + 1)
        sample = []
        for _ in range(L):
            name = random.choice(vocab)
            value = random.uniform(-2.0, 2.0)
            sample.append((name, value))
        data.append(sample)
    return data


class StrFloatDataset(Dataset):
    """Dataset for samples that are list[tuple[str, float]].

    Each sample is a variable-length list of (name, value) pairs. Names are
    drawn from a fixed vocabulary (`names_list`) of length 14. The dataset
    converts inputs to two tensors:
      - names: (n_samples, max_len), dtype=torch.long (0 is padding)
      - values: (n_samples, max_len), dtype=torch.float

    Targets Y have shape (n_samples, output_dim).
    """

    def __init__(self, data: List[List[Tuple[str, float]]], names_list: Optional[List[str]] = None, max_len: int = 15, output_dim: int = 25):
        self.data = data
        self.max_len = max_len
        self.output_dim = output_dim

        self.W = np.random.random((self.output_dim, self.output_dim))
        self.b = np.random.random((self.output_dim,))

        # default vocabulary of 14 names if none provided
        if names_list is None:
            self.names_list = [f"name{i + 1}" for i in range(14)]
        else:
            self.names_list = list(names_list)

        # map name, reserve 0 for padding
        self.name2idx = {n: i + 1 for i, n in enumerate(self.names_list)}

        # build tensors now
        self.names_tensor, self.values_tensor, self.Y = self._build_tensors()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.names_tensor[idx], self.values_tensor[idx], self.Y[idx]

    def _encode_sample(self, sample: List[Tuple[str, float]]) -> Tuple[List[int], List[float]]:
        """Encode a single sample into fixed-length name indices and values."""
        names_idx = [0] * self.max_len
        values = [0.0] * self.max_len

        for i, (n, v) in enumerate(sample[: self.max_len]):
            names_idx[i] = self.name2idx.get(n, 0)
            values[i] = float(v)

        return names_idx, values

    def _compute_targets(self, names: np.ndarray, values: np.ndarray) -> np.ndarray:
        """Construct more complex, sample-independent targets with combinatorial name effects.

        For each sample we build a high-dimensional feature vector that contains:
          - per-name counts and per-name value sums
          - pairwise name-count interactions and pairwise value interactions

        Then each sample is projected to `output_dim` using a sample-specific random
        projection (seeded by the global `seed` and sample index) so that targets
        across different samples are not tied together by a single global transform.
        This also creates strong within-sample combinatorial effects from `names`.
        """
        n_samples, max_len = names.shape
        V = len(self.names_list)

        # per-name counts and per-name summed values
        name_counts = np.zeros((n_samples, V), dtype=float)
        name_value_sums = np.zeros((n_samples, V), dtype=float)
        for i_name in range(V):
            mask = names == (i_name + 1)
            name_counts[:, i_name] = mask.sum(axis=1)
            # sum values where that name appears
            name_value_sums[:, i_name] = (values * mask).sum(axis=1)

        # pairwise combination effects (unordered pairs)
        pair_idx = [(a, b) for a in range(V) for b in range(a + 1, V)]
        num_pairs = len(pair_idx)
        pair_counts = np.zeros((n_samples, num_pairs), dtype=float)
        pair_value_inter = np.zeros((n_samples, num_pairs), dtype=float)
        for p_idx, (a, b) in enumerate(pair_idx):
            pair_counts[:, p_idx] = name_counts[:, a] * name_counts[:, b]
            pair_value_inter[:, p_idx] = name_value_sums[:, a] * name_value_sums[:, b]

        # assemble feature vector per sample
        features = np.concatenate([name_counts, name_value_sums, pair_counts, pair_value_inter], axis=1)
        high_dim = features.shape[1]

        # build targets per-sample using a sample-specific random projection
        Y = np.zeros((n_samples, self.output_dim), dtype=float)
        base_seed = 10007
        for i in range(n_samples):
            rng = np.random.RandomState(seed + base_seed + i)
            P = rng.normal(loc=0.0, scale=1.0, size=(high_dim, self.output_dim))
            b = rng.normal(loc=0.0, scale=0.1, size=(self.output_dim,))
            y = np.tanh(features[i] @ P + b)
            # small per-sample noise
            y += rng.normal(scale=0.01, size=y.shape)
            Y[i] = y

        return Y

    def _build_tensors(self):
        encoded = [self._encode_sample(sample) for sample in self.data]
        names = np.array([ni for ni, _ in encoded], dtype=int)
        values = np.array([vals for _, vals in encoded], dtype=float)
        Y = self._compute_targets(names, values)
        names_tensor = torch.tensor(names, dtype=torch.int).to(device)
        values_tensor = torch.tensor(values, dtype=torch.float).to(device)
        Y_tensor = torch.tensor(Y, dtype=torch.float).to(device)
        return names_tensor, values_tensor, Y_tensor


data = generate_synthetic_data(num_samples=n_samples, vocab=vocab, max_len=15)
ds = StrFloatDataset(data, names_list=vocab)

train_size = int(n_samples * 0.8)
test_size = n_samples - train_size

train_dataset, test_dataset = random_split(ds, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

if __name__ == "__main__":
    data = generate_synthetic_data(num_samples=n_samples, vocab=vocab, max_len=15)
    ds = StrFloatDataset(data, names_list=vocab)
    print("names shape:", ds.names_tensor.shape)
    print("values shape:", ds.values_tensor.shape)
    print("Y shape:", ds.Y.shape)
    # show first sample
    print("first names:", ds.names_tensor[0])
    print("first values:", ds.values_tensor[0])
    print("first Y (trim):", ds.Y[0][:6])
