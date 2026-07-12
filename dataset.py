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
        """Construct richer, order-invariant targets with strong within-sample combinational effects.

        Each sample is summarized by features that do not depend on the sequence order of
        names. Instead, the targets rely on:
          - per-name counts and aggregated value statistics
          - pairwise name interactions
          - triple name interactions
          - global sample-level statistics

        This makes the target mapping permutation-invariant and much harder to fit.
        """
        n_samples, _ = names.shape
        V = len(self.names_list)

        # Per-name statistics
        name_counts = np.zeros((n_samples, V), dtype=float)
        name_value_sums = np.zeros((n_samples, V), dtype=float)
        name_value_abs_sums = np.zeros((n_samples, V), dtype=float)
        name_positive_sums = np.zeros((n_samples, V), dtype=float)
        name_negative_sums = np.zeros((n_samples, V), dtype=float)

        for i_name in range(V):
            mask = names == (i_name + 1)
            name_counts[:, i_name] = mask.sum(axis=1)
            name_value_sums[:, i_name] = (values * mask).sum(axis=1)
            name_value_abs_sums[:, i_name] = (np.abs(values) * mask).sum(axis=1)
            name_positive_sums[:, i_name] = np.where(mask, np.maximum(values, 0.0), 0.0).sum(axis=1)
            name_negative_sums[:, i_name] = np.where(mask, np.minimum(values, 0.0), 0.0).sum(axis=1)

        # Pairwise interactions (unordered pairs)
        pair_idx = [(a, b) for a in range(V) for b in range(a + 1, V)]
        num_pairs = len(pair_idx)
        pair_count_inter = np.zeros((n_samples, num_pairs), dtype=float)
        pair_value_inter = np.zeros((n_samples, num_pairs), dtype=float)
        pair_cross_inter = np.zeros((n_samples, num_pairs), dtype=float)
        pair_presence_inter = np.zeros((n_samples, num_pairs), dtype=float)
        for p_idx, (a, b) in enumerate(pair_idx):
            pair_count_inter[:, p_idx] = name_counts[:, a] * name_counts[:, b]
            pair_value_inter[:, p_idx] = name_value_sums[:, a] * name_value_sums[:, b]
            pair_cross_inter[:, p_idx] = name_counts[:, a] * name_value_sums[:, b] + name_counts[:, b] * name_value_sums[:, a]
            pair_presence_inter[:, p_idx] = ((name_counts[:, a] > 0) & (name_counts[:, b] > 0)).astype(float)

        # Triple interactions (unordered triples)
        triple_idx = [(a, b, c) for a in range(V) for b in range(a + 1, V) for c in range(b + 1, V)]
        num_triples = len(triple_idx)
        triple_count_inter = np.zeros((n_samples, num_triples), dtype=float)
        triple_value_inter = np.zeros((n_samples, num_triples), dtype=float)
        triple_presence_inter = np.zeros((n_samples, num_triples), dtype=float)
        for t_idx, (a, b, c) in enumerate(triple_idx):
            triple_count_inter[:, t_idx] = name_counts[:, a] * name_counts[:, b] * name_counts[:, c]
            triple_value_inter[:, t_idx] = name_value_sums[:, a] * name_value_sums[:, b] * name_value_sums[:, c]
            triple_presence_inter[:, t_idx] = ((name_counts[:, a] > 0) & (name_counts[:, b] > 0) & (name_counts[:, c] > 0)).astype(float)

        # Global sample statistics
        sample_length = (names > 0).sum(axis=1).astype(float)
        total_value_sum = values.sum(axis=1)
        total_value_abs = np.abs(values).sum(axis=1)
        total_value_sq = np.square(values).sum(axis=1)
        value_variance = np.var(values, axis=1)

        # Assemble order-invariant feature vector
        features = np.concatenate(
            [
                name_counts,
                name_value_sums,
                name_value_abs_sums,
                name_positive_sums,
                name_negative_sums,
                pair_count_inter,
                pair_value_inter,
                pair_cross_inter,
                pair_presence_inter,
                triple_count_inter,
                triple_value_inter,
                triple_presence_inter,
                sample_length[:, None],
                total_value_sum[:, None],
                total_value_abs[:, None],
                total_value_sq[:, None],
                value_variance[:, None],
            ],
            axis=1,
        )
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        # direct sample-level target construction based on counts and value statistics
        # Output dimension is formed by concatenating:
        #   - per-name count features (V dims)
        #   - per-name total value features (V dims)
        #   - per-name absolute value totals (V dims)
        #   - sample length, value sum, abs sum, square sum, variance
        # If output_dim is larger than this base size, we repeat or tile values.
        V = len(self.names_list)
        base_features = np.concatenate(
            [
                name_counts,
                name_value_sums,
                name_value_abs_sums,
                sample_length[:, None],
                total_value_sum[:, None],
                total_value_abs[:, None],
                total_value_sq[:, None],
                value_variance[:, None],
            ],
            axis=1,
        )
        base_features = np.nan_to_num(base_features, nan=0.0, posinf=0.0, neginf=0.0)

        target_dim = self.output_dim
        if base_features.shape[1] >= target_dim:
            Y = base_features[:, :target_dim]
        else:
            repeats = int(np.ceil(target_dim / base_features.shape[1]))
            Y = np.tile(base_features, (1, repeats))[:, :target_dim]

        # normalize each target dimension to roughly [-1, 1]
        Y_mean = Y.mean(axis=0, keepdims=True)
        Y_std = Y.std(axis=0, keepdims=True)
        Y_std[Y_std < 1e-6] = 1.0
        Y = (Y - Y_mean) / Y_std

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
    # tensors may be on GPU
    names_t = ds.names_tensor
    values_t = ds.values_tensor
    Y_t = ds.Y
    print("names shape:", names_t.shape)
    print("values shape:", values_t.shape)
    print("Y shape:", Y_t.shape)
    # show first sample
    print("first names:", names_t[0])
    print("first values:", values_t[0])
    print("first Y (trim):", Y_t[0][:6])

    # compute and print statistics for Y
    Y = Y_t.cpu().numpy() if isinstance(Y_t, torch.Tensor) else np.array(Y_t)
    overall_mean = float(Y.mean())
    overall_std = float(Y.std())
    per_dim_mean = Y.mean(axis=0)
    per_dim_std = Y.std(axis=0)
    y_min = float(Y.min())
    y_max = float(Y.max())
    p05 = float(np.percentile(Y, 5))
    p95 = float(np.percentile(Y, 95))

    print(f"Y overall mean: {overall_mean:.6f}, std: {overall_std:.6f}")
    print(f"Y min/max: {y_min:.6f} / {y_max:.6f}")
    print(f"Y 5/95 percentiles: {p05:.6f} / {p95:.6f}")
    print("Y per-dim mean (first 8):", np.round(per_dim_mean[:8], 6))
    print("Y per-dim std  (first 8):", np.round(per_dim_std[:8], 6))

    # save summary to file
    np.savez(
        "Y_stats.npz",
        overall_mean=overall_mean,
        overall_std=overall_std,
        per_dim_mean=per_dim_mean,
        per_dim_std=per_dim_std,
        y_min=y_min,
        y_max=y_max,
        p05=p05,
        p95=p95,
    )
    print("Saved Y statistics to Y_stats.npz")
