import torch
import torch.nn as nn


class RBFLayer(nn.Module):
    def __init__(self, input_dim, num_centers, sigma=None):
        super(RBFLayer, self).__init__()
        self.input_dim = input_dim
        self.num_centers = num_centers

        # 初始化中心点（可学习参数）
        self.centers = nn.Parameter(torch.randn(num_centers, input_dim))

        # 宽度参数（可学习或固定）
        if sigma is None:
            self.sigma = nn.Parameter(torch.ones(1))  # 共享的可学习 sigma
        else:
            self.sigma = sigma  # 固定值（如预计算的平均距离）

    def forward(self, x):
        # 计算输入与中心点的欧氏距离（广播机制）
        # x.shape: (batch_size, input_dim)
        # centers.shape: (num_centers, input_dim)
        x_expanded = x.unsqueeze(1)  # (batch_size, 1, input_dim)
        centers_expanded = self.centers.unsqueeze(0)  # (1, num_centers, input_dim)
        distances = torch.norm(x_expanded - centers_expanded, dim=2)  # (batch_size, num_centers)

        # 高斯径向基函数
        return torch.exp(-(distances**2) / (2 * self.sigma**2))


class MLPBaseline(nn.Module):
    """Baseline MLP that embeds names, scales by values, flattens and predicts Y."""

    def __init__(self, vocab_size=15, emb_dim=16, max_len=15, hidden=128, output_dim=25):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.max_len = max_len
        self.flatten_dim = emb_dim * max_len
        self.net = nn.Sequential(
            nn.Linear(self.flatten_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, output_dim),
        )

    def forward(self, names, values):
        # names: (B, L), values: (B, L)
        emb = self.emb(names)  # (B, L, emb_dim)
        scaled = emb * values.unsqueeze(-1)  # broadcast
        flat = scaled.view(scaled.size(0), -1)
        return self.net(flat)


class EmbeddingRNNModel(nn.Module):
    """Embed names, scale by values, run a bi-GRU, aggregate and predict."""

    def __init__(self, vocab_size=15, emb_dim=32, rnn_hidden=64, output_dim=25):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.rnn = nn.GRU(input_size=emb_dim, hidden_size=rnn_hidden, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(
            nn.Linear(rnn_hidden * 2, rnn_hidden),
            nn.ReLU(),
            nn.Linear(rnn_hidden, output_dim),
        )

    def forward(self, names, values):
        # names: (B, L), values: (B, L)
        mask = names != 0
        emb = self.emb(names) * values.unsqueeze(-1)
        out, _ = self.rnn(emb)  # (B, L, 2*rnn_hidden)

        # aggregate by masked mean
        mask_f = mask.unsqueeze(-1).float()
        summed = (out * mask_f).sum(dim=1)
        denom = mask_f.sum(dim=1).clamp(min=1.0)
        mean = summed / denom
        return self.head(mean)


class TransformerSetModel(nn.Module):
    """Transformer-style model treating the sequence as a set with attention.

    Uses a small Transformer encoder with padding mask, then aggregates.
    """

    def __init__(self, vocab_size=15, emb_dim=32, d_model=64, nhead=4, num_layers=2, output_dim=25):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.project = nn.Linear(emb_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=128, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, output_dim))

    def forward(self, names, values):
        # names: (B, L), values: (B, L)
        mask = names == 0  # True where padded
        emb = self.emb(names) * values.unsqueeze(-1)
        x = self.project(emb)
        # transformer accepts src_key_padding_mask: (B, L) with True for padding
        x = self.transformer(x, src_key_padding_mask=mask)

        # masked mean pooling
        mask_f = (~mask).unsqueeze(-1).float()
        summed = (x * mask_f).sum(dim=1)
        denom = mask_f.sum(dim=1).clamp(min=1.0)
        mean = summed / denom
        return self.head(mean)
