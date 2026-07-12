import torch
import torch.nn as nn


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
        emb = self.emb(names) * values.unsqueeze(-1)  # (B, L, emb_dim)
        flat = emb.view(emb.size(0), -1)
        return self.net(flat)


class EmbeddingRNN(nn.Module):
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
        emb = self.emb(names) * values.unsqueeze(-1)  # (B, L, emb_dim)
        out, _ = self.rnn(emb)  # (B, L, 2*rnn_hidden)

        # aggregate by masked mean
        # mask 表示有效位置
        # summed 表示每个样本的有效位置的输出向量的和
        # denom 表示每个样本的有效位置的数量，
        # mean 表示每个样本的有效位置的输出向量的平均值
        mask = (names != 0).unsqueeze(-1).float()  # (B, L, 1)
        summed = (out * mask).sum(dim=1)  # (B, 2*rnn_hidden)
        denom = mask.sum(dim=1).clamp(min=1.0)  # (B, 1)
        mean = summed / denom  # (B, 2*rnn_hidden)
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
