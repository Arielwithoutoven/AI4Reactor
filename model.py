from random import random

import torch
from torch import nn


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


