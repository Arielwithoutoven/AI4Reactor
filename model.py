from random import random

import torch
from torch import nn


class SubNet(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.2):
        """hidden_dims (list): 隐藏层维度列表（例如[256, 128]表示两个隐藏层）
        dropout (float, optional): dropout 概率
        """
        super(SubNet, self).__init__()
        layers = []
        current_dim = input_dim

        # 隐藏层
        for h_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, h_dim))
            # layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())  # Todo
            layers.append(nn.Dropout(dropout))
            current_dim = h_dim

        # 输出层
        layers.append(nn.Linear(current_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class CascadedNet(nn.Module):
    def __init__(self, input_dim, subnetwork_params_list):
        """
        Args:
            input_dim: 初始输入维度
            subnetwork_params_list: 每个子网络的参数列表，每个元素为元组：
                (hidden_dims, output_dim)
        """
        super(CascadedNet, self).__init__()
        self.subnetworks = nn.ModuleList()
        current_dim = input_dim

        # 创建子网络
        for hidden_dims, output_dim in subnetwork_params_list:
            subnet = SubNet(current_dim, hidden_dims, output_dim)
            self.subnetworks.append(subnet)
            current_dim = output_dim  # 下一个子网络的输入维度是当前子网络的输出

    def forward(self, x):
        for subnet in self.subnetworks:
            x = subnet(x)
        return x


class CascadedFFN(nn.Module):
    def __init__(self, input_dim, output_dim, activation=nn.ReLU()):
        super(CascadedFFN, self).__init__()
        self.input_dim = input_dim  # 输入特征维度
        self.output_dim = output_dim  # 输出维度
        self.activation = activation  # 隐藏层激活函数
        self.hidden_layers = nn.ModuleList()  # 动态存储隐藏层
        self.output_layer = None  # 动态调整的输出层

    def add_neuron(self):
        """动态添加一个新的隐藏神经元"""
        # 计算新神经元的输入维度: 原始输入 + 所有已有隐藏层输出
        new_input_dim = self.input_dim + len(self.hidden_layers)

        # 创建新的隐藏层神经元（单神经元全连接层）
        new_layer = nn.Linear(new_input_dim, 1)
        self.hidden_layers.append(new_layer)

        # 动态调整输出层：连接所有隐藏层输出
        new_output_input_dim = len(self.hidden_layers)
        self.output_layer = nn.Linear(new_output_input_dim, self.output_dim)

        # 冻结旧隐藏层的参数（仅训练新添加的层）
        for param in self.hidden_layers[:-1].parameters():
            param.requires_grad = False

    def forward(self, x):
        hidden_outputs = []
        current_input = x

        # 逐层计算隐藏层输出
        for i, layer in enumerate(self.hidden_layers):
            if i > 0:
                # 拼接原始输入和之前所有隐藏层输出
                combined_input = torch.cat([x] + hidden_outputs[:i], dim=1)
                current_input = combined_input
            # 计算当前隐藏层输出（含激活函数）
            h = self.activation(layer(current_input))
            hidden_outputs.append(h)

        # 汇总所有隐藏层输出到最终输出层
        final_hidden = torch.cat(hidden_outputs, dim=1)
        output = self.output_layer(final_hidden)
        return output


class ResidualBlock(nn.Module):
    def __init__(self, input_dim, output_dim, dropout, activation):
        # 采用 PreActivation 模式
        super(ResidualBlock, self).__init__()
        self.fc = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            activation,
            nn.Linear(input_dim, output_dim),
            nn.Dropout(dropout),
            nn.BatchNorm1d(output_dim),
            activation,
            nn.Linear(output_dim, output_dim),
        )
        self.activation = activation
        self.short_cut = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()
        self._init_weights()

    def _init_weights(self):
        for layer in self.fc:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, mode="fan_in", nonlinearity="relu")
                nn.init.constant_(layer.bias, 0)
        if isinstance(self.short_cut, nn.Linear):
            nn.init.xavier_normal_(self.short_cut.weight)
            nn.init.constant_(self.short_cut.bias, 0)

    def forward(self, x):
        residual = self.short_cut(x)
        x = self.fc(x)
        x += residual
        return self.activation(x)


class ComplexMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: list = [512, 256, 128, 64],
        dropout: float = 0.3,
        activation: str = nn.GELU(),
    ):
        super(ComplexMLP, self).__init__()

        # 动态构建隐藏层
        layers = []
        prev_dim = input_dim
        for idx, hidden_dim in enumerate(hidden_dims):
            # 添加残差层
            layers.append(ResidualBlock(prev_dim, hidden_dim, dropout=dropout, activation=activation))
            prev_dim = hidden_dim

        self.hidden_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(prev_dim, output_dim)

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        for layer in self.hidden_layers:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, mode="fan_in", nonlinearity="relu")
                nn.init.constant_(layer.bias, 0)
        nn.init.xavier_normal_(self.output_layer.weight)
        nn.init.constant_(self.output_layer.bias, 0)

    def forward(self, x):
        x = self.hidden_layers(x)
        x = self.output_layer(x)
        return x
