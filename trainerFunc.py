import os

import torch
from d2l import torch as d2l

from config import *
from model import CascadedFFN


class Accumulator:  # @save
    """在n个变量上累加"""

    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def evaluate_loss(net, data_iter, loss):
    """计算在指定数据集上模型的误差"""
    if isinstance(net, torch.nn.Module):
        net.eval()  # 将模型设置为评估模式
    metric = Accumulator(2)  # loss、预测总数
    with torch.no_grad():
        for X, y in data_iter:
            y_hat = net(X)
            assert y_hat.shape == y.shape
            l = loss(y_hat, y)
            metric.add(l.sum(), y.numel())
    return metric[0] / metric[1]


def train_epoch_ch(net, train_iter, loss, updater):  # @save
    """训练模型一个迭代周期（定义见第3章）"""
    if isinstance(net, torch.nn.Module):
        net.train()
    # 训练损失总和、样本数
    metric = Accumulator(2)
    for X, y in train_iter:
        # 计算梯度并更新参数
        y_hat = net(X)
        assert y_hat.shape == y.shape
        l = loss(y_hat, y)
        metric.add(float(l.sum()), y.numel())
        if isinstance(updater, torch.optim.Optimizer):
            # 使用PyTorch内置的优化器和损失函数
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            # 使用定制的优化器和损失函数
            l.sum().backward()
            updater(X.shape[0])
    # 返回训练损失
    return (metric[0] / metric[1],)


def train_ch(net, train_iter, test_iter, loss, num_epochs, updater):
    animator = d2l.Animator(
        xlabel="epoch",
        ylabel="loss",
        yscale="log",
        xlim=[1, num_epochs],
        ylim=[1e-2, 1e4],
        legend=["train", "test"],
        figsize=(14.0, 5.0)
    )
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch(net, train_iter, loss, updater)
        test_loss = evaluate_loss(net, test_iter, loss)
        animator.add(epoch + 1, train_metrics + (test_loss,))
    # (train_loss,) = train_metrics
    # assert train_loss < 0.5, train_loss


if __name__ == "__main__":
    if os.path.exists(model_name):
        CFFN = CascadedFFN().load_state_dict(torch.load(model_name))
    else:
        CFFN = CascadedFFN(input_dim, output_dim)

    torch.save(CFFN.parameters, model_name)
