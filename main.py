import torch
import torch.optim as optim
from torch.nn import MSELoss

from config import device, learning_rate, num_epochs
from dataset import test_loader, train_loader
from models import EmbeddingRNN, MLPBaseline, TransformerSetModel


# 训练函数
def train(model, optimizer, train_iter, test_iter, num_epochs=num_epochs):
    train_losses = []
    test_losses = []
    with open(f"{model.__class__.__name__}.txt", "w") as f:
        for epoch in range(num_epochs):
            model.train()
            train_loss = 0.0
            for *X, targets in train_iter:
                optimizer.zero_grad()
                outputs = model(*X)
                _train_loss = criterion(outputs, targets)
                _train_loss.backward()
                train_loss += _train_loss.item()
                optimizer.step()
            train_loss /= len(train_iter)
            train_losses.append(train_loss)

            model.eval()
            test_loss = 0.0
            with torch.no_grad():
                for *X, targets in test_iter:
                    outputs = model(*X)
                    test_loss += criterion(outputs, targets).item()
            test_loss /= len(test_iter)
            test_losses.append(test_loss)
            if (epoch + 1) % 50 == 0:
                f.write(f"Epoch [{epoch + 1}/{num_epochs}], TrainLoss: {train_loss:.4f}, TestLoss: {test_loss:.4f}\n")
            if (epoch + 1) % 50 == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}], TrainLoss: {train_loss:.4f}, TestLoss: {test_loss:.4f}")
    return train_losses, test_losses


def plot_loss_distribution(model_instance, test_loader, criterion, bins=50, save_name=None):
    import matplotlib.pyplot as plt
    import numpy as np

    model_instance.eval()
    losses = []
    with torch.no_grad():
        for *X, targets in test_loader:
            outputs = model_instance(*X)
            diff = outputs - targets
            if diff.dim() > 1:
                per_sample = diff.view(diff.size(0), -1).pow(2).mean(dim=1)
            else:
                per_sample = diff.pow(2)
            losses.extend(per_sample.cpu().numpy())

    losses = np.array(losses)
    plt.figure(figsize=(8, 5))
    plt.hist(losses, bins=bins, alpha=0.7)
    plt.xlabel("MSE per sample")
    plt.ylabel("Count")
    plt.title(f"Loss Distribution ({model.__name__})")
    if save_name is None:
        save_name = f"{model.__name__}_loss_distribution.png"
    plt.savefig(save_name)
    plt.close()


# 绘制训练和测试损失曲线
def plot_loss_curve(train_losses, test_losses, path):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(test_losses, label="Test Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Testing Loss Curve")
    plt.yscale("log")
    plt.legend()
    plt.savefig(path)
    plt.close()


if __name__ == "__main__":
    for model in [MLPBaseline, EmbeddingRNN, TransformerSetModel]:
        print(f"Training model: {model.__name__}")
        # 损失函数
        criterion = MSELoss()
        # 模型
        model_instance = model().to(device)
        # 优化器
        optimizer = optim.Adam(model_instance.parameters(), lr=learning_rate)
        train_losses, test_losses = train(model_instance, optimizer, train_loader, test_loader)

        torch.save(model_instance, f"{model.__name__}.pth")

        path = f"{model.__name__}_loss_curve.png"
        plot_loss_curve(train_losses, test_losses, path)

        # 绘制 loss 分布（每个样本的 MSE）
        plot_loss_distribution(model_instance, test_loader, criterion)
