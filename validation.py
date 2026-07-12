import torch

from main import test_loader

criterion = torch.nn.MSELoss()


# 绘制 loss 分布（每个样本的 MSE）
def plot_loss_distribution(model, test_loader, criterion, bins=50, save_name=None):
    import matplotlib.pyplot as plt
    import numpy as np

    model.eval()
    losses = []
    with torch.no_grad():
        for *X, targets in test_loader:
            outputs = model(*X)
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
    plt.title(f"Loss Distribution ({model.__class__.__name__})")
    if save_name is None:
        save_name = f"{model.__class__.__name__}_loss_distribution.png"
    plt.savefig(save_name)
    plt.close()


# model = torch.load("MLPBaseline.pth", weights_only=False)
# model = torch.load("EmbeddingRNNModel.pth", weights_only=False)
model = torch.load("TransformerSetModel.pth", weights_only=False)
plot_loss_distribution(model, test_loader, criterion)
