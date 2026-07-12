import torch

from main import test_loader

criterion = torch.nn.MSELoss()


# model = torch.load("MLPBaseline.pth", weights_only=False)
# model = torch.load("EmbeddingRNNModel.pth", weights_only=False)
# model = torch.load("TransformerSetModel.pth", weights_only=False)
plot_loss_distribution(model, test_loader, criterion)
