import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


num_epochs = 500  # 迭代次数
n_samples = 7000  # 样本总数
batch_size = 128  # 一次训练所用样本数

learning_rate = 1e-4
