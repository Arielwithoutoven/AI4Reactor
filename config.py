import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_size = 1400
test_size = 600
num_examples = train_size + test_size

num_epochs = 500  # 迭代次数

batch_size = 512  # 一次训练所用样本数

learning_rate = 0.005
