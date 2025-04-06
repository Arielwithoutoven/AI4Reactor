import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_size = 700
test_size = 300
num_examples = train_size + test_size

num_epochs = 500  # 迭代次数

batch_size = 256  # 一次训练所用样本数

learning_rate = 0.003

input_dim = 1
output_dim = 1

model_name = "CascadedFFN.params"
dataset_name = "dataset"

subnetwork_params = [
    # ([128, 256], 128),
    ([256, 128], 64),
    ([64, 16], 8),
    ([8], output_dim)
]
