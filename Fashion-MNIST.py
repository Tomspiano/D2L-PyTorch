import numpy as np
import torch

from d2lPytorch import fashionMNIST as fmnist

batch_size = 256

num_inputs = 784
num_outputs = 10

fmnist.load_data(batch_size)

W = torch.tensor(np.random.normal(0, .01, (num_inputs, num_outputs)), dtype=torch.float)
b = torch.zeros(num_outputs, dtype=torch.float)

W.requires_grad_()
b.requires_grad_()

y_hat = torch.tensor([[.1, .3, .6], [.3, .2, .3]])
y = torch.LongTensor([0, 2])
y_hat.gather(1, y.view(-1, 1))
