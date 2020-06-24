import torch
import numpy as np
from modules import fashionMNIST as fmnist

from modules import d2lScratch as scratch

import torch.nn as nn
from torch.nn import init
from modules import d2lCustom as custom


def scratch_ver(num_inputs, num_hiddens, num_outputs, train_iter, test_iter, eps, batch_size):
    # epoch 10, loss 0.0012, train acc 0.885, test acc 0.849
    # if eps=1e-3, learning rate = 128 (0.5)

    lr = 0.5 * batch_size

    W1 = torch.tensor(np.random.normal(0, .01, (num_inputs, num_hiddens)), dtype=torch.float)
    b1 = torch.zeros(num_hiddens, dtype=torch.float)
    W2 = torch.tensor(np.random.normal(0, .01, (num_hiddens, num_outputs)), dtype=torch.float)
    b2 = torch.zeros(num_outputs, dtype=torch.float)

    params = [W1, b1, W2, b2]
    for param in params:
        param.requires_grad_()

    loss = nn.CrossEntropyLoss()

    fmnist.train(scratch.mlp, train_iter, test_iter, loss, eps, batch_size, params, lr)


def custom_ver(num_inputs, num_hiddens, num_outputs, train_iter, test_iter, eps, batch_size):
    # epoch 16, loss 0.0010, train acc 0.900, test acc 0.875
    # if eps=1e-3, learning rate = 0.5

    net = nn.Sequential(
            custom.FlattenLayer(),
            nn.Linear(num_inputs, num_hiddens),
            nn.ReLU(),
            nn.Linear(num_hiddens, num_outputs)
    )

    for params in net.parameters():
        init.normal_(params, mean=0, std=.01)

    loss = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(net.parameters(), lr=.5)

    fmnist.train(net, train_iter, test_iter, loss, eps, batch_size, None, None, optimizer)


def main():
    batch_size = 256

    num_inputs = 784
    num_outputs = 10
    num_hiddens = 256

    eps = 1e-3
    # eps = 1e-1

    root = './Datasets'
    train_iter, test_iter = fmnist.load_data(batch_size, root=root)

    mode = eval(input('0[Scratch Version], 1[Custom Version]: '))
    if mode:
        custom_ver(num_inputs, num_hiddens, num_outputs, train_iter, test_iter, eps, batch_size)
    else:
        scratch_ver(num_inputs, num_hiddens, num_outputs, train_iter, test_iter, eps, batch_size)


if __name__ == '__main__':
    main()
