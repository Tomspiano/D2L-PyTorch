import torch
from modules import base
from modules import fashionMNIST as fmnist

from modules import d2lScratch as scratch

from torch import nn
from torch.nn import init
from modules import d2lCustom as custom


def scratch_ver(num_inputs, num_hiddens, num_outputs, train_iter, test_iter, eps, batch_size):
    # epoch 14, loss 0.0011, train acc 0.897, test acc 0.856, time 20.7 sec
    # if eps = 1e-3, learning rate = 0.5

    lr = 0.5

    W1 = nn.Parameter(torch.randn(num_inputs, num_hiddens, requires_grad=True) * 0.01)
    b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))
    W2 = nn.Parameter(torch.randn(num_hiddens, num_outputs, requires_grad=True) * 0.01)
    b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))

    params = [W1, b1, W2, b2]

    mlp = scratch.MLPNet(params)

    loss = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(params, lr)

    base.train(mlp.net, train_iter, test_iter, loss, eps, batch_size, optimizer=optimizer)


def custom_ver(num_inputs, num_hiddens, num_outputs, train_iter, test_iter, eps, batch_size):
    # epoch 16, loss 0.0010, train acc 0.900, test acc 0.863, time 22.5 sec
    # if eps = 1e-3, learning rate = 0.5

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

    base.train(net, train_iter, test_iter, loss, eps, batch_size, optimizer=optimizer)


def main():
    batch_size = 256

    num_inputs, num_outputs, num_hiddens = 784, 10, 256

    eps = 1e-3
    # eps = 1e-1

    root = '../../Datasets'
    train_iter, test_iter = fmnist.load_data(batch_size, root=root)

    mode = eval(input('0[Scratch Version], 1[Custom Version]: '))
    if mode:
        custom_ver(num_inputs, num_hiddens, num_outputs, train_iter, test_iter, eps, batch_size)
    else:
        scratch_ver(num_inputs, num_hiddens, num_outputs, train_iter, test_iter, eps, batch_size)


if __name__ == '__main__':
    main()
