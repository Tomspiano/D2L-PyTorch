import torch
from modules import base
from modules import fashionMNIST as fmnist
from torch import nn
from torch.nn import init
from modules import d2lCustom as custom


def train(num_inputs, num_hiddens, num_outputs, train_iter, test_iter, eps):
    # epoch 46, loss 0.155, train acc 0.943, test acc 0.883, 546.2 examples/sec
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

    base.train(net, train_iter, test_iter, loss, eps=eps, num_epochs=50, optimizer=optimizer)


def main():
    batch_size = 256

    num_inputs, num_outputs, num_hiddens = 784, 10, 256

    eps = 1e-3
    # eps = 1e-1

    root = '../../Datasets'
    train_iter, test_iter = fmnist.load_data(batch_size, root=root)

    train(num_inputs, num_hiddens, num_outputs, train_iter, test_iter, eps)


if __name__ == '__main__':
    main()
