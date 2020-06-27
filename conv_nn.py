import torch
from torch import nn
from modules import base
from modules import fashionMNIST as fmnist
from modules import d2lCustom as custom


def main():
    batch_size = 256

    lr, eps = .001, 1e-3
    # lr, eps = .001, 1e-1

    root = './Datasets'
    train_iter, test_iter = fmnist.load_data(batch_size, root=root)

    net = custom.LeNet()
    optimizer = torch.optim.Adam(net.parameters(), lr)
    loss = nn.CrossEntropyLoss()

    base.train(net, train_iter, test_iter, loss, eps, batch_size, optimizer=optimizer)
    # epoch 23, loss 0.0015, train acc 0.859, test acc 0.851
    # if eps = 1e-3, learning rate = 0.001


if __name__ == '__main__':
    main()
