import torch
from torch import nn
from modules import base
from modules import fashionMNIST as fmnist
from modules import d2lCustom as custom


def main():
    batch_size = 256

    lr, eps = .001, 1e-3
    # lr, eps = .001, 1e-1

    root = '../../Datasets'
    train_iter, test_iter = fmnist.load_data(batch_size, root=root)

    net = custom.LeNet()
    optimizer = torch.optim.Adam(net.parameters(), lr)
    loss = nn.CrossEntropyLoss()

    base.train(net, train_iter, test_iter, loss, eps=eps, num_epochs=50, optimizer=optimizer)
    # epoch 29, loss 0.338, train acc 0.875, test acc 0.863, 93.3 examples/sec
    # if eps = 1e-3, learning rate = 0.001


if __name__ == '__main__':
    main()
