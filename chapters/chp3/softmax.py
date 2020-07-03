import torch
from modules import base
from modules import fashionMNIST as fmnist
from torch import nn
from torch.nn import init
from collections import OrderedDict
from modules import d2lCustom as custom


def train(num_inputs, num_outputs, train_iter, test_iter, eps):
    # epoch 11, loss 0.443, train acc 0.849, test acc 0.833, 9418.4 examples/sec
    # if eps = 1e-3, learning rate = 0.1

    net = nn.Sequential(
            OrderedDict([
                ('flatten', custom.FlattenLayer()),
                ('linear', nn.Linear(num_inputs, num_outputs))
            ])
    )

    init.normal_(net.linear.weight, mean=0, std=.01)
    init.constant_(net.linear.bias, val=0)

    loss = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(net.parameters(), lr=.1)

    base.train(net, train_iter, test_iter, loss, eps=eps, optimizer=optimizer)

    return net


def main():
    batch_size = 256

    num_inputs, num_outputs = 784, 10

    eps = 1e-3
    # eps = 1e-1

    root = '../../Datasets'
    train_iter, test_iter = fmnist.load_data(batch_size, root=root)

    X, y = next(iter(test_iter))
    true_labels = fmnist.get_labels(y.numpy())

    net = train(num_inputs, num_outputs, train_iter, test_iter, eps)
    pred_labels = fmnist.get_labels(net(X).argmax(dim=1).numpy())

    titles = []
    for true, pred in zip(true_labels, pred_labels):
        title = true + '\n'
        if true != pred:
            title += 'X: ' + pred
        titles.append(title)

    fmnist.show(X, nrows=5, ncols=10, titles=titles)


if __name__ == '__main__':
    main()
