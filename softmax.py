import torch
import numpy as np
from modules import fashionMNIST as fmnist

from modules import d2lScratch as scratch

import torch.nn as nn
from torch.nn import init
from collections import OrderedDict
from modules import d2lCustom as custom


def scratch_ver(num_inputs, num_outputs, train_iter, test_iter, eps, batch_size):
    # epoch 9, loss 0.4533, train acc 0.846, test acc 0.833
    # if eps=1e-3, learning rate = 0.1

    lr = 0.1

    W = torch.tensor(np.random.normal(0, .01, (num_inputs, num_outputs)), dtype=torch.float)
    b = torch.zeros(num_outputs, dtype=torch.float)

    W.requires_grad_()
    b.requires_grad_()

    fmnist.train(scratch.softmax_regression, train_iter, test_iter, scratch.cross_entropy, eps, batch_size,
                 [W, b], lr)

    return [W, b]


def custom_ver(num_inputs, num_outputs, train_iter, test_iter, eps, batch_size):
    # epoch 12, loss 0.0017, train acc 0.850, test acc 0.813
    # if eps=1e-3, learning rate = 0.1

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

    fmnist.train(net, train_iter, test_iter, loss, eps, batch_size, None, None, optimizer)

    return net


def main():
    batch_size = 256

    num_inputs = 784
    num_outputs = 10

    eps = 1e-3
    # eps = 1e-1

    root = './Datasets'
    train_iter, test_iter = fmnist.load_data(batch_size, root=root)

    mode = eval(input('0[Scratch Version], 1[Custom Version]: '))
    if mode:
        cus_net = custom_ver(num_inputs, num_outputs, train_iter, test_iter, eps, batch_size)
    else:
        params = scratch_ver(num_inputs, num_outputs, train_iter, test_iter, eps, batch_size)

    # predict
    X, y = next(iter(test_iter))

    true_labels = fmnist.get_labels(y.numpy())

    if mode:
        pred_labels = fmnist.get_labels(scratch.softmax_regression(X, cus_net).argmax(dim=1).numpy())
    else:
        pred_labels = fmnist.get_labels(scratch.softmax_regression(X, params=params).argmax(dim=1).numpy())

    titles = []
    for true, pred in zip(true_labels, pred_labels):
        title = true + '\n'
        if true != pred:
            title += 'X: ' + pred
        titles.append(title)

    fmnist.show(X, titles)


if __name__ == '__main__':
    main()
