import torch
import numpy as np
from modules import base
from modules import fashionMNIST as fmnist

from modules import d2lScratch as scratch

from torch import nn
from torch.nn import init
from collections import OrderedDict
from modules import d2lCustom as custom


# overfitting
## inverted_dropout

def scratch_ver(num_inputs, num_hiddens, num_outputs, train_iter, test_iter, eps, batch_size, drop_prob):
    # epoch 16, loss 0.0012, train acc 0.885, test acc 0.860
    # if eps=1e-3, learning rate = 128 (0.5)

    lr = 0.5 * batch_size

    params = []
    pre_nrows = num_inputs
    cnt = len(num_hiddens) + 1
    for i in range(cnt):
        if i == cnt - 1:
            Wt = torch.tensor(np.random.normal(0, .01, (pre_nrows, num_outputs)), dtype=torch.float, requires_grad=True)
            bt = torch.zeros(num_outputs, dtype=torch.float, requires_grad=True)
        else:
            Wt = torch.tensor(np.random.normal(0, .01, (pre_nrows, num_hiddens[i])), dtype=torch.float,
                              requires_grad=True)
            bt = torch.zeros(num_hiddens[i], dtype=torch.float, requires_grad=True)

        params += [Wt, bt]
        if i != cnt - 1:
            pre_nrows = num_hiddens[i]

    dropout = scratch.DropoutNet(params, drop_prob)

    loss = nn.CrossEntropyLoss()

    base.train(dropout.net, train_iter, test_iter, loss, eps, batch_size, params, lr)


def custom_ver(num_inputs, num_hiddens, num_outputs, train_iter, test_iter, eps, batch_size, drop_prob):
    # epoch 13, loss 0.0013, train acc 0.880, test acc 0.872
    # if eps=1e-3, learning rate = 0.5

    cnt = len(num_hiddens) + 1
    od = OrderedDict()
    od['flatten'] = custom.FlattenLayer()
    od['linear_0'] = nn.Linear(num_inputs, num_hiddens[0])
    for i in range(1, cnt):
        od['relu_%d' % i] = nn.ReLU()
        od['dropout_%d' % i] = nn.Dropout(drop_prob[i - 1])
        if i == cnt - 1:
            od['linear_%d' % i] = nn.Linear(num_hiddens[i - 1], num_outputs)
        else:
            od['linear_%d' % i] = nn.Linear(num_hiddens[i - 1], num_hiddens[i])

    net = nn.Sequential(od)

    for params in net.parameters():
        init.normal_(params, mean=0, std=.01)

    loss = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(net.parameters(), lr=.5)

    base.train(net, train_iter, test_iter, loss, eps, batch_size, optimizer=optimizer)


def main():
    batch_size = 256

    num_inputs, num_outputs, num_hiddens = 784, 10, [256, 256]
    drop_prob = [.2, .5]

    eps = 1e-3
    # eps = 1e-1

    root = './Datasets'
    train_iter, test_iter = fmnist.load_data(batch_size, root=root)

    mode = eval(input('0[Scratch Version], 1[Custom Version]: '))
    if mode:
        custom_ver(num_inputs, num_hiddens, num_outputs, train_iter, test_iter, eps, batch_size, drop_prob)
    else:
        scratch_ver(num_inputs, num_hiddens, num_outputs, train_iter, test_iter, eps, batch_size, drop_prob)


if __name__ == '__main__':
    main()
