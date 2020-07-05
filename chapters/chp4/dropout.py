import torch
from modules import base
from modules import fashionMNIST as fmnist
from torch import nn
from torch.nn import init
from collections import OrderedDict
from modules import d2lCustom as custom


# overfitting
## inverted_dropout

def train(num_inputs, num_hiddens, num_outputs, train_iter, test_iter, eps, drop_prob):
    # epoch 25, loss 0.272, train acc 0.898, test acc 0.860, 467.3 examples/sec
    # if eps = 1e-3, learning rate = 0.5
    cnt = len(num_hiddens) + 1
    odict = OrderedDict()
    odict['flatten'] = custom.FlattenLayer()
    odict['linear_0'] = nn.Linear(num_inputs, num_hiddens[0])
    for i in range(1, cnt):
        odict['relu_%d' % i] = nn.ReLU()
        odict['dropout_%d' % i] = nn.Dropout(drop_prob[i - 1])
        if i == cnt - 1:
            odict['linear_%d' % i] = nn.Linear(num_hiddens[i - 1], num_outputs)
        else:
            odict['linear_%d' % i] = nn.Linear(num_hiddens[i - 1], num_hiddens[i])

    net = nn.Sequential(odict)

    for params in net.parameters():
        init.normal_(params, mean=0, std=.01)

    loss = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(net.parameters(), lr=.5)

    base.train(net, train_iter, test_iter, loss, eps=eps, optimizer=optimizer)


def main():
    batch_size = 256

    num_inputs, num_outputs, num_hiddens = 784, 10, [256, 256]
    drop_prob = [.2, .5]

    eps = 1e-3
    # eps = 1e-1

    root = '../../Datasets'
    train_iter, test_iter = fmnist.load_data(batch_size, root=root)

    train(num_inputs, num_hiddens, num_outputs, train_iter, test_iter, eps, drop_prob)


if __name__ == '__main__':
    main()
