import torch
from torch import nn
import torch.utils.data as Data
import numpy as np
from modules import base
from modules import d2lScratch as scratch


# overfitting

def scratch_ver(num_inputs, train_iter, train_features, test_features, train_labels, test_labels, loss, num_epochs,
                lr, wd):
    # L2 norm of w:  0.017779173329472542
    # if num_epochs = 100, learning rate = 0.003, weight decay = 15
    w = torch.randn((num_inputs, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)

    linear = scratch.LinearNet([w, b])
    net = linear.net

    train_ls, test_ls = [], []
    for _ in range(num_epochs):
        for X, y in train_iter:
            ls = loss(net(X), y) + wd * scratch.l2_penalty(w)
            ls.sum()
            if w.grad is not None:
                w.grad.data.zero_()
                b.grad.data.zero_()

            ls.backward()

            scratch.sgd([w, b], lr, 1)

        train_ls.append(loss(net(train_features), train_labels).mean().item())
        test_ls.append(loss(net(test_features), test_labels).mean().item())

    print('L2 norm of w: ', w.norm().item())
    base.semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'loss',
                  range(1, num_epochs + 1), test_ls, ['train', 'test'],
                  figsize=(10, 8))


def custom_ver(num_inputs, train_iter, train_features, test_features, train_labels, test_labels, loss, num_epochs,
               lr, wd):
    # L2 norm of w:  0.017779173329472542
    # if num_epochs = 100, learning rate = 0.003, weight decay = 15
    net = nn.Linear(num_inputs, 1)

    nn.init.normal_(net.weight)
    nn.init.normal_(net.bias)

    optimizer_w = torch.optim.SGD([net.weight], lr, weight_decay=wd)
    optimizer_b = torch.optim.SGD([net.bias], lr)

    train_ls, test_ls = [], []
    for _ in range(num_epochs):
        for X, y in train_iter:
            ls = loss(net(X), y).mean()
            optimizer_w.zero_grad()
            optimizer_b.zero_grad()

            ls.backward()

            optimizer_w.step()
            optimizer_b.step()

        train_ls.append(loss(net(train_features), train_labels).mean().item())
        test_ls.append(loss(net(test_features), test_labels).mean().item())

    print('L2 norm of w: ', net.weight.data.norm().item())
    base.semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'loss',
                  range(1, num_epochs + 1), test_ls, ['train', 'test'],
                  figsize=(10, 8))


def main():
    n_train, n_test, num_inputs = 20, 100, 200
    true_w, true_b = torch.ones(num_inputs, 1) * .01, .05

    features = torch.randn((n_train + n_test, num_inputs))
    labels = torch.matmul(features, true_w) + true_b
    labels += torch.tensor(np.random.normal(0, .01, labels.size()), dtype=torch.float)
    train_features, test_features = features[:n_train, :], features[n_train:, :]
    train_labels, test_labels = labels[:n_train], labels[n_train:]

    batch_size, num_epochs, lr, wd = 1, 100, .003, 15

    dataset = Data.TensorDataset(train_features, train_labels)
    train_iter = Data.DataLoader(dataset, batch_size, shuffle=True)

    mode = eval(input('0[Scratch Version], 1[Custom Version]: '))
    if mode:
        custom_ver(num_inputs, train_iter, train_features, test_features, train_labels, test_labels,
                   scratch.squared_loss, num_epochs, lr, wd)
    else:
        scratch_ver(num_inputs, train_iter, train_features, test_features, train_labels, test_labels,
                    scratch.squared_loss, num_epochs, lr, wd)


if __name__ == '__main__':
    main()
