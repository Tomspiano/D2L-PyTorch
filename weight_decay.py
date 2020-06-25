import torch
from torch import nn
import torch.utils.data as Data
import numpy as np
from modules import d2lScratch as scratch


# overfitting
def weight_decay_fit(num_inputs, train_iter, train_features, test_features, train_labels, test_labels, loss, num_epochs,
                     lr, wd):
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
    scratch.semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'loss',
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

    weight_decay_fit(num_inputs, train_iter, train_features, test_features, train_labels, test_labels,
                     scratch.squared_loss, num_epochs, lr, wd)
    # L2 norm of w:  0.019665690138936043


if __name__ == '__main__':
    main()
