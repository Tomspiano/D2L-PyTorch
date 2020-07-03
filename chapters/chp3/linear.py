import torch
from torch import nn
from modules import base
from modules import d2lScratch as scratch


def squared_loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape))**2 / 2


def main():
    true_w = torch.tensor([2, -3.4])
    true_b = 4.2
    features, labels = base.synthetic_data(true_w, true_b, 1000)
    labels = labels.reshape(-1, 1)

    batch_size = 10
    data_iter = base.load_array((features, labels), batch_size)

    num_epochs = 3  # Number of iterations

    optimizer = None
    params = []
    mode = eval(input('0[Scratch Version], 1[Custom Version]: '))
    if mode:
        net = nn.Sequential(nn.Linear(2, 1))
        net[0].weight.data.uniform_(0.0, 0.01)
        net[0].bias.data.fill_(0)

        loss = nn.MSELoss()

        optimizer = torch.optim.SGD(net.parameters(), lr=.03)
    else:
        w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
        b = torch.zeros(1, requires_grad=True)
        params.extend([w, b])

        net = scratch.LinearNet([w, b]).net

        loss = squared_loss  # 0.5 (y-y')^2

    for epoch in range(num_epochs):
        # Assuming the number of examples can be divided by the batch size, all
        # the examples in the training data set are used once in one epoch
        # iteration. The features and tags of mini-batch examples are given by X
        # and y respectively
        for X, y in data_iter:
            ls = loss(net(X), y).sum()  # Minibatch loss in X and y

            if mode:
                optimizer.zero_grad()
            ls.backward()  # Compute gradient on ls with respect to [w,b]
            if mode:
                optimizer.step()
            else:
                scratch.sgd(params, .03, batch_size)

        with torch.no_grad():
            train_ls = loss(net(features), labels)
            print(f'epoch {epoch + 1}, loss {float(train_ls.mean())}')


if __name__ == '__main__':
    main()
