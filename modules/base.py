import random
from torch import nn
from IPython import display
from matplotlib import pyplot as plt
from modules import d2lScratch as scratch

'''''
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)  # readout sequence is random
    for i in range(0, num_examples, batch_size):
        j = torch.LongTensor(indices[i: min(i + batch_size, num_examples)])
        yield features.index_select(0, j), labels.index_select(0, j)
'''


def svg_display():
    # use scalable vector graphics to display the figures
    display.set_matplotlib_formats('svg')


def set_size(figsize=(3.5, 2.5)):
    svg_display()
    # set the size of the figure
    plt.rcParams['figure.figsize'] = figsize


def semilogy(x_vals, y_vals, x_label, y_label, x2_vals=None, y2_vals=None, legend=None, figsize=(3.5, 2.5)):
    set_size(figsize)

    plt.semilogy(x_vals, y_vals)
    if x2_vals and y2_vals:
        plt.semilogy(x2_vals, y2_vals, linestyle=':')
        plt.legend(legend)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()


def train(net, train_iter, test_iter, loss, eps, batch_size,
          params=None, lr=None, optimizer=None):
    train_acc_rate = 0
    epoch = 0

    while True:
        train_loss_sum, train_acc_sum, n = .0, .0, 0

        for X, y in train_iter:
            y_hat = net(X)
            ls = loss(y_hat, y).sum()

            # reset gradient
            if optimizer is not None:
                optimizer.zero_grad()
            elif params is not None and params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()

            ls.backward()
            if optimizer is None:
                scratch.sgd(params, lr, batch_size)
            else:
                optimizer.step()

            train_loss_sum += ls.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
            n += y.shape[0]

        test_acc = accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
              % (epoch + 1, train_loss_sum / n, train_acc_sum / n, test_acc))

        if abs(train_acc_sum / n - train_acc_rate) < eps:
            break
        else:
            epoch += 1
            train_acc_rate = train_acc_sum / n


# evaluate
def accuracy(data_iter, net):
    acc_sum, n = .0, 0
    for X, y in data_iter:
        if isinstance(net, nn.Module):
            net.eval()
            acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
            net.train()
        else:
            if 'is_training' in net.__code__.co_varnames:
                acc_sum += (net(X, is_training=False).argmax(dim=1) == y).float().sum().item()
            else:
                acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()

        n += y.shape[0]
    return acc_sum / n
