import random
import torch
from IPython import display
from matplotlib import pyplot as plt


def svg_display():
    # use scalable vector graphics to display the figures
    display.set_matplotlib_formats('svg')


'''''
def set_size(figsize=(3.5, 2.5)):
    svg_display()
    # set the size of the figure
    plt.rcParams['figure.figsize'] = figsize
'''

'''''
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)  # readout sequence is random
    for i in range(0, num_examples, batch_size):
        j = torch.LongTensor(indices[i: min(i + batch_size, num_examples)])
        yield features.index_select(0, j), labels.index_select(0, j)
'''


# regression
## linear regression
def linear_regression(X, w, b):
    return torch.mm(X, w) + b


## softmax regression
def softmax(X):
    X_exp = X.exp()
    partition = X_exp.sum(dim=1, keepdim=True)
    return X_exp / partition


def softmax_regression(X, net=None, params=None):  # params = [W, b]
    if params is None:
        return softmax(net(X))
    num_inputs = X.shape[-1] * X.shape[-2]
    return softmax(torch.mm(X.view(-1, num_inputs), params[0]) + params[1])


## multilayer perceptron
def mlp(X, params=None):  # params = [W1, b1, W1, b1]
    num_inputs = X.shape[-1] * X.shape[-2]
    H = relu(torch.mm(X.view(-1, num_inputs), params[0]) + params[1])
    return torch.mm(H, params[2]) + params[3]


# loss
def squared_loss(y_hat, y):
    return (y_hat - y.view(y_hat.size()))**2 / 2


def cross_entropy(y_hat, y):
    return - torch.log(y_hat.gather(1, y.view(-1, 1)))


# optimize
def sgd(params, lr, batch_size):
    # mini-batch stochastic gradient descent
    for param in params:
        param.data -= lr * param.grad / batch_size


# evaluate
def accuracy(data_iter, net, params=None):
    acc_sum, n = .0, 0
    for X, y in data_iter:
        if params is None:
            acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
        else:
            acc_sum += (net(X, params=params).argmax(dim=1) == y).float().sum().item()
        n += y.shape[0]
    return acc_sum / n


# activation function
def relu(X):
    return torch.max(X, torch.zeros(X.shape, dtype=torch.float))
