import random
import torch
from IPython import display
from matplotlib import pyplot as plt


def svg_display():
    # use scalable vector graphics to display the figures
    display.set_matplotlib_formats('svg')


def set_size(figsize=(3.5, 2.5)):
    svg_display()
    # set the size of the figure
    plt.rcParams['figure.figsize'] = figsize


def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)  # readout sequence is random
    for i in range(0, num_examples, batch_size):
        j = torch.LongTensor(indices[i: min(i + batch_size, num_examples)])
        yield features.index_select(0, j), labels.index_select(0, j)


# regression
## linear regression
def linear_regression(X, w, b):
    return torch.mm(X, w) + b


## softmax regression
def softmax(X):
    X_exp = X.exp()
    partition = X_exp.sum(dim=1, keepdim=True)
    return X_exp/partition


def softmax_regression(X, W, b, num_inputs=None):
    if num_inputs==None:
        num_inputs=X.shape[0]
    return softmax(torch.mm(X.view(-1, num_inputs), W) + b)


# loss
def squared_loss(y_hat, y):
    return (y_hat - y.view(y_hat.size()))**2/2


def cross_entropy(y_hat, y):
    return - torch.log(y_hat.gather(1, y.view(-1, 1)))


# optimize
def sgd(params, lr, batch_size):
    # mini-batch stochastic gradient descent
    for param in params:
        param.data -= lr*param.grad/batch_size


# evaluate
def accuracy(data_iter, net):
    acc_sum, n = .0, 0
    for X, y in data_iter:
        acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
        n += y.shape[0]
    return acc_sum/n

# train
