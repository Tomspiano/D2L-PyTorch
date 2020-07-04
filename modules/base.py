import torch
from torch import nn
from torch.utils import data as Data
import time
import numpy as np
from IPython import display
from matplotlib import pyplot as plt


# display
def use_svg_display():
    """Use the svg format to display a plot in Jupyter."""
    display.set_matplotlib_formats('svg')


def set_figsize(figsize=(3.5, 2.5)):
    """Set the figure size for matplotlib."""
    use_svg_display()
    plt.rcParams['figure.figsize'] = figsize


def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """Set the axes for matplotlib."""
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()


def has_one_axis(X):
    # Return True if `X` (ndarray or list) has 1 axis
    return (hasattr(X, "ndim") and X.ndim == 1 or isinstance(X, list)
            and not hasattr(X[0], "__len__"))


def plot(X, Y=None, xlabel=None, ylabel=None, legend=None, xlim=None,
         ylim=None, xscale='linear', yscale='linear',
         fmts=('-', 'm--', 'g-.', 'r:'), figsize=(3.5, 2.5), axes=None):
    """Plot data instances."""
    if legend is None:
        legend = []

    set_figsize(figsize)
    axes = axes if axes else plt.gca()

    if has_one_axis(X):
        X = [X]
    if Y is None:
        X, Y = [[]] * len(X), X
    elif has_one_axis(Y):
        Y = [Y]
    if len(X) != len(Y):
        X = X * len(Y)
    axes.cla()
    for x, y, fmt in zip(X, Y, fmts):
        if len(x):
            axes.plot(x, y, fmt)
        else:
            axes.plot(y, fmt)
    set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)


# data
def synthetic_data(w, b, num_examples):
    """Generate y = X w + b + noise."""
    X = torch.zeros(size=(num_examples, len(w))).normal_()
    y = torch.matmul(X, w) + b
    y += torch.zeros(size=y.shape).normal_(std=0.01)
    return X, y


def load_array(data_arrays, batch_size, is_train=True):
    """Construct a PyTorch data loader"""
    dataset = Data.TensorDataset(*data_arrays)
    return Data.DataLoader(dataset, batch_size, shuffle=is_train)


# classes
class Accumulator:
    """Sum a list of numbers over time."""

    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class Animator:
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear', fmts=None,
                 nrows=1, ncols=1, figsize=(3.5, 2.5)):
        """Incrementally plot multiple lines."""
        if legend is None:
            legend = []
        use_svg_display()
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # Use a lambda to capture arguments
        self.config_axes = lambda: set_axes(
                self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        """Add multiple data points into the figure."""
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        if not self.fmts:
            self.fmts = ['-'] * n
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        # display.display(self.fig)
        # display.clear_output(wait=True)

    @staticmethod
    def show():
        plt.show()


class Timer:
    """Record multiple running times."""

    def __init__(self):
        self.times = []
        self.tik = 0.0
        self.start()

    def start(self):
        """Start the timer."""
        self.tik = time.time()

    def stop(self):
        """Stop the timer and record the time in a list."""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """Return the average time."""
        return sum(self.times) / len(self.times)

    def sum(self):
        """Return the sum of time."""
        return sum(self.times)

    def cumsum(self):
        """Return the accumulated time."""
        return np.array(self.times).cumsum().tolist()


# use GPU
def try_gpu(i=0):
    """Return gpu(i) if exists, otherwise return cpu()."""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


# train
def train(net, train_iter, test_iter, loss, eps=None, num_epochs=0,
          optimizer=None, device=try_gpu(),
          checkpoint_path=None, checkpoint=None):
    print('training on', device)
    if isinstance(net, nn.Module):
        net.to(device)

    epoch = 0 if checkpoint is None else checkpoint['epoch']

    timer = Timer()
    train_accurate = 0
    while True:
        metric = Accumulator(3)  # train_loss, train_acc, num_examples

        for X, y in train_iter:
            timer.start()

            # reset gradient
            optimizer.zero_grad()

            X, y = X.to(device), y.to(device)

            y_hat = net(X)
            ls = loss(y_hat, y)
            ls.backward()

            optimizer.step()

            with torch.no_grad():
                metric.add(ls * X.shape[0], accuracy(y_hat, y), X.shape[0])

            timer.stop()

        epoch += 1
        if checkpoint_path is not None:
            ckpt = {
                'net'      : net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch'    : epoch
            }
            torch.save(ckpt, checkpoint_path)

        train_loss, train_acc = metric[0] / metric[2], metric[1] / metric[2]
        t = metric[2] / timer.sum()
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.3f, train acc %.3f, test acc %.3f, %.1f examples/sec'
              % (epoch, train_loss, train_acc, test_acc, t))

        if abs(train_acc - train_accurate) < eps and epoch >= num_epochs:
            break
        else:
            train_accurate = train_acc


# evaluate
def accuracy(y_hat, y):
    if y_hat.shape[1] > 1:
        return float((y_hat.argmax(axis=1).type(torch.float32) ==
                      y.type(torch.float32)).sum())
    else:
        return float((y_hat.type(torch.int32) == y.type(torch.int32)).sum())


def evaluate_accuracy(data_iter, net, device=None):
    if not device and isinstance(net, nn.Module):
        device = next(iter(net.parameters())).device
    metric = Accumulator(2)  # acc_sum, n
    for X, y in data_iter:
        if device is not None:
            X, y = X.to(device), y.to(device)
        metric.add(accuracy(net(X), y), sum(y.shape))
    return metric[0] / metric[1]


def evaluate_loss(net, data_iter, loss):
    """Evaluate the loss of a model on the given dataset."""
    metric = Accumulator(2)  # Sum of losses, no. of examples
    for X, y in data_iter:
        ls = loss(net(X), y)
        metric.add(ls.sum(), ls.numel())
    return metric[0] / metric[1]
