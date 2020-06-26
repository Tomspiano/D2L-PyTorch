import torch
import torch.utils.data as Data
import torchvision
import numpy as np
import math
from matplotlib import pyplot as plt
from modules import base


def get_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


def show(images, labels):
    # show at most 25 images
    base.svg_display()

    ncols = 5
    # print(len(images))
    nrows = min(5, math.ceil(len(images) / ncols))
    # _, axes = plt.subplots(nrows, ncols, figsize=(12, 12))

    n = min(len(images), ncols * nrows)
    for i in range(1, n + 1):
        plt.subplot(nrows, ncols, i)
        plt.axis('off')
        plt.imshow(images[i - 1].view((28, 28)).numpy())
        plt.title(labels[i - 1])
        i += 1

    plt.tight_layout()
    plt.show()


def load_data(batch_size, resize=None, root='../'):
    trans = []
    if resize:
        trans.append(torchvision.transforms.Resize(resize))
    trans.append(torchvision.transforms.ToTensor())

    transform = torchvision.transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(root, download=True, transform=transform)
    mnist_test = torchvision.datasets.FashionMNIST(root, train=False, download=True, transform=transform)

    num_workers = 3
    train_iter = Data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=num_workers)
    test_iter = Data.DataLoader(mnist_test, batch_size, shuffle=False, num_workers=num_workers)

    return train_iter, test_iter


def test():
    batch_size = 20

    num_inputs = 784
    num_outputs = 10

    # eps, lr = 5, 0.1

    root = '../Datasets'
    train_iter, test_iter = load_data(batch_size=batch_size, root=root)

    W = torch.tensor(np.random.normal(0, .01, (num_inputs, num_outputs)), dtype=torch.float)
    b = torch.zeros(num_outputs, dtype=torch.float)

    W.requires_grad_()
    b.requires_grad_()
    '''''
    base.train(scratch.softmax_regression, train_iter, test_iter, scratch.cross_entropy, eps, batch_size,[W, b], lr)
    '''''
    X, y = next(iter(test_iter))

    true_labels = get_labels(y.numpy())
    titles = [true for true in true_labels]

    show(X[0:5], titles[0:5])


if __name__ == '__main__':
    test()
