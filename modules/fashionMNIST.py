import torch.utils.data as Data
import torchvision
import sys
from matplotlib import pyplot as plt


def get_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


def show(images, nrows, ncols, titles=None, scale=1.5):
    """Plot a list of images."""
    figsize = (ncols * scale, nrows * scale)
    _, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, images)):
        if 'asnumpy' in dir(img):
            img = img.view(img.shape[1], img.shape[2]).asnumpy()
        if 'numpy' in dir(img):
            img = img.view(img.shape[1], img.shape[2]).numpy()
        ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])

    plt.tight_layout()
    plt.show()
    # return axes


def get_dataloader_workers(num_workers=4):
    # 0 means no additional process is used to speed up the reading of data.
    if sys.platform.startswith('win'):
        return 0
    else:
        return num_workers


def load_data(batch_size, resize=None, root='./'):
    """Download the Fashion-MNIST dataset and then load into memory."""
    trans = [torchvision.transforms.Resize(resize)] if resize else []
    trans.append(torchvision.transforms.ToTensor())
    transform = torchvision.transforms.Compose(trans)

    mnist_train = torchvision.datasets.FashionMNIST(
            root, train=True, download=True, transform=transform)
    mnist_test = torchvision.datasets.FashionMNIST(
            root, train=False, download=True, transform=transform)

    num_workers = get_dataloader_workers()
    train_iter = Data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=num_workers)
    test_iter = Data.DataLoader(mnist_test, batch_size, shuffle=False, num_workers=num_workers)

    return train_iter, test_iter


def test():
    batch_size = 256

    root = '../Datasets'
    train_iter, test_iter = load_data(batch_size, root=root)

    X, y = next(iter(test_iter))
    true_labels = get_labels(y.numpy())

    show(X, nrows=5, ncols=3, titles=true_labels)


if __name__ == '__main__':
    test()
