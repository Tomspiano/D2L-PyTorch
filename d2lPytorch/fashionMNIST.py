import torch.utils.data
import torchvision
from matplotlib import pyplot as plt

from d2lPytorch import d2lBase as d2l


def get_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


def show(images, labels):
    d2l.svg_display()
    _, figs = plt.subplots(1, len(images), figsize=(12, 12))
    for f, image, label in zip(figs, images, labels):
        f.imshow(image.view((28, 28)).numpy())
        f.set_title(label)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    plt.show()


def load_data(batch_size, resize=None, root='../Datasets/FashionMNIST'):
    trans = []
    if resize:
        trans.append(torchvision.transforms.Resize(resize))
    trans.append(torchvision.transforms.ToTensor())

    transform = torchvision.transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(root, download=True,
                                                    transform=transform)
    mnist_test = torchvision.datasets.FashionMNIST(root, train=False, download=True,
                                                   transform=transform)

    num_workers = 3
    train_iter = torch.utils.data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(mnist_test, batch_size, shuffle=False, num_workers=num_workers)

    return train_iter, test_iter
