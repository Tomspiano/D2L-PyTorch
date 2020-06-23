import torch
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


def load_data(batch_size, resize=None, root='./Datasets/FashionMNIST'):
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


def train(net, train_iter, test_iter, loss, num_epochs, batch_size, params=None, lr=None, optimizer=None):
    for epoch in range(num_epochs):
        train_loss_sum, train_acc_sum, n = .0, .0, 0

        for X, y in train_iter:
            y_hat = net(X, params[0], params[1])
            ls = loss(y_hat, y).sum()

            # reset gradient
            if optimizer is not None:
                optimizer.zero_grad()
            elif params is not None and params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()

            ls.backward()
            if optimizer is None:
                d2l.sgd(params, lr, batch_size)
            else:
                optimizer.step()

            train_loss_sum += ls.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
            n += y.shape[0]

        test_acc = d2l.accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
              % (epoch + 1, train_loss_sum / n, train_acc_sum / n, test_acc))
