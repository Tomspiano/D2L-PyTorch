import torch
from torch import nn


class FlattenLayer(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(x):
        return x.view(x.shape[0], -1)


class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(kernel_size))
        self.bias = nn.Parameter(torch.zeros(1))

    @staticmethod
    def corr2d(x, K):
        """Compute 2D cross-correlation."""
        h, w = K.shape
        Y = torch.zeros((x.shape[0] - h + 1, x.shape[1] - w + 1))
        for i in range(Y.shape[0]):
            for j in range(Y.shape[1]):
                Y[i, j] = (x[i:i + h, j:j + w] * K).sum()

        return Y

    def forward(self, x):
        return self.corr2d(x, self.weight) + self.bias


class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
                nn.Conv2d(1, 6, kernel_size=5, padding=2),
                nn.Sigmoid(),
                nn.AvgPool2d(2, 2),

                nn.Conv2d(6, 16, 5),
                nn.Sigmoid(),
                nn.AvgPool2d(2, 2)
        )
        self.fc = nn.Sequential(
                nn.Linear(16 * 5 * 5, 120),
                nn.Sigmoid(),
                nn.Linear(120, 84),
                nn.Sigmoid(),
                nn.Linear(84, 10)
        )

    def forward(self, img):
        feature = self.conv(img)
        output = self.fc(feature.view(img.shape[0], -1))
        return output


class AlexNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
                # capture the object
                nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1), nn.ReLU(),
                nn.MaxPool2d(3, 2),

                # Make the convolution window smaller, set padding to 2 for consistent
                # height and width across the input and output, and increase the
                # number of output channels
                nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),
                nn.MaxPool2d(3, 2),

                # Use three successive convolutional layers and a smaller convolution
                # window. Except for the final convolutional layer, the number of
                # output channels is further increased. Pooling layers are not used to
                # reduce the height and width of input after the first two
                # convolutional layers
                nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
                nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
                nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
                nn.MaxPool2d(3, 2)
        )

        # the number of outputs of the fully connected layer is several times larger
        # than that in LeNet.
        # Use the dropout layer to mitigate overfitting
        self.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(256 * 5 * 5, 4096), nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(4096, 4096), nn.ReLU(),
                # the number of classes in Fashion-MNIST is 10
                nn.Linear(4096, 10),
        )

    def forward(self, img):
        feature = self.conv(img)
        output = self.fc(feature.view(img.shape[0], -1))
        return output


class VGG11(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))

        # The convulational layer part
        conv_blks = []
        in_channels = 1
        for (num_convs, out_channels) in self.conv_arch:
            conv_blks.append(self.vgg_block(num_convs, in_channels, out_channels))
            in_channels = out_channels
        # The fully connected layer part
        self.net = nn.Sequential(
                *conv_blks, FlattenLayer(),
                nn.Linear(in_channels * 7 * 7, 4096), nn.ReLU(), nn.Dropout(0.5),
                nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
                nn.Linear(4096, 10)
        )

    @staticmethod
    def vgg_block(num_convs, in_channels, out_channels):
        layers = []
        for _ in range(num_convs):
            layers.append(nn.Conv2d(in_channels, out_channels,
                                    kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            in_channels = out_channels
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def arch(x, layers):
    for layer in layers:
        x = layer(x)
        print(layer.__class__.__name__, 'output shape:\t', x.shape)
    return x


if __name__ == '__main__':
    size = (1, 1, 224, 224)
    nets = [VGG11().net]
    X = torch.randn(size, dtype=torch.float32)
    for net in nets:
        X = arch(X, net)
        if net is nets[-1]:
            break
        X = FlattenLayer().forward(X)
        print('FlattenLayer output shape:\t', X.shape)
