import torch
from torch import nn
from torch.nn import functional as F


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


class NiN(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
                self.nin_block(1, 96, kernel_size=11, strides=4, padding=0),
                nn.MaxPool2d(3, stride=2),
                self.nin_block(96, 256, kernel_size=5, strides=1, padding=2),
                nn.MaxPool2d(3, stride=2),
                self.nin_block(256, 384, kernel_size=3, strides=1, padding=1),
                nn.MaxPool2d(3, stride=2),
                nn.Dropout(0.5),
                # There are 10 label classes
                self.nin_block(384, 10, kernel_size=3, strides=1, padding=1),
                # The global average pooling layer automatically sets the window shape
                # to the height and width of the input
                nn.AdaptiveMaxPool2d((1, 1)),
                # Transform the four-dimensional output into two-dimensional output
                # with a shape of (batch size, 10)
                FlattenLayer()
        )

    @staticmethod
    def nin_block(in_channels, out_channels, kernel_size, strides, padding):
        return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, strides, padding),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)


class Inception(nn.Module):
    # c1 - c4 are the number of output channels for each layer in the path
    def __init__(self, in_channels, c1, c2, c3, c4):
        super().__init__()
        # Path 1 is a single 1 x 1 convolutional layer
        self.p1_1 = nn.Conv2d(in_channels, c1, kernel_size=1)
        # Path 2 is a 1 x 1 convolutional layer followed by a 3 x 3
        # convolutional layer
        self.p2_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1)
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
        # Path 3 is a 1 x 1 convolutional layer followed by a 5 x 5
        # convolutional layer
        self.p3_1 = nn.Conv2d(in_channels, c3[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)
        # Path 4 is a 3 x 3 maximum pooling layer followed by a 1 x 1
        # convolutional layer
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = nn.Conv2d(in_channels, c4, kernel_size=1)

    def forward(self, x):
        p1 = F.relu(self.p1_1(x))
        p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
        p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
        p4 = F.relu(self.p4_2(self.p4_1(x)))
        # Concatenate the outputs on the channel dimension
        return torch.cat((p1, p2, p3, p4), dim=1)


class GoogLeNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.b1 = nn.Sequential(
                nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.b2 = nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(64, 192, kernel_size=3, padding=1),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.b3 = nn.Sequential(
                Inception(192, 64, (96, 128), (16, 32), 32),
                Inception(256, 128, (128, 192), (32, 96), 64),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.b4 = nn.Sequential(
                Inception(480, 192, (96, 208), (16, 48), 64),
                Inception(512, 160, (112, 224), (24, 64), 64),
                Inception(512, 128, (128, 256), (24, 64), 64),
                Inception(512, 112, (144, 288), (32, 64), 64),
                Inception(528, 256, (160, 320), (32, 128), 128),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.b5 = nn.Sequential(
                Inception(832, 256, (160, 320), (32, 128), 128),
                Inception(832, 384, (192, 384), (48, 128), 128),
                nn.AdaptiveMaxPool2d((1, 1)),
                FlattenLayer()
        )

        self.net = nn.Sequential(self.b1, self.b2, self.b3, self.b4, self.b5, nn.Linear(1024, 10))

    def forward(self, x):
        return self.net(x)


def arch(x, layers):
    for layer in layers:
        x = layer(x)
        print(layer.__class__.__name__, 'output shape:\t', x.shape)
    return x


if __name__ == '__main__':
    size = (1, 1, 96, 96)
    nets = [GoogLeNet().net]
    X = torch.randn(size, dtype=torch.float32)
    for net in nets:
        X = arch(X, net)
        if net is nets[-1]:
            break
        X = FlattenLayer().forward(X)
        print('FlattenLayer output shape:\t', X.shape)
