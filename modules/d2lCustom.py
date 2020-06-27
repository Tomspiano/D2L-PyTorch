import torch
from torch import nn


class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()

    @staticmethod
    def forward(x):
        return x.view(x.shape[0], -1)


class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        super(Conv2D, self).__init__()
        self.weight = nn.Parameter(torch.randn(kernel_size))
        self.bias = nn.Parameter(torch.randn(1))

    @staticmethod
    def corr2d(X, K):
        h, w = K.shape
        Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
        for i in range(Y.shape[0]):
            for j in range(Y.shape[1]):
                Y[i, j] = (X[i:i + h, j:j + w] * K).sum()

        return Y

    def forward(self, x):
        return self.corr2d(x, self.weight) + self.bias


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv = nn.Sequential(
                nn.Conv2d(1, 6, 5),
                nn.Sigmoid(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(6, 16, 5),
                nn.Sigmoid(),
                nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
                nn.Linear(16 * 4 * 4, 120),
                nn.Sigmoid(),
                nn.Linear(120, 84),
                nn.Sigmoid(),
                nn.Linear(84, 10)
        )

    def forward(self, img):
        feature = self.conv(img)
        output = self.fc(feature.view(img.shape[0], -1))
        return output
