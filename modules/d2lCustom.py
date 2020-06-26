from torch import nn


class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()

    @staticmethod
    def forward(x):
        return x.view(x.shape[0], -1)
