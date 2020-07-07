import os
import torch
from torch import nn
from modules import base
from modules import fashionMNIST as fmnist
from modules import d2lCustom as custom


def main():
    batch_size = 128

    lr, eps = .0001, 1e-3
    # lr, eps = .0001, 1e-1

    root = '../../Datasets'
    train_iter, test_iter = fmnist.load_data(batch_size, 224, root)

    net = custom.VGG11(4)
    optimizer = torch.optim.Adam(net.parameters(), lr)
    ckpt_path = '../../checkpoint/vgg.pt'
    ckpt = None
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path)

        net.load_state_dict(ckpt['net'])
        optimizer.load_state_dict(ckpt['optimizer'])
    loss = nn.CrossEntropyLoss()

    base.train(net, train_iter, test_iter, loss, eps=eps,
               optimizer=optimizer, checkpoint_path=ckpt_path, checkpoint=ckpt)
    # epoch 15, loss 0.046, train acc 0.983, test acc 0.926, 5.8 examples/sec
    # if eps = 1e-3, learning rate = 0.0001
    # overfitting


if __name__ == '__main__':
    main()
