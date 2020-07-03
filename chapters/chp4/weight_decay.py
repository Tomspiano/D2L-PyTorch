import torch
from torch import nn
from modules import base
from modules import d2lScratch as scratch


# overfitting

def scratch_ver(num_inputs, train_iter, test_iter, num_epochs,
                lr, wd, batch_size):
    # L2 norm of w:  0.0032210401259362698
    # if num_epochs = 100, learning rate = 0.003, weight decay = 100
    w = torch.normal(0, 1, (num_inputs, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)

    net = scratch.LinearNet([w, b]).net
    loss = scratch.squared_loss

    animator = base.Animator(xlabel='epochs', ylabel='loss', yscale='log',
                             xlim=[1, num_epochs], legend=['train', 'test'])
    for epoch in range(num_epochs):
        for X, y in train_iter:
            with torch.enable_grad():
                # The L2 norm penalty term has been added, and broadcasting
                # makes l2_penalty(w) a vector whose length is batch_size
                ls = loss(net(X), y) + wd * scratch.l2_penalty(w)
            ls.sum().backward()
            scratch.sgd([w, b], lr, batch_size)

        animator.add(epoch, (base.evaluate_loss(net, train_iter, loss),
                             base.evaluate_loss(net, test_iter, loss)))

    animator.show()
    print('L2 norm of w: ', torch.norm(w).item())


def custom_ver(num_inputs, train_iter, test_iter, num_epochs,
               lr, wd):
    # L2 norm of w:  0.007476740516722202
    # if num_epochs = 100, learning rate = 0.003, weight decay = 100
    net = nn.Linear(num_inputs, 1)

    nn.init.normal_(net.weight)
    nn.init.normal_(net.bias)

    loss = nn.MSELoss()

    # The bias parameter has not decayed. Bias names generally end with "bias"
    trainer = torch.optim.SGD([
        {"params": net.weight, 'weight_decay': wd},
        {"params": net.bias}], lr=lr)

    animator = base.Animator(xlabel='epochs', ylabel='loss', yscale='log',
                             xlim=[1, num_epochs], legend=['train', 'test'])
    for epoch in range(num_epochs):
        for X, y in train_iter:
            with torch.enable_grad():
                trainer.zero_grad()
                ls = loss(net(X), y)
            ls.backward()
            trainer.step()

        animator.add(epoch, (base.evaluate_loss(net, train_iter, loss),
                             base.evaluate_loss(net, test_iter, loss)))

    animator.show()
    print('L2 norm of w: ', net.weight.data.norm().item())


def main():
    n_train, n_test, num_inputs = 20, 100, 200
    batch_size, num_epochs, lr, wd = 5, 100, .003, 100

    true_w, true_b = torch.ones(num_inputs, 1) * .01, .05

    train_data = base.synthetic_data(true_w, true_b, n_train)
    train_iter = base.load_array(train_data, batch_size)
    test_data = base.synthetic_data(true_w, true_b, n_test)
    test_iter = base.load_array(test_data, batch_size, is_train=False)

    mode = eval(input('0[Scratch Version], 1[Custom Version]: '))
    if mode:
        custom_ver(num_inputs, train_iter, test_iter, num_epochs, lr, wd)
    else:
        scratch_ver(num_inputs, train_iter, test_iter, num_epochs, lr, wd, batch_size)


if __name__ == '__main__':
    main()
