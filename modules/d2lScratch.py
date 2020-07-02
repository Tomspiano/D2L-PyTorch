import torch


# activation function
def relu(X):
    return torch.max(X, torch.zeros_like(X))


# net
## linear regression
class LinearNet:
    def __init__(self, params):
        self.params = params  # params = [w, b]

    def net(self, X):
        return torch.matmul(X, self.params[0]) + self.params[1]


## softmax regression
class SoftmaxNet:
    def __init__(self, params):
        self.params = params  # params = [W, b]

    @staticmethod
    def softmax(X):
        X_exp = torch.exp(X)
        partition = torch.sum(X_exp, dim=1, keepdim=True)
        return X_exp / partition

    def net(self, X):
        num_inputs = X.shape[-1] * X.shape[-2]
        return self.softmax(torch.matmul(X.reshape((-1, num_inputs)), self.params[0]) + self.params[1])


## multilayer perceptron
class MLPNet:
    def __init__(self, params):
        self.params = params  # params = [W1, b1, W2, b2]

    def net(self, X):
        num_inputs = X.shape[-1] * X.shape[-2]
        H = relu((X.reshape((-1, num_inputs)) @ self.params[0] + self.params[1]))
        # Here '@' stands for dot product operation

        return H @ self.params[2] + self.params[3]


## inverted dropout
class DropoutNet:
    def __init__(self, params, drop_prob):
        self.params = params  # params = [W1, b1, W2, b2, ...]
        self.drop_prob = drop_prob

    @staticmethod
    def dropout(X, drop_prob):
        assert 0 <= drop_prob <= 1

        keep_prob = 1 - drop_prob
        if keep_prob == 0:
            return torch.zeros_like(X)
        if drop_prob == 0:
            return X

        mask = (torch.rand(X.shape) < keep_prob).float()
        return mask * X / keep_prob

    def net(self, X, is_training=True):
        num_inputs = X.shape[-1] * X.shape[-2]

        H = []
        pre_mat = X.view(-1, num_inputs)
        cnt = len(self.drop_prob)
        for i in range(cnt):
            Ht = (torch.matmul(pre_mat, self.params[2 * i]) + self.params[2 * i + 1]).relu()
            if is_training:
                Ht = self.dropout(Ht, self.drop_prob[i])

            H.append(Ht)
            pre_mat = Ht

        return torch.matmul(H[-1], self.params[-2]) + self.params[-1]


## weight decay
def l2_penalty(w):
    return torch.sum(w**2) / 2


# loss
def squared_loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape))**2 / 2


def cross_entropy(y_hat, y):
    return - torch.log(y_hat[range(len(y_hat)), y])


# optimize
def sgd(params, lr, batch_size):
    # mini-batch stochastic gradient descent
    for param in params:
        param.data.sub_(lr * param.grad / batch_size)
        param.grad.data.zero_()
