import torch


# initialize parameters
def init_params(num_inputs):
    w = torch.randn((num_inputs, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    return [w, b]


# activation function
def relu(X):
    return torch.max(X, torch.zeros(X.shape, dtype=torch.float))


# net
## linear regression
class LinearNet:
    def __init__(self, params):
        self.params = params  # params = [w, b]

    def net(self, X):
        return torch.mm(X, self.params[0]) + self.params[1]


## softmax regression
class SoftmaxNet:
    def __init__(self, params):
        self.params = params  # params = [W, b]

    @staticmethod
    def softmax(X):
        X_exp = X.exp()
        partition = X_exp.sum(dim=1, keepdim=True)
        return X_exp / partition

    def net(self, X):
        num_inputs = X.shape[-1] * X.shape[-2]
        return self.softmax(torch.mm(X.view(-1, num_inputs), self.params[0]) + self.params[1])


## multilayer perceptron
class MLPNet:
    def __init__(self, params):
        self.params = params  # params = [W1, b1, W2, b2]

    def net(self, X):
        num_inputs = X.shape[-1] * X.shape[-2]
        H = relu(torch.matmul(X.view(-1, num_inputs), self.params[0]) + self.params[1])
        return torch.matmul(H, self.params[2]) + self.params[3]


## inverted dropout
class DropoutNet:
    def __init__(self, params, drop_prob):
        self.params = params  # params = [W1, b1, W2, b2, ...]
        self.drop_prob = drop_prob

    @staticmethod
    def dropout(X, drop_prob):
        assert 0 <= drop_prob <= 1

        X = X.float()
        keep_prob = 1 - drop_prob

        if keep_prob == 0:
            return torch.zeros_like(X)

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


# loss
def squared_loss(y_hat, y):
    return (y_hat - y.view(y_hat.size()))**2 / 2


def cross_entropy(y_hat, y):
    return - torch.log(y_hat.gather(1, y.view(-1, 1)))


# overfitting
## penalty
def l2_penalty(w):
    return (w**2).sum() / 2



# optimize
def sgd(params, lr, batch_size):
    # mini-batch stochastic gradient descent
    for param in params:
        param.data -= lr * param.grad / batch_size
