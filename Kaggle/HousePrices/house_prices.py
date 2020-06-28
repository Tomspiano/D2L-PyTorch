# Jun. 28th, 2020
# Score: 0.14508
import torch
from torch import nn
from torch.nn import init
import torch.utils.data as Data
import pandas as pd
from modules import base


def get_net(feature_num):
    hidden_num = 32, 8
    drop_prob = .001
    net = nn.Sequential(
            nn.Linear(feature_num, hidden_num[0]),

            nn.ReLU(),
            nn.Dropout(drop_prob),
            nn.Linear(hidden_num[0], hidden_num[1]),

            nn.ReLU(),
            nn.Linear(hidden_num[1], 1)
    )

    for params in net.parameters():
        init.normal_(params, mean=0, std=.01)

    return net


def log_rmse(net, features, labels, loss):
    with torch.no_grad():
        clipped_preds = torch.max(net(features), torch.tensor(1.0))
        rmse = torch.sqrt(loss(clipped_preds.log(), labels.log()))
    return rmse.item()


def train(net, train_features, train_labels, test_features, test_labels,
          loss, learning_rate, weight_decay, batch_size):
    train_ls, test_ls = [], []
    dataset = Data.TensorDataset(train_features, train_labels)
    train_iter = Data.DataLoader(dataset, batch_size, shuffle=True)

    optimizer = torch.optim.Adam(params=net.parameters(), lr=learning_rate, weight_decay=weight_decay)

    net = net.float()

    num_epochs = 0
    while True:
        for X, y in train_iter:
            ls = loss(net(X.float()), y.float())
            optimizer.zero_grad()
            ls.backward()
            optimizer.step()
        train_ls.append(log_rmse(net, train_features, train_labels, loss))
        if test_labels is not None:
            test_ls.append(log_rmse(net, test_features, test_labels, loss))

        num_epochs += 1

        if len(train_ls) > 1:
            if abs(train_ls[-1] - train_ls[-2]) < 1e-3:
                break

    return train_ls, test_ls, num_epochs


def get_k_fold_data(k, i, X, y):
    assert k > 1

    fold_size = X.shape[0] // k
    X_train, y_train = None, None
    X_valid, y_valid = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx, :], y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.cat((X_train, X_part), dim=0)
            y_train = torch.cat((y_train, y_part), dim=0)

    return X_train, y_train, X_valid, y_valid


def k_fold(k, X_train, y_train,
           learning_rate, weight_decay, batch_size):
    loss = nn.MSELoss()
    train_loss_sum, valid_loss_sum = 0, 0

    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)
        net = get_net(X_train.shape[1])
        train_ls, valid_ls, num_epochs = train(net, *data, loss, learning_rate,
                                               weight_decay, batch_size)
        train_loss_sum += train_ls[-1]
        valid_loss_sum += valid_ls[-1]
        if i == k - 1:
            base.semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'rmse',
                          range(1, num_epochs + 1), valid_ls,
                          ['train', 'valid'])
        print('fold %d, train rmse %f, valid rmse %f' % (i, train_ls[-1], valid_ls[-1]))

    return train_loss_sum / k, valid_loss_sum / k


def train_and_pred(train_features, test_features, train_labels, test_data,
                   lr, weight_decay, batch_size):
    loss = nn.MSELoss()

    net = get_net(train_features.shape[1])

    train_ls, _, num_epochs = train(net, train_features, train_labels, None, None,
                                    loss, lr, weight_decay, batch_size)

    base.semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'rmse')
    print('train rmse %f' % train_ls[-1])

    preds = net(test_features).detach().numpy()
    test_data['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])
    submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)
    submission.to_csv('./submission.csv', index=False)


def main():
    torch.set_default_tensor_type(torch.FloatTensor)

    path = './data'

    train_data = pd.read_csv(path + '/train.csv')
    test_data = pd.read_csv(path + '/test.csv')

    all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))

    # data preprocessing
    numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
    all_features[numeric_features] = all_features[numeric_features].apply(
            lambda x: (x - x.mean()) / (x.std()))
    all_features[numeric_features] = all_features[numeric_features].fillna(0)

    all_features = pd.get_dummies(all_features, dummy_na=True)

    n_train = train_data.shape[0]
    train_features = torch.tensor(all_features[:n_train].values, dtype=torch.float)
    test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float)
    train_labels = torch.tensor(train_data.SalePrice.values, dtype=torch.float)
    train_labels = train_labels.view(train_labels.shape[0], -1)

    # model selection
    k, lr, weight_decay, batch_size = 5, .1, 100, 64

    train_l, valid_l = k_fold(k, train_features, train_labels,
                              lr, weight_decay, batch_size)
    print('%d-fold validation: avg train rmse %f, avg valid rmse %f'
          % (k, train_l, valid_l))

    # predict and submit
    train_and_pred(train_features, test_features, train_labels, test_data,
                   lr, weight_decay, batch_size)


if __name__ == '__main__':
    main()
