import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils import data
from sklearn.metrics import mean_squared_error, mean_absolute_error


from utils.process import process_data

def load_iter(X, y, batch_size = 32):
    X = torch.FloatTensor(X)
    y = torch.FloatTensor(y)

    dataset = data.TensorDataset(X, y)
    return data.DataLoader(dataset, batch_size, shuffle = False)

def train_epoch(model, train_iter, optimizer, loss, device):
    model = model.to(device)
    model.train()
    metric = Accumulator(4)
    for X, y in train_iter:
        optimizer.zero_grad()
        X, y = X.to(device), y.to(device)
        y_hat = model(X)
        l = loss(y_hat, y)
        RMSE, MAE = calc_metrics(y_hat.cpu().detach().numpy(), y.cpu().detach().numpy())
        l.backward()
        optimizer.step()
        metric.add(RMSE, MAE, float(l.sum()), y.numel())
    return metric[0] / metric[-1], metric[1] / metric[-1], metric[2] / metric[-1]
    
def test_epoch(model, test_iter, loss, device):
    with torch.no_grad():
        model = model.to(device)
        model.eval()
        metric = Accumulator(4)
        for X, y in test_iter:
            X, y = X.to(device), y.to(device)
            y_hat = model(X)
            l = loss(y_hat, y)
            RMSE, MAE = calc_metrics(y_hat.cpu().detach().numpy(), y.cpu().detach().numpy())
            metric.add(RMSE, MAE, float(l.sum()), y.numel())
    return metric[0] / metric[-1], metric[1] / metric[-1], metric[2] / metric[-1]

def calc_metrics(y_hat, y):
    rmse = np.sqrt(mean_squared_error(y_hat, y))
    mae = mean_absolute_error(y_hat, y)
    return rmse, mae

def plot(train, test, mode):
    plt.cla()
    plt.plot(train, label = 'train ' + mode)
    plt.plot(test, label = 'test ' + mode)
    plt.legend()
    plt.title(mode)
    plt.xlabel('epoch')
    plt.savefig('./results/rnn/pics/' + mode + '.png')

class RnnNet(nn.Module):
    def __init__(self, num_inputs, num_hiddens, num_layers):
        super(RnnNet, self).__init__()
        self.rnn_layer = nn.GRU(
            num_inputs,
            num_hiddens,
            num_layers = num_layers
        )

        self.LinearSeq = nn.Sequential(
            nn.Linear(num_hiddens, 128),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.Linear(128, 32),
            nn.BatchNorm1d(32), nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, X):
        X, _ = self.rnn_layer(X)
        # print(X.shape)
        return self.LinearSeq(X[:, -1, :]).reshape(-1)

class Accumulator:
    """在n个变量上累加"""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

if __name__ == '__main__':
    data_path = 'dataset.csv'
    n_in, n_out, validation_split = 120, 6, 0.2
    train_X, train_y, test_X, test_y, scaler, pca = \
        process_data(data_path, n_in, n_out, validation_split, 
                     dropnan = True, use_rnn = True)

    # 生成数据集
    batch_size = 32
    train_iter = load_iter(train_X, train_y, batch_size)
    test_iter = load_iter(test_X, test_y, batch_size)

    # X, y = next(iter(train_iter))
    # print(f'X.shape: {X.shape}; y.shape: {y.shape}')

    kwargs = {
        'num_inputs': n_out,
        'num_hiddens': 256,
        'num_layers': 2
    }

    model = RnnNet(**kwargs)

    optimizer = torch.optim.Adam(model.parameters())
    loss = nn.MSELoss()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    epochs = 10
    train_loss_lst, test_loss_lst = [], []
    train_rmse_lst, test_rmse_lst = [], []
    train_mae_lst, test_mae_lst = [], []
    for epoch in range(epochs):
        train_rmse, train_mae, train_loss = train_epoch(model, train_iter, optimizer, loss, device)
        test_rmse, test_mae, test_loss = test_epoch(model, test_iter, loss, device)
        print('---------------------------------------')
        print(f'epoch: {epoch}, train loss: {train_loss:.5f}, train rmse: {train_rmse:.5f}, train mae: {train_mae:.5f}')
        print(f'epoch: {epoch}, test loss: {test_loss:.5f}, test rmse: {test_rmse:.5f}, test mae: {test_mae:.5f}')
        train_loss_lst.append(train_loss)
        test_loss_lst.append(test_loss)

        train_rmse_lst.append(train_rmse)
        test_rmse_lst.append(test_rmse)

        train_mae_lst.append(train_mae)
        test_mae_lst.append(test_mae)
    
    plot(train_loss_lst, test_loss_lst, 'loss')
    plot(train_rmse_lst, test_rmse_lst, 'RMSE')
    plot(train_mae_lst, test_mae_lst, 'MAE')


