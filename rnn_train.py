import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils import data

from utils.process import process_data


def load_iter(X, y, batch_size = 32):
    X = torch.FloatTensor(X)
    y = torch.FloatTensor(y)

    dataset = data.TensorDataset(X, y)
    return data.DataLoader(dataset, batch_size, shuffle = False)

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

    X, y = next(iter(train_iter))
    print(f'X.shape: {X.shape}; y.shape: {y.shape}')

    kwargs = {
        'num_inputs': X.shape[-1],
        'num_hiddens': 256,
        'num_layers': 2
    }

    model = RnnNet(**kwargs)
    print(model(X).shape)


