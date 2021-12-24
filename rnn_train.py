import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch import nn

from utils.process import *
from utils.RnnNet import RnnNet

from rnn_test import evaluate, load_model

def train_epoch(model, train_iter, scaler, optimizer, loss, device):
    model = model.to(device)
    model.train()
    metric = Accumulator(4)
    i = 0
    for X, y in train_iter:
        optimizer.zero_grad()
        X, y = X.to(device), y.to(device)
        y_hat = model(X)

        l = loss(y_hat, y)
        l.backward()
        optimizer.step()

        RMSE, MAE = calc_metrics(y_hat.cpu().detach().numpy(), y.cpu().detach().numpy(), scaler)
        metric.add(RMSE * y.numel(), MAE * y.numel(), float(l.sum()), y.numel())

    return metric[0] / metric[-1], metric[1] / metric[-1], metric[2] / metric[-1]
    
def test_epoch(model, test_iter, scaler, loss, device):
    model = model.to(device)
    model.eval()
    metric = Accumulator(4)

    with torch.no_grad():
        for X, y in test_iter:
            X, y = X.to(device), y.to(device)
            y_hat = model(X)
            l = loss(y_hat, y)
            RMSE, MAE = calc_metrics(y_hat.cpu().detach().numpy(), y.cpu().detach().numpy(), scaler)
            metric.add(RMSE * y.numel(), MAE * y.numel(), float(l.sum()), y.numel())

    return metric[0] / metric[-1], metric[1] / metric[-1], metric[2] / metric[-1]

def train(model, epochs, scaler):
    optimizer = torch.optim.Adam(model.parameters())
    loss = nn.MSELoss()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    train_loss_lst, test_loss_lst = [], []
    train_rmse_lst, test_rmse_lst = [], []
    train_mae_lst, test_mae_lst = [], []
    for epoch in range(epochs):
        train_rmse, train_mae, train_loss = train_epoch(model, train_iter, scaler, optimizer, loss, device)
        test_rmse, test_mae, test_loss = test_epoch(model, test_iter, scaler, loss, device)
        print('---------------------------------------')
        print(f'epoch: {epoch}, train loss: {train_loss:.5f}, train rmse: {train_rmse:.5f}, train mae: {train_mae:.5f}')
        print(f'epoch: {epoch}, test loss : {test_loss:.5f}, test rmse : {test_rmse:.5f}, test mae : {test_mae:.5f}')
        
        train_loss_lst.append(train_loss)
        train_rmse_lst.append(train_rmse)
        train_mae_lst.append(train_mae)

        test_loss_lst.append(test_loss)
        test_rmse_lst.append(test_rmse)
        test_mae_lst.append(test_mae)

        if epoch % 10 == 0:
            torch.save(model.state_dict(), './results/rnn/models/RNN_epochs_' + str(epochs) + '.pth')
    
    plot_metrics(train_loss_lst, test_loss_lst, 'loss')
    plot_metrics(train_rmse_lst, test_rmse_lst, 'RMSE')
    plot_metrics(train_mae_lst, test_mae_lst, 'MAE')

    torch.save(model.state_dict(), './results/rnn/models/RNN_epochs_' + str(epochs) + '.pth')

epochs = 50

if __name__ == '__main__':
    data_path = 'dataset.csv'
    n_in, n_out, validation_split = 360, 60, 0.2
    train_X, train_y, test_X, test_y, scaler, _ = \
        process_data(data_path, n_in, n_out, validation_split, 
                     mode = 'minmax', dropnan = True, use_rnn = True)

    # 生成数据集
    batch_size = 32
    train_iter = load_iter(train_X, train_y, batch_size)
    test_iter = load_iter(test_X, test_y, batch_size)

    # 生成模型
    
    kwargs = {
        'num_inputs': 6,
        'num_hiddens': 128,
        'num_layers': 3
    }
    # model = RnnNet(**kwargs)

    model = load_model('./results/rnn/models/RNN_epochs_300.pth', kwargs)

    # 训练
    train(model, epochs, scaler)

    model_path = './results/rnn/models/RNN_epochs_' + str(epochs) + '.pth'
    
    print('train set:', end = ' ')
    train_pic_path = './results/rnn/pics/train.png'
    evaluate(train_X, train_y, kwargs, scaler, model_path, train_pic_path)
    test_pic_path = './results/rnn/pics/test.png'
    print('test set:', end = ' ')
    evaluate(test_X, test_y, kwargs, scaler, model_path, test_pic_path)

