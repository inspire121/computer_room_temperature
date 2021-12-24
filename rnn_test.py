import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils import data

from utils.process import *
from utils.RnnNet import RnnNet

def load_model(path, kwargs, mode = 'eval', device = 'cpu'):

    model = RnnNet(**kwargs)

    # 加载模型到指定设备
    if device == 'cpu':
        model.load_state_dict(torch.load(path, map_location='cpu'))
    else:
        model.load_state_dict(torch.load(path))

    # 指定模式
    if mode == 'train':
        model.train()
    else:
        model.eval()
    return model

def evaluate(X, Y, kwargs, scaler, model_path, pic_path = None):
    Y_hat = []
    model = load_model(model_path, kwargs)
    with torch.no_grad():
        for x, y in zip(X, Y):
            x = torch.FloatTensor(x[np.newaxis])
            y_hat = model(x).cpu().detach().numpy()
            Y_hat.append(y_hat)
    Y_hat = np.asarray(Y_hat, dtype = float)
    RMSE, MAE = calc_metrics(Y_hat, Y, scaler)
    print(f'RMSE: {RMSE:.3f}, MAE: {MAE:.3f}')

    if pic_path is not None:
        plt.cla()
        plt.plot(Y, label = 'labels')
        plt.plot(Y_hat, label = 'predictions')
        plt.legend()
        plt.savefig(pic_path)

if __name__ == '__main__':
    from rnn_train import epochs
    data_path = 'dataset.csv'
    n_in, n_out, validation_split = 120, 60, 0.2
    train_X, train_y, test_X, test_y, scaler, _ = \
        process_data(data_path, n_in, n_out, validation_split, 
                     dropnan = True, use_rnn = True)

    from rnn_train import epochs
    model_path = './results/rnn/models/RNN_epochs_' + str(epochs) + '.pth'
    
    print('train set:', end = ' ')
    train_pic_path = './results/rnn/pics/train.png'
    evaluate(train_X, train_y, scaler, model_path, train_pic_path)
    test_pic_path = './results/rnn/pics/test.png'
    print('test set:', end = ' ')
    evaluate(test_X, test_y, scaler, model_path, test_pic_path)
    
    
    