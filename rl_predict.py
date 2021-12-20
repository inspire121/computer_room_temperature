import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from torch.cuda.memory import reset_peak_memory_stats
from rl_train import train

from utils.process import process_predict_data
from utils.env import Env
from rl_test import *

n_in, n_out = 120, 6
data_path = 'dataset.csv'
train_X, train_y, test_X, test_y, scaler, pca = \
        process_data(data_path, n_in, n_out, 
                     dropnan = True, use_rnn = False)

def predict(sequence):
    n_in, n_out = 120, 6
    
    features = process_predict_data(sequence, n_in, n_out, scaler, pca)
    # print(features.shape)
    # env = Env(train_X, train_y)
    # state = env.reset()
    # print(len(state))
    policy_path = './results/rl/models/TD3'
    policy = load_policy(23, 1, policy_path)

    predictions = policy.select_action(features[-1])
    print(predictions)
    temp1, temp2 = np.ones((2,)), np.ones((3,))
    seq = np.concatenate((temp1, predictions, temp2), axis=0).reshape((1, -1))
    seq = scaler.inverse_transform(seq).reshape(-1)

    return seq[2]

if __name__ == '__main__':
    test = np.random.uniform(0, 1, size = (360, 6))
    print(predict(test))
