import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

from utils.process import *
from utils.env import Env
from utils.TD3 import TD3
from utils.replayer import ReplayBuffer
from sklearn.metrics import mean_squared_error, mean_absolute_error


def load_evaluations(path):
    '''生成学习过程中的奖励曲线'''
    evaluations = np.load('./results/rl/evaluations.npy')
    plt.cla()
    plt.plot(evaluations[2:])
    plt.ylabel('evaluations reward')
    plt.savefig(path)

def load_policy(state_dim, action_dim, path):
    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": 1,
    }

    policy = TD3(**kwargs)
    policy.load(path)
    return policy

def evaluate(X, y, scaler, path = None):
    # 环境生成
    env = Env(X, y)

    # 加载策略
    state_dim, action_dim = env.observation_dim, env.action_dim
    policy_path = './results/rl/models/TD3'
    policy = load_policy(state_dim, action_dim, policy_path)

    state, done = env.reset(), False
    labels = env.check
    predictions = []
    avg_reward = 0
    while not done:
        action = policy.select_action(np.array(state))
        predictions.append(action)
        state, reward, done = env.step(action)
        avg_reward += reward

    RMSE, MAE = calc_metrics(predictions, labels, scaler)
    print(f'RMSE: {RMSE:.3f}, MAE: {MAE:.3f} ')

    if path is not None:
        plt.cla()
        plt.plot(labels, label = 'labels')
        plt.plot(predictions, label = 'predictions')
        plt.legend()
        plt.savefig(path)
    
    return RMSE, MAE
 
if __name__ == '__main__':
    # 学习过程中的奖励曲线
    evaluations_path = './results/rl/pics/evaluations.png'
    load_evaluations(evaluations_path)

    # 数据读取
    data_path = 'dataset.csv'
    n_in, n_out, validation_split = 60, 60, 0.2
    train_X, train_y, test_X, test_y, scaler, pca = \
        process_data(data_path, n_in, n_out, validation_split, 
                     dropnan = True, use_rnn = False)

    # 训练集评估
    train_path = './results/rl/pics/train.png'
    print('train set:')
    evaluate(train_X, train_y, scaler, train_path)


    # 测试集评估
    test_path = './results/rl/pics/test.png'
    print('test set :')
    evaluate(test_X, test_y, scaler, test_path)
    

    
