import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

from utils.process import process_data, process_predict_data
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
    discount = 0.99        # 折扣因子
    tau = 0.005
    policy_noise = 0.2
    noise_clip = 0.5
    policy_freq = 2

    max_action = 1.0

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "discount": discount,
        "tau": tau,
        "policy_freq": policy_freq
    }

    kwargs["policy_noise"] = policy_noise * max_action
    kwargs["noise_clip"] = noise_clip * max_action

    policy = TD3(**kwargs)
    policy.load(path)
    return policy

def calc_metrics(predicted_data, true_data, scaler):
    predicted_data = np.array(predicted_data, dtype=float).reshape(-1)
    true_data = np.array(true_data, dtype=float).reshape(-1)

    assert predicted_data.shape == true_data.shape
    predicted_data = predicted_data.reshape(predicted_data.shape[0],1)
    true_data = true_data.reshape(true_data.shape[0],1)

    # scaler只能对整体反归一化，构造两个空矩阵
    temp_array_1 = np.ones((len(predicted_data),2))
    temp_array_2 = np.ones((len(predicted_data),3))

    # 反归一化
    predicted_seq = np.concatenate((temp_array_1, predicted_data,temp_array_2), axis=1)
    predicted_seq = scaler.inverse_transform(predicted_seq)
    predicted_data = predicted_seq[:,2]

    true_seq = np.concatenate((temp_array_1, true_data,temp_array_2), axis=1)
    true_seq = scaler.inverse_transform(true_seq)
    true_data = true_seq[:,2]

    # 计算RMSE,MAE
    rmse = math.sqrt(mean_squared_error(true_data,predicted_data))
    mae = mean_absolute_error(true_data,predicted_data)
    print("RMSE:{}".format(round(rmse,3)))
    print("MAE: {}".format(round(mae, 3)))

def evaluate(X, y, scaler, path):
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

    plt.cla()
    plt.plot(labels, label = 'labels')
    plt.plot(predictions, label = 'predictions')
    plt.legend()
    plt.savefig(path)

    calc_metrics(predictions, labels, scaler)
 

if __name__ == '__main__':
    # 学习过程中的奖励曲线
    evaluations_path = './results/rl/pics/evaluations.png'
    load_evaluations(evaluations_path)

    # 数据读取
    data_path = 'dataset.csv'
    n_in, n_out, validation_split = 120, 6, 0.2
    train_X, train_y, test_X, test_y, scaler, pca = \
        process_data(data_path, n_in, n_out, validation_split, 
                     dropnan = True, use_rnn = False)

    # 训练集评估
    train_path = './results/rl/pics/train.png'
    print('train set:')
    evaluate(train_X, train_y, scaler, train_path)

    print('--------------------------------------')

    # 测试集评估
    test_path = './results/rl/pics/test.png'
    print('test set:')
    evaluate(test_X, test_y, scaler, test_path)
    

    
