import numpy as np
import matplotlib.pyplot as plt
from torch import optim

from utils.process import process_data
from utils.env import Env
from utils.TD3 import TD3
from utils.replayer import ReplayBuffer

from rl_test import *

import time
from datetime import datetime

def eval_policy(policy, env, eval_episodes=10):
    avg_reward = 0
    for _ in range(eval_episodes):
        state, done = env.reset(), False
        while not done:
            action = policy.select_action(np.array(state))
            state, reward, done = env.step(action)
            avg_reward += reward
    avg_reward /= eval_episodes

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    return avg_reward

def train(train_X, train_y, minibatch_size, start_timesteps, eval_freq, max_timesteps):
    # 相关参数设置
    
    expl_noise = 0.1        # 噪音
    batch_size = 256
    

    env = Env(train_X, train_y, minibatch_size = minibatch_size, train = True)
    state_dim = env.observation_dim
    action_dim = env.action_dim
    max_action = 1.0

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
    }

    policy = TD3(**kwargs)
    replay_buffer = ReplayBuffer(state_dim, action_dim)
    evaluations = [eval_policy(policy, env)]

    state, done = env.reset(), False
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0

    for t in range(int(max_timesteps)):
        episode_timesteps += 1
        
        # Select action randomly or according to policy
        if t < start_timesteps:
            action = np.random.uniform(np.array([0]), np.array([1]))
        else:
            action = (
                policy.select_action(np.array(state))
                + np.random.normal(0, max_action * expl_noise, size=action_dim)
            )#.clip(0, 1)
            
        # Perform action
        next_state, reward, done = env.step(action)
        done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0
        
        # Store data in replay buffer
        replay_buffer.add(state, action, next_state, reward, done_bool)

        
        state = next_state
        episode_reward += reward
        
        # Train agent after collecting sufficient data
        if t >= start_timesteps:
            policy.train(replay_buffer, batch_size)
            
        if done: 
            # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
            print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
            # Reset environment
            state, done = env.reset(), False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1 
        
        # Evaluate episode
        if (t + 1) % eval_freq == 0:
            evaluations.append(eval_policy(policy, env))
            np.save(f'./results/rl/evaluations', evaluations)
            policy.save(f"./results/rl/models/TD3")
            state, done = env.reset(), False

if __name__ == '__main__':
    # 数据读取与存储
    data_path = 'dataset.csv'
    n_in, n_out, validation_split = 120, 60, 0.2
    train_X, train_y, test_X, test_y, scaler, pca = \
        process_data(data_path, n_in, n_out, validation_split, 
                    dropnan = True, use_rnn = False)

    kwargs = {
        'train_X': train_X,
        'train_y': train_y,
        'start_timesteps': 500,  
        'eval_freq': 500,
        'minibatch_size': 10,       
        'max_timesteps': 40000
    }

    # 训练
    train(**kwargs)

    # 学习过程中的奖励曲线
    evaluations_path = './results/rl/pics/evaluations.png'
    load_evaluations(evaluations_path)

    # 训练集评估
    train_path = './results/rl/pics/train.png'
    print('train set:', end = ' ')
    train_rmse, train_mae = evaluate(train_X, train_y, scaler, train_path)


    # 测试集评估
    test_path = './results/rl/pics/test.png'
    print('test set: ', end = ' ')
    test_rmse, test_mae = evaluate(test_X, test_y, scaler, test_path)
    
    # 记录训练结果
    time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    steps = 'steps: ' + str(kwargs['max_timesteps'])
    minibatch_size = 'minibatch_size: ' + str(kwargs['minibatch_size'])
    start_timesteps = 'start_timesteps： ' + str(kwargs['start_timesteps'])
    train_metrics = 'train set: RMSE: {:.3f}, MAE: {:.3f}'.format(train_rmse, train_mae)
    test_metrics = 'test set : RMSE: {:.3f}, MAE: {:.3f}'.format(test_rmse, test_mae)

    lines = [time_str, steps, start_timesteps, minibatch_size, train_metrics, test_metrics]
    write_log(lines, './results/rl')
        

    



