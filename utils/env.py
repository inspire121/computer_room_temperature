import numpy as np

class Env:
    def __init__(self, X, y, train = False, minibatch_size = 500):
        
        self.memory = X
        self.labels = y
        
        self.observation_space = np.array([[]], dtype = float)
        self.check = np.array([[]], dtype = float)
        
        self.observation_dim = self.memory.shape[1]
        self.action_dim = 1
        self.action_low = np.array([0.], dtype=float)
        self.action_high = np.array([1.], dtype=float)
        
        self.i = 0
        self.train = train
        
        if train:
            self._max_episode_steps = minibatch_size
        else:
            self._max_episode_steps = self.memory.shape[0]
        
        self.minibatch_size = minibatch_size
        
    def reset(self):
        self.i = 0
        
        if not self.train:
            self.observation_space = self.memory.copy()
            self.check = self.labels.copy()
        else:
            indices = np.random.choice(np.arange(self.memory.shape[0]), size = self.minibatch_size, replace = False)
            self.observation_space = self.memory[indices]
            self.check = self.labels[indices]
            
        return self.observation_space[self.i].tolist()
    
    def step(self, action):
        action_num = float(action)
        reward = 0
        MSE = (action_num - self.check[self.i])**2
        MAE = abs(action_num - self.check[self.i])
        reward = -MAE - MSE
        
        self.i += 1
        if self.i == self.observation_space.shape[0]:
            done = True
            observation = self.observation_space[self.i-1].tolist()
        else:
            done = False
            observation = self.observation_space[self.i].tolist()
       
        return observation, reward, done