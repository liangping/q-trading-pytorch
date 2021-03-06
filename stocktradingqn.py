# -*- coding: utf-8 -*-
"""StockTradingQN.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/github/nandahkrishna/StockTrading/blob/master/TradingDeepQNetwork.ipynb
"""

# Commented out IPython magic to ensure Python compatibility.
import os
import time
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
# %matplotlib inline

aapl = pd.read_csv('data/^GSPC.csv')
aapl.head()

aapl.shape

# train = aapl.iloc[5000:5400].reset_index(drop=True)
# test = aapl.iloc[5400:5800].reset_index(drop=True)
train = aapl
test = pd.read_csv('data/^GSPC_2011.csv')

class TradingEnvironment:
    def __init__(self, data, his_len=60):
        self.data = data
        self.his_len = his_len
        self.profits = 0
        self.positions = []
        self.position_value = 0
        self.current_t = 0
        self.done = False
        self.history = [0 for i in range(self.his_len)]
        self.reset()

    def reset(self):
        self.current_t = 0
        self.profits = 0
        self.positions = []
        self.position_value = 0
        self.done = False
        self.history = [0 for i in range(self.his_len)]
        return [self.position_value] + self.history
  
    def step(self, action):
        reward = 0
        if action == 1:
            self.positions.append(self.data.iloc[self.current_t, :]['Close'])
        elif action == 2:
            if len(self.positions) == 0:
                reward = -1
            else:
                profits = 0
                for p in self.positions:
                    profits += self.data.iloc[self.current_t, :]['Close'] - p
                reward += profits
                self.profits += profits
                self.positions = []
        self.current_t += 1
        self.position_value = 0
        for p in self.positions:
            self.position_value += self.data.iloc[self.current_t, :]['Close'] - p
        self.history.pop(0)
        self.history.append(self.data.iloc[self.current_t, :]['Close'] -
                            self.data.iloc[self.current_t - 1, :]['Close'])
        if self.current_t == len(self.data) - 1:
            done = True
        if reward > 0:
            reward = 10
        elif reward < 0:
            reward = -1
        return [self.position_value] + self.history, reward, self.done


env = TradingEnvironment(train)


class Q_Network(nn.Module):
    def __init__(self, inputs=61, actions=3):
        super(Q_Network, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(inputs, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, actions)
        )

    def forward(self, x):
        h = self.net(x)
        return h

Q = Q_Network()
Q_b = copy.deepcopy(Q)

loss = nn.MSELoss()
optimizer = optim.Adam(list(Q.parameters()), lr=0.001)

epochs = 50
step_max = len(env.data) - 1
mem_size = 200
batch_size = 50
gamma = 0.97

memory = []
total_step = 0
total_rewards = []
total_losses = []
epsilon = 1.0
epsilon_min = 0.1
epsilon_decrease = 1e-3
start_reduce_epsilon = 200
train_freq = 10
update_q_freq = 20
show_log_freq = 5

torch.autograd.set_detect_anomaly(True)

start = time.time()
for epoch in range(epochs):
    pobs = env.reset()
    step = 0
    done = False
    total_reward = 0
    total_loss = 0
    while not done and step < step_max:
        pact = np.random.randint(3)
        if np.random.rand() > epsilon:
            pact = Q(torch.from_numpy(np.array(pobs, dtype=np.float32).reshape(1, -1)))
            pact = np.argmax(pact.data)
            pact = pact.numpy()
        obs,reward, done = env.step(pact)
        memory.append([pobs, pact, reward, obs, done])
        if len(memory) > mem_size:
            memory.pop(0)
        if len(memory) == mem_size:
            if total_step % train_freq == 0:
                shuffled_memory = np.random.permutation(memory)
                memory_idx = range(len(shuffled_memory))
                for i in memory_idx[::batch_size]:
                    batch = np.array(shuffled_memory[i:i + batch_size])
                    b_pobs = np.array(batch[:, 0].tolist(), dtype=np.float32).reshape(batch_size, -1)
                    b_pact = np.array(batch[:, 1].tolist(), dtype=np.int32)
                    b_reward = np.array(batch[:, 2].tolist(), dtype=np.int32)
                    b_obs = np.array(batch[:, 3].tolist(), dtype=np.float32).reshape(batch_size, -1)
                    b_done = np.array(batch[:, 4].tolist(), dtype=np.bool)
                    q = Q(torch.from_numpy(b_pobs))
                    q_ = Q_b(torch.from_numpy(b_obs))
                    maxq = np.max(q_.data.numpy(),axis=1)
                    target = copy.deepcopy(q.data)
                    for j in range(batch_size):
                        target[j, b_pact[j]] = b_reward[j] + gamma * maxq[j] * (not b_done[j])
                        Q.zero_grad()
                        loss_val = loss(q, target)
                        total_loss += loss_val.data.item()
                        loss_val.backward(retain_graph=True)
                        # loss_val.backward()
                        optimizer.step()
        if total_step % update_q_freq == 0:
            Q_b = copy.deepcopy(Q)
        if epsilon > epsilon_min and total_step > start_reduce_epsilon:
            epsilon -= epsilon_decrease
        total_reward += reward
        pobs = obs
        step += 1
        total_step += 1
    total_rewards.append(total_reward)
    total_losses.append(total_loss)
    if (epoch + 1) % show_log_freq == 0:
        log_reward = sum(total_rewards[epoch + 1 - show_log_freq:]) / show_log_freq
        log_loss = sum(total_losses[epoch + 1 - show_log_freq:]) / show_log_freq
        elapsed_time = time.time() - start
        start = time.time()
        print('\t'.join(map(str, [epoch + 1, epsilon, total_step, log_reward, log_loss, elapsed_time])))

test_env = TradingEnvironment(test)
pobs = env.reset()
test_acts = []
test_rewards = []
for _ in range(len(test_env.data) - 1):
    pact = Q(torch.from_numpy(np.array(pobs, dtype=np.float32).reshape(1, -1)))
    pact = np.argmax(pact.data)
    test_acts.append(pact.item())
    obs, reward, done = test_env.step(pact.numpy())
    test_rewards.append(reward)
    pobs = obs
    if done:
        break
test_worth = test_env.position_value

test_worth

test_acts