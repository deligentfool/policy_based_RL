import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from torch.distributions import Categorical
import gym
import numpy as np

class net(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(net, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.fc1 = nn.Linear(self.input_dim, 256)
        self.fc2 = nn.Linear(256, self.output_dim)

        self.rewards = []
        self.log_probs = []

    def forward(self, input):
        x = self.fc1(input)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.softmax(x, 1)
        return x

    def act(self, input):
        probs = self.forward(input)
        dist = Categorical(probs)
        action = dist.sample()
        self.log_probs.append(dist.log_prob(action))
        return action.item()


class reinforce(object):
    def __init__(self, env, gamma, learning_rate, episode, render):
        self.env = env
        self.observation_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.episode = episode
        self.render = render
        self.net = net(self.observation_dim, self.action_dim)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)
        self.total_returns = []
        self.weight_reward = None

    def train(self, ):
        loss_list = []
        total_returns = torch.FloatTensor(self.total_returns)
        eps = np.finfo(np.float32).eps.item()
        total_returns = (total_returns - total_returns.mean()) / (total_returns.std() + eps)
        for log_prob, total_return in zip(self.net.log_probs, total_returns):
            loss_list.append(- log_prob * total_return)
        loss = torch.cat(loss_list, 0).sum()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def run(self, ):
        for i in range(self.episode):
            obs = self.env.reset()
            total_reward = 0
            if self.render:
                self.env.render()
            while True:
                action = self.net.act(torch.FloatTensor(np.expand_dims(obs, 0)))
                next_obs, reward, done, info = self.env.step(action)
                self.net.rewards.append(reward)
                total_reward += reward
                if self.render:
                    self.env.render()
                obs = next_obs
                if done:
                    R = 0
                    if self.weight_reward:
                        self.weight_reward = 0.99 * self.weight_reward + 0.01 * total_reward
                    else:
                        self.weight_reward = total_reward
                    for r in reversed(self.net.rewards):
                        R = R * self.gamma + r
                        self.total_returns.append(R)
                    self.total_returns = list(reversed(self.total_returns))
                    self.train()
                    del self.net.rewards[:]
                    del self.net.log_probs[:]
                    del self.total_returns[:]
                    print('episode: {}  reward: {:.1f}  weight_reward: {:.2f}'.format(i+1, total_reward, self.weight_reward))
                    break


if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    env = env.unwrapped
    test = reinforce(env, gamma=0.99, learning_rate=1e-2, episode=100000, render=False)
    test.run()