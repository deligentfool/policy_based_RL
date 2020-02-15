import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import gym
from collections import deque
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter


class gae_trajectory_buffer(object):
    def __init__(self, capacity, gamma, lam):
        self.capacity = capacity
        self.gamma = gamma
        self.lam = lam
        self.memory = deque(maxlen=self.capacity)
        # * [obs, act, rew, don, val, ret, adv]

    def store(self, obs, act, rew, don, val):
        obs = np.expand_dims(obs, 0)
        self.memory.append([obs, act, rew, don, val])

    def process(self):
        R = 0
        Adv = 0
        Value_previous = 0
        for traj in reversed(list(self.memory)):
            R = self.gamma * R * (1 - traj[3]) + traj[4]
            traj.append(R)
            # * the generalized advantage estimator(GAE)
            delta = traj[2] + Value_previous * self.gamma * (1 - traj[3]) - traj[4]
            Adv = delta + (1 - traj[3]) * Adv * self.gamma * self.lam
            traj.append(Adv)
            Value_previous = traj[4]

    def get(self):
        obs, act, rew, don, val, ret, adv = zip(* self.memory)
        act = np.expand_dims(act, 1)
        rew = np.expand_dims(rew, 1)
        don = np.expand_dims(don, 1)
        val = np.expand_dims(val, 1)
        ret = np.expand_dims(ret, 1)
        adv = np.array(adv)
        adv = (adv - adv.mean()) / adv.std()
        adv = np.expand_dims(adv, 1)
        return np.concatenate(obs, 0), act, rew, don, val, ret, adv

    def __len__(self):
        return len(self.memory)


class policy_net(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(policy_net, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fc1 = nn.Linear(self.input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, self.output_dim)

    def forward(self, input):
        x = torch.tanh(self.fc1(input))
        x = torch.tanh(self.fc2(x))
        mu = self.fc3(x)
        return mu

    def act(self, input):
        mu = self.forward(input)
        sigma = torch.ones_like(mu)
        dist = Normal(mu, sigma)
        action = dist.sample().detach().item()
        return action

    def get_distribution(self, input):
        mu = self.forward(input)
        sigma = torch.ones_like(mu)
        dist = Normal(mu, sigma)
        return dist


class value_net(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(value_net, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.fc1 = nn.Linear(self.input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, self.output_dim)

    def forward(self, input):
        x = torch.tanh(self.fc1(input))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        return x


class ppo_clip(object):
    def __init__(self, env, episode, learning_rate, gamma, lam, epsilon, capacity, render, log, value_update_iter, policy_update_iter):
        super(ppo_clip, self).__init__()
        self.env = env
        self.episode = episode
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.lam = lam
        self.epsilon = epsilon
        self.capacity = capacity
        self.render = render
        self.log = log
        self.value_update_iter = value_update_iter
        self.policy_update_iter = policy_update_iter

        self.observation_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.policy_net = policy_net(self.observation_dim, self.action_dim)
        self.value_net = value_net(self.observation_dim, 1)
        self.value_optimizer = torch.optim.Adam(self.value_net.parameters(), lr=self.learning_rate)
        self.policy_optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.buffer = gae_trajectory_buffer(capacity=self.capacity, gamma=self.gamma, lam=self.lam)
        self.count = 0
        self.train_count = 0
        self.weight_reward = None
        self.writer = SummaryWriter('runs/ppo_clip_pendulum')

    def train(self):
        obs, act, rew, don, val, ret, adv = self.buffer.get()

        obs = torch.FloatTensor(obs)
        act = torch.FloatTensor(act)
        rew = torch.FloatTensor(rew)
        don = torch.FloatTensor(don)
        val = torch.FloatTensor(val)
        ret = torch.FloatTensor(ret)
        adv = torch.FloatTensor(adv)

        old_dist = self.policy_net.get_distribution(obs)
        old_log_probs = old_dist.log_prob(act).detach()
        value_loss_buffer = []
        for _ in range(self.value_update_iter):
            value = self.value_net.forward(obs)
            value_loss = (ret - value).pow(2).mean()
            value_loss_buffer.append(value_loss.item())
            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()
            if self.log:
                self.writer.add_scalar('value_loss', np.mean(value_loss_buffer), self.train_count)

        policy_loss_buffer = []
        for _ in range(self.policy_update_iter):
            dist = self.policy_net.get_distribution(obs)
            log_probs = dist.log_prob(act)
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1. - self.epsilon, 1. + self.epsilon) * adv
            policy_loss = - torch.min(surr1, surr2).mean()
            policy_loss_buffer.append(policy_loss.item())
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()
            if self.log:
                self.writer.add_scalar('policy_loss', np.mean(policy_loss_buffer), self.train_count)

    def run(self):
        for i in range(self.episode):
            obs = self.env.reset()
            total_reward = 0
            if self.render:
                self.env.render()
            while True:
                action = self.policy_net.act(torch.FloatTensor(np.expand_dims(obs, 0)))
                next_obs, reward, done, _ = self.env.step([action])
                if self.render:
                    self.env.render()
                value = self.value_net.forward(torch.FloatTensor(np.expand_dims(obs, 0))).detach().item()
                self.buffer.store(obs, action, reward, done, value)
                self.count += 1
                total_reward += reward
                obs = next_obs
                if self.count % self.capacity == 0:
                    self.buffer.process()
                    self.train_count += 1
                    self.train()
                if done:
                    if not self.weight_reward:
                        self.weight_reward = total_reward
                    else:
                        self.weight_reward = self.weight_reward * 0.9 + total_reward * 0.1
                    if self.log:
                        self.writer.add_scalar('weight_reward', self.weight_reward, i+1)
                        self.writer.add_scalar('reward', total_reward, i+1)
                    print('episode: {}  reward: {:.2f}  weight_reward: {:.2f}  train_step: {}'.format(i+1, total_reward, self.weight_reward, self.train_count))
                    break


if __name__ == '__main__':
    env = gym.make('Pendulum-v0')
    test = ppo_clip(env=env,
                    episode=10000,
                    learning_rate=1e-3,
                    gamma=0.99,
                    lam=0.97,
                    epsilon=0.2,
                    capacity=2000,
                    render=False,
                    log=False,
                    value_update_iter=10,
                    policy_update_iter=10)
    test.run()
