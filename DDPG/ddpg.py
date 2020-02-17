import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import random
import numpy as np
from collections import deque
import gym
import math
from torch.utils.tensorboard import SummaryWriter

class replay_buffer(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=self.capacity)

    def store(self, observation, action, reward, next_observation, done):
        observation = np.expand_dims(observation, 0)
        next_observation = np.expand_dims(next_observation, 0)
        self.memory.append([observation, action, reward, next_observation, done])

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        observation, action, reward, next_observation, done = zip(* batch)
        return np.concatenate(observation, 0), action, reward, np.concatenate(next_observation, 0), done

    def __len__(self):
        return len(self.memory)


class policy_net(nn.Module):
    # * deterministic actor network, output a deterministic value as the selected action
    def __init__(self, input_dim, output_dim):
        super(policy_net, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.fc1 = nn.Linear(self.input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, self.output_dim)

    def forward(self, input):
        x = F.relu(self.fc1(input))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def act(self, input):
        action = self.forward(input).detach().item()
        return action



class value_net(nn.Module):
    def __init__(self, input1_dim, input2_dim, output_dim):
        super(value_net, self).__init__()
        self.input1_dim = input1_dim
        self.input2_dim = input2_dim
        self.output_dim = output_dim

        self.fc1 = nn.Linear(self.input1_dim + self.input2_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, self.output_dim)

    def forward(self, input1, input2):
        # * concatentate the observation and action as the input
        x = torch.cat([input1, input2], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ddpg(object):
    def __init__(self, env, episode, learning_rate, gamma, capacity, batch_size, value_iter, policy_iter, epsilon_init, decay, epsilon_min, rho, max_a, min_a, render, log):
        self.env = env
        self.episode = episode
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.capacity = capacity
        self.batch_size = batch_size
        self.value_iter = value_iter
        self.policy_iter = policy_iter
        self.epsilon_init = epsilon_init
        self.decay = decay
        self.epsilon_min = epsilon_min
        self.rho = rho
        self.max_a = max_a
        self.min_a = min_a
        self.render = render
        self.log = log

        self.observation_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.policy_net = policy_net(self.observation_dim, self.action_dim)
        self.target_policy_net = policy_net(self.observation_dim, self.action_dim)
        self.value_net = value_net(self.observation_dim, self.action_dim, 1)
        self.target_value_net = value_net(self.observation_dim, self.action_dim, 1)
        self.policy_optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.value_optimizer = torch.optim.Adam(self.value_net.parameters(), lr=self.learning_rate)
        self.target_policy_net.load_state_dict(self.policy_net.state_dict())
        self.target_value_net.load_state_dict(self.value_net.state_dict())
        self.buffer = replay_buffer(self.capacity)
        self.writer = SummaryWriter('runs/ddpg')
        self.count = 0
        self.train_count = 0
        self.weight_reward = 0
        self.epsilon = lambda x: self.epsilon_min + (self.epsilon_init - self.epsilon_min) * math.exp(- x / self.decay)

    def soft_update(self):
        for param, target_param in zip(self.value_net.parameters(), self.target_value_net.parameters()):
            target_param.detach().copy_(self.rho * target_param.detach() + (1. - self.rho) * param.detach())
        for param, target_param in zip(self.policy_net.parameters(), self.target_policy_net.parameters()):
            target_param.detach().copy_(self.rho * target_param.detach() + (1. - self.rho) * param.detach())

    def train(self):
        observation, action, reward, next_observation, done = self.buffer.sample(self.batch_size)

        observation = torch.FloatTensor(observation)
        action = torch.FloatTensor(action).unsqueeze(1)
        reward = torch.FloatTensor(reward).unsqueeze(1)
        next_observation = torch.FloatTensor(next_observation)
        done = torch.FloatTensor(done).unsqueeze(1)

        value_loss_buffer = []
        for _ in range(self.value_iter):
            target_next_action = self.target_policy_net.forward(next_observation)
            target_next_value = self.target_value_net.forward(next_observation, target_next_action)
            q_target = reward + self.gamma * (1 - done) * target_next_value
            q_target = q_target.detach()
            q = self.value_net.forward(observation, action)
            value_loss = (q - q_target).pow(2).mean()
            value_loss_buffer.append(value_loss.detach().item())

            self.value_optimizer.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
            self.value_optimizer.step()

        policy_loss_buffer = []
        for _ in range(self.policy_iter):
            current_action = self.policy_net.forward(observation)
            policy_loss = (- self.value_net.forward(observation, current_action)).mean()
            policy_loss_buffer.append(policy_loss.detach().item())

            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), 0.5)
            self.policy_optimizer.step()

        if self.log:
            self.writer.add_scalar('value_loss', np.mean(value_loss_buffer), self.train_count)
            self.writer.add_scalar('policy_loss', np.mean(policy_loss_buffer), self.train_count)

        self.soft_update()

    def run(self):
        for i in range(self.episode):
            obs = self.env.reset()
            total_reward = 0
            if self.render:
                self.env.render()
            while True:
                action = self.policy_net.act(torch.FloatTensor(np.expand_dims(obs, 0)))
                action = action + np.random.randn() * self.epsilon(self.count)
                action = np.clip(action, self.min_a, self.max_a)
                next_obs, reward, done, _ = self.env.step([action])
                if self.render:
                    self.env.render()
                self.buffer.store(obs, action, reward, next_obs, done)
                self.count += 1
                total_reward += reward
                obs = next_obs

                if done:
                    self.train_count += 1
                    self.train()
                    if not self.weight_reward:
                        self.weight_reward = total_reward
                    else:
                        self.weight_reward = self.weight_reward * 0.99 + total_reward * 0.01
                    if self.log:
                        self.writer.add_scalar('reward', total_reward, i + 1)
                        self.writer.add_scalar('weight_reward', self.weight_reward, i + 1)
                    print('episode: {}  reward: {:.2f}  weight_reward: {:.2f}'.format(i + 1, total_reward, self.weight_reward))
                    break


if __name__ == '__main__':
    env = gym.make('Pendulum-v0')
    test = ddpg(env=env,
                episode=10000,
                learning_rate=1e-3,
                gamma=0.99,
                capacity=10000,
                batch_size=64,
                value_iter=10,
                policy_iter=10,
                epsilon_init=1.,
                decay=10000,
                epsilon_min=0.01,
                rho=0.995,
                max_a=2.,
                min_a=-2.,
                render=False,
                log=False)
    test.run()