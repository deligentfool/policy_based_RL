import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import random
from collections import deque
import numpy as np
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
        x = torch.cat([input1, input2], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class td3(object):
    def __init__(self, env, batch_size, learning_rate, exploration, episode, gamma, capacity, rho, update_iter, policy_delay, epsilon_init, decay, epsilon_min, max_a, min_a, noisy_range, render, log):
        self.env = env
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.exploration = exploration
        self.episode = episode
        self.gamma = gamma
        self.capacity = capacity
        self.rho = rho
        self.update_iter = update_iter
        self.policy_delay = policy_delay
        self.epsilon_init = epsilon_init
        self.decay = decay
        self.epsilon_min = epsilon_min
        self.max_a = max_a
        self.min_a = min_a
        self.noisy_range = noisy_range
        self.render = render
        self.log = log

        self.observation_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]

        self.value_net1 = value_net(self.observation_dim, self.action_dim, 1)
        self.value_net2 = value_net(self.observation_dim, self.action_dim, 1)
        self.target_value_net1 = value_net(self.observation_dim, self.action_dim, 1)
        self.target_value_net2 = value_net(self.observation_dim, self.action_dim, 1)
        self.policy_net = policy_net(self.observation_dim, self.action_dim)
        self.target_policy_net = policy_net(self.observation_dim, self.action_dim)
        self.target_value_net1.load_state_dict(self.value_net1.state_dict())
        self.target_value_net2.load_state_dict(self.value_net2.state_dict())
        self.target_policy_net.load_state_dict(self.policy_net.state_dict())

        self.buffer = replay_buffer(capacity=self.capacity)

        self.value_optimizer1 = torch.optim.Adam(self.value_net1.parameters(), lr=self.learning_rate)
        self.value_optimizer2 = torch.optim.Adam(self.value_net2.parameters(), lr=self.learning_rate)
        self.policy_optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)

        self.weight_reward = None
        self.count = 0
        self.train_count = 0
        self.epsilon = lambda x: self.epsilon_min + (self.epsilon_init - self.epsilon_min) * math.exp(- x / self.decay)
        self.writer = SummaryWriter('runs/td3')

    def soft_update(self):
        for param, target_param in zip(self.value_net1.parameters(), self.target_value_net1.parameters()):
            target_param.detach().copy_(param.detach() * (1 - self.rho) + target_param.detach() * self.rho)
        for param, target_param in zip(self.value_net2.parameters(), self.target_value_net2.parameters()):
            target_param.detach().copy_(param.detach() * (1 - self.rho) + target_param.detach() * self.rho)
        for param, target_param in zip(self.policy_net.parameters(), self.target_policy_net.parameters()):
            target_param.detach().copy_(param.detach() * (1 - self.rho) + target_param.detach() * self.rho)

    def train(self):
        value1_loss_buffer = []
        value2_loss_buffer = []
        policy_loss_buffer = []
        for iter in range(self.update_iter):
            observation, action, reward, next_observation, done = self.buffer.sample(self.batch_size)

            observation = torch.FloatTensor(observation)
            action = torch.FloatTensor(action).unsqueeze(1)
            reward = torch.FloatTensor(reward).unsqueeze(1)
            next_observation = torch.FloatTensor(next_observation)
            done = torch.FloatTensor(done).unsqueeze(1)

            target_next_action = self.target_policy_net.forward(next_observation)
            target_next_action = target_next_action + np.clip(np.random.randn() * self.epsilon(self.count), - self.noisy_range, self.noisy_range)
            target_next_action = torch.clamp(target_next_action, self.min_a, self.max_a).detach()

            q_min = torch.min(self.target_value_net1.forward(next_observation, target_next_action), self.target_value_net2.forward(next_observation, target_next_action))
            target_q = reward + (1 - done) * self.gamma * q_min.detach()
            q1 = self.value_net1.forward(observation, action)
            q2 = self.value_net2.forward(observation, action)
            value_loss1 = (q1 - target_q).pow(2).mean()
            value_loss2 = (q2 - target_q).pow(2).mean()
            value1_loss_buffer.append(value_loss1.detach().item())
            value2_loss_buffer.append(value_loss2.detach().item())

            self.value_optimizer1.zero_grad()
            value_loss1.backward()
            torch.nn.utils.clip_grad_norm_(self.value_net1.parameters(), 0.5)
            self.value_optimizer1.step()

            self.value_optimizer2.zero_grad()
            value_loss2.backward()
            torch.nn.utils.clip_grad_norm_(self.value_net2.parameters(), 0.5)
            self.value_optimizer2.step()

            if (iter + 1) % self.policy_delay == 0:
                current_action = self.policy_net.forward(observation)
                policy_loss = (- self.value_net1.forward(observation, current_action)).mean()
                policy_loss_buffer.append(policy_loss.detach().item())

                self.policy_optimizer.zero_grad()
                policy_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.)
                self.policy_optimizer.step()

            self.soft_update()
        if self.log:
            self.writer.add_scalar('value1_loss', np.mean(value1_loss_buffer), self.train_count)
            self.writer.add_scalar('value2_loss', np.mean(value2_loss_buffer), self.train_count)
            self.writer.add_scalar('policy_loss', np.mean(policy_loss_buffer), self.train_count)

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
                    if i > self.exploration:
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
    test = td3(env=env,
               batch_size=100,
               learning_rate=1e-3,
               exploration=300,
               episode=10000,
               gamma=0.99,
               capacity=10000,
               rho=0.995,
               update_iter=10,
               policy_delay=2,
               epsilon_init=1.,
               decay=10000,
               epsilon_min=0.01,
               max_a=2.,
               min_a=-2.,
               noisy_range=0.5,
               render=False,
               log=False)
    test.run()