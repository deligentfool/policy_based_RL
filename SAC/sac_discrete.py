import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
from collections import deque
import random
from torch.utils.tensorboard import SummaryWriter
import numpy as np


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


class value_net(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(value_net, self).__init__()
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


class policy_net(nn.Module):
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
        policy = F.softmax(self.fc3(x), dim=-1)
        return policy

    def act(self, input):
        policy = self.forward(input)
        dist = torch.distributions.Categorical(policy)
        action = dist.sample()
        return action[0].item()


class sac_discrete(object):
    def __init__(self, env, batch_size, learning_rate, exploration, episode, gamma, alpha, auto_entropy_tuning, capacity, rho, update_iter, update_every, render, log):
        self.env = env
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.exploration = exploration
        self.episode = episode
        self.gamma = gamma
        self.auto_entropy_tuning = auto_entropy_tuning
        if not self.auto_entropy_tuning:
            self.alpha = alpha
        else:
            self.log_alpha = torch.zeros(1, requires_grad=True)
            self.alpha = self.log_alpha.exp()
            # * set the max possible entropy as the target entropy
            self.target_entropy = -np.log((1. / self.env.action_space.n)) * 0.98
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.learning_rate, eps=1e-4)
        self.capacity = capacity
        self.rho = rho
        self.update_iter = update_iter
        self.update_every = update_every
        self.render = render
        self.log = log

        self.observation_dim = self.env.observation_space.shape[0]
        self.action_num = self.env.action_space.n

        self.value_net1 = value_net(self.observation_dim, self.action_num)
        self.value_net2 = value_net(self.observation_dim, self.action_num)
        self.target_value_net1 = value_net(self.observation_dim, self.action_num)
        self.target_value_net2 = value_net(self.observation_dim, self.action_num)
        self.policy_net = policy_net(self.observation_dim, self.action_num)
        self.target_value_net1.load_state_dict(self.value_net1.state_dict())
        self.target_value_net2.load_state_dict(self.value_net2.state_dict())

        self.buffer = replay_buffer(capacity=self.capacity)

        self.value_optimizer1 = torch.optim.Adam(self.value_net1.parameters(), lr=self.learning_rate)
        self.value_optimizer2 = torch.optim.Adam(self.value_net2.parameters(), lr=self.learning_rate)
        self.policy_optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)

        self.weight_reward = None
        self.count = 0
        self.train_count = 0
        self.writer = SummaryWriter('runs/sac_discrete')

    def soft_update(self):
        for param, target_param in zip(self.value_net1.parameters(), self.target_value_net1.parameters()):
            target_param.detach().copy_(param.detach() * (1 - self.rho) + target_param.detach() * self.rho)
        for param, target_param in zip(self.value_net2.parameters(), self.target_value_net2.parameters()):
            target_param.detach().copy_(param.detach() * (1 - self.rho) + target_param.detach() * self.rho)

    def train(self):
        observation, action, reward, next_observation, done = self.buffer.sample(self.batch_size)

        observation = torch.FloatTensor(observation)
        action = torch.LongTensor(action).unsqueeze(1)
        reward = torch.FloatTensor(reward).unsqueeze(1)
        next_observation = torch.FloatTensor(next_observation)
        done = torch.FloatTensor(done).unsqueeze(1)

        value_loss1_buffer = []
        value_loss2_buffer = []
        policy_loss_buffer = []
        for _ in range(self.update_iter):
            policy = self.policy_net.forward(next_observation)
            target_q_value1 = self.target_value_net1.forward(next_observation)
            target_q_value2 = self.target_value_net2.forward(next_observation)
            # * calculate the expectation directly
            target_q = reward + (1 - done) * self.gamma * (policy * (torch.min(target_q_value1, target_q_value2) - self.alpha * policy.log())).mean(dim=1).unsqueeze(-1)
            target_q = target_q.detach()

            q1 = self.value_net1.forward(observation).gather(dim=1, index=action)
            q2 = self.value_net2.forward(observation).gather(dim=1, index=action)
            value_loss1 = (q1 - target_q).pow(2).mean()
            value_loss2 = (q2 - target_q).pow(2).mean()
            value_loss1_buffer.append(value_loss1.detach().item())
            value_loss2_buffer.append(value_loss2.detach().item())

            self.value_optimizer1.zero_grad()
            value_loss1.backward()
            nn.utils.clip_grad_norm_(self.value_net1.parameters(), 0.5)
            self.value_optimizer1.step()

            self.value_optimizer2.zero_grad()
            value_loss2.backward()
            nn.utils.clip_grad_norm_(self.value_net2.parameters(), 0.5)
            self.value_optimizer2.step()

            # * calculate the expectation directly
            policy_loss = policy * (self.alpha * policy.log() - torch.min(target_q_value1, target_q_value2).detach())
            policy_loss = policy_loss.mean()

            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
            self.policy_optimizer.step()

            if self.auto_entropy_tuning:
                self.alpha_optimizer.zero_grad()
                entropy_loss = -(self.log_alpha * (policy.log() + self.target_entropy).detach()).mean()
                entropy_loss.backward()
                nn.utils.clip_grad_norm_([self.log_alpha], 0.2)
                self.alpha_optimizer.step()

                self.alpha = self.log_alpha.exp()

            self.soft_update()
        if self.log:
            self.writer.add_scalar('value_loss1', np.mean(value_loss1_buffer), self.train_count)
            self.writer.add_scalar('value_loss2', np.mean(value_loss2_buffer), self.train_count)
            self.writer.add_scalar('policy_loss', np.mean(policy_loss_buffer), self.train_count)

    def run(self):
        for i in range(self.episode):
            obs = self.env.reset()
            total_reward = 0
            if self.render:
                self.env.render()
            while True:
                if i >= self.exploration:
                    action = self.policy_net.act(torch.FloatTensor(np.expand_dims(obs, 0)))
                else:
                    action = random.choice(list(range(self.action_num)))
                next_obs, reward, done, _ = self.env.step(action)
                if self.render:
                    self.env.render()
                self.buffer.store(obs, action, reward, next_obs, done)
                self.count += 1
                total_reward += reward
                obs = next_obs

                if (self.count % self.update_every) == 0 and i >= self.exploration:
                    self.train_count += 1
                    self.train()
                if done:
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
    env = gym.make('CartPole-v1').unwrapped
    test = sac_discrete(
        env=env,
        batch_size=64,
        learning_rate=3e-4,
        exploration=3000,
        episode=10000,
        gamma=0.99,
        alpha=None,
        auto_entropy_tuning=True,
        capacity=100000,
        rho=0.995,
        update_iter=3,
        update_every=5,
        render=False,
        log=False
    )
    test.run()
