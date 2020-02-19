import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import random
import numpy as np
import gym
from torch.utils.tensorboard import SummaryWriter


class policy_net(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(policy_net, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.fc1 = nn.Linear(self.input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, self.output_dim)

        self.log_probs = []
        self.rewards = []

    def forward(self, input):
        x = F.relu(self.fc1(input))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x, 1)

    def act(self, input):
        prob = self.forward(input)
        dist = Categorical(prob)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        self.log_probs.append(log_prob)
        return action.detach().item()


class q_value_net(nn.Module):
    # * different with A2C, this is a q value network that the input is observation and action
    def __init__(self, input1_dim, input2_dim, output_dim):
        super(q_value_net, self).__init__()
        self.input1_dim = input1_dim
        self.input2_dim = input2_dim
        self.output_dim = output_dim

        self.fc1 = nn.Linear(self.input1_dim + self.input2_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, self.output_dim)

    def forward(self, input1, input2):
        x = torch.cat([input1, input2], 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x


class actor_critic(object):
    def __init__(self, env, learning_rate, episode, render):
        self.env = env
        self.observation_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n
        self.learning_rate = learning_rate
        self.episode = episode
        self.render = render
        self.policy_net = policy_net(self.observation_dim, self.action_dim)
        self.q_value_net = q_value_net(self.observation_dim, self.action_dim, 1)
        self.policy_optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.value_optimizer = torch.optim.Adam(self.q_value_net.parameters(), lr=self.learning_rate)
        self.values_buffer = []
        self.next_observation_buffer = []
        self.writer = SummaryWriter('runs/actor_critic')
        self.weight_reward = None
        self.count = 0

    def train(self, ):
        values = torch.cat(self.values_buffer, 0)
        log_probs = torch.cat(self.policy_net.log_probs, 0).unsqueeze(1)
        rewards = torch.FloatTensor(self.policy_net.rewards).unsqueeze(1)
        next_observation = torch.FloatTensor(self.next_observation_buffer)

        policy_loss = (- log_probs * values)
        policy_loss = policy_loss.sum()
        self.writer.add_scalar('policy_loss', policy_loss, self.count)
        self.policy_optimizer.zero_grad()
        policy_loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.1)
        self.policy_optimizer.step()

        # * find the max value in all actions
        q_stack = None
        for action in range(self.action_dim):
            action = self.one_hot(action)
            action = torch.FloatTensor(action)
            action = action.expand(values.size(0), 2)
            tmp = self.q_value_net.forward(next_observation, action)
            if q_stack is None:
                q_stack = tmp
            else:
                q_stack = torch.cat([q_stack, tmp], 1)
        q_max = q_stack.max(1)[0].unsqueeze(1)
        value_loss = (rewards + q_max - values).pow(2).sum()
        self.writer.add_scalar('value_loss', value_loss, self.count)
        self.value_optimizer.zero_grad()
        value_loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(self.q_value_net.parameters(), 0.1)
        self.value_optimizer.step()

    def one_hot(self, action):
        one_hot_action = np.zeros(self.action_dim)
        one_hot_action[action] = 1
        return one_hot_action

    def run(self, ):
        for i in range(self.episode):
            obs = self.env.reset()
            total_reward = 0
            if self.render:
                self.env.render()
            while True:
                action = self.policy_net.act(torch.FloatTensor(np.expand_dims(obs, 0)))
                self.values_buffer.append(self.q_value_net.forward(torch.FloatTensor(np.expand_dims(obs, 0)), torch.FloatTensor(np.expand_dims(self.one_hot(action), 0))))
                next_obs, reward, done, info = self.env.step(action)
                self.policy_net.rewards.append(reward)
                self.next_observation_buffer.append(next_obs)
                self.count += 1
                total_reward += reward
                if self.render:
                    self.env.render()
                obs = next_obs
                if done:
                    if self.weight_reward:
                        self.weight_reward = 0.99 * self.weight_reward + 0.01 * total_reward
                    else:
                        self.weight_reward = total_reward
                    R = 0
                    self.train()
                    del self.policy_net.rewards[:]
                    del self.policy_net.log_probs[:]
                    del self.values_buffer[:]
                    del self.next_observation_buffer[:]
                    print('episode: {}  reward: {:.1f}  weight_reward: {:.2f}'.format(i+1, total_reward, self.weight_reward))
                    break


if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    env = env.unwrapped
    test = actor_critic(env, learning_rate=1e-3, episode=100000, render=False)
    test.run()