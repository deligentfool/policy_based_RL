import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
import numpy as np
import random
from collections import deque
from torch.utils.tensorboard import SummaryWriter


class replay_buffer(object):
    # * a different implement of replay buffer
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=self.capacity)
        self.memory.append([])

    def store(self, observation, action, reward, policy, done):
        observation = np.expand_dims(observation, 0)
        self.memory[-1].append([observation, action, reward, policy, done])

    def sample(self, batch_size=None):
        if not batch_size:
            batch = self.memory[-1]
        else:
            batch_list = random.sample(list(self.memory)[: -1], batch_size)
            batch = []
            for i in batch_list:
                batch.extend(i)

        observation, action, reward, policy, done = zip(* batch)
        return np.concatenate(observation, 0), action, reward, np.concatenate(policy, 0), done

    def create(self):
        self.memory.append([])

    def __len__(self):
        return len(self.memory)


class policy_value_net(nn.Module):
    # * a network for the discrete case
    def __init__(self, observation_dim, action_dim):
        super(policy_value_net, self).__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim

        self.policy_fc1 = nn.Linear(self.observation_dim, 128)
        self.policy_fc2 = nn.Linear(128, 128)
        self.policy_fc3 = nn.Linear(128, self.action_dim)

        self.value_fc1 = nn.Linear(self.observation_dim, 128)
        self.value_fc2 = nn.Linear(128, 128)
        self.value_fc3 = nn.Linear(128, self.action_dim)

    def forward(self, observation):
        policy_x = F.tanh(self.policy_fc1(observation))
        policy_x = F.tanh(self.policy_fc2(policy_x))
        policy_x = self.policy_fc3(policy_x)
        policy = F.softmax(policy_x, 1).clamp(max=1-1e-20)

        q_value_x = F.tanh(self.value_fc1(observation))
        q_value_x = F.tanh(self.value_fc2(q_value_x))
        q_value = self.value_fc3(q_value_x)

        value = (policy * q_value).sum(1, keepdim=True)
        return policy, q_value, value


class acer(object):
    # * without trust region policy optimization
    def __init__(self, env, episode, capacity, learning_rate, exploration, c, gamma, batch_size, entropy_weight, replay_ratio, render, log):
        self.env = env
        self.episode = episode
        self.capacity = capacity
        self.learning_rate = learning_rate
        self.exploration = exploration
        self.c = c
        self.gamma = gamma
        self.batch_size = batch_size
        self.entropy_weight = entropy_weight
        self.replay_ratio = replay_ratio
        self.render = render
        self.log = log

        self.observation_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n
        self.net = policy_value_net(self.observation_dim, self.action_dim)
        self.buffer = replay_buffer(self.capacity)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)
        self.weight_reward = None
        self.writer = SummaryWriter('runs/acer_cartpole')
        self.train_count = 0

    def compute_loss(self, policies, q_values, values, actions, rewards, retrace, dones, behavior_policies):
        loss = 0
        for i in reversed(range(policies.size(0))):
            rho = (policies[i] / behavior_policies[i]).detach()

            retrace = rewards[i] + self.gamma * retrace * (1. - dones[i])
            advantage = retrace - values[i].squeeze()

            log_policy_action = policies[i].gather(0, actions[i]).log()
            rho_action = rho.gather(0, actions[i])
            actor_loss = -torch.clamp(rho_action, max=self.c).detach() * log_policy_action * advantage.detach()
            rho_correction = torch.clamp(1 - self.c / rho, min=0.).detach()
            actor_loss -= (rho_correction * policies[i].log() * (q_values[i] - values[i]).detach()).sum()

            entropy = self.entropy_weight * -(policies[i] * policies[i].log()).sum()
            critic_loss = (retrace - q_values[i].gather(0, actions[i])).pow(2).sum()

            loss += (critic_loss + actor_loss - entropy)

            retrace = torch.clamp(rho_action, max=self.c).detach() * (retrace - q_values[i].gather(0, actions[i])) + values[i]
            retrace = retrace.squeeze().detach()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def on_policy_train(self, next_observation):
        observations, actions, rewards, behavior_policies, dones = self.buffer.sample()

        observations = torch.FloatTensor(observations)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        behavior_policies = torch.FloatTensor(behavior_policies)
        dones = torch.FloatTensor(dones)

        policies, q_values, values = self.net.forward(observations)

        _, _, retrace = self.net.forward(torch.FloatTensor(np.expand_dims(next_observation, 0)))
        retrace = retrace.squeeze().detach()
        loss = self.compute_loss(policies, q_values, values, actions, rewards, retrace, dones, behavior_policies)
        if self.log:
            self.writer.add_scalar('on_policy_loss', loss, self.train_count)

    def off_policy_train(self):
        loss_list = []
        for _ in range(np.random.poisson(self.replay_ratio)):
            observations, actions, rewards, behavior_policies, dones = self.buffer.sample(self.batch_size)

            observations = torch.FloatTensor(observations)
            actions = torch.LongTensor(actions)
            rewards = torch.FloatTensor(rewards)
            behavior_policies = torch.FloatTensor(behavior_policies)
            dones = torch.FloatTensor(dones)

            policies, q_values, values = self.net.forward(observations)

            _, _, retrace = self.net.forward(observations[-1].unsqueeze(0))
            retrace = retrace.squeeze().detach()
            loss = self.compute_loss(policies, q_values, values, actions, rewards, retrace, dones, behavior_policies)
            loss_list.append(loss)
        if self.log:
            self.writer.add_scalar('off_policy_loss', np.mean(loss_list), self.train_count)

    def run(self):
        for i in range(self.episode):
            obs = self.env.reset()
            total_reward = 0
            if self.render:
                self.env.render()
            while True:
                policy, _, _ = self.net.forward(torch.FloatTensor(np.expand_dims(obs, 0)))
                action = policy.multinomial(1).item()
                next_obs, reward, done, info = self.env.step(action)
                total_reward += reward
                if self.render:
                    self.env.render()
                policy = policy.detach().numpy()
                self.buffer.store(obs, action, reward / 10., policy, done)
                obs = next_obs

                if done:
                    if len(self.buffer) > self.exploration:
                        self.on_policy_train(next_obs)
                        self.off_policy_train()
                        self.train_count += 1
                    if not self.weight_reward:
                        self.weight_reward = total_reward
                    else:
                        self.weight_reward = 0.99 * self.weight_reward + 0.01 * total_reward
                    self.buffer.create()
                    if self.log:
                        self.writer.add_scalar('reward', total_reward, i + 1)
                        self.writer.add_scalar('weight_reward', self.weight_reward, i + 1)
                    print('episode: {}  reward: {}  weight_reward: {:.2f}'.format(i + 1, total_reward, self.weight_reward))
                    break


if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    env = env.unwrapped
    test = acer(env=env,
                episode=10000,
                capacity=10000,
                learning_rate=1e-3,
                exploration=1000,
                c=1.,
                gamma=0.99,
                batch_size=16,
                entropy_weight=1e-4,
                replay_ratio=2,
                render=False,
                log=False)
    test.run()