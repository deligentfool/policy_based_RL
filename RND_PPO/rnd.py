import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import gym
from collections import deque
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter


class gae_trajectory_buffer(object):
    def __init__(self, capacity, gamma, lam):
        self.capacity = capacity
        self.gamma = gamma
        self.lam = lam
        self.memory = deque(maxlen=self.capacity)
        # * [obs, next_obs, act, rew, don, val, ret, adv]

    def store(self, obs, next_obs, act, rew, don, val):
        obs = np.expand_dims(obs, 0)
        next_obs = np.expand_dims(next_obs, 0)
        self.memory.append([obs, next_obs, act, rew, don, val])

    def process(self):
        R = 0
        Adv = 0
        Value_previous = 0
        for traj in reversed(list(self.memory)):
            R = self.gamma * R * (1 - traj[4]) + traj[5]
            traj.append(R)
            # * the generalized advantage estimator(GAE)
            delta = traj[3] + Value_previous * self.gamma * (1 - traj[4]) - traj[5]
            Adv = delta + (1 - traj[4]) * Adv * self.gamma * self.lam
            traj.append(Adv)
            Value_previous = traj[5]

    def get(self):
        obs, next_obs, act, rew, don, val, ret, adv = zip(* self.memory)
        act = np.expand_dims(act, 1)
        rew = np.expand_dims(rew, 1)
        don = np.expand_dims(don, 1)
        val = np.expand_dims(val, 1)
        ret = np.expand_dims(ret, 1)
        adv = np.array(adv)
        adv = (adv - adv.mean()) / adv.std()
        adv = np.expand_dims(adv, 1)
        return np.concatenate(obs, 0), np.concatenate(next_obs, 0), act, rew, don, val, ret, adv

    def __len__(self):
        return len(self.memory)

    def clear(self):
        self.memory.clear()


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
        x = self.fc3(x)
        return F.softmax(x, 1)

    def act(self, input):
        probs = self.forward(input)
        dist = Categorical(probs)
        action = dist.sample()
        action = action.detach().item()
        return action


class value_net(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(value_net, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.fc1 = nn.Linear(self.input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.int_layer = nn.Linear(128, self.output_dim)
        self.ext_layer = nn.Linear(128, self.output_dim)

    def forward(self, input):
        x = F.relu(self.fc1(input))
        x = F.relu(self.fc2(x))
        value_int = self.int_layer(x)
        value_ext = self.ext_layer(x)
        return value_int, value_ext


class rnd(nn.Module):
    def __init__(self, input_dim):
        super(rnd, self).__init__()
        self.input_dim = input_dim

        self.predictor = nn.Sequential(
            nn.Linear(self.input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256)
        )

        self.target = nn.Sequential(
            nn.Linear(self.input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 256)
        )

        for param in self.target.parameters():
            param.requires_grad = False

    def forward(self, input):
        pre_feature = self.predictor(input)
        tar_feature = self.target(input)
        return pre_feature, tar_feature

    def calc_int_reward(self, input):
        pre_feature, tar_feature = self.forward(input)
        int_reward = 0.5 * (pre_feature - tar_feature).pow(2).sum(-1)
        return int_reward.detach().numpy()

class ppo_clip(object):
    def __init__(self, env, episode, learning_rate, gamma, lam, epsilon, capacity, render, log, value_update_iter, policy_update_iter, int_coef, ext_coef, rnd_update_prop):
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
        self.int_coef = int_coef
        self.ext_coef = ext_coef
        self.rnd_update_prop = rnd_update_prop

        self.observation_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n

        self.policy_net = policy_net(self.observation_dim, self.action_dim)
        self.value_net = value_net(self.observation_dim, 1)
        self.rnd = rnd(self.observation_dim)
        self.value_optimizer = torch.optim.Adam(self.value_net.parameters(), lr=self.learning_rate)
        self.policy_optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.rnd_optimizer = torch.optim.Adam(self.rnd.predictor.parameters(), lr=self.learning_rate)
        self.int_buffer = gae_trajectory_buffer(capacity=self.capacity, gamma=self.gamma, lam=self.lam)
        self.ext_buffer = gae_trajectory_buffer(capacity=self.capacity, gamma=self.gamma, lam=self.lam)

        self.count = 0
        self.train_count = 0
        self.weight_reward = None
        self.writer = SummaryWriter('runs/ppo_clip_rnd')

    def train(self):
        obs, next_obs, act, int_rew, don, _, _, int_adv = self.int_buffer.get()
        _, _, _, ext_rew, _, _, _, ext_adv = self.int_buffer.get()

        obs = torch.FloatTensor(obs)
        next_obs = torch.FloatTensor(next_obs)
        act = torch.LongTensor(act)
        int_rew = torch.FloatTensor(int_rew)
        ext_rew = torch.FloatTensor(ext_rew)
        don = torch.FloatTensor(don)
        int_adv = torch.FloatTensor(int_adv).squeeze(1)
        ext_adv = torch.FloatTensor(ext_adv).squeeze(1)
        adv = self.int_coef * int_adv + self.ext_coef * ext_adv

        old_probs = self.policy_net.forward(obs)
        old_probs = old_probs.gather(1, act).squeeze(1).detach()
        value_loss_buffer = []
        policy_loss_buffer = []
        rnd_loss_buffer = []
        for _ in range(self.value_update_iter):
            value_int, value_ext = self.value_net.forward(obs)
            next_value_int, next_value_ext = self.value_net.forward(next_obs)
            # * intrinsic value net
            int_td_target = int_rew + self.gamma * next_value_int * (1 - don)
            int_value_loss = F.mse_loss(int_td_target.detach(), value_int)
            # * external value net
            ext_td_target = ext_rew + self.gamma * next_value_ext * (1 - don)
            ext_value_loss = F.mse_loss(ext_td_target.detach(), value_ext)
            value_loss = 0.5 * (int_value_loss + ext_value_loss)

            value_loss_buffer.append(value_loss.item())
            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()

            probs = self.policy_net.forward(obs)
            probs = probs.gather(1, act).squeeze(1)
            ratio = probs / old_probs
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1. - self.epsilon, 1. + self.epsilon) * adv
            policy_loss = - torch.min(surr1, surr2).mean()
            policy_loss_buffer.append(policy_loss.item())
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()

            pre_feature, tar_feature = self.rnd.forward(obs)
            rnd_loss = (pre_feature - tar_feature.detach()).pow(2).mean(-1)
            mask = torch.rand(len(rnd_loss))
            mask = torch.FloatTensor((mask < self.rnd_update_prop).float())
            rnd_loss = (rnd_loss * mask).sum() / torch.max(mask.sum(), torch.FloatTensor([1.]))
            rnd_loss_buffer.append(rnd_loss)
            self.rnd_optimizer.zero_grad()
            rnd_loss.backward()
            self.rnd_optimizer.step()
        if self.log:
            self.writer.add_scalar('rnd_loss', np.mean(policy_loss_buffer), self.train_count)
            self.writer.add_scalar('policy_loss', np.mean(policy_loss_buffer), self.train_count)
            self.writer.add_scalar('value_loss', np.mean(value_loss_buffer), self.train_count)

    def run(self):
        for i in range(self.episode):
            obs = self.env.reset()
            total_reward = 0
            if self.render:
                self.env.render()
            while True:
                action = self.policy_net.act(torch.FloatTensor(np.expand_dims(obs, 0)))
                next_obs, ext_reward, done, _ = self.env.step(action)
                int_reward = self.rnd.calc_int_reward(torch.FloatTensor(np.expand_dims(obs, 0)))[0]
                if self.render:
                    self.env.render()
                value_int, value_ext = self.value_net.forward(torch.FloatTensor(np.expand_dims(obs, 0)))
                value_int = value_int.detach().item()
                value_ext = value_ext.detach().item()
                self.ext_buffer.store(obs, next_obs, action, ext_reward, done, value_ext)
                self.int_buffer.store(obs, next_obs, action, int_reward, done, value_int)
                self.count += 1
                total_reward += ext_reward
                obs = next_obs
                if self.count % self.capacity == 0:
                    self.int_buffer.process()
                    self.ext_buffer.process()
                    self.train_count += 1
                    self.train()
                    self.int_buffer.clear()
                    self.ext_buffer.clear()
                if done:
                    if not self.weight_reward:
                        self.weight_reward = total_reward
                    else:
                        self.weight_reward = self.weight_reward * 0.99 + total_reward * 0.01
                    if self.log:
                        self.writer.add_scalar('weight_reward', self.weight_reward, i+1)
                        self.writer.add_scalar('reward', total_reward, i+1)
                    print('episode: {}  reward: {:.2f}  weight_reward: {:.2f}  train_step: {}'.format(i+1, total_reward, self.weight_reward, self.train_count))
                    break


if __name__ == '__main__':
    env = gym.make('CartPole-v1').unwrapped
    test = ppo_clip(
        env=env,
        episode=10000,
        learning_rate=1e-3,
        gamma=0.99,
        lam=0.97,
        epsilon=0.2,
        capacity=20,
        render=False,
        log=False,
        value_update_iter=10,
        policy_update_iter=10,
        int_coef=1.,
        ext_coef=2.,
        rnd_update_prop=0.25
    )
    test.run()
