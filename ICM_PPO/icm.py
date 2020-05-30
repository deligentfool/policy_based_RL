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
        # * [obs, act, next_obs, rew, don, val, ret, adv]

    def store(self, obs, act, rew, don, val, next_obs):
        obs = np.expand_dims(obs, 0)
        next_obs = np.expand_dims(next_obs, 0)
        self.memory.append([obs, act, next_obs, rew, don, val])

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
        obs, act, next_obs, rew, don, val, ret, adv = zip(* self.memory)
        act = np.expand_dims(act, 1)
        rew = np.expand_dims(rew, 1)
        don = np.expand_dims(don, 1)
        val = np.expand_dims(val, 1)
        ret = np.expand_dims(ret, 1)
        adv = np.array(adv)
        adv = (adv - adv.mean()) / adv.std()
        adv = np.expand_dims(adv, 1)
        return np.concatenate(obs, 0), act, np.concatenate(next_obs, 0), rew, don, val, ret, adv

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
        self.fc3 = nn.Linear(128, self.output_dim)

    def forward(self, input):
        x = F.relu(self.fc1(input))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class icm(nn.Module):
    def __init__(self, observation_dim, action_dim, state_dim, reset_time):
        super(icm, self).__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.reset_time = reset_time

        self.feature = nn.Sequential(
            nn.Linear(self.observation_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, self.state_dim)
        )

        self.inverse_net = nn.Sequential(
            nn.Linear(2 * self.state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, self.action_dim)
        )

        self.forward_net_1 = nn.Sequential(
            nn.Linear(self.state_dim + self.action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256)
        )

        self.reset_net = [
            nn.Sequential(
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU()
            )
        ] * 2 * self.reset_time

        self.forward_net_2 = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, self.state_dim)
        )

    def forward(self, observation, action, next_observation):
        state = self.feature(observation)
        next_state = self.feature(next_observation)
        cat_state = torch.cat([state, next_state], 1)
        pred_action = self.inverse_net(cat_state)
        pred_action = torch.softmax(pred_action, 1)
        pred_state = self.forward_net_1(torch.cat([state, action], 1))
        for i in range(self.reset_time):
            pred_state_tmp = self.reset_net[2 * i](pred_state)
            pred_state = self.reset_net[2 * i + 1](pred_state_tmp) + pred_state
        pred_state = self.forward_net_2(pred_state)
        return pred_action, pred_state, next_state

    def intrinsic_reward(self, observation, action, next_observation):
        state = self.feature(observation)
        next_state = self.feature(next_observation)
        pred_state = self.forward_net_1(torch.cat([state, action], 1))
        for i in range(self.reset_time):
            pred_state_tmp = self.reset_net[2 * i](pred_state)
            pred_state = self.reset_net[2 * i + 1](pred_state_tmp) + pred_state
        pred_state = self.forward_net_2(pred_state)
        r_i = (pred_state - next_state).pow(2).sum()
        return r_i.detach().item()


class icm_ppo(object):
    def __init__(self, env, episode, learning_rate, gamma, lam, epsilon, capacity, render, log, value_update_iter, policy_update_iter,  state_dim, reset_time, intrinsic_weight):
        super(icm_ppo, self).__init__()
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
        self.state_dim = state_dim
        self.reset_time = reset_time
        self.intrinsic_weight = intrinsic_weight

        self.observation_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n
        self.policy_net = policy_net(self.observation_dim, self.action_dim)
        self.value_net = value_net(self.observation_dim, 1)
        self.icm_net = icm(self.observation_dim, self.action_dim, self.state_dim, self.reset_time)
        self.value_optimizer = torch.optim.Adam(self.value_net.parameters(), lr=self.learning_rate)
        self.policy_optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.icm_optimizer = torch.optim.Adam(self.icm_net.parameters(), lr=self.learning_rate)
        self.mse_func = torch.nn.MSELoss()
        self.buffer = gae_trajectory_buffer(capacity=self.capacity, gamma=self.gamma, lam=self.lam)
        self.count = 0
        self.train_count = 0
        self.weight_reward = None
        self.writer = SummaryWriter('runs/icm_ppo_cartpole')

    def train(self):
        obs, act, next_obs, rew, don, val, ret, adv = self.buffer.get()

        obs = torch.FloatTensor(obs)
        act = torch.LongTensor(act)
        next_obs = torch.FloatTensor(next_obs)
        rew = torch.FloatTensor(rew)
        don = torch.FloatTensor(don)
        val = torch.FloatTensor(val)
        ret = torch.FloatTensor(ret)
        adv = torch.FloatTensor(adv).squeeze(1)
        act_one_hot = torch.zeros(act.size(0), self.action_dim).scatter(1, act, 1)

        old_probs = self.policy_net.forward(obs)
        old_probs = old_probs.gather(1, act).squeeze(1).detach()
        value_loss_buffer = []
        for _ in range(self.value_update_iter):
            value = self.value_net.forward(obs)
            td_target = rew + self.gamma * self.value_net.forward(next_obs) * (1 - don)
            value_loss = F.smooth_l1_loss(td_target.detach(), value)
            value_loss_buffer.append(value_loss.item())
            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()
            if self.log:
                self.writer.add_scalar('value_loss', np.mean(value_loss_buffer), self.train_count)

        policy_loss_buffer = []
        for _ in range(self.policy_update_iter):
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
            if self.log:
                self.writer.add_scalar('policy_loss', np.mean(policy_loss_buffer), self.train_count)

        pred_action, pred_state, next_state = self.icm_net.forward(obs, act_one_hot, next_obs)
        forward_loss = self.mse_func(pred_state, next_state.detach())
        inverse_loss = self.mse_func(pred_action, act_one_hot)
        icm_loss = forward_loss + inverse_loss
        self.icm_optimizer.zero_grad()
        icm_loss.backward()
        self.icm_optimizer.step()

    def run(self):
        for i in range(self.episode):
            obs = self.env.reset()
            total_reward = 0
            if self.render:
                self.env.render()
            while True:
                action = self.policy_net.act(torch.FloatTensor(np.expand_dims(obs, 0)))
                next_obs, reward, done, _ = self.env.step(action)
                if self.render:
                    self.env.render()
                value = self.value_net.forward(torch.FloatTensor(np.expand_dims(obs, 0))).detach().item()
                action_one_hot = np.zeros([1, self.action_dim])
                action_one_hot[0, action] = 1
                action_one_hot = torch.FloatTensor(action_one_hot)
                intrinsic_reward = self.intrinsic_weight * self.icm_net.intrinsic_reward(torch.FloatTensor(np.expand_dims(obs, 0)), action_one_hot, torch.FloatTensor(np.expand_dims(next_obs, 0)))
                reward = max(intrinsic_reward, 0.1) + reward
                self.buffer.store(obs, action, reward / 100., done, value, next_obs)
                self.count += 1
                total_reward += reward
                obs = next_obs
                #if self.count % self.capacity == 0:
                if done:
                    self.buffer.process()
                    self.train_count += 1
                    self.train()
                    self.buffer.clear()
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
    env = gym.make('CartPole-v0').unwrapped
    test = icm_ppo(
        env=env,
        episode=10000,
        learning_rate=1e-3,
        gamma=0.99,
        lam=0.97,
        epsilon=0.2,
        capacity=20000,
        render=False,
        log=False,
        value_update_iter=3,
        policy_update_iter=3,
        state_dim=256,
        reset_time=2,
        intrinsic_weight=1e-6
    )
    test.run()
