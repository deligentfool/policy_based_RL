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
    def __init__(self, observation_dim, action_dim, quant_num, cosine_num):
        super(value_net, self).__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.quant_num = quant_num
        self.cosine_num = cosine_num

        self.feature_layer = nn.Sequential(
            nn.Linear(self.observation_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )

        self.cosine_layer = nn.Sequential(
            nn.Linear(self.cosine_num, 128),
            nn.ReLU()
        )

        self.psi_layer = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.action_dim)
        )

        self.quantile_fraction_layer = nn.Sequential(
            nn.Linear(128, self.quant_num),
            nn.Softmax(dim=-1)
        )

    def calc_state_embedding(self, observation):
        return self.feature_layer(observation)

    def calc_quantile_fraction(self, state_embedding):
        assert not state_embedding.requires_grad
        q = self.quantile_fraction_layer(state_embedding.detach())
        tau_0 = torch.zeros(q.size(0), 1)
        tau = torch.cat([tau_0, q], dim=-1)
        tau = torch.cumsum(tau, dim=-1)
        entropy = torch.distributions.Categorical(probs=q).entropy()
        tau_hat = ((tau[:, :-1] + tau[:, 1:]) / 2.).detach()
        return tau, tau_hat, entropy

    def calc_quantile_value(self, tau, state_embedding):
        assert not tau.requires_grad
        quants = torch.arange(0, self.cosine_num, 1.0).unsqueeze(0).unsqueeze(0)
        cos_trans = torch.cos(quants * tau.unsqueeze(-1).detach() * np.pi)
        # * cos_trans: [batch_size, quant_num, cosine_num]
        rand_feat = self.cosine_layer(cos_trans)
        # * rand_feat: [batch_size, quant_num, 128]
        x = state_embedding.unsqueeze(1)
        # * x: [batch_size, 1, 128]
        x = x * rand_feat
        # * x: [batch_size, quant_num, 128]
        value = self.psi_layer(x).transpose(1, 2)
        # * value: [batch_size, action_dim, quant_num]
        return value

    def act(self, observation, epsilon):
        if random.random() > epsilon:
            state_embedding = self.calc_state_embedding(observation)
            tau, tau_hat, _ = self.calc_quantile_fraction(state_embedding.detach())
            q_value = self.calc_q_value(state_embedding, tau, tau_hat)
            action = q_value.max(1)[1].detach().item()
        else:
            action = random.choice(list(range(self.action_dim)))
        return action

    def calc_sa_quantile_value(self, state_embedding, action, tau):
        sa_quantile_value = self.calc_quantile_value(tau.detach(), state_embedding)
        sa_quantile_value = sa_quantile_value.gather(1, action.unsqueeze(-1).expand(sa_quantile_value.size(0), 1, sa_quantile_value.size(-1))).squeeze(1)
        return sa_quantile_value

    def calc_q_value(self, state_embedding, tau, tau_hat):
        tau_delta = tau[:, 1:] - tau[:, :-1]
        tau_hat_value = self.calc_quantile_value(tau_hat.detach(), state_embedding)
        q_value = (tau_delta.unsqueeze(1) * tau_hat_value).sum(-1).detach()
        return q_value


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
    def __init__(self, env, batch_size, value_learning_rate, policy_learning_rate, quantile_learning_rate, quant_num, cosine_num, exploration, episode, gamma, alpha, auto_entropy_tuning, capacity, rho, update_iter, update_every, render, log, k=1.):
        self.env = env
        self.batch_size = batch_size
        self.value_learning_rate = value_learning_rate
        self.policy_learning_rate = policy_learning_rate
        self.quantile_learning_rate = quantile_learning_rate
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
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.value_learning_rate, eps=1e-4)
        self.capacity = capacity
        self.rho = rho
        self.update_iter = update_iter
        self.update_every = update_every
        self.render = render
        self.log = log
        self.quant_num = quant_num
        self.cosine_num = cosine_num
        self.k = k

        self.observation_dim = self.env.observation_space.shape[0]
        self.action_num = self.env.action_space.n

        self.value_net1 = value_net(self.observation_dim, self.action_num, self.quant_num, self.cosine_num)
        self.value_net2 = value_net(self.observation_dim, self.action_num, self.quant_num, self.cosine_num)
        self.target_value_net1 = value_net(self.observation_dim, self.action_num, self.quant_num, self.cosine_num)
        self.target_value_net2 = value_net(self.observation_dim, self.action_num, self.quant_num, self.cosine_num)
        self.policy_net = policy_net(self.observation_dim, self.action_num)
        self.target_value_net1.load_state_dict(self.value_net1.state_dict())
        self.target_value_net2.load_state_dict(self.value_net2.state_dict())

        self.buffer = replay_buffer(capacity=self.capacity)

        self.value_net1_params = list(self.value_net1.feature_layer.parameters()) + list(self.value_net1.cosine_layer.parameters()) + list(self.value_net1.psi_layer.parameters())
        self.value_net2_params = list(self.value_net2.feature_layer.parameters()) + list(self.value_net2.cosine_layer.parameters()) + list(self.value_net2.psi_layer.parameters())
        self.value_optimizer1 = torch.optim.Adam(self.value_net1_params, lr=self.value_learning_rate)
        self.value_optimizer2 = torch.optim.Adam(self.value_net2_params, lr=self.value_learning_rate)
        self.quantile_optimizer1 = torch.optim.RMSprop(self.value_net1.quantile_fraction_layer.parameters(), lr=self.quantile_learning_rate)
        self.quantile_optimizer2 = torch.optim.RMSprop(self.value_net2.quantile_fraction_layer.parameters(), lr=self.quantile_learning_rate)
        self.policy_optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.policy_learning_rate)

        self.weight_reward = None
        self.count = 0
        self.train_count = 0
        self.writer = SummaryWriter('run/dsac_discrete')

    def soft_update(self):
        for param, target_param in zip(self.value_net1.parameters(), self.target_value_net1.parameters()):
            target_param.detach().copy_(param.detach() * (1 - self.rho) + target_param.detach() * self.rho)
        for param, target_param in zip(self.value_net2.parameters(), self.target_value_net2.parameters()):
            target_param.detach().copy_(param.detach() * (1 - self.rho) + target_param.detach() * self.rho)

    def calc_quantile_value_loss(self, tau, value, target_value):
        # * calculate quantile value loss
        # * get the quantile huber loss
        assert not tau.requires_grad
        u = target_value.unsqueeze(-2) - value.unsqueeze(-1)
        huber_loss = 0.5 * u.abs().clamp(min=0., max=self.k).pow(2)
        huber_loss = huber_loss + self.k * (u.abs() - u.abs().clamp(min=0., max=self.k) - 0.5 * self.k)
        quantile_loss = (tau.unsqueeze(-1) - (u < 0).float()).abs() * huber_loss
        loss = quantile_loss.mean()
        return loss

    def calc_quantile_fraction_loss(self, net, embedding, actions, tau, tau_hat):
        # * calculate quantile fraction loss
        assert not tau_hat.requires_grad
        sa_quantile_hat = net.calc_sa_quantile_value(embedding, actions, tau_hat).detach()
        sa_quantile = net.calc_sa_quantile_value(embedding, actions, tau[:, 1:-1]).detach()
        gradient_tau = 2 * sa_quantile - sa_quantile_hat[:, :-1] - sa_quantile_hat[:, 1:]
        return (gradient_tau.detach() * tau[:, 1: -1]).sum(1).mean()

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

            state_embedding1 = self.value_net1.calc_state_embedding(observation)
            tau1, tau_hat1, entropy1 = self.value_net1.calc_quantile_fraction(state_embedding1.detach())
            dist1 = self.value_net1.calc_quantile_value(tau_hat1.detach(), state_embedding1)
            dist1 = dist1.gather(1, action.unsqueeze(-1).expand(self.batch_size, 1, dist1.size(2))).squeeze()
            # * use tau_hat to calculate the quantile value
            target_next_state_embedding1 = self.target_value_net1.calc_state_embedding(next_observation)
            # * double q
            eval_next_state_embedding1 = self.value_net1.calc_state_embedding(next_observation)
            next_tau1, next_tau_hat1, _ = self.value_net1.calc_quantile_fraction(eval_next_state_embedding1.detach())
            target_action1 = self.value_net1.calc_q_value(eval_next_state_embedding1, next_tau1, next_tau_hat1).max(1)[1].detach()
            target_dist1 = self.target_value_net1.calc_quantile_value(tau_hat1.detach(), target_next_state_embedding1)
            target_dist1 = target_dist1.gather(1, target_action1.unsqueeze(-1).unsqueeze(-1).expand(self.batch_size, 1, target_dist1.size(2))).squeeze()
            target_dist1 = reward + self.gamma * target_dist1 * (1. - done)
            target_q_value1 = self.target_value_net1.calc_q_value(target_next_state_embedding1, tau1, tau_hat1)
            target_q_value1 = reward + self.gamma * target_q_value1 * (1. - done)
            #value = target_dist1.gather(1, action.unsqueeze(-1).expand(self.batch_size, 1, target_dist1.size(2))).squeeze()

            state_embedding2 = self.value_net2.calc_state_embedding(observation)
            tau2, tau_hat2, entropy2 = self.value_net2.calc_quantile_fraction(state_embedding2.detach())
            dist2 = self.value_net2.calc_quantile_value(tau_hat2.detach(), state_embedding2)
            dist2 = dist2.gather(1, action.unsqueeze(-1).expand(self.batch_size, 1, dist2.size(2))).squeeze()
            # * use tau_hat to calculate the quantile value
            target_next_state_embedding2 = self.target_value_net2.calc_state_embedding(next_observation)
            eval_next_state_embedding2 = self.value_net2.calc_state_embedding(next_observation)
            next_tau2, next_tau_hat2, _ = self.value_net2.calc_quantile_fraction(eval_next_state_embedding2.detach())
            target_action2 = self.value_net2.calc_q_value(eval_next_state_embedding2, next_tau2, next_tau_hat2).max(1)[1].detach()
            target_dist2 = self.target_value_net2.calc_quantile_value(tau_hat2.detach(), target_next_state_embedding2)
            target_dist2 = target_dist2.gather(1, target_action2.unsqueeze(-1).unsqueeze(-1).expand(self.batch_size, 1, target_dist2.size(2))).squeeze()
            target_dist2 = reward + self.gamma * target_dist2 * (1. - done)
            target_q_value2 = self.target_value_net2.calc_q_value(target_next_state_embedding2, tau2, tau_hat2)
            target_q_value2 = reward + self.gamma * target_q_value2 * (1. - done)
            # * calculate the expectation directly

            value_loss1 = self.calc_quantile_value_loss(tau_hat1.detach(), dist1, target_dist1)
            value_loss2 = self.calc_quantile_value_loss(tau_hat2.detach(), dist2, target_dist2)
            value_loss1_buffer.append(value_loss1.detach().item())
            value_loss2_buffer.append(value_loss2.detach().item())

            quantile_loss1 = self.calc_quantile_fraction_loss(self.value_net1, state_embedding1, action, tau1, tau_hat1)
            quantile_loss2 = self.calc_quantile_fraction_loss(self.value_net2, state_embedding2, action, tau2, tau_hat2)

            self.quantile_optimizer1.zero_grad()
            quantile_loss1.backward(retain_graph=True)
            self.quantile_optimizer1.step()

            self.quantile_optimizer2.zero_grad()
            quantile_loss2.backward(retain_graph=True)
            self.quantile_optimizer2.step()

            self.value_optimizer1.zero_grad()
            value_loss1.backward()
            nn.utils.clip_grad_norm_(self.value_net1_params, 10)
            self.value_optimizer1.step()

            self.value_optimizer2.zero_grad()
            value_loss2.backward()
            nn.utils.clip_grad_norm_(self.value_net2_params, 10)
            self.value_optimizer2.step()

            # * calculate the expectation directly
            policy_loss = policy * (self.alpha * policy.log() - torch.min(target_q_value1, target_q_value2).detach())
            policy_loss = policy_loss.mean()

            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10)
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
        value_learning_rate=3e-4,
        policy_learning_rate=3e-4,
        quantile_learning_rate=2.5e-9,
        quant_num=32,
        cosine_num=32,
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
