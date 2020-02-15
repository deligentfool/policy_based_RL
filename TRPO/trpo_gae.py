import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
import gym
from collections import deque
import random
from torch.utils.tensorboard import SummaryWriter


class gae_trajectory_buffer(object):
    def __init__(self, capacity, gamma, lam):
        self.capacity = capacity
        self.memory = deque(maxlen=self.capacity)
        # * [observation, action, reward, done, value, return, advantage]
        self.gamma = gamma
        self.lam = lam

    def store(self, observation, action, reward, done, value):
        observation = np.expand_dims(observation, 0)
        self.memory.append([observation, action, reward, done, value])

    def process(self):
        R = 0
        Adv = 0
        Value_previous = 0
        for traj in reversed(list(self.memory)):
            R = R * self.gamma * (1 - traj[3]) + traj[2]
            traj.append(R)
            # * the generalized advantage estimator(GAE)
            delta = traj[2] + self.gamma * (1 - traj[3]) * Value_previous - traj[4]
            Adv = delta + self.gamma * self.lam * Adv * (1 - traj[3])
            Value_previous = traj[4]
            traj.append(Adv)

    def get(self):
        observation, action, reward, done, value, ret, advantage = zip(* list(self.memory))
        observation = np.concatenate(observation, 0)
        action = np.expand_dims(action, 1)
        reward = np.expand_dims(reward, 1)
        done = np.expand_dims(done, 1)
        value = np.expand_dims(value, 1)
        ret = np.expand_dims(ret, 1)
        advantage = np.array(advantage)
        advantage = (advantage - advantage.mean()) / advantage.std()
        advantage = np.expand_dims(advantage, 1)
        return observation, action, reward, done, value, ret, advantage

    def clear(self):
        self.memory.clear()

    def __len__(self):
        return len(self.memory)


class gaussian_policy_net(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(gaussian_policy_net, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.fc1 = nn.Linear(self.input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, self.output_dim)

    def forward(self, input):
        x = F.tanh(self.fc1(input))
        x = F.tanh(self.fc2(x))
        mu = self.fc3(x)
        sigma = torch.ones_like(mu)
        #log_sigma = torch.zeros_like(mu)
        #sigma = torch.exp(log_sigma)
        return mu, sigma

    def act(self, input):
        mu, sigma = self.forward(input)
        dist = Normal(mu, sigma)
        action = dist.sample().detach().item()
        return action

    def distribute(self, input):
        mu, sigma = self.forward(input)
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
        x = F.tanh(self.fc1(input))
        x = F.tanh(self.fc2(x))
        x = self.fc3(x)
        return x


class trpo(object):
    def __init__(self, env, capacity, gamma, learning_rate, render, sample_size, episode, lam, delta, value_train_iter, policy_train_iter, method, backtrack_coeff, backtrack_alpha, training, log):
        self.env = env
        self.gamma = gamma
        self.lam = lam
        self.delta = delta
        self.capacity = capacity
        self.learning_rate = learning_rate
        self.render = render
        self.sample_size = sample_size
        self.episode = episode
        self.value_train_iter = value_train_iter
        self.policy_train_iter = policy_train_iter
        self.method = method
        self.backtrack_coeff = backtrack_coeff
        self.backtrack_alpha = backtrack_alpha
        self.training = training

        self.observation_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.policy_net = gaussian_policy_net(self.observation_dim, self.action_dim)
        self.old_policy_net = gaussian_policy_net(self.observation_dim, self.action_dim)
        self.value_net = value_net(self.observation_dim, 1)
        self.buffer = gae_trajectory_buffer(capacity=self.capacity, gamma=self.gamma, lam=self.lam)
        self.value_optimizer = torch.optim.Adam(self.value_net.parameters(), lr=self.learning_rate)
        self.old_policy_optimizer = torch.optim.Adam(self.old_policy_net.parameters(), lr=self.learning_rate)
        self.policy_optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.count = 0
        self.train_count = 0
        self.weight_reward = None
        self.writer = SummaryWriter('runs/trpo_gae')
        self.log = log

    def guassian_kl(self, old_policy, policy, obs):
        # * calculate the guassian distribution kl
        mu_old, sigma_old = old_policy.forward(obs)
        mu_old, sigma_old = mu_old.detach(), sigma_old.detach()

        mu, sigma = policy.forward(obs)

        kl = torch.log(sigma / sigma_old) + (sigma_old.pow(2) + (mu_old - mu).pow(2)) / (2. * sigma.pow(2)) - 0.5
        return kl.sum(-1, keepdim=True).mean()

    def flatten_grad(self, grads, hessian=False):
        grad_flat = []
        if hessian == False:
            for grad in grads:
                grad_flat.append(grad.view(-1))
            grad_flat = torch.cat(grad_flat, 0)
        else:
            for grad in grads:
                grad_flat.append(grad.contiguous().view(-1))
            grad_flat = torch.cat(grad_flat, 0).detach()
        return grad_flat

    def flatten_param(self, params):
        param_flat = []
        for param in params:
            param_flat.append(param.view(-1))
        return torch.cat(param_flat, 0).detach()

    def hessian_vector_product(self, obs, p, damping_coeff=0.1):
        # * calculate the production of hessian matrix with a vector
        # * obs : observation
        # * p : a vector
        kl = self.guassian_kl(self.old_policy_net, self.policy_net, obs)
        kl_grad = torch.autograd.grad(kl, self.policy_net.parameters(), create_graph=True)
        kl_grad = self.flatten_grad(kl_grad)

        kl_grad_p = (kl_grad * p).sum()
        kl_hessian = torch.autograd.grad(kl_grad_p, self.policy_net.parameters())
        kl_hessian = self.flatten_grad(kl_hessian, hessian=True)
        return kl_hessian + p * damping_coeff

    def conjugate_gradient(self, obs, b, cg_iters=10, eps=1e-8, residual_tol=1e-10):
        # * calculate the search direction with conjugate gradient method, find the x that makes hx = g
        # * obs : observation
        # * b : gradient
        x = torch.zeros_like(b)
        r = b.clone()
        p = r.clone()
        rTr = torch.dot(r, r)

        for _ in range(cg_iters):
            Ap = self.hessian_vector_product(obs, p)
            alpha = rTr / (torch.dot(p, Ap) + eps)
            x = x + alpha * p
            r = r - alpha * Ap

            new_rTr = torch.dot(r, r)
            beta = new_rTr / rTr
            p = r + beta * p
            rTr = new_rTr

            if rTr < residual_tol:
                break
        return x

    def update_model(self, model, params):
        index = 0
        for param in model.parameters():
            param_length = param.view(-1).size(0)
            new_param = params[index: index + param_length]
            new_param = new_param.view(param.size())
            param.detach().copy_(new_param)
            index += param_length

    def train(self):
        self.train_count += 1
        obs, act, rew, do, val, ret, adv = self.buffer.get()

        obs = torch.FloatTensor(obs)
        act = torch.FloatTensor(act)
        rew = torch.FloatTensor(rew)
        do = torch.FloatTensor(do)
        val = torch.FloatTensor(val)
        ret = torch.FloatTensor(ret)
        adv = torch.FloatTensor(adv)

        dist_old = self.policy_net.distribute(obs)
        log_prob_old = dist_old.log_prob(act).detach()
        dist = self.policy_net.distribute(obs)
        log_prob = dist.log_prob(act)
        value = self.value_net.forward(obs)

        ratio_old = torch.exp(log_prob - log_prob_old)
        policy_loss_old = (ratio_old * adv).mean()
        value_loss = (value - ret).pow(2).mean()
        self.writer.add_scalar('value_loss', value_loss, self.train_count)
        self.writer.add_scalar('policy_loss_old', policy_loss_old, self.train_count)

        for _ in range(self.value_train_iter):
            self.value_optimizer.zero_grad()
            value_loss.backward(retain_graph=True)
            self.value_optimizer.step()

        gradient = torch.autograd.grad(policy_loss_old, self.policy_net.parameters())
        gradient = self.flatten_grad(gradient)

        search_dir = self.conjugate_gradient(obs, gradient)
        # * search_dir is x in paper
        xhx = torch.dot(self.hessian_vector_product(obs, search_dir), search_dir)
        step_size = torch.sqrt((2. * self.delta) / xhx)
        old_params = self.flatten_param(self.policy_net.parameters())
        self.update_model(self.old_policy_net, old_params)

        if self.method == 'npg':
            params = old_params + step_size * search_dir
            self.update_model(self.policy_net, params)

        elif self.method == 'trpo':
            full_improve = (gradient * step_size * search_dir).sum(0, keepdim=True)
            dist_old = self.old_policy_net.distribute(obs)

            for i in range(self.policy_train_iter):
                params = old_params + self.backtrack_coeff * step_size * search_dir
                self.update_model(self.policy_net, params)

                dist = self.policy_net.distribute(obs)
                log_prob = dist.log_prob(act)
                ratio = torch.exp(log_prob - log_prob_old)
                policy_loss = (ratio * adv).mean()
                loss_improve = policy_loss - policy_loss_old
                full_improve = full_improve * self.backtrack_coeff
                improve_condition = loss_improve / full_improve

                kl = self.guassian_kl(self.old_policy_net, self.policy_net, obs)

                if kl < self.delta and improve_condition > self.backtrack_alpha:
                    self.writer.add_scalar('improve_condition', improve_condition, self.train_count)
                    self.writer.add_scalar('kl', kl, self.train_count)
                    self.writer.add_scalar('backtrack_coeff', self.backtrack_coeff, self.train_count)
                    break
                else:
                    if i == self.policy_train_iter - 1:
                        params = self.flatten_param(self.old_policy_net.parameters())
                        self.update_model(self.policy_net, params)
                        self.writer.add_scalar('improve_condition', improve_condition, self.train_count)
                        self.writer.add_scalar('kl', kl, self.train_count)
                        self.writer.add_scalar('backtrack_coeff', 0., self.train_count)
                self.backtrack_coeff = self.backtrack_coeff * 0.5
            self.backtrack_coeff = 1.

    def run(self):
        for i in range(self.episode):
            total_reward = 0
            obs = self.env.reset()
            if self.render:
                self.env.render()
            while True:
                self.count += 1
                if self.training:
                    action = self.policy_net.act(torch.FloatTensor(np.expand_dims(obs, 0)))
                    next_obs, reward, done, _ = self.env.step([action])
                    val = self.value_net.forward(torch.FloatTensor(np.expand_dims(obs, 0))).detach().item()
                    self.buffer.store(obs, action, reward, done, val)
                    if self.count % self.capacity == 0:
                        self.buffer.process()
                        self.train()
                else:
                    action = self.policy_net.act(torch.FloatTensor(np.expand_dims(obs)))
                    next_obs, reward, done, _ = self.env.step(action)

                total_reward += reward
                obs = next_obs
                if done:
                    if not self.weight_reward:
                        self.weight_reward = total_reward
                    else:
                        self.weight_reward = self.weight_reward * 0.99 + total_reward * 0.01
                    if self.log:
                        self.writer.add_scalar('weight_reward', self.weight_reward, i + 1)
                        self.writer.add_scalar('reward', total_reward, i + 1)
                    print('episode: {}  reward: {:.2f}  weight_reward: {:.2f}  train_step: {}'.format(i + 1, total_reward, self.weight_reward, self.train_count))
                    break


if __name__ == '__main__':
    env = gym.make('Pendulum-v0')
    test = trpo(env=env,
                capacity=2000,
                gamma=0.99,
                learning_rate=1e-3,
                render=False,
                sample_size=64,
                episode=5000,
                lam=0.97,
                delta=1e-2,
                value_train_iter=80,
                policy_train_iter=10,
                method='trpo',
                backtrack_coeff=1.,
                backtrack_alpha=0.5,
                training=True,
                log=False)
    test.run()