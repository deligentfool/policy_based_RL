import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import torch.multiprocessing as mp
import os
from utils import pull_and_push, record, set_init
from SharedAdam import SharedAdam
os.environ["OMP_NUM_THREADS"] = "1"


class Net(nn.Module):
    def __init__(self, action_d, observation_d):
        super(Net, self).__init__()
        self.action_d = action_d
        self.observation_d = observation_d
        self.policy_layer_1 = nn.Linear(self.observation_d, 256)
        self.policy_layer_2 = nn.Linear(256, self.action_d)
        self.value_layer_1 = nn.Linear(self.observation_d, 256)
        self.value_layer_2 = nn.Linear(256, 1)
        set_init([
            self.policy_layer_1, self.policy_layer_2, self.value_layer_1,
            self.value_layer_2
        ])
        self.distribution = torch.distributions.Categorical

    def forward(self, x):
        pl_1 = F.relu6(self.policy_layer_1(x))
        policy = F.softmax(self.policy_layer_2(pl_1), dim=1)
        vl_1 = F.relu6(self.value_layer_1(x))
        value = self.value_layer_2(vl_1)
        return policy, value

    def choose_action(self, s):
        self.eval()
        prob, _ = self.forward(s)
        prob = prob.data
        m = self.distribution(prob)
        return m.sample().numpy()[0]

    def loss_func(self, s, a, value_target):
        self.train()
        prob, value = self.forward(s)
        td_error = value_target - value
        critic_loss = td_error.pow(2)

        m = self.distribution(prob)
        log_pob = m.log_prob(a)
        exp_v = log_pob * td_error.detach().squeeze()
        actor_loss = -exp_v
        loss = (critic_loss + actor_loss).mean()
        return loss


class Worker(mp.Process):
    def __init__(self, global_net, optimizer, global_episode_counter,
                 global_reward, res_queue, name, max_episode,
                 update_global_iteration, gamma):
        super(Worker, self).__init__()
        self.name = 'w' + name
        self.global_episode_counter = global_episode_counter
        self.global_reward = global_reward
        self.res_queue = res_queue
        self.global_net = global_net
        self.optimizer = optimizer
        self.max_episode = max_episode
        self.update_global_iteration = update_global_iteration
        self.gamma = gamma
        self.env = gym.make('CartPole-v0')
        self.env = self.env.unwrapped
        self.action_d = env.action_space.n
        self.observation_d = env.observation_space.shape[0]
        self.local_net = Net(self.action_d, self.observation_d)

    def run(self):
        total_step = 1
        while self.global_episode_counter.value < self.max_episode:
            s = self.env.reset()
            buffer_s, buffer_a, buffer_r = [], [], []
            episode_reward = 0
            while True:
                if self.name == 'w0':
                    self.env.render()
                a = self.local_net.choose_action(
                    torch.Tensor(s).view(-1, self.observation_d))
                s_, r, done, _ = self.env.step(a)
                if done:
                    r = -1
                episode_reward += r
                buffer_a.append(a)
                buffer_r.append(r)
                buffer_s.append(s)
                if total_step % self.update_global_iteration == 0 or done:
                    # sync
                    pull_and_push(self.optimizer, self.local_net,
                                  self.global_net, done, s_, buffer_s,
                                  buffer_a, buffer_r, self.gamma)
                    buffer_s, buffer_a, buffer_r = [], [], []
                    if done:
                        # record
                        record(self.global_episode_counter, self.global_reward,
                               episode_reward, self.res_queue, self.name)
                        break
                s = s_
                total_step += 1
        self.res_queue.put(None)


if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    action_d = env.action_space.n
    observation_d = env.observation_space.shape[0]
    global_net = Net(action_d, observation_d)
    optimizer = SharedAdam(global_net.parameters(), lr=0.0001)
    global_episode_counter, global_reward, res_queue = mp.Value(
        'i', 0), mp.Value('d', 0.), mp.Queue()
    workers = [
        Worker(global_net,
               optimizer, global_episode_counter, global_reward, res_queue,
               str(i), 10000, 10, 0.9) for i in range(mp.cpu_count())
    ]
    [worker.start() for worker in workers]
    res = []
    while True:
        r = res_queue.get()
        if r is not None:
            res.append(r)
        else:
            break
    [worker.join() for worker in workers]


    import matplotlib.pyplot as plt
    plt.plot(res)
    plt.ylabel('Moving average ep reward')
    plt.xlabel('Step')
    plt.show()