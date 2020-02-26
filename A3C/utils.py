import torch
import numpy as np
import torch.nn as nn


def pull_and_push(opt, local_net, global_net, done, s_, b_s, b_a, b_r, gamma):
    if done:
        v_s_ = 0
    else:
        _, v_s_ = local_net.forward(torch.Tensor([s_]))
        v_s_ = v_s_.data.numpy()[0, 0]
    buffer_v_target = []
    for r in reversed(b_r):
        v_s_ = r + gamma * v_s_
        buffer_v_target.append(v_s_)
    buffer_v_target.reverse()

    loss = local_net.loss_func(torch.Tensor(b_s),
                               torch.Tensor(b_a).view(-1, 1),
                               torch.Tensor(buffer_v_target).view(-1, 1))
    opt.zero_grad()
    loss.backward()
    for l_p, g_p in zip(local_net.parameters(), global_net.parameters()):
        g_p._grad = l_p.grad
    opt.step()
    local_net.load_state_dict(global_net.state_dict())


def record(global_ep, global_ep_r, ep_r, res_queue, name):
    with global_ep.get_lock():
        global_ep.value += 1
    with global_ep_r.get_lock():
        if global_ep_r.value == 0.:
            global_ep_r.value = ep_r
        else:
            global_ep_r.value = global_ep_r.value * 0.99 + ep_r * 0.01
    res_queue.put(global_ep_r.value)
    print(
        name,
        "Ep:",
        global_ep.value,
        "| Ep_r: %.0f" % global_ep_r.value,
    )


def set_init(layers):
    for layer in layers:
        nn.init.normal_(layer.weight, mean=0., std=0.1)
        nn.init.constant_(layer.bias, 0.)