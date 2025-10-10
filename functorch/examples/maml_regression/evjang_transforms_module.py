# Eric Jang originally wrote an implementation of MAML in JAX
# (https://github.com/ericjang/maml-jax).
# We translated his implementation from JAX to PyTorch.

import math

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import torch
from functorch import grad, make_functional, vmap
from torch import nn
from torch.nn import functional as F


mpl.use("Agg")


class ThreeLayerNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(1, 40)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(40, 40)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(40, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x


# TODO: Use F.mse_loss


def mse_loss(x, y):
    return torch.mean((x - y) ** 2)


net, params = make_functional(ThreeLayerNet())
opt = torch.optim.Adam(params, lr=1e-3)
alpha = 0.1

K = 20
losses = []
num_tasks = 4


def sample_tasks(outer_batch_size, inner_batch_size):
    # Select amplitude and phase for the task
    As = []
    phases = []
    for _ in range(outer_batch_size):
        As.append(np.random.uniform(low=0.1, high=0.5))
        phases.append(np.random.uniform(low=0.0, high=np.pi))

    def get_batch():
        xs, ys = [], []
        for A, phase in zip(As, phases):
            x = np.random.uniform(low=-5.0, high=5.0, size=(inner_batch_size, 1))
            y = A * np.sin(x + phase)
            xs.append(x)
            ys.append(y)
        return torch.tensor(xs, dtype=torch.float), torch.tensor(ys, dtype=torch.float)

    x1, y1 = get_batch()
    x2, y2 = get_batch()
    return x1, y1, x2, y2


for it in range(20000):
    loss2 = 0.0
    opt.zero_grad()

    def get_loss_for_task(x1, y1, x2, y2):
        def inner_loss(params, x1, y1):
            f = net(params, x1)
            loss = mse_loss(f, y1)
            return loss

        grads = grad(inner_loss)(params, x1, y1)
        new_params = [(params[i] - alpha * grads[i]) for i in range(len(params))]

        v_f = net(new_params, x2)
        return mse_loss(v_f, y2)

    task = sample_tasks(num_tasks, K)
    inner_losses = vmap(get_loss_for_task)(task[0], task[1], task[2], task[3])
    loss2 = sum(inner_losses) / len(inner_losses)
    loss2.backward()

    opt.step()

    if it % 100 == 0:
        print(f"Iteration {it:d} -- Outer Loss: {loss2:.4f}")
    losses.append(loss2.detach())

t_A = torch.tensor(0.0).uniform_(0.1, 0.5)
t_b = torch.tensor(0.0).uniform_(0.0, math.pi)

t_x = torch.empty(4, 1).uniform_(-5, 5)
t_y = t_A * torch.sin(t_x + t_b)

opt.zero_grad()

t_params = params
for k in range(5):
    t_f = net(t_params, t_x)
    t_loss = F.l1_loss(t_f, t_y)

    grads = torch.autograd.grad(t_loss, t_params, create_graph=True)
    t_params = [(t_params[i] - alpha * grads[i]) for i in range(len(params))]


test_x = torch.arange(-2 * math.pi, 2 * math.pi, step=0.01).unsqueeze(1)
test_y = t_A * torch.sin(test_x + t_b)

test_f = net(t_params, test_x)

plt.plot(test_x.data.numpy(), test_y.data.numpy(), label="sin(x)")
plt.plot(test_x.data.numpy(), test_f.data.numpy(), label="net(x)")
plt.plot(t_x.data.numpy(), t_y.data.numpy(), "o", label="Examples")
plt.legend()
plt.savefig("maml-sine.png")
plt.figure()
plt.plot(np.convolve(losses, [0.05] * 20))
plt.savefig("losses.png")
