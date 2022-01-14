
import torch

import copy
import lazy_tensor_core
import lazy_tensor_core.debug.metrics as metrics
import torch.optim as optim
import lazy_tensor_core.core.lazy_model as ltm

lazy_tensor_core._LAZYC._ltc_init_ts_backend()

class M(torch.nn.Module):
    def __init__(self):
        super(M, self).__init__()
        self.bias = torch.nn.Parameter(torch.rand(2, 2, requires_grad=True))

    def forward(self, input):
        return input * self.bias

class Optimizer:
    def __init__(self, parameters):
        params = list(parameters)
        assert len(params) == 1
        self.param = params[0]

    def step(self):
        diff = torch.ops.lazy_cuda.optim(self.param, self.param.grad, {"test_key" : torch.rand(1)})

    def zero_grad(self):
        if self.param.grad is not None:
            self.param.grad.zero_()

dev = 'lazy'
x = torch.rand(2, 2, device=dev, requires_grad=True)

model = M().to(device = dev)
model.train()
print(model.bias.requires_grad)
print(x.requires_grad)
optimizer = Optimizer(model.parameters())
niter = 3
for _ in range(niter):
    optimizer.zero_grad()
    pred = model(x)
    y = torch.rand_like(pred).to(device = dev)
    (y - pred).sum().backward()
    optimizer.step()
    ltm.mark_step()