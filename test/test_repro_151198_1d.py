import torch
import torch.nn as nn
import torch.nn.functional as F
from torch._inductor import config
import os

config.fallback_random = True
torch.set_grad_enabled(False)
torch.manual_seed(0)


class Model(torch.nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.pad = torch.nn.ReflectionPad1d(1)  # this fails
        # self.pad = torch.nn.ReplicationPad1d(1)  # this also fails
        # self.pad = torch.nn.CircularPad1d(1)  # this works
        # self.pad = torch.nn.ZeroPad1d(1)  # this works

    def forward(self, x):
        x = self.pad(x)
        x = x.view(x.size(0), -1)
        x = torch.chunk(x, 2, dim=1)
        return x[0] - x[1]


model = Model().eval().cuda()


x = torch.randn(1, 3, 4).cuda()

inputs = [x]


def run_test(model, inputs, backend):
    if backend != "eager":
        model = torch.compile(model, backend=backend)
    torch.manual_seed(0)
    output = model(*inputs)
    return output


output = run_test(model, inputs, 'eager')
c_output = run_test(model, inputs, 'inductor')
fp64 = run_test(model.to(dtype=torch.float64), [x.to(dtype=torch.float64)], 'eager')


print(output)
print(c_output)
print(fp64)
print(torch.allclose(output, c_output, 1e-3, 1e-3, equal_nan=True))
print(torch._dynamo.utils.same(output, c_output, fp64))
print(torch.max(torch.abs(output - c_output)))
