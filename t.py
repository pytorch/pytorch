import copy

from triton.testing import do_bench

import torch
import torch._inductor.config as inductor_config
from torch import nn

inductor_config.benchmark_kernel = True
inductor_config.triton.unique_kernel_names = True

torch.set_default_device("cuda")

B = 32
T = 1024
D = 768
V = 50257


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(D, V)

    def forward(self, x):
        return self.linear(x)


ref_model = Model().to(torch.bfloat16)
opt_model = copy.deepcopy(ref_model)
ce = nn.CrossEntropyLoss()


def f(m, x, label):
    ce(m(x).view(-1, V), label.view(-1)).backward()


opt_f = torch.compile(f)

x = torch.randn(B, T, D).to(torch.bfloat16)
label = torch.randint(0, V, (B, T)).to(torch.int64)

f(ref_model, x, label)
ref_grad = ref_model.linear.weight.grad

opt_f(opt_model, x, label)
act_grad = opt_model.linear.weight.grad
assert torch.allclose(
    ref_grad, act_grad, atol=1e-3, rtol=1e-3
), f"{ref_grad=}\n{act_grad=}"

torch.cuda.reset_peak_memory_stats()
for _ in range(3):
    opt_f(opt_model, x, label)

ms = do_bench(lambda: opt_f(opt_model, x, label))
peak_mem = torch.cuda.max_memory_allocated() / 10**9

print(f"{ms=}, {peak_mem=:.3f} GB")
print("bye")
