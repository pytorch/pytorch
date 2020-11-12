import torch
import time
import os
os.environ["OMP_NUM_THREADS"] = "1"

@torch.jit.script
def f(x, y):
    return x + y

a = torch.tensor([1])
b = torch.tensor([1])

for _ in range(3):
    a = f(a, b)

tic = time.perf_counter()

for _ in range(1000000):
     a = f(a, b)

toc = time.perf_counter()
print("No fusion: ", toc - tic)

torch._C._jit_override_can_fuse_on_cpu(True)
torch._C._jit_set_texpr_fuser_enabled(True)
torch._C._debug_set_fusion_group_inlining(False)

@torch.jit.script
def f(x, y):
    return x + y

for _ in range(3):
    a = f(a, b)

tic = time.perf_counter()

for _ in range(1000000):
     a = f(a, b)

toc = time.perf_counter()
print("With fusion: ", toc - tic)
print(torch.jit.last_executed_optimized_graph())
