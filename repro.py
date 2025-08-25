import torch
import torch._dynamo

# Required for symbolic shape tracing
torch._dynamo.config.capture_scalar_outputs = True

# Define op with SymInt[] instead of int[]
torch.library.define("ezyang::split_with_sizes_and_clone", "(Tensor input, SymInt[] sizes) -> Tensor[]")

# Implementation of the op
def split_with_sizes_and_clone(input, sizes):
    return [t.clone() for t in torch.ops.aten.split_with_sizes.default(input, sizes)]

torch.library.impl("ezyang::split_with_sizes_and_clone", "default", split_with_sizes_and_clone)

# Use new name for abstract registration
@torch.library.register_fake("ezyang::split_with_sizes_and_clone")
def split_with_sizes_and_clone_abstract(input, sizes):
    rs = torch.ops.aten.split_with_sizes.default(input, sizes)
    return [input.new_empty(r.size()) for r in rs]

# Compiled test function
@torch.compile()
def f(sz, x):
    s0, s1 = sz.tolist()
    r0, r1 = torch.ops.ezyang.split_with_sizes_and_clone.default(x, [s0, s1])
    return torch.ops.aten.sort.default(r1)

# Repro input
N = 7312
S0 = 420
S1 = N - S0
f(torch.tensor([S0, S1]), torch.randn(N))
