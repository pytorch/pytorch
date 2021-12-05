import torch
import time
from functorch.compile import memory_efficient_operator_authoring, clear_compile_cache
import benchmark_helper


batch_size = 32
seq_len = 196
hidden_size = 1024
def bias_dropout_res_layernorm(input, bias, residual):
    a = torch.add(input, bias)
    b = torch.nn.functional.dropout(a, p=0.7, training=True)
    c = b + residual
    d = torch.nn.functional.layer_norm(c, normalized_shape=(hidden_size,))
    return d


fn = bias_dropout_res_layernorm

clear_compile_cache()

# Set inputs
device = "cuda"
dtype = torch.float16
# batch_size = 2
# seq_len = 4
# hidden_size = 3
input = torch.randn(
    batch_size, seq_len, hidden_size, requires_grad=True, device=device, dtype=dtype
)
bias = torch.randn(hidden_size, requires_grad=True, device=device, dtype=dtype)
residual = torch.randn(
    batch_size, seq_len, hidden_size, requires_grad=False, device=device, dtype=dtype
)


# Get the optimized function
opt_fn = memory_efficient_operator_authoring(fn, compiler_name="torchscript_nvfuser")

# Use this to print the graphs for NVFuser
with torch.jit.fuser("fuser2"):
    for _ in range(10):
        fwd = opt_fn(input, bias, residual)
        loss = fwd.sum()
        loss.backward()

# Profile cuda kernels
benchmark_helper.profile_cuda_kernels(fn, (input, bias, residual), "Eager")
with torch.jit.fuser("fuser2"):
    benchmark_helper.profile_cuda_kernels(
        opt_fn, (input, bias, residual), "AOTAutograd"
    )


# Time it with Torch Timer
benchmark_helper.time_with_torch_timer(fn, (input, bias, residual), "Eager")
with torch.jit.fuser("fuser2"):
    benchmark_helper.time_with_torch_timer(
        opt_fn, (input, bias, residual), "AOTAutograd"
    )

# Time it with manual Timer
benchmark_helper.time_with_manual_timer(fn, (input, bias, residual), "Eager")
with torch.jit.fuser("fuser2"):
    benchmark_helper.time_with_manual_timer(
        opt_fn, (input, bias, residual), "AOTAutograd"
    )
