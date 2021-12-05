import torch
import time
from functorch.compile import memory_efficient_operator_authoring, clear_compile_cache
import benchmark_helper


batch_size = 8192
hidden_size = 512
def layernorm_sigmoid(inp):
    a = torch.nn.functional.layer_norm(inp, normalized_shape=(hidden_size,))
    b = torch.sigmoid(a)
    return b


fn = layernorm_sigmoid

clear_compile_cache()

# Set inputs
device = "cuda"
dtype = torch.float16
# batch_size = 2
# seq_len = 4
# hidden_size = 3
inp = torch.randn(
    batch_size, hidden_size, requires_grad=True, device=device, dtype=dtype
)


# Get the optimized function
opt_fn = memory_efficient_operator_authoring(fn, compiler_name="torchscript_nvfuser")

# Use this to print the graphs for NVFuser
with torch.jit.fuser("fuser2"):
    for _ in range(10):
        fwd = opt_fn(inp)
        loss = fwd.sum()
        loss.backward()

# Profile cuda kernels
benchmark_helper.profile_cuda_kernels(fn, (inp,), "Eager")
with torch.jit.fuser("fuser2"):
    benchmark_helper.profile_cuda_kernels(
        opt_fn, (inp,), "AOTAutograd"
    )


# Time it with Torch Timer
benchmark_helper.time_with_torch_timer(fn, (inp,), "Eager")
with torch.jit.fuser("fuser2"):
    benchmark_helper.time_with_torch_timer(
        opt_fn, (inp,), "AOTAutograd"
    )

# Time it with manual Timer
benchmark_helper.time_with_manual_timer(fn, (inp,), "Eager")
with torch.jit.fuser("fuser2"):
    benchmark_helper.time_with_manual_timer(
        opt_fn, (inp,), "AOTAutograd"
    )
