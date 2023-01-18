import torch
from functorch.compile import memory_efficient_pointwise_fusion
import benchmark_helper

# ALL comments regarding the patetrns


def bias_gelu_dropout(input, bias):
    a = torch.add(input, bias)
    b = torch.nn.functional.gelu(a)
    c = torch.nn.functional.dropout(b, p=0.6, training=True)
    return c


def aot_fn(input, bias):
    a = torch.add(input, bias)
    b = a * 0.5 * (1.0 + torch.tanh(0.79788456 * a * (1 + 0.044715 * a * a)))
    c = torch.nn.functional.dropout(b, p=0.6, training=True)
    return c


fn = bias_gelu_dropout


# Set inputs
device = "cuda"
dtype = torch.float16
batch_size = 32
seq_len = 196
intermediate_size = 4096
# batch_size = 2
# seq_len = 4
# intermediate_size = 3
input = torch.randn(
    batch_size,
    seq_len,
    intermediate_size,
    requires_grad=True,
    device=device,
    dtype=dtype,
)
bias = torch.randn(intermediate_size, requires_grad=True, device=device, dtype=dtype)


# Get the optimized function
opt_fn = memory_efficient_pointwise_fusion(
    aot_fn, compiler_name="torchscript_nvfuser"
)


# Profile cuda kernels
benchmark_helper.profile_cuda_kernels(fn, (input, bias), "Eager")
with torch.jit.fuser("fuser2"):
    benchmark_helper.profile_cuda_kernels(opt_fn, (input, bias), "AOTAutograd")


# Time it with Torch Timer
benchmark_helper.time_with_torch_timer(fn, (input, bias), "Eager")
with torch.jit.fuser("fuser2"):
    benchmark_helper.time_with_torch_timer(opt_fn, (input, bias), "AOTAutograd")

# Time it with manual Timer
benchmark_helper.time_with_manual_timer(fn, (input, bias), "Eager")
with torch.jit.fuser("fuser2"):
    benchmark_helper.time_with_manual_timer(opt_fn, (input, bias), "AOTAutograd")
