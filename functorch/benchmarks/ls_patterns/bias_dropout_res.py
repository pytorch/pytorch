import torch
import time
from functorch.compile import memory_efficient_pointwise_fusion, clear_compile_cache
import benchmark_helper

### ALL comments regarding the patetrns

# @brief fused bias, dropout, and residual at the end of Attention and FFN,
# store dropped position in mask, it's not in-place
#
# @param total_count total elements
# @param ratio drop ratio
# @param out [batch_size, seq_len, hidden_size], float and __half
# @param in [batch_size, seq_len, hidden_size], float and __half
# @param mask [batch_size, seq_len, hidden_size], uint8 type
# @param bias [hidden_size], ffn bias
# @param residual [batch_size, seq_len, hidden_size], float and __half
#
#   output4.x = (input4.x + b4.x) * scale * m[0] + res4.x;
#   output4.y = (input4.y + b4.y) * scale * m[1] + res4.y;
#   output4.z = (input4.z + b4.z) * scale * m[2] + res4.z;
#   output4.w = (input4.w + b4.w) * scale * m[3] + res4.w;
#
#   out4[i] = output4;


# def f(a, b, c):
#
#     # 5 reads in total = 3 primary input reads + 2 intermediate input reads
#     # 4 writes in total = 1 for each op + 1 saved mask
#     x = a + b
#     y = dropout(x)
#     z = y + c
#     return z
#
#
# def f_backward(dz):
#     # 3 reads in total = 1 input read for dz and 1 intermediate saved read for mask + 1 for sum
#     # 3 writes in total = 1 for each op + 1 for the dc
#     dy = dz
#     dc = dz
#
#     dx = masked_scale(dy, self.saved_mask)
#
#     da = dx
#     db = dx.sum()
#
#     return (da, db, dc)
#
#
#
#
# For fused bwd, it is 2 reads + 3 writes. Considering, writes to be not on the critical path. Max we could benefit is 3 reads/2 writes = 1.5x
#
#  graph(%self : __torch__.torch.fx.graph_module.___torch_mangle_2.GraphModule,
#        %lt.1 : Tensor,
#        %tangents_1.1 : Tensor):
#    %4 : int[] = prim::Constant[value=[1024]]()
#    %37 : bool = prim::CudaFusionGuard[types=[Float(32, 196, 1024, strides=[200704, 1024, 1], requires_grad=0, device=cuda:0), Bool(32, 196, 1024, strides=[200704, 1024, 1], requires_grad=0, device=cuda:0)]](%tangents_1.1, %lt.1)
#    %35 : Tensor, %36 : Tensor = prim::If(%37)
#      block0():
#        %sum_1.4 : Tensor, %mul_1.4 : Tensor = prim::CudaFusionGroup_0(%tangents_1.1, %lt.1)
#        -> (%sum_1.4, %mul_1.4)
#      block1():
#        %sum_1.1 : Tensor, %mul_1.1 : Tensor = prim::FallbackGraph_1(%tangents_1.1, %lt.1)
#        -> (%sum_1.1, %mul_1.1)
#    %view.1 : Tensor = aten::view(%35, %4) # <eval_with_key>.10:10:11
#    %23 : Tensor[] = prim::ListConstruct(%36, %view.1, %tangents_1.1)
#    return (%23)
#  with prim::CudaFusionGroup_0 = graph(%8 : Float(32, 196, 1024, strides=[200704, 1024, 1], requires_grad=0, device=cuda:0),
#        %11 : Bool(32, 196, 1024, strides=[200704, 1024, 1], requires_grad=0, device=cuda:0)):
#    %3 : NoneType = prim::Constant()
#    %2 : bool = prim::Constant[value=1]() # <eval_with_key>.10:9:46
#    %1 : int[] = prim::Constant[value=[0, 1]]()
#    %6 : float = prim::Constant[value=3.333333333333333]()
#    %type_as_1.1 : Float(32, 196, 1024, strides=[200704, 1024, 1], requires_grad=0, device=cuda:0) = aten::type_as(%11, %8) # <eval_with_key>.10:5:16
#    %mul.1 : Float(32, 196, 1024, strides=[200704, 1024, 1], requires_grad=0, device=cuda:0) = aten::mul(%8, %type_as_1.1) # <eval_with_key>.10:6:10
#    %mul_1.1 : Float(32, 196, 1024, strides=[200704, 1024, 1], requires_grad=0, device=cuda:0) = aten::mul(%mul.1, %6) # <eval_with_key>.10:8:12
#    %sum_1.1 : Float(1, 1, 1024, strides=[1024, 1024, 1], requires_grad=0, device=cuda:0) = aten::sum(%mul_1.1, %1, %2, %3) # <eval_with_key>.10:9:12
#    return (%sum_1.1, %mul_1.1)
#
#


def dropout_res_bias(input, bias, residual):
    a = torch.add(input, bias)
    b = torch.nn.functional.dropout(a, p=0.7, training=True)
    c = b + residual
    return c


fn = dropout_res_bias

clear_compile_cache()

# Set inputs
device = "cuda"
dtype = torch.float16
batch_size = 32
seq_len = 196
hidden_size = 1024
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
opt_fn = memory_efficient_pointwise_fusion(fn, compiler_name="torchscript_nvfuser")

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
