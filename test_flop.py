import triton
import triton.language as tl

import torch
from torch.utils.flop_counter import (
    _FlopCounterMode,
    FlopCounterMode,
    register_flop_formula_for_triton_kernel,
)


# D = 1024


# @torch._library.triton.triton_op("mylib::kernel1", mutates_args=())
# def kernel1() -> None:
#     print("kernel1 IN HERE")


# @torch._library.triton.triton_op("mylib::kernel2", mutates_args=())
# def kernel2() -> None:
#     print("kernel2 IN HERE")


# @torch._library.triton.triton_op("mylib::op2", mutates_args=())
# def op2() -> None:
#     print("op2 IN HERE")
#     # Assume 1:1 mapping between op and kernel
#     torch.ops.mylib.kernel1()


# @torch._library.triton.triton_op("mylib::op", mutates_args=())
# def op() -> None:
#     print("op IN HERE")
#     torch.ops.mylib.kernel1()
#     torch.ops.mylib.kernel2()


# def op_decompose(mode, *args, **kwargs):
#     with mode:
#         print("op_decompose IN HERE")
#         # Call the decomposed operations
#         torch.ops.mylib.kernel1()
#         torch.ops.mylib.kernel2()
#         # Return None since the original op returns None


# torch.library.register_torch_dispatch("mylib::op", _FlopCounterMode, op_decompose)


# @register_flop_formula_for_triton_kernel(torch.ops.mylib.kernel1)
# def compute_flops_kernel1(*args, **kwargs) -> int:
#     return 1


# @register_flop_formula_for_triton_kernel(torch.ops.mylib.kernel2)
# def compute_flops_kernel2(*args, **kwargs) -> int:
#     return 2


# with FlopCounterMode() as m:
#     torch.ops.mylib.op()
#     torch.ops.mylib.kernel1()
#     torch.ops.mylib.kernel2()
#     # torch.ops.mylib.op2()
# print(m)


@triton.jit
def sin_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    out = tl.sin(x)
    tl.store(out_ptr + offsets, out, mask=mask)


x = torch.randn(3, device="cuda")
out = torch.empty(3, device="cuda")


@register_flop_formula_for_triton_kernel(sin_kernel)
def compute_sin_kerenel_flops(*args, **kwargs) -> int:
    # dummy implementation
    return 2


def sin_grid(meta):
    return (triton.cdiv(3, meta["BLOCK_SIZE"]),)


with FlopCounterMode() as m:
    n_elements = 3
    torch.library.wrap_triton(sin_kernel)[sin_grid](x, out, 3, 256)


# Now, wrap in a triton op and do the decomp
@torch._library.triton.triton_op("mylib::sin_op", mutates_args=())
def op() -> None:
    n_elements = 3
    torch.library.wrap_triton(sin_kernel)[sin_grid](x, out, 3, 256)


def op_decompose(mode, *args, **kwargs):
    with mode:
        n_elements = 3
        torch.library.wrap_triton(sin_kernel)[sin_grid](x, out, 3, 256)

torch.library.register_torch_dispatch("mylib::sin_op", _FlopCounterMode, op_decompose)

# Should now output 2 flops; previously would be 0
with FlopCounterMode() as m:
    torch.ops.mylib.sin_op()