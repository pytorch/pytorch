import torch
from torch.utils.flop_counter import (
    _FlopCounterMode,
    FlopCounterMode,
    register_flop_formula_for_triton_kernel,
)


D = 1024


@torch._library.triton.triton_op("mylib::kernel1", mutates_args=())
def kernel1() -> None:
    print("kernel1 IN HERE")


@torch._library.triton.triton_op("mylib::kernel2", mutates_args=())
def kernel2() -> None:
    print("kernel2 IN HERE")


@torch._library.triton.triton_op("mylib::op2", mutates_args=())
def op2() -> None:
    print("op2 IN HERE")
    # Assume 1:1 mapping between op and kernel
    torch.ops.mylib.kernel1()


@torch._library.triton.triton_op("mylib::op", mutates_args=())
def op() -> None:
    print("op IN HERE")
    torch.ops.mylib.kernel1()
    torch.ops.mylib.kernel2()


def op_decompose(mode, *args, **kwargs):
    with mode:
        print("op_decompose IN HERE")
        # Call the decomposed operations
        torch.ops.mylib.kernel1()
        torch.ops.mylib.kernel2()
        # Return None since the original op returns None


torch.library.register_torch_dispatch("mylib::op", _FlopCounterMode, op_decompose)


@register_flop_formula_for_triton_kernel(torch.ops.mylib.kernel1)
def compute_flops_kernel1(*args, **kwargs) -> int:
    return 1


@register_flop_formula_for_triton_kernel(torch.ops.mylib.kernel2)
def compute_flops_kernel2(*args, **kwargs) -> int:
    return 2


with FlopCounterMode() as m:
    torch.ops.mylib.op()
    torch.ops.mylib.kernel1()
    torch.ops.mylib.kernel2()
    # torch.ops.mylib.op2()
print(m)
