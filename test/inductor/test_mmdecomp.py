# Owner(s): ["module: nn"]

import math
from typing import List, Tuple, Union

import torch
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_nn import NNTestCase
from torch.testing._internal.common_utils import IS_WINDOWS, parametrize, run_tests


default_atol = {
    torch.float16: 1e-3,
    torch.bfloat16: float("infinity"),
    torch.float32: 1e-5,
}
default_rtol = {
    torch.float16: 1e-3,
    torch.bfloat16: float("infinity"),
    torch.float32: 1.3e-6,
}


def rand_math_tensor(
    shape: Tuple[Union[int, List[int]]],
    device: str,
    dtype: torch.dtype,
    requires_grad: bool = False,
    packed: bool = False,
) -> torch.Tensor:
    """Creates rand dense or nested tensor with given shape and type.

    Args:
        shape (Tuple[int]): Shape of Tensor to construct
        device (str): which device to create tensor on
        dtype (torch.dtype): Tensors' dtype
        requires_grad (bool, optional): Tensors grad status. Defaults to False.
        packed (bool, optional): Whether to create a single QKV packed or not. Defaults to False.

    Returns:
        torch.Tensor: A new tensor
    """
    return torch.randn(shape, device=device, dtype=dtype, requires_grad=requires_grad)


def init_tensor(tensor_list, **kwargs) -> torch.Tensor:
    return torch.Tensor(tensor_list).to(**kwargs)


def run_comp_nocomp(function, *inputs, **kwargs):
    c_function = torch.compile(function)

    f_res = function(*inputs)
    cf_res = c_function(*inputs)

    if not (math.isinf(kwargs.get("atol", 0.0)) or math.isinf(kwargs.get("rtol", 0.0))):
        torch.testing.assert_close(f_res, cf_res, **kwargs)


# The test functions are used by several tests
def torch_mm(a, b):
    return torch.mm(a, b)


def torch_addmm(add, b, c):
    return torch.addmm(add, b, c)


def torch_bmm(a, b):
    return torch.bmm(a, b)


def torch_baddbmm(add, b, c, alpha, beta):
    return torch.baddbmm(add, b, c, alpha=alpha, beta=beta)


# The shapes we test on
ts_list = [
    (1, 32, 32, 1),
    (1, 10, 10, 1),
    (1, 3, 3, 1),
    (32, 1, 1, 32),
    (3, 1, 1, 3),
    (4, 1, 1, 9),
    (9, 1, 1, 4),
]


class TestDecomp(NNTestCase):
    _do_cuda_memory_leak_check = True
    _do_cuda_non_default_stream = True

    @parametrize("dtype", [torch.float, torch.bfloat16])
    def test_simple_mm(self, device, dtype):
        fudge = 10
        rtol = default_rtol[dtype] * fudge
        atol = default_atol[dtype] * fudge

        for t_size in ts_list:
            ((a1_0, a1_1, a2_0, a2_1)) = t_size

            t1 = rand_math_tensor((a1_0, a1_1), dtype=dtype, device=device)
            t2 = rand_math_tensor((a2_0, a2_1), dtype=dtype, device=device)
            tadd = rand_math_tensor((a1_0, a2_1), dtype=dtype, device=device)

            run_comp_nocomp(torch_mm, t1, t2, rtol=rtol, atol=atol)
            run_comp_nocomp(torch_addmm, tadd, t1, t2, rtol=rtol, atol=atol)

    @parametrize("dtype", [torch.float, torch.bfloat16])
    @parametrize("bs", [1, 2, 4, 10])
    def test_batched_mm(self, device, dtype, bs):
        fudge = 3
        rtol = default_rtol[dtype] * fudge
        atol = default_atol[dtype] * fudge

        for t_size in ts_list:
            ((a1_0, a1_1, a2_0, a2_1)) = t_size

            t1 = rand_math_tensor((bs, a1_0, a1_1), dtype=dtype, device=device)
            t2 = rand_math_tensor((bs, a2_0, a2_1), dtype=dtype, device=device)
            tadd = rand_math_tensor((bs, a1_0, a2_1), dtype=dtype, device=device)

            run_comp_nocomp(torch_bmm, t1, t2, rtol=rtol, atol=atol)

            for alpha in (0, 1, -1, 0.5, -0.5):
                for beta in (0, 1, -1, 0.5, -0.5):
                    run_comp_nocomp(
                        torch_baddbmm, tadd, t1, t2, alpha, beta, rtol=rtol, atol=atol
                    )

    @parametrize("dtype", [torch.float, torch.bfloat16, torch.int])
    def test_some(self, device, dtype):
        run_comp_nocomp(
            torch_mm,
            init_tensor([[1], [2], [3], [4]], dtype=dtype, device=device),
            init_tensor([[1, 2, 3, 4]], dtype=dtype, device=device),
        )
        run_comp_nocomp(
            torch_mm,
            init_tensor([[1, 2, 3, 4]], dtype=dtype, device=device),
            init_tensor([[1], [2], [3], [4]], dtype=dtype, device=device),
        )

    @parametrize("dtype", [torch.float, torch.bfloat16, torch.int])
    @parametrize("bs", [1, 2, 4, 10])
    def test_some_batched(self, device, dtype, bs):
        run_comp_nocomp(
            torch_bmm,
            init_tensor([[[1], [2], [3], [4]]] * bs, dtype=dtype, device=device),
            init_tensor([[[1, 2, 3, 4]]] * bs, dtype=dtype, device=device),
        )
        run_comp_nocomp(
            torch_bmm,
            init_tensor([[[1, 2, 3, 4]]] * bs, dtype=dtype, device=device),
            init_tensor([[[1], [2], [3], [4]]] * bs, dtype=dtype, device=device),
        )


device_types = ("cpu",)
instantiate_device_type_tests(TestDecomp, globals(), only_for=device_types)

if __name__ == "__main__":
    # We don't support torch.compile() on Windows presently
    if not IS_WINDOWS:
        run_tests()
