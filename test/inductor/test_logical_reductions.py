# Owner(s): ["module: inductor"]

import torch
from torch._inductor.test_case import run_tests, TestCase
from torch.testing._internal.common_device_type import (
    dtypes,
    instantiate_device_type_tests,
)
from torch.testing._internal.common_utils import parametrize


class TestLogicalReductions(TestCase):
    @parametrize("op", (torch.any, torch.all))
    @parametrize("dim", ((), []))
    @parametrize("keepdim", (False, True))
    @parametrize("layout", ("contiguous", "transposed", "empty", "scalar"))
    @dtypes(torch.bool, torch.uint8, torch.int64, torch.float32)
    def test_empty_dim(self, device, dtype, op, dim, keepdim, layout):
        x = torch.tensor([[0, 1, 2], [3, 0, 4]], device=device, dtype=dtype)
        x = {
            "contiguous": x,
            "transposed": x.t(),
            "empty": x[:0],
            "scalar": x[0, 1],
        }[layout]

        def fn(x):
            return op(x, dim=dim, keepdim=keepdim)

        expected_dtype = torch.uint8 if dtype == torch.uint8 else torch.bool
        expected = x.ne(0).to(expected_dtype)
        actual = torch.compile(fn, fullgraph=True)(x)
        self.assertEqual(actual, expected)

    @parametrize("op", (torch.any, torch.all))
    @parametrize("dim", ((), []))
    def test_empty_dim_does_not_alias_input(self, device, op, dim):
        def fn(x):
            return op(x, dim=dim)

        x = torch.tensor([[True, False], [False, True]], device=device)
        expected = x.clone()
        actual = torch.compile(fn, fullgraph=True)(x)
        x.logical_not_()
        self.assertEqual(actual, expected)

    @parametrize("op", (torch.any, torch.all))
    @parametrize("dim", (None, 0, -1, (0, 1)))
    @parametrize("keepdim", (False, True))
    def test_reduction_dims(self, device, op, dim, keepdim):
        def fn(x):
            return op(x, dim=dim, keepdim=keepdim)

        x = torch.tensor([[True, False, True], [False, True, False]], device=device)
        self.assertEqual(torch.compile(fn, fullgraph=True)(x), fn(x))


instantiate_device_type_tests(
    TestLogicalReductions, globals(), only_for=("cpu", "cuda", "xpu"), allow_xpu=True
)


if __name__ == "__main__":
    run_tests()
