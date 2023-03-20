# Owner(s): ["module: inductor"]
import functools
import itertools
import unittest

import torch
from torch._dynamo.test_case import run_tests, TestCase
from torch._dynamo.utils import counters
from torch.nn import functional as F
from torch.testing._internal.common_utils import IS_LINUX, TEST_WITH_SLOW
from torch.testing._internal.inductor_utils import HAS_CPU, HAS_CUDA

slow = functools.partial(unittest.skipIf, not TEST_WITH_SLOW, "too slow")


unary_list = {
    torch.nn.ReLU(): 2,
    torch.nn.Sigmoid(): 2,
    torch.nn.Tanh(): 2,
    torch.nn.Hardswish(): 6,
    torch.nn.LeakyReLU(0.1, inplace=False): 4,
    torch.nn.Hardtanh(min_val=-0.5, max_val=4, inplace=False): 3,
    torch.nn.GELU(approximate="none"): 6,
    torch.nn.GELU(approximate="tanh"): 10,
    torch.nn.ReLU6(): 3,
    torch.nn.SiLU(): 3,
    torch.nn.Hardsigmoid(): 5,
    lambda x: F.relu(x): 2,
    lambda x: F.sigmoid(x): 2,
    lambda x: F.tanh(x): 2,
    lambda x: F.hardswish(x): 6,
    lambda x: F.leaky_relu(x, 0.1): 4,
    lambda x: F.hardtanh(x, min_val=-0.5, max_val=4): 3,
    lambda x: F.gelu(x, approximate="none"): 6,
    lambda x: F.gelu(x, approximate="tanh"): 10,
    lambda x: F.relu6(x): 3,
    lambda x: F.silu(x): 3,
    lambda x: F.hardsigmoid(x): 5,
    lambda x: torch.relu(x): 2,
    lambda x: torch.sigmoid(x): 2,
    lambda x: torch.tanh(x): 2,
    lambda x: x.relu(): 2,
    lambda x: x.sigmoid(): 2,
    lambda x: x.tanh(): 2,
}


class TestPaternMatcher(TestCase):
    if HAS_CUDA:

        def test_mm_plus_mm(self):
            def fn(a, b, c, d):
                return torch.add(torch.mm(a, b), torch.mm(c, d))

            args = [
                torch.randn(16, 16, device="cuda"),
                torch.randn(16, 16, device="cuda"),
                torch.randn(16, 16, device="cuda"),
                torch.randn(16, 16, device="cuda"),
            ]
            expected = fn(*args)
            actual = torch.compile(fn)(*args)
            torch.testing.assert_close(actual, expected)
            self.assertEqual(counters["inductor"]["pattern_matcher_count"], 1)
            self.assertEqual(counters["inductor"]["pattern_matcher_nodes"], 3)

        def test_addmm(self):
            def fn(a, b, c):
                return torch.add(a, torch.mm(b, c)), torch.mm(b, c) + a

            args_list = [
                (
                    torch.randn(16, 16, device="cuda"),
                    torch.randn(16, 16, device="cuda"),
                    torch.randn(16, 16, device="cuda"),
                ),
                (
                    torch.randn(16, 16, device="cuda"),
                    torch.randn(1, 16, device="cuda"),
                    torch.randn(16, 16, device="cuda"),
                ),
                (
                    torch.randn(1, 16, 16, device="cuda"),
                    torch.randn(16, 16, device="cuda"),
                    torch.randn(16, 16, device="cuda"),
                ),
                (
                    4,
                    torch.randn(16, 16, device="cuda"),
                    torch.randn(16, 16, device="cuda"),
                ),
            ]
            for args in args_list:
                counters.clear()
                e1, e2 = fn(*args)
                a1, a2 = torch.compile(fn)(*args)
                torch.testing.assert_close(a1, e1)
                torch.testing.assert_close(a2, e2)
                self.assertEqual(counters["inductor"]["pattern_matcher_count"], 2)
                self.assertEqual(counters["inductor"]["pattern_matcher_nodes"], 4)

        def test_cat_mm(self):
            def fn(a, b, c):
                return torch.cat(
                    [
                        torch.mm(a, b),
                        torch.mm(b, c),
                        torch.mm(a, c),
                    ],
                    1,
                )

            args = [
                torch.randn(16, 16, device="cuda"),
                torch.randn(16, 16, device="cuda"),
                torch.randn(16, 16, device="cuda"),
            ]
            expected = fn(*args)
            actual = torch.compile(fn)(*args)
            torch.testing.assert_close(actual, expected)
            self.assertEqual(counters["inductor"]["pattern_matcher_count"], 1)
            self.assertEqual(counters["inductor"]["pattern_matcher_nodes"], 4)

        def test_cat_addmm(self):
            def fn(a, b, c):
                return torch.cat(
                    [
                        torch.addmm(a, b, c),
                        torch.addmm(b, c, a),
                        torch.addmm(c, a, b),
                    ],
                    1,
                )

            args = [
                torch.randn(16, 16, device="cuda"),
                torch.randn(16, 16, device="cuda"),
                torch.randn(16, 16, device="cuda"),
            ]
            expected = fn(*args)
            actual = torch.compile(fn)(*args)
            torch.testing.assert_close(actual, expected)
            self.assertEqual(counters["inductor"]["pattern_matcher_count"], 1)
            self.assertEqual(counters["inductor"]["pattern_matcher_nodes"], 4)

        def test_cat_slice_cat(self):
            def fn(a, b):
                cat_1 = torch.ops.aten.cat.default([a, b], 1)
                slice_1 = torch.ops.aten.slice.Tensor(cat_1, 0, 0, 9223372036854775807)
                slice_2 = torch.ops.aten.slice.Tensor(slice_1, 1, 0, 19)
                return torch.ops.aten.cat.default([cat_1, slice_2], 1)

            args = [
                torch.randn(2, 32, device="cuda"),
                torch.randn(2, 16, device="cuda"),
            ]
            expected = fn(*args)
            actual = torch.compile(fn)(*args)
            torch.testing.assert_close(actual, expected)
            self.assertEqual(counters["inductor"]["pattern_matcher_count"], 1)
            self.assertEqual(counters["inductor"]["pattern_matcher_nodes"], 4)

            counters.clear()
            args = [
                torch.randn(2, 8, device="cuda"),
                torch.randn(2, 16, device="cuda"),
            ]
            expected = fn(*args)
            actual = torch.compile(fn)(*args)
            torch.testing.assert_close(actual, expected)
            self.assertEqual(counters["inductor"]["pattern_matcher_count"], 1)
            self.assertEqual(counters["inductor"]["pattern_matcher_nodes"], 4)

    if HAS_CPU:

        @slow()
        def test_conv2d_unary(self):
            class M(torch.nn.Module):
                def __init__(
                    self,
                    unary_fn,
                    **kwargs,
                ):
                    super().__init__()
                    self.conv = torch.nn.Conv2d(3, 16, kernel_size=3, stride=1)
                    self.unary_fn = unary_fn

                def forward(self, x):
                    x = self.conv(x)
                    return self.unary_fn(x)

            test_memory_format = [torch.contiguous_format, torch.channels_last]
            options = itertools.product(
                unary_list.keys(),
                test_memory_format,
            )

            for (
                unary_fn,
                memory_format,
            ) in options:
                x_shape = (1, 3, 56, 56)
                mod = M(
                    unary_fn,
                ).eval()

                # TODO: add bf16 test for cpu path?
                # TODO: this test fails when requires_grad=False
                v = (
                    torch.randn(x_shape, dtype=torch.float32, requires_grad=True)
                    .add(1)
                    .to(memory_format=memory_format)
                )
                with torch.no_grad():
                    expected = mod(v)
                    actual = torch.compile(mod)(v)
                    torch.testing.assert_close(actual, expected)
                    self.assertEqual(counters["inductor"]["pattern_matcher_count"], 1)
                    self.assertEqual(
                        counters["inductor"]["pattern_matcher_nodes"],
                        unary_list[unary_fn],
                    )
                    counters.clear()


if __name__ == "__main__":
    if IS_LINUX and (HAS_CUDA or HAS_CPU):
        run_tests()
