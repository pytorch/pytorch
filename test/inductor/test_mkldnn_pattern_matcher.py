# Owner(s): ["module: inductor"]
import functools
import itertools

import torch
from torch._dynamo.test_case import run_tests, TestCase
from torch._dynamo.utils import counters
from torch.nn import functional as F
from torch.testing._internal.common_utils import IS_LINUX
from torch.testing._internal.inductor_utils import HAS_CPU


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

unary_list_bf16 = {
    torch.nn.ReLU(): 2,
    torch.nn.Sigmoid(): 2,
    torch.nn.Tanh(): 2,
    lambda x: F.relu(x): 2,
    lambda x: F.sigmoid(x): 2,
    lambda x: F.tanh(x): 2,
    lambda x: torch.relu(x): 2,
    lambda x: torch.sigmoid(x): 2,
    lambda x: torch.tanh(x): 2,
    lambda x: x.relu(): 2,
    lambda x: x.sigmoid(): 2,
    lambda x: x.tanh(): 2,
}


# For OneDNN bf16 path, OneDNN requires the cpu has intel avx512 with avx512bw,
# avx512vl, and avx512dq at least. So we will skip the test case if one processor
# is not meet the requirement.
@functools.lru_cache(maxsize=None)
def has_bf16_support():
    import sys

    if sys.platform != "linux":
        return False
    with open("/proc/cpuinfo", encoding="ascii") as f:
        lines = f.read()
    return all(word in lines for word in ["avx512bw", "avx512vl", "avx512dq"])


class TestPaternMatcher(TestCase):
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

    def test_linear_unary(self):
        class M(torch.nn.Module):
            def __init__(
                self,
                unary_fn,
                in_features,
                out_features,
                bias,
                **kwargs,
            ):
                super().__init__()
                self.linear = torch.nn.Linear(
                    in_features,
                    out_features,
                    bias,
                    **kwargs,
                )
                self.unary_fn = unary_fn

            def forward(self, x):
                x = self.linear(x)
                return self.unary_fn(x)

        options = itertools.product(unary_list_bf16, [True, False])
        dtype = torch.bfloat16
        if has_bf16_support():
            for unary_fn, bias in options:
                mod = M(unary_fn, 10, 30, bias=bias).eval()
                # only fuse for linear when the dtype is bf16
                mod = mod.to(dtype)
                v = torch.randn(2, 10).to(dtype)
                with torch.no_grad():
                    expected = mod(v)
                    actual = torch.compile(mod)(v)
                    torch.testing.assert_close(actual, expected)
                    self.assertEqual(counters["inductor"]["pattern_matcher_count"], 1)
                    self.assertEqual(
                        counters["inductor"]["pattern_matcher_nodes"],
                        unary_list_bf16[unary_fn],
                    )
                    counters.clear()


if __name__ == "__main__":
    if IS_LINUX and HAS_CPU and torch._C.has_mkldnn:
        run_tests()
