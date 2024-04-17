# Owner(s): ["oncall: cpu inductor"]
import functools
import unittest
from unittest.mock import patch

import torch
import torch._dynamo.config as dynamo_config
import torch._inductor.config as inductor_config
import torch._inductor.select_algorithm as select_algorithm
from torch._dynamo.utils import counters
from torch._inductor.test_case import run_tests, TestCase

from torch.testing._internal.common_utils import IS_MACOS, TEST_MKL

aten = torch.ops.aten


def patches(fn):
    def skip_cache(self, choices, name, key, generate):
        return generate(choices)

    for patcher in [
        dynamo_config.patch(verbose=True),
        inductor_config.patch(debug=True, max_autotune=True, epilogue_fusion=True),
        patch.object(select_algorithm, "VERIFY", dict(atol=1e-4, rtol=1e-4)),
        patch.object(select_algorithm.AlgorithmSelectorCache, "lookup", skip_cache),
    ]:
        fn = patcher(fn)

    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        counters.clear()
        torch.manual_seed(12345)
        return fn(*args, **kwargs)

    return wrapped


class TestSelectAlgorithm(TestCase):
    @inductor_config.patch({"freezing": True})
    @patches
    @torch.no_grad
    @unittest.skipIf(not TEST_MKL, "Test requires MKL")
    def test_linear_fp32_cpu(self):
        class M(torch.nn.Module):
            def __init__(self, bias):
                super().__init__()
                self.linear = torch.nn.Linear(10, 32, bias)

            @torch.compile
            def forward(self, x):
                return self.linear(x)

        for bias in [True, False]:
            counters.clear()
            mod = M(bias=bias).eval()
            v = torch.randn(2, 10)
            mod(v)
            self.assertEqual(counters["inductor"]["select_algorithm_autotune"], 1)


@dynamo_config.patch({"dynamic_shapes": True, "assume_static_by_default": False})
class TestDynamicSelectAlgorithm(TestCase):
    test_linear_fp32_dynamic_shapes_cpu = TestSelectAlgorithm.test_linear_fp32_cpu


if __name__ == "__main__":
    from torch.testing._internal.inductor_utils import HAS_CPU

    if HAS_CPU and not IS_MACOS:
        run_tests()
