# Owner(s): ["module: inductor"]
import functools
from unittest.mock import patch

import torch
import torch._dynamo.config as dynamo_config
import torch._inductor.config as inductor_config
import torch._inductor.select_algorithm as select_algorithm
import torch.nn.functional as F
from torch._dynamo.test_case import run_tests, TestCase
from torch._dynamo.utils import counters
from torch.testing._internal.common_utils import IS_LINUX
from torch.testing._internal.inductor_utils import HAS_CUDA

torch.backends.cuda.matmul.allow_tf32 = False


def patches(fn):
    def skip_cache(self, choices, name, key, generate):
        return {choice: generate(choice) for choice in choices}

    for patcher in [
        dynamo_config.patch(verbose=True),
        inductor_config.patch(debug=True, max_autotune=True, epilogue_fusion=True),
        patch.object(select_algorithm, "VERIFY", dict(atol=1e-4, rtol=1e-4)),
        patch.object(select_algorithm.AlgorithmSelectorCache, "lookup", skip_cache),
        torch.backends.cudnn.flags(allow_tf32=False),
    ]:
        fn = patcher(fn)

    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        counters.clear()
        torch.manual_seed(12345)
        assert (
            not torch.backends.cuda.matmul.allow_tf32
        ), "correctness testing is allergic to tf32"
        return fn(*args, **kwargs)

    return wrapped


class TestSelectAlgorithm(TestCase):
    @patches
    def test_linear_relu(self):
        @torch.compile
        def foo(input, weight, bias):
            return F.relu(F.linear(input, weight, bias))

        foo(
            torch.randn(64, 32, device="cuda"),
            torch.randn(16, 32, device="cuda"),
            torch.randn(16, device="cuda"),
        )
        # Autotuning checks correctness of each version
        self.assertEqual(counters["inductor"]["select_algorithm_autotune"], 1)
        # It would be nice to assert this got fused into a single kernel, but that
        # only happens if we select a triton template (and not aten).

    @patches
    def test_addmm(self):
        @torch.compile
        def foo(input, weight, bias):
            return torch.addmm(bias, input, weight)

        inps = (
            torch.randn(20, 33, device="cuda"),
            torch.randn(33, 16, device="cuda"),
            torch.randn(20, 16, device="cuda"),
        )

        foo(*inps)
        # Autotuning checks correctness of each version
        self.assertEqual(counters["inductor"]["select_algorithm_autotune"], 1)

    @patch.object(select_algorithm, "VERIFY", dict(atol=5e-2, rtol=5e-2))
    @patches
    def test_addmm_fp16(self):
        @torch.compile
        def foo(input, weight, bias):
            return torch.addmm(bias, input, weight)

        inps = (
            torch.randn(2, 320, device="cuda", dtype=torch.half),
            torch.randn(320, 320, device="cuda", dtype=torch.half).t(),
            torch.empty(320, device="cuda", dtype=torch.half),
        )

        foo(*inps)
        # Autotuning checks correctness of each version
        self.assertEqual(counters["inductor"]["select_algorithm_autotune"], 1)

    @patches
    def test_mm(self):
        @torch.compile
        def foo(a, b):
            return torch.mm(a, b)

        foo(
            torch.randn(8, 32, device="cuda"),
            torch.randn(32, 8, device="cuda"),
        )
        self.assertEqual(counters["inductor"]["select_algorithm_autotune"], 1)

    @patches
    def test__int_mm(self):
        @torch.compile
        def foo(a, b):
            return torch._int_mm(a, b)

        foo(
            torch.randint(-10, 10, (64, 32), device="cuda", dtype=torch.int8),
            torch.randint(-10, 10, (32, 64), device="cuda", dtype=torch.int8),
        )
        self.assertEqual(counters["inductor"]["select_algorithm_autotune"], 1)

    @patches
    def test_mm_skip(self):
        @torch.compile
        def foo(a, b):
            return torch.mm(a, b)

        foo(
            torch.randn(8, 32, device="cuda", dtype=torch.float64),
            torch.randn(32, 8, device="cuda", dtype=torch.float64),
        )
        # float64 not supported by tl.dot()
        self.assertEqual(counters["inductor"]["select_algorithm_autotune"], 0)

    @patches
    def test_bmm(self):
        @torch.compile
        def foo(a, b):
            return torch.bmm(a, b)

        foo(
            torch.randn(2, 8, 32, device="cuda"),
            torch.randn(2, 32, 8, device="cuda"),
        )
        # Autotuning checks correctness of each version
        self.assertEqual(counters["inductor"]["select_algorithm_autotune"], 1)

    @patches
    def test_mm_not_even_k(self):
        @torch.compile
        def foo(a, b):
            return torch.mm(a, b)

        foo(
            torch.randn(11, 22, device="cuda"),
            torch.randn(22, 33, device="cuda"),
        )
        self.assertEqual(counters["inductor"]["select_algorithm_autotune"], 1)

    @patches
    def test_baddbmm(self):
        @torch.compile
        def foo(a, b, c):
            return torch.baddbmm(c, a, b)

        foo(
            torch.randn(2, 8, 32, device="cuda"),
            torch.randn(2, 32, 8, device="cuda"),
            torch.randn(2, 1, 8, device="cuda"),
        )
        # Autotuning checks correctness of each version
        self.assertEqual(counters["inductor"]["select_algorithm_autotune"], 1)

    @patches
    def test_mm_plus_mm(self):
        @torch.compile
        def foo(a, b, c, d):
            return (a @ b) + (c @ d)

        foo(
            torch.randn(32, 32, device="cuda"),
            torch.randn(32, 32, device="cuda"),
            torch.randn(32, 32, device="cuda"),
            torch.randn(32, 32, device="cuda"),
        )
        # Autotuning checks correctness of each version
        self.assertEqual(counters["inductor"]["select_algorithm_autotune"], 1)


if __name__ == "__main__":
    from torch._inductor.utils import is_big_gpu

    if IS_LINUX and HAS_CUDA and is_big_gpu(0):
        run_tests()
