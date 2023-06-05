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
from torch._inductor.autotune_process import BenchmarkRequest

from torch.testing._internal.common_utils import IS_LINUX, TEST_WITH_ROCM
from torch.testing._internal.inductor_utils import HAS_CUDA

aten = torch.ops.aten


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
    def check_counter(self, counter, expected):
        if not inductor_config.cpp_wrapper:
            self.assertEqual(counter, expected)
        elif not dynamo_config.dynamic_shapes:
            # cpp_wrapper for the CUDA backend runs two passes
            self.assertEqual(counter, 2 * expected)

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
        self.check_counter(counters["inductor"]["select_algorithm_autotune"], 1)
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
        self.check_counter(counters["inductor"]["select_algorithm_autotune"], 1)

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
        self.check_counter(counters["inductor"]["select_algorithm_autotune"], 1)

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

    @patches
    def test_convolution1(self):
        @torch.compile
        def foo(x, w, b):
            return aten.convolution(
                x + 1,
                w,
                b,
                stride=(2, 3),
                padding=(4, 5),
                dilation=(1, 1),
                transposed=False,
                output_padding=(0, 0),
                groups=1,
            )

        foo(
            torch.randn(2, 33, 34, 41, device="cuda"),
            torch.randn(34, 33, 3, 3, device="cuda"),
            torch.randn(34, device="cuda"),
        )
        # Autotuning checks correctness of each version
        self.assertEqual(counters["inductor"]["select_algorithm_autotune"], 1)

    @patches
    def test_mm_dropout(self):
        @torch.compile
        def fn(x1, x2, seed):
            mm_4 = torch.ops.aten.mm.default(x2, x1)
            rnd = torch.ops.prims.inductor_random.default(mm_4.shape, seed, "rand")
            return mm_4 * rnd

        # sizes picked so triton autotuning wins
        fn(
            torch.randn(512, 1024, dtype=torch.float16, device="cuda"),
            torch.randn(384, 512, dtype=torch.float16, device="cuda"),
            torch.tensor(12345, device="cuda"),
        )
        self.assertEqual(counters["inductor"]["select_algorithm_autotune"], 1)

    @patches
    @torch._inductor.config.patch(conv_1x1_as_mm=False)
    def test_convolution2(self):
        @torch.compile
        def foo(x, w, b):
            return aten.convolution(
                x,
                w,
                b,
                stride=(1, 1),
                padding=(0, 0),
                dilation=(1, 1),
                transposed=False,
                output_padding=(0, 0),
                groups=1,
            )

        foo(
            torch.randn(1, 33, 16, 16, device="cuda"),
            torch.randn(34, 33, 1, 1, device="cuda"),
            torch.randn(34, device="cuda"),
        )
        # Autotuning checks correctness of each version
        self.assertEqual(counters["inductor"]["select_algorithm_autotune"], 1)

    @patches
    @torch._inductor.config.patch(conv_1x1_as_mm=True)
    def test_convolution_as_mm(self):
        @torch.compile
        def foo(x, w, b):
            return aten.convolution(
                x + 1,
                w,
                b,
                stride=(1, 1),
                padding=(0, 0),
                dilation=(1, 1),
                transposed=False,
                output_padding=(0, 0),
                groups=1,
            )

        foo(
            torch.randn(2, 33, 16, 16, device="cuda"),
            torch.randn(34, 33, 1, 1, device="cuda"),
            torch.randn(34, device="cuda"),
        )
        # Autotuning checks correctness of each version
        self.assertEqual(counters["inductor"]["select_algorithm_autotune"], 1)

    def test_TritonTemplateCaller_str(self):
        """
        Make sure str(TritonTemplateCaller) does not raise exceptions.
        """
        module_path = "abc.py"
        bmreq = BenchmarkRequest(
            module_path=module_path,
            module_cache_key=None,
            kernel_name=None,
            grid=None,
            extra_args=None,
            num_stages=None,
            num_warps=None,
            input_tensors=None,
            output_tensor=None,
        )
        caller = select_algorithm.TritonTemplateCaller(
            None, None, None, None, "extra", bmreq
        )
        caller_str = str(caller)
        self.assertEqual(caller_str, f"TritonTemplateCaller({module_path}, extra)")


if __name__ == "__main__":
    from torch._inductor.utils import is_big_gpu

    if IS_LINUX and HAS_CUDA and is_big_gpu(0) and not TEST_WITH_ROCM:
        run_tests()
