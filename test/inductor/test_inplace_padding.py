# Owner(s): ["module: inductor"]
import os
import sys
import unittest

import torch
from torch import nn
from torch._dynamo.utils import same
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import run_and_get_code
from torch.testing import FileCheck
from torch.testing._internal.common_utils import serialTest
from torch.testing._internal.inductor_utils import (
    GPU_TYPE,
    HAS_GPU,
    requires_cuda_with_enough_memory,
)


# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)

# TODO move check_model to a common module since it's quite often to
# be used by new test cases.
from inductor.test_torchinductor import check_model
from torch._dynamo.testing import rand_strided
from torch._inductor import config as inductor_config


aten = torch.ops.aten


def num_inplace_padding():
    from torch._dynamo.utils import counters

    return counters["inductor"]["inplace_padding"]


enable_inplace_padding = True
if os.environ.get("TORCHINDUCTOR_INPLACE_PADDING") is not None:
    enable_inplace_padding = os.environ.get("TORCHINDUCTOR_INPLACE_PADDING") == "1"

DO_PERF_TEST = os.environ.get("DO_PERF_TEST") == "1"


@inductor_config.patch(inplace_padding=enable_inplace_padding)
class InplacePaddingTest(TestCase):
    def test_skip_pad_due_to_fusion(self):
        """
        If the padding can be fused with downstream op, there would
        be little benefit to do inplace padding.
        """

        def f(x):
            x = aten.constant_pad_nd(x, (0, 8, 0, 0), 12345.0)
            return x.sum(dim=-1)

        M, N = 2048, 2048
        x = rand_strided((M, N), (N + 10, 1), device=GPU_TYPE)
        check_model(self, f, (x,), atol=1e-3, rtol=1e-3)

        self.assertEqual(num_inplace_padding(), 0)

    def test_skip_pad_input(self):
        """
        Don't apply the padding to graph input since Inductor does not
        allocatae the input and can not guarantee enough trailing space
        for padding.
        """

        def f(x, y):
            x = aten.constant_pad_nd(x, (0, 8, 0, 0), 12345.0)
            return x @ y

        M, N = 2048, 2048
        x = rand_strided((M, N), (N + 10, 1), device=GPU_TYPE)
        y = torch.randn(N + 8, M, device=GPU_TYPE)
        check_model(self, f, (x, y), atol=1e-2, rtol=1e-2)

        self.assertEqual(num_inplace_padding(), 0)

    def test_pad_non_zero(self):
        def f(x):
            x = x + 1
            x = aten.constant_pad_nd(x, (0, 1, 0, 0), 12345.0)

            return x @ x

        # 'odd' shape on purpose to pad intermediate buffer's strides
        x = torch.randn(2048, 2047, device=GPU_TYPE)

        ref = f(x)
        act, (code,) = run_and_get_code(torch.compile(f), x)

        # When we allocate the 2048x2047 tensor for the output of 'x + 1'
        # Instead of doing
        #   empty_strided_cuda((2048, 2047), (2048, 1), torch.float32)
        # (note the stride is already padded)
        # We do
        #   empty_strided_cuda((2048, 2048), (2048, 1), torch.float32).
        #     as_strided((2048, 2047), (2048, 1))
        # . This will allocate an extra item for the last row so that
        # inplace padding would be safe without accessing out of bound
        # memory.
        FileCheck().check_regex(
            r"empty_strided.*\(\(2048, 2048\), \(2048, 1\), torch.float32\)."
            r"as_strided\(\(2048, 2047\), \(2048, 1\)\)"
        ).run(code)

        self.assertTrue(torch.allclose(ref, act, atol=1e-2, rtol=1e-2))
        self.assertEqual(num_inplace_padding(), 1)

    @inductor_config.patch(cpp_wrapper=True)
    def test_pad_non_zero_cpp_wrapper(self):
        def f(x):
            x = x + 1
            x = aten.constant_pad_nd(x, (0, 1, 0, 0), 12345.0)

            return x @ x

        # 'odd' shape on purpose to pad intermediate buffer's strides
        x = torch.randn(2048, 2047, device=GPU_TYPE)

        ref = f(x)
        from torch._inductor.codegen.cpp_wrapper_gpu import CppWrapperGpu

        orig_generate_and_run_autotune_block = (
            CppWrapperGpu.generate_and_run_autotune_block
        )
        compile_time_autotune_called = False

        def mock_generate_and_run_autotune_block(wrapper):
            nonlocal compile_time_autotune_called
            compile_time_autotune_called = True
            out = orig_generate_and_run_autotune_block(wrapper)
            call_code = wrapper.kernel_autotune_calls.getvalue()
            FileCheck().check(
                f"buf0 = generate_example_value((2048, 2047), (2048, 1), '{GPU_TYPE}:0', torch.float32, 0, (2048, 2048))"
            ).run(call_code)
            return out

        with unittest.mock.patch.object(
            CppWrapperGpu,
            "generate_and_run_autotune_block",
            mock_generate_and_run_autotune_block,
        ):
            act, (code,) = run_and_get_code(torch.compile(f), x)

        # Buf0 should be over-allocated and then strided.
        FileCheck().check_regex(
            r"aoti_torch_as_strided\(buf0_handle, .*, &buf0_handle_restrided\)"
        ).run(code)

        self.assertTrue(torch.allclose(ref, act, atol=1e-2, rtol=1e-2))

        self.assertEqual(num_inplace_padding(), 1)
        self.assertTrue(compile_time_autotune_called)

    def test_pad_too_large(self):
        def f(x, y):
            x = aten.constant_pad_nd(x, (0, 8, 0, 0), 12345.0)
            return x @ y

        M, N = 2048, 2048
        x = rand_strided((M, N), (N + 5, 1), device=GPU_TYPE)
        y = torch.randn(N + 8, M, device=GPU_TYPE)
        check_model(self, f, (x, y), atol=1e-2, rtol=1e-2)

        self.assertEqual(num_inplace_padding(), 0)

    @inductor_config.patch(can_inplace_pad_graph_input=True)
    def test_mutating_padding_input(self):
        """
        Even if `aten.constant_pad_nd` input get inplace updated,
        doing inplace-padding still generates the correct result.
        """

        def f(x, y):
            x2 = aten.constant_pad_nd(x, (0, 8, 0, 0), 12345.0)
            x.add_(5)
            return x2 @ y

        M, N = 2048, 2048
        x = rand_strided((M, N + 10), (N + 10, 1), device=GPU_TYPE).as_strided(
            (M, N), (N + 10, 1)
        )
        y = torch.randn(N + 8, M, device=GPU_TYPE)
        check_model(self, f, (x, y), atol=1e-2, rtol=1e-2)

        self.assertEqual(num_inplace_padding(), 1)

    def test_mutating_padding_output(self):
        """
        Inplace padding does not take effect since the `aten.add_` op
        cause the user of the padding output to be not matmul. We skip
        inplace-padding in this case.
        """

        def f(x, y):
            x = aten.constant_pad_nd(x, (0, 8, 0, 0), 12345.0)
            x.add_(1)
            return x @ y

        M, N = 2048, 2048
        x = rand_strided((M, N), (N + 10, 1), device=GPU_TYPE)
        y = torch.randn(N + 8, M, device=GPU_TYPE)
        # 1e-3 tolerance may fail on CI A10G GPU.
        check_model(self, f, (x, y), atol=1e-2, rtol=1e-2)

        self.assertEqual(num_inplace_padding(), 0)

    @requires_cuda_with_enough_memory(2e10)
    @inductor_config.patch(force_shape_pad=True)
    @serialTest()
    def test_linear_and_cel(self):
        # Use nan for torch.empty
        torch.use_deterministic_algorithms(True)
        torch.utils.deterministic.fill_uninitialized_empty = True
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

        B, T, C, V = 32, 1024, 768, 50257

        linear = nn.Linear(C, V).bfloat16().to(device=GPU_TYPE)
        ce = torch.nn.CrossEntropyLoss()

        def f(x, y):
            x.grad = None
            linear.weight.grad = None
            linear.bias.grad = None

            loss = ce(linear(x), y)
            loss.backward()
            return loss

        x = torch.randn(B * T, C, requires_grad=True).cuda().bfloat16()
        x.retain_grad()
        y = torch.randint(0, V, (B * T,)).cuda()

        opt_f = torch.compile(f)

        expect = (f(x, y), x.grad, linear.weight.grad, linear.bias.grad)
        actual = (opt_f(x, y), x.grad, linear.weight.grad, linear.bias.grad)
        assert same(expect, actual, tol=1e-2), f"ref:\n{expect}\nact:\n{actual}"

        # We may disable inplace_padding via env-var to test perf.
        self.assertEqual(num_inplace_padding(), int(inductor_config.inplace_padding))

        if DO_PERF_TEST:
            from triton.testing import do_bench

            ms = do_bench(lambda: opt_f(x, y))
            print(f"{inductor_config.inplace_padding=} {ms=:.3f}")

    # Enable Max-Autotune to repro this test failure:
    #   https://github.com/pytorch/pytorch/pull/140249#issuecomment-2556079406
    @inductor_config.patch(max_autotune=True)
    def test_linear_and_cel_max_autotune(self):
        self.test_linear_and_cel()


if __name__ == "__main__":
    if HAS_GPU:
        run_tests()
