# Owner(s): ["module: dsl-native-ops"]

import unittest
from unittest import mock

import torch
import torch.nn.functional as F
from torch._native.registry import native_decomp_table
from torch._vendor.quack.mx_utils import to_blocked
from torch.nn.functional import ScalingType, SwizzleType
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    skipIfNoCuteDSL,
    TestCase,
)


def _make_mxfp8(rows: int, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    q = torch.randn(rows, k, device="cuda").clamp(-400, 400).to(torch.float8_e4m3fn)
    exponents = torch.randint(-3, 4, (rows, k // 32), device="cuda")
    scales = (exponents + 127).to(torch.uint8).view(torch.float8_e8m0fnu)
    return q, to_blocked(scales)


def _scaled_mm_args(m: int, n: int, k: int):
    q_input, input_scale = _make_mxfp8(m, k)
    weight, weight_scale = _make_mxfp8(n, k)
    args = (
        q_input,
        weight.T,
        input_scale,
        ScalingType.BlockWise1x32,
        weight_scale,
        ScalingType.BlockWise1x32,
    )
    kwargs = {
        "swizzle_a": SwizzleType.SWIZZLE_32_4_4,
        "swizzle_b": SwizzleType.SWIZZLE_32_4_4,
        "output_dtype": torch.bfloat16,
    }
    return args, kwargs


@unittest.skipUnless(TEST_CUDA, "CUDA required")
@skipIfNoCuteDSL
class TestCuTeDSLMxfp8ScaledMM(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        if torch.cuda.get_device_capability() not in {(10, 0), (10, 3)}:
            raise unittest.SkipTest("SM100 or SM103 required")

    @parametrize("m", range(1, 9))
    def test_correctness(self, m: int) -> None:
        torch.manual_seed(m)
        args, kwargs = _scaled_mm_args(m, 128, 1024)
        with torch.backends.python_native.cutedsl.disabled():
            expected = F.scaled_mm(*args, **kwargs)
        actual = F.scaled_mm(*args, **kwargs)
        self.assertEqual(actual, expected, atol=1.0, rtol=0.05)

    def test_multiple_output_tiles(self) -> None:
        torch.manual_seed(123)
        args, kwargs = _scaled_mm_args(3, 512, 1024)
        with torch.backends.python_native.cutedsl.disabled():
            expected = F.scaled_mm(*args, **kwargs)
        actual = F.scaled_mm(*args, **kwargs)
        self.assertEqual(actual, expected, atol=1.0, rtol=0.05)

    def test_dispatches_through_native_kernel(self) -> None:
        from torch._native.ops.scaled_mm import cutedsl_kernel

        args, kwargs = _scaled_mm_args(3, 128, 1024)
        with mock.patch.object(
            cutedsl_kernel,
            "mxfp8_small_m_scaled_mm",
            wraps=cutedsl_kernel.mxfp8_small_m_scaled_mm,
        ) as launch:
            F.scaled_mm(*args, **kwargs)
        launch.assert_called_once()

    def test_dynamic_m_reuses_one_compilation(self) -> None:
        from torch._native.ops.scaled_mm import cutedsl_kernel

        before = cutedsl_kernel._compile_mxfp8_small_m.cache_info()
        for m in (1, 8):
            args, kwargs = _scaled_mm_args(m, 256, 1024)
            F.scaled_mm(*args, **kwargs)
        after = cutedsl_kernel._compile_mxfp8_small_m.cache_info()
        self.assertEqual(after.currsize - before.currsize, 1)
        self.assertEqual(
            (after.hits + after.misses) - (before.hits + before.misses), 2
        )
        self.assertGreaterEqual(after.hits - before.hits, 1)

    def test_cuda_graph_replay_uses_updated_input(self) -> None:
        args, kwargs = _scaled_mm_args(3, 128, 1024)
        graph = torch.cuda.CUDAGraph()
        torch.cuda.synchronize()
        with torch.cuda.graph(graph):
            output = F.scaled_mm(*args, **kwargs)
        args[0].zero_()
        graph.replay()
        torch.cuda.synchronize()
        self.assertEqual(output, torch.zeros_like(output))

    def test_export_routes_to_native_op(self) -> None:
        class ScaledMM(torch.nn.Module):
            def forward(self, q_input, weight_t, input_scale, weight_scale):
                return F.scaled_mm(
                    q_input,
                    weight_t,
                    input_scale,
                    ScalingType.BlockWise1x32,
                    weight_scale,
                    ScalingType.BlockWise1x32,
                    swizzle_a=SwizzleType.SWIZZLE_32_4_4,
                    swizzle_b=SwizzleType.SWIZZLE_32_4_4,
                    output_dtype=torch.bfloat16,
                )

        args, _ = _scaled_mm_args(3, 128, 1024)
        exported = torch.export.export(
            ScaledMM(), (args[0], args[1], args[2], args[4])
        ).run_decompositions(native_decomp_table())
        targets = [
            str(node.target)
            for node in exported.graph_module.graph.nodes
            if node.op == "call_function"
        ]
        self.assertTrue(any("_native._scaled_mm_v2" in target for target in targets))

    def test_unsupported_m_falls_through(self) -> None:
        args, kwargs = _scaled_mm_args(9, 128, 1024)
        with torch.backends.python_native.cutedsl.disabled():
            expected = F.scaled_mm(*args, **kwargs)
        actual = F.scaled_mm(*args, **kwargs)
        self.assertEqual(actual, expected)


instantiate_parametrized_tests(TestCuTeDSLMxfp8ScaledMM)

if __name__ == "__main__":
    run_tests()
