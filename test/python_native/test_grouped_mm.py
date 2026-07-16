# Owner(s): ["module: dsl-native-ops"]

import unittest

import torch
from torch._native.common_utils import check_native_jit_disabled
from torch.testing._internal.common_cuda import SM120OrLater, TEST_CUDA
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    skipIfNoCuteDSL,
    TEST_WITH_ROCM,
    TestCase,
)


def _reference_grouped_mm(
    mat_a: torch.Tensor,
    mat_b: torch.Tensor,
    offs: torch.Tensor,
) -> torch.Tensor:
    if mat_a.dim() == 2 and mat_b.dim() == 2:
        outs = []
        start = 0
        for end in offs.cpu().tolist():
            outs.append(mat_a[:, start:end] @ mat_b[start:end])
            start = end
        return torch.stack(outs)

    outs = []
    start = 0
    for group, end in enumerate(offs.cpu().tolist()):
        outs.append(mat_a[start:end] @ mat_b[group])
        start = end
    return torch.cat(outs, dim=0)


def _make_grouped_mm_inputs(dtype: torch.dtype, varlen_k: bool):
    torch.manual_seed(0)
    sizes = [64, 96, 128, 32]
    if varlen_k:
        m, n, k = 128, 128, sum(sizes)
        mat_a = torch.randn(k, m, device="cuda", dtype=dtype).t()
        mat_b = torch.randn(k, n, device="cuda", dtype=dtype)
    else:
        g, m, n, k = len(sizes), sum(sizes), 128, 128
        mat_a = torch.randn(m, k, device="cuda", dtype=dtype)
        mat_b = torch.randn(g, k, n, device="cuda", dtype=dtype)
    offs = (
        torch.tensor(sizes, device="cuda", dtype=torch.int32).cumsum(0).to(torch.int32)
    )
    return mat_a, mat_b, offs


def _assert_quack_path_supported(
    test: TestCase,
    mat_a: torch.Tensor,
    mat_b: torch.Tensor,
    offs: torch.Tensor,
) -> None:
    from torch._native.ops.grouped_mm.impl import _grouped_mm_sm12x_cond

    test.assertTrue(_grouped_mm_sm12x_cond(mat_a, mat_b, offs=offs))


@unittest.skipIf(TEST_WITH_ROCM, "QuACK grouped_mm is CUDA-only")
@unittest.skipUnless(TEST_CUDA and SM120OrLater, "SM120 CUDA required")
@unittest.skipIf(check_native_jit_disabled(), "Native DSL ops disabled")
@skipIfNoCuteDSL
class TestGroupedMm(TestCase):
    @parametrize("dtype", (torch.float16, torch.bfloat16))
    def test_sm120_varlen_k_grouped_mm(self, dtype):
        mat_a, mat_b, offs = _make_grouped_mm_inputs(dtype, True)
        _assert_quack_path_supported(self, mat_a, mat_b, offs)

        actual = torch._grouped_mm(mat_a, mat_b, offs=offs)
        self.assertEqual(actual, _reference_grouped_mm(mat_a, mat_b, offs))

    @parametrize("dtype", (torch.float16, torch.bfloat16))
    def test_sm120_varlen_k_grouped_mm_cudagraph(self, dtype):
        mat_a, mat_b, offs = _make_grouped_mm_inputs(dtype, True)
        _assert_quack_path_supported(self, mat_a, mat_b, offs)

        warmup_stream = torch.cuda.Stream()
        warmup_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup_stream):
            for _ in range(3):
                torch._grouped_mm(mat_a, mat_b, offs=offs)
        torch.cuda.current_stream().wait_stream(warmup_stream)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = torch._grouped_mm(mat_a, mat_b, offs=offs)
        graph.replay()

        self.assertEqual(actual, _reference_grouped_mm(mat_a, mat_b, offs))

    @parametrize("dtype", (torch.float16, torch.bfloat16))
    def test_sm120_varlen_m_grouped_mm(self, dtype):
        mat_a, mat_b, offs = _make_grouped_mm_inputs(dtype, False)
        _assert_quack_path_supported(self, mat_a, mat_b, offs)

        actual = torch._grouped_mm(mat_a, mat_b, offs=offs)
        self.assertEqual(actual, _reference_grouped_mm(mat_a, mat_b, offs))

    @parametrize("dtype", (torch.float16, torch.bfloat16))
    def test_sm120_varlen_m_grouped_mm_cudagraph(self, dtype):
        mat_a, mat_b, offs = _make_grouped_mm_inputs(dtype, False)
        _assert_quack_path_supported(self, mat_a, mat_b, offs)

        warmup_stream = torch.cuda.Stream()
        warmup_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup_stream):
            for _ in range(3):
                torch._grouped_mm(mat_a, mat_b, offs=offs)
        torch.cuda.current_stream().wait_stream(warmup_stream)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = torch._grouped_mm(mat_a, mat_b, offs=offs)
        graph.replay()

        self.assertEqual(actual, _reference_grouped_mm(mat_a, mat_b, offs))


instantiate_parametrized_tests(TestGroupedMm)


if __name__ == "__main__":
    run_tests()
