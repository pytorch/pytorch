# Owner(s): ["module: dsl-native-ops"]

import unittest

import torch
import torch._native  # noqa: F401
from torch.testing._internal.common_cuda import SM120OrLater, TEST_CUDA
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


def _reference_grouped_mm(
    mat_a: torch.Tensor,
    mat_b: torch.Tensor,
    offs: torch.Tensor,
) -> torch.Tensor:
    outs = []
    start = 0
    for group, end in enumerate(offs.cpu().tolist()):
        outs.append(mat_a[start:end] @ mat_b[group])
        start = end
    return torch.cat(outs, dim=0)


def _make_grouped_mm_inputs(dtype: torch.dtype):
    torch.manual_seed(0)
    sizes = [128, 192, 160]
    mat_a = torch.randn(sum(sizes), 128, device="cuda", dtype=dtype)
    mat_b = torch.randn(len(sizes), 128, 128, device="cuda", dtype=dtype)
    offs = torch.tensor(sizes, device="cuda", dtype=torch.int32).cumsum(0).to(torch.int32)
    return mat_a, mat_b, offs


@unittest.skipUnless(TEST_CUDA and SM120OrLater, "SM120 CUDA required")
class TestGroupedMm(TestCase):
    @parametrize("dtype", (torch.float16, torch.bfloat16))
    def test_sm120_varlen_m_grouped_mm(self, dtype):
        mat_a, mat_b, offs = _make_grouped_mm_inputs(dtype)

        actual = torch._grouped_mm(mat_a, mat_b, offs=offs)
        expected = _reference_grouped_mm(mat_a, mat_b, offs)

        atol = 2e-2 if dtype is torch.float16 else 2e-1
        self.assertEqual(actual, expected, atol=atol, rtol=0)

    @parametrize("dtype", (torch.float16, torch.bfloat16))
    def test_sm120_varlen_m_grouped_mm_cudagraph(self, dtype):
        mat_a, mat_b, offs = _make_grouped_mm_inputs(dtype)

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

        expected = _reference_grouped_mm(mat_a, mat_b, offs)
        atol = 2e-2 if dtype is torch.float16 else 2e-1
        self.assertEqual(actual, expected, atol=atol, rtol=0)


instantiate_parametrized_tests(TestGroupedMm)


if __name__ == "__main__":
    run_tests()
