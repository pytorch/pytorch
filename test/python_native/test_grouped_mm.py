# Owner(s): ["module: dsl-native-ops"]

import unittest
from unittest import mock

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
@unittest.skipUnless(TEST_CUDA, "CUDA required")
class TestGroupedMmCond(TestCase):
    def test_sm120_cond_rejects_invalid_native_override_inputs(self):
        from torch._native.ops.grouped_mm.impl import _grouped_mm_sm12x_cond

        mat_a, mat_b, offs = _make_grouped_mm_inputs(torch.float16, True)
        mat_a_3d_case, mat_b_3d, offs_3d = _make_grouped_mm_inputs(torch.float16, False)
        mat_a_bf16, _, _ = _make_grouped_mm_inputs(torch.bfloat16, True)
        mat_a_fp32, mat_b_fp32, _ = _make_grouped_mm_inputs(torch.float32, True)
        strided_offs = torch.empty(
            offs.numel() * 2, device="cuda", dtype=torch.int32
        ).as_strided((offs.numel(),), (2,))
        invalid_stride_mat_a = torch.empty(
            1024, device="cuda", dtype=torch.float16
        ).as_strided((16, 16), (2, 3))
        invalid_stride_mat_b = torch.empty(
            1024, device="cuda", dtype=torch.float16
        ).as_strided((16, 16), (2, 3))

        with mock.patch("torch.cuda.get_device_capability", return_value=(12, 0)):
            self.assertTrue(_grouped_mm_sm12x_cond(mat_a, mat_b, offs=offs))
            self.assertTrue(
                _grouped_mm_sm12x_cond(mat_a_3d_case, mat_b_3d, offs=offs_3d)
            )

            for name, kwargs in (
                ("mismatched_input_dtypes", {"self": mat_a_bf16}),
                ("float32_input_dtype", {"self": mat_a_fp32, "mat2": mat_b_fp32}),
                ("out_dtype_mismatch", {"out_dtype": torch.float32}),
                ("offs_none", {"offs": None}),
                ("bias_not_none", {"bias": torch.empty_like(mat_b)}),
                ("offs_wrong_numel", {"mat2": mat_b_3d, "offs": offs_3d[:-1]}),
                ("offs_dim_gt_one", {"offs": offs.reshape(2, 2)}),
                ("offs_stride_gt_one", {"offs": strided_offs}),
                ("offs_no_elements", {"offs": offs[:0]}),
                ("zero_dim_mat_a", {"self": mat_a[0, 0]}),
                ("three_dim_mat_a", {"self": mat_a.unsqueeze(0)}),
                ("zero_dim_mat_b", {"mat2": mat_b[0, 0]}),
                ("one_dim_mat_b", {"mat2": mat_b[:, 0]}),
                ("four_dim_mat_b", {"mat2": mat_b_3d.unsqueeze(0)}),
                ("invalid_stride_mat_a", {"self": invalid_stride_mat_a}),
                ("invalid_stride_mat_b", {"mat2": invalid_stride_mat_b}),
            ):
                with self.subTest(name=name):
                    args = {"self": mat_a, "mat2": mat_b, "offs": offs}
                    args.update(kwargs)
                    self.assertFalse(_grouped_mm_sm12x_cond(**args))


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
