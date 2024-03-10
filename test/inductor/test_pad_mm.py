# Owner(s): ["module: inductor"]
import unittest

import torch

import torch._inductor.config as inductor_config

from torch._dynamo.test_case import run_tests, TestCase
from torch._dynamo.testing import rand_strided
from torch._inductor.fx_passes.pad_mm import get_alignment_size, get_padded_length
from torch._inductor.utils import run_and_get_code
from torch.testing import FileCheck
from torch.testing._internal.inductor_utils import HAS_CUDA


class PadMMTest(TestCase):
    def test_pad_preserves_output_stride(
        self,
        m=2613,
        n=1029,
        k=1023,
        batch_size=3,
    ):
        with unittest.mock.patch(
            "torch._inductor.fx_passes.pad_mm._skip_do_bench_times", True
        ):
            mat1 = torch.ones((batch_size, m, k), device="cuda", dtype=torch.float16)
            mat2 = torch.ones((batch_size, k, n), device="cuda", dtype=torch.float16)
            expected_alignment = get_alignment_size(mat1)

            assert expected_alignment == 8, "Alignment for float16 should be 8"
            bmm_expected_result = torch.bmm(mat1, mat2)
            for keep_output_stride in [False, True]:
                with inductor_config.patch(
                    {
                        "keep_output_stride": keep_output_stride,
                    }
                ):
                    # reset dynamo cache. If we don't do that, the keep_output_stride
                    # setting can be ignored due to caching
                    torch._dynamo.reset()
                    bmm_compiled_result = torch.compile(
                        lambda mat1, mat2: torch.bmm(mat1, mat2), dynamic=False
                    )(mat1, mat2)
                    assert torch.allclose(
                        bmm_compiled_result, bmm_expected_result
                    ), "Compiled BMM results are not identical"
                    if keep_output_stride:
                        assert (
                            bmm_compiled_result.stride() == bmm_expected_result.stride()
                        ), "config.keep_output_stride is being violated by shape padding"
                    # BMM outputs are made contiguous, and therefore not aligned in preexisting impl.

            bias = torch.ones((m, n), device="cuda", dtype=torch.float16)
            bias_vec = torch.ones(n, device="cuda", dtype=torch.float16)
            mat1 = torch.ones((m, k), device="cuda", dtype=torch.float16)
            mat2 = torch.ones((k, n), device="cuda", dtype=torch.float16)
            expected_alignment = get_alignment_size(mat1)

            assert expected_alignment == 8, "Alignment for float16 should be 8"

            for keep_output_stride in [False, True]:
                with inductor_config.patch(
                    {
                        "keep_output_stride": keep_output_stride,
                    }
                ):
                    # reset dynamo cache. If we don't do that, the keep_output_stride
                    # setting can be ignored due to caching
                    torch._dynamo.reset()
                    mm_expected_result = torch.mm(mat1, mat2)
                    mm_compiled_result = torch.compile(
                        lambda mat1, mat2: torch.mm(mat1, mat2),
                        dynamic=False,
                    )(mat1, mat2)
                    assert torch.allclose(
                        mm_compiled_result, mm_expected_result
                    ), "Compiled BMM results are not identical"

                    if keep_output_stride:
                        assert (
                            mm_compiled_result.stride() == mm_expected_result.stride()
                        ), "config.keep_output_stride is being violated by shape padding."
                    else:
                        assert (
                            mm_compiled_result.stride() != mm_expected_result.stride()
                        ), "shape padding was not applied"

                    for bias_tensor in [bias, bias_vec]:
                        addmm_expected_result = torch.addmm(bias_tensor, mat1, mat2)
                        addmm_compiled_result = torch.compile(
                            lambda bias, mat1, mat2: torch.addmm(bias, mat1, mat2),
                            dynamic=False,
                        )(bias_tensor, mat1, mat2)
                        assert torch.allclose(
                            addmm_compiled_result, addmm_expected_result
                        ), "Compiled BMM results are not identical"
                        if keep_output_stride:
                            assert (
                                addmm_compiled_result.stride()
                                == addmm_expected_result.stride()
                            ), "config.keep_output_stride is being violated by shape padding"
                            # ADDMM outputs are made contiguous, and therefore not aligned in preexisting impl.

    @inductor_config.patch(max_autotune=True, max_autotune_gemm_backends="TRITON")
    @inductor_config.patch(
        max_autotune=True,
        max_autotune_gemm_backends="TRITON",
        shape_padding=True,
        keep_output_stride=False,
    )
    def test_pad_mm_dyn_m(self):
        M = 40
        K1 = 581
        K2 = 49
        N = 30

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.w = rand_strided(
                    (K2, N), (1, K2), device="cuda", dtype=torch.float32
                )

            def forward(self, a):
                a1 = torch.narrow(a, 1, 0, K2)
                return torch.mm(a1, self.w)

        fn = Model().cuda()
        a = rand_strided((M, K1), (K1, 1), device="cuda", dtype=torch.float32)
        aligned_k = get_padded_length(K2, get_alignment_size(a)) + K2
        torch._dynamo.mark_dynamic(a, 0)
        with unittest.mock.patch(
            "torch._inductor.fx_passes.pad_mm._skip_do_bench_times", True
        ):
            res1 = fn(a)
            compiled_fn = torch.compile(fn)
            res2, (code,) = run_and_get_code(compiled_fn, a)
            FileCheck().check(f"K = {aligned_k}").run(code)
        self.assertEqual(res1, res2)

    @inductor_config.patch(max_autotune=True, max_autotune_gemm_backends="TRITON")
    @inductor_config.patch(
        max_autotune=True,
        max_autotune_gemm_backends="TRITON",
        shape_padding=True,
        keep_output_stride=False,
    )
    def test_cat_pad_mm_dyn_m(self):
        M1 = 128
        M2 = 40
        K1 = 129
        K2 = 111
        N = 100

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.w = rand_strided(
                    (K2, N), (1, K2), device="cuda", dtype=torch.float32
                )

            def forward(self, a, b):
                c = torch.cat([a, b], dim=0)
                a1 = torch.narrow(c, 1, 0, K2)
                return torch.mm(a1, self.w)

        fn = Model().cuda()
        a = rand_strided((M1, K1), (K1, 1), device="cuda", dtype=torch.float32)
        b = rand_strided((M2, K1), (K1, 1), device="cuda", dtype=torch.float32)
        torch._dynamo.mark_dynamic(a, 0)
        torch._dynamo.mark_dynamic(b, 0)
        aligned_k = get_padded_length(K2, get_alignment_size(a)) + K2
        with unittest.mock.patch(
            "torch._inductor.fx_passes.pad_mm._skip_do_bench_times", True
        ):
            res1 = fn(a, b)
            compiled_fn = torch.compile(fn)
            res2, (code,) = run_and_get_code(compiled_fn, a, b)
            FileCheck().check(f"K = {aligned_k}").run(code)
        self.assertEqual(res1, res2)

    @inductor_config.patch(max_autotune=True, max_autotune_gemm_backends="TRITON")
    @inductor_config.patch(
        max_autotune=True,
        max_autotune_gemm_backends="TRITON",
        shape_padding=True,
        keep_output_stride=False,
    )
    def test_pad_mm_dyn_n(self):
        M = 20
        K = 81
        N = 30

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, a, b):
                return torch.mm(a, b)

        fn = Model().cuda()
        a = rand_strided((M, K), (K, 1), device="cuda", dtype=torch.float32)
        b = rand_strided((K, N), (1, K), device="cuda", dtype=torch.float32)
        aligned_k = get_padded_length(K, get_alignment_size(a)) + K
        torch._dynamo.mark_dynamic(b, 1)
        with unittest.mock.patch(
            "torch._inductor.fx_passes.pad_mm._skip_do_bench_times", True
        ):
            res1 = fn(a, b)
            compiled_fn = torch.compile(fn)
            res2, (code,) = run_and_get_code(compiled_fn, a, b)
            FileCheck().check(f"K = {aligned_k}").run(code)
        self.assertEqual(res1, res2)

    @inductor_config.patch(max_autotune=True, max_autotune_gemm_backends="TRITON")
    @inductor_config.patch(
        max_autotune=True,
        max_autotune_gemm_backends="TRITON",
        shape_padding=True,
        keep_output_stride=False,
    )
    def test_pad_mm_dyn_k(self):
        M = 21
        K = 80
        N = 30

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, a, b):
                return torch.mm(a, b)

        fn = Model().cuda()
        a = rand_strided((M, K), (K, 1), device="cuda", dtype=torch.float32)
        b = rand_strided((K, N), (1, K), device="cuda", dtype=torch.float32)
        aligned_m = get_padded_length(M, get_alignment_size(a)) + M
        torch._dynamo.mark_dynamic(a, 1)
        torch._dynamo.mark_dynamic(b, 0)
        with unittest.mock.patch(
            "torch._inductor.fx_passes.pad_mm._skip_do_bench_times", True
        ):
            res1 = fn(a, b)
            compiled_fn = torch.compile(fn)
            res2, (code,) = run_and_get_code(compiled_fn, a, b)
            FileCheck().check(f"M = {aligned_m}").run(code)
        self.assertEqual(res1, res2)

    def test_pad_mm_dyn_mnk(self):
        M = 20
        K = 81
        N = 30

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, a, b):
                return torch.mm(a, b)

        fn = Model().cuda()
        a = rand_strided((M, K), (K, 1), device="cuda", dtype=torch.float32)
        b = rand_strided((K, N), (1, K), device="cuda", dtype=torch.float32)
        torch._dynamo.mark_dynamic(a, 0)
        torch._dynamo.mark_dynamic(a, 1)
        torch._dynamo.mark_dynamic(b, 0)
        torch._dynamo.mark_dynamic(b, 1)
        with unittest.mock.patch(
            "torch._inductor.fx_passes.pad_mm._skip_do_bench_times", True
        ):
            res1 = fn(a, b)
            compiled_fn = torch.compile(fn)
            res2, (code,) = run_and_get_code(compiled_fn, a, b)
        self.assertEqual(res1, res2)

    @inductor_config.patch(max_autotune=True, max_autotune_gemm_backends="TRITON")
    @inductor_config.patch(
        max_autotune=True,
        max_autotune_gemm_backends="TRITON",
        shape_padding=True,
        keep_output_stride=False,
    )
    def test_pad_bmm_dyn_b(self):
        B = 10
        M = 128
        K = 33
        N = 40

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, a, b):
                return torch.bmm(a, b)

        fn = Model().cuda()
        a = torch.randn(B, M, K, device="cuda", dtype=torch.float32)
        b = torch.randn(B, K, N, device="cuda", dtype=torch.float32)
        aligned_k = get_padded_length(K, get_alignment_size(a)) + K
        torch._dynamo.mark_dynamic(a, 0)
        torch._dynamo.mark_dynamic(b, 0)
        with unittest.mock.patch(
            "torch._inductor.fx_passes.pad_mm._skip_do_bench_times", True
        ):
            res1 = fn(a, b)
            compiled_fn = torch.compile(fn)
            res2, (code,) = run_and_get_code(compiled_fn, a, b)
            FileCheck().check(f"K = {aligned_k}").run(code)
        self.assertEqual(res1, res2)

    @inductor_config.patch(max_autotune=True, max_autotune_gemm_backends="TRITON")
    @inductor_config.patch(
        max_autotune=True,
        max_autotune_gemm_backends="TRITON",
        shape_padding=True,
        keep_output_stride=False,
    )
    def test_pad_bmm_dyn_k(self):
        B = 10
        M = 128
        K = 40
        N = 41

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, a, b):
                return torch.bmm(a, b)

        fn = Model().cuda()
        a = torch.randn(B, M, K, device="cuda", dtype=torch.float32)
        b = torch.randn(B, K, N, device="cuda", dtype=torch.float32)
        aligned_n = get_padded_length(N, get_alignment_size(b)) + N
        torch._dynamo.mark_dynamic(a, 2)
        torch._dynamo.mark_dynamic(b, 1)
        with unittest.mock.patch(
            "torch._inductor.fx_passes.pad_mm._skip_do_bench_times", True
        ):
            res1 = fn(a, b)
            compiled_fn = torch.compile(fn)
            res2, (code,) = run_and_get_code(compiled_fn, a, b)
            FileCheck().check(f"N = {aligned_n}").run(code)
        self.assertEqual(res1, res2)

    @inductor_config.patch(max_autotune=True, max_autotune_gemm_backends="TRITON")
    @inductor_config.patch(
        max_autotune=True,
        max_autotune_gemm_backends="TRITON",
        shape_padding=True,
        keep_output_stride=False,
    )
    def test_pad_bmm_dyn_bm(self):
        B = 10
        M = 128
        K = 40
        N = 41

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, a, b):
                return torch.bmm(a, b)

        fn = Model().cuda()
        a = torch.randn(B, M, K, device="cuda", dtype=torch.float32)
        b = torch.randn(B, K, N, device="cuda", dtype=torch.float32)
        aligned_n = get_padded_length(N, get_alignment_size(b)) + N
        torch._dynamo.mark_dynamic(a, 0)
        torch._dynamo.mark_dynamic(a, 1)
        torch._dynamo.mark_dynamic(b, 0)
        with unittest.mock.patch(
            "torch._inductor.fx_passes.pad_mm._skip_do_bench_times", True
        ):
            res1 = fn(a, b)
            compiled_fn = torch.compile(fn)
            res2, (code,) = run_and_get_code(compiled_fn, a, b)
            FileCheck().check(f"N = {aligned_n}").run(code)
        self.assertEqual(res1, res2)

    @inductor_config.patch(max_autotune=True, max_autotune_gemm_backends="TRITON")
    @inductor_config.patch(
        max_autotune=True,
        max_autotune_gemm_backends="TRITON",
        shape_padding=True,
        keep_output_stride=False,
    )
    def test_pad_addmm_dyn_m(self):
        M = 128
        K = 33
        N = 40

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, a, b, c):
                return torch.addmm(a, b, c)

        fn = Model().cuda()
        a = torch.randn(M, N, device="cuda", dtype=torch.float32)
        b = torch.randn(M, K, device="cuda", dtype=torch.float32)
        c = torch.randn(K, N, device="cuda", dtype=torch.float32)
        aligned_k = get_padded_length(K, get_alignment_size(b)) + K
        torch._dynamo.mark_dynamic(a, 0)
        torch._dynamo.mark_dynamic(b, 0)
        with unittest.mock.patch(
            "torch._inductor.fx_passes.pad_mm._skip_do_bench_times", True
        ):
            res1 = fn(a, b, c)
            compiled_fn = torch.compile(fn)
            res2, (code,) = run_and_get_code(compiled_fn, a, b, c)
            FileCheck().check(f"K = {aligned_k}").run(code)
        self.assertEqual(res1, res2)

    @inductor_config.patch(max_autotune=True, max_autotune_gemm_backends="TRITON")
    @inductor_config.patch(
        max_autotune=True,
        max_autotune_gemm_backends="TRITON",
        keep_output_stride=False,
        shape_padding=True,
    )
    def test_pad_addmm_dyn_mn(self):
        M = 128
        K = 33
        N = 40

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, a, b, c):
                return torch.addmm(a, b, c)

        fn = Model().cuda()
        a = torch.randn(M, N, device="cuda", dtype=torch.float32)
        b = torch.randn(M, K, device="cuda", dtype=torch.float32)
        c = torch.randn(K, N, device="cuda", dtype=torch.float32)
        torch._dynamo.mark_dynamic(a, 0)
        torch._dynamo.mark_dynamic(a, 1)
        torch._dynamo.mark_dynamic(b, 0)
        torch._dynamo.mark_dynamic(c, 1)
        with unittest.mock.patch(
            "torch._inductor.fx_passes.pad_mm._skip_do_bench_times", True
        ):
            res1 = fn(a, b, c)
            compiled_fn = torch.compile(fn)
            res2, (code,) = run_and_get_code(compiled_fn, a, b, c)
            # no padding
            FileCheck().check(f"K = {K}").run(code)
        self.assertEqual(res1, res2)


if __name__ == "__main__":
    if HAS_CUDA:
        run_tests()
