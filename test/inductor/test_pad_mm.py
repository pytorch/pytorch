# Owner(s): ["module: inductor"]
import unittest
from unittest.mock import patch

import torch

import torch._inductor.config as inductor_config

from torch._dynamo.test_case import run_tests
from torch._dynamo.testing import rand_strided

from torch._inductor import config
from torch._inductor.fx_passes.pad_mm import (
    addmm_replace,
    bmm_replace,
    call_addmm,
    call_bmm,
    call_mm,
    get_alignment_size,
    get_padded_length,
    mm_replace,
    should_pad_common,
)
from torch._inductor.utils import run_and_get_code
from torch.testing import FileCheck
from torch.testing._internal.common_utils import parametrize, TestCase
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
        force_shape_pad=True,
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

    @inductor_config.patch(
        max_autotune=True,
        max_autotune_gemm_backends="TRITON",
        force_shape_pad=True,
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

    @inductor_config.patch(
        max_autotune=True,
        max_autotune_gemm_backends="TRITON",
        force_shape_pad=True,
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

    @inductor_config.patch(
        max_autotune=True,
        max_autotune_gemm_backends="TRITON",
        force_shape_pad=True,
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

    @inductor_config.patch(
        max_autotune=True,
        max_autotune_gemm_backends="TRITON",
        force_shape_pad=True,
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

    @inductor_config.patch(
        max_autotune=True,
        max_autotune_gemm_backends="TRITON",
        force_shape_pad=True,
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

    @inductor_config.patch(
        max_autotune=True,
        max_autotune_gemm_backends="TRITON",
        force_shape_pad=True,
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

    @inductor_config.patch(
        max_autotune=True,
        max_autotune_gemm_backends="TRITON",
        force_shape_pad=True,
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

    @inductor_config.patch(
        max_autotune=True,
        max_autotune_gemm_backends="TRITON",
        keep_output_stride=False,
        force_shape_pad=True,
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

    @staticmethod
    def _check_tensor_alignment(tensor, expected_alignment):
        contiguous_dim_count = 0
        for stride in reversed(tensor.stride()):
            assert (stride == 1) or (
                stride % expected_alignment == 0
            ), f"Expected all non-contiguous strides of tensor with shape {tensor.shape} and strides {tensor.stride()} to be aligned to a multiple of {expected_alignment}"  # noqa: B950
            if stride == 1:
                contiguous_dim_count += 1

    @parametrize(
        "m,n,k", [(1, 1, 1), (16, 32, 64), (17, 33, 65), (16, 15, 8), (15, 32, 16)]
    )
    @parametrize("shape_pad_use_transpose", (False, True))
    @parametrize("keep_output_stride", (False, True))
    def test_pad_nobatch(
        self,
        m=6,
        n=9,
        k=11,
        shape_pad_use_transpose: bool = True,
        keep_output_stride=False,
    ):
        with config.patch(
            {
                "shape_pad_use_transpose": shape_pad_use_transpose,
                "force_shape_pad": True,
                "keep_output_stride": keep_output_stride,
            }
        ):
            mat1 = torch.ones((m, k), device="cuda", dtype=torch.float16)
            mat2 = torch.ones((k, n), device="cuda", dtype=torch.float16)
            bias = torch.ones((m, n), device="cuda", dtype=torch.float16)
            expected_alignment = get_alignment_size(mat1)
            assert (
                expected_alignment >= 2
            ), "Expected alignment should be greater or equal to 2"
            assert should_pad_common(
                mat1, mat2
            ), "This should pass the common padding criteria"
            assert should_pad_common(
                mat1, mat2, bias
            ), "This should pass the common padding criteria"

            orig_addmm = call_addmm
            called_checked = False

            def aten_addmm_checked(b, m1, m2, *args, **kwargs):
                nonlocal called_checked
                self._check_tensor_alignment(b, expected_alignment)
                self._check_tensor_alignment(m1, expected_alignment)
                self._check_tensor_alignment(m2, expected_alignment)
                res = orig_addmm(b, m1, m2, *args, **kwargs)
                self._check_tensor_alignment(res, expected_alignment)
                called_checked = True
                return res

            with patch(
                "torch._inductor.fx_passes.pad_mm.call_addmm", aten_addmm_checked
            ):
                addmm_result = addmm_replace(bias, mat1, mat2)

            addmm_expected_result = torch.addmm(bias, mat1, mat2)
            assert torch.allclose(
                addmm_result, addmm_expected_result
            ), "ADDMM results are not identical"

            addmm_compiled_result = torch.compile(
                lambda bias, mat1, mat2: torch.addmm(bias, mat1, mat2), dynamic=False
            )(bias, mat1, mat2)
            assert torch.allclose(
                addmm_compiled_result, addmm_expected_result
            ), "Compiled ADDMM results are not identical"

            if not keep_output_stride:
                self._check_tensor_alignment(addmm_compiled_result, expected_alignment)
            else:
                assert (
                    addmm_compiled_result.stride() == addmm_expected_result.stride()
                ), "config.keep_output_stride is being violated by shape padding"

            orig_mm = call_mm
            called_checked = False

            def aten_mm_checked(m1, m2, *args, **kwargs):
                nonlocal called_checked
                self._check_tensor_alignment(m1, expected_alignment)
                self._check_tensor_alignment(m2, expected_alignment)
                res = orig_mm(m1, m2, *args, **kwargs)
                self._check_tensor_alignment(res, expected_alignment)
                called_checked = True
                return res

            with patch("torch._inductor.fx_passes.pad_mm.call_mm", aten_mm_checked):
                mm_result = mm_replace(mat1, mat2)
            assert called_checked, "patched / checked aten.mm was not called at all"

            mm_expected_result = torch.mm(mat1, mat2)
            assert torch.allclose(
                mm_result, mm_expected_result
            ), "MM results are not identical"

            mm_compiled_result = torch.compile(lambda m1, m2: m1 @ m2, dynamic=False)(
                mat1, mat2
            )
            assert torch.allclose(
                mm_compiled_result, mm_expected_result
            ), "Compiled MM results are not identical"

            if not keep_output_stride:
                self._check_tensor_alignment(mm_compiled_result, expected_alignment)
            else:
                assert (
                    mm_compiled_result.stride() == mm_expected_result.stride()
                ), "config.keep_output_stride is being violated by shape padding"

    @parametrize(
        "m,n,k,batch_size",
        [
            (1, 1, 1, 8),
            (16, 32, 64, 8),
            (17, 33, 65, 7),
            (16, 33, 64, 4),
            (15, 32, 62, 3),
        ],
    )
    @parametrize("shape_pad_use_transpose", (False, True))
    @parametrize("keep_output_stride", (False, True))
    def test_pad_batch(
        self,
        m=6,
        n=9,
        k=11,
        batch_size=3,
        shape_pad_use_transpose: bool = True,
        keep_output_stride=False,
    ):
        with config.patch(
            {
                "shape_pad_use_transpose": shape_pad_use_transpose,
                "force_shape_pad": True,
                "keep_output_stride": keep_output_stride,
            }
        ):
            mat1 = torch.ones((batch_size, m, k), device="cuda", dtype=torch.float16)
            mat2 = torch.ones((batch_size, k, n), device="cuda", dtype=torch.float16)
            expected_alignment = get_alignment_size(mat1)

            assert expected_alignment == 8, "Alignment for float16 should be 8"
            assert should_pad_common(
                mat1, mat2
            ), "This should pass the common padding criteria"

            orig_bmm = call_bmm
            called_checked = False

            def aten_bmm_checked(m1, m2, *args, **kwargs):
                nonlocal called_checked
                self._check_tensor_alignment(m1, expected_alignment)
                self._check_tensor_alignment(m2, expected_alignment)
                res = orig_bmm(m1, m2, *args, **kwargs)
                self._check_tensor_alignment(res, expected_alignment)
                called_checked = True
                return res

            with patch("torch._inductor.fx_passes.pad_mm.call_bmm", aten_bmm_checked):
                bmm_result = bmm_replace(mat1, mat2)

            bmm_expected_result = torch.bmm(mat1, mat2)

            assert torch.allclose(
                bmm_result, bmm_expected_result
            ), "BMM results are not identical"
            self._check_tensor_alignment(bmm_result, expected_alignment)

            bmm_compiled_result = torch.compile(
                lambda mat1, mat2: torch.bmm(mat1, mat2), dynamic=False
            )(mat1, mat2)
            assert torch.allclose(
                bmm_compiled_result, bmm_expected_result
            ), "Compiled BMM results are not identical"

            if not keep_output_stride:
                self._check_tensor_alignment(bmm_compiled_result, expected_alignment)
            else:
                assert (
                    bmm_compiled_result.stride() == bmm_expected_result.stride()
                ), "config.keep_output_stride is being violated by shape padding"


if __name__ == "__main__":
    if HAS_CUDA:
        run_tests()
