# Owner(s): ["module: linear algebra"]

import unittest

import torch
import torch.nn.functional as F
from torch.testing._internal.common_cuda import SM80OrLater
from torch.testing._internal.common_device_type import (
    dtypes,
    instantiate_device_type_tests,
    onlyAccelerator,
)
from torch.testing._internal.common_utils import (
    decorateIf,
    parametrize,
    run_tests,
    TestCase,
)


class TestMatmul(TestCase):
    def setUp(self):
        super().setUp()
        torch.backends.cuda.matmul.allow_tf32 = False

    def tearDown(self):
        torch.backends.cuda.matmul.allow_tf32 = True
        super().tearDown()

    def grouped_mm_helper(self, alist, blist, gOlist, agradlist, bgradlist, outlist):
        for a, b, gO, agrad, bgrad, out in zip(
            alist, blist, gOlist, agradlist, bgradlist, outlist
        ):
            a = a.clone().detach().requires_grad_()
            b = b.clone().detach().requires_grad_()
            out_ref = torch.mm(a, b.t())
            out_ref.backward(gO)
            self.assertEqual(out, out_ref)
            if agrad is not None:
                self.assertEqual(agrad, a.grad)
                self.assertEqual(bgrad, b.grad)

    @onlyAccelerator
    @decorateIf(
        unittest.skipIf(not SM80OrLater, "Grouped gemm requires SM80+ on CUDA"),
        lambda params: "cuda" in params["device"],
    )
    @parametrize("strided", [False, True])
    @parametrize("a_row_major", [False, True])
    @parametrize("b_row_major", [False, True])
    @dtypes(torch.bfloat16, torch.float32, torch.float16)
    def test_grouped_gemm_2d_2d(self, device, strided, a_row_major, b_row_major, dtype):
        m, n, k, n_groups = 16, 32, 64, 4
        if a_row_major:
            a = torch.randn(
                m, k * n_groups + k * int(strided), device=device, dtype=dtype
            )[:, : k * n_groups]
        else:
            a = torch.randn(
                k * n_groups + k * int(strided), m, device=device, dtype=dtype
            ).t()[:, : k * n_groups]

        if b_row_major:
            b = torch.randn(
                n, k * n_groups + k * int(strided), device=device, dtype=dtype
            )[:, : k * n_groups]
        else:
            b = torch.randn(
                k * n_groups + k * int(strided), n, device=device, dtype=dtype
            ).t()[:, : k * n_groups]

        a.requires_grad_(True)
        b.requires_grad_(True)
        offs = torch.arange(k, n_groups * k + 1, k, device=device, dtype=torch.int32)

        f = F.grouped_mm
        out = f(a, b.t(), offs=offs, out_dtype=dtype)
        gO = torch.rand_like(out)
        out.backward(gO)
        offs_cpu = offs.cpu()
        alist, blist, agradlist, bgradlist = [], [], [], []
        start = 0
        for i in range(n_groups):
            alist.append(a[:, start : offs_cpu[i]])
            blist.append(b[:, start : offs_cpu[i]])
            agradlist.append(a.grad[:, start : offs_cpu[i]])
            bgradlist.append(b.grad[:, start : offs_cpu[i]])
            start = offs_cpu[i]
        self.grouped_mm_helper(alist, blist, gO, agradlist, bgradlist, out)

    @onlyAccelerator
    @decorateIf(
        unittest.skipIf(not SM80OrLater, "Grouped gemm requires SM80+ on CUDA"),
        lambda params: "cuda" in params["device"],
    )
    @parametrize("strided", [False, True])
    @parametrize("a_row_major", [False, True])
    @parametrize("b_row_major", [False, True])
    @dtypes(torch.bfloat16, torch.float32, torch.float16)
    def test_grouped_gemm_2d_3d(self, device, strided, a_row_major, b_row_major, dtype):
        s_int = int(strided)
        m, n, k, n_groups = 16, 32, 64, 4
        if a_row_major:
            a = torch.randn(m * n_groups, k * (1 + s_int), device=device, dtype=dtype)[
                :, :k
            ]
        else:
            a = torch.randn(
                k, (m + 2 * s_int) * n_groups, device=device, dtype=dtype
            ).t()[: m * n_groups, :]

        if b_row_major:
            b = torch.randn(
                n_groups * (1 + s_int), n, k * (1 + s_int), device=device, dtype=dtype
            )[:: (1 + s_int), :, :k]
        else:
            b = torch.randn(
                n_groups * (1 + s_int), k * (1 + s_int), n, device=device, dtype=dtype
            ).transpose(-2, -1)[:: (1 + s_int), :, :k]

        a.requires_grad_(True)
        b.requires_grad_(True)

        a_contig = a if a_row_major else a.t()
        self.assertTrue(a_contig.is_contiguous() is not strided)
        b_contig = b if b_row_major else b.transpose(-2, -1)
        self.assertTrue(b_contig.is_contiguous() is not strided)
        for check_zero_size in (False, True):
            if check_zero_size and n_groups <= 1:
                continue

            a.grad = None
            b.grad = None
            offs = torch.arange(
                m, n_groups * m + 1, m, device=device, dtype=torch.int32
            )
            if check_zero_size:
                offs[0] = offs[1]

            f = F.grouped_mm
            out = f(a, b.transpose(-2, -1), offs=offs, out_dtype=dtype)
            gO = torch.rand_like(out)
            if not check_zero_size:
                out.backward(gO)
            offs_cpu = offs.cpu()
            alist, agradlist, gOlist, outlist = [], [], [], []
            bgradlist = [None] * n_groups if check_zero_size else b.grad
            start = 0
            for i in range(n_groups):
                alist.append(a[start : offs_cpu[i]])
                agradlist.append(
                    None if check_zero_size else a.grad[start : offs_cpu[i]]
                )
                outlist.append(out[start : offs_cpu[i]])
                gOlist.append(gO[start : offs_cpu[i]])
                start = offs_cpu[i]
            self.grouped_mm_helper(alist, b, gOlist, agradlist, bgradlist, outlist)

    @onlyAccelerator
    @decorateIf(
        unittest.skipIf(not SM80OrLater, "Grouped gemm requires SM80+ on CUDA"),
        lambda params: "cuda" in params["device"],
    )
    @parametrize("strided", [False, True])
    @parametrize("a_row_major", [False, True])
    @parametrize("b_row_major", [False, True])
    @dtypes(torch.bfloat16, torch.float32, torch.float16)
    def test_grouped_gemm_3d_3d(self, device, strided, a_row_major, b_row_major, dtype):
        s_int = int(strided)
        m, n, k, n_groups = 16, 32, 64, 4
        if a_row_major:
            a = torch.randn(
                n_groups * (1 + s_int), m, k * (1 + s_int), device=device, dtype=dtype
            )[:: (1 + s_int), :, :k]
        else:
            a = torch.randn(
                n_groups * (1 + s_int), k * (1 + s_int), m, device=device, dtype=dtype
            ).transpose(-2, -1)[:: (1 + s_int), :, :k]
        if b_row_major:
            b = torch.randn(
                n_groups * (1 + s_int), n, k * (1 + s_int), device=device, dtype=dtype
            )[:: (1 + s_int), :, :k]
        else:
            b = torch.randn(
                n_groups * (1 + s_int), k * (1 + s_int), n, device=device, dtype=dtype
            ).transpose(-2, -1)[:: (1 + s_int), :, :k]
        a.requires_grad_(True)
        b.requires_grad_(True)

        a_contig = a if a_row_major else a.transpose(-2, -1)
        self.assertTrue(a_contig.is_contiguous() is not strided)
        b_contig = b if b_row_major else b.transpose(-2, -1)
        self.assertTrue(b_contig.is_contiguous() is not strided)

        f = F.grouped_mm
        out = f(a, b.transpose(-2, -1), out_dtype=dtype)
        gO = torch.rand_like(out)
        out.backward(gO)
        self.grouped_mm_helper(a, b, gO, a.grad, b.grad, out)

    @onlyAccelerator
    @decorateIf(
        unittest.skipIf(not SM80OrLater, "Grouped gemm requires SM80+ on CUDA"),
        lambda params: "cuda" in params["device"],
    )
    @parametrize("strided", [False, True])
    @parametrize("a_row_major", [False, True])
    @parametrize("b_row_major", [False, True])
    @dtypes(torch.bfloat16, torch.float32, torch.float16)
    def test_grouped_gemm_3d_2d(self, device, strided, a_row_major, b_row_major, dtype):
        s_int = int(strided)
        m, n, k, n_groups = 16, 32, 64, 4
        if a_row_major:
            a = torch.randn(
                n_groups * (1 + s_int), m, k * (1 + s_int), device=device, dtype=dtype
            )[:: (1 + s_int), :, :k]
        else:
            a = torch.randn(
                n_groups * (1 + s_int), k * (1 + s_int), m, device=device, dtype=dtype
            ).transpose(-2, -1)[:: (1 + s_int), :, :k]
        if b_row_major:
            b = torch.randn(n * n_groups, k * (1 + s_int), device=device, dtype=dtype)[
                :, :k
            ]
        else:
            b = torch.randn(
                k, n * (n_groups + s_int), device=device, dtype=dtype
            ).transpose(-2, -1)[: n * n_groups, :]

        a.requires_grad_(True)
        b.requires_grad_(True)

        a_contig = a if a_row_major else a.transpose(-2, -1)
        self.assertTrue(a_contig.is_contiguous() is not strided)
        b_contig = b if b_row_major else b.transpose(-2, -1)
        self.assertTrue(b_contig.is_contiguous() is not strided)
        for check_zero_size in (False, True):
            if check_zero_size and n_groups <= 1:
                continue

            offs = torch.arange(
                n, n_groups * n + 1, n, device=device, dtype=torch.int32
            )
            if check_zero_size:
                offs[0] = offs[1]

            f = F.grouped_mm
            out = f(a, b.transpose(-2, -1), offs=offs, out_dtype=dtype)
            gO = torch.rand_like(out)
            if not check_zero_size:
                out.backward(gO)
            offs_cpu = offs.cpu()
            blist, outlist, bgradlist, gOlist = [], [], [], []
            agradlist = [None] * n_groups if check_zero_size else a.grad
            start = 0
            for i in range(n_groups):
                blist.append(b[start : offs_cpu[i]])
                bgradlist.append(b.grad[start : offs_cpu[i]])
                outlist.append(out[:, start : offs_cpu[i]])
                gOlist.append(gO[:, start : offs_cpu[i]])
                start = offs_cpu[i]
            self.grouped_mm_helper(a, blist, gOlist, agradlist, bgradlist, outlist)

    @onlyAccelerator
    @parametrize(
        "ops",
        [
            ("mm", torch.mm),
            ("bmm", torch.bmm),
            ("addmm", torch.addmm),
            ("baddbmm", torch.baddbmm),
        ],
    )
    def test_input_dimension_checking_out_dtype(self, device, ops):
        op_name, op = ops
        B = 2
        M, N, K = 32, 32, 32

        def is_addmm():
            return "add" in op_name

        def is_batched():
            return "bmm" in op_name

        if is_batched():
            a = torch.randn(B, M, K, device=device, dtype=torch.bfloat16)
            mismatch_k_b = torch.randn(B, K + 1, N, device=device, dtype=torch.bfloat16)
            c = torch.randn(B, M, N, device=device, dtype=torch.bfloat16)
            extra_dim_b = a.clone().unsqueeze(0)

            mismatch_k_err = (
                "Expected size for first two dimensions of batch2 tensor to be"
            )
            extra_dim_err = "batch2 must be a 3D tensor"
        else:
            a = torch.randn(M, K, device=device, dtype=torch.bfloat16)
            mismatch_k_b = torch.randn(K + 1, N, device=device, dtype=torch.bfloat16)
            c = torch.randn(M, N, device=device, dtype=torch.bfloat16)
            extra_dim_b = a.clone().unsqueeze(0)

            mismatch_k_err = "mat1 and mat2 shapes cannot be multiplied"
            extra_dim_err = "mat2 must be a matrix, got 3-D tensor"

        # Test mismatch K
        with self.assertRaisesRegex(RuntimeError, mismatch_k_err):
            if is_addmm():
                op(c, a, mismatch_k_b, out_dtype=torch.float32)
            else:
                op(a, mismatch_k_b, out_dtype=torch.float32)

        # Test extra dimension
        with self.assertRaisesRegex(RuntimeError, extra_dim_err):
            if is_addmm():
                op(c, a, extra_dim_b, out_dtype=torch.float32)
            else:
                op(c, extra_dim_b, out_dtype=torch.float32)

        if is_batched():
            with self.assertRaisesRegex(
                RuntimeError,
                "Expected size for first two dimensions of batch2 tensor to be",
            ):
                mismatch_batch_dim_b = torch.randn(
                    B + 1, K, N, device=device, dtype=torch.bfloat16
                )
                if is_addmm():
                    op(c, a, mismatch_batch_dim_b, out_dtype=torch.float32)
                else:
                    op(a, mismatch_batch_dim_b, out_dtype=torch.float32)


instantiate_device_type_tests(TestMatmul, globals(), except_for="cpu")

if __name__ == "__main__":
    TestCase._default_dtype_check_enabled = True
    run_tests()
