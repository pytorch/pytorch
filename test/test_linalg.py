import torch
import numpy as np

import sys
import subprocess
import os
import unittest
import itertools
import warnings
import math
from math import inf, nan, isnan
import random
from random import randrange
from itertools import product
from functools import reduce

from torch.testing._internal.common_utils import \
    (TestCase, run_tests, TEST_SCIPY, IS_MACOS, IS_WINDOWS, slowTest,
     TEST_WITH_ASAN, make_tensor, TEST_WITH_ROCM, IS_FBCODE, IS_REMOTE_GPU,
     wrapDeterministicFlagAPITest, iter_indices, gradcheck, gradgradcheck)
from torch.testing._internal.common_device_type import \
    (instantiate_device_type_tests, dtypes,
     onlyCPU, skipCUDAIf, skipCUDAIfNoMagma, skipCPUIfNoLapack, precisionOverride,
     skipCUDAIfNoMagmaAndNoCusolver, skipCUDAIfRocm, onlyOnCPUAndCUDA, dtypesIfCUDA,
     onlyCUDA)
from torch.testing import floating_and_complex_types, floating_types, all_types
from torch.testing._internal.common_cuda import SM53OrLater, tf32_on_and_off, CUDA11OrLater, CUDA9

# Protects against includes accidentally setting the default dtype
# NOTE: jit_metaprogramming_utils sets the default dtype to double!
torch.set_default_dtype(torch.float32)
assert torch.get_default_dtype() is torch.float32

if TEST_SCIPY:
    import scipy

class TestLinalg(TestCase):
    exact_dtype = True

    @dtypes(torch.float, torch.cfloat)
    @precisionOverride({torch.float: 1e-06, torch.cfloat: 1e-06})
    @tf32_on_and_off(5e-3)
    def test_inner(self, device, dtype):
        def check(a_sizes_, b_sizes_):
            for a_sizes, b_sizes in ((a_sizes_, b_sizes_), (b_sizes_, a_sizes_)):
                a = torch.randn(a_sizes, dtype=dtype, device=device)
                b = torch.randn(b_sizes, dtype=dtype, device=device)
                res = torch.inner(a, b)
                ref = np.inner(a.cpu().numpy(), b.cpu().numpy())
                self.assertEqual(res.cpu(), torch.from_numpy(np.array(ref)))
                out = torch.zeros_like(res)
                torch.inner(a, b, out=out)
                self.assertEqual(res, out)

        check([], [])                       # scalar x scalar
        check([], [0])                      # scalar x empty
        check([], [3])                      # scalar x 1D
        check([], [2, 3, 4])                # scalar x 3D

        check([0], [0])                     # empty x empty
        check([0], [2, 0])                  # empty x 2D

        check([2], [2])                     # 1D x 1D
        check([2], [3, 1, 2])               # 1D x 3D
        check([2], [3, 0, 2])               # 1D x 3D empty

        check([1, 2], [3, 2])               # 2D x 2D
        check([1, 2], [3, 4, 2])            # 2D x 3D
        check([2, 1, 3, 2], [1, 3, 2, 2])   # 4D x 4D

        # Test discontiguous input
        a = torch.randn(3, 2, device=device, dtype=dtype).transpose_(0, 1)
        b = torch.randn(4, 3, device=device, dtype=dtype)[::2, :]
        self.assertFalse(a.is_contiguous() or b.is_contiguous())
        self.assertEqual(a.inner(b).cpu().numpy(), np.inner(a.cpu().numpy(), b.cpu().numpy()))

        # Test error message
        with self.assertRaisesRegex(RuntimeError,
                                    r"inner\(\) the last dimension must match on both "
                                    r"input tensors but got shapes \[2, 3\] and \[2, 2\]"):
            torch.randn(2, 3, device=device, dtype=dtype).inner(torch.randn(2, 2, device=device, dtype=dtype))

    # Tests torch.outer, and its alias, torch.ger, vs. NumPy
    @precisionOverride({torch.bfloat16: 1e-1})
    @dtypes(*(torch.testing.get_all_dtypes()))
    def test_outer(self, device, dtype):
        def run_test_case(a, b):
            if dtype == torch.bfloat16:
                a_np = a.to(torch.double).cpu().numpy()
                b_np = b.to(torch.double).cpu().numpy()
            else:
                a_np = a.cpu().numpy()
                b_np = b.cpu().numpy()
            expected = np.outer(a_np, b_np)

            self.assertEqual(torch.outer(a, b), expected)
            self.assertEqual(torch.Tensor.outer(a, b), expected)

            self.assertEqual(torch.ger(a, b), expected)
            self.assertEqual(torch.Tensor.ger(a, b), expected)

            # test out variant
            out = torch.empty(a.size(0), b.size(0), device=device, dtype=dtype)
            torch.outer(a, b, out=out)
            self.assertEqual(out, expected)

            out = torch.empty(a.size(0), b.size(0), device=device, dtype=dtype)
            torch.ger(a, b, out=out)
            self.assertEqual(out, expected)

        a = torch.randn(50).to(device=device, dtype=dtype)
        b = torch.randn(50).to(device=device, dtype=dtype)
        run_test_case(a, b)

        # test 0 strided tensor
        zero_strided = torch.randn(1).to(device=device, dtype=dtype).expand(50)
        run_test_case(zero_strided, b)
        run_test_case(a, zero_strided)

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(torch.float32, torch.float64, torch.complex64, torch.complex128)
    def test_cholesky(self, device, dtype):
        from torch.testing._internal.common_utils import random_hermitian_pd_matrix

        def run_test(shape, batch, contiguous):
            A = random_hermitian_pd_matrix(shape, *batch, dtype=dtype, device=device)
            if A.numel() > 0 and not contiguous:
                A = A.transpose(-2, -1)
                self.assertFalse(A.is_contiguous())
            expected_L = np.linalg.cholesky(A.cpu().numpy())
            actual_L = torch.linalg.cholesky(A)

            # For fp32 individual entries in matrices can differ between PyTorch and NumPy
            # Let's compare the norms of matrices instead
            if A.numel() > 0 and dtype in [torch.float32, torch.complex64]:
                # axis is specified to calculate matrix norm for batched input
                expected_norm = np.linalg.norm(expected_L, ord=1, axis=(-2, -1))
                actual_norm = torch.linalg.norm(actual_L, ord=1, axis=(-2, -1))
                # Compare the norms with standard tolerances
                self.assertEqual(actual_norm, expected_norm)
                # and individual values with a higher tolerance
                self.assertEqual(actual_L, expected_L, atol=1e-2, rtol=1e-5)
            else:
                self.assertEqual(actual_L, expected_L)

        shapes = (0, 3, 5)
        batches = ((), (3, ), (2, 2))
        larger_input_case = [(100, (5, ), True)]
        for shape, batch, contiguous in list(itertools.product(shapes, batches, (True, False))) + larger_input_case:
            run_test(shape, batch, contiguous)

        # check the out= variant
        A = random_hermitian_pd_matrix(3, 3, dtype=dtype, device=device)
        out = torch.empty_like(A)
        ans = torch.linalg.cholesky(A, out=out)
        self.assertEqual(ans, out)
        expected = torch.linalg.cholesky(A)
        self.assertEqual(expected, out)

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(torch.float32, torch.float64, torch.complex64, torch.complex128)
    def test_cholesky_errors_and_warnings(self, device, dtype):
        from torch.testing._internal.common_utils import random_hermitian_pd_matrix

        # cholesky requires the input to be a square matrix or batch of square matrices
        A = torch.randn(2, 3, device=device, dtype=dtype)
        with self.assertRaisesRegex(RuntimeError, r'must be batches of square matrices'):
            torch.linalg.cholesky(A)
        A = torch.randn(2, 2, 3, device=device, dtype=dtype)
        with self.assertRaisesRegex(RuntimeError, r'must be batches of square matrices'):
            torch.linalg.cholesky(A)
        with self.assertRaisesRegex(np.linalg.LinAlgError, r'Last 2 dimensions of the array must be square'):
            np.linalg.cholesky(A.cpu().numpy())

        # cholesky requires the input to be at least 2 dimensional tensor
        A = torch.randn(2, device=device, dtype=dtype)
        with self.assertRaisesRegex(RuntimeError, r'must have at least 2 dimensions'):
            torch.linalg.cholesky(A)
        with self.assertRaisesRegex(np.linalg.LinAlgError,
                                    r'1-dimensional array given\. Array must be at least two-dimensional'):
            np.linalg.cholesky(A.cpu().numpy())

        # if the input matrix is singular, an error should be raised
        A = torch.eye(3, 3, dtype=dtype, device=device)
        A[-1, -1] = 0  # Now A is singular
        with self.assertRaisesRegex(RuntimeError, r'U\(3,3\) is zero, singular U\.'):
            torch.linalg.cholesky(A)
        with self.assertRaisesRegex(np.linalg.LinAlgError, r'Matrix is not positive definite'):
            np.linalg.cholesky(A.cpu().numpy())

        # if at least one matrix in the batch is singular, an error should be raised
        A = torch.eye(3, 3, dtype=dtype, device=device)
        A = A.reshape((1, 3, 3))
        A = A.repeat(5, 1, 1)
        A[4, -1, -1] = 0  # Now A[4] is singular
        with self.assertRaisesRegex(RuntimeError, r'For batch 4: U\(3,3\) is zero, singular U\.'):
            torch.linalg.cholesky(A)

        # if out tensor with wrong shape is passed a warning is given
        A = random_hermitian_pd_matrix(3, dtype=dtype, device=device)
        out = torch.empty(2, 3, dtype=dtype, device=device)
        with warnings.catch_warnings(record=True) as w:
            # Trigger warning
            torch.linalg.cholesky(A, out=out)
            # Check warning occurs
            self.assertEqual(len(w), 1)
            self.assertTrue("An output with one or more elements was resized" in str(w[-1].message))

        # dtypes should match
        out = torch.empty_like(A).to(torch.int)
        with self.assertRaisesRegex(RuntimeError, "result dtype Int does not match self dtype"):
            torch.linalg.cholesky(A, out=out)

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(torch.float64, torch.complex128)
    def test_cholesky_autograd(self, device, dtype):
        def func(root):
            x = 0.5 * (root + root.transpose(-1, -2).conj())
            return torch.linalg.cholesky(x)

        def run_test(shape):
            root = torch.rand(*shape, dtype=dtype, device=device, requires_grad=True)
            root = root + torch.eye(shape[-1], dtype=dtype, device=device)

            gradcheck(func, root)
            gradgradcheck(func, root)

            root = torch.rand(*shape, dtype=dtype, device=device)
            root = torch.matmul(root, root.transpose(-1, -2).conj())
            root.requires_grad_()
            chol = torch.linalg.cholesky(root).sum().backward()
            self.assertEqual(root.grad, root.grad.transpose(-1, -2).conj())  # Check the gradient is hermitian

        shapes = ((3, 3), (4, 3, 2, 2))
        for shape in shapes:
            run_test(shape)

    # NOTE: old_cholesky* tests were moved here from test_torch.py and test_autograd.py
    @slowTest
    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(torch.double)
    def test_old_cholesky_batched_many_batches(self, device, dtype):
        from torch.testing._internal.common_utils import random_symmetric_pd_matrix

        def cholesky_test_helper(n, batchsize, device, upper):
            A = random_symmetric_pd_matrix(n, batchsize, dtype=dtype, device=device)
            chol_fact = torch.cholesky(A, upper=upper)
            if upper:
                # Correctness check
                self.assertEqual(A, chol_fact.transpose(-2, -1).matmul(chol_fact))
                # Upper triangular check
                self.assertEqual(chol_fact, chol_fact.triu())
            else:
                # Correctness check
                self.assertEqual(A, chol_fact.matmul(chol_fact.transpose(-2, -1)))
                # Lower triangular check
                self.assertEqual(chol_fact, chol_fact.tril())

        for upper, batchsize in itertools.product([True, False], [262144, 524288]):
            cholesky_test_helper(2, batchsize, device, upper)

    @precisionOverride({torch.float32: 1e-4, torch.complex64: 1e-4})
    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(torch.float32, torch.float64, torch.complex64, torch.complex128)
    def test_old_cholesky_batched(self, device, dtype):
        from torch.testing._internal.common_utils import random_hermitian_pd_matrix

        def cholesky_test_helper(n, batch_dims, upper):
            A = random_hermitian_pd_matrix(n, *batch_dims, dtype=dtype, device=device)
            cholesky_exp = torch.stack([m.cholesky(upper=upper) for m in A.reshape(-1, n, n)])
            cholesky_exp = cholesky_exp.reshape_as(A)
            self.assertEqual(cholesky_exp, torch.cholesky(A, upper=upper))

        for upper, batchsize in itertools.product([True, False], [(3,), (3, 4), (2, 3, 4)]):
            cholesky_test_helper(3, batchsize, upper)

    @precisionOverride({torch.float32: 1e-4, torch.complex64: 1e-4})
    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(torch.float32, torch.float64, torch.complex64, torch.complex128)
    @tf32_on_and_off(0.01)
    def test_old_cholesky(self, device, dtype):
        from torch.testing._internal.common_utils import random_hermitian_pd_matrix

        A = random_hermitian_pd_matrix(10, dtype=dtype, device=device)

        # default Case
        C = torch.cholesky(A)
        B = torch.mm(C, C.t().conj())
        self.assertEqual(A, B, atol=1e-14, rtol=0)

        # test Upper Triangular
        U = torch.cholesky(A, True)
        B = torch.mm(U.t().conj(), U)
        self.assertEqual(A, B, atol=1e-14, rtol=0, msg='cholesky (upper) did not allow rebuilding the original matrix')

        # test Lower Triangular
        L = torch.cholesky(A, False)
        B = torch.mm(L, L.t().conj())
        self.assertEqual(A, B, atol=1e-14, rtol=0, msg='cholesky (lower) did not allow rebuilding the original matrix')

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(torch.float32, torch.float64, torch.complex64, torch.complex128)
    def test_old_cholesky_empty(self, device, dtype):
        def run_test(upper):
            A = torch.empty(0, 0, dtype=dtype, device=device)
            chol = torch.cholesky(A, upper)
            chol_A = torch.matmul(chol, chol.t().conj())
            self.assertEqual(A, chol_A)
        for upper in [True, False]:
            run_test(upper)

    @onlyCPU
    @skipCPUIfNoLapack
    @dtypes(torch.float64, torch.complex128)
    def test_old_cholesky_autograd(self, device, dtype):
        def func(root, upper):
            x = 0.5 * (root + root.transpose(-1, -2).conj())
            return torch.cholesky(x, upper)

        def run_test(upper, dims):
            root = torch.rand(*dims, dtype=dtype, device=device, requires_grad=True)
            root = root + torch.eye(dims[-1])

            gradcheck(func, [root, upper])
            gradgradcheck(func, [root, upper])

            root = torch.rand(*dims, dtype=dtype, device=device)
            root = torch.matmul(root, root.transpose(-1, -2).conj())
            root.requires_grad_()
            chol = root.cholesky().sum().backward()
            self.assertEqual(root.grad, root.grad.transpose(-1, -2).conj())  # Check the gradient is hermitian

        for upper, dims in itertools.product([True, False], [(3, 3), (4, 3, 2, 2)]):
            run_test(upper, dims)

    def _test_addr_vs_numpy(self, device, dtype, beta=1, alpha=1):
        def check(m, a, b, beta, alpha):
            if dtype == torch.bfloat16:
                a_np = a.to(torch.double).cpu().numpy()
                b_np = b.to(torch.double).cpu().numpy()
                m_np = m.to(torch.double).cpu().numpy()
            else:
                a_np = a.cpu().numpy()
                b_np = b.cpu().numpy()
                m_np = m.cpu().numpy()
            if beta == 0:
                expected = alpha * np.outer(a_np, b_np)
            else:
                expected = beta * m_np + alpha * np.outer(a_np, b_np)

            res = torch.addr(m, a, b, beta=beta, alpha=alpha)
            self.assertEqual(res, expected)

            # Test out variant
            out = torch.empty_like(res)
            torch.addr(m, a, b, beta=beta, alpha=alpha, out=out)
            self.assertEqual(out, expected)

        m = make_tensor((50, 50), device=device, dtype=dtype, low=-2, high=2)
        a = make_tensor((50,), device=device, dtype=dtype, low=-2, high=2)
        b = make_tensor((50,), device=device, dtype=dtype, low=-2, high=2)

        check(m, a, b, beta, alpha)

        # test transpose
        m_transpose = torch.transpose(m, 0, 1)
        check(m_transpose, a, b, beta, alpha)

        # test 0 strided tensor
        zero_strided = make_tensor((1,), device=device, dtype=dtype, low=-2, high=2).expand(50)
        check(m, zero_strided, b, beta, alpha)

        # test scalar
        m_scalar = torch.tensor(1, device=device, dtype=dtype)
        check(m_scalar, a, b, beta, alpha)

        # test nans and infs are not propagated to the output when beta == 0
        float_and_complex_dtypes = torch.testing.get_all_fp_dtypes() + torch.testing.get_all_complex_dtypes()
        if beta == 0 and dtype in float_and_complex_dtypes:
            m[0][10] = m[10][10] = m[20][20] = float('inf')
            m[1][10] = m[11][10] = m[21][20] = float('nan')
        check(m, a, b, 0, alpha)

    @dtypes(torch.bool)
    def test_addr_bool(self, device, dtype):
        self._test_addr_vs_numpy(device, dtype, beta=True, alpha=False)
        self._test_addr_vs_numpy(device, dtype, beta=False, alpha=True)
        self._test_addr_vs_numpy(device, dtype, beta=False, alpha=False)
        self._test_addr_vs_numpy(device, dtype, beta=True, alpha=True)

    @dtypes(*(torch.testing.get_all_int_dtypes()))
    def test_addr_integral(self, device, dtype):
        with self.assertRaisesRegex(RuntimeError,
                                    'argument beta must not be a floating point number.'):
            self._test_addr_vs_numpy(device, dtype, beta=2., alpha=1)
        with self.assertRaisesRegex(RuntimeError,
                                    'argument alpha must not be a floating point number.'):
            self._test_addr_vs_numpy(device, dtype, beta=2, alpha=1.)
        with self.assertRaisesRegex(RuntimeError,
                                    'Boolean beta only supported for Boolean results.'):
            self._test_addr_vs_numpy(device, dtype, beta=True, alpha=1)
        with self.assertRaisesRegex(RuntimeError,
                                    'Boolean alpha only supported for Boolean results.'):
            self._test_addr_vs_numpy(device, dtype, beta=2, alpha=True)

        # when beta is zero
        self._test_addr_vs_numpy(device, dtype, beta=0, alpha=2)
        # when beta is not zero
        self._test_addr_vs_numpy(device, dtype, beta=2, alpha=2)

    @precisionOverride({torch.bfloat16: 1e-1})
    @dtypes(*(torch.testing.get_all_fp_dtypes() + torch.testing.get_all_complex_dtypes()))
    def test_addr_float_and_complex(self, device, dtype):
        with self.assertRaisesRegex(RuntimeError,
                                    'Boolean beta only supported for Boolean results.'):
            self._test_addr_vs_numpy(device, dtype, beta=True, alpha=1)
        with self.assertRaisesRegex(RuntimeError,
                                    'Boolean alpha only supported for Boolean results.'):
            self._test_addr_vs_numpy(device, dtype, beta=2, alpha=True)

        # when beta is zero
        self._test_addr_vs_numpy(device, dtype, beta=0., alpha=2)
        # when beta is not zero
        self._test_addr_vs_numpy(device, dtype, beta=0.5, alpha=2)
        if dtype in torch.testing.get_all_complex_dtypes():
            self._test_addr_vs_numpy(device, dtype, beta=(0 + 0.1j), alpha=(0.2 - 0.2j))

    @dtypes(*itertools.product(torch.testing.get_all_dtypes(),
                               torch.testing.get_all_dtypes()))
    def test_outer_type_promotion(self, device, dtypes):
        a = torch.randn(5).to(device=device, dtype=dtypes[0])
        b = torch.randn(5).to(device=device, dtype=dtypes[1])
        for op in (torch.outer, torch.Tensor.outer, torch.ger, torch.Tensor.ger):
            result = op(a, b)
            self.assertEqual(result.dtype, torch.result_type(a, b))

    @dtypes(*itertools.product(torch.testing.get_all_dtypes(),
                               torch.testing.get_all_dtypes(),
                               torch.testing.get_all_dtypes()))
    def test_addr_type_promotion(self, device, dtypes):
        a = make_tensor((5,), device=device, dtype=dtypes[0], low=-2, high=2)
        b = make_tensor((5,), device=device, dtype=dtypes[1], low=-2, high=2)
        m = make_tensor((5, 5), device=device, dtype=dtypes[2], low=-2, high=2)

        desired_dtype = torch.promote_types(torch.promote_types(dtypes[0], dtypes[1]),
                                            dtypes[2])
        for op in (torch.addr, torch.Tensor.addr):
            result = op(m, a, b)
            self.assertEqual(result.dtype, desired_dtype)

    # Tests migrated from test_torch.py
    # 1) test the shape of the result tensor when there is empty input tensor
    # 2) test the Runtime Exception when there is scalar input tensor
    def test_outer_ger_addr_legacy_tests(self, device):
        for size in ((0, 0), (0, 5), (5, 0)):
            a = torch.rand(size[0], device=device)
            b = torch.rand(size[1], device=device)

            self.assertEqual(torch.outer(a, b).shape, size)
            self.assertEqual(torch.ger(a, b).shape, size)

            m = torch.empty(size, device=device)
            self.assertEqual(torch.addr(m, a, b).shape, size)

        m = torch.randn(5, 6, device=device)
        a = torch.randn(5, device=device)
        b = torch.tensor(6, device=device)
        self.assertRaises(RuntimeError, lambda: torch.outer(a, b))
        self.assertRaises(RuntimeError, lambda: torch.outer(b, a))
        self.assertRaises(RuntimeError, lambda: torch.ger(a, b))
        self.assertRaises(RuntimeError, lambda: torch.ger(b, a))
        self.assertRaises(RuntimeError, lambda: torch.addr(m, a, b))
        self.assertRaises(RuntimeError, lambda: torch.addr(m, b, a))

    # Tests torch.det and its alias, torch.linalg.det, vs. NumPy
    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(torch.double, torch.cdouble)
    # NOTE: This test, and many others in this file that use magma, are currently skipped for ROCm.
    # See: https://github.com/pytorch/pytorch/issues/51303
    @skipCUDAIfRocm
    def test_det(self, device, dtype):
        tensors = (
            torch.randn((2, 2), device=device, dtype=dtype),
            torch.randn((129, 129), device=device, dtype=dtype),
            torch.randn((3, 52, 52), device=device, dtype=dtype),
            torch.randn((4, 2, 26, 26), device=device, dtype=dtype))


        ops = (torch.det, torch.Tensor.det,
               torch.linalg.det)
        for t in tensors:
            expected = np.linalg.det(t.cpu().numpy())
            for op in ops:
                actual = op(t)
                self.assertEqual(actual, expected)
                self.compare_with_numpy(op, np.linalg.det, t)

        # NOTE: det requires a 2D+ tensor
        t = torch.randn(1, device=device, dtype=dtype)
        with self.assertRaises(RuntimeError):
            op(t)

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(torch.float32, torch.float64, torch.complex64, torch.complex128)
    @precisionOverride({torch.float32: 1e-4, torch.complex64: 1e-4})
    def test_eigh(self, device, dtype):
        from torch.testing._internal.common_utils import random_hermitian_matrix

        def run_test(shape, batch, uplo):
            matrix = random_hermitian_matrix(shape, *batch, dtype=dtype, device=device)
            expected_w, expected_v = np.linalg.eigh(matrix.cpu().numpy(), UPLO=uplo)
            actual_w, actual_v = torch.linalg.eigh(matrix, UPLO=uplo)
            self.assertEqual(actual_w, expected_w)
            # sign of eigenvectors is not unique and therefore absolute values are compared
            self.assertEqual(abs(actual_v), abs(expected_v))
            # additionally we can flip the sign and then compare the values
            # let's choose the convention that the first element of the eigenvector should be positive,
            # otherwise flip the sign of the eigenvector
            if matrix.numel() > 0:
                sign = np.sign(expected_v[..., 0, :]).reshape(batch + (1, shape))
                expected_v = sign * expected_v
                torch_real_slice = actual_v[..., 0, :].real if dtype.is_complex else actual_v[..., 0, :]
                sign = torch.sign(torch_real_slice).reshape(batch + (1, shape))
                actual_v = sign * actual_v
                self.assertEqual(actual_v, expected_v)

            # check the out= variant
            out_w = torch.empty_like(actual_w)
            out_v = torch.empty_like(actual_v)
            ans_w, ans_v = torch.linalg.eigh(matrix, UPLO=uplo, out=(out_w, out_v))
            self.assertEqual(ans_w, out_w)
            self.assertEqual(ans_v, out_v)
            self.assertEqual(ans_w, actual_w)
            self.assertEqual(abs(ans_v), abs(actual_v))

        shapes = (0, 3, 5)
        batches = ((), (3, ), (2, 2))
        uplos = ["U", "L"]
        for shape, batch, uplo in itertools.product(shapes, batches, uplos):
            run_test(shape, batch, uplo)

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(torch.float32, torch.float64, torch.complex64, torch.complex128)
    @precisionOverride({torch.float32: 1e-4, torch.complex64: 1e-4})
    def test_eigh_lower_uplo(self, device, dtype):
        def run_test(shape, batch, uplo):
            # check lower case uplo
            # use non-symmetric input to check whether uplo argument is working as intended
            matrix = torch.randn(shape, shape, *batch, dtype=dtype, device=device)
            expected_w, expected_v = np.linalg.eigh(matrix.cpu().numpy(), UPLO=uplo)
            actual_w, actual_v = torch.linalg.eigh(matrix, UPLO=uplo)
            self.assertEqual(actual_w, expected_w)
            self.assertEqual(abs(actual_v), abs(expected_v))

        uplos = ["u", "l"]
        for uplo in uplos:
            run_test(3, (2, 2), uplo)

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(torch.float32, torch.float64, torch.complex64, torch.complex128)
    def test_eigh_errors_and_warnings(self, device, dtype):
        from torch.testing._internal.common_utils import random_hermitian_matrix

        # eigh requires a square matrix
        t = torch.randn(2, 3, device=device, dtype=dtype)
        with self.assertRaisesRegex(RuntimeError, "must be batches of square matrices"):
            torch.linalg.eigh(t)

        # eigh requires 'uplo' parameter to be 'U' or 'L'
        t = torch.randn(3, 3, device=device, dtype=dtype)
        for uplo in ["a", "wrong"]:
            with self.assertRaisesRegex(RuntimeError, "be \'L\' or \'U\'"):
                torch.linalg.eigh(t, UPLO=uplo)
            with self.assertRaisesRegex(ValueError, "be \'L\' or \'U\'"):
                np.linalg.eigh(t.cpu().numpy(), UPLO=uplo)

        # if non-empty out tensor with wrong shape is passed a warning is given
        a = random_hermitian_matrix(3, dtype=dtype, device=device)
        real_dtype = a.real.dtype if dtype.is_complex else dtype
        out_w = torch.empty(7, 7, dtype=real_dtype, device=device)
        out_v = torch.empty(7, 7, dtype=dtype, device=device)
        with warnings.catch_warnings(record=True) as w:
            # Trigger warning
            torch.linalg.eigh(a, out=(out_w, out_v))
            # Check warning occurs
            self.assertEqual(len(w), 2)
            self.assertTrue("An output with one or more elements was resized" in str(w[-2].message))
            self.assertTrue("An output with one or more elements was resized" in str(w[-1].message))

        # dtypes should match
        out_w = torch.empty_like(a).to(torch.int)
        out_v = torch.empty_like(a)
        with self.assertRaisesRegex(RuntimeError, "dtype Int does not match self dtype"):
            torch.linalg.eigh(a, out=(out_w, out_v))

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(torch.float32, torch.float64, torch.complex64, torch.complex128)
    @precisionOverride({torch.float32: 1e-4, torch.complex64: 1e-4})
    def test_eigh_non_contiguous(self, device, dtype):
        from torch.testing._internal.common_utils import random_hermitian_matrix

        def run_test(matrix, uplo):
            self.assertFalse(matrix.is_contiguous())
            expected_w, expected_v = np.linalg.eigh(matrix.cpu().numpy(), UPLO=uplo)
            actual_w, actual_v = torch.linalg.eigh(matrix, UPLO=uplo)
            self.assertEqual(actual_w, expected_w)
            # sign of eigenvectors is not unique and therefore absolute values are compared
            self.assertEqual(abs(actual_v), abs(expected_v))

        def run_test_permuted(shape, batch, uplo):
            # check for permuted / transposed inputs
            matrix = random_hermitian_matrix(shape, *batch, dtype=dtype, device=device)
            matrix = matrix.transpose(-2, -1)
            run_test(matrix, uplo)

        def run_test_skipped_elements(shape, batch, uplo):
            # check for inputs with skipped elements
            matrix = random_hermitian_matrix(shape, *batch, dtype=dtype, device=device)
            matrix = matrix[::2]
            run_test(matrix, uplo)

        shapes = (3, 5)
        batches = ((4, ), (4, 2))
        uplos = ["U", "L"]
        for shape, batch, uplo in itertools.product(shapes, batches, uplos):
            run_test_permuted(shape, batch, uplo)
            run_test_skipped_elements(shape, batch, uplo)

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(torch.float64, torch.complex128)
    def test_eigh_autograd(self, device, dtype):
        from torch.testing._internal.common_utils import random_hermitian_matrix

        def func(x, uplo):
            x = 0.5 * (x + x.conj().transpose(-2, -1))
            return torch.linalg.eigh(x, UPLO=uplo)

        def func_grad_w(x, uplo):
            return func(x, uplo)[0]

        def func_grad_v(x, uplo):
            # gauge invariant loss function
            return abs(func(x, uplo)[1])

        def run_test(dims, uplo):
            x = torch.randn(*dims, dtype=dtype, device=device, requires_grad=True)

            gradcheck(func_grad_w, [x, uplo])
            gradgradcheck(func_grad_w, [x, uplo])

            gradcheck(func_grad_v, [x, uplo])
            gradgradcheck(func_grad_v, [x, uplo])

            x = random_hermitian_matrix(dims[-1], *dims[:-2]).requires_grad_()
            w, v = torch.linalg.eigh(x)
            (w.sum() + abs(v).sum()).backward()
            self.assertEqual(x.grad, x.grad.conj().transpose(-1, -2))  # Check the gradient is Hermitian

        for dims, uplo in itertools.product([(3, 3), (2, 3, 3)], ["L", "U"]):
            run_test(dims, uplo)

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(torch.float32, torch.float64, torch.complex64, torch.complex128)
    @precisionOverride({torch.float32: 1e-4, torch.complex64: 1e-4})
    def test_eigvalsh(self, device, dtype):
        from torch.testing._internal.common_utils import random_hermitian_matrix

        def run_test(shape, batch, uplo):
            matrix = random_hermitian_matrix(shape, *batch, dtype=dtype, device=device)
            expected_w = np.linalg.eigvalsh(matrix.cpu().numpy(), UPLO=uplo)
            actual_w = torch.linalg.eigvalsh(matrix, UPLO=uplo)
            self.assertEqual(actual_w, expected_w)

            # check the out= variant
            out = torch.empty_like(actual_w)
            ans = torch.linalg.eigvalsh(matrix, UPLO=uplo, out=out)
            self.assertEqual(ans, out)
            self.assertEqual(ans, actual_w)

        shapes = (0, 3, 5)
        batches = ((), (3, ), (2, 2))
        uplos = ["U", "L"]
        for shape, batch, uplo in itertools.product(shapes, batches, uplos):
            run_test(shape, batch, uplo)

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(torch.float32, torch.float64, torch.complex64, torch.complex128)
    def test_eigvalsh_errors_and_warnings(self, device, dtype):
        # eigvalsh requires a square matrix
        t = torch.randn(2, 3, device=device, dtype=dtype)
        with self.assertRaisesRegex(RuntimeError, "must be batches of square matrices"):
            torch.linalg.eigvalsh(t)

        # eigvalsh requires 'uplo' parameter to be 'U' or 'L'
        t = torch.randn(3, 3, device=device, dtype=dtype)
        for uplo in ["a", "wrong"]:
            with self.assertRaisesRegex(RuntimeError, "be \'L\' or \'U\'"):
                torch.linalg.eigvalsh(t, UPLO=uplo)
            with self.assertRaisesRegex(ValueError, "be \'L\' or \'U\'"):
                np.linalg.eigvalsh(t.cpu().numpy(), UPLO=uplo)

        # if non-empty out tensor with wrong shape is passed a warning is given
        real_dtype = t.real.dtype if dtype.is_complex else dtype
        out = torch.empty_like(t).to(real_dtype)
        with warnings.catch_warnings(record=True) as w:
            # Trigger warning
            torch.linalg.eigvalsh(t, out=out)
            # Check warning occurs
            self.assertEqual(len(w), 1)
            self.assertTrue("An output with one or more elements was resized" in str(w[-1].message))

        # dtypes should match
        out = torch.empty_like(t).to(torch.int)
        with self.assertRaisesRegex(RuntimeError, "result dtype Int does not match self dtype"):
            torch.linalg.eigvalsh(t, out=out)

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(torch.float32, torch.float64, torch.complex64, torch.complex128)
    @precisionOverride({torch.float32: 1e-4, torch.complex64: 1e-4})
    def test_eigvalsh_non_contiguous(self, device, dtype):
        from torch.testing._internal.common_utils import random_hermitian_matrix

        def run_test(matrix, uplo):
            self.assertFalse(matrix.is_contiguous())
            expected_w = np.linalg.eigvalsh(matrix.cpu().numpy(), UPLO=uplo)
            actual_w = torch.linalg.eigvalsh(matrix, UPLO=uplo)
            self.assertEqual(actual_w, expected_w)

        def run_test_permuted(shape, batch, uplo):
            # check for permuted / transposed inputs
            matrix = random_hermitian_matrix(shape, *batch, dtype=dtype, device=device)
            matrix = matrix.transpose(-2, -1)
            run_test(matrix, uplo)

        def run_test_skipped_elements(shape, batch, uplo):
            # check for inputs with skipped elements
            matrix = random_hermitian_matrix(shape, *batch, dtype=dtype, device=device)
            matrix = matrix[::2]
            run_test(matrix, uplo)

        shapes = (3, 5)
        batches = ((4, ), (4, 2))
        uplos = ["U", "L"]
        for shape, batch, uplo in itertools.product(shapes, batches, uplos):
            run_test_permuted(shape, batch, uplo)
            run_test_skipped_elements(shape, batch, uplo)

    @dtypes(torch.float32, torch.float64, torch.complex64, torch.complex128)
    def test_kron(self, device, dtype):

        def run_test_case(a_shape, b_shape):
            a = torch.rand(a_shape, dtype=dtype, device=device)
            b = torch.rand(b_shape, dtype=dtype, device=device)

            expected = np.kron(a.cpu().numpy(), b.cpu().numpy())
            result = torch.kron(a, b)
            self.assertEqual(result, expected)

            # check the out= variant
            out = torch.empty_like(result)
            ans = torch.kron(a, b, out=out)
            self.assertEqual(ans, out)
            self.assertEqual(ans, result)

        shapes = [(4,), (2, 2), (1, 2, 3), (1, 2, 3, 3)]
        for a_shape, b_shape in itertools.product(shapes, reversed(shapes)):
            run_test_case(a_shape, b_shape)

    @dtypes(torch.float32, torch.float64, torch.complex64, torch.complex128)
    def test_kron_non_contiguous(self, device, dtype):

        def run_test_transposed(a_shape, b_shape):
            # check for transposed case
            a = torch.rand(a_shape, dtype=dtype, device=device).transpose(-2, -1)
            b = torch.rand(b_shape, dtype=dtype, device=device).transpose(-2, -1)
            self.assertFalse(a.is_contiguous())
            self.assertFalse(b.is_contiguous())

            expected = np.kron(a.cpu().numpy(), b.cpu().numpy())
            result = torch.kron(a, b)
            self.assertEqual(result, expected)

            # check the out= variant
            out = torch.empty(result.transpose(-2, -1).shape, dtype=dtype, device=device).transpose(-2, -1)
            self.assertFalse(out.is_contiguous())
            ans = torch.kron(a, b, out=out)
            self.assertEqual(ans, out)
            self.assertEqual(ans, result)

        def run_test_skipped_elements(a_shape, b_shape):
            # check for transposed case
            a = torch.rand(2 * a_shape[0], *a_shape[1:], dtype=dtype, device=device)[::2]
            b = torch.rand(2 * b_shape[0], *b_shape[1:], dtype=dtype, device=device)[::2]
            self.assertFalse(a.is_contiguous())
            self.assertFalse(b.is_contiguous())

            expected = np.kron(a.cpu().numpy(), b.cpu().numpy())
            result = torch.kron(a, b)
            self.assertEqual(result, expected)

            # check the out= variant
            out = torch.empty(2 * result.shape[0], *result.shape[1:], dtype=dtype, device=device)[::2]
            self.assertFalse(out.is_contiguous())
            ans = torch.kron(a, b, out=out)
            self.assertEqual(ans, out)
            self.assertEqual(ans, result)

        shapes = [(2, 2), (2, 2, 3), (2, 2, 3, 3)]
        for a_shape, b_shape in itertools.product(shapes, reversed(shapes)):
            # run_test_transposed(a_shape, b_shape)
            run_test_skipped_elements(a_shape, b_shape)

        # Test that kron perserve memory format
        a = torch.randn(1, 2, 3, 4, dtype=dtype, device=device).contiguous(memory_format=torch.channels_last)
        b = torch.randn(1, 2, 3, 4, dtype=dtype, device=device).contiguous(memory_format=torch.channels_last)
        c = torch.kron(a, b)
        self.assertTrue(c.is_contiguous(memory_format=torch.channels_last))
        torch.kron(a, b, out=c)
        self.assertTrue(c.is_contiguous(memory_format=torch.channels_last))
        c = c.contiguous(memory_format=torch.contiguous_format)
        torch.kron(a, b, out=c)
        self.assertTrue(c.is_contiguous(memory_format=torch.contiguous_format))


    @dtypes(torch.float32, torch.float64, torch.complex64, torch.complex128)
    def test_kron_empty(self, device, dtype):

        def run_test_case(empty_shape):
            a = torch.eye(3, dtype=dtype, device=device)
            b = torch.empty(empty_shape, dtype=dtype, device=device)
            result = torch.kron(a, b)
            expected = np.kron(a.cpu().numpy(), b.cpu().numpy())
            self.assertEqual(result, expected)

            # NumPy doesn't work if the first argument is empty
            result = torch.kron(b, a)
            self.assertEqual(result.shape, expected.shape)

        empty_shapes = [(0,), (2, 0), (1, 0, 3)]
        for empty_shape in empty_shapes:
            run_test_case(empty_shape)

    @dtypes(torch.float32, torch.float64, torch.complex64, torch.complex128)
    def test_kron_errors_and_warnings(self, device, dtype):
        # if non-empty out tensor with wrong shape is passed a warning is given
        a = torch.eye(3, dtype=dtype, device=device)
        b = torch.ones((2, 2), dtype=dtype, device=device)
        out = torch.empty_like(a)
        with warnings.catch_warnings(record=True) as w:
            # Trigger warning
            torch.kron(a, b, out=out)
            # Check warning occurs
            self.assertEqual(len(w), 1)
            self.assertTrue("An output with one or more elements was resized" in str(w[-1].message))

        # dtypes should match
        out = torch.empty_like(a).to(torch.int)
        with self.assertRaisesRegex(RuntimeError, "can't be cast to the desired output type"):
            torch.kron(a, b, out=out)

    # This test confirms that torch.linalg.norm's dtype argument works
    # as expected, according to the function's documentation
    @skipCUDAIfNoMagma
    @skipCUDAIfRocm
    def test_norm_dtype(self, device):
        def run_test_case(input_size, ord, keepdim, from_dtype, to_dtype):
            # Determine the best dtype to use for comparisons between tensors
            # of two different types
            def get_compare_dtype(type0, type1):
                types_32bit_based = [torch.float, torch.cfloat]
                is_complex = type0.is_complex or type1.is_complex

                if type0 in types_32bit_based or type1 in types_32bit_based:
                    return torch.cfloat if is_complex else torch.float
                else:
                    return torch.cdouble if is_complex else torch.double

            compare_dtype = get_compare_dtype(from_dtype, to_dtype)

            def get_value_type(dtype):
                if dtype == torch.cfloat:
                    return torch.float
                elif dtype == torch.cdouble:
                    return torch.double
                elif dtype == torch.complex32:
                    return torch.float16
                else:
                    return dtype

            msg = (
                f'input_size={input_size}, ord={ord}, keepdim={keepdim}, '
                f'from_dtype={from_dtype}, to_dtype={to_dtype}')
            input = torch.randn(*input_size, dtype=from_dtype, device=device)
            result = torch.linalg.norm(input, ord, keepdim=keepdim)
            if from_dtype.is_complex:
                # By default, norm downgrades a complex input to the corresponding real number type
                self.assertEqual(result.dtype, get_value_type(from_dtype), msg=msg)
            else:
                self.assertEqual(result.dtype, from_dtype, msg=msg)

            result_out = torch.empty((), dtype=to_dtype, device=device)
            torch.linalg.norm(input, ord, keepdim=keepdim, out=result_out)
            self.assertEqual(result_out.dtype, to_dtype, msg=msg)
            self.assertEqual(result.to(compare_dtype), result_out.to(compare_dtype), msg=msg)

            result_with_dtype = torch.linalg.norm(input, ord, keepdim=keepdim, dtype=to_dtype)
            self.assertEqual(result_with_dtype.dtype, to_dtype, msg=msg)

            if from_dtype.is_complex:
                result_convert_first = torch.linalg.norm(input.to(to_dtype), ord, keepdim=keepdim)
                self.assertEqual(result_with_dtype.to(compare_dtype), result_convert_first.to(compare_dtype), msg=msg)
            else:
                self.assertEqual(result.to(compare_dtype), result_with_dtype.to(compare_dtype), msg=msg)

            result_out_with_dtype = torch.empty_like(result_with_dtype)
            torch.linalg.norm(input, ord, keepdim=keepdim, dtype=to_dtype, out=result_out_with_dtype)
            self.assertEqual(result_out_with_dtype.dtype, to_dtype, msg=msg)
            self.assertEqual(result_with_dtype, result_out_with_dtype, msg=msg)

        ord_vector = [0, 0.1, -0.1, 1, -1, 2, -2, 3, -3, 4.5, -4.5, inf, -inf, None]
        ord_matrix = ['fro', 'nuc', 1, -1, 2, -2, inf, -inf, None]
        S = 10
        test_cases = [
            ((S, ), ord_vector),
            ((S, S), ord_matrix),
        ]
        for keepdim in [True, False]:
            for input_size, ord_settings in test_cases:
                for ord in ord_settings:
                    dtypes = [torch.float, torch.double, torch.cfloat, torch.cdouble]
                    for from_dtype, to_dtype in itertools.product(dtypes, dtypes):
                        run_test_case(input_size, ord, keepdim, from_dtype, to_dtype)

        # Make sure that setting dtype != out.dtype raises an error
        dtype_pairs = [
            (torch.float, torch.double),
            (torch.double, torch.float),
            (torch.cfloat, torch.cdouble),
            (torch.cdouble, torch.cfloat),
        ]
        for keepdim in [True, False]:
            for input_size, ord_settings in test_cases:
                for ord in ord_settings:
                    for dtype, out_dtype in dtype_pairs:
                        input = torch.rand(*input_size)
                        result = torch.Tensor().to(out_dtype)
                        with self.assertRaisesRegex(RuntimeError, r'provided dtype must match dtype of result'):
                            torch.linalg.norm(input, ord=ord, keepdim=keepdim, dtype=dtype, out=result)

    # This test compares torch.linalg.norm and numpy.linalg.norm to ensure that
    # their vector norm results match
    @dtypes(torch.float, torch.double)
    def test_norm_vector(self, device, dtype):
        def run_test_case(input, p, dim, keepdim):
            result = torch.linalg.norm(input, ord, dim, keepdim)
            input_numpy = input.cpu().numpy()
            result_numpy = np.linalg.norm(input_numpy, ord, dim, keepdim)

            msg = f'input.size()={input.size()}, ord={ord}, dim={dim}, keepdim={keepdim}, dtype={dtype}'
            self.assertEqual(result, result_numpy, msg=msg)

            result_out = torch.empty_like(result)
            torch.linalg.norm(input, ord, dim, keepdim, out=result_out)
            self.assertEqual(result, result_out, msg=msg)

        ord_vector = [0, 1, -1, 2, -2, 3, -3, 4.5, -4.5, inf, -inf, None]
        S = 10
        test_cases = [
            # input size, p settings, dim
            ((S, ), ord_vector, None),
            ((S, ), ord_vector, 0),
            ((S, S, S), ord_vector, 0),
            ((S, S, S), ord_vector, 1),
            ((S, S, S), ord_vector, 2),
            ((S, S, S), ord_vector, -1),
            ((S, S, S), ord_vector, -2),
        ]
        L = 1_000_000
        if dtype == torch.double:
            test_cases.append(((L, ), ord_vector, None))
        for keepdim in [True, False]:
            for input_size, ord_settings, dim in test_cases:
                input = torch.randn(*input_size, dtype=dtype, device=device)
                for ord in ord_settings:
                    run_test_case(input, ord, dim, keepdim)

    # This test compares torch.linalg.norm and numpy.linalg.norm to ensure that
    # their matrix norm results match
    @skipCUDAIfNoMagma
    @dtypes(torch.float, torch.double)
    @precisionOverride({torch.float32: 2e-5})
    def test_norm_matrix(self, device, dtype):
        def run_test_case(input, p, dim, keepdim):
            result = torch.linalg.norm(input, ord, dim, keepdim)
            input_numpy = input.cpu().numpy()
            result_numpy = np.linalg.norm(input_numpy, ord, dim, keepdim)

            msg = f'input.size()={input.size()}, ord={ord}, dim={dim}, keepdim={keepdim}, dtype={dtype}'
            self.assertEqual(result, result_numpy, msg=msg)

            result_out = torch.empty_like(result)
            torch.linalg.norm(input, ord, dim, keepdim, out=result_out)
            self.assertEqual(result, result_out, msg=msg)

        ord_matrix = [1, -1, 2, -2, inf, -inf, 'nuc', 'fro', None]
        S = 10
        test_cases = [
            # input size, p settings, dim
            ((S, S), ord_matrix, None),
            ((S, S), ord_matrix, (0, 1)),
            ((S, S), ord_matrix, (1, 0)),
            ((S, S, S, S), ord_matrix, (2, 0)),
            ((S, S, S, S), ord_matrix, (-1, -2)),
            ((S, S, S, S), ord_matrix, (-1, -3)),
            ((S, S, S, S), ord_matrix, (-3, 2)),
        ]
        L = 1_000
        if dtype == torch.double:
            test_cases.append(((L, L), ord_matrix, None))
        for keepdim in [True, False]:
            for input_size, ord_settings, dim in test_cases:
                input = torch.randn(*input_size, dtype=dtype, device=device)
                for ord in ord_settings:
                    run_test_case(input, ord, dim, keepdim)

    @skipCPUIfNoLapack
    @skipCUDAIfNoMagma
    @dtypes(torch.float32, torch.float64, torch.complex64, torch.complex128)
    @precisionOverride({torch.float32: 1e-3})
    @skipCUDAIfRocm
    def test_cond(self, device, dtype):
        def run_test_case(input, p):
            result = torch.linalg.cond(input, p)
            result_numpy = np.linalg.cond(input.cpu().numpy(), p)
            self.assertEqual(result, result_numpy, rtol=1e-2, atol=self.precision)

            # test out= variant
            out = torch.empty_like(result)
            ans = torch.linalg.cond(input, p, out=out)
            self.assertEqual(ans, out)
            self.assertEqual(ans, result)

        norm_types = [1, -1, 2, -2, inf, -inf, 'fro', 'nuc', None]
        input_sizes = [(32, 32), (2, 3, 3, 3)]
        for input_size in input_sizes:
            input = torch.randn(*input_size, dtype=dtype, device=device)
            for p in norm_types:
                run_test_case(input, p)

        # test empty batch sizes
        input_sizes = [(0, 3, 3), (0, 2, 5, 5)]
        for input_size in input_sizes:
            input = torch.randn(*input_size, dtype=dtype, device=device)
            for p in norm_types:
                run_test_case(input, p)

        # test non-square input
        input_sizes = [(16, 32), (32, 16), (2, 3, 5, 3), (2, 3, 3, 5)]
        for input_size in input_sizes:
            input = torch.randn(*input_size, dtype=dtype, device=device)
            for p in [2, -2, None]:
                run_test_case(input, p)

        # test for singular input
        a = torch.eye(3, dtype=dtype, device=device)
        a[-1, -1] = 0  # make 'a' singular
        for p in norm_types:
            run_test_case(a, p)

        # test for 0x0 matrices. NumPy doesn't work for such input, we return 0
        input_sizes = [(0, 0), (2, 5, 0, 0)]
        for input_size in input_sizes:
            input = torch.randn(*input_size, dtype=dtype, device=device)
            for p in ['fro', 2]:
                expected_dtype = a.real.dtype if dtype.is_complex else dtype
                expected = torch.zeros(input_size[:-2], dtype=expected_dtype, device=device)
                actual = torch.linalg.cond(input, p)
                self.assertEqual(actual, expected)

    @skipCPUIfNoLapack
    @skipCUDAIfNoMagma
    @dtypes(torch.float32, torch.float64, torch.complex64, torch.complex128)
    @precisionOverride({torch.float32: 1e-3})
    def test_cond_errors_and_warnings(self, device, dtype):
        norm_types = [1, -1, 2, -2, inf, -inf, 'fro', 'nuc', None]

        # cond expects the input to be at least 2-dimensional
        a = torch.ones(3, dtype=dtype, device=device)
        for p in norm_types:
            with self.assertRaisesRegex(RuntimeError, r'supports matrices or batches of matrices'):
                torch.linalg.cond(a, p)

        # for some norm types cond expects the input to be square
        a = torch.ones(3, 2, dtype=dtype, device=device)
        norm_types = [1, -1, inf, -inf, 'fro', 'nuc']
        for p in norm_types:
            with self.assertRaisesRegex(RuntimeError, r'supports square matrices or batches of square matrices'):
                torch.linalg.cond(a, p)

        # if non-empty out tensor with wrong shape is passed a warning is given
        a = torch.ones((2, 2), dtype=dtype, device=device)
        for p in ['fro', 2]:
            real_dtype = a.real.dtype if dtype.is_complex else dtype
            out = torch.empty(a.shape, dtype=real_dtype, device=device)
            with warnings.catch_warnings(record=True) as w:
                # Trigger warning
                torch.linalg.cond(a, p, out=out)
                # Check warning occurs
                self.assertEqual(len(w), 1)
                self.assertTrue("An output with one or more elements was resized" in str(w[-1].message))

        # dtypes should match
        out = torch.empty_like(a).to(torch.int)
        for p in ['fro', 2]:
            with self.assertRaisesRegex(RuntimeError, "result dtype Int does not match"):
                torch.linalg.cond(a, p, out=out)

        # for batched input if at least one matrix in the batch is not invertible,
        # we can't get the result for all other (possibly) invertible matrices in the batch without an explicit for loop.
        # this should change when at::inverse works with silent errors
        # NumPy works fine in this case because it's possible to silence the error and get the inverse matrix results
        # possibly filled with NANs
        batch_dim = 3
        a = torch.eye(3, 3, dtype=dtype, device=device)
        a = a.reshape((1, 3, 3))
        a = a.repeat(batch_dim, 1, 1)
        a[0, -1, -1] = 0  # now a[0] is singular
        for p in [1, -1, inf, -inf, 'fro', 'nuc']:
            with self.assertRaisesRegex(RuntimeError, "linalg_cond does not support yet"):
                torch.linalg.cond(a, p)

        # check invalid norm type
        a = torch.ones(3, 3, dtype=dtype, device=device)
        for p in ['wrong_norm', 5]:
            with self.assertRaisesRegex(RuntimeError, f"linalg_cond got an invalid norm type: {p}"):
                torch.linalg.cond(a, p)

    # This test calls torch.linalg.norm and numpy.linalg.norm with illegal arguments
    # to ensure that they both throw errors
    @dtypes(torch.float, torch.double)
    def test_norm_errors(self, device, dtype):
        def run_error_test_case(input, ord, dim, keepdim, error_type, error_regex):
            test_case_info = (
                f'test case input.size()={input.size()}, ord={ord}, dim={dim}, '
                f'keepdim={keepdim}, dtype={dtype}')

            with self.assertRaisesRegex(error_type, error_regex, msg=test_case_info):
                torch.linalg.norm(input, ord, dim, keepdim)

            input_numpy = input.cpu().numpy()

            msg = f'numpy does not raise error but pytorch does, for case "{test_case_info}"'
            with self.assertRaises(Exception, msg=test_case_info):
                np.linalg.norm(input_numpy, ord, dim, keepdim)

        S = 10
        error_test_cases = [
            # input size, p settings, dim, error type, error regex
            ((S, ), ['fro'], None, RuntimeError, r'order "fro" can only be used if either len\(dim\) == 2'),
            ((S, ), ['nuc'], None, RuntimeError, r'order "nuc" can only be used if either len\(dim\) == 2'),
            ((S, S), [3.5], None, RuntimeError, r'Order 3.5 not supported for matrix norm'),
            ((S, S), [0], None, RuntimeError, r'Order 0 not supported for matrix norm'),
            ((S, S), ['nuc'], 0, RuntimeError, r'order "nuc" can only be used if either len\(dim\) == 2'),
            ((S, S), ['fro'], 0, RuntimeError, r'order "fro" can only be used if either len\(dim\) == 2'),
            ((S, S), ['nuc'], (0, 0), RuntimeError, r'duplicate or invalid dimensions'),
            ((S, S), ['fro', 0], (0, 0), RuntimeError, r'Expected dims to be different'),
            ((S, S), ['fro', 'nuc', 0], (0, 4), IndexError, r'Dimension out of range'),
            ((S, ), [0], (4, ), IndexError, r'Dimension out of range'),
            ((S, ), [None], (0, 0), RuntimeError, r'Expected dims to be different, got this instead'),
            ((S, S, S), [1], (0, 1, 2), RuntimeError, r"'dim' must specify 1 or 2 dimensions"),
            ((S, S, S), [1], None, RuntimeError, r"'dim' must specify 1 or 2 dimensions"),
            ((S, S), ['garbage'], (0, 1), RuntimeError, r'Invalid norm order: garbage'),
        ]
        for keepdim in [True, False]:
            for input_size, ord_settings, dim, error_type, error_regex in error_test_cases:
                input = torch.randn(*input_size, dtype=dtype, device=device)
                for ord in ord_settings:
                    run_error_test_case(input, ord, dim, keepdim, error_type, error_regex)

    # Test complex number inputs for linalg.norm
    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(torch.cfloat, torch.cdouble)
    @precisionOverride({torch.cfloat: 2e-4})
    def test_norm_complex(self, device, dtype):
        def gen_error_message(input_size, ord, keepdim, dim=None):
            return "complex norm failed for input size %s, ord=%s, keepdim=%s, dim=%s" % (
                input_size, ord, keepdim, dim)

        vector_ords = [None, 0, 1, 2, 3, inf, -1, -2, -3, -inf]
        matrix_ords = [None, 'fro', 'nuc', 1, 2, inf, -1, -2, -inf]

        # Test supported ords
        for keepdim in [False, True]:
            # vector norm
            x = torch.randn(25, device=device, dtype=dtype)
            xn = x.cpu().numpy()
            for ord in vector_ords:
                res = torch.linalg.norm(x, ord, keepdim=keepdim).cpu()
                expected = np.linalg.norm(xn, ord, keepdims=keepdim)
                msg = gen_error_message(x.size(), ord, keepdim)
                self.assertEqual(res.shape, expected.shape, msg=msg)
                self.assertEqual(res, expected, msg=msg)

                res_out = torch.Tensor().to(device)
                torch.linalg.norm(x, ord, keepdim=keepdim, out=res_out)
                self.assertEqual(res_out.shape, expected.shape, msg=msg)
                self.assertEqual(res_out.cpu(), expected, msg=msg)

            # matrix norm
            x = torch.randn(25, 25, device=device, dtype=dtype)
            xn = x.cpu().numpy()
            for ord in matrix_ords:
                res = torch.linalg.norm(x, ord, keepdim=keepdim).cpu()
                expected = np.linalg.norm(xn, ord, keepdims=keepdim)
                msg = gen_error_message(x.size(), ord, keepdim)
                self.assertEqual(res.shape, expected.shape, msg=msg)
                self.assertEqual(res, expected, msg=msg)

                res_out = torch.Tensor().to(device)
                torch.linalg.norm(x, ord, keepdim=keepdim, out=res_out)
                self.assertEqual(res_out.shape, expected.shape, msg=msg)
                self.assertEqual(res_out.cpu(), expected, msg=msg)

    # Test that linal.norm gives the same result as numpy when inputs
    # contain extreme values (inf, -inf, nan)
    @skipCUDAIf(True, r"GPU Test is blocking torch.svd https://github.com/pytorch/pytorch/pull/48436")
    @unittest.skipIf(IS_WINDOWS, "Skipped on Windows!")
    @unittest.skipIf(IS_MACOS, "Skipped on MacOS!")
    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    def test_norm_extreme_values(self, device):
        vector_ords = [0, 1, 2, 3, inf, -1, -2, -3, -inf]
        matrix_ords = ['fro', 'nuc', 1, 2, inf, -1, -2, -inf]
        vectors = []
        matrices = []
        for pair in itertools.product([inf, -inf, 0.0, nan, 1.0], repeat=2):
            vectors.append(list(pair))
            matrices.append([[pair[0], pair[1]]])
            matrices.append([[pair[0]], [pair[1]]])
        for vector in vectors:
            x = torch.tensor(vector).to(device)
            x_n = x.cpu().numpy()
            for ord in vector_ords:
                msg = f'ord={ord}, vector={vector}'
                result = torch.linalg.norm(x, ord=ord)
                result_n = np.linalg.norm(x_n, ord=ord)
                self.assertEqual(result, result_n, msg=msg)

        # TODO: Remove this function once the broken cases are fixed
        def is_broken_matrix_norm_case(ord, x):
            if self.device_type == 'cuda':
                if x.size() == torch.Size([1, 2]):
                    if ord in ['nuc', 2, -2] and isnan(x[0][0]) and x[0][1] == 1:
                        # These cases are broken because of an issue with svd
                        # https://github.com/pytorch/pytorch/issues/43567
                        return True
            return False

        for matrix in matrices:
            x = torch.tensor(matrix).to(device)
            x_n = x.cpu().numpy()
            for ord in matrix_ords:
                msg = f'ord={ord}, matrix={matrix}'
                result = torch.linalg.norm(x, ord=ord)
                result_n = np.linalg.norm(x_n, ord=ord)

                if is_broken_matrix_norm_case(ord, x):
                    continue
                else:
                    self.assertEqual(result, result_n, msg=msg)

    # Test degenerate shape results match numpy for linalg.norm vector norms
    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @unittest.skipIf(TEST_WITH_ASAN, "Skipped on ASAN since it checks for undefined behavior.")
    @dtypes(torch.float, torch.double, torch.cfloat, torch.cdouble)
    def test_norm_vector_degenerate_shapes(self, device, dtype):
        def run_test_case(input, ord, dim, keepdim, should_error):
            msg = f'input.size()={input.size()}, ord={ord}, dim={dim}, keepdim={keepdim}, dtype={dtype}'
            input_numpy = input.cpu().numpy()
            if should_error:
                with self.assertRaises(ValueError):
                    np.linalg.norm(input_numpy, ord, dim, keepdim)
                with self.assertRaises(RuntimeError):
                    torch.linalg.norm(input, ord, dim, keepdim)
            else:
                result_numpy = np.linalg.norm(input_numpy, ord, dim, keepdim)
                result = torch.linalg.norm(input, ord, dim, keepdim)
                self.assertEqual(result, result_numpy, msg=msg)

        ord_vector = [0, 0.5, 1, 2, 3, inf, -0.5, -1, -2, -3, -inf, None]
        S = 10
        test_cases = [
            # input size, p settings that cause error, dim
            ((0, ), [inf, -inf], None),
            ((0, S), [inf, -inf], 0),
            ((0, S), [], 1),
            ((S, 0), [], 0),
            ((S, 0), [inf, -inf], 1),
        ]
        for keepdim in [True, False]:
            for input_size, error_ords, dim in test_cases:
                input = torch.randn(*input_size, dtype=dtype, device=device)
                for ord in ord_vector:
                    run_test_case(input, ord, dim, keepdim, ord in error_ords)

    # Test degenerate shape results match numpy for linalg.norm matrix norms
    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(torch.float, torch.double, torch.cfloat, torch.cdouble)
    def test_norm_matrix_degenerate_shapes(self, device, dtype):
        def run_test_case(input, ord, dim, keepdim, should_error):
            msg = f'input.size()={input.size()}, ord={ord}, dim={dim}, keepdim={keepdim}, dtype={dtype}'
            input_numpy = input.cpu().numpy()
            if should_error:
                with self.assertRaises(ValueError):
                    np.linalg.norm(input_numpy, ord, dim, keepdim)
                with self.assertRaises(RuntimeError):
                    torch.linalg.norm(input, ord, dim, keepdim)
            else:
                result_numpy = np.linalg.norm(input_numpy, ord, dim, keepdim)
                result = torch.linalg.norm(input, ord, dim, keepdim)
                self.assertEqual(result, result_numpy, msg=msg)

        ord_matrix = ['fro', 'nuc', 1, 2, inf, -1, -2, -inf, None]
        S = 10
        test_cases = [
            # input size, p settings that cause error, dim
            ((0, 0), [1, 2, inf, -1, -2, -inf], None),
            ((0, S), [2, inf, -2, -inf], None),
            ((S, 0), [1, 2, -1, -2], None),
            ((S, S, 0), [], (0, 1)),
            ((1, S, 0), [], (0, 1)),
            ((0, 0, S), [1, 2, inf, -1, -2, -inf], (0, 1)),
            ((0, 0, S), [1, 2, inf, -1, -2, -inf], (1, 0)),
        ]
        for keepdim in [True, False]:
            for input_size, error_ords, dim in test_cases:
                input = torch.randn(*input_size, dtype=dtype, device=device)
                for ord in ord_matrix:
                    run_test_case(input, ord, dim, keepdim, ord in error_ords)

    def test_norm_fastpaths(self, device):
        x = torch.randn(3, 5, device=device)

        # slow path
        result = torch.linalg.norm(x, 4.5, 1)
        expected = torch.pow(x.abs().pow(4.5).sum(1), 1.0 / 4.5)
        self.assertEqual(result, expected)

        # fast 0-norm
        result = torch.linalg.norm(x, 0, 1)
        expected = (x != 0).type_as(x).sum(1)
        self.assertEqual(result, expected)

        # fast 1-norm
        result = torch.linalg.norm(x, 1, 1)
        expected = x.abs().sum(1)
        self.assertEqual(result, expected)

        # fast 2-norm
        result = torch.linalg.norm(x, 2, 1)
        expected = torch.sqrt(x.pow(2).sum(1))
        self.assertEqual(result, expected)

        # fast 3-norm
        result = torch.linalg.norm(x, 3, 1)
        expected = torch.pow(x.pow(3).abs().sum(1), 1.0 / 3.0)
        self.assertEqual(result, expected)

    @skipCPUIfNoLapack
    @skipCUDAIfNoMagma
    @dtypes(*floating_and_complex_types())
    def test_eig_basic(self, device, dtype):
        a = torch.tensor([[1.96, 0.00, 0.00, 0.00, 0.00],
                          [-6.49, 3.80, 0.00, 0.00, 0.00],
                          [-0.47, -6.39, 4.17, 0.00, 0.00],
                          [-7.20, 1.50, -1.51, 5.70, 0.00],
                          [-0.65, -6.34, 2.67, 1.80, -7.10]],
                         dtype=dtype, device=device).t()
        e = torch.eig(a)[0]
        ee, vv = torch.eig(a, True)
        te = torch.tensor((), dtype=dtype, device=device)
        tv = torch.tensor((), dtype=dtype, device=device)
        eee, vvv = torch.eig(a, True, out=(te, tv))
        self.assertEqual(e, ee, atol=1e-12, rtol=0)
        self.assertEqual(ee, eee, atol=1e-12, rtol=0)
        self.assertEqual(ee, te, atol=1e-12, rtol=0)
        self.assertEqual(vv, vvv, atol=1e-12, rtol=0)
        self.assertEqual(vv, tv, atol=1e-12, rtol=0)
        #
        # compare with numpy
        np_e, np_v = np.linalg.eig(a.cpu().numpy())
        if dtype.is_complex:
            self.assertEqual(ee, np_e)
        else:
            # np_e.shape == (n, 2), where each column contain the real and
            # imaginary parts of the result
            self.assertEqual(ee[:, 0], np_e)  # real part
            self.assertEqual(ee[:, 1], torch.zeros(ee.shape[0], dtype=dtype))  # imaginary part
        self.assertEqual(vv, np_v)

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(torch.complex64, torch.complex128)
    def test_eig_backward_complex(self, device, dtype):
        # torch.eig's backward is not supported yet for complex types. We
        # should kill this test once it's implemented.
        a = torch.tensor([[1., 2], [3, 4]], device=device, dtype=dtype, requires_grad=True)
        with self.assertRaisesRegex(RuntimeError,
                                    "eig does not support automatic differentiation for outputs with complex dtype"):
            e, v = torch.eig(a, True)

    @skipCPUIfNoLapack
    @skipCUDAIfNoMagma
    @dtypes(torch.double, torch.float)
    def test_eig_reuse(self, device, dtype):
        X = torch.randn(4, 4, dtype=dtype, device=device)
        X = torch.mm(X.t(), X)
        e = torch.zeros(4, 2, dtype=dtype, device=device)
        v = torch.zeros(4, 4, dtype=dtype, device=device)
        torch.eig(X, True, out=(e, v))
        Xhat = np.matmul(np.matmul(v.cpu(), torch.diag(e.select(1, 0)).cpu()), v.t().cpu())
        if dtype is torch.float:
            atol = 1e-7
            rtol = 1e-5
        else:
            atol = 1e-8
            rtol = 0
        self.assertEqual(X, Xhat, atol=atol, rtol=rtol, msg='VeV\' wrong')
        self.assertTrue(v.is_contiguous(), 'V is not contiguous')

        torch.eig(X, True, out=(e, v))
        Xhat = np.matmul(v.cpu(), np.matmul(e.select(1, 0).diag().cpu(), v.t().cpu()))
        self.assertEqual(X, Xhat, atol=atol, rtol=rtol, msg='VeV\' wrong')
        self.assertTrue(v.is_contiguous(), 'V is not contiguous')

    @skipCPUIfNoLapack
    @skipCUDAIfNoMagma
    @dtypes(torch.double, torch.float)
    def test_eig_non_contiguous(self, device, dtype):
        X = torch.randn(4, 4, dtype=dtype, device=device)
        X = torch.mm(X.t(), X)
        e = torch.zeros(4, 2, 2, dtype=dtype, device=device)[:, 1]
        v = torch.zeros(4, 2, 4, dtype=dtype, device=device)[:, 1]
        self.assertFalse(v.is_contiguous(), 'V is contiguous')
        self.assertFalse(e.is_contiguous(), 'E is contiguous')
        torch.eig(X, True, out=(e, v))
        Xhat = np.matmul(np.matmul(v.cpu(), torch.diag(e.cpu().select(1, 0))), v.t().cpu())
        if dtype is torch.float:
            atol = 1e-7
            rtol = 1e-5
        else:
            atol = 1e-8
            rtol = 0
        self.assertEqual(X, Xhat, atol=atol, rtol=rtol, msg='VeV\' wrong')

    @skipCPUIfNoLapack
    @skipCUDAIfNoMagma
    @dtypes(torch.double, torch.float)
    def test_eig_invalid_input(self, device, dtype):
        # test invalid input
        self.assertRaisesRegex(
            RuntimeError,
            'input should be 2 dimensional',
            lambda: torch.eig(torch.ones((2))))
        self.assertRaisesRegex(
            RuntimeError,
            'input should be square',
            lambda: torch.eig(torch.ones((2, 3))))
        self.assertRaisesRegex(
            RuntimeError,
            'input should not contain infs or NaNs',
            lambda: torch.eig(np.inf * torch.ones((2, 2))))
        self.assertRaisesRegex(
            RuntimeError,
            'input should not contain infs or NaNs',
            lambda: torch.eig(np.nan * torch.ones((2, 2))))

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(torch.double, torch.float)
    def test_eig_out(self, device, dtype):
        # the out version of torch.eig needs to be tested manually: we can't
        # use the "test_out=True" parameter to tensor_op_tests because the
        # signature is irregular (since we have *two* output vectors)
        t = torch.randn(10, 10, dtype=dtype, device=device)
        evals, evecs = torch.eig(t, eigenvectors=True)
        #
        # check that the out= version computes the same values as the normal one
        out_evals = torch.empty_like(evals)
        out_evecs = torch.empty_like(evecs)
        evals2, evecs2 = torch.eig(t, eigenvectors=True, out=(out_evals, out_evecs))
        # check that the out tensors were used in-place
        self.assertEqual(evals2.data_ptr(), out_evals.data_ptr())
        self.assertEqual(evecs2.data_ptr(), out_evecs.data_ptr())
        # check that the result is the same as the non-out version
        self.assertEqual(evals, out_evals)
        self.assertEqual(evecs, out_evecs)
        #
        # check what happens in the eigenvectors=False case
        out_evals = torch.empty_like(evals)
        out_evecs = torch.tensor([1, 2, 3], dtype=dtype, device=device)
        evals2, evecs2 = torch.eig(t, eigenvectors=False, out=(out_evals, out_evecs))
        # check that the out_evals was used in-place
        self.assertEqual(evals2.data_ptr(), out_evals.data_ptr())
        self.assertEqual(evals, out_evals)
        # check that out_evecs was NOT touched at all
        assert out_evecs.tolist() == [1, 2, 3]
        #
        # check that we complain if we pass an out vector of the wrong dtype
        wrong_out = torch.empty((0, 0), dtype=int)
        with self.assertRaisesRegex(RuntimeError, r"Expected .* but got .*"):
            torch.eig(t, eigenvectors=True, out=(wrong_out, out_evecs))
        with self.assertRaisesRegex(RuntimeError, r"Expected .* but got .*"):
            torch.eig(t, eigenvectors=True, out=(out_evals, wrong_out))

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    def test_norm_old(self, device):
        def gen_error_message(input_size, p, keepdim, dim=None):
            return "norm failed for input size %s, p=%s, keepdim=%s, dim=%s" % (
                input_size, p, keepdim, dim)

        for keepdim in [False, True]:
            # full reduction
            x = torch.randn(25, device=device)
            xn = x.cpu().numpy()
            for p in [0, 1, 2, 3, 4, inf, -inf, -1, -2, -3, 1.5]:
                res = x.norm(p, keepdim=keepdim).cpu()
                expected = np.linalg.norm(xn, p, keepdims=keepdim)
                self.assertEqual(res, expected, atol=1e-5, rtol=0, msg=gen_error_message(x.size(), p, keepdim))

            # one dimension
            x = torch.randn(25, 25, device=device)
            xn = x.cpu().numpy()
            for p in [0, 1, 2, 3, 4, inf, -inf, -1, -2, -3]:
                dim = 1
                res = x.norm(p, dim, keepdim=keepdim).cpu()
                expected = np.linalg.norm(xn, p, dim, keepdims=keepdim)
                msg = gen_error_message(x.size(), p, keepdim, dim)
                self.assertEqual(res.shape, expected.shape, msg=msg)
                self.assertEqual(res, expected, msg=msg)

            # matrix norm
            for p in ['fro', 'nuc']:
                res = x.norm(p, keepdim=keepdim).cpu()
                expected = np.linalg.norm(xn, p, keepdims=keepdim)
                msg = gen_error_message(x.size(), p, keepdim)
                self.assertEqual(res.shape, expected.shape, msg=msg)
                self.assertEqual(res, expected, msg=msg)

            # zero dimensions
            x = torch.randn((), device=device)
            xn = x.cpu().numpy()
            res = x.norm(keepdim=keepdim).cpu()
            expected = np.linalg.norm(xn, keepdims=keepdim)
            msg = gen_error_message(x.size(), None, keepdim)
            self.assertEqual(res.shape, expected.shape, msg=msg)
            self.assertEqual(res, expected, msg=msg)

            # larger tensor sanity check
            self.assertEqual(
                2 * torch.norm(torch.ones(10000), keepdim=keepdim),
                torch.norm(torch.ones(40000), keepdim=keepdim))

            # matrix norm with non-square >2-D tensors, all combinations of reduction dims
            x = torch.randn(5, 6, 7, 8, device=device)
            xn = x.cpu().numpy()
            for p in ['fro', 'nuc']:
                for dim in itertools.product(*[list(range(4))] * 2):
                    if dim[0] == dim[1]:
                        continue
                    res = x.norm(p=p, dim=dim, keepdim=keepdim).cpu()
                    expected = np.linalg.norm(xn, ord=p, axis=dim, keepdims=keepdim)
                    msg = gen_error_message(x.size(), p, keepdim, dim)
                    self.assertEqual(res.shape, expected.shape, msg=msg)
                    self.assertEqual(res, expected, msg=msg)

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    def test_norm_complex_old(self, device):
        def gen_error_message(input_size, p, keepdim, dim=None):
            return "complex norm failed for input size %s, p=%s, keepdim=%s, dim=%s" % (
                input_size, p, keepdim, dim)

        for keepdim in [False, True]:
            # vector norm
            x = torch.randn(25, device=device) + 1j * torch.randn(25, device=device)
            xn = x.cpu().numpy()
            for p in [0, 1, 2, 3, inf, -1, -2, -3, -inf]:
                res = x.norm(p, keepdim=keepdim).cpu()
                expected = np.linalg.norm(xn, p, keepdims=keepdim)
                msg = gen_error_message(x.size(), p, keepdim)
                self.assertEqual(res.shape, expected.shape, msg=msg)
                self.assertEqual(res, expected, msg=msg)

            # matrix norm
            x = torch.randn(25, 25, device=device) + 1j * torch.randn(25, 25, device=device)
            xn = x.cpu().numpy()
            for p in ['nuc', 'fro']:
                res = x.norm(p, keepdim=keepdim).cpu()
                expected = np.linalg.norm(xn, p, keepdims=keepdim)
                msg = gen_error_message(x.size(), p, keepdim)
                self.assertEqual(res.shape, expected.shape, msg=msg)
                self.assertEqual(res, expected, msg=msg, rtol=1.3e-6, atol=3e-4)

    # Ensure torch.norm with p='fro' and p=2 give the same results for mutually supported input combinations
    @dtypes(torch.float)
    @skipCUDAIfRocm
    def test_norm_fro_2_equivalence_old(self, device, dtype):
        input_sizes = [
            (0,),
            (10,),
            (0, 0),
            (4, 30),
            (0, 45),
            (100, 0),
            (45, 10, 23),
            (0, 23, 59),
            (23, 0, 37),
            (34, 58, 0),
            (0, 0, 348),
            (0, 3434, 0),
            (0, 0, 0),
            (5, 3, 8, 1, 3, 5)]

        for input_size in input_sizes:
            a = make_tensor(input_size, device, dtype, low=-9, high=9)

            # Try full reduction
            dim_settings = [None]

            # Try all possible 1-D reductions
            dim_settings += list(range(-a.dim(), a.dim()))

            def wrap_dim(dim, ndims):
                assert (dim < ndims) and (dim >= -ndims)
                if dim >= 0:
                    return dim
                else:
                    return dim + ndims

            # Try all possible 2-D reductions
            dim_settings += [
                (d0, d1) for d0, d1 in itertools.combinations(range(-a.dim(), a.dim()), 2)
                if wrap_dim(d0, a.dim()) != wrap_dim(d1, a.dim())]

            for dim in dim_settings:
                for keepdim in [True, False]:
                    a_norm_2 = torch.norm(a, p=2, dim=dim, keepdim=keepdim)
                    a_norm_fro = torch.norm(a, p='fro', dim=dim, keepdim=keepdim)
                    self.assertEqual(a_norm_fro, a_norm_2)

    @skipCUDAIfNoMagma
    def test_nuclear_norm_axes_small_brute_force_old(self, device):
        def check_single_nuclear_norm(x, axes):
            if self.device_type != 'cpu' and randrange(100) < 95:
                return  # too many cpu <==> device copies

            a = np.array(x.cpu(), copy=False)
            expected = np.linalg.norm(a, "nuc", axis=axes)

            ans = torch.norm(x, "nuc", dim=axes)
            self.assertTrue(ans.is_contiguous())
            self.assertEqual(ans.shape, expected.shape)
            self.assertEqual(ans.cpu(), expected, rtol=1e-02, atol=1e-03, equal_nan=True)

            out = torch.zeros(expected.shape, dtype=x.dtype, device=x.device)
            ans = torch.norm(x, "nuc", dim=axes, out=out)
            self.assertIs(ans, out)
            self.assertTrue(ans.is_contiguous())
            self.assertEqual(ans.shape, expected.shape)
            self.assertEqual(ans.cpu(), expected, rtol=1e-02, atol=1e-03, equal_nan=True)

        for n in range(1, 3):
            for m in range(1, 3):
                for axes in itertools.permutations([0, 1], 2):
                    # 2d, inner dimensions C
                    x = torch.randn(n, m, device=device)
                    check_single_nuclear_norm(x, axes)

                    # 2d, inner dimensions Fortran
                    x = torch.randn(m, n, device=device).transpose(-1, -2)
                    check_single_nuclear_norm(x, axes)

                    # 2d, inner dimensions non-contiguous
                    x = torch.randn(n, 2 * m, device=device)[:, ::2]
                    check_single_nuclear_norm(x, axes)

                    # 2d, all dimensions non-contiguous
                    x = torch.randn(7 * n, 2 * m, device=device)[::7, ::2]
                    check_single_nuclear_norm(x, axes)

                for o in range(1, 3):
                    for axes in itertools.permutations([0, 1, 2], 2):
                        # 3d, inner dimensions C
                        x = torch.randn(o, n, m, device=device)
                        check_single_nuclear_norm(x, axes)

                        # 3d, inner dimensions Fortran
                        x = torch.randn(o, m, n, device=device).transpose(-1, -2)
                        check_single_nuclear_norm(x, axes)

                        # 3d, inner dimensions non-contiguous
                        x = torch.randn(o, n, 2 * m, device=device)[:, :, ::2]
                        check_single_nuclear_norm(x, axes)

                        # 3d, all dimensions non-contiguous
                        x = torch.randn(7 * o, 5 * n, 2 * m, device=device)[::7, ::5, ::2]
                        check_single_nuclear_norm(x, axes)

                    for r in range(1, 3):
                        for axes in itertools.permutations([0, 1, 2, 3], 2):
                            # 4d, inner dimensions C
                            x = torch.randn(r, o, n, m, device=device)
                            check_single_nuclear_norm(x, axes)

                            # 4d, inner dimensions Fortran
                            x = torch.randn(r, o, n, m, device=device).transpose(-1, -2)
                            check_single_nuclear_norm(x, axes)

                            # 4d, inner dimensions non-contiguous
                            x = torch.randn(r, o, n, 2 * m, device=device)[:, :, :, ::2]
                            check_single_nuclear_norm(x, axes)

                            # 4d, all dimensions non-contiguous
                            x = torch.randn(7 * r, 5 * o, 11 * n, 2 * m, device=device)[::7, ::5, ::11, ::2]
                            check_single_nuclear_norm(x, axes)

    @skipCUDAIfNoMagma
    def test_nuclear_norm_exceptions_old(self, device):
        for lst in [], [1], [1, 2]:
            x = torch.tensor(lst, dtype=torch.double, device=device)
            for axes in (), (0,):
                self.assertRaises(RuntimeError, torch.norm, x, "nuc", axes)
            self.assertRaises(IndexError, torch.norm, x, "nuc", (0, 1))

        x = torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.double, device=device)
        self.assertRaisesRegex(RuntimeError, "duplicate or invalid", torch.norm, x, "nuc", (0, 0))
        self.assertRaisesRegex(IndexError, "Dimension out of range", torch.norm, x, "nuc", (0, 2))

    # ~~~ tests for torch.svd ~~~
    @skipCUDAIfNoMagmaAndNoCusolver
    @skipCPUIfNoLapack
    @dtypes(torch.double)
    def test_svd(self, device, dtype):
        def run_test(dims, some, compute_uv):
            x = torch.randn(*dims, dtype=dtype, device=device)
            outu = torch.empty(0, dtype=dtype, device=device)
            outs = torch.empty(0, dtype=dtype, device=device)
            outv = torch.empty(0, dtype=dtype, device=device)
            torch.svd(x, some=some, compute_uv=compute_uv, out=(outu, outs, outv))

            if compute_uv:
                if some:
                    x_recon = torch.matmul(outu, torch.matmul(outs.diag_embed(), outv.transpose(-2, -1)))
                    self.assertEqual(x, x_recon, atol=1e-8, rtol=0, msg='Incorrect reconstruction using U @ diag(S) @ V.T')
                else:
                    narrow_u = outu[..., :min(*dims[-2:])]
                    narrow_v = outv[..., :min(*dims[-2:])]
                    x_recon = torch.matmul(narrow_u, torch.matmul(outs.diag_embed(), narrow_v.transpose(-2, -1)))
                    self.assertEqual(x, x_recon, atol=1e-8, rtol=0, msg='Incorrect reconstruction using U @ diag(S) @ V.T')
            else:
                _, singvals, _ = torch.svd(x, compute_uv=True)
                self.assertEqual(singvals, outs, msg='Singular values mismatch')
                self.assertEqual(outu, torch.zeros_like(outu), msg='U not zero')
                self.assertEqual(outv, torch.zeros_like(outv), msg='V not zero')

            resu, ress, resv = torch.svd(x, some=some, compute_uv=compute_uv)
            self.assertEqual(resu, outu, msg='outputs of svd and svd with out differ')
            self.assertEqual(ress, outs, msg='outputs of svd and svd with out differ')
            self.assertEqual(resv, outv, msg='outputs of svd and svd with out differ')

            # test non-contiguous
            x = torch.randn(*dims, dtype=dtype, device=device)
            if x.numel() > 0:
                n_dim = len(dims)
                # Reverse the batch dimensions and the matrix dimensions and then concat them
                x = x.permute(tuple(range(n_dim - 3, -1, -1)) + (n_dim - 1, n_dim - 2))
                assert not x.is_contiguous(), "x is intentionally non-contiguous"
                resu, ress, resv = torch.svd(x, some=some, compute_uv=compute_uv)
                if compute_uv:
                    if some:
                        x_recon = torch.matmul(resu, torch.matmul(ress.diag_embed(), resv.transpose(-2, -1)))
                        self.assertEqual(x, x_recon, atol=1e-8, rtol=0, msg='Incorrect reconstruction using U @ diag(S) @ V.T')
                    else:
                        narrow_u = resu[..., :min(*dims[-2:])]
                        narrow_v = resv[..., :min(*dims[-2:])]
                        x_recon = torch.matmul(narrow_u, torch.matmul(ress.diag_embed(), narrow_v.transpose(-2, -1)))
                        self.assertEqual(x, x_recon, atol=1e-8, rtol=0, msg='Incorrect reconstruction using U @ diag(S) @ V.T')
                else:
                    _, singvals, _ = torch.svd(x, compute_uv=True)
                    self.assertEqual(singvals, ress, msg='Singular values mismatch')
                    self.assertEqual(resu, torch.zeros_like(resu), msg='U not zero')
                    self.assertEqual(resv, torch.zeros_like(resv), msg='V not zero')

        shapes = [(0, 0), (5, 0), (0, 5),  # empty matrices
                  (0, 0, 0), (0, 5, 5), (0, 5, 3),  # zero batch dimension
                  (3, 3), (5, 3, 3), (7, 5, 3, 3),  # square matrices
                  (7, 3), (5, 7, 3), (7, 5, 7, 3),  # fat matrices
                  (3, 7), (5, 3, 7), (7, 5, 3, 7)]  # thin matrices
        for dims, some, compute_uv in product(shapes, [True, False], [True, False]):
            run_test(dims, some, compute_uv)

    @skipCUDAIfNoMagmaAndNoCusolver
    @skipCPUIfNoLapack
    @dtypes(torch.float)
    def test_svd_no_singularvectors(self, device, dtype):
        for size in [(5, 5), (5, 20), (20, 5)]:
            a = torch.randn(*size, device=device, dtype=dtype)
            u, s_expect, v = torch.svd(a)
            u, s_actual, v = torch.svd(a, compute_uv=False)
            self.assertEqual(s_expect, s_actual, msg="Singular values don't match")

    @skipCUDAIfNoMagmaAndNoCusolver
    @skipCPUIfNoLapack
    @dtypes(torch.double)
    @skipCUDAIfRocm
    def test_svd_lowrank(self, device, dtype):
        from torch.testing._internal.common_utils import random_lowrank_matrix, random_sparse_matrix

        def run_subtest(actual_rank, matrix_size, batches, device, svd_lowrank, **options):
            density = options.pop('density', 1)
            if isinstance(matrix_size, int):
                rows = columns = matrix_size
            else:
                rows, columns = matrix_size
            if density == 1:
                a_input = random_lowrank_matrix(actual_rank, rows, columns, *batches, device=device, dtype=dtype)
                a = a_input
            else:
                assert batches == ()
                a_input = random_sparse_matrix(rows, columns, density, device=device, dtype=dtype)
                a = a_input.to_dense()

            q = min(*size)
            u, s, v = svd_lowrank(a_input, q=q, **options)

            # check if u, s, v is a SVD
            u, s, v = u[..., :q], s[..., :q], v[..., :q]
            A = u.matmul(s.diag_embed()).matmul(v.transpose(-2, -1))
            self.assertEqual(A, a)

            # check if svd_lowrank produces same singular values as torch.svd
            U, S, V = torch.svd(a)
            self.assertEqual(s.shape, S.shape)
            self.assertEqual(u.shape, U.shape)
            self.assertEqual(v.shape, V.shape)
            self.assertEqual(s, S)

            if density == 1:
                # actual_rank is known only for dense inputs
                #
                # check if pairs (u, U) and (v, V) span the same
                # subspaces, respectively
                u, s, v = u[..., :actual_rank], s[..., :actual_rank], v[..., :actual_rank]
                U, S, V = U[..., :actual_rank], S[..., :actual_rank], V[..., :actual_rank]
                self.assertEqual(u.transpose(-2, -1).matmul(U).det().abs(), torch.ones(batches, device=device, dtype=dtype))
                self.assertEqual(v.transpose(-2, -1).matmul(V).det().abs(), torch.ones(batches, device=device, dtype=dtype))

        all_batches = [(), (1,), (3,), (2, 3)]
        for actual_rank, size, all_batches in [
                (2, (17, 4), all_batches),
                (4, (17, 4), all_batches),
                (4, (17, 17), all_batches),
                (10, (100, 40), all_batches),
                (7, (1000, 1000), [()]),
        ]:
            # dense input
            for batches in all_batches:
                run_subtest(actual_rank, size, batches, device, torch.svd_lowrank)
                if size != size[::-1]:
                    run_subtest(actual_rank, size[::-1], batches, device, torch.svd_lowrank)

        # sparse input
        for size in [(17, 4), (4, 17), (17, 17), (100, 40), (40, 100), (1000, 1000)]:
            for density in [0.005, 0.1]:
                run_subtest(None, size, (), device, torch.svd_lowrank, density=density)

        # jitting support
        jitted = torch.jit.script(torch.svd_lowrank)
        actual_rank, size, batches = 2, (17, 4), ()
        run_subtest(actual_rank, size, batches, device, jitted)

    @skipCUDAIfNoMagmaAndNoCusolver
    @skipCPUIfNoLapack
    @dtypes(torch.cfloat)
    def test_svd_complex(self, device, dtype):
        # this test verifies that torch.svd really returns V and not V.conj()
        # see: https://github.com/pytorch/pytorch/issues/45821
        t = torch.randn((10, 10), dtype=dtype, device=device)
        U, S, V = torch.svd(t, some=False)
        # verify that t ≈ t2
        # t2 = U @ diag(S) @ Vᴴ
        # Vᴴ is the conjugate transpose of V
        t2 = U @ torch.diag(S).type(dtype) @ V.conj().T
        self.assertEqual(t, t2)

    def _test_svd_helper(self, shape, some, col_maj, device, dtype):
        cpu_tensor = torch.randn(shape, device='cpu').to(dtype)
        device_tensor = cpu_tensor.to(device=device)
        if col_maj:
            cpu_tensor = cpu_tensor.t()
            device_tensor = device_tensor.t()
        cpu_result = torch.svd(cpu_tensor, some=some)
        device_result = torch.svd(device_tensor, some=some)
        m = min(cpu_tensor.shape[-2:])
        # torch.svd returns torch.return_types.svd which is a tuple of (U, V, S).
        # - When some==False, U[..., m:] can be arbitrary.
        # - When some==True, U shape: [..., m], V shape: [m, m]
        # - Signs are not deterministic. If the sign of a column of U is changed
        #   then the corresponding column of the V has to be changed.
        # Thus here we only compare result[..., :m].abs() from CPU and device.
        for x, y in zip(cpu_result, device_result):
            self.assertEqual(x[..., :m].abs(), y[..., :m].abs(), atol=1e-5, rtol=0)

    @skipCUDAIfNoMagmaAndNoCusolver
    @skipCPUIfNoLapack
    @dtypes(*floating_and_complex_types())
    def test_svd_square(self, device, dtype):
        self._test_svd_helper((10, 10), True, False, device, dtype)

    @skipCUDAIfNoMagmaAndNoCusolver
    @skipCPUIfNoLapack
    @dtypes(*floating_types())
    def test_svd_square_col_maj(self, device, dtype):
        self._test_svd_helper((10, 10), True, True, device, dtype)

    @skipCUDAIfNoMagmaAndNoCusolver
    @skipCPUIfNoLapack
    @dtypes(*floating_types())
    def test_svd_tall_some(self, device, dtype):
        self._test_svd_helper((20, 5), True, False, device, dtype)

    @skipCUDAIfNoMagmaAndNoCusolver
    @skipCPUIfNoLapack
    @dtypes(*floating_types())
    def test_svd_tall_all(self, device, dtype):
        self._test_svd_helper((20, 5), False, False, device, dtype)

    @skipCUDAIfNoMagmaAndNoCusolver
    @skipCPUIfNoLapack
    @dtypes(*floating_types())
    def test_svd_tall_some_col_maj(self, device, dtype):
        self._test_svd_helper((5, 20), True, True, device, dtype)

    @skipCUDAIfNoMagmaAndNoCusolver
    @skipCPUIfNoLapack
    @dtypes(*floating_types())
    def test_svd_tall_all_col_maj(self, device, dtype):
        self._test_svd_helper((5, 20), False, True, device, dtype)

    # ~~~ tests for torch.linalg.svd ~~~
    @skipCUDAIfNoMagmaAndNoCusolver
    @skipCPUIfNoLapack
    @dtypes(torch.float, torch.double, torch.cfloat, torch.cdouble)
    def test_linalg_svd_compute_uv(self, device, dtype):
        """
        Test the default case, compute_uv=True. Here we have the very same behavior as
        numpy
        """
        t = torch.randn((10, 11), device=device, dtype=dtype)
        np_t = t.cpu().numpy()
        for full_matrices in (True, False):
            # check linalg.svd vs numpy
            expected = np.linalg.svd(np_t, full_matrices, compute_uv=True)
            actual = torch.linalg.svd(t, full_matrices, compute_uv=True)
            # sign/phase of the singular vectors is not unique and therefore absolute values are compared
            self.assertEqual(abs(actual[0]), abs(expected[0]))
            self.assertEqual(actual[1], expected[1])
            self.assertEqual(abs(actual[2]), abs(expected[2]))
            # check linalg.svd vs linalg.svd(out=...)
            out = (torch.empty_like(actual[0]),
                   torch.empty_like(actual[1]),
                   torch.empty_like(actual[2]))
            out2 = torch.linalg.svd(t, full_matrices, compute_uv=True, out=out)
            self.assertEqual(actual, out)
            self.assertEqual(actual, out2)

    @skipCUDAIfNoMagmaAndNoCusolver
    @skipCPUIfNoLapack
    @dtypes(torch.float, torch.double, torch.cfloat, torch.cdouble)
    def test_linalg_svd_no_compute_uv(self, device, dtype):
        """
        Test the compute_uv=False case. Here we have a different return type than
        numpy: numpy returns S, we return (empty, S, empty)
        """
        t = torch.randn((10, 11), device=device, dtype=dtype)
        np_t = t.cpu().numpy()

        def is_empty(x):
            return x.numel() == 0 and x.dtype == t.dtype and x.device == t.device

        for full_matrices in (True, False):
            # check linalg.svd vs numpy
            np_s = np.linalg.svd(np_t, full_matrices, compute_uv=False)
            USV = torch.linalg.svd(t, full_matrices, compute_uv=False)
            assert is_empty(USV.U)
            self.assertEqual(USV.S, np_s)
            assert is_empty(USV.V)
            # check linalg.svd vs linalg.svd(out=...)
            out = (torch.empty_like(USV.U), torch.empty_like(USV.S), torch.empty_like(USV.V))
            USV = torch.linalg.svd(t, full_matrices, compute_uv=False, out=out)
            assert USV.U is out[0]
            assert USV.S is out[1]
            assert USV.V is out[2]
            self.assertEqual(USV.S, np_s)

    @skipCUDAIfNoMagmaAndNoCusolver
    @skipCPUIfNoLapack
    @onlyCUDA
    @dtypes(torch.float)
    def test_linalg_svd_out_different_device(self, device, dtype):
        t = torch.randn(5, 7, device=device, dtype=dtype)  # this is on cuda
        u = torch.empty((5, 5), device='cpu', dtype=dtype)
        s = torch.empty((5,), device='cpu', dtype=dtype)
        v = torch.empty((7, 7), device='cpu', dtype=dtype)
        with self.assertRaisesRegex(RuntimeError, 'svd output tensor U is on the wrong device: expected cuda:.* got cpu'):
            torch.linalg.svd(t, out=(u, s, v))

    def cholesky_solve_test_helper(self, A_dims, b_dims, upper, device, dtype):
        from torch.testing._internal.common_utils import random_hermitian_pd_matrix

        b = torch.randn(*b_dims, dtype=dtype, device=device)
        A = random_hermitian_pd_matrix(*A_dims, dtype=dtype, device=device)
        L = torch.cholesky(A, upper=upper)
        return b, A, L

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(torch.float32, torch.float64, torch.complex64, torch.complex128)
    @precisionOverride({torch.float32: 1e-3, torch.complex64: 1e-3,
                        torch.float64: 1e-8, torch.complex128: 1e-8})
    def test_cholesky_solve(self, device, dtype):
        for (k, n), upper in itertools.product(zip([2, 3, 5], [3, 5, 7]), [True, False]):
            b, A, L = self.cholesky_solve_test_helper((n,), (n, k), upper, device, dtype)
            x = torch.cholesky_solve(b, L, upper=upper)
            self.assertEqual(b, np.matmul(A.cpu(), x.cpu()))

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(torch.float32, torch.float64, torch.complex64, torch.complex128)
    @precisionOverride({torch.float32: 1e-3, torch.complex64: 1e-3,
                        torch.float64: 1e-8, torch.complex128: 1e-8})
    def test_cholesky_solve_batched(self, device, dtype):
        def cholesky_solve_batch_helper(A_dims, b_dims, upper):
            b, A, L = self.cholesky_solve_test_helper(A_dims, b_dims, upper, device, dtype)
            x_exp_list = []
            for i in range(b_dims[0]):
                x_exp_list.append(torch.cholesky_solve(b[i], L[i], upper=upper))
            x_exp = torch.stack(x_exp_list)  # Stacked output
            x_act = torch.cholesky_solve(b, L, upper=upper)  # Actual output
            self.assertEqual(x_act, x_exp)  # Equality check
            Ax = np.matmul(A.cpu(), x_act.cpu())
            self.assertEqual(b, Ax)  # Correctness check

        for upper, batchsize in itertools.product([True, False], [1, 3, 4]):
            cholesky_solve_batch_helper((5, batchsize), (batchsize, 5, 10), upper)

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(torch.float32, torch.float64, torch.complex64, torch.complex128)
    def test_cholesky_solve_batched_non_contiguous(self, device, dtype):
        from numpy.linalg import solve
        from torch.testing._internal.common_utils import random_hermitian_pd_matrix

        for upper in [True, False]:
            A = random_hermitian_pd_matrix(2, 2, dtype=dtype, device='cpu')
            b = torch.randn(2, 2, 2, dtype=dtype, device='cpu')
            x_exp = solve(A.permute(0, 2, 1).numpy(), b.permute(2, 1, 0).numpy())
            A = A.to(device).permute(0, 2, 1)
            b = b.to(device).permute(2, 1, 0)
            assert not A.is_contiguous() and not b.is_contiguous(), "contiguous inputs"
            L = torch.cholesky(A, upper)
            x = torch.cholesky_solve(b, L, upper=upper)
            self.assertEqual(x, x_exp)

    @slowTest
    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(torch.float32, torch.float64, torch.complex64, torch.complex128)
    @precisionOverride({torch.float32: 1e-3, torch.complex64: 1e-3,
                        torch.float64: 1e-8, torch.complex128: 1e-8})
    def test_cholesky_solve_batched_many_batches(self, device, dtype):
        for A_dims, b_dims in zip([(5, 256, 256), (5,)], [(5, 10), (512, 512, 5, 10)]):
            for upper in [True, False]:
                b, A, L = self.cholesky_solve_test_helper(A_dims, b_dims, upper, device, dtype)
                x = torch.cholesky_solve(b, L, upper)
                Ax = torch.matmul(A, x)
                self.assertEqual(Ax, b.expand_as(Ax))

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(torch.float32, torch.float64, torch.complex64, torch.complex128)
    @precisionOverride({torch.float32: 1e-3, torch.complex64: 1e-3,
                        torch.float64: 1e-8, torch.complex128: 1e-8})
    def test_cholesky_solve_batched_broadcasting(self, device, dtype):
        from numpy.linalg import solve
        from torch.testing._internal.common_utils import random_hermitian_pd_matrix

        def run_test(A_dims, b_dims, upper):
            A_matrix_size = A_dims[-1]
            A_batch_dims = A_dims[:-2]
            A = random_hermitian_pd_matrix(A_matrix_size, *A_batch_dims,
                                           dtype=dtype, device='cpu')
            b = torch.randn(*b_dims, dtype=dtype, device='cpu')
            x_exp = torch.tensor(solve(A.numpy(), b.numpy()), dtype=dtype, device=device)
            A, b = A.to(dtype=dtype, device=device), b.to(dtype=dtype, device=device)
            L = torch.cholesky(A, upper)
            x = torch.cholesky_solve(b, L, upper=upper)
            self.assertEqual(x, x_exp)
            # https://github.com/pytorch/pytorch/issues/42695
            x = torch.cholesky_solve(b, L, upper=upper, out=x)
            self.assertEqual(x, x_exp)

        # test against numpy.linalg.solve
        for upper in [True, False]:
            run_test((2, 1, 3, 4, 4), (2, 1, 3, 4, 6), upper)  # no broadcasting
            run_test((2, 1, 3, 4, 4), (4, 6), upper)  # broadcasting b
            run_test((4, 4), (2, 1, 3, 4, 2), upper)  # broadcasting A
            run_test((1, 3, 1, 4, 4), (2, 1, 3, 4, 5), upper)  # broadcasting A & b

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(torch.float64, torch.complex128)
    def test_cholesky_solve_autograd(self, device, dtype):
        def run_test(A_dims, B_dims, upper):
            root = torch.randn(*A_dims, device=device, dtype=dtype).requires_grad_()
            b = torch.randn(*B_dims, device=device, dtype=dtype).requires_grad_()

            def func(root, b, upper):
                if upper:
                    A = root.triu()
                else:
                    A = root.tril()
                return torch.cholesky_solve(b, A, upper)

            gradcheck(func, [root, b, upper])
            # TODO(#50743): the following fails with batched grad testing
            gradgradcheck(func, [root, b, upper], atol=1e-3, check_batched_grad=False)

        for (a_size, b_size), upper in itertools.product([((3, 3), (3, 4)), ((3, 3), (3, 2)),
                                                          ((2, 3, 3), (2, 3, 4)), ((2, 3, 3), (2, 3, 2))],
                                                         [True, False]):
            run_test(a_size, b_size, upper)

    @skipCUDAIfNoMagmaAndNoCusolver
    @skipCPUIfNoLapack
    @dtypes(torch.float32, torch.float64, torch.complex64, torch.complex128)
    @precisionOverride({torch.float32: 2e-3, torch.complex64: 2e-3,
                        torch.float64: 1e-8, torch.complex128: 1e-8})
    @skipCUDAIfRocm
    def test_inverse(self, device, dtype):
        from torch.testing._internal.common_utils import random_fullrank_matrix_distinct_singular_value

        def run_test(torch_inverse, matrix, batches, n):
            matrix_inverse = torch_inverse(matrix)

            # Compare against NumPy output
            # NumPy uses 'gesv' LAPACK routine solving the equation A A_inv = I
            # But in PyTorch 'gertf' + 'getri' is used causing element-wise differences
            expected = np.linalg.inv(matrix.cpu().numpy())
            self.assertEqual(matrix_inverse, expected, atol=self.precision, rtol=self.precision)

            # Additional correctness tests, check matrix*matrix_inverse == identity
            identity = torch.eye(n, dtype=dtype, device=device)
            self.assertEqual(identity.expand_as(matrix), np.matmul(matrix.cpu(), matrix_inverse.cpu()))
            self.assertEqual(identity.expand_as(matrix), np.matmul(matrix_inverse.cpu(), matrix.cpu()))

            # check the out= variant
            # prepare the expected out tensor
            matrix_inverse_out = torch.empty(*batches, n, n, dtype=dtype, device=device)
            matrix_inverse_out_t = matrix_inverse_out.transpose(-2, -1).clone(memory_format=torch.contiguous_format)
            matrix_inverse_out = matrix_inverse_out_t.transpose(-2, -1)
            ans = torch_inverse(matrix, out=matrix_inverse_out)
            self.assertEqual(matrix_inverse_out, ans, atol=0, rtol=0)
            self.assertEqual(matrix_inverse_out, matrix_inverse, atol=0, rtol=0)

            # batched matrices: 3+ dimensional tensors, check matrix_inverse same as single-inverse for each matrix
            if matrix.ndim > 2 and batches[0] != 0:
                expected_inv_list = []
                p = int(np.prod(batches))  # use `p` instead of -1, so that the test works for empty input as well
                for mat in matrix.contiguous().view(p, n, n):
                    expected_inv_list.append(torch_inverse(mat))
                expected_inv = torch.stack(expected_inv_list).view(*batches, n, n)
                if self.device_type == 'cuda' and dtype in [torch.float32, torch.complex64]:
                    # single-inverse is done using cuSOLVER, while batched inverse is done using MAGMA
                    # individual values can be significantly different for fp32, hence rather high rtol is used
                    # the important thing is that torch_inverse passes above checks with identity
                    self.assertEqual(matrix_inverse, expected_inv, atol=1e-1, rtol=1e-2)
                else:
                    self.assertEqual(matrix_inverse, expected_inv)

        for torch_inverse in [torch.inverse, torch.linalg.inv]:
            for batches, n in itertools.product(
                [[], [0], [1], [2], [4], [2, 3]],
                [0, 5, 64]
            ):
                matrices = random_fullrank_matrix_distinct_singular_value(n, *batches, dtype=dtype).to(device)
                run_test(torch_inverse, matrices, batches, n)

                # test non-contiguous input
                run_test(torch_inverse, matrices.transpose(-2, -1), batches, n)
                if n > 0:
                    run_test(
                        torch_inverse,
                        random_fullrank_matrix_distinct_singular_value(n * 2, *batches, dtype=dtype).to(device)
                        .view(-1, n * 2, n * 2)[:, ::2, ::2].view(*batches, n, n),
                        batches, n
                    )

    @slowTest
    @skipCUDAIfNoMagmaAndNoCusolver
    @skipCPUIfNoLapack
    @dtypes(torch.float32, torch.float64, torch.complex64, torch.complex128)
    @precisionOverride({torch.float32: 2e-3, torch.complex64: 2e-3,
                        torch.float64: 1e-5, torch.complex128: 1e-5})
    def test_inverse_many_batches(self, device, dtype):
        from torch.testing._internal.common_utils import random_fullrank_matrix_distinct_singular_value

        def test_inverse_many_batches_helper(torch_inverse, b, n):
            matrices = random_fullrank_matrix_distinct_singular_value(b, n, n, dtype=dtype).to(device)
            matrices_inverse = torch_inverse(matrices)

            # Compare against NumPy output
            expected = np.linalg.inv(matrices.cpu().numpy())
            self.assertEqual(matrices_inverse, expected, atol=self.precision, rtol=1e-3)

        for torch_inverse in [torch.inverse, torch.linalg.inv]:
            test_inverse_many_batches_helper(torch_inverse, 5, 256)
            test_inverse_many_batches_helper(torch_inverse, 3, 512)
            test_inverse_many_batches_helper(torch_inverse, 64, 64)

    @skipCUDAIfNoMagmaAndNoCusolver
    @skipCPUIfNoLapack
    @onlyOnCPUAndCUDA   # TODO: XLA doesn't raise exception
    @dtypes(torch.float32, torch.float64, torch.complex64, torch.complex128)
    def test_inverse_errors(self, device, dtype):
        # inverse expects batches of square matrices as input
        with self.assertRaisesRegex(RuntimeError, "must be batches of square matrices"):
            torch.inverse(torch.randn(2, 3, 4, 3))

        # if input is not invertible, RuntimeError is raised mentioning the first non-invertible batch
        def run_test_singular_input(batch_dim, n):
            x = torch.eye(3, 3, dtype=dtype, device=device).reshape((1, 3, 3)).repeat(batch_dim, 1, 1)
            x[n, -1, -1] = 0
            with self.assertRaisesRegex(RuntimeError, rf'For batch {n}: U\(3,3\) is zero'):
                torch.inverse(x)

        for params in [(1, 0), (2, 0), (2, 1), (4, 0), (4, 2), (10, 2)]:
            run_test_singular_input(*params)

    @precisionOverride({torch.float32: 1e-3, torch.complex64: 1e-3, torch.float64: 1e-7, torch.complex128: 1e-7})
    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(torch.float32, torch.float64, torch.complex64, torch.complex128)
    def test_pinv(self, device, dtype):
        from torch.testing._internal.common_utils import random_hermitian_pd_matrix

        def run_test_main(A, hermitian):
            # Testing against definition for pseudo-inverses
            A_pinv = torch.linalg.pinv(A, hermitian=hermitian)
            np_A = A.cpu().numpy()
            np_A_pinv = A_pinv.cpu().numpy()
            if A.numel() > 0:
                self.assertEqual(A, np_A @ np_A_pinv @ np_A, atol=self.precision, rtol=self.precision)
                self.assertEqual(A_pinv, np_A_pinv @ np_A @ np_A_pinv, atol=self.precision, rtol=self.precision)
                self.assertEqual(np_A @ np_A_pinv, (np_A @ np_A_pinv).conj().swapaxes(-2, -1))
                self.assertEqual(np_A_pinv @ np_A, (np_A_pinv @ np_A).conj().swapaxes(-2, -1))
            else:
                self.assertEqual(A.shape, A_pinv.shape[:-2] + (A_pinv.shape[-1], A_pinv.shape[-2]))

            # Check out= variant
            out = torch.empty_like(A_pinv)
            ans = torch.linalg.pinv(A, hermitian=hermitian, out=out)
            self.assertEqual(ans, out)
            self.assertEqual(ans, A_pinv)

        def run_test_numpy(A, hermitian):
            # Check against NumPy output
            # Test float rcond, and specific value for each matrix
            rconds = [float(torch.rand(1)), ]
            # Test different types of rcond tensor
            for rcond_type in all_types():
                rconds.append(torch.rand(A.shape[:-2], dtype=torch.double, device=device).to(rcond_type))
            # Test broadcasting of rcond
            if A.ndim > 2:
                rconds.append(torch.rand(A.shape[-3], device=device))
            for rcond in rconds:
                actual = torch.linalg.pinv(A, rcond=rcond, hermitian=hermitian)
                numpy_rcond = rcond if isinstance(rcond, float) else rcond.cpu().numpy()
                expected = np.linalg.pinv(A.cpu().numpy(), rcond=numpy_rcond, hermitian=hermitian)
                self.assertEqual(actual, expected, atol=self.precision, rtol=1e-5)

        for sizes in [(5, 5), (3, 5, 5), (3, 2, 5, 5),  # square matrices
                      (3, 2), (5, 3, 2), (2, 5, 3, 2),  # fat matrices
                      (2, 3), (5, 2, 3), (2, 5, 2, 3),  # thin matrices
                      (0, 0), (0, 2), (2, 0), (3, 0, 0), (0, 3, 0), (0, 0, 3)]:  # zero numel matrices
            A = torch.randn(*sizes, dtype=dtype, device=device)
            hermitian = False
            run_test_main(A, hermitian)
            run_test_numpy(A, hermitian)

        # Check hermitian = True
        for sizes in [(5, 5), (3, 5, 5), (3, 2, 5, 5),  # square matrices
                      (0, 0), (3, 0, 0), ]:  # zero numel square matrices
            A = random_hermitian_pd_matrix(sizes[-1], *sizes[:-2], dtype=dtype, device=device)
            hermitian = True
            run_test_main(A, hermitian)
            run_test_numpy(A, hermitian)

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(torch.float32, torch.float64, torch.complex64, torch.complex128)
    def test_pinv_errors_and_warnings(self, device, dtype):
        # pinv requires at least 2D tensor
        a = torch.randn(1, device=device, dtype=dtype)
        with self.assertRaisesRegex(RuntimeError, "expected a tensor with 2 or more dimensions"):
            torch.linalg.pinv(a)

        # if non-empty out tensor with wrong shape is passed a warning is given
        a = torch.randn(3, 3, dtype=dtype, device=device)
        out = torch.empty(7, 7, dtype=dtype, device=device)
        with warnings.catch_warnings(record=True) as w:
            # Trigger warning
            torch.linalg.pinv(a, out=out)
            # Check warning occurs
            self.assertEqual(len(w), 1)
            self.assertTrue("An output with one or more elements was resized" in str(w[-1].message))

        # dtypes of out and input should match
        out = torch.empty_like(a).to(torch.int)
        with self.assertRaisesRegex(RuntimeError, "dtype Int does not match the expected dtype"):
            torch.linalg.pinv(a, out=out)

        if torch.cuda.is_available():
            # device of out and input should match
            wrong_device = 'cpu' if self.device_type != 'cpu' else 'cuda'
            out = torch.empty_like(a).to(wrong_device)
            with self.assertRaisesRegex(RuntimeError, "Expected result and input to be on the same device"):
                torch.linalg.pinv(a, out=out)

            # device of rcond and input should match
            wrong_device = 'cpu' if self.device_type != 'cpu' else 'cuda'
            rcond = torch.full((), 1e-2, device=wrong_device)
            with self.assertRaisesRegex(RuntimeError, "Expected rcond and input to be on the same device"):
                torch.linalg.pinv(a, rcond=rcond)

        # rcond can't be complex
        rcond = torch.full((), 1j, device=device)
        with self.assertRaisesRegex(RuntimeError, "rcond tensor of complex type is not supported"):
            torch.linalg.pinv(a, rcond=rcond)

    @skipCUDAIfNoMagmaAndNoCusolver
    @skipCPUIfNoLapack
    @dtypes(torch.float32, torch.float64, torch.complex64, torch.complex128)
    def test_inv_errors(self, device, dtype):
        # inv expects batches of square matrices as input
        a = torch.randn(2, 3, 4, 3, dtype=dtype, device=device)
        with self.assertRaisesRegex(RuntimeError, "must be batches of square matrices"):
            torch.linalg.inv(a)

        # inv requires the input to be at least 2 dimensional tensor
        a = torch.randn(2, device=device, dtype=dtype)
        with self.assertRaisesRegex(RuntimeError, "must have at least 2 dimensions"):
            torch.linalg.inv(a)

        # if input is not invertible, RuntimeError is raised mentioning the first non-invertible batch
        def run_test_singular_input(batch_dim, n):
            a = torch.eye(3, 3, dtype=dtype, device=device).reshape((1, 3, 3)).repeat(batch_dim, 1, 1)
            a[n, -1, -1] = 0
            with self.assertRaisesRegex(RuntimeError, rf"For batch {n}: U\(3,3\) is zero"):
                torch.linalg.inv(a)

        for params in [(1, 0), (2, 0), (2, 1), (4, 0), (4, 2), (10, 2)]:
            run_test_singular_input(*params)

        # if non-empty out tensor with wrong shape is passed an error is thrown
        a = torch.randn(2, 3, 3, device=device, dtype=dtype)
        out = torch.empty(1, device=device, dtype=dtype)
        with self.assertRaisesRegex(RuntimeError, "does not match input shape"):
            torch.linalg.inv(a, out=out)

        # dtypes should match
        out = torch.empty_like(a).to(torch.int)
        with self.assertRaisesRegex(RuntimeError, "result dtype Int does not match input dtype"):
            torch.linalg.inv(a, out=out)

        # device should match
        if torch.cuda.is_available():
            wrong_device = 'cpu' if self.device_type != 'cpu' else 'cuda'
            out = torch.empty(0, device=wrong_device, dtype=dtype)
            with self.assertRaisesRegex(RuntimeError, "does not match input device"):
                torch.linalg.inv(a, out=out)

    def solve_test_helper(self, A_dims, b_dims, device, dtype):
        from torch.testing._internal.common_utils import random_fullrank_matrix_distinct_singular_value

        b = torch.randn(*b_dims, dtype=dtype, device=device)
        A = random_fullrank_matrix_distinct_singular_value(*A_dims, dtype=dtype).to(device)
        return b, A

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(torch.float32, torch.float64, torch.complex64, torch.complex128)
    @precisionOverride({torch.float32: 1e-3, torch.complex64: 1e-3})
    @skipCUDAIfRocm
    def test_solve(self, device, dtype):
        def run_test(n, batch, rhs):
            A_dims = (n, *batch)
            b_dims = (*batch, n, *rhs)
            b, A = self.solve_test_helper(A_dims, b_dims, device, dtype)

            # Correctness test
            x = torch.linalg.solve(A, b)
            if rhs == ():
                Ax = np.matmul(A.cpu(), x.unsqueeze(-1).cpu())
                Ax.squeeze_(-1)
            else:
                Ax = np.matmul(A.cpu(), x.cpu())
            self.assertEqual(b.expand_as(Ax), Ax)

            # Check against NumPy
            expected = np.linalg.solve(A.cpu().numpy(), b.expand_as(x).cpu().numpy())
            self.assertEqual(x, expected)

            # Check out= variant
            if rhs == ():
                out = torch.empty_like(x.unsqueeze(-1))
            else:
                out = torch.empty_like(x)
            ans = torch.linalg.solve(A, b, out=out)
            self.assertEqual(ans, out)
            self.assertEqual(x, out)

            # Check empty out
            out = torch.empty(0, dtype=dtype, device=device)
            ans = torch.linalg.solve(A, b, out=out)
            self.assertEqual(ans, out)
            self.assertEqual(x, out)

        batches = [(), (0, ), (3, ), (2, 3)]
        ns = [0, 5, 32]
        nrhs = [(), (1, ), (5, )]
        for n, batch, rhs in itertools.product(ns, batches, nrhs):
            run_test(n, batch, rhs)

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(torch.float32, torch.float64, torch.complex64, torch.complex128)
    @precisionOverride({torch.float32: 1e-3, torch.complex64: 1e-3})
    @skipCUDAIfRocm
    def test_solve_batched_non_contiguous(self, device, dtype):
        from torch.testing._internal.common_utils import random_fullrank_matrix_distinct_singular_value
        A = random_fullrank_matrix_distinct_singular_value(2, 2, dtype=dtype).to(device).permute(1, 0, 2)
        b = torch.randn(2, 2, 2, dtype=dtype, device=device).permute(2, 1, 0)
        self.assertFalse(A.is_contiguous())
        self.assertFalse(b.is_contiguous())
        actual = torch.linalg.solve(A, b)
        expected = np.linalg.solve(A.cpu().numpy(), b.cpu().numpy())
        self.assertEqual(actual, expected)

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(torch.float32, torch.float64, torch.complex64, torch.complex128)
    def test_solve_errors(self, device, dtype):
        # solve expects batches of square matrices as input
        with self.assertRaisesRegex(RuntimeError, "must be batches of square matrices"):
            a = torch.randn(2, 3, 4, 3, dtype=dtype, device=device)
            b = torch.randn(2, 3, 4, 1, dtype=dtype, device=device)
            torch.linalg.solve(a, b)

        # solve expects compatible shapes for A x = b
        with self.assertRaisesRegex(RuntimeError, "Incompatible matrix sizes"):
            a = torch.randn(2, 3, 3, 3, dtype=dtype, device=device)
            b = torch.randn(2, 3, 2, 1, dtype=dtype, device=device)
            torch.linalg.solve(a, b)

        # if input is not solvable, RuntimeError is raised mentioning the first non-solvable batch
        def run_test_singular_input(batch_dim, n):
            a = torch.eye(3, 3, dtype=dtype, device=device).reshape((1, 3, 3)).repeat(batch_dim, 1, 1)
            a[n, -1, -1] = 0
            b = torch.randn(batch_dim, 3, 1, dtype=dtype, device=device)
            with self.assertRaisesRegex(RuntimeError, rf'For batch {n}: U\(3,3\) is zero'):
                torch.linalg.solve(a, b)

        for params in [(1, 0), (2, 0), (2, 1), (4, 0), (4, 2), (10, 2)]:
            run_test_singular_input(*params)

        # if out is non-empty then it should have correct sizes
        with self.assertRaisesRegex(RuntimeError, r'does not match broadcasted other shape'):
            out = torch.empty(1, dtype=dtype, device=device)
            A = torch.eye(3, dtype=dtype, device=device)
            b = torch.randn(3, 1, dtype=dtype, device=device)
            torch.linalg.solve(A, b, out=out)

        # if out is non-empty then it should also be Fortran contiguous
        with self.assertRaisesRegex(RuntimeError, r'tensor must be in batched column major'):
            out = torch.zeros(2, 2, 2, dtype=dtype, device=device).permute(2, 1, 0)
            self.assertFalse(out.is_contiguous())
            A = torch.eye(2, dtype=dtype, device=device).reshape((1, 2, 2)).repeat(2, 1, 1)
            b = torch.randn(2, 2, 2, dtype=dtype, device=device)
            torch.linalg.solve(A, b, out=out)

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(torch.float32, torch.float64, torch.complex64, torch.complex128)
    def test_old_solve(self, device, dtype):
        for (k, n) in zip([2, 3, 5], [3, 5, 7]):
            b, A = self.solve_test_helper((n,), (n, k), device, dtype)
            x = torch.solve(b, A)[0]
            self.assertEqual(b, np.matmul(A.cpu(), x.cpu()))

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(torch.float32, torch.float64, torch.complex64, torch.complex128)
    @skipCUDAIfRocm
    def test_old_solve_batched(self, device, dtype):
        def solve_batch_helper(A_dims, b_dims):
            b, A = self.solve_test_helper(A_dims, b_dims, device, dtype)
            x_exp_list = []
            for i in range(b_dims[0]):
                x_exp_list.append(torch.solve(b[i], A[i])[0])
            x_exp = torch.stack(x_exp_list)  # Stacked output
            x_act = torch.solve(b, A)[0]  # Actual output
            self.assertEqual(x_exp, x_act)  # Equality check
            Ax = np.matmul(A.cpu(), x_act.cpu())
            self.assertEqual(b, Ax)

        for batchsize in [1, 3, 4]:
            solve_batch_helper((5, batchsize), (batchsize, 5, 10))

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(torch.float32, torch.float64, torch.complex64, torch.complex128)
    @skipCUDAIfRocm
    def test_old_solve_batched_non_contiguous(self, device, dtype):
        from numpy.linalg import solve
        from torch.testing._internal.common_utils import random_fullrank_matrix_distinct_singular_value
        A = random_fullrank_matrix_distinct_singular_value(2, 2, dtype=dtype).to(device).permute(1, 0, 2)
        b = torch.randn(2, 2, 2, dtype=dtype, device=device).permute(2, 1, 0)
        x, _ = torch.solve(b, A)
        x_exp = solve(A.cpu().numpy(), b.cpu().numpy())
        self.assertEqual(x, x_exp)

    @slowTest
    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(torch.float32, torch.float64, torch.complex64, torch.complex128)
    def test_old_solve_batched_many_batches(self, device, dtype):
        for A_dims, b_dims in zip([(5, 256, 256), (3, )], [(5, 1), (512, 512, 3, 1)]):
            b, A = self.solve_test_helper(A_dims, b_dims, device, dtype)
            x, _ = torch.solve(b, A)
            Ax = torch.matmul(A, x)
            self.assertEqual(Ax, b.expand_as(x))

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(torch.float32, torch.float64, torch.complex64, torch.complex128)
    @skipCUDAIfRocm
    def test_old_solve_batched_broadcasting(self, device, dtype):
        from numpy.linalg import solve

        def run_test(A_dims, b_dims):
            A_matrix_size = A_dims[-1]
            A_batch_dims = A_dims[:-2]
            b, A = self.solve_test_helper((A_matrix_size,) + A_batch_dims, b_dims, device, dtype)
            x, _ = torch.solve(b, A)
            x_exp = solve(A.cpu().numpy(), b.cpu().numpy())
            self.assertEqual(x, x_exp)

        # test against numpy.linalg.solve
        run_test((2, 1, 3, 4, 4), (2, 1, 3, 4, 6))  # no broadcasting
        run_test((2, 1, 3, 4, 4), (4, 6))  # broadcasting b
        run_test((4, 4), (2, 1, 3, 4, 2))  # broadcasting A
        run_test((1, 3, 1, 4, 4), (2, 1, 3, 4, 5))  # broadcasting A & b

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(torch.float, torch.double, torch.cfloat, torch.cdouble)
    @precisionOverride({torch.float: 1e-4, torch.cfloat: 1e-4})
    def test_tensorsolve(self, device, dtype):
        def run_test(a_shape, dims):
            a = torch.randn(a_shape, dtype=dtype, device=device)
            b = torch.randn(a_shape[:2], dtype=dtype, device=device)
            result = torch.linalg.tensorsolve(a, b, dims=dims)
            expected = np.linalg.tensorsolve(a.cpu().numpy(), b.cpu().numpy(), axes=dims)
            self.assertEqual(result, expected)

            # check the out= variant
            out = torch.empty_like(result)
            ans = torch.linalg.tensorsolve(a, b, dims=dims, out=out)
            self.assertEqual(ans, out)
            self.assertEqual(ans, result)

        a_shapes = [(2, 3, 6), (3, 4, 4, 3)]
        dims = [None, (0, 2)]
        for a_shape, d in itertools.product(a_shapes, dims):
            run_test(a_shape, d)

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(torch.float, torch.double, torch.cfloat, torch.cdouble)
    def test_tensorsolve_empty(self, device, dtype):
        # Check for empty inputs. NumPy does not work for these cases.
        a = torch.empty(0, 0, 1, 2, 3, 0, dtype=dtype, device=device)
        b = torch.empty(a.shape[:2], dtype=dtype, device=device)
        x = torch.linalg.tensorsolve(a, b)
        self.assertEqual(torch.tensordot(a, x, dims=len(x.shape)), b)

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(torch.float, torch.double, torch.cfloat, torch.cdouble)
    @precisionOverride({torch.float: 1e-4, torch.cfloat: 1e-4})
    def test_tensorsolve_non_contiguous(self, device, dtype):
        def run_test_permuted(a_shape, dims):
            # check for permuted / transposed inputs
            a = torch.randn(a_shape, dtype=dtype, device=device)
            a = a.movedim((0, 2), (-2, -1))
            self.assertFalse(a.is_contiguous())
            b = torch.randn(a.shape[:2], dtype=dtype, device=device)
            b = b.t()
            self.assertFalse(b.is_contiguous())
            result = torch.linalg.tensorsolve(a, b, dims=dims)
            expected = np.linalg.tensorsolve(a.cpu().numpy(), b.cpu().numpy(), axes=dims)
            self.assertEqual(result, expected)

        def run_test_skipped_elements(a_shape, dims):
            # check for inputs with skipped elements
            a = torch.randn(a_shape, dtype=dtype, device=device)
            a = a[::2]
            self.assertFalse(a.is_contiguous())
            b = torch.randn(a_shape[:2], dtype=dtype, device=device)
            b = b[::2]
            self.assertFalse(b.is_contiguous())
            result = torch.linalg.tensorsolve(a, b, dims=dims)
            expected = np.linalg.tensorsolve(a.cpu().numpy(), b.cpu().numpy(), axes=dims)
            self.assertEqual(result, expected)

            # check non-contiguous out
            out = torch.empty(2 * result.shape[0], *result.shape[1:], dtype=dtype, device=device)[::2]
            self.assertFalse(out.is_contiguous())
            ans = torch.linalg.tensorsolve(a, b, dims=dims, out=out)
            self.assertEqual(ans, out)
            self.assertEqual(ans, result)

        a_shapes = [(2, 3, 6), (3, 4, 4, 3)]
        dims = [None, (0, 2)]
        for a_shape, d in itertools.product(a_shapes, dims):
            run_test_permuted(a_shape, d)

        a_shapes = [(4, 3, 6), (6, 4, 4, 3)]
        dims = [None, (0, 2)]
        for a_shape, d in itertools.product(a_shapes, dims):
            run_test_skipped_elements(a_shape, d)

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(torch.float32)
    def test_tensorsolve_errors_and_warnings(self, device, dtype):
        # tensorsolve expects the input that can be reshaped to a square matrix
        a = torch.eye(2 * 3 * 4).reshape((2 * 3, 4, 2, 3, 4))
        b = torch.randn(8, 4)
        self.assertTrue(np.prod(a.shape[2:]) != np.prod(b.shape))
        with self.assertRaisesRegex(RuntimeError, r'Expected self to satisfy the requirement'):
            torch.linalg.tensorsolve(a, b)

        # if non-empty out tensor with wrong shape is passed a warning is given
        out = torch.empty_like(a)
        b = torch.randn(6, 4)
        with warnings.catch_warnings(record=True) as w:
            # Trigger warning
            torch.linalg.tensorsolve(a, b, out=out)
            # Check warning occurs
            self.assertEqual(len(w), 1)
            self.assertTrue("An output with one or more elements was resized" in str(w[-1].message))

        # dtypes should match
        out = torch.empty_like(a).to(torch.int)
        with self.assertRaisesRegex(RuntimeError, "result dtype Int does not match self dtype"):
            torch.linalg.tensorsolve(a, b, out=out)

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(torch.float32, torch.float64, torch.complex64, torch.complex128)
    @precisionOverride({torch.float: 1e-3, torch.cfloat: 1e-3})
    @skipCUDAIfRocm
    def test_tensorinv(self, device, dtype):

        def run_test(a_shape, ind):
            a = torch.randn(a_shape, dtype=dtype, device=device)
            a_numpy = a.cpu().numpy()
            result = torch.linalg.tensorinv(a, ind=ind)
            expected = np.linalg.tensorinv(a_numpy, ind=ind)
            self.assertEqual(result, expected)

            # check the out= variant
            out = torch.empty_like(result)
            ans = torch.linalg.tensorinv(a, ind=ind, out=out)
            self.assertEqual(ans, out)
            self.assertEqual(ans, result)

        # compare to NumPy output
        run_test((12, 3, 4), ind=1)
        run_test((3, 8, 24), ind=2)
        run_test((18, 3, 3, 2), ind=1)
        run_test((1, 4, 2, 2), ind=2)
        run_test((2, 3, 5, 30), ind=3)
        run_test((24, 2, 2, 3, 2), ind=1)
        run_test((3, 4, 2, 3, 2), ind=2)
        run_test((1, 2, 3, 2, 3), ind=3)
        run_test((3, 2, 1, 2, 12), ind=4)

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(torch.float32, torch.float64, torch.complex64, torch.complex128)
    @precisionOverride({torch.float: 1e-3, torch.cfloat: 1e-3})
    @skipCUDAIfRocm
    def test_tensorinv_non_contiguous(self, device, dtype):

        def run_test(a_shape, ind):
            # check for permuted (transposed) case
            a = torch.randn(a_shape, dtype=dtype, device=device)
            permutation = list(range(0, a.ndim))
            a = a.permute(permutation[ind:] + permutation[:ind])
            self.assertFalse(a.is_contiguous())
            a_numpy = a.cpu().numpy()
            result = torch.linalg.tensorinv(a, ind=a.ndim - ind)
            expected = np.linalg.tensorinv(a_numpy, ind=a.ndim - ind)
            self.assertEqual(result, expected)

        def run_test_skipped_elements(a_shape, ind):
            # check for input with skipped elements
            a = torch.randn(a_shape, dtype=dtype, device=device)
            a = a[::2]
            self.assertFalse(a.is_contiguous())
            a_numpy = a.cpu().numpy()
            result = torch.linalg.tensorinv(a, ind=ind)
            expected = np.linalg.tensorinv(a_numpy, ind=ind)
            self.assertEqual(result, expected)

            # check non-contiguous out
            out = torch.empty(2 * result.shape[0], *result.shape[1:], dtype=dtype, device=device)[::2]
            self.assertFalse(out.is_contiguous())
            ans = torch.linalg.tensorinv(a, ind=ind, out=out)
            self.assertEqual(ans, out)
            self.assertEqual(ans, result)

        run_test((12, 3, 4), ind=1)
        run_test((3, 8, 24), ind=2)
        run_test((18, 3, 3, 2), ind=1)
        run_test((1, 4, 2, 2), ind=2)
        run_test((2, 3, 5, 30), ind=3)
        run_test((24, 2, 2, 3, 2), ind=1)
        run_test((3, 4, 2, 3, 2), ind=2)
        run_test((1, 2, 3, 2, 3), ind=3)
        run_test((3, 2, 1, 2, 12), ind=4)

        run_test_skipped_elements((12, 3, 2), ind=1)
        run_test_skipped_elements((18, 3, 3, 1), ind=1)

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(torch.float32, torch.float64, torch.complex64, torch.complex128)
    def test_tensorinv_empty(self, device, dtype):
        for ind in range(1, 4):
            # Check for empty inputs. NumPy does not work for these cases.
            a = torch.empty(0, 0, 1, 2, 3, 0, dtype=dtype, device=device)
            a_inv = torch.linalg.tensorinv(a, ind=ind)
            self.assertEqual(a_inv.shape, a.shape[ind:] + a.shape[:ind])

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(torch.float32, torch.float64, torch.complex64, torch.complex128)
    def test_tensorinv_errors_and_warnings(self, device, dtype):

        def check_shape(a_shape, ind):
            # tensorinv requires the input to satisfy
            # prod(a.shape[ind:]) == prod(a.shape[:ind])
            a = torch.randn(a_shape)
            with self.assertRaisesRegex(RuntimeError, "Expected self to satisfy the requirement"):
                torch.linalg.tensorinv(a, ind=ind)

        def check_ind(a_shape, ind):
            a = torch.randn(a_shape)
            with self.assertRaisesRegex(RuntimeError, "Expected a strictly positive integer"):
                torch.linalg.tensorinv(a, ind=ind)

        def check_out(a_shape, ind):
            # if non-empty out tensor with wrong shape is passed a warning is given
            a = torch.randn(a_shape)
            out = torch.empty_like(a)
            with warnings.catch_warnings(record=True) as w:
                # Trigger warning
                torch.linalg.tensorinv(a, ind=ind, out=out)
                # Check warning occurs
                self.assertEqual(len(w), 1)
                self.assertTrue("An output with one or more elements was resized" in str(w[-1].message))

            # dtypes should match
            out = torch.empty_like(a).to(torch.int)
            with self.assertRaisesRegex(RuntimeError, "result dtype Int does not match self dtype"):
                torch.linalg.tensorinv(a, ind=ind, out=out)

        # test for invalid shape
        check_shape((2, 3, 4), ind=1)
        check_shape((1, 2, 3, 4), ind=3)

        # test for invalid ind
        check_ind((12, 3, 4), ind=-1)
        check_ind((18, 3, 3, 2), ind=0)

        # test for invalid out tensor
        check_out((12, 3, 4), ind=1)
        check_out((3, 8, 24), ind=2)

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(torch.float32, torch.float64, torch.complex64, torch.complex128)
    def test_tensorinv_singular_input(self, device, dtype):

        def check_singular_input(a_shape, ind):
            prod_ind_end = np.prod(a_shape[ind:])
            a = torch.eye(prod_ind_end, dtype=dtype, device=device)
            a[-1, -1] = 0   # Now `a` is singular
            a = a.reshape(a_shape)
            with self.assertRaisesRegex(RuntimeError, "Failed to invert the input tensor, because it is singular"):
                torch.linalg.tensorinv(a, ind=ind)

        # test for non-invertible input
        check_singular_input((12, 3, 4), ind=1)
        check_singular_input((3, 6, 18), ind=2)

    def _test_dot_vdot_vs_numpy(self, device, dtype, torch_fn, np_fn):
        def check(x, y):
            # Compare with numpy
            res = torch_fn(x, y)
            ref = torch.from_numpy(np.array(np_fn(x.cpu().numpy(), y.cpu().numpy())))
            self.assertEqual(res.cpu(), ref)

            # Test out variant
            out = torch.empty_like(res)
            torch_fn(x, y, out=out)
            self.assertEqual(out, res)

        # Empty
        x = torch.tensor([], dtype=dtype, device=device)
        y = torch.tensor([], dtype=dtype, device=device)
        check(x, y)

        # Contiguous
        x = torch.randn(10, dtype=dtype, device=device)
        y = torch.randn(10, dtype=dtype, device=device)
        check(x, y)

        # 0 strided
        y = torch.randn(1, dtype=dtype, device=device).expand(10)
        check(x, y)

        # 2 strided
        check(x[::2], y[::2])

    @dtypes(torch.float, torch.cfloat)
    @precisionOverride({torch.cfloat: 1e-4, torch.float32: 5e-5})
    def test_dot_vs_numpy(self, device, dtype):
        self._test_dot_vdot_vs_numpy(device, dtype, torch.dot, np.dot)

    @dtypes(torch.float, torch.cfloat)
    @precisionOverride({torch.cfloat: 1e-4, torch.float32: 5e-5})
    def test_vdot_vs_numpy(self, device, dtype):
        self._test_dot_vdot_vs_numpy(device, dtype, torch.vdot, np.vdot)

    def _test_dot_vdot_invalid_args(self, device, torch_fn, complex_dtypes=False):
        def check(x, y, regex):
            with self.assertRaisesRegex(RuntimeError, regex):
                torch_fn(x, y)

        if complex_dtypes:
            x = torch.randn(1, dtype=torch.cfloat, device=device)
            y = torch.randn(3, dtype=torch.cdouble, device=device)
        else:
            x = torch.randn(1, dtype=torch.float, device=device)
            y = torch.randn(3, dtype=torch.double, device=device)

        check(x, y, 'dot : expected both vectors to have same dtype')
        check(x.reshape(1, 1), y, '1D tensors expected')
        check(x.expand(9), y.to(x.dtype), 'inconsistent tensor size')

        if self.device_type != 'cpu':
            x_cpu = x.expand(3).cpu()
            check(x_cpu, y.to(x.dtype), 'expected all tensors to be on the same device')

    @onlyOnCPUAndCUDA
    def test_vdot_invalid_args(self, device):
        self._test_dot_vdot_invalid_args(device, torch.vdot)
        self._test_dot_vdot_invalid_args(device, torch.vdot, complex_dtypes=True)

    @onlyOnCPUAndCUDA
    def test_dot_invalid_args(self, device):
        self._test_dot_vdot_invalid_args(device, torch.dot)
        self._test_dot_vdot_invalid_args(device, torch.dot, complex_dtypes=True)

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(torch.float32, torch.float64, torch.complex64, torch.complex128)
    def test_matrix_rank(self, device, dtype):
        matrix_rank = torch.linalg.matrix_rank

        def run_test(shape0, shape1, batch):
            a = torch.randn(*batch, shape0, shape1, dtype=dtype, device=device)
            rank_a = matrix_rank(a)

            self.assertEqual(rank_a, matrix_rank(a.conj().transpose(-2, -1)))
            aaH = torch.matmul(a, a.conj().transpose(-2, -1))
            rank_aaH = matrix_rank(aaH)
            rank_aaH_hermitian = matrix_rank(aaH, hermitian=True)
            self.assertEqual(rank_aaH, rank_aaH_hermitian)
            aHa = torch.matmul(a.conj().transpose(-2, -1), a)
            self.assertEqual(matrix_rank(aHa), matrix_rank(aHa, hermitian=True))

            # check against NumPy
            self.assertEqual(rank_a, np.linalg.matrix_rank(a.cpu().numpy()))
            self.assertEqual(matrix_rank(a, 0.01), np.linalg.matrix_rank(a.cpu().numpy(), 0.01))

            self.assertEqual(rank_aaH, np.linalg.matrix_rank(aaH.cpu().numpy()))
            self.assertEqual(matrix_rank(aaH, 0.01), np.linalg.matrix_rank(aaH.cpu().numpy(), 0.01))

            # hermitian flag for NumPy was added in 1.14.0
            if np.lib.NumpyVersion(np.__version__) >= '1.14.0':
                self.assertEqual(rank_aaH_hermitian,
                                 np.linalg.matrix_rank(aaH.cpu().numpy(), hermitian=True))
                self.assertEqual(matrix_rank(aaH, 0.01, True),
                                 np.linalg.matrix_rank(aaH.cpu().numpy(), 0.01, True))

            # check out= variant
            out = torch.empty(a.shape[:-2], dtype=torch.int64, device=device)
            ans = matrix_rank(a, out=out)
            self.assertEqual(ans, out)
            self.assertEqual(ans, rank_a)

        shapes = (3, 13)
        batches = ((), (0, ), (4, ), (3, 5, ))
        for (shape0, shape1), batch in zip(itertools.product(shapes, reversed(shapes)), batches):
            run_test(shape0, shape1, batch)

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(torch.float32, torch.float64, torch.complex64, torch.complex128)
    def test_matrix_rank_empty(self, device, dtype):
        matrix_rank = torch.linalg.matrix_rank

        # NumPy doesn't work for input with no elements
        def run_test(shape0, shape1, batch):
            a = torch.randn(*batch, shape0, shape1, dtype=dtype, device=device)
            rank_a = matrix_rank(a)
            expected = torch.zeros(batch, dtype=torch.int64, device=device)

            self.assertEqual(rank_a, matrix_rank(a.conj().transpose(-2, -1)))

            aaH = torch.matmul(a, a.conj().transpose(-2, -1))
            rank_aaH = matrix_rank(aaH)
            rank_aaH_hermitian = matrix_rank(aaH, hermitian=True)
            self.assertEqual(rank_aaH, rank_aaH_hermitian)

            aHa = torch.matmul(a.conj().transpose(-2, -1), a)
            self.assertEqual(matrix_rank(aHa), matrix_rank(aHa, hermitian=True))

            self.assertEqual(rank_a, expected)
            self.assertEqual(matrix_rank(a, 0.01), expected)

            self.assertEqual(rank_aaH, expected)
            self.assertEqual(matrix_rank(aaH, 0.01), expected)

            self.assertEqual(rank_aaH_hermitian, expected)
            self.assertEqual(matrix_rank(aaH, 0.01, True), expected)

        batches = ((), (4, ), (3, 5, ))
        for batch in batches:
            run_test(0, 0, batch)
            run_test(0, 3, batch)
            run_test(3, 0, batch)

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(torch.float32, torch.float64, torch.complex64, torch.complex128)
    def test_matrix_rank_basic(self, device, dtype):
        matrix_rank = torch.linalg.matrix_rank

        a = torch.eye(10, dtype=dtype, device=device)
        self.assertEqual(matrix_rank(a).item(), 10)
        self.assertEqual(matrix_rank(a, hermitian=True).item(), 10)

        a[5, 5] = 0
        self.assertEqual(matrix_rank(a).item(), 9)
        self.assertEqual(matrix_rank(a, hermitian=True).item(), 9)

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(torch.float32, torch.float64, torch.complex64, torch.complex128)
    def test_old_matrix_rank(self, device, dtype):
        a = torch.eye(10, dtype=dtype, device=device)
        self.assertEqual(torch.matrix_rank(a).item(), 10)
        self.assertEqual(torch.matrix_rank(a, True).item(), 10)

        a[5, 5] = 0
        self.assertEqual(torch.matrix_rank(a).item(), 9)
        self.assertEqual(torch.matrix_rank(a, True).item(), 9)

        a = torch.randn(24, 42, dtype=dtype, device=device)
        self.assertEqual(torch.matrix_rank(a), torch.matrix_rank(a.t()))
        aaT = torch.mm(a, a.conj().t())
        self.assertEqual(torch.matrix_rank(aaT), torch.matrix_rank(aaT, True))
        aTa = torch.mm(a.conj().t(), a)
        self.assertEqual(torch.matrix_rank(aTa), torch.matrix_rank(aTa, True))

        a = torch.randn(35, 75, dtype=dtype, device=device)
        self.assertEqual(torch.matrix_rank(a), np.linalg.matrix_rank(a.cpu().numpy()))
        self.assertEqual(torch.matrix_rank(a, 0.01), np.linalg.matrix_rank(a.cpu().numpy(), 0.01))

        aaT = torch.mm(a, a.conj().t())
        self.assertEqual(torch.matrix_rank(aaT), np.linalg.matrix_rank(aaT.cpu().numpy()))
        self.assertEqual(torch.matrix_rank(aaT, 0.01), np.linalg.matrix_rank(aaT.cpu().numpy(), 0.01))

        if np.lib.NumpyVersion(np.__version__) >= '1.14.0':
            self.assertEqual(torch.matrix_rank(aaT, True), np.linalg.matrix_rank(aaT.cpu().numpy(), True))
            self.assertEqual(torch.matrix_rank(aaT, 0.01, True), np.linalg.matrix_rank(aaT.cpu().numpy(), 0.01, True))

    @onlyOnCPUAndCUDA
    @dtypes(torch.float, torch.double, torch.cfloat, torch.cdouble)
    @precisionOverride({torch.float: 1e-03, torch.cfloat: 1e-03})
    def test_multi_dot(self, device, dtype):
        def check(*shapes):
            tensors = [torch.randn(shape, device=device, dtype=dtype) for shape in shapes]
            np_arrays = [tensor.cpu().numpy() for tensor in tensors]
            res = torch.linalg.multi_dot(tensors).cpu()
            ref = torch.from_numpy(np.array(np.linalg.multi_dot(np_arrays)))
            self.assertEqual(res, ref)

        # test for inputs with empty dimensions
        check([0], [0])
        check([2], [2, 0])
        check([1, 0], [0])
        check([0, 2], [2, 1])
        check([2, 2], [2, 0])
        check([2, 0], [0, 3])
        check([0, 0], [0, 1])
        check([4, 2], [2, 0], [0, 3], [3, 2])

        # test variable output shapes
        check([2], [2])
        check([1, 2], [2])
        check([2], [2, 1])
        check([1, 2], [2, 1])
        check([3, 2], [2, 4])

        # test multiple input tensors
        check([3], [3, 4], [4, 2], [2, 5], [5])
        check([1, 2], [2, 2], [2, 3], [3, 1])

        # test large tensors
        check([10, 100], [100, 5], [5, 50])

        # test discontiguous input
        shapes = [[3, 2], [2, 2], [2, 3], [3, 4]]
        tensors = [torch.randn(shape, device=device, dtype=dtype) for shape in shapes]
        tensors[1] = tensors[1].t()
        np_arrays = [tensor.cpu().numpy() for tensor in tensors]
        res = torch.linalg.multi_dot(tensors).cpu()
        ref = torch.from_numpy(np.array(np.linalg.multi_dot(np_arrays)))
        self.assertEqual(res, ref)

        # test out variant
        out = torch.empty(0, device=device, dtype=dtype)
        torch.linalg.multi_dot(tensors, out=out)
        self.assertEqual(res, out)
        out = torch.zeros_like(res.t(), memory_format=torch.contiguous_format).t()
        self.assertFalse(out.is_contiguous())
        out_strides = out.stride()
        torch.linalg.multi_dot(tensors, out=out)
        self.assertEqual(res, out)
        self.assertEqual(out_strides, out.stride())

    @precisionOverride({torch.float32: 5e-6, torch.complex64: 5e-6})
    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(torch.float32, torch.float64, torch.complex64, torch.complex128)
    def test_qr(self, device, dtype):
        def run_test(tensor_dims, some):
            A = torch.randn(*tensor_dims, dtype=dtype, device=device)
            Q, R = torch.qr(A, some=some)

            # Check0: Q[-2:] = (m, n_columns), R[-2:] = (n_columns, n)
            m, n = tensor_dims[-2:]
            n_columns = m if (not some) and m > n else min(m, n)
            self.assertEqual(Q.size(-2), m)
            self.assertEqual(R.size(-1), n)
            self.assertEqual(Q.size(-1), n_columns)

            A_ = A.cpu().numpy()
            Q_ = Q.cpu().numpy()
            R_ = R.cpu().numpy()

            # Check1: A = QR
            self.assertEqual(A_, np.matmul(Q_, R_))

            # Check2: A = QR (with out)
            Q_out, R_out = torch.full_like(Q, math.nan), torch.full_like(R, math.nan)
            torch.qr(A, some=some, out=(Q_out, R_out))
            Q_out_ = Q_out.cpu().numpy()
            R_out_ = R_out.cpu().numpy()
            self.assertEqual(A_, np.matmul(Q_out_, R_out_))

            # Check3: Q == Q_out, R == R_out
            self.assertEqual(Q_, Q_out_)
            self.assertEqual(R_, R_out_)

            # Check4: Q^{T}Q = I, triu(R) = R
            eye = torch.eye(n_columns, device=device, dtype=dtype).expand(Q.shape[:-2] + (n_columns, n_columns)).cpu().numpy()
            self.assertEqual(np.matmul(Q_.swapaxes(-1, -2).conj(), Q_), eye)
            self.assertEqual(R.triu(), R)

        tensor_dims_list = [(3, 5), (5, 5), (5, 3),  # Single matrix
                            (7, 3, 5), (7, 5, 5), (7, 5, 3),  # 3-dim Tensors
                            (7, 5, 3, 5), (7, 5, 5, 5), (7, 5, 5, 3)]  # 4-dim Tensors
        for tensor_dims, some in itertools.product(tensor_dims_list, [True, False]):
            run_test(tensor_dims, some)

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(torch.float, torch.double, torch.cfloat, torch.cdouble)
    def test_qr_vs_numpy(self, device, dtype):
        """
        test torch.linalg.qr vs numpy.linalg.qr
        """
        sizes_to_test = [
            (7, 5),
            (5, 7),
            (5, 0),    # empty
            (0, 5),    # empty
        ]
        for size in sizes_to_test:
            t = torch.randn(size, device=device, dtype=dtype)
            np_t = t.cpu().numpy()
            for mode in ['reduced', 'complete']:
                exp_q, exp_r = np.linalg.qr(np_t, mode=mode)
                q, r = torch.linalg.qr(t, mode=mode)
                self.assertEqual(q, exp_q)
                self.assertEqual(r, exp_r)
            #
            # for mode='r' we need a special logic because numpy returns only r
            exp_r = np.linalg.qr(np_t, mode='r')
            q, r = torch.linalg.qr(t, mode='r')
            # check that q is empty
            self.assertEqual(q.shape, (0,))
            self.assertEqual(q.dtype, t.dtype)
            self.assertEqual(q.device, t.device)
            # check r
            self.assertEqual(r, exp_r)

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(torch.float)
    def test_linalg_qr_autograd_errors(self, device, dtype):
        # torch.linalg.qr(mode='r') returns only 'r' and discards 'q', but
        # without 'q' you cannot compute the backward pass. Check that
        # linalg_qr_backward complains cleanly in that case.
        inp = torch.randn((5, 7), device=device, dtype=dtype, requires_grad=True)
        q, r = torch.linalg.qr(inp, mode='r')
        self.assertEqual(q.shape, (0,))  # empty tensor
        b = torch.sum(r)
        with self.assertRaisesRegex(RuntimeError,
                                    "The derivative of qr is not implemented when mode='r'"):
            b.backward()
        #
        inp = torch.randn((7, 5), device=device, dtype=dtype, requires_grad=True)
        q, r = torch.linalg.qr(inp, mode='complete')
        b = torch.sum(r)
        with self.assertRaisesRegex(RuntimeError,
                                    "The derivative of qr is not implemented when mode='complete' and nrows > ncols"):
            b.backward()

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(torch.float, torch.double, torch.cfloat, torch.cdouble)
    def test_qr_batched(self, device, dtype):
        """
        test torch.linalg.qr vs numpy.linalg.qr. We need some special logic
        because numpy does not support batched qr
        """
        def np_qr_batched(a, mode):
            """poor's man batched version of np.linalg.qr"""
            all_q = []
            all_r = []
            for matrix in a:
                result = np.linalg.qr(matrix, mode=mode)
                if mode == 'r':
                    all_r.append(result)
                else:
                    q, r = result
                    all_q.append(q)
                    all_r.append(r)
            if mode == 'r':
                return np.array(all_r)
            else:
                return np.array(all_q), np.array(all_r)

        t = torch.randn((3, 7, 5), device=device, dtype=dtype)
        np_t = t.cpu().numpy()
        for mode in ['reduced', 'complete']:
            exp_q, exp_r = np_qr_batched(np_t, mode=mode)
            q, r = torch.linalg.qr(t, mode=mode)
            self.assertEqual(q, exp_q)
            self.assertEqual(r, exp_r)
        # for mode='r' we need a special logic because numpy returns only r
        exp_r = np_qr_batched(np_t, mode='r')
        q, r = torch.linalg.qr(t, mode='r')
        # check that q is empty
        self.assertEqual(q.shape, (0,))
        self.assertEqual(q.dtype, t.dtype)
        self.assertEqual(q.device, t.device)
        # check r
        self.assertEqual(r, exp_r)

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(torch.float, torch.double, torch.cfloat, torch.cdouble)
    def test_qr_out(self, device, dtype):
        """
        test torch.linalg.qr(out=...) vs torch.lingalg.qr
        """
        sizes_to_test = [
            (7, 5),
            (5, 7),
            (5, 0),    # empty
            (0, 5),    # empty
        ]
        for size in sizes_to_test:
            t = torch.randn(size, device=device, dtype=dtype)
            np_t = t.cpu().numpy()
            for mode in ['reduced', 'complete', 'r']:
                q, r = torch.linalg.qr(t, mode=mode)
                out = (torch.empty((0), dtype=dtype, device=device),
                       torch.empty((0), dtype=dtype, device=device))
                q2, r2 = torch.linalg.qr(t, mode=mode, out=out)
                self.assertIs(q2, out[0])
                self.assertIs(r2, out[1])
                self.assertEqual(q2, q)
                self.assertEqual(r2, r)

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(torch.float)
    def test_qr_error_cases(self, device, dtype):
        t1 = torch.randn(5, device=device, dtype=dtype)
        with self.assertRaisesRegex(RuntimeError, 'qr input should have at least 2 dimensions, but has 1 dimensions instead'):
            torch.linalg.qr(t1)
        t2 = torch.randn((5, 7), device=device, dtype=dtype)
        with self.assertRaisesRegex(RuntimeError, "qr received unrecognized mode 'hello'"):
            torch.linalg.qr(t2, mode='hello')

    @dtypes(torch.double, torch.cdouble)
    def test_einsum(self, device, dtype):
        def check(equation, *operands):
            ref = np.einsum(equation, *[operand.cpu().numpy() for operand in operands])
            res = torch.einsum(equation, operands)
            self.assertEqual(res.cpu(), torch.from_numpy(np.array(ref)))

            # Check autograd
            ops = [op.detach().requires_grad_() for op in operands]
            self.assertTrue(gradcheck(lambda *ops: torch.einsum(equation, ops), ops))
            for op in ops:
                self.assertTrue(op._version == 0)

        # Test cases from https://gist.github.com/rockt/15ee013889d65342088e9260a377dc8f
        x = torch.rand(5, device=device, dtype=dtype)
        y = torch.rand(7, device=device, dtype=dtype)
        A = torch.randn(3, 5, device=device, dtype=dtype)
        B = torch.randn(2, 5, device=device, dtype=dtype)
        C = torch.randn(2, 3, 5, device=device, dtype=dtype)
        D = torch.randn(2, 5, 7, device=device, dtype=dtype)
        E = torch.randn(7, 9, device=device, dtype=dtype)
        F = torch.randn(2, 3, 3, 5, device=device, dtype=dtype)
        G = torch.randn(5, 4, 6, device=device, dtype=dtype)
        H = torch.randn(4, 4, device=device, dtype=dtype)
        I = torch.rand(2, 3, 2, device=device, dtype=dtype)

        # Note: gradcheck fails if the same input is given multiple times which is why the
        # calls to clone below. (see https://github.com/pytorch/pytorch/issues/9282)

        # Vector operations
        check('i->', x)                     # sum
        check('i,i->', x, x.clone())        # dot
        check('i,i->i', x, x.clone())       # vector element-wisem mul
        check('i,j->ij', x, y)              # outer

        # Matrix operations
        check("ij->ji", A)                  # transpose
        check("ij->j", A)                   # row sum
        check("ij->i", A)                   # col sum
        check("ij,ij->ij", A, A.clone())    # matrix element-wise mul
        check("ij,j->i", A, x)              # matrix vector multiplication
        check("ij,kj->ik", A, B)            # matmul
        check("ij,ab->ijab", A, E)          # matrix outer product

        # Tensor operations
        check("aij,ajk->aik", C, D)         # batch matmul
        check("ijk,jk->i", C, A)            # tensor matrix contraction
        check("aij,jk->aik", D, E)          # tensor matrix contraction
        check("abcd,dfg->abcfg", F, G)      # tensor tensor contraction
        check("ijk,jk->ik", C, A)           # tensor matrix contraction with double indices
        check("ijk,jk->ij", C, A)           # tensor matrix contraction with double indices
        check("ijk,ik->j", C, B)            # non contiguous
        check("ijk,ik->jk", C, B)           # non contiguous with double indices

        # Test diagonals
        check("ii", H)                      # trace
        check("ii->i", H)                   # diagonal
        check('iji->j', I)                  # non-contiguous trace
        check('ngrg...->nrg...', torch.rand((2, 1, 3, 1, 4), device=device, dtype=dtype))

        # Test ellipsis
        check("i...->...", H)
        check("ki,...k->i...", A.t(), B)
        check("k...,jk->...", A.t(), B)
        check('...ik, ...j -> ...ij', C, x)
        check('bik,k...j->i...j', C, torch.rand(5, 3, device=device, dtype=dtype))
        check('i...j, ij... -> ...ij', C, torch.rand(2, 5, 2, 3, device=device, dtype=dtype))

        # torch.bilinear with discontiguous tensors
        l = torch.randn(10, 5, device=device, dtype=dtype).transpose(0, 1)
        r = torch.randn(20, 5, device=device, dtype=dtype).transpose(0, 1)
        w = torch.randn(15, 10, 20, device=device, dtype=dtype)
        check("bn,anm,bm->ba", l, w, r)

        # with strided tensors
        check("bn,anm,bm->ba", l[:, ::2], w[:, ::2, ::2], r[:, ::2])

    @dtypes(torch.double, torch.cdouble)
    def test_einsum_random(self, device, dtype):
        def check(equation, *operands):
            ref = np.einsum(equation, *[op.cpu().numpy() for op in operands])
            res = torch.einsum(equation, operands)
            self.assertEqual(res.cpu(), torch.from_numpy(np.array(ref)))

        for _ in range(20):
            # Create a random number of input operands, each with a random
            # number of dimensions randomly labeled.
            op_labels = []
            valid_labels = set()
            for _ in range(random.randint(1, 3)):
                labels = np.random.randint(0, 10, random.randint(1, 5))
                op_labels.append(labels)
                valid_labels.update(labels)
            label_size = np.random.randint(1, 5, 10)
            ell_sizes = np.random.randint(1, 5, 3)

            # Build equation and tensors from input operand labels.
            ops = []
            equation = ''
            for labels in op_labels:
                sizes = [label_size[label] for label in labels]
                labels = [chr(ord('a') + label) for label in labels]

                # Add ellipsis dimensions at random
                ell_num_dim = random.randint(0, 3)
                if ell_num_dim > 0:
                    ell_index = random.randint(0, len(labels))
                    sizes[ell_index:ell_index] = ell_sizes[-ell_num_dim:]
                    labels.insert(ell_index, "...")

                equation += ''.join(labels) + ','
                ops.append(torch.rand(sizes, device=device, dtype=dtype))
            equation = equation[:-1]

            # Test with implicit output
            check(equation, *ops)

            # Randomly choose some labels to be part of the output
            out_labels = np.unique(np.random.choice(list(valid_labels), random.randint(1, len(valid_labels))))
            out_labels = [chr(ord('a') + label) for label in out_labels]
            ell_index = random.randint(0, len(out_labels))
            out_labels.insert(ell_index, '...')
            equation += '->' + ''.join(out_labels)

            # Randomly test the output
            check(equation, *ops)

    def test_einsum_corner_cases(self, device):
        def check(equation, *operands, expected_output):
            tensors = [torch.tensor(operand, dtype=torch.float32, device=device) if not isinstance(operand, tuple)
                       else torch.rand(operand, dtype=torch.float32, device=device) for operand in operands]
            output = torch.einsum(equation, tensors)
            self.assertEqual(output, torch.tensor(expected_output, dtype=torch.float32, device=device))

        # Test equation variantions
        check(' ', 1, expected_output=1)
        check(' -> ', 1, expected_output=1)
        check(' , ', 2, 2, expected_output=4)
        check(' , , ', 2, 2, 2, expected_output=8)
        check(' , -> ', 2, 2, expected_output=4)
        check(' i ', [1], expected_output=[1])
        check(' i -> ', [1], expected_output=1)
        check(' i -> i ', [1], expected_output=[1])
        check(' i , i ', [2], [2], expected_output=4)
        check(' i , i -> i ', [2], [2], expected_output=[4])

        # Test tensors with 0 size dimensions
        check('i', [], expected_output=[])
        check(' i j -> j', [[], []], expected_output=[])
        check('ij->i', [[], []], expected_output=[0., 0.])
        check(' i j k  ,  k  -> i j ', (3, 0, 6), (6,), expected_output=[[], [], []])

        # Test broadcasting
        check('i,j', [2], [1, 2], expected_output=[[2, 4]])
        check('i,ij->ij', [1, 2], [[1, 2, 3], [2, 3, 4]], expected_output=[[1, 2, 3], [4, 6, 8]])

        # Test ellipsis broadcasting
        check('...', 1, expected_output=1)
        check('...->', 1, expected_output=1)
        check('...->...', 1, expected_output=1)
        check('...', [1], expected_output=[1])
        check('...->', [1], expected_output=1)
        check('i...->i', [1], expected_output=[1])
        check('i...->...i', [1], expected_output=[1])
        check('...a->', [[2], [4]], expected_output=6)
        check('a...b->ab', [[[1], [2]], [[3], [4]]], expected_output=[[3], [7]])

    def test_einsum_error_cases(self, device):
        def check(equation, operands, regex, exception=RuntimeError):
            with self.assertRaisesRegex(exception, r'einsum\(\) ' + regex):
                torch.einsum(equation, operands)

        x = torch.rand(2)
        y = torch.rand(2, 3)

        check('', [], r'must provide at least one operand')
        check('. ..', [x], r'found \'.\' for operand 0 that is not part of any ellipsis')
        check('... ...', [x], r'found \'.\' for operand 0 for which an ellipsis was already found')
        check('A', [x], r'operand subscript must be in range \[a, z\] but found A for operand 0')
        check(',', [x], r'fewer operands were provided than specified in the equation')
        check('', [x, x], r'more operands were provided than specified in the equation')
        check('', [x], r'the number of subscripts in the equation \(0\) does not match the number '
                       r'of dimensions \(1\) for operand 0 and no ellipsis was given')
        check('ai', [x], r'the number of subscripts in the equation \(2\) does not match the number '
                         r'of dimensions \(1\) for operand 0 and no ellipsis was given')
        check('ai...', [x], r'the number of subscripts in the equation \(2\) is more than the number '
                            r'of dimensions \(1\) for operand 0')
        check('a->... .', [x], r'found \'.\' for output but an ellipsis \(...\) was already found')
        check('a->..', [x], r'found \'.\' for output that is not part of any ellipsis \(...\)')
        check('a->A', [x], r'subscripts must be in range \[a, z\] but found A for the output')
        check('a->aa', [x], r'output subscript a appears more than once in the output')
        check('a->i', [x], r'output subscript i does not appear in the equation for any input operand')
        check('aa', [y], r'subscript a is repeated for operand 0 but the sizes don\'t match, 3 != 2')
        check('a, ba', [x, y], r'operands do not broadcast with remapped shapes \[original->remapped\]: '
                               r'\[2\]->\[1, 2\] \[2, 3\]->\[2, 3\]')

    def triangular_solve_test_helper(self, A_dims, b_dims, upper, unitriangular,
                                     device, dtype):
        triangle_function = torch.triu if upper else torch.tril
        b = torch.randn(*b_dims, dtype=dtype, device=device)
        A = torch.randn(*A_dims, dtype=dtype, device=device)
        # create positive definite matrix
        A = torch.matmul(A, A.transpose(-2, -1))
        A_triangular = triangle_function(A)
        if unitriangular:
            A_triangular.diagonal(dim1=-2, dim2=-1).fill_(1.)
        return b, A_triangular

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(torch.float32, torch.float64, torch.complex64, torch.complex128)
    @precisionOverride({torch.float32: 1e-3, torch.complex64: 1e-3,
                        torch.float64: 1e-8, torch.complex128: 1e-8})
    def test_triangular_solve(self, device, dtype):
        ks = [0, 1, 3]
        ns = [0, 5]
        for (k, n), (upper, unitriangular, transpose) in itertools.product(zip(ks, ns),
                                                                           itertools.product([True, False], repeat=3)):
            b, A = self.triangular_solve_test_helper((n, n), (n, k), upper,
                                                     unitriangular, device, dtype)
            x = torch.triangular_solve(b, A, upper=upper, unitriangular=unitriangular, transpose=transpose)[0]
            if transpose:
                self.assertEqual(b, np.matmul(A.t().cpu(), x.cpu()))
            else:
                self.assertEqual(b, np.matmul(A.cpu(), x.cpu()))

    @skipCPUIfNoLapack
    @skipCUDAIfNoMagma
    @dtypes(torch.float32, torch.float64, torch.complex64, torch.complex128)
    @precisionOverride({torch.float32: 1e-3, torch.complex64: 1e-3,
                        torch.float64: 1e-8, torch.complex128: 1e-8})
    def test_triangular_solve_batched(self, device, dtype):
        def triangular_solve_batch_helper(A_dims, b_dims, upper, unitriangular, transpose):
            b, A = self.triangular_solve_test_helper(A_dims, b_dims, upper,
                                                     unitriangular, device, dtype)
            x_exp_list = []
            for i in range(b_dims[0]):
                x_exp_list.append(torch.triangular_solve(b[i], A[i], upper=upper,
                                                         unitriangular=unitriangular,
                                                         transpose=transpose)[0])
            x_exp = torch.stack(x_exp_list)  # Stacked output
            x_act = torch.triangular_solve(b, A, upper=upper,
                                           unitriangular=unitriangular,
                                           transpose=transpose)[0]  # Actual output
            self.assertEqual(x_act, x_exp)  # Equality check
            if transpose:
                A = A.transpose(-2, -1)

            Ax = np.matmul(A.cpu(), x_act.cpu())
            self.assertEqual(b, Ax)

        def triangular_solve_zero_batch_helper(A_dims, b_dims, upper, unitriangular, transpose):
            b, A = self.triangular_solve_test_helper(A_dims, b_dims, upper,
                                                     unitriangular, device, dtype)
            x = torch.triangular_solve(b, A, upper=upper,
                                       unitriangular=unitriangular,
                                       transpose=transpose)[0]
            self.assertTrue(x.shape == b.shape)

        for upper, unitriangular, transpose in itertools.product([True, False], repeat=3):
            batchsize = 3
            triangular_solve_batch_helper((batchsize, 5, 5), (batchsize, 5, 10),
                                          upper, unitriangular, transpose)

            # test empty input
            triangular_solve_batch_helper((batchsize, 0, 0), (batchsize, 0, 10),
                                          upper, unitriangular, transpose)
            triangular_solve_batch_helper((batchsize, 0, 0), (batchsize, 0, 0),
                                          upper, unitriangular, transpose)

            # test zero batch case
            batchsize = 0
            triangular_solve_zero_batch_helper((batchsize, 5, 5), (batchsize, 5, 10),
                                               upper, unitriangular, transpose)


    @slowTest
    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(torch.float32, torch.float64, torch.complex64, torch.complex128)
    @precisionOverride({torch.float32: 1e-3, torch.complex64: 1e-3,
                        torch.float64: 1e-8, torch.complex128: 1e-8})
    def test_triangular_solve_batched_many_batches(self, device, dtype):
        for upper, transpose, unitriangular in itertools.product([True, False], repeat=3):
            # test batched A case
            b, A = self.triangular_solve_test_helper((256, 256, 5, 5), (5, 1),
                                                     upper, unitriangular, device, dtype)
            x, _ = torch.triangular_solve(b, A,
                                          upper=upper, transpose=transpose, unitriangular=unitriangular)
            if transpose:
                A = A.transpose(-2, -1)

            Ax = torch.matmul(A, x)

            rtol = 1e-2 if dtype in [torch.float32, torch.complex64] else self.precision
            self.assertEqual(Ax, b.expand_as(Ax), atol=self.precision, rtol=rtol)

            # test batched b case
            b, A = self.triangular_solve_test_helper((3, 3), (512, 512, 3, 1),
                                                     upper, unitriangular, device, dtype)
            x, _ = torch.triangular_solve(b, A, upper=upper, transpose=transpose,
                                          unitriangular=unitriangular)
            if transpose:
                A = A.transpose(-2, -1)

            self.assertEqual(torch.matmul(A, x), b)

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @unittest.skipIf(not TEST_SCIPY, "SciPy not found")
    @dtypes(torch.float32, torch.float64, torch.complex64, torch.complex128)
    def test_triangular_solve_batched_broadcasting(self, device, dtype):
        from scipy.linalg import solve_triangular as tri_solve

        def scipy_tri_solve_batched(A, B, upper, trans, diag):
            batch_dims_A, batch_dims_B = A.shape[:-2], B.shape[:-2]
            single_dim_A, single_dim_B = A.shape[-2:], B.shape[-2:]
            expand_dims = tuple(torch._C._infer_size(torch.Size(batch_dims_A),
                                                     torch.Size(batch_dims_B)))
            expand_A = np.broadcast_to(A, expand_dims + single_dim_A)
            expand_B = np.broadcast_to(B, expand_dims + single_dim_B)
            flat_A = expand_A.reshape((-1,) + single_dim_A)
            flat_B = expand_B.reshape((-1,) + single_dim_B)
            flat_X = np.vstack([tri_solve(a, b, lower=(not upper), trans=int(trans), unit_diagonal=diag)
                                for a, b in zip(flat_A, flat_B)])
            return flat_X.reshape(expand_B.shape)

        def run_test(A_dims, b_dims, device, upper, transpose, unitriangular):
            b, A = self.triangular_solve_test_helper(A_dims, b_dims, upper,
                                                     unitriangular, device, dtype)
            x_exp = torch.as_tensor(scipy_tri_solve_batched(A.cpu().numpy(), b.cpu().numpy(),
                                                            upper, transpose, unitriangular))
            x = torch.triangular_solve(b, A, upper=upper, transpose=transpose, unitriangular=unitriangular)[0]

            self.assertEqual(x, x_exp.to(device))

        for upper, transpose, unitriangular in itertools.product([True, False], repeat=3):
            # test against scipy.linalg.solve_triangular
            run_test((2, 1, 3, 4, 4), (2, 1, 3, 4, 6), device, upper, transpose, unitriangular)  # no broadcasting
            run_test((2, 1, 3, 4, 4), (4, 6), device, upper, transpose, unitriangular)  # broadcasting b
            run_test((4, 4), (2, 1, 3, 4, 2), device, upper, transpose, unitriangular)  # broadcasting A
            run_test((1, 3, 1, 4, 4), (2, 1, 3, 4, 5), device, upper, transpose, unitriangular)  # broadcasting A & b

    @onlyCPU
    @skipCPUIfNoLapack
    @dtypes(torch.float32, torch.float64, torch.complex64, torch.complex128)
    def test_triangular_solve_singular(self, device, dtype):
        b = torch.rand(3, 1, dtype=dtype, device=device)
        A = torch.eye(3, 3, dtype=dtype, device=device)
        A[-1, -1] = 0  # Now A is singular
        err_str = r"triangular_solve_cpu: U\(3,3\) is zero, singular U\."
        with self.assertRaisesRegex(RuntimeError, err_str):
            torch.triangular_solve(b, A)

    def check_single_matmul(self, x, y, shape):
        a = np.array(x, copy=False)
        b = np.array(y, copy=False)
        expected = np.matmul(a, b)

        ans = torch.matmul(x, y)
        self.assertTrue(ans.is_contiguous())
        self.assertTrue(np.array_equal(ans, expected))

        out = torch.zeros(*shape, dtype=torch.int64).to(x.device)
        ans = torch.matmul(x, y, out=out)
        self.assertIs(ans, out)
        self.assertTrue(ans.is_contiguous())
        self.assertTrue(np.array_equal(ans, expected))

    # TODO: update to run on CUDA, too
    @onlyCPU
    def test_matmul_small_brute_force_1d_Nd(self, device):
        # Issue #20452: range(0, 10) does not work.
        n = 1
        for m in range(1, 8):
            for p in range(1, 8):
                for o in range(1, 5):
                    # 1d, 3d, inner dimensions C
                    x = torch.arange(m, device=device)
                    y = torch.arange(o * m * p, device=device).reshape(o, m, p)
                    self.check_single_matmul(x, y, (o, n, p))

                    # 1d, 3d, inner dimensions Fortran
                    x = torch.arange(m, device=device)
                    y = torch.arange(o * p * m, device=device).reshape(o, p, m).transpose(-1, -2)
                    self.check_single_matmul(x, y, (o, n, p))

                    # 1d, 3d, inner dimensions non-contiguous
                    x = torch.arange(2 * m, device=device)[::2]
                    y = torch.arange(o * m * 2 * p, device=device).reshape(o, m, 2 * p)[:, :, ::2]
                    self.check_single_matmul(x, y, (o, n, p))

                    for r in range(1, 5):
                        # 1d, 4d, inner dimensions C
                        x = torch.arange(m)
                        y = torch.arange(r * o * m * p, device=device).reshape(r, o, m, p)
                        self.check_single_matmul(x, y, (r, o, n, p))

                        # 1d, 4d, inner dimensions Fortran
                        x = torch.arange(m)
                        y = torch.arange(r * o * p * m, device=device).reshape(r, o, p, m).transpose(-1, -2)
                        self.check_single_matmul(x, y, (r, o, n, p))

                        # 1d, 4d, inner dimensions non-contiguous
                        x = torch.arange(2 * m, device=device)[::2]
                        y = torch.arange(r * o * m * 2 * p, device=device).reshape(r, o, m, 2 * p)[:, :, :, ::2]
                        self.check_single_matmul(x, y, (r, o, n, p))

    # TODO: update to run on CUDA, too
    @onlyCPU
    def test_matmul_small_brute_force_2d_Nd(self, device):
        # Issue #20452: range(0, 10) does not work.
        for n in range(1, 5):
            for m in range(1, 5):
                for p in range(1, 5):
                    for o in range(1, 3):
                        # 2d, 3d, inner dimensions C
                        x = torch.arange(n * m, device=device).reshape(n, m)
                        y = torch.arange(o * m * p, device=device).reshape(o, m, p)
                        self.check_single_matmul(x, y, (o, n, p))

                        # 2d, 3d, inner dimensions Fortran
                        x = torch.arange(m * n, device=device).reshape(m, n).transpose(-1, -2)
                        y = torch.arange(o * p * m, device=device).reshape(o, p, m).transpose(-1, -2)
                        self.check_single_matmul(x, y, (o, n, p))

                        # 2d, 3d, inner dimensions non-contiguous
                        x = torch.arange(n * 2 * m, device=device).reshape(n, 2 * m)[:, ::2]
                        y = torch.arange(o * m * 2 * p, device=device).reshape(o, m, 2 * p)[:, :, ::2]
                        self.check_single_matmul(x, y, (o, n, p))

                        for r in range(1, 2):
                            # 2d, 4d, inner dimensions C
                            x = torch.arange(n * m, device=device).reshape(n, m)
                            y = torch.arange(r * o * m * p, device=device).reshape(r, o, m, p)
                            self.check_single_matmul(x, y, (r, o, n, p))

                            # 2d, 4d, inner dimensions Fortran
                            x = torch.arange(m * n, device=device).reshape(m, n).transpose(-1, -2)
                            y = torch.arange(r * o * p * m, device=device).reshape(r, o, p, m).transpose(-1, -2)
                            self.check_single_matmul(x, y, (r, o, n, p))

                            # 2d, 4d, inner dimensions non-contiguous
                            x = torch.arange(n * 2 * m, device=device).reshape(n, 2 * m)[:, ::2]
                            y = torch.arange(r * o * m * 2 * p, device=device).reshape(r, o, m, 2 * p)[:, :, :, ::2]
                            self.check_single_matmul(x, y, (r, o, n, p))

    def test_linear_algebra_scalar_raises(self, device) -> None:
        m = torch.randn(5, 5, device=device)
        v = torch.randn(5, device=device)
        s = torch.tensor(7, device=device)
        self.assertRaises(RuntimeError, lambda: torch.mv(m, s))
        self.assertRaises(RuntimeError, lambda: torch.addmv(v, m, s))

    @onlyCPU
    @dtypes(torch.float)
    def test_cross(self, device, dtype):
        x = torch.rand(100, 3, 100, dtype=dtype, device=device)
        y = torch.rand(100, 3, 100, dtype=dtype, device=device)
        res1 = torch.cross(x, y)
        res2 = torch.tensor((), dtype=dtype, device=device)
        torch.cross(x, y, out=res2)
        self.assertEqual(res1, res2)

    @onlyCPU
    @dtypes(torch.float)
    def test_cross_with_and_without_dim(self, device, dtype):
        x = torch.rand(100, 3, dtype=dtype, device=device)
        y = torch.rand(100, 3, dtype=dtype, device=device)
        res1 = torch.cross(x, y, dim=1)
        res2 = torch.cross(x, y, dim=-1)
        res3 = torch.cross(x, y)
        self.assertEqual(res1, res2)
        self.assertEqual(res1, res3)

    def test_cross_errors(self, device):
        self.assertRaisesRegex(
            RuntimeError, "inconsistent tensors dimensions",
            lambda: torch.cross(torch.rand(100, 3, device=device), torch.rand(100, 3, 10, device=device)))
        self.assertRaisesRegex(
            RuntimeError, "inconsistent tensors sizes",
            lambda: torch.cross(torch.rand(5, 3, device=device), torch.rand(3, 5, device=device)))
        self.assertRaisesRegex(
            RuntimeError, "no dimension of size 3 in input",
            lambda: torch.cross(torch.rand(5, 4, device=device), torch.rand(5, 4, device=device)))
        self.assertRaisesRegex(
            RuntimeError, "dimension 0 does not have size 3",
            lambda: torch.cross(torch.rand(5, 4, 3, device=device), torch.rand(5, 4, 3, device=device), dim=0))
        self.assertRaisesRegex(
            RuntimeError, "dimension -1 does not have size 3",
            lambda: torch.cross(torch.rand(5, 3, 4, device=device), torch.rand(5, 3, 4, device=device), dim=-1))
        self.assertRaisesRegex(
            IndexError, "Dimension out of range",
            lambda: torch.cross(torch.rand(5, 3, 4, device=device), torch.rand(5, 3, 4, device=device), dim=-5))

    def test_renorm(self, device):
        m1 = torch.randn(10, 5, device=device)
        res1 = torch.tensor((), device=device)

        def renorm(matrix, value, dim, max_norm):
            m1 = matrix.transpose(dim, 0).contiguous()
            # collapse non-dim dimensions.
            m2 = m1.clone().resize_(m1.size(0), int(math.floor(m1.nelement() / m1.size(0))))
            norms = m2.norm(value, 1, True)
            # clip
            new_norms = norms.clone()
            new_norms[torch.gt(norms, max_norm)] = max_norm
            new_norms.div_(norms.add_(1e-7))
            # renormalize
            m1.mul_(new_norms.expand_as(m1))
            return m1.transpose(dim, 0)

        # note that the axis fed to torch.renorm is different (2~=1)
        maxnorm = m1.norm(2, 1).mean()
        m2 = renorm(m1, 2, 1, maxnorm)
        m1.renorm_(2, 1, maxnorm)
        self.assertEqual(m1, m2, atol=1e-5, rtol=0)
        self.assertEqual(m1.norm(2, 0), m2.norm(2, 0), atol=1e-5, rtol=0)

        m1 = torch.randn(3, 4, 5, device=device)
        m2 = m1.transpose(1, 2).contiguous().clone().resize_(15, 4)
        maxnorm = m2.norm(2, 0).mean()
        m2 = renorm(m2, 2, 1, maxnorm)
        m1.renorm_(2, 1, maxnorm)
        m3 = m1.transpose(1, 2).contiguous().clone().resize_(15, 4)
        self.assertEqual(m3, m2)
        self.assertEqual(m3.norm(2, 0), m2.norm(2, 0))

    # TODO: make this work on CUDA, too
    @onlyCPU
    @skipCPUIfNoLapack
    def test_ormqr(self, device):
        mat1 = torch.randn(7, 7)
        mat2 = torch.randn(7, 7)
        q, r = torch.qr(mat1)
        m, tau = torch.geqrf(mat1)
        out_holder = torch.empty_like(mat1)

        res1 = torch.mm(q, mat2)
        res2 = torch.ormqr(m, tau, mat2, left=True, transpose=False)
        torch.ormqr(m, tau, mat2, out=out_holder)
        self.assertEqual(res1, res2)
        self.assertEqual(res2, out_holder)

        res1 = torch.mm(mat2, q)
        res2 = torch.ormqr(m, tau, mat2, left=False, transpose=False)
        torch.ormqr(m, tau, mat2, left=False, transpose=False, out=out_holder)
        self.assertEqual(res1, res2)
        self.assertEqual(res2, out_holder)

        res1 = torch.mm(q.t(), mat2)
        res2 = torch.ormqr(m, tau, mat2, left=True, transpose=True)
        torch.ormqr(m, tau, mat2, left=True, transpose=True, out=out_holder)
        self.assertEqual(res1, res2)
        self.assertEqual(res2, out_holder)

        res1 = torch.mm(mat2, q.t())
        res2 = torch.ormqr(m, tau, mat2, left=False, transpose=True)
        torch.ormqr(m, tau, mat2, left=False, transpose=True, out=out_holder)
        self.assertEqual(res1, res2)
        self.assertEqual(res2, out_holder)

    @skipCUDAIfRocm
    def test_blas_empty(self, device):
        def fn(torchfn, *args, test_out=False, **kwargs):
            def call_torch_fn(*args, **kwargs):
                return torchfn(*tuple(torch.randn(shape, device=device) if isinstance(shape, tuple) else shape
                                      for shape in args), **kwargs)
            result = call_torch_fn(*args, **kwargs)
            if not test_out:
                return result
            else:
                out = torch.full_like(result, math.nan)
                out1 = call_torch_fn(*args, **kwargs, out=out)
                return out

        # mm, addmm
        self.assertEqual((0, 0), fn(torch.mm, (0, 0), (0, 0)).shape)
        self.assertEqual((0, 5), fn(torch.mm, (0, 0), (0, 5)).shape)
        self.assertEqual((5, 0), fn(torch.mm, (5, 0), (0, 0)).shape)
        self.assertEqual((3, 0), fn(torch.mm, (3, 2), (2, 0)).shape)
        self.assertEqual(torch.zeros((5, 6), device=device), fn(torch.mm, (5, 0), (0, 6)))
        self.assertEqual(torch.zeros((5, 6), device=device), fn(torch.mm, (5, 0), (0, 6), test_out=True))

        self.assertEqual((0, 0), fn(torch.addmm, (0, 0), (0, 0), (0, 0)).shape)
        self.assertEqual((0, 1), fn(torch.addmm, (1, ), (0, 17), (17, 1)).shape)
        t = torch.randn((5, 6), device=device)
        self.assertEqual(t, fn(torch.addmm, t, (5, 0), (0, 6)))
        self.assertEqual(t, fn(torch.addmm, t, (5, 0), (0, 6), test_out=True))

        # mv, addmv
        self.assertEqual((0,), fn(torch.mv, (0, 0), (0,)).shape)
        self.assertEqual((0,), fn(torch.mv, (0, 2), (2,)).shape)
        self.assertEqual(torch.zeros((3,), device=device), fn(torch.mv, (3, 0), (0,)))
        self.assertEqual(torch.zeros((3,), device=device), fn(torch.mv, (3, 0), (0,), test_out=True))

        self.assertEqual((0,), fn(torch.addmv, (0,), (0, 0), (0,)).shape)
        t = torch.randn((3,), device=device)
        self.assertEqual(t, fn(torch.addmv, t, (3, 0), (0,)))
        self.assertEqual(t, fn(torch.addmv, t, (3, 0), (0,), test_out=True))

        # bmm, baddbmm
        self.assertEqual((0, 0, 0), fn(torch.bmm, (0, 0, 0), (0, 0, 0)).shape)
        self.assertEqual((3, 0, 5), fn(torch.bmm, (3, 0, 0), (3, 0, 5)).shape)
        self.assertEqual((0, 5, 6), fn(torch.bmm, (0, 5, 0), (0, 0, 6)).shape)
        self.assertEqual(torch.zeros((3, 5, 6), device=device), fn(torch.bmm, (3, 5, 0), (3, 0, 6)))
        self.assertEqual(torch.zeros((3, 5, 6), device=device), fn(torch.bmm, (3, 5, 0), (3, 0, 6), test_out=True))

        self.assertEqual((0, 0, 0), fn(torch.baddbmm, (0, 0, 0), (0, 0, 0), (0, 0, 0)).shape)
        self.assertEqual((3, 0, 5), fn(torch.baddbmm, (3, 0, 5), (3, 0, 0), (3, 0, 5)).shape)
        self.assertEqual((0, 5, 6), fn(torch.baddbmm, (0, 5, 6), (0, 5, 0), (0, 0, 6)).shape)
        self.assertEqual((3, 5, 6), fn(torch.baddbmm, (3, 5, 6), (3, 5, 0), (3, 0, 6)).shape)
        c = torch.arange(30, dtype=torch.float32, device=device).reshape(3, 2, 5)
        self.assertEqual(-2 * c, fn(torch.baddbmm, c, (3, 2, 0), (3, 0, 5), beta=-2))  # Issue #33467
        self.assertEqual(-2 * c, fn(torch.baddbmm, c, (3, 2, 0), (3, 0, 5), beta=-2, test_out=True))  # Issue #33467

        # addbmm
        self.assertEqual((0, 0), fn(torch.addbmm, (0, 0), (0, 0, 0), (0, 0, 0)).shape)
        self.assertEqual((0, 5), fn(torch.addbmm, (0, 5), (3, 0, 0), (3, 0, 5)).shape)
        t = torch.randn((5, 6), device=device)
        self.assertEqual(t, fn(torch.addbmm, t, (0, 5, 0), (0, 0, 6)))
        self.assertEqual(t, fn(torch.addbmm, t, (0, 5, 0), (0, 0, 6), test_out=True))

        # matmul
        self.assertEqual(torch.tensor(0., device=device), fn(torch.matmul, (0,), (0,)))
        self.assertEqual(torch.tensor(0., device=device), fn(torch.matmul, (0,), (0,), test_out=True))
        self.assertEqual((0, 0), fn(torch.matmul, (0, 0), (0, 0)).shape)
        self.assertEqual((0, 0, 0), fn(torch.matmul, (0, 0, 0), (0, 0, 0)).shape)
        self.assertEqual((5, 0, 0), fn(torch.matmul, (5, 0, 0), (5, 0, 0)).shape)
        self.assertEqual(torch.zeros((5, 3, 4), device=device), fn(torch.matmul, (5, 3, 0), (5, 0, 4)))
        self.assertEqual(torch.zeros((5, 3, 4), device=device), fn(torch.matmul, (5, 3, 0), (5, 0, 4), test_out=True))

        # dot
        self.assertEqual(torch.tensor(0., device=device), fn(torch.dot, (0,), (0,)))
        self.assertEqual(torch.tensor(0., device=device), fn(torch.dot, (0,), (0,), test_out=True))

        if torch._C.has_lapack:
            # lu
            A_LU, pivots = fn(torch.lu, (0, 5, 5))
            self.assertEqual([(0, 5, 5), (0, 5)], [A_LU.shape, pivots.shape])
            A_LU, pivots = fn(torch.lu, (0, 0, 0))
            self.assertEqual([(0, 0, 0), (0, 0)], [A_LU.shape, pivots.shape])
            A_LU, pivots = fn(torch.lu, (2, 0, 0))
            self.assertEqual([(2, 0, 0), (2, 0)], [A_LU.shape, pivots.shape])

    @skipCUDAIfRocm
    @dtypesIfCUDA(torch.cfloat, torch.cdouble,
                  *torch.testing.get_all_fp_dtypes(include_half=not CUDA9, include_bfloat16=(CUDA11OrLater and SM53OrLater)))
    @dtypes(*(set(torch.testing.get_all_dtypes()) - {torch.half, torch.bool}))
    def test_blas_alpha_beta_empty(self, device, dtype):
        # This test is disabled on CUDA 9 due to:
        # See: https://github.com/pytorch/pytorch/issues/31006
        if dtype is torch.bfloat16 and self.device_type == 'xla':
            # TODO (@zasdfgbnm): this causes the following error on test
            # TestTorchDeviceTypeXLA.test_blas_alpha_beta_empty_xla_bfloat16:
            #
            #   RuntimeError: _th_equal not supported on CPUType for BFloat16
            return
        # ensure beta is respected
        value = 11
        input = torch.full((2,), value, dtype=dtype, device=device)
        mat = torch.ones((2, 0), dtype=dtype, device=device)
        vec = torch.ones((0,), dtype=dtype, device=device)
        out = torch.empty((2,), dtype=dtype, device=device)
        if dtype.is_complex:
            alpha = 6 + 7j
            beta = 3 + 4j
        else:
            alpha = 6
            beta = 3
        self.assertEqual(torch.full((2,), beta * value, dtype=dtype, device=device),
                         torch.addmv(input=input, mat=mat, vec=vec, alpha=alpha, beta=beta))
        self.assertEqual(torch.full((2,), beta * value, dtype=dtype, device=device),
                         torch.addmv(input=input, mat=mat, vec=vec, alpha=alpha, beta=beta, out=out))

        # torch.addmm
        input = torch.full((2, 3), value, dtype=dtype, device=device)
        mat2 = torch.ones((0, 3), dtype=dtype, device=device)
        out = torch.empty((2, 3), dtype=dtype, device=device)
        self.assertEqual(torch.full((2, 3), beta * value, dtype=dtype, device=device),
                         torch.addmm(input=input, mat1=mat, mat2=mat2, alpha=alpha, beta=beta))
        self.assertEqual(torch.full((2, 3), beta * value, dtype=dtype, device=device),
                         torch.addmm(input=input, mat1=mat, mat2=mat2, alpha=alpha, beta=beta, out=out))

    @dtypes(*(torch.testing.get_all_complex_dtypes() + torch.testing.get_all_fp_dtypes()))
    def test_blas_nan_out(self, device, dtype):
        # These functions should work correctly with NaN filled outputs,
        # but need special handling, see [NOTE: cpu_zero]
        b = 3
        n = 5
        m = 7
        p = 11

        # torch.mv
        nm = torch.randn((m, n), device=device).t()
        _m = torch.randn((), device=device).expand(m)
        _m_out = torch.full((m,), float('nan'), device=device)
        self.assertEqual(torch.mv(nm, _m), torch.mv(nm, _m, out=_m_out))
        self.assertEqual(0, torch.isnan(torch.mv(nm, _m)).sum())

        # torch.mm
        mp = torch.randn((p, m), device=device).t()
        np_out = torch.full((n, p), float('nan'), device=device)
        self.assertEqual(torch.mm(nm, mp), torch.mm(nm, mp, out=np_out))

        # torch.bmm
        bnm = torch.randn((b, m, n), device=device).transpose(1, 2)
        bmp = torch.randn((b, p, m), device=device).transpose(1, 2)
        bnp_out = torch.full((b, n, p), float('nan'), device=device)
        self.assertEqual(torch.bmm(bnm, bmp), torch.bmm(bnm, bmp, out=bnp_out))

    @onlyCPU  # not supported by CUBLAS
    def test_blas_mv_large_input(self, device):
        # This would previously fail if the allocated output had NaNs, see:
        # https://github.com/pytorch/pytorch/issues/31663 and [NOTE: cpu_zero]
        n = 3000
        m = 200

        nm = torch.randn((m, n), device=device).t()
        _m = torch.randn((), device=device).expand(m)
        _m_out = torch.full((m,), 0., device=device)

        self.assertEqual(torch.mv(nm, _m), torch.mv(nm, _m, out=_m_out))

    @onlyCPU
    def test_renorm_ps(self, device):
        # full reduction
        x = torch.randn(5, 5)
        xn = x.numpy()
        for p in [1, 2, 3, 4, inf]:
            res = x.renorm(p, 1, 1)
            expected = x / x.norm(p, 0, keepdim=True).clamp(min=1)
            self.assertEqual(res, expected, msg="renorm failed for {}-norm".format(p))

    @onlyCPU
    @skipCPUIfNoLapack
    @dtypes(torch.float32, torch.float64, torch.complex64, torch.complex128)
    def test_orgqr(self, device, dtype):
        def generate_reflectors_and_tau(A):
            """
            This function uses numpy.linalg.qr with mode "raw" to extract output of LAPACK's geqrf.
            There is torch.geqrf function but it doesn't work with complex-valued input.
            """
            if A.numel() > 0:
                A_cpu = A.cpu()
                flattened_batch_shape = [-1, *A_cpu.shape[-2:]]
                reflectors = torch.empty_like(A_cpu).view(*flattened_batch_shape)
                tau_shape = [*A_cpu.shape[:-2], A_cpu.shape[-1]]
                tau = torch.empty(tau_shape, dtype=dtype).view(-1, A_cpu.shape[-1])
                for A_i, reflectors_i, tau_i in zip(A_cpu.contiguous().view(*flattened_batch_shape), reflectors, tau):
                    reflectors_tmp, tau_i[:] = map(torch.from_numpy, np.linalg.qr(A_i, mode='raw'))
                    reflectors_i[:] = reflectors_tmp.T
                reflectors = reflectors.view(*A_cpu.shape)
                tau = tau.view(tau_shape)
                return reflectors.to(A.device), tau.to(A.device)

            reflectors = torch.empty_like(A)
            tau = torch.empty(*A.shape[:-2], A.shape[-1], dtype=dtype, device=device)
            return reflectors, tau

        def run_test(shape):
            A = torch.randn(*shape, dtype=dtype, device=device)
            reflectors, tau = generate_reflectors_and_tau(A)
            expected, _ = torch.linalg.qr(A)
            actual = torch.orgqr(reflectors, tau)
            # torch.linalg.qr does not work correctly for zero batch dimension tensors
            # see https://github.com/pytorch/pytorch/issues/50576
            if (A.numel() > 0):
                self.assertEqual(expected, actual)
            else:
                self.assertTrue(actual.shape == shape)

            out = torch.empty_like(A)
            ans = torch.orgqr(reflectors, tau, out=out)
            self.assertEqual(ans, out)
            if (A.numel() > 0):
                self.assertEqual(expected, out)

        shapes = [(0, 0), (5, 0),  # Empty matrix
                  (5, 5), (5, 3),  # Single matrix
                  (0, 0, 0), (0, 5, 5), (0, 5, 3),  # Zero batch dimension tensors
                  (2, 5, 5), (2, 5, 3),  # 3-dim tensors
                  (2, 1, 5, 5), (2, 1, 5, 3)]  # 4-dim tensors
        for shape in shapes:
            run_test(shape)

    @onlyCPU
    @skipCPUIfNoLapack
    def test_orgqr_errors_and_warnings(self, device):
        test_cases = [
            # input1 size, input2 size, error regex
            ((10,), (2,), r"input must have at least 2 dimensions"),
            ((10, 6), (20,), r"input.shape\[-1\] must be greater than or equal to tau.shape\[-1\]"),
            ((6, 10), (5,), r"input.shape\[-2\] must be greater than or equal to input.shape\[-1\]"),
        ]
        for a_size, tau_size, error_regex in test_cases:
            a = torch.rand(*a_size, device=device)
            tau = torch.rand(*tau_size, device=device)
            with self.assertRaisesRegex(RuntimeError, error_regex):
                torch.orgqr(a, tau)

        # if out tensor with wrong shape is passed a warning is given
        reflectors = torch.randn(3, 3, device=device)
        tau = torch.randn(3, device=device)
        out = torch.empty(2, 3, device=device)
        with warnings.catch_warnings(record=True) as w:
            # Trigger warning
            torch.orgqr(reflectors, tau, out=out)
            # Check warning occurs
            self.assertEqual(len(w), 1)
            self.assertTrue("An output with one or more elements was resized" in str(w[-1].message))

        # dtypes should match
        out = torch.empty_like(reflectors).to(torch.int)
        with self.assertRaisesRegex(RuntimeError, "result dtype Int does not match the expected dtype"):
            torch.orgqr(reflectors, tau, out=out)

        with self.assertRaisesRegex(RuntimeError, "tau dtype Int does not match input dtype"):
            torch.orgqr(reflectors, tau.to(torch.int))

        # TODO: enable the following tests when orgqr is implemented for CUDA
        if torch.cuda.is_available():
            with self.assertRaisesRegex(RuntimeError, "the operator doesn't exist for this backend"):
                # device of out and input should match
                wrong_device = 'cpu' if self.device_type != 'cpu' else 'cuda'
                out = torch.empty_like(reflectors).to(wrong_device)
                # with self.assertRaisesRegex(RuntimeError, "Expected result and input to be on the same device"):
                torch.orgqr(reflectors, tau, out=out)

                # device of tau and input should match
                wrong_device = 'cpu' if self.device_type != 'cpu' else 'cuda'
                tau = tau.to(wrong_device)
                # with self.assertRaisesRegex(RuntimeError, "Expected input and tau to be on the same device"):
                torch.orgqr(reflectors, tau)

    @precisionOverride({torch.complex64: 5e-6})
    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(torch.double, torch.cfloat, torch.cdouble)
    @skipCUDAIfRocm
    def test_lu(self, device, dtype):
        from torch.testing._internal.common_utils import random_matrix

        def run_test(device, pivot):
            def run_subtest(matrix_size, batches, device, pivot, singular=False, a=None):
                if isinstance(matrix_size, int):
                    rows = columns = matrix_size
                else:
                    rows, columns = matrix_size
                if a is None:
                    a = random_matrix(rows, columns, *batches, **dict(singular=singular, dtype=dtype)).to(device)
                a_LU_info, pivots_info, info_ = a.lu(pivot=pivot, get_infos=True)
                self.assertEqual(a_LU_info.size(), torch.Size(batches + (rows, columns)))
                self.assertEqual(pivots_info.size(), torch.Size(batches + (min(rows, columns),)))
                self.assertEqual(info_.size(), torch.Size(batches))
                # If a randomly generated input matrix is singular,
                # then info_ contains indices i such that U[i, i] ==
                # 0. This however conveys that the factorization was
                # successful albeit with a singular input. Therefore,
                # we require info.min() >= 0
                self.assertGreaterEqual(info_.min(), 0)
                a_LU, pivots = a.lu(pivot=pivot)
                self.assertEqual(a_LU, a_LU_info)
                self.assertEqual(pivots_info, pivots)


                P, L, U = torch.lu_unpack(a_LU, pivots)
                P_ = P.cpu().numpy()
                L_ = L.cpu().numpy()
                U_ = U.cpu().numpy()

                self.assertEqual(np.matmul(P_, np.matmul(L_, U_)), a)

                if self.device_type == 'cuda':
                    # lu without pivoting is implemented only for cuda device
                    a_LU_info_nopiv, nopiv, info_nopiv = a.lu(pivot=False, get_infos=True)
                    P_nopiv, L_nopiv, U_nopiv = torch.lu_unpack(a_LU_info_nopiv, nopiv)
                    P_nopiv_ = P_nopiv.cpu().numpy()
                    L_nopiv_ = L_nopiv.cpu().numpy()
                    U_nopiv_ = U_nopiv.cpu().numpy()

                    self.assertEqual(np.matmul(P_nopiv_, np.matmul(L_nopiv_, U_nopiv_)), a)

                    k = min(rows, columns)
                    self.assertEqual(nopiv, torch.arange(1, 1 + k, device=device, dtype=torch.int32).expand(a.shape[:-2] + (k, )))
                    if not singular:
                        # It is not guaranteed that LU factorization
                        # without pivoting is able to determine if a
                        # matrix is singular while LU factorization
                        # with pivoting is. Therefore, we require the
                        # equality of info-s only for non-singular
                        # matrices.
                        # NOTE: infor_ is reshaped because info_nopiv might have
                        # squashed batch dimensions for complex types on CUDA,
                        # see the TODOs above.
                        self.assertEqual(info_.reshape(info_nopiv.shape), info_nopiv)

            for ms, batch in itertools.product([3, 5, 7, (4, 2), (3, 4)], [(), (2,), (3,), (3, 5)]):
                run_subtest(ms, batch, device, pivot)
                run_subtest(ms, batch, device, pivot, singular=True)

                # Reproducer of a magma bug, see https://bitbucket.org/icl/magma/issues/13/getrf_batched-kernel-produces-nans-on
                a = torch.ones(batch + (ms if isinstance(ms, tuple) else (ms, ms)), dtype=torch.double, device=device)
                run_subtest(ms, batch, device, pivot, singular=True, a=a)

            # Info should be positive for rank deficient matrices
            a = torch.ones(5, 3, 3, device=device)
            self.assertGreater(a.lu(pivot=pivot, get_infos=True)[2][0], 0)

        run_test(device, True)

        if self.device_type == 'cpu':
            # Error checking, no pivoting variant on CPU
            with self.assertRaisesRegex(RuntimeError, 'lu without pivoting is not implemented on the CPU'):
                torch.lu(torch.empty(1, 2, 2), pivot=False)
        else:
            run_test(device, False)

    @skipCPUIfNoLapack
    @skipCUDAIfNoMagma
    @dtypes(torch.double)
    @skipCUDAIfRocm
    def test_lu_unpack(self, device, dtype):
        def run_test(pivot):
            for shape in ((3, 3), (5, 3, 3), (7, 3, 5, 5), (7, 5, 3, 3, 3)):
                a = torch.randn(*shape, dtype=dtype, device=device)
                a_lu, p = torch.lu(a, pivot=pivot)
                p_ref, l_ref, u_ref = torch.lu_unpack(a_lu, p)
                self.assertEqual(p_ref.matmul(l_ref.matmul(u_ref)), a)

        run_test(True)

        if self.device_type == 'cuda':
            run_test(False)

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(torch.double)
    @skipCUDAIfRocm
    def test_lobpcg_basic(self, device, dtype):
        self._test_lobpcg_method(device, dtype, 'basic')

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(torch.double)
    @skipCUDAIfRocm
    def test_lobpcg_ortho(self, device, dtype):
        self._test_lobpcg_method(device, dtype, 'ortho')

    def _test_lobpcg_method(self, device, dtype, method):
        from torch.testing._internal.common_utils import random_symmetric_pd_matrix, random_sparse_pd_matrix
        from torch._linalg_utils import matmul, qform
        from torch._lobpcg import lobpcg

        def test_tracker(worker):
            k = worker.iparams['k']
            nc = worker.ivars['converged_count']
            if k <= nc:
                tol = worker.fparams['tol']
                rerr = worker.tvars['rerr']
                X = worker.X
                E = worker.E
                B = worker.B
                A = worker.A
                dtype = X.dtype
                device = X.device

                # Check convergence
                self.assertLessEqual(rerr[:k].max(), tol)

                # Check B-orthogonality
                I = torch.eye(k, k, dtype=dtype, device=device)
                self.assertEqual(qform(B, X[:, :k]), I)

                # Check block equation
                self.assertEqual(qform(A, X[:, :k]) / E[:k], I, atol=0.2, rtol=0)

        orig_lobpcg = lobpcg

        def lobpcg(*args, **kwargs):
            kwargs['tracker'] = test_tracker
            kwargs['niter'] = 1000
            kwargs['method'] = method
            kwargs['tol'] = 1e-8
            return orig_lobpcg(*args, **kwargs)
        prec = 5e-4

        # check dense input
        mm = torch.matmul
        for batches in [(), (2,), (2, 3)]:
            for m, n, k in [
                    (9, 3, 1),
                    (9, 3, 2),
                    (9, 2, 2),
                    (100, 15, 5),
            ]:
                # skip tests that are known to fail with the basic
                # LOBPCG method due to calling cholesky on singular
                # input
                if method == 'basic' and (m, n, k) in [(9, 2, 2), (100, 15, 5)]:
                    continue
                A = random_symmetric_pd_matrix(m, *batches, device=device, dtype=dtype)
                B = random_symmetric_pd_matrix(m, *batches, device=device, dtype=dtype)

                # classical eigenvalue problem, smallest eigenvalues
                E, V = lobpcg(A, k=k, n=n, largest=False)
                self.assertEqual(E.shape, batches + (k,))
                self.assertEqual(V.shape, batches + (m, k))
                self.assertEqual(matmul(A, V), mm(V, E.diag_embed()), atol=prec, rtol=0)
                e = torch.symeig(A)[0]
                e_smallest = e[..., :k]
                self.assertEqual(E, e_smallest)

                # classical eigenvalue problem, largest eigenvalues
                E, V = lobpcg(A, k=k, n=n, largest=True)
                e_largest, _ = torch.sort(e[..., -k:], descending=True)
                self.assertEqual(E, e_largest, atol=prec, rtol=0)
                self.assertEqual(matmul(A, V), mm(V, E.diag_embed()), atol=prec, rtol=0)

                # generalized eigenvalue problem, smallest eigenvalues
                E, V = lobpcg(A, B=B, k=k, n=n, largest=False)
                self.assertEqual(matmul(A, V), mm(matmul(B, V), E.diag_embed()), atol=prec, rtol=0)

                # generalized eigenvalue problem, largest eigenvalues
                E, V = lobpcg(A, B=B, k=k, n=n, largest=True)
                self.assertEqual(matmul(A, V) / E.max(), mm(matmul(B, V), (E / E.max()).diag_embed()),
                                 atol=prec, rtol=0)

        # check sparse input
        for m, n, k, density in [
                (5, 1, 1, 0.8),
                (9, 3, 2, 0.5),
                (100, 1, 1, 0.1),
                (1000, 7, 3, 0.01),
        ]:
            # skip tests that are known to fail with the basic LOBCG
            # method due to insufficient accuracy
            if method == 'basic' and (m, n, k, density) in [(1000, 7, 3, 0.01)]:
                continue
            A = random_sparse_pd_matrix(m, density=density, device=device, dtype=dtype)
            B = random_sparse_pd_matrix(m, density=density, device=device, dtype=dtype)
            A_eigenvalues = torch.arange(1, m + 1, dtype=dtype) / m
            e_smallest = A_eigenvalues[..., :k]
            e_largest, _ = torch.sort(A_eigenvalues[..., -k:], descending=True)

            # classical eigenvalue problem, smallest eigenvalues
            E, V = lobpcg(A, k=k, n=n, largest=False)
            self.assertEqual(E, e_smallest)
            self.assertEqual(matmul(A, V), mm(V, E.diag_embed()), atol=prec, rtol=0)

            # classical eigenvalue problem, largest eigenvalues
            E, V = lobpcg(A, k=k, n=n, largest=True)
            self.assertEqual(matmul(A, V), mm(V, E.diag_embed()), atol=prec, rtol=0)
            self.assertEqual(E, e_largest)

            # generalized eigenvalue problem, smallest eigenvalues
            E, V = lobpcg(A, B=B, k=k, n=n, largest=False)
            self.assertEqual(matmul(A, V), matmul(B, mm(V, E.diag_embed())), atol=prec, rtol=0)

            # generalized eigenvalue problem, largest eigenvalues
            E, V = lobpcg(A, B=B, k=k, n=n, largest=True)
            self.assertEqual(matmul(A, V) / E.max(), mm(matmul(B, V), (E / E.max()).diag_embed()),
                             atol=prec, rtol=0)

    @skipCPUIfNoLapack
    @onlyCPU
    @dtypes(torch.double)
    def test_lobpcg_torchscript(self, device, dtype):
        from torch.testing._internal.common_utils import random_sparse_pd_matrix
        from torch._linalg_utils import matmul as mm

        lobpcg = torch.jit.script(torch.lobpcg)

        m = 500
        k = 5
        A1 = random_sparse_pd_matrix(m, density=2.0 / m, device=device, dtype=dtype)
        X1 = torch.randn((m, k), dtype=dtype, device=device)
        E1, V1 = lobpcg(A1, X=X1)
        eq_err = torch.norm((mm(A1, V1) - V1 * E1), 2) / E1.max()
        self.assertLess(eq_err, 1e-6)

    @unittest.skipIf(not TEST_SCIPY or (TEST_SCIPY and scipy.__version__ < '1.4.1'), "Scipy not found or older than 1.4.1")
    @skipCPUIfNoLapack
    @onlyCPU
    @dtypes(torch.double)
    def test_lobpcg_scipy(self, device, dtype):
        """Compare torch and scipy.sparse.linalg implementations of lobpcg
        """
        import time
        from torch.testing._internal.common_utils import random_sparse_pd_matrix
        from torch._linalg_utils import matmul as mm
        from scipy.sparse.linalg import lobpcg as scipy_lobpcg
        import scipy.sparse

        def toscipy(A):
            if A.layout == torch.sparse_coo:
                values = A.coalesce().values().cpu().numpy().copy()
                indices = A.coalesce().indices().cpu().numpy().copy()
                return scipy.sparse.coo_matrix((values, (indices[0], indices[1])), A.shape)
            return A.cpu().numpy().copy()

        niter = 1000
        repeat = 10
        m = 500   # size of the square matrix
        k = 7     # the number of requested eigenpairs
        A1 = random_sparse_pd_matrix(m, density=2.0 / m, device=device, dtype=dtype)
        B1 = random_sparse_pd_matrix(m, density=2.0 / m, device=device, dtype=dtype)
        X1 = torch.randn((m, k), dtype=dtype, device=device)

        A2 = toscipy(A1)
        B2 = toscipy(B1)
        X2 = toscipy(X1)

        lambdas1 = []

        def tracker(worker):
            lambdas1.append(worker.E[:])

        tol = 1e-8
        # tol for scipy lobpcg will be choosed so that the number of
        # iterations will be equal or very close to pytorch lobpcg
        # (that is around 170-180)

        # Standard eigenvalue problem
        E1, V1 = torch.lobpcg(A1, X=X1, niter=niter, largest=True, tracker=tracker, tol=tol)
        E2, V2, lambdas2 = scipy_lobpcg(A2, X2, maxiter=niter, largest=True, retLambdaHistory=True, tol=1.1 * tol)
        iters1 = len(lambdas1)
        iters2 = len(lambdas2)
        self.assertLess(abs(iters1 - iters2), 0.05 * max(iters1, iters2))

        E2a, V2a = scipy_lobpcg(A2, X2, maxiter=niter, largest=False)

        eq_err = torch.norm((mm(A1, V1) - V1 * E1), 2) / E1.max()
        eq_err_scipy = (abs(A2.dot(V2) - V2 * E2)**2).sum() ** 0.5 / E2.max()
        self.assertLess(eq_err, 1e-6)        # std
        self.assertLess(eq_err_scipy, 1e-6)  # std

        self.assertEqual(E1, torch.from_numpy(E2.copy()))

        # Generalized eigenvalue problem
        lambdas1 = []

        def tracker(worker):
            lambdas1.append(worker.E[:])

        E1, V1 = torch.lobpcg(A1, B=B1, X=X1, niter=niter, largest=True, tracker=tracker, tol=tol)
        E2, V2, lambdas2 = scipy_lobpcg(A2, X2, B=B2, maxiter=niter, largest=True, retLambdaHistory=True, tol=39 * tol)
        E2a, V2a = scipy_lobpcg(A2, X2, B=B2, maxiter=niter, largest=False)
        iters1 = len(lambdas1)
        iters2 = len(lambdas2)
        self.assertLess(abs(iters1 - iters2), 0.05 * max(iters1, iters2))

        eq_err = torch.norm((mm(A1, V1) - mm(B1, V1) * E1), 2) / E1.max()
        eq_err_scipy = (abs(A2.dot(V2) - B2.dot(V2) * E2)**2).sum() ** 0.5 / E2.max()
        self.assertLess(eq_err, 1e-6)        # general
        self.assertLess(eq_err_scipy, 1e-6)  # general

        self.assertEqual(E1, torch.from_numpy(E2.copy()))

        # Timings
        elapsed_ortho = 0
        elapsed_ortho_general = 0
        elapsed_scipy = 0
        elapsed_general_scipy = 0
        for i in range(repeat):
            start = time.time()
            torch.lobpcg(A1, X=X1, niter=niter, method='ortho', tol=tol)
            end = time.time()
            elapsed_ortho += end - start

            start = time.time()
            torch.lobpcg(A1, X=X1, B=B1, niter=niter, method='ortho', tol=tol)
            end = time.time()
            elapsed_ortho_general += end - start

            start = time.time()
            scipy_lobpcg(A2, X2, maxiter=niter, tol=1.1 * tol)
            end = time.time()
            elapsed_scipy += end - start

            start = time.time()
            scipy_lobpcg(A2, X2, B=B2, maxiter=niter, tol=39 * tol)
            end = time.time()
            elapsed_general_scipy += end - start

        elapsed_ortho_ms = 1000.0 * elapsed_ortho / repeat
        elapsed_ortho_general_ms = 1000.0 * elapsed_ortho_general / repeat
        elapsed_scipy_ms = 1000.0 * elapsed_scipy / repeat
        elapsed_general_scipy_ms = 1000.0 * elapsed_general_scipy / repeat

        print('''
CPU timings: torch.lobpcg vs scipy.sparse.linalg.lobpcg
-------------------------------------------------------
              | standard    | generalized | method
torch.lobpcg  | {:10.2f}  | {:10.2f}  | ortho
scipy_lobpcg  | {:10.2f}  | {:10.2f}  | N/A
-(input size: {:4}, eigenpairs:{:2}, units: ms per call)-
        '''.format(elapsed_ortho_ms, elapsed_ortho_general_ms,
                   elapsed_scipy_ms, elapsed_general_scipy_ms,
                   m, k))

        # Handling of very small tolerence
        tol = 1e-100

        lambdas1 = []

        def tracker(worker):
            lambdas1.append(worker.E[:])

        E1, V1 = torch.lobpcg(A1, X=X1, niter=niter, largest=True, tracker=tracker, tol=tol)
        iters1 = len(lambdas1)
        eq_err = torch.norm((mm(A1, V1) - V1 * E1), 2) / E1.max()

        try:
            E2, V2, lambdas2 = scipy_lobpcg(A2, X2, maxiter=niter, largest=True, retLambdaHistory=True, tol=tol)
            iters2 = len(lambdas2)
            eq_err_scipy = (abs(A2.dot(V2) - V2 * E2)**2).sum() ** 0.5 / E2.max()
        except Exception as msg:
            print('Calling scipy_lobpcg failed [standard]:', msg)
            iters2 = -1
            eq_err_scipy = -1

        lambdas1 = []

        def tracker(worker):
            lambdas1.append(worker.E[:])

        E1, V1 = torch.lobpcg(A1, X=X1, B=B1, niter=niter, largest=True, tracker=tracker, tol=tol)
        iters1_general = len(lambdas1)
        eq_err_general = torch.norm((mm(A1, V1) - mm(B1, V1) * E1), 2) / E1.max()

        try:
            E2, V2, lambdas2 = scipy_lobpcg(A2, X2, B=B2, maxiter=niter, largest=True, retLambdaHistory=True, tol=tol)
            iters2_general = len(lambdas2)
            eq_err_general_scipy = (abs(A2.dot(V2) - B2.dot(V2) * E2)**2).sum() ** 0.5 / E2.max()
        except Exception as msg:
            print('Calling scipy_lobpcg failed [generalized]:', msg)
            iters2_general = -1
            eq_err_general_scipy = -1

        print('''\
Handling of small tol={:6.0e}: torch.lobpcg vs scipy.sparse.linalg.lobpcg
----------------------------------------------------------------------------
              | standard    | generalized |  niter | method
torch.lobpcg  | {:10.2e}  | {:10.2e}  | {:6} | ortho
scipy_lobpcg  | {:10.2e}  | {:10.2e}  | {:6} | N/A
---(input size: {:4}, eigenpairs:{:2}, units: relative error, maxiter={:4})---
'''.format(tol, eq_err, eq_err_general, iters1, eq_err_scipy, eq_err_general_scipy, iters2, m, k, niter))

    def _test_addmm_addmv(self, f, t, m, v, *, alpha=None, beta=None, transpose_out=False):
        dtype = t.dtype
        numpy_dtype = dtype
        if dtype in {torch.bfloat16}:
            numpy_dtype = torch.float
        if dtype.is_complex:
            alpha = 0.9 + 0.3j if alpha is None else alpha
            beta = 0.5 + 0.6j if beta is None else beta
        else:
            alpha = 1.2 if alpha is None else alpha
            beta = 0.8 if beta is None else beta
        res1 = f(t, m, v, alpha=alpha, beta=beta)
        res2 = torch.full_like(res1, math.nan)
        if transpose_out:
            res2 = res2.t().clone(memory_format=torch.contiguous_format).t()
        f(t, m, v, alpha=alpha, beta=beta, out=res2)
        res3 = alpha * (m.to(numpy_dtype).cpu().numpy() @ v.to(numpy_dtype).cpu().numpy())
        if beta != 0:
            res3 += (beta * t).to(numpy_dtype).cpu().numpy()
        res3 = torch.from_numpy(res3).to(dtype)
        self.assertEqual(res1, res2)
        self.assertEqual(res1, res3)

    @precisionOverride({torch.bfloat16: 1e-0, torch.half: 5e-4, torch.float: 1e-4, torch.double: 1e-8,
                        torch.cfloat: 1e-4, torch.cdouble: 1e-8})
    @dtypesIfCUDA(*torch.testing.get_all_complex_dtypes(),
                  *torch.testing.get_all_fp_dtypes(include_bfloat16=(TEST_WITH_ROCM or (CUDA11OrLater and SM53OrLater)),
                                                   include_half=(not TEST_WITH_ROCM)))
    @dtypes(torch.bfloat16, torch.float, torch.double, torch.cfloat, torch.cdouble)
    def test_addmv(self, device, dtype):
        # have to use torch.randn(...).to(bfloat16) instead of
        # torch.randn(..., dtype=bfloat16). randn does not support
        # bfloat16 yet.
        ts = [
            torch.randn(10, device=device).to(dtype),
            torch.randn(1, device=device).to(dtype).expand(10),
        ]
        vs = [
            torch.randn(100, device=device).to(dtype),
            torch.ones(1, device=device).to(dtype).expand(100),  # to reduce errors for low precision
        ]
        ms = [
            # 0d
            torch.ones((), device=device).to(dtype).expand(10, 100),  # to reduce errors for low precision
            # 1d
            torch.randn((1, 100), device=device).to(dtype).expand(10, 100),
            # this initialization reduces errors for low precision for broadcasted matrices
            # by making sure that intermediate and result values are exactly representable
            # in low precision type
            torch.randint(3, (10, 1), dtype=torch.float, device=device).to(dtype).expand(10, 100),
            # 2d
            torch.randn((10, 100), device=device).to(dtype),
            torch.randn((100, 10), device=device).to(dtype).t(),
        ]
        for m, v, t in itertools.product(ms, vs, ts):
            self._test_addmm_addmv(torch.addmv, t, m, v)
        # Test beta=0, t=nan
        t = torch.full((10,), math.nan, device=device).to(dtype)
        for m, v in itertools.product(ms, vs):
            self._test_addmm_addmv(torch.addmv, t, m, v, beta=0)

    @dtypesIfCUDA(*torch.testing.get_all_fp_dtypes(include_bfloat16=(TEST_WITH_ROCM or (CUDA11OrLater and SM53OrLater))))
    @dtypes(torch.float, torch.double)
    def test_addmv_rowmajor_colmajor_incx_incy_lda(self, device, dtype):
        # tests (o, s)*(s).  o is output size, s is summed size.
        o = 5
        s = 3
        a_data = torch.arange(1, o * s + 1, device=device, dtype=dtype).view(o, s)
        x_data = torch.arange(1, s + 1, 1, device=device, dtype=dtype)
        y_data = torch.ones(o, device=device, dtype=dtype)
        control = torch.tensor([15., 33., 51., 69., 87.], device=device, dtype=dtype)

        def _test(row_major, incx, incy, lda_tail):
            if row_major:
                a_storage = torch.full((o, s + lda_tail), float('nan'), device=device, dtype=dtype)
            else:
                a_storage = torch.full((s, o + lda_tail), float('nan'), device=device, dtype=dtype).permute(1, 0)
            a = a_storage[:o, :s].copy_(a_data)

            x_storage = torch.full((s, incx), float('nan'), device=device, dtype=dtype)
            x = x_storage[:, 0].copy_(x_data)

            y_storage = torch.full((o, incy), float('nan'), device=device, dtype=dtype)
            y = y_storage[:, 0].copy_(y_data)

            self._test_addmm_addmv(torch.addmv, y, a, x)

        for row_major, incx, incy, lda_tail in itertools.product((False, True), (1, 2), (1, 2), (0, 1)):
            _test(row_major, incx, incy, lda_tail)

    @precisionOverride({torch.double: 1e-8, torch.float: 1e-4, torch.bfloat16: 0.6,
                        torch.half: 1e-1, torch.cfloat: 1e-4, torch.cdouble: 1e-8})
    @dtypesIfCUDA(*torch.testing.get_all_complex_dtypes(),
                  *torch.testing.get_all_fp_dtypes(include_bfloat16=(TEST_WITH_ROCM or (CUDA11OrLater and SM53OrLater))))
    @dtypes(*torch.testing.get_all_complex_dtypes(), *torch.testing.get_all_fp_dtypes())
    @tf32_on_and_off(0.05)
    def test_addmm(self, device, dtype):
        M = torch.randn(10, 25, device=device).to(dtype)
        m1 = torch.randn(10, 50, device=device).to(dtype)
        m2 = torch.randn(50, 25, device=device).to(dtype)
        self._test_addmm_addmv(torch.addmm, M, m1, m2)

        # Test 0-strided
        M = torch.randn(10, 1, device=device).to(dtype).expand(10, 25)
        m1 = torch.randn(10, 1, device=device).to(dtype).expand(10, 50)
        m2 = torch.randn(50, 25, device=device).to(dtype)
        self._test_addmm_addmv(torch.addmm, M, m1, m2)

        # Test beta=0, M=nan
        M = torch.full((10, 25), math.nan, device=device).to(dtype)
        m1 = torch.randn(10, 50, device=device).to(dtype)
        m2 = torch.randn(50, 25, device=device).to(dtype)
        self._test_addmm_addmv(torch.addmm, M, m1, m2, beta=0)

        # Test transpose
        for t1, t2, t3, t4 in itertools.product([True, False], repeat=4):
            def maybe_transpose(cond, m):
                if not cond:
                    return m
                return m.t().clone(memory_format=torch.contiguous_format).t()

            M = maybe_transpose(t1, torch.randn(10, 25, device=device).to(dtype))
            m1 = maybe_transpose(t2, torch.randn(10, 50, device=device).to(dtype))
            m2 = maybe_transpose(t3, torch.randn(50, 25, device=device).to(dtype))
            self._test_addmm_addmv(torch.addmm, M, m1, m2, transpose_out=t4)

    @dtypes(torch.float, torch.double)
    @dtypesIfCUDA(*([torch.float, torch.double] +
                    ([] if TEST_WITH_ROCM else torch.testing.get_all_complex_dtypes())))
    @tf32_on_and_off(0.005)
    def test_addmm_sizes(self, device, dtype):
        for m in [0, 1, 25]:
            for n in [0, 1, 10]:
                for k in [0, 1, 8]:
                    M = torch.randn(n, m, device=device).to(dtype)
                    m1 = torch.randn(n, k, device=device).to(dtype)
                    m2 = torch.randn(k, m, device=device).to(dtype)
                    self._test_addmm_addmv(torch.addmm, M, m1, m2)

    @unittest.skipIf(IS_FBCODE and IS_REMOTE_GPU, "cublas runtime error")
    @onlyCUDA
    def test_matmul_45724(self, device):
        # https://github.com/pytorch/pytorch/issues/45724
        a = torch.rand(65537, 22, 64, device=device, dtype=torch.half)
        b = torch.rand(65537, 64, 22, device=device, dtype=torch.half)
        c = torch.full((65537, 22, 22), math.nan, dtype=torch.half, device=device)
        cpu_result = torch.matmul(a.cpu().float(), b.cpu().float()).cuda().half()
        torch.matmul(a, b, out=c)
        self.assertEqual(c, cpu_result)

    @slowTest
    @onlyOnCPUAndCUDA
    @dtypes(torch.float32, torch.float64, torch.bfloat16, torch.int32, torch.int64, torch.cfloat, torch.cdouble)
    @dtypesIfCUDA(torch.float32, torch.float64, torch.cfloat, torch.cdouble)
    @tf32_on_and_off(0.01)
    def test_mm(self, device, dtype):
        def _test_mm(n, m, p, dtype, genf):
            # helper function
            def matrixmultiply(mat1, mat2):
                n = mat1.size(0)
                m = mat1.size(1)
                p = mat2.size(1)
                res = torch.zeros(n, p, dtype=dtype, device=device)
                for i, j in iter_indices(res):
                    res[i, j] = sum(mat1[i, k] * mat2[k, j] for k in range(m))
                return res

            # contiguous case
            mat1 = genf(n, m)
            mat2 = genf(m, p)
            res = torch.mm(mat1, mat2)

            res2 = matrixmultiply(mat1, mat2)
            self.assertEqual(res, res2)

            # non contiguous case 1
            mat1 = genf(n, m)
            mat2 = genf(p, m).t()
            res = torch.mm(mat1, mat2)

            res2 = matrixmultiply(mat1, mat2)
            self.assertEqual(res, res2)

            # non contiguous case 2
            mat1 = genf(m, n).t()
            mat2 = genf(m, p)
            res = torch.mm(mat1, mat2)

            res2 = matrixmultiply(mat1, mat2)
            self.assertEqual(res, res2)

            # non contiguous case 3
            mat1 = genf(m, n).t()
            mat2 = genf(p, m).t()
            res = torch.mm(mat1, mat2)

            res2 = matrixmultiply(mat1, mat2)
            self.assertEqual(res, res2)

            # test with zero stride
            mat1 = genf(n, m)
            mat2 = genf(m, 1).expand(m, p)
            res = torch.mm(mat1, mat2)

            res2 = matrixmultiply(mat1, mat2)
            self.assertEqual(res, res2)

            # explicitly exercise the _out variant in torch.mm().
            # contiguous case
            mat1 = genf(n, m)
            mat2 = genf(m, p)
            res = genf(n, p)
            torch.mm(mat1, mat2, out=res)

            res2 = matrixmultiply(mat1, mat2)
            self.assertEqual(res, res2)

            # explicitly exercise the _out variant in torch.mm().
            # non contiguous case 3
            mat1 = genf(m, n).t()
            mat2 = genf(p, m).t()
            res = genf(n, p)
            torch.mm(mat1, mat2, out=res)

            res2 = matrixmultiply(mat1, mat2)
            self.assertEqual(res, res2)

        def genf_int(x, y):
            return torch.randint(0, 100, (x, y), dtype=dtype, device=device)

        def genf_bfloat(x, y):
            return torch.randn(x, y, dtype=torch.float32, device=device).to(dtype)

        def genf_float(x, y):
            return torch.randn(x, y, dtype=dtype, device=device)

        for (n, m, p) in [(20, 10, 5), (15, 5, 10), (5, 18, 10)]:
            if (dtype == torch.int32) or (dtype == torch.int64):
                genf = genf_int
            elif (dtype == torch.bfloat16):
                genf = genf_bfloat
            else:
                genf = genf_float

            _test_mm(n, m, p, dtype, genf)

    @onlyOnCPUAndCUDA
    @dtypes(torch.float32, torch.float64)
    def test_strided_mm_bmm(self, device, dtype):
        # Tests strided view case with stride smaller than corresponding dimension size
        x = torch.tensor([[1., 2., 3.], [4., 5., 6.]], dtype=dtype, device=device)
        new_shape = [2, 2, 2]
        new_stride = [3, 1, 1]
        sx = torch.as_strided(x, size=new_shape, stride=new_stride)

        torch_fn = lambda x: torch.bmm(x, x)  # noqa: E731
        np_fn = lambda x: np.matmul(x, x)  # noqa: E731
        self.compare_with_numpy(torch_fn, np_fn, sx)

        torch_fn = lambda x: torch.mm(x, x)  # noqa: E731
        self.compare_with_numpy(torch_fn, np_fn, sx[0])

    @precisionOverride({torch.half: 0.05, torch.bfloat16: 0.05})
    @skipCUDAIf(torch.version.cuda == "10.1", "flaky on CUDA 10.1")
    @onlyOnCPUAndCUDA
    @dtypes(*torch.testing.get_all_fp_dtypes(), *torch.testing.get_all_complex_dtypes())
    @tf32_on_and_off(0.05)
    def test_bmm(self, device, dtype):
        if self.device_type == 'cuda' and dtype is torch.bfloat16 and CUDA11OrLater and not SM53OrLater:
            # cuBLAS does not guarantee BFloat16 support on SM < 53.
            # So on PyTorch, we consider BFloat16 support on SM < 53 as
            # undefined bahavior
            return

        batch_sizes = [1, 10]
        M, N, O = 23, 8, 12
        numpy_dtype = dtype if dtype != torch.bfloat16 else torch.float32

        is_supported = True
        if dtype == torch.bfloat16 and self.device_type == 'cuda':
            is_supported = TEST_WITH_ROCM or (CUDA11OrLater and SM53OrLater)

        if not is_supported:
            for num_batches in batch_sizes:
                b1 = torch.randn(num_batches, M, N, device=device).to(dtype)
                b2 = torch.randn(num_batches, N, O, device=device).to(dtype)
                self.assertRaisesRegex(RuntimeError, "type|Type|not implemented|CUBLAS_STATUS_NOT_SUPPORTED",
                                       lambda: torch.bmm(b1, b2))
            return

        def invert_perm(p):
            d = {x: i for i, x in enumerate(p)}
            return (d[0], d[1], d[2])

        def generate_inputs(num_batches):
            # transposed tensors
            for perm1, perm2 in itertools.product(itertools.permutations((0, 1, 2)), repeat=2):
                b1 = make_tensor((num_batches, M, N), device, dtype, low=-1, high=1)
                b2 = make_tensor((num_batches, N, O), device, dtype, low=-1, high=1)
                b1 = b1.permute(perm1).contiguous().permute(invert_perm(perm1))
                b2 = b2.permute(perm2).contiguous().permute(invert_perm(perm2))
                yield b1, b2
            # broadcasting tensors
            for b1, b2, b3, b4, b5, b6 in itertools.product((True, False), repeat=6):
                shape1 = (num_batches if b1 else 1, M if b2 else 1, N if b3 else 1)
                shape2 = (num_batches if b4 else 1, N if b5 else 1, O if b6 else 1)
                b1 = make_tensor(shape1, device, dtype, low=-1, high=1).expand(num_batches, M, N)
                b2 = make_tensor(shape2, device, dtype, low=-1, high=1).expand(num_batches, N, O)
                yield b1, b2
            # zero-sized tensors
            for z1, z2, z3, z4 in itertools.product((True, False), repeat=4):
                shape1 = (num_batches if z1 else 0, M if z2 else 0, N if z3 else 0)
                shape2 = (num_batches if z1 else 0, N if z3 else 0, O if z4 else 0)
                b1 = torch.randn(shape1, dtype=dtype, device=device)
                b2 = torch.randn(shape2, dtype=dtype, device=device)
                yield b1, b2

        for num_batches in batch_sizes:
            for (b1, b2), perm3 in itertools.product(generate_inputs(num_batches), itertools.permutations((0, 1, 2))):
                res1 = torch.bmm(b1, b2)
                res2 = torch.full((num_batches, M, O), math.nan, dtype=dtype, device=device) \
                    .permute(perm3).contiguous().permute(invert_perm(perm3))
                torch.bmm(b1, b2, out=res2)
                expect = torch.from_numpy(
                    b1.to(numpy_dtype).cpu().numpy() @ b2.to(numpy_dtype).cpu().numpy()).to(device=device, dtype=dtype)
                self.assertEqual(expect, res1)
                self.assertEqual(expect, res2)

                if self.device_type == 'cuda':
                    # check that mixed arguments are rejected
                    self.assertRaises(RuntimeError, lambda: torch.bmm(b1, b2.cpu()))
                    self.assertRaises(RuntimeError, lambda: torch.bmm(b1.cpu(), b2))
                    self.assertRaises(RuntimeError, lambda: torch.bmm(b1, b2, out=res2.cpu()))

    @unittest.skipIf(IS_FBCODE and IS_REMOTE_GPU, "cublas runtime error")
    @onlyCUDA
    @wrapDeterministicFlagAPITest
    def test_cublas_config_deterministic_error(self, device):
        test_cases = [
            # (function, (tensor sizes))
            ('mm', ((2, 2), (2, 2),)),
            ('mv', ((2, 2), (2,),)),
            ('bmm', ((1, 2, 2), (1, 2, 2),))]

        test_configs = [
            # (CuBLAS workspace config, is deterministic)
            ('garbage', False),
            (None, False),
            (':4096:8', True),
            (':16:8', True)]

        cublas_var_name = 'CUBLAS_WORKSPACE_CONFIG'
        is_cuda10_2_or_higher = (
            (torch.version.cuda is not None)
            and ([int(x) for x in torch.version.cuda.split(".")] >= [10, 2]))

        def test_case_info(fn_name, config):
            return f'function "{fn_name}" with config "{"" if config is None else config}"'

        # Create processes to test each combination of test cases and config settings
        processes = []
        for fn_name, arg_sizes in test_cases:
            for config, is_config_deterministic in test_configs:
                env = os.environ.copy()
                if config is None:
                    if env.get(cublas_var_name) is not None:
                        del env[cublas_var_name]
                else:
                    env[cublas_var_name] = config
                should_throw_error = is_cuda10_2_or_higher and not is_config_deterministic
                script = f"""
import torch
torch.use_deterministic_algorithms(True)
fn = torch.{fn_name}
arg_sizes = {arg_sizes}
device = '{device}'
should_throw_error = {should_throw_error}
args = []
for arg_size in arg_sizes:
    args.append(torch.randn(*arg_size, device=device))
try:
    fn(*args)
except RuntimeError as e:
    if not should_throw_error:
        raise RuntimeError('Did not expect any error to be raised')
    elif 'Deterministic behavior was enabled with either' not in str(e):
        raise RuntimeError('Expected a CuBLAS nondeterministic error, but got a different error')
else:
    if should_throw_error:
        raise RuntimeError('Expected a CuBLAS nondeterministic error, but it was not raised')

"""
                try:
                    subprocess.check_output(
                        [sys.executable, '-c', script],
                        stderr=subprocess.STDOUT,
                        # On Windows, opening the subprocess with the default CWD makes `import torch`
                        # fail, so just set CWD to this script's directory
                        cwd=os.path.dirname(os.path.realpath(__file__)),
                        env=env)
                except subprocess.CalledProcessError as e:
                    self.fail(msg=(
                        f'Subprocess exception while attempting to run {test_case_info(fn_name, config)}:\n'
                        + e.output.decode("utf-8")))

    def _test_addbmm_baddbmm(self, func, b1, b2, ref, out_tensor):
        getattr(out_tensor, func + "_")(b1, b2)
        self.assertEqual(out_tensor, ref)
        res3 = out_tensor.clone()

        with self.maybeWarnsRegex(
                UserWarning, f"This overload of {func}_ is deprecated"):
            getattr(out_tensor, func + "_")(1, b1, b2)
        self.assertEqual(out_tensor, ref * 2),
        getattr(res3, func + "_")(b1, b2, beta=1)
        self.assertEqual(out_tensor, res3)

        with self.maybeWarnsRegex(
                UserWarning, f"This overload of {func}_ is deprecated"):
            getattr(out_tensor, func + "_")(1., .5, b1, b2)
        self.assertEqual(out_tensor, ref * 2.5)
        getattr(res3, func + "_")(b1, b2, beta=1., alpha=.5)
        self.assertEqual(out_tensor, res3)

        with self.maybeWarnsRegex(
                UserWarning, f"This overload of {func} is deprecated"):
            self.assertEqual(out_tensor, getattr(torch, func)(1, out_tensor, 0, b1, b2))

        res4 = getattr(torch, func)(out_tensor, b1, b2, beta=1, alpha=.5)
        self.assertEqual(res4, ref * 3),

        nan = torch.full_like(out_tensor, math.nan)
        res5 = getattr(torch, func)(nan, b1, b2, beta=0, alpha=1)
        self.assertEqual(res5, ref)

        if b1.is_complex():
            res6 = getattr(torch, func)(out_tensor, b1, b2, beta=.1j, alpha=.5j)
            self.assertEqual(res6, out_tensor * .1j + .5j * ref)
        else:
            res6 = getattr(torch, func)(out_tensor, b1, b2, beta=.1, alpha=.5)
            self.assertEqual(res6, out_tensor * .1 + .5 * ref)

        res7 = torch.full_like(out_tensor, math.nan)
        getattr(torch, func)(nan, b1, b2, beta=0, out=res7)
        self.assertEqual(res7, ref)

    @precisionOverride({torch.half: 0.05, torch.bfloat16: 0.05})
    @onlyOnCPUAndCUDA
    @dtypes(*torch.testing.get_all_fp_dtypes(), *torch.testing.get_all_complex_dtypes())
    @tf32_on_and_off(0.05)
    def test_addbmm(self, device, dtype):
        if self.device_type == 'cuda' and dtype is torch.bfloat16 and CUDA11OrLater and not SM53OrLater:
            # cuBLAS does not guarantee BFloat16 support on SM < 53.
            # So on PyTorch, we consider BFloat16 support on SM < 53 as
            # undefined bahavior
            return

        num_batches = 2
        M, N, O = 2, 3, 4

        is_supported = True
        if dtype == torch.bfloat16:
            if self.device_type == 'cpu':
                self.precision = 1  # 43 vs 43.75
            else:
                is_supported = TEST_WITH_ROCM or (CUDA11OrLater and SM53OrLater)

        if not is_supported:
            b1 = make_tensor((num_batches, M, N), device, dtype, low=-1, high=1)
            b2 = make_tensor((num_batches, N, O), device, dtype, low=-1, high=1)
            t = make_tensor((M, O), device, dtype, low=-1, high=1)
            self.assertRaisesRegex(RuntimeError, "type|Type|not implemented|CUBLAS_STATUS_NOT_SUPPORTED",
                                   lambda: torch.addbmm(t, b1, b2))
            return

        def invert_perm(p):
            d = {x: i for i, x in enumerate(p)}
            return (d[0], d[1], d[2])

        def generate_tensor():
            numpy_dtype = dtype if dtype != torch.bfloat16 else torch.float32
            # transposed tensors
            for perm1, perm2 in itertools.product(itertools.permutations((0, 1, 2)), repeat=2):
                for perm3 in itertools.permutations((0, 1)):
                    b1 = make_tensor((num_batches, M, N), device, dtype, low=-1, high=1)
                    b2 = make_tensor((num_batches, N, O), device, dtype, low=-1, high=1)
                    b1 = b1.permute(perm1).contiguous().permute(invert_perm(perm1))
                    b2 = b2.permute(perm2).contiguous().permute(invert_perm(perm2))
                    ref = torch.from_numpy(
                        b1.to(numpy_dtype).cpu().numpy() @ b2.to(numpy_dtype).cpu().numpy()
                    ).to(device=device, dtype=dtype).sum(0)
                    out_tensor = torch.zeros_like(ref).permute(perm3).contiguous().permute(perm3)
                    yield b1, b2, ref, out_tensor
            # broadcasting tensors
            for s1, s2, s3, s4, s5, s6 in itertools.product((True, False), repeat=6):
                shape1 = (num_batches if s1 else 1, M if s2 else 1, N if s3 else 1)
                shape2 = (num_batches if s4 else 1, N if s5 else 1, O if s6 else 1)
                b1 = make_tensor(shape1, device, dtype, low=-1, high=1).expand(num_batches, M, N)
                b2 = make_tensor(shape2, device, dtype, low=-1, high=1).expand(num_batches, N, O)
                ref = torch.from_numpy(
                    b1.to(numpy_dtype).cpu().numpy() @ b2.to(numpy_dtype).cpu().numpy()
                ).to(device=device, dtype=dtype).sum(0)
                out_tensor = torch.zeros_like(ref)
                yield b1, b2, ref, out_tensor
            # zero-sized tensors
            for z1, z2, z3, z4 in itertools.product((True, False), repeat=4):
                shape1 = (num_batches if z1 else 0, M if z2 else 0, N if z3 else 0)
                shape2 = (num_batches if z1 else 0, N if z3 else 0, O if z4 else 0)
                b1 = make_tensor(shape1, device, dtype, low=-1, high=1)
                b2 = make_tensor(shape2, device, dtype, low=-1, high=1)
                ref = torch.from_numpy(
                    b1.to(numpy_dtype).cpu().numpy() @ b2.to(numpy_dtype).cpu().numpy()
                ).to(device=device, dtype=dtype).sum(0)
                out_tensor = torch.zeros_like(ref)
                yield b1, b2, ref, out_tensor

        for b1, b2, ref, out_tensor in generate_tensor():
            self._test_addbmm_baddbmm("addbmm", b1, b2, ref, out_tensor)

    @precisionOverride({torch.half: 0.1, torch.bfloat16: 0.5})
    @onlyOnCPUAndCUDA
    @dtypes(*torch.testing.get_all_fp_dtypes(), *torch.testing.get_all_complex_dtypes())
    @tf32_on_and_off(0.05)
    def test_baddbmm(self, device, dtype):
        if self.device_type == 'cuda' and dtype is torch.bfloat16 and CUDA11OrLater and not SM53OrLater:
            # cuBLAS does not guarantee BFloat16 support on SM < 53.
            # So on PyTorch, we consider BFloat16 support on SM < 53 as
            # undefined bahavior
            return

        num_batches = 10
        M, N, O = 12, 8, 5

        is_supported = True
        if dtype == torch.bfloat16 and self.device_type == 'cuda':
            is_supported = TEST_WITH_ROCM or (CUDA11OrLater and SM53OrLater)

        if not is_supported:
            b1 = make_tensor((num_batches, M, N), device, dtype, low=-1, high=1)
            b2 = make_tensor((num_batches, N, O), device, dtype, low=-1, high=1)
            t = make_tensor((num_batches, M, O), device, dtype, low=-1, high=1)
            self.assertRaisesRegex(RuntimeError, "type|Type|not implemented|CUBLAS_STATUS_NOT_SUPPORTED",
                                   lambda: torch.baddbmm(t, b1, b2))
            return

        def invert_perm(p):
            d = {x: i for i, x in enumerate(p)}
            return (d[0], d[1], d[2])

        def generate_tensor():
            numpy_dtype = dtype if dtype != torch.bfloat16 else torch.float32
            # transposed tensors
            for perm1, perm2, perm3 in itertools.product(itertools.permutations((0, 1, 2)), repeat=3):
                b1 = make_tensor((num_batches, M, N), device, dtype, low=-1, high=1)
                b2 = make_tensor((num_batches, N, O), device, dtype, low=-1, high=1)
                b1 = b1.permute(perm1).contiguous().permute(invert_perm(perm1))
                b2 = b2.permute(perm2).contiguous().permute(invert_perm(perm2))
                ref = torch.from_numpy(
                    b1.to(numpy_dtype).cpu().numpy() @ b2.to(numpy_dtype).cpu().numpy()).to(device=device, dtype=dtype)
                out_tensor = torch.zeros_like(ref)
                out_tensor = out_tensor.permute(perm3).contiguous().permute(invert_perm(perm3))
                yield b1, b2, ref, out_tensor
            # broadcasting tensors
            for s1, s2, s3, s4, s5, s6 in itertools.product((True, False), repeat=6):
                shape1 = (num_batches if s1 else 1, M if s2 else 1, N if s3 else 1)
                shape2 = (num_batches if s4 else 1, N if s5 else 1, O if s6 else 1)
                b1 = make_tensor(shape1, device, dtype, low=-1, high=1).expand(num_batches, M, N)
                b2 = make_tensor(shape2, device, dtype, low=-1, high=1).expand(num_batches, N, O)
                ref = torch.from_numpy(
                    b1.to(numpy_dtype).cpu().numpy() @ b2.to(numpy_dtype).cpu().numpy()).to(device=device, dtype=dtype)
                out_tensor = torch.zeros_like(ref)
                yield b1, b2, ref, out_tensor
            # zero-sized tensors
            for z1, z2, z3, z4 in itertools.product((True, False), repeat=4):
                shape1 = (num_batches if z1 else 0, M if z2 else 0, N if z3 else 0)
                shape2 = (num_batches if z1 else 0, N if z3 else 0, O if z4 else 0)
                b1 = make_tensor(shape1, device, dtype, low=-2, high=2)
                b2 = make_tensor(shape2, device, dtype, low=-2, high=2)
                ref = torch.from_numpy(
                    b1.to(numpy_dtype).cpu().numpy() @ b2.to(numpy_dtype).cpu().numpy()).to(device=device, dtype=dtype)
                out_tensor = torch.zeros_like(ref)
                yield b1, b2, ref, out_tensor

        for b1, b2, ref, out_tensor in generate_tensor():
            self._test_addbmm_baddbmm("baddbmm", b1, b2, ref, out_tensor)

    # TODO: update to compare against NumPy
    @onlyCUDA
    def test_solve_methods_arg_device(self, device):
        for b_device, A_device in itertools.product(['cpu', device], repeat=2):
            if b_device == A_device:
                continue

            b = torch.randn(3, 1, device=b_device)
            A = torch.randn(3, 3, device=A_device)
            err_str = "Expected b and A to be on the same device"
            with self.assertRaisesRegex(RuntimeError, err_str):
                torch.solve(b, A)

            with self.assertRaisesRegex(RuntimeError, err_str):
                torch.cholesky_solve(b, A)

            with self.assertRaisesRegex(RuntimeError, err_str):
                torch.triangular_solve(b, A)

            # b and A have to be modified to match accepted inputs sizes for lu_solve
            b = b.unsqueeze(0)
            A = A.unsqueeze(0)
            with self.assertRaisesRegex(RuntimeError, err_str):
                torch.lu_solve(b, A, torch.rand(A.shape[:-1], device=A_device).int())

            # This checks if a suitable error message is thrown
            # when LU output and pivots are on the same device
            with self.assertRaisesRegex(RuntimeError,
                                        "Expected LU_pivots and LU_data to be on the same device"):
                torch.lu_solve(b, A, torch.rand(A.shape[:-1], device=b_device).int())

    @precisionOverride({torch.float32: 5e-3, torch.complex64: 1e-3})
    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(torch.float32, torch.float64, torch.complex64, torch.complex128)
    def test_pinverse(self, device, dtype):
        from torch.testing._internal.common_utils import random_fullrank_matrix_distinct_singular_value as fullrank

        def run_test(M):
            # Testing against definition for pseudo-inverses
            MPI = torch.pinverse(M)
            MPI_ = MPI.cpu().numpy()
            M_ = M.cpu().numpy()
            if M.numel() > 0:
                self.assertEqual(M_, np.matmul(np.matmul(M_, MPI_), M_))
                self.assertEqual(MPI_, np.matmul(np.matmul(MPI_, M_), MPI_))
                self.assertEqual(np.matmul(M_, MPI_), np.matmul(M_, MPI_).swapaxes(-2, -1).conj())
                self.assertEqual(np.matmul(MPI_, M_), np.matmul(MPI_, M_).swapaxes(-2, -1).conj())
            else:
                self.assertEqual(M.shape, MPI.shape[:-2] + (MPI.shape[-1], MPI.shape[-2]))
        for sizes in [(5, 5), (3, 5, 5), (3, 7, 5, 5),  # square matrices
                      (3, 2), (5, 3, 2), (7, 5, 3, 2),  # fat matrices
                      (2, 3), (5, 2, 3), (7, 5, 2, 3),  # thin matrices
                      (0, 0), (0, 2), (2, 0), (3, 0, 0), (0, 3, 0), (0, 0, 3)]:  # zero numel matrices
            M = torch.randn(*sizes, dtype=dtype, device=device)
            run_test(M)

        # Test inverse and pseudo-inverse for invertible matrix
        for sizes in [(5, 5), (3, 5, 5), (3, 7, 5, 5)]:
            matsize = sizes[-1]
            batchdims = sizes[:-2]
            M = fullrank(matsize, *batchdims, dtype=dtype, device=device)
            self.assertEqual(torch.eye(matsize, dtype=dtype, device=device).expand(sizes), M.pinverse().matmul(M),
                             atol=1e-7, rtol=0, msg='pseudo-inverse for invertible matrix')

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(torch.double)
    def test_matrix_power(self, device, dtype):
        def run_test(M, sign=1):
            if sign == -1:
                M = M.inverse()
            MP2 = torch.matrix_power(M, 2)
            self.assertEqual(MP2, torch.matmul(M, M))

            MP3 = torch.matrix_power(M, 3)
            self.assertEqual(MP3, torch.matmul(MP2, M))

            MP4 = torch.matrix_power(M, 4)
            self.assertEqual(MP4, torch.matmul(MP2, MP2))

            MP6 = torch.matrix_power(M, 6)
            self.assertEqual(MP6, torch.matmul(MP3, MP3))

            MP0 = torch.matrix_power(M, 0)
            self.assertEqual(MP0, torch.eye(M.size(-2), dtype=dtype).expand_as(M))

        # Single matrix
        M = torch.randn(5, 5, dtype=dtype, device=device)
        run_test(M)

        # Batch matrices
        M = torch.randn(3, 3, 3, dtype=dtype, device=device)
        run_test(M)

        # Many batch matrices
        M = torch.randn(2, 3, 3, 3, dtype=dtype, device=device)
        run_test(M)

        # This is for negative powers
        from torch.testing._internal.common_utils import random_fullrank_matrix_distinct_singular_value
        M = random_fullrank_matrix_distinct_singular_value(5, dtype=dtype, device=device)
        run_test(M, sign=-1)

        M = random_fullrank_matrix_distinct_singular_value(3, 3, dtype=dtype, device=device)
        run_test(M, sign=-1)

        M = random_fullrank_matrix_distinct_singular_value(3, 2, 3, dtype=dtype, device=device)
        run_test(M, sign=-1)

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(torch.float, torch.complex64)
    def test_matrix_exp_utils(self, device, dtype):
        # test linear combination
        def run_test(coeff_shape, data_shape):
            coeffs = torch.rand(*coeff_shape, device=device, dtype=torch.float)
            x = torch.rand(coeff_shape[1], *data_shape, device=device, dtype=dtype)

            res1 = torch._compute_linear_combination(x, coeffs)
            res2 = (x.unsqueeze(0) * coeffs.view(*coeff_shape, *([1] * len(data_shape)))).sum(1)
            self.assertEqual(res1, res2, atol=1e-5, rtol=0.0)

            # check `out=` version
            res3 = torch.zeros(coeff_shape[0], *data_shape, device=device, dtype=dtype)
            torch._compute_linear_combination(x, coeffs, out=res3)
            self.assertEqual(res1, res3, atol=1e-5, rtol=0.0)

            res4 = torch.ones(coeff_shape[0], *data_shape, device=device, dtype=dtype)
            torch._compute_linear_combination(x, coeffs, out=res4)
            self.assertEqual(res1, res4 - 1.0, atol=1e-5, rtol=0.0)

            res5 = torch.ones(coeff_shape[0], *data_shape, device=device, dtype=dtype)
            res5_clone = res5.clone()
            torch._compute_linear_combination(x, coeffs, out=res5)
            self.assertEqual(res1, res5 - res5_clone, atol=1e-5, rtol=0.0)

        run_test([1, 3], [2, 2])
        run_test([3, 1], [2, 2])
        run_test([1, 10], [10, 10])
        run_test([10, 1], [10, 10])
        run_test([5, 3], [2, 2])
        run_test([5, 3], [100, 100])
        run_test([3, 4], [3, 3, 3])
        run_test([3, 4], [3, 3, 3, 3])

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(torch.float, torch.double, torch.complex64, torch.complex128)
    def test_matrix_exp_boundary_cases(self, device, dtype):

        with self.assertRaisesRegex(RuntimeError, "expected a tensor of floating or complex types"):
            torch.randn(3, 3).type(torch.int).matrix_exp()

        with self.assertRaisesRegex(RuntimeError, "with dim at least 2"):
            torch.randn(3).matrix_exp()

        with self.assertRaisesRegex(RuntimeError, "expected a tensor of squared matrices"):
            torch.randn(3, 2, 1).matrix_exp()

        # check 1x1 matrices
        x = torch.randn(3, 3, 1, 1)
        mexp = x.matrix_exp()
        self.assertEqual(mexp, x.exp())

    @slowTest
    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(torch.float, torch.double, torch.cfloat, torch.cdouble)
    def test_matrix_exp_analytic(self, device, dtype):
        # check zero matrix
        x = torch.zeros(20, 20, dtype=dtype, device=device)
        self.assertTrue((x.matrix_exp() == torch.eye(20, 20, dtype=dtype, device=device)).all().item())

        def normalize_to_1_operator_norm(sample, desired_norm):
            sample_norm, _ = sample.abs().sum(-2).max(-1)
            sample_to_1_norm = sample / sample_norm.unsqueeze(-1).unsqueeze(-1)
            return sample_to_1_norm * desired_norm

        def gen_good_cond_number_matrices(*n):
            """
            Generates a diagonally-domimant matrix
            with the eigenvalues centered at 1
            and the radii at most (n[-1] - 1) / (n[-2] ** 2)
            """
            identity = torch.eye(n[-2], n[-1], dtype=dtype, device=device).expand(*n)
            x = torch.rand(*n, dtype=dtype, device=device) / (n[-1] ** 2)
            x = (x - x * identity) + identity
            return x

        def run_test(*n):
            if dtype == torch.float:
                thetas = [
                    1.192092800768788e-07,  # deg 1
                    5.978858893805233e-04,  # deg 2
                    5.116619363445086e-02,  # deg 4
                    5.800524627688768e-01,  # deg 8
                    1.461661507209034e+00,  # deg 12
                    3.010066362817634e+00   # deg 18
                ]
            else:  # if torch.double
                thetas = [
                    2.220446049250313e-16,  # deg 1
                    2.580956802971767e-08,  # deg 2
                    3.397168839976962e-04,  # deg 4
                    4.991228871115323e-02,  # deg 8
                    2.996158913811580e-01,  # deg 12
                    1.090863719290036e+00   # deg 18
                ]

            # generate input
            q = gen_good_cond_number_matrices(*n)
            q_ = q.cpu().numpy()
            qinv = torch.inverse(q)
            qinv_ = qinv.cpu().numpy()
            d = torch.randn(n[:-1], dtype=dtype, device=device)
            x = torch.from_numpy(
                np.matmul(q_, np.matmul(torch.diag_embed(d).cpu().numpy(), qinv_))).to(device)
            x_norm, _ = x.abs().sum(-2).max(-1)

            # test simple analytic whatever norm generated
            mexp = x.matrix_exp()
            mexp_analytic = np.matmul(
                q_,
                np.matmul(
                    torch.diag_embed(d.exp()).cpu().numpy(),
                    qinv_
                )
            )
            self.assertEqual(mexp, mexp_analytic, atol=1e-3, rtol=0.0)

            # generate norms to test different degree expansions
            sample_norms = []
            for i in range(len(thetas) - 1):
                sample_norms.append(0.5 * (thetas[i] + thetas[i + 1]))
            sample_norms = [thetas[0] / 2] + sample_norms + [thetas[-1] * 2]

            # matrices to equal norm
            for sample_norm in sample_norms:
                x_normalized = normalize_to_1_operator_norm(x, sample_norm)

                mexp = x_normalized.matrix_exp()
                mexp_analytic = np.matmul(
                    q_,
                    np.matmul(
                        torch.diag_embed((d / x_norm.unsqueeze(-1) * sample_norm).exp()).cpu().numpy(),
                        qinv_
                    )
                )
                self.assertEqual(mexp, mexp_analytic, atol=1e-3, rtol=0.0)

        # single matrix
        run_test(2, 2)
        run_test(3, 3)
        run_test(4, 4)
        run_test(5, 5)
        run_test(100, 100)
        run_test(200, 200)

        # small batch of matrices
        run_test(3, 2, 2)
        run_test(3, 3, 3)
        run_test(3, 4, 4)
        run_test(3, 5, 5)
        run_test(3, 100, 100)
        run_test(3, 200, 200)

        # large batch of matrices
        run_test(3, 3, 2, 2)
        run_test(3, 3, 3, 3)
        run_test(3, 3, 4, 4)
        run_test(3, 3, 5, 5)
        run_test(3, 3, 100, 100)
        run_test(3, 3, 200, 200)

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(torch.float, torch.double)
    def test_matrix_exp_batch(self, device, dtype):

        def run_test(*n):
            tensors_batch = torch.zeros(n, dtype=dtype, device=device)
            tensors_batch = tensors_batch.view(-1, n[-2], n[-1])

            num_matrices = tensors_batch.size(0)
            tensors_list = []
            for i in range(num_matrices):
                tensors_list.append(torch.randn(n[-2], n[-1], dtype=dtype, device=device))

            for i in range(num_matrices):
                tensors_batch[i, ...] = tensors_list[i]

            tensors_exp_map = (x.matrix_exp() for x in tensors_list)
            tensors_exp_batch = tensors_batch.matrix_exp()

            for i, tensor_exp in enumerate(tensors_exp_map):
                self.assertEqual(tensors_exp_batch[i, ...], tensor_exp)

        # small batch of matrices
        run_test(3, 2, 2)
        run_test(3, 3, 3)
        run_test(3, 4, 4)
        run_test(3, 5, 5)

        # large batch of matrices
        run_test(3, 3, 2, 2)
        run_test(3, 3, 3, 3)
        run_test(3, 3, 4, 4)
        run_test(3, 3, 5, 5)

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(torch.float, torch.double, torch.cfloat, torch.cdouble)
    def test_matrix_exp_compare_with_taylor(self, device, dtype):

        def normalize_to_1_operator_norm(sample, desired_norm):
            sample_norm, _ = sample.abs().sum(-2).max(-1)
            sample_to_1_norm = sample / sample_norm.unsqueeze(-1).unsqueeze(-1)
            return sample_to_1_norm * desired_norm

        def gen_good_cond_number_matrices(*n):
            """
            Generates a diagonally-domimant matrix
            with the eigenvalues centered at 1
            and the radii at most (n[-1] - 1) / (n[-2] ** 2)
            """
            identity = torch.eye(n[-2], n[-1], dtype=dtype, device=device).expand(*n)
            x = torch.rand(*n, dtype=dtype, device=device) / (n[-1] ** 2)
            x = (x - x * identity) + identity
            return x

        def get_taylor_approximation(a, deg):
            a_ = a.cpu().numpy()
            identity = torch.eye(a.size(-2), a.size(-1), dtype=dtype, device=device).expand_as(a)
            res = identity.cpu().numpy()
            taylor_term = identity.cpu().numpy()

            for i in range(1, deg + 1):
                taylor_term = np.matmul(a_, taylor_term) / i
                res = res + taylor_term

            return res

        def scale_square(a, deg):
            if a.abs().pow(2).sum().sqrt() < 1.0:
                return get_taylor_approximation(a, 12)
            else:
                s = int(torch.log2(a.abs().pow(2).sum().sqrt()).ceil().item())
                b = a / (2 ** s)
                b = get_taylor_approximation(b, 18)
                for _ in range(s):
                    b = np.matmul(b, b)
                return torch.from_numpy(b).to(a.device)

        def run_test(*n):
            degs = [1, 2, 4, 8, 12, 18]
            if dtype == torch.float:
                thetas = [
                    1.192092800768788e-07,  # deg 1
                    5.978858893805233e-04,  # deg 2
                    5.116619363445086e-02,  # deg 4
                    5.800524627688768e-01,  # deg 8
                    1.461661507209034e+00,  # deg 12
                    3.010066362817634e+00   # deg 18
                ]
            else:  # if torch.double
                thetas = [
                    2.220446049250313e-16,  # deg 1
                    2.580956802971767e-08,  # deg 2
                    3.397168839976962e-04,  # deg 4
                    4.991228871115323e-02,  # deg 8
                    2.996158913811580e-01,  # deg 12
                    1.090863719290036e+00   # deg 18
                ]

            # generate norms to test different degree expansions
            sample_norms = []
            for i in range(len(thetas) - 1):
                sample_norms.append(0.5 * (thetas[i] + thetas[i + 1]))
            sample_norms = [thetas[0] / 2] + sample_norms + [thetas[-1] * 2]
            degs = [degs[0]] + degs

            for sample_norm, deg in zip(sample_norms, degs):
                x = gen_good_cond_number_matrices(*n)
                x = normalize_to_1_operator_norm(x, sample_norm)

                mexp = x.matrix_exp()
                mexp_taylor = scale_square(x, deg)

                self.assertEqual(mexp, mexp_taylor, atol=1e-2, rtol=0.0)

        # single matrix
        run_test(2, 2)
        run_test(3, 3)
        run_test(4, 4)
        run_test(5, 5)

        # small batch of matrices
        run_test(3, 2, 2)
        run_test(3, 3, 3)
        run_test(3, 4, 4)
        run_test(3, 5, 5)

        # large batch of matrices
        run_test(3, 3, 2, 2)
        run_test(3, 3, 3, 3)
        run_test(3, 3, 4, 4)
        run_test(3, 3, 5, 5)

    @dtypes(torch.double)
    def test_chain_matmul(self, device, dtype):
        def product(matrices):
            for mat in matrices[1:]:
                matrices[0] = matrices[0].mm(mat)
            return matrices[0]

        def run_test(p):
            matrices = []
            for (pi, pi_1) in zip(p[:-1], p[1:]):
                matrices.append(torch.randn(pi, pi_1, dtype=dtype, device=device))
            self.assertEqual(torch.chain_matmul(*matrices), product(matrices))

        run_test([10, 20, 30, 5])
        run_test([15, 5, 10, 20, 25])

        with self.assertRaisesRegex(RuntimeError, "chain_matmul: Expected one or more matrices"):
            torch.chain_matmul()

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(torch.float32, torch.float64, torch.complex64, torch.complex128)
    @precisionOverride({torch.float32: 1e-3, torch.complex64: 1e-3,
                        torch.float64: 1e-8, torch.complex128: 1e-8})
    def test_slogdet(self, device, dtype):
        from torch.testing._internal.common_utils import (random_hermitian_matrix, random_hermitian_psd_matrix,
                                                          random_hermitian_pd_matrix, random_square_matrix_of_rank)

        # mat_chars denotes matrix characteristics
        # possible values are: hermitian, hermitian_psd, hermitian_pd, singular, non_singular
        def run_test(matsize, batchdims, mat_chars):
            num_matrices = np.prod(batchdims)
            list_of_matrices = []
            if num_matrices != 0:
                for idx in range(num_matrices):
                    mat_type = idx % len(mat_chars)
                    if mat_chars[mat_type] == 'hermitian':
                        list_of_matrices.append(random_hermitian_matrix(matsize, dtype=dtype, device=device))
                    elif mat_chars[mat_type] == 'hermitian_psd':
                        list_of_matrices.append(random_hermitian_psd_matrix(matsize, dtype=dtype, device=device))
                    elif mat_chars[mat_type] == 'hermitian_pd':
                        list_of_matrices.append(random_hermitian_pd_matrix(matsize, dtype=dtype, device=device))
                    elif mat_chars[mat_type] == 'singular':
                        list_of_matrices.append(torch.ones(matsize, matsize, dtype=dtype, device=device))
                    elif mat_chars[mat_type] == 'non_singular':
                        list_of_matrices.append(random_square_matrix_of_rank(matsize, matsize, dtype=dtype, device=device))
                full_tensor = torch.stack(list_of_matrices, dim=0).reshape(batchdims + (matsize, matsize))
            else:
                full_tensor = torch.randn(*batchdims, matsize, matsize, dtype=dtype, device=device)

            actual_value = torch.linalg.slogdet(full_tensor)
            expected_value = np.linalg.slogdet(full_tensor.cpu().numpy())
            self.assertEqual(expected_value[0], actual_value[0], atol=self.precision, rtol=self.precision)
            self.assertEqual(expected_value[1], actual_value[1], atol=self.precision, rtol=self.precision)

            # test out=variant
            sign_out = torch.empty_like(actual_value[0])
            logabsdet_out = torch.empty_like(actual_value[1])
            ans = torch.linalg.slogdet(full_tensor, out=(sign_out, logabsdet_out))
            self.assertEqual(ans[0], sign_out)
            self.assertEqual(ans[1], logabsdet_out)
            self.assertEqual(sign_out, actual_value[0])
            self.assertEqual(logabsdet_out, actual_value[1])

        for matsize, batchdims in itertools.product([0, 3, 5], [(0,), (3,), (5, 3)]):
            run_test(matsize, batchdims, mat_chars=['hermitian_pd'])
            run_test(matsize, batchdims, mat_chars=['singular'])
            run_test(matsize, batchdims, mat_chars=['non_singular'])
            run_test(matsize, batchdims, mat_chars=['hermitian', 'hermitian_pd', 'hermitian_psd'])
            run_test(matsize, batchdims, mat_chars=['singular', 'non_singular'])

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(torch.float32, torch.float64, torch.complex64, torch.complex128)
    def test_slogdet_errors_and_warnings(self, device, dtype):
        # slogdet requires the input to be a square matrix or batch of square matrices
        a = torch.randn(2, 3, device=device, dtype=dtype)
        with self.assertRaisesRegex(RuntimeError, r'must be batches of square matrices'):
            torch.linalg.slogdet(a)

        # slogdet requires the input to be at least 2 dimensional tensor
        a = torch.randn(2, device=device, dtype=dtype)
        with self.assertRaisesRegex(RuntimeError, r'must have at least 2 dimensions'):
            torch.linalg.slogdet(a)

        # slogdet requires the input to be of float, double, cfloat or cdouble types
        a = torch.randn(2, 2, device=device, dtype=torch.bfloat16)
        with self.assertRaisesRegex(RuntimeError, r'of float, double, cfloat or cdouble types'):
            torch.linalg.slogdet(a)

        # if non-empty out tensor with wrong shape is passed a warning is given
        a = torch.randn(2, 3, 3, device=device, dtype=dtype)
        sign_out = torch.empty(1, device=device, dtype=dtype)
        real_dtype = a.real.dtype if dtype.is_complex else dtype
        logabsdet_out = torch.empty(1, device=device, dtype=real_dtype)
        with warnings.catch_warnings(record=True) as w:
            # Trigger warning
            torch.linalg.slogdet(a, out=(sign_out, logabsdet_out))
            # Check warning occurs
            self.assertEqual(len(w), 1)
            self.assertTrue("An output with one or more elements was resized" in str(w[-1].message))

        # dtypes should match
        sign_out = torch.empty_like(a).to(torch.int)
        logabsdet_out = torch.empty_like(a).to(torch.int)
        with self.assertRaisesRegex(RuntimeError, "sign dtype Int does not match input dtype"):
            torch.linalg.slogdet(a, out=(sign_out, logabsdet_out))

        sign_out = torch.empty(0, device=device, dtype=dtype)
        with self.assertRaisesRegex(RuntimeError, "logabsdet dtype Int does not match the expected dtype"):
            torch.linalg.slogdet(a, out=(sign_out, logabsdet_out))

        # device should match
        if torch.cuda.is_available():
            wrong_device = 'cpu' if self.device_type != 'cpu' else 'cuda'
            sign_out = torch.empty(0, device=wrong_device, dtype=dtype)
            logabsdet_out = torch.empty(0, device=wrong_device, dtype=real_dtype)
            with self.assertRaisesRegex(RuntimeError, "Expected sign, logabsdet and input to be on the same device"):
                torch.linalg.slogdet(a, out=(sign_out, logabsdet_out))

    @slowTest
    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(torch.double)
    def test_det_logdet_slogdet(self, device, dtype):
        def reference_slogdet(M):
            sdet, logabsdet = np.linalg.slogdet(M.detach().cpu().numpy())
            return M.new_tensor(sdet), M.new_tensor(logabsdet)

        def test_single_det(M, target, desc):
            target_sdet, target_logabsdet = target

            det = M.det()
            logdet = M.logdet()
            sdet, logabsdet = M.slogdet()
            linalg_sdet, linalg_logabsdet = torch.linalg.slogdet(M)

            # Test det
            self.assertEqual(det, target_sdet * target_logabsdet.exp(),
                             atol=1e-7, rtol=0, msg='{} (det)'.format(desc))

            # Test slogdet
            # Compare the overall value rather than individual parts because of
            # precision issues when det is near zero.
            self.assertEqual(sdet * logabsdet.exp(), target_sdet * target_logabsdet.exp(),
                             atol=1e-7, rtol=0, msg='{} (slogdet)'.format(desc))
            self.assertEqual(linalg_sdet * linalg_logabsdet.exp(), target_sdet * target_logabsdet.exp(),
                             atol=1e-7, rtol=0, msg='{} (linalg_slogdet)'.format(desc))

            # Test logdet
            # Compare logdet against our own pytorch slogdet because they should
            # be consistent, while it may behave slightly differently with other
            # slogdet implementations when det is near zero due to precision
            # issues.
            if sdet.item() < 0:
                self.assertTrue(logdet.item() != logdet.item(), '{} (logdet negative case)'.format(desc))
            else:
                self.assertEqual(logdet.exp(), target_logabsdet.exp(),
                                 atol=1e-7, rtol=0, msg='{} (logdet non-negative case)'.format(desc))

        eye = torch.eye(5, dtype=dtype, device=device)
        test_single_det(eye, (torch.ones((), dtype=dtype, device=device), torch.zeros((), dtype=dtype, device=device)), 'identity')
        # Testing bug in #34061 (https://github.com/pytorch/pytorch/issues/34061)
        for n in range(250, 551, 100):
            mat = torch.randn(n, n, dtype=dtype, device=device)
            q, _ = torch.qr(mat)
            ref_det, ref_logabsdet = reference_slogdet(q)
            test_single_det(q, (ref_det, ref_logabsdet), 'orthogonal')

        def test(M):
            assert M.size(0) >= 5, 'this helper fn assumes M to be at least 5x5'
            M = M.to(device)

            ref_M_sdet, ref_M_logabsdet = reference_slogdet(M)

            test_single_det(M, (ref_M_sdet, ref_M_logabsdet), 'basic')
            if ref_M_logabsdet.exp().item() >= 1e-6:  # skip singular
                M_inv = M.inverse()
                test_single_det(M_inv, reference_slogdet(M_inv), 'inverse')

            test_single_det(M, (ref_M_sdet, ref_M_logabsdet), 'transpose')

            for x in [0, 2, 4]:
                for scale in [-2, -0.1, 0, 10]:
                    if scale > 0:
                        target = ref_M_sdet, ref_M_logabsdet + math.log(scale)
                    elif scale == 0:
                        target = torch.zeros_like(ref_M_sdet), torch.full_like(ref_M_logabsdet, -inf)
                    else:
                        target = ref_M_sdet.neg(), ref_M_logabsdet + math.log(-scale)

                    # dim 0
                    M_clone = M.clone()
                    M_clone[:, x] *= scale
                    test_single_det(M_clone, target, 'scale a row')
                    # dim 1
                    M_clone = M.clone()
                    M_clone[x, :] *= scale
                    test_single_det(M_clone, target, 'scale a column')

            for x1, x2 in [(0, 3), (4, 1), (3, 2)]:
                assert x1 != x2, 'x1 and x2 needs to be different for this test'
                target = torch.zeros_like(ref_M_sdet), torch.full_like(ref_M_logabsdet, -inf)
                # dim 0
                M_clone = M.clone()
                M_clone[:, x2] = M_clone[:, x1]
                test_single_det(M_clone, target, 'two rows are same')
                # dim 1
                M_clone = M.clone()
                M_clone[x2, :] = M_clone[x1, :]
                test_single_det(M_clone, target, 'two columns are same')

                for scale1, scale2 in [(0.3, -1), (0, 2), (10, 0.1)]:
                    det_scale = scale1 * scale2 * -1
                    if det_scale > 0:
                        target = ref_M_sdet, ref_M_logabsdet + math.log(det_scale)
                    elif det_scale == 0:
                        target = torch.zeros_like(ref_M_sdet), torch.full_like(ref_M_logabsdet, -inf)
                    else:
                        target = ref_M_sdet.neg(), ref_M_logabsdet + math.log(-det_scale)

                    # dim 0
                    M_clone = M.clone()
                    t = M_clone[:, x1] * scale1
                    M_clone[:, x1] += M_clone[:, x2] * scale2
                    M_clone[:, x2] = t
                    test_single_det(M_clone, target, 'exchanging rows')
                    # dim 1
                    M_clone = M.clone()
                    t = M_clone[x1, :] * scale1
                    M_clone[x1, :] += M_clone[x2, :] * scale2
                    M_clone[x2, :] = t
                    test_single_det(M_clone, target, 'exchanging columns')

        def get_random_mat_scale(n):
            # For matrices with values i.i.d. with 0 mean, unit variance, and
            # subexponential tail, we have:
            #   E[log det(A^2)] \approx log((n-1)!)
            #
            # Notice:
            #   log Var[det(A)] = log E[det(A^2)] >= E[log det(A^2)]
            #
            # So:
            #   stddev[det(A)] >= sqrt( (n-1)! )
            #
            # We use this as an intuitive guideline to scale random generated
            # matrices so our closeness tests can work more robustly:
            #   scale by sqrt( (n-1)! )^(-1/n) = ( (n-1)! )^(-1/(2n))
            #
            # source: https://arxiv.org/pdf/1112.0752.pdf

            # TODO: technically we need subexponential distn for this to hold,
            #       but we mostly use gaussian entries below. Consider switching
            #       to Chi-sq if this turns out not stable enough, since Chi-sq
            #       is easy enough to sample from.
            return math.factorial(n - 1) ** (-1.0 / (2 * n))

        for n in [5, 10, 25]:
            scale = get_random_mat_scale(n)
            test(torch.randn(n, n, dtype=dtype, device=device) * scale)
            r = torch.randn(n, n, dtype=dtype, device=device) * scale
            # symmetric psd
            test(r.mm(r.t()))
            # symmetric pd
            r = torch.randn(n, n, dtype=dtype, device=device) * scale
            test(r.mm(r.t()) + torch.eye(n, dtype=dtype, device=device) * 1e-6)
            # symmetric
            r = torch.randn(n, n, dtype=dtype, device=device) * scale
            for i in range(n):
                for j in range(i):
                    r[i, j] = r[j, i]
            test(r)
            # non-contiguous
            test((torch.randn(n, n, n + 1, dtype=dtype, device=device) * scale)[:, 2, 1:])
            # det = 0
            r = torch.randn(n, n, dtype=dtype, device=device) * scale
            u, s, v = r.svd()
            if reference_slogdet(u)[0] < 0:
                u = -u
            if reference_slogdet(v)[0] < 0:
                v = -v
            s[0] *= -1
            s[-1] = 0
            test(u.mm(s.diag()).mm(v))

        # Small values to test numerical stability. Note that we don't scale
        # this matrix.
        r = torch.randn(512, 512, dtype=dtype, device=device)
        u, s, v = r.svd()
        s.fill_(1. / (100 * s.numel()))
        test(u.mm(s.diag()).mm(v))

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(torch.double)
    @skipCUDAIfRocm
    def test_det_logdet_slogdet_batched(self, device, dtype):
        from torch.testing._internal.common_utils import (random_symmetric_matrix, random_symmetric_psd_matrix,
                                                          random_symmetric_pd_matrix, random_square_matrix_of_rank)

        # mat_chars denotes matrix characteristics
        # possible values are: sym, sym_psd, sym_pd, sing, non_sym
        def run_test(matsize, batchdims, mat_chars):
            num_matrices = reduce(lambda x, y: x * y, batchdims, 1)
            list_of_matrices = []

            for idx in range(num_matrices):
                mat_type = idx % len(mat_chars)
                if mat_chars[mat_type] == 'sym':
                    list_of_matrices.append(random_symmetric_matrix(matsize, dtype=dtype, device=device))
                elif mat_chars[mat_type] == 'sym_psd':
                    list_of_matrices.append(random_symmetric_psd_matrix(matsize, dtype=dtype, device=device))
                elif mat_chars[mat_type] == 'sym_pd':
                    list_of_matrices.append(random_symmetric_pd_matrix(matsize, dtype=dtype, device=device))
                elif mat_chars[mat_type] == 'sing':
                    list_of_matrices.append(torch.ones(matsize, matsize, dtype=dtype, device=device))
                elif mat_chars[mat_type] == 'non_sing':
                    list_of_matrices.append(random_square_matrix_of_rank(matsize, matsize, dtype=dtype, device=device))
            full_tensor = torch.stack(list_of_matrices, dim=0).reshape(batchdims + (matsize, matsize))
            # Scaling adapted from `get_random_mat_scale` in _test_det_logdet_slogdet
            full_tensor *= (math.factorial(matsize - 1) ** (-1.0 / (2 * matsize)))

            for fn in [torch.det, torch.logdet, torch.slogdet, torch.linalg.slogdet]:
                expected_value = []
                actual_value = fn(full_tensor)
                for full_idx in itertools.product(*map(lambda x: list(range(x)), batchdims)):
                    expected_value.append(fn(full_tensor[full_idx]))

                if fn == torch.slogdet or fn == torch.linalg.slogdet:
                    sign_value = torch.stack([tup[0] for tup in expected_value], dim=0).reshape(batchdims)
                    expected_value = torch.stack([tup[1] for tup in expected_value], dim=0).reshape(batchdims)
                    self.assertEqual(sign_value, actual_value[0])
                    self.assertEqual(expected_value, actual_value[1])
                else:
                    expected_value = torch.stack(expected_value, dim=0).reshape(batchdims)
                    self.assertEqual(actual_value, expected_value)

        for matsize, batchdims in itertools.product([3, 5], [(3,), (5, 3)]):
            run_test(matsize, batchdims, mat_chars=['sym_pd'])
            run_test(matsize, batchdims, mat_chars=['sing'])
            run_test(matsize, batchdims, mat_chars=['non_sing'])
            run_test(matsize, batchdims, mat_chars=['sym', 'sym_pd', 'sym_psd'])
            run_test(matsize, batchdims, mat_chars=['sing', 'non_sing'])

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(torch.float32, torch.float64, torch.complex64, torch.complex128)
    def test_cholesky_inverse(self, device, dtype):
        from torch.testing._internal.common_utils import random_hermitian_pd_matrix

        def run_test(shape, batch, upper, contiguous):
            A = random_hermitian_pd_matrix(shape, *batch, dtype=dtype, device=device)
            if A.numel() > 0 and not contiguous:
                A = A.transpose(-2, -1)
                self.assertFalse(A.is_contiguous())
            L = torch.linalg.cholesky(A)
            expected_inverse = torch.inverse(A)
            L = L.conj().transpose(-2, -1) if upper else L
            actual_inverse = torch.cholesky_inverse(L, upper)
            self.assertEqual(actual_inverse, expected_inverse)

        shapes = (0, 3, 5)
        batches = ((), (0,), (3, ), (2, 2))
        for shape, batch, upper, contiguous in list(itertools.product(shapes, batches, (True, False), (True, False))):
            run_test(shape, batch, upper, contiguous)

        # check the out= variant
        A = random_hermitian_pd_matrix(3, 2, dtype=dtype, device=device)
        L = torch.linalg.cholesky(A)

        # There are two code paths currently for the out= variant
        # 1. When 'out' tensor is in Fortran (column-major) memory format
        # then the fast route is taken and the storage is reused directly in the computations
        # 2. When 'out' tensor is not in Fortran format then a temporary tensor is allocated internally
        # and the result is copied from the temporary tensor to 'out' tensor

        # This test checks the first code path
        out = torch.empty_like(A)
        out_t = out.transpose(-2, -1).clone(memory_format=torch.contiguous_format)
        out = out_t.transpose(-2, -1)
        ans = torch.cholesky_inverse(L, out=out)
        self.assertEqual(ans, out)
        expected = torch.inverse(A)
        self.assertEqual(expected, out)

        # This test checks the second code path
        out = torch.empty_like(A)
        ans = torch.cholesky_inverse(L, out=out)
        self.assertEqual(ans, out)
        expected = torch.inverse(A)
        self.assertEqual(expected, out)

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(torch.float32, torch.float64, torch.complex64, torch.complex128)
    def test_cholesky_inverse_errors_and_warnings(self, device, dtype):
        # cholesky_inverse requires the input to be at least 2 dimensional tensor
        a = torch.randn(2, device=device, dtype=dtype)
        with self.assertRaisesRegex(RuntimeError, "must have at least 2 dimensions"):
            torch.cholesky_inverse(a)

        # cholesky_inverse requires a square matrix
        a = torch.randn(2, 3, device=device, dtype=dtype)
        with self.assertRaisesRegex(RuntimeError, "must be batches of square matrices"):
            torch.cholesky_inverse(a)

        # if non-empty out tensor with wrong shape is passed a warning is given
        a = torch.randn(3, 3, device=device, dtype=dtype)
        out = torch.empty(2, 3, device=device, dtype=dtype)
        with warnings.catch_warnings(record=True) as w:
            # Trigger warning
            torch.cholesky_inverse(a, out=out)
            # Check warning occurs
            self.assertEqual(len(w), 1)
            self.assertTrue("An output with one or more elements was resized" in str(w[-1].message))

        # dtypes should match
        out = torch.empty_like(a).to(torch.int)
        with self.assertRaisesRegex(RuntimeError, "dtype Int does not match input dtype"):
            torch.cholesky_inverse(a, out=out)

        # device should match
        if torch.cuda.is_available():
            wrong_device = 'cpu' if self.device_type != 'cpu' else 'cuda'
            out = torch.empty(0, device=wrong_device, dtype=dtype)
            with self.assertRaisesRegex(RuntimeError, "does not match input device"):
                torch.cholesky_inverse(a, out=out)

        # cholesky_inverse raises an error for invalid inputs on CPU
        # for example if at least one diagonal element is zero
        a = torch.randn(3, 3, device=device, dtype=dtype)
        a[1, 1] = 0
        if self.device_type == 'cpu':
            with self.assertRaisesRegex(RuntimeError, r"cholesky_inverse: U\(2,2\) is zero, singular U\."):
                torch.cholesky_inverse(a)
        # cholesky_inverse on GPU does not raise an error for this case
        elif self.device_type == 'cuda':
            out = torch.cholesky_inverse(a)
            self.assertTrue(out.isinf().any() or out.isnan().any())

    def _select_broadcastable_dims(self, dims_full=None):
        # select full dimensionality
        if dims_full is None:
            dims_full = []
            ndims = random.randint(1, 4)
            dims_full = [random.randint(1, 8) for _ in range(ndims)]
        else:
            ndims = len(dims_full)

        # select actual dimensions for ops:
        # larger: full ndims, individual sizes may be reduced
        # smaller: possibly reduced ndims, sizes may be reduced
        smaller_ndims = random.randint(1, ndims)
        dims_small = []
        dims_large = []
        for i in range(ndims - 1, -1, -1):
            j = random.randint(1, 3)
            if j == 1:  # no reduced singleton dimension
                ds = dims_full[i]
                dl = dims_full[i]
            elif j == 2:  # larger may have reduced singleton dimension
                ds = dims_full[i]
                dl = 1 if len(dims_small) < smaller_ndims else dims_full[i]
            elif j == 3:  # smaller may have reduced singleton dimension
                ds = 1
                dl = dims_full[i]
            dims_large = [dl] + dims_large
            if len(dims_small) < smaller_ndims:
                dims_small = [ds] + dims_small
        return (dims_small, dims_large, dims_full)

    def test_broadcast_fused_matmul(self, device):
        fns = ["baddbmm", "addbmm", "addmm", "addmv", "addr"]

        for fn in fns:
            batch_dim = random.randint(1, 8)
            n_dim = random.randint(1, 8)
            m_dim = random.randint(1, 8)
            p_dim = random.randint(1, 8)

            def dims_full_for_fn():
                if fn == "baddbmm":
                    return ([batch_dim, n_dim, p_dim], [batch_dim, n_dim, m_dim], [batch_dim, m_dim, p_dim])
                elif fn == "addbmm":
                    return ([n_dim, p_dim], [batch_dim, n_dim, m_dim], [batch_dim, m_dim, p_dim])
                elif fn == "addmm":
                    return ([n_dim, p_dim], [n_dim, m_dim], [m_dim, p_dim])
                elif fn == "addmv":
                    return ([n_dim], [n_dim, m_dim], [m_dim])
                elif fn == "addr":
                    return ([n_dim, m_dim], [n_dim], [m_dim])
                else:
                    raise AssertionError("unknown function")

            (t0_dims_full, t1_dims, t2_dims) = dims_full_for_fn()
            (t0_dims_small, _, _) = self._select_broadcastable_dims(t0_dims_full)

            t0_small = torch.randn(*t0_dims_small, device=device).float()
            t1 = torch.randn(*t1_dims, device=device).float()
            t2 = torch.randn(*t2_dims, device=device).float()

            t0_full = t0_small.expand(*t0_dims_full).to(device)

            fntorch = getattr(torch, fn)
            r0 = fntorch(t0_small, t1, t2)
            r1 = fntorch(t0_full, t1, t2)
            self.assertEqual(r0, r1)

    @tf32_on_and_off(0.001)
    def test_broadcast_batched_matmul(self, device):
        n_dim = random.randint(1, 8)
        m_dim = random.randint(1, 8)
        p_dim = random.randint(1, 8)
        full_batch_dims = [random.randint(1, 3) for i in range(random.randint(1, 3))]
        (batch_dims_small, _, _) = self._select_broadcastable_dims(full_batch_dims)

        def verify_batched_matmul(full_lhs, one_dimensional):
            if not one_dimensional:
                lhs_dims = [n_dim, m_dim]
                rhs_dims = [m_dim, p_dim]
                result_dims = [n_dim, p_dim]
            else:
                lhs_dims = [n_dim, m_dim] if full_lhs else [m_dim]
                rhs_dims = [m_dim, p_dim] if not full_lhs else [m_dim]
                result_dims = [n_dim] if full_lhs else [p_dim]

            lhs_mat_dims = lhs_dims if len(lhs_dims) != 1 else [1, m_dim]
            rhs_mat_dims = rhs_dims if len(rhs_dims) != 1 else [m_dim, 1]
            full_mat_dims = lhs_mat_dims if full_lhs else rhs_mat_dims
            dim0_dims = rhs_dims if full_lhs else lhs_dims
            small_dims = batch_dims_small + (rhs_mat_dims if full_lhs else lhs_mat_dims)

            small = torch.randn(*(small_dims), device=device).float()
            dim0 = torch.randn(*(dim0_dims), device=device).float()
            full = torch.randn(*(full_batch_dims + full_mat_dims), device=device).float()
            if not one_dimensional:
                (lhsTensors, rhsTensors) = ((full,), (small, dim0)) if full_lhs else ((small, dim0), (full,))
            else:
                (lhsTensors, rhsTensors) = ((full,), (dim0,)) if full_lhs else ((dim0,), (full,))

            def maybe_squeeze_result(l, r, result):
                if len(lhs_dims) == 1 and l.dim() != 1:
                    return result.squeeze(-2)
                elif len(rhs_dims) == 1 and r.dim() != 1:
                    return result.squeeze(-1)
                else:
                    return result

            for lhs in lhsTensors:
                lhs_expanded = lhs.expand(*(torch.Size(full_batch_dims) + torch.Size(lhs_mat_dims)))
                lhs_expanded_matmul_fn = lhs_expanded.matmul
                for rhs in rhsTensors:
                    rhs_expanded = ((rhs if len(rhs_dims) != 1 else rhs.unsqueeze(-1)).
                                    expand(*(torch.Size(full_batch_dims) + torch.Size(rhs_mat_dims))))
                    truth = maybe_squeeze_result(lhs_expanded, rhs_expanded, lhs_expanded_matmul_fn(rhs_expanded))
                    for l in (lhs, lhs_expanded):
                        for r in (rhs, rhs_expanded):
                            l_matmul_fn = l.matmul
                            result = maybe_squeeze_result(l, r, l_matmul_fn(r))
                            self.assertEqual(truth, result)
                            # test torch.matmul function as well
                            torch_result = maybe_squeeze_result(l, r, torch.matmul(l, r))
                            self.assertEqual(truth, torch_result)
                            # test torch.matmul with out
                            out = torch.zeros_like(torch_result)
                            torch.matmul(l, r, out=out)
                            self.assertEqual(truth, maybe_squeeze_result(l, r, out))

                # compare to bmm
                bmm_result = (torch.bmm(lhs_expanded.contiguous().view(-1, *lhs_mat_dims),
                                        rhs_expanded.contiguous().view(-1, *rhs_mat_dims)))
                self.assertEqual(truth.view(-1, *result_dims), bmm_result.view(-1, *result_dims))

        for indices in itertools.product((True, False), repeat=2):
            verify_batched_matmul(*indices)

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(torch.float32, torch.float64, torch.complex64, torch.complex128)
    @skipCUDAIfRocm
    def test_lu_solve_batched_non_contiguous(self, device, dtype):
        from numpy.linalg import solve
        from torch.testing._internal.common_utils import random_fullrank_matrix_distinct_singular_value

        A = random_fullrank_matrix_distinct_singular_value(2, 2, dtype=dtype, device='cpu')
        b = torch.randn(2, 2, 2, dtype=dtype, device='cpu')
        x_exp = torch.as_tensor(solve(A.permute(0, 2, 1).numpy(), b.permute(2, 1, 0).numpy())).to(device)
        A = A.to(device).permute(0, 2, 1)
        b = b.to(device).permute(2, 1, 0)
        assert not A.is_contiguous() and not b.is_contiguous(), "contiguous inputs"
        LU_data, LU_pivots = torch.lu(A)
        x = torch.lu_solve(b, LU_data, LU_pivots)
        self.assertEqual(x, x_exp)

    def lu_solve_test_helper(self, A_dims, b_dims, pivot, device, dtype):
        from torch.testing._internal.common_utils import random_fullrank_matrix_distinct_singular_value

        b = torch.randn(*b_dims, dtype=dtype, device=device)
        A = random_fullrank_matrix_distinct_singular_value(*A_dims, dtype=dtype).to(device)
        LU_data, LU_pivots, info = torch.lu(A, get_infos=True, pivot=pivot)
        self.assertEqual(info, torch.zeros_like(info))
        return b, A, LU_data, LU_pivots

    @skipCPUIfNoLapack
    @skipCUDAIfNoMagma
    @dtypes(torch.float32, torch.float64, torch.complex64, torch.complex128)
    @precisionOverride({torch.float32: 1e-3, torch.complex64: 1e-3,
                        torch.float64: 1e-8, torch.complex128: 1e-8})
    def test_lu_solve(self, device, dtype):
        def sub_test(pivot):
            for k, n in zip([2, 3, 5], [3, 5, 7]):
                b, A, LU_data, LU_pivots = self.lu_solve_test_helper((n,), (n, k), pivot, device, dtype)
                x = torch.lu_solve(b, LU_data, LU_pivots)
                self.assertEqual(b, np.matmul(A.cpu(), x.cpu()))

        sub_test(True)
        if self.device_type == 'cuda':
            sub_test(False)

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(torch.float32, torch.float64, torch.complex64, torch.complex128)
    @precisionOverride({torch.float32: 1e-3, torch.complex64: 1e-3,
                        torch.float64: 1e-8, torch.complex128: 1e-8})
    @skipCUDAIfRocm
    def test_lu_solve_batched(self, device, dtype):
        def sub_test(pivot):
            def lu_solve_batch_test_helper(A_dims, b_dims, pivot):
                b, A, LU_data, LU_pivots = self.lu_solve_test_helper(A_dims, b_dims, pivot, device, dtype)
                x_exp_list = []
                for i in range(b_dims[0]):
                    x_exp_list.append(torch.lu_solve(b[i], LU_data[i], LU_pivots[i]))
                x_exp = torch.stack(x_exp_list)  # Stacked output
                x_act = torch.lu_solve(b, LU_data, LU_pivots)  # Actual output
                self.assertEqual(x_exp, x_act)  # Equality check
                Ax = np.matmul(A.cpu(), x_act.cpu())
                self.assertEqual(b, Ax)

            for batchsize in [1, 3, 4]:
                lu_solve_batch_test_helper((5, batchsize), (batchsize, 5, 10), pivot)

        # Tests tensors with 0 elements
        b = torch.randn(3, 0, 3, dtype=dtype, device=device)
        A = torch.randn(3, 0, 0, dtype=dtype, device=device)
        LU_data, LU_pivots = torch.lu(A)
        self.assertEqual(torch.empty_like(b), b.lu_solve(LU_data, LU_pivots))

        sub_test(True)
        if self.device_type == 'cuda':
            sub_test(False)

    @slowTest
    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(torch.float32, torch.float64, torch.complex64, torch.complex128)
    def test_lu_solve_batched_many_batches(self, device, dtype):
        def run_test(A_dims, b_dims):
            b, A, LU_data, LU_pivots = self.lu_solve_test_helper(A_dims, b_dims, True, device, dtype)
            x = torch.lu_solve(b, LU_data, LU_pivots)
            Ax = torch.matmul(A, x)
            self.assertEqual(Ax, b.expand_as(Ax))

        run_test((5, 65536), (65536, 5, 10))
        run_test((5, 262144), (262144, 5, 10))

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(torch.float32, torch.float64, torch.complex64, torch.complex128)
    @skipCUDAIfRocm
    def test_lu_solve_batched_broadcasting(self, device, dtype):
        from numpy.linalg import solve
        from torch.testing._internal.common_utils import random_fullrank_matrix_distinct_singular_value

        def run_test(A_dims, b_dims, pivot=True):
            A_matrix_size = A_dims[-1]
            A_batch_dims = A_dims[:-2]
            A = random_fullrank_matrix_distinct_singular_value(A_matrix_size, *A_batch_dims, dtype=dtype)
            b = torch.randn(*b_dims, dtype=dtype)
            x_exp = torch.as_tensor(solve(A.numpy(), b.numpy())).to(dtype=dtype, device=device)
            A, b = A.to(device), b.to(device)
            LU_data, LU_pivots = torch.lu(A, pivot=pivot)
            x = torch.lu_solve(b, LU_data, LU_pivots)
            self.assertEqual(x, x_exp)

        # test against numpy.linalg.solve
        run_test((2, 1, 3, 4, 4), (2, 1, 3, 4, 6))  # no broadcasting
        run_test((2, 1, 3, 4, 4), (4, 6))  # broadcasting b
        run_test((4, 4), (2, 1, 3, 4, 2))  # broadcasting A
        run_test((1, 3, 1, 4, 4), (2, 1, 3, 4, 5))  # broadcasting A & b

    @precisionOverride({torch.float32: 1e-5, torch.complex64: 1e-5})
    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(torch.float32, torch.float64, torch.complex64, torch.complex128)
    def test_symeig(self, device, dtype):
        from torch.testing._internal.common_utils import random_hermitian_matrix

        def run_test(dims, eigenvectors, upper):
            x = random_hermitian_matrix(*dims, dtype=dtype, device=device)
            if dtype.is_complex:
                real_dtype = torch.float32 if dtype is torch.complex64 else torch.float64
            else:
                real_dtype = dtype
            oute = torch.empty(dims[1:] + dims[:1], dtype=real_dtype, device=device)
            outv = torch.empty(dims[1:] + dims[:1] * 2, dtype=dtype, device=device)
            torch.symeig(x, eigenvectors=eigenvectors, upper=upper, out=(oute, outv))

            if eigenvectors:
                outv_ = outv.cpu().numpy()
                x_recon = np.matmul(np.matmul(outv_, torch.diag_embed(oute.to(dtype)).cpu().numpy()),
                                    outv_.swapaxes(-2, -1).conj())
                self.assertEqual(x, x_recon, atol=1e-8, rtol=0, msg='Incorrect reconstruction using V @ diag(e) @ V.T')
            else:
                eigvals, _ = torch.symeig(x, eigenvectors=True, upper=upper)
                self.assertEqual(eigvals, oute, msg='Eigenvalues mismatch')
                self.assertEqual(torch.empty(0, device=device, dtype=dtype), outv, msg='Eigenvector matrix not empty')

            rese, resv = x.symeig(eigenvectors=eigenvectors, upper=upper)
            self.assertEqual(rese, oute, msg="outputs of symeig and symeig with out don't match")
            self.assertEqual(resv, outv, msg="outputs of symeig and symeig with out don't match")

            # test non-contiguous
            x = random_hermitian_matrix(*dims, dtype=dtype, device=device)
            n_dim = len(dims) + 1
            # Reverse the batch dimensions and the matrix dimensions and then concat them
            x = x.permute(tuple(range(n_dim - 3, -1, -1)) + (n_dim - 1, n_dim - 2))
            assert not x.is_contiguous(), "x is intentionally non-contiguous"
            rese, resv = torch.symeig(x, eigenvectors=eigenvectors, upper=upper)
            if eigenvectors:
                resv_ = resv.cpu().numpy()
                x_recon = np.matmul(np.matmul(resv_, torch.diag_embed(rese.to(dtype)).cpu().numpy()),
                                    resv_.swapaxes(-2, -1).conj())
                self.assertEqual(x, x_recon, atol=1e-8, rtol=0, msg='Incorrect reconstruction using V @ diag(e) @ V.T')
            else:
                eigvals, _ = torch.symeig(x, eigenvectors=True, upper=upper)
                self.assertEqual(eigvals, rese, msg='Eigenvalues mismatch')
                self.assertEqual(torch.empty(0, device=device, dtype=dtype), resv, msg='Eigenvector matrix not empty')

        batch_dims_set = [(), (3,), (3, 5), (5, 3, 5)]
        for batch_dims, eigenvectors, upper in itertools.product(batch_dims_set, (True, False), (True, False)):
            run_test((5,) + batch_dims, eigenvectors, upper)

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    def test_pca_lowrank(self, device):
        from torch.testing._internal.common_utils import random_lowrank_matrix, random_sparse_matrix

        dtype = torch.double

        def run_subtest(guess_rank, actual_rank, matrix_size, batches, device, pca, **options):
            density = options.pop('density', 1)
            if isinstance(matrix_size, int):
                rows = columns = matrix_size
            else:
                rows, columns = matrix_size
            if density == 1:
                a_input = random_lowrank_matrix(actual_rank, rows, columns, *batches, device=device, dtype=dtype)
                a = a_input
            else:
                a_input = random_sparse_matrix(rows, columns, density, device=device, dtype=dtype)
                a = a_input.to_dense()

            u, s, v = pca(a_input, q=guess_rank, **options)

            self.assertEqual(s.shape[-1], guess_rank)
            self.assertEqual(u.shape[-2], rows)
            self.assertEqual(u.shape[-1], guess_rank)
            self.assertEqual(v.shape[-1], guess_rank)
            self.assertEqual(v.shape[-2], columns)

            A1 = u.matmul(s.diag_embed()).matmul(v.transpose(-2, -1))
            ones_m1 = torch.ones(batches + (rows, 1), dtype=a.dtype, device=device)
            c = a.sum(axis=-2) / rows
            c = c.reshape(batches + (1, columns))
            A2 = a - ones_m1.matmul(c)
            self.assertEqual(A1, A2)

            if density == 1:
                # actual rank is known only for dense input
                detect_rank = (s.abs() > 1e-5).sum(axis=-1)
                self.assertEqual(actual_rank * torch.ones(batches, device=device, dtype=torch.int64), detect_rank)
                U, S, V = torch.svd(A2)
                self.assertEqual(s[..., :actual_rank], S[..., :actual_rank])

        all_batches = [(), (1,), (3,), (2, 3)]
        for actual_rank, size, all_batches in [
                (2, (17, 4), all_batches),
                (2, (100, 4), all_batches),
                (6, (100, 40), all_batches),
                (12, (1000, 1000), [()]),
        ]:
            for batches in all_batches:
                for guess_rank in [
                        actual_rank,
                        actual_rank + 2,
                        actual_rank + 6,
                ]:
                    if guess_rank <= min(*size):
                        run_subtest(guess_rank, actual_rank, size, batches, device, torch.pca_lowrank)
                        run_subtest(guess_rank, actual_rank, size[::-1], batches, device, torch.pca_lowrank)

        # sparse input
        for guess_rank, size in [
                (4, (17, 4)), (4, (4, 17)), (16, (17, 17)),
                (21, (100, 40)), (20, (40, 100)), (600, (1000, 1000))]:
            for density in [0.005, 0.1]:
                run_subtest(guess_rank, None, size, (), device, torch.pca_lowrank, density=density)

        # jitting support
        jitted = torch.jit.script(torch.pca_lowrank)
        guess_rank, actual_rank, size, batches = 2, 2, (17, 4), ()
        run_subtest(guess_rank, actual_rank, size, batches, device, jitted)

    # Ensure that nuclear_norm's out variant gives the same result as the non-out
    @onlyOnCPUAndCUDA
    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(torch.float32, torch.float64)
    def test_nuclear_norm_out(self, device, dtype):
        test_cases = [
            # input size, dim
            ((25, 25), None),
            ((25, 25), (0, 1)),
            ((25, 25), (1, 0)),
            ((25, 25, 25), (2, 0)),
            ((25, 25, 25), (0, 1)),
        ]
        for keepdim in [False, True]:
            for input_size, dim in test_cases:
                msg = f'input_size: {input_size}, dim: {dim}, keepdim: {keepdim}'
                x = torch.randn(*input_size, device=device, dtype=dtype)
                result_out = torch.empty(0, device=device, dtype=dtype)
                if dim is None:
                    result = torch.nuclear_norm(x, keepdim=keepdim)
                    torch.nuclear_norm(x, keepdim=keepdim, out=result_out)
                else:
                    result = torch.nuclear_norm(x, keepdim=keepdim, dim=dim)
                    torch.nuclear_norm(x, keepdim=keepdim, dim=dim, out=result_out)
                self.assertEqual(result, result_out, msg=msg)

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    def test_geqrf(self, device):
        a = torch.randn(5, 5, device=device)
        b, c = torch.geqrf(a)
        b_placeholder, c_placeholder = torch.empty_like(b), torch.empty_like(c)
        torch.geqrf(a, out=(b_placeholder, c_placeholder))
        self.assertEqual(b, b_placeholder)
        self.assertEqual(c, c_placeholder)

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(torch.double)
    def test_lstsq(self, device, dtype):
        def _test_underdetermined(a, b, expectedNorm):
            # underdetermined systems are only supported on CPU
            if self.device_type != 'cpu':
                return

            m = a.size()[0]
            n = a.size()[1]
            assert(m <= n)

            a_copy = a.clone()
            b_copy = b.clone()
            res1 = torch.lstsq(b, a)[0]
            self.assertEqual(a, a_copy, atol=0, rtol=0)
            self.assertEqual(b, b_copy, atol=0, rtol=0)
            self.assertEqual((torch.mm(a, res1) - b).norm(), expectedNorm, atol=1e-8, rtol=0)

            ta = torch.tensor((), dtype=dtype, device=device)
            tb = torch.tensor((), dtype=dtype, device=device)
            res2 = torch.lstsq(b, a, out=(tb, ta))[0]
            self.assertEqual(a, a_copy, atol=0, rtol=0)
            self.assertEqual(b, b_copy, atol=0, rtol=0)
            self.assertEqual((torch.mm(a, res1) - b).norm(), expectedNorm, atol=1e-8, rtol=0)

            res3 = torch.lstsq(b, a, out=(b, a))[0]
            self.assertEqual((torch.mm(a_copy, b) - b_copy).norm(), expectedNorm, atol=1e-8, rtol=0)
            self.assertEqual(res1, tb, atol=0, rtol=0)
            self.assertEqual(res1, b, atol=0, rtol=0)
            self.assertEqual(res1, res2, atol=0, rtol=0)
            self.assertEqual(res1, res3, atol=0, rtol=0)

        def _test_overdetermined(a, b, expectedNorm):
            m = a.size()[0]
            n = a.size()[1]
            assert(m > n)

            def check_norm(a, b, expected_norm, gels_result):
                # Checks |ax - b| and the residual info from the result

                # The first n rows is the least square solution.
                # Rows n to m-1 contain residual information.
                x = gels_result[:n]
                resid_info = gels_result[n:]

                resid_norm = (torch.mm(a, x) - b).norm()
                self.assertEqual(resid_norm, expectedNorm, atol=1e-8, rtol=0)
                self.assertEqual(resid_info.norm(), resid_norm, atol=1e-8, rtol=0)

            a_copy = a.clone()
            b_copy = b.clone()
            res1 = torch.lstsq(b, a)[0]
            self.assertEqual(a, a_copy, atol=0, rtol=0)
            self.assertEqual(b, b_copy, atol=0, rtol=0)
            check_norm(a, b, expectedNorm, res1)

            ta = torch.tensor((), dtype=dtype, device=device)
            tb = torch.tensor((), dtype=dtype, device=device)
            res2 = torch.lstsq(b, a, out=(tb, ta))[0]
            self.assertEqual(a, a_copy, atol=0, rtol=0)
            self.assertEqual(b, b_copy, atol=0, rtol=0)
            check_norm(a, b, expectedNorm, res2)

            res3 = torch.lstsq(b, a, out=(b, a))[0]
            check_norm(a_copy, b_copy, expectedNorm, res3)

            self.assertEqual(res1, tb, atol=0, rtol=0)
            self.assertEqual(res1, b, atol=0, rtol=0)
            self.assertEqual(res1, res2, atol=0, rtol=0)
            self.assertEqual(res1, res3, atol=0, rtol=0)

        # basic test
        expectedNorm = 0
        a = torch.tensor(((1.44, -9.96, -7.55, 8.34),
                          (-7.84, -0.28, 3.24, 8.09),
                          (-4.39, -3.24, 6.27, 5.28),
                          (4.53, 3.83, -6.64, 2.06)), dtype=dtype, device=device).t()
        b = torch.tensor(((8.58, 8.26, 8.48, -5.28),
                          (9.35, -4.43, -0.70, -0.26)), dtype=dtype, device=device).t()
        _test_underdetermined(a, b, expectedNorm)

        # test overdetermined
        expectedNorm = 17.390200628863
        a = torch.tensor(((1.44, -9.96, -7.55, 8.34, 7.08, -5.45),
                          (-7.84, -0.28, 3.24, 8.09, 2.52, -5.70),
                          (-4.39, -3.24, 6.27, 5.28, 0.74, -1.19),
                          (4.53, 3.83, -6.64, 2.06, -2.47, 4.70)), dtype=dtype, device=device).t()
        b = torch.tensor(((8.58, 8.26, 8.48, -5.28, 5.72, 8.93),
                          (9.35, -4.43, -0.70, -0.26, -7.36, -2.52)), dtype=dtype, device=device).t()
        _test_overdetermined(a, b, expectedNorm)

        # test underdetermined
        expectedNorm = 0
        a = torch.tensor(((1.44, -9.96, -7.55),
                          (-7.84, -0.28, 3.24),
                          (-4.39, -3.24, 6.27),
                          (4.53, 3.83, -6.64)), dtype=dtype, device=device).t()
        b = torch.tensor(((8.58, 8.26, 8.48),
                          (9.35, -4.43, -0.70)), dtype=dtype, device=device).t()
        _test_underdetermined(a, b, expectedNorm)

        # test reuse
        expectedNorm = 0
        a = torch.tensor(((1.44, -9.96, -7.55, 8.34),
                          (-7.84, -0.28, 3.24, 8.09),
                          (-4.39, -3.24, 6.27, 5.28),
                          (4.53, 3.83, -6.64, 2.06)), dtype=dtype, device=device).t()
        b = torch.tensor(((8.58, 8.26, 8.48, -5.28),
                          (9.35, -4.43, -0.70, -0.26)), dtype=dtype, device=device).t()
        ta = torch.tensor((), dtype=dtype, device=device)
        tb = torch.tensor((), dtype=dtype, device=device)
        torch.lstsq(b, a, out=(tb, ta))
        self.assertEqual((torch.mm(a, tb) - b).norm(), expectedNorm, atol=1e-8, rtol=0)
        torch.lstsq(b, a, out=(tb, ta))
        self.assertEqual((torch.mm(a, tb) - b).norm(), expectedNorm, atol=1e-8, rtol=0)
        torch.lstsq(b, a, out=(tb, ta))
        self.assertEqual((torch.mm(a, tb) - b).norm(), expectedNorm, atol=1e-8, rtol=0)

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    def test_lapack_empty(self, device):
        # FIXME: these are just a selection of LAPACK functions -- we need a general strategy here.
        # The LAPACK functions themselves generally do NOT work with zero sized dimensions, although
        # numpy/sci often has a direct wrapper (e.g. lu_factor) and a wrapper that "does the right thing"
        # (e.g. lu).  We often name our functions identically to the lapack function, so it will take work
        # to name / migrate-to better wrappers.
        def fn(torchfn, *args):
            return torchfn(*tuple(torch.randn(shape, device=device) if isinstance(shape, tuple) else shape
                                  for shape in args))

        # inverse, pinverse
        self.assertEqual((0, 0), fn(torch.inverse, (0, 0)).shape)
        self.assertEqual((5, 0), fn(torch.pinverse, (0, 5)).shape)
        self.assertEqual((0, 5), fn(torch.pinverse, (5, 0)).shape)
        self.assertEqual((0, 0), fn(torch.pinverse, (0, 0)).shape)

        # det, logdet, slogdet
        self.assertEqual(torch.tensor(1., device=device), fn(torch.det, (0, 0)))
        self.assertEqual(torch.tensor(0., device=device), fn(torch.logdet, (0, 0)))
        self.assertEqual((torch.tensor(1., device=device), torch.tensor(0., device=device)),
                         fn(torch.slogdet, (0, 0)))

        # eig, symeig
        evalues, evectors = fn(torch.eig, (0, 0), True)
        self.assertEqual([(0, 2), (0, 0)], [evalues.shape, evectors.shape])
        evalues, evectors = fn(torch.symeig, (0, 0), True)
        self.assertEqual([(0,), (0, 0)], [evalues.shape, evectors.shape])

        # qr
        q, r = fn(torch.qr, (3, 0), True)
        self.assertEqual([(3, 0), (0, 0)], [q.shape, r.shape])
        q, r = fn(torch.qr, (0, 3), True)
        self.assertEqual([(0, 0), (0, 3)], [q.shape, r.shape])
        q, r = fn(torch.qr, (3, 0), False)
        self.assertEqual([(3, 3), (3, 0)], [q.shape, r.shape])

        # lstsq
        self.assertRaises(RuntimeError, lambda: torch.lstsq(torch.randn(0, 0), torch.randn(0, 0)))
        self.assertRaises(RuntimeError, lambda: torch.lstsq(torch.randn(0,), torch.randn(0, 0)))

    @tf32_on_and_off(0.005)
    def test_tensordot(self, device):
        a = torch.arange(60., device=device).reshape(3, 4, 5)
        b = torch.arange(24., device=device).reshape(4, 3, 2)
        c = torch.tensordot(a, b, dims=([1, 0], [0, 1])).cpu()
        cn = torch.from_numpy(np.tensordot(a.cpu().numpy(), b.cpu().numpy(),
                                           axes=([1, 0], [0, 1])))
        self.assertEqual(c, cn)

        cout = torch.zeros((5, 2))
        torch.tensordot(a, b, dims=([1, 0], [0, 1]), out=cout).cpu()
        self.assertEqual(c, cout)

        a = torch.randn(2, 3, 4, 5, device=device)
        b = torch.randn(4, 5, 6, 7, device=device)
        c = torch.tensordot(a, b, dims=2).cpu()
        cn = torch.from_numpy(np.tensordot(a.cpu().numpy(), b.cpu().numpy(),
                                           axes=2))

        with self.assertRaisesRegex(RuntimeError, "expects dims >= 0"):
            torch.tensordot(a, b, dims=-1)

        self.assertEqual(c, cn)
        c = torch.tensordot(a, b).cpu()
        cn = torch.from_numpy(np.tensordot(a.cpu().numpy(), b.cpu().numpy()))
        self.assertEqual(c, cn)


instantiate_device_type_tests(TestLinalg, globals())

if __name__ == '__main__':
    run_tests()
