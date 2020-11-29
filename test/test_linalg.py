import torch
import unittest
import itertools
import warnings
from math import inf, nan, isnan
from random import randrange

from torch.testing._internal.common_utils import \
    (TestCase, run_tests, TEST_NUMPY, TEST_SCIPY, IS_MACOS, IS_WINDOWS, slowTest, TEST_WITH_ASAN, make_tensor)
from torch.testing._internal.common_device_type import \
    (instantiate_device_type_tests, dtypes,
     onlyCPU, skipCUDAIf, skipCUDAIfNoMagma, skipCPUIfNoLapack, precisionOverride,
     skipCUDAIfNoMagmaAndNoCusolver, skipCUDAIfRocm, onlyOnCPUAndCUDA)
from torch.testing._internal.common_cuda import tf32_on_and_off
from torch.testing._internal.jit_metaprogramming_utils import gen_script_fn_and_args
from torch.autograd import gradcheck, gradgradcheck

if TEST_NUMPY:
    import numpy as np

class TestLinalg(TestCase):
    exact_dtype = True

    # Tests torch.outer, and its alias, torch.ger, vs. NumPy
    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
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
    @skipCUDAIf(True, "See issue #26789.")
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
    @dtypes(torch.float32, torch.float64, torch.complex64, torch.complex128)
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
    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    @dtypes(torch.double, torch.cdouble)
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
        with self.assertRaisesRegex(RuntimeError, "result dtype Int does not match self dtype"):
            torch.kron(a, b, out=out)

    # This test confirms that torch.linalg.norm's dtype argument works
    # as expected, according to the function's documentation
    @skipCUDAIfNoMagma
    def test_norm_dtype(self, device):
        def run_test_case(input_size, ord, keepdim, from_dtype, to_dtype, compare_dtype):
            msg = (
                f'input_size={input_size}, ord={ord}, keepdim={keepdim}, '
                f'from_dtype={from_dtype}, to_dtype={to_dtype}')
            input = torch.randn(*input_size, dtype=from_dtype, device=device)
            result = torch.linalg.norm(input, ord, keepdim=keepdim, dtype=from_dtype)
            self.assertEqual(result.dtype, from_dtype, msg=msg)
            result_converted = torch.linalg.norm(input, ord, keepdim=keepdim, dtype=to_dtype)
            self.assertEqual(result_converted.dtype, to_dtype, msg=msg)
            self.assertEqual(result.to(compare_dtype), result_converted.to(compare_dtype), msg=msg)

            result_out_converted = torch.empty_like(result_converted)
            torch.linalg.norm(input, ord, keepdim=keepdim, dtype=to_dtype, out=result_out_converted)
            self.assertEqual(result_out_converted.dtype, to_dtype, msg=msg)
            self.assertEqual(result_converted, result_out_converted, msg=msg)

        ord_vector = [0, 1, -1, 2, -2, 3, -3, 4.5, -4.5, inf, -inf, None]
        ord_matrix = ['fro', 'nuc', 1, -1, 2, -2, inf, -inf, None]
        S = 10
        test_cases = [
            ((S, ), ord_vector),
            ((S, S), ord_matrix),
        ]
        for keepdim in [True, False]:
            for input_size, ord_settings in test_cases:
                for ord in ord_settings:
                    # float to double
                    run_test_case(input_size, ord, keepdim, torch.float, torch.double, torch.float)
                    # double to float
                    run_test_case(input_size, ord, keepdim, torch.double, torch.double, torch.float)

        # Make sure that setting dtype != out.dtype raises an error
        dtype_pairs = [
            (torch.float, torch.double),
            (torch.double, torch.float),
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
    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
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
    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    @dtypes(torch.float, torch.double)
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

    # Test autograd and jit functionality for linalg functions.
    # TODO: Once support for linalg functions is added to method_tests in common_methods_invocations.py,
    #       the `test_cases` entries below should be moved there. These entries are in a similar format,
    #       so they should work with minimal changes.
    @dtypes(torch.float, torch.double)
    def test_autograd_and_jit(self, device, dtype):
        torch.manual_seed(0)
        S = 10
        NO_ARGS = None  # NOTE: refer to common_methods_invocations.py if you need this feature
        test_cases = [
            # NOTE: Not all the features from common_methods_invocations.py are functional here, since this
            #       is only a temporary solution.
            # (
            #   method name,
            #   input size/constructing fn,
            #   args (tuple represents shape of a tensor arg),
            #   test variant name (will be used at test name suffix),    // optional
            #   (should_check_autodiff[bool], nonfusible_nodes, fusible_nodes) for autodiff, // optional
            #   indices for possible dim arg,                            // optional
            #   fn mapping output to part that should be gradcheck'ed,   // optional
            #   kwargs                                                   // optional
            # )
            ('norm', (S,), (), 'default_1d'),
            ('norm', (S, S), (), 'default_2d'),
            ('norm', (S, S, S), (), 'default_3d'),
            ('norm', (S,), (inf,), 'vector_inf'),
            ('norm', (S,), (3.5,), 'vector_3_5'),
            ('norm', (S,), (0.5,), 'vector_0_5'),
            ('norm', (S,), (2,), 'vector_2'),
            ('norm', (S,), (1,), 'vector_1'),
            ('norm', (S,), (0,), 'vector_0'),
            ('norm', (S,), (-inf,), 'vector_neg_inf'),
            ('norm', (S,), (-3.5,), 'vector_neg_3_5'),
            ('norm', (S,), (-0.5,), 'vector_neg_0_5'),
            ('norm', (S,), (2,), 'vector_neg_2'),
            ('norm', (S,), (1,), 'vector_neg_1'),
            ('norm', (S, S), (inf,), 'matrix_inf'),
            ('norm', (S, S), (2,), 'matrix_2', (), NO_ARGS, [skipCPUIfNoLapack, skipCUDAIfNoMagma]),
            ('norm', (S, S), (1,), 'matrix_1'),
            ('norm', (S, S), (-inf,), 'matrix_neg_inf'),
            ('norm', (S, S), (-2,), 'matrix_neg_2', (), NO_ARGS, [skipCPUIfNoLapack, skipCUDAIfNoMagma]),
            ('norm', (S, S), (-1,), 'matrix_neg_1'),
            ('norm', (S, S), ('fro',), 'fro'),
            ('norm', (S, S), ('fro', [0, 1]), 'fro_dim'),
            ('norm', (S, S), ('nuc',), 'nuc', (), NO_ARGS, [skipCPUIfNoLapack, skipCUDAIfNoMagma]),
            ('norm', (S, S), ('nuc', [0, 1]), 'nuc_dim', (), NO_ARGS, [skipCPUIfNoLapack, skipCUDAIfNoMagma]),
        ]
        for test_case in test_cases:
            func_name = test_case[0]
            func = getattr(torch.linalg, func_name)
            input_size = test_case[1]
            args = list(test_case[2])
            test_case_name = test_case[3] if len(test_case) >= 4 else None
            mapping_funcs = list(test_case[6]) if len(test_case) >= 7 else None

            # Skip a test if a decorator tells us to
            if mapping_funcs is not None:
                def decorated_func(self, device, dtype):
                    pass
                for mapping_func in mapping_funcs:
                    decorated_func = mapping_func(decorated_func)
                try:
                    decorated_func(self, device, dtype)
                except unittest.SkipTest:
                    continue

            msg = f'function name: {func_name}, case name: {test_case_name}'

            # Test JIT
            input = torch.randn(*input_size, dtype=dtype, device=device)
            input_script = input.clone().detach()
            script_method, tensors = gen_script_fn_and_args("linalg.norm", "functional", input_script, *args)
            self.assertEqual(
                func(input, *args),
                script_method(input_script),
                msg=msg)

            # Test autograd
            # gradcheck is only designed to work with torch.double inputs
            if dtype == torch.double:
                input = torch.randn(*input_size, dtype=dtype, device=device, requires_grad=True)

                def run_func(input):
                    return func(input, *args)
                self.assertTrue(gradcheck(run_func, input), msg=msg)

    # This test calls torch.linalg.norm and numpy.linalg.norm with illegal arguments
    # to ensure that they both throw errors
    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
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

    # Test complex number inputs for linalg.norm. Some cases are not supported yet, so
    # this test also verifies that those cases raise an error.
    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @unittest.skipIf(not TEST_NUMPY, "Numpy not found")
    @dtypes(torch.cfloat, torch.cdouble)
    def test_norm_complex(self, device, dtype):
        def gen_error_message(input_size, ord, keepdim, dim=None):
            return "complex norm failed for input size %s, ord=%s, keepdim=%s, dim=%s" % (
                input_size, ord, keepdim, dim)

        if self.device_type == 'cpu':
            supported_vector_ords = [0, 1, 3, inf, -1, -2, -3, -inf]
            supported_matrix_ords = ['nuc', 1, 2, inf, -1, -2, -inf]
            unsupported_vector_ords = [
                (2, r'norm with p=2 not supported for complex tensors'),
                (None, r'norm with p=2 not supported for complex tensors'),
            ]
            unsupported_matrix_ords = [
                ('fro', r'frobenius norm not supported for complex tensors'),
                (None, r'norm with p=2 not supported for complex tensors'),
            ]

        elif self.device_type == 'cuda':
            supported_vector_ords = [inf, -inf]
            supported_matrix_ords = [1, inf, -1, -inf]
            unsupported_vector_ords = [
                (0, r'norm_cuda" not implemented for \'Complex'),
                (1, r'norm_cuda" not implemented for \'Complex'),
                (2, r'norm with p=2 not supported for complex tensors'),
                (-1, r'norm_cuda" not implemented for \'Complex'),
                (-2, r'norm_cuda" not implemented for \'Complex'),
                (None, r'norm with p=2 not supported for complex tensors'),
            ]
            unsupported_matrix_ords = [
                (None, r'norm with p=2 not supported for complex tensors'),
                ('fro', r'frobenius norm not supported for complex tensors'),
            ]

        # Test supported ords
        for keepdim in [False, True]:
            # vector norm
            x = torch.randn(25, device=device, dtype=dtype)
            xn = x.cpu().numpy()
            for ord in supported_vector_ords:
                res = torch.linalg.norm(x, ord, keepdim=keepdim).cpu()
                expected = np.linalg.norm(xn, ord, keepdims=keepdim)
                msg = gen_error_message(x.size(), ord, keepdim)
                self.assertEqual(res.shape, expected.shape, msg=msg)
                self.assertEqual(res, expected, msg=msg)

            # matrix norm
            x = torch.randn(25, 25, device=device, dtype=dtype)
            xn = x.cpu().numpy()
            for ord in supported_matrix_ords:
                # TODO: Need to fix abort when nuclear norm is given cdouble input:
                #       "double free or corruption (!prev) Aborted (core dumped)"
                if ord == 'nuc' and dtype == torch.cdouble:
                    continue
                res = torch.linalg.norm(x, ord, keepdim=keepdim).cpu()
                expected = np.linalg.norm(xn, ord, keepdims=keepdim)
                msg = gen_error_message(x.size(), ord, keepdim)
                self.assertEqual(res.shape, expected.shape, msg=msg)
                self.assertEqual(res, expected, msg=msg)

        # Test unsupported ords
        # vector norm
        x = torch.randn(25, device=device, dtype=dtype)
        for ord, error_msg in unsupported_vector_ords:
            with self.assertRaisesRegex(RuntimeError, error_msg):
                torch.linalg.norm(x, ord)

        # matrix norm
        x = torch.randn(25, 25, device=device, dtype=dtype)
        for ord, error_msg in unsupported_matrix_ords:
            with self.assertRaisesRegex(RuntimeError, error_msg):
                torch.linalg.norm(x, ord)

    # Test that linal.norm gives the same result as numpy when inputs
    # contain extreme values (inf, -inf, nan)
    @unittest.skipIf(IS_WINDOWS, "Skipped on Windows!")
    @unittest.skipIf(IS_MACOS, "Skipped on MacOS!")
    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @unittest.skipIf(not TEST_NUMPY, "Numpy not found")
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
                if dtype in [torch.cfloat, torch.cdouble] and ord in [2, None]:
                    # TODO: Once these ord values have support for complex numbers,
                    #       remove this error test case
                    with self.assertRaises(RuntimeError):
                        torch.linalg.norm(input, ord, dim, keepdim)
                    return
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
    @unittest.skipIf(not TEST_NUMPY, "Numpy not found")
    @dtypes(torch.float, torch.double, torch.cfloat, torch.cdouble)
    def test_norm_matrix_degenerate_shapes(self, device, dtype):
        def run_test_case(input, ord, dim, keepdim, should_error):
            if dtype in [torch.cfloat, torch.cdouble] and ord in ['fro', None]:
                # TODO: Once these ord values have support for complex numbers,
                #       remove this error test case
                with self.assertRaises(RuntimeError):
                    torch.linalg.norm(input, ord, dim, keepdim)
                return
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

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @unittest.skipIf(not TEST_NUMPY, "Numpy not found")
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
    @unittest.skipIf(not TEST_NUMPY, "Numpy not found")
    def test_norm_complex_old(self, device):
        def gen_error_message(input_size, p, keepdim, dim=None):
            return "complex norm failed for input size %s, p=%s, keepdim=%s, dim=%s" % (
                input_size, p, keepdim, dim)

        if device == 'cpu':
            for keepdim in [False, True]:
                # vector norm
                x = torch.randn(25, device=device) + 1j * torch.randn(25, device=device)
                xn = x.cpu().numpy()
                for p in [0, 1, 3, inf, -1, -2, -3, -inf]:
                    res = x.norm(p, keepdim=keepdim).cpu()
                    expected = np.linalg.norm(xn, p, keepdims=keepdim)
                    msg = gen_error_message(x.size(), p, keepdim)
                    self.assertEqual(res.shape, expected.shape, msg=msg)
                    self.assertEqual(res, expected, msg=msg)

                # matrix norm
                x = torch.randn(25, 25, device=device) + 1j * torch.randn(25, 25, device=device)
                xn = x.cpu().numpy()
                for p in ['nuc']:
                    res = x.norm(p, keepdim=keepdim).cpu()
                    expected = np.linalg.norm(xn, p, keepdims=keepdim)
                    msg = gen_error_message(x.size(), p, keepdim)
                    self.assertEqual(res.shape, expected.shape, msg=msg)
                    self.assertEqual(res, expected, msg=msg)

            # TODO: remove error test and add functionality test above when 2-norm support is added
            with self.assertRaisesRegex(RuntimeError, r'norm with p=2 not supported for complex tensors'):
                x = torch.randn(2, device=device, dtype=torch.complex64).norm(p=2)

            # TODO: remove error test and add functionality test above when frobenius support is added
            with self.assertRaisesRegex(RuntimeError, r'frobenius norm not supported for complex tensors'):
                x = torch.randn(2, 2, device=device, dtype=torch.complex64).norm(p='fro')

        elif device == 'cuda':
            with self.assertRaisesRegex(RuntimeError, r'"norm_cuda" not implemented for \'ComplexFloat\''):
                (1j * torch.randn(25)).norm()

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
    @unittest.skipIf(not TEST_NUMPY, "Numpy not found")
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

    @skipCUDAIfNoMagmaAndNoCusolver
    @skipCPUIfNoLapack
    @dtypes(torch.float32, torch.float64, torch.complex64, torch.complex128)
    @precisionOverride({torch.float32: 2e-3, torch.complex64: 2e-3})
    def test_inverse(self, device, dtype):
        from torch.testing._internal.common_utils import random_fullrank_matrix_distinct_singular_value

        def run_test(matrix, batches, n):
            matrix_inverse = torch.inverse(matrix)

            # Compare against NumPy output
            # NumPy uses 'gesv' LAPACK routine solving the equation A A_inv = I
            # But in PyTorch 'gertf' + 'getri' is used causing element-wise differences
            expected = np.linalg.inv(matrix.cpu().numpy())
            self.assertEqual(matrix_inverse, expected, atol=self.precision, rtol=1e-4)

            # Additional correctness tests, check matrix*matrix_inverse == identity
            identity = torch.eye(n, dtype=dtype, device=device)
            self.assertEqual(identity.expand_as(matrix), torch.matmul(matrix, matrix_inverse))
            self.assertEqual(identity.expand_as(matrix), torch.matmul(matrix_inverse, matrix))

            # check the out= variant
            matrix_inverse_out = torch.empty(*batches, n, n, dtype=dtype, device=device)
            ans = torch.inverse(matrix, out=matrix_inverse_out)
            self.assertEqual(matrix_inverse_out, ans, atol=0, rtol=0)
            self.assertEqual(matrix_inverse_out, matrix_inverse, atol=0, rtol=0)

            # batched matrices: 3+ dimensional tensors, check matrix_inverse same as single-inverse for each matrix
            if matrix.ndim > 2:
                expected_inv_list = []
                p = int(np.prod(batches))  # use `p` instead of -1, so that the test works for empty input as well
                for mat in matrix.contiguous().view(p, n, n):
                    expected_inv_list.append(torch.inverse(mat))
                expected_inv = torch.stack(expected_inv_list).view(*batches, n, n)
                if self.device_type == 'cuda' and dtype in [torch.float32, torch.complex64]:
                    # single-inverse is done using cuSOLVER, while batched inverse is done using MAGMA
                    # individual values can be significantly different for fp32, hence rather high rtol is used
                    # the important thing is that torch.inverse passes above checks with identity
                    self.assertEqual(matrix_inverse, expected_inv, atol=1e-1, rtol=1e-2)
                else:
                    self.assertEqual(matrix_inverse, expected_inv)

        for batches, n in itertools.product(
            [[], [1], [4], [2, 3]],
            [0, 5, 64]
        ):
            # large batch size and large matrix size will be tested in test_inverse_many_batches (slow test)
            if batches and batches[0] == 32 and n == 256:
                continue
            matrices = random_fullrank_matrix_distinct_singular_value(n, *batches, dtype=dtype).to(device)
            run_test(matrices, batches, n)

            # test non-contiguous input
            run_test(matrices.transpose(-2, -1), batches, n)
            if n > 0:
                run_test(
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

        def test_inverse_many_batches_helper(b, n):
            matrices = random_fullrank_matrix_distinct_singular_value(b, n, n, dtype=dtype).to(device)
            matrices_inverse = torch.inverse(matrices)

            # Compare against NumPy output
            expected = np.linalg.inv(matrices.cpu().numpy())
            self.assertEqual(matrices_inverse, expected, atol=self.precision, rtol=1e-4)

        test_inverse_many_batches_helper(5, 256)
        test_inverse_many_batches_helper(3, 512)
        test_inverse_many_batches_helper(64, 64)

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

    def solve_test_helper(self, A_dims, b_dims, device, dtype):
        from torch.testing._internal.common_utils import random_fullrank_matrix_distinct_singular_value

        b = torch.randn(*b_dims, dtype=dtype, device=device)
        A = random_fullrank_matrix_distinct_singular_value(*A_dims, dtype=dtype).to(device)
        return b, A

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(torch.float32, torch.float64, torch.complex64, torch.complex128)
    def test_solve(self, device, dtype):
        for (k, n) in zip([2, 3, 5], [3, 5, 7]):
            b, A = self.solve_test_helper((n,), (n, k), device, dtype)
            x = torch.solve(b, A)[0]
            self.assertEqual(b, A.mm(x))

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(torch.float32, torch.float64, torch.complex64, torch.complex128)
    def test_solve_batched(self, device, dtype):
        def solve_batch_helper(A_dims, b_dims):
            b, A = self.solve_test_helper(A_dims, b_dims, device, dtype)
            x_exp_list = []
            for i in range(b_dims[0]):
                x_exp_list.append(torch.solve(b[i], A[i])[0])
            x_exp = torch.stack(x_exp_list)  # Stacked output
            x_act = torch.solve(b, A)[0]  # Actual output
            self.assertEqual(x_exp, x_act)  # Equality check
            Ax = torch.matmul(A, x_act)
            self.assertEqual(b, Ax)

        for batchsize in [1, 3, 4]:
            solve_batch_helper((5, batchsize), (batchsize, 5, 10))

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(torch.float32, torch.float64, torch.complex64, torch.complex128)
    def test_solve_batched_non_contiguous(self, device, dtype):
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
    def test_solve_batched_many_batches(self, device, dtype):
        for A_dims, b_dims in zip([(5, 256, 256), (3, )], [(5, 1), (512, 512, 3, 1)]):
            b, A = self.solve_test_helper(A_dims, b_dims, device, dtype)
            x, _ = torch.solve(b, A)
            Ax = torch.matmul(A, x)
            self.assertEqual(Ax, b.expand_as(x))

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(torch.float32, torch.float64, torch.complex64, torch.complex128)
    def test_solve_batched_broadcasting(self, device, dtype):
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
        for (k, n), (upper, unitriangular, transpose) in itertools.product(zip([2, 3, 5], [3, 5, 7]),
                                                                           itertools.product([True, False], repeat=3)):
            b, A = self.triangular_solve_test_helper((n, n), (n, k), upper,
                                                     unitriangular, device, dtype)
            x = torch.triangular_solve(b, A, upper=upper, unitriangular=unitriangular, transpose=transpose)[0]
            if transpose:
                self.assertEqual(b, A.t().mm(x))
            else:
                self.assertEqual(b, A.mm(x))

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

            Ax = torch.matmul(A, x_act)
            self.assertEqual(b, Ax)

        for (upper, unitriangular, transpose), batchsize in itertools.product(itertools.product(
                [True, False], repeat=3), [1, 3, 4]):
            triangular_solve_batch_helper((batchsize, 5, 5), (batchsize, 5, 10),
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

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(torch.float64, torch.complex128)
    def test_triangular_solve_autograd(self, device, dtype):
        def run_test(A_dims, B_dims):
            A = torch.rand(*A_dims, dtype=dtype).requires_grad_()
            b = torch.rand(*B_dims, dtype=dtype).requires_grad_()

            for upper, transpose, unitriangular in itertools.product((True, False), repeat=3):
                def func(A, b):
                    return torch.triangular_solve(b, A, upper, transpose, unitriangular)

                gradcheck(func, [A, b])
                gradgradcheck(func, [A, b])

        run_test((3, 3), (3, 4))
        run_test((3, 3), (3, 2))
        run_test((2, 3, 3), (2, 3, 4))
        run_test((2, 3, 3), (2, 3, 2))

instantiate_device_type_tests(TestLinalg, globals())

if __name__ == '__main__':
    run_tests()
