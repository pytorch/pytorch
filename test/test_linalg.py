# -*- coding: utf-8 -*-
# Owner(s): ["module: linear algebra"]

import torch
import numpy as np

import unittest
import itertools
import warnings
import math
from math import inf, nan, isnan
import random
from random import randrange
from itertools import product
from functools import reduce, partial

from torch.testing._internal.common_utils import \
    (TestCase, run_tests, TEST_SCIPY, IS_MACOS, IS_WINDOWS, slowTest,
     TEST_WITH_ASAN, TEST_WITH_ROCM, IS_FBCODE, IS_REMOTE_GPU, iter_indices,
     make_fullrank_matrices_with_distinct_singular_values,
     freeze_rng_state, IS_ARM64, IS_SANDCASTLE, TEST_OPT_EINSUM, parametrize, skipIfTorchDynamo,
     setLinalgBackendsToDefaultFinally)
from torch.testing._internal.common_device_type import \
    (instantiate_device_type_tests, dtypes, has_cusolver, has_hipsolver,
     onlyCPU, skipCUDAIf, skipCUDAIfNoMagma, skipCPUIfNoLapack, precisionOverride,
     skipCUDAIfNoMagmaAndNoCusolver, skipCUDAIfRocm, onlyNativeDeviceTypes, dtypesIfCUDA,
     onlyCUDA, skipCUDAVersionIn, skipMeta, skipCUDAIfNoCusolver, dtypesIfMPS, largeTensorTest)
from torch.testing import make_tensor
from torch.testing._internal.common_dtype import (
    all_types, all_types_and_complex_and, floating_and_complex_types, integral_types,
    floating_and_complex_types_and, floating_types_and, complex_types,
)
from torch.testing._internal.common_cuda import SM53OrLater, tf32_on_and_off, _get_magma_version, \
    _get_torch_cuda_version
from torch.distributions.binomial import Binomial
import torch.backends.opt_einsum as opt_einsum

# Protects against includes accidentally setting the default dtype
# NOTE: jit_metaprogramming_utils sets the default dtype to double!
torch.set_default_dtype(torch.float32)
assert torch.get_default_dtype() is torch.float32

if TEST_SCIPY:
    import scipy


@unittest.skipIf(IS_ARM64, "Issue with numpy version on arm")
class TestLinalg(TestCase):
    def setUp(self):
        super(self.__class__, self).setUp()
        torch.backends.cuda.matmul.allow_tf32 = False

    def tearDown(self):
        torch.backends.cuda.matmul.allow_tf32 = True
        super(self.__class__, self).tearDown()

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

        # Test error message
        with self.assertRaisesRegex(RuntimeError,
                                    r"inner\(\) the last dimension must match on both "
                                    r"input tensors but got shapes \[2, 3\] and \[2, 2\]"):
            torch.randn(2, 3, device=device, dtype=dtype).inner(torch.randn(2, 2, device=device, dtype=dtype))

    # Tests torch.outer, and its alias, torch.ger, vs. NumPy
    @precisionOverride({torch.bfloat16: 1e-1})
    @dtypes(*all_types_and_complex_and(torch.half, torch.bfloat16, torch.bool))
    def test_outer(self, device, dtype):
        def run_test_case(a, b):
            if dtype == torch.bfloat16:
                a_np = a.to(torch.double).cpu().numpy()
                b_np = b.to(torch.double).cpu().numpy()
                exact_dtype = False
            else:
                a_np = a.cpu().numpy()
                b_np = b.cpu().numpy()
                exact_dtype = True
            expected = np.outer(a_np, b_np)

            self.assertEqual(torch.outer(a, b), expected, exact_dtype=False)
            self.assertEqual(torch.Tensor.outer(a, b), expected, exact_dtype=False)

            self.assertEqual(torch.ger(a, b), expected, exact_dtype=False)
            self.assertEqual(torch.Tensor.ger(a, b), expected, exact_dtype=False)

            # test out variant
            out = torch.empty(a.size(0), b.size(0), device=device, dtype=dtype)
            torch.outer(a, b, out=out)
            self.assertEqual(out, expected, exact_dtype=False)

            out = torch.empty(a.size(0), b.size(0), device=device, dtype=dtype)
            torch.ger(a, b, out=out)
            self.assertEqual(out, expected, exact_dtype=False)

        a = torch.randn(50).to(device=device, dtype=dtype)
        b = torch.randn(50).to(device=device, dtype=dtype)
        run_test_case(a, b)

        # test 0 strided tensor
        zero_strided = torch.randn(1).to(device=device, dtype=dtype).expand(50)
        run_test_case(zero_strided, b)
        run_test_case(a, zero_strided)

    def test_matrix_rank_removed_error(self, device):
        a = make_tensor(5, 5, device=device, dtype=torch.float32)
        with self.assertRaisesRegex(RuntimeError, "This function was deprecated since version 1.9 and is now removed"):
            torch.matrix_rank(a)

    def test_solve_removed_error(self, device):
        a = make_tensor(5, 5, device=device, dtype=torch.float32)
        b = make_tensor(5, 1, device=device, dtype=torch.float32)
        with self.assertRaisesRegex(RuntimeError, "This function was deprecated since version 1.9 and is now removed"):
            torch.solve(b, a)
        with self.assertRaisesRegex(RuntimeError, "This function was deprecated since version 1.9 and is now removed"):
            b.solve(a)

    def test_eig_removed_error(self, device):
        a = make_tensor(5, 5, device=device, dtype=torch.float32)
        with self.assertRaisesRegex(RuntimeError, "This function was deprecated since version 1.9 and is now removed"):
            torch.eig(a)
        with self.assertRaisesRegex(RuntimeError, "This function was deprecated since version 1.9 and is now removed"):
            a.eig()

    def test_symeig_removed_error(self, device):
        a = make_tensor(5, 5, device=device, dtype=torch.float32)
        with self.assertRaisesRegex(RuntimeError, "This function was deprecated since version 1.9 and is now removed"):
            torch.symeig(a)
        with self.assertRaisesRegex(RuntimeError, "This function was deprecated since version 1.9 and is now removed"):
            a.symeig()

    def test_lstsq_removed_error(self, device):
        a = make_tensor(5, 5, device=device, dtype=torch.float32)
        with self.assertRaisesRegex(RuntimeError, "This function was deprecated since version 1.9 and is now removed"):
            torch.lstsq(a, a)
        with self.assertRaisesRegex(RuntimeError, "This function was deprecated since version 1.9 and is now removed"):
            a.lstsq(a)

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(torch.float, torch.double, torch.cfloat, torch.cdouble)
    def test_linalg_lstsq(self, device, dtype):
        from torch.testing._internal.common_utils import random_well_conditioned_matrix
        if self.device_type == 'cpu':
            drivers = ('gels', 'gelsy', 'gelsd', 'gelss', None)
        else:
            drivers = ('gels', None)

        def check_solution_correctness(a, b, sol):
            sol2 = a.pinverse() @ b
            self.assertEqual(sol, sol2, atol=1e-5, rtol=1e-5)

        def check_correctness_ref(a, b, res, ref, driver="default"):
            def apply_if_not_empty(t, f):
                if t.numel():
                    return f(t)
                else:
                    return t

            def select_if_not_empty(t, i):
                selected = apply_if_not_empty(t, lambda x: x.select(0, i))
                return selected

            m = a.size(-2)
            n = a.size(-1)
            nrhs = b.size(-1)
            batch_size = int(np.prod(a.shape[:-2]))
            if batch_size == 0:
                batch_size = 1
            a_3d = a.view(batch_size, m, n)
            b_3d = b.view(batch_size, m, nrhs)

            solution_3d = res.solution.view(batch_size, n, nrhs)
            residuals_2d = apply_if_not_empty(res.residuals, lambda t: t.view(-1, nrhs))
            rank_1d = apply_if_not_empty(res.rank, lambda t: t.view(-1))
            singular_values_2d = res.singular_values.view(batch_size, res.singular_values.shape[-1])

            if a.numel() > 0:
                for i in range(batch_size):
                    sol, residuals, rank, singular_values = ref(
                        a_3d.select(0, i).numpy(),
                        b_3d.select(0, i).numpy()
                    )
                    # Singular values are None when lapack_driver='gelsy' in SciPy
                    if singular_values is None:
                        singular_values = []
                    self.assertEqual(sol, solution_3d.select(0, i), atol=1e-5, rtol=1e-5)
                    self.assertEqual(rank, select_if_not_empty(rank_1d, i), atol=1e-5, rtol=1e-5)
                    self.assertEqual(singular_values, singular_values_2d.select(0, i), atol=1e-5, rtol=1e-5)

                    # SciPy and NumPy operate only on non-batched input and
                    # return an empty array with shape (0,) if rank(a) != n
                    # in PyTorch the batched inputs are supported and
                    # matrices in the batched input can have different ranks
                    # we compute residuals only if all matrices have rank == n
                    # see https://github.com/pytorch/pytorch/issues/56483
                    if m > n:
                        if torch.all(rank_1d == n):
                            self.assertEqual(
                                residuals, select_if_not_empty(residuals_2d, i), atol=1e-5, rtol=1e-5, exact_dtype=False
                            )
                        else:
                            self.assertTrue(residuals_2d.numel() == 0)

            else:
                self.assertEqual(res.solution.shape, (*a.shape[:-2], n, nrhs))
                self.assertEqual(res.rank.shape, a.shape[:-2])

                # residuals are not always computed (and have non-zero shape)
                if m > n and driver != "gelsy":
                    self.assertEqual(res.residuals.shape, (*a.shape[:-2], 0))
                else:
                    self.assertEqual(res.residuals.shape, (0, ))

                # singular_values are not always computed (and have non-zero shape)
                if driver == "default" or driver == "gelsd" or driver == "gelss":
                    self.assertEqual(res.singular_values.shape, (*a.shape[:-2], min(m, n)))
                else:
                    self.assertEqual(res.singular_values.shape, (0, ))

        def check_correctness_scipy(a, b, res, driver, cond):
            # SciPy provides 3 driver options: gelsd, gelss, gelsy
            if TEST_SCIPY and driver in ('gelsd', 'gelss', 'gelsy'):
                import scipy.linalg

                def scipy_ref(a, b):
                    return scipy.linalg.lstsq(a, b, lapack_driver=driver, cond=cond)
                check_correctness_ref(a, b, res, scipy_ref, driver=driver)

        def check_correctness_numpy(a, b, res, driver, rcond):
            # NumPy uses only gelsd routine
            if driver == 'gelsd':

                def numpy_ref(a, b):
                    return np.linalg.lstsq(a, b, rcond=rcond)
                check_correctness_ref(a, b, res, numpy_ref)

        ms = [2 ** i for i in range(5)]
        m_ge_n_sizes = [(m, m // 2) for m in ms] + [(m, m) for m in ms]
        # cases m < n are only supported on CPU and for cuSOLVER path on CUDA
        m_l_n_sizes = [(m // 2, m) for m in ms]
        include_m_l_n_case = (has_cusolver() or device == 'cpu')
        matrix_sizes = m_ge_n_sizes + (m_l_n_sizes if include_m_l_n_case else [])
        batches = [(), (2,), (2, 2), (2, 2, 2)]
        # we generate matrices with singular values sampled from a normal distribution,
        # that is why we use `cond=1.0`, the mean to cut roughly half of all
        # the singular values and compare whether torch.linalg.lstsq agrees with
        # SciPy and NumPy.
        # if rcond is True then set value for it based on the used algorithm
        # rcond == -1 or any other negative value forces LAPACK to use machine precision tolerance
        rconds = (None, True, -1)

        for batch, matrix_size, driver, rcond in itertools.product(batches, matrix_sizes, drivers, rconds):
            # keep the rcond value if it is None or -1, set the driver specific value if it is True
            if rcond and rcond != -1:
                if driver in ('gelss', 'gelsd'):
                    # SVD based algorithm; set to zero roughly half of all the singular values
                    rcond = 1.0
                else:
                    # driver == 'gelsy'
                    # QR based algorithm; setting the value too high might lead to non-unique solutions and flaky tests
                    # so we skip this case
                    continue

            # specifying rcond value has no effect for gels driver so no need to run the tests again
            if driver == 'gels' and rcond is not None:
                continue

            shape = batch + matrix_size
            a = random_well_conditioned_matrix(*shape, dtype=dtype, device=device)
            b = torch.rand(*shape, dtype=dtype, device=device)

            m = a.size(-2)
            n = a.size(-1)
            res = torch.linalg.lstsq(a, b, rcond=rcond, driver=driver)
            sol = res.solution

            # Only checks gelsd, gelss, gelsy drivers
            check_correctness_scipy(a, b, res, driver, rcond)

            # Only checks gelsd driver
            check_correctness_numpy(a, b, res, driver, rcond)

            # gels driver is not checked by comparing to NumPy or SciPy implementation
            # because NumPy and SciPy do not implement this driver
            if driver == 'gels' and rcond is None:
                check_solution_correctness(a, b, sol)

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(torch.float, torch.double, torch.cfloat, torch.cdouble)
    def test_linalg_lstsq_batch_broadcasting(self, device, dtype):
        from torch.testing._internal.common_utils import random_well_conditioned_matrix

        def check_correctness(a, b):
            sol = torch.linalg.lstsq(a, b).solution
            sol2 = a.pinverse() @ b
            self.assertEqual(sol, sol2, rtol=1e-5, atol=1e-5)

        ms = [2 ** i for i in range(5)]
        batches = [(), (0,), (2,), (2, 2), (2, 2, 2)]
        # the case when a single matrix is batch-broadcasted over the rhs
        for m, batch in itertools.product(ms, batches):
            a = random_well_conditioned_matrix(m, m, dtype=dtype, device=device).view(*([1] * len(batch)), m, m)
            b = torch.rand(*(batch + (m, m)), dtype=dtype, device=device)
            check_correctness(a, b)

        # cases with broadcastable shapes
        for m in ms:
            a = random_well_conditioned_matrix(1, 3, 1, 3, m, m, dtype=dtype, device=device)
            b = torch.rand(3, 1, 3, 1, m, m // 2, dtype=dtype, device=device)
            check_correctness(a, b)

            # rhs are vectors, not matrices in this test
            b = torch.rand(3, 1, 3, 1, m, dtype=dtype, device=device)
            # unsqueeze for b because `check_correctness` checks against
            # a.pinverse() @ b, which requires b to be a matrix
            check_correctness(a, b.unsqueeze(-1))

            a = random_well_conditioned_matrix(3, 1, 3, 1, m, m, dtype=dtype, device=device)
            b = torch.rand(1, 3, 1, 3, m, m // 2, dtype=dtype, device=device)
            check_correctness(a, b)

            # rhs are vectors, not matrices in this test
            b = torch.rand(1, 3, 1, 3, m, dtype=dtype, device=device)
            check_correctness(a, b.unsqueeze(-1))

    @skipCPUIfNoLapack
    @skipCUDAIfNoMagma
    @dtypes(torch.float, torch.double, torch.cfloat, torch.cdouble)
    def test_linalg_lstsq_input_checks(self, device, dtype):
        # check empty inputs
        # empty batches
        a = torch.rand(0, 0, 3, 3, dtype=dtype, device=device)
        b = torch.rand(0, 0, 3, 2, dtype=dtype, device=device)
        self.assertEqual(
            torch.linalg.lstsq(a, b)[0],
            torch.zeros(0, 0, 3, 2, dtype=dtype, device=device)
        )
        # empty a and b
        a = torch.rand(2, 2, 0, 0, dtype=dtype, device=device)
        b = torch.rand(2, 2, 0, 0, dtype=dtype, device=device)
        self.assertEqual(
            torch.linalg.lstsq(a, b)[0],
            torch.zeros(2, 2, 0, 0, dtype=dtype, device=device)
        )
        # empty a and b
        a = torch.rand(2, 2, 3, 0, dtype=dtype, device=device)
        b = torch.rand(2, 2, 3, 0, dtype=dtype, device=device)
        self.assertEqual(
            torch.linalg.lstsq(a, b)[0],
            torch.zeros(2, 2, 0, 0, dtype=dtype, device=device)
        )
        # empty a but not b
        a = torch.rand(2, 2, 3, 0, dtype=dtype, device=device)
        b = torch.rand(2, 2, 3, 2, dtype=dtype, device=device)
        self.assertEqual(
            torch.linalg.lstsq(a, b)[0],
            torch.zeros(2, 2, 0, 2, dtype=dtype, device=device)
        )

        # empty a and b
        if torch.device(device).type == 'cpu':
            # only CPU since CUDA does not support overdetermined systems
            a = torch.rand(2, 2, 0, 3, dtype=dtype, device=device)
            b = torch.rand(2, 2, 0, 3, dtype=dtype, device=device)
            self.assertEqual(
                torch.linalg.lstsq(a, b)[0],
                torch.zeros(2, 2, 3, 3, dtype=dtype, device=device)
            )

        a = torch.rand(2, 3, dtype=dtype, device=device)
        b = torch.rand(3, dtype=dtype, device=device)

        with self.assertRaisesRegex(RuntimeError, 'input must have at least 2 dimensions'):
            torch.linalg.lstsq(b, b)

        with self.assertRaisesRegex(RuntimeError, 'other must have at least 1 dimension'):
            torch.linalg.lstsq(a, torch.tensor(1, dtype=dtype, device=device))

        with self.assertRaisesRegex(RuntimeError, r'input.size\(-2\) should match other.size\(-1\)'):
            torch.linalg.lstsq(a, b)

        with self.assertRaisesRegex(RuntimeError, r'input.size\(-2\) should match other.size\(-2\)'):
            torch.linalg.lstsq(a, b.unsqueeze(-1))

        def complement_device(device):
            if device == 'cpu' and torch.cuda.is_available():
                return 'cuda'
            else:
                return 'cpu'

        a = torch.rand(2, 2, 2, 2, dtype=dtype, device=device)
        b = torch.rand(2, 2, 2, dtype=dtype, device=complement_device(device))
        if a.device != b.device:
            with self.assertRaisesRegex(RuntimeError, 'be on the same device'):
                torch.linalg.lstsq(a, b)

        b = (torch.rand(2, 2, 2, dtype=dtype, device=device) * 100).long()
        with self.assertRaisesRegex(RuntimeError, 'the same dtype'):
            torch.linalg.lstsq(a, b)

        a = torch.rand(2, 2, 2, 2, dtype=dtype, device=device)
        b = torch.rand(2, 2, 2, dtype=dtype, device=device)

        if device != 'cpu':
            with self.assertRaisesRegex(RuntimeError, '`driver` other than `gels` is not supported on CUDA'):
                torch.linalg.lstsq(a, b, driver='fictitious_driver')
        # if on cpu
        else:
            with self.assertRaisesRegex(RuntimeError, r'parameter `driver` should be one of \(gels, gelsy, gelsd, gelss\)'):
                torch.linalg.lstsq(a, b, driver='fictitious_driver')

        # cuSOLVER path supports underdetermined systems
        version = torch.testing._internal.common_cuda._get_torch_cuda_version()
        cusolver_not_available = (version < (10, 1))

        if device != 'cpu' and cusolver_not_available:
            a = torch.rand(2, 3, dtype=dtype, device=device)
            b = torch.rand(2, 1, dtype=dtype, device=device)
            with self.assertRaisesRegex(RuntimeError, r'only overdetermined systems'):
                torch.linalg.lstsq(a, b)

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(*floating_and_complex_types())
    def test_cholesky(self, device, dtype):
        from torch.testing._internal.common_utils import random_hermitian_pd_matrix

        def run_test(shape, batch, contiguous):
            A = random_hermitian_pd_matrix(shape, *batch, dtype=dtype, device=device)
            if A.numel() > 0 and not contiguous:
                A = A.mT
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

        # check the upper= variant
        expected = torch.linalg.cholesky(A).mH
        actual = torch.linalg.cholesky(A, upper=True)
        self.assertEqual(expected, actual)

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(*floating_and_complex_types())
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

        # if the input matrix is not positive definite, an error should be raised
        A = torch.eye(3, 3, dtype=dtype, device=device)
        A[-1, -1] = 0  # Now A is not positive definite
        with self.assertRaisesRegex(torch.linalg.LinAlgError, r'minor of order 3 is not positive-definite'):
            torch.linalg.cholesky(A)
        with self.assertRaisesRegex(np.linalg.LinAlgError, r'Matrix is not positive definite'):
            np.linalg.cholesky(A.cpu().numpy())

        # if at least one matrix in the batch is singular, an error should be raised
        A = torch.eye(3, 3, dtype=dtype, device=device)
        A = A.reshape((1, 3, 3))
        A = A.repeat(5, 1, 1)
        A[4, -1, -1] = 0  # Now A[4] is not positive definite
        with self.assertRaisesRegex(torch.linalg.LinAlgError, r'\(Batch element 4\): The factorization could not be completed'):
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

        # dtypes should be safely castable
        out = torch.empty(*A.shape, dtype=torch.int, device=device)
        with self.assertRaisesRegex(RuntimeError, "but got int instead"):
            torch.linalg.cholesky(A, out=out)

        # device should match
        if torch.cuda.is_available():
            wrong_device = 'cpu' if self.device_type != 'cpu' else 'cuda'
            out = torch.empty(0, device=wrong_device, dtype=dtype)
            with self.assertRaisesRegex(RuntimeError, "Expected all tensors to be on the same device"):
                torch.linalg.cholesky(A, out=out)

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
                self.assertEqual(A, chol_fact.mT.matmul(chol_fact))
                # Upper triangular check
                self.assertEqual(chol_fact, chol_fact.triu())
            else:
                # Correctness check
                self.assertEqual(A, chol_fact.matmul(chol_fact.mT))
                # Lower triangular check
                self.assertEqual(chol_fact, chol_fact.tril())

        for upper, batchsize in itertools.product([True, False], [262144, 524288]):
            cholesky_test_helper(2, batchsize, device, upper)

    @precisionOverride({torch.float32: 1e-4, torch.complex64: 1e-4})
    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(*floating_and_complex_types())
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
    @dtypes(*floating_and_complex_types())
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
    @dtypes(*floating_and_complex_types())
    def test_old_cholesky_empty(self, device, dtype):
        def run_test(upper):
            A = torch.empty(0, 0, dtype=dtype, device=device)
            chol = torch.cholesky(A, upper)
            chol_A = torch.matmul(chol, chol.t().conj())
            self.assertEqual(A, chol_A)
        for upper in [True, False]:
            run_test(upper)

    # Test for issue
    # https://github.com/pytorch/pytorch/issues/57032
    # torch.cholesky with upper=True for batched CUDA inputs was wrong
    # it was using the lower triangular part instead of the upper one
    @onlyCUDA
    @skipCUDAIfNoMagma
    @dtypes(*floating_and_complex_types())
    def test_old_cholesky_batched_upper(self, device, dtype):
        from torch.testing._internal.common_utils import random_hermitian_pd_matrix

        batchsize = 2
        A = random_hermitian_pd_matrix(3, batchsize, dtype=dtype, device=device)
        A_triu = A.triu()  # fill the lower triangular part with zero

        U = torch.cholesky(A_triu, upper=True)

        reconstruct_A = U.mH @ U
        self.assertEqual(A, reconstruct_A)

    @skipCUDAIfNoMagmaAndNoCusolver
    @skipCPUIfNoLapack
    @dtypes(*floating_and_complex_types())
    def test_cholesky_ex(self, device, dtype):
        from torch.testing._internal.common_utils import random_hermitian_pd_matrix

        def run_test(n, batch):
            A = random_hermitian_pd_matrix(n, *batch, dtype=dtype, device=device)
            expected_L = np.linalg.cholesky(A.cpu().numpy())
            expected_info = torch.zeros(A.shape[:-2], dtype=torch.int32, device=device)
            actual_L, actual_info = torch.linalg.cholesky_ex(A)

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
            self.assertEqual(actual_info, expected_info)

        ns = (0, 3, 5)
        batches = ((), (2, ), (2, 1))
        for n, batch in itertools.product(ns, batches):
            run_test(n, batch)

    @skipCUDAIfNoMagmaAndNoCusolver
    @skipCPUIfNoLapack
    @dtypes(*floating_and_complex_types())
    def test_cholesky_ex_non_pd(self, device, dtype):
        # if the input matrix is not positive definite, info with positive integer is returned
        A = torch.eye(3, 3, dtype=dtype, device=device)
        A[-1, -1] = 0  # Now A is singular
        _, info = torch.linalg.cholesky_ex(A)
        self.assertEqual(info, 3)
        with self.assertRaisesRegex(torch.linalg.LinAlgError, r'minor of order 3 is not positive-definite'):
            torch.linalg.cholesky_ex(A, check_errors=True)

        # if at least one matrix in the batch is not positive definite,
        # batched info with positive integer for the corresponding matrix is returned
        A = torch.eye(3, 3, dtype=dtype, device=device)
        A = A.reshape((1, 3, 3))
        A = A.repeat(5, 1, 1)
        A[3, -2, -2] = 0  # Now A[3] is singular
        _, info = torch.linalg.cholesky_ex(A)

        expected_info = torch.zeros(A.shape[:-2], dtype=torch.int32, device=device)
        expected_info[3] = 2
        self.assertEqual(info, expected_info)
        with self.assertRaisesRegex(torch.linalg.LinAlgError, r'\(Batch element 3\): The factorization could not be completed'):
            torch.linalg.cholesky_ex(A, check_errors=True)

    def _test_addr_vs_numpy(self, device, dtype, beta=1, alpha=1):
        def check(m, a, b, beta, alpha):
            if dtype == torch.bfloat16:
                a_np = a.to(torch.double).cpu().numpy()
                b_np = b.to(torch.double).cpu().numpy()
                m_np = m.to(torch.double).cpu().numpy()
                exact_dtype = False
            else:
                a_np = a.cpu().numpy()
                b_np = b.cpu().numpy()
                m_np = m.cpu().numpy()
                exact_dtype = True
            if beta == 0:
                expected = alpha * np.outer(a_np, b_np)
            else:
                expected = beta * m_np + alpha * np.outer(a_np, b_np)

            res = torch.addr(m, a, b, beta=beta, alpha=alpha)
            self.assertEqual(res, expected, exact_dtype=exact_dtype)

            # Test out variant
            out = torch.empty_like(res)
            torch.addr(m, a, b, beta=beta, alpha=alpha, out=out)
            self.assertEqual(out, expected, exact_dtype=exact_dtype)

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
        float_and_complex_dtypes = floating_and_complex_types_and(torch.half, torch.bfloat16)
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

    @dtypes(*integral_types())
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
    @dtypes(*floating_and_complex_types_and(torch.half, torch.bfloat16))
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
        if dtype in complex_types():
            self._test_addr_vs_numpy(device, dtype, beta=(0 + 0.1j), alpha=(0.2 - 0.2j))

    @dtypes(*itertools.product(all_types_and_complex_and(torch.half, torch.bfloat16, torch.bool),
                               all_types_and_complex_and(torch.half, torch.bfloat16, torch.bool)))
    def test_outer_type_promotion(self, device, dtypes):
        a = torch.randn(5).to(device=device, dtype=dtypes[0])
        b = torch.randn(5).to(device=device, dtype=dtypes[1])
        for op in (torch.outer, torch.Tensor.outer, torch.ger, torch.Tensor.ger):
            result = op(a, b)
            self.assertEqual(result.dtype, torch.result_type(a, b))

    # don't use @dtypes decorator to avoid generating ~1700 tests per device
    def test_addr_type_promotion(self, device):
        for dtypes0, dtypes1, dtypes2 in product(all_types_and_complex_and(torch.half, torch.bfloat16, torch.bool), repeat=3):
            a = make_tensor((5,), device=device, dtype=dtypes0, low=-2, high=2)
            b = make_tensor((5,), device=device, dtype=dtypes1, low=-2, high=2)
            m = make_tensor((5, 5), device=device, dtype=dtypes2, low=-2, high=2)

            desired_dtype = torch.promote_types(torch.promote_types(dtypes0, dtypes1),
                                                dtypes2)
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
    @dtypes(*floating_and_complex_types())
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
            # additionally we can multiply the eigenvector with a phase factor e^{i\phi} and then compare the values
            # let's choose the convention that the first element of the eigenvectors from torch and numpy be the same
            # for real inputs, this phase factor is plus or minus one
            if matrix.numel() > 0:
                phase = torch.from_numpy(expected_v[..., 0, :]).to(device=device).div(actual_v[..., 0, :])
                actual_v_rotated = actual_v * phase.unsqueeze(-2).expand_as(actual_v)
                self.assertEqual(actual_v_rotated, expected_v)

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
    @dtypes(*floating_and_complex_types())
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
    @dtypes(*floating_and_complex_types())
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

        # dtypes should be safely castable
        out_w = torch.empty(0, dtype=real_dtype, device=device)
        out_v = torch.empty(0, dtype=torch.int, device=device)
        with self.assertRaisesRegex(RuntimeError, "but got int instead"):
            torch.linalg.eigh(a, out=(out_w, out_v))

        out_w = torch.empty(0, dtype=torch.int, device=device)
        out_v = torch.empty(0, dtype=dtype, device=device)
        with self.assertRaisesRegex(RuntimeError, "but got int instead"):
            torch.linalg.eigh(a, out=(out_w, out_v))

        # device should match
        if torch.cuda.is_available():
            wrong_device = 'cpu' if self.device_type != 'cpu' else 'cuda'
            out_w = torch.empty(0, device=wrong_device, dtype=dtype)
            out_v = torch.empty(0, device=device, dtype=dtype)
            with self.assertRaisesRegex(RuntimeError, "tensors to be on the same device"):
                torch.linalg.eigh(a, out=(out_w, out_v))
            out_w = torch.empty(0, device=device, dtype=dtype)
            out_v = torch.empty(0, device=wrong_device, dtype=dtype)
            with self.assertRaisesRegex(RuntimeError, "tensors to be on the same device"):
                torch.linalg.eigh(a, out=(out_w, out_v))


    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(*floating_and_complex_types())
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
    @dtypes(*floating_and_complex_types())
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

        # dtypes should be safely castable
        out = torch.empty(0, dtype=torch.int, device=device)
        with self.assertRaisesRegex(RuntimeError, "but got int instead"):
            torch.linalg.eigvalsh(t, out=out)

        # device should match
        if torch.cuda.is_available():
            wrong_device = 'cpu' if self.device_type != 'cpu' else 'cuda'
            out = torch.empty(0, device=wrong_device, dtype=dtype)
            with self.assertRaisesRegex(RuntimeError, "tensors to be on the same device"):
                torch.linalg.eigvalsh(t, out=out)

    @dtypes(*floating_and_complex_types())
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

    @dtypes(*floating_and_complex_types())
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

    @dtypes(*floating_and_complex_types())
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
    @dtypes(torch.float, torch.double, torch.cfloat, torch.cdouble, torch.bfloat16, torch.float16)
    def test_norm_dtype(self, device, dtype):
        make_arg = partial(make_tensor, dtype=dtype, device=device)

        def run_test_case(input_size, ord, keepdim, to_dtype):
            msg = (
                f'input_size={input_size}, ord={ord}, keepdim={keepdim}, '
                f'dtype={dtype}, to_dtype={to_dtype}')
            input = make_arg(input_size)
            result = torch.linalg.norm(input, ord, keepdim=keepdim)
            self.assertEqual(result.dtype, input.real.dtype, msg=msg)

            result_out = torch.empty((0), dtype=result.dtype, device=device)
            torch.linalg.norm(input, ord, keepdim=keepdim, out=result_out)
            self.assertEqual(result, result_out, msg=msg)

            result = torch.linalg.norm(input.to(to_dtype), ord, keepdim=keepdim)
            result_with_dtype = torch.linalg.norm(input, ord, keepdim=keepdim, dtype=to_dtype)
            self.assertEqual(result, result_with_dtype, msg=msg)

            result_out_with_dtype = torch.empty_like(result_with_dtype)
            torch.linalg.norm(input, ord, keepdim=keepdim, dtype=to_dtype, out=result_out_with_dtype)
            self.assertEqual(result_with_dtype, result_out_with_dtype, msg=msg)

        ord_vector = [0, 1, -1, 2, -2, 3, -3, 4.5, -4.5, inf, -inf, None]

        # In these orders we are computing the 10-th power and 10-th root of numbers.
        # We avoid them for half-precision types as it makes the tests above too badly conditioned
        if dtype != torch.float16 and dtype != torch.bfloat16:
            ord_vector.extend([0.1, -0.1])
        ord_matrix = ['fro', 'nuc', 1, -1, 2, -2, inf, -inf, None]
        S = 10

        if dtype == torch.cfloat:
            norm_dtypes = (torch.cfloat, torch.cdouble)
        elif dtype == torch.cdouble:
            norm_dtypes = (torch.cdouble,)
        elif dtype in (torch.float16, torch.bfloat16, torch.float):
            norm_dtypes = (torch.float, torch.double)
        elif dtype == torch.double:
            norm_dtypes = (torch.double,)
        else:
            raise RuntimeError("Unsupported dtype")

        for ord, keepdim, norm_dtype in product(ord_vector, (True, False), norm_dtypes):
            run_test_case((S,) , ord, keepdim, norm_dtype)

        for ord, keepdim, norm_dtype in product(ord_matrix, (True, False), norm_dtypes):
            if ord in [2, -2, 'nuc']:
                # We need torch.svdvals
                if dtype == torch.float16 or dtype == torch.bfloat16:
                    continue

                # We need LAPACK or equivalent
                if ((torch.device(device).type == 'cuda' and not torch.cuda.has_magma and not has_cusolver()) or
                   (torch.device(device).type == 'cpu' and not torch._C.has_lapack)):
                    continue
            run_test_case((S, S) , ord, keepdim, norm_dtype)

    # This test confirms torch.linalg.norm bfloat16 and half get right result.
    @dtypes(torch.bfloat16, torch.float16)
    def test_norm_bfloat16_and_half(self, device, dtype):
        make_arg = partial(make_tensor, dtype=dtype, device=device)

        def run_test_case(input_size, ord, keepdim):
            msg = (
                f'input_size={input_size}, ord={ord}, keepdim={keepdim}, '
                f'dtype={dtype}')
            input = make_arg(input_size).fill_(1)
            result_ref = torch.linalg.norm(input.float(), ord, keepdim=keepdim).to(dtype=dtype)
            result = torch.linalg.norm(input, ord, keepdim=keepdim)
            self.assertEqual(result_ref, result, msg=msg)

        ord_vector = [0, 1, -1, 2, -2, 3, -3, 4.5, -4.5, inf, -inf, None]
        for S, ord, keepdim in product((10, 2049), ord_vector, (True, False)):
            run_test_case((S,) , ord, keepdim, )

    @dtypes(torch.float, torch.double, torch.cfloat, torch.cdouble, torch.bfloat16, torch.float16)
    def test_vector_norm(self, device, dtype):
        # This test compares torch.linalg.vector_norm's output with
        # torch.linalg.norm given a flattened tensor
        ord_vector = [0, 0.9, 1, 2, 3, inf, -0.5, -1, -2, -3, -inf]
        input_sizes = [
            (10, ),
            (4, 5),
            (3, 4, 5),
            (0, ),
            (0, 10),
            (0, 0),
            (10, 0, 10),
        ]

        def vector_norm_reference(input, ord, dim=None, keepdim=False, dtype=None):
            if dim is None:
                input_maybe_flat = input.flatten(0, -1)
            else:
                input_maybe_flat = input

            result = torch.linalg.norm(input_maybe_flat, ord, dim=dim, keepdim=keepdim, dtype=dtype)
            if keepdim and dim is None:
                result = result.reshape([1] * input.dim())
            return result

        def run_test_case(input, ord, dim, keepdim, norm_dtype):
            if (input.numel() == 0 and
                (ord < 0. or ord == inf) and
               (dim is None or input.shape[dim] == 0)):
                # The operation does not have an identity.
                error_msg = "linalg.vector_norm cannot compute"
                with self.assertRaisesRegex(RuntimeError, error_msg):
                    torch.linalg.vector_norm(input, ord, dim=dim, keepdim=keepdim)
            else:
                msg = (f'input.size()={input.size()}, ord={ord}, dim={dim}, '
                       f'keepdim={keepdim}, dtype={dtype}, norm_dtype={norm_dtype}')
                result_dtype_reference = vector_norm_reference(input, ord, dim=dim, keepdim=keepdim, dtype=norm_dtype)
                result_dtype = torch.linalg.vector_norm(input, ord, dim=dim, keepdim=keepdim, dtype=norm_dtype)
                if dtype.is_complex:
                    result_dtype_reference = result_dtype_reference.real
                self.assertEqual(result_dtype, result_dtype_reference, msg=msg)

                if norm_dtype is not None:
                    ref = torch.linalg.vector_norm(input.to(norm_dtype), ord, dim=dim, keepdim=keepdim)
                    actual = torch.linalg.vector_norm(input, ord, dim=dim, keepdim=keepdim, dtype=norm_dtype)
                    self.assertEqual(ref, actual, msg=msg)

        if dtype == torch.cfloat:
            norm_dtypes = (None, torch.cfloat, torch.cdouble)
        elif dtype == torch.cdouble:
            norm_dtypes = (None, torch.cdouble)
        elif dtype in (torch.float16, torch.bfloat16, torch.float):
            norm_dtypes = (None, torch.float, torch.double)
        elif dtype == torch.double:
            norm_dtypes = (None, torch.double)
        else:
            raise RuntimeError("Unsupported dtype")

        for input_size, ord, keepdim, norm_dtype in product(input_sizes, ord_vector, [True, False], norm_dtypes):
            input = make_tensor(input_size, dtype=dtype, device=device, low=-9, high=9)
            for dim in [None, random.randint(0, len(input_size) - 1)]:
                run_test_case(
                    input,
                    ord,
                    dim,
                    keepdim,
                    norm_dtype)

    def test_vector_norm_dim_tuple_arg(self, device):
        test_cases = [
            # input size, dim, error, error message
            ((4, ), (0, ), None, None),
            ((4, ), (1, ), IndexError, r'Dimension out of range'),
            ((4, ), (-2, ), IndexError, r'Dimension out of range'),
            ((4, 3), (0, -1), None, None),
            ((4, 3), (0, 0), RuntimeError, r'dim 0 appears multiple times in the list of dims'),
            ((4, 3), (0, -2), RuntimeError, r'dim 0 appears multiple times in the list of dims'),
            ((4, 3), (0, 1.0), TypeError, r"argument 'dim' must be tuple of ints"),
            ((4, 3), (None, ), TypeError, r"argument 'dim' must be tuple of ints"),
        ]
        for input_size, dim_tuple, error, error_msg in test_cases:
            input = torch.randn(input_size, device=device)
            # vector_norm should accept a tuple or a list for dim arg
            for dim in [dim_tuple, list(dim_tuple)]:
                if error is None:
                    torch.linalg.vector_norm(input, dim=dim)
                else:
                    with self.assertRaises(error):
                        torch.linalg.vector_norm(input, dim=dim)

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

        ord_vector = [0, 1, -1, 2, -2, 3, -3, 4.5, -4.5, inf, -inf]
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

    # This test compares torch.linalg.norm, torch.linalg.matrix_norm and numpy.linalg.norm to
    # ensure that their matrix norm results match.
    @skipMeta  # https://github.com/pytorch/pytorch/issues/54082
    @skipCUDAIfNoMagma
    @dtypes(torch.float, torch.double)
    @precisionOverride({torch.float32: 2e-4})
    def test_norm_matrix(self, device, dtype):
        make_arg = partial(make_tensor, dtype=dtype, device=device)

        def run_test_case(input, ord, dim, keepdim):
            msg = f'input.size()={input.size()}, ord={ord}, dim={dim}, keepdim={keepdim}, dtype={dtype}'
            result = torch.linalg.norm(input, ord, dim, keepdim)
            input_numpy = input.cpu().numpy()
            result_numpy = np.linalg.norm(input_numpy, ord, dim, keepdim)

            result = torch.linalg.norm(input, ord, dim, keepdim)
            self.assertEqual(result, result_numpy, msg=msg)
            if ord is not None and dim is not None:
                result = torch.linalg.matrix_norm(input, ord, dim, keepdim)
                self.assertEqual(result, result_numpy, msg=msg)

        ord_matrix = [1, -1, 2, -2, inf, -inf, 'nuc', 'fro']
        S = 10
        test_cases = [
            # input size, dim
            ((S, S), None),
            ((S, S), (0, 1)),
            ((S, S), (1, 0)),
            ((S, S, S, S), (2, 0)),
            ((S, S, S, S), (-1, -2)),
            ((S, S, S, S), (-1, -3)),
            ((S, S, S, S), (-3, 2)),
        ]

        for (shape, dim), keepdim, ord in product(test_cases, [True, False], ord_matrix):
            if ord in [2, -2, 'nuc']:
                # We need torch.svdvals
                if dtype == torch.float16 or dtype == torch.bfloat16:
                    continue
                # We need LAPACK or equivalent
                if ((torch.device(device).type == 'cuda' and not torch.cuda.has_magma and not has_cusolver()) or
                   (torch.device(device).type == 'cpu' and not torch._C.has_lapack)):
                    continue
            run_test_case(make_arg(shape), ord, dim, keepdim)


    @onlyCUDA
    @dtypes(torch.bfloat16, torch.float16)
    def test_norm_fused_type_promotion(self, device, dtype):
        x = torch.randn(10, device=device, dtype=dtype)

        def profile_and_check(fn, x, kwargs):
            with torch.profiler.profile(activities=(torch.profiler.ProfilerActivity.CPU,)) as p:
                fn(x, **kwargs, dtype=torch.float)
            # smoke check that profiler returned some events
            self.assertTrue("aten::linalg_vector_norm" in (e.name for e in p.events()))
            # test that there was no explicit copy
            self.assertFalse("aten::to" in (e.name for e in p.events()))

        for f, kwargs, in zip((torch.linalg.vector_norm, torch.norm), ({}, {"p" : 2})):
            profile_and_check(f, x, kwargs)

    @skipMeta  # https://github.com/pytorch/pytorch/issues/53739
    @skipCPUIfNoLapack
    @skipCUDAIfNoMagma
    @dtypes(*floating_and_complex_types())
    @precisionOverride({torch.float32: 1e-3})
    def test_cond(self, device, dtype):
        def run_test_case(input, p):
            result = torch.linalg.cond(input, p)
            result_numpy = np.linalg.cond(input.cpu().numpy(), p)
            self.assertEqual(result, result_numpy, rtol=1e-2, atol=self.precision, exact_dtype=False)
            self.assertEqual(result.shape, result_numpy.shape)

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
            try:
                run_test_case(a, p)
            except np.linalg.LinAlgError:
                # Numpy may fail to converge for some BLAS backends (although this is very rare)
                # See the discussion in https://github.com/pytorch/pytorch/issues/67675
                pass

        # test for 0x0 matrices. NumPy doesn't work for such input, we return 0
        input_sizes = [(0, 0), (2, 5, 0, 0)]
        for input_size in input_sizes:
            input = torch.randn(*input_size, dtype=dtype, device=device)
            for p in ['fro', 2]:
                expected_dtype = a.real.dtype if dtype.is_complex else dtype
                expected = torch.zeros(input_size[:-2], dtype=expected_dtype, device=device)
                actual = torch.linalg.cond(input, p)
                self.assertEqual(actual, expected)

    @skipMeta  # https://github.com/pytorch/pytorch/issues/53739
    @skipCPUIfNoLapack
    @skipCUDAIfNoMagma
    @dtypes(*floating_and_complex_types())
    @precisionOverride({torch.float32: 1e-3})
    def test_cond_errors_and_warnings(self, device, dtype):
        norm_types = [1, -1, 2, -2, inf, -inf, 'fro', 'nuc', None]

        # cond expects the input to be at least 2-dimensional
        a = torch.ones(3, dtype=dtype, device=device)
        for p in norm_types:
            with self.assertRaisesRegex(RuntimeError, r'at least 2 dimensions'):
                torch.linalg.cond(a, p)

        # for some norm types cond expects the input to be square
        a = torch.ones(3, 2, dtype=dtype, device=device)
        norm_types = [1, -1, inf, -inf, 'fro', 'nuc']
        for p in norm_types:
            with self.assertRaisesRegex(RuntimeError, r'must be batches of square matrices'):
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

        # dtypes should be safely castable
        out = torch.empty(0, dtype=torch.int, device=device)
        for p in ['fro', 2]:
            with self.assertRaisesRegex(RuntimeError, "but got result with dtype Int"):
                torch.linalg.cond(a, p, out=out)

        # device should match
        if torch.cuda.is_available():
            wrong_device = 'cpu' if self.device_type != 'cpu' else 'cuda'
            out = torch.empty(0, dtype=dtype, device=wrong_device)
            for p in ['fro', 2]:
                with self.assertRaisesRegex(RuntimeError, "tensors to be on the same device"):
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
        a[1, -1, -1] = 0  # now a[1] is singular
        for p in [1, -1, inf, -inf, 'fro', 'nuc']:
            result = torch.linalg.cond(a, p)
            self.assertEqual(result[1], float('inf'))

        # check invalid norm type
        a = torch.ones(3, 3, dtype=dtype, device=device)
        for p in ['wrong_norm', 5]:
            with self.assertRaisesRegex(RuntimeError, f"linalg.cond got an invalid norm type: {p}"):
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
            ((S, ), ['fro', 'nuc'], None, RuntimeError, r'A must have at least 2 dimensions'),
            ((S, S), [3.5], None, RuntimeError, r'matrix_norm: Order 3.5 not supported'),
            ((S, S), [0], None, RuntimeError, r'matrix_norm: Order 0 not supported'),
            ((S, S), ['fail'], None, RuntimeError, r'matrix_norm: Order fail not supported'),
            ((S, S), ['fro', 'nuc'], 0, RuntimeError, r'matrix_norm: dim must be a 2-tuple'),
            ((S, S), ['fro', 'nuc', 2], (0, 0), RuntimeError, r'dims must be different'),
            ((S, S), ['fro', 'nuc', 2], (-1, 1), RuntimeError, r'dims must be different'),
            ((S, S), ['fro', 'nuc', 2], (0, 4), IndexError, r'Dimension out of range'),
            ((S, ), [0], (4, ), IndexError, r'Dimension out of range'),
            ((S, ), [None], (0, 0), RuntimeError, r'dim 0 appears multiple times'),
            ((S, S, S), [1], (0, 1, 2), RuntimeError, r"If dim is specified, it must be of length 1 or 2."),
            ((S, S, S), [1], None, RuntimeError, r"If dim is not specified but ord is, the input must be 1D or 2D"),
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
    @precisionOverride({torch.cfloat: 5e-4})
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
                self.assertEqual(res, expected, msg=msg, exact_dtype=False)

                res_out = torch.tensor([], device=device, dtype=res.dtype)
                torch.linalg.norm(x, ord, keepdim=keepdim, out=res_out)
                self.assertEqual(res_out.shape, expected.shape, msg=msg)
                self.assertEqual(res_out, expected, msg=msg)

            # matrix norm
            x = torch.randn(25, 25, device=device, dtype=dtype)
            xn = x.cpu().numpy()
            for ord in matrix_ords:
                res = torch.linalg.norm(x, ord, keepdim=keepdim).cpu()
                expected = np.linalg.norm(xn, ord, keepdims=keepdim)
                msg = gen_error_message(x.size(), ord, keepdim)
                self.assertEqual(res.shape, expected.shape, msg=msg)
                self.assertEqual(res, expected, msg=msg, exact_dtype=False)

                res_out = torch.tensor([], device=device, dtype=res.dtype)
                torch.linalg.norm(x, ord, keepdim=keepdim, out=res_out)
                self.assertEqual(res_out.shape, expected.shape, msg=msg)
                self.assertEqual(res_out, expected, msg=msg)

    # Test that linal.vector_norm gives the same result as numpy when inputs
    # contain extreme values (inf, -inf, nan)
    def test_vector_norm_extreme_values(self, device):
        vector_ords = [0, 1, 2, 3, inf, -1, -2, -3, -inf]
        vectors = []
        for pair in itertools.product([inf, -inf, 0.0, nan, 1.0], repeat=2):
            vectors.append(list(pair))
        for vector in vectors:
            x = torch.tensor(vector, device=device)
            x_n = x.cpu().numpy()
            for ord in vector_ords:
                msg = f'ord={ord}, vector={vector}'
                result = torch.linalg.vector_norm(x, ord=ord)
                result_n = np.linalg.norm(x_n, ord=ord)
                self.assertEqual(result, result_n, msg=msg)

    @skipCUDAIfNoMagmaAndNoCusolver
    @skipCPUIfNoLapack
    @dtypes(torch.float, torch.double)
    @precisionOverride({torch.float32: 2e-5})
    def test_matrix_norm(self, device, dtype):
        # Test only inputs for which torch.linalg.matrix_norm diverges from torch.linalg.norm
        A = make_tensor((2, 2, 2), dtype=dtype, device=device)

        with self.assertRaisesRegex(RuntimeError, r'linalg.matrix_norm:.*must have at least 2 dimensions.*'):
            torch.linalg.matrix_norm(make_tensor((2,), dtype=dtype, device=device))
        with self.assertRaisesRegex(RuntimeError, r'linalg.matrix_norm:.*must be a 2-tuple.*'):
            torch.linalg.matrix_norm(A, dim=(0,))
        with self.assertRaisesRegex(RuntimeError, r'.*not supported.*'):
            torch.linalg.matrix_norm(A, ord=0)
        with self.assertRaisesRegex(RuntimeError, r'.*not supported.*'):
            torch.linalg.matrix_norm(A, ord=3.0)

        # Test dim=None behavior
        ref = torch.linalg.norm(A, dim=(-2, -1))
        res = torch.linalg.matrix_norm(A)
        self.assertEqual(ref, res)

    # Test that linal.norm gives the same result as numpy when inputs
    # contain extreme values (inf, -inf, nan)
    @unittest.skipIf(IS_WINDOWS, "Skipped on Windows!")
    @unittest.skipIf(IS_MACOS, "Skipped on MacOS!")
    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    def test_norm_extreme_values(self, device):
        vector_ords = [0, 1, 2, 3, inf, -1, -2, -3, -inf]
        # matrix_ords 'nuc', 2, -2 are skipped currently
        # See issue https://github.com/pytorch/pytorch/issues/71911
        matrix_ords = ['fro', 1, inf, -1, -inf]
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
                if ord in ['nuc', 2, -2]:
                    # These cases are broken because of another issue with svd
                    # https://github.com/pytorch/pytorch/issues/52633
                    return True
            return False

        for matrix in matrices:
            x = torch.tensor(matrix).to(device)
            x_n = x.cpu().numpy()
            for ord in matrix_ords:
                msg = f'ord={ord}, matrix={matrix}'
                if is_broken_matrix_norm_case(ord, x):
                    continue
                else:
                    result_n = np.linalg.norm(x_n, ord=ord)
                    result = torch.linalg.norm(x, ord=ord)
                    self.assertEqual(result, result_n, msg=msg)

    # Test degenerate shape results match numpy for linalg.norm vector norms
    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @unittest.skipIf(TEST_WITH_ASAN, "Skipped on ASAN since it checks for undefined behavior.")
    @dtypes(torch.float, torch.double, torch.cfloat, torch.cdouble)
    def test_norm_vector_degenerate_shapes(self, device, dtype):
        def run_test_case(input, ord, dim, keepdim):
            msg = f'input.size()={input.size()}, ord={ord}, dim={dim}, keepdim={keepdim}, dtype={dtype}'
            if (input.numel() == 0 and
                (ord < 0. or ord == inf) and
               (dim is None or input.shape[dim] == 0)):
                with self.assertRaises(RuntimeError):
                    torch.linalg.norm(input, ord, dim, keepdim)
            else:
                input_numpy = input.cpu().numpy()
                result_numpy = np.linalg.norm(input_numpy, ord, dim, keepdim)
                result = torch.linalg.norm(input, ord, dim, keepdim)
                self.assertEqual(result, result_numpy, msg=msg)

        ord_vector = [0, 0.5, 1, 2, 3, inf, -0.5, -1, -2, -3, -inf]
        S = 10
        test_cases = [
            # input size, dim
            ((0, ), None),
            ((0, S), 0),
            ((0, S), 1),
            ((S, 0), 0),
            ((S, 0), 1),
        ]
        for keepdim in [True, False]:
            for input_size, dim in test_cases:
                input = torch.randn(*input_size, dtype=dtype, device=device)
                for ord in ord_vector:
                    run_test_case(input, ord, dim, keepdim)

    # Test degenerate shape results match numpy for linalg.norm matrix norms
    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(torch.float, torch.double, torch.cfloat, torch.cdouble)
    def test_norm_matrix_degenerate_shapes(self, device, dtype):
        def run_test_case(input, ord, dim, keepdim, should_error):
            msg = f'input.size()={input.size()}, ord={ord}, dim={dim}, keepdim={keepdim}, dtype={dtype}'
            input_numpy = input.cpu().numpy()
            ops = [torch.linalg.norm]

            if ord is not None and dim is not None:
                ops.append(torch.linalg.matrix_norm)

            if should_error:
                with self.assertRaises(ValueError):
                    np.linalg.norm(input_numpy, ord, dim, keepdim)
                for op in ops:
                    with self.assertRaises(IndexError):
                        op(input, ord, dim, keepdim)
            else:
                result_numpy = np.linalg.norm(input_numpy, ord, dim, keepdim)
                for op in ops:
                    result = op(input, ord, dim, keepdim)
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
    # NumPy computes only in float64 and complex128 precisions
    # for float32 or complex64 results might be very different from float64 or complex128
    @dtypes(torch.float64, torch.complex128)
    def test_eig_numpy(self, device, dtype):
        def run_test(shape, *, symmetric=False):
            from torch.testing._internal.common_utils import random_symmetric_matrix

            if not dtype.is_complex and symmetric:
                # for symmetric real-valued inputs eigenvalues and eigenvectors have imaginary part equal to zero
                # unlike NumPy the result is not cast to float32 or float64 dtype in this case
                a = random_symmetric_matrix(shape[-1], *shape[:-2], dtype=dtype, device=device)
            else:
                a = make_tensor(shape, dtype=dtype, device=device)

            actual = torch.linalg.eig(a)

            # compare with NumPy
            # the eigenvalues are not necessarily ordered
            # so order of NumPy and PyTorch can be different
            expected = np.linalg.eig(a.cpu().numpy())

            # sort NumPy output
            ind = np.argsort(expected[0], axis=-1)[::-1]
            expected = (np.take_along_axis(expected[0], ind, axis=-1), np.take_along_axis(expected[1], ind[:, None], axis=-1))

            # sort PyTorch output
            # torch.argsort doesn't work with complex inputs, NumPy sorting on CPU is used instead
            # RuntimeError: _th_sort not supported on CUDAType for ComplexDouble
            # RuntimeError: "sorting_kernel_method_name" not implemented for 'ComplexDouble'
            ind = np.argsort(actual[0].cpu().numpy(), axis=-1)[::-1]
            actual_np = [x.cpu().numpy() for x in actual]
            sorted_actual = (
                np.take_along_axis(actual_np[0], ind, axis=-1),
                np.take_along_axis(actual_np[1], ind[:, None], axis=-1))

            self.assertEqual(expected[0], sorted_actual[0], exact_dtype=False)
            self.assertEqual(abs(expected[1]), abs(sorted_actual[1]), exact_dtype=False)

        shapes = [(0, 0),  # Empty matrix
                  (5, 5),  # Single matrix
                  (0, 0, 0), (0, 5, 5),  # Zero batch dimension tensors
                  (2, 5, 5),  # 3-dim tensors
                  (2, 1, 5, 5)]  # 4-dim tensors
        for shape in shapes:
            run_test(shape)
            run_test(shape, symmetric=True)

    @onlyCUDA
    @skipCUDAIfNoMagma
    @dtypes(*floating_and_complex_types())
    def test_eig_compare_backends(self, device, dtype):
        def run_test(shape, *, symmetric=False):
            from torch.testing._internal.common_utils import random_symmetric_matrix

            if not dtype.is_complex and symmetric:
                # for symmetric real-valued inputs eigenvalues and eigenvectors have imaginary part equal to zero
                a = random_symmetric_matrix(shape[-1], *shape[:-2], dtype=dtype, device=device)
            else:
                a = make_tensor(shape, dtype=dtype, device=device)

            actual = torch.linalg.eig(a)

            complementary_device = 'cpu'

            # compare with CPU
            expected = torch.linalg.eig(a.to(complementary_device))
            self.assertEqual(expected[0], actual[0])
            self.assertEqual(expected[1], actual[1])

        shapes = [(0, 0),  # Empty matrix
                  (5, 5),  # Single matrix
                  (0, 0, 0), (0, 5, 5),  # Zero batch dimension tensors
                  (2, 5, 5),  # 3-dim tensors
                  (2, 1, 5, 5)]  # 4-dim tensors
        for shape in shapes:
            run_test(shape)
            run_test(shape, symmetric=True)

    @slowTest
    @onlyCUDA
    @skipCUDAIfNoMagma
    @dtypes(torch.float32)
    def test_eig_check_magma(self, device, dtype):
        # For CUDA inputs only matrices of size larger than 2048x2048 actually call MAGMA library
        shape = (2049, 2049)
        a = make_tensor(shape, dtype=dtype, device=device)
        w, v = torch.linalg.eig(a)
        # check correctness using eigendecomposition identity
        self.assertEqual(a.to(v.dtype) @ v, w * v, atol=1e-3, rtol=1e-3)

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(*floating_and_complex_types())
    def test_eig_errors_and_warnings(self, device, dtype):
        # eig requires the input to be at least 2 dimensional tensor
        a = make_tensor(2, dtype=dtype, device=device)
        with self.assertRaisesRegex(RuntimeError, "must have at least 2 dimensions"):
            torch.linalg.eig(a)

        # eig requires a square matrix
        a = make_tensor((2, 3), dtype=dtype, device=device)
        with self.assertRaisesRegex(RuntimeError, "must be batches of square matrices"):
            torch.linalg.eig(a)

        # if out tensor with floating dtype is passed for complex output an error is thrown
        if not dtype.is_complex:
            # The characteristic equation is p(λ) = λ^2 − 2λ + 5 = 0, with roots λ = 1±2i
            a = torch.tensor([[3., -2.], [4., -1.]], dtype=dtype, device=device)
            out0 = torch.empty(0, device=device, dtype=dtype)
            out1 = torch.empty(0, device=device, dtype=dtype)
            with self.assertRaisesRegex(RuntimeError, "Expected eigenvalues to be safely castable"):
                torch.linalg.eig(a, out=(out0, out1))

            out0 = torch.empty(0, device=device, dtype=torch.complex128)
            with self.assertRaisesRegex(RuntimeError, "Expected eigenvectors to be safely castable"):
                torch.linalg.eig(a, out=(out0, out1))

        # dtypes should be safely castable
        a = make_tensor((3, 3), dtype=dtype, device=device)
        out0 = torch.empty(0, dtype=torch.int, device=device)
        out1 = torch.empty(0, dtype=torch.int, device=device)
        with self.assertRaisesRegex(RuntimeError, "but got eigenvalues with dtype Int"):
            torch.linalg.eig(a, out=(out0, out1))

        out0 = torch.empty(0, dtype=torch.complex128, device=device)
        with self.assertRaisesRegex(RuntimeError, "but got eigenvectors with dtype Int"):
            torch.linalg.eig(a, out=(out0, out1))

        # if non-empty out tensor with wrong shape is passed a warning is given
        a = make_tensor((3, 3), dtype=dtype, device=device)
        out0 = torch.empty(1, device=device, dtype=torch.complex128)
        out1 = torch.empty(1, device=device, dtype=torch.complex128)
        with warnings.catch_warnings(record=True) as w:
            # Trigger warning
            torch.linalg.eig(a, out=(out0, out1))
            # Check warning occurs
            self.assertEqual(len(w), 2)
            self.assertTrue("An output with one or more elements was resized" in str(w[-1].message))
            self.assertTrue("An output with one or more elements was resized" in str(w[-2].message))

        # device should match
        if torch.cuda.is_available():
            wrong_device = 'cpu' if self.device_type != 'cpu' else 'cuda'
            out_w = torch.empty(0, device=wrong_device, dtype=torch.complex128)
            out_v = torch.empty(0, device=device, dtype=torch.complex128)
            with self.assertRaisesRegex(RuntimeError, "tensors to be on the same device"):
                torch.linalg.eig(a, out=(out_w, out_v))
            out_w = torch.empty(0, device=device, dtype=torch.complex128)
            out_v = torch.empty(0, device=wrong_device, dtype=torch.complex128)
            with self.assertRaisesRegex(RuntimeError, "tensors to be on the same device"):
                torch.linalg.eig(a, out=(out_w, out_v))

    @skipCPUIfNoLapack
    @skipCUDAIfNoMagma
    @dtypes(*floating_and_complex_types())
    def test_eig_with_nan(self, device, dtype):
        for val in [np.inf, np.nan]:
            for batch_dim in [(), (10,)]:
                a = make_tensor((*batch_dim, 5, 5), device=device, dtype=dtype)
                a[..., -1, -1] = val

                with self.assertRaisesRegex(RuntimeError, "torch.linalg.eig: input tensor should not"):
                    torch.linalg.eig(a)

    @skipCPUIfNoLapack
    @skipCUDAIfNoMagma
    # NumPy computes only in float64 and complex128 precisions
    # for float32 or complex64 results might be very different from float64 or complex128
    @dtypes(torch.float64, torch.complex128)
    def test_eigvals_numpy(self, device, dtype):
        def run_test(shape, *, symmetric=False):
            from torch.testing._internal.common_utils import random_symmetric_matrix

            if not dtype.is_complex and symmetric:
                # for symmetric real-valued inputs eigenvalues and eigenvectors have imaginary part equal to zero
                # unlike NumPy the result is not cast to float32 or float64 dtype in this case
                a = random_symmetric_matrix(shape[-1], *shape[:-2], dtype=dtype, device=device)
            else:
                a = make_tensor(shape, dtype=dtype, device=device)

            actual = torch.linalg.eigvals(a)

            # compare with NumPy
            # the eigenvalues are not necessarily ordered
            # so order of NumPy and PyTorch can be different
            expected = np.linalg.eigvals(a.cpu().numpy())

            # sort NumPy output
            ind = np.argsort(expected, axis=-1)[::-1]
            expected = np.take_along_axis(expected, ind, axis=-1)

            # sort PyTorch output
            # torch.argsort doesn't work with complex inputs, NumPy sorting on CPU is used instead
            # RuntimeError: _th_sort not supported on CUDAType for ComplexDouble
            # RuntimeError: "sorting_kernel_method_name" not implemented for 'ComplexDouble'
            ind = np.argsort(actual.cpu().numpy(), axis=-1)[::-1]
            actual_np = actual.cpu().numpy()
            sorted_actual = np.take_along_axis(actual_np, ind, axis=-1)

            self.assertEqual(expected, sorted_actual, exact_dtype=False)

        shapes = [(0, 0),  # Empty matrix
                  (5, 5),  # Single matrix
                  (0, 0, 0), (0, 5, 5),  # Zero batch dimension tensors
                  (2, 5, 5),  # 3-dim tensors
                  (2, 1, 5, 5)]  # 4-dim tensors
        for shape in shapes:
            run_test(shape)
            run_test(shape, symmetric=True)

    @onlyCUDA
    @skipCUDAIfNoMagma
    @dtypes(*floating_and_complex_types())
    def test_eigvals_compare_backends(self, device, dtype):
        def run_test(shape, *, symmetric=False):
            from torch.testing._internal.common_utils import random_symmetric_matrix

            if not dtype.is_complex and symmetric:
                # for symmetric real-valued inputs eigenvalues and eigenvectors have imaginary part equal to zero
                a = random_symmetric_matrix(shape[-1], *shape[:-2], dtype=dtype, device=device)
            else:
                a = make_tensor(shape, dtype=dtype, device=device)

            actual = torch.linalg.eigvals(a)

            complementary_device = 'cpu'

            # compare with CPU
            expected = torch.linalg.eigvals(a.to(complementary_device))
            self.assertEqual(expected, actual)

            # check out= variant
            complex_dtype = dtype
            if not dtype.is_complex:
                complex_dtype = torch.complex128 if dtype == torch.float64 else torch.complex64
            out = torch.empty(0, dtype=complex_dtype, device=device)
            ans = torch.linalg.eigvals(a, out=out)
            self.assertEqual(ans, out)
            self.assertEqual(expected.to(complex_dtype), out)

            # check non-contiguous out
            if a.numel() > 0:
                out = torch.empty(2 * shape[0], *shape[1:-1], dtype=complex_dtype, device=device)[::2]
                self.assertFalse(out.is_contiguous())
                ans = torch.linalg.eigvals(a, out=out)
                self.assertEqual(ans, out)
                self.assertEqual(expected.to(complex_dtype), out)

        shapes = [(0, 0),  # Empty matrix
                  (5, 5),  # Single matrix
                  (0, 0, 0), (0, 5, 5),  # Zero batch dimension tensors
                  (2, 5, 5),  # 3-dim tensors
                  (2, 1, 5, 5)]  # 4-dim tensors
        for shape in shapes:
            run_test(shape)
            run_test(shape, symmetric=True)

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(*floating_and_complex_types())
    def test_eigvals_errors_and_warnings(self, device, dtype):
        # eig requires the input to be at least 2 dimensional tensor
        a = make_tensor(2, dtype=dtype, device=device)
        with self.assertRaisesRegex(RuntimeError, "must have at least 2 dimensions"):
            torch.linalg.eigvals(a)

        # eig requires a square matrix
        a = make_tensor((2, 3), dtype=dtype, device=device)
        with self.assertRaisesRegex(RuntimeError, "must be batches of square matrices"):
            torch.linalg.eigvals(a)

        # if out tensor with floating dtype is passed for complex output an error is thrown
        if not dtype.is_complex:
            # The characteristic equation is p(λ) = λ^2 − 2λ + 5 = 0, with roots λ = 1±2i
            a = torch.tensor([[3., -2.], [4., -1.]], dtype=dtype, device=device)
            out = torch.empty(0, device=device, dtype=dtype)
            with self.assertRaisesRegex(RuntimeError, "Expected eigenvalues to be safely castable"):
                torch.linalg.eigvals(a, out=out)

        # dtypes should be safely castable
        a = make_tensor((3, 3), dtype=dtype, device=device)
        out = torch.empty(0, dtype=torch.int, device=device)
        with self.assertRaisesRegex(RuntimeError, "but got eigenvalues with dtype Int"):
            torch.linalg.eigvals(a, out=out)

        # if non-empty out tensor with wrong shape is passed a warning is given
        out = torch.empty(1, device=device, dtype=torch.complex128)
        with warnings.catch_warnings(record=True) as w:
            # Trigger warning
            torch.linalg.eigvals(a, out=out)
            # Check warning occurs
            self.assertEqual(len(w), 1)
            self.assertTrue("An output with one or more elements was resized" in str(w[-1].message))

        # device should match
        if torch.cuda.is_available():
            wrong_device = 'cpu' if self.device_type != 'cpu' else 'cuda'
            out_w = torch.empty(0, device=wrong_device, dtype=torch.complex128)
            with self.assertRaisesRegex(RuntimeError, "tensors to be on the same device"):
                torch.linalg.eigvals(a, out=out_w)

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    def test_norm_old(self, device):
        def gen_error_message(input_size, p, keepdim, dim=None):
            return "norm failed for input size %s, p=%s, keepdim=%s, dim=%s" % (
                input_size, p, keepdim, dim)

        # 'nuc' norm uses SVD, and thus its precsion is much lower than other norms.
        # test_svd takes @precisionOverride({torch.float: 1e-4, torch.cfloat: 2e-4}),
        # and here we are doing the same thing for nuc norm.
        class PrecisionContext(object):
            def __init__(self, test, norm):
                self.norm = norm
                self.saved_overrides = getattr(test, 'precision_overrides', None)
                self.target_test = test

            def __enter__(self):
                if 'nuc' != self.norm:
                    return None
                self.target_test.precision_overrides = {torch.float: 1e-4, torch.cfloat: 2e-4}
                return self.target_test.precision_overrides

            def __exit__(self, type, value, tb) -> bool:
                if 'nuc' != self.norm:
                    return True
                if self.saved_overrides is None:
                    delattr(self.target_test, 'precision_overrides')
                else:
                    self.target_test.precision_overrides = self.saved_overrides
                return True

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
                with PrecisionContext(self, p):
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
                    with PrecisionContext(self, p):
                        self.assertEqual(res.shape, expected.shape, msg=msg)
                        self.assertEqual(res, expected, msg=msg)

    # Test that torch.norm with p=+/-inf propagates NaN
    def test_norm_old_nan_propagation(self, device):
        ords = [inf, -inf]
        for pair in itertools.product([0.0, nan, 1.0], repeat=2):
            x = torch.tensor(list(pair), device=device)
            for ord in ords:
                result = torch.norm(x, p=ord)
                result_check = torch.linalg.norm(x, ord=ord)
                self.assertEqual(result, result_check)

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
                self.assertEqual(res, expected, msg=msg, rtol=4e-6, atol=6e-4)

    # Ensure torch.norm with p='fro' and p=2 give the same results for mutually supported input combinations
    @dtypes(torch.float)
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
            a = make_tensor(input_size, dtype=dtype, device=device, low=-9, high=9)

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

    @skipIfTorchDynamo("Not a TorchDynamo suitable test")
    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
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
                    x = torch.randn(m, n, device=device).mT
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
                        x = torch.randn(o, m, n, device=device).mT
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
                            x = torch.randn(r, o, n, m, device=device).mT
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
            self.assertRaises(RuntimeError, torch.norm, x, "nuc", (0, 1))

        x = torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.double, device=device)
        self.assertRaisesRegex(RuntimeError, "must be different", torch.norm, x, "nuc", (0, 0))
        self.assertRaisesRegex(IndexError, "Dimension out of range", torch.norm, x, "nuc", (0, 2))

    @skipCUDAIfNoCusolver
    @skipCPUIfNoLapack
    @dtypes(torch.double)
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
            A = u.matmul(s.diag_embed()).matmul(v.mT)
            self.assertEqual(A, a, rtol=1e-7, atol=2e-7)

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
                self.assertEqual(u.mT.matmul(U).det().abs(), torch.ones(batches, device=device, dtype=dtype))
                self.assertEqual(v.mT.matmul(V).det().abs(), torch.ones(batches, device=device, dtype=dtype))

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
    @precisionOverride({torch.float: 1e-4, torch.cfloat: 2e-4})
    @setLinalgBackendsToDefaultFinally
    @dtypes(*floating_and_complex_types())
    def test_svd(self, device, dtype):
        # tests linalg.svd, svd, linalg.svdvals
        make_arg = partial(make_tensor, dtype=dtype, device=device)

        backends = ["default"]

        if torch.device(device).type == 'cuda':
            if torch.cuda.has_magma:
                backends.append("magma")
            if has_cusolver() or has_hipsolver():
                backends.append("cusolver")

        ns = (12, 4, 2, 0)
        batches = ((), (0,), (1,), (2,), (2, 1), (0, 2))
        drivers = (None, 'gesvd', 'gesvdj', 'gesvda')

        for backend in backends:
            torch.backends.cuda.preferred_linalg_library(backend)

            for batch, m, n, driver in product(batches, ns, ns, drivers):
                if not (backend == 'cusolver' or driver is None):
                    # only test cases below and skip otherwise:
                    # - backend == 'cusolver' (driver can be anything)
                    # - backend != 'cusolver' (driver should only be None)
                    continue

                shape = batch + (m, n)
                k = min(m, n)
                A = make_arg(shape)
                U, S, Vh = torch.linalg.svd(A, full_matrices=False, driver=driver)
                self.assertEqual((U @ S.to(A.dtype).diag_embed()) @ Vh, A)

                U_f, S_f, Vh_f = torch.linalg.svd(A, full_matrices=True, driver=driver)
                self.assertEqual(S_f, S)
                self.assertEqual((U_f[..., :k] @ S_f.to(A.dtype).diag_embed()) @ Vh_f[..., :k, :], A)

                S_s = torch.linalg.svdvals(A, driver=driver)
                self.assertEqual(S_s, S)

                U, S, V = torch.svd(A, some=True)
                self.assertEqual((U @ S.to(A.dtype).diag_embed()) @ V.mH, A)

                U_f, S_f, V_f = torch.svd(A, some=False)
                self.assertEqual(S_f, S)
                self.assertEqual((U_f[..., :k] @ S_f.to(A.dtype).diag_embed()) @ V_f[..., :k].mH, A)

                S_s = torch.svd(A, compute_uv=False).S
                self.assertEqual(S_s, S)

    @skipCUDAIfNoMagmaAndNoCusolver
    @skipCPUIfNoLapack
    @dtypes(torch.complex128)
    def test_invariance_error_spectral_decompositions(self, device, dtype):
        make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=True)
        A = make_arg((3, 3))
        with self.assertRaisesRegex(RuntimeError, "ill-defined"):
            U, _, Vh = torch.linalg.svd(A, full_matrices=False)
            (U + Vh).sum().abs().backward()

        A = make_arg((3, 3))
        with self.assertRaisesRegex(RuntimeError, "ill-defined"):
            V = torch.linalg.eig(A).eigenvectors
            V.sum().abs().backward()

        A = make_arg((3, 3))
        A = A + A.mH
        with self.assertRaisesRegex(RuntimeError, "ill-defined"):
            Q = torch.linalg.eigh(A).eigenvectors
            Q.sum().abs().backward()

    @skipCUDAIfNoCusolver  # MAGMA backend doesn't work in this case
    @precisionOverride({torch.float: 1e-4, torch.cfloat: 1e-4})
    @skipCPUIfNoLapack
    @dtypes(*floating_and_complex_types())
    def test_svd_memory_allocation(self, device, dtype):
        # test for https://github.com/pytorch/pytorch/issues/61949
        # the problem was that tensors of incorrect size were allocated and then narrowed
        m = 3
        n = 2**20
        a = make_tensor((m, n), dtype=dtype, device=device)
        # the following should run without errors
        S = torch.linalg.svdvals(a)
        result = torch.linalg.svd(a, full_matrices=False)
        self.assertEqual(result.S, S)

    def cholesky_solve_test_helper(self, A_dims, b_dims, upper, device, dtype):
        from torch.testing._internal.common_utils import random_hermitian_pd_matrix

        b = torch.randn(*b_dims, dtype=dtype, device=device)
        A = random_hermitian_pd_matrix(*A_dims, dtype=dtype, device=device)
        L = torch.cholesky(A, upper=upper)
        return b, A, L

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(*floating_and_complex_types())
    @precisionOverride({torch.float32: 1e-3, torch.complex64: 1e-3,
                        torch.float64: 1e-8, torch.complex128: 1e-8})
    def test_cholesky_solve(self, device, dtype):
        for (k, n), upper in itertools.product(zip([2, 3, 5], [3, 5, 7]), [True, False]):
            b, A, L = self.cholesky_solve_test_helper((n,), (n, k), upper, device, dtype)
            x = torch.cholesky_solve(b, L, upper=upper)
            self.assertEqual(b, np.matmul(A.cpu(), x.cpu()))

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(*floating_and_complex_types())
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

    @slowTest
    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(*floating_and_complex_types())
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
    @dtypes(*floating_and_complex_types())
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
            L = torch.linalg.cholesky(A, upper=upper)
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
    @dtypes(*floating_and_complex_types())
    def test_cholesky_solve_out_errors_and_warnings(self, device, dtype):
        # dtypes should be safely castable
        a = torch.eye(2, dtype=dtype, device=device)
        b = torch.randn(2, 1, dtype=dtype, device=device)
        out = torch.empty(0, dtype=torch.int, device=device)
        with self.assertRaisesRegex(RuntimeError, "but got result with dtype Int"):
            torch.cholesky_solve(b, a, out=out)

        # device should match
        if torch.cuda.is_available():
            wrong_device = 'cpu' if self.device_type != 'cpu' else 'cuda'
            out = torch.empty(0, dtype=dtype, device=wrong_device)
            with self.assertRaisesRegex(RuntimeError, "tensors to be on the same device"):
                torch.cholesky_solve(b, a, out=out)

        # if out tensor with wrong shape is passed a warning is given
        with warnings.catch_warnings(record=True) as w:
            out = torch.empty(1, dtype=dtype, device=device)
            # Trigger warning
            torch.cholesky_solve(b, a, out=out)
            # Check warning occurs
            self.assertEqual(len(w), 1)
            self.assertTrue("An output with one or more elements was resized" in str(w[-1].message))

    @skipCUDAIfNoMagmaAndNoCusolver
    @skipCPUIfNoLapack
    @dtypes(*floating_and_complex_types())
    @precisionOverride({torch.float32: 2e-3, torch.complex64: 2e-3,
                        torch.float64: 1e-8, torch.complex128: 1e-8})
    def test_inverse(self, device, dtype):
        make_fullrank = make_fullrank_matrices_with_distinct_singular_values
        make_arg = partial(make_fullrank, device=device, dtype=dtype)

        def run_test(torch_inverse, matrix, batches, n):
            matrix_inverse = torch_inverse(matrix)

            # Compare against NumPy output
            # NumPy uses 'gesv' LAPACK routine solving the equation A A_inv = I
            # But in PyTorch 'gertf' + 'getrs' is used. As such, there may be some element-wise differences
            expected = np.linalg.inv(matrix.cpu().numpy())
            self.assertEqual(matrix_inverse, expected, atol=self.precision, rtol=self.precision)

            # Additional correctness tests, check matrix*matrix_inverse == identity
            identity = torch.eye(n, dtype=dtype, device=device)
            self.assertEqual(identity.expand_as(matrix), np.matmul(matrix.cpu(), matrix_inverse.cpu()))
            self.assertEqual(identity.expand_as(matrix), np.matmul(matrix_inverse.cpu(), matrix.cpu()))

            # check the out= variant
            # prepare the expected out tensor
            matrix_inverse_out = torch.empty(*batches, n, n, dtype=dtype, device=device)
            matrix_inverse_out_t = matrix_inverse_out.mT.clone(memory_format=torch.contiguous_format)
            matrix_inverse_out = matrix_inverse_out_t.mT
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

        # helper function for testing torch.linalg.inv_ex
        def test_inv_ex(input, out=None):
            if out is not None:
                info = torch.empty(0, dtype=torch.int32, device=device)
                return torch.linalg.inv_ex(input, out=(out, info)).inverse
            return torch.linalg.inv_ex(input).inverse

        for torch_inverse in [torch.inverse, torch.linalg.inv, test_inv_ex]:
            for batches, n in itertools.product(
                [[], [0], [2], [2, 1]],
                [0, 5]
            ):
                matrices = make_arg(*batches, n, n)
                run_test(torch_inverse, matrices, batches, n)

                # test non-contiguous input
                run_test(torch_inverse, matrices.mT, batches, n)
                if n > 0:
                    run_test(
                        torch_inverse,
                        make_arg(*batches, 2 * n, 2 * n)
                        .view(-1, n * 2, n * 2)[:, ::2, ::2].view(*batches, n, n),
                        batches, n
                    )

    @skipCUDAIfNoMagmaAndNoCusolver
    @skipCPUIfNoLapack
    @dtypes(*floating_and_complex_types())
    def test_inv_ex_info_device(self, device, dtype):
        A = torch.eye(3, 3, dtype=dtype, device=device)
        info = torch.linalg.inv_ex(A).info
        self.assertTrue(info.device == A.device)

    @skipCUDAIfNoMagmaAndNoCusolver
    @skipCPUIfNoLapack
    @dtypes(*floating_and_complex_types())
    def test_inv_ex_singular(self, device, dtype):
        # if the input matrix is not invertible, info with positive integer is returned
        A = torch.eye(3, 3, dtype=dtype, device=device)
        A[-1, -1] = 0  # Now A is singular
        info = torch.linalg.inv_ex(A).info
        self.assertEqual(info, 3)
        with self.assertRaisesRegex(torch.linalg.LinAlgError,
                                    r'diagonal element 3 is zero, the inversion could not be completed'):
            torch.linalg.inv_ex(A, check_errors=True)

        # if at least one matrix in the batch is not positive definite,
        # batched info with positive integer for the corresponding matrix is returned
        A = torch.eye(3, 3, dtype=dtype, device=device)
        A = A.reshape((1, 3, 3))
        A = A.repeat(5, 1, 1)
        A[3, -2, -2] = 0  # Now A[3] is singular
        info = torch.linalg.inv_ex(A).info

        expected_info = torch.zeros(A.shape[:-2], dtype=torch.int32, device=device)
        expected_info[3] = 2
        self.assertEqual(info, expected_info)
        with self.assertRaisesRegex(torch.linalg.LinAlgError, r'\(Batch element 3\): The diagonal element 2 is zero'):
            torch.linalg.inv_ex(A, check_errors=True)

    @slowTest
    @skipCUDAIfNoMagmaAndNoCusolver
    @skipCPUIfNoLapack
    @skipCUDAIfRocm
    @dtypes(*floating_and_complex_types())
    @precisionOverride({torch.float32: 2e-3, torch.complex64: 2e-3,
                        torch.float64: 1e-5, torch.complex128: 1e-5})
    def test_inverse_many_batches(self, device, dtype):
        make_fullrank = make_fullrank_matrices_with_distinct_singular_values
        make_arg = partial(make_fullrank, device=device, dtype=dtype)

        def test_inverse_many_batches_helper(torch_inverse, b, n):
            matrices = make_arg(b, n, n)
            matrices_inverse = torch_inverse(matrices)

            # Compare against NumPy output
            expected = np.linalg.inv(matrices.cpu().numpy())
            self.assertEqual(matrices_inverse, expected, atol=self.precision, rtol=1e-3)

        for torch_inverse in [torch.inverse, torch.linalg.inv]:
            test_inverse_many_batches_helper(torch_inverse, 5, 256)
            test_inverse_many_batches_helper(torch_inverse, 3, 512)

    @skipCUDAIfNoMagmaAndNoCusolver
    @skipCPUIfNoLapack
    @onlyNativeDeviceTypes   # TODO: XLA doesn't raise exception
    @dtypes(*floating_and_complex_types())
    def test_inverse_errors(self, device, dtype):
        # inverse expects batches of square matrices as input
        with self.assertRaisesRegex(RuntimeError, "must be batches of square matrices"):
            torch.inverse(torch.randn(2, 3, 4, 3))

        # if input is not invertible, RuntimeError is raised mentioning the first non-invertible batch
        def run_test_singular_input(batch_dim, n):
            x = torch.eye(3, 3, dtype=dtype, device=device).reshape((1, 3, 3)).repeat(batch_dim, 1, 1)
            x[n, -1, -1] = 0
            with self.assertRaisesRegex(torch.linalg.LinAlgError, rf'\(Batch element {n}\): The diagonal element 3 is zero'):
                torch.inverse(x)

        for params in [(1, 0), (2, 0), (2, 1), (4, 0), (4, 2), (10, 2)]:
            run_test_singular_input(*params)

    @unittest.skipIf(IS_FBCODE or IS_SANDCASTLE, "Test fails for float64 on GPU (P100, V100) on Meta infra")
    @skipCUDAIfNoMagmaAndNoCusolver
    @skipCPUIfNoLapack
    @onlyNativeDeviceTypes   # TODO: XLA doesn't raise exception
    @skipCUDAIfRocm
    @dtypes(*floating_and_complex_types())
    def test_inverse_errors_large(self, device, dtype):
        # Test batched inverse of singular matrices reports errors without crashing (gh-51930)
        x = torch.empty((8, 10, 616, 616), dtype=dtype, device=device)
        x[:] = torch.eye(616, dtype=dtype, device=device)
        x[..., 10, 10] = 0
        with self.assertRaisesRegex(torch.linalg.LinAlgError, r'\(Batch element 0\): The diagonal element 11 is zero'):
            torch.inverse(x)

    @precisionOverride({torch.float32: 1e-3, torch.complex64: 1e-3, torch.float64: 1e-7, torch.complex128: 1e-7})
    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(*floating_and_complex_types())
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
                torch_rtol = torch.linalg.pinv(A, rtol=rcond, hermitian=hermitian)
                self.assertEqual(actual, torch_rtol)
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
    @dtypes(*floating_and_complex_types())
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

        # dtypes of out and input should be safely castable
        out = torch.empty_like(a).to(torch.int)
        with self.assertRaisesRegex(RuntimeError, "but got result with dtype Int"):
            torch.linalg.pinv(a, out=out)

        if torch.cuda.is_available():
            # device of out and input should match
            wrong_device = 'cpu' if self.device_type != 'cpu' else 'cuda'
            out = torch.empty_like(a).to(wrong_device)
            with self.assertRaisesRegex(RuntimeError, "Expected result and input tensors to be on the same device"):
                torch.linalg.pinv(a, out=out)

            # device of rcond and input should match
            wrong_device = 'cpu' if self.device_type != 'cpu' else 'cuda'
            rcond = torch.full((), 1e-2, device=wrong_device)
            with self.assertRaisesRegex(RuntimeError, "Expected all tensors to be on the same device"):
                torch.linalg.pinv(a, rcond=rcond)

        # rcond can't be complex
        rcond = torch.full((), 1j, device=device)
        with self.assertRaisesRegex(RuntimeError, "rcond tensor of complex type is not supported"):
            torch.linalg.pinv(a, rcond=rcond)

        # atol can't be complex
        atol = torch.full((), 1j, device=device)
        with self.assertRaisesRegex(RuntimeError, "atol tensor of complex type is not supported"):
            torch.linalg.pinv(a, atol=atol)

        # rtol can't be complex
        rtol = torch.full((), 1j, device=device)
        with self.assertRaisesRegex(RuntimeError, "rtol tensor of complex type is not supported"):
            torch.linalg.pinv(a, rtol=rtol)

    @skipCUDAIfNoMagmaAndNoCusolver
    @skipCPUIfNoLapack
    @dtypes(*floating_and_complex_types())
    def test_inv_errors_and_warnings(self, device, dtype):
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
            with self.assertRaisesRegex(torch.linalg.LinAlgError, rf"\(Batch element {n}\): The diagonal element 3 is zero"):
                torch.linalg.inv(a)

        for params in [(1, 0), (2, 0), (2, 1), (4, 0), (4, 2), (10, 2)]:
            run_test_singular_input(*params)

        # dtypes should match
        a = torch.eye(2, dtype=dtype, device=device)
        out = torch.empty(0, dtype=torch.int, device=device)
        with self.assertRaisesRegex(RuntimeError, "but got int instead"):
            torch.linalg.inv(a, out=out)

        # device should match
        if torch.cuda.is_available():
            wrong_device = 'cpu' if self.device_type != 'cpu' else 'cuda'
            out = torch.empty(0, device=wrong_device, dtype=dtype)
            with self.assertRaisesRegex(RuntimeError, "tensors to be on the same device"):
                torch.linalg.inv(a, out=out)

        # if out tensor with wrong shape is passed a warning is given
        with warnings.catch_warnings(record=True) as w:
            a = torch.eye(2, dtype=dtype, device=device)
            out = torch.empty(1, dtype=dtype, device=device)
            # Trigger warning
            torch.linalg.inv(a, out=out)
            # Check warning occurs
            self.assertEqual(len(w), 1)
            self.assertTrue("An output with one or more elements was resized" in str(w[-1].message))

        # if out tensor in batched column major format but with wrong a warning is given
        with warnings.catch_warnings(record=True) as w:
            a = torch.eye(2, dtype=dtype, device=device)
            out = torch.empty(3, 3, dtype=dtype, device=device)
            out = out.mT.clone(memory_format=torch.contiguous_format)
            out = out.mT
            self.assertTrue(out.mT.is_contiguous())
            # Trigger warning
            torch.linalg.inv(a, out=out)
            # Check warning occurs
            self.assertEqual(len(w), 1)
            self.assertTrue("An output with one or more elements was resized" in str(w[-1].message))

    def solve_test_helper(self, A_dims, b_dims, device, dtype):
        make_fullrank = make_fullrank_matrices_with_distinct_singular_values
        make_A = partial(make_fullrank, device=device, dtype=dtype)

        b = torch.randn(*b_dims, dtype=dtype, device=device)
        A = make_A(*A_dims)
        return b, A

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(*floating_and_complex_types())
    @precisionOverride({torch.float32: 1e-3, torch.complex64: 1e-3})
    def test_solve(self, device, dtype):
        def run_test(n, batch, rhs):
            A_dims = (*batch, n, n)
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

        batches = [(), (0, ), (3, ), (2, 3)]
        ns = [0, 5, 32]
        nrhs = [(), (1, ), (5, )]
        for n, batch, rhs in itertools.product(ns, batches, nrhs):
            run_test(n, batch, rhs)

    @skipCUDAIfNoMagmaAndNoCusolver
    @skipCPUIfNoLapack
    @dtypes(*floating_and_complex_types())
    def test_solve_batched_broadcasting(self, device, dtype):
        from numpy.linalg import solve

        def run_test(A_dims, B_dims):
            A_matrix_size = A_dims[-1]
            A_batch_dims = A_dims[:-2]
            B, A = self.solve_test_helper(A_batch_dims + (A_matrix_size, A_matrix_size), B_dims, device, dtype)
            actual = torch.linalg.solve(A, B)
            expected = solve(A.cpu().numpy(), B.cpu().numpy())
            self.assertEqual(actual, expected)

        # test against numpy.linalg.solve
        run_test((5, 5), (2, 0, 5, 3))  # broadcasting with 0 batch dim
        run_test((2, 0, 5, 5), (5, 3))  # broadcasting with 0 batch dim
        run_test((2, 1, 3, 4, 4), (4, 6))  # broadcasting B
        run_test((4, 4), (2, 1, 3, 4, 2))  # broadcasting A
        run_test((1, 3, 1, 4, 4), (2, 1, 3, 4, 5))  # broadcasting A & B

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
    @dtypes(torch.float32)
    def test_tensorsolve_errors_and_warnings(self, device, dtype):
        # tensorsolve expects the input that can be reshaped to a square matrix
        a = torch.eye(2 * 3 * 4, dtype=dtype, device=device).reshape((2 * 3, 4, 2, 3, 4))
        b = torch.randn(8, 4, dtype=dtype, device=device)
        self.assertTrue(np.prod(a.shape[2:]) != np.prod(b.shape))
        with self.assertRaisesRegex(RuntimeError, r'Expected self to satisfy the requirement'):
            torch.linalg.tensorsolve(a, b)

        # if non-empty out tensor with wrong shape is passed a warning is given
        out = torch.empty_like(a)
        b = torch.randn(6, 4, dtype=dtype, device=device)
        with warnings.catch_warnings(record=True) as w:
            # Trigger warning
            torch.linalg.tensorsolve(a, b, out=out)
            # Check warning occurs
            self.assertEqual(len(w), 1)
            self.assertTrue("An output with one or more elements was resized" in str(w[-1].message))

        # dtypes should be safely castable
        out = torch.empty_like(a).to(torch.int)
        with self.assertRaisesRegex(RuntimeError, "but got result with dtype Int"):
            torch.linalg.tensorsolve(a, b, out=out)

        # device should match
        if torch.cuda.is_available():
            wrong_device = 'cpu' if self.device_type != 'cpu' else 'cuda'
            out = torch.empty(0, dtype=dtype, device=wrong_device)
            with self.assertRaisesRegex(RuntimeError, "tensors to be on the same device"):
                torch.linalg.tensorsolve(a, b, out=out)

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(*floating_and_complex_types())
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

    @skipMeta  # See https://github.com/pytorch/pytorch/issues/53739
    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(*floating_and_complex_types())
    def test_tensorinv_empty(self, device, dtype):
        for ind in range(1, 4):
            # Check for empty inputs. NumPy does not work for these cases.
            a = torch.empty(0, 0, 1, 2, 3, 0, dtype=dtype, device=device)
            a_inv = torch.linalg.tensorinv(a, ind=ind)
            self.assertEqual(a_inv.shape, a.shape[ind:] + a.shape[:ind])

    @skipMeta  # See https://github.com/pytorch/pytorch/issues/53739
    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(*floating_and_complex_types())
    def test_tensorinv_errors_and_warnings(self, device, dtype):

        def check_shape(a_shape, ind):
            # tensorinv requires the input to satisfy
            # prod(a.shape[ind:]) == prod(a.shape[:ind])
            a = torch.randn(a_shape, dtype=dtype, device=device)
            with self.assertRaisesRegex(RuntimeError, "Expected self to satisfy the requirement"):
                torch.linalg.tensorinv(a, ind=ind)

        def check_ind(a_shape, ind):
            a = torch.randn(a_shape, dtype=dtype, device=device)
            with self.assertRaisesRegex(RuntimeError, "Expected a strictly positive integer"):
                torch.linalg.tensorinv(a, ind=ind)

        def check_out(a_shape, ind):
            # if non-empty out tensor with wrong shape is passed a warning is given
            a = torch.randn(a_shape, dtype=dtype, device=device)
            out = torch.empty_like(a)
            with warnings.catch_warnings(record=True) as w:
                # Trigger warning
                torch.linalg.tensorinv(a, ind=ind, out=out)
                # Check warning occurs
                self.assertEqual(len(w), 1)
                self.assertTrue("An output with one or more elements was resized" in str(w[-1].message))

            # dtypes should be safely castable
            out = torch.empty(0, dtype=torch.int, device=device)
            with self.assertRaisesRegex(RuntimeError, "but got result with dtype Int"):
                torch.linalg.tensorinv(a, ind=ind, out=out)

            # device should match
            if torch.cuda.is_available():
                wrong_device = 'cpu' if self.device_type != 'cpu' else 'cuda'
                out = torch.empty(0, dtype=dtype, device=wrong_device)
                with self.assertRaisesRegex(RuntimeError, "tensors to be on the same device"):
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
    @dtypes(*floating_and_complex_types())
    def test_tensorinv_singular_input(self, device, dtype):

        def check_singular_input(a_shape, ind):
            prod_ind_end = np.prod(a_shape[ind:])
            a = torch.eye(prod_ind_end, dtype=dtype, device=device)
            a[-1, -1] = 0   # Now `a` is singular
            a = a.reshape(a_shape)
            with self.assertRaisesRegex(torch.linalg.LinAlgError, "The diagonal element"):
                torch.linalg.tensorinv(a, ind=ind)

        # test for non-invertible input
        check_singular_input((12, 3, 4), ind=1)
        check_singular_input((3, 6, 18), ind=2)

    def _test_dot_vdot_vs_numpy(self, device, dtype, torch_fn, np_fn):
        def check(x, y):
            # Compare with numpy
            res = torch_fn(x, y)
            if x.dtype == torch.bfloat16:
                ref = torch.from_numpy(np.array(np_fn(x.cpu().float().numpy(), y.cpu().float().numpy())))
            else:
                ref = torch.from_numpy(np.array(np_fn(x.cpu().numpy(), y.cpu().numpy())))
            if res.dtype == torch.bfloat16:
                self.assertEqual(res.cpu(), ref.bfloat16())
            else:
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
        x = 0.1 * torch.randn(5000, dtype=dtype, device=device)
        y = 0.1 * torch.randn(5000, dtype=dtype, device=device)
        check(x, y)

        # 0 strided
        y = 0.1 * torch.randn(1, dtype=dtype, device=device).expand(5000)
        check(x, y)

        # 2 strided
        check(x[::2], y[::2])

    @dtypes(torch.float, torch.cfloat, torch.bfloat16)
    @dtypesIfCUDA(torch.float, torch.cfloat)
    @precisionOverride({torch.cfloat: 1e-4, torch.float32: 5e-5, torch.bfloat16: 1e-0})
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
            check(x_cpu, y.to(x.dtype), 'Expected all tensors to be on the same device')

    @onlyNativeDeviceTypes
    def test_vdot_invalid_args(self, device):
        self._test_dot_vdot_invalid_args(device, torch.vdot)
        self._test_dot_vdot_invalid_args(device, torch.vdot, complex_dtypes=True)

    @onlyNativeDeviceTypes
    def test_dot_invalid_args(self, device):
        self._test_dot_vdot_invalid_args(device, torch.dot)
        self._test_dot_vdot_invalid_args(device, torch.dot, complex_dtypes=True)

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(*floating_and_complex_types())
    def test_matrix_rank(self, device, dtype):
        matrix_rank = torch.linalg.matrix_rank

        def run_test(shape0, shape1, batch):
            a = torch.randn(*batch, shape0, shape1, dtype=dtype, device=device)
            rank_a = matrix_rank(a)

            self.assertEqual(rank_a, matrix_rank(a.mH))
            aaH = torch.matmul(a, a.mH)
            rank_aaH = matrix_rank(aaH)
            rank_aaH_hermitian = matrix_rank(aaH, hermitian=True)
            self.assertEqual(rank_aaH, rank_aaH_hermitian)
            aHa = torch.matmul(a.mH, a)
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
    @dtypes(*floating_and_complex_types())
    def test_matrix_rank_atol(self, device, dtype):

        def run_test_atol(shape0, shape1, batch):
            a = make_tensor((*batch, shape0, shape1), dtype=dtype, device=device)
            # Check against NumPy output
            # Test float tol, and specific value for each matrix
            tolerances = [float(torch.rand(1)), ]
            # Test different types of tol tensor
            for tol_type in all_types():
                tolerances.append(make_tensor(a.shape[:-2], dtype=tol_type, device=device, low=0))
            # Test broadcasting of tol
            if a.ndim > 2:
                tolerances.append(make_tensor(a.shape[-3], dtype=torch.float32, device=device, low=0))
            for tol in tolerances:
                actual = torch.linalg.matrix_rank(a, atol=tol)
                actual_tol = torch.linalg.matrix_rank(a, tol=tol)
                self.assertEqual(actual, actual_tol)
                numpy_tol = tol if isinstance(tol, float) else tol.cpu().numpy()
                expected = np.linalg.matrix_rank(a.cpu().numpy(), tol=numpy_tol)
                self.assertEqual(actual, expected)

        shapes = (3, 13)
        batches = ((), (0, ), (4, ), (3, 5, ))
        for (shape0, shape1), batch in zip(itertools.product(shapes, reversed(shapes)), batches):
            run_test_atol(shape0, shape1, batch)

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(torch.float64)
    def test_matrix_rank_atol_rtol(self, device, dtype):
        make_fullrank = make_fullrank_matrices_with_distinct_singular_values
        make_arg = partial(make_fullrank, device=device, dtype=dtype)

        # creates a matrix with singular values rank=n and singular values in range [2/3, 3/2]
        # the singular values are 1 + 1/2, 1 - 1/3, 1 + 1/4, 1 - 1/5, ...
        n = 9
        a = make_arg(n, n)

        # test float and tensor variants
        for tol_value in [0.81, torch.tensor(0.81, device=device)]:
            # using rtol (relative tolerance) takes into account the largest singular value (1.5 in this case)
            result = torch.linalg.matrix_rank(a, rtol=tol_value)
            self.assertEqual(result, 2)  # there are 2 singular values above 1.5*0.81 = 1.215

            # atol is used directly to compare with singular values
            result = torch.linalg.matrix_rank(a, atol=tol_value)
            self.assertEqual(result, 7)  # there are 7 singular values above 0.81

            # when both are specified the maximum tolerance is used
            result = torch.linalg.matrix_rank(a, atol=tol_value, rtol=tol_value)
            self.assertEqual(result, 2)  # there are 2 singular values above max(0.81, 1.5*0.81)

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @skipCUDAVersionIn([(11, 6), (11, 7)])  # https://github.com/pytorch/pytorch/issues/75391
    @dtypes(*floating_and_complex_types())
    def test_matrix_rank_empty(self, device, dtype):
        matrix_rank = torch.linalg.matrix_rank

        # NumPy doesn't work for input with no elements
        def run_test(shape0, shape1, batch):
            a = torch.randn(*batch, shape0, shape1, dtype=dtype, device=device)
            rank_a = matrix_rank(a)
            expected = torch.zeros(batch, dtype=torch.int64, device=device)

            self.assertEqual(rank_a, matrix_rank(a.mH))

            aaH = torch.matmul(a, a.mH)
            rank_aaH = matrix_rank(aaH)
            rank_aaH_hermitian = matrix_rank(aaH, hermitian=True)
            self.assertEqual(rank_aaH, rank_aaH_hermitian)

            aHa = torch.matmul(a.mH, a)
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
    @dtypes(*floating_and_complex_types())
    def test_matrix_rank_out_errors_and_warnings(self, device, dtype):
        # dtypes should be safely castable
        a = torch.eye(2, dtype=dtype, device=device)
        out = torch.empty(0, dtype=torch.bool, device=device)
        with self.assertRaisesRegex(RuntimeError, "but got result with dtype Bool"):
            torch.linalg.matrix_rank(a, out=out)

        # device should match
        if torch.cuda.is_available():
            wrong_device = 'cpu' if self.device_type != 'cpu' else 'cuda'
            out = torch.empty(0, dtype=dtype, device=wrong_device)
            with self.assertRaisesRegex(RuntimeError, "tensors to be on the same device"):
                torch.linalg.matrix_rank(a, out=out)

        # if out tensor with wrong shape is passed a warning is given
        with warnings.catch_warnings(record=True) as w:
            out = torch.empty(3, dtype=dtype, device=device)
            # Trigger warning
            torch.linalg.matrix_rank(a, out=out)
            # Check warning occurs
            self.assertEqual(len(w), 1)
            self.assertTrue("An output with one or more elements was resized" in str(w[-1].message))

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(*floating_and_complex_types())
    def test_matrix_rank_basic(self, device, dtype):
        matrix_rank = torch.linalg.matrix_rank

        a = torch.eye(10, dtype=dtype, device=device)
        self.assertEqual(matrix_rank(a).item(), 10)
        self.assertEqual(matrix_rank(a, hermitian=True).item(), 10)

        a[5, 5] = 0
        self.assertEqual(matrix_rank(a).item(), 9)
        self.assertEqual(matrix_rank(a, hermitian=True).item(), 9)

    @onlyNativeDeviceTypes
    @dtypes(torch.double)
    # This tests only the cases where torch.chain_matmul differs from torch.linalg.multi_dot which this is an "alias" for.
    def test_chain_matmul(self, device, dtype):
        # chain_matmul accepts a single input tensor while multi_dot does not
        t = make_tensor((2, 2), dtype=dtype, device=device)
        self.assertEqual(t, torch.chain_matmul(t))
        with self.assertRaisesRegex(RuntimeError, r"chain_matmul\(\): Expected one or more matrices"):
            torch.chain_matmul()

        # chain_matmul expects all tensors to be 2D whereas multi_dot allows the first and last tensors to
        # be either 1D or 2D
        with self.assertRaisesRegex(RuntimeError, r"Tensor dimension is 1, expected 2 instead"):
            torch.chain_matmul(make_tensor(1, dtype=dtype, device=device), make_tensor(1, dtype=dtype, device=device))

    @onlyNativeDeviceTypes
    @dtypes(torch.double, torch.cdouble)
    def test_multi_dot(self, device, dtype):
        def check(*shapes):
            tensors = [make_tensor(shape, dtype=dtype, device=device) for shape in shapes]
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
        check([10, 20], [20, 30], [30, 5])

    @onlyNativeDeviceTypes
    @dtypes(torch.float)
    def test_multi_dot_errors(self, device, dtype):
        def check(tensors, out, msg):
            with self.assertRaisesRegex(RuntimeError, msg):
                torch.linalg.multi_dot(tensors, out=out)

        a = make_tensor(2, dtype=dtype, device=device)

        check([], None, "expected at least 2 tensors")
        check([a], None, "expected at least 2 tensors")

        check([torch.tensor(1, device=device, dtype=dtype), a], None, "the first tensor must be 1D or 2D")
        check([a, torch.tensor(1, device=device, dtype=dtype)], None, "the last tensor must be 1D or 2D")

        check([a, a, a], None, "tensor 1 must be 2D")
        check([a, make_tensor((2, 2, 2), dtype=dtype, device=device), a], None, "tensor 1 must be 2D")

        check([a, make_tensor(2, dtype=torch.double, device=device)], None, "all tensors must have be the same dtype")
        check([a, a], torch.empty(0, device=device, dtype=torch.double), "expected out tensor to have dtype")

        if self.device_type == 'cuda':
            check([a, make_tensor(2, dtype=dtype, device="cpu")], None, "all tensors must be on the same device")
            check([a, a], torch.empty(0, dtype=dtype), "expected out tensor to be on device")

        check([a, make_tensor(3, dtype=dtype, device=device)], None, "cannot be multiplied")
        check([a, make_tensor((3, 2), dtype=dtype, device=device), a], None, "cannot be multiplied")

    @precisionOverride({torch.float32: 5e-6, torch.complex64: 5e-6})
    @skipCUDAIfNoCusolver
    @skipCPUIfNoLapack
    @dtypes(*floating_and_complex_types())
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

        tensor_dims_list = [(0, 5), (0, 0), (5, 0),  # Empty Tensors
                            (2, 1, 0, 5), (2, 1, 0, 0), (2, 1, 5, 0), (2, 0, 5, 5),  # Batched empty Tensors
                            (3, 5), (5, 5), (5, 3),  # Single matrix
                            (7, 3, 5), (7, 5, 5), (7, 5, 3),  # 3-dim Tensors
                            (7, 5, 3, 5), (7, 5, 5, 5), (7, 5, 5, 3)]  # 4-dim Tensors
        for tensor_dims, some in itertools.product(tensor_dims_list, [True, False]):
            run_test(tensor_dims, some)

    @skipCUDAIfNoCusolver
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

    @skipCUDAIfNoCusolver
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
                                    "The derivative of linalg.qr depends on Q"):
            b.backward()
        #
        inp = torch.randn((7, 5), device=device, dtype=dtype, requires_grad=True)
        q, r = torch.linalg.qr(inp, mode='complete')
        b = torch.sum(r)
        with self.assertRaisesRegex(RuntimeError,
                                    "The QR decomposition is not differentiable when mode='complete' and nrows > ncols"):
            b.backward()

    @skipCUDAIfNoCusolver
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

    @skipCUDAIfNoCusolver
    @skipCPUIfNoLapack
    @dtypes(torch.float)
    def test_qr_error_cases(self, device, dtype):
        t1 = torch.randn(5, device=device, dtype=dtype)
        with self.assertRaisesRegex(RuntimeError, 'linalg.qr: The input tensor A must have at least 2 dimensions.'):
            torch.linalg.qr(t1)
        t2 = torch.randn((5, 7), device=device, dtype=dtype)
        with self.assertRaisesRegex(RuntimeError, "qr received unrecognized mode 'hello'"):
            torch.linalg.qr(t2, mode='hello')

    def _check_einsum(self, *args, np_args=None):
        if np_args is None:
            np_args = [arg.cpu().numpy() if isinstance(arg, torch.Tensor) else arg for arg in args]
        ref = np.einsum(*np_args)
        res = torch.einsum(*args)
        self.assertEqual(ref, res)

        # Check that the other variations for opt_einsum work too
        if TEST_OPT_EINSUM:
            with opt_einsum.flags(enabled=False):
                res = torch.einsum(*args)
                self.assertEqual(ref, res)

            with opt_einsum.flags(enabled=True, strategy='greedy'):
                res = torch.einsum(*args)
                self.assertEqual(ref, res)

            with opt_einsum.flags(enabled=True, strategy='optimal'):
                res = torch.einsum(*args)
                self.assertEqual(ref, res)

    @dtypes(torch.double, torch.cdouble)
    def test_einsum(self, device, dtype):
        # Test cases from https://gist.github.com/rockt/15ee013889d65342088e9260a377dc8f
        x = make_tensor((5,), dtype=dtype, device=device)
        y = make_tensor((7,), dtype=dtype, device=device)
        A = make_tensor((3, 5), dtype=dtype, device=device)
        B = make_tensor((2, 5), dtype=dtype, device=device)
        C = make_tensor((2, 3, 5), dtype=dtype, device=device)
        D = make_tensor((2, 5, 7), dtype=dtype, device=device)
        E = make_tensor((7, 9), dtype=dtype, device=device)
        F = make_tensor((2, 3, 3, 5), dtype=dtype, device=device)
        G = make_tensor((5, 4, 6), dtype=dtype, device=device)
        H = make_tensor((4, 4), dtype=dtype, device=device)
        I = make_tensor((2, 3, 2), dtype=dtype, device=device)

        # Vector operations
        self._check_einsum('i->', x)                     # sum
        self._check_einsum('i,i->', x, x)                # dot
        self._check_einsum('i,i->i', x, x)               # vector element-wisem mul
        self._check_einsum('i,j->ij', x, y)              # outer

        # Matrix operations
        self._check_einsum("ij->ji", A)                  # transpose
        self._check_einsum("ij->j", A)                   # row sum
        self._check_einsum("ij->i", A)                   # col sum
        self._check_einsum("ij,ij->ij", A, A)            # matrix element-wise mul
        self._check_einsum("ij,j->i", A, x)              # matrix vector multiplication
        self._check_einsum("ij,kj->ik", A, B)            # matmul
        self._check_einsum("ij,ab->ijab", A, E)          # matrix outer product

        # Tensor operations
        self._check_einsum("Aij,Ajk->Aik", C, D)         # batch matmul
        self._check_einsum("ijk,jk->i", C, A)            # tensor matrix contraction
        self._check_einsum("aij,jk->aik", D, E)          # tensor matrix contraction
        self._check_einsum("abCd,dFg->abCFg", F, G)      # tensor tensor contraction
        self._check_einsum("ijk,jk->ik", C, A)           # tensor matrix contraction with double indices
        self._check_einsum("ijk,jk->ij", C, A)           # tensor matrix contraction with double indices
        self._check_einsum("ijk,ik->j", C, B)            # non contiguous
        self._check_einsum("ijk,ik->jk", C, B)           # non contiguous with double indices

        # Test diagonals
        self._check_einsum("ii", H)                      # trace
        self._check_einsum("ii->i", H)                   # diagonal
        self._check_einsum('iji->j', I)                  # non-contiguous trace
        self._check_einsum('ngrg...->nrg...', make_tensor((2, 1, 3, 1, 4), dtype=dtype, device=device))

        # Test ellipsis
        self._check_einsum("i...->...", H)
        self._check_einsum("ki,...k->i...", A.t(), B)
        self._check_einsum("k...,jk->...", A.t(), B)
        self._check_einsum('...ik, ...j -> ...ij', C, x)
        self._check_einsum('Bik,k...j->i...j', C, make_tensor((5, 3), dtype=dtype, device=device))
        self._check_einsum('i...j, ij... -> ...ij', C, make_tensor((2, 5, 2, 3), dtype=dtype, device=device))

        # torch.bilinear with noncontiguous tensors
        l = make_tensor((5, 10), dtype=dtype, device=device, noncontiguous=True)
        r = make_tensor((5, 20), dtype=dtype, device=device, noncontiguous=True)
        w = make_tensor((15, 10, 20), dtype=dtype, device=device)
        self._check_einsum("bn,anm,bm->ba", l, w, r)

        # with strided tensors
        self._check_einsum("bn,Anm,bm->bA", l[:, ::2], w[:, ::2, ::2], r[:, ::2])

        # test multiple inputs
        self._check_einsum("...,be,b...,beg,gi,bc...->bi...", A, B, C, D, E, F)

    @dtypes(torch.double, torch.cdouble)
    def test_einsum_sublist_format(self, device, dtype):
        x = make_tensor((5,), dtype=dtype, device=device)
        y = make_tensor((7,), dtype=dtype, device=device)
        A = make_tensor((3, 5), dtype=dtype, device=device)
        B = make_tensor((2, 5), dtype=dtype, device=device)
        C = make_tensor((2, 1, 3, 1, 4), dtype=dtype, device=device)

        self._check_einsum(x, [0])
        self._check_einsum(x, [0], [])
        self._check_einsum(x, [0], y, [1], [0, 1])
        self._check_einsum(A, [0, 1], [1, 0])
        self._check_einsum(A, [0, 1], x, [1], [0])
        self._check_einsum(A, [0, 1], B, [2, 1])
        self._check_einsum(A, [0, 1], B, [2, 1], [0, 2])
        self._check_einsum(C, [0, 1, 2, 1, Ellipsis], [0, 2, 1, Ellipsis])
        self._check_einsum(A.t(), [0, 1], B, [Ellipsis, 0])
        self._check_einsum(A.t(), [0, 1], B, [Ellipsis, 0], [1, Ellipsis])
        self._check_einsum(A.t(), [0, Ellipsis], B, [1, 0], [Ellipsis])

        # torch.bilinear with noncontiguous tensors
        l = make_tensor((5, 10), dtype=dtype, device=device, noncontiguous=True)
        r = make_tensor((5, 20), dtype=dtype, device=device, noncontiguous=True)
        w = make_tensor((15, 10, 20), dtype=dtype, device=device)
        self._check_einsum(l, [40, 41], w, [2, 41, 50], r, [40, 50], [40, 2])

    @dtypes(torch.double, torch.cdouble)
    def test_einsum_random(self, device, dtype):
        def convert_label(label):
            if label == ...:
                return '...'
            elif label < 26:
                return chr(ord('A') + label)
            else:
                return chr(ord('a') + label - 26)

        def convert_sublist(sublist):
            return ''.join(convert_label(label) for label in sublist)

        def test(n=10,                       # how many tests to generate
                 n_labels=5,                 # how many labels available
                 min_ops=1, max_ops=4,       # min and max number of operands per test
                 min_dims=1, max_dims=3,     # min and max number of dimensions per operand
                 min_size=1, max_size=8,     # min and max size of each dimension
                 max_out_dim=3,              # max number of dimensions for the output
                 enable_diagonals=True,      # controls if labels can be repeated for diagonals
                 ellipsis_prob=0.5,          # probability of including ellipsis in operand
                 broadcasting_prob=0.1):     # probability of turning some dim sizes 1 for broadcasting

            all_labels = torch.arange(52)

            assert 0 <= n
            assert 0 <= n_labels < len(all_labels)
            assert 0 < min_ops <= max_ops
            assert 0 <= min_dims <= max_dims
            assert 0 <= min_size <= max_size
            assert 0 <= max_out_dim
            assert enable_diagonals or max_dims <= n_labels

            for _ in range(n):

                # Select a subset of labels for this test and give them random sizes
                possible_labels = all_labels[torch.randperm(len(all_labels))[:n_labels]]
                labels_size = torch.randint_like(all_labels, min_size, max_size + 1)
                ellipsis_shape = torch.randint(min_size, max_size + 1, (max_dims - min_dims,))

                operands = []
                sublists = []

                ell_size = 0
                valid_labels = set()

                # create random input operands
                for _ in range(random.randint(min_ops, max_ops)):
                    n_dim = random.randint(min_dims, max_dims)
                    labels_idx = torch.ones(len(possible_labels)).multinomial(n_dim, enable_diagonals)
                    labels = possible_labels[labels_idx]
                    valid_labels.update(labels.tolist())
                    shape = labels_size[labels]

                    # turn some dimensions to size 1 for testing broadcasting
                    mask = Binomial(probs=broadcasting_prob).sample((n_dim,))
                    broadcast_labels = torch.unique(labels[mask == 1])
                    shape[(labels[..., None] == broadcast_labels).any(-1)] = 1

                    labels = labels.tolist()
                    shape = shape.tolist()

                    # include ellipsis if not all dimensions were assigned a label already
                    if n_dim < max_dims and torch.rand(1) < ellipsis_prob:
                        ell_num_dim = random.randint(1, max_dims - n_dim)
                        ell_size = max(ell_size, ell_num_dim)
                        ell_shape = ellipsis_shape[-ell_num_dim:]
                        # again, turn some dimensions to size 1 for broadcasting
                        mask = Binomial(probs=broadcasting_prob).sample((ell_num_dim,))
                        ell_shape[mask == 1] = 1
                        ell_index = random.randint(0, n_dim)
                        shape[ell_index:ell_index] = ell_shape
                        labels.insert(ell_index, ...)

                    operands.append(make_tensor(shape, dtype=dtype, device=device))
                    sublists.append(labels)

                # NumPy has a bug with the sublist format so for now we compare PyTorch sublist
                # implementation against the equation format implementation of NumPy
                # see https://github.com/numpy/numpy/issues/10926
                np_operands = [op.cpu().numpy() for op in operands]

                # test equation format
                equation = ','.join(convert_sublist(l) for l in sublists)
                self._check_einsum(equation, *operands, np_args=(equation, *np_operands))

                # test sublist format
                args = [*itertools.chain(*zip(operands, sublists))]
                self._check_einsum(*args, np_args=(equation, *np_operands))

                # generate an explicit output
                out_sublist = []
                num_out_labels = max(0, random.randint(0, min(max_out_dim, len(valid_labels))) - ell_size)
                if num_out_labels > 0:
                    out_labels_idx = torch.ones(len(valid_labels)).multinomial(num_out_labels)
                    out_sublist = torch.tensor(list(valid_labels))[out_labels_idx].tolist()
                out_sublist.insert(random.randint(0, num_out_labels), ...)

                # test equation format with explicit output
                equation += '->' + convert_sublist(out_sublist)
                self._check_einsum(equation, *operands, np_args=(equation, *np_operands))

                # test sublist format with explicit output
                args.append(out_sublist)
                self._check_einsum(*args, np_args=(equation, *np_operands))

        test(500)

    def test_einsum_corner_cases(self, device):
        def check(equation, *operands, expected_output):
            tensors = [torch.tensor(operand, device=device, dtype=torch.float32) if not isinstance(operand, tuple)
                       else make_tensor(operand, dtype=torch.float32, device=device) for operand in operands]
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
        check('z...->z', [1], expected_output=[1])
        check('Z...->...Z', [1], expected_output=[1])
        check('...a->', [[2], [4]], expected_output=6)
        check('a...b->ab', [[[1], [2]], [[3], [4]]], expected_output=[[3], [7]])

    def test_einsum_error_cases(self, device):
        def check(*args, regex, exception=RuntimeError):
            with self.assertRaisesRegex(exception, r'einsum\(\):.*' + regex):
                torch.einsum(*args)

        x = make_tensor((2,), dtype=torch.float32, device=device)
        y = make_tensor((2, 3), dtype=torch.float32, device=device)

        check('', [], regex=r'at least one operand', exception=ValueError)
        check('. ..', [x], regex=r'found \'.\' for operand 0 that is not part of any ellipsis')
        check('... ...', [x], regex=r'found \'.\' for operand 0 for which an ellipsis was already found')
        check('1', [x], regex=r'invalid subscript given at index 0')
        check(',', [x], regex=r'fewer operands were provided than specified in the equation')
        check('', [x, x], regex=r'more operands were provided than specified in the equation')
        check('', [x], regex=r'the number of subscripts in the equation \(0\) does not match the number '
              r'of dimensions \(1\) for operand 0 and no ellipsis was given')
        check('ai', [x], regex=r'the number of subscripts in the equation \(2\) does not match the number '
              r'of dimensions \(1\) for operand 0 and no ellipsis was given')
        check('ai...', [x], regex=r'the number of subscripts in the equation \(2\) is more than the number '
              r'of dimensions \(1\) for operand 0')
        check('a->... .', [x], regex=r'found \'.\' for output but an ellipsis \(...\) was already found')
        check('a->..', [x], regex=r'found \'.\' for output that is not part of any ellipsis \(...\)')
        check('a->1', [x], regex=r'invalid subscript given at index 3')
        check('a->aa', [x], regex=r'output subscript a appears more than once in the output')
        check('a->i', [x], regex=r'output subscript i does not appear in the equation for any input operand')
        check('aa', [y], regex=r'subscript a is repeated for operand 0 but the sizes don\'t match, 3 != 2')
        check('...,...', [x, y], regex=r'does not broadcast')
        check('a,a', [x, make_tensor((3,), dtype=torch.float32, device=device)], regex=r'does not broadcast')
        check('a, ba', [x, y], regex=r'subscript a has size 3 for operand 1 which does not broadcast with previously'
              r' seen size 2')

        check(x, [-1], regex=r'not within the valid range \[0, 52\)', exception=ValueError)
        check(x, [52], regex=r'not within the valid range \[0, 52\)', exception=ValueError)

    def _gen_shape_inputs_linalg_triangular_solve(self, shape, dtype, device, well_conditioned=False):
        make_arg = partial(make_tensor, dtype=dtype, device=device)
        make_randn = partial(torch.randn, dtype=dtype, device=device)
        b, n, k = shape
        for left, uni, expand_a, tr_a, conj_a, expand_b, tr_b, conj_b in product((True, False), repeat=8):
            # expand means that we generate a batch of matrices with a stride of zero in the batch dimension
            if (conj_a or conj_b) and not dtype.is_complex:
                continue
            # We just expand on the batch size
            if (expand_a or expand_b) and b == 1:
                continue

            size_a = (b, n, n) if left else (b, k, k)
            size_b = (b, n, k) if not tr_b else (b, k, n)

            # If expand_a or expand_b, we'll expand them to the correct size later
            if b == 1 or expand_a:
                size_a = size_a[1:]
            if b == 1 or expand_b:
                size_b = size_b[1:]

            if well_conditioned:
                PLU = torch.linalg.lu(make_randn(*size_a))
                if uni:
                    # A = L from PLU
                    A = PLU[1].transpose(-2, -1).contiguous()
                else:
                    # A = U from PLU
                    A = PLU[2].contiguous()
            else:
                A = make_arg(size_a)
                A.triu_()

            diag = A.diagonal(0, -2, -1)
            if uni:
                diag.fill_(1.)
            else:
                diag[diag.abs() < 1e-6] = 1.

            B = make_arg(size_b)

            if tr_a:
                A.transpose_(-2, -1)
            if tr_b:
                B.transpose_(-2, -1)
            if conj_a:
                A = A.conj()
            if conj_b:
                B = B.conj()
            if expand_a:
                A = A.expand(b, *size_a)
            if expand_b:
                B = B.expand(b, n, k)
            yield A, B, left, not tr_a, uni

    def _test_linalg_solve_triangular(self, A, B, upper, left, uni):
        X = torch.linalg.solve_triangular(A, B, upper=upper, left=left, unitriangular=uni)
        if left:
            self.assertEqual(A @ X, B)
        else:
            self.assertEqual(X @ A, B)
        out = B
        # B may be expanded
        if not B.is_contiguous() and not B.transpose(-2, -1).is_contiguous():
            out = B.clone()
        torch.linalg.solve_triangular(A, B, upper=upper, left=left, unitriangular=uni, out=out)
        self.assertEqual(X, out)

    @dtypes(*floating_and_complex_types())
    @precisionOverride({torch.float32: 1e-1, torch.complex64: 1e-1,
                        torch.float64: 1e-8, torch.complex128: 1e-8})
    def test_linalg_solve_triangular(self, device, dtype):
        if TEST_WITH_ROCM and dtype is torch.float32:
            raise unittest.SkipTest("Skipping for ROCm for Magma backend; unskip when hipSolver backend is enabled")
        # This exercises the API + BLAS CPU + batched cuBLAS
        ks = (3, 1, 0)
        ns = (5, 0)
        bs = (1, 2, 0)

        gen_inputs = self._gen_shape_inputs_linalg_triangular_solve
        for b, n, k in product(bs, ns, ks):
            for A, B, left, upper, uni in gen_inputs((b, n, k), dtype, device):
                self._test_linalg_solve_triangular(A, B, upper, left, uni)

    @unittest.skipIf(IS_FBCODE or IS_SANDCASTLE, "Test fails for float64 on GPU (P100, V100) on Meta infra")
    @onlyCUDA
    @skipCUDAIfNoMagma  # Magma needed for the PLU decomposition
    @skipCUDAIfRocm  # There is a memory access bug in rocBLAS in the (non-batched) solve_triangular
    @dtypes(*floating_and_complex_types())
    @precisionOverride({torch.float32: 1e-2, torch.complex64: 1e-2,
                        torch.float64: 1e-8, torch.complex128: 1e-8})
    def test_linalg_solve_triangular_large(self, device, dtype):
        # Exercises magma and cublas
        magma = (9, 513, 1)
        iterative_cublas = (2, 64, 1)

        gen_inputs = self._gen_shape_inputs_linalg_triangular_solve
        for shape in (magma, iterative_cublas):
            for A, B, left, upper, uni in gen_inputs(shape, dtype, device, well_conditioned=True):
                self._test_linalg_solve_triangular(A, B, upper, left, uni)

    @dtypes(*floating_and_complex_types())
    @precisionOverride({torch.float32: 1e-2, torch.complex64: 1e-2,
                        torch.float64: 1e-8, torch.complex128: 1e-8})
    def test_linalg_solve_triangular_broadcasting(self, device, dtype):
        make_arg = partial(make_tensor, dtype=dtype, device=device)

        sizes = (((2, 1, 3, 4, 4), (2, 1, 3, 4, 6)),
                 ((2, 1, 3, 4, 4), (4, 6)),
                 ((4, 4), (2, 1, 3, 4, 2)),
                 ((1, 3, 1, 4, 4), (2, 1, 3, 4, 5)))
        for size_A, size_B in sizes:
            for left, upper, uni in itertools.product([True, False], repeat=3):
                A = make_arg(size_A)
                if upper:
                    A.triu_()
                else:
                    A.tril_()
                diag = A.diagonal(0, -2, -1)
                if uni:
                    diag.fill_(1.)
                else:
                    diag[diag.abs() < 1e-6] = 1.
                B = make_arg(size_B)
                if not left:
                    B.transpose_(-2, -1)

                X = torch.linalg.solve_triangular(A, B, upper=upper, left=left, unitriangular=uni)
                if left:
                    B_other = A @ X
                else:
                    B_other = X @ A

                self.assertEqual(*torch.broadcast_tensors(B, B_other))

    def triangular_solve_test_helper(self, A_dims, b_dims, upper, unitriangular,
                                     device, dtype):
        triangle_function = torch.triu if upper else torch.tril
        b = torch.randn(*b_dims, dtype=dtype, device=device)
        A = torch.randn(*A_dims, dtype=dtype, device=device)
        # create positive definite matrix
        A = torch.matmul(A, A.mT)
        A_triangular = triangle_function(A)
        if unitriangular:
            A_triangular.diagonal(dim1=-2, dim2=-1).fill_(1.)
        return b, A_triangular

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(*floating_and_complex_types())
    @precisionOverride({torch.float32: 1e-3, torch.complex64: 1e-3,
                        torch.float64: 1e-8, torch.complex128: 1e-8})
    def test_triangular_solve(self, device, dtype):
        ks = [0, 1, 3]
        ns = [0, 5]
        for k, n, (upper, unitriangular, transpose) in itertools.product(ks, ns,
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
    @dtypes(*floating_and_complex_types())
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
                A = A.mT

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
    @dtypes(*floating_and_complex_types())
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
                A = A.mT

            Ax = torch.matmul(A, x)

            rtol = 1e-2 if dtype in [torch.float32, torch.complex64] else self.precision
            self.assertEqual(Ax, b.expand_as(Ax), atol=self.precision, rtol=rtol)

            # test batched b case
            b, A = self.triangular_solve_test_helper((3, 3), (512, 512, 3, 1),
                                                     upper, unitriangular, device, dtype)
            x, _ = torch.triangular_solve(b, A, upper=upper, transpose=transpose,
                                          unitriangular=unitriangular)
            if transpose:
                A = A.mT

            self.assertEqual(torch.matmul(A, x), b)

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @unittest.skipIf(not TEST_SCIPY, "SciPy not found")
    @dtypes(*floating_and_complex_types())
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

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(*floating_and_complex_types())
    def test_triangular_solve_out_errors_and_warnings(self, device, dtype):
        # dtypes should be safely castable
        a = torch.eye(2, dtype=dtype, device=device)
        b = torch.randn(2, 1, dtype=dtype, device=device)
        out = torch.empty_like(b).to(torch.int)
        clone_a = torch.empty_like(a)
        with self.assertRaisesRegex(RuntimeError, "Expected out tensor to have dtype"):
            torch.triangular_solve(b, a, out=(out, clone_a))

        out = torch.empty_like(b)
        clone_a = clone_a.to(torch.int)
        with self.assertRaisesRegex(RuntimeError, "Expected out tensor to have dtype"):
            torch.triangular_solve(b, a, out=(out, clone_a))

        # device should match
        if torch.cuda.is_available():
            wrong_device = 'cpu' if self.device_type != 'cpu' else 'cuda'
            out = torch.empty(0, dtype=dtype, device=wrong_device)
            clone_a = torch.empty_like(a)
            with self.assertRaisesRegex(RuntimeError, "tensors to be on the same device"):
                torch.triangular_solve(b, a, out=(out, clone_a))
            out = torch.empty(0, dtype=dtype, device=device)
            clone_a = torch.empty_like(a).to(wrong_device)
            with self.assertRaisesRegex(RuntimeError, "tensors to be on the same device"):
                torch.triangular_solve(b, a, out=(out, clone_a))

        # Trigger the WARN_ONCE deprecation error
        torch.triangular_solve(b, a)

        # if out tensor with wrong shape is passed a warning is given
        with warnings.catch_warnings(record=True) as w:
            out = torch.empty(1, dtype=dtype, device=device)
            clone_a = torch.empty(1, dtype=dtype, device=device)
            # Trigger warning
            torch.triangular_solve(b, a, out=(out, clone_a))
            # Check warning occurs
            self.assertEqual(len(w), 2)
            self.assertTrue("An output with one or more elements was resized" in str(w[0].message))
            self.assertTrue("An output with one or more elements was resized" in str(w[1].message))


    def check_single_matmul(self, x, y):

        def assertEqual(answer, expected):
            if x.dtype.is_floating_point or x.dtype.is_complex:
                k = max(x.shape[-1], 1)  # Scale the atol with the size of the matrix
                self.assertEqual(answer, expected,
                                 msg=f"{x.shape} x {y.shape} = {answer.shape}",
                                 atol=k * 5e-5,
                                 rtol=1e-4)
            else:
                self.assertEqual(answer, expected, msg=f"{x.shape} x {y.shape} = {answer.shape}")

        # test x @ y
        expected = np.matmul(x.cpu(), y.cpu())
        ans = torch.matmul(x, y)
        self.assertTrue(ans.is_contiguous())
        assertEqual(ans, expected)

        # test out
        out = torch.empty_like(ans)
        ans = torch.matmul(x, y, out=out)
        self.assertIs(ans, out)
        self.assertTrue(ans.is_contiguous())
        assertEqual(ans, expected)

    def gen_sizes_matmul(self, x_dim, y_dim=4, matrix_size=4, batch_size=3):
        """
        Generates sequences of tuples (x, y) of with size(x) = x_dim and
        size(y) <= y_dim that are compatible wrt. matmul
        """
        assert x_dim >= 1
        assert y_dim >= 2
        x = x_dim
        for y in range(1, y_dim + 1):
            for batch, mn in product(product(range(batch_size), repeat=max(x - 2, y - 2, 0)),
                                     product(range(matrix_size), repeat=min(y, 2))):
                if x == 1:
                    size_x = mn[:1]
                    size_y = batch + mn
                    yield size_x, size_y
                else:
                    for k in range(matrix_size):
                        size_x = (k,) + mn[:1]
                        if x > 2:
                            size_x = batch[-(x - 2):] + size_x
                        size_y = mn
                        if y > 2:
                            size_y = batch[-(y - 2):] + size_y
                        yield size_x, size_y

    @dtypesIfCUDA(torch.float, torch.complex64)  # Integer matmul just supported on CPU
    @dtypes(torch.int64, torch.float, torch.complex64)
    def test_matmul_small_brute_force_1d_Nd(self, device, dtype):
        make_arg = partial(make_tensor, device=device, dtype=dtype)

        for (size_x, size_y), nctg_x, nctg_y in product(self.gen_sizes_matmul(1), (True, False), (True, False)):
            x = make_arg(size_x, noncontiguous=nctg_x)
            y = make_arg(size_y, noncontiguous=nctg_y)
            self.check_single_matmul(x, y)

    @dtypesIfCUDA(torch.float, torch.complex64)  # Integer matmul just supported on CPU
    @dtypes(torch.int64, torch.float, torch.complex64)
    def test_matmul_small_brute_force_2d_Nd(self, device, dtype):
        make_arg = partial(make_tensor, device=device, dtype=dtype)

        for (size_x, size_y), nctg_x, nctg_y in product(self.gen_sizes_matmul(2), (True, False), (True, False)):
            x = make_arg(size_x, noncontiguous=nctg_x)
            y = make_arg(size_y, noncontiguous=nctg_y)
            self.check_single_matmul(x, y)

    @dtypesIfCUDA(torch.float, torch.complex64)  # Integer matmul just supported on CPU
    @dtypes(torch.int64, torch.float, torch.complex64)
    def test_matmul_small_brute_force_3d_Nd(self, device, dtype):
        make_arg = partial(make_tensor, device=device, dtype=dtype)

        for (size_x, size_y), nctg_x, nctg_y in product(self.gen_sizes_matmul(3), (True, False), (True, False)):
            x = make_arg(size_x, noncontiguous=nctg_x)
            y = make_arg(size_y, noncontiguous=nctg_y)
            self.check_single_matmul(x, y)

    # 4GB should do, but we run tests in parallel in CI, so let's be generous
    @largeTensorTest('16GB', device='cuda')
    def test_large_bmm_mm_backward(self, device):
        A = torch.randn([1024, 2, 1024], device="cuda").mT.contiguous().mT
        B = torch.randn([1024, 65536], device="cuda", requires_grad=True)
        G = torch.randn([1024, 2, 65536], device="cuda")

        # Should not create an intermediary tensor of size [1024, 1024, 65536] (256GB of memory) and OOM
        (A @ B).backward(G)

    # 4GB should do, but we run tests in parallel in CI, so let's be generous
    @largeTensorTest('16GB', device='cuda')
    def test_large_bmm_backward(self, device):
        A = torch.randn([1024, 2, 1024], device="cuda").mT.contiguous().mT
        B = torch.randn([1, 1024, 65536], device="cuda", requires_grad=True)
        G = torch.randn([1024, 2, 65536], device="cuda")

        # Should not create an intermediary tensor of size [1024, 1024, 65536] (256GB of memory) and OOM
        (A @ B).backward(G)

    def test_linear_algebra_scalar_raises(self, device) -> None:
        m = torch.randn(5, 5, device=device)
        v = torch.randn(5, device=device)
        s = torch.tensor(7, device=device)
        self.assertRaises(RuntimeError, lambda: torch.mv(m, s))
        self.assertRaises(RuntimeError, lambda: torch.addmv(v, m, s))

    @dtypes(torch.float32, torch.complex64)
    def test_cross(self, device, dtype):
        x = torch.rand(100, 3, 100, dtype=dtype, device=device)
        y = torch.rand(100, 3, 100, dtype=dtype, device=device)
        res1 = torch.cross(x, y)
        res2 = torch.tensor((), dtype=dtype, device=device)
        torch.cross(x, y, out=res2)
        self.assertEqual(res1, res2)

    @dtypes(torch.float32, torch.complex64)
    def test_linalg_cross(self, device, dtype):
        x = torch.rand(100, 3, 100, dtype=dtype, device=device)
        y = torch.rand(100, 3, 100, dtype=dtype, device=device)
        res1 = torch.linalg.cross(x, y, dim=1)
        res2 = torch.tensor((), dtype=dtype, device=device)
        torch.linalg.cross(x, y, dim=1, out=res2)
        self.assertEqual(res1, res2)

        # test for broadcastable inputs
        x = torch.rand(1, 3, 2, dtype=dtype, device=device)
        y = torch.rand(4, 3, 1, dtype=dtype, device=device)
        res1 = torch.linalg.cross(x, y, dim=1)
        res2 = torch.tensor((), dtype=dtype, device=device)
        torch.linalg.cross(x, y, dim=1, out=res2)
        self.assertEqual(res1, res2)

    @dtypes(torch.float32, torch.complex64)
    def test_cross_with_and_without_dim(self, device, dtype):
        x = torch.rand(100, 3, dtype=dtype, device=device)
        y = torch.rand(100, 3, dtype=dtype, device=device)
        res1 = torch.cross(x, y, dim=1)
        res2 = torch.cross(x, y, dim=-1)
        res3 = torch.cross(x, y)
        self.assertEqual(res1, res2)
        self.assertEqual(res1, res3)

    @dtypes(torch.float32, torch.complex64)
    def test_linalg_cross_with_and_without_dim(self, device, dtype):
        x = torch.rand(100, 3, dtype=dtype, device=device)
        y = torch.rand(100, 3, dtype=dtype, device=device)
        res1 = torch.linalg.cross(x, y, dim=1)
        res2 = torch.linalg.cross(x, y, dim=-1)
        res3 = torch.linalg.cross(x, y)
        self.assertEqual(res1, res2)
        self.assertEqual(res1, res3)

    def test_renorm(self, device):
        m1 = torch.randn(20, 20, device=device)  # big enough to exercise vectorized path
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

    @skipCPUIfNoLapack
    @skipCUDAIfNoCusolver
    @dtypes(*floating_and_complex_types())
    def test_ormqr(self, device, dtype):

        def run_test(batch, m, n, fortran_contiguous):
            A = make_tensor((*batch, m, n), dtype=dtype, device=device)
            reflectors, tau = torch.geqrf(A)
            if not fortran_contiguous:
                self.assertTrue(reflectors.mT.is_contiguous())
                reflectors = reflectors.contiguous()

            # Q is of size m x m
            Q, _ = torch.linalg.qr(A, mode='complete')
            C_right = make_tensor((*batch, m, n), dtype=dtype, device=device)
            C_left = make_tensor((*batch, n, m), dtype=dtype, device=device)

            expected = Q @ C_right
            actual = torch.ormqr(reflectors, tau, C_right, left=True, transpose=False)
            self.assertEqual(expected, actual)

            expected = C_left @ Q
            actual = torch.ormqr(reflectors, tau, C_left, left=False, transpose=False)
            self.assertEqual(expected, actual)

            expected = Q.mH @ C_right
            actual = torch.ormqr(reflectors, tau, C_right, left=True, transpose=True)
            self.assertEqual(expected, actual)

            expected = C_left @ Q.mH
            actual = torch.ormqr(reflectors, tau, C_left, left=False, transpose=True)
            self.assertEqual(expected, actual)

            # if tau is all zeros then the implicit matrix Q is the identity matrix
            # so the actual result should be C_right in this case
            zero_tau = torch.zeros_like(tau)
            actual = torch.ormqr(reflectors, zero_tau, C_right, left=True, transpose=False)
            self.assertEqual(C_right, actual)

        batches = [(), (0, ), (2, ), (2, 1)]
        ns = [5, 2, 0]
        for batch, (m, n), fortran_contiguous in product(batches, product(ns, ns), [True, False]):
            run_test(batch, m, n, fortran_contiguous)

    @skipCPUIfNoLapack
    @skipCUDAIfNoCusolver
    @dtypes(*floating_and_complex_types())
    def test_ormqr_errors_and_warnings(self, device, dtype):
        test_cases = [
            # input1 size, input2 size, input3 size, error regex
            ((10,), (2,), (2,), r"input must have at least 2 dimensions"),
            ((2, 2), (2,), (2,), r"other must have at least 2 dimensions"),
            ((10, 6), (20,), (10, 6), r"other.shape\[-2\] must be greater than or equal to tau.shape\[-1\]"),
            ((6, 6), (5,), (5, 5), r"other.shape\[-2\] must be equal to input.shape\[-2\]"),
            ((1, 2, 2), (2, 2), (1, 2, 2), r"batch dimensions of tau to be equal to input.shape\[:-2\]"),
            ((1, 2, 2), (1, 2), (2, 2, 2), r"batch dimensions of other to be equal to input.shape\[:-2\]"),
        ]
        for a_size, tau_size, c_size, error_regex in test_cases:
            a = make_tensor(a_size, dtype=dtype, device=device)
            tau = make_tensor(tau_size, dtype=dtype, device=device)
            c = make_tensor(c_size, dtype=dtype, device=device)
            with self.assertRaisesRegex(RuntimeError, error_regex):
                torch.ormqr(a, tau, c)

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

    @precisionOverride({torch.double: 1e-8, torch.float: 1e-4, torch.bfloat16: 0.6,
                        torch.half: 1e-1, torch.cfloat: 1e-4, torch.cdouble: 1e-8})
    @dtypesIfCUDA(*floating_and_complex_types_and(
                  torch.half,
                  *[torch.bfloat16] if SM53OrLater else []
                  ))
    @dtypes(*all_types_and_complex_and(torch.bfloat16))
    def test_corner_cases_of_cublasltmatmul(self, device, dtype):
        # common case
        M = torch.randn(128, device=device).to(dtype)
        m1 = torch.randn(2048, 2400, device=device).to(dtype)
        m2 = torch.randn(128, 2400, device=device).to(dtype)
        torch.nn.functional.linear(m1, m2, M)
        # Ntrans_B has ld >> rows
        m1 = torch.rand([128, 2400]).to(dtype).to(device).t()
        m2 = torch.rand([2048, 25272]).to(dtype).to(device).t()[21940:24340]
        M = torch.rand([128]).to(dtype).to(device)
        torch.addmm(M, m2.t(), m1)
        # trans_A has ld >> rows
        m1 = torch.rand([128, 25272]).to(dtype).to(device)[:, 21940:24340].t()
        m2 = torch.randn(2048, 2400, device=device).to(dtype)
        M = torch.rand([128]).to(dtype).to(device)
        torch.addmm(M, m2, m1)
        # large tensor dim > 65535
        M = torch.randn(16, device=device).to(dtype)
        m1 = torch.randn(32, 131071 , device=device).to(dtype)
        m2 = torch.randn(16, 131071, device=device).to(dtype)
        torch.nn.functional.linear(m1, m2, M)

    @dtypesIfCUDA(*floating_and_complex_types_and(
                  torch.half,
                  *[torch.bfloat16] if SM53OrLater else []
                  ))
    @dtypes(*all_types_and_complex_and(torch.bfloat16))
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

    @dtypes(*floating_and_complex_types_and(torch.half, torch.bfloat16))
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

    @skipCPUIfNoLapack
    @skipCUDAIfNoCusolver
    @dtypes(*floating_and_complex_types())
    def test_householder_product(self, device, dtype):
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
            actual = torch.linalg.householder_product(reflectors, tau)
            # torch.linalg.qr does not work correctly for zero batch dimension tensors
            # see https://github.com/pytorch/pytorch/issues/50576
            if (A.numel() > 0):
                self.assertEqual(expected, actual)
            else:
                self.assertTrue(actual.shape == shape)

            # if tau is empty and A is not the result should be a matrix with ones on the diagonal
            if (A.numel() > 0):
                tau_empty = torch.empty(*shape[:-2], 0, dtype=dtype, device=device)
                identity_mat = torch.zeros_like(reflectors)
                identity_mat.diagonal(dim1=-1, dim2=-2)[:] = 1
                actual = torch.linalg.householder_product(reflectors, tau_empty)
                self.assertEqual(actual, identity_mat)

            out = torch.empty_like(A)
            ans = torch.linalg.householder_product(reflectors, tau, out=out)
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

    @skipCPUIfNoLapack
    @skipCUDAIfNoCusolver
    def test_householder_product_errors_and_warnings(self, device):
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
                torch.linalg.householder_product(a, tau)

        # if out tensor with wrong shape is passed a warning is given
        reflectors = torch.randn(3, 3, device=device)
        tau = torch.randn(3, device=device)
        out = torch.empty(2, 3, device=device)
        with warnings.catch_warnings(record=True) as w:
            # Trigger warning
            torch.linalg.householder_product(reflectors, tau, out=out)
            # Check warning occurs
            self.assertEqual(len(w), 1)
            self.assertTrue("An output with one or more elements was resized" in str(w[-1].message))

        # dtypes should be safely castable
        out = torch.empty_like(reflectors).to(torch.int)
        with self.assertRaisesRegex(RuntimeError, "but got result with dtype Int"):
            torch.linalg.householder_product(reflectors, tau, out=out)

        with self.assertRaisesRegex(RuntimeError, "tau dtype Int does not match input dtype"):
            torch.linalg.householder_product(reflectors, tau.to(torch.int))

        if torch.cuda.is_available():
            # device of out and input should match
            wrong_device = 'cpu' if self.device_type != 'cpu' else 'cuda'
            out = torch.empty_like(reflectors).to(wrong_device)
            with self.assertRaisesRegex(RuntimeError, "Expected all tensors to be on the same device"):
                torch.linalg.householder_product(reflectors, tau, out=out)

            # device of tau and input should match
            wrong_device = 'cpu' if self.device_type != 'cpu' else 'cuda'
            tau = tau.to(wrong_device)
            with self.assertRaisesRegex(RuntimeError, "Expected all tensors to be on the same device"):
                torch.linalg.householder_product(reflectors, tau)

    @precisionOverride({torch.float32: 1e-2, torch.complex64: 1e-2})
    @skipCUDAIfNoMagmaAndNoCusolver
    @skipCPUIfNoLapack
    @dtypes(*floating_and_complex_types())
    def test_linalg_lu_family(self, device, dtype):
        # Tests torch.lu
        #       torch.linalg.lu_factor
        #       torch.linalg.lu_factor_ex
        #       torch.lu_unpack
        #       torch.linalg.lu_solve
        #       torch.linalg.solve
        make_arg_full = partial(make_fullrank_matrices_with_distinct_singular_values, device=device, dtype=dtype)
        make_arg = partial(make_tensor, device=device, dtype=dtype)

        def run_test(A, pivot, singular, fn):
            k = min(A.shape[-2:])
            batch = A.shape[:-2]
            check_errors = (fn == torch.linalg.lu_factor)
            if singular and check_errors:
                # It may or may not throw as the LU decomposition without pivoting
                # may still succeed for singular matrices
                try:
                    LU, pivots = fn(A, pivot=pivot)
                except RuntimeError:
                    return
            else:
                LU, pivots = fn(A, pivot=pivot)[:2]

            self.assertEqual(LU.size(), A.shape)
            self.assertEqual(pivots.size(), batch + (k,))

            if not pivot:
                self.assertEqual(pivots, torch.arange(1, 1 + k, device=device, dtype=torch.int32).expand(batch + (k, )))

            P, L, U = torch.lu_unpack(LU, pivots, unpack_pivots=pivot)

            self.assertEqual(P @ L @ U if pivot else L @ U, A)

            PLU = torch.linalg.lu(A, pivot=pivot)
            self.assertEqual(P, PLU.P)
            self.assertEqual(L, PLU.L)
            self.assertEqual(U, PLU.U)

            if not singular and A.size(-2) == A.size(-1):
                nrhs = ((), (1,), (3,))
                for left, rhs in product((True, False), nrhs):
                    # Vector case when left = False is not allowed
                    if not left and rhs == ():
                        continue
                    if left:
                        shape_B = A.shape[:-1] + rhs
                    else:
                        shape_B = A.shape[:-2] + rhs + A.shape[-1:]
                    B = make_arg(shape_B)

                    # Test linalg.lu_solve. It does not support vectors as rhs
                    # See https://github.com/pytorch/pytorch/pull/74045#issuecomment-1112304913
                    if rhs != ():
                        for adjoint in (True, False):
                            X = torch.linalg.lu_solve(LU, pivots, B, left=left, adjoint=adjoint)
                            A_adj = A.mH if adjoint else A
                            if left:
                                self.assertEqual(B, A_adj @ X)
                            else:
                                self.assertEqual(B, X @ A_adj)

                    # Test linalg.solve
                    X = torch.linalg.solve(A, B, left=left)
                    X_ = X.unsqueeze(-1) if rhs == () else X
                    B_ = B.unsqueeze(-1) if rhs == () else B
                    if left:
                        self.assertEqual(B_, A @ X_)
                    else:
                        self.assertEqual(B_, X_ @ A)


        sizes = ((3, 3), (5, 5), (4, 2), (3, 4), (0, 0), (0, 1), (1, 0))
        batches = ((0,), (), (1,), (2,), (3,), (1, 0), (3, 5))
        # Non pivoting just implemented for CUDA
        pivots = (True, False) if self.device_type == "cuda" else (True,)
        fns = (partial(torch.lu, get_infos=True), torch.linalg.lu_factor, torch.linalg.lu_factor_ex)
        for ms, batch, pivot, singular, fn in itertools.product(sizes, batches, pivots, (True, False), fns):
            shape = batch + ms
            A = make_arg(shape) if singular else make_arg_full(*shape)
            # Just do one of them on singular matrices
            if A.numel() == 0 and not singular:
                continue
            run_test(A, pivot, singular, fn)

            # Reproducer of a magma bug,
            # see https://bitbucket.org/icl/magma/issues/13/getrf_batched-kernel-produces-nans-on
            # This is also a bug in cuSOLVER < 11.3
            if (dtype == torch.double
               and singular
               and (torch.version.cuda is None or
                    torch.version.cuda.split('.') >= ["11", "3"])):
                A = torch.ones(batch + ms, dtype=dtype, device=device)
                run_test(A, pivot, singular, fn)

        # Info should be positive for rank deficient matrices
        A = torch.ones(5, 3, 3, device=device)
        self.assertTrue((torch.linalg.lu_factor_ex(A, pivot=True).info >= 0).all())

        if self.device_type == 'cpu':
            # Error checking, no pivoting variant on CPU
            fns = [torch.lu, torch.linalg.lu_factor, torch.linalg.lu_factor_ex, torch.linalg.lu]
            for f in fns:
                with self.assertRaisesRegex(RuntimeError, 'LU without pivoting is not implemented on the CPU'):
                    f(torch.empty(1, 2, 2), pivot=False)


    @precisionOverride({torch.float32: 1e-2, torch.complex64: 1e-2})
    @skipCUDAIfNoMagmaAndNoCusolver
    @skipCPUIfNoLapack
    @setLinalgBackendsToDefaultFinally
    @dtypes(*floating_and_complex_types())
    def test_linalg_lu_solve(self, device, dtype):
        make_arg = partial(make_tensor, dtype=dtype, device=device)

        backends = ["default"]

        if torch.device(device).type == 'cuda':
            if torch.cuda.has_magma:
                backends.append("magma")
            if has_cusolver():
                backends.append("cusolver")

        def gen_matrices():
            rhs = 3
            ns = (5, 2, 0)
            batches = ((), (0,), (1,), (2,), (2, 1), (0, 2))
            for batch, n in product(batches, ns):
                yield make_arg(batch + (n, n)), make_arg(batch + (n, rhs))
            # Shapes to exercise all the paths
            shapes = ((1, 64), (2, 128), (1025, 2))
            for b, n in shapes:
                yield make_arg((b, n, n)), make_arg((b, n, rhs))


        for A, B in gen_matrices():
            LU, pivots = torch.linalg.lu_factor(A)
            for backend in backends:
                torch.backends.cuda.preferred_linalg_library(backend)

                for left, adjoint in product((True, False), repeat=2):
                    B_left = B if left else B.mT
                    X = torch.linalg.lu_solve(LU, pivots, B_left, left=left, adjoint=adjoint)
                    A_adj = A.mH if adjoint else A
                    if left:
                        self.assertEqual(B_left, A_adj @ X)
                    else:
                        self.assertEqual(B_left, X @ A_adj)


    @onlyCPU
    @dtypes(*floating_and_complex_types())
    def test_linalg_lu_cpu_errors(self, device, dtype):
        # Square tests
        sample = torch.randn(3, 2, 2, device=device, dtype=dtype)
        B = torch.randn(3, 2, 2, device=device, dtype=dtype)
        LU, pivots = torch.linalg.lu_factor(sample)

        # This should run without issues
        torch.linalg.lu_solve(LU, pivots, B, adjoint=True)
        torch.lu_unpack(LU, pivots)

        pivots[0] = 0
        with self.assertRaisesRegex(RuntimeError, r"greater or equal to 1"):
            torch.linalg.lu_solve(LU, pivots, B, adjoint=True)
        with self.assertRaisesRegex(RuntimeError, r"between 1 and LU.size\(-2\)."):
            torch.lu_unpack(LU, pivots)

        pivots[0] = 3
        with self.assertRaisesRegex(RuntimeError, r"smaller or equal to LU.size\(-2\)"):
            torch.linalg.lu_solve(LU, pivots, B, adjoint=True)
        with self.assertRaisesRegex(RuntimeError, r"between 1 and LU.size\(-2\)."):
            torch.lu_unpack(LU, pivots)

        # Rectangular tests
        sample = torch.randn(3, 4, 2, device=device, dtype=dtype)
        B = torch.randn(3, 4, 2, device=device, dtype=dtype)
        LU, pivots = torch.linalg.lu_factor(sample)

        # This should run without issues
        torch.lu_unpack(LU, pivots)

        pivots[0] = 0
        with self.assertRaisesRegex(RuntimeError, r"between 1 and LU.size\(-2\)."):
            torch.lu_unpack(LU, pivots)

        pivots[0] = 5
        with self.assertRaisesRegex(RuntimeError, r"between 1 and LU.size\(-2\)."):
            torch.lu_unpack(LU, pivots)


        # Rectangular tests
        sample = torch.randn(2, 3, 5, device=device, dtype=dtype)
        B = torch.randn(2, 3, 5, device=device, dtype=dtype)
        LU, pivots = torch.linalg.lu_factor(sample)

        # This should run without issues
        torch.lu_unpack(LU, pivots)

        pivots[0] = 0
        with self.assertRaisesRegex(RuntimeError, r"between 1 and LU.size\(-2\)."):
            torch.lu_unpack(LU, pivots)

        pivots[0] = 4
        with self.assertRaisesRegex(RuntimeError, r"between 1 and LU.size\(-2\)."):
            torch.lu_unpack(LU, pivots)


    @skipCPUIfNoLapack
    @skipCUDAIfNoMagma
    @dtypes(torch.double)
    def test_lu_unpack_check_input(self, device, dtype):
        x = torch.rand(5, 5, 5, device=device, dtype=dtype)
        lu_data, lu_pivots = torch.linalg.lu_factor(x)

        with self.assertRaisesRegex(RuntimeError, "torch.int32 dtype"):
            torch.lu_unpack(lu_data, lu_pivots.long())

        # check that onces flags are unset, Nones are returned
        p, l, u = torch.lu_unpack(lu_data, lu_pivots, unpack_data=False)
        self.assertTrue(l.numel() == 0 and u.numel() == 0)
        p, l, u = torch.lu_unpack(lu_data, lu_pivots, unpack_pivots=False)
        self.assertTrue(p.numel() == 0)
        p, l, u = torch.lu_unpack(lu_data, lu_pivots, unpack_data=False, unpack_pivots=False)
        self.assertTrue(p.numel() == 0 and l.numel() == 0 and u.numel() == 0)

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(torch.double)
    def test_lobpcg_basic(self, device, dtype):
        self._test_lobpcg_method(device, dtype, 'basic')

    @skipCUDAIfNoCusolver
    @skipCPUIfNoLapack
    @dtypes(torch.double)
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
                e = torch.linalg.eigvalsh(A)
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

    def _test_addmm_addmv(self, f, t, m, v, *, alpha=None, beta=None, transpose_out=False, activation=None):
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
        if activation == "relu":
            res3 = res3 * (res3 > 0)
        else:
            assert activation is None, f"unsupported activation {activation}"
        res3 = torch.from_numpy(res3).to(dtype)
        self.assertEqual(res1, res2)
        self.assertEqual(res1, res3)

    @precisionOverride({torch.bfloat16: 1e-0, torch.half: 5e-4, torch.float: 1e-4, torch.double: 1e-8,
                        torch.cfloat: 1e-4, torch.cdouble: 1e-8})
    @dtypesIfCUDA(*floating_and_complex_types_and(
                  *[torch.bfloat16] if TEST_WITH_ROCM or SM53OrLater else [],
                  torch.half))
    @dtypes(torch.bfloat16, torch.float, torch.double, torch.cfloat, torch.cdouble)
    def test_addmv(self, device, dtype):
        # have to use torch.randn(...).to(bfloat16) instead of
        # torch.randn(..., dtype=bfloat16). randn does not support
        # bfloat16 yet.
        # "*0.2" to reduce errors for low precision
        ts = [
            0.2 * torch.randn(50, device=device).to(dtype),
            0.2 * torch.randn(1, device=device).to(dtype).expand(50),
        ]
        vs = [
            0.2 * torch.randn(100, device=device).to(dtype),
            0.2 * torch.ones(1, device=device).to(dtype).expand(100),  # to reduce errors for low precision
        ]
        ms = [
            # 0d
            0.2 * torch.ones((), device=device).to(dtype).expand(50, 100),  # to reduce errors for low precision
            # 1d
            0.2 * torch.randn((1, 100), device=device).to(dtype).expand(50, 100),
            # this initialization reduces errors for low precision for broadcasted matrices
            # by making sure that intermediate and result values are exactly representable
            # in low precision type
            0.2 * torch.randint(3, (50, 1), dtype=torch.float, device=device).to(dtype).expand(50, 100),
            # 2d
            0.2 * torch.randn((50, 100), device=device).to(dtype),
            0.2 * torch.randn((100, 50), device=device).to(dtype).t(),
        ]
        for m, v, t in itertools.product(ms, vs, ts):
            self._test_addmm_addmv(torch.addmv, t, m, v)
        # Test beta=0, t=nan
        t = torch.full((50,), math.nan, device=device).to(dtype)
        for m, v in itertools.product(ms, vs):
            self._test_addmm_addmv(torch.addmv, t, m, v, beta=0)

    @dtypesIfCUDA(*floating_types_and(*[torch.bfloat16] if TEST_WITH_ROCM or
                  SM53OrLater else []))
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

    def _test_addmm_impl(self, func, activation, device, dtype):
        M = torch.randn(10, 25, device=device).to(dtype)
        m1 = torch.randn(10, 50, device=device).to(dtype)
        m2 = torch.randn(50, 25, device=device).to(dtype)
        self._test_addmm_addmv(func, M, m1, m2, activation=activation)

        # Test 0-strided
        M = torch.randn(10, 1, device=device).to(dtype).expand(10, 25)
        m1 = torch.randn(10, 1, device=device).to(dtype).expand(10, 50)
        m2 = torch.randn(50, 25, device=device).to(dtype)
        self._test_addmm_addmv(func, M, m1, m2, activation=activation)

        # Test beta=0, M=nan
        M = torch.full((10, 25), math.nan, device=device).to(dtype)
        m1 = torch.randn(10, 50, device=device).to(dtype)
        m2 = torch.randn(50, 25, device=device).to(dtype)
        self._test_addmm_addmv(func, M, m1, m2, beta=0, activation=activation)

        # Test transpose
        for t1, t2, t3, t4 in itertools.product([True, False], repeat=4):
            def maybe_transpose(cond, m):
                if not cond:
                    return m
                return m.t().clone(memory_format=torch.contiguous_format).t()

            M = maybe_transpose(t1, torch.randn(10, 25, device=device).to(dtype))
            m1 = maybe_transpose(t2, torch.randn(10, 50, device=device).to(dtype))
            m2 = maybe_transpose(t3, torch.randn(50, 25, device=device).to(dtype))
            self._test_addmm_addmv(func, M, m1, m2, transpose_out=t4, activation=activation)

    @precisionOverride({torch.double: 1e-8, torch.float: 1e-4, torch.bfloat16: 0.6,
                        torch.half: 1e-1, torch.cfloat: 1e-4, torch.cdouble: 1e-8})
    @dtypesIfMPS(torch.float32)
    @dtypesIfCUDA(*floating_and_complex_types_and(
                  *[torch.bfloat16] if TEST_WITH_ROCM or SM53OrLater else []))
    @dtypes(*floating_and_complex_types_and(torch.bfloat16))
    @tf32_on_and_off(0.05)
    def test_addmm(self, device, dtype):
        self._test_addmm_impl(torch.addmm, None, device, dtype)

    @precisionOverride({torch.double: 1e-8, torch.float: 1e-4, torch.bfloat16: 0.6,
                        torch.half: 1e-1, torch.cfloat: 1e-4, torch.cdouble: 1e-8})
    @dtypesIfCUDA(*floating_types_and(
                  *[torch.bfloat16] if TEST_WITH_ROCM or SM53OrLater else []))
    @dtypes(*floating_types_and(torch.bfloat16))
    @tf32_on_and_off(0.05)
    def test_addmm_activation(self, device, dtype):
        self._test_addmm_impl(torch._addmm_activation, "relu", device, dtype)

    @dtypes(torch.float, torch.double)
    @dtypesIfCUDA(*floating_and_complex_types())
    @tf32_on_and_off(0.005)
    def test_addmm_sizes(self, device, dtype):
        for m in [0, 1, 25]:
            for n in [0, 1, 10]:
                for k in [0, 1, 8]:
                    M = torch.randn(n, m, device=device).to(dtype)
                    m1 = torch.randn(n, k, device=device).to(dtype)
                    m2 = torch.randn(k, m, device=device).to(dtype)
                    self._test_addmm_addmv(torch.addmm, M, m1, m2)

                    m1 = torch.randn(n, k + 1, device=device).to(dtype)
                    m2 = torch.randn(k, m, device=device).to(dtype)
                    self.assertRaisesRegex(RuntimeError, f"{n}x{k + 1}.*{k}x{m}", lambda: torch.addmm(M, m1, m2))
                    self.assertRaisesRegex(RuntimeError, f"{n}x{k + 1}.*{k}x{m}", lambda: torch.mm(m1, m2))

    @dtypes(torch.half)
    @onlyCUDA
    def test_addmm_baddbmm_overflow(self, device, dtype):
        orig = torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        inp = torch.zeros(128, 128, dtype=torch.half, device=device)
        mat1 = torch.ones(128, 1000, dtype=torch.half, device=device) * 100
        mat2 = torch.ones(1000, 128, dtype=torch.half, device=device) * 100
        out = torch.addmm(inp, mat1, mat2, alpha=0.001, beta=0.)
        # just check for no overflow on ROCM
        if TEST_WITH_ROCM:
            self.assertFalse(out.isinf().any())
        else:
            self.assertTrue((out == 10000.).all())
        inp = torch.zeros(3, 128, 128, dtype=torch.half, device=device)
        mat1 = torch.ones(3, 128, 1000, dtype=torch.half, device=device) * 100
        mat2 = torch.ones(3, 1000, 128, dtype=torch.half, device=device) * 100
        out = torch.baddbmm(inp, mat1, mat2, alpha=0.001, beta=0.)
        if TEST_WITH_ROCM:
            self.assertFalse(out.isinf().any())
        else:
            self.assertTrue((out == 10000.).all())
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = orig

    @dtypes(torch.float)
    def test_baddbmm_nan_input_with_zero_beta(self, device, dtype):
        for shape in [[3, 2, 2], [2, 20, 20]]:
            mat1, mat2 = [torch.randn(shape, dtype=dtype, device=device) for _ in range(2)]
            inputs = [torch.randn(shape, dtype=dtype, device=device),
                      torch.randn(shape, dtype=dtype, device=device).fill_(torch.nan)]
            outs = [None, torch.randn(shape, dtype=dtype, device=device),
                    torch.randn(shape, dtype=dtype, device=device).fill_(torch.nan)]
            options = itertools.product(inputs, outs)
            for input, out in options:
                y_ref = torch.bmm(mat1, mat2)
                y = torch.baddbmm(input, mat1, mat2, beta=0.0, out=out)
                self.assertEqual(y_ref, y)


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

    @unittest.skipIf(IS_WINDOWS, "Skipped on Windows!")
    @unittest.skipIf(IS_FBCODE and IS_REMOTE_GPU, "cublas runtime error")
    @onlyCUDA
    @parametrize("k", [16, 32])
    @parametrize("n", [16, 32])
    @parametrize("use_transpose_a", [True, False])
    @parametrize("use_transpose_b", [True, False])
    def test__int_mm(self, device, k, n, use_transpose_a, use_transpose_b):
        if TEST_WITH_ROCM:
            self.skipTest("_int_mm not compiled for ROCM")

        def genf_int_float(x, y, use_transpose):
            if use_transpose:
                x, y = y, x
            x_int8 = torch.randint(-10, 10, (x, y), dtype=torch.int8, device=device)
            x_float = x_int8.to(torch.float32)
            if use_transpose:
                return x_int8.t(), x_float.t()
            return x_int8, x_float

        def _test(m, k, n, transpose_a, transpose_b, test_equal=True):
            a_int8, a_float = genf_int_float(m, k, transpose_a)
            b_int8, b_float = genf_int_float(k, n, transpose_b)
            c_int32 = torch._int_mm(a_int8, b_int8)
            self.assertTrue(c_int32.dtype is torch.int32)
            self.assertEqual(c_int32.device, torch.device(device))
            if test_equal:
                self.assertEqual(c_int32.float(), torch.mm(a_float, b_float))
            else:
                self.assertNotEqual(c_int32.float(), torch.mm(a_float, b_float))
            c_int32_result = c_int32.new_empty(c_int32.size())
            # Checking out variant
            torch._int_mm(a_int8, b_int8, out=c_int32_result)
            if test_equal:
                self.assertEqual(c_int32_result.float(), torch.mm(a_float, b_float))
            else:
                self.assertNotEqual(c_int32_result.float(), torch.mm(a_float, b_float))

        # NOTE: We're just exercising terrible failures here.
        version = _get_torch_cuda_version()
        SM86OrLater = torch.cuda.is_available() and torch.cuda.get_device_capability() >= (8, 6)
        if version == (11, 7):
            if not use_transpose_a and use_transpose_b:
                if SM86OrLater:
                    _test(17, k, n, use_transpose_a, use_transpose_b, False)
                else:
                    with self.assertRaisesRegex(RuntimeError,
                                                "CUDA error: CUBLAS_STATUS_NOT_SUPPORTED when calling cublasLtMatmul"):
                        _test(17, k, n, use_transpose_a, use_transpose_b, False)

            if use_transpose_a and not use_transpose_b:
                with self.assertRaisesRegex(RuntimeError,
                                            "CUDA error: CUBLAS_STATUS_NOT_SUPPORTED when calling cublasLtMatmul"):
                    _test(17, k, n, use_transpose_a, use_transpose_b)

            if use_transpose_a and use_transpose_b:
                with self.assertRaisesRegex(RuntimeError,
                                            "CUDA error: CUBLAS_STATUS_NOT_SUPPORTED when calling cublasLtMatmul"):
                    _test(17, k, n, use_transpose_a, use_transpose_b)

            if not use_transpose_a and not use_transpose_b:
                if SM86OrLater:
                    _test(17, k, n, use_transpose_a, use_transpose_b)
                else:
                    with self.assertRaisesRegex(RuntimeError,
                                                "CUDA error: CUBLAS_STATUS_NOT_SUPPORTED when calling cublasLtMatmul"):
                        _test(17, k, n, use_transpose_a, use_transpose_b)
        else:
            with self.assertRaisesRegex(RuntimeError, "_int_mm_out_cuda not compiled for CUDA"):
                _test(17, k, n, use_transpose_a, use_transpose_b, False)

    @unittest.skipIf(IS_WINDOWS, "Skipped on Windows!")
    @unittest.skipIf(IS_FBCODE and IS_REMOTE_GPU, "cublas runtime error")
    @onlyCUDA
    def test__int_mm_errors(self, device):
        if TEST_WITH_ROCM:
            self.skipTest("_int_mm not compiled for ROCM")

        version = _get_torch_cuda_version()
        if version != (11, 7):
            self.skipTest("_int_mm only compiled for CUDA 11.7")

        def genf_int(x, y):
            return torch.empty((x, y), dtype=torch.int8, device=device)

        def _gen_pair(m, k, n):
            return genf_int(m, k), genf_int(k, n)

        self.assertRaisesRegex(RuntimeError,
                               r"self.size\(0\) needs to be greater than 16, but got 16",
                               lambda: torch._int_mm(*_gen_pair(16, 8, 32)))
        self.assertRaisesRegex(RuntimeError,
                               r"self.size\(1\) needs to be greater than 0 and a multiple of 8, but got 7",
                               lambda: torch._int_mm(*_gen_pair(17, 7, 32)))
        self.assertRaisesRegex(RuntimeError,
                               r"self.size\(1\) needs to match mat2.size\(0\) but got 8 and 7",
                               lambda: torch._int_mm(genf_int(17, 8), genf_int(7, 32)))
        self.assertRaisesRegex(RuntimeError,
                               r"mat2.size\(1\) needs to be greater than 0 and a multiple of 8, but got 31",
                               lambda: torch._int_mm(*_gen_pair(17, 8, 31)))
        self.assertRaisesRegex(RuntimeError,
                               r"expected scalar type Char but found Float",
                               lambda: torch._int_mm(genf_int(17, 8).float(), genf_int(8, 32)))
        self.assertRaisesRegex(RuntimeError,
                               r"expected scalar type Char but found Float",
                               lambda: torch._int_mm(genf_int(17, 8), genf_int(8, 32).float()))
        self.assertRaisesRegex(RuntimeError,
                               r"Expected result dtype to be of type kInt but got float",
                               lambda: torch._int_mm(genf_int(17, 8), genf_int(8, 32), out=genf_int(16, 32).float()))
        self.assertRaisesRegex(RuntimeError,
                               r"Expected result.size\(0\) to be 17 but got 15",
                               lambda: torch._int_mm(genf_int(17, 8), genf_int(8, 32), out=genf_int(15, 32).int()))
        self.assertRaisesRegex(RuntimeError,
                               r"Expected result.size\(0\) to be 17 but got 16",
                               lambda: torch._int_mm(genf_int(17, 8), genf_int(8, 32), out=genf_int(16, 31).int()))

    @slowTest
    @onlyNativeDeviceTypes
    # bfloat16 doesn't have sufficient precision to pass this test
    @dtypes(torch.float32, torch.float64, torch.int32, torch.int64, torch.cfloat, torch.cdouble)
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
            return torch.randn(x, y, dtype=torch.float32, device=device).to(dtype) * 0.1

        def genf_float(x, y):
            return torch.randn(x, y, dtype=dtype, device=device)

        for (n, m, p) in [(20, 10, 15), (15, 20, 10), (25, 18, 10)]:
            if (dtype == torch.int32) or (dtype == torch.int64):
                genf = genf_int
            elif (dtype == torch.bfloat16):
                genf = genf_bfloat
            else:
                genf = genf_float

            _test_mm(n, m, p, dtype, genf)

    @onlyNativeDeviceTypes
    def test_mm_bmm_non_memory_dense(self, device):
        def _slice(tensor, fn):
            return fn(tensor)[..., ::2]
        A = torch.randn(3, 6, dtype=torch.cfloat, device=device)
        B = torch.randn(3, 3, dtype=torch.cfloat, device=device)
        out = torch.empty(3, 3, device=device, dtype=torch.complex64).t()
        out1 = torch.empty(3, 3, device=device, dtype=torch.complex64).t()
        A_conj = _slice(A, torch.conj)
        A_conj_physical = _slice(A, torch.conj_physical)

        self.assertEqual(torch.mm(A_conj, B, out=out), torch.mm(A_conj_physical, B, out=out))
        self.assertEqual(torch.mm(A_conj.t(), B, out=out), torch.mm(A_conj_physical.t(), B, out=out))

        Ab = torch.randn(2, 3, 6, dtype=torch.cfloat, device=device)
        Bb = torch.randn(2, 3, 3, dtype=torch.cfloat, device=device)
        Bb_ = torch.randn(1, 3, 3, dtype=torch.cfloat, device=device).expand(2, 3, 3)
        out_b = torch.empty(2, 3, 3, device=device, dtype=torch.complex64).mT

        Ab_conj = _slice(Ab, torch.conj)
        Ab_conj_physical = _slice(Ab, torch.conj_physical)

        def t_b(tensor):
            return tensor.mT

        self.assertEqual(torch.bmm(Ab_conj, Bb, out=out_b), torch.bmm(Ab_conj_physical, Bb, out=out_b))
        self.assertEqual(torch.bmm(t_b(Ab_conj), Bb, out=out_b), torch.bmm(t_b(Ab_conj_physical), Bb, out=out_b))

        # test broadcasting
        self.assertEqual(torch.bmm(Ab_conj, Bb_, out=out_b), torch.bmm(Ab_conj_physical, Bb_, out=out_b))
        self.assertEqual(torch.bmm(t_b(Ab_conj), Bb_, out=out_b), torch.bmm(t_b(Ab_conj_physical), Bb_, out=out_b))

    @onlyNativeDeviceTypes
    def test_mm_empty_inputs_mixed_dtype_errors(self, device):
        a = torch.randint(0, 10, [1, 10], dtype=torch.int16, device=device)
        b = torch.randn(10, 20, dtype=torch.float32, device=device)
        with self.assertRaisesRegex(RuntimeError, "expected .* and .* to have the same dtype, but got:"):
            torch.mm(a, b)

    @onlyNativeDeviceTypes
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
    @onlyNativeDeviceTypes
    @dtypes(*floating_and_complex_types_and(torch.bfloat16))
    @tf32_on_and_off(0.05)
    def test_bmm(self, device, dtype):
        if self.device_type == 'cuda' and dtype is torch.bfloat16 and not SM53OrLater:
            # cuBLAS does not guarantee BFloat16 support on SM < 53.
            # So on PyTorch, we consider BFloat16 support on SM < 53 as
            # undefined bahavior
            return

        batch_sizes = [1, 10]
        M, N, O = 23, 15, 12
        numpy_dtype = dtype if dtype != torch.bfloat16 else torch.float32

        is_supported = True
        if dtype == torch.bfloat16 and self.device_type == 'cuda':
            is_supported = TEST_WITH_ROCM or SM53OrLater

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
                b1 = make_tensor((num_batches, M, N), dtype=dtype, device=device, low=-0.1, high=0.1)
                b2 = make_tensor((num_batches, N, O), dtype=dtype, device=device, low=-0.1, high=0.1)
                b1 = b1.permute(perm1).contiguous().permute(invert_perm(perm1))
                b2 = b2.permute(perm2).contiguous().permute(invert_perm(perm2))
                yield b1, b2
            # broadcasting tensors
            for b1, b2, b3, b4, b5, b6 in itertools.product((True, False), repeat=6):
                shape1 = (num_batches if b1 else 1, M if b2 else 1, N if b3 else 1)
                shape2 = (num_batches if b4 else 1, N if b5 else 1, O if b6 else 1)
                b1 = make_tensor(shape1, dtype=dtype, device=device, low=-0.1, high=0.1).expand(num_batches, M, N)
                b2 = make_tensor(shape2, dtype=dtype, device=device, low=-0.1, high=0.1).expand(num_batches, N, O)
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

    def _test_addbmm_baddbmm(self, func, b1, b2, ref, out_tensor):
        getattr(out_tensor, func + "_")(b1, b2)
        self.assertEqual(out_tensor, ref)
        res3 = out_tensor.clone()

        with self.assertWarnsOnceRegex(
                UserWarning, f"This overload of {func}_ is deprecated"):
            getattr(out_tensor, func + "_")(1, b1, b2)
        self.assertEqual(out_tensor, ref * 2),
        getattr(res3, func + "_")(b1, b2, beta=1)
        self.assertEqual(out_tensor, res3)

        with self.assertWarnsOnceRegex(
                UserWarning, f"This overload of {func}_ is deprecated"):
            getattr(out_tensor, func + "_")(1., .5, b1, b2)
        self.assertEqual(out_tensor, ref * 2.5)
        getattr(res3, func + "_")(b1, b2, beta=1., alpha=.5)
        self.assertEqual(out_tensor, res3)

        with self.assertWarnsOnceRegex(
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
    @onlyNativeDeviceTypes
    @dtypes(*floating_and_complex_types_and(torch.bfloat16))
    @tf32_on_and_off(0.05)
    def test_addbmm(self, device, dtype):
        if self.device_type == 'cuda' and dtype is torch.bfloat16 and not SM53OrLater:
            # cuBLAS does not guarantee BFloat16 support on SM < 53.
            # So on PyTorch, we consider BFloat16 support on SM < 53 as
            # undefined bahavior
            return

        num_batches = 2
        M, N, O = 16, 17, 18

        is_supported = True
        if dtype == torch.bfloat16:
            if self.device_type == 'cpu':
                self.precision = 1  # 43 vs 43.75
            else:
                is_supported = TEST_WITH_ROCM or SM53OrLater

        if not is_supported:
            b1 = make_tensor((num_batches, M, N), dtype=dtype, device=device, low=-1, high=1)
            b2 = make_tensor((num_batches, N, O), dtype=dtype, device=device, low=-1, high=1)
            t = make_tensor((M, O), dtype=dtype, device=device, low=-1, high=1)
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
                    b1 = make_tensor((num_batches, M, N), dtype=dtype, device=device, low=-1, high=1) * 0.1
                    b2 = make_tensor((num_batches, N, O), dtype=dtype, device=device, low=-1, high=1) * 0.1
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
                b1 = make_tensor(shape1, dtype=dtype, device=device, low=-1, high=1).expand(num_batches, M, N) * 0.1
                b2 = make_tensor(shape2, dtype=dtype, device=device, low=-1, high=1).expand(num_batches, N, O) * 0.1
                ref = torch.from_numpy(
                    b1.to(numpy_dtype).cpu().numpy() @ b2.to(numpy_dtype).cpu().numpy()
                ).to(device=device, dtype=dtype).sum(0)
                out_tensor = torch.zeros_like(ref)
                yield b1, b2, ref, out_tensor
            # zero-sized tensors
            for z1, z2, z3, z4 in itertools.product((True, False), repeat=4):
                shape1 = (num_batches if z1 else 0, M if z2 else 0, N if z3 else 0)
                shape2 = (num_batches if z1 else 0, N if z3 else 0, O if z4 else 0)
                b1 = make_tensor(shape1, dtype=dtype, device=device, low=-1, high=1) * 0.1
                b2 = make_tensor(shape2, dtype=dtype, device=device, low=-1, high=1) * 0.1
                ref = torch.from_numpy(
                    b1.to(numpy_dtype).cpu().numpy() @ b2.to(numpy_dtype).cpu().numpy()
                ).to(device=device, dtype=dtype).sum(0)
                out_tensor = torch.zeros_like(ref)
                yield b1, b2, ref, out_tensor

        for b1, b2, ref, out_tensor in generate_tensor():
            self._test_addbmm_baddbmm("addbmm", b1, b2, ref, out_tensor)

    @precisionOverride({torch.half: 0.1, torch.bfloat16: 0.5})
    @onlyNativeDeviceTypes
    @dtypes(*floating_and_complex_types_and(torch.bfloat16))
    @tf32_on_and_off(0.05)
    def test_baddbmm(self, device, dtype):
        if self.device_type == 'cuda' and dtype is torch.bfloat16 and not SM53OrLater:
            # cuBLAS does not guarantee BFloat16 support on SM < 53.
            # So on PyTorch, we consider BFloat16 support on SM < 53 as
            # undefined bahavior
            return

        num_batches = 10
        M, N, O = 12, 8, 50

        is_supported = True
        if dtype == torch.bfloat16 and self.device_type == 'cuda':
            is_supported = TEST_WITH_ROCM or SM53OrLater

        if not is_supported:
            b1 = make_tensor((num_batches, M, N), dtype=dtype, device=device, low=-1, high=1)
            b2 = make_tensor((num_batches, N, O), dtype=dtype, device=device, low=-1, high=1)
            t = make_tensor((num_batches, M, O), dtype=dtype, device=device, low=-1, high=1)
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
                b1 = make_tensor((num_batches, M, N), dtype=dtype, device=device, low=-1, high=1)
                b2 = make_tensor((num_batches, N, O), dtype=dtype, device=device, low=-1, high=1)
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
                b1 = make_tensor(shape1, dtype=dtype, device=device, low=-1, high=1).expand(num_batches, M, N)
                b2 = make_tensor(shape2, dtype=dtype, device=device, low=-1, high=1).expand(num_batches, N, O)
                ref = torch.from_numpy(
                    b1.to(numpy_dtype).cpu().numpy() @ b2.to(numpy_dtype).cpu().numpy()).to(device=device, dtype=dtype)
                out_tensor = torch.zeros_like(ref)
                yield b1, b2, ref, out_tensor
            # zero-sized tensors
            for z1, z2, z3, z4 in itertools.product((True, False), repeat=4):
                shape1 = (num_batches if z1 else 0, M if z2 else 0, N if z3 else 0)
                shape2 = (num_batches if z1 else 0, N if z3 else 0, O if z4 else 0)
                b1 = make_tensor(shape1, dtype=dtype, device=device, low=-2, high=2)
                b2 = make_tensor(shape2, dtype=dtype, device=device, low=-2, high=2)
                ref = torch.from_numpy(
                    b1.to(numpy_dtype).cpu().numpy() @ b2.to(numpy_dtype).cpu().numpy()).to(device=device, dtype=dtype)
                out_tensor = torch.zeros_like(ref)
                yield b1, b2, ref, out_tensor

        for b1, b2, ref, out_tensor in generate_tensor():
            self._test_addbmm_baddbmm("baddbmm", b1, b2, ref, out_tensor)

    @precisionOverride({torch.float32: 5e-3, torch.complex64: 1e-3})
    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(*floating_and_complex_types())
    def test_pinverse(self, device, dtype):
        make_fullrank = make_fullrank_matrices_with_distinct_singular_values
        make_arg = partial(make_fullrank, device=device, dtype=dtype)

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
            M = make_arg(*batchdims, matsize, matsize)
            self.assertEqual(torch.eye(matsize, dtype=dtype, device=device).expand(sizes), M.pinverse().matmul(M),
                             atol=1e-7, rtol=0, msg='pseudo-inverse for invertible matrix')

    @skipCPUIfNoLapack
    @skipCUDAIfNoMagmaAndNoCusolver
    @dtypes(torch.double, torch.cdouble)
    def test_matrix_power_non_negative(self, device, dtype):
        def check(*size):
            t = make_tensor(size, dtype=dtype, device=device)
            for n in range(8):
                res = torch.linalg.matrix_power(t, n)
                ref = np.linalg.matrix_power(t.cpu().numpy(), n)
                self.assertEqual(res.cpu(), torch.from_numpy(ref))

        check(0, 0)
        check(1, 1)
        check(5, 5)
        check(0, 3, 3)
        check(2, 3, 3)

    @skipCPUIfNoLapack
    @skipCUDAIfNoMagmaAndNoCusolver
    @dtypes(torch.double, torch.cdouble)
    def test_matrix_power_negative(self, device, dtype):
        make_fullrank = make_fullrank_matrices_with_distinct_singular_values
        make_arg = partial(make_fullrank, device=device, dtype=dtype)

        def check(*size):
            t = make_arg(*size)
            for n in range(-7, 0):
                res = torch.linalg.matrix_power(t, n)
                ref = np.linalg.matrix_power(t.cpu().numpy(), n)
                self.assertEqual(res.cpu(), torch.from_numpy(ref))

        check(0, 0)
        check(5, 5)
        check(2, 0, 0)
        check(0, 3, 3)
        check(2, 3, 3)
        check(2, 3, 5, 5)

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(torch.float, torch.complex64)
    def test_linalg_matrix_exp_utils(self, device, dtype):
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

        # Regression test for https://github.com/pytorch/pytorch/issues/94124
        with self.assertRaises(RuntimeError):
            x = torch.rand([], device=device, dtype=dtype)
            coeffs = torch.rand([2, 2], device=device, dtype=dtype)
            res = torch._compute_linear_combination(x, coeffs)

    @onlyCPU
    @skipCPUIfNoLapack
    @dtypes(torch.complex64)
    def test_linalg_matrix_exp_no_warnings(self, device, dtype):
        # this tests https://github.com/pytorch/pytorch/issues/80948
        with freeze_rng_state():
            torch.manual_seed(42)
            tens = 0.5 * torch.randn(10, 3, 3, dtype=dtype, device=device)
            tens = (0.5 * (tens.transpose(-1, -2) + tens))
            with warnings.catch_warnings(record=True) as w:
                tens.imag = torch.matrix_exp(tens.imag)
                self.assertFalse(len(w))

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(torch.float, torch.double, torch.complex64, torch.complex128)
    def test_linalg_matrix_exp_boundary_cases(self, device, dtype):
        expm = torch.linalg.matrix_exp

        with self.assertRaisesRegex(RuntimeError, "Expected a floating point or complex tensor"):
            expm(torch.randn(3, 3).type(torch.int))

        with self.assertRaisesRegex(RuntimeError, "must have at least 2 dimensions"):
            expm(torch.randn(3))

        with self.assertRaisesRegex(RuntimeError, "must be batches of square matrices"):
            expm(torch.randn(3, 2, 1))

        # check 1x1 matrices
        x = torch.randn(3, 3, 1, 1)
        self.assertEqual(expm(x), x.exp())

    @slowTest
    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(torch.float, torch.double, torch.cfloat, torch.cdouble)
    def test_linalg_matrix_exp_analytic(self, device, dtype):
        expm = torch.linalg.matrix_exp
        # check zero matrix
        x = torch.zeros(20, 20, dtype=dtype, device=device)
        self.assertTrue((expm(x) == torch.eye(20, 20, dtype=dtype, device=device)).all().item())

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
            mexp = expm(x)
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

                mexp = expm(x_normalized)
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
    def test_linalg_matrix_exp_batch(self, device, dtype):

        def run_test(*n):
            tensors_batch = torch.zeros(n, dtype=dtype, device=device)
            tensors_batch = tensors_batch.view(-1, n[-2], n[-1])

            num_matrices = tensors_batch.size(0)
            tensors_list = []
            for i in range(num_matrices):
                tensors_list.append(torch.randn(n[-2], n[-1], dtype=dtype, device=device))

            for i in range(num_matrices):
                tensors_batch[i, ...] = tensors_list[i]

            tensors_exp_map = (torch.linalg.matrix_exp(x) for x in tensors_list)
            tensors_exp_batch = torch.linalg.matrix_exp(tensors_batch)

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
    def test_linalg_matrix_exp_compare_with_taylor(self, device, dtype):

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

                mexp = torch.linalg.matrix_exp(x)
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

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(*floating_and_complex_types())
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
    @dtypes(*floating_and_complex_types())
    def test_slogdet_errors_and_warnings(self, device, dtype):
        # slogdet requires the input to be a square matrix or batch of square matrices
        a = torch.randn(2, 3, device=device, dtype=dtype)
        with self.assertRaisesRegex(RuntimeError, r'must be batches of square matrices'):
            torch.linalg.slogdet(a)

        # slogdet requires the input to be at least 2 dimensional tensor
        a = torch.randn(2, device=device, dtype=dtype)
        with self.assertRaisesRegex(RuntimeError, r'must have at least 2 dimensions'):
            torch.linalg.slogdet(a)

        a = torch.randn(2, 2, device=device, dtype=torch.bfloat16)
        with self.assertRaisesRegex(RuntimeError, r'Low precision dtypes not supported'):
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

        # device should match
        if torch.cuda.is_available():
            wrong_device = 'cpu' if self.device_type != 'cpu' else 'cuda'
            sign_out = torch.empty(0, device=wrong_device, dtype=dtype)
            logabsdet_out = torch.empty(0, device=wrong_device, dtype=real_dtype)
            with self.assertRaisesRegex(RuntimeError, "tensors to be on the same device"):
                torch.linalg.slogdet(a, out=(sign_out, logabsdet_out))

    @skipCUDAIf(torch.version.cuda is not None
                and torch.version.cuda.split(".") < ["11", "3"], "There's a bug in cuSOLVER < 11.3")
    # FIXME One of the backends of lu_factor fails in windows. I haven't investigated which or why
    # https://github.com/pytorch/pytorch/issues/75225
    @unittest.skipIf(IS_WINDOWS, "Skipped on Windows!")
    @skipCUDAIfNoCusolver
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
                             atol=1e-6, rtol=0, msg='{} (det)'.format(desc))

            # Test slogdet
            # Compare the overall value rather than individual parts because of
            # precision issues when det is near zero.
            self.assertEqual(sdet * logabsdet.exp(), target_sdet * target_logabsdet.exp(),
                             atol=1e-6, rtol=0, msg='{} (slogdet)'.format(desc))
            self.assertEqual(linalg_sdet * linalg_logabsdet.exp(), target_sdet * target_logabsdet.exp(),
                             atol=1e-6, rtol=0, msg='{} (linalg_slogdet)'.format(desc))

            # Test logdet
            # Compare logdet against our own pytorch slogdet because they should
            # be consistent, while it may behave slightly differently with other
            # slogdet implementations when det is near zero due to precision
            # issues.
            if sdet.item() < 0:
                self.assertTrue(logdet.item() != logdet.item(), '{} (logdet negative case)'.format(desc))
            else:
                self.assertEqual(logdet.exp(), target_logabsdet.exp(),
                                 atol=1e-6, rtol=0, msg='{} (logdet non-negative case)'.format(desc))

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
                for full_idx in itertools.product(*(list(range(x)) for x in batchdims)):
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
    @dtypes(*floating_and_complex_types())
    def test_cholesky_inverse(self, device, dtype):
        from torch.testing._internal.common_utils import random_hermitian_pd_matrix

        def run_test(shape, batch, upper, contiguous):
            A = random_hermitian_pd_matrix(shape, *batch, dtype=dtype, device=device)
            if A.numel() > 0 and not contiguous:
                A = A.mT
                self.assertFalse(A.is_contiguous())
            L = torch.linalg.cholesky(A)
            expected_inverse = torch.inverse(A)
            L = L.mH if upper else L
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
        out_t = out.mT.clone(memory_format=torch.contiguous_format)
        out = out_t.mT
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
    @dtypes(*floating_and_complex_types())
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

        # dtypes should be safely castable
        out = torch.empty(*a.shape, dtype=torch.int, device=device)
        with self.assertRaisesRegex(RuntimeError, "but got result with dtype Int"):
            torch.cholesky_inverse(a, out=out)

        # device should match
        if torch.cuda.is_available():
            wrong_device = 'cpu' if self.device_type != 'cpu' else 'cuda'
            out = torch.empty(0, device=wrong_device, dtype=dtype)
            with self.assertRaisesRegex(RuntimeError, "Expected all tensors to be on the same device"):
                torch.cholesky_inverse(a, out=out)

        # cholesky_inverse raises an error for invalid inputs on CPU
        # for example if at least one diagonal element is zero
        a = torch.randn(3, 3, device=device, dtype=dtype)
        a[1, 1] = 0
        if self.device_type == 'cpu':
            with self.assertRaisesRegex(torch.linalg.LinAlgError, r"cholesky_inverse: The diagonal element 2 is zero"):
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

    def lu_solve_test_helper(self, A_dims, b_dims, pivot, device, dtype):
        make_fullrank = make_fullrank_matrices_with_distinct_singular_values
        make_A = partial(make_fullrank, device=device, dtype=dtype)

        b = torch.randn(*b_dims, dtype=dtype, device=device)
        A = make_A(*A_dims)
        LU_data, LU_pivots, info = torch.linalg.lu_factor_ex(A)
        self.assertEqual(info, torch.zeros_like(info))
        return b, A, LU_data, LU_pivots

    @skipCPUIfNoLapack
    @skipCUDAIfNoMagmaAndNoCusolver
    @dtypes(*floating_and_complex_types())
    @precisionOverride({torch.float32: 1e-3, torch.complex64: 1e-3,
                        torch.float64: 1e-8, torch.complex128: 1e-8})
    def test_lu_solve(self, device, dtype):
        def sub_test(pivot):
            for k, n in zip([2, 3, 5], [3, 5, 7]):
                b, A, LU_data, LU_pivots = self.lu_solve_test_helper((n, n), (n, k), pivot, device, dtype)
                x = torch.lu_solve(b, LU_data, LU_pivots)
                self.assertEqual(b, np.matmul(A.cpu(), x.cpu()))

        sub_test(True)
        if self.device_type == 'cuda':
            sub_test(False)

    @skipCPUIfNoLapack
    @skipCUDAIfNoMagmaAndNoCusolver
    @dtypes(*floating_and_complex_types())
    @precisionOverride({torch.float32: 1e-3, torch.complex64: 1e-3,
                        torch.float64: 1e-8, torch.complex128: 1e-8})
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
                lu_solve_batch_test_helper((batchsize, 5, 5), (batchsize, 5, 10), pivot)

        # Tests tensors with 0 elements
        b = torch.randn(3, 0, 3, dtype=dtype, device=device)
        A = torch.randn(3, 0, 0, dtype=dtype, device=device)
        LU_data, LU_pivots = torch.linalg.lu_factor(A)
        self.assertEqual(torch.empty_like(b), b.lu_solve(LU_data, LU_pivots))

        sub_test(True)
        if self.device_type == 'cuda':
            sub_test(False)

    @skipCUDAIfRocm  # ROCm: test was exceptionally slow, even for slow tests. Skip until triage.
    @slowTest
    @skipCPUIfNoLapack
    @skipCUDAIfNoMagmaAndNoCusolver
    @dtypes(*floating_and_complex_types())
    def test_lu_solve_batched_many_batches(self, device, dtype):
        def run_test(A_dims, b_dims):
            b, A, LU_data, LU_pivots = self.lu_solve_test_helper(A_dims, b_dims, True, device, dtype)
            x = torch.lu_solve(b, LU_data, LU_pivots)
            Ax = torch.matmul(A, x)
            self.assertEqual(Ax, b.expand_as(Ax))

        run_test((65536, 5, 5), (65536, 5, 10))
        run_test((262144, 5, 5), (262144, 5, 10))

    @skipCPUIfNoLapack
    @skipCUDAIfNoMagmaAndNoCusolver
    @dtypes(*floating_and_complex_types())
    def test_lu_solve_batched_broadcasting(self, device, dtype):
        make_fullrank = make_fullrank_matrices_with_distinct_singular_values
        make_A = partial(make_fullrank, device=device, dtype=dtype)

        def run_test(A_dims, b_dims, pivot=True):
            A_matrix_size = A_dims[-1]
            A_batch_dims = A_dims[:-2]
            A = make_A(*A_batch_dims, A_matrix_size, A_matrix_size)
            b = make_tensor(b_dims, dtype=dtype, device=device)
            x_exp = np.linalg.solve(A.cpu(), b.cpu())
            LU_data, LU_pivots = torch.linalg.lu_factor(A)
            x = torch.lu_solve(b, LU_data, LU_pivots)
            self.assertEqual(x, x_exp)

        # test against numpy.linalg.solve
        run_test((2, 1, 3, 4, 4), (2, 1, 3, 4, 6))  # no broadcasting
        run_test((2, 1, 3, 4, 4), (4, 6))  # broadcasting b
        run_test((4, 4), (2, 1, 3, 4, 2))  # broadcasting A
        run_test((1, 3, 1, 4, 4), (2, 1, 3, 4, 5))  # broadcasting A & b

    @onlyCUDA
    @skipCUDAIfNoMagma
    @dtypes(*floating_and_complex_types())
    # this tests https://github.com/pytorch/pytorch/issues/36921
    def test_lu_solve_large_matrices(self, device, dtype):
        def run_test(A_dims, b_dims):
            b, A, LU_data, LU_pivots = self.lu_solve_test_helper(A_dims, b_dims, True, device, dtype)
            x = torch.lu_solve(b, LU_data, LU_pivots)
            Ax = torch.matmul(A, x)
            self.assertEqual(Ax, b.expand_as(Ax))

        run_test((1, 1), (1, 1, 1025))

    @skipCUDAIfNoCusolver
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

            A1 = u.matmul(s.diag_embed()).matmul(v.mT)
            ones_m1 = torch.ones(batches + (rows, 1), dtype=a.dtype, device=device)
            c = a.sum(axis=-2) / rows
            c = c.reshape(batches + (1, columns))
            A2 = a - ones_m1.matmul(c)
            self.assertEqual(A1, A2)

            if density == 1:
                # actual rank is known only for dense input
                detect_rank = (s.abs() > 1e-5).sum(axis=-1)
                self.assertEqual(actual_rank * torch.ones(batches, device=device, dtype=torch.int64), detect_rank)
                S = torch.linalg.svdvals(A2)
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
    @onlyNativeDeviceTypes
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

    @skipCUDAIfNoMagmaAndNoCusolver
    @skipCPUIfNoLapack
    @dtypes(*floating_and_complex_types())
    def test_geqrf(self, device, dtype):

        def run_test(shape):
            # numpy.linalg.qr with mode = 'raw' computes the same operation as torch.geqrf
            # so this test compares against that function
            A = make_tensor(shape, dtype=dtype, device=device)

            # numpy.linalg.qr doesn't work with batched input
            m, n = A.shape[-2:]
            tau_size = "n" if m > n else "m"
            np_dtype = A.cpu().numpy().dtype
            ot = [np_dtype, np_dtype]
            numpy_geqrf_batched = np.vectorize(
                lambda x: np.linalg.qr(x, mode='raw'),
                otypes=ot,
                signature=f'(m,n)->(n,m),({tau_size})')

            expected = numpy_geqrf_batched(A.cpu())
            actual = torch.geqrf(A)

            # numpy.linalg.qr returns transposed result
            self.assertEqual(expected[0].swapaxes(-2, -1), actual[0])
            self.assertEqual(expected[1], actual[1])

        batches = [(), (0, ), (2, ), (2, 1)]
        ns = [5, 2, 0]
        for batch, (m, n) in product(batches, product(ns, ns)):
            run_test((*batch, m, n))

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

    @tf32_on_and_off(0.005)
    def test_tensordot(self, device):
        a = torch.arange(60., device=device).reshape(3, 4, 5)
        b = torch.arange(24., device=device).reshape(4, 3, 2)
        c = torch.tensordot(a, b, dims=([1, 0], [0, 1])).cpu()
        cn = torch.from_numpy(np.tensordot(a.cpu().numpy(), b.cpu().numpy(),
                                           axes=([1, 0], [0, 1])))
        self.assertEqual(c, cn)

        cout = torch.zeros((5, 2), device=device)
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

        a = torch.tensordot(torch.tensor(0.), torch.tensor(0.), 0)
        an = torch.from_numpy(np.tensordot(np.zeros((), dtype=np.float32), np.zeros((), dtype=np.float32), 0))
        self.assertEqual(a, an)

    @skipCUDAIfNoCusolver
    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @skipCUDAIfRocm
    @dtypes(*floating_and_complex_types())
    def test_ldl_factor(self, device, dtype):
        from torch.testing._internal.common_utils import random_hermitian_pd_matrix

        def run_test(shape, batch, hermitian):
            A = random_hermitian_pd_matrix(shape, *batch, dtype=dtype, device=device)
            actual_factors, actual_pivots, info = torch.linalg.ldl_factor_ex(A, hermitian=hermitian)
            actual_L = torch.tril(actual_factors, diagonal=-1)
            actual_L.diagonal(0, -2, -1).fill_(1.0)

            # This test is designed only for inputs with 1x1 block diagonal matrix D.
            # That is for positive definite input matrices, the pivots tensor is always > 0.
            # If negative pivots are encountered, it means that the input matrix is not positive definite.
            # And matrix D is a 2x2 block diagonal matrix.
            self.assertTrue((actual_pivots > 0).all())

            # Construct a 1x1 block diagonal matrix D from factors.
            actual_D = torch.diag_embed(actual_factors.diagonal(0, -2, -1))

            def T(x):
                return x.mH if hermitian else x.mT
            A_reconstructed = actual_L @ actual_D @ T(actual_L)

            def symmetric(A):
                return A.tril() + A.tril(-1).mT

            self.assertEqual(symmetric(A) if not hermitian else A, A_reconstructed)

            # Now test against SciPy implementation
            if TEST_SCIPY:
                from scipy.linalg import ldl as scipy_ldl
                A_np = A.cpu().numpy()
                np_dtype = A_np.dtype
                scipy_ldl_batched = np.vectorize(
                    lambda x: scipy_ldl(x, hermitian=hermitian, lower=True),
                    otypes=[np_dtype, np_dtype, np.dtype('int64')],
                    signature='(m,m)->(m,m),(m,m),(m)')

                expected = scipy_ldl_batched(A_np)
                expected_L, expected_D, expected_pivots = expected

                if expected_pivots.ndim > 1:
                    permuted_expected_L = np.stack(
                        [expected_L[i][expected_pivots[i], :] for i in range(expected_pivots.shape[0])]
                    )
                else:
                    permuted_expected_L = expected_L[expected_pivots, :]
                self.assertEqual(actual_L, permuted_expected_L)
                self.assertEqual(actual_D, expected_D)
            else:
                self.assertEqual(actual_factors.shape, A.shape)
                self.assertEqual(actual_pivots.shape, A.shape[:-1])
                self.assertEqual(info.shape, A.shape[:-2])

        # hermitian=True for complex inputs on CUDA is supported only with MAGMA 2.5.4+
        magma_254_available = self.device_type == 'cuda' and _get_magma_version() >= (2, 5, 4)
        hermitians = (True, False) if dtype.is_complex and (self.device_type == 'cpu' or magma_254_available) else (False,)

        shapes = (5,)
        batches = ((), (4,),)
        for shape, batch, hermitian in itertools.product(shapes, batches, hermitians):
            run_test(shape, batch, hermitian)

    @skipCUDAIfNoCusolver
    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @skipCUDAIfRocm
    @skipCUDAIf(_get_torch_cuda_version() < (11, 4), "not available before CUDA 11.3.1")
    @dtypes(*floating_and_complex_types())
    def test_ldl_solve(self, device, dtype):
        from torch.testing._internal.common_utils import random_hermitian_pd_matrix

        def run_test(shape, batch, nrhs, hermitian):
            A = random_hermitian_pd_matrix(shape, *batch, dtype=dtype, device=device)
            B = make_tensor((*A.shape[:-1], nrhs), dtype=dtype, device=device)
            factors, pivots, info = torch.linalg.ldl_factor_ex(A, hermitian=hermitian)
            X = torch.linalg.ldl_solve(factors, pivots, B, hermitian=hermitian)

            def symmetric(A):
                return A.tril() + A.tril(-1).mT

            # verify A @ X == B
            expected_B = symmetric(A) @ X if not hermitian else A @ X
            self.assertEqual(B, expected_B)

        # hermitian=True is not supported on CUDA yet
        hermitians = (True, False) if dtype.is_complex and self.device_type == 'cpu' else (False,)

        shapes = (5,)
        batches = ((), (4,), (2, 2))
        nrhss = (1, 7)
        for shape, batch, nrhs, hermitian in itertools.product(shapes, batches, nrhss, hermitians):
            run_test(shape, batch, nrhs, hermitian)

    @onlyCUDA
    @skipCUDAIfNoMagma
    @skipCUDAIfNoCusolver
    @setLinalgBackendsToDefaultFinally
    def test_preferred_linalg_library(self):
        # The main purpose of this test is to make sure these "backend" calls work normally without raising exceptions.
        x = torch.randint(2, 5, (2, 4, 4), device='cuda', dtype=torch.double)

        torch.backends.cuda.preferred_linalg_library('cusolver')
        out1 = torch.linalg.inv(x)

        torch.backends.cuda.preferred_linalg_library('magma')
        out2 = torch.linalg.inv(x)

        torch.backends.cuda.preferred_linalg_library('default')
        # Although linalg preferred flags doesn't affect CPU currently,
        # we set this to make sure the flag can switch back to default normally.
        out_ref = torch.linalg.inv(x.cpu())

        self.assertEqual(out_ref, out1.cpu())
        self.assertEqual(out1, out2)

    def test_permute_matmul(self):
        a = torch.ones([2, 5, 24, 24])
        b = torch.ones([3, 2, 5, 24, 24])
        c = a.permute(0, 1, 3, 2).matmul(b)
        self.assertEqual([c.min(), c.max(), c.sum()], [24, 24, 414720])

    def test_bfloat16_accumulation_with_ref_path(self):
        # fix https://github.com/pytorch/pytorch/issues/95125
        # and https://github.com/pytorch/pytorch/issues/83863
        # for bf16 accumulation in gemm ref path
        def check_correctness(fn, *args):
            expected = fn(*args).bfloat16()
            with torch.backends.mkldnn.flags(enabled=False):
                def test():
                    bf16_args = (arg.bfloat16() for arg in args)
                    tmp_result = fn(*bf16_args)
                    return tmp_result
                c = test()
                assert (torch.all(c == expected)), "Incorrect result with\n" \
                                                   f"expected: {expected}\n" \
                                                   f"got: {c}\n"
        # test matmul
        for transa in [True, False]:
            for transb in [True, False]:
                a = torch.ones(300, 300)
                b = torch.ones(300, 300)
                if transa:
                    a = a.transpose(0, 1).contiguous().transpose(0, 1)
                if transb:
                    b = b.transpose(0, 1).contiguous().transpose(0, 1)
                check_correctness(torch.matmul, a, b)
        # test bmm
        a = torch.ones(1, 1, 300)
        b = torch.ones(1, 300, 1)
        check_correctness(torch.bmm, a, b)
        # test baddbmm
        a = torch.ones(1, 1, 300)
        b = torch.ones(1, 300, 1)
        c = torch.ones(1, 1, 1)
        check_correctness(torch.baddbmm, c, a, b)
        # test mv/addmv
        for trans in [True, False]:
            c = torch.ones(300) * -300
            a = torch.ones(300, 300)
            if trans:
                a = a.transpose(0, 1).contiguous().transpose(0, 1)
            b = torch.ones(300)
            check_correctness(torch.mv, a, b)
            check_correctness(torch.addmv, c, a, b)
        # test dot
        a = torch.ones(300)
        b = torch.ones(300)
        check_correctness(torch.dot, a, b)

instantiate_device_type_tests(TestLinalg, globals())

if __name__ == '__main__':
    run_tests()
