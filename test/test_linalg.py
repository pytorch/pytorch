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
     onlyCPU, skipCUDAIfNoMagma, skipCPUIfNoLapack, precisionOverride,
     skipCUDAIfNoMagmaAndNoCusolver, skipCUDAIfRocm, onlyOnCPUAndCUDA)
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

    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    @precisionOverride({torch.bfloat16: 1e-1})
    @dtypes(*(torch.testing.get_all_dtypes()))
    def test_addr(self, device, dtype):
        def run_test_case(m, a, b, beta=1, alpha=1):
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

            self.assertEqual(torch.addr(m, a, b, beta=beta, alpha=alpha), expected)
            self.assertEqual(torch.Tensor.addr(m, a, b, beta=beta, alpha=alpha), expected)

            result_dtype = torch.addr(m, a, b, beta=beta, alpha=alpha).dtype
            out = torch.empty_like(m, dtype=result_dtype)
            torch.addr(m, a, b, beta=beta, alpha=alpha, out=out)
            self.assertEqual(out, expected)

        a = torch.randn(50).to(device=device, dtype=dtype)
        b = torch.randn(50).to(device=device, dtype=dtype)
        m = torch.randn(50, 50).to(device=device, dtype=dtype)

        # when beta is zero
        run_test_case(m, a, b, beta=0., alpha=2)

        # when beta is not zero
        run_test_case(m, a, b, beta=0.5, alpha=2)

        # test transpose
        m_transpose = torch.transpose(m, 0, 1)
        run_test_case(m_transpose, a, b, beta=0.5, alpha=2)

        # test 0 strided tensor
        zero_strided = torch.randn(1).to(device=device, dtype=dtype).expand(50)
        run_test_case(m, zero_strided, b, beta=0.5, alpha=2)

        # test scalar
        m_scalar = torch.tensor(1, device=device, dtype=dtype)
        run_test_case(m_scalar, a, b)

    @dtypes(*itertools.product(torch.testing.get_all_dtypes(),
                               torch.testing.get_all_dtypes()))
    def test_outer_type_promotion(self, device, dtypes):
        a = torch.randn(5).to(device=device, dtype=dtypes[0])
        b = torch.randn(5).to(device=device, dtype=dtypes[1])
        for op in (torch.outer, torch.Tensor.outer, torch.ger, torch.Tensor.ger):
            result = op(a, b)
            self.assertEqual(result.dtype, torch.result_type(a, b))

    @dtypes(*itertools.product(torch.testing.get_all_dtypes(),
                               torch.testing.get_all_dtypes()))
    def test_addr_type_promotion(self, device, dtypes):
        a = torch.randn(5).to(device=device, dtype=dtypes[0])
        b = torch.randn(5).to(device=device, dtype=dtypes[1])
        m = torch.randn(5, 5).to(device=device,
                                 dtype=torch.result_type(a, b))
        for op in (torch.addr, torch.Tensor.addr):
            # pass the integer 1 to the torch.result_type as both
            # the default values of alpha and beta are integers (alpha=1, beta=1)
            desired_dtype = torch.result_type(m, 1)
            result = op(m, a, b)
            self.assertEqual(result.dtype, desired_dtype)

            desired_dtype = torch.result_type(m, 2.)
            result = op(m, a, b, beta=0, alpha=2.)
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
            # TODO(@ivanyashchuk): remove this once batched matmul is avaiable on CUDA for complex dtypes
            if self.device_type == 'cuda' and dtype.is_complex:
                result_identity_list1 = []
                result_identity_list2 = []
                p = int(np.prod(batches))  # use `p` instead of -1, so that the test works for empty input as well
                for m, m_inv in zip(matrix.contiguous().view(p, n, n), matrix_inverse.contiguous().view(p, n, n)):
                    result_identity_list1.append(torch.matmul(m, m_inv))
                    result_identity_list2.append(torch.matmul(m_inv, m))
                result_identity1 = torch.stack(result_identity_list1).view(*batches, n, n)
                result_identity2 = torch.stack(result_identity_list2).view(*batches, n, n)
                self.assertEqual(identity.expand_as(matrix), result_identity1)
                self.assertEqual(identity.expand_as(matrix), result_identity2)
            else:
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
            # TODO(@ivanyashchuk): remove this once batched matmul is avaiable on CUDA for complex dtypes
            if self.device_type == 'cuda' and dtype.is_complex:
                Ax = torch.matmul(A.cpu(), x_act.cpu()).to(device)
            else:
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
            # TODO(@ivanyashchuk): remove this once batched matmul is avaiable on CUDA for complex dtypes
            if self.device_type == 'cuda' and dtype.is_complex:
                Ax = torch.matmul(A.cpu(), x.cpu()).to(device)
            else:
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
        # TODO(@ivanyashchuk): remove this once batched matmul is avaiable on CUDA for complex dtypes
        if self.device_type == 'cuda' and dtype.is_complex:
            A_tmp = torch.empty_like(A).view(-1, *A_dims[-2:])
            for A_i, A_tmp_i in zip(A.contiguous().view(-1, *A_dims[-2:]), A_tmp):
                torch.matmul(A_i, A_i.t(), out=A_tmp_i)
            A = A_tmp.view(*A_dims)
        else:
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

            # TODO(@ivanyashchuk): remove this once batched matmul is avaiable on CUDA for complex dtypes
            if self.device_type == 'cuda' and dtype.is_complex:
                Ax = torch.empty_like(x_act).view(-1, *b_dims[-2:])
                for A_i, x_i, Ax_i in zip(A.contiguous().view(-1, *A_dims[-2:]),
                                          x_act.contiguous().view(-1, *b_dims[-2:]), Ax):
                    torch.matmul(A_i, x_i, out=Ax_i)
                Ax = Ax.view(*x_act.shape)
            else:
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

            # TODO(@ivanyashchuk): remove this once batched matmul is avaiable on CUDA for complex dtypes
            if self.device_type == 'cuda' and dtype.is_complex:
                Ax = torch.empty_like(x).view(-1, 5, 1)
                for A_i, x_i, Ax_i in zip(A.contiguous().view(-1, 5, 5), x.contiguous().view(-1, 5, 1), Ax):
                    torch.matmul(A_i, x_i, out=Ax_i)
                Ax = Ax.view(256, 256, 5, 1)
            else:
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
