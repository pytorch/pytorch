import torch
import unittest
import itertools
import warnings
from math import inf, nan, isnan

from torch.testing._internal.common_utils import \
    (TestCase, run_tests, TEST_NUMPY)
from torch.testing._internal.common_device_type import \
    (instantiate_device_type_tests, dtypes, skipCUDAIfNoMagma, skipCPUIfNoLapack)
from torch.testing._internal.jit_metaprogramming_utils import gen_script_fn_and_args
from torch.autograd import gradcheck

if TEST_NUMPY:
    import numpy as np

class TestLinalg(TestCase):
    exact_dtype = True

    # TODO: test out variant
    # Tests torch.ger, and its alias, torch.outer, vs. NumPy
    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    @dtypes(torch.float)
    def test_outer(self, device, dtype):
        a = torch.randn(50, device=device, dtype=dtype)
        b = torch.randn(50, device=device, dtype=dtype)

        ops = (torch.ger, torch.Tensor.ger,
               torch.outer, torch.Tensor.outer)

        expected = np.outer(a.cpu().numpy(), b.cpu().numpy())
        for op in ops:
            actual = op(a, b)
            self.assertEqual(actual, expected)

    # Tests torch.det and its alias, torch.linalg.det, vs. NumPy
    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    @dtypes(torch.double)
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

        # NOTE: det requires a 2D+ tensor
        t = torch.randn(1, device=device, dtype=dtype)
        with self.assertRaises(IndexError):
            op(t)

    # This test confirms that torch.linalg.norm's dtype argument works
    # as expected, according to the function's documentation
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
        ord_matrix = [1, -1, 2, -2, inf, -inf, None]
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

        # TODO: Once dtype arg is supported in nuclear and frobenius norms, remove the following test
        #       and  add 'nuc' and 'fro' to ord_matrix above
        for ord in ['nuc', 'fro']:
            input = torch.randn(10, 10, device=device)
            with self.assertRaisesRegex(RuntimeError, f"ord=\'{ord}\' does not yet support the dtype argument"):
                torch.linalg.norm(input, ord, dtype=torch.float)

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
            ('norm', (S,), (2,), 'vector_2'),
            ('norm', (S,), (1,), 'vector_1'),
            ('norm', (S,), (0,), 'vector_0'),
            ('norm', (S,), (-inf,), 'vector_neg_inf'),
            ('norm', (S,), (-3.5,), 'vector_neg_3_5'),
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
                (2, r'"svd_cuda" not implemented for \'Complex'),
                (-2, r'"svd_cuda" not implemented for \'Complex'),
                ('nuc', r'"svd_cuda" not implemented for \'Complex'),
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
                    self.assertNotEqual(result, result_n, msg=msg)
                else:
                    self.assertEqual(result, result_n, msg=msg)

    # Test degenerate shape results match numpy for linalg.norm vector norms
    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @unittest.skipIf(not TEST_NUMPY, "Numpy not found")
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

    def test_norm_deprecated(self, device):
        expected_message = (
            r'torch.norm is deprecated and may be removed in a future PyTorch release. '
            r'Use torch.linalg.norm instead.')
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            for func in [torch.norm, torch.functional.norm]:
                func(torch.rand(10, device=device))
        self.assertEqual(len(w), 2)
        for wi in w:
            self.assertEqual(str(wi.message), expected_message)

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

instantiate_device_type_tests(TestLinalg, globals())

if __name__ == '__main__':
    run_tests()
