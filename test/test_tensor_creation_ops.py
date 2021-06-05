import torch
import numpy as np

import sys
import math
import warnings
import unittest
from itertools import product, combinations, combinations_with_replacement, permutations
import random

from torch.testing._internal.common_utils import (
    TestCase, run_tests, do_test_empty_full, TEST_WITH_ROCM, suppress_warnings,
    torch_to_numpy_dtype_dict, slowTest, make_tensor, TEST_SCIPY, IS_MACOS, IS_PPC,
    IS_WINDOWS)
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests, deviceCountAtLeast, onlyOnCPUAndCUDA,
    onlyCPU, largeTensorTest, precisionOverride, dtypes,
    onlyCUDA, skipCPUIf, dtypesIfCUDA, dtypesIfCPU, skipMeta)

# TODO: refactor tri_tests_args, _compare_trilu_indices, run_additional_tri_tests
from torch.testing._internal.common_methods_invocations import (
    tri_tests_args, _compare_trilu_indices, run_additional_tri_tests)


# TODO: replace with make_tensor
def _generate_input(shape, dtype, device, with_extremal):
    if shape == ():
        x = torch.tensor((), dtype=dtype, device=device)
    else:
        if dtype.is_floating_point or dtype.is_complex:
            # work around torch.randn not being implemented for bfloat16
            if dtype == torch.bfloat16:
                x = torch.randn(*shape, device=device) * random.randint(30, 100)
                x = x.to(torch.bfloat16)
            else:
                x = torch.randn(*shape, dtype=dtype, device=device) * random.randint(30, 100)
            x[torch.randn(*shape) > 0.5] = 0
            if with_extremal and dtype.is_floating_point:
                # Use extremal values
                x[torch.randn(*shape) > 0.5] = float('nan')
                x[torch.randn(*shape) > 0.5] = float('inf')
                x[torch.randn(*shape) > 0.5] = float('-inf')
            elif with_extremal and dtype.is_complex:
                x[torch.randn(*shape) > 0.5] = complex('nan')
                x[torch.randn(*shape) > 0.5] = complex('inf')
                x[torch.randn(*shape) > 0.5] = complex('-inf')
        elif dtype == torch.bool:
            x = torch.zeros(shape, dtype=dtype, device=device)
            x[torch.randn(*shape) > 0.5] = True
        else:
            x = torch.randint(15, 100, shape, dtype=dtype, device=device)

    return x


# TODO: replace with make_tensor
def _rand_shape(dim, min_size, max_size):
    shape = []
    for i in range(dim):
        shape.append(random.randint(min_size, max_size))
    return tuple(shape)

# Test suite for tensor creation ops
#
# Includes creation functions like torch.eye, random creation functions like
#   torch.rand, and *like functions like torch.ones_like.
# DOES NOT INCLUDE view ops, which are tested in TestViewOps (currently in
#   test_torch.py) OR numpy interop (which is also still tested in test_torch.py)
#
# See https://pytorch.org/docs/master/torch.html#creation-ops

class TestTensorCreation(TestCase):
    exact_dtype = True

    @onlyCPU
    @dtypes(torch.float)
    def test_diag_embed(self, device, dtype):
        x = torch.arange(3 * 4, dtype=dtype, device=device).view(3, 4)
        result = torch.diag_embed(x)
        expected = torch.stack([torch.diag(r) for r in x], 0)
        self.assertEqual(result, expected)

        result = torch.diag_embed(x, offset=1, dim1=0, dim2=2)
        expected = torch.stack([torch.diag(r, 1) for r in x], 1)
        self.assertEqual(result, expected)

    def test_cat_mem_overlap(self, device):
        x = torch.rand((1, 3), device=device).expand((6, 3))
        y = torch.rand((3, 3), device=device)
        with self.assertRaisesRegex(RuntimeError, 'unsupported operation'):
            torch.cat([y, y], out=x)

    @onlyOnCPUAndCUDA
    def test_vander(self, device):
        x = torch.tensor([1, 2, 3, 5], device=device)

        self.assertEqual((0, 0), torch.vander(torch.tensor([]), 0).shape)

        with self.assertRaisesRegex(RuntimeError, "N must be non-negative."):
            torch.vander(x, N=-1)

        with self.assertRaisesRegex(RuntimeError, "x must be a one-dimensional tensor."):
            torch.vander(torch.stack((x, x)))

    @onlyOnCPUAndCUDA
    @dtypes(torch.bool, torch.uint8, torch.int8, torch.short, torch.int, torch.long,
            torch.float, torch.double,
            torch.cfloat, torch.cdouble)
    def test_vander_types(self, device, dtype):
        if dtype is torch.uint8:
            # Note: no negative uint8 values
            X = [[1, 2, 3, 5], [0, 1 / 3, 1, math.pi, 3 / 7]]
        elif dtype is torch.bool:
            # Note: see https://github.com/pytorch/pytorch/issues/37398
            # for why this is necessary.
            X = [[True, True, True, True], [False, True, True, True, True]]
        elif dtype in [torch.cfloat, torch.cdouble]:
            X = [[1 + 1j, 1 + 0j, 0 + 1j, 0 + 0j],
                 [2 + 2j, 3 + 2j, 4 + 3j, 5 + 4j]]
        else:
            X = [[1, 2, 3, 5], [-math.pi, 0, 1 / 3, 1, math.pi, 3 / 7]]

        N = [None, 0, 1, 3]
        increasing = [False, True]

        for x, n, inc in product(X, N, increasing):
            numpy_dtype = torch_to_numpy_dtype_dict[dtype]
            pt_x = torch.tensor(x, device=device, dtype=dtype)
            np_x = np.array(x, dtype=numpy_dtype)

            pt_res = torch.vander(pt_x, increasing=inc) if n is None else torch.vander(pt_x, n, inc)
            np_res = np.vander(np_x, n, inc)

            self.assertEqual(
                pt_res,
                torch.from_numpy(np_res),
                atol=1e-3,
                rtol=0,
                exact_dtype=False)

    def test_cat_all_dtypes_and_devices(self, device):
        for dt in torch.testing.get_all_dtypes():
            x = torch.tensor([[1, 2], [3, 4]], dtype=dt, device=device)

            expected1 = torch.tensor([[1, 2], [3, 4], [1, 2], [3, 4]], dtype=dt, device=device)
            self.assertEqual(torch.cat((x, x), 0), expected1)

            expected2 = torch.tensor([[1, 2, 1, 2], [3, 4, 3, 4]], dtype=dt, device=device)
            self.assertEqual(torch.cat((x, x), 1), expected2)

    def test_fill_all_dtypes_and_devices(self, device):
        for dt in torch.testing.get_all_dtypes():
            for x in [torch.tensor((10, 10), dtype=dt, device=device),
                      torch.empty(10000, dtype=dt, device=device)]:  # large tensor
                numel = x.numel()
                bound = 100 if dt in (torch.uint8, torch.int8) else 2000
                for n in range(-bound, bound, bound // 10):
                    x.fill_(n)
                    self.assertEqual(x, torch.tensor([n] * numel, dtype=dt, device=device))
                    self.assertEqual(dt, x.dtype)

    def test_roll(self, device):
        numbers = torch.arange(1, 9, device=device)

        single_roll = numbers.roll(1, 0)
        expected = torch.tensor([8, 1, 2, 3, 4, 5, 6, 7], device=device)
        self.assertEqual(single_roll, expected, msg="{} did not equal expected result".format(single_roll))

        roll_backwards = numbers.roll(-2, 0)
        expected = torch.tensor([3, 4, 5, 6, 7, 8, 1, 2], device=device)
        self.assertEqual(roll_backwards, expected, msg="{} did not equal expected result".format(roll_backwards))

        data = numbers.view(2, 2, 2)
        rolled = data.roll(1, 0)
        expected = torch.tensor([5, 6, 7, 8, 1, 2, 3, 4], device=device).view(2, 2, 2)
        self.assertEqual(expected, rolled, msg="{} did not equal expected result: {}".format(rolled, expected))

        data = data.view(2, 4)
        # roll a loop until back where started
        loop_rolled = data.roll(2, 0).roll(4, 1)
        self.assertEqual(data, loop_rolled, msg="{} did not equal the original: {}".format(loop_rolled, data))
        # multiple inverse loops
        self.assertEqual(data, data.roll(-20, 0).roll(-40, 1))
        self.assertEqual(torch.tensor([8, 1, 2, 3, 4, 5, 6, 7], device=device), numbers.roll(1, 0))

        # test non-contiguous
        # strided equivalent to numbers.as_strided(size=(4, 2), stride=(1, 4))
        strided = numbers.view(2, 4).transpose(0, 1)
        self.assertFalse(strided.is_contiguous(), "this test needs a non-contiguous tensor")
        expected = torch.tensor([4, 8, 1, 5, 2, 6, 3, 7]).view(4, 2)
        rolled = strided.roll(1, 0)
        self.assertEqual(expected, rolled,
                         msg="non contiguous tensor rolled to {} instead of {} ".format(rolled, expected))

        # test roll with no dimension specified
        expected = numbers.roll(1, 0).view(2, 4)
        self.assertEqual(expected, data.roll(1), msg="roll with no dims should flatten and roll.")
        self.assertEqual(expected, data.roll(1, dims=None), msg="roll with no dims should flatten and roll.")

        # test roll over multiple dimensions
        expected = torch.tensor([[7, 8, 5, 6], [3, 4, 1, 2]], device=device)
        double_rolled = data.roll(shifts=(2, -1), dims=(1, 0))
        self.assertEqual(double_rolled, expected,
                         msg="should be able to roll over two dimensions, got {}".format(double_rolled))

        self.assertRaisesRegex(RuntimeError, "required", lambda: data.roll(shifts=(), dims=()))
        self.assertRaisesRegex(RuntimeError, "required", lambda: data.roll(shifts=(), dims=1))
        # shifts/dims should align
        self.assertRaisesRegex(RuntimeError, "align", lambda: data.roll(shifts=(1, 2), dims=(1,)))
        self.assertRaisesRegex(RuntimeError, "align", lambda: data.roll(shifts=(1,), dims=(1, 2)))

        # test bool tensor
        t = torch.zeros(6, dtype=torch.bool, device=device)
        t[0] = True
        t[3] = True
        self.assertEqual(torch.tensor([False, True, False, False, True, False]), t.roll(1, 0))

        # test complex tensor
        t = torch.tensor([1, 2 + 1j, 3.5, 4. + 2j, 5j, 6.], device=device)
        t[0] = 1 + 0.5j
        t[3] = 4.
        expected = torch.tensor([6., 1 + 0.5j, 2 + 1j, 3.5, 4., 5j], device=device)
        self.assertEqual(expected, t.roll(1, 0))

    @slowTest
    def test_triu_tril(self, device):
        def gen_mask(shape, diagonal, device, upper):
            mask = torch.zeros(*shape[-2:]).byte()
            for i in range(shape[-2]):
                for j in range(shape[-1]):
                    cond = j - i < diagonal if upper else j - i > diagonal
                    if cond:
                        mask[i, j] = 1
            return mask.expand(*shape).to(device)

        torch_functions = {True: torch.triu, False: torch.tril}
        numpy_functions = {True: np.triu, False: np.tril}

        # TODO: remove this when bool and half are supported for torch.where
        def bool_half_compat_where(pred, true_tensor, false_tensor, dtype):
            if dtype == torch.bool or dtype == torch.half:
                return torch.where(pred.byte(), true_tensor.byte(), false_tensor.byte()).to(dtype=dtype)
            else:
                return torch.where(pred, true_tensor, false_tensor)

        def run_test(shape, device, diagonal, dtype):
            x = torch.empty(*shape, device=device, dtype=dtype).fill_(2)

            for upper in [True, False]:
                # normal test with mask
                torch_tri_func = torch_functions[upper]
                res1 = torch_tri_func(x, diagonal=diagonal)
                res2 = torch.empty(0, device=device, dtype=dtype)
                torch_tri_func(x, diagonal=diagonal, out=res2)
                exp_mask = gen_mask(shape, diagonal, device, upper)
                expected = bool_half_compat_where(exp_mask, torch.tensor(0).type_as(x), x, dtype)
                self.assertEqual(res1, res2, atol=0, rtol=0)
                self.assertEqual(expected, res1, atol=0, rtol=0)

                # non-contiguous and expanded tensors test
                if 0 not in shape:
                    for s in range(-len(shape), -1):
                        # non-contiguous tensors
                        x_nc = x.clone().transpose(s, s + 1)
                        exp_mask = gen_mask(x_nc.size(), diagonal, device, upper)
                        if 1 not in shape:
                            assert not x_nc.is_contiguous(), "x is intentionally non-contiguous"
                        exp_nc = bool_half_compat_where(exp_mask, torch.tensor(0).type_as(x), x_nc, dtype)
                        self.assertEqual(torch_tri_func(x_nc, diagonal), exp_nc, atol=0, rtol=0)
                        x_nc_is_contiguous = x_nc.is_contiguous()
                        if upper:
                            self.assertEqual(x_nc.triu_(diagonal), exp_nc, atol=0, rtol=0)
                        else:
                            self.assertEqual(x_nc.tril_(diagonal), exp_nc, atol=0, rtol=0)

                        self.assertTrue(x_nc.is_contiguous() == x_nc_is_contiguous,
                                        "contiguity of x_nc should not be changed")

                    # expanded tensors
                    expanded_size = (x.size(0),) + x.size()
                    x_expanded = x.clone().expand(*expanded_size)
                    if x.size(0) != 1:
                        assert 0 in x_expanded.stride(), "x intentionally has 0 in its stride"
                    output = torch_tri_func(x_expanded, diagonal)
                    self.assertEqual(output, expected.expand(expanded_size), atol=0, rtol=0)
                    if x.size(0) != 1:
                        self.assertTrue(0 in x_expanded.stride(),
                                        "geometry of x_expanded should be the same")
                    if upper:
                        self.assertEqual(output, x_expanded.triu_(diagonal), atol=0, rtol=0)
                    else:
                        self.assertEqual(output, x_expanded.tril_(diagonal), atol=0, rtol=0)

                # numpy test
                numpy_tri_func = numpy_functions[upper]
                self.assertEqual(numpy_tri_func(x.to('cpu').numpy(), diagonal), res1.cpu().numpy())

        diagonals = [-2, -1, 0, 1, 2]
        shapes = [(3, 3), (5, 3, 3), (7, 5, 3, 3),  # square matrices
                  (7, 3), (5, 7, 3), (7, 5, 7, 3),  # fat matrices
                  (3, 7), (5, 3, 7), (7, 5, 3, 7),  # thin matrices
                  (3, 0), (0, 3, 3), (3, 3, 0, 0),  # no numel matrices
                  (3, 1), (5, 3, 1), (7, 5, 3, 1),  # very fat matrices
                  (1, 3), (5, 1, 3), (7, 5, 1, 3),  # very thin matrices
                  (1, 3, 3, 3), (3, 1, 3, 3, 3)]    # unsqueezed batch dimensions
        dtypes = [dtype for dtype in torch.testing.get_all_dtypes() if dtype != torch.bfloat16]
        for s, d, dtype in product(shapes, diagonals, dtypes):
            run_test(s, device, d, dtype)

    def test_diagflat(self, device):
        dtype = torch.float32
        # Basic sanity test
        x = torch.randn((100,), dtype=dtype, device=device)
        result = torch.diagflat(x)
        expected = torch.diag(x)
        self.assertEqual(result, expected)

        # Test offset
        x = torch.randn((100,), dtype=dtype, device=device)
        result = torch.diagflat(x, 17)
        expected = torch.diag(x, 17)
        self.assertEqual(result, expected)

        # Test where input has more than one dimension
        x = torch.randn((2, 3, 4), dtype=dtype, device=device)
        result = torch.diagflat(x)
        expected = torch.diag(x.contiguous().view(-1))
        self.assertEqual(result, expected)

        # Noncontig input
        x = torch.randn((2, 3, 4), dtype=dtype, device=device).transpose(2, 0)
        self.assertFalse(x.is_contiguous())
        result = torch.diagflat(x)
        expected = torch.diag(x.contiguous().view(-1))
        self.assertEqual(result, expected)

        # Complex number support
        result = torch.diagflat(torch.ones(4, dtype=torch.complex128))
        expected = torch.eye(4, dtype=torch.complex128)
        self.assertEqual(result, expected)

    def test_block_diag(self, device):
        def block_diag_workaround(*arrs):
            arrs_expanded = []
            for a in arrs:
                if a.dim() == 2:
                    arrs_expanded.append(a)
                elif a.dim() == 1:
                    arrs_expanded.append(a.expand(1, a.size(0)))
                elif a.dim() == 0:
                    arrs_expanded.append(a.expand(1, 1))
            shapes = torch.tensor([a.shape for a in arrs_expanded], device=device)
            out = torch.zeros(
                torch.sum(shapes, dim=0).tolist(),
                dtype=arrs_expanded[0].dtype,
                device=device
            )
            r, c = 0, 0
            for i, (rr, cc) in enumerate(shapes):
                out[r:r + rr, c:c + cc] = arrs_expanded[i]
                r += rr
                c += cc
            return out

        tensors = [
            torch.rand((2, 2), device=device),
            torch.rand((2, 3), device=device),
            torch.rand(10, device=device),
            torch.rand((8, 1), device=device),
            torch.rand(1, device=device)[0]
        ]
        result = torch.block_diag(*tensors)
        result_check = block_diag_workaround(*tensors)
        self.assertEqual(result, result_check)

        tensor = torch.rand(1, device=device)[0]
        result = torch.block_diag(tensor)
        result_check = tensor.expand(1, 1)
        self.assertEqual(result, result_check)

        tensor = torch.rand(10, device=device)
        result = torch.block_diag(tensor)
        result_check = tensor.expand(1, tensor.size(0))
        self.assertEqual(result, result_check)

        result = torch.block_diag()
        result_check = torch.empty(1, 0, device=device)
        self.assertEqual(result, result_check)
        self.assertEqual(result.device.type, 'cpu')

        test_dtypes = [
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.float32,
            torch.float64,
            torch.complex64,
            torch.complex128
        ]
        # Test pairs of different dtypes
        for dtype1 in test_dtypes:
            for dtype2 in test_dtypes:
                a = torch.tensor(1, device=device, dtype=dtype1)
                b = torch.tensor(2, device=device, dtype=dtype2)
                result = torch.block_diag(a, b)
                result_dtype = torch.result_type(a, b)
                result_check = torch.tensor([[1, 0], [0, 2]], device=device, dtype=result_dtype)
                self.assertEqual(result, result_check)

        with self.assertRaisesRegex(
            RuntimeError,
            "torch.block_diag: Input tensors must have 2 or fewer dimensions. Input 1 has 3 dimensions"
        ):
            torch.block_diag(torch.tensor(5), torch.tensor([[[6]]]))

        with self.assertRaisesRegex(
            RuntimeError,
            "torch.block_diag: Input tensors must have 2 or fewer dimensions. Input 0 has 4 dimensions"
        ):
            torch.block_diag(torch.tensor([[[[6]]]]))

        if device != 'cpu':
            with self.assertRaisesRegex(
                RuntimeError,
                (
                    "torch.block_diag: input tensors must all be on the same device."
                    " Input 0 is on device cpu and input 1 is on device "
                )
            ):
                torch.block_diag(torch.ones(2, 2).cpu(), torch.ones(2, 2, device=device))

    @unittest.skipIf(not TEST_SCIPY, "Scipy not found")
    def test_block_diag_scipy(self, device):
        import scipy.linalg
        scipy_tensors_list = [
            [
                1,
                [2],
                [],
                [3, 4, 5],
                [[], []],
                [[6], [7.3]]
            ],
            [
                [[1, 2], [3, 4]],
                [1]
            ],
            [
                [[4, 9], [7, 10]],
                [4.6, 9.12],
                [1j + 3]
            ],
            []
        ]

        expected_torch_types = [
            torch.float32,
            torch.int64,
            torch.complex64,
            torch.float32
        ]

        expected_scipy_types = [
            torch.float64,
            # windows scipy block_diag returns int32 types
            torch.int32 if IS_WINDOWS else torch.int64,
            torch.complex128,
            torch.float64
        ]

        for scipy_tensors, torch_type, scipy_type in zip(scipy_tensors_list, expected_torch_types, expected_scipy_types):
            torch_tensors = [torch.tensor(t, device=device) for t in scipy_tensors]
            torch_result = torch.block_diag(*torch_tensors)
            self.assertEqual(torch_result.dtype, torch_type)

            scipy_result = torch.tensor(
                scipy.linalg.block_diag(*scipy_tensors),
                device=device
            )
            self.assertEqual(scipy_result.dtype, scipy_type)
            scipy_result = scipy_result.to(torch_type)

            self.assertEqual(torch_result, scipy_result)

    @onlyOnCPUAndCUDA
    @dtypes(torch.float32, torch.float64)
    def test_torch_complex(self, device, dtype):
        real = torch.tensor([1, 2], device=device, dtype=dtype)
        imag = torch.tensor([3, 4], device=device, dtype=dtype)
        z = torch.complex(real, imag)
        complex_dtype = torch.complex64 if dtype == torch.float32 else torch.complex128
        self.assertEqual(torch.tensor([1.0 + 3.0j, 2.0 + 4.0j], dtype=complex_dtype), z)

    @onlyOnCPUAndCUDA
    @dtypes(torch.float32, torch.float64)
    def test_torch_polar(self, device, dtype):
        abs = torch.tensor([1, 2, -3, -4.5, 1, 1], device=device, dtype=dtype)
        angle = torch.tensor([math.pi / 2, 5 * math.pi / 4, 0, -11 * math.pi / 6, math.pi, -math.pi],
                             device=device, dtype=dtype)
        z = torch.polar(abs, angle)
        complex_dtype = torch.complex64 if dtype == torch.float32 else torch.complex128
        self.assertEqual(torch.tensor([1j, -1.41421356237 - 1.41421356237j, -3,
                                       -3.89711431703 - 2.25j, -1, -1],
                                      dtype=complex_dtype),
                         z, atol=1e-5, rtol=1e-5)

    @onlyOnCPUAndCUDA
    @dtypes(torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64,
            torch.float16, torch.complex64, torch.complex128, torch.bool)
    def test_torch_complex_floating_dtype_error(self, device, dtype):
        for op in (torch.complex, torch.polar):
            a = torch.tensor([1, 2], device=device, dtype=dtype)
            b = torch.tensor([3, 4], device=device, dtype=dtype)
            error = r"Expected both inputs to be Float or Double tensors but " \
                    r"got [A-Za-z]+ and [A-Za-z]+"
        with self.assertRaisesRegex(RuntimeError, error):
            op(a, b)

    @onlyOnCPUAndCUDA
    @dtypes(torch.float32, torch.float64)
    def test_torch_complex_same_dtype_error(self, device, dtype):

        def dtype_name(dtype):
            return 'Float' if dtype == torch.float32 else 'Double'

        for op in (torch.complex, torch.polar):
            other_dtype = torch.float64 if dtype == torch.float32 else torch.float32
            a = torch.tensor([1, 2], device=device, dtype=dtype)
            b = torch.tensor([3, 4], device=device, dtype=other_dtype)
            error = "Expected object of scalar type {} but got scalar type " \
                    "{} for second argument".format(dtype_name(dtype),
                                                    dtype_name(other_dtype))
            with self.assertRaisesRegex(RuntimeError, error):
                op(a, b)

    @onlyOnCPUAndCUDA
    @dtypes(torch.float32, torch.float64)
    def test_torch_complex_out_dtype_error(self, device, dtype):

        def dtype_name(dtype):
            return 'Float' if dtype == torch.float32 else 'Double'

        def complex_dtype_name(dtype):
            return 'ComplexFloat' if dtype == torch.complex64 else 'ComplexDouble'

        for op in (torch.complex, torch.polar):
            a = torch.tensor([1, 2], device=device, dtype=dtype)
            b = torch.tensor([3, 4], device=device, dtype=dtype)
            out = torch.zeros(2, device=device, dtype=dtype)
            expected_dtype = torch.complex64 if dtype == torch.float32 else torch.complex128
            error = "Expected object of scalar type {} but got scalar type " \
                    "{} for argument 'out'".format(
                        complex_dtype_name(expected_dtype), dtype_name(dtype))
            with self.assertRaisesRegex(RuntimeError, error):
                op(a, b, out=out)

    def test_cat_empty_legacy(self, device):
        # FIXME: this is legacy behavior and should be removed
        # when we support empty tensors with arbitrary sizes
        dtype = torch.float32

        x = torch.randn((4, 3, 32, 32), dtype=dtype, device=device)
        empty = torch.randn((0,), dtype=dtype, device=device)

        res1 = torch.cat([x, empty], dim=1)
        res2 = torch.cat([empty, x], dim=1)
        self.assertEqual(res1, res2)

        res1 = torch.cat([empty, empty], dim=1)
        self.assertEqual(res1, empty)

        with self.assertRaisesRegex(RuntimeError,
                                    'non-empty list of Tensors'):
            torch.cat([], dim=1)

    def test_cat_empty(self, device):
        dtype = torch.float32

        x = torch.randn((4, 3, 32, 32), dtype=dtype, device=device)
        empty = torch.randn((4, 0, 32, 32), dtype=dtype, device=device)

        res1 = torch.cat([x, empty], dim=1)
        res2 = torch.cat([empty, x], dim=1)
        self.assertEqual(res1, res2)

        res1 = torch.cat([empty, empty], dim=1)
        self.assertEqual(res1, empty)

        # check non-legacy-behavior (sizes don't match)
        empty = torch.randn((4, 0, 31, 32), dtype=dtype, device=device)
        self.assertRaises(RuntimeError, lambda: torch.cat([x, empty], dim=1))
        self.assertRaises(RuntimeError, lambda: torch.cat([empty, x], dim=1))

        # check non-legacy-behavior (dimensions don't match)
        empty = torch.randn((4, 0), dtype=dtype, device=device)
        self.assertRaises(RuntimeError, lambda: torch.cat([x, empty], dim=1))
        self.assertRaises(RuntimeError, lambda: torch.cat([empty, x], dim=1))

    def test_cat_out(self, device):
        x = torch.zeros((0), device=device)
        y = torch.randn((4, 6), device=device)

        with self.assertRaisesRegex(
                RuntimeError, r"unsupported operation:.* input tensor 0"):
            torch.cat([x, y], dim=0, out=x)

        with self.assertRaisesRegex(
                RuntimeError, r"unsupported operation:.* input tensor 1"):
            torch.cat([x, y], dim=0, out=y)

        z = torch.zeros((4, 6), device=device)
        with self.assertRaisesRegex(
                RuntimeError, r"unsupported operation:.* input tensor 1"):
            torch.cat([y, z], out=z[:2, :])

        w = y.view(-1).clone()
        a = torch.cat([w[:2], w[4:6]])
        b = torch.cat([w[:2], w[4:6]], out=w[6:10])
        self.assertEqual(a, b)
        self.assertEqual(w[:6], y.view(-1)[:6])

        # Case:
        # Reference: https://github.com/pytorch/pytorch/issues/49878
        for dim in [0, 1]:
            x = torch.zeros((10, 5, 2), device=device)

            random_length = random.randint(1, 4)
            y = x.narrow(dim, 0, x.shape[dim] - random_length)
            val = torch.full_like(y[0], 3., device=device)

            if dim == 0:
                self.assertTrue(y.is_contiguous())
            else:
                self.assertFalse(y.is_contiguous())

            torch.cat((val[None],) * y.shape[0], dim=0, out=y)

            expected_y = torch.cat((val[None],) * y.shape[0], dim=0)
            expected_x = torch.zeros((10, 5, 2), device=device)
            if dim == 0:
                expected_x[:x.shape[dim] - random_length, :, :] = expected_y
            elif dim == 1:
                expected_x[:, :x.shape[dim] - random_length, :] = expected_y

            self.assertEqual(y, expected_y)
            self.assertEqual(x, expected_x)

    def test_cat_out_channels_last(self, device):
        x = torch.randn((4, 3, 8, 8))
        y = torch.randn(x.shape)
        res1 = torch.cat((x, y))
        z = res1.clone().contiguous(memory_format=torch.channels_last)
        res2 = torch.cat((x, y), out=z)
        self.assertEqual(res1, res2)

    @onlyOnCPUAndCUDA
    def test_cat_in_channels_last(self, device):
        for dim in range(4):
            x = torch.randn((4, 15, 8, 8), device=device)
            y = torch.randn(x.shape, device=device)
            res1 = torch.cat((x, y), dim=dim)
            x = x.clone().contiguous(memory_format=torch.channels_last)
            y = y.clone().contiguous(memory_format=torch.channels_last)
            res2 = torch.cat((x, y), dim=dim)
            self.assertTrue(res2.is_contiguous(memory_format=torch.channels_last))
            self.assertEqual(res1, res2)

            # Size larger than grain size.
            x = torch.randn((4, 15, 256, 256), device=device)
            y = torch.randn(x.shape, device=device)
            res1 = torch.cat((x, y), dim=dim)
            x = x.clone().contiguous(memory_format=torch.channels_last)
            y = y.clone().contiguous(memory_format=torch.channels_last)
            res2 = torch.cat((x, y), dim=dim)
            self.assertTrue(res2.is_contiguous(memory_format=torch.channels_last))
            self.assertEqual(res1, res2)

    @onlyOnCPUAndCUDA
    def test_cat_preserve_channels_last(self, device):
        x = torch.randn((4, 3, 8, 8), device=device)
        y = torch.randn(x.shape, device=device)
        res1 = torch.cat((x, y))
        res2 = torch.cat((x.contiguous(memory_format=torch.channels_last), y.contiguous(memory_format=torch.channels_last)))
        self.assertEqual(res1, res2)
        self.assertTrue(res2.is_contiguous(memory_format=torch.channels_last))
        # discontiguous channels-last inputs
        x = torch.arange(24, dtype=torch.float, device=device).reshape(2, 2, 3, 2).to(memory_format=torch.channels_last)
        x1 = x[:, :, :2]
        x2 = x[:, :, 1:]
        res1 = torch.cat((x1, x2), dim=-1)
        res2 = torch.cat((x1.contiguous(), x2.contiguous()), dim=-1)
        self.assertEqual(res1, res2)
        self.assertTrue(res1.is_contiguous(memory_format=torch.channels_last))


    @onlyCUDA
    @deviceCountAtLeast(2)
    def test_cat_different_devices(self, devices):
        cuda0 = torch.randn((3, 3), device=devices[0])
        cuda1 = torch.randn((3, 3), device=devices[1])
        with self.assertRaisesRegex(RuntimeError,
                                    "Expected all tensors to be on the same device"):
            torch.cat((cuda0, cuda1))

        with self.assertRaisesRegex(RuntimeError,
                                    "Expected all tensors to be on the same device"):
            torch.cat((cuda0, cuda0), out=cuda1)

    @onlyCUDA
    def test_cat_stack_cross_devices(self, device):
        cuda = torch.randn((3, 3), device=device)
        cpu = torch.randn((3, 3), device='cpu')
        out_cpu = cpu.clone()
        out_cuda = cuda.clone()
        with self.assertRaisesRegex(RuntimeError,
                                    "Expected all tensors to be on the same device"):
            torch.cat((cuda, cpu))
        with self.assertRaisesRegex(RuntimeError,
                                    "Expected all tensors to be on the same device"):
            torch.cat((cpu, cuda))

        with self.assertRaisesRegex(RuntimeError,
                                    "Expected all tensors to be on the same device"):
            torch.cat((cpu, cuda), out=out_cuda)

        with self.assertRaisesRegex(RuntimeError,
                                    "Expected all tensors to be on the same device"):
            torch.cat((cpu, cpu), out=out_cuda)

        with self.assertRaisesRegex(RuntimeError,
                                    "Expected all tensors to be on the same device"):
            torch.cat((cuda, cuda), out=out_cpu)

        # Stack
        with self.assertRaisesRegex(RuntimeError,
                                    "Expected all tensors to be on the same device"):
            torch.stack((cuda, cpu))
        with self.assertRaisesRegex(RuntimeError,
                                    "Expected all tensors to be on the same device"):
            torch.stack((cpu, cuda))

        with self.assertRaisesRegex(RuntimeError,
                                    "Expected all tensors to be on the same device"):
            torch.stack((cpu, cuda), out=out_cuda)

        with self.assertRaisesRegex(RuntimeError,
                                    "Expected all tensors to be on the same device"):
            torch.stack((cpu, cpu), out=out_cuda)

        with self.assertRaisesRegex(RuntimeError,
                                    "Expected all tensors to be on the same device"):
            torch.stack((cuda, cuda), out=out_cpu)

    # TODO: reconcile with other cat tests
    # TODO: Compare with a NumPy reference instead of CPU
    @onlyCUDA
    def test_cat(self, device):
        SIZE = 10
        for dim in range(-3, 3):
            pos_dim = dim if dim >= 0 else 3 + dim
            x = torch.rand(13, SIZE, SIZE, device=device).transpose(0, pos_dim)
            y = torch.rand(17, SIZE, SIZE, device=device).transpose(0, pos_dim)
            z = torch.rand(19, SIZE, SIZE, device=device).transpose(0, pos_dim)

            res1 = torch.cat((x, y, z), dim)
            self.assertEqual(res1.narrow(pos_dim, 0, 13), x, atol=0, rtol=0)
            self.assertEqual(res1.narrow(pos_dim, 13, 17), y, atol=0, rtol=0)
            self.assertEqual(res1.narrow(pos_dim, 30, 19), z, atol=0, rtol=0)

        x = torch.randn(20, SIZE, SIZE, device=device)
        self.assertEqual(torch.cat(torch.split(x, 7)), x)
        self.assertEqual(torch.cat(torch.chunk(x, 7)), x)

        y = torch.randn(1, SIZE, SIZE, device=device)
        z = torch.cat([x, y])
        self.assertEqual(z.size(), (21, SIZE, SIZE))

    # TODO: update this test to compare against NumPy instead of CPU
    @onlyCUDA
    @dtypesIfCUDA(torch.half, torch.float, torch.double)
    @dtypes(torch.float, torch.double)
    def test_device_rounding(self, device, dtype):
        # test half-to-even
        a = [-5.8, -3.5, -2.3, -1.5, -0.5, 0.5, 1.5, 2.3, 3.5, 5.8]
        res = [-6., -4., -2., -2., 0., 0., 2., 2., 4., 6.]

        a_tensor = torch.tensor(a, device=device).round()
        res_tensor = torch.tensor(res, device='cpu')
        self.assertEqual(a_tensor, res_tensor)

    # Note: This test failed on XLA since its test cases are created by empty_strided which
    #       doesn't support overlapping sizes/strides in XLA impl
    @onlyOnCPUAndCUDA
    def test_like_fn_stride_proparation_vs_tensoriterator_unary_op(self, device):
        # Test like functions against tensoriterator based unary operator (exp) to
        # make sure the returned tensor from like function follows the same stride propergation
        # rule as what tensoriterator does for unary operator. The like function's  output strides
        # is computed on CPU side always, no need to test GPU here.

        def compare_helper_(like_fn, t):
            te = torch.exp(t)
            tl = like_fn(t)
            self.assertEqual(te.stride(), tl.stride())
            self.assertEqual(te.size(), tl.size())

        like_fns = [
            lambda t, **kwargs: torch.zeros_like(t, **kwargs),
            lambda t, **kwargs: torch.ones_like(t, **kwargs),
            lambda t, **kwargs: torch.randint_like(t, 10, 100, **kwargs),
            lambda t, **kwargs: torch.randint_like(t, 100, **kwargs),
            lambda t, **kwargs: torch.randn_like(t, **kwargs),
            lambda t, **kwargs: torch.rand_like(t, **kwargs),
            lambda t, **kwargs: torch.full_like(t, 7, **kwargs),
            lambda t, **kwargs: torch.empty_like(t, **kwargs)]

        # dense non-overlapping tensor,
        # non-dense non-overlapping sliced tensor
        # non-dense non-overlapping gapped tensor
        # non-dense non-overlapping 0 strided tensor
        # non-dense overlapping general tensor
        # non-dense overlapping sliced tensor
        # non-dense overlapping gapped tensor
        # non-dense overlapping 0 strided tensor
        # non-dense overlapping equal strides
        tset = (
            torch.randn(4, 3, 2, device=device),
            torch.randn(4, 3, 2, device=device)[:, :, ::2],
            torch.empty_strided((4, 3, 2), (10, 3, 1), device=device).fill_(1.0),
            torch.empty_strided((4, 3, 2), (10, 0, 3), device=device).fill_(1.0),
            torch.empty_strided((4, 3, 2), (10, 1, 2), device=device).fill_(1.0),
            torch.empty_strided((4, 3, 2), (4, 2, 1), device=device)[:, :, ::2].fill_(1.0),
            torch.empty_strided((4, 3, 2), (10, 1, 1), device=device).fill_(1.0),
            torch.empty_strided((4, 1, 1, 2), (10, 0, 0, 2), device=device).fill_(1.0),
            torch.empty_strided((4, 2, 3), (10, 3, 3), device=device).fill_(1.0))

        for like_fn in like_fns:
            for t in tset:
                for p in permutations(range(t.dim())):
                    tp = t.permute(p)
                    compare_helper_(like_fn, tp)

    def _hvd_split_helper(self, torch_fn, np_fn, op_name, inputs, device, dtype, dim):
        dimension_error_message = op_name + " requires a tensor with at least "
        divisibiliy_error_message = op_name + " attempted to split along dimension "

        for shape, arg in inputs:
            direction = dim - (len(shape) == 1 and dim == 1)
            bound = dim + 2 * (dim == 0) + (dim == 2)
            error_expected = len(shape) < bound or (not isinstance(arg, list) and shape[direction] % arg != 0)

            t = make_tensor(shape, device, dtype)
            t_np = t.cpu().numpy()

            if not error_expected:
                self.assertEqual(torch_fn(t, arg), np_fn(t_np, arg))
            else:
                self.assertRaises(RuntimeError, lambda: torch_fn(t, arg))
                self.assertRaises(ValueError, lambda: np_fn(t, arg))
                expected_error_message = dimension_error_message if len(shape) < bound else divisibiliy_error_message
                self.assertRaisesRegex(RuntimeError, expected_error_message, lambda: torch_fn(t, arg))

    @onlyOnCPUAndCUDA
    @dtypes(torch.long, torch.float32, torch.complex64)
    def test_hsplit(self, device, dtype):
        inputs = (
            ((), 3),
            ((), [2, 4, 6]),
            ((6,), 2),
            ((6,), 4),
            ((6,), [2, 5]),
            ((6,), [7, 9]),
            ((3, 8), 4),
            ((3, 8), 5),
            ((3, 8), [1, 5]),
            ((3, 8), [3, 8]),
            ((5, 5, 5), 2),
            ((5, 5, 5), [1, 4]),
            ((5, 0, 5), 3),
            ((5, 5, 0), [2, 6]),
        )
        self._hvd_split_helper(torch.hsplit, np.hsplit, "torch.hsplit", inputs, device, dtype, 1)

    @onlyOnCPUAndCUDA
    @dtypes(torch.long, torch.float32, torch.complex64)
    def test_vsplit(self, device, dtype):
        inputs = (
            ((6,), 2),
            ((6,), 4),
            ((6, 5), 2),
            ((6, 5), 4),
            ((6, 5), [1, 2, 3]),
            ((6, 5), [1, 5, 9]),
            ((6, 5, 5), 2),
            ((6, 0, 5), 2),
            ((5, 0, 5), [1, 5]),
        )
        self._hvd_split_helper(torch.vsplit, np.vsplit, "torch.vsplit", inputs, device, dtype, 0)

    @onlyOnCPUAndCUDA
    @dtypes(torch.long, torch.float32, torch.complex64)
    def test_dsplit(self, device, dtype):
        inputs = (
            ((6,), 4),
            ((6, 6), 3),
            ((5, 5, 6), 2),
            ((5, 5, 6), 4),
            ((5, 5, 6), [1, 2, 3]),
            ((5, 5, 6), [1, 5, 9]),
            ((5, 5, 0), 2),
            ((5, 0, 6), 4),
            ((5, 0, 6), [1, 2, 3]),
            ((5, 5, 6), [1, 5, 9]),
        )
        self._hvd_split_helper(torch.dsplit, np.dsplit, "torch.dsplit", inputs, device, dtype, 2)

    def _test_special_stacks(self, dim, at_least_dim, torch_fn, np_fn, device, dtype):
        # Test error for non-tuple argument
        t = torch.randn(10)
        with self.assertRaisesRegex(TypeError, "must be tuple of Tensors, not Tensor"):
            torch_fn(t)
        # Test error for a single array
        with self.assertRaisesRegex(TypeError, "must be tuple of Tensors, not Tensor"):
            torch_fn((t))

        # Test 0-D
        num_tensors = random.randint(1, 5)
        input_t = [torch.tensor(random.uniform(0, 10), device=device, dtype=dtype) for i in range(num_tensors)]
        actual = torch_fn(input_t)
        expected = np_fn([input.cpu().numpy() for input in input_t])
        self.assertEqual(actual, expected)

        for ndims in range(1, 5):
            base_shape = list(_rand_shape(ndims, min_size=1, max_size=5))
            for i in range(ndims):
                shape = list(base_shape)
                num_tensors = random.randint(1, 5)
                torch_input = []
                # Create tensors with shape being different along one axis only
                for param in range(num_tensors):
                    shape[i] = random.randint(1, 5)
                    torch_input.append(_generate_input(tuple(shape), dtype, device, with_extremal=False))

                # Determine if input tensors have valid dimensions.
                valid_dim = True
                for k in range(len(torch_input) - 1):
                    for tdim in range(ndims):
                        # Test whether all tensors have the same shape except in concatenating dimension
                        # Unless the number of dimensions is less than the corresponding at_least function dimension
                        # Since the original concatenating dimension would shift after applying at_least and would no
                        # longer be the concatenating dimension
                        if (ndims < at_least_dim or tdim != dim) and torch_input[k].size()[tdim] != torch_input[k + 1].size()[tdim]:
                            valid_dim = False

                # Special case for hstack is needed since hstack works differently when ndims is 1
                if valid_dim or (torch_fn is torch.hstack and ndims == 1):
                    # Valid dimensions, test against numpy
                    np_input = [input.cpu().numpy() for input in torch_input]
                    actual = torch_fn(torch_input)
                    expected = np_fn(np_input)
                    self.assertEqual(actual, expected)
                else:
                    # Invalid dimensions, test for error
                    with self.assertRaisesRegex(RuntimeError, "Sizes of tensors must match except in dimension"):
                        torch_fn(torch_input)
                    with self.assertRaises(ValueError):
                        np_input = [input.cpu().numpy() for input in torch_input]
                        np_fn(np_input)

    @onlyOnCPUAndCUDA
    @dtypes(*(torch.testing.get_all_int_dtypes() + torch.testing.get_all_fp_dtypes(include_bfloat16=False) +
              torch.testing.get_all_complex_dtypes()))
    def test_hstack_column_stack(self, device, dtype):
        ops = ((torch.hstack, np.hstack), (torch.column_stack, np.column_stack))
        for torch_op, np_op in ops:
            self._test_special_stacks(1, 1, torch_op, np_op, device, dtype)

        # Test torch.column_stack with combinations of 1D and 2D tensors input
        one_dim_tensor = torch.arange(0, 10).to(dtype=dtype, device=device)
        two_dim_tensor = torch.arange(0, 100).to(dtype=dtype, device=device).reshape(10, 10)
        inputs = two_dim_tensor, one_dim_tensor, two_dim_tensor, one_dim_tensor
        torch_result = torch.column_stack(inputs)

        np_inputs = [input.cpu().numpy() for input in inputs]
        np_result = np.column_stack(np_inputs)

        self.assertEqual(np_result,
                         torch_result)

    @onlyOnCPUAndCUDA
    @dtypes(*(torch.testing.get_all_int_dtypes() + torch.testing.get_all_fp_dtypes(include_bfloat16=False) +
              torch.testing.get_all_complex_dtypes()))
    def test_vstack_row_stack(self, device, dtype):
        ops = ((torch.vstack, np.vstack), (torch.row_stack, np.row_stack))
        for torch_op, np_op in ops:
            self._test_special_stacks(0, 2, torch_op, np_op, device, dtype)
            for i in range(5):
                # Test dimension change for 1D tensor of size (N) and 2D tensor of size (1, N)
                n = random.randint(1, 10)
                input_a = _generate_input((n,), dtype, device, with_extremal=False)
                input_b = _generate_input((1, n), dtype, device, with_extremal=False)
                torch_input = [input_a, input_b]
                np_input = [input.cpu().numpy() for input in torch_input]
                actual = torch_op(torch_input)
                expected = np_op(np_input)
                self.assertEqual(actual, expected)

    @onlyOnCPUAndCUDA
    @dtypes(*(torch.testing.get_all_int_dtypes() + torch.testing.get_all_fp_dtypes(include_bfloat16=False) +
              torch.testing.get_all_complex_dtypes()))
    def test_dstack(self, device, dtype):
        self._test_special_stacks(2, 3, torch.dstack, np.dstack, device, dtype)
        for i in range(5):
            # Test dimension change for 1D tensor of size (N), 2D tensor of size (1, N), and 3D tensor of size (1, N, 1)
            n = random.randint(1, 10)
            input_a = _generate_input((n,), dtype, device, with_extremal=False)
            input_b = _generate_input((1, n), dtype, device, with_extremal=False)
            input_c = _generate_input((1, n, 1), dtype, device, with_extremal=False)
            torch_input = [input_a, input_b, input_c]
            np_input = [input.cpu().numpy() for input in torch_input]
            actual = torch.dstack(torch_input)
            expected = np.dstack(np_input)
            self.assertEqual(actual, expected)

            # Test dimension change for 2D tensor of size (M, N) and 3D tensor of size (M, N, 1)
            m = random.randint(1, 10)
            n = random.randint(1, 10)
            input_a = _generate_input((m, n), dtype, device, with_extremal=False)
            input_b = _generate_input((m, n, 1), dtype, device, with_extremal=False)
            torch_input = [input_a, input_b]
            np_input = [input.cpu().numpy() for input in torch_input]
            actual = torch.dstack(torch_input)
            expected = np.dstack(np_input)
            self.assertEqual(actual, expected)

    @dtypes(torch.int32, torch.int64)
    def test_large_linspace(self, device, dtype):
        start = torch.iinfo(dtype).min
        end = torch.iinfo(dtype).max & ~0xfff
        steps = 15
        x = torch.linspace(start, end, steps, dtype=dtype, device=device)
        self.assertGreater(x[1] - x[0], (end - start) / steps)

    @dtypes(torch.float32, torch.float64)
    def test_unpack_double(self, device, dtype):
        # Reference: https://github.com/pytorch/pytorch/issues/33111
        vals = (2 ** 24 + 1, 2 ** 53 + 1,
                np.iinfo(np.int64).max, np.iinfo(np.uint64).max, np.iinfo(np.uint64).max + 1,
                -1e500, 1e500)
        for val in vals:
            t = torch.tensor(val, dtype=dtype, device=device)
            a = np.array(val, dtype=torch_to_numpy_dtype_dict[dtype])
            self.assertEqual(t, torch.from_numpy(a))

    def _float_to_int_conversion_helper(self, vals, device, dtype):
        a = np.array(vals, dtype=np.float32).astype(torch_to_numpy_dtype_dict[dtype])
        t = torch.tensor(vals, device=device, dtype=torch.float).to(dtype)
        self.assertEqual(torch.from_numpy(a), t.cpu())

    # Checks that float->integer casts don't produce undefined behavior errors.
    # Note: In C++, casting from a floating value to an integral dtype
    # is undefined if the floating point value is not within the integral
    # dtype's dynamic range. This can (and should) cause undefined behavior
    # errors with UBSAN. These casts are deliberate in PyTorch, however, and
    # NumPy has the same behavior.
    @onlyOnCPUAndCUDA
    @unittest.skipIf(IS_MACOS, "Test is broken on MacOS, see https://github.com/pytorch/pytorch/issues/38752")
    @unittest.skipIf(IS_PPC, "Test is borken on PowerPC, see https://github.com/pytorch/pytorch/issues/39671")
    @dtypes(torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
    def test_float_to_int_conversion_finite(self, device, dtype):
        min = torch.finfo(torch.float).min
        max = torch.finfo(torch.float).max

        # Note: CUDA max float -> integer conversion is divergent on some dtypes
        vals = (min, -2, -1.5, -.5, 0, .5, 1.5, 2, max)
        if self.device_type == 'cuda':
            if torch.version.hip:
                # HIP min float -> int64 conversion is divergent
                vals = (-2, -1.5, -.5, 0, .5, 1.5, 2)
            else:
                vals = (min, -2, -1.5, -.5, 0, .5, 1.5, 2)

        self._float_to_int_conversion_helper(vals, device, dtype)

    # Note: CUDA will fail this test on most dtypes, often dramatically.
    @onlyCPU
    @dtypes(torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
    def test_float_to_int_conversion_nonfinite(self, device, dtype):
        vals = (float('-inf'), float('inf'), float('nan'))

        self._float_to_int_conversion_helper(vals, device, dtype)

    # TODO: re-enable this test
    @unittest.skipIf(True, "real and imag not implemented for complex")
    @onlyOnCPUAndCUDA
    def test_complex_type_conversions(self, device):
        dtypes = [torch.float, torch.complex64, torch.complex128]
        for from_type in dtypes:
            for to_type in dtypes:
                from_tensor = torch.randn(4, dtype=from_type, device=device)
                to_tensor = from_tensor.to(to_type)
                if from_type.is_complex and not to_type.is_complex:
                    self.assertEqual(torch.real(from_tensor), to_tensor, exact_dtype=False)
                elif not from_type.is_complex and to_type.is_complex:
                    self.assertEqual(from_tensor, torch.real(to_tensor), exact_dtype=False)
                    self.assertEqual(torch.zeros_like(torch.imag(to_tensor)), torch.imag(to_tensor), exact_dtype=False)
                else:
                    self.assertEqual(from_tensor, to_tensor, exact_dtype=False)

    @slowTest
    @onlyCPU
    def test_cat_big(self, device):
        SIZE1 = 6500
        SIZE2 = 4500
        concat_list = []
        concat_list.append(torch.ones((SIZE1, 1024 * 512), dtype=torch.uint8, device=device))
        concat_list.append(torch.ones((SIZE2, 1024 * 512), dtype=torch.uint8, device=device))
        result = torch.cat(concat_list)
        self.assertEqual(result.size(0), SIZE1 + SIZE2)

    @onlyCPU
    def test_cat_bad_input_sizes(self, device):
        x = torch.randn(2, 1, device=device)
        y = torch.randn(2, 1, 1, device=device)
        z = torch.randn(2, 1, 1, device=device)
        self.assertRaises(RuntimeError, lambda: torch.cat([x, y, z]))

        x = torch.randn(2, 1, 2, device=device)
        y = torch.randn(2, 1, 1, device=device)
        z = torch.randn(2, 2, 1, device=device)
        self.assertRaises(RuntimeError, lambda: torch.cat([x, y, z], dim=1))

    @onlyCPU
    @dtypes(torch.half, torch.double, torch.int)
    def test_cat2(self, device, dtype):
        SIZE = 10
        for dim in range(-3, 3):
            pos_dim = dim if dim >= 0 else 3 + dim
            x = torch.randint(low=-100, high=100, size=(13, SIZE, SIZE), device=device).to(dtype).transpose(0, pos_dim)
            y = torch.randint(low=-100, high=100, size=(17, SIZE, SIZE), device=device).to(dtype).transpose(0, pos_dim)
            z = torch.randint(low=-100, high=100, size=(19, SIZE, SIZE), device=device).to(dtype).transpose(0, pos_dim)

            res1 = torch.cat((x, y, z), dim)
            self.assertEqual(res1.narrow(pos_dim, 0, 13), x, atol=0, rtol=0)
            self.assertEqual(res1.narrow(pos_dim, 13, 17), y, atol=0, rtol=0)
            self.assertEqual(res1.narrow(pos_dim, 30, 19), z, atol=0, rtol=0)

        x = torch.randint(low=-100, high=100, size=(20, SIZE, SIZE), device=device).to(dtype)
        self.assertEqual(torch.cat(torch.split(x, 7)), x)
        self.assertEqual(torch.cat(torch.chunk(x, 7)), x)

        y = torch.randint(low=-100, high=100, size=(1, SIZE, SIZE), device=device).to(dtype)
        z = torch.cat([x, y])
        self.assertEqual(z.size(), (21, SIZE, SIZE))

        self.assertRaises(RuntimeError, lambda: torch.cat([]))
        self.assertRaisesRegex(TypeError, 'got None', lambda: torch.cat([x, None]))

    @onlyCPU
    def test_cat_scalars(self, device):
        x = torch.tensor(0, device=device)
        y = torch.tensor(1, device=device)
        with self.assertRaisesRegex(RuntimeError, 'zero-dimensional.*cannot be concatenated'):
            torch.cat([x, y])

    def test_zeros_dtype_out_match(self, device):
        d = torch.tensor((2, 3), device=device, dtype=torch.double)
        self.assertRaises(RuntimeError, lambda: torch.zeros((2, 3), device=device, dtype=torch.float32, out=d))

    # TODO: update to work on CUDA, too
    @onlyCPU
    def test_trilu_indices(self, device):
        for test_args in tri_tests_args:
            _compare_trilu_indices(self, *test_args)
        run_additional_tri_tests(self, 'cpu')

        # test default options
        x = torch.ones(
            3, 3, dtype=torch.long, device='cpu', layout=torch.strided)
        self.assertEqual(
            x.tril(0).nonzero().transpose(0, 1), torch.tril_indices(3, 3))
        self.assertEqual(
            x.triu(0).nonzero().transpose(0, 1), torch.triu_indices(3, 3))

        # test stride 0 cases
        x = torch.ones(
            3, 1, 3, 3, dtype=torch.long, device='cpu', layout=torch.strided)
        output = x.triu(2).expand(3, 3, 3, 3)
        b = x.clone().expand(3, 3, 3, 3)
        self.assertEqual(b.triu(2), output)
        self.assertRaises(RuntimeError, lambda: b.triu_(2))

    # TODO: update to work on CUDA, too
    @onlyCPU
    def test_stack(self, device):
        for dtype in (torch.half, torch.double, torch.int):
            x = torch.randint(low=-100, high=100, size=(2, 3, 4)).to(dtype)
            y = torch.randint(low=-100, high=100, size=(2, 3, 4)).to(dtype)
            z = torch.randint(low=-100, high=100, size=(2, 3, 4)).to(dtype)
            for dim in range(4):
                res = torch.stack((x, y, z), dim)
                res_neg = torch.stack((x, y, z), dim - 4)
                expected_size = x.size()[:dim] + (3,) + x.size()[dim:]
                self.assertEqual(res, res_neg)
                self.assertEqual(res.size(), expected_size)
                self.assertEqual(res.select(dim, 0), x, atol=0, rtol=0)
                self.assertEqual(res.select(dim, 1), y, atol=0, rtol=0)
                self.assertEqual(res.select(dim, 2), z, atol=0, rtol=0)

    # TODO: update to work on CUDA, too
    @onlyCPU
    def test_stack_out(self, device):
        for dtype in (torch.half, torch.double, torch.int):
            x = torch.randint(low=-100, high=100, size=(2, 3, 4)).to(dtype)
            y = torch.randint(low=-100, high=100, size=(2, 3, 4)).to(dtype)
            z = torch.randint(low=-100, high=100, size=(2, 3, 4)).to(dtype)
            for dim in range(4):
                expected_size = x.size()[:dim] + (3,) + x.size()[dim:]
                res_out = x.new(expected_size)
                res_neg_out = x.new(expected_size)
                res_out_dp = res_out.data_ptr()
                res_out_neg_dp = res_neg_out.data_ptr()
                torch.stack((x, y, z), dim, out=res_out)
                torch.stack((x, y, z), dim - 4, out=res_neg_out)
                self.assertEqual(res_out, res_neg_out)
                self.assertEqual(res_out.size(), expected_size)
                self.assertEqual(res_out_dp, res_out.data_ptr())
                self.assertEqual(res_out_neg_dp, res_neg_out.data_ptr())
                self.assertEqual(res_out.select(dim, 0), x, atol=0, rtol=0)
                self.assertEqual(res_out.select(dim, 1), y, atol=0, rtol=0)
                self.assertEqual(res_out.select(dim, 2), z, atol=0, rtol=0)

    def test_repeat_interleave(self, device):
        x = torch.tensor([0, 1, 2, 3], device=device)
        expected = torch.tensor([1, 2, 2, 3, 3, 3], device=device)
        self.assertEqual(torch.repeat_interleave(x), expected)

        with self.assertRaises(RuntimeError):
            torch.repeat_interleave(torch.arange(4, device=device).reshape(2, 2))

        with self.assertRaises(RuntimeError):
            torch.repeat_interleave(torch.arange(4.0, device=device))

        with self.assertRaises(RuntimeError):
            torch.repeat_interleave(torch.tensor([1, 2, -1, 3, 4], device=device))

        y = torch.tensor([[1, 2], [3, 4]], device=device)

        y1_v1 = torch.repeat_interleave(y, 2)
        y1_v2 = torch.repeat_interleave(y, torch.tensor(2, device=device))
        y1_v3 = torch.repeat_interleave(y, torch.tensor([2], device=device))
        y1_expect = torch.tensor([1, 1, 2, 2, 3, 3, 4, 4], device=device)
        self.assertEqual(y1_v1, y1_expect)
        self.assertEqual(y1_v2, y1_expect)
        self.assertEqual(y1_v3, y1_expect)

        y2 = torch.repeat_interleave(y, 3, dim=1)
        y2_expect = torch.tensor([[1, 1, 1, 2, 2, 2],
                                  [3, 3, 3, 4, 4, 4]], device=device)
        self.assertEqual(y2, y2_expect)

        y3 = torch.repeat_interleave(y, torch.tensor([1, 2], device=device), dim=0)
        y3_expect = torch.tensor([[1, 2],
                                  [3, 4],
                                  [3, 4]], device=device)
        self.assertEqual(y3, y3_expect)

        with self.assertRaises(RuntimeError):
            torch.repeat_interleave(y, torch.tensor([1, 2, 3], device=device), dim=0)

        with self.assertRaises(RuntimeError):
            torch.repeat_interleave(y, torch.arange(9, device=device).reshape(3, 3), dim=0)

        # test zero sized dimension
        x = torch.zeros((5, 0), device=device)
        y = torch.repeat_interleave(x, repeats=3, dim=1)
        self.assertEqual(y, x.new_zeros(5, 0, device=device))

        x = torch.tensor([], dtype=torch.int64, device=device)
        y = torch.repeat_interleave(x, x)
        self.assertEqual(y, x)

    # TODO: udpate to work on CUDA, too
    @onlyCPU
    def test_new_methods_requires_grad(self, device):
        size = (10,)
        test_cases = [
            # method name, args
            ('new_full', [size, 1]),
            ('new_empty', [size]),
            ('new_zeros', [size]),
        ]
        for method_name, args in test_cases:
            x = torch.randn(size)
            for requires_grad in [True, False]:
                x_new = x.__getattribute__(method_name)(*args, requires_grad=requires_grad)
                self.assertEqual(x_new.requires_grad, requires_grad)
            x = torch.randint(10, size)
            with self.assertRaisesRegex(
                    RuntimeError,
                    r'Only Tensors of floating point and complex dtype can require gradients'):
                x_new = x.__getattribute__(method_name)(*args, requires_grad=True)

    # TODO: update to work on CUDA, too?
    @onlyCPU
    def test_tensor_from_sequence(self, device):
        class MockSequence(object):
            def __init__(self, lst):
                self.lst = lst

            def __len__(self):
                return len(self.lst)

            def __getitem__(self, item):
                raise TypeError

        class GoodMockSequence(MockSequence):
            def __getitem__(self, item):
                return self.lst[item]

        bad_mock_seq = MockSequence([1.0, 2.0, 3.0])
        good_mock_seq = GoodMockSequence([1.0, 2.0, 3.0])
        with self.assertRaisesRegex(ValueError, 'could not determine the shape'):
            torch.tensor(bad_mock_seq)
        self.assertEqual(torch.tensor([1.0, 2.0, 3.0]), torch.tensor(good_mock_seq))

    # TODO: update to work on CUDA, too?
    @onlyCPU
    def test_simple_scalar_cast(self, device):
        ok = [torch.tensor([1.5]), torch.zeros(1, 1, 1, 1)]
        ok_values = [1.5, 0]

        not_ok = map(torch.Tensor, [[], [1, 2], [[1, 2], [3, 4]]])

        for tensor, value in zip(ok, ok_values):
            self.assertEqual(int(tensor), int(value))
            self.assertEqual(float(tensor), float(value))
            self.assertEqual(complex(tensor), complex(value))

        self.assertEqual(complex(torch.tensor(1.5j)), 1.5j)

        for tensor in not_ok:
            self.assertRaises(ValueError, lambda: int(tensor))
            self.assertRaises(ValueError, lambda: float(tensor))
            self.assertRaises(ValueError, lambda: complex(tensor))

        self.assertRaises(RuntimeError, lambda: float(torch.tensor(1.5j)))
        self.assertRaises(RuntimeError, lambda: int(torch.tensor(1.5j)))

    # TODO: update to work on CUDA, too?
    @onlyCPU
    def test_offset_scalar_cast(self, device):
        x = torch.tensor([1., 2., 3.])
        y = x[2:]
        self.assertEqual(int(y), 3)

    def test_meshgrid(self, device):
        a = torch.tensor(1, device=device)
        b = torch.tensor([1, 2, 3], device=device)
        c = torch.tensor([1, 2], device=device)
        grid_a, grid_b, grid_c = torch.meshgrid([a, b, c])
        self.assertEqual(grid_a.shape, torch.Size([1, 3, 2]))
        self.assertEqual(grid_b.shape, torch.Size([1, 3, 2]))
        self.assertEqual(grid_c.shape, torch.Size([1, 3, 2]))
        grid_a2, grid_b2, grid_c2 = torch.meshgrid(a, b, c)
        self.assertEqual(grid_a2.shape, torch.Size([1, 3, 2]))
        self.assertEqual(grid_b2.shape, torch.Size([1, 3, 2]))
        self.assertEqual(grid_c2.shape, torch.Size([1, 3, 2]))
        expected_grid_a = torch.ones(1, 3, 2, dtype=torch.int64, device=device)
        expected_grid_b = torch.tensor([[[1, 1],
                                         [2, 2],
                                         [3, 3]]], device=device)
        expected_grid_c = torch.tensor([[[1, 2],
                                         [1, 2],
                                         [1, 2]]], device=device)
        self.assertTrue(grid_a.equal(expected_grid_a))
        self.assertTrue(grid_b.equal(expected_grid_b))
        self.assertTrue(grid_c.equal(expected_grid_c))
        self.assertTrue(grid_a2.equal(expected_grid_a))
        self.assertTrue(grid_b2.equal(expected_grid_b))
        self.assertTrue(grid_c2.equal(expected_grid_c))

    def test_cartesian_prod(self, device):
        a = torch.tensor([1], device=device)
        b = torch.tensor([1, 2, 3], device=device)
        c = torch.tensor([1, 2], device=device)
        prod = torch.cartesian_prod(a, b, c)
        expected = torch.tensor(list(product([a], b, c)), device=device)
        self.assertEqual(expected, prod)

        # test 0 size input
        d = torch.empty(0, dtype=b.dtype, device=device)
        prod = torch.cartesian_prod(a, b, c, d)
        expected = torch.empty(0, 4, dtype=b.dtype, device=device)
        self.assertEqual(expected, prod)

        # test single input
        prod = torch.cartesian_prod(b)
        self.assertEqual(b, prod)

    def test_combinations(self, device):
        a = torch.tensor([1, 2, 3], device=device)

        c = torch.combinations(a, r=1)
        expected = torch.tensor(list(combinations(a, r=1)), device=device)
        self.assertEqual(c, expected)

        c = torch.combinations(a, r=1, with_replacement=True)
        expected = torch.tensor(list(combinations_with_replacement(a, r=1)), device=device)
        self.assertEqual(c, expected)

        c = torch.combinations(a)
        expected = torch.tensor(list(combinations(a, r=2)), device=device)
        self.assertEqual(c, expected)

        c = torch.combinations(a, with_replacement=True)
        expected = torch.tensor(list(combinations_with_replacement(a, r=2)), device=device)
        self.assertEqual(c, expected)

        c = torch.combinations(a, r=3)
        expected = torch.tensor(list(combinations(a, r=3)), device=device)
        self.assertEqual(c, expected)

        c = torch.combinations(a, r=4)
        expected = torch.empty(0, 4, dtype=a.dtype, device=device)
        self.assertEqual(c, expected)

        c = torch.combinations(a, r=5)
        expected = torch.empty(0, 5, dtype=a.dtype, device=device)
        self.assertEqual(c, expected)

        # test empty imput
        a = torch.empty(0, device=device)
        c1 = torch.combinations(a)
        c2 = torch.combinations(a, with_replacement=True)
        expected = torch.empty(0, 2, dtype=a.dtype, device=device)
        self.assertEqual(c1, expected)
        self.assertEqual(c2, expected)

    def test_linlogspace_mem_overlap(self, device):
        x = torch.rand(1, device=device).expand(10)
        with self.assertRaisesRegex(RuntimeError, 'unsupported operation'):
            torch.linspace(1, 10, 10, out=x)

        with self.assertRaisesRegex(RuntimeError, 'unsupported operation'):
            torch.logspace(1, 10, 10, out=x)

    def test_ctor_with_numpy_array(self, device):
        correct_dtypes = [
            np.double,
            np.float,
            np.float16,
            np.int64,
            np.int32,
            np.int16,
            np.int8,
            np.uint8,
            np.bool,
        ]

        incorrect_byteorder = '>' if sys.byteorder == 'little' else '<'
        incorrect_dtypes = [incorrect_byteorder + t for t in ['d', 'f']]

        for dtype in correct_dtypes:
            array = np.array([1, 2, 3, 4], dtype=dtype)

            # Upcast
            tensor = torch.DoubleTensor(array).to(device)
            for i in range(len(array)):
                self.assertEqual(tensor[i], array[i])

            # Downcast (sometimes)
            tensor = torch.FloatTensor(array).to(device)
            for i in range(len(array)):
                self.assertEqual(tensor[i], array[i])

            tensor = torch.HalfTensor(array).to(device)
            for i in range(len(array)):
                self.assertEqual(tensor[i], array[i])

    @dtypes(torch.float, torch.double, torch.int8, torch.int16, torch.int32, torch.int64)
    def test_random(self, device, dtype):
        # This test is flaky with p<=(2/(ub-lb))^200=6e-36
        t = torch.empty(200, dtype=dtype, device=device)
        lb = 1
        ub = 4

        t.fill_(-1)
        t.random_(lb, ub)
        self.assertEqual(t.min(), lb)
        self.assertEqual(t.max(), ub - 1)

        t.fill_(-1)
        t.random_(ub)
        self.assertEqual(t.min(), 0)
        self.assertEqual(t.max(), ub - 1)

    def test_random_bool(self, device):
        size = 2000
        t = torch.empty(size, dtype=torch.bool, device=device)

        t.fill_(False)
        t.random_()
        self.assertEqual(t.min(), False)
        self.assertEqual(t.max(), True)
        self.assertTrue(0.4 < (t.eq(True)).to(torch.int).sum().item() / size < 0.6)

        t.fill_(True)
        t.random_()
        self.assertEqual(t.min(), False)
        self.assertEqual(t.max(), True)
        self.assertTrue(0.4 < (t.eq(True)).to(torch.int).sum().item() / size < 0.6)

    def test_random_from_to_bool(self, device):
        size = 2000

        int64_min_val = torch.iinfo(torch.int64).min
        int64_max_val = torch.iinfo(torch.int64).max

        min_val = 0
        max_val = 1

        froms = [int64_min_val, -42, min_val - 1, min_val, max_val, max_val + 1, 42]
        tos = [-42, min_val - 1, min_val, max_val, max_val + 1, 42, int64_max_val]

        for from_ in froms:
            for to_ in tos:
                t = torch.empty(size, dtype=torch.bool, device=device)
                if to_ > from_:
                    if not (min_val <= from_ <= max_val):
                        self.assertRaisesRegex(
                            RuntimeError,
                            "from is out of bounds",
                            lambda: t.random_(from_, to_)
                        )
                    elif not (min_val <= (to_ - 1) <= max_val):
                        self.assertRaisesRegex(
                            RuntimeError,
                            "to - 1 is out of bounds",
                            lambda: t.random_(from_, to_)
                        )
                    else:
                        t.random_(from_, to_)
                        range_ = to_ - from_
                        delta = 1
                        self.assertTrue(from_ <= t.to(torch.int).min() < (from_ + delta))
                        self.assertTrue((to_ - delta) <= t.to(torch.int).max() < to_)
                else:
                    self.assertRaisesRegex(
                        RuntimeError,
                        "random_ expects 'from' to be less than 'to', but got from=" + str(from_) + " >= to=" + str(to_),
                        lambda: t.random_(from_, to_)
                    )

    @dtypes(*(torch.testing.get_all_int_dtypes() + torch.testing.get_all_fp_dtypes()))
    def test_random_full_range(self, device, dtype):
        size = 2000
        alpha = 0.1

        int64_min_val = torch.iinfo(torch.int64).min
        int64_max_val = torch.iinfo(torch.int64).max

        if dtype == torch.double:
            fp_limit = 2**53
        elif dtype == torch.float:
            fp_limit = 2**24
        elif dtype == torch.half:
            fp_limit = 2**11
        elif dtype == torch.bfloat16:
            fp_limit = 2**8
        else:
            fp_limit = 0

        t = torch.empty(size, dtype=dtype, device=device)

        if dtype in [torch.float, torch.double, torch.half, torch.bfloat16]:
            from_ = int(max(-fp_limit, int64_min_val))
            to_inc_ = int(min(fp_limit, int64_max_val))
        else:
            from_ = int(max(torch.iinfo(dtype).min, int64_min_val))
            to_inc_ = int(min(torch.iinfo(dtype).max, int64_max_val))
        range_ = to_inc_ - from_ + 1

        t.random_(from_, None)
        delta = max(1, alpha * range_)
        self.assertTrue(from_ <= t.to(torch.double).min() < (from_ + delta))
        self.assertTrue((to_inc_ - delta) < t.to(torch.double).max() <= to_inc_)

    @dtypes(*(torch.testing.get_all_int_dtypes() + torch.testing.get_all_fp_dtypes()))
    def test_random_from_to(self, device, dtype):
        size = 2000
        alpha = 0.1

        int64_min_val = torch.iinfo(torch.int64).min
        int64_max_val = torch.iinfo(torch.int64).max

        if dtype in [torch.float, torch.double, torch.half]:
            min_val = int(max(torch.finfo(dtype).min, int64_min_val))
            max_val = int(min(torch.finfo(dtype).max, int64_max_val))
            froms = [min_val, -42, 0, 42]
            tos = [-42, 0, 42, max_val >> 1]
        elif dtype == torch.bfloat16:
            min_val = int64_min_val
            max_val = int64_max_val
            froms = [min_val, -42, 0, 42]
            tos = [-42, 0, 42, max_val >> 1]
        elif dtype == torch.uint8:
            min_val = torch.iinfo(dtype).min
            max_val = torch.iinfo(dtype).max
            froms = [int64_min_val, -42, min_val - 1, min_val, 42, max_val, max_val + 1]
            tos = [-42, min_val - 1, min_val, 42, max_val, max_val + 1, int64_max_val]
        elif dtype == torch.int64:
            min_val = int64_min_val
            max_val = int64_max_val
            froms = [min_val, -42, 0, 42]
            tos = [-42, 0, 42, max_val]
        else:
            min_val = torch.iinfo(dtype).min
            max_val = torch.iinfo(dtype).max
            froms = [int64_min_val, min_val - 1, min_val, -42, 0, 42, max_val, max_val + 1]
            tos = [min_val - 1, min_val, -42, 0, 42, max_val, max_val + 1, int64_max_val]

        if dtype == torch.double:
            fp_limit = 2**53
        elif dtype == torch.float:
            fp_limit = 2**24
        elif dtype == torch.half:
            fp_limit = 2**11
        elif dtype == torch.bfloat16:
            fp_limit = 2**8
        else:
            fp_limit = 0

        for from_ in froms:
            for to_ in tos:
                t = torch.empty(size, dtype=dtype, device=device)
                if to_ > from_:
                    if not (min_val <= from_ <= max_val):
                        self.assertRaisesRegex(
                            RuntimeError,
                            "from is out of bounds",
                            lambda: t.random_(from_, to_)
                        )
                    elif not (min_val <= (to_ - 1) <= max_val):
                        self.assertRaisesRegex(
                            RuntimeError,
                            "to - 1 is out of bounds",
                            lambda: t.random_(from_, to_)
                        )
                    else:
                        if dtype.is_floating_point and (
                                not (-fp_limit <= from_ <= fp_limit) or not (-fp_limit <= (to_ - 1) <= fp_limit)):
                            if not (-fp_limit <= from_ <= fp_limit):
                                self.assertWarnsRegex(UserWarning, "from is out of bounds",
                                                      lambda: t.random_(from_, to_))
                            if not (-fp_limit <= (to_ - 1) <= fp_limit):
                                self.assertWarnsRegex(UserWarning, "to - 1 is out of bounds",
                                                      lambda: t.random_(from_, to_))
                        else:
                            t.random_(from_, to_)
                            range_ = to_ - from_
                            delta = max(1, alpha * range_)
                            if dtype == torch.bfloat16:
                                # Less strict checks because of rounding errors
                                # TODO investigate rounding errors
                                self.assertTrue(from_ <= t.to(torch.double).min() < (from_ + delta))
                                self.assertTrue((to_ - delta) < t.to(torch.double).max() <= to_)
                            else:
                                self.assertTrue(from_ <= t.to(torch.double).min() < (from_ + delta))
                                self.assertTrue((to_ - delta) <= t.to(torch.double).max() < to_)
                else:
                    self.assertRaisesRegex(
                        RuntimeError,
                        "random_ expects 'from' to be less than 'to', but got from=" + str(from_) + " >= to=" + str(to_),
                        lambda: t.random_(from_, to_)
                    )

    @dtypes(*(torch.testing.get_all_int_dtypes() + torch.testing.get_all_fp_dtypes()))
    def test_random_to(self, device, dtype):
        size = 2000
        alpha = 0.1

        int64_min_val = torch.iinfo(torch.int64).min
        int64_max_val = torch.iinfo(torch.int64).max

        if dtype in [torch.float, torch.double, torch.half]:
            min_val = int(max(torch.finfo(dtype).min, int64_min_val))
            max_val = int(min(torch.finfo(dtype).max, int64_max_val))
            tos = [-42, 0, 42, max_val >> 1]
        elif dtype == torch.bfloat16:
            min_val = int64_min_val
            max_val = int64_max_val
            tos = [-42, 0, 42, max_val >> 1]
        elif dtype == torch.uint8:
            min_val = torch.iinfo(dtype).min
            max_val = torch.iinfo(dtype).max
            tos = [-42, min_val - 1, min_val, 42, max_val, max_val + 1, int64_max_val]
        elif dtype == torch.int64:
            min_val = int64_min_val
            max_val = int64_max_val
            tos = [-42, 0, 42, max_val]
        else:
            min_val = torch.iinfo(dtype).min
            max_val = torch.iinfo(dtype).max
            tos = [min_val - 1, min_val, -42, 0, 42, max_val, max_val + 1, int64_max_val]

        from_ = 0
        for to_ in tos:
            t = torch.empty(size, dtype=dtype, device=device)
            if to_ > from_:
                if not (min_val <= (to_ - 1) <= max_val):
                    self.assertRaisesRegex(
                        RuntimeError,
                        "to - 1 is out of bounds",
                        lambda: t.random_(from_, to_)
                    )
                else:
                    t.random_(to_)
                    range_ = to_ - from_
                    delta = max(1, alpha * range_)
                    if dtype == torch.bfloat16:
                        # Less strict checks because of rounding errors
                        # TODO investigate rounding errors
                        self.assertTrue(from_ <= t.to(torch.double).min() < (from_ + delta))
                        self.assertTrue((to_ - delta) < t.to(torch.double).max() <= to_)
                    else:
                        self.assertTrue(from_ <= t.to(torch.double).min() < (from_ + delta))
                        self.assertTrue((to_ - delta) <= t.to(torch.double).max() < to_)
            else:
                self.assertRaisesRegex(
                    RuntimeError,
                    "random_ expects 'from' to be less than 'to', but got from=" + str(from_) + " >= to=" + str(to_),
                    lambda: t.random_(from_, to_)
                )

    @dtypes(*(torch.testing.get_all_int_dtypes() + torch.testing.get_all_fp_dtypes()))
    def test_random_default(self, device, dtype):
        size = 2000
        alpha = 0.1

        if dtype == torch.float:
            to_inc = 1 << 24
        elif dtype == torch.double:
            to_inc = 1 << 53
        elif dtype == torch.half:
            to_inc = 1 << 11
        elif dtype == torch.bfloat16:
            to_inc = 1 << 8
        else:
            to_inc = torch.iinfo(dtype).max

        t = torch.empty(size, dtype=dtype, device=device)
        t.random_()
        self.assertTrue(0 <= t.to(torch.double).min() < alpha * to_inc)
        self.assertTrue((to_inc - alpha * to_inc) < t.to(torch.double).max() <= to_inc)

    # TODO: this test should be updated
    @onlyOnCPUAndCUDA
    def test_empty_full(self, device):
        torch_device = torch.device(device)
        device_type = torch_device.type

        if device_type == 'cpu':
            do_test_empty_full(self, torch.testing.get_all_math_dtypes('cpu'), torch.strided, torch_device)
        if device_type == 'cuda':
            do_test_empty_full(self, torch.testing.get_all_math_dtypes('cpu'), torch.strided, None)
            do_test_empty_full(self, torch.testing.get_all_math_dtypes('cpu'), torch.strided, torch_device)

    # TODO: this test should be updated
    @suppress_warnings
    @onlyOnCPUAndCUDA
    @deviceCountAtLeast(1)
    def test_tensor_device(self, devices):
        device_type = torch.device(devices[0]).type
        if device_type == 'cpu':
            self.assertEqual('cpu', torch.tensor(5).device.type)
            self.assertEqual('cpu',
                             torch.ones((2, 3), dtype=torch.float32, device='cpu').device.type)
            self.assertEqual('cpu',
                             torch.ones((2, 3), dtype=torch.float32, device='cpu:0').device.type)
            self.assertEqual('cpu',
                             torch.tensor(torch.ones((2, 3), dtype=torch.float32), device='cpu:0').device.type)
            self.assertEqual('cpu', torch.tensor(np.random.randn(2, 3), device='cpu').device.type)
        if device_type == 'cuda':
            self.assertEqual('cuda:0', str(torch.tensor(5).cuda(0).device))
            self.assertEqual('cuda:0', str(torch.tensor(5).cuda('cuda:0').device))
            self.assertEqual('cuda:0',
                             str(torch.tensor(5, dtype=torch.int64, device=0).device))
            self.assertEqual('cuda:0',
                             str(torch.tensor(5, dtype=torch.int64, device='cuda:0').device))
            self.assertEqual('cuda:0',
                             str(torch.tensor(torch.ones((2, 3), dtype=torch.float32), device='cuda:0').device))

            self.assertEqual('cuda:0', str(torch.tensor(np.random.randn(2, 3), device='cuda:0').device))

            for device in devices:
                with torch.cuda.device(device):
                    device_string = 'cuda:' + str(torch.cuda.current_device())
                    self.assertEqual(device_string,
                                     str(torch.tensor(5, dtype=torch.int64, device='cuda').device))

            with self.assertRaises(RuntimeError):
                torch.tensor(5).cuda('cpu')
            with self.assertRaises(RuntimeError):
                torch.tensor(5).cuda('cpu:0')

            if len(devices) > 1:
                self.assertEqual('cuda:1', str(torch.tensor(5).cuda(1).device))
                self.assertEqual('cuda:1', str(torch.tensor(5).cuda('cuda:1').device))
                self.assertEqual('cuda:1',
                                 str(torch.tensor(5, dtype=torch.int64, device=1).device))
                self.assertEqual('cuda:1',
                                 str(torch.tensor(5, dtype=torch.int64, device='cuda:1').device))
                self.assertEqual('cuda:1',
                                 str(torch.tensor(torch.ones((2, 3), dtype=torch.float32),
                                     device='cuda:1').device))

                self.assertEqual('cuda:1',
                                 str(torch.tensor(np.random.randn(2, 3), device='cuda:1').device))

    # TODO: this test should be updated
    @onlyOnCPUAndCUDA
    def test_as_strided_neg(self, device):
        error = r'as_strided: Negative strides are not supported at the ' \
                r'moment, got strides: \[-?[0-9]+(, -?[0-9]+)*\]'
        with self.assertRaisesRegex(RuntimeError, error):
            torch.as_strided(torch.ones(3, 3, device=device), (1, 1), (2, -1))
        with self.assertRaisesRegex(RuntimeError, error):
            torch.as_strided(torch.ones(14, device=device), (2,), (-11,))

    # TODO: this test should be updated
    def test_zeros(self, device):
        res1 = torch.zeros(100, 100, device=device)
        res2 = torch.tensor((), device=device)
        torch.zeros(100, 100, device=device, out=res2)

        self.assertEqual(res1, res2)

        boolTensor = torch.zeros(2, 2, device=device, dtype=torch.bool)
        expected = torch.tensor([[False, False], [False, False]],
                                device=device, dtype=torch.bool)
        self.assertEqual(boolTensor, expected)

        halfTensor = torch.zeros(1, 1, device=device, dtype=torch.half)
        expected = torch.tensor([[0.]], device=device, dtype=torch.float16)
        self.assertEqual(halfTensor, expected)

        bfloat16Tensor = torch.zeros(1, 1, device=device, dtype=torch.bfloat16)
        expected = torch.tensor([[0.]], device=device, dtype=torch.bfloat16)
        self.assertEqual(bfloat16Tensor, expected)

        complexTensor = torch.zeros(2, 2, device=device, dtype=torch.complex64)
        expected = torch.tensor([[0., 0.], [0., 0.]], device=device, dtype=torch.complex64)
        self.assertEqual(complexTensor, expected)

    # TODO: this test should be updated
    def test_zeros_out(self, device):
        shape = (3, 4)
        out = torch.zeros(shape, device=device)
        torch.zeros(shape, device=device, out=out)

        # change the dtype, layout, device
        with self.assertRaises(RuntimeError):
            torch.zeros(shape, device=device, dtype=torch.int64, out=out)
        with self.assertRaises(RuntimeError):
            torch.zeros(shape, device=device, layout=torch.sparse_coo, out=out)

        # leave them the same
        self.assertEqual(torch.zeros(shape, device=device),
                         torch.zeros(shape, device=device, dtype=out.dtype, out=out))
        self.assertEqual(torch.zeros(shape, device=device),
                         torch.zeros(shape, device=device, layout=torch.strided, out=out))
        self.assertEqual(torch.zeros(shape, device=device),
                         torch.zeros(shape, device=device, out=out))

    # TODO: this test should be updated
    def test_ones(self, device):
        res1 = torch.ones(100, 100, device=device)
        res2 = torch.tensor((), device=device)
        torch.ones(100, 100, device=device, out=res2)
        self.assertEqual(res1, res2)

        # test boolean tensor
        res1 = torch.ones(1, 2, device=device, dtype=torch.bool)
        expected = torch.tensor([[True, True]], device=device, dtype=torch.bool)
        self.assertEqual(res1, expected)

    # TODO: this test should be updated
    @onlyCPU
    def test_constructor_dtypes(self, device):
        default_type = torch.tensor([]).type()
        self.assertIs(torch.tensor([]).dtype, torch.get_default_dtype())

        self.assertIs(torch.uint8, torch.ByteTensor.dtype)
        self.assertIs(torch.float32, torch.FloatTensor.dtype)
        self.assertIs(torch.float64, torch.DoubleTensor.dtype)

        torch.set_default_tensor_type('torch.FloatTensor')
        self.assertIs(torch.float32, torch.get_default_dtype())
        self.assertIs(torch.FloatStorage, torch.Storage)

        torch.set_default_dtype(torch.float64)
        self.assertIs(torch.float64, torch.get_default_dtype())
        self.assertIs(torch.DoubleStorage, torch.Storage)

        torch.set_default_tensor_type(torch.FloatTensor)
        self.assertIs(torch.float32, torch.get_default_dtype())
        self.assertIs(torch.FloatStorage, torch.Storage)

        if torch.cuda.is_available():
            torch.set_default_tensor_type(torch.cuda.FloatTensor)
            self.assertIs(torch.float32, torch.get_default_dtype())
            self.assertIs(torch.float32, torch.cuda.FloatTensor.dtype)
            self.assertIs(torch.cuda.FloatStorage, torch.Storage)

            torch.set_default_dtype(torch.float64)
            self.assertIs(torch.float64, torch.get_default_dtype())
            self.assertIs(torch.cuda.DoubleStorage, torch.Storage)

        # don't support integral or sparse default types.
        self.assertRaises(TypeError, lambda: torch.set_default_tensor_type('torch.IntTensor'))
        self.assertRaises(TypeError, lambda: torch.set_default_dtype(torch.int64))

        # don't allow passing dtype to set_default_tensor_type
        self.assertRaises(TypeError, lambda: torch.set_default_tensor_type(torch.float32))

        torch.set_default_tensor_type(default_type)

    # TODO: this test should be updated
    @onlyCPU
    def test_constructor_device_legacy(self, device):
        self.assertRaises(RuntimeError, lambda: torch.FloatTensor(device='cuda'))
        self.assertRaises(RuntimeError, lambda: torch.FloatTensor(torch.Size([2, 3, 4]), device='cuda'))
        self.assertRaises(RuntimeError, lambda: torch.FloatTensor((2.0, 3.0), device='cuda'))

        self.assertRaises(RuntimeError, lambda: torch.Tensor(device='cuda'))
        self.assertRaises(RuntimeError, lambda: torch.Tensor(torch.Size([2, 3, 4]), device='cuda'))
        self.assertRaises(RuntimeError, lambda: torch.Tensor((2.0, 3.0), device='cuda'))

        # Tensor constructor/new with Tensor argument shouldn't work with device specified
        i = torch.tensor([1], device='cpu')
        self.assertRaises(RuntimeError, lambda: torch.Tensor(i, device='cpu'))
        self.assertRaises(RuntimeError, lambda: i.new(i, device='cpu'))
        self.assertRaises(RuntimeError, lambda: torch.Tensor(i, device='cuda'))
        self.assertRaises(RuntimeError, lambda: i.new(i, device='cuda'))

        x = torch.randn((3,), device='cpu')
        self.assertRaises(RuntimeError, lambda: x.new(device='cuda'))
        self.assertRaises(RuntimeError, lambda: x.new(torch.Size([2, 3, 4]), device='cuda'))
        self.assertRaises(RuntimeError, lambda: x.new((2.0, 3.0), device='cuda'))

        if torch.cuda.is_available():
            self.assertRaises(RuntimeError, lambda: torch.cuda.FloatTensor(device='cpu'))
            self.assertRaises(RuntimeError, lambda: torch.cuda.FloatTensor(torch.Size([2, 3, 4]), device='cpu'))
            self.assertRaises(RuntimeError, lambda: torch.cuda.FloatTensor((2.0, 3.0), device='cpu'))

            # Tensor constructor/new with Tensor argument shouldn't work with device specified
            i = torch.tensor([1], device='cuda')
            self.assertRaises(RuntimeError, lambda: torch.Tensor(i, device='cuda'))
            self.assertRaises(RuntimeError, lambda: i.new(i, device='cuda'))
            self.assertRaises(RuntimeError, lambda: torch.Tensor(i, device='cpu'))
            self.assertRaises(RuntimeError, lambda: i.new(i, device='cpu'))

            default_type = torch.Tensor().type()
            torch.set_default_tensor_type(torch.cuda.FloatTensor)
            self.assertRaises(RuntimeError, lambda: torch.Tensor(device='cpu'))
            self.assertRaises(RuntimeError, lambda: torch.Tensor(torch.Size([2, 3, 4]), device='cpu'))
            self.assertRaises(RuntimeError, lambda: torch.Tensor((2.0, 3.0), device='cpu'))
            torch.set_default_tensor_type(torch.cuda.FloatTensor)
            torch.set_default_tensor_type(default_type)

            x = torch.randn((3,), device='cuda')
            self.assertRaises(RuntimeError, lambda: x.new(device='cpu'))
            self.assertRaises(RuntimeError, lambda: x.new(torch.Size([2, 3, 4]), device='cpu'))
            self.assertRaises(RuntimeError, lambda: x.new((2.0, 3.0), device='cpu'))

    # TODO: this test should be updated
    @suppress_warnings
    @onlyCPU
    def test_tensor_factory(self, device):
        # TODO: This test probably doesn't make too much sense now that
        # torch.tensor has been established for a while; it makes more
        # sense to test the legacy behavior in terms of the new behavior
        expected = torch.Tensor([1, 1])
        # test data
        res1 = torch.tensor([1, 1])
        self.assertEqual(res1, expected, exact_dtype=False)

        res1 = torch.tensor([1, 1], dtype=torch.int)
        self.assertEqual(res1, expected, exact_dtype=False)
        self.assertIs(torch.int, res1.dtype)

        # test copy
        res2 = torch.tensor(expected)
        self.assertEqual(res2, expected)
        res2[1] = 2
        self.assertEqual(expected, torch.ones_like(expected))

        res2 = torch.tensor(expected, dtype=torch.int)
        self.assertEqual(res1, expected, exact_dtype=False)
        self.assertIs(torch.int, res1.dtype)

        # test copy with numpy
        for dtype in [np.float64, np.int64, np.int8, np.uint8]:
            a = np.array([5.]).astype(dtype)
            res1 = torch.tensor(a)
            self.assertEqual(5., res1[0].item())
            a[0] = 7.
            self.assertEqual(5., res1[0].item())

        # test boolean tensor
        a = torch.tensor([True, True, False, True, True], dtype=torch.bool)
        b = torch.tensor([-1, -1.1, 0, 1, 1.1], dtype=torch.bool)
        self.assertEqual(a, b)
        c = torch.tensor([-0.1, -1.1, 0, 1, 0.1], dtype=torch.bool)
        self.assertEqual(a, c)
        d = torch.tensor((-.3, 0, .3, 1, 3 / 7), dtype=torch.bool)
        e = torch.tensor((True, False, True, True, True), dtype=torch.bool)
        self.assertEqual(e, d)
        f = torch.tensor((-1, 0, -1.1, 1, 1.1), dtype=torch.bool)
        self.assertEqual(e, f)

        int64_max = torch.iinfo(torch.int64).max
        int64_min = torch.iinfo(torch.int64).min
        float64_max = torch.finfo(torch.float64).max
        float64_min = torch.finfo(torch.float64).min
        g_1 = torch.tensor((float('nan'), 0, int64_min, int64_max, int64_min - 1), dtype=torch.bool)
        self.assertEqual(e, g_1)
        g_2 = torch.tensor((int64_max + 1, 0, (int64_max + 1) * 2, (int64_max + 1) * 2 + 1, float64_min), dtype=torch.bool)
        self.assertEqual(e, g_2)
        g_3 = torch.tensor((float64_max, 0, float64_max + 1, float64_min - 1, float64_max + 1e291), dtype=torch.bool)
        self.assertEqual(e, g_3)

        h = torch.tensor([True, False, False, True, False, True, True], dtype=torch.bool)
        i = torch.tensor([1e-323, 1e-324, 0j, 1e-323j, 1e-324j, 1 + 2j, -1j], dtype=torch.bool)
        self.assertEqual(h, i)
        j = torch.tensor((True, True, True, True), dtype=torch.bool)
        k = torch.tensor((1e323, -1e323, float('inf'), -float('inf')), dtype=torch.bool)
        self.assertEqual(j, k)

    # TODO: this test should be updated
    @suppress_warnings
    @onlyCPU
    def test_tensor_factory_copy_var(self, device):
        def check_copy(copy, is_leaf, requires_grad, data_ptr=None):
            if data_ptr is None:
                data_ptr = copy.data_ptr
            self.assertEqual(copy, source, exact_dtype=False)
            self.assertTrue(copy.is_leaf == is_leaf)
            self.assertTrue(copy.requires_grad == requires_grad)
            self.assertTrue(copy.data_ptr == data_ptr)

        source = torch.randn(5, 5, dtype=torch.double, requires_grad=True)
        # test torch.tensor()
        check_copy(torch.tensor(source), True, False)
        check_copy(torch.tensor(source, requires_grad=False), True, False)
        check_copy(torch.tensor(source, requires_grad=True), True, True)

        # test tensor.new_tensor()
        copy = torch.randn(1)
        check_copy(copy.new_tensor(source), True, False)
        check_copy(copy.new_tensor(source, requires_grad=False), True, False)
        check_copy(copy.new_tensor(source, requires_grad=True), True, True)

        # test torch.as_tensor()
        check_copy(torch.as_tensor(source), source.is_leaf, source.requires_grad, source.data_ptr)  # not copy
        check_copy(torch.as_tensor(source, dtype=torch.float), False, True)  # copy and keep the graph

    # TODO: this test should be updated
    @onlyCPU
    def test_tensor_factory_type_inference(self, device):
        def test_inference(default_dtype):
            saved_dtype = torch.get_default_dtype()
            torch.set_default_dtype(default_dtype)
            default_complex_dtype = torch.complex64 if default_dtype == torch.float32 else torch.complex128
            self.assertIs(default_dtype, torch.tensor(()).dtype)
            self.assertIs(default_dtype, torch.tensor(5.).dtype)
            self.assertIs(torch.int64, torch.tensor(5).dtype)
            self.assertIs(torch.bool, torch.tensor(True).dtype)
            self.assertIs(torch.int32, torch.tensor(5, dtype=torch.int32).dtype)
            self.assertIs(default_dtype, torch.tensor(((7, 5), (9, 5.))).dtype)
            self.assertIs(default_dtype, torch.tensor(((5., 5), (3, 5))).dtype)
            self.assertIs(torch.int64, torch.tensor(((5, 3), (3, 5))).dtype)
            self.assertIs(default_complex_dtype, torch.tensor(((5, 3 + 2j), (3, 5 + 4j))).dtype)

            self.assertIs(torch.float64, torch.tensor(np.array(())).dtype)
            self.assertIs(torch.float64, torch.tensor(np.array(5.)).dtype)
            if np.array(5).dtype == np.int64:  # np long, which can be 4 bytes (e.g. on windows)
                self.assertIs(torch.int64, torch.tensor(np.array(5)).dtype)
            else:
                self.assertIs(torch.int32, torch.tensor(np.array(5)).dtype)
            self.assertIs(torch.uint8, torch.tensor(np.array(3, dtype=np.uint8)).dtype)
            self.assertIs(default_dtype, torch.tensor(((7, np.array(5)), (np.array(9), 5.))).dtype)
            self.assertIs(torch.float64, torch.tensor(((7, 5), (9, np.array(5.)))).dtype)
            self.assertIs(torch.int64, torch.tensor(((5, np.array(3)), (np.array(3), 5))).dtype)
            torch.set_default_dtype(saved_dtype)

        test_inference(torch.float64)
        test_inference(torch.float32)

    # TODO: this test should be updated
    @suppress_warnings
    @onlyCPU
    def test_new_tensor(self, device):
        expected = torch.autograd.Variable(torch.ByteTensor([1, 1]))
        # test data
        res1 = expected.new_tensor([1, 1])
        self.assertEqual(res1, expected)
        res1 = expected.new_tensor([1, 1], dtype=torch.int)
        self.assertEqual(res1, expected, exact_dtype=False)
        self.assertIs(torch.int, res1.dtype)

        # test copy
        res2 = expected.new_tensor(expected)
        self.assertEqual(res2, expected)
        res2[1] = 2
        self.assertEqual(expected, torch.ones_like(expected))
        res2 = expected.new_tensor(expected, dtype=torch.int)
        self.assertEqual(res2, expected, exact_dtype=False)
        self.assertIs(torch.int, res2.dtype)

        # test copy with numpy
        a = np.array([5.])
        res1 = torch.tensor(a)
        res1 = res1.new_tensor(a)
        self.assertEqual(5., res1[0].item())
        a[0] = 7.
        self.assertEqual(5., res1[0].item())

        if torch.cuda.device_count() >= 2:
            expected = expected.cuda(1)
            res1 = expected.new_tensor([1, 1])
            self.assertEqual(res1.get_device(), expected.get_device())
            res1 = expected.new_tensor([1, 1], dtype=torch.int)
            self.assertIs(torch.int, res1.dtype)
            self.assertEqual(res1.get_device(), expected.get_device())

            res2 = expected.new_tensor(expected)
            self.assertEqual(res2.get_device(), expected.get_device())
            res2 = expected.new_tensor(expected, dtype=torch.int)
            self.assertIs(torch.int, res1.dtype)
            self.assertEqual(res2.get_device(), expected.get_device())
            res2 = expected.new_tensor(expected, dtype=torch.int, device=0)
            self.assertIs(torch.int, res1.dtype)
            self.assertEqual(res2.get_device(), 0)

            res1 = expected.new_tensor(1)
            self.assertEqual(res1.get_device(), expected.get_device())
            res1 = expected.new_tensor(1, dtype=torch.int)
            self.assertIs(torch.int, res1.dtype)
            self.assertEqual(res1.get_device(), expected.get_device())

    # TODO: this test should be updated
    @onlyCPU
    def test_as_tensor(self, device):
        # from python data
        x = [[0, 1], [2, 3]]
        self.assertEqual(torch.tensor(x), torch.as_tensor(x))
        self.assertEqual(torch.tensor(x, dtype=torch.float32), torch.as_tensor(x, dtype=torch.float32))

        # python data with heterogeneous types
        z = [0, 'torch']
        with self.assertRaisesRegex(TypeError, "invalid data type"):
            torch.tensor(z)
            torch.as_tensor(z)

        # python data with self-referential lists
        z = [0]
        z += [z]
        with self.assertRaisesRegex(TypeError, "self-referential lists are incompatible"):
            torch.tensor(z)
            torch.as_tensor(z)

        z = [[1, 2], z]
        with self.assertRaisesRegex(TypeError, "self-referential lists are incompatible"):
            torch.tensor(z)
            torch.as_tensor(z)

        # from tensor (doesn't copy unless type is different)
        y = torch.tensor(x)
        self.assertIs(y, torch.as_tensor(y))
        self.assertIsNot(y, torch.as_tensor(y, dtype=torch.float32))
        if torch.cuda.is_available():
            self.assertIsNot(y, torch.as_tensor(y, device='cuda'))
            y_cuda = y.to('cuda')
            self.assertIs(y_cuda, torch.as_tensor(y_cuda))
            self.assertIs(y_cuda, torch.as_tensor(y_cuda, device='cuda'))

        # doesn't copy
        for dtype in [np.float64, np.int64, np.int8, np.uint8]:
            n = np.random.rand(5, 6).astype(dtype)
            n_astensor = torch.as_tensor(n)
            self.assertEqual(torch.tensor(n), n_astensor)
            n_astensor[0][0] = 25.7
            self.assertEqual(torch.tensor(n), n_astensor)

        # changing dtype causes copy
        n = np.random.rand(5, 6).astype(np.float32)
        n_astensor = torch.as_tensor(n, dtype=torch.float64)
        self.assertEqual(torch.tensor(n, dtype=torch.float64), n_astensor)
        n_astensor[0][1] = 250.8
        self.assertNotEqual(torch.tensor(n, dtype=torch.float64), n_astensor)

        # changing device causes copy
        if torch.cuda.is_available():
            n = np.random.randn(5, 6)
            n_astensor = torch.as_tensor(n, device='cuda')
            self.assertEqual(torch.tensor(n, device='cuda'), n_astensor)
            n_astensor[0][2] = 250.9
            self.assertNotEqual(torch.tensor(n, device='cuda'), n_astensor)

    # TODO: this test should be updated
    @suppress_warnings
    def test_range(self, device):
        res1 = torch.range(0, 1, device=device)
        res2 = torch.tensor((), device=device)
        torch.range(0, 1, device=device, out=res2)
        self.assertEqual(res1, res2, atol=0, rtol=0)

        # Check range for non-contiguous tensors.
        x = torch.zeros(2, 3, device=device)
        torch.range(0, 3, device=device, out=x.narrow(1, 1, 2))
        res2 = torch.tensor(((0, 0, 1), (0, 2, 3)), device=device, dtype=torch.float32)
        self.assertEqual(x, res2, atol=1e-16, rtol=0)

        # Check negative
        res1 = torch.tensor((1, 0), device=device, dtype=torch.float32)
        res2 = torch.tensor((), device=device)
        torch.range(1, 0, -1, device=device, out=res2)
        self.assertEqual(res1, res2, atol=0, rtol=0)

        # Equal bounds
        res1 = torch.ones(1, device=device)
        res2 = torch.tensor((), device=device)
        torch.range(1, 1, -1, device=device, out=res2)
        self.assertEqual(res1, res2, atol=0, rtol=0)
        torch.range(1, 1, 1, device=device, out=res2)
        self.assertEqual(res1, res2, atol=0, rtol=0)

    # TODO: this test should be updated
    def test_range_warning(self, device):
        with warnings.catch_warnings(record=True) as w:
            torch.range(0, 10, device=device)
            self.assertEqual(len(w), 1)

    # TODO: this test should be updated
    @onlyCPU
    def test_arange(self, device):
        res = torch.tensor(range(10000))
        res1 = torch.arange(0, 10000)  # Use a larger number so vectorized code can be triggered
        res2 = torch.tensor([], dtype=torch.int64)
        torch.arange(0, 10000, out=res2)
        self.assertEqual(res, res1, atol=0, rtol=0)
        self.assertEqual(res, res2, atol=0, rtol=0)

        # Vectorization on non-contiguous tensors
        res = torch.rand(3, 3, 300000).to(torch.int64)
        res = res.permute(2, 0, 1)
        torch.arange(0, 300000 * 3 * 3, out=res)
        self.assertEqual(res.flatten(), torch.arange(0, 300000 * 3 * 3))

        # Check arange with only one argument
        res1 = torch.arange(10)
        res2 = torch.arange(0, 10)
        self.assertEqual(res1, res2, atol=0, rtol=0)

        # Check arange for non-contiguous tensors.
        x = torch.zeros(2, 3)
        torch.arange(0, 4, out=x.narrow(1, 1, 2))
        res2 = torch.tensor(((0., 0., 1.), (0., 2., 3.)))
        self.assertEqual(x, res2, atol=1e-16, rtol=0)

        # Check negative
        res1 = torch.tensor((1., 0.))
        res2 = torch.tensor([])
        torch.arange(1, -1, -1, out=res2)
        self.assertEqual(res1, res2, atol=0, rtol=0)

        # Equal bounds
        res1 = torch.ones(1)
        res2 = torch.tensor([])
        torch.arange(1, 0, -1, out=res2)
        self.assertEqual(res1, res2, atol=0, rtol=0)
        torch.arange(1, 2, 1, out=res2)
        self.assertEqual(res1, res2, atol=0, rtol=0)

        # FloatTensor
        res1 = torch.arange(0.6, 0.89, 0.1, out=torch.FloatTensor())
        self.assertEqual(res1, [0.6, 0.7, 0.8])
        res1 = torch.arange(1, 10, 0.3, out=torch.FloatTensor())
        self.assertEqual(res1.size(0), 30)
        self.assertEqual(res1[0], 1)
        self.assertEqual(res1[29], 9.7)

        # DoubleTensor
        res1 = torch.arange(0.6, 0.89, 0.1, out=torch.DoubleTensor())
        self.assertEqual(res1, [0.6, 0.7, 0.8])
        res1 = torch.arange(1, 10, 0.3, out=torch.DoubleTensor())
        self.assertEqual(res1.size(0), 30)
        self.assertEqual(res1[0], 1)
        self.assertEqual(res1[29], 9.7)

        # Bool Input matching numpy semantics
        r = torch.arange(True)
        self.assertEqual(r[0], 0)
        r2 = torch.arange(False)
        self.assertEqual(len(r2), 0)
        self.assertEqual(r.dtype, torch.int64)
        self.assertEqual(r2.dtype, torch.int64)

        # Check that it's exclusive
        r = torch.arange(0, 5)
        self.assertEqual(r.min(), 0)
        self.assertEqual(r.max(), 4)
        self.assertEqual(r.numel(), 5)

        r = torch.arange(0, 5, 2)
        self.assertEqual(r.min(), 0)
        self.assertEqual(r.max(), 4)
        self.assertEqual(r.numel(), 3)

        r1 = torch.arange(0, 5 + 1e-6)
        # NB: without the dtype, we'll infer output type to be int64
        r2 = torch.arange(0, 5, dtype=torch.float32)
        r3 = torch.arange(0, 5 - 1e-6)
        self.assertEqual(r1[:-1], r2, atol=0, rtol=0)
        self.assertEqual(r2, r3, atol=0, rtol=0)

        r1 = torch.arange(10, -1 + 1e-6, -1)
        # NB: without the dtype, we'll infer output type to be int64
        r2 = torch.arange(10, -1, -1, dtype=torch.float32)
        r3 = torch.arange(10, -1 - 1e-6, -1)
        self.assertEqual(r1, r2, atol=0, rtol=0)
        self.assertEqual(r2, r3[:-1], atol=0, rtol=0)

        # Test Rounding Errors
        line = torch.zeros(size=(1, 49))
        self.assertWarnsRegex(UserWarning, 'The out tensor will be resized',
                              lambda: torch.arange(-1, 1, 2. / 49, dtype=torch.float32, out=line))
        self.assertEqual(line.shape, [50])

        x = torch.empty(1).expand(10)
        self.assertRaises(RuntimeError, lambda: torch.arange(10, out=x))
        msg = "unsupported range"
        self.assertRaisesRegex(RuntimeError, msg, lambda: torch.arange(0, float('inf')))
        self.assertRaisesRegex(RuntimeError, msg, lambda: torch.arange(float('inf')))

        for device in torch.testing.get_all_device_types():
            self.assertRaisesRegex(RuntimeError, msg, lambda: torch.arange(-5, float('nan'), device=device))
            # check with step size
            self.assertRaisesRegex(RuntimeError, msg, lambda: torch.arange(0, float('-inf'), -1, device=device))
            self.assertRaisesRegex(RuntimeError, msg, lambda: torch.arange(0, float('inf'), device=device))
            self.assertRaisesRegex(RuntimeError, msg, lambda: torch.arange(float('-inf'), 10, device=device))
            self.assertRaisesRegex(RuntimeError, msg, lambda: torch.arange(float('nan'), 10, device=device))
            self.assertRaisesRegex(RuntimeError, msg, lambda: torch.arange(float('inf'), device=device))
            self.assertRaisesRegex(RuntimeError, msg, lambda: torch.arange(float('nan'), device=device))

            self.assertRaisesRegex(
                RuntimeError, "overflow",
                lambda: torch.arange(1.175494351e-38, 3.402823466e+38, device=device))

            # check that it holds a consistent output shape on precision-cornered step sizes
            d = torch.arange(-4.0, 4.0, 0.01, dtype=torch.float32, device=device)
            self.assertEqual(d.shape[0], 800)

    # TODO: this test should be updated
    @onlyCPU
    def test_arange_inference(self, device):
        saved_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float32)
        # end only
        self.assertIs(torch.float32, torch.arange(1.).dtype)
        self.assertIs(torch.float32, torch.arange(torch.tensor(1.)).dtype)
        self.assertIs(torch.float32, torch.arange(torch.tensor(1., dtype=torch.float64)).dtype)

        self.assertIs(torch.int64, torch.arange(1).dtype)
        self.assertIs(torch.int64, torch.arange(torch.tensor(1)).dtype)
        self.assertIs(torch.int64, torch.arange(torch.tensor(1, dtype=torch.int16)).dtype)

        # start, end, [step]
        self.assertIs(torch.float32, torch.arange(1., 3).dtype)
        self.assertIs(torch.float32, torch.arange(torch.tensor(1., dtype=torch.float64), 3).dtype)
        self.assertIs(torch.float32, torch.arange(1, 3.).dtype)
        self.assertIs(torch.float32, torch.arange(torch.tensor(1, dtype=torch.int16), torch.tensor(3.)).dtype)
        self.assertIs(torch.float32, torch.arange(1, 3, 1.).dtype)
        self.assertIs(torch.float32,
                      torch.arange(torch.tensor(1),
                                   torch.tensor(3, dtype=torch.int16),
                                   torch.tensor(1., dtype=torch.float64)).dtype)

        self.assertIs(torch.int64, torch.arange(1, 3).dtype)
        self.assertIs(torch.int64, torch.arange(torch.tensor(1), 3).dtype)
        self.assertIs(torch.int64, torch.arange(torch.tensor(1), torch.tensor(3, dtype=torch.int16)).dtype)
        self.assertIs(torch.int64, torch.arange(1, 3, 1).dtype)
        self.assertIs(torch.int64,
                      torch.arange(torch.tensor(1),
                                   torch.tensor(3),
                                   torch.tensor(1, dtype=torch.int16)).dtype)
        torch.set_default_dtype(saved_dtype)

    # cannot call storage() on meta tensor
    @skipMeta
    def test_empty_strided(self, device):
        for shape in [(2, 3, 4), (0, 2, 0)]:
            # some of these cases are pretty strange, just verifying that if as_strided
            # allows them then empty_strided can as well.
            for strides in [(12, 4, 1), (2, 4, 6), (0, 0, 0)]:
                empty_strided = torch.empty_strided(shape, strides, device=device)
                # as_strided checks the storage size is big enough to support such a strided tensor;
                # instead of repeating this calculation, we just use empty_strided which does the same
                # calculation when setting the storage size.
                as_strided = torch.empty(empty_strided.storage().size(),
                                         device=device).as_strided(shape, strides)
                self.assertEqual(empty_strided.shape, as_strided.shape)
                self.assertEqual(empty_strided.stride(), as_strided.stride())

    def test_new_empty_strided(self, device):
        def _test(sizes, strides, dtype):
            x = torch.zeros(5, 5, dtype=dtype, device=device)
            result = x.new_empty_strided(sizes, strides)
            expected = torch.empty_strided(sizes, strides, dtype=x.dtype, device=x.device)
            self.assertEqual(result.shape, expected.shape)
            self.assertEqual(result.stride(), expected.stride())
            self.assertEqual(result.dtype, expected.dtype)
            self.assertEqual(result.device, expected.device)

        _test([2, 3], [3, 1], torch.float)
        _test([5, 3], [0, 1], torch.int)
        _test([], [], torch.float)

        # Some really weird cases
        for shape in [(2, 3, 4), (0, 2, 0)]:
            for strides in [(12, 4, 1), (2, 4, 6), (0, 0, 0)]:
                _test(shape, strides, torch.float)

    def test_strided_mismatched_stride_shape(self, device):
        for shape, strides in [((1, ), ()), ((1, 2), (1, ))]:
            with self.assertRaisesRegex(RuntimeError, "mismatch in length of strides and shape"):
                torch.tensor(0.42, device=device).as_strided(shape, strides)

            with self.assertRaisesRegex(RuntimeError, "mismatch in length of strides and shape"):
                torch.tensor(0.42, device=device).as_strided_(shape, strides)

    def test_empty_tensor_props(self, device):
        sizes = [(0,), (0, 3), (5, 0), (5, 0, 3, 0, 2), (0, 3, 0, 2), (0, 5, 0, 2, 0)]
        for size in sizes:
            x = torch.empty(tuple(size), device=device)
            self.assertEqual(size, x.shape)
            self.assertTrue(x.is_contiguous())
            size_ones_instead_of_zeros = (x if x != 0 else 1 for x in size)
            y = torch.empty(tuple(size_ones_instead_of_zeros), device=device)
            self.assertEqual(x.stride(), y.stride())

    def test_eye(self, device):
        for dtype in torch.testing.get_all_dtypes():
            if dtype == torch.bfloat16:
                continue
            # Test the RuntimeError is raised when either m or n is a negative number
            for n, m in ((-1, 1), (1, -1), (-1, -1)):
                with self.assertRaisesRegex(RuntimeError, 'must be greater or equal to'):
                    torch.eye(n, m, device=device, dtype=dtype)

            # Test when the `m` parameter is not provided
            for n in (3, 5, 7):
                res1 = torch.eye(n, device=device, dtype=dtype)
                naive_eye = torch.zeros(n, n, dtype=dtype, device=device)
                naive_eye.diagonal(dim1=-2, dim2=-1).fill_(1)
                self.assertEqual(naive_eye, res1)

                # Check eye_out outputs
                res2 = torch.empty(0, device=device, dtype=dtype)
                torch.eye(n, out=res2)
                self.assertEqual(res1, res2)

            for n, m in product([3, 5, 7], repeat=2):
                # Construct identity using diagonal and fill
                res1 = torch.eye(n, m, device=device, dtype=dtype)
                naive_eye = torch.zeros(n, m, dtype=dtype, device=device)
                naive_eye.diagonal(dim1=-2, dim2=-1).fill_(1)
                self.assertEqual(naive_eye, res1)

                # Check eye_out outputs
                res2 = torch.empty(0, device=device, dtype=dtype)
                torch.eye(n, m, out=res2)
                self.assertEqual(res1, res2)

    @precisionOverride({torch.float: 1e-8, torch.double: 1e-10})
    @dtypes(*(torch.testing.get_all_fp_dtypes(include_half=False, include_bfloat16=False) +
              torch.testing.get_all_complex_dtypes()))
    def test_linspace_vs_numpy(self, device, dtype):
        start = -0.0316082797944545745849609375 + (0.8888888888j if dtype.is_complex else 0)
        end = .0315315723419189453125 + (0.444444444444j if dtype.is_complex else 0)

        for steps in [1, 2, 3, 5, 11, 256, 257, 2**22]:
            t = torch.linspace(start, end, steps, device=device, dtype=dtype)
            a = np.linspace(start, end, steps, dtype=torch_to_numpy_dtype_dict[dtype])
            t = t.cpu()
            self.assertEqual(t, torch.from_numpy(a))
            self.assertTrue(t[0].item() == a[0])
            self.assertTrue(t[steps - 1].item() == a[steps - 1])

    def _test_linspace_logspace_complex_helper(self, torch_fn, np_fn, device, dtype):
        start = torch.randn(1, dtype=dtype).item()
        end = (start + torch.randn(1, dtype=dtype) + random.randint(5, 15)).item()

        def test_fn(torch_fn, numpy_fn, steps):
            t = torch_fn(start, end, steps, device=device)
            a = numpy_fn(start, end, steps, dtype=torch_to_numpy_dtype_dict[dtype])
            t = t.cpu()
            self.assertEqual(t, torch.from_numpy(a))

        for steps in [1, 2, 3, 5, 11, 256, 257, 2**22]:
            test_fn(torch.linspace, np.linspace, steps)

    @dtypes(torch.complex64)
    def test_linspace_vs_numpy_complex(self, device, dtype):
        self._test_linspace_logspace_complex_helper(torch.linspace, np.linspace,
                                                    device, dtype)

    @dtypes(torch.complex64)
    def test_logspace_vs_numpy_complex(self, device, dtype):
        self._test_linspace_logspace_complex_helper(torch.logspace, np.logspace,
                                                    device, dtype)

    @precisionOverride({torch.float: 1e-6, torch.double: 1e-10})
    @dtypes(*torch.testing.get_all_fp_dtypes(include_half=False, include_bfloat16=False))
    def test_logspace_vs_numpy(self, device, dtype):
        start = -0.0316082797944545745849609375
        end = .0315315723419189453125

        for steps in [1, 2, 3, 5, 11, 256, 257, 2**22]:
            t = torch.logspace(start, end, steps, device=device, dtype=dtype)
            a = np.logspace(start, end, steps, dtype=torch_to_numpy_dtype_dict[dtype])
            t = t.cpu()
            self.assertEqual(t, torch.from_numpy(a))
            self.assertEqual(t[0], a[0])
            self.assertEqual(t[steps - 1], a[steps - 1])

    def _linspace_logspace_warning_helper(self, op, device, dtype):
        with self.assertWarnsOnceRegex(UserWarning, "Not providing a value for .+"):
            op(0, 10, device=device, dtype=dtype)

    @dtypes(torch.float)
    def test_linspace_steps_warning(self, device, dtype):
        self._linspace_logspace_warning_helper(torch.linspace, device, dtype)

    @dtypes(torch.float)
    def test_logspace_steps_warning(self, device, dtype):
        self._linspace_logspace_warning_helper(torch.logspace, device, dtype)

    @onlyCUDA
    @largeTensorTest('16GB')
    def test_range_factories_64bit_indexing(self, device):
        bigint = 2 ** 31 + 1
        t = torch.arange(bigint, dtype=torch.long, device=device)
        self.assertEqual(t[-1].item(), bigint - 1)
        del t
        t = torch.linspace(0, 1, bigint, dtype=torch.float, device=device)
        self.assertEqual(t[-1].item(), 1)
        del t
        t = torch.logspace(0, 1, bigint, 2, dtype=torch.float, device=device)
        self.assertEqual(t[-1].item(), 2)
        del t

    @onlyOnCPUAndCUDA
    def test_tensor_ctor_device_inference(self, device):
        torch_device = torch.device(device)
        values = torch.tensor((1, 2, 3), device=device)

        # Tests tensor and as_tensor
        # Note: warnings are suppressed (suppresses warnings)
        for op in (torch.tensor, torch.as_tensor):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.assertEqual(op(values).device, torch_device)
                self.assertEqual(op(values, dtype=torch.float64).device, torch_device)

                if self.device_type == 'cuda':
                    with torch.cuda.device(device):
                        self.assertEqual(op(values.cpu()).device, torch.device('cpu'))

        # Tests sparse ctor
        indices = torch.tensor([[0, 1, 1],
                                [2, 0, 1],
                                [2, 1, 0]], device=device)
        sparse_size = (3, 3, 3)

        sparse_default = torch.sparse_coo_tensor(indices, values, sparse_size)
        self.assertEqual(sparse_default.device, torch_device)

        sparse_with_dtype = torch.sparse_coo_tensor(indices, values, sparse_size, dtype=torch.float64)
        self.assertEqual(sparse_with_dtype.device, torch_device)

        if self.device_type == 'cuda':
            with torch.cuda.device(device):
                sparse_with_dtype = torch.sparse_coo_tensor(indices.cpu(), values.cpu(),
                                                            sparse_size, dtype=torch.float64)
                self.assertEqual(sparse_with_dtype.device, torch.device('cpu'))

    @onlyOnCPUAndCUDA
    @precisionOverride({torch.bfloat16: 5e-2, torch.half: 1e-3})
    @unittest.skipIf(not TEST_SCIPY, "Scipy not found")
    @dtypesIfCUDA(torch.float, torch.double, torch.bfloat16, torch.half, torch.long)
    @dtypesIfCPU(torch.float, torch.double, torch.long)
    def test_signal_window_functions(self, device, dtype):
        import scipy.signal as signal

        def test(name, kwargs):
            torch_method = getattr(torch, name + '_window')
            if not dtype.is_floating_point:
                with self.assertRaisesRegex(RuntimeError, r'floating point'):
                    torch_method(3, dtype=dtype)
                return
            for size in [0, 1, 2, 5, 10, 50, 100, 1024, 2048]:
                for periodic in [True, False]:
                    res = torch_method(size, periodic=periodic, **kwargs, device=device, dtype=dtype)
                    # NB: scipy always returns a float64 result
                    ref = torch.from_numpy(signal.get_window((name, *(kwargs.values())), size, fftbins=periodic))
                    self.assertEqual(res, ref, exact_dtype=False)
            with self.assertRaisesRegex(RuntimeError, r'not implemented for sparse types'):
                torch_method(3, layout=torch.sparse_coo)
            self.assertTrue(torch_method(3, requires_grad=True).requires_grad)
            self.assertFalse(torch_method(3).requires_grad)

        for window in ['hann', 'hamming', 'bartlett', 'blackman']:
            test(window, kwargs={})

        for num_test in range(50):
            test('kaiser', kwargs={'beta': random.random() * 30})

    def test_tensor_factories_empty(self, device):
        # ensure we can create empty tensors from each factory function
        shapes = [(5, 0, 1), (0,), (0, 0, 1, 0, 2, 0, 0)]

        for shape in shapes:
            for dt in torch.testing.get_all_dtypes():

                self.assertEqual(shape, torch.zeros(shape, device=device, dtype=dt).shape)
                self.assertEqual(shape, torch.zeros_like(torch.zeros(shape, device=device, dtype=dt)).shape)
                self.assertEqual(shape, torch.full(shape, 3, device=device, dtype=dt).shape)
                self.assertEqual(shape, torch.full_like(torch.zeros(shape, device=device, dtype=dt), 3).shape)
                self.assertEqual(shape, torch.ones(shape, device=device, dtype=dt).shape)
                self.assertEqual(shape, torch.ones_like(torch.zeros(shape, device=device, dtype=dt)).shape)
                self.assertEqual(shape, torch.empty(shape, device=device, dtype=dt).shape)
                self.assertEqual(shape, torch.empty_like(torch.zeros(shape, device=device, dtype=dt)).shape)
                self.assertEqual(shape, torch.empty_strided(shape, (0,) * len(shape), device=device, dtype=dt).shape)

                if dt == torch.bool:
                    self.assertEqual(shape, torch.randint(2, shape, device=device, dtype=dt).shape)
                    self.assertEqual(shape, torch.randint_like(torch.zeros(shape, device=device, dtype=dt), 2).shape)
                elif dt.is_complex:
                    self.assertRaises(RuntimeError, lambda: torch.randint(6, shape, device=device, dtype=dt).shape)
                else:
                    self.assertEqual(shape, torch.randint(6, shape, device=device, dtype=dt).shape)
                    self.assertEqual(shape, torch.randint_like(torch.zeros(shape, device=device, dtype=dt), 6).shape)

                if dt not in {torch.double, torch.float, torch.half, torch.bfloat16, torch.complex64, torch.complex128}:
                    self.assertRaises(RuntimeError, lambda: torch.rand(shape, device=device, dtype=dt).shape)

                if dt == torch.double or dt == torch.float or dt.is_complex:
                    self.assertEqual(shape, torch.randn(shape, device=device, dtype=dt).shape)
                    self.assertEqual(shape, torch.randn_like(torch.zeros(shape, device=device, dtype=dt)).shape)

        self.assertEqual((0,), torch.arange(0, device=device).shape)
        self.assertEqual((0, 0), torch.eye(0, device=device).shape)
        self.assertEqual((0, 0), torch.eye(0, 0, device=device).shape)
        self.assertEqual((5, 0), torch.eye(5, 0, device=device).shape)
        self.assertEqual((0, 5), torch.eye(0, 5, device=device).shape)
        self.assertEqual((0,), torch.linspace(1, 1, 0, device=device).shape)
        self.assertEqual((0,), torch.logspace(1, 1, 0, device=device).shape)
        self.assertEqual((0,), torch.randperm(0, device=device).shape)
        self.assertEqual((0,), torch.bartlett_window(0, device=device).shape)
        self.assertEqual((0,), torch.bartlett_window(0, periodic=False, device=device).shape)
        self.assertEqual((0,), torch.hamming_window(0, device=device).shape)
        self.assertEqual((0,), torch.hann_window(0, device=device).shape)
        self.assertEqual((0,), torch.kaiser_window(0, device=device).shape)
        self.assertEqual((1, 1, 0), torch.tensor([[[]]], device=device).shape)
        self.assertEqual((1, 1, 0), torch.as_tensor([[[]]], device=device).shape)

    @onlyCUDA
    def test_tensor_factory_gpu_type_inference(self, device):
        saved_type = torch.tensor([]).type()
        torch.set_default_tensor_type(torch.cuda.DoubleTensor)
        torch.set_default_dtype(torch.float32)
        self.assertIs(torch.float32, torch.tensor(0.).dtype)
        self.assertEqual(torch.device(device), torch.tensor(0.).device)
        torch.set_default_dtype(torch.float64)
        self.assertIs(torch.float64, torch.tensor(0.).dtype)
        self.assertEqual(torch.device(device), torch.tensor(0.).device)
        torch.set_default_tensor_type(saved_type)

    @onlyCUDA
    def test_tensor_factory_gpu_type(self, device):
        saved_type = torch.tensor([]).type()
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
        x = torch.zeros((5, 5))
        self.assertIs(torch.float32, x.dtype)
        self.assertTrue(x.is_cuda)
        torch.set_default_tensor_type(torch.cuda.DoubleTensor)
        x = torch.zeros((5, 5))
        self.assertIs(torch.float64, x.dtype)
        self.assertTrue(x.is_cuda)
        torch.set_default_tensor_type(saved_type)

    @skipCPUIf(True, 'compares device with cpu')
    @dtypes(torch.int, torch.long, torch.float, torch.double)
    def test_arange_device_vs_cpu(self, device, dtype):
        cpu_tensor = torch.arange(0, 10, dtype=dtype, device='cpu')
        device_tensor = torch.arange(0, 10, dtype=dtype, device=device)
        self.assertEqual(cpu_tensor, device_tensor)

    @onlyCUDA
    def test_arange_bfloat16(self, device):
        ref_tensor = torch.tensor([0, 1, 2, 3], dtype=torch.bfloat16, device=device)
        bfloat16_tensor = torch.arange(0, 4, dtype=torch.bfloat16, device=device)
        self.assertEqual(ref_tensor, bfloat16_tensor)

        # step=2
        ref_tensor = torch.tensor([0, 2, 4], dtype=torch.bfloat16, device=device)
        bfloat16_tensor = torch.arange(0, 6, step=2, dtype=torch.bfloat16, device=device)
        self.assertEqual(ref_tensor, bfloat16_tensor)

    @dtypes(*torch.testing.get_all_dtypes(include_bool=False, include_half=False))
    @dtypesIfCUDA(*torch.testing.get_all_dtypes(include_bool=False, include_half=True))
    def test_linspace(self, device, dtype):
        _from = random.random()
        to = _from + random.random()
        res1 = torch.linspace(_from, to, 137, device=device, dtype=dtype)
        res2 = torch.tensor((), device=device, dtype=dtype)
        torch.linspace(_from, to, 137, dtype=dtype, out=res2)
        self.assertEqual(res1, res2, atol=0, rtol=0)

        # small tensor
        self.assertEqual(torch.linspace(10, 20, 11, device=device, dtype=dtype),
                         torch.tensor(list(range(10, 21)), device=device, dtype=dtype))
        # large tensor
        if dtype not in (torch.int8, torch.uint8):
            self.assertEqual(torch.linspace(10, 2000, 1991, device=device, dtype=dtype),
                             torch.tensor(list(range(10, 2001)), device=device, dtype=dtype))

        # Vectorization on non-contiguous tensors
        if dtype not in (torch.int8, torch.uint8):  # int8 and uint8 are too small for this test
            res = torch.rand(3, 3, 1000, device=device).to(dtype)
            res = res.permute(2, 0, 1)
            torch.linspace(0, 1000 * 3 * 3, 1000 * 3 * 3, out=res)
            self.assertEqual(res.flatten(), torch.linspace(0, 1000 * 3 * 3, 1000 * 3 * 3, device=device, dtype=dtype))

        self.assertRaises(RuntimeError, lambda: torch.linspace(0, 1, -1, device=device, dtype=dtype))
        # steps = 1
        self.assertEqual(torch.linspace(0, 1, 1, device=device, dtype=dtype),
                         torch.zeros(1, device=device, dtype=dtype), atol=0, rtol=0)
        # steps = 0
        self.assertEqual(torch.linspace(0, 1, 0, device=device, dtype=dtype).numel(), 0, atol=0, rtol=0)

        # Check linspace for generating the correct output for each dtype.
        start = 0 if dtype == torch.uint8 else -100
        expected_lin = torch.tensor([start + .5 * i for i in range(401)], device=device, dtype=torch.double)
        actual_lin = torch.linspace(start, start + 200, 401, device=device, dtype=dtype)
        # If on GPU, allow for minor error depending on dtype.
        tol = 0.
        if device != 'cpu':
            if dtype == torch.half:
                tol = 1e-1
            elif dtype == torch.float:
                tol = 1e-5
            elif dtype == torch.double:
                tol = 1e-10

        self.assertEqual(expected_lin.to(dtype), actual_lin, atol=tol, rtol=0)

        # Check linspace for generating with start > end.
        self.assertEqual(torch.linspace(2, 0, 3, device=device, dtype=dtype),
                         torch.tensor((2, 1, 0), device=device, dtype=dtype),
                         atol=0, rtol=0)

        # Check for race condition (correctness when applied on a large tensor).
        if dtype not in (torch.int8, torch.uint8, torch.int16, torch.half, torch.bfloat16):
            y = torch.linspace(0, 999999 + (999999j if dtype.is_complex else 0),
                               1000000, device=device, dtype=dtype)
            if dtype.is_complex:
                cond = torch.logical_and(y[:-1].real < y[1:].real, y[:-1].imag < y[1:].imag)
            else:
                cond = y[:-1] < y[1:]
            correct = all(cond)
            self.assertTrue(correct)

        # Check linspace for non-contiguous tensors.
        x = torch.zeros(2, 3, device=device, dtype=dtype)
        y = torch.linspace(0, 3, 4, out=x.narrow(1, 1, 2), dtype=dtype)
        self.assertEqual(x, torch.tensor(((0, 0, 1), (0, 2, 3)), device=device, dtype=dtype), atol=0, rtol=0)

    def _test_linspace_logspace_deduction_helper(self, fn, device):
        for start, end in [(1, 2), (1., 2), (1., -2.), (1j, 2j), (0., 2j), (1j, 2)]:
            dtype = torch.float32
            if isinstance(start, complex) or isinstance(end, complex):
                dtype = torch.cfloat

            if dtype == torch.cfloat:
                # TODO(kshitij12345): Fix unnecessary warning
                # Reference: https://github.com/pytorch/pytorch/issues/53171
                with self.assertWarnsRegex(UserWarning,
                                           "As either `start` or `stop` is complex"):
                    self.assertEqual(fn(start, end, steps=100, device=device).dtype, dtype)
            else:
                self.assertEqual(fn(start, end, steps=100, device=device).dtype, dtype)

    def test_linspace_deduction(self, device):
        # Test deduction from input parameters.
        self._test_linspace_logspace_deduction_helper(torch.linspace, device)

    def test_logspace_deduction(self, device):
        # Test deduction from input parameters.
        self._test_linspace_logspace_deduction_helper(torch.logspace, device)

    # The implementation of linspace+logspace goes through a different path
    # when the steps arg is equal to 0 or 1. For other values of `steps`
    # they call specialized linspace (or logspace) kernels.
    LINSPACE_LOGSPACE_SPECIAL_STEPS = [0, 1]

    # NOTE [Linspace+Logspace precision override]
    # Our Linspace and logspace torch.half CUDA kernels are not very precise.
    # Since linspace/logspace are deterministic, we can compute an expected
    # amount of error (by testing without a precision override), adding a tiny
    # amount (EPS) to that, and using that value as the override.
    LINSPACE_LOGSPACE_EXTRA_EPS = 1e-5

    # Compares linspace device vs. cpu
    def _test_linspace(self, device, dtype, steps):
        a = torch.linspace(0, 10, steps=steps, dtype=dtype, device=device)
        b = torch.linspace(0, 10, steps=steps)
        self.assertEqual(a, b, exact_dtype=False)

    # See NOTE [Linspace+Logspace precision override]
    @skipCPUIf(True, "compares with CPU")
    @precisionOverride({torch.half: 0.0039 + LINSPACE_LOGSPACE_EXTRA_EPS})
    @dtypes(*(torch.testing.get_all_fp_dtypes() + torch.testing.get_all_complex_dtypes()))
    def test_linspace_device_vs_cpu(self, device, dtype):
        self._test_linspace(device, dtype, steps=10)

    @skipCPUIf(True, "compares with CPU")
    @dtypes(*(torch.testing.get_all_fp_dtypes() + torch.testing.get_all_complex_dtypes()))
    def test_linspace_special_steps(self, device, dtype):
        for steps in self.LINSPACE_LOGSPACE_SPECIAL_STEPS:
            self._test_linspace(device, dtype, steps=steps)

    # Compares logspace device vs cpu
    def _test_logspace(self, device, dtype, steps):
        a = torch.logspace(1, 1.1, steps=steps, dtype=dtype, device=device)
        b = torch.logspace(1, 1.1, steps=steps)
        self.assertEqual(a, b, exact_dtype=False)

    # Compares logspace device vs cpu
    def _test_logspace_base2(self, device, dtype, steps):
        a = torch.logspace(1, 1.1, steps=steps, base=2, dtype=dtype, device=device)
        b = torch.logspace(1, 1.1, steps=steps, base=2)
        self.assertEqual(a, b, exact_dtype=False)

    # See NOTE [Linspace+Logspace precision override]
    @skipCPUIf(True, "compares with CPU")
    @precisionOverride({torch.half: 0.025 + LINSPACE_LOGSPACE_EXTRA_EPS})
    @dtypesIfCUDA(torch.half, torch.float, torch.double)
    @dtypes(torch.float, torch.double)
    def test_logspace_device_vs_cpu(self, device, dtype):
        self._test_logspace(device, dtype, steps=10)

    # See NOTE [Linspace+Logspace precision override]
    @skipCPUIf(True, "compares with CPU")
    @precisionOverride({torch.half: 0.0201 + LINSPACE_LOGSPACE_EXTRA_EPS})
    @dtypesIfCUDA(torch.half, torch.float, torch.double)
    @dtypes(torch.float, torch.double)
    def test_logspace_base2(self, device, dtype):
        self._test_logspace_base2(device, dtype, steps=10)

    @skipCPUIf(True, "compares with CPU")
    @dtypesIfCUDA(torch.half, torch.float, torch.double)
    @dtypes(torch.float, torch.double)
    def test_logspace_special_steps(self, device, dtype):
        for steps in self.LINSPACE_LOGSPACE_SPECIAL_STEPS:
            self._test_logspace(device, dtype, steps=steps)
            self._test_logspace_base2(device, dtype, steps=steps)

    @dtypes(*torch.testing.get_all_dtypes(include_bool=False, include_half=False, include_complex=False))
    @dtypesIfCUDA(*((torch.testing.get_all_int_dtypes() + [torch.float32, torch.float16, torch.bfloat16])
                    if TEST_WITH_ROCM
                    else torch.testing.get_all_dtypes(include_bool=False, include_half=True, include_complex=False)))
    def test_logspace(self, device, dtype):
        _from = random.random()
        to = _from + random.random()
        res1 = torch.logspace(_from, to, 137, device=device, dtype=dtype)
        res2 = torch.tensor((), device=device, dtype=dtype)
        torch.logspace(_from, to, 137, device=device, dtype=dtype, out=res2)
        self.assertEqual(res1, res2, atol=0, rtol=0)
        self.assertRaises(RuntimeError, lambda: torch.logspace(0, 1, -1, device=device, dtype=dtype))
        self.assertEqual(torch.logspace(0, 1, 1, device=device, dtype=dtype),
                         torch.ones(1, device=device, dtype=dtype), atol=0, rtol=0)

        # Check precision - start, stop and base are chosen to avoid overflow
        # steps is chosen so that step size is not subject to rounding error
        # a tolerance is needed for gpu tests due to differences in computation
        atol = None
        rtol = None
        if self.device_type == 'cpu':
            atol = 0
            rtol = 0
        self.assertEqual(torch.tensor([2. ** (i / 8.) for i in range(49)], device=device, dtype=dtype),
                         torch.logspace(0, 6, steps=49, base=2, device=device, dtype=dtype),
                         atol=atol, rtol=rtol)

        # Check non-default base=2
        self.assertEqual(torch.logspace(1, 1, 1, 2, device=device, dtype=dtype),
                         torch.ones(1, device=device, dtype=dtype) * 2)
        self.assertEqual(torch.logspace(0, 2, 3, 2, device=device, dtype=dtype),
                         torch.tensor((1, 2, 4), device=device, dtype=dtype))

        # Check logspace_ for generating with start > end.
        self.assertEqual(torch.logspace(1, 0, 2, device=device, dtype=dtype),
                         torch.tensor((10, 1), device=device, dtype=dtype), atol=0, rtol=0)

        # Check logspace_ for non-contiguous tensors.
        x = torch.zeros(2, 3, device=device, dtype=dtype)
        y = torch.logspace(0, 3, 4, base=2, device=device, dtype=dtype, out=x.narrow(1, 1, 2))
        self.assertEqual(x, torch.tensor(((0, 1, 2), (0, 4, 8)), device=device, dtype=dtype), atol=0, rtol=0)

    @onlyOnCPUAndCUDA
    @dtypes(torch.half, torch.float, torch.double)
    def test_full_inference(self, device, dtype):
        size = (2, 2)

        prev_default = torch.get_default_dtype()
        torch.set_default_dtype(dtype)

        # Tests bool fill value inference
        t = torch.full(size, True)
        self.assertEqual(t.dtype, torch.bool)

        # Tests integer fill value inference
        t = torch.full(size, 1)
        self.assertEqual(t.dtype, torch.long)

        # Tests float fill value inference
        t = torch.full(size, 1.)
        self.assertEqual(t.dtype, dtype)

        # Tests complex inference
        t = torch.full(size, (1 + 1j))
        ctype = torch.complex128 if dtype is torch.double else torch.complex64
        self.assertEqual(t.dtype, ctype)

        torch.set_default_dtype(prev_default)

    def test_full_out(self, device):
        size = (5,)
        o = torch.empty(size, device=device, dtype=torch.long)

        # verifies dtype/out conflict throws a RuntimeError
        with self.assertRaises(RuntimeError):
            torch.full(o.shape, 1., dtype=torch.float, out=o)

        # verifies out dtype overrides inference
        self.assertEqual(torch.full(o.shape, 1., out=o).dtype, o.dtype)
        self.assertEqual(torch.full(size, 1, out=o).dtype, o.dtype)

    # check that warning for numpy being not writable is suppressed
    # when a copy of it is being created.
    # see issue #47160
    def test_tensor_from_non_writable_numpy(self, device):
        with warnings.catch_warnings(record=True) as w:
            a = np.arange(5.)
            a.flags.writeable = False
            t = torch.tensor(a)
            self.assertEqual(len(w), 0)


# Class for testing random tensor creation ops, like torch.randint
class TestRandomTensorCreation(TestCase):
    exact_dtype = True

    # TODO: add torch.complex64, torch.complex128
    @dtypes(torch.float, torch.double)
    def test_normal(self, device, dtype):

        def helper(self, device, dtype, ptype, t_transform, std_transform):
            q = torch.empty(100, 100, dtype=dtype, device=device)

            q.normal_()
            self.assertEqual(t_transform(q).mean(), 0, atol=0.2, rtol=0)
            self.assertEqual(t_transform(q).std(), std_transform(1), atol=0.2, rtol=0)

            q.normal_(2, 3)
            self.assertEqual(t_transform(q).mean(), 2, atol=0.3, rtol=0)
            self.assertEqual(t_transform(q).std(), std_transform(3), atol=0.3, rtol=0)

            q = torch.empty(100, 100, dtype=dtype, device=device)
            q_row1 = q[0:1].clone()
            q[99:100].normal_()
            self.assertEqual(t_transform(q[99:100]).mean(), 0, atol=0.2, rtol=0)
            self.assertEqual(t_transform(q[99:100]).std(), std_transform(1), atol=0.2, rtol=0)
            self.assertEqual(t_transform(q[0:1]).clone(), t_transform(q_row1))

            mean = torch.empty(100, 100, dtype=dtype, device=device)
            mean[:50].fill_(ptype(0))
            mean[50:].fill_(ptype(1))

            std = torch.empty(100, 100, dtype=torch.float, device=device)
            std[:, :50] = 4
            std[:, 50:] = 1

            r = torch.normal(mean)
            self.assertEqual(r.dtype, dtype)
            self.assertEqual(str(r.device), device)
            self.assertEqual(t_transform(r[:50]).mean(), 0, atol=0.2, rtol=0)
            self.assertEqual(t_transform(r[50:]).mean(), 1, atol=0.2, rtol=0)
            self.assertEqual(t_transform(r).std(), std_transform(1), atol=0.2, rtol=0)

            r.fill_(42)
            r = torch.normal(mean, 3)
            self.assertEqual(r.dtype, dtype)
            self.assertEqual(str(r.device), device)
            self.assertEqual(t_transform(r[:50]).mean(), 0, atol=0.2, rtol=0)
            self.assertEqual(t_transform(r[50:]).mean(), 1, atol=0.2, rtol=0)
            self.assertEqual(t_transform(r).std(), std_transform(3), atol=0.2, rtol=0)

            r.fill_(42)
            torch.normal(mean, 3, out=r)
            self.assertEqual(r.dtype, dtype)
            self.assertEqual(str(r.device), device)
            self.assertEqual(t_transform(r[:50]).mean(), 0, atol=0.2, rtol=0)
            self.assertEqual(t_transform(r[50:]).mean(), 1, atol=0.2, rtol=0)
            self.assertEqual(t_transform(r).std(), std_transform(3), atol=0.2, rtol=0)

            r.fill_(42)
            r = torch.normal(2, std)
            self.assertFalse(r.dtype.is_complex)
            self.assertEqual(str(r.device), device)
            self.assertEqual(r.mean(), 2, atol=0.2, rtol=0)
            self.assertEqual(r[:, :50].std(), 4, atol=0.3, rtol=0)
            self.assertEqual(r[:, 50:].std(), 1, atol=0.2, rtol=0)

            r.fill_(42)
            torch.normal(2, std, out=r)
            self.assertFalse(r.dtype.is_complex)
            self.assertEqual(str(r.device), device)
            self.assertEqual(r.mean(), 2, atol=0.2, rtol=0)
            self.assertEqual(r[:, :50].std(), 4, atol=0.3, rtol=0)
            self.assertEqual(r[:, 50:].std(), 1, atol=0.2, rtol=0)

            r.fill_(42)
            r = torch.normal(mean, std)
            self.assertEqual(r.dtype, dtype)
            self.assertEqual(str(r.device), device)
            self.assertEqual(t_transform(r[:50]).mean(), 0, atol=0.2, rtol=0)
            self.assertEqual(t_transform(r[50:]).mean(), 1, atol=0.2, rtol=0)
            self.assertEqual(t_transform(r[:, :50]).std(), std_transform(4), atol=0.3, rtol=0)
            self.assertEqual(t_transform(r[:, 50:]).std(), std_transform(1), atol=0.2, rtol=0)

            r.fill_(42)
            torch.normal(mean, std, out=r)
            self.assertEqual(r.dtype, dtype)
            self.assertEqual(str(r.device), device)
            self.assertEqual(t_transform(r[:50]).mean(), 0, atol=0.2, rtol=0)
            self.assertEqual(t_transform(r[50:]).mean(), 1, atol=0.2, rtol=0)
            self.assertEqual(t_transform(r[:, :50]).std(), std_transform(4), atol=0.3, rtol=0)
            self.assertEqual(t_transform(r[:, 50:]).std(), std_transform(1), atol=0.2, rtol=0)

            r.fill_(42)
            r = torch.normal(2, 3, (100, 100), dtype=dtype, device=device)
            self.assertEqual(r.dtype, dtype)
            self.assertEqual(str(r.device), device)
            self.assertEqual(t_transform(r).mean(), 2, atol=0.3, rtol=0)
            self.assertEqual(t_transform(r).std(), std_transform(3), atol=0.3, rtol=0)

            r.fill_(42)
            torch.normal(2, 3, (100, 100), dtype=dtype, device=device, out=r)
            self.assertEqual(r.dtype, dtype)
            self.assertEqual(str(r.device), device)
            self.assertEqual(t_transform(r).mean(), 2, atol=0.3, rtol=0)
            self.assertEqual(t_transform(r).std(), std_transform(3), atol=0.3, rtol=0)

            # float std 0 with float mean
            r.fill_(42)
            torch.normal(2, 0, (10, 10), dtype=dtype, device=device, out=r)
            self.assertEqual(r.dtype, dtype)
            self.assertEqual(str(r.device), device)
            self.assertTrue(r.eq(2).all())

            # float std 0 with tensor mean
            r.fill_(42)
            mean_rand = torch.randn(10, 10, dtype=dtype, device=device)
            torch.normal(mean_rand, 0, out=r)
            self.assertEqual(r.dtype, dtype)
            self.assertEqual(str(r.device), device)
            self.assertEqual(mean_rand, r, atol=0, rtol=0)

            # tensor std 0 with float mean
            r.fill_(42)
            std_zeros = torch.zeros(10, 10, dtype=dtype, device=device)
            torch.normal(2, std_zeros, out=r)
            self.assertEqual(r.dtype, dtype)
            self.assertEqual(str(r.device), device)
            self.assertTrue(r.eq(2).all())

            # tensor std 0 with tensor mean
            r.fill_(42)
            torch.normal(mean_rand, std_zeros, out=r)
            self.assertEqual(r.dtype, dtype)
            self.assertEqual(str(r.device), device)
            self.assertEqual(mean_rand, r, atol=0, rtol=0)

        if dtype.is_complex:
            helper(self, device, dtype, lambda x: complex(x, x),
                   lambda t: torch.real(t).to(torch.float), lambda mean: mean / math.sqrt(2))
            helper(self, device, dtype, lambda x: complex(x, x),
                   lambda t: torch.imag(t).to(torch.float), lambda mean: mean / math.sqrt(2))
            self.assertRaisesRegex(
                RuntimeError, "normal expects standard deviation to be non-complex",
                lambda: torch.normal(0, torch.empty(100, 100, dtype=dtype, device=device)))
            out = torch.empty(100, 100, dtype=dtype, device=device)
            self.assertRaisesRegex(
                RuntimeError, "normal expects standard deviation to be non-complex",
                lambda: torch.normal(0, torch.empty(100, 100, dtype=dtype, device=device), out=out))
        else:
            helper(self, device, dtype, lambda x: x, lambda t: t, lambda mean: mean)

    # Ensure that normal raises appropriate error when `std` < 0
    def test_normal_std_error(self, device):
        a = torch.tensor(0, dtype=torch.float32, device=device)
        std = torch.tensor(-1, dtype=torch.float32, device=device)

        for input in [0, a]:
            with self.assertRaisesRegex(RuntimeError, r'normal_ expects std >= 0.0'):
                torch.normal(input, -1, (10,))

            with self.assertRaisesRegex(RuntimeError, r'normal expects all elements of std >= 0.0'):
                torch.normal(input, std)

    @dtypes(torch.float, torch.double, torch.half)
    @dtypesIfCUDA(torch.float, torch.double, torch.half, torch.bfloat16)
    def test_uniform_from_to(self, device, dtype):
        size = 2000
        alpha = 0.1

        float_min = torch.finfo(torch.float).min
        float_max = torch.finfo(torch.float).max
        double_min = torch.finfo(torch.double).min
        double_max = torch.finfo(torch.double).max

        if dtype == torch.bfloat16:
            min_val = -3.389531389251535e+38
            max_val = 3.389531389251535e+38
        else:
            min_val = torch.finfo(dtype).min
            max_val = torch.finfo(dtype).max

        values = [double_min, float_min, -42, 0, 42, float_max, double_max]

        for from_ in values:
            for to_ in values:
                t = torch.empty(size, dtype=dtype, device=device)
                if not (min_val <= from_ <= max_val) or not (min_val <= to_ <= max_val):
                    pass
                elif to_ < from_:
                    self.assertRaisesRegex(
                        RuntimeError,
                        "uniform_ expects to return",
                        lambda: t.uniform_(from_, to_)
                    )
                elif to_ - from_ > max_val:
                    self.assertRaisesRegex(
                        RuntimeError,
                        "uniform_ expects to-from",
                        lambda: t.uniform_(from_, to_)
                    )
                else:
                    t.uniform_(from_, to_)
                    range_ = to_ - from_
                    if not (dtype == torch.bfloat16) and not (
                            dtype == torch.half and device == 'cpu') and not torch.isnan(t).all():
                        delta = alpha * range_
                        double_t = t.to(torch.double)
                        if range_ == 0:
                            self.assertTrue(double_t.min() == from_)
                            self.assertTrue(double_t.max() == to_)
                        elif dtype == torch.half:
                            self.assertTrue(from_ <= double_t.min() <= (from_ + delta))
                            self.assertTrue((to_ - delta) <= double_t.max() <= to_)
                        else:
                            self.assertTrue(from_ <= double_t.min() <= (from_ + delta))
                            self.assertTrue((to_ - delta) <= double_t.max() < to_)

    def test_random_neg_values(self, device):
        SIZE = 10
        signed_dtypes = [torch.double, torch.float, torch.long, torch.int, torch.short]
        for dtype in signed_dtypes:
            res = torch.rand(SIZE, SIZE).to(device=device, dtype=dtype)
            res.random_(-10, -1)
            self.assertLessEqual(res.max().item(), 9)
            self.assertGreaterEqual(res.min().item(), -10)

    # TODO: this test should be updated
    @onlyCPU
    def test_randint_inference(self, device):
        size = (2, 1)
        for args in [(3,), (1, 3)]:  # (low,) and (low, high)
            self.assertIs(torch.int64, torch.randint(*args, size=size).dtype)
            self.assertIs(torch.int64, torch.randint(*args, size=size, layout=torch.strided).dtype)
            self.assertIs(torch.int64, torch.randint(*args, size=size, generator=torch.default_generator).dtype)
            self.assertIs(torch.float32, torch.randint(*args, size=size, dtype=torch.float32).dtype)
            out = torch.empty(size, dtype=torch.float32)
            self.assertIs(torch.float32, torch.randint(*args, size=size, out=out).dtype)
            self.assertIs(torch.float32, torch.randint(*args, size=size, out=out, dtype=torch.float32).dtype)
            out = torch.empty(size, dtype=torch.int64)
            self.assertIs(torch.int64, torch.randint(*args, size=size, out=out).dtype)
            self.assertIs(torch.int64, torch.randint(*args, size=size, out=out, dtype=torch.int64).dtype)

    # TODO: this test should be updated
    @onlyCPU
    def test_randint(self, device):
        SIZE = 100

        def seed(generator):
            if generator is None:
                torch.manual_seed(123456)
            else:
                generator.manual_seed(123456)
            return generator

        for generator in (None, torch.Generator()):
            generator = seed(generator)
            res1 = torch.randint(0, 6, (SIZE, SIZE), generator=generator)
            res2 = torch.empty((), dtype=torch.int64)
            generator = seed(generator)
            torch.randint(0, 6, (SIZE, SIZE), generator=generator, out=res2)
            generator = seed(generator)
            res3 = torch.randint(6, (SIZE, SIZE), generator=generator)
            res4 = torch.empty((), dtype=torch.int64)
            generator = seed(generator)
            torch.randint(6, (SIZE, SIZE), out=res4, generator=generator)
            self.assertEqual(res1, res2)
            self.assertEqual(res1, res3)
            self.assertEqual(res1, res4)
            self.assertEqual(res2, res3)
            self.assertEqual(res2, res4)
            self.assertEqual(res3, res4)
            self.assertTrue((res1 < 6).all().item())
            self.assertTrue((res1 >= 0).all().item())

    @dtypes(torch.half, torch.float, torch.bfloat16, torch.double,
            torch.complex32, torch.complex64, torch.complex128)
    def test_randn(self, device, dtype):
        SIZE = 100
        for size in [0, SIZE]:
            torch.manual_seed(123456)
            res1 = torch.randn(size, size, dtype=dtype, device=device)
            res2 = torch.tensor([], dtype=dtype, device=device)
            torch.manual_seed(123456)
            torch.randn(size, size, out=res2)
            self.assertEqual(res1, res2)

    @dtypes(torch.float, torch.double, torch.complex64, torch.complex128)
    def test_rand(self, device, dtype):
        SIZE = 100
        for size in [0, SIZE]:
            torch.manual_seed(123456)
            res1 = torch.rand(size, size, dtype=dtype, device=device)
            res2 = torch.tensor([], dtype=dtype, device=device)
            torch.manual_seed(123456)
            torch.rand(size, size, out=res2)
            self.assertEqual(res1, res2)

    def test_randperm(self, device):
        if device == 'cpu' or device == 'meta':
            rng_device = None
        else:
            # TODO: This won't actually work for non-CUDA device
            # see https://github.com/pytorch/pytorch/issues/54282
            rng_device = [device]

        # Test core functionality. On CUDA, different value of n has different
        # code path
        for n in (5, 100, 50000, 100000):
            # Ensure both integer and floating-point numbers are tested. Half follows an execution path that is
            # different from others on CUDA.
            for dtype in (torch.long, torch.half, torch.float):
                if n > 2049 and dtype == torch.half:  # Large n for torch.half will raise an exception, do not test here.
                    continue
                with torch.random.fork_rng(devices=rng_device):
                    res1 = torch.randperm(n, dtype=dtype, device=device)
                res2 = torch.empty(0, dtype=dtype, device=device)
                torch.randperm(n, out=res2, dtype=dtype, device=device)
                self.assertEqual(res1, res2, atol=0, rtol=0)
                self.assertEqual(res1.sort().values.long(), torch.arange(n, device=device))

        # Default type is long
        for n in (100, 10000):
            self.assertEqual(torch.randperm(n, device=device).dtype, torch.long)

        # randperm of 0 elements is an empty tensor
        res1 = torch.randperm(0)
        res2 = torch.tensor(5, dtype=dtype, device=device)
        torch.randperm(0, out=res2)
        self.assertEqual(res1.numel(), 0)
        self.assertEqual(res2.numel(), 0)

        # Test exceptions when n is too large for a floating point type
        for dtype, small_n, large_n in ((torch.uint8, 2**8, 2**8 + 1),
                                        (torch.half, 2**11 + 1, 2**11 + 2),
                                        (torch.float, 2**24 + 1, 2**24 + 2),
                                        (torch.double, 2**25,  # 2**53 + 1 is too large to run
                                         2**53 + 2)):
            res = torch.empty(0, dtype=dtype, device=device)
            torch.randperm(small_n, out=res)  # No exception expected
            self.assertRaises(RuntimeError, lambda: torch.randperm(large_n, out=res, device=device))

        # Test non-contiguous tensors
        for n in (4, 5, 6, 10, 20):
            non_contiguous_tensor = torch.zeros((2, 3), dtype=torch.long, device=device).t()
            self.assertFalse(non_contiguous_tensor.is_contiguous())
            with torch.random.fork_rng(devices=rng_device):
                res = torch.randperm(n, dtype=torch.long, device=device)
            torch.randperm(n, out=non_contiguous_tensor)
            self.assertEqual(non_contiguous_tensor, res)
            self.assertEqual(res.sort().values.long(), torch.arange(n, device=device))

    # Test exceptions when device and generator types are incompatible
    @onlyCUDA
    def test_randperm_device_compatibility(self, device):
        cuda_gen = torch.Generator(device='cuda')
        cpu_gen = torch.Generator(device='cpu')

        # n=0 is a special case that we don't need to use generator, thus no error even if
        # device and generator don't match
        torch.randperm(0, device='cuda:0', generator=torch.Generator(device='cuda:1'))
        if torch.cuda.device_count() > 1:
            torch.randperm(0, device='cuda:1', generator=torch.Generator(device='cuda:0'))
        torch.randperm(0, device='cuda', generator=torch.Generator(device='cpu'))
        torch.randperm(0, device='cpu', generator=torch.Generator(device='cuda'))

        for n in (1, 3, 100, 30000):
            torch.randperm(n, device='cuda', generator=torch.Generator(device='cuda:0'))
            torch.randperm(n, device='cuda:0', generator=torch.Generator(device='cuda'))
            # For cuda:0 to match cuda:1, we are making consistent device type matching
            # behavior just like torch.randint. Longer term, generator should ignore
            # device ordinal, since it's not used anyway.
            torch.randint(low=0, high=n + 1, size=(1,), device="cuda:0", generator=torch.Generator(device='cuda:1'))
            torch.randperm(n, device='cuda:0', generator=torch.Generator(device='cuda:1'))
            if torch.cuda.device_count() > 1:
                torch.randint(low=0, high=n + 1, size=(1,), device="cuda:1", generator=torch.Generator(device='cuda:0'))
                torch.randperm(n, device='cuda:1', generator=torch.Generator(device='cuda:0'))

            regex = 'Expected a .* device type for generator but found .*'
            cuda_t = torch.tensor(n, device='cuda')
            self.assertRaisesRegex(RuntimeError, regex, lambda: torch.randperm(n, device='cuda', generator=cpu_gen))
            self.assertRaisesRegex(RuntimeError, regex, lambda: torch.randperm(n, device='cuda', generator=cpu_gen, out=cuda_t))
            cpu_t = torch.tensor(n, device='cpu')
            self.assertRaisesRegex(RuntimeError, regex, lambda: torch.randperm(n, device='cpu', generator=cuda_gen))
            self.assertRaisesRegex(RuntimeError, regex, lambda: torch.randperm(n, device='cpu', generator=cuda_gen, out=cpu_t))
            self.assertRaisesRegex(RuntimeError, regex, lambda: torch.randperm(n, generator=cuda_gen))  # implicitly on CPU

# Class for testing *like ops, like torch.ones_like
class TestLikeTensorCreation(TestCase):
    exact_dtype = True

    # TODO: this test should be updated
    def test_ones_like(self, device):
        expected = torch.ones(100, 100, device=device)

        res1 = torch.ones_like(expected)
        self.assertEqual(res1, expected)

        # test boolean tensor
        expected = torch.tensor([True, True], device=device, dtype=torch.bool)
        res1 = torch.ones_like(expected)
        self.assertEqual(res1, expected)

    # TODO: this test should be updated
    @onlyCPU
    def test_empty_like(self, device):
        x = torch.autograd.Variable(torch.tensor([]))
        y = torch.autograd.Variable(torch.randn(4, 4))
        z = torch.autograd.Variable(torch.IntTensor([1, 2, 3]))
        for a in (x, y, z):
            self.assertEqual(torch.empty_like(a).shape, a.shape)
            self.assertEqualTypeString(torch.empty_like(a), a)

    def test_zeros_like(self, device):
        expected = torch.zeros((100, 100,), device=device)

        res1 = torch.zeros_like(expected)
        self.assertEqual(res1, expected)

    @deviceCountAtLeast(2)
    def test_zeros_like_multiple_device(self, devices):
        expected = torch.zeros(100, 100, device=devices[0])
        x = torch.randn(100, 100, device=devices[1], dtype=torch.float32)
        output = torch.zeros_like(x)
        self.assertEqual(output, expected)

    @deviceCountAtLeast(2)
    def test_ones_like_multiple_device(self, devices):
        expected = torch.ones(100, 100, device=devices[0])
        x = torch.randn(100, 100, device=devices[1], dtype=torch.float32)
        output = torch.ones_like(x)
        self.assertEqual(output, expected)

    # Full-like precedence is the explicit dtype then the dtype of the "like"
    # tensor.
    @onlyOnCPUAndCUDA
    def test_full_like_inference(self, device):
        size = (2, 2)
        like = torch.empty((5,), device=device, dtype=torch.long)

        self.assertEqual(torch.full_like(like, 1.).dtype, torch.long)
        self.assertEqual(torch.full_like(like, 1., dtype=torch.complex64).dtype,
                         torch.complex64)

instantiate_device_type_tests(TestTensorCreation, globals())
instantiate_device_type_tests(TestRandomTensorCreation, globals())
instantiate_device_type_tests(TestLikeTensorCreation, globals())

if __name__ == '__main__':
    run_tests()
