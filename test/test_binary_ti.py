import torch
import numpy as np

import itertools
from itertools import product
import math
import random
import unittest
import warnings
import operator
from functools import partial

from torch._six import inf, nan
from torch.testing._internal.common_utils import (
    TestCase, iter_indices, TEST_WITH_ASAN, run_tests,
    torch_to_numpy_dtype_dict, make_tensor, TEST_SCIPY, set_default_dtype)
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests, onlyCUDA, onlyCPU, dtypes, dtypesIfCUDA,
    dtypesIfCPU, deviceCountAtLeast, precisionOverride, onlyOnCPUAndCUDA,
    skipCUDAIfRocm, skipIf)
from torch.testing import all_types_and_complex_and

if TEST_SCIPY:
    import scipy.special

# TODO: remove this
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

# TODO: refactor this out
# Converts half/bfloat16 dtype to float when device is cpu
def _convert_t(dtype, device):
    if device == 'cpu' and dtype in {torch.half, torch.bfloat16}:
        return torch.float
    return dtype

# TODO: revise the tests to use make_tensor in common_utils.py instead
# Returns a tensor of the requested shape, dtype, and device
# Requesting a half CPU tensor returns a float CPU tensor with
# values representable by a half.
# Initialization uses randint for non-float types and randn for float types.
def _make_tensor(shape, dtype, device, fill_ones=False) -> torch.Tensor:
    # Returns a tensor filled with ones
    if fill_ones:
        return torch.ones(*shape, dtype=_convert_t(dtype, device), device=device)

    # Returns a tensor with random integer values
    if not (dtype.is_floating_point or dtype.is_complex):
        t = torch.randint(0, 10, shape, device=device)
        if dtype != torch.uint8:
            t = t - 5  # generate negative values also
        return t.to(_convert_t(dtype, device))

    # Populates the CPU tensor with floats representable as half/bfloat16
    if dtype == torch.half and device == 'cpu':
        return torch.randn(*shape, dtype=torch.float, device=device).half().float()
    if dtype == torch.bfloat16 and device == 'cpu':
        return torch.randn(*shape, dtype=torch.float, device=device).bfloat16().float()

    # Default: returns a tensor with random float values
    return torch.randn(shape, dtype=dtype, device=device).to(dtype=dtype)

# TODO: update to use opinfos consistently
class TestBinaryUfuncs(TestCase):

    def test_add_broadcast_empty(self, device):
        # empty + empty
        self.assertRaises(RuntimeError, lambda: torch.randn(5, 0, device=device) + torch.randn(0, 5, device=device))
        self.assertEqual(torch.randn(5, 0, device=device), torch.randn(0, device=device) + torch.randn(5, 0, device=device))
        self.assertEqual(torch.randn(5, 0, 0, device=device), torch.randn(0, device=device) + torch.randn(5, 0, 1, device=device))

        # scalar + empty
        self.assertEqual(torch.randn(5, 0, 6, device=device), torch.randn((), device=device) + torch.randn(5, 0, 6, device=device))

        # non-empty, empty
        self.assertEqual(torch.randn(0, device=device), torch.randn(0, device=device) + torch.randn(1, device=device))
        self.assertEqual(torch.randn(0, 7, 0, 6, 5, 0, 7, device=device),
                         torch.randn(0, 7, 0, 6, 5, 0, 1, device=device) + torch.randn(1, 1, 5, 1, 7, device=device))
        self.assertRaises(RuntimeError, lambda: torch.randn(7, 0, device=device) + torch.randn(2, 1, device=device))

    def test_basic(self, device):
        a = torch.randn(5, 7, device=device)
        b = torch.randn(5, 7, device=device)
        ab = a - (-b) # bypass plus

        # functional convention
        c = torch.add(a, b)
        self.assertEqual(c, ab)

        # inplace
        c = a.clone()
        c.add_(b)
        self.assertEqual(c, ab)

        # out convention with correct size
        c = torch.zeros(5, 7)
        torch.add(a, b, out=c)
        self.assertEqual(c, ab)

        # out convention with incorrect size
#        c = torch.zeros(1, 2)
#        torch.add(a, b, out=c)
#        self.assertEqual(c, ab)


    def test_add(self, device):
        dtypes = [torch.float, torch.double] + torch.testing.get_all_complex_dtypes()
        for dtype in dtypes:
            # [res] torch.add([res,] tensor1, tensor2)
            m1 = torch.randn(100, 100, dtype=dtype, device=device)
            v1 = torch.randn(100, dtype=dtype, device=device)

            # contiguous
            res1 = torch.add(m1[4], v1)
            res2 = res1.clone().zero_()
            for i in range(m1.size(1)):
                res2[i] = m1[4, i] + v1[i]
            self.assertEqual(res1, res2)

            m1 = torch.randn(100, 100, device=device)
            v1 = torch.randn(100, device=device)

            # non-contiguous
#            res1 = torch.add(m1[:, 4], v1)
#            res2 = res1.clone().zero_()
#            for i in range(m1.size(0)):
#                res2[i] = m1[i, 4] + v1[i]
#            self.assertEqual(res1, res2)

            # [res] torch.add([res,] tensor, value)
            m1 = torch.randn(10, 10, device=device)

            # contiguous
#            res1 = m1.clone()
#            res1[3].add_(2)
#            res2 = m1.clone()
#            for i in range(m1.size(1)):
#                res2[3, i] = res2[3, i] + 2
#            self.assertEqual(res1, res2)

            # non-contiguous
#            m1 = torch.randn(10, 10, device=device)
#            res1 = m1.clone()
#            res1[:, 3].add_(2)
#            res2 = m1.clone()
#            for i in range(m1.size(0)):
#                res2[i, 3] = res2[i, 3] + 2
#            self.assertEqual(res1, res2)

            # inter-type
            m1 = torch.randn(10, 10, dtype=dtype, device=device)
            self.assertEqual(m1 + 3, m1 + torch.tensor(3))
            self.assertEqual(3 + m1, torch.tensor(3) + m1)

            # contiguous + non-contiguous
            m1 = torch.randn(10, 10, dtype=dtype, device=device)
            m2 = torch.randn(10, 10, dtype=dtype, device=device).t()
            res = m1 + m2
            self.assertTrue(res.is_contiguous())
            self.assertEqual(res, m1 + m2.contiguous())

            # 1d + empty
            m1 = torch.tensor([1.0], dtype=dtype, device=device)
            m2 = torch.tensor([], dtype=dtype, device=device)
            self.assertEqual(m1 + m2, [])

        # inter-type unint8
        one = torch.tensor(1, dtype=torch.uint8, device=device)
        self.assertEqual(torch.add(one, 1), 2)
        self.assertEqual(torch.add(one, 1).dtype, torch.uint8)

        # bool
        m1 = torch.tensor([True, False, False, True, False, False], dtype=torch.bool, device=device)
        m2 = torch.tensor([True, True, False, False, False, True], dtype=torch.bool, device=device)
        expected = torch.tensor([True, True, False, True, False, True], dtype=torch.bool, device=device)
        self.assertEqual(m1 + m2, expected)

        # fused multiply add
        a = torch.zeros(2, 3, dtype=torch.bool, device=device)
        res = torch.add(a, a, alpha=0)
        expected = torch.zeros(2, 3, device=device).bool()
        self.assertEqual(res, expected)

        # bfloat16
        m1 = torch.tensor([1., 2.], dtype=torch.bfloat16)
        m2 = torch.tensor([3., 4.], dtype=torch.bfloat16)
        self.assertEqual(m1 + m2, torch.tensor([4., 6.], dtype=torch.bfloat16))

        # different alpha types
        m1 = torch.tensor([2 + 3j, 4 + 5j], dtype=torch.complex64, device=device)
        m2 = torch.tensor([4 + 5j, 2 + 3j], dtype=torch.complex64, device=device)
        # add complex numbers with float alpha
        res = torch.add(m1, m2, alpha=0.1)
        expected = torch.tensor([2.4000 + 3.5000j, 4.2000 + 5.3000j], dtype=torch.complex64, device=device)
        self.assertEqual(res, expected)

        # add complex numbers with complex alpha
        res = torch.add(m1, m2, alpha=complex(0.1, 0.2))
        expected = torch.tensor([1.4000 + 4.3000j, 3.6000 + 5.7000j], dtype=torch.complex64, device=device)
        self.assertEqual(res, expected)

        # add complex numbers with integer alpha
        res = torch.add(m1, m2, alpha=2)
        expected = torch.tensor([10. + 13.j, 8. + 11.j], dtype=torch.complex64, device=device)
        self.assertEqual(res, expected)

        # mismatched alpha
        m1 = torch.tensor([1], dtype=torch.int8, device=device)
        m2 = torch.tensor([2], dtype=torch.int8, device=device)
        self.assertRaisesRegex(RuntimeError,
                               r"Boolean alpha only supported for Boolean results\.",
                               lambda: torch.add(m1, m2, alpha=True))
        self.assertRaisesRegex(RuntimeError,
                               r"For integral input tensors, argument alpha must not be a floating point number\.",
                               lambda: torch.add(m1, m2, alpha=1.0))

        # mismatched alpha, float / double tensor and complex alpha
        msg = r"For non-complex input tensors, argument alpha must not be a complex number\."
        m1 = torch.tensor([3., 4.], device=device)
        m2 = torch.tensor([4., 3.], device=device)
        self.assertRaisesRegex(RuntimeError, msg,
                               lambda: torch.add(m1, m2, alpha=complex(0.1, 0.2)))

        m1 = torch.tensor([3., 4.], dtype=torch.double, device=device)
        m2 = torch.tensor([4., 3.], dtype=torch.double, device=device)
        self.assertRaisesRegex(RuntimeError, msg,
                               lambda: torch.add(m1, m2, alpha=complex(0.1, 0.2)))

        # complex
        m1 = torch.tensor((4.0000 + 4.0000j), dtype=torch.complex64)
        m2 = torch.tensor(4., dtype=torch.float64)
        self.assertRaisesRegex(RuntimeError, r"result type ComplexFloat can't be cast to the desired output type Double",
                               lambda: torch.add(m1, m1, out=m2))

instantiate_device_type_tests(TestBinaryUfuncs, globals(), except_for='meta')

if __name__ == '__main__':
    run_tests()
