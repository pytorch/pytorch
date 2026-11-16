# ======= BEGIN Dynamo patch =======
# Owner(s): ["module: dynamo"]

# ruff: noqa
# flake8: noqa

# Test copied from
# https://raw.githubusercontent.com/python/cpython/refs/tags/v3.13.5/Lib/test/test_augassign.py

import sys
import torch
import torch._dynamo.test_case
import unittest
from torch._dynamo.test_case import CPythonTestCase
from torch.testing._internal.common_utils import run_tests

# ======= END DYNAMO PATCH =======

# Augmented assignment test.

import unittest


class AugAssignTest(CPythonTestCase):
    def testBasic(self):
        x = 2
        x += 1
        x *= 2
        x **= 2
        x -= 8
        x //= 5
        x %= 3
        x &= 2
        x |= 5
        x ^= 1
        x /= 2
        self.assertEqual(x, 3.0)

    def test_with_unpacking(self):
        self.assertRaises(SyntaxError, compile, "x, b += 3", "<test>", "exec")

    def testInList(self):
        x = [2]
        x[0] += 1
        x[0] *= 2
        x[0] **= 2
        x[0] -= 8
        x[0] //= 5
        x[0] %= 3
        x[0] &= 2
        x[0] |= 5
        x[0] ^= 1
        x[0] /= 2
        self.assertEqual(x[0], 3.0)

    def testInDict(self):
        x = {0: 2}
        x[0] += 1
        x[0] *= 2
        x[0] **= 2
        x[0] -= 8
        x[0] //= 5
        x[0] %= 3
        x[0] &= 2
        x[0] |= 5
        x[0] ^= 1
        x[0] /= 2
        self.assertEqual(x[0], 3.0)

    def testSequences(self):
        x = [1,2]
        x += [3,4]
        x *= 2

        self.assertEqual(x, [1, 2, 3, 4, 1, 2, 3, 4])

        x = [1, 2, 3]
        y = x
        x[1:2] *= 2
        y[1:2] += [1]

        self.assertEqual(x, [1, 2, 1, 2, 3])
        self.assertTrue(x is y)

    def testCustomMethods1(self):

        class aug_test:
            def __init__(self, value):
                self.val = value
            def __radd__(self, val):
                return self.val + val
            def __add__(self, val):
                return aug_test(self.val + val)

        class aug_test2(aug_test):
            def __iadd__(self, val):
                self.val = self.val + val
                return self

        class aug_test3(aug_test):
            def __iadd__(self, val):
                return aug_test3(self.val + val)

        class aug_test4(aug_test3):
            """Blocks inheritance, and fallback to __add__"""
            __iadd__ = None

        x = aug_test(1)
        y = x
        x += 10

        self.assertIsInstance(x, aug_test)
        self.assertTrue(y is not x)
        self.assertEqual(x.val, 11)

        x = aug_test2(2)
        y = x
        x += 10

        self.assertTrue(y is x)
        self.assertEqual(x.val, 12)

        x = aug_test3(3)
        y = x
        x += 10

        self.assertIsInstance(x, aug_test3)
        self.assertTrue(y is not x)
        self.assertEqual(x.val, 13)

        x = aug_test4(4)
        with self.assertRaises(TypeError):
            x += 10


    def testCustomMethods2(test_self):
        output = []

        class testall:
            def __add__(self, val):
                output.append("__add__ called")
            def __radd__(self, val):
                output.append("__radd__ called")
            def __iadd__(self, val):
                output.append("__iadd__ called")
                return self

            def __sub__(self, val):
                output.append("__sub__ called")
            def __rsub__(self, val):
                output.append("__rsub__ called")
            def __isub__(self, val):
                output.append("__isub__ called")
                return self

            def __mul__(self, val):
                output.append("__mul__ called")
            def __rmul__(self, val):
                output.append("__rmul__ called")
            def __imul__(self, val):
                output.append("__imul__ called")
                return self

            def __matmul__(self, val):
                output.append("__matmul__ called")
            def __rmatmul__(self, val):
                output.append("__rmatmul__ called")
            def __imatmul__(self, val):
                output.append("__imatmul__ called")
                return self

            def __floordiv__(self, val):
                output.append("__floordiv__ called")
                return self
            def __ifloordiv__(self, val):
                output.append("__ifloordiv__ called")
                return self
            def __rfloordiv__(self, val):
                output.append("__rfloordiv__ called")
                return self

            def __truediv__(self, val):
                output.append("__truediv__ called")
                return self
            def __rtruediv__(self, val):
                output.append("__rtruediv__ called")
                return self
            def __itruediv__(self, val):
                output.append("__itruediv__ called")
                return self

            def __mod__(self, val):
                output.append("__mod__ called")
            def __rmod__(self, val):
                output.append("__rmod__ called")
            def __imod__(self, val):
                output.append("__imod__ called")
                return self

            def __pow__(self, val):
                output.append("__pow__ called")
            def __rpow__(self, val):
                output.append("__rpow__ called")
            def __ipow__(self, val):
                output.append("__ipow__ called")
                return self

            def __or__(self, val):
                output.append("__or__ called")
            def __ror__(self, val):
                output.append("__ror__ called")
            def __ior__(self, val):
                output.append("__ior__ called")
                return self

            def __and__(self, val):
                output.append("__and__ called")
            def __rand__(self, val):
                output.append("__rand__ called")
            def __iand__(self, val):
                output.append("__iand__ called")
                return self

            def __xor__(self, val):
                output.append("__xor__ called")
            def __rxor__(self, val):
                output.append("__rxor__ called")
            def __ixor__(self, val):
                output.append("__ixor__ called")
                return self

            def __rshift__(self, val):
                output.append("__rshift__ called")
            def __rrshift__(self, val):
                output.append("__rrshift__ called")
            def __irshift__(self, val):
                output.append("__irshift__ called")
                return self

            def __lshift__(self, val):
                output.append("__lshift__ called")
            def __rlshift__(self, val):
                output.append("__rlshift__ called")
            def __ilshift__(self, val):
                output.append("__ilshift__ called")
                return self

        x = testall()
        x + 1
        1 + x
        x += 1

        x - 1
        1 - x
        x -= 1

        x * 1
        1 * x
        x *= 1

        x @ 1
        1 @ x
        x @= 1

        x / 1
        1 / x
        x /= 1

        x // 1
        1 // x
        x //= 1

        x % 1
        1 % x
        x %= 1

        x ** 1
        1 ** x
        x **= 1

        x | 1
        1 | x
        x |= 1

        x & 1
        1 & x
        x &= 1

        x ^ 1
        1 ^ x
        x ^= 1

        x >> 1
        1 >> x
        x >>= 1

        x << 1
        1 << x
        x <<= 1

        test_self.assertEqual(output, '''\
__add__ called
__radd__ called
__iadd__ called
__sub__ called
__rsub__ called
__isub__ called
__mul__ called
__rmul__ called
__imul__ called
__matmul__ called
__rmatmul__ called
__imatmul__ called
__truediv__ called
__rtruediv__ called
__itruediv__ called
__floordiv__ called
__rfloordiv__ called
__ifloordiv__ called
__mod__ called
__rmod__ called
__imod__ called
__pow__ called
__rpow__ called
__ipow__ called
__or__ called
__ror__ called
__ior__ called
__and__ called
__rand__ called
__iand__ called
__xor__ called
__rxor__ called
__ixor__ called
__rshift__ called
__rrshift__ called
__irshift__ called
__lshift__ called
__rlshift__ called
__ilshift__ called
'''.splitlines())

if __name__ == '__main__':
    run_tests()
