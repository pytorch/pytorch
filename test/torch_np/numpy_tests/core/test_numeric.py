# Owner(s): ["module: dynamo"]

import functools
import itertools
import math
import platform
import sys
import warnings

import numpy
import pytest


IS_WASM = False
HAS_REFCOUNT = True

import operator
from unittest import expectedFailure as xfail, skipIf as skipif, SkipTest

from hypothesis import given, strategies as st
from hypothesis.extra import numpy as hynp
from pytest import raises as assert_raises

from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    subtest,
    TEST_WITH_TORCHDYNAMO,
    TestCase,
    xfailIfTorchDynamo,
    xpassIfTorchDynamo_np,
)


# If we are going to trace through these, we should use NumPy
# If testing on eager mode, we use torch._numpy
if TEST_WITH_TORCHDYNAMO:
    import numpy as np
    from numpy.random import rand, randint, randn
    from numpy.testing import (
        assert_,
        assert_allclose,
        assert_almost_equal,
        assert_array_almost_equal,
        assert_array_equal,
        assert_equal,
        assert_warns,  # assert_array_max_ulp, HAS_REFCOUNT, IS_WASM
    )
else:
    import torch._numpy as np
    from torch._numpy.random import rand, randint, randn
    from torch._numpy.testing import (
        assert_,
        assert_allclose,
        assert_almost_equal,
        assert_array_almost_equal,
        assert_array_equal,
        assert_equal,
        assert_warns,  # assert_array_max_ulp, HAS_REFCOUNT, IS_WASM
    )


skip = functools.partial(skipif, True)


@instantiate_parametrized_tests
class TestResize(TestCase):
    def test_copies(self):
        A = np.array([[1, 2], [3, 4]])
        Ar1 = np.array([[1, 2, 3, 4], [1, 2, 3, 4]])
        assert_equal(np.resize(A, (2, 4)), Ar1)

        Ar2 = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
        assert_equal(np.resize(A, (4, 2)), Ar2)

        Ar3 = np.array([[1, 2, 3], [4, 1, 2], [3, 4, 1], [2, 3, 4]])
        assert_equal(np.resize(A, (4, 3)), Ar3)

    def test_repeats(self):
        A = np.array([1, 2, 3])
        Ar1 = np.array([[1, 2, 3, 1], [2, 3, 1, 2]])
        assert_equal(np.resize(A, (2, 4)), Ar1)

        Ar2 = np.array([[1, 2], [3, 1], [2, 3], [1, 2]])
        assert_equal(np.resize(A, (4, 2)), Ar2)

        Ar3 = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]])
        assert_equal(np.resize(A, (4, 3)), Ar3)

    def test_zeroresize(self):
        A = np.array([[1, 2], [3, 4]])
        Ar = np.resize(A, (0,))
        assert_array_equal(Ar, np.array([]))
        assert_equal(A.dtype, Ar.dtype)

        Ar = np.resize(A, (0, 2))
        assert_equal(Ar.shape, (0, 2))

        Ar = np.resize(A, (2, 0))
        assert_equal(Ar.shape, (2, 0))

    def test_reshape_from_zero(self):
        # See also gh-6740
        A = np.zeros(0, dtype=np.float32)
        Ar = np.resize(A, (2, 1))
        assert_array_equal(Ar, np.zeros((2, 1), Ar.dtype))
        assert_equal(A.dtype, Ar.dtype)

    def test_negative_resize(self):
        A = np.arange(0, 10, dtype=np.float32)
        new_shape = (-10, -1)
        with pytest.raises((RuntimeError, ValueError)):
            np.resize(A, new_shape=new_shape)


@instantiate_parametrized_tests
class TestNonarrayArgs(TestCase):
    # check that non-array arguments to functions wrap them in arrays
    def test_choose(self):
        choices = [[0, 1, 2], [3, 4, 5], [5, 6, 7]]
        tgt = [5, 1, 5]
        a = [2, 0, 1]

        out = np.choose(a, choices)
        assert_equal(out, tgt)

    def test_clip(self):
        arr = [-1, 5, 2, 3, 10, -4, -9]
        out = np.clip(arr, 2, 7)
        tgt = [2, 5, 2, 3, 7, 2, 2]
        assert_equal(out, tgt)

    @xpassIfTorchDynamo_np  # (reason="TODO implement compress(...)")
    def test_compress(self):
        arr = [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]
        tgt = [[5, 6, 7, 8, 9]]
        out = np.compress([0, 1], arr, axis=0)
        assert_equal(out, tgt)

    def test_count_nonzero(self):
        arr = [[0, 1, 7, 0, 0], [3, 0, 0, 2, 19]]
        tgt = np.array([2, 3])
        out = np.count_nonzero(arr, axis=1)
        assert_equal(out, tgt)

    def test_cumproduct(self):
        A = [[1, 2, 3], [4, 5, 6]]
        assert_(np.all(np.cumprod(A) == np.array([1, 2, 6, 24, 120, 720])))

    def test_diagonal(self):
        a = [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]
        out = np.diagonal(a)
        tgt = [0, 5, 10]

        assert_equal(out, tgt)

    def test_mean(self):
        A = [[1, 2, 3], [4, 5, 6]]
        assert_(np.mean(A) == 3.5)
        assert_(np.all(np.mean(A, 0) == np.array([2.5, 3.5, 4.5])))
        assert_(np.all(np.mean(A, 1) == np.array([2.0, 5.0])))

        #    with warnings.catch_warnings(record=True) as w:
        #        warnings.filterwarnings('always', '', RuntimeWarning)
        assert_(np.isnan(np.mean([])))

    #        assert_(w[0].category is RuntimeWarning)

    def test_ptp(self):
        a = [3, 4, 5, 10, -3, -5, 6.0]
        assert_equal(np.ptp(a, axis=0), 15.0)

    def test_prod(self):
        arr = [[1, 2, 3, 4], [5, 6, 7, 9], [10, 3, 4, 5]]
        tgt = [24, 1890, 600]

        assert_equal(np.prod(arr, axis=-1), tgt)

    def test_ravel(self):
        a = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
        tgt = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        assert_equal(np.ravel(a), tgt)

    def test_repeat(self):
        a = [1, 2, 3]
        tgt = [1, 1, 2, 2, 3, 3]

        out = np.repeat(a, 2)
        assert_equal(out, tgt)

    def test_reshape(self):
        arr = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
        tgt = [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]]
        assert_equal(np.reshape(arr, (2, 6)), tgt)

    def test_round(self):
        arr = [1.56, 72.54, 6.35, 3.25]
        tgt = [1.6, 72.5, 6.4, 3.2]
        assert_equal(np.around(arr, decimals=1), tgt)
        s = np.float64(1.0)
        assert_equal(s.round(), 1.0)

    def test_round_2(self):
        s = np.float64(1.0)
        assert_(isinstance(s.round(), (np.float64, np.ndarray)))

    @xpassIfTorchDynamo_np  # (reason="scalar instances")
    @parametrize(
        "dtype",
        [
            np.int8,
            np.int16,
            np.int32,
            np.int64,
            np.uint8,
            np.float16,
            np.float32,
            np.float64,
        ],
    )
    def test_dunder_round(self, dtype):
        s = dtype(1)
        assert_(isinstance(round(s), int))
        assert_(isinstance(round(s, None), int))
        assert_(isinstance(round(s, ndigits=None), int))
        assert_equal(round(s), 1)
        assert_equal(round(s, None), 1)
        assert_equal(round(s, ndigits=None), 1)

    @parametrize(
        "val, ndigits",
        [
            # pytest.param(
            #    2**31 - 1, -1, marks=pytest.mark.xfail(reason="Out of range of int32")
            # ),
            subtest((2**31 - 1, -1), decorators=[xpassIfTorchDynamo_np]),
            subtest(
                (2**31 - 1, 1 - math.ceil(math.log10(2**31 - 1))),
                decorators=[xpassIfTorchDynamo_np],
            ),
            subtest(
                (2**31 - 1, -math.ceil(math.log10(2**31 - 1))),
                decorators=[xpassIfTorchDynamo_np],
            ),
        ],
    )
    def test_dunder_round_edgecases(self, val, ndigits):
        assert_equal(round(val, ndigits), round(np.int32(val), ndigits))

    @xfail  # (reason="scalar instances")
    def test_dunder_round_accuracy(self):
        f = np.float64(5.1 * 10**73)
        assert_(isinstance(round(f, -73), np.float64))
        assert_array_max_ulp(round(f, -73), 5.0 * 10**73)
        assert_(isinstance(round(f, ndigits=-73), np.float64))
        assert_array_max_ulp(round(f, ndigits=-73), 5.0 * 10**73)

        i = np.int64(501)
        assert_(isinstance(round(i, -2), np.int64))
        assert_array_max_ulp(round(i, -2), 500)
        assert_(isinstance(round(i, ndigits=-2), np.int64))
        assert_array_max_ulp(round(i, ndigits=-2), 500)

    @xfail  # (raises=AssertionError, reason="gh-15896")
    def test_round_py_consistency(self):
        f = 5.1 * 10**73
        assert_equal(round(np.float64(f), -73), round(f, -73))

    def test_searchsorted(self):
        arr = [-8, -5, -1, 3, 6, 10]
        out = np.searchsorted(arr, 0)
        assert_equal(out, 3)

    def test_size(self):
        A = [[1, 2, 3], [4, 5, 6]]
        assert_(np.size(A) == 6)
        assert_(np.size(A, 0) == 2)
        assert_(np.size(A, 1) == 3)

    def test_squeeze(self):
        A = [[[1, 1, 1], [2, 2, 2], [3, 3, 3]]]
        assert_equal(np.squeeze(A).shape, (3, 3))
        assert_equal(np.squeeze(np.zeros((1, 3, 1))).shape, (3,))
        assert_equal(np.squeeze(np.zeros((1, 3, 1)), axis=0).shape, (3, 1))
        assert_equal(np.squeeze(np.zeros((1, 3, 1)), axis=-1).shape, (1, 3))
        assert_equal(np.squeeze(np.zeros((1, 3, 1)), axis=2).shape, (1, 3))
        assert_equal(np.squeeze([np.zeros((3, 1))]).shape, (3,))
        assert_equal(np.squeeze([np.zeros((3, 1))], axis=0).shape, (3, 1))
        assert_equal(np.squeeze([np.zeros((3, 1))], axis=2).shape, (1, 3))
        assert_equal(np.squeeze([np.zeros((3, 1))], axis=-1).shape, (1, 3))

    def test_std(self):
        A = [[1, 2, 3], [4, 5, 6]]
        assert_almost_equal(np.std(A), 1.707825127659933)
        assert_almost_equal(np.std(A, 0), np.array([1.5, 1.5, 1.5]))
        assert_almost_equal(np.std(A, 1), np.array([0.81649658, 0.81649658]))

        #  with warnings.catch_warnings(record=True) as w:
        #      warnings.filterwarnings('always', '', RuntimeWarning)
        assert_(np.isnan(np.std([])))

    #      assert_(w[0].category is RuntimeWarning)

    def test_swapaxes(self):
        tgt = [[[0, 4], [2, 6]], [[1, 5], [3, 7]]]
        a = [[[0, 1], [2, 3]], [[4, 5], [6, 7]]]
        out = np.swapaxes(a, 0, 2)
        assert_equal(out, tgt)

    def test_sum(self):
        m = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        tgt = [[6], [15], [24]]
        out = np.sum(m, axis=1, keepdims=True)

        assert_equal(tgt, out)

    def test_take(self):
        tgt = [2, 3, 5]
        indices = [1, 2, 4]
        a = [1, 2, 3, 4, 5]

        out = np.take(a, indices)
        assert_equal(out, tgt)

    def test_trace(self):
        c = [[1, 2], [3, 4], [5, 6]]
        assert_equal(np.trace(c), 5)

    def test_transpose(self):
        arr = [[1, 2], [3, 4], [5, 6]]
        tgt = [[1, 3, 5], [2, 4, 6]]
        assert_equal(np.transpose(arr, (1, 0)), tgt)

    def test_var(self):
        A = [[1, 2, 3], [4, 5, 6]]
        assert_almost_equal(np.var(A), 2.9166666666666665)
        assert_almost_equal(np.var(A, 0), np.array([2.25, 2.25, 2.25]))
        assert_almost_equal(np.var(A, 1), np.array([0.66666667, 0.66666667]))

        #  with warnings.catch_warnings(record=True) as w:
        #      warnings.filterwarnings('always', '', RuntimeWarning)
        assert_(np.isnan(np.var([])))

    #      assert_(w[0].category is RuntimeWarning)


@xfail  # (reason="TODO")
class TestIsscalar(TestCase):
    def test_isscalar(self):
        assert_(np.isscalar(3.1))
        assert_(np.isscalar(np.int16(12345)))
        assert_(np.isscalar(False))
        assert_(np.isscalar("numpy"))
        assert_(not np.isscalar([3.1]))
        assert_(not np.isscalar(None))

        # PEP 3141
        from fractions import Fraction

        assert_(np.isscalar(Fraction(5, 17)))
        from numbers import Number

        assert_(np.isscalar(Number()))


class TestBoolScalar(TestCase):
    def test_logical(self):
        f = np.False_
        t = np.True_
        s = "xyz"
        assert_((t and s) is s)
        assert_((f and s) is f)

    def test_bitwise_or_eq(self):
        f = np.False_
        t = np.True_
        assert_((t | t) == t)
        assert_((f | t) == t)
        assert_((t | f) == t)
        assert_((f | f) == f)

    def test_bitwise_or_is(self):
        f = np.False_
        t = np.True_
        assert_(bool(t | t) is bool(t))
        assert_(bool(f | t) is bool(t))
        assert_(bool(t | f) is bool(t))
        assert_(bool(f | f) is bool(f))

    def test_bitwise_and_eq(self):
        f = np.False_
        t = np.True_
        assert_((t & t) == t)
        assert_((f & t) == f)
        assert_((t & f) == f)
        assert_((f & f) == f)

    def test_bitwise_and_is(self):
        f = np.False_
        t = np.True_
        assert_(bool(t & t) is bool(t))
        assert_(bool(f & t) is bool(f))
        assert_(bool(t & f) is bool(f))
        assert_(bool(f & f) is bool(f))

    def test_bitwise_xor_eq(self):
        f = np.False_
        t = np.True_
        assert_((t ^ t) == f)
        assert_((f ^ t) == t)
        assert_((t ^ f) == t)
        assert_((f ^ f) == f)

    def test_bitwise_xor_is(self):
        f = np.False_
        t = np.True_
        assert_(bool(t ^ t) is bool(f))
        assert_(bool(f ^ t) is bool(t))
        assert_(bool(t ^ f) is bool(t))
        assert_(bool(f ^ f) is bool(f))


class TestBoolArray(TestCase):
    def setUp(self):
        super().setUp()
        # offset for simd tests
        self.t = np.array([True] * 41, dtype=bool)[1::]
        self.f = np.array([False] * 41, dtype=bool)[1::]
        self.o = np.array([False] * 42, dtype=bool)[2::]
        self.nm = self.f.copy()
        self.im = self.t.copy()
        self.nm[3] = True
        self.nm[-2] = True
        self.im[3] = False
        self.im[-2] = False

    def test_all_any(self):
        assert_(self.t.all())
        assert_(self.t.any())
        assert_(not self.f.all())
        assert_(not self.f.any())
        assert_(self.nm.any())
        assert_(self.im.any())
        assert_(not self.nm.all())
        assert_(not self.im.all())
        # check bad element in all positions
        for i in range(256 - 7):
            d = np.array([False] * 256, dtype=bool)[7::]
            d[i] = True
            assert_(np.any(d))
            e = np.array([True] * 256, dtype=bool)[7::]
            e[i] = False
            assert_(not np.all(e))
            assert_array_equal(e, ~d)
        # big array test for blocked libc loops
        for i in list(range(9, 6000, 507)) + [7764, 90021, -10]:
            d = np.array([False] * 100043, dtype=bool)
            d[i] = True
            assert_(np.any(d), msg=f"{i!r}")
            e = np.array([True] * 100043, dtype=bool)
            e[i] = False
            assert_(not np.all(e), msg=f"{i!r}")

    def test_logical_not_abs(self):
        assert_array_equal(~self.t, self.f)
        assert_array_equal(np.abs(~self.t), self.f)
        assert_array_equal(np.abs(~self.f), self.t)
        assert_array_equal(np.abs(self.f), self.f)
        assert_array_equal(~np.abs(self.f), self.t)
        assert_array_equal(~np.abs(self.t), self.f)
        assert_array_equal(np.abs(~self.nm), self.im)
        np.logical_not(self.t, out=self.o)
        assert_array_equal(self.o, self.f)
        np.abs(self.t, out=self.o)
        assert_array_equal(self.o, self.t)

    def test_logical_and_or_xor(self):
        assert_array_equal(self.t | self.t, self.t)
        assert_array_equal(self.f | self.f, self.f)
        assert_array_equal(self.t | self.f, self.t)
        assert_array_equal(self.f | self.t, self.t)
        np.logical_or(self.t, self.t, out=self.o)
        assert_array_equal(self.o, self.t)
        assert_array_equal(self.t & self.t, self.t)
        assert_array_equal(self.f & self.f, self.f)
        assert_array_equal(self.t & self.f, self.f)
        assert_array_equal(self.f & self.t, self.f)
        np.logical_and(self.t, self.t, out=self.o)
        assert_array_equal(self.o, self.t)
        assert_array_equal(self.t ^ self.t, self.f)
        assert_array_equal(self.f ^ self.f, self.f)
        assert_array_equal(self.t ^ self.f, self.t)
        assert_array_equal(self.f ^ self.t, self.t)
        np.logical_xor(self.t, self.t, out=self.o)
        assert_array_equal(self.o, self.f)

        assert_array_equal(self.nm & self.t, self.nm)
        assert_array_equal(self.im & self.f, False)
        assert_array_equal(self.nm & True, self.nm)
        assert_array_equal(self.im & False, self.f)
        assert_array_equal(self.nm | self.t, self.t)
        assert_array_equal(self.im | self.f, self.im)
        assert_array_equal(self.nm | True, self.t)
        assert_array_equal(self.im | False, self.im)
        assert_array_equal(self.nm ^ self.t, self.im)
        assert_array_equal(self.im ^ self.f, self.im)
        assert_array_equal(self.nm ^ True, self.im)
        assert_array_equal(self.im ^ False, self.im)


@xfailIfTorchDynamo
class TestBoolCmp(TestCase):
    def setUp(self):
        super().setUp()
        self.f = np.ones(256, dtype=np.float32)
        self.ef = np.ones(self.f.size, dtype=bool)
        self.d = np.ones(128, dtype=np.float64)
        self.ed = np.ones(self.d.size, dtype=bool)
        # generate values for all permutation of 256bit simd vectors
        s = 0
        for i in range(32):
            self.f[s : s + 8] = [i & 2**x for x in range(8)]
            self.ef[s : s + 8] = [(i & 2**x) != 0 for x in range(8)]
            s += 8
        s = 0
        for i in range(16):
            self.d[s : s + 4] = [i & 2**x for x in range(4)]
            self.ed[s : s + 4] = [(i & 2**x) != 0 for x in range(4)]
            s += 4

        self.nf = self.f.copy()
        self.nd = self.d.copy()
        self.nf[self.ef] = np.nan
        self.nd[self.ed] = np.nan

        self.inff = self.f.copy()
        self.infd = self.d.copy()
        self.inff[::3][self.ef[::3]] = np.inf
        self.infd[::3][self.ed[::3]] = np.inf
        self.inff[1::3][self.ef[1::3]] = -np.inf
        self.infd[1::3][self.ed[1::3]] = -np.inf
        self.inff[2::3][self.ef[2::3]] = np.nan
        self.infd[2::3][self.ed[2::3]] = np.nan
        self.efnonan = self.ef.copy()
        self.efnonan[2::3] = False
        self.ednonan = self.ed.copy()
        self.ednonan[2::3] = False

        self.signf = self.f.copy()
        self.signd = self.d.copy()
        self.signf[self.ef] *= -1.0
        self.signd[self.ed] *= -1.0
        self.signf[1::6][self.ef[1::6]] = -np.inf
        self.signd[1::6][self.ed[1::6]] = -np.inf
        self.signf[3::6][self.ef[3::6]] = -np.nan
        self.signd[3::6][self.ed[3::6]] = -np.nan
        self.signf[4::6][self.ef[4::6]] = -0.0
        self.signd[4::6][self.ed[4::6]] = -0.0

    def test_float(self):
        # offset for alignment test
        for i in range(4):
            assert_array_equal(self.f[i:] > 0, self.ef[i:])
            assert_array_equal(self.f[i:] - 1 >= 0, self.ef[i:])
            assert_array_equal(self.f[i:] == 0, ~self.ef[i:])
            assert_array_equal(-self.f[i:] < 0, self.ef[i:])
            assert_array_equal(-self.f[i:] + 1 <= 0, self.ef[i:])
            r = self.f[i:] != 0
            assert_array_equal(r, self.ef[i:])
            r2 = self.f[i:] != np.zeros_like(self.f[i:])
            r3 = 0 != self.f[i:]
            assert_array_equal(r, r2)
            assert_array_equal(r, r3)
            # check bool == 0x1
            assert_array_equal(r.view(np.int8), r.astype(np.int8))
            assert_array_equal(r2.view(np.int8), r2.astype(np.int8))
            assert_array_equal(r3.view(np.int8), r3.astype(np.int8))

            # isnan on amd64 takes the same code path
            assert_array_equal(np.isnan(self.nf[i:]), self.ef[i:])
            assert_array_equal(np.isfinite(self.nf[i:]), ~self.ef[i:])
            assert_array_equal(np.isfinite(self.inff[i:]), ~self.ef[i:])
            assert_array_equal(np.isinf(self.inff[i:]), self.efnonan[i:])
            assert_array_equal(np.signbit(self.signf[i:]), self.ef[i:])

    def test_double(self):
        # offset for alignment test
        for i in range(2):
            assert_array_equal(self.d[i:] > 0, self.ed[i:])
            assert_array_equal(self.d[i:] - 1 >= 0, self.ed[i:])
            assert_array_equal(self.d[i:] == 0, ~self.ed[i:])
            assert_array_equal(-self.d[i:] < 0, self.ed[i:])
            assert_array_equal(-self.d[i:] + 1 <= 0, self.ed[i:])
            r = self.d[i:] != 0
            assert_array_equal(r, self.ed[i:])
            r2 = self.d[i:] != np.zeros_like(self.d[i:])
            r3 = 0 != self.d[i:]
            assert_array_equal(r, r2)
            assert_array_equal(r, r3)
            # check bool == 0x1
            assert_array_equal(r.view(np.int8), r.astype(np.int8))
            assert_array_equal(r2.view(np.int8), r2.astype(np.int8))
            assert_array_equal(r3.view(np.int8), r3.astype(np.int8))

            # isnan on amd64 takes the same code path
            assert_array_equal(np.isnan(self.nd[i:]), self.ed[i:])
            assert_array_equal(np.isfinite(self.nd[i:]), ~self.ed[i:])
            assert_array_equal(np.isfinite(self.infd[i:]), ~self.ed[i:])
            assert_array_equal(np.isinf(self.infd[i:]), self.ednonan[i:])
            assert_array_equal(np.signbit(self.signd[i:]), self.ed[i:])


@xpassIfTorchDynamo_np  # (reason="TODO")
class TestSeterr(TestCase):
    def test_default(self):
        err = np.geterr()
        assert_equal(
            err, dict(divide="warn", invalid="warn", over="warn", under="ignore")
        )

    def test_set(self):
        err = np.seterr()
        old = np.seterr(divide="print")
        assert_(err == old)
        new = np.seterr()
        assert_(new["divide"] == "print")
        np.seterr(over="raise")
        assert_(np.geterr()["over"] == "raise")
        assert_(new["divide"] == "print")
        np.seterr(**old)
        assert_(np.geterr() == old)

    @xfail
    @skipif(IS_WASM, reason="no wasm fp exception support")
    @skipif(platform.machine() == "armv5tel", reason="See gh-413.")
    def test_divide_err(self):
        with assert_raises(FloatingPointError):
            np.array([1.0]) / np.array([0.0])

        np.seterr(divide="ignore")
        np.array([1.0]) / np.array([0.0])

    @skipif(IS_WASM, reason="no wasm fp exception support")
    def test_errobj(self):
        olderrobj = np.geterrobj()
        self.called = 0
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                np.seterrobj([20000, 1, None])
                np.array([1.0]) / np.array([0.0])
                assert_equal(len(w), 1)

            def log_err(*args):
                self.called += 1
                extobj_err = args
                assert_(len(extobj_err) == 2)
                assert_("divide" in extobj_err[0])

            np.seterrobj([20000, 3, log_err])
            np.array([1.0]) / np.array([0.0])
            assert_equal(self.called, 1)

            np.seterrobj(olderrobj)
            np.divide(1.0, 0.0, extobj=[20000, 3, log_err])
            assert_equal(self.called, 2)
        finally:
            np.seterrobj(olderrobj)
            del self.called


@xfail  # (reason="TODO")
@instantiate_parametrized_tests
class TestFloatExceptions(TestCase):
    def assert_raises_fpe(self, fpeerr, flop, x, y):
        ftype = type(x)
        try:
            flop(x, y)
            assert_(False, f"Type {ftype} did not raise fpe error '{fpeerr}'.")
        except FloatingPointError as exc:
            assert_(
                str(exc).find(fpeerr) >= 0,
                f"Type {ftype} raised wrong fpe error '{exc}'.",
            )

    def assert_op_raises_fpe(self, fpeerr, flop, sc1, sc2):
        # Check that fpe exception is raised.
        #
        # Given a floating operation `flop` and two scalar values, check that
        # the operation raises the floating point exception specified by
        # `fpeerr`. Tests all variants with 0-d array scalars as well.

        self.assert_raises_fpe(fpeerr, flop, sc1, sc2)
        self.assert_raises_fpe(fpeerr, flop, sc1[()], sc2)
        self.assert_raises_fpe(fpeerr, flop, sc1, sc2[()])
        self.assert_raises_fpe(fpeerr, flop, sc1[()], sc2[()])

    # Test for all real and complex float types
    @skipif(IS_WASM, reason="no wasm fp exception support")
    @parametrize("typecode", np.typecodes["AllFloat"])
    def test_floating_exceptions(self, typecode):
        # Test basic arithmetic function errors
        ftype = np.dtype(typecode).type
        if np.dtype(ftype).kind == "f":
            # Get some extreme values for the type
            fi = np.finfo(ftype)
            ft_tiny = fi._machar.tiny
            ft_max = fi.max
            ft_eps = fi.eps
            underflow = "underflow"
            divbyzero = "divide by zero"
        else:
            # 'c', complex, corresponding real dtype
            rtype = type(ftype(0).real)
            fi = np.finfo(rtype)
            ft_tiny = ftype(fi._machar.tiny)
            ft_max = ftype(fi.max)
            ft_eps = ftype(fi.eps)
            # The complex types raise different exceptions
            underflow = ""
            divbyzero = ""
        overflow = "overflow"
        invalid = "invalid"

        # The value of tiny for double double is NaN, so we need to
        # pass the assert
        if not np.isnan(ft_tiny):
            self.assert_raises_fpe(underflow, operator.truediv, ft_tiny, ft_max)
            self.assert_raises_fpe(underflow, operator.mul, ft_tiny, ft_tiny)
        self.assert_raises_fpe(overflow, operator.mul, ft_max, ftype(2))
        self.assert_raises_fpe(overflow, operator.truediv, ft_max, ftype(0.5))
        self.assert_raises_fpe(overflow, operator.add, ft_max, ft_max * ft_eps)
        self.assert_raises_fpe(overflow, operator.sub, -ft_max, ft_max * ft_eps)
        self.assert_raises_fpe(overflow, np.power, ftype(2), ftype(2**fi.nexp))
        self.assert_raises_fpe(divbyzero, operator.truediv, ftype(1), ftype(0))
        self.assert_raises_fpe(invalid, operator.truediv, ftype(np.inf), ftype(np.inf))
        self.assert_raises_fpe(invalid, operator.truediv, ftype(0), ftype(0))
        self.assert_raises_fpe(invalid, operator.sub, ftype(np.inf), ftype(np.inf))
        self.assert_raises_fpe(invalid, operator.add, ftype(np.inf), ftype(-np.inf))
        self.assert_raises_fpe(invalid, operator.mul, ftype(0), ftype(np.inf))

    @skipif(IS_WASM, reason="no wasm fp exception support")
    def test_warnings(self):
        # test warning code path
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            np.divide(1, 0.0)
            assert_equal(len(w), 1)
            assert_("divide by zero" in str(w[0].message))
            np.array(1e300) * np.array(1e300)
            assert_equal(len(w), 2)
            assert_("overflow" in str(w[-1].message))
            np.array(np.inf) - np.array(np.inf)
            assert_equal(len(w), 3)
            assert_("invalid value" in str(w[-1].message))
            np.array(1e-300) * np.array(1e-300)
            assert_equal(len(w), 4)
            assert_("underflow" in str(w[-1].message))


class TestTypes(TestCase):
    def check_promotion_cases(self, promote_func):
        # tests that the scalars get coerced correctly.
        b = np.bool_(0)
        i8, i16, i32, i64 = np.int8(0), np.int16(0), np.int32(0), np.int64(0)
        u8 = np.uint8(0)
        f32, f64 = np.float32(0), np.float64(0)
        c64, c128 = np.complex64(0), np.complex128(0)

        # coercion within the same kind
        assert_equal(promote_func(i8, i16), np.dtype(np.int16))
        assert_equal(promote_func(i32, i8), np.dtype(np.int32))
        assert_equal(promote_func(i16, i64), np.dtype(np.int64))
        assert_equal(promote_func(f32, f64), np.dtype(np.float64))
        assert_equal(promote_func(c128, c64), np.dtype(np.complex128))

        # coercion between kinds
        assert_equal(promote_func(b, i32), np.dtype(np.int32))
        assert_equal(promote_func(b, u8), np.dtype(np.uint8))
        assert_equal(promote_func(i8, u8), np.dtype(np.int16))
        assert_equal(promote_func(u8, i32), np.dtype(np.int32))

        assert_equal(promote_func(f32, i16), np.dtype(np.float32))
        assert_equal(promote_func(f32, c64), np.dtype(np.complex64))
        assert_equal(promote_func(c128, f32), np.dtype(np.complex128))

        # coercion between scalars and 1-D arrays
        assert_equal(promote_func(np.array([b]), i8), np.dtype(np.int8))
        assert_equal(promote_func(np.array([b]), u8), np.dtype(np.uint8))
        assert_equal(promote_func(np.array([b]), i32), np.dtype(np.int32))
        assert_equal(promote_func(c64, np.array([f64])), np.dtype(np.complex128))
        assert_equal(
            promote_func(np.complex64(3j), np.array([f64])), np.dtype(np.complex128)
        )

        # coercion between scalars and 1-D arrays, where
        # the scalar has greater kind than the array
        assert_equal(promote_func(np.array([b]), f64), np.dtype(np.float64))
        assert_equal(promote_func(np.array([b]), i64), np.dtype(np.int64))
        assert_equal(promote_func(np.array([i8]), f64), np.dtype(np.float64))

    def check_promotion_cases_2(self, promote_func):
        # these are failing because of the "scalars do not upcast arrays" rule
        # Two first tests (i32 + f32 -> f64, and i64+f32 -> f64) xfail
        # until ufuncs implement the proper type promotion (ufunc loops?)
        i8, i32, i64 = np.int8(0), np.int32(0), np.int64(0)
        f32, f64 = np.float32(0), np.float64(0)
        c128 = np.complex128(0)

        assert_equal(promote_func(i32, f32), np.dtype(np.float64))
        assert_equal(promote_func(i64, f32), np.dtype(np.float64))

        assert_equal(promote_func(np.array([i8]), i64), np.dtype(np.int8))
        assert_equal(promote_func(f64, np.array([f32])), np.dtype(np.float32))

        # float and complex are treated as the same "kind" for
        # the purposes of array-scalar promotion, so that you can do
        # (0j + float32array) to get a complex64 array instead of
        # a complex128 array.
        assert_equal(promote_func(np.array([f32]), c128), np.dtype(np.complex64))

    def test_coercion(self):
        def res_type(a, b):
            return np.add(a, b).dtype

        self.check_promotion_cases(res_type)

        # Use-case: float/complex scalar * bool/int8 array
        #           shouldn't narrow the float/complex type
        for a in [np.array([True, False]), np.array([-3, 12], dtype=np.int8)]:
            b = 1.234 * a
            assert_equal(b.dtype, np.dtype("f8"), f"array type {a.dtype}")
            b = np.float64(1.234) * a
            assert_equal(b.dtype, np.dtype("f8"), f"array type {a.dtype}")
            b = np.float32(1.234) * a
            assert_equal(b.dtype, np.dtype("f4"), f"array type {a.dtype}")
            b = np.float16(1.234) * a
            assert_equal(b.dtype, np.dtype("f2"), f"array type {a.dtype}")

            b = 1.234j * a
            assert_equal(b.dtype, np.dtype("c16"), f"array type {a.dtype}")
            b = np.complex128(1.234j) * a
            assert_equal(b.dtype, np.dtype("c16"), f"array type {a.dtype}")
            b = np.complex64(1.234j) * a
            assert_equal(b.dtype, np.dtype("c8"), f"array type {a.dtype}")

        # The following use-case is problematic, and to resolve its
        # tricky side-effects requires more changes.
        #
        # Use-case: (1-t)*a, where 't' is a boolean array and 'a' is
        #            a float32, shouldn't promote to float64
        #
        # a = np.array([1.0, 1.5], dtype=np.float32)
        # t = np.array([True, False])
        # b = t*a
        # assert_equal(b, [1.0, 0.0])
        # assert_equal(b.dtype, np.dtype('f4'))
        # b = (1-t)*a
        # assert_equal(b, [0.0, 1.5])
        # assert_equal(b.dtype, np.dtype('f4'))
        #
        # Probably ~t (bitwise negation) is more proper to use here,
        # but this is arguably less intuitive to understand at a glance, and
        # would fail if 't' is actually an integer array instead of boolean:
        #
        # b = (~t)*a
        # assert_equal(b, [0.0, 1.5])
        # assert_equal(b.dtype, np.dtype('f4'))

    @xpassIfTorchDynamo_np  # (reason="'Scalars do not upcast arrays' rule")
    def test_coercion_2(self):
        def res_type(a, b):
            return np.add(a, b).dtype

        self.check_promotion_cases_2(res_type)

    def test_result_type(self):
        self.check_promotion_cases(np.result_type)

    @skip(reason="array(None) not supported")
    def test_tesult_type_2(self):
        assert_(np.result_type(None) == np.dtype(None))

    @skip(reason="no endianness in dtypes")
    def test_promote_types_endian(self):
        # promote_types should always return native-endian types
        assert_equal(np.promote_types("<i8", "<i8"), np.dtype("i8"))
        assert_equal(np.promote_types(">i8", ">i8"), np.dtype("i8"))

    def test_can_cast(self):
        assert_(np.can_cast(np.int32, np.int64))
        assert_(np.can_cast(np.float64, complex))
        assert_(not np.can_cast(complex, float))

        assert_(np.can_cast("i8", "f8"))
        assert_(not np.can_cast("i8", "f4"))

        assert_(np.can_cast("i8", "i8", "no"))

    @skip(reason="no endianness in dtypes")
    def test_can_cast_2(self):
        assert_(not np.can_cast("<i8", ">i8", "no"))

        assert_(np.can_cast("<i8", ">i8", "equiv"))
        assert_(not np.can_cast("<i4", ">i8", "equiv"))

        assert_(np.can_cast("<i4", ">i8", "safe"))
        assert_(not np.can_cast("<i8", ">i4", "safe"))

        assert_(np.can_cast("<i8", ">i4", "same_kind"))
        assert_(not np.can_cast("<i8", ">u4", "same_kind"))

        assert_(np.can_cast("<i8", ">u4", "unsafe"))

        assert_raises(TypeError, np.can_cast, "i4", None)
        assert_raises(TypeError, np.can_cast, None, "i4")

        # Also test keyword arguments
        assert_(np.can_cast(from_=np.int32, to=np.int64))

    @xpassIfTorchDynamo_np  # (reason="value-based casting?")
    def test_can_cast_values(self):
        # gh-5917
        for dt in [np.int8, np.int16, np.int32, np.int64] + [
            np.uint8,
            np.uint16,
            np.uint32,
            np.uint64,
        ]:
            ii = np.iinfo(dt)
            assert_(np.can_cast(ii.min, dt))
            assert_(np.can_cast(ii.max, dt))
            assert_(not np.can_cast(ii.min - 1, dt))
            assert_(not np.can_cast(ii.max + 1, dt))

        for dt in [np.float16, np.float32, np.float64, np.longdouble]:
            fi = np.finfo(dt)
            assert_(np.can_cast(fi.min, dt))
            assert_(np.can_cast(fi.max, dt))


# Custom exception class to test exception propagation in fromiter
class NIterError(Exception):
    pass


@skip(reason="NP_VER: fails on CI")
@xpassIfTorchDynamo_np  # (reason="TODO")
@instantiate_parametrized_tests
class TestFromiter(TestCase):
    def makegen(self):
        return (x**2 for x in range(24))

    def test_types(self):
        ai32 = np.fromiter(self.makegen(), np.int32)
        ai64 = np.fromiter(self.makegen(), np.int64)
        af = np.fromiter(self.makegen(), float)
        assert_(ai32.dtype == np.dtype(np.int32))
        assert_(ai64.dtype == np.dtype(np.int64))
        assert_(af.dtype == np.dtype(float))

    def test_lengths(self):
        expected = np.array(list(self.makegen()))
        a = np.fromiter(self.makegen(), int)
        a20 = np.fromiter(self.makegen(), int, 20)
        assert_(len(a) == len(expected))
        assert_(len(a20) == 20)
        assert_raises(ValueError, np.fromiter, self.makegen(), int, len(expected) + 10)

    def test_values(self):
        expected = np.array(list(self.makegen()))
        a = np.fromiter(self.makegen(), int)
        a20 = np.fromiter(self.makegen(), int, 20)
        assert_(np.all(a == expected, axis=0))
        assert_(np.all(a20 == expected[:20], axis=0))

    def load_data(self, n, eindex):
        # Utility method for the issue 2592 tests.
        # Raise an exception at the desired index in the iterator.
        for e in range(n):
            if e == eindex:
                raise NIterError(f"error at index {eindex}")
            yield e

    @parametrize("dtype", [int])
    @parametrize("count, error_index", [(10, 5), (10, 9)])
    def test_2592(self, count, error_index, dtype):
        # Test iteration exceptions are correctly raised. The data/generator
        # has `count` elements but errors at `error_index`
        iterable = self.load_data(count, error_index)
        with pytest.raises(NIterError):
            np.fromiter(iterable, dtype=dtype, count=count)

    @skip(reason="NP_VER: fails on CI")
    def test_empty_result(self):
        class MyIter:
            def __length_hint__(self):
                return 10

            def __iter__(self):
                return iter([])  # actual iterator is empty.

        res = np.fromiter(MyIter(), dtype="d")
        assert res.shape == (0,)
        assert res.dtype == "d"

    def test_too_few_items(self):
        msg = "iterator too short: Expected 10 but iterator had only 3 items."
        with pytest.raises(ValueError, match=msg):
            np.fromiter([1, 2, 3], count=10, dtype=int)

    def test_failed_itemsetting(self):
        with pytest.raises(TypeError):
            np.fromiter([1, None, 3], dtype=int)

        # The following manages to hit somewhat trickier code paths:
        iterable = ((2, 3, 4) for i in range(5))
        with pytest.raises(ValueError):
            np.fromiter(iterable, dtype=np.dtype((int, 2)))


@instantiate_parametrized_tests
class TestNonzeroAndCountNonzero(TestCase):
    def test_count_nonzero_list(self):
        lst = [[0, 1, 2, 3], [1, 0, 0, 6]]
        assert np.count_nonzero(lst) == 5
        assert_array_equal(np.count_nonzero(lst, axis=0), np.array([1, 1, 1, 2]))
        assert_array_equal(np.count_nonzero(lst, axis=1), np.array([3, 2]))

    def test_nonzero_trivial(self):
        assert_equal(np.count_nonzero(np.array([])), 0)
        assert_equal(np.count_nonzero(np.array([], dtype="?")), 0)
        assert_equal(np.nonzero(np.array([])), ([],))

        assert_equal(np.count_nonzero(np.array([0])), 0)
        assert_equal(np.count_nonzero(np.array([0], dtype="?")), 0)
        assert_equal(np.nonzero(np.array([0])), ([],))

        assert_equal(np.count_nonzero(np.array([1])), 1)
        assert_equal(np.count_nonzero(np.array([1], dtype="?")), 1)
        assert_equal(np.nonzero(np.array([1])), ([0],))

    def test_nonzero_trivial_differs(self):
        # numpy returns a python int, we return a 0D array
        assert isinstance(np.count_nonzero([]), np.ndarray)

    def test_nonzero_zerod(self):
        assert_equal(np.count_nonzero(np.array(0)), 0)
        assert_equal(np.count_nonzero(np.array(0, dtype="?")), 0)

        assert_equal(np.count_nonzero(np.array(1)), 1)
        assert_equal(np.count_nonzero(np.array(1, dtype="?")), 1)

    def test_nonzero_zerod_differs(self):
        # numpy returns a python int, we return a 0D array
        assert isinstance(np.count_nonzero(np.array(1)), np.ndarray)

    def test_nonzero_onedim(self):
        x = np.array([1, 0, 2, -1, 0, 0, 8])
        assert_equal(np.count_nonzero(x), 4)
        assert_equal(np.count_nonzero(x), 4)
        assert_equal(np.nonzero(x), ([0, 2, 3, 6],))

    def test_nonzero_onedim_differs(self):
        # numpy returns a python int, we return a 0D array
        x = np.array([1, 0, 2, -1, 0, 0, 8])
        assert isinstance(np.count_nonzero(x), np.ndarray)

    def test_nonzero_twodim(self):
        x = np.array([[0, 1, 0], [2, 0, 3]])
        assert_equal(np.count_nonzero(x.astype("i1")), 3)
        assert_equal(np.count_nonzero(x.astype("i2")), 3)
        assert_equal(np.count_nonzero(x.astype("i4")), 3)
        assert_equal(np.count_nonzero(x.astype("i8")), 3)
        assert_equal(np.nonzero(x), ([0, 1, 1], [1, 0, 2]))

        x = np.eye(3)
        assert_equal(np.count_nonzero(x.astype("i1")), 3)
        assert_equal(np.count_nonzero(x.astype("i2")), 3)
        assert_equal(np.count_nonzero(x.astype("i4")), 3)
        assert_equal(np.count_nonzero(x.astype("i8")), 3)
        assert_equal(np.nonzero(x), ([0, 1, 2], [0, 1, 2]))

    def test_sparse(self):
        # test special sparse condition boolean code path
        for i in range(20):
            c = np.zeros(200, dtype=bool)
            c[i::20] = True
            assert_equal(np.nonzero(c)[0], np.arange(i, 200 + i, 20))

            c = np.zeros(400, dtype=bool)
            c[10 + i : 20 + i] = True
            c[20 + i * 2] = True
            assert_equal(
                np.nonzero(c)[0],
                np.concatenate((np.arange(10 + i, 20 + i), [20 + i * 2])),
            )

    def test_count_nonzero_axis(self):
        # Basic check of functionality
        m = np.array([[0, 1, 7, 0, 0], [3, 0, 0, 2, 19]])

        expected = np.array([1, 1, 1, 1, 1])
        assert_array_equal(np.count_nonzero(m, axis=0), expected)

        expected = np.array([2, 3])
        assert_array_equal(np.count_nonzero(m, axis=1), expected)

        assert isinstance(np.count_nonzero(m, axis=1), np.ndarray)

        assert_raises(ValueError, np.count_nonzero, m, axis=(1, 1))
        assert_raises(TypeError, np.count_nonzero, m, axis="foo")
        assert_raises(np.AxisError, np.count_nonzero, m, axis=3)
        assert_raises(TypeError, np.count_nonzero, m, axis=np.array([[1], [2]]))

    @parametrize("typecode", "efdFDBbhil?")
    def test_count_nonzero_axis_all_dtypes(self, typecode):
        # More thorough test that the axis argument is respected
        # for all dtypes and responds correctly when presented with
        # either integer or tuple arguments for axis

        m = np.zeros((3, 3), dtype=typecode)
        n = np.ones(1, dtype=typecode)

        m[0, 0] = n[0]
        m[1, 0] = n[0]

        expected = np.array([2, 0, 0], dtype=np.intp)
        result = np.count_nonzero(m, axis=0)
        assert_array_equal(result, expected)
        assert expected.dtype == result.dtype

        expected = np.array([1, 1, 0], dtype=np.intp)
        result = np.count_nonzero(m, axis=1)
        assert_array_equal(result, expected)
        assert expected.dtype == result.dtype

        expected = np.array(2)
        assert_array_equal(np.count_nonzero(m, axis=(0, 1)), expected)
        assert_array_equal(np.count_nonzero(m, axis=None), expected)
        assert_array_equal(np.count_nonzero(m), expected)

    def test_countnonzero_axis_empty(self):
        a = np.array([[0, 0, 1], [1, 0, 1]])
        assert_equal(np.count_nonzero(a, axis=()), a.astype(bool))

    def test_countnonzero_keepdims(self):
        a = np.array([[0, 0, 1, 0], [0, 3, 5, 0], [7, 9, 2, 0]])
        assert_array_equal(np.count_nonzero(a, axis=0, keepdims=True), [[1, 2, 3, 0]])
        assert_array_equal(np.count_nonzero(a, axis=1, keepdims=True), [[1], [2], [3]])
        assert_array_equal(np.count_nonzero(a, keepdims=True), [[6]])
        assert isinstance(np.count_nonzero(a, axis=1, keepdims=True), np.ndarray)


class TestIndex(TestCase):
    def test_boolean(self):
        a = rand(3, 5, 8)
        V = rand(5, 8)
        g1 = randint(0, 5, size=15)
        g2 = randint(0, 8, size=15)
        V[g1, g2] = -V[g1, g2]
        assert_(
            (np.array([a[0][V > 0], a[1][V > 0], a[2][V > 0]]) == a[:, V > 0]).all()
        )

    def test_boolean_edgecase(self):
        a = np.array([], dtype="int32")
        b = np.array([], dtype="bool")
        c = a[b]
        assert_equal(c, [])
        assert_equal(c.dtype, np.dtype("int32"))


@xpassIfTorchDynamo_np  # (reason="TODO")
class TestBinaryRepr(TestCase):
    def test_zero(self):
        assert_equal(np.binary_repr(0), "0")

    def test_positive(self):
        assert_equal(np.binary_repr(10), "1010")
        assert_equal(np.binary_repr(12522), "11000011101010")
        assert_equal(np.binary_repr(10736848), "101000111101010011010000")

    def test_negative(self):
        assert_equal(np.binary_repr(-1), "-1")
        assert_equal(np.binary_repr(-10), "-1010")
        assert_equal(np.binary_repr(-12522), "-11000011101010")
        assert_equal(np.binary_repr(-10736848), "-101000111101010011010000")

    def test_sufficient_width(self):
        assert_equal(np.binary_repr(0, width=5), "00000")
        assert_equal(np.binary_repr(10, width=7), "0001010")
        assert_equal(np.binary_repr(-5, width=7), "1111011")

    def test_neg_width_boundaries(self):
        # see gh-8670

        # Ensure that the example in the issue does not
        # break before proceeding to a more thorough test.
        assert_equal(np.binary_repr(-128, width=8), "10000000")

        for width in range(1, 11):
            num = -(2 ** (width - 1))
            exp = "1" + (width - 1) * "0"
            assert_equal(np.binary_repr(num, width=width), exp)

    def test_large_neg_int64(self):
        # See gh-14289.
        assert_equal(np.binary_repr(np.int64(-(2**62)), width=64), "11" + "0" * 62)


@xpassIfTorchDynamo_np  # (reason="TODO")
class TestBaseRepr(TestCase):
    def test_base3(self):
        assert_equal(np.base_repr(3**5, 3), "100000")

    def test_positive(self):
        assert_equal(np.base_repr(12, 10), "12")
        assert_equal(np.base_repr(12, 10, 4), "000012")
        assert_equal(np.base_repr(12, 4), "30")
        assert_equal(np.base_repr(3731624803700888, 36), "10QR0ROFCEW")

    def test_negative(self):
        assert_equal(np.base_repr(-12, 10), "-12")
        assert_equal(np.base_repr(-12, 10, 4), "-000012")
        assert_equal(np.base_repr(-12, 4), "-30")

    def test_base_range(self):
        with assert_raises(ValueError):
            np.base_repr(1, 1)
        with assert_raises(ValueError):
            np.base_repr(1, 37)


class TestArrayComparisons(TestCase):
    def test_array_equal(self):
        res = np.array_equal(np.array([1, 2]), np.array([1, 2]))
        assert_(res)
        assert_(type(res) is bool)
        res = np.array_equal(np.array([1, 2]), np.array([1, 2, 3]))
        assert_(not res)
        assert_(type(res) is bool)
        res = np.array_equal(np.array([1, 2]), np.array([3, 4]))
        assert_(not res)
        assert_(type(res) is bool)
        res = np.array_equal(np.array([1, 2]), np.array([1, 3]))
        assert_(not res)
        assert_(type(res) is bool)

    def test_array_equal_equal_nan(self):
        # Test array_equal with equal_nan kwarg
        a1 = np.array([1, 2, np.nan])
        a2 = np.array([1, np.nan, 2])
        a3 = np.array([1, 2, np.inf])

        # equal_nan=False by default
        assert_(not np.array_equal(a1, a1))
        assert_(np.array_equal(a1, a1, equal_nan=True))
        assert_(not np.array_equal(a1, a2, equal_nan=True))
        # nan's not conflated with inf's
        assert_(not np.array_equal(a1, a3, equal_nan=True))
        # 0-D arrays
        a = np.array(np.nan)
        assert_(not np.array_equal(a, a))
        assert_(np.array_equal(a, a, equal_nan=True))
        # Non-float dtype - equal_nan should have no effect
        a = np.array([1, 2, 3], dtype=int)
        assert_(np.array_equal(a, a))
        assert_(np.array_equal(a, a, equal_nan=True))
        # Multi-dimensional array
        a = np.array([[0, 1], [np.nan, 1]])
        assert_(not np.array_equal(a, a))
        assert_(np.array_equal(a, a, equal_nan=True))
        # Complex values
        a, b = [np.array([1 + 1j])] * 2
        a.real, b.imag = np.nan, np.nan
        assert_(not np.array_equal(a, b, equal_nan=False))
        assert_(np.array_equal(a, b, equal_nan=True))

    def test_none_compares_elementwise(self):
        a = np.ones(3)
        assert_equal(a.__eq__(None), [False, False, False])
        assert_equal(a.__ne__(None), [True, True, True])

    def test_array_equiv(self):
        res = np.array_equiv(np.array([1, 2]), np.array([1, 2]))
        assert_(res)
        assert_(type(res) is bool)
        res = np.array_equiv(np.array([1, 2]), np.array([1, 2, 3]))
        assert_(not res)
        assert_(type(res) is bool)
        res = np.array_equiv(np.array([1, 2]), np.array([3, 4]))
        assert_(not res)
        assert_(type(res) is bool)
        res = np.array_equiv(np.array([1, 2]), np.array([1, 3]))
        assert_(not res)
        assert_(type(res) is bool)

        res = np.array_equiv(np.array([1, 1]), np.array([1]))
        assert_(res)
        assert_(type(res) is bool)
        res = np.array_equiv(np.array([1, 1]), np.array([[1], [1]]))
        assert_(res)
        assert_(type(res) is bool)
        res = np.array_equiv(np.array([1, 2]), np.array([2]))
        assert_(not res)
        assert_(type(res) is bool)
        res = np.array_equiv(np.array([1, 2]), np.array([[1], [2]]))
        assert_(not res)
        assert_(type(res) is bool)
        res = np.array_equiv(
            np.array([1, 2]), np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        )
        assert_(not res)
        assert_(type(res) is bool)


@instantiate_parametrized_tests
class TestClip(TestCase):
    def setUp(self):
        super().setUp()
        self.nr = 5
        self.nc = 3

    def fastclip(self, a, m, M, out=None, casting=None):
        if out is None:
            if casting is None:
                return a.clip(m, M)
            else:
                return a.clip(m, M, casting=casting)
        else:
            if casting is None:
                return a.clip(m, M, out)
            else:
                return a.clip(m, M, out, casting=casting)

    def clip(self, a, m, M, out=None):
        # use slow-clip
        selector = np.less(a, m) + 2 * np.greater(a, M)
        return selector.choose((a, m, M), out=out)

    # Handy functions
    def _generate_data(self, n, m):
        return randn(n, m)

    def _generate_data_complex(self, n, m):
        return randn(n, m) + 1.0j * rand(n, m)

    def _generate_flt_data(self, n, m):
        return (randn(n, m)).astype(np.float32)

    def _neg_byteorder(self, a):
        a = np.asarray(a)
        if sys.byteorder == "little":
            a = a.astype(a.dtype.newbyteorder(">"))
        else:
            a = a.astype(a.dtype.newbyteorder("<"))
        return a

    def _generate_non_native_data(self, n, m):
        data = randn(n, m)
        data = self._neg_byteorder(data)
        assert_(not data.dtype.isnative)
        return data

    def _generate_int_data(self, n, m):
        return (10 * rand(n, m)).astype(np.int64)

    def _generate_int32_data(self, n, m):
        return (10 * rand(n, m)).astype(np.int32)

    # Now the real test cases

    @parametrize("dtype", "?bhilBfd")
    def test_ones_pathological(self, dtype):
        # for preservation of behavior described in
        # gh-12519; amin > amax behavior may still change
        # in the future
        arr = np.ones(10, dtype=dtype)
        expected = np.zeros(10, dtype=dtype)
        actual = np.clip(arr, 1, 0)
        assert_equal(actual, expected)

    @parametrize("dtype", "eFD")
    def test_ones_pathological_2(self, dtype):
        if dtype in "FD":
            # FIXME: make xfail
            raise SkipTest("torch.clamp not implemented for complex types")
        # for preservation of behavior described in
        # gh-12519; amin > amax behavior may still change
        # in the future
        arr = np.ones(10, dtype=dtype)
        expected = np.zeros(10, dtype=dtype)
        actual = np.clip(arr, 1, 0)
        assert_equal(actual, expected)

    def test_simple_double(self):
        # Test native double input with scalar min/max.
        a = self._generate_data(self.nr, self.nc)
        m = 0.1
        M = 0.6
        ac = self.fastclip(a, m, M)
        act = self.clip(a, m, M)
        assert_array_equal(ac, act)

    def test_simple_int(self):
        # Test native int input with scalar min/max.
        a = self._generate_int_data(self.nr, self.nc)
        a = a.astype(int)
        m = -2
        M = 4
        ac = self.fastclip(a, m, M)
        act = self.clip(a, m, M)
        assert_array_equal(ac, act)

    def test_array_double(self):
        # Test native double input with array min/max.
        a = self._generate_data(self.nr, self.nc)
        m = np.zeros(a.shape)
        M = m + 0.5
        ac = self.fastclip(a, m, M)
        act = self.clip(a, m, M)
        assert_array_equal(ac, act)

    @xpassIfTorchDynamo_np  # (reason="byteorder not supported in torch")
    def test_simple_nonnative(self):
        # Test non native double input with scalar min/max.
        # Test native double input with non native double scalar min/max.
        a = self._generate_non_native_data(self.nr, self.nc)
        m = -0.5
        M = 0.6
        ac = self.fastclip(a, m, M)
        act = self.clip(a, m, M)
        assert_array_equal(ac, act)

        # Test native double input with non native double scalar min/max.
        a = self._generate_data(self.nr, self.nc)
        m = -0.5
        M = self._neg_byteorder(0.6)
        assert_(not M.dtype.isnative)
        ac = self.fastclip(a, m, M)
        act = self.clip(a, m, M)
        assert_array_equal(ac, act)

    @xpassIfTorchDynamo_np  # (reason="clamp not supported for complex")
    def test_simple_complex(self):
        # Test native complex input with native double scalar min/max.
        # Test native input with complex double scalar min/max.
        a = 3 * self._generate_data_complex(self.nr, self.nc)
        m = -0.5
        M = 1.0
        ac = self.fastclip(a, m, M)
        act = self.clip(a, m, M)
        assert_array_equal(ac, act)

        # Test native input with complex double scalar min/max.
        a = 3 * self._generate_data(self.nr, self.nc)
        m = -0.5 + 1.0j
        M = 1.0 + 2.0j
        ac = self.fastclip(a, m, M)
        act = self.clip(a, m, M)
        assert_array_equal(ac, act)

    @xfail  # (reason="clamp not supported for complex")
    def test_clip_complex(self):
        # Address Issue gh-5354 for clipping complex arrays
        # Test native complex input without explicit min/max
        # ie, either min=None or max=None
        a = np.ones(10, dtype=complex)
        m = a.min()
        M = a.max()
        am = self.fastclip(a, m, None)
        aM = self.fastclip(a, None, M)
        assert_array_equal(am, a)
        assert_array_equal(aM, a)

    def test_clip_non_contig(self):
        # Test clip for non contiguous native input and native scalar min/max.
        a = self._generate_data(self.nr * 2, self.nc * 3)
        a = a[::2, ::3]
        assert_(not a.flags["F_CONTIGUOUS"])
        assert_(not a.flags["C_CONTIGUOUS"])
        ac = self.fastclip(a, -1.6, 1.7)
        act = self.clip(a, -1.6, 1.7)
        assert_array_equal(ac, act)

    def test_simple_out(self):
        # Test native double input with scalar min/max.
        a = self._generate_data(self.nr, self.nc)
        m = -0.5
        M = 0.6
        ac = np.zeros(a.shape)
        act = np.zeros(a.shape)
        self.fastclip(a, m, M, ac)
        self.clip(a, m, M, act)
        assert_array_equal(ac, act)

    #   @xpassIfTorchDynamo_np  # (reason="casting not supported")
    @parametrize(
        "casting",
        [
            subtest(None, decorators=[xfail]),
            subtest("unsafe", decorators=[xpassIfTorchDynamo_np]),
        ],
    )
    def test_simple_int32_inout(self, casting):
        # Test native int32 input with double min/max and int32 out.
        a = self._generate_int32_data(self.nr, self.nc)
        m = np.float64(0)
        M = np.float64(2)
        ac = np.zeros(a.shape, dtype=np.int32)
        act = ac.copy()
        if casting is not None:
            # explicitly passing "unsafe" will silence warning
            self.fastclip(a, m, M, ac, casting=casting)
        self.clip(a, m, M, act)
        assert_array_equal(ac, act)

    def test_simple_int64_out(self):
        # Test native int32 input with int32 scalar min/max and int64 out.
        a = self._generate_int32_data(self.nr, self.nc)
        m = np.int32(-1)
        M = np.int32(1)
        ac = np.zeros(a.shape, dtype=np.int64)
        act = ac.copy()
        self.fastclip(a, m, M, ac)
        self.clip(a, m, M, act)
        assert_array_equal(ac, act)

    @xfail  # (reason="FIXME arrays not equal")
    def test_simple_int64_inout(self):
        # Test native int32 input with double array min/max and int32 out.
        a = self._generate_int32_data(self.nr, self.nc)
        m = np.zeros(a.shape, np.float64)
        M = np.float64(1)
        ac = np.zeros(a.shape, dtype=np.int32)
        act = ac.copy()
        self.clip(a, m, M, act)
        assert_array_equal(ac, act)

    @xfail  # (reason="FIXME arrays not equal")
    def test_simple_int32_out(self):
        # Test native double input with scalar min/max and int out.
        a = self._generate_data(self.nr, self.nc)
        m = -1.0
        M = 2.0
        ac = np.zeros(a.shape, dtype=np.int32)
        act = ac.copy()
        self.clip(a, m, M, act)
        assert_array_equal(ac, act)

    def test_simple_inplace_01(self):
        # Test native double input with array min/max in-place.
        a = self._generate_data(self.nr, self.nc)
        ac = a.copy()
        m = np.zeros(a.shape)
        M = 1.0
        self.fastclip(a, m, M, a)
        self.clip(a, m, M, ac)
        assert_array_equal(a, ac)

    def test_simple_inplace_02(self):
        # Test native double input with scalar min/max in-place.
        a = self._generate_data(self.nr, self.nc)
        ac = a.copy()
        m = -0.5
        M = 0.6
        self.fastclip(a, m, M, a)
        self.clip(ac, m, M, ac)
        assert_array_equal(a, ac)

    def test_noncontig_inplace(self):
        # Test non contiguous double input with double scalar min/max in-place.
        a = self._generate_data(self.nr * 2, self.nc * 3)
        a = a[::2, ::3]
        assert_(not a.flags["F_CONTIGUOUS"])
        assert_(not a.flags["C_CONTIGUOUS"])
        ac = a.copy()
        m = -0.5
        M = 0.6
        self.fastclip(a, m, M, a)
        self.clip(ac, m, M, ac)
        assert_array_equal(a, ac)

    def test_type_cast_01(self):
        # Test native double input with scalar min/max.
        a = self._generate_data(self.nr, self.nc)
        m = -0.5
        M = 0.6
        ac = self.fastclip(a, m, M)
        act = self.clip(a, m, M)
        assert_array_equal(ac, act)

    def test_type_cast_02(self):
        # Test native int32 input with int32 scalar min/max.
        a = self._generate_int_data(self.nr, self.nc)
        a = a.astype(np.int32)
        m = -2
        M = 4
        ac = self.fastclip(a, m, M)
        act = self.clip(a, m, M)
        assert_array_equal(ac, act)

    def test_type_cast_03(self):
        # Test native int32 input with float64 scalar min/max.
        a = self._generate_int32_data(self.nr, self.nc)
        m = -2
        M = 4
        ac = self.fastclip(a, np.float64(m), np.float64(M))
        act = self.clip(a, np.float64(m), np.float64(M))
        assert_array_equal(ac, act)

    def test_type_cast_04(self):
        # Test native int32 input with float32 scalar min/max.
        a = self._generate_int32_data(self.nr, self.nc)
        m = np.float32(-2)
        M = np.float32(4)
        act = self.fastclip(a, m, M)
        ac = self.clip(a, m, M)
        assert_array_equal(ac, act)

    def test_type_cast_05(self):
        # Test native int32 with double arrays min/max.
        a = self._generate_int_data(self.nr, self.nc)
        m = -0.5
        M = 1.0
        ac = self.fastclip(a, m * np.zeros(a.shape), M)
        act = self.clip(a, m * np.zeros(a.shape), M)
        assert_array_equal(ac, act)

    @xpassIfTorchDynamo_np  # (reason="newbyteorder not supported")
    def test_type_cast_06(self):
        # Test native with NON native scalar min/max.
        a = self._generate_data(self.nr, self.nc)
        m = 0.5
        m_s = self._neg_byteorder(m)
        M = 1.0
        act = self.clip(a, m_s, M)
        ac = self.fastclip(a, m_s, M)
        assert_array_equal(ac, act)

    @xpassIfTorchDynamo_np  # (reason="newbyteorder not supported")
    def test_type_cast_07(self):
        # Test NON native with native array min/max.
        a = self._generate_data(self.nr, self.nc)
        m = -0.5 * np.ones(a.shape)
        M = 1.0
        a_s = self._neg_byteorder(a)
        assert_(not a_s.dtype.isnative)
        act = a_s.clip(m, M)
        ac = self.fastclip(a_s, m, M)
        assert_array_equal(ac, act)

    @xpassIfTorchDynamo_np  # (reason="newbyteorder not supported")
    def test_type_cast_08(self):
        # Test NON native with native scalar min/max.
        a = self._generate_data(self.nr, self.nc)
        m = -0.5
        M = 1.0
        a_s = self._neg_byteorder(a)
        assert_(not a_s.dtype.isnative)
        ac = self.fastclip(a_s, m, M)
        act = a_s.clip(m, M)
        assert_array_equal(ac, act)

    @xpassIfTorchDynamo_np  # (reason="newbyteorder not supported")
    def test_type_cast_09(self):
        # Test native with NON native array min/max.
        a = self._generate_data(self.nr, self.nc)
        m = -0.5 * np.ones(a.shape)
        M = 1.0
        m_s = self._neg_byteorder(m)
        assert_(not m_s.dtype.isnative)
        ac = self.fastclip(a, m_s, M)
        act = self.clip(a, m_s, M)
        assert_array_equal(ac, act)

    def test_type_cast_10(self):
        # Test native int32 with float min/max and float out for output argument.
        a = self._generate_int_data(self.nr, self.nc)
        b = np.zeros(a.shape, dtype=np.float32)
        m = np.float32(-0.5)
        M = np.float32(1)
        act = self.clip(a, m, M, out=b)
        ac = self.fastclip(a, m, M, out=b)
        assert_array_equal(ac, act)

    @xpassIfTorchDynamo_np  # (reason="newbyteorder not supported")
    def test_type_cast_11(self):
        # Test non native with native scalar, min/max, out non native
        a = self._generate_non_native_data(self.nr, self.nc)
        b = a.copy()
        b = b.astype(b.dtype.newbyteorder(">"))
        bt = b.copy()
        m = -0.5
        M = 1.0
        self.fastclip(a, m, M, out=b)
        self.clip(a, m, M, out=bt)
        assert_array_equal(b, bt)

    def test_type_cast_12(self):
        # Test native int32 input and min/max and float out
        a = self._generate_int_data(self.nr, self.nc)
        b = np.zeros(a.shape, dtype=np.float32)
        m = np.int32(0)
        M = np.int32(1)
        act = self.clip(a, m, M, out=b)
        ac = self.fastclip(a, m, M, out=b)
        assert_array_equal(ac, act)

    def test_clip_with_out_simple(self):
        # Test native double input with scalar min/max
        a = self._generate_data(self.nr, self.nc)
        m = -0.5
        M = 0.6
        ac = np.zeros(a.shape)
        act = np.zeros(a.shape)
        self.fastclip(a, m, M, ac)
        self.clip(a, m, M, act)
        assert_array_equal(ac, act)

    @xfail  # (reason="FIXME arrays not equal")
    def test_clip_with_out_simple2(self):
        # Test native int32 input with double min/max and int32 out
        a = self._generate_int32_data(self.nr, self.nc)
        m = np.float64(0)
        M = np.float64(2)
        ac = np.zeros(a.shape, dtype=np.int32)
        act = ac.copy()
        self.clip(a, m, M, act)
        assert_array_equal(ac, act)

    def test_clip_with_out_simple_int32(self):
        # Test native int32 input with int32 scalar min/max and int64 out
        a = self._generate_int32_data(self.nr, self.nc)
        m = np.int32(-1)
        M = np.int32(1)
        ac = np.zeros(a.shape, dtype=np.int64)
        act = ac.copy()
        self.fastclip(a, m, M, ac)
        self.clip(a, m, M, act)
        assert_array_equal(ac, act)

    @xfail  # (reason="FIXME arrays not equal")
    def test_clip_with_out_array_int32(self):
        # Test native int32 input with double array min/max and int32 out
        a = self._generate_int32_data(self.nr, self.nc)
        m = np.zeros(a.shape, np.float64)
        M = np.float64(1)
        ac = np.zeros(a.shape, dtype=np.int32)
        act = ac.copy()
        self.clip(a, m, M, act)
        assert_array_equal(ac, act)

    @xfail  # (reason="FIXME arrays not equal")
    def test_clip_with_out_array_outint32(self):
        # Test native double input with scalar min/max and int out
        a = self._generate_data(self.nr, self.nc)
        m = -1.0
        M = 2.0
        ac = np.zeros(a.shape, dtype=np.int32)
        act = ac.copy()
        self.clip(a, m, M, act)
        assert_array_equal(ac, act)

    def test_clip_with_out_transposed(self):
        # Test that the out argument works when transposed
        a = np.arange(16).reshape(4, 4)
        out = np.empty_like(a).T
        a.clip(4, 10, out=out)
        expected = self.clip(a, 4, 10)
        assert_array_equal(out, expected)

    def test_clip_with_out_memory_overlap(self):
        # Test that the out argument works when it has memory overlap
        a = np.arange(16).reshape(4, 4)
        ac = a.copy()
        a[:-1].clip(4, 10, out=a[1:])
        expected = self.clip(ac[:-1], 4, 10)
        assert_array_equal(a[1:], expected)

    def test_clip_inplace_array(self):
        # Test native double input with array min/max
        a = self._generate_data(self.nr, self.nc)
        ac = a.copy()
        m = np.zeros(a.shape)
        M = 1.0
        self.fastclip(a, m, M, a)
        self.clip(a, m, M, ac)
        assert_array_equal(a, ac)

    def test_clip_inplace_simple(self):
        # Test native double input with scalar min/max
        a = self._generate_data(self.nr, self.nc)
        ac = a.copy()
        m = -0.5
        M = 0.6
        self.fastclip(a, m, M, a)
        self.clip(a, m, M, ac)
        assert_array_equal(a, ac)

    def test_clip_func_takes_out(self):
        # Ensure that the clip() function takes an out=argument.
        a = self._generate_data(self.nr, self.nc)
        ac = a.copy()
        m = -0.5
        M = 0.6
        a2 = np.clip(a, m, M, out=a)
        self.clip(a, m, M, ac)
        assert_array_equal(a2, ac)
        assert_(a2 is a)

    @skip(reason="Edge case; Wait until deprecation graduates")
    def test_clip_nan(self):
        d = np.arange(7.0)
        with assert_warns(DeprecationWarning):
            assert_equal(d.clip(min=np.nan), d)
        with assert_warns(DeprecationWarning):
            assert_equal(d.clip(max=np.nan), d)
        with assert_warns(DeprecationWarning):
            assert_equal(d.clip(min=np.nan, max=np.nan), d)
        with assert_warns(DeprecationWarning):
            assert_equal(d.clip(min=-2, max=np.nan), d)
        with assert_warns(DeprecationWarning):
            assert_equal(d.clip(min=np.nan, max=10), d)

    @parametrize(
        "amin, amax",
        [
            # two scalars
            (1, 0),
            # mix scalar and array
            (1, np.zeros(10)),
            # two arrays
            (np.ones(10), np.zeros(10)),
        ],
    )
    def test_clip_value_min_max_flip(self, amin, amax):
        a = np.arange(10, dtype=np.int64)
        # requirement from ufunc_docstrings.py
        expected = np.minimum(np.maximum(a, amin), amax)
        actual = np.clip(a, amin, amax)
        assert_equal(actual, expected)

    @parametrize(
        "arr, amin, amax",
        [
            # problematic scalar nan case from hypothesis
            (
                np.zeros(10, dtype=np.int64),
                np.array(np.nan),
                np.zeros(10, dtype=np.int32),
            ),
        ],
    )
    def test_clip_scalar_nan_propagation(self, arr, amin, amax):
        # enforcement of scalar nan propagation for comparisons
        # called through clip()
        expected = np.minimum(np.maximum(arr, amin), amax)
        actual = np.clip(arr, amin, amax)
        assert_equal(actual, expected)

    @skip  # hypothesis hynp.from_dtype fails on CI (versions?)
    @given(
        data=st.data(),
        arr=hynp.arrays(
            dtype=hynp.integer_dtypes() | hynp.floating_dtypes(),
            shape=hynp.array_shapes(),
        ),
    )
    def test_clip_property(self, data, arr):
        """A property-based test using Hypothesis.

        This aims for maximum generality: it could in principle generate *any*
        valid inputs to np.clip, and in practice generates much more varied
        inputs than human testers come up with.

        Because many of the inputs have tricky dependencies - compatible dtypes
        and mutually-broadcastable shapes - we use `st.data()` strategy draw
        values *inside* the test function, from strategies we construct based
        on previous values.  An alternative would be to define a custom strategy
        with `@st.composite`, but until we have duplicated code inline is fine.

        That accounts for most of the function; the actual test is just three
        lines to calculate and compare actual vs expected results!
        """
        numeric_dtypes = hynp.integer_dtypes() | hynp.floating_dtypes()
        # Generate shapes for the bounds which can be broadcast with each other
        # and with the base shape.  Below, we might decide to use scalar bounds,
        # but it's clearer to generate these shapes unconditionally in advance.
        in_shapes, result_shape = data.draw(
            hynp.mutually_broadcastable_shapes(num_shapes=2, base_shape=arr.shape)
        )
        # Scalar `nan` is deprecated due to the differing behaviour it shows.
        s = numeric_dtypes.flatmap(lambda x: hynp.from_dtype(x, allow_nan=False))
        amin = data.draw(
            s
            | hynp.arrays(
                dtype=numeric_dtypes, shape=in_shapes[0], elements={"allow_nan": False}
            )
        )
        amax = data.draw(
            s
            | hynp.arrays(
                dtype=numeric_dtypes, shape=in_shapes[1], elements={"allow_nan": False}
            )
        )

        # Then calculate our result and expected result and check that they're
        # equal!  See gh-12519 and gh-19457 for discussion deciding on this
        # property and the result_type argument.
        result = np.clip(arr, amin, amax)
        t = np.result_type(arr, amin, amax)
        expected = np.minimum(amax, np.maximum(arr, amin, dtype=t), dtype=t)
        assert result.dtype == t
        assert_array_equal(result, expected)


class TestAllclose(TestCase):
    rtol = 1e-5
    atol = 1e-8

    def tst_allclose(self, x, y):
        assert_(np.allclose(x, y), f"{x} and {y} not close")

    def tst_not_allclose(self, x, y):
        assert_(not np.allclose(x, y), f"{x} and {y} shouldn't be close")

    def test_ip_allclose(self):
        # Parametric test factory.
        arr = np.array([100, 1000])
        aran = np.arange(125).reshape((5, 5, 5))

        atol = self.atol
        rtol = self.rtol

        data = [
            ([1, 0], [1, 0]),
            ([atol], [0]),
            ([1], [1 + rtol + atol]),
            (arr, arr + arr * rtol),
            (arr, arr + arr * rtol + atol * 2),
            (aran, aran + aran * rtol),
            (np.inf, np.inf),
            (np.inf, [np.inf]),
        ]

        for x, y in data:
            self.tst_allclose(x, y)

    def test_ip_not_allclose(self):
        # Parametric test factory.
        aran = np.arange(125).reshape((5, 5, 5))

        atol = self.atol
        rtol = self.rtol

        data = [
            ([np.inf, 0], [1, np.inf]),
            ([np.inf, 0], [1, 0]),
            ([np.inf, np.inf], [1, np.inf]),
            ([np.inf, np.inf], [1, 0]),
            ([-np.inf, 0], [np.inf, 0]),
            ([np.nan, 0], [np.nan, 0]),
            ([atol * 2], [0]),
            ([1], [1 + rtol + atol * 2]),
            (aran, aran + aran * atol + atol * 2),
            (np.array([np.inf, 1]), np.array([0, np.inf])),
        ]

        for x, y in data:
            self.tst_not_allclose(x, y)

    def test_no_parameter_modification(self):
        x = np.array([np.inf, 1])
        y = np.array([0, np.inf])
        np.allclose(x, y)
        assert_array_equal(x, np.array([np.inf, 1]))
        assert_array_equal(y, np.array([0, np.inf]))

    def test_min_int(self):
        # Could make problems because of abs(min_int) == min_int
        min_int = np.iinfo(np.int_).min
        a = np.array([min_int], dtype=np.int_)
        assert_(np.allclose(a, a))

    def test_equalnan(self):
        x = np.array([1.0, np.nan])
        assert_(np.allclose(x, x, equal_nan=True))


class TestIsclose(TestCase):
    rtol = 1e-5
    atol = 1e-8

    def _setup(self):
        atol = self.atol
        rtol = self.rtol
        arr = np.array([100, 1000])
        aran = np.arange(125).reshape((5, 5, 5))

        self.all_close_tests = [
            ([1, 0], [1, 0]),
            ([atol], [0]),
            ([1], [1 + rtol + atol]),
            (arr, arr + arr * rtol),
            (arr, arr + arr * rtol + atol),
            (aran, aran + aran * rtol),
            (np.inf, np.inf),
            (np.inf, [np.inf]),
            ([np.inf, -np.inf], [np.inf, -np.inf]),
        ]
        self.none_close_tests = [
            ([np.inf, 0], [1, np.inf]),
            ([np.inf, -np.inf], [1, 0]),
            ([np.inf, np.inf], [1, -np.inf]),
            ([np.inf, np.inf], [1, 0]),
            ([np.nan, 0], [np.nan, -np.inf]),
            ([atol * 2], [0]),
            ([1], [1 + rtol + atol * 2]),
            (aran, aran + rtol * 1.1 * aran + atol * 1.1),
            (np.array([np.inf, 1]), np.array([0, np.inf])),
        ]
        self.some_close_tests = [
            ([np.inf, 0], [np.inf, atol * 2]),
            ([atol, 1, 1e6 * (1 + 2 * rtol) + atol], [0, np.nan, 1e6]),
            (np.arange(3), [0, 1, 2.1]),
            (np.nan, [np.nan, np.nan, np.nan]),
            ([0], [atol, np.inf, -np.inf, np.nan]),
            (0, [atol, np.inf, -np.inf, np.nan]),
        ]
        self.some_close_results = [
            [True, False],
            [True, False, False],
            [True, True, False],
            [False, False, False],
            [True, False, False, False],
            [True, False, False, False],
        ]

    def test_ip_isclose(self):
        self._setup()
        tests = self.some_close_tests
        results = self.some_close_results
        for (x, y), result in zip(tests, results):
            assert_array_equal(np.isclose(x, y), result)

    def tst_all_isclose(self, x, y):
        assert_(np.all(np.isclose(x, y)), f"{x} and {y} not close")

    def tst_none_isclose(self, x, y):
        msg = "%s and %s shouldn't be close"
        assert_(not np.any(np.isclose(x, y)), msg % (x, y))

    def tst_isclose_allclose(self, x, y):
        msg = "isclose.all() and allclose aren't same for %s and %s"
        msg2 = "isclose and allclose aren't same for %s and %s"
        if np.isscalar(x) and np.isscalar(y):
            assert_(np.isclose(x, y) == np.allclose(x, y), msg=msg2 % (x, y))
        else:
            assert_array_equal(np.isclose(x, y).all(), np.allclose(x, y), msg % (x, y))

    def test_ip_all_isclose(self):
        self._setup()
        for x, y in self.all_close_tests:
            self.tst_all_isclose(x, y)

    def test_ip_none_isclose(self):
        self._setup()
        for x, y in self.none_close_tests:
            self.tst_none_isclose(x, y)

    def test_ip_isclose_allclose(self):
        self._setup()
        tests = self.all_close_tests + self.none_close_tests + self.some_close_tests
        for x, y in tests:
            self.tst_isclose_allclose(x, y)

    def test_equal_nan(self):
        assert_array_equal(np.isclose(np.nan, np.nan, equal_nan=True), [True])
        arr = np.array([1.0, np.nan])
        assert_array_equal(np.isclose(arr, arr, equal_nan=True), [True, True])

    @xfailIfTorchDynamo  # scalars vs 0D
    def test_scalar_return(self):
        assert_(np.isscalar(np.isclose(1, 1)))

    def test_no_parameter_modification(self):
        x = np.array([np.inf, 1])
        y = np.array([0, np.inf])
        np.isclose(x, y)
        assert_array_equal(x, np.array([np.inf, 1]))
        assert_array_equal(y, np.array([0, np.inf]))

    def test_non_finite_scalar(self):
        # GH7014, when two scalars are compared the output should also be a
        # scalar
        # XXX: test modified since there are array scalars
        assert_(np.isclose(np.inf, -np.inf).item() is False)
        assert_(np.isclose(0, np.inf).item() is False)


class TestStdVar(TestCase):
    def setUp(self):
        super().setUp()
        self.A = np.array([1, -1, 1, -1])
        self.real_var = 1

    def test_basic(self):
        assert_almost_equal(np.var(self.A), self.real_var)
        assert_almost_equal(np.std(self.A) ** 2, self.real_var)

    def test_scalars(self):
        assert_equal(np.var(1), 0)
        assert_equal(np.std(1), 0)

    def test_ddof1(self):
        assert_almost_equal(
            np.var(self.A, ddof=1), self.real_var * len(self.A) / (len(self.A) - 1)
        )
        assert_almost_equal(
            np.std(self.A, ddof=1) ** 2, self.real_var * len(self.A) / (len(self.A) - 1)
        )

    def test_ddof2(self):
        assert_almost_equal(
            np.var(self.A, ddof=2), self.real_var * len(self.A) / (len(self.A) - 2)
        )
        assert_almost_equal(
            np.std(self.A, ddof=2) ** 2, self.real_var * len(self.A) / (len(self.A) - 2)
        )

    def test_out_scalar(self):
        d = np.arange(10)
        out = np.array(0.0)
        r = np.std(d, out=out)
        assert_(r is out)
        assert_array_equal(r, out)
        r = np.var(d, out=out)
        assert_(r is out)
        assert_array_equal(r, out)
        r = np.mean(d, out=out)
        assert_(r is out)
        assert_array_equal(r, out)


class TestStdVarComplex(TestCase):
    def test_basic(self):
        A = np.array([1, 1.0j, -1, -1.0j])
        real_var = 1
        assert_almost_equal(np.var(A), real_var)
        assert_almost_equal(np.std(A) ** 2, real_var)

    def test_scalars(self):
        assert_equal(np.var(1j), 0)
        assert_equal(np.std(1j), 0)


class TestCreationFuncs(TestCase):
    # Test ones, zeros, empty and full.

    def setUp(self):
        super().setUp()
        dtypes = {np.dtype(tp) for tp in "efdFDBbhil?"}
        self.dtypes = dtypes
        self.orders = {
            "C": "c_contiguous"
        }  # XXX: reeenable when implemented, 'F': 'f_contiguous'}
        self.ndims = 10

    def check_function(self, func, fill_value=None):
        par = ((0, 1, 2), range(self.ndims), self.orders, self.dtypes)
        fill_kwarg = {}
        if fill_value is not None:
            fill_kwarg = {"fill_value": fill_value}

        for size, ndims, order, dtype in itertools.product(*par):
            shape = ndims * [size]

            arr = func(shape, order=order, dtype=dtype, **fill_kwarg)

            assert_equal(arr.dtype, dtype)
            assert_(getattr(arr.flags, self.orders[order]))

            if fill_value is not None:
                val = fill_value
                assert_equal(arr, dtype.type(val))

    def test_zeros(self):
        self.check_function(np.zeros)

    def test_ones(self):
        self.check_function(np.ones)

    def test_empty(self):
        self.check_function(np.empty)

    def test_full(self):
        self.check_function(np.full, 0)
        self.check_function(np.full, 1)

    @skipif(TEST_WITH_TORCHDYNAMO, reason="fails with dynamo")
    @skipif(not HAS_REFCOUNT, reason="Python lacks refcounts")
    def test_for_reference_leak(self):
        # Make sure we have an object for reference
        dim = 1
        beg = sys.getrefcount(dim)
        np.zeros([dim] * 10)
        assert_(sys.getrefcount(dim) == beg)
        np.ones([dim] * 10)
        assert_(sys.getrefcount(dim) == beg)
        np.empty([dim] * 10)
        assert_(sys.getrefcount(dim) == beg)
        np.full([dim] * 10, 0)
        assert_(sys.getrefcount(dim) == beg)


@skip(reason="implement order etc")  # FIXME: make xfail
@instantiate_parametrized_tests
class TestLikeFuncs(TestCase):
    """Test ones_like, zeros_like, empty_like and full_like"""

    def setUp(self):
        super().setUp()
        self.data = [
            # Array scalars
            (np.array(3.0), None),
            (np.array(3), "f8"),
            # 1D arrays
            (np.arange(6, dtype="f4"), None),
            (np.arange(6), "c16"),
            # 2D C-layout arrays
            (np.arange(6).reshape(2, 3), None),
            (np.arange(6).reshape(3, 2), "i1"),
            # 2D F-layout arrays
            (np.arange(6).reshape((2, 3), order="F"), None),
            (np.arange(6).reshape((3, 2), order="F"), "i1"),
            # 3D C-layout arrays
            (np.arange(24).reshape(2, 3, 4), None),
            (np.arange(24).reshape(4, 3, 2), "f4"),
            # 3D F-layout arrays
            (np.arange(24).reshape((2, 3, 4), order="F"), None),
            (np.arange(24).reshape((4, 3, 2), order="F"), "f4"),
            # 3D non-C/F-layout arrays
            (np.arange(24).reshape(2, 3, 4).swapaxes(0, 1), None),
            (np.arange(24).reshape(4, 3, 2).swapaxes(0, 1), "?"),
        ]
        self.shapes = [
            (),
            (5,),
            (
                5,
                6,
            ),
            (
                5,
                6,
                7,
            ),
        ]

    def compare_array_value(self, dz, value, fill_value):
        if value is not None:
            if fill_value:
                # Conversion is close to what np.full_like uses
                # but we  may want to convert directly in the future
                # which may result in errors (where this does not).
                z = np.array(value).astype(dz.dtype)
                assert_(np.all(dz == z))
            else:
                assert_(np.all(dz == value))

    def check_like_function(self, like_function, value, fill_value=False):
        if fill_value:
            fill_kwarg = {"fill_value": value}
        else:
            fill_kwarg = {}
        for d, dtype in self.data:
            # default (K) order, dtype
            dz = like_function(d, dtype=dtype, **fill_kwarg)
            assert_equal(dz.shape, d.shape)
            assert_equal(
                np.array(dz.strides) * d.dtype.itemsize,
                np.array(d.strides) * dz.dtype.itemsize,
            )
            assert_equal(d.flags.c_contiguous, dz.flags.c_contiguous)
            assert_equal(d.flags.f_contiguous, dz.flags.f_contiguous)
            if dtype is None:
                assert_equal(dz.dtype, d.dtype)
            else:
                assert_equal(dz.dtype, np.dtype(dtype))
            self.compare_array_value(dz, value, fill_value)

            # C order, default dtype
            dz = like_function(d, order="C", dtype=dtype, **fill_kwarg)
            assert_equal(dz.shape, d.shape)
            assert_(dz.flags.c_contiguous)
            if dtype is None:
                assert_equal(dz.dtype, d.dtype)
            else:
                assert_equal(dz.dtype, np.dtype(dtype))
            self.compare_array_value(dz, value, fill_value)

            # F order, default dtype
            dz = like_function(d, order="F", dtype=dtype, **fill_kwarg)
            assert_equal(dz.shape, d.shape)
            assert_(dz.flags.f_contiguous)
            if dtype is None:
                assert_equal(dz.dtype, d.dtype)
            else:
                assert_equal(dz.dtype, np.dtype(dtype))
            self.compare_array_value(dz, value, fill_value)

            # A order
            dz = like_function(d, order="A", dtype=dtype, **fill_kwarg)
            assert_equal(dz.shape, d.shape)
            if d.flags.f_contiguous:
                assert_(dz.flags.f_contiguous)
            else:
                assert_(dz.flags.c_contiguous)
            if dtype is None:
                assert_equal(dz.dtype, d.dtype)
            else:
                assert_equal(dz.dtype, np.dtype(dtype))
            self.compare_array_value(dz, value, fill_value)

            # Test the 'shape' parameter
            for s in self.shapes:
                for o in "CFA":
                    sz = like_function(d, dtype=dtype, shape=s, order=o, **fill_kwarg)
                    assert_equal(sz.shape, s)
                    if dtype is None:
                        assert_equal(sz.dtype, d.dtype)
                    else:
                        assert_equal(sz.dtype, np.dtype(dtype))
                    if o == "C" or (o == "A" and d.flags.c_contiguous):
                        assert_(sz.flags.c_contiguous)
                    elif o == "F" or (o == "A" and d.flags.f_contiguous):
                        assert_(sz.flags.f_contiguous)
                    self.compare_array_value(sz, value, fill_value)

                if d.ndim != len(s):
                    assert_equal(
                        np.argsort(
                            like_function(
                                d, dtype=dtype, shape=s, order="K", **fill_kwarg
                            ).strides
                        ),
                        np.argsort(np.empty(s, dtype=dtype, order="C").strides),
                    )
                else:
                    assert_equal(
                        np.argsort(
                            like_function(
                                d, dtype=dtype, shape=s, order="K", **fill_kwarg
                            ).strides
                        ),
                        np.argsort(d.strides),
                    )

    def test_ones_like(self):
        self.check_like_function(np.ones_like, 1)

    def test_zeros_like(self):
        self.check_like_function(np.zeros_like, 0)

    def test_empty_like(self):
        self.check_like_function(np.empty_like, None)

    def test_filled_like(self):
        self.check_like_function(np.full_like, 0, True)
        self.check_like_function(np.full_like, 1, True)
        self.check_like_function(np.full_like, 1000, True)
        self.check_like_function(np.full_like, 123.456, True)
        # Inf to integer casts cause invalid-value errors: ignore them.
        self.check_like_function(np.full_like, np.inf, True)

    @parametrize("likefunc", [np.empty_like, np.full_like, np.zeros_like, np.ones_like])
    @parametrize("dtype", [str, bytes])
    def test_dtype_str_bytes(self, likefunc, dtype):
        # Regression test for gh-19860
        a = np.arange(16).reshape(2, 8)
        b = a[:, ::2]  # Ensure b is not contiguous.
        kwargs = {"fill_value": ""} if likefunc == np.full_like else {}
        result = likefunc(b, dtype=dtype, **kwargs)
        if dtype == str:
            assert result.strides == (16, 4)
        else:
            # dtype is bytes
            assert result.strides == (4, 1)


class TestCorrelate(TestCase):
    def _setup(self, dt):
        self.x = np.array([1, 2, 3, 4, 5], dtype=dt)
        self.xs = np.arange(1, 20)[::3]
        self.y = np.array([-1, -2, -3], dtype=dt)
        self.z1 = np.array([-3.0, -8.0, -14.0, -20.0, -26.0, -14.0, -5.0], dtype=dt)
        self.z1_4 = np.array([-2.0, -5.0, -8.0, -11.0, -14.0, -5.0], dtype=dt)
        self.z1r = np.array([-15.0, -22.0, -22.0, -16.0, -10.0, -4.0, -1.0], dtype=dt)
        self.z2 = np.array([-5.0, -14.0, -26.0, -20.0, -14.0, -8.0, -3.0], dtype=dt)
        self.z2r = np.array([-1.0, -4.0, -10.0, -16.0, -22.0, -22.0, -15.0], dtype=dt)
        self.zs = np.array(
            [-3.0, -14.0, -30.0, -48.0, -66.0, -84.0, -102.0, -54.0, -19.0], dtype=dt
        )

    def test_float(self):
        self._setup(float)
        z = np.correlate(self.x, self.y, "full")
        assert_array_almost_equal(z, self.z1)
        z = np.correlate(self.x, self.y[:-1], "full")
        assert_array_almost_equal(z, self.z1_4)
        z = np.correlate(self.y, self.x, "full")
        assert_array_almost_equal(z, self.z2)
        z = np.correlate(np.flip(self.x), self.y, "full")
        assert_array_almost_equal(z, self.z1r)
        z = np.correlate(self.y, np.flip(self.x), "full")
        assert_array_almost_equal(z, self.z2r)
        z = np.correlate(self.xs, self.y, "full")
        assert_array_almost_equal(z, self.zs)

    def test_no_overwrite(self):
        d = np.ones(100)
        k = np.ones(3)
        np.correlate(d, k)
        assert_array_equal(d, np.ones(100))
        assert_array_equal(k, np.ones(3))

    def test_complex(self):
        x = np.array([1, 2, 3, 4 + 1j], dtype=complex)
        y = np.array([-1, -2j, 3 + 1j], dtype=complex)
        r_z = np.array([3 - 1j, 6, 8 + 1j, 11 + 5j, -5 + 8j, -4 - 1j], dtype=complex)
        r_z = np.flip(r_z).conjugate()
        z = np.correlate(y, x, mode="full")
        assert_array_almost_equal(z, r_z)

    def test_zero_size(self):
        with pytest.raises((ValueError, RuntimeError)):
            np.correlate(np.array([]), np.ones(1000), mode="full")
        with pytest.raises((ValueError, RuntimeError)):
            np.correlate(np.ones(1000), np.array([]), mode="full")

    @skip(reason="do not implement deprecated behavior")
    def test_mode(self):
        d = np.ones(100)
        k = np.ones(3)
        default_mode = np.correlate(d, k, mode="valid")
        with assert_warns(DeprecationWarning):
            valid_mode = np.correlate(d, k, mode="v")
        assert_array_equal(valid_mode, default_mode)
        # integer mode
        with assert_raises(ValueError):
            np.correlate(d, k, mode=-1)
        assert_array_equal(np.correlate(d, k, mode=0), valid_mode)
        # illegal arguments
        with assert_raises(TypeError):
            np.correlate(d, k, mode=None)


class TestConvolve(TestCase):
    def test_object(self):
        d = [1.0] * 100
        k = [1.0] * 3
        assert_array_almost_equal(np.convolve(d, k)[2:-2], np.full(98, 3))

    def test_no_overwrite(self):
        d = np.ones(100)
        k = np.ones(3)
        np.convolve(d, k)
        assert_array_equal(d, np.ones(100))
        assert_array_equal(k, np.ones(3))

    @skip(reason="do not implement deprecated behavior")
    def test_mode(self):
        d = np.ones(100)
        k = np.ones(3)
        default_mode = np.convolve(d, k, mode="full")
        with assert_warns(DeprecationWarning):
            full_mode = np.convolve(d, k, mode="f")
        assert_array_equal(full_mode, default_mode)
        # integer mode
        with assert_raises(ValueError):
            np.convolve(d, k, mode=-1)
        assert_array_equal(np.convolve(d, k, mode=2), full_mode)
        # illegal arguments
        with assert_raises(TypeError):
            np.convolve(d, k, mode=None)

    def test_numpy_doc_examples(self):
        conv = np.convolve([1, 2, 3], [0, 1, 0.5])
        assert_allclose(conv, [0.0, 1.0, 2.5, 4.0, 1.5], atol=1e-15)

        conv = np.convolve([1, 2, 3], [0, 1, 0.5], "same")
        assert_allclose(conv, [1.0, 2.5, 4.0], atol=1e-15)

        conv = np.convolve([1, 2, 3], [0, 1, 0.5], "valid")
        assert_allclose(conv, [2.5], atol=1e-15)


class TestDtypePositional(TestCase):
    def test_dtype_positional(self):
        np.empty((2,), bool)


@instantiate_parametrized_tests
class TestArgwhere(TestCase):
    @parametrize("nd", [0, 1, 2])
    def test_nd(self, nd):
        # get an nd array with multiple elements in every dimension
        x = np.empty((2,) * nd, dtype=bool)

        # none
        x[...] = False
        assert_equal(np.argwhere(x).shape, (0, nd))

        # only one
        x[...] = False
        x.ravel()[0] = True
        assert_equal(np.argwhere(x).shape, (1, nd))

        # all but one
        x[...] = True
        x.ravel()[0] = False
        assert_equal(np.argwhere(x).shape, (x.size - 1, nd))

        # all
        x[...] = True
        assert_equal(np.argwhere(x).shape, (x.size, nd))

    def test_2D(self):
        x = np.arange(6).reshape((2, 3))
        assert_array_equal(np.argwhere(x > 1), [[0, 2], [1, 0], [1, 1], [1, 2]])

    def test_list(self):
        assert_equal(np.argwhere([4, 0, 2, 1, 3]), [[0], [2], [3], [4]])


@xpassIfTorchDynamo_np  # (reason="TODO")
class TestStringFunction(TestCase):
    def test_set_string_function(self):
        a = np.array([1])
        np.set_string_function(lambda x: "FOO", repr=True)
        assert_equal(repr(a), "FOO")
        np.set_string_function(None, repr=True)
        assert_equal(repr(a), "array([1])")

        np.set_string_function(lambda x: "FOO", repr=False)
        assert_equal(str(a), "FOO")
        np.set_string_function(None, repr=False)
        assert_equal(str(a), "[1]")


class TestRoll(TestCase):
    def test_roll1d(self):
        x = np.arange(10)
        xr = np.roll(x, 2)
        assert_equal(xr, np.array([8, 9, 0, 1, 2, 3, 4, 5, 6, 7]))

    def test_roll2d(self):
        x2 = np.reshape(np.arange(10), (2, 5))
        x2r = np.roll(x2, 1)
        assert_equal(x2r, np.array([[9, 0, 1, 2, 3], [4, 5, 6, 7, 8]]))

        x2r = np.roll(x2, 1, axis=0)
        assert_equal(x2r, np.array([[5, 6, 7, 8, 9], [0, 1, 2, 3, 4]]))

        x2r = np.roll(x2, 1, axis=1)
        assert_equal(x2r, np.array([[4, 0, 1, 2, 3], [9, 5, 6, 7, 8]]))

        # Roll multiple axes at once.
        x2r = np.roll(x2, 1, axis=(0, 1))
        assert_equal(x2r, np.array([[9, 5, 6, 7, 8], [4, 0, 1, 2, 3]]))

        x2r = np.roll(x2, (1, 0), axis=(0, 1))
        assert_equal(x2r, np.array([[5, 6, 7, 8, 9], [0, 1, 2, 3, 4]]))

        x2r = np.roll(x2, (-1, 0), axis=(0, 1))
        assert_equal(x2r, np.array([[5, 6, 7, 8, 9], [0, 1, 2, 3, 4]]))

        x2r = np.roll(x2, (0, 1), axis=(0, 1))
        assert_equal(x2r, np.array([[4, 0, 1, 2, 3], [9, 5, 6, 7, 8]]))

        x2r = np.roll(x2, (0, -1), axis=(0, 1))
        assert_equal(x2r, np.array([[1, 2, 3, 4, 0], [6, 7, 8, 9, 5]]))

        x2r = np.roll(x2, (1, 1), axis=(0, 1))
        assert_equal(x2r, np.array([[9, 5, 6, 7, 8], [4, 0, 1, 2, 3]]))

        x2r = np.roll(x2, (-1, -1), axis=(0, 1))
        assert_equal(x2r, np.array([[6, 7, 8, 9, 5], [1, 2, 3, 4, 0]]))

        # Roll the same axis multiple times.
        x2r = np.roll(x2, 1, axis=(0, 0))
        assert_equal(x2r, np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]))

        x2r = np.roll(x2, 1, axis=(1, 1))
        assert_equal(x2r, np.array([[3, 4, 0, 1, 2], [8, 9, 5, 6, 7]]))

        # Roll more than one turn in either direction.
        x2r = np.roll(x2, 6, axis=1)
        assert_equal(x2r, np.array([[4, 0, 1, 2, 3], [9, 5, 6, 7, 8]]))

        x2r = np.roll(x2, -4, axis=1)
        assert_equal(x2r, np.array([[4, 0, 1, 2, 3], [9, 5, 6, 7, 8]]))

    def test_roll_empty(self):
        x = np.array([])
        assert_equal(np.roll(x, 1), np.array([]))


class TestRollaxis(TestCase):
    # expected shape indexed by (axis, start) for array of
    # shape (1, 2, 3, 4)
    tgtshape = {
        (0, 0): (1, 2, 3, 4),
        (0, 1): (1, 2, 3, 4),
        (0, 2): (2, 1, 3, 4),
        (0, 3): (2, 3, 1, 4),
        (0, 4): (2, 3, 4, 1),
        (1, 0): (2, 1, 3, 4),
        (1, 1): (1, 2, 3, 4),
        (1, 2): (1, 2, 3, 4),
        (1, 3): (1, 3, 2, 4),
        (1, 4): (1, 3, 4, 2),
        (2, 0): (3, 1, 2, 4),
        (2, 1): (1, 3, 2, 4),
        (2, 2): (1, 2, 3, 4),
        (2, 3): (1, 2, 3, 4),
        (2, 4): (1, 2, 4, 3),
        (3, 0): (4, 1, 2, 3),
        (3, 1): (1, 4, 2, 3),
        (3, 2): (1, 2, 4, 3),
        (3, 3): (1, 2, 3, 4),
        (3, 4): (1, 2, 3, 4),
    }

    def test_exceptions(self):
        a = np.arange(1 * 2 * 3 * 4).reshape(1, 2, 3, 4)
        assert_raises(np.AxisError, np.rollaxis, a, -5, 0)
        assert_raises(np.AxisError, np.rollaxis, a, 0, -5)
        assert_raises(np.AxisError, np.rollaxis, a, 4, 0)
        assert_raises(np.AxisError, np.rollaxis, a, 0, 5)

    @xfail  # XXX: ndarray.attributes
    def test_results(self):
        a = np.arange(1 * 2 * 3 * 4).reshape(1, 2, 3, 4).copy()
        aind = np.indices(a.shape)
        assert_(a.flags["OWNDATA"])
        for i, j in self.tgtshape:
            # positive axis, positive start
            res = np.rollaxis(a, axis=i, start=j)
            i0, i1, i2, i3 = aind[np.array(res.shape) - 1]
            assert_(np.all(res[i0, i1, i2, i3] == a))
            assert_(res.shape == self.tgtshape[(i, j)], str((i, j)))
            assert_(not res.flags["OWNDATA"])

            # negative axis, positive start
            ip = i + 1
            res = np.rollaxis(a, axis=-ip, start=j)
            i0, i1, i2, i3 = aind[np.array(res.shape) - 1]
            assert_(np.all(res[i0, i1, i2, i3] == a))
            assert_(res.shape == self.tgtshape[(4 - ip, j)])
            assert_(not res.flags["OWNDATA"])

            # positive axis, negative start
            jp = j + 1 if j < 4 else j
            res = np.rollaxis(a, axis=i, start=-jp)
            i0, i1, i2, i3 = aind[np.array(res.shape) - 1]
            assert_(np.all(res[i0, i1, i2, i3] == a))
            assert_(res.shape == self.tgtshape[(i, 4 - jp)])
            assert_(not res.flags["OWNDATA"])

            # negative axis, negative start
            ip = i + 1
            jp = j + 1 if j < 4 else j
            res = np.rollaxis(a, axis=-ip, start=-jp)
            i0, i1, i2, i3 = aind[np.array(res.shape) - 1]
            assert_(np.all(res[i0, i1, i2, i3] == a))
            assert_(res.shape == self.tgtshape[(4 - ip, 4 - jp)])
            assert_(not res.flags["OWNDATA"])


class TestMoveaxis(TestCase):
    def test_move_to_end(self):
        x = np.random.randn(5, 6, 7)
        for source, expected in [
            (0, (6, 7, 5)),
            (1, (5, 7, 6)),
            (2, (5, 6, 7)),
            (-1, (5, 6, 7)),
        ]:
            actual = np.moveaxis(x, source, -1).shape
            assert_(actual, expected)

    def test_move_new_position(self):
        x = np.random.randn(1, 2, 3, 4)
        for source, destination, expected in [
            (0, 1, (2, 1, 3, 4)),
            (1, 2, (1, 3, 2, 4)),
            (1, -1, (1, 3, 4, 2)),
        ]:
            actual = np.moveaxis(x, source, destination).shape
            assert_(actual, expected)

    def test_preserve_order(self):
        x = np.zeros((1, 2, 3, 4))
        for source, destination in [
            (0, 0),
            (3, -1),
            (-1, 3),
            ([0, -1], [0, -1]),
            ([2, 0], [2, 0]),
            (range(4), range(4)),
        ]:
            actual = np.moveaxis(x, source, destination).shape
            assert_(actual, (1, 2, 3, 4))

    def test_move_multiples(self):
        x = np.zeros((0, 1, 2, 3))
        for source, destination, expected in [
            ([0, 1], [2, 3], (2, 3, 0, 1)),
            ([2, 3], [0, 1], (2, 3, 0, 1)),
            ([0, 1, 2], [2, 3, 0], (2, 3, 0, 1)),
            ([3, 0], [1, 0], (0, 3, 1, 2)),
            ([0, 3], [0, 1], (0, 3, 1, 2)),
        ]:
            actual = np.moveaxis(x, source, destination).shape
            assert_(actual, expected)

    def test_errors(self):
        x = np.random.randn(1, 2, 3)
        assert_raises(np.AxisError, np.moveaxis, x, 3, 0)  # 'source.*out of bounds',
        assert_raises(np.AxisError, np.moveaxis, x, -4, 0)  # 'source.*out of bounds',
        assert_raises(
            np.AxisError,
            np.moveaxis,
            x,
            0,
            5,  # 'destination.*out of bounds',
        )
        assert_raises(
            ValueError,
            np.moveaxis,
            x,
            [0, 0],
            [0, 1],  # 'repeated axis in `source`',
        )
        assert_raises(
            ValueError,  # 'repeated axis in `destination`',
            np.moveaxis,
            x,
            [0, 1],
            [1, 1],
        )
        assert_raises(
            (ValueError, RuntimeError),  # 'must have the same number',
            np.moveaxis,
            x,
            0,
            [0, 1],
        )
        assert_raises(
            (ValueError, RuntimeError),  # 'must have the same number',
            np.moveaxis,
            x,
            [0, 1],
            [0],
        )

        x = [1, 2, 3]
        result = np.moveaxis(x, 0, 0)
        assert_(x, list(result))
        assert_(isinstance(result, np.ndarray))


class TestCross(TestCase):
    def test_2x2(self):
        u = [1, 2]
        v = [3, 4]
        z = -2
        cp = np.cross(u, v)
        assert_equal(cp, z)
        cp = np.cross(v, u)
        assert_equal(cp, -z)

    def test_2x3(self):
        u = [1, 2]
        v = [3, 4, 5]
        z = np.array([10, -5, -2])
        cp = np.cross(u, v)
        assert_equal(cp, z)
        cp = np.cross(v, u)
        assert_equal(cp, -z)

    def test_3x3(self):
        u = [1, 2, 3]
        v = [4, 5, 6]
        z = np.array([-3, 6, -3])
        cp = np.cross(u, v)
        assert_equal(cp, z)
        cp = np.cross(v, u)
        assert_equal(cp, -z)

    def test_broadcasting(self):
        # Ticket #2624 (Trac #2032)
        u = np.tile([1, 2], (11, 1))
        v = np.tile([3, 4], (11, 1))
        z = -2
        assert_equal(np.cross(u, v), z)
        assert_equal(np.cross(v, u), -z)
        assert_equal(np.cross(u, u), 0)

        u = np.tile([1, 2], (11, 1)).T
        v = np.tile([3, 4, 5], (11, 1))
        z = np.tile([10, -5, -2], (11, 1))
        assert_equal(np.cross(u, v, axisa=0), z)
        assert_equal(np.cross(v, u.T), -z)
        assert_equal(np.cross(v, v), 0)

        u = np.tile([1, 2, 3], (11, 1)).T
        v = np.tile([3, 4], (11, 1)).T
        z = np.tile([-12, 9, -2], (11, 1))
        assert_equal(np.cross(u, v, axisa=0, axisb=0), z)
        assert_equal(np.cross(v.T, u.T), -z)
        assert_equal(np.cross(u.T, u.T), 0)

        u = np.tile([1, 2, 3], (5, 1))
        v = np.tile([4, 5, 6], (5, 1)).T
        z = np.tile([-3, 6, -3], (5, 1))
        assert_equal(np.cross(u, v, axisb=0), z)
        assert_equal(np.cross(v.T, u), -z)
        assert_equal(np.cross(u, u), 0)

    def test_broadcasting_shapes(self):
        u = np.ones((2, 1, 3))
        v = np.ones((5, 3))
        assert_equal(np.cross(u, v).shape, (2, 5, 3))
        u = np.ones((10, 3, 5))
        v = np.ones((2, 5))
        assert_equal(np.cross(u, v, axisa=1, axisb=0).shape, (10, 5, 3))
        assert_raises(np.AxisError, np.cross, u, v, axisa=1, axisb=2)
        assert_raises(np.AxisError, np.cross, u, v, axisa=3, axisb=0)
        u = np.ones((10, 3, 5, 7))
        v = np.ones((5, 7, 2))
        assert_equal(np.cross(u, v, axisa=1, axisc=2).shape, (10, 5, 3, 7))
        assert_raises(np.AxisError, np.cross, u, v, axisa=-5, axisb=2)
        assert_raises(np.AxisError, np.cross, u, v, axisa=1, axisb=-4)
        # gh-5885
        u = np.ones((3, 4, 2))
        for axisc in range(-2, 2):
            assert_equal(np.cross(u, u, axisc=axisc).shape, (3, 4))

    @skipif(numpy.__version__ < "1.24", reason="fix landed in NumPy 1.24")
    def test_uint8_int32_mixed_dtypes(self):
        # regression test for gh-19138
        u = np.array([[195, 8, 9]], np.uint8)
        v = np.array([250, 166, 68], np.int32)
        z = np.array([[950, 11010, -30370]], dtype=np.int32)
        assert_equal(np.cross(v, u), z)
        assert_equal(np.cross(u, v), -z)


class TestOuterMisc(TestCase):
    def test_outer_out_param(self):
        arr1 = np.ones((5,))
        arr2 = np.ones((2,))
        arr3 = np.linspace(-2, 2, 5)
        out1 = np.empty(shape=(5, 5))
        out2 = np.empty(shape=(2, 5))
        res1 = np.outer(arr1, arr3, out1)
        assert_equal(res1, out1)
        assert_equal(np.outer(arr2, arr3, out2), out2)


@instantiate_parametrized_tests
class TestIndices(TestCase):
    def test_simple(self):
        [x, y] = np.indices((4, 3))
        assert_array_equal(x, np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]]))
        assert_array_equal(y, np.array([[0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2]]))

    def test_single_input(self):
        [x] = np.indices((4,))
        assert_array_equal(x, np.array([0, 1, 2, 3]))

        [x] = np.indices((4,), sparse=True)
        assert_array_equal(x, np.array([0, 1, 2, 3]))

    def test_scalar_input(self):
        assert_array_equal([], np.indices(()))
        assert_array_equal([], np.indices((), sparse=True))
        assert_array_equal([[]], np.indices((0,)))
        assert_array_equal([[]], np.indices((0,), sparse=True))

    def test_sparse(self):
        [x, y] = np.indices((4, 3), sparse=True)
        assert_array_equal(x, np.array([[0], [1], [2], [3]]))
        assert_array_equal(y, np.array([[0, 1, 2]]))

    @parametrize("dtype", [np.int32, np.int64, np.float32, np.float64])
    @parametrize("dims", [(), (0,), (4, 3)])
    def test_return_type(self, dtype, dims):
        inds = np.indices(dims, dtype=dtype)
        assert_(inds.dtype == dtype)

        for arr in np.indices(dims, dtype=dtype, sparse=True):
            assert_(arr.dtype == dtype)


@xpassIfTorchDynamo_np  # (reason="TODO")
class TestRequire(TestCase):
    flag_names = [
        "C",
        "C_CONTIGUOUS",
        "CONTIGUOUS",
        "F",
        "F_CONTIGUOUS",
        "FORTRAN",
        "A",
        "ALIGNED",
        "W",
        "WRITEABLE",
        "O",
        "OWNDATA",
    ]

    def generate_all_false(self, dtype):
        arr = np.zeros((2, 2), [("junk", "i1"), ("a", dtype)])
        arr.setflags(write=False)
        a = arr["a"]
        assert_(not a.flags["C"])
        assert_(not a.flags["F"])
        assert_(not a.flags["O"])
        assert_(not a.flags["W"])
        assert_(not a.flags["A"])
        return a

    def set_and_check_flag(self, flag, dtype, arr):
        if dtype is None:
            dtype = arr.dtype
        b = np.require(arr, dtype, [flag])
        assert_(b.flags[flag])
        assert_(b.dtype == dtype)

        # a further call to np.require ought to return the same array
        # unless OWNDATA is specified.
        c = np.require(b, None, [flag])
        if flag[0] != "O":
            assert_(c is b)
        else:
            assert_(c.flags[flag])

    def test_require_each(self):
        id = ["f8", "i4"]
        fd = [None, "f8", "c16"]
        for idtype, fdtype, flag in itertools.product(id, fd, self.flag_names):
            a = self.generate_all_false(idtype)
            self.set_and_check_flag(flag, fdtype, a)

    def test_unknown_requirement(self):
        a = self.generate_all_false("f8")
        assert_raises(KeyError, np.require, a, None, "Q")

    def test_non_array_input(self):
        a = np.require([1, 2, 3, 4], "i4", ["C", "A", "O"])
        assert_(a.flags["O"])
        assert_(a.flags["C"])
        assert_(a.flags["A"])
        assert_(a.dtype == "i4")
        assert_equal(a, [1, 2, 3, 4])

    def test_C_and_F_simul(self):
        a = self.generate_all_false("f8")
        assert_raises(ValueError, np.require, a, None, ["C", "F"])


@xpassIfTorchDynamo_np  # (reason="TODO")
class TestBroadcast(TestCase):
    def test_broadcast_in_args(self):
        # gh-5881
        arrs = [
            np.empty((6, 7)),
            np.empty((5, 6, 1)),
            np.empty((7,)),
            np.empty((5, 1, 7)),
        ]
        mits = [
            np.broadcast(*arrs),
            np.broadcast(np.broadcast(*arrs[:0]), np.broadcast(*arrs[0:])),
            np.broadcast(np.broadcast(*arrs[:1]), np.broadcast(*arrs[1:])),
            np.broadcast(np.broadcast(*arrs[:2]), np.broadcast(*arrs[2:])),
            np.broadcast(arrs[0], np.broadcast(*arrs[1:-1]), arrs[-1]),
        ]
        for mit in mits:
            assert_equal(mit.shape, (5, 6, 7))
            assert_equal(mit.ndim, 3)
            assert_equal(mit.nd, 3)
            assert_equal(mit.numiter, 4)
            for a, ia in zip(arrs, mit.iters):
                assert_(a is ia.base)

    def test_broadcast_single_arg(self):
        # gh-6899
        arrs = [np.empty((5, 6, 7))]
        mit = np.broadcast(*arrs)
        assert_equal(mit.shape, (5, 6, 7))
        assert_equal(mit.ndim, 3)
        assert_equal(mit.nd, 3)
        assert_equal(mit.numiter, 1)
        assert_(arrs[0] is mit.iters[0].base)

    def test_number_of_arguments(self):
        arr = np.empty((5,))
        for j in range(35):
            arrs = [arr] * j
            if j > 32:
                assert_raises(ValueError, np.broadcast, *arrs)
            else:
                mit = np.broadcast(*arrs)
                assert_equal(mit.numiter, j)

    def test_broadcast_error_kwargs(self):
        # gh-13455
        arrs = [np.empty((5, 6, 7))]
        mit = np.broadcast(*arrs)
        mit2 = np.broadcast(*arrs, **{})  # noqa: PIE804
        assert_equal(mit.shape, mit2.shape)
        assert_equal(mit.ndim, mit2.ndim)
        assert_equal(mit.nd, mit2.nd)
        assert_equal(mit.numiter, mit2.numiter)
        assert_(mit.iters[0].base is mit2.iters[0].base)

        assert_raises(ValueError, np.broadcast, 1, **{"x": 1})  # noqa: PIE804

    @skip(reason="error messages do not match.")
    def test_shape_mismatch_error_message(self):
        with assert_raises(
            ValueError,
            match=r"arg 0 with shape \(1, 3\) and arg 2 with shape \(2,\)",
        ):
            np.broadcast([[1, 2, 3]], [[4], [5]], [6, 7])


class TestTensordot(TestCase):
    def test_zero_dimension(self):
        # Test resolution to issue #5663
        a = np.zeros((3, 0))
        b = np.zeros((0, 4))
        td = np.tensordot(a, b, (1, 0))
        assert_array_equal(td, np.dot(a, b))

    def test_zero_dimension_einsum(self):
        # Test resolution to issue #5663
        a = np.zeros((3, 0))
        b = np.zeros((0, 4))
        td = np.tensordot(a, b, (1, 0))
        assert_array_equal(td, np.einsum("ij,jk", a, b))

    def test_zero_dimensional(self):
        # gh-12130
        arr_0d = np.array(1)
        ret = np.tensordot(
            arr_0d, arr_0d, ([], [])
        )  # contracting no axes is well defined
        assert_array_equal(ret, arr_0d)


if __name__ == "__main__":
    run_tests()
