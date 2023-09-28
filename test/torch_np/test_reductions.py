# Owner(s): ["module: dynamo"]

import pytest

import torch._numpy as np
from pytest import raises as assert_raises

from torch._numpy import _util
from torch._numpy.testing import (
    assert_allclose,
    assert_almost_equal,
    assert_array_equal,
    assert_equal,
)


class TestFlatnonzero:
    def test_basic(self):
        x = np.arange(-2, 3)
        assert_equal(np.flatnonzero(x), [0, 1, 3, 4])


class TestAny:
    def test_basic(self):
        y1 = [0, 0, 1, 0]
        y2 = [0, 0, 0, 0]
        y3 = [1, 0, 1, 0]
        assert np.any(y1)
        assert np.any(y3)
        assert not np.any(y2)

    def test_nd(self):
        y1 = [[0, 0, 0], [0, 1, 0], [1, 1, 0]]
        assert np.any(y1)
        assert_equal(np.any(y1, axis=0), [1, 1, 0])
        assert_equal(np.any(y1, axis=1), [0, 1, 1])
        assert_equal(np.any(y1), True)
        assert isinstance(np.any(y1, axis=1), np.ndarray)

    # YYY: deduplicate
    def test_method_vs_function(self):
        y = np.array([[0, 1, 0, 3], [1, 0, 2, 0]])
        assert_equal(np.any(y), y.any())


class TestAll:
    def test_basic(self):
        y1 = [0, 1, 1, 0]
        y2 = [0, 0, 0, 0]
        y3 = [1, 1, 1, 1]
        assert not np.all(y1)
        assert np.all(y3)
        assert not np.all(y2)
        assert np.all(~np.array(y2))

    def test_nd(self):
        y1 = [[0, 0, 1], [0, 1, 1], [1, 1, 1]]
        assert not np.all(y1)
        assert_equal(np.all(y1, axis=0), [0, 0, 1])
        assert_equal(np.all(y1, axis=1), [0, 0, 1])
        assert_equal(np.all(y1), False)

    def test_method_vs_function(self):
        y = np.array([[0, 1, 0, 3], [1, 0, 2, 0]])
        assert_equal(np.all(y), y.all())


class TestMean:
    def test_mean(self):
        A = [[1, 2, 3], [4, 5, 6]]
        assert np.mean(A) == 3.5
        assert np.all(np.mean(A, 0) == np.array([2.5, 3.5, 4.5]))
        assert np.all(np.mean(A, 1) == np.array([2.0, 5.0]))

        # XXX: numpy emits a warning on empty slice
        assert np.isnan(np.mean([]))

        m = np.asarray(A)
        assert np.mean(A) == m.mean()

    def test_mean_values(self):
        # rmat = np.random.random((4, 5))
        rmat = np.arange(20, dtype=float).reshape((4, 5))
        cmat = rmat + 1j * rmat

        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            for mat in [rmat, cmat]:
                for axis in [0, 1]:
                    tgt = mat.sum(axis=axis)
                    res = np.mean(mat, axis=axis) * mat.shape[axis]
                    assert_allclose(res, tgt)

                for axis in [None]:
                    tgt = mat.sum(axis=axis)
                    res = np.mean(mat, axis=axis) * mat.size
                    assert_allclose(res, tgt)

    def test_mean_float16(self):
        # This fail if the sum inside mean is done in float16 instead
        # of float32.
        assert np.mean(np.ones(100000, dtype="float16")) == 1

    @pytest.mark.xfail(reason="XXX: mean(..., where=...) not implemented")
    def test_mean_where(self):
        a = np.arange(16).reshape((4, 4))
        wh_full = np.array(
            [
                [False, True, False, True],
                [True, False, True, False],
                [True, True, False, False],
                [False, False, True, True],
            ]
        )
        wh_partial = np.array([[False], [True], [True], [False]])
        _cases = [
            (1, True, [1.5, 5.5, 9.5, 13.5]),
            (0, wh_full, [6.0, 5.0, 10.0, 9.0]),
            (1, wh_full, [2.0, 5.0, 8.5, 14.5]),
            (0, wh_partial, [6.0, 7.0, 8.0, 9.0]),
        ]
        for _ax, _wh, _res in _cases:
            assert_allclose(a.mean(axis=_ax, where=_wh), np.array(_res))
            assert_allclose(np.mean(a, axis=_ax, where=_wh), np.array(_res))

        a3d = np.arange(16).reshape((2, 2, 4))
        _wh_partial = np.array([False, True, True, False])
        _res = [[1.5, 5.5], [9.5, 13.5]]
        assert_allclose(a3d.mean(axis=2, where=_wh_partial), np.array(_res))
        assert_allclose(np.mean(a3d, axis=2, where=_wh_partial), np.array(_res))

        with pytest.warns(RuntimeWarning) as w:
            assert_allclose(
                a.mean(axis=1, where=wh_partial), np.array([np.nan, 5.5, 9.5, np.nan])
            )
        with pytest.warns(RuntimeWarning) as w:
            assert_equal(a.mean(where=False), np.nan)
        with pytest.warns(RuntimeWarning) as w:
            assert_equal(np.mean(a, where=False), np.nan)


class TestSum:
    def test_sum(self):
        m = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        tgt = [[6], [15], [24]]
        out = np.sum(m, axis=1, keepdims=True)
        assert_equal(tgt, out)

        am = np.asarray(m)
        assert_equal(np.sum(m), am.sum())

    def test_sum_stability(self):
        a = np.ones(500, dtype=np.float32)
        zero = np.zeros(1, dtype="float32")[0]
        assert_allclose((a / 10.0).sum() - a.size / 10.0, zero, atol=1.5e-4)

        a = np.ones(500, dtype=np.float64)
        assert_allclose((a / 10.0).sum() - a.size / 10.0, 0.0, atol=1.5e-13)

    def test_sum_boolean(self):
        a = np.arange(7) % 2 == 0
        res = a.sum()
        assert_equal(res, 4)

        res_float = a.sum(dtype=np.float64)
        assert_allclose(res_float, 4.0, atol=1e-15)
        assert res_float.dtype == "float64"

    @pytest.mark.xfail(reason="sum: does not warn on overflow")
    def test_sum_dtypes_warnings(self):
        for dt in (int, np.float16, np.float32, np.float64):
            for v in (0, 1, 2, 7, 8, 9, 15, 16, 19, 127, 128, 1024, 1235):
                # warning if sum overflows, which it does in float16
                import warnings

                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always", RuntimeWarning)

                    tgt = dt(v * (v + 1) / 2)
                    overflow = not np.isfinite(tgt)
                    assert_equal(len(w), 1 * overflow)

                    d = np.arange(1, v + 1, dtype=dt)

                    assert_almost_equal(np.sum(d), tgt)
                    assert_equal(len(w), 2 * overflow)

                    assert_almost_equal(np.sum(np.flip(d)), tgt)
                    assert_equal(len(w), 3 * overflow)

    def test_sum_dtypes_2(self):
        for dt in (int, np.float16, np.float32, np.float64):
            d = np.ones(500, dtype=dt)
            assert_almost_equal(np.sum(d[::2]), 250.0)
            assert_almost_equal(np.sum(d[1::2]), 250.0)
            assert_almost_equal(np.sum(d[::3]), 167.0)
            assert_almost_equal(np.sum(d[1::3]), 167.0)
            assert_almost_equal(np.sum(np.flip(d)[::2]), 250.0)

            assert_almost_equal(np.sum(np.flip(d)[1::2]), 250.0)

            assert_almost_equal(np.sum(np.flip(d)[::3]), 167.0)
            assert_almost_equal(np.sum(np.flip(d)[1::3]), 167.0)

            # sum with first reduction entry != 0
            d = np.ones((1,), dtype=dt)
            d += d
            assert_almost_equal(d, 2.0)

    @pytest.mark.parametrize("dt", [np.complex64, np.complex128])
    def test_sum_complex_1(self, dt):
        for v in (0, 1, 2, 7, 8, 9, 15, 16, 19, 127, 128, 1024, 1235):
            tgt = dt(v * (v + 1) / 2) - dt((v * (v + 1) / 2) * 1j)
            d = np.empty(v, dtype=dt)
            d.real = np.arange(1, v + 1)
            d.imag = -np.arange(1, v + 1)
            assert_allclose(np.sum(d), tgt, atol=1.5e-5)
            assert_allclose(np.sum(np.flip(d)), tgt, atol=1.5e-7)

    @pytest.mark.parametrize("dt", [np.complex64, np.complex128])
    def test_sum_complex_2(self, dt):
        d = np.ones(500, dtype=dt) + 1j
        assert_allclose(np.sum(d[::2]), 250.0 + 250j, atol=1.5e-7)
        assert_allclose(np.sum(d[1::2]), 250.0 + 250j, atol=1.5e-7)
        assert_allclose(np.sum(d[::3]), 167.0 + 167j, atol=1.5e-7)
        assert_allclose(np.sum(d[1::3]), 167.0 + 167j, atol=1.5e-7)
        assert_allclose(np.sum(np.flip(d)[::2]), 250.0 + 250j, atol=1.5e-7)
        assert_allclose(np.sum(np.flip(d)[1::2]), 250.0 + 250j, atol=1.5e-7)
        assert_allclose(np.sum(np.flip(d)[::3]), 167.0 + 167j, atol=1.5e-7)
        assert_allclose(np.sum(np.flip(d)[1::3]), 167.0 + 167j, atol=1.5e-7)
        # sum with first reduction entry != 0
        d = np.ones((1,), dtype=dt) + 1j
        d += d
        assert_allclose(d, 2.0 + 2j, atol=1.5e-7)

    @pytest.mark.xfail(reason="initial=... need implementing")
    def test_sum_initial(self):
        # Integer, single axis
        assert_equal(np.sum([3], initial=2), 5)

        # Floating point
        assert_almost_equal(np.sum([0.2], initial=0.1), 0.3)

        # Multiple non-adjacent axes
        assert_equal(
            np.sum(np.ones((2, 3, 5), dtype=np.int64), axis=(0, 2), initial=2),
            [12, 12, 12],
        )

    @pytest.mark.xfail(reason="where=... need implementing")
    def test_sum_where(self):
        # More extensive tests done in test_reduction_with_where.
        assert_equal(np.sum([[1.0, 2.0], [3.0, 4.0]], where=[True, False]), 4.0)
        assert_equal(
            np.sum([[1.0, 2.0], [3.0, 4.0]], axis=0, initial=5.0, where=[True, False]),
            [9.0, 5.0],
        )


class _GenericReductionsTestMixin:
    """Run a set of generic tests to verify that self.func acts like a
    reduction operation.

    Specifically, this class checks axis=... and keepdims=... parameters.
    To check the out=... parameter, see the _GenericHasOutTestMixin class below.

    To use: subclass, define self.func and self.allowed_axes.
    """

    def test_bad_axis(self):
        # Basic check of functionality
        m = np.array([[0, 1, 7, 0, 0], [3, 0, 0, 2, 19]])

        assert_raises(TypeError, self.func, m, axis="foo")
        assert_raises(np.AxisError, self.func, m, axis=3)
        assert_raises(TypeError, self.func, m, axis=np.array([[1], [2]]))
        assert_raises(TypeError, self.func, m, axis=1.5)

        # TODO: add tests with np.int32(3) etc, when implemented

    def test_array_axis(self):
        a = np.array([[0, 1, 7, 0, 0], [3, 0, 0, 2, 19]])
        assert_equal(self.func(a, axis=np.array(-1)), self.func(a, axis=-1))

        with assert_raises(TypeError):
            self.func(a, axis=np.array([1, 2]))

    def test_axis_empty_generic(self):
        a = np.array([[0, 0, 1], [1, 0, 1]])
        assert_array_equal(
            self.func(a, axis=()), self.func(np.expand_dims(a, axis=0), axis=0)
        )

    def test_axis_bad_tuple(self):
        # Basic check of functionality
        m = np.array([[0, 1, 7, 0, 0], [3, 0, 0, 2, 19]])
        with assert_raises(ValueError):
            self.func(m, axis=(1, 1))

    def _check_keepdims_generic(self, axis):
        a = np.arange(2 * 3 * 4).reshape((2, 3, 4))
        with_keepdims = self.func(a, axis, keepdims=True)
        expanded = np.expand_dims(self.func(a, axis=axis), axis=axis)
        assert_array_equal(with_keepdims, expanded)

    def test_keepdims_generic(self):
        for axis in self.allowed_axes:
            self._check_keepdims_generic(axis)

    def test_keepdims_generic_axis_none(self):
        a = np.arange(2 * 3 * 4).reshape((2, 3, 4))
        with_keepdims = self.func(a, axis=None, keepdims=True)
        scalar = self.func(a, axis=None)
        expanded = np.full((1,) * a.ndim, fill_value=scalar)
        assert_array_equal(with_keepdims, expanded)


class _GenericHasOutTestMixin:
    """Tests for reduction functions which support out=... parameter."""

    def test_out_scalar(self):
        # out no axis: scalar
        a = np.arange(2 * 3 * 4).reshape((2, 3, 4))

        result = self.func(a)
        out = np.empty_like(result)
        result_with_out = self.func(a, out=out)

        assert result_with_out is out
        assert_array_equal(result, result_with_out)

    def _check_out_axis(self, axis, dtype, keepdims):
        # out with axis
        a = np.arange(2 * 3 * 4).reshape((2, 3, 4))
        result = self.func(a, axis=axis, keepdims=keepdims).astype(dtype)

        out = np.empty_like(result, dtype=dtype)
        result_with_out = self.func(a, axis=axis, keepdims=keepdims, out=out)

        assert result_with_out is out
        assert result_with_out.dtype == dtype
        assert_array_equal(result, result_with_out)

        # TODO: what if result.dtype != out.dtype; does out typecast the result?

        # out of wrong shape (any/out does not broadcast)
        # np.any(m, out=np.empty_like(m)) raises a ValueError (wrong number
        # of dimensions.)
        # pytorch.any emits a warning and resizes the out array.
        # Here we follow pytorch, since the result is a superset
        # of the numpy functionality

    @pytest.mark.parametrize("keepdims", [True, False, None])
    @pytest.mark.parametrize("dtype", [bool, "int32", "float64"])
    def test_out_axis(self, dtype, keepdims):
        for axis in self.allowed_axes + [None]:
            self._check_out_axis(axis, dtype, keepdims)

    def _check_keepdims_out(self, axis):
        """Check the expicit shape of out w/keepdims=True"""
        d = np.ones((3, 5, 7, 11))
        if axis is None:
            shape_out = (1,) * d.ndim
        else:
            axis_norm = _util.normalize_axis_tuple(axis, d.ndim)
            shape_out = tuple(
                1 if i in axis_norm else d.shape[i] for i in range(d.ndim)
            )
        out = np.empty(shape_out)
        result = self.func(d, axis=axis, keepdims=True, out=out)
        assert result is out
        assert_equal(result.shape, shape_out)

    def test_keepdims_out(self):
        allowed_axes = [ax for ax in self.allowed_axes if ax != ()]
        for axis in allowed_axes + [
            None,
        ]:
            self._check_keepdims_out(axis)


class TestAnyGeneric(_GenericReductionsTestMixin, _GenericHasOutTestMixin):
    def setup_method(self):
        self.func = np.any
        self.allowed_axes = [
            0,
            1,
            2,
            -1,
            -2,
        ]


class TestAllGeneric(_GenericReductionsTestMixin, _GenericHasOutTestMixin):
    def setup_method(self):
        self.func = np.all
        self.allowed_axes = [
            0,
            1,
            2,
            -1,
            -2,
        ]


class TestCountNonzeroGeneric(_GenericReductionsTestMixin):
    # count_nonzero does not have the out=... argument
    def setup_method(self):
        self.func = np.count_nonzero
        self.allowed_axes = [0, 1, 2, -1, -2, (0, 1), (1, 0), (0, 1, 2), (1, -1, 0)]


class TestArgminGeneric(_GenericReductionsTestMixin, _GenericHasOutTestMixin):
    def setup_method(self):
        self.func = np.argmin
        self.allowed_axes = [
            0,
            1,
            2,
            -1,
            -2,
        ]


class TestArgmaxGeneric(_GenericReductionsTestMixin, _GenericHasOutTestMixin):
    def setup_method(self):
        self.func = np.argmax
        self.allowed_axes = [
            0,
            1,
            2,
            -1,
            -2,
        ]


class TestAmaxGeneric(_GenericReductionsTestMixin, _GenericHasOutTestMixin):
    def setup_method(self):
        self.func = np.amax
        self.allowed_axes = [0, 1, 2, -1, -2, (0, 1), (1, 0), (0, 1, 2), (1, -1, 0)]


class TestAminGeneric(_GenericReductionsTestMixin, _GenericHasOutTestMixin):
    def setup_method(self):
        self.func = np.amin
        self.allowed_axes = [0, 1, 2, -1, -2, (0, 1), (1, 0), (0, 1, 2), (1, -1, 0)]


class TestMeanGeneric(_GenericReductionsTestMixin, _GenericHasOutTestMixin):
    def setup_method(self):
        self.func = np.mean
        self.allowed_axes = [0, 1, 2, -1, -2, (0, 1), (1, 0), (0, 1, 2), (1, -1, 0)]


class TestSumGeneric(_GenericReductionsTestMixin, _GenericHasOutTestMixin):
    def setup_method(self):
        self.func = np.sum
        self.allowed_axes = [0, 1, 2, -1, -2, (0, 1), (1, 0), (0, 1, 2), (1, -1, 0)]


class TestProdGeneric(_GenericReductionsTestMixin, _GenericHasOutTestMixin):
    def setup_method(self):
        self.func = np.prod
        self.allowed_axes = [
            0,
            1,
            2,
            -1,
            -2,
        ]


class TestStdGeneric(_GenericReductionsTestMixin, _GenericHasOutTestMixin):
    def setup_method(self):
        self.func = np.std
        self.allowed_axes = [0, 1, 2, -1, -2, (0, 1), (1, 0), (0, 1, 2), (1, -1, 0)]


class TestVarGeneric(_GenericReductionsTestMixin, _GenericHasOutTestMixin):
    def setup_method(self):
        self.func = np.var
        self.allowed_axes = [0, 1, 2, -1, -2, (0, 1), (1, 0), (0, 1, 2), (1, -1, 0)]


class _GenericCumSumProdTestMixin:
    """Run a set of generic tests to verify that self.func acts like a
    reduction operation.

    Specifically, this class checks axis=... and keepdims=... parameters.
    To check the out=... parameter, see the _GenericHasOutTestMixin class below.

    To use: subclass, define self.func and self.allowed_axes.
    """

    def test_bad_axis(self):
        # Basic check of functionality
        m = np.array([[0, 1, 7, 0, 0], [3, 0, 0, 2, 19]])

        assert_raises(TypeError, self.func, m, axis="foo")
        assert_raises(np.AxisError, self.func, m, axis=3)
        assert_raises(TypeError, self.func, m, axis=np.array([[1], [2]]))
        assert_raises(TypeError, self.func, m, axis=1.5)

        # TODO: add tests with np.int32(3) etc, when implemented

    def test_array_axis(self):
        a = np.array([[0, 1, 7, 0, 0], [3, 0, 0, 2, 19]])
        assert_equal(self.func(a, axis=np.array(-1)), self.func(a, axis=-1))

        with assert_raises(TypeError):
            self.func(a, axis=np.array([1, 2]))

    def test_axis_empty_generic(self):
        a = np.array([[0, 0, 1], [1, 0, 1]])
        assert_array_equal(self.func(a, axis=None), self.func(a.ravel(), axis=0))

    def test_axis_bad_tuple(self):
        # Basic check of functionality
        m = np.array([[0, 1, 7, 0, 0], [3, 0, 0, 2, 19]])
        with assert_raises(TypeError):
            self.func(m, axis=(1, 1))


class TestCumProdGeneric(_GenericCumSumProdTestMixin):
    def setup_method(self):
        self.func = np.cumprod


class TestCumSumGeneric(_GenericCumSumProdTestMixin):
    def setup_method(self):
        self.func = np.cumsum
