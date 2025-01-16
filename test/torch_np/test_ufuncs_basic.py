# Owner(s): ["module: dynamo"]

"""
Poking around ufunc casting/broadcasting/dtype/out behavior.

The goal is to validate on numpy, and tests should work when replacing
>>> import numpy as no

by
>>> import torch._numpy as np
"""
import operator
from unittest import skipIf as skip, SkipTest

from pytest import raises as assert_raises

from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TEST_WITH_TORCHDYNAMO,
    TestCase,
)


if TEST_WITH_TORCHDYNAMO:
    import numpy as np
    from numpy.testing import assert_equal
else:
    import torch._numpy as np
    from torch._numpy.testing import assert_equal


parametrize_unary_ufuncs = parametrize("ufunc", [np.sin])
parametrize_casting = parametrize(
    "casting", ["no", "equiv", "safe", "same_kind", "unsafe"]
)


@instantiate_parametrized_tests
class TestUnaryUfuncs(TestCase):
    def get_x(self, ufunc):
        return np.arange(5, dtype="float64")

    @parametrize_unary_ufuncs
    def test_scalar(self, ufunc):
        # check that ufunc accepts a scalar and the result is convertible to scalar
        x = self.get_x(ufunc)[0]
        float(ufunc(x))

    @skip(True, reason="XXX: unary ufuncs ignore the dtype=... parameter")
    @parametrize_unary_ufuncs
    def test_x_and_dtype(self, ufunc):
        x = self.get_x(ufunc)
        res = ufunc(x, dtype="float")
        assert res.dtype == np.dtype("float")

    @skip(True, reason="XXX: unary ufuncs ignore the dtype=... parameter")
    @parametrize_casting
    @parametrize_unary_ufuncs
    @parametrize("dtype", ["float64", "complex128", "float32"])
    def test_x_and_dtype_casting(self, ufunc, casting, dtype):
        x = self.get_x(ufunc)
        if not np.can_cast(x, dtype, casting=casting):
            with assert_raises(TypeError):
                ufunc(x, dtype=dtype, casting=casting)
        else:
            assert ufunc(x, dtype=dtype, casting=casting).dtype == dtype

    @parametrize_casting
    @parametrize_unary_ufuncs
    @parametrize("out_dtype", ["float64", "complex128", "float32"])
    def test_x_and_out_casting(self, ufunc, casting, out_dtype):
        x = self.get_x(ufunc)
        out = np.empty_like(x, dtype=out_dtype)
        if not np.can_cast(x, out_dtype, casting=casting):
            with assert_raises(TypeError):
                ufunc(x, out=out, casting=casting)
        else:
            result = ufunc(x, out=out, casting=casting)
            assert result.dtype == out_dtype
            assert result is out

    @parametrize_unary_ufuncs
    def test_x_and_out_broadcast(self, ufunc):
        x = self.get_x(ufunc)
        out = np.empty((x.shape[0], x.shape[0]))

        x_b = np.broadcast_to(x, out.shape)

        res_out = ufunc(x, out=out)
        res_bcast = ufunc(x_b)
        # TODO: switching the order causes a graph break, failing the test.
        # See test/dynamo/test_misc.py -k test_numpy_graph_break
        assert res_out is out
        assert_equal(res_out, res_bcast)

        out = np.empty((1, x.shape[0]))
        x_b = np.broadcast_to(x, out.shape)

        res_out = ufunc(x, out=out)
        res_bcast = ufunc(x_b)
        assert res_out is out
        assert_equal(res_out, res_bcast)


ufunc_op_iop_numeric = [
    (np.add, operator.__add__, operator.__iadd__),
    (np.subtract, operator.__sub__, operator.__isub__),
    (np.multiply, operator.__mul__, operator.__imul__),
]

ufuncs_with_dunders = [ufunc for ufunc, _, _ in ufunc_op_iop_numeric]
numeric_binary_ufuncs = [
    np.float_power,
    np.power,
]

# these are not implemented for complex inputs
no_complex = [
    np.floor_divide,
    np.hypot,
    np.arctan2,
    np.copysign,
    np.fmax,
    np.fmin,
    np.fmod,
    np.heaviside,
    np.logaddexp,
    np.logaddexp2,
    np.maximum,
    np.minimum,
]

parametrize_binary_ufuncs = parametrize(
    "ufunc", ufuncs_with_dunders + numeric_binary_ufuncs + no_complex
)


# TODO: these snowflakes need special handling
"""
 'bitwise_and',
 'bitwise_or',
 'bitwise_xor',
 'equal',
 'lcm',
 'ldexp',
 'left_shift',
 'less',
 'less_equal',
 'gcd',
 'greater',
 'greater_equal',
 'logical_and',
 'logical_or',
 'logical_xor',
 'matmul',
 'not_equal',
"""


@instantiate_parametrized_tests
class TestBinaryUfuncs(TestCase):
    def get_xy(self, ufunc):
        return np.arange(5, dtype="float64"), np.arange(8, 13, dtype="float64")

    @parametrize_binary_ufuncs
    def test_scalar(self, ufunc):
        # check that ufunc accepts a scalar and the result is convertible to scalar
        xy = self.get_xy(ufunc)
        x, y = xy[0][0], xy[1][0]
        float(ufunc(x, y))

    @parametrize_binary_ufuncs
    def test_vector_vs_scalar(self, ufunc):
        x, y = self.get_xy(ufunc)
        assert_equal(ufunc(x, y), [ufunc(a, b) for a, b in zip(x, y)])

    @parametrize_casting
    @parametrize_binary_ufuncs
    @parametrize("out_dtype", ["float64", "complex128", "float32"])
    def test_xy_and_out_casting(self, ufunc, casting, out_dtype):
        x, y = self.get_xy(ufunc)
        out = np.empty_like(x, dtype=out_dtype)

        if ufunc in no_complex and np.issubdtype(out_dtype, np.complexfloating):
            raise SkipTest(f"{ufunc} does not accept complex.")

        can_cast_x = np.can_cast(x, out_dtype, casting=casting)
        can_cast_y = np.can_cast(y, out_dtype, casting=casting)

        if not (can_cast_x and can_cast_y):
            with assert_raises(TypeError):
                ufunc(x, out=out, casting=casting)
        else:
            result = ufunc(x, y, out=out, casting=casting)
            assert result.dtype == out_dtype
            assert result is out

    @parametrize_binary_ufuncs
    def test_xy_and_out_broadcast(self, ufunc):
        x, y = self.get_xy(ufunc)
        y = y[:, None]
        out = np.empty((2, y.shape[0], x.shape[0]))

        x_b = np.broadcast_to(x, out.shape)
        y_b = np.broadcast_to(y, out.shape)

        res_out = ufunc(x, y, out=out)
        res_bcast = ufunc(x_b, y_b)

        # TODO: switching the order causes a graph break, failing the test.
        # See test/dynamo/test_misc.py -k test_numpy_graph_break
        assert res_out is out
        assert_equal(res_out, res_bcast)


dtypes_numeric = [np.int32, np.float32, np.float64, np.complex128]


@instantiate_parametrized_tests
class TestNdarrayDunderVsUfunc(TestCase):
    """Test ndarray dunders which delegate to ufuncs, vs ufuncs."""

    @parametrize("ufunc, op, iop", ufunc_op_iop_numeric)
    def test_basic(self, ufunc, op, iop):
        """basic op/rop/iop, no dtypes, no broadcasting"""

        # __add__
        a = np.array([1, 2, 3])
        assert_equal(op(a, 1), ufunc(a, 1))
        assert_equal(op(a, a.tolist()), ufunc(a, a.tolist()))
        assert_equal(op(a, a), ufunc(a, a))

        # __radd__
        a = np.array([1, 2, 3])
        assert_equal(op(1, a), ufunc(1, a))
        assert_equal(op(a.tolist(), a), ufunc(a, a.tolist()))

        # __iadd__
        a0 = np.array([2, 4, 6])
        a = a0.copy()

        iop(a, 2)  # modifies a in-place
        assert_equal(a, op(a0, 2))

        a0 = np.array([2, 4, 6])
        a = a0.copy()
        iop(a, a)
        assert_equal(a, op(a0, a0))

    @parametrize("ufunc, op, iop", ufunc_op_iop_numeric)
    @parametrize("other_dtype", dtypes_numeric)
    def test_other_scalar(self, ufunc, op, iop, other_dtype):
        """Test op/iop/rop when the other argument is a scalar of a different dtype."""
        a = np.array([1, 2, 3])
        b = other_dtype(3)

        if ufunc in no_complex and issubclass(other_dtype, np.complexfloating):
            raise SkipTest(f"{ufunc} does not accept complex.")

        # __op__
        result = op(a, b)
        assert_equal(result, ufunc(a, b))

        if result.dtype != np.result_type(a, b):
            assert result.dtype == np.result_type(a, b)

        # __rop__
        result = op(b, a)
        assert_equal(result, ufunc(b, a))
        if result.dtype != np.result_type(a, b):
            assert result.dtype == np.result_type(a, b)

        # __iop__ : casts the result to self.dtype, raises if cannot
        can_cast = np.can_cast(
            np.result_type(a.dtype, other_dtype), a.dtype, casting="same_kind"
        )
        if can_cast:
            a0 = a.copy()
            result = iop(a, b)
            assert_equal(result, ufunc(a0, b))
            if result.dtype != np.result_type(a, b):
                assert result.dtype == np.result_type(a0, b)

        else:
            with assert_raises((TypeError, RuntimeError)):  # XXX np.UFuncTypeError
                iop(a, b)

    @parametrize("ufunc, op, iop", ufunc_op_iop_numeric)
    @parametrize("other_dtype", dtypes_numeric)
    def test_other_array(self, ufunc, op, iop, other_dtype):
        """Test op/iop/rop when the other argument is an array of a different dtype."""
        a = np.array([1, 2, 3])
        b = np.array([5, 6, 7], dtype=other_dtype)

        if ufunc in no_complex and issubclass(other_dtype, np.complexfloating):
            raise SkipTest(f"{ufunc} does not accept complex.")

        # __op__
        result = op(a, b)
        assert_equal(result, ufunc(a, b))
        if result.dtype != np.result_type(a, b):
            assert result.dtype == np.result_type(a, b)

        # __rop__(other array)
        result = op(b, a)
        assert_equal(result, ufunc(b, a))
        if result.dtype != np.result_type(a, b):
            assert result.dtype == np.result_type(a, b)

        # __iop__
        can_cast = np.can_cast(
            np.result_type(a.dtype, other_dtype), a.dtype, casting="same_kind"
        )
        if can_cast:
            a0 = a.copy()
            result = iop(a, b)
            assert_equal(result, ufunc(a0, b))
            if result.dtype != np.result_type(a, b):
                assert result.dtype == np.result_type(a0, b)
        else:
            with assert_raises((TypeError, RuntimeError)):  # XXX np.UFuncTypeError
                iop(a, b)

    @parametrize("ufunc, op, iop", ufunc_op_iop_numeric)
    def test_other_array_bcast(self, ufunc, op, iop):
        """Test op/rop/iop with broadcasting"""
        # __op__
        a = np.array([1, 2, 3])
        result_op = op(a, a[:, None])
        result_ufunc = ufunc(a, a[:, None])
        assert result_op.shape == result_ufunc.shape
        assert_equal(result_op, result_ufunc)

        if result_op.dtype != result_ufunc.dtype:
            assert result_op.dtype == result_ufunc.dtype

        # __rop__
        a = np.array([1, 2, 3])
        result_op = op(a[:, None], a)
        result_ufunc = ufunc(a[:, None], a)
        assert result_op.shape == result_ufunc.shape
        assert_equal(result_op, result_ufunc)

        if result_op.dtype != result_ufunc.dtype:
            assert result_op.dtype == result_ufunc.dtype

        # __iop__ : in-place ops (`self += other` etc) do not broadcast self
        b = a[:, None].copy()
        with assert_raises((ValueError, RuntimeError)):  # XXX ValueError in numpy
            iop(a, b)

        # however, `self += other` broadcasts other
        aa = np.broadcast_to(a, (3, 3)).copy()
        aa0 = aa.copy()

        result = iop(aa, a)
        result_ufunc = ufunc(aa0, a)

        assert result.shape == result_ufunc.shape
        assert_equal(result, result_ufunc)

        if result_op.dtype != result_ufunc.dtype:
            assert result_op.dtype == result_ufunc.dtype


class TestUfuncDtypeKwd(TestCase):
    def test_binary_ufunc_dtype(self):
        # default computation uses float64:
        r64 = np.add(1, 1e-15)
        assert r64.dtype == "float64"
        assert r64 - 1 > 0

        # force the float32 dtype: loss of precision
        r32 = np.add(1, 1e-15, dtype="float32")
        assert r32.dtype == "float32"
        assert r32 == 1

        # now force the cast
        rb = np.add(1.0, 1e-15, dtype=bool, casting="unsafe")
        assert rb.dtype == bool

    def test_binary_ufunc_dtype_and_out(self):
        # all in float64: no precision loss
        out64 = np.empty(2, dtype=np.float64)
        r64 = np.add([1.0, 2.0], 1.0e-15, out=out64)

        assert (r64 != [1.0, 2.0]).all()
        assert r64.dtype == np.float64

        # all in float32: loss of precision, result is float32
        out32 = np.empty(2, dtype=np.float32)
        r32 = np.add([1.0, 2.0], 1.0e-15, dtype=np.float32, out=out32)
        assert (r32 == [1, 2]).all()
        assert r32.dtype == np.float32

        # dtype is float32, so computation is in float32: precision loss
        # the result is then cast to float64
        out64 = np.empty(2, dtype=np.float64)
        r = np.add([1.0, 2.0], 1.0e-15, dtype=np.float32, out=out64)
        assert (r == [1, 2]).all()
        assert r.dtype == np.float64

        # Internal computations are in float64, but the final cast to out.dtype
        # truncates the precision => precision loss.
        out32 = np.empty(2, dtype=np.float32)
        r = np.add([1.0, 2.0], 1.0e-15, dtype=np.float64, out=out32)
        assert (r == [1, 2]).all()
        assert r.dtype == np.float32


if __name__ == "__main__":
    run_tests()
