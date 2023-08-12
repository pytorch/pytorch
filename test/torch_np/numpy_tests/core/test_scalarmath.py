# Owner(s): ["module: dynamo"]

import contextlib
import itertools
import operator
import sys
import warnings

# from numpy._utils import _pep440
import pytest

# from hypothesis import given, settings
# from hypothesis.strategies import sampled_from
# from hypothesis.extra import numpy as hynp

import torch._numpy as np
from pytest import raises as assert_raises
from torch._numpy.testing import (
    _gen_alignment_data,
    assert_,
    assert_almost_equal,
    assert_equal,
    #    assert_array_equal, suppress_warnings, _gen_alignment_data,
    #    assert_warns,
)

IS_PYPY = False

types = [
    np.bool_,
    np.byte,
    np.ubyte,
    np.short,
    np.intc,
    np.int_,
    np.longlong,
    np.single,
    np.double,
    np.csingle,
    np.cdouble,
]

floating_types = np.floating.__subclasses__()
complex_floating_types = np.complexfloating.__subclasses__()

objecty_things = [object(), None]

reasonable_operators_for_scalars = [
    operator.lt,
    operator.le,
    operator.eq,
    operator.ne,
    operator.ge,
    operator.gt,
    operator.add,
    operator.floordiv,
    operator.mod,
    operator.mul,
    operator.pow,
    operator.sub,
    operator.truediv,
]


# This compares scalarmath against ufuncs.


class TestTypes:
    def test_types(self):
        for atype in types:
            a = atype(1)
            assert_(a == 1, f"error with {atype!r}: got {a!r}")

    def test_type_add(self):
        # list of types
        for k, atype in enumerate(types):
            a_scalar = atype(3)
            a_array = np.array([3], dtype=atype)
            for l, btype in enumerate(types):
                b_scalar = btype(1)
                b_array = np.array([1], dtype=btype)
                c_scalar = a_scalar + b_scalar
                c_array = a_array + b_array
                # It was comparing the type numbers, but the new ufunc
                # function-finding mechanism finds the lowest function
                # to which both inputs can be cast - which produces 'l'
                # when you do 'q' + 'b'.  The old function finding mechanism
                # skipped ahead based on the first argument, but that
                # does not produce properly symmetric results...
                assert_equal(
                    c_scalar.dtype,
                    c_array.dtype,
                    "error with types (%d/'%s' + %d/'%s')"
                    % (k, np.dtype(atype).name, l, np.dtype(btype).name),
                )

    def test_type_create(self):
        for k, atype in enumerate(types):
            a = np.array([1, 2, 3], atype)
            b = atype([1, 2, 3])
            assert_equal(a, b)

    def test_leak(self):
        # test leak of scalar objects
        # a leak would show up in valgrind as still-reachable of ~2.6MB
        for i in range(200000):
            np.add(1, 1)


'''
@pytest.mark.slow
@settings(max_examples=10000, deadline=2000)
@given(sampled_from(reasonable_operators_for_scalars),
       hynp.arrays(dtype=hynp.scalar_dtypes(), shape=()),
       hynp.arrays(dtype=hynp.scalar_dtypes(), shape=()))
def test_array_scalar_ufunc_equivalence(op, arr1, arr2):
    """
    This is a thorough test attempting to cover important promotion paths
    and ensuring that arrays and scalars stay as aligned as possible.
    However, if it creates troubles, it should maybe just be removed.
    """
    scalar1 = arr1[()]
    scalar2 = arr2[()]
    assert isinstance(scalar1, np.generic)
    assert isinstance(scalar2, np.generic)

    if arr1.dtype.kind == "c" or arr2.dtype.kind == "c":
        comp_ops = {operator.ge, operator.gt, operator.le, operator.lt}
        if op in comp_ops and (np.isnan(scalar1) or np.isnan(scalar2)):
            pytest.xfail("complex comp ufuncs use sort-order, scalars do not.")

    # ignore fpe's since they may just mismatch for integers anyway.
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        # Comparisons DeprecationWarnings replacing errors (2022-03):
        warnings.simplefilter("error", DeprecationWarning)
        try:
            res = op(arr1, arr2)
        except Exception as e:
            with pytest.raises(type(e)):
                op(scalar1, scalar2)
        else:
            scalar_res = op(scalar1, scalar2)
            assert_array_equal(scalar_res, res)
'''


class TestBaseMath:
    def test_blocked(self):
        # test alignments offsets for simd instructions
        # alignments for vz + 2 * (vs - 1) + 1
        for dt, sz in [(np.float32, 11), (np.float64, 7), (np.int32, 11)]:
            for out, inp1, inp2, msg in _gen_alignment_data(
                dtype=dt, type="binary", max_size=sz
            ):
                exp1 = np.ones_like(inp1)
                inp1[...] = np.ones_like(inp1)
                inp2[...] = np.zeros_like(inp2)
                assert_almost_equal(np.add(inp1, inp2), exp1, err_msg=msg)
                assert_almost_equal(np.add(inp1, 2), exp1 + 2, err_msg=msg)
                assert_almost_equal(np.add(1, inp2), exp1, err_msg=msg)

                np.add(inp1, inp2, out=out)
                assert_almost_equal(out, exp1, err_msg=msg)

                inp2[...] += np.arange(inp2.size, dtype=dt) + 1
                assert_almost_equal(
                    np.square(inp2), np.multiply(inp2, inp2), err_msg=msg
                )
                # skip true divide for ints
                if dt != np.int32:
                    assert_almost_equal(
                        np.reciprocal(inp2), np.divide(1, inp2), err_msg=msg
                    )

                inp1[...] = np.ones_like(inp1)
                np.add(inp1, 2, out=out)
                assert_almost_equal(out, exp1 + 2, err_msg=msg)
                inp2[...] = np.ones_like(inp2)
                np.add(2, inp2, out=out)
                assert_almost_equal(out, exp1 + 2, err_msg=msg)

    @pytest.mark.xfail(reason="pytorch does not have .view")
    def test_lower_align(self):
        # check data that is not aligned to element size
        # i.e doubles are aligned to 4 bytes on i386
        d = np.zeros(23 * 8, dtype=np.int8)[4:-4].view(np.float64)
        o = np.zeros(23 * 8, dtype=np.int8)[4:-4].view(np.float64)
        assert_almost_equal(d + d, d * 2)
        np.add(d, d, out=o)
        np.add(np.ones_like(d), d, out=o)
        np.add(d, np.ones_like(d), out=o)
        np.add(np.ones_like(d), d)
        np.add(d, np.ones_like(d))


class TestPower:
    def test_small_types(self):
        for t in [np.int8, np.int16, np.float16]:
            a = t(3)
            b = a**4
            assert_(b == 81, f"error with {t!r}: got {b!r}")

    def test_large_types(self):
        for t in [np.int32, np.int64, np.float32, np.float64]:
            a = t(51)
            b = a**4
            msg = f"error with {t!r}: got {b!r}"
            if np.issubdtype(t, np.integer):
                assert_(b == 6765201, msg)
            else:
                assert_almost_equal(b, 6765201, err_msg=msg)

    @pytest.mark.xfail(reason="Value-based casting: (2)**(-2) -> 0 in pytorch.")
    def test_integers_to_negative_integer_power(self):
        # Note that the combination of uint64 with a signed integer
        # has common type np.float64. The other combinations should all
        # raise a ValueError for integer ** negative integer.
        exp = [np.array(-1, dt)[()] for dt in "bhil"]

        # 1 ** -1 possible special case
        base = [np.array(1, dt)[()] for dt in "bhilB"]
        for i1, i2 in itertools.product(base, exp):
            if i1.dtype != np.uint64:
                assert_raises(ValueError, operator.pow, i1, i2)
            else:
                res = operator.pow(i1, i2)
                assert_(res.dtype.type is np.float64)
                assert_almost_equal(res, 1.0)

        # -1 ** -1 possible special case
        base = [np.array(-1, dt)[()] for dt in "bhil"]
        for i1, i2 in itertools.product(base, exp):
            if i1.dtype != np.uint64:
                assert_raises(ValueError, operator.pow, i1, i2)
            else:
                res = operator.pow(i1, i2)
                assert_(res.dtype.type is np.float64)
                assert_almost_equal(res, -1.0)

        # 2 ** -1 perhaps generic
        base = [np.array(2, dt)[()] for dt in "bhilB"]
        for i1, i2 in itertools.product(base, exp):
            if i1.dtype != np.uint64:
                assert_raises(ValueError, operator.pow, i1, i2)
            else:
                res = operator.pow(i1, i2)
                assert_(res.dtype.type is np.float64)
                assert_almost_equal(res, 0.5)

    def test_mixed_types(self):
        typelist = [
            np.int8,
            np.int16,
            np.float16,
            np.float32,
            np.float64,
            np.int8,
            np.int16,
            np.int32,
            np.int64,
        ]
        for t1 in typelist:
            for t2 in typelist:
                a = t1(3)
                b = t2(2)
                result = a**b
                msg = ("error with {!r} and {!r}:" "got {!r}, expected {!r}").format(
                    t1, t2, result, 9
                )
                if np.issubdtype(np.dtype(result), np.integer):
                    assert_(result == 9, msg)
                else:
                    assert_almost_equal(result, 9, err_msg=msg)

    def test_modular_power(self):
        # modular power is not implemented, so ensure it errors
        a = 5
        b = 4
        c = 10
        expected = pow(a, b, c)  # noqa: F841
        for t in (np.int32, np.float32, np.complex64):
            # note that 3-operand power only dispatches on the first argument
            assert_raises(TypeError, operator.pow, t(a), b, c)
            assert_raises(TypeError, operator.pow, np.array(t(a)), b, c)


def floordiv_and_mod(x, y):
    return (x // y, x % y)


def _signs(dt):
    if dt in np.typecodes["UnsignedInteger"]:
        return (+1,)
    else:
        return (+1, -1)


class TestModulus:
    def test_modulus_basic(self):
        dt = np.typecodes["AllInteger"] + np.typecodes["Float"]
        for op in [floordiv_and_mod, divmod]:
            for dt1, dt2 in itertools.product(dt, dt):
                for sg1, sg2 in itertools.product(_signs(dt1), _signs(dt2)):
                    fmt = "op: %s, dt1: %s, dt2: %s, sg1: %s, sg2: %s"
                    msg = fmt % (op.__name__, dt1, dt2, sg1, sg2)
                    a = np.array(sg1 * 71, dtype=dt1)[()]
                    b = np.array(sg2 * 19, dtype=dt2)[()]
                    div, rem = op(a, b)
                    assert_equal(div * b + rem, a, err_msg=msg)
                    if sg2 == -1:
                        assert_(b < rem <= 0, msg)
                    else:
                        assert_(b > rem >= 0, msg)

    @pytest.mark.slow
    def test_float_modulus_exact(self):
        # test that float results are exact for small integers. This also
        # holds for the same integers scaled by powers of two.
        nlst = list(range(-127, 0))
        plst = list(range(1, 128))
        dividend = nlst + [0] + plst
        divisor = nlst + plst
        arg = list(itertools.product(dividend, divisor))
        tgt = [divmod(*t) for t in arg]

        a, b = np.array(arg, dtype=int).T
        # convert exact integer results from Python to float so that
        # signed zero can be used, it is checked.
        tgtdiv, tgtrem = np.array(tgt, dtype=float).T
        tgtdiv = np.where((tgtdiv == 0.0) & ((b < 0) ^ (a < 0)), -0.0, tgtdiv)
        tgtrem = np.where((tgtrem == 0.0) & (b < 0), -0.0, tgtrem)

        for op in [floordiv_and_mod, divmod]:
            for dt in np.typecodes["Float"]:
                msg = f"op: {op.__name__}, dtype: {dt}"
                fa = a.astype(dt)
                fb = b.astype(dt)
                # use list comprehension so a_ and b_ are scalars
                div, rem = zip(*[op(a_, b_) for a_, b_ in zip(fa, fb)])
                assert_equal(div, tgtdiv, err_msg=msg)
                assert_equal(rem, tgtrem, err_msg=msg)

    def test_float_modulus_roundoff(self):
        # gh-6127
        dt = np.typecodes["Float"]
        for op in [floordiv_and_mod, divmod]:
            for dt1, dt2 in itertools.product(dt, dt):
                for sg1, sg2 in itertools.product((+1, -1), (+1, -1)):
                    fmt = "op: %s, dt1: %s, dt2: %s, sg1: %s, sg2: %s"
                    msg = fmt % (op.__name__, dt1, dt2, sg1, sg2)
                    a = np.array(sg1 * 78 * 6e-8, dtype=dt1)[()]
                    b = np.array(sg2 * 6e-8, dtype=dt2)[()]
                    div, rem = op(a, b)
                    # Equal assertion should hold when fmod is used
                    assert_equal(div * b + rem, a, err_msg=msg)
                    if sg2 == -1:
                        assert_(b < rem <= 0, msg)
                    else:
                        assert_(b > rem >= 0, msg)

    @pytest.mark.parametrize("dt", np.typecodes["Float"])
    def test_float_modulus_corner_cases(self, dt):
        if dt == "e":
            pytest.xfail(
                reason="RuntimeError: 'nextafter_cpu' not implemented for 'Half'"
            )

        b = np.array(1.0, dtype=dt)
        a = np.nextafter(np.array(0.0, dtype=dt), -b)
        rem = operator.mod(a, b)
        assert_(rem <= b, "dt: %s" % dt)
        rem = operator.mod(-a, -b)
        assert_(rem >= -b, "dt: %s" % dt)

        # Check nans, inf
        #     with suppress_warnings() as sup:
        #         sup.filter(RuntimeWarning, "invalid value encountered in remainder")
        #         sup.filter(RuntimeWarning, "divide by zero encountered in remainder")
        #         sup.filter(RuntimeWarning, "divide by zero encountered in floor_divide")
        #         sup.filter(RuntimeWarning, "divide by zero encountered in divmod")
        #         sup.filter(RuntimeWarning, "invalid value encountered in divmod")
        for dt in np.typecodes["Float"]:
            fone = np.array(1.0, dtype=dt)
            fzer = np.array(0.0, dtype=dt)
            finf = np.array(np.inf, dtype=dt)
            fnan = np.array(np.nan, dtype=dt)
            rem = operator.mod(fone, fzer)
            assert_(np.isnan(rem), "dt: %s" % dt)
            # MSVC 2008 returns NaN here, so disable the check.
            # rem = operator.mod(fone, finf)
            # assert_(rem == fone, 'dt: %s' % dt)
            rem = operator.mod(fone, fnan)
            assert_(np.isnan(rem), "dt: %s" % dt)
            rem = operator.mod(finf, fone)
            assert_(np.isnan(rem), "dt: %s" % dt)
            for op in [floordiv_and_mod, divmod]:
                div, mod = op(fone, fzer)
                assert_(np.isinf(div)) and assert_(np.isnan(mod))


class TestComplexDivision:
    @pytest.mark.skip(reason="With pytorch, 1/(0+0j) is nan + nan*j, not inf + nan*j")
    def test_zero_division(self):
        with np.errstate(all="ignore"):
            for t in [np.complex64, np.complex128]:
                a = t(0.0)
                b = t(1.0)
                assert_(np.isinf(b / a))
                b = t(complex(np.inf, np.inf))
                assert_(np.isinf(b / a))
                b = t(complex(np.inf, np.nan))
                assert_(np.isinf(b / a))
                b = t(complex(np.nan, np.inf))
                assert_(np.isinf(b / a))
                b = t(complex(np.nan, np.nan))
                assert_(np.isnan(b / a))
                b = t(0.0)
                assert_(np.isnan(b / a))

    def test_signed_zeros(self):
        with np.errstate(all="ignore"):
            for t in [np.complex64, np.complex128]:
                # tupled (numerator, denominator, expected)
                # for testing as expected == numerator/denominator
                data = (
                    ((0.0, -1.0), (0.0, 1.0), (-1.0, -0.0)),
                    ((0.0, -1.0), (0.0, -1.0), (1.0, -0.0)),
                    ((0.0, -1.0), (-0.0, -1.0), (1.0, 0.0)),
                    ((0.0, -1.0), (-0.0, 1.0), (-1.0, 0.0)),
                    ((0.0, 1.0), (0.0, -1.0), (-1.0, 0.0)),
                    ((0.0, -1.0), (0.0, -1.0), (1.0, -0.0)),
                    ((-0.0, -1.0), (0.0, -1.0), (1.0, -0.0)),
                    ((-0.0, 1.0), (0.0, -1.0), (-1.0, -0.0)),
                )
                for cases in data:
                    n = cases[0]
                    d = cases[1]
                    ex = cases[2]
                    result = t(complex(n[0], n[1])) / t(complex(d[0], d[1]))
                    # check real and imag parts separately to avoid comparison
                    # in array context, which does not account for signed zeros
                    assert_equal(result.real, ex[0])
                    assert_equal(result.imag, ex[1])

    def test_branches(self):
        with np.errstate(all="ignore"):
            for t in [np.complex64, np.complex128]:
                # tupled (numerator, denominator, expected)
                # for testing as expected == numerator/denominator
                data = list()

                # trigger branch: real(fabs(denom)) > imag(fabs(denom))
                # followed by else condition as neither are == 0
                data.append(((2.0, 1.0), (2.0, 1.0), (1.0, 0.0)))

                # trigger branch: real(fabs(denom)) > imag(fabs(denom))
                # followed by if condition as both are == 0
                # is performed in test_zero_division(), so this is skipped

                # trigger else if branch: real(fabs(denom)) < imag(fabs(denom))
                data.append(((1.0, 2.0), (1.0, 2.0), (1.0, 0.0)))

                for cases in data:
                    n = cases[0]
                    d = cases[1]
                    ex = cases[2]
                    result = t(complex(n[0], n[1])) / t(complex(d[0], d[1]))
                    # check real and imag parts separately to avoid comparison
                    # in array context, which does not account for signed zeros
                    assert_equal(result.real, ex[0])
                    assert_equal(result.imag, ex[1])


class TestConversion:
    def test_int_from_long(self):
        # NB: this test assumes that the default fp type is float64
        l = [1e6, 1e12, 1e18, -1e6, -1e12, -1e18]
        li = [10**6, 10**12, 10**18, -(10**6), -(10**12), -(10**18)]
        for T in [None, np.float64, np.int64]:
            a = np.array(l, dtype=T)
            assert_equal([int(_m) for _m in a], li)

    @pytest.mark.xfail(reason="pytorch does not emit this warning.")
    def test_iinfo_long_values_1(self):
        for code in "bBh":
            with pytest.warns(DeprecationWarning):
                res = np.array(np.iinfo(code).max + 1, dtype=code)
            tgt = np.iinfo(code).min
            assert_(res == tgt)

    def test_iinfo_long_values_2(self):
        for code in np.typecodes["AllInteger"]:
            res = np.array(np.iinfo(code).max, dtype=code)
            tgt = np.iinfo(code).max
            assert_(res == tgt)

        for code in np.typecodes["AllInteger"]:
            res = np.dtype(code).type(np.iinfo(code).max)
            tgt = np.iinfo(code).max
            assert_(res == tgt)

    def test_int_raise_behaviour(self):
        def overflow_error_func(dtype):
            dtype(np.iinfo(dtype).max + 1)

        for code in [np.int_, np.longlong]:
            assert_raises(RuntimeError, overflow_error_func, code)

    def test_numpy_scalar_relational_operators(self):
        # All integer
        for dt1 in np.typecodes["AllInteger"]:
            assert_(1 > np.array(0, dtype=dt1)[()], f"type {dt1} failed")
            assert_(not 1 < np.array(0, dtype=dt1)[()], f"type {dt1} failed")

            for dt2 in np.typecodes["AllInteger"]:
                assert_(
                    np.array(1, dtype=dt1)[()] > np.array(0, dtype=dt2)[()],
                    f"type {dt1} and {dt2} failed",
                )
                assert_(
                    not np.array(1, dtype=dt1)[()] < np.array(0, dtype=dt2)[()],
                    f"type {dt1} and {dt2} failed",
                )

        # Signed integers and floats
        for dt1 in "bhl" + np.typecodes["Float"]:
            assert_(1 > np.array(-1, dtype=dt1)[()], f"type {dt1} failed")
            assert_(not 1 < np.array(-1, dtype=dt1)[()], f"type {dt1} failed")
            assert_(-1 == np.array(-1, dtype=dt1)[()], f"type {dt1} failed")

            for dt2 in "bhl" + np.typecodes["Float"]:
                assert_(
                    np.array(1, dtype=dt1)[()] > np.array(-1, dtype=dt2)[()],
                    f"type {dt1} and {dt2} failed",
                )
                assert_(
                    not np.array(1, dtype=dt1)[()] < np.array(-1, dtype=dt2)[()],
                    f"type {dt1} and {dt2} failed",
                )
                assert_(
                    np.array(-1, dtype=dt1)[()] == np.array(-1, dtype=dt2)[()],
                    f"type {dt1} and {dt2} failed",
                )

    def test_numpy_scalar_relational_operators_2(self):
        # Unsigned integers
        for dt1 in "B":
            assert_(-1 < np.array(1, dtype=dt1)[()], f"type {dt1} failed")
            assert_(not -1 > np.array(1, dtype=dt1)[()], f"type {dt1} failed")
            assert_(-1 != np.array(1, dtype=dt1)[()], f"type {dt1} failed")

            # unsigned vs signed
            for dt2 in "bhil":
                assert_(
                    np.array(1, dtype=dt1)[()] > np.array(-1, dtype=dt2)[()],
                    f"type {dt1} and {dt2} failed",
                )
                assert_(
                    not np.array(1, dtype=dt1)[()] < np.array(-1, dtype=dt2)[()],
                    f"type {dt1} and {dt2} failed",
                )
                assert_(
                    np.array(1, dtype=dt1)[()] != np.array(-1, dtype=dt2)[()],
                    f"type {dt1} and {dt2} failed",
                )

    def test_scalar_comparison_to_none(self):
        # Scalars should just return False and not give a warnings.
        # The comparisons are flagged by pep8, ignore that.
        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings("always", "", FutureWarning)
            assert_(np.float32(1) is not None)
            assert_(np.float32(1) is not None)
        assert_(len(w) == 0)


# class TestRepr:
#    def test_repr(self):
#        for t in types:
#            val = t(1197346475.0137341)
#            val_repr = repr(val)
#            val2 = eval(val_repr)
#            assert_equal( val, val2 )


@pytest.mark.xfail(reason="can delegate repr to pytorch")
class TestRepr:
    def _test_type_repr(self, t):
        finfo = np.finfo(t)
        last_fraction_bit_idx = finfo.nexp + finfo.nmant
        last_exponent_bit_idx = finfo.nexp
        storage_bytes = np.dtype(t).itemsize * 8
        # could add some more types to the list below
        for which in ["small denorm", "small norm"]:
            # Values from https://en.wikipedia.org/wiki/IEEE_754
            constr = np.array([0x00] * storage_bytes, dtype=np.uint8)
            if which == "small denorm":
                byte = last_fraction_bit_idx // 8
                bytebit = 7 - (last_fraction_bit_idx % 8)
                constr[byte] = 1 << bytebit
            elif which == "small norm":
                byte = last_exponent_bit_idx // 8
                bytebit = 7 - (last_exponent_bit_idx % 8)
                constr[byte] = 1 << bytebit
            else:
                raise ValueError("hmm")
            val = constr.view(t)[0]
            val_repr = repr(val)
            val2 = t(eval(val_repr))
            if not (val2 == 0 and val < 1e-100):
                assert_equal(val, val2)

    def test_float_repr(self):
        # long double test cannot work, because eval goes through a python
        # float
        for t in [np.float32, np.float64]:
            self._test_type_repr(t)


@pytest.mark.skip(reason="Array scalars do not decay to python scalars.")
class TestMultiply:
    def test_seq_repeat(self):
        # Test that basic sequences get repeated when multiplied with
        # numpy integers. And errors are raised when multiplied with others.
        # Some of this behaviour may be controversial and could be open for
        # change.
        accepted_types = set(np.typecodes["AllInteger"])
        deprecated_types = {"?"}
        forbidden_types = set(np.typecodes["All"]) - accepted_types - deprecated_types
        forbidden_types -= {"V"}  # can't default-construct void scalars

        for seq_type in (list, tuple):
            seq = seq_type([1, 2, 3])
            for numpy_type in accepted_types:
                i = np.dtype(numpy_type).type(2)
                assert_equal(seq * i, seq * int(i))
                assert_equal(i * seq, int(i) * seq)

            for numpy_type in deprecated_types:
                i = np.dtype(numpy_type).type()
                assert_equal(
                    assert_warns(DeprecationWarning, operator.mul, seq, i), seq * int(i)
                )
                assert_equal(
                    assert_warns(DeprecationWarning, operator.mul, i, seq), int(i) * seq
                )

            for numpy_type in forbidden_types:
                i = np.dtype(numpy_type).type()
                assert_raises(TypeError, operator.mul, seq, i)
                assert_raises(TypeError, operator.mul, i, seq)

    def test_no_seq_repeat_basic_array_like(self):
        # Test that an array-like which does not know how to be multiplied
        # does not attempt sequence repeat (raise TypeError).
        # See also gh-7428.
        class ArrayLike:
            def __init__(self, arr):
                self.arr = arr

            def __array__(self):
                return self.arr

        # Test for simple ArrayLike above and memoryviews (original report)
        for arr_like in (ArrayLike(np.ones(3)), memoryview(np.ones(3))):
            assert_array_equal(arr_like * np.float32(3.0), np.full(3, 3.0))
            assert_array_equal(np.float32(3.0) * arr_like, np.full(3, 3.0))
            assert_array_equal(arr_like * np.int_(3), np.full(3, 3))
            assert_array_equal(np.int_(3) * arr_like, np.full(3, 3))


class TestNegative:
    def test_exceptions(self):
        a = np.ones((), dtype=np.bool_)[()]
        # XXX: TypeError from numpy, RuntimeError from torch
        assert_raises((TypeError, RuntimeError), operator.neg, a)

    def test_result(self):
        types = np.typecodes["AllInteger"] + np.typecodes["AllFloat"]
        # with suppress_warnings() as sup:
        #     sup.filter(RuntimeWarning)
        for dt in types:
            a = np.ones((), dtype=dt)[()]
            if dt in np.typecodes["UnsignedInteger"]:
                st = np.dtype(dt).type
                max = st(np.iinfo(dt).max)
                assert_equal(operator.neg(a), max)
            else:
                assert_equal(operator.neg(a) + a, 0)


class TestSubtract:
    def test_exceptions(self):
        a = np.ones((), dtype=np.bool_)[()]
        with assert_raises((TypeError, RuntimeError)):  # XXX: TypeError from numpy
            operator.sub(a, a)  # RuntimeError from torch

    def test_result(self):
        types = np.typecodes["AllInteger"] + np.typecodes["AllFloat"]
        #        with suppress_warnings() as sup:
        #            sup.filter(RuntimeWarning)
        for dt in types:
            a = np.ones((), dtype=dt)[()]
            assert_equal(operator.sub(a, a), 0)


class TestAbs:
    def _test_abs_func(self, absfunc, test_dtype):
        x = test_dtype(-1.5)
        assert_equal(absfunc(x), 1.5)
        x = test_dtype(0.0)
        res = absfunc(x)
        # assert_equal() checks zero signedness
        assert_equal(res, 0.0)
        x = test_dtype(-0.0)
        res = absfunc(x)
        assert_equal(res, 0.0)

        x = test_dtype(np.finfo(test_dtype).max)
        assert_equal(absfunc(x), x.real)

        #      with suppress_warnings() as sup:
        #          sup.filter(UserWarning)
        x = test_dtype(np.finfo(test_dtype).tiny)
        assert_equal(absfunc(x), x.real)

        x = test_dtype(np.finfo(test_dtype).min)
        assert_equal(absfunc(x), -x.real)

    @pytest.mark.parametrize("dtype", floating_types + complex_floating_types)
    def test_builtin_abs(self, dtype):
        self._test_abs_func(abs, dtype)

    @pytest.mark.parametrize("dtype", floating_types + complex_floating_types)
    def test_numpy_abs(self, dtype):
        self._test_abs_func(np.abs, dtype)


class TestBitShifts:
    @pytest.mark.parametrize("type_code", np.typecodes["AllInteger"])
    @pytest.mark.parametrize("op", [operator.rshift, operator.lshift], ids=[">>", "<<"])
    def test_shift_all_bits(self, type_code, op):
        """Shifts where the shift amount is the width of the type or wider"""
        # gh-2449
        dt = np.dtype(type_code)
        nbits = dt.itemsize * 8
        for val in [5, -5]:
            for shift in [nbits, nbits + 4]:
                val_scl = np.array(val).astype(dt)[()]
                shift_scl = dt.type(shift)

                res_scl = op(val_scl, shift_scl)
                if val_scl < 0 and op is operator.rshift:
                    # sign bit is preserved
                    assert_equal(res_scl, -1)
                else:
                    if type_code in ("i", "l") and shift == np.iinfo(type_code).bits:
                        pytest.xfail("https://github.com/pytorch/pytorch/issues/70904")
                    assert_equal(res_scl, 0)

                # Result on scalars should be the same as on arrays
                val_arr = np.array([val_scl] * 32, dtype=dt)
                shift_arr = np.array([shift] * 32, dtype=dt)
                res_arr = op(val_arr, shift_arr)
                assert_equal(res_arr, res_scl)


@pytest.mark.skip(reason="Will rely on pytest for hashing")
class TestHash:
    @pytest.mark.parametrize("type_code", np.typecodes["AllInteger"])
    def test_integer_hashes(self, type_code):
        scalar = np.dtype(type_code).type
        for i in range(128):
            assert hash(i) == hash(scalar(i))

    @pytest.mark.parametrize("type_code", np.typecodes["AllFloat"])
    def test_float_and_complex_hashes(self, type_code):
        scalar = np.dtype(type_code).type
        for val in [np.pi, np.inf, 3, 6.0]:
            numpy_val = scalar(val)
            # Cast back to Python, in case the NumPy scalar has less precision
            if numpy_val.dtype.kind == "c":
                val = complex(numpy_val)
            else:
                val = float(numpy_val)
            assert val == numpy_val
            assert hash(val) == hash(numpy_val)

        if hash(float(np.nan)) != hash(float(np.nan)):
            # If Python distinguishes different NaNs we do so too (gh-18833)
            assert hash(scalar(np.nan)) != hash(scalar(np.nan))

    @pytest.mark.parametrize("type_code", np.typecodes["Complex"])
    def test_complex_hashes(self, type_code):
        # Test some complex valued hashes specifically:
        scalar = np.dtype(type_code).type
        for val in [np.pi + 1j, np.inf - 3j, 3j, 6.0 + 1j]:
            numpy_val = scalar(val)
            assert hash(complex(numpy_val)) == hash(numpy_val)


@contextlib.contextmanager
def recursionlimit(n):
    o = sys.getrecursionlimit()
    try:
        sys.setrecursionlimit(n)
        yield
    finally:
        sys.setrecursionlimit(o)


"""
@given(sampled_from(objecty_things),
       sampled_from(reasonable_operators_for_scalars),
       sampled_from(types))
def test_operator_object_left(o, op, type_):
    try:
        with recursionlimit(200):
            op(o, type_(1))
    except TypeError:
        pass



@given(sampled_from(objecty_things),
       sampled_from(reasonable_operators_for_scalars),
       sampled_from(types))
def test_operator_object_right(o, op, type_):
    try:
        with recursionlimit(200):
            op(type_(1), o)
    except TypeError:
        pass


@given(sampled_from(reasonable_operators_for_scalars),
       sampled_from(types),
       sampled_from(types))
def test_operator_scalars(op, type1, type2):
    try:
        op(type1(1), type2(1))
    except TypeError:
        pass
"""


@pytest.mark.xfail(reason="pytorch does not warn on overflow")
@pytest.mark.parametrize("dtype", np.typecodes["AllInteger"])
@pytest.mark.parametrize(
    "operation",
    [
        lambda min, max: max + max,
        lambda min, max: min - max,
        lambda min, max: max * max,
    ],
    ids=["+", "-", "*"],
)
def test_scalar_integer_operation_overflow(dtype, operation):
    st = np.dtype(dtype).type
    min = st(np.iinfo(dtype).min)
    max = st(np.iinfo(dtype).max)

    with pytest.warns(RuntimeWarning, match="overflow encountered"):
        operation(min, max)


@pytest.mark.xfail(reason="pytorch does not warn on overflow")
@pytest.mark.parametrize("dtype", np.typecodes["Integer"])
@pytest.mark.parametrize(
    "operation",
    [
        lambda min, neg_1: -min,
        lambda min, neg_1: abs(min),
        lambda min, neg_1: min * neg_1,
        pytest.param(
            lambda min, neg_1: min // neg_1,
            marks=pytest.mark.skip(reason="broken on some platforms"),
        ),
    ],
    ids=["neg", "abs", "*", "//"],
)
def test_scalar_signed_integer_overflow(dtype, operation):
    # The minimum signed integer can "overflow" for some additional operations
    st = np.dtype(dtype).type
    min = st(np.iinfo(dtype).min)
    neg_1 = st(-1)

    with pytest.warns(RuntimeWarning, match="overflow encountered"):
        operation(min, neg_1)


@pytest.mark.xfail(reason="pytorch does not warn on overflow")
@pytest.mark.parametrize("dtype", np.typecodes["UnsignedInteger"])
def test_scalar_unsigned_integer_overflow(dtype):
    val = np.dtype(dtype).type(8)
    with pytest.warns(RuntimeWarning, match="overflow encountered"):
        -val

    zero = np.dtype(dtype).type(0)
    -zero  # does not warn


@pytest.mark.xfail(reason="pytorch raises RuntimeError on division by zero")
@pytest.mark.parametrize("dtype", np.typecodes["AllInteger"])
@pytest.mark.parametrize(
    "operation",
    [
        lambda val, zero: val // zero,
        lambda val, zero: val % zero,
    ],
    ids=["//", "%"],
)
def test_scalar_integer_operation_divbyzero(dtype, operation):
    st = np.dtype(dtype).type
    val = st(100)
    zero = st(0)

    with pytest.warns(RuntimeWarning, match="divide by zero"):
        operation(val, zero)


ops_with_names = [
    ("__lt__", "__gt__", operator.lt, True),
    ("__le__", "__ge__", operator.le, True),
    ("__eq__", "__eq__", operator.eq, True),
    # Note __op__ and __rop__ may be identical here:
    ("__ne__", "__ne__", operator.ne, True),
    ("__gt__", "__lt__", operator.gt, True),
    ("__ge__", "__le__", operator.ge, True),
    ("__floordiv__", "__rfloordiv__", operator.floordiv, False),
    ("__truediv__", "__rtruediv__", operator.truediv, False),
    ("__add__", "__radd__", operator.add, False),
    ("__mod__", "__rmod__", operator.mod, False),
    ("__mul__", "__rmul__", operator.mul, False),
    ("__pow__", "__rpow__", operator.pow, False),
    ("__sub__", "__rsub__", operator.sub, False),
]


@pytest.mark.skip(reason="We do not support subclassing scalars.")
@pytest.mark.parametrize(["__op__", "__rop__", "op", "cmp"], ops_with_names)
@pytest.mark.parametrize("sctype", [np.float32, np.float64])
def test_subclass_deferral(sctype, __op__, __rop__, op, cmp):
    """
    This test covers scalar subclass deferral.  Note that this is exceedingly
    complicated, especially since it tends to fall back to the array paths and
    these additionally add the "array priority" mechanism.

    The behaviour was modified subtly in 1.22 (to make it closer to how Python
    scalars work).  Due to its complexity and the fact that subclassing NumPy
    scalars is probably a bad idea to begin with.  There is probably room
    for adjustments here.
    """

    class myf_simple1(sctype):
        pass

    class myf_simple2(sctype):
        pass

    def op_func(self, other):
        return __op__

    def rop_func(self, other):
        return __rop__

    myf_op = type("myf_op", (sctype,), {__op__: op_func, __rop__: rop_func})

    # inheritance has to override, or this is correctly lost:
    res = op(myf_simple1(1), myf_simple2(2))
    assert type(res) == sctype or type(res) == np.bool_
    assert op(myf_simple1(1), myf_simple2(2)) == op(1, 2)  # inherited

    # Two independent subclasses do not really define an order.  This could
    # be attempted, but we do not since Python's `int` does neither:
    assert op(myf_op(1), myf_simple1(2)) == __op__
    assert op(myf_simple1(1), myf_op(2)) == op(1, 2)  # inherited


@pytest.mark.skip(reason="We do not support subclassing scalars.")
@pytest.mark.parametrize(["__op__", "__rop__", "op", "cmp"], ops_with_names)
@pytest.mark.parametrize("subtype", [float, int, complex, np.float16])
# @np._no_nep50_warning()
def test_pyscalar_subclasses(subtype, __op__, __rop__, op, cmp):
    def op_func(self, other):
        return __op__

    def rop_func(self, other):
        return __rop__

    # Check that deferring is indicated using `__array_ufunc__`:
    myt = type(
        "myt", (subtype,), {__op__: op_func, __rop__: rop_func, "__array_ufunc__": None}
    )

    # Just like normally, we should never presume we can modify the float.
    assert op(myt(1), np.float64(2)) == __op__
    assert op(np.float64(1), myt(2)) == __rop__

    if op in {operator.mod, operator.floordiv} and subtype == complex:
        return  # module is not support for complex.  Do not test.

    if __rop__ == __op__:
        return

    # When no deferring is indicated, subclasses are handled normally.
    myt = type("myt", (subtype,), {__rop__: rop_func})

    # Check for float32, as a float subclass float64 may behave differently
    res = op(myt(1), np.float16(2))
    expected = op(subtype(1), np.float16(2))
    assert res == expected
    assert type(res) == type(expected)
    res = op(np.float32(2), myt(1))
    expected = op(np.float32(2), subtype(1))
    assert res == expected
    assert type(res) == type(expected)
