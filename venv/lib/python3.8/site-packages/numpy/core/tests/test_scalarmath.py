import sys
import warnings
import itertools
import operator
import platform
import pytest

import numpy as np
from numpy.testing import (
    assert_, assert_equal, assert_raises, assert_almost_equal,
    assert_array_equal, IS_PYPY, suppress_warnings, _gen_alignment_data,
    assert_warns, assert_raises_regex,
    )

types = [np.bool_, np.byte, np.ubyte, np.short, np.ushort, np.intc, np.uintc,
         np.int_, np.uint, np.longlong, np.ulonglong,
         np.single, np.double, np.longdouble, np.csingle,
         np.cdouble, np.clongdouble]

floating_types = np.floating.__subclasses__()
complex_floating_types = np.complexfloating.__subclasses__()


# This compares scalarmath against ufuncs.

class TestTypes:
    def test_types(self):
        for atype in types:
            a = atype(1)
            assert_(a == 1, "error with %r: got %r" % (atype, a))

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
                assert_equal(c_scalar.dtype, c_array.dtype,
                           "error with types (%d/'%c' + %d/'%c')" %
                            (k, np.dtype(atype).char, l, np.dtype(btype).char))

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


class TestBaseMath:
    def test_blocked(self):
        # test alignments offsets for simd instructions
        # alignments for vz + 2 * (vs - 1) + 1
        for dt, sz in [(np.float32, 11), (np.float64, 7), (np.int32, 11)]:
            for out, inp1, inp2, msg in _gen_alignment_data(dtype=dt,
                                                            type='binary',
                                                            max_size=sz):
                exp1 = np.ones_like(inp1)
                inp1[...] = np.ones_like(inp1)
                inp2[...] = np.zeros_like(inp2)
                assert_almost_equal(np.add(inp1, inp2), exp1, err_msg=msg)
                assert_almost_equal(np.add(inp1, 2), exp1 + 2, err_msg=msg)
                assert_almost_equal(np.add(1, inp2), exp1, err_msg=msg)

                np.add(inp1, inp2, out=out)
                assert_almost_equal(out, exp1, err_msg=msg)

                inp2[...] += np.arange(inp2.size, dtype=dt) + 1
                assert_almost_equal(np.square(inp2),
                                    np.multiply(inp2, inp2),  err_msg=msg)
                # skip true divide for ints
                if dt != np.int32:
                    assert_almost_equal(np.reciprocal(inp2),
                                        np.divide(1, inp2),  err_msg=msg)

                inp1[...] = np.ones_like(inp1)
                np.add(inp1, 2, out=out)
                assert_almost_equal(out, exp1 + 2, err_msg=msg)
                inp2[...] = np.ones_like(inp2)
                np.add(2, inp2, out=out)
                assert_almost_equal(out, exp1 + 2, err_msg=msg)

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
            b = a ** 4
            assert_(b == 81, "error with %r: got %r" % (t, b))

    def test_large_types(self):
        for t in [np.int32, np.int64, np.float32, np.float64, np.longdouble]:
            a = t(51)
            b = a ** 4
            msg = "error with %r: got %r" % (t, b)
            if np.issubdtype(t, np.integer):
                assert_(b == 6765201, msg)
            else:
                assert_almost_equal(b, 6765201, err_msg=msg)

    def test_integers_to_negative_integer_power(self):
        # Note that the combination of uint64 with a signed integer
        # has common type np.float64. The other combinations should all
        # raise a ValueError for integer ** negative integer.
        exp = [np.array(-1, dt)[()] for dt in 'bhilq']

        # 1 ** -1 possible special case
        base = [np.array(1, dt)[()] for dt in 'bhilqBHILQ']
        for i1, i2 in itertools.product(base, exp):
            if i1.dtype != np.uint64:
                assert_raises(ValueError, operator.pow, i1, i2)
            else:
                res = operator.pow(i1, i2)
                assert_(res.dtype.type is np.float64)
                assert_almost_equal(res, 1.)

        # -1 ** -1 possible special case
        base = [np.array(-1, dt)[()] for dt in 'bhilq']
        for i1, i2 in itertools.product(base, exp):
            if i1.dtype != np.uint64:
                assert_raises(ValueError, operator.pow, i1, i2)
            else:
                res = operator.pow(i1, i2)
                assert_(res.dtype.type is np.float64)
                assert_almost_equal(res, -1.)

        # 2 ** -1 perhaps generic
        base = [np.array(2, dt)[()] for dt in 'bhilqBHILQ']
        for i1, i2 in itertools.product(base, exp):
            if i1.dtype != np.uint64:
                assert_raises(ValueError, operator.pow, i1, i2)
            else:
                res = operator.pow(i1, i2)
                assert_(res.dtype.type is np.float64)
                assert_almost_equal(res, .5)

    def test_mixed_types(self):
        typelist = [np.int8, np.int16, np.float16,
                    np.float32, np.float64, np.int8,
                    np.int16, np.int32, np.int64]
        for t1 in typelist:
            for t2 in typelist:
                a = t1(3)
                b = t2(2)
                result = a**b
                msg = ("error with %r and %r:"
                       "got %r, expected %r") % (t1, t2, result, 9)
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
    if dt in np.typecodes['UnsignedInteger']:
        return (+1,)
    else:
        return (+1, -1)


class TestModulus:

    def test_modulus_basic(self):
        dt = np.typecodes['AllInteger'] + np.typecodes['Float']
        for op in [floordiv_and_mod, divmod]:
            for dt1, dt2 in itertools.product(dt, dt):
                for sg1, sg2 in itertools.product(_signs(dt1), _signs(dt2)):
                    fmt = 'op: %s, dt1: %s, dt2: %s, sg1: %s, sg2: %s'
                    msg = fmt % (op.__name__, dt1, dt2, sg1, sg2)
                    a = np.array(sg1*71, dtype=dt1)[()]
                    b = np.array(sg2*19, dtype=dt2)[()]
                    div, rem = op(a, b)
                    assert_equal(div*b + rem, a, err_msg=msg)
                    if sg2 == -1:
                        assert_(b < rem <= 0, msg)
                    else:
                        assert_(b > rem >= 0, msg)

    def test_float_modulus_exact(self):
        # test that float results are exact for small integers. This also
        # holds for the same integers scaled by powers of two.
        nlst = list(range(-127, 0))
        plst = list(range(1, 128))
        dividend = nlst + [0] + plst
        divisor = nlst + plst
        arg = list(itertools.product(dividend, divisor))
        tgt = list(divmod(*t) for t in arg)

        a, b = np.array(arg, dtype=int).T
        # convert exact integer results from Python to float so that
        # signed zero can be used, it is checked.
        tgtdiv, tgtrem = np.array(tgt, dtype=float).T
        tgtdiv = np.where((tgtdiv == 0.0) & ((b < 0) ^ (a < 0)), -0.0, tgtdiv)
        tgtrem = np.where((tgtrem == 0.0) & (b < 0), -0.0, tgtrem)

        for op in [floordiv_and_mod, divmod]:
            for dt in np.typecodes['Float']:
                msg = 'op: %s, dtype: %s' % (op.__name__, dt)
                fa = a.astype(dt)
                fb = b.astype(dt)
                # use list comprehension so a_ and b_ are scalars
                div, rem = zip(*[op(a_, b_) for  a_, b_ in zip(fa, fb)])
                assert_equal(div, tgtdiv, err_msg=msg)
                assert_equal(rem, tgtrem, err_msg=msg)

    def test_float_modulus_roundoff(self):
        # gh-6127
        dt = np.typecodes['Float']
        for op in [floordiv_and_mod, divmod]:
            for dt1, dt2 in itertools.product(dt, dt):
                for sg1, sg2 in itertools.product((+1, -1), (+1, -1)):
                    fmt = 'op: %s, dt1: %s, dt2: %s, sg1: %s, sg2: %s'
                    msg = fmt % (op.__name__, dt1, dt2, sg1, sg2)
                    a = np.array(sg1*78*6e-8, dtype=dt1)[()]
                    b = np.array(sg2*6e-8, dtype=dt2)[()]
                    div, rem = op(a, b)
                    # Equal assertion should hold when fmod is used
                    assert_equal(div*b + rem, a, err_msg=msg)
                    if sg2 == -1:
                        assert_(b < rem <= 0, msg)
                    else:
                        assert_(b > rem >= 0, msg)

    def test_float_modulus_corner_cases(self):
        # Check remainder magnitude.
        for dt in np.typecodes['Float']:
            b = np.array(1.0, dtype=dt)
            a = np.nextafter(np.array(0.0, dtype=dt), -b)
            rem = operator.mod(a, b)
            assert_(rem <= b, 'dt: %s' % dt)
            rem = operator.mod(-a, -b)
            assert_(rem >= -b, 'dt: %s' % dt)

        # Check nans, inf
        with suppress_warnings() as sup:
            sup.filter(RuntimeWarning, "invalid value encountered in remainder")
            for dt in np.typecodes['Float']:
                fone = np.array(1.0, dtype=dt)
                fzer = np.array(0.0, dtype=dt)
                finf = np.array(np.inf, dtype=dt)
                fnan = np.array(np.nan, dtype=dt)
                rem = operator.mod(fone, fzer)
                assert_(np.isnan(rem), 'dt: %s' % dt)
                # MSVC 2008 returns NaN here, so disable the check.
                #rem = operator.mod(fone, finf)
                #assert_(rem == fone, 'dt: %s' % dt)
                rem = operator.mod(fone, fnan)
                assert_(np.isnan(rem), 'dt: %s' % dt)
                rem = operator.mod(finf, fone)
                assert_(np.isnan(rem), 'dt: %s' % dt)

    def test_inplace_floordiv_handling(self):
        # issue gh-12927
        # this only applies to in-place floordiv //=, because the output type
        # promotes to float which does not fit
        a = np.array([1, 2], np.int64)
        b = np.array([1, 2], np.uint64)
        pattern = 'could not be coerced to provided output parameter'
        with assert_raises_regex(TypeError, pattern):
            a //= b


class TestComplexDivision:
    def test_zero_division(self):
        with np.errstate(all="ignore"):
            for t in [np.complex64, np.complex128]:
                a = t(0.0)
                b = t(1.0)
                assert_(np.isinf(b/a))
                b = t(complex(np.inf, np.inf))
                assert_(np.isinf(b/a))
                b = t(complex(np.inf, np.nan))
                assert_(np.isinf(b/a))
                b = t(complex(np.nan, np.inf))
                assert_(np.isinf(b/a))
                b = t(complex(np.nan, np.nan))
                assert_(np.isnan(b/a))
                b = t(0.)
                assert_(np.isnan(b/a))

    def test_signed_zeros(self):
        with np.errstate(all="ignore"):
            for t in [np.complex64, np.complex128]:
                # tupled (numerator, denominator, expected)
                # for testing as expected == numerator/denominator
                data = (
                    (( 0.0,-1.0), ( 0.0, 1.0), (-1.0,-0.0)),
                    (( 0.0,-1.0), ( 0.0,-1.0), ( 1.0,-0.0)),
                    (( 0.0,-1.0), (-0.0,-1.0), ( 1.0, 0.0)),
                    (( 0.0,-1.0), (-0.0, 1.0), (-1.0, 0.0)),
                    (( 0.0, 1.0), ( 0.0,-1.0), (-1.0, 0.0)),
                    (( 0.0,-1.0), ( 0.0,-1.0), ( 1.0,-0.0)),
                    ((-0.0,-1.0), ( 0.0,-1.0), ( 1.0,-0.0)),
                    ((-0.0, 1.0), ( 0.0,-1.0), (-1.0,-0.0))
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
                data.append((( 2.0, 1.0), ( 2.0, 1.0), (1.0, 0.0)))

                # trigger branch: real(fabs(denom)) > imag(fabs(denom))
                # followed by if condition as both are == 0
                # is performed in test_zero_division(), so this is skipped

                # trigger else if branch: real(fabs(denom)) < imag(fabs(denom))
                data.append((( 1.0, 2.0), ( 1.0, 2.0), (1.0, 0.0)))

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
        l = [1e6, 1e12, 1e18, -1e6, -1e12, -1e18]
        li = [10**6, 10**12, 10**18, -10**6, -10**12, -10**18]
        for T in [None, np.float64, np.int64]:
            a = np.array(l, dtype=T)
            assert_equal([int(_m) for _m in a], li)

        a = np.array(l[:3], dtype=np.uint64)
        assert_equal([int(_m) for _m in a], li[:3])

    def test_iinfo_long_values(self):
        for code in 'bBhH':
            res = np.array(np.iinfo(code).max + 1, dtype=code)
            tgt = np.iinfo(code).min
            assert_(res == tgt)

        for code in np.typecodes['AllInteger']:
            res = np.array(np.iinfo(code).max, dtype=code)
            tgt = np.iinfo(code).max
            assert_(res == tgt)

        for code in np.typecodes['AllInteger']:
            res = np.typeDict[code](np.iinfo(code).max)
            tgt = np.iinfo(code).max
            assert_(res == tgt)

    def test_int_raise_behaviour(self):
        def overflow_error_func(dtype):
            np.typeDict[dtype](np.iinfo(dtype).max + 1)

        for code in 'lLqQ':
            assert_raises(OverflowError, overflow_error_func, code)

    def test_int_from_infinite_longdouble(self):
        # gh-627
        x = np.longdouble(np.inf)
        assert_raises(OverflowError, int, x)
        with suppress_warnings() as sup:
            sup.record(np.ComplexWarning)
            x = np.clongdouble(np.inf)
            assert_raises(OverflowError, int, x)
            assert_equal(len(sup.log), 1)

    @pytest.mark.skipif(not IS_PYPY, reason="Test is PyPy only (gh-9972)")
    def test_int_from_infinite_longdouble___int__(self):
        x = np.longdouble(np.inf)
        assert_raises(OverflowError, x.__int__)
        with suppress_warnings() as sup:
            sup.record(np.ComplexWarning)
            x = np.clongdouble(np.inf)
            assert_raises(OverflowError, x.__int__)
            assert_equal(len(sup.log), 1)

    @pytest.mark.skipif(np.finfo(np.double) == np.finfo(np.longdouble),
                        reason="long double is same as double")
    @pytest.mark.skipif(platform.machine().startswith("ppc"),
                        reason="IBM double double")
    def test_int_from_huge_longdouble(self):
        # Produce a longdouble that would overflow a double,
        # use exponent that avoids bug in Darwin pow function.
        exp = np.finfo(np.double).maxexp - 1
        huge_ld = 2 * 1234 * np.longdouble(2) ** exp
        huge_i = 2 * 1234 * 2 ** exp
        assert_(huge_ld != np.inf)
        assert_equal(int(huge_ld), huge_i)

    def test_int_from_longdouble(self):
        x = np.longdouble(1.5)
        assert_equal(int(x), 1)
        x = np.longdouble(-10.5)
        assert_equal(int(x), -10)

    def test_numpy_scalar_relational_operators(self):
        # All integer
        for dt1 in np.typecodes['AllInteger']:
            assert_(1 > np.array(0, dtype=dt1)[()], "type %s failed" % (dt1,))
            assert_(not 1 < np.array(0, dtype=dt1)[()], "type %s failed" % (dt1,))

            for dt2 in np.typecodes['AllInteger']:
                assert_(np.array(1, dtype=dt1)[()] > np.array(0, dtype=dt2)[()],
                        "type %s and %s failed" % (dt1, dt2))
                assert_(not np.array(1, dtype=dt1)[()] < np.array(0, dtype=dt2)[()],
                        "type %s and %s failed" % (dt1, dt2))

        #Unsigned integers
        for dt1 in 'BHILQP':
            assert_(-1 < np.array(1, dtype=dt1)[()], "type %s failed" % (dt1,))
            assert_(not -1 > np.array(1, dtype=dt1)[()], "type %s failed" % (dt1,))
            assert_(-1 != np.array(1, dtype=dt1)[()], "type %s failed" % (dt1,))

            #unsigned vs signed
            for dt2 in 'bhilqp':
                assert_(np.array(1, dtype=dt1)[()] > np.array(-1, dtype=dt2)[()],
                        "type %s and %s failed" % (dt1, dt2))
                assert_(not np.array(1, dtype=dt1)[()] < np.array(-1, dtype=dt2)[()],
                        "type %s and %s failed" % (dt1, dt2))
                assert_(np.array(1, dtype=dt1)[()] != np.array(-1, dtype=dt2)[()],
                        "type %s and %s failed" % (dt1, dt2))

        #Signed integers and floats
        for dt1 in 'bhlqp' + np.typecodes['Float']:
            assert_(1 > np.array(-1, dtype=dt1)[()], "type %s failed" % (dt1,))
            assert_(not 1 < np.array(-1, dtype=dt1)[()], "type %s failed" % (dt1,))
            assert_(-1 == np.array(-1, dtype=dt1)[()], "type %s failed" % (dt1,))

            for dt2 in 'bhlqp' + np.typecodes['Float']:
                assert_(np.array(1, dtype=dt1)[()] > np.array(-1, dtype=dt2)[()],
                        "type %s and %s failed" % (dt1, dt2))
                assert_(not np.array(1, dtype=dt1)[()] < np.array(-1, dtype=dt2)[()],
                        "type %s and %s failed" % (dt1, dt2))
                assert_(np.array(-1, dtype=dt1)[()] == np.array(-1, dtype=dt2)[()],
                        "type %s and %s failed" % (dt1, dt2))

    def test_scalar_comparison_to_none(self):
        # Scalars should just return False and not give a warnings.
        # The comparisons are flagged by pep8, ignore that.
        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings('always', '', FutureWarning)
            assert_(not np.float32(1) == None)
            assert_(not np.str_('test') == None)
            # This is dubious (see below):
            assert_(not np.datetime64('NaT') == None)

            assert_(np.float32(1) != None)
            assert_(np.str_('test') != None)
            # This is dubious (see below):
            assert_(np.datetime64('NaT') != None)
        assert_(len(w) == 0)

        # For documentation purposes, this is why the datetime is dubious.
        # At the time of deprecation this was no behaviour change, but
        # it has to be considered when the deprecations are done.
        assert_(np.equal(np.datetime64('NaT'), None))


#class TestRepr:
#    def test_repr(self):
#        for t in types:
#            val = t(1197346475.0137341)
#            val_repr = repr(val)
#            val2 = eval(val_repr)
#            assert_equal( val, val2 )


class TestRepr:
    def _test_type_repr(self, t):
        finfo = np.finfo(t)
        last_fraction_bit_idx = finfo.nexp + finfo.nmant
        last_exponent_bit_idx = finfo.nexp
        storage_bytes = np.dtype(t).itemsize*8
        # could add some more types to the list below
        for which in ['small denorm', 'small norm']:
            # Values from https://en.wikipedia.org/wiki/IEEE_754
            constr = np.array([0x00]*storage_bytes, dtype=np.uint8)
            if which == 'small denorm':
                byte = last_fraction_bit_idx // 8
                bytebit = 7-(last_fraction_bit_idx % 8)
                constr[byte] = 1 << bytebit
            elif which == 'small norm':
                byte = last_exponent_bit_idx // 8
                bytebit = 7-(last_exponent_bit_idx % 8)
                constr[byte] = 1 << bytebit
            else:
                raise ValueError('hmm')
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


if not IS_PYPY:
    # sys.getsizeof() is not valid on PyPy
    class TestSizeOf:

        def test_equal_nbytes(self):
            for type in types:
                x = type(0)
                assert_(sys.getsizeof(x) > x.nbytes)

        def test_error(self):
            d = np.float32()
            assert_raises(TypeError, d.__sizeof__, "a")


class TestMultiply:
    def test_seq_repeat(self):
        # Test that basic sequences get repeated when multiplied with
        # numpy integers. And errors are raised when multiplied with others.
        # Some of this behaviour may be controversial and could be open for
        # change.
        accepted_types = set(np.typecodes["AllInteger"])
        deprecated_types = {'?'}
        forbidden_types = (
            set(np.typecodes["All"]) - accepted_types - deprecated_types)
        forbidden_types -= {'V'}  # can't default-construct void scalars

        for seq_type in (list, tuple):
            seq = seq_type([1, 2, 3])
            for numpy_type in accepted_types:
                i = np.dtype(numpy_type).type(2)
                assert_equal(seq * i, seq * int(i))
                assert_equal(i * seq, int(i) * seq)

            for numpy_type in deprecated_types:
                i = np.dtype(numpy_type).type()
                assert_equal(
                    assert_warns(DeprecationWarning, operator.mul, seq, i),
                    seq * int(i))
                assert_equal(
                    assert_warns(DeprecationWarning, operator.mul, i, seq),
                    int(i) * seq)

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
            assert_array_equal(arr_like * np.float32(3.), np.full(3, 3.))
            assert_array_equal(np.float32(3.) * arr_like, np.full(3, 3.))
            assert_array_equal(arr_like * np.int_(3), np.full(3, 3))
            assert_array_equal(np.int_(3) * arr_like, np.full(3, 3))


class TestNegative:
    def test_exceptions(self):
        a = np.ones((), dtype=np.bool_)[()]
        assert_raises(TypeError, operator.neg, a)

    def test_result(self):
        types = np.typecodes['AllInteger'] + np.typecodes['AllFloat']
        with suppress_warnings() as sup:
            sup.filter(RuntimeWarning)
            for dt in types:
                a = np.ones((), dtype=dt)[()]
                assert_equal(operator.neg(a) + a, 0)


class TestSubtract:
    def test_exceptions(self):
        a = np.ones((), dtype=np.bool_)[()]
        assert_raises(TypeError, operator.sub, a, a)

    def test_result(self):
        types = np.typecodes['AllInteger'] + np.typecodes['AllFloat']
        with suppress_warnings() as sup:
            sup.filter(RuntimeWarning)
            for dt in types:
                a = np.ones((), dtype=dt)[()]
                assert_equal(operator.sub(a, a), 0)


class TestAbs:
    def _test_abs_func(self, absfunc):
        for tp in floating_types + complex_floating_types:
            x = tp(-1.5)
            assert_equal(absfunc(x), 1.5)
            x = tp(0.0)
            res = absfunc(x)
            # assert_equal() checks zero signedness
            assert_equal(res, 0.0)
            x = tp(-0.0)
            res = absfunc(x)
            assert_equal(res, 0.0)

            x = tp(np.finfo(tp).max)
            assert_equal(absfunc(x), x.real)

            x = tp(np.finfo(tp).tiny)
            assert_equal(absfunc(x), x.real)

            x = tp(np.finfo(tp).min)
            assert_equal(absfunc(x), -x.real)

    def test_builtin_abs(self):
        self._test_abs_func(abs)

    def test_numpy_abs(self):
        self._test_abs_func(np.abs)


class TestBitShifts:

    @pytest.mark.parametrize('type_code', np.typecodes['AllInteger'])
    @pytest.mark.parametrize('op',
        [operator.rshift, operator.lshift], ids=['>>', '<<'])
    def test_shift_all_bits(self, type_code, op):
        """ Shifts where the shift amount is the width of the type or wider """
        # gh-2449
        dt = np.dtype(type_code)
        nbits = dt.itemsize * 8
        for val in [5, -5]:
            for shift in [nbits, nbits + 4]:
                val_scl = dt.type(val)
                shift_scl = dt.type(shift)
                res_scl = op(val_scl, shift_scl)
                if val_scl < 0 and op is operator.rshift:
                    # sign bit is preserved
                    assert_equal(res_scl, -1)
                else:
                    assert_equal(res_scl, 0)

                # Result on scalars should be the same as on arrays
                val_arr = np.array([val]*32, dtype=dt)
                shift_arr = np.array([shift]*32, dtype=dt)
                res_arr = op(val_arr, shift_arr)
                assert_equal(res_arr, res_scl)
