"""
This file adds basic tests to test the NEP 50 style promotion compatibility
mode.  Most of these test are likely to be simply deleted again once NEP 50
is adopted in the main test suite.  A few may be moved elsewhere.
"""

import operator
import threading
import warnings

import numpy as np

import pytest
import hypothesis
from hypothesis import strategies

from numpy.testing import assert_array_equal, IS_WASM


@pytest.fixture(scope="module", autouse=True)
def _weak_promotion_enabled():
    state = np._get_promotion_state()
    np._set_promotion_state("weak_and_warn")
    yield
    np._set_promotion_state(state)


@pytest.mark.skipif(IS_WASM, reason="wasm doesn't have support for fp errors")
def test_nep50_examples():
    with pytest.warns(UserWarning, match="result dtype changed"):
        res = np.uint8(1) + 2
    assert res.dtype == np.uint8

    with pytest.warns(UserWarning, match="result dtype changed"):
        res = np.array([1], np.uint8) + np.int64(1)
    assert res.dtype == np.int64

    with pytest.warns(UserWarning, match="result dtype changed"):
        res = np.array([1], np.uint8) + np.array(1, dtype=np.int64)
    assert res.dtype == np.int64

    with pytest.warns(UserWarning, match="result dtype changed"):
        # Note: For "weak_and_warn" promotion state the overflow warning is
        #       unfortunately not given (because we use the full array path).
        with np.errstate(over="raise"):
            res = np.uint8(100) + 200
    assert res.dtype == np.uint8

    with pytest.warns(Warning) as recwarn:
        res = np.float32(1) + 3e100

    # Check that both warnings were given in the one call:
    warning = str(recwarn.pop(UserWarning).message)
    assert warning.startswith("result dtype changed")
    warning = str(recwarn.pop(RuntimeWarning).message)
    assert warning.startswith("overflow")
    assert len(recwarn) == 0  # no further warnings
    assert np.isinf(res)
    assert res.dtype == np.float32

    # Changes, but we don't warn for it (too noisy)
    res = np.array([0.1], np.float32) == np.float64(0.1)
    assert res[0] == False

    # Additional test, since the above silences the warning:
    with pytest.warns(UserWarning, match="result dtype changed"):
        res = np.array([0.1], np.float32) + np.float64(0.1)
    assert res.dtype == np.float64

    with pytest.warns(UserWarning, match="result dtype changed"):
        res = np.array([1.], np.float32) + np.int64(3)
    assert res.dtype == np.float64


@pytest.mark.parametrize("dtype", np.typecodes["AllInteger"])
def test_nep50_weak_integers(dtype):
    # Avoids warning (different code path for scalars)
    np._set_promotion_state("weak")
    scalar_type = np.dtype(dtype).type

    maxint = int(np.iinfo(dtype).max)

    with np.errstate(over="warn"):
        with pytest.warns(RuntimeWarning):
            res = scalar_type(100) + maxint
    assert res.dtype == dtype

    # Array operations are not expected to warn, but should give the same
    # result dtype.
    res = np.array(100, dtype=dtype) + maxint
    assert res.dtype == dtype


@pytest.mark.parametrize("dtype", np.typecodes["AllFloat"])
def test_nep50_weak_integers_with_inexact(dtype):
    # Avoids warning (different code path for scalars)
    np._set_promotion_state("weak")
    scalar_type = np.dtype(dtype).type

    too_big_int = int(np.finfo(dtype).max) * 2

    if dtype in "dDG":
        # These dtypes currently convert to Python float internally, which
        # raises an OverflowError, while the other dtypes overflow to inf.
        # NOTE: It may make sense to normalize the behavior!
        with pytest.raises(OverflowError):
            scalar_type(1) + too_big_int

        with pytest.raises(OverflowError):
            np.array(1, dtype=dtype) + too_big_int
    else:
        # NumPy uses (or used) `int -> string -> longdouble` for the
        # conversion.  But Python may refuse `str(int)` for huge ints.
        # In that case, RuntimeWarning would be correct, but conversion
        # fails earlier (seems to happen on 32bit linux, possibly only debug).
        if dtype in "gG":
            try:
                str(too_big_int)
            except ValueError:
                pytest.skip("`huge_int -> string -> longdouble` failed")

        # Otherwise, we overflow to infinity:
        with pytest.warns(RuntimeWarning):
            res = scalar_type(1) + too_big_int
        assert res.dtype == dtype
        assert res == np.inf

        with pytest.warns(RuntimeWarning):
            # We force the dtype here, since windows may otherwise pick the
            # double instead of the longdouble loop.  That leads to slightly
            # different results (conversion of the int fails as above).
            res = np.add(np.array(1, dtype=dtype), too_big_int, dtype=dtype)
        assert res.dtype == dtype
        assert res == np.inf


@pytest.mark.parametrize("op", [operator.add, operator.pow])
def test_weak_promotion_scalar_path(op):
    # Some additional paths exercising the weak scalars.
    np._set_promotion_state("weak")

    # Integer path:
    res = op(np.uint8(3), 5)
    assert res == op(3, 5)
    assert res.dtype == np.uint8 or res.dtype == bool

    with pytest.raises(OverflowError):
        op(np.uint8(3), 1000)

    # Float path:
    res = op(np.float32(3), 5.)
    assert res == op(3., 5.)
    assert res.dtype == np.float32 or res.dtype == bool


def test_nep50_complex_promotion():
    np._set_promotion_state("weak")

    with pytest.warns(RuntimeWarning, match=".*overflow"):
        res = np.complex64(3) + complex(2**300)

    assert type(res) == np.complex64


def test_nep50_integer_conversion_errors():
    # Do not worry about warnings here (auto-fixture will reset).
    np._set_promotion_state("weak")
    # Implementation for error paths is mostly missing (as of writing)
    with pytest.raises(OverflowError, match=".*uint8"):
        np.array([1], np.uint8) + 300

    with pytest.raises(OverflowError, match=".*uint8"):
        np.uint8(1) + 300

    # Error message depends on platform (maybe unsigned int or unsigned long)
    with pytest.raises(OverflowError,
            match="Python integer -1 out of bounds for uint8"):
        np.uint8(1) + -1


def test_nep50_integer_regression():
    # Test the old integer promotion rules.  When the integer is too large,
    # we need to keep using the old-style promotion.
    np._set_promotion_state("legacy")
    arr = np.array(1)
    assert (arr + 2**63).dtype == np.float64
    assert (arr[()] + 2**63).dtype == np.float64


def test_nep50_with_axisconcatenator():
    # I promised that this will be an error in the future in the 1.25
    # release notes;  test this (NEP 50 opt-in makes the deprecation an error).
    np._set_promotion_state("weak")

    with pytest.raises(OverflowError):
        np.r_[np.arange(5, dtype=np.int8), 255]


@pytest.mark.parametrize("ufunc", [np.add, np.power])
@pytest.mark.parametrize("state", ["weak", "weak_and_warn"])
def test_nep50_huge_integers(ufunc, state):
    # Very large integers are complicated, because they go to uint64 or
    # object dtype.  This tests covers a few possible paths (some of which
    # cannot give the NEP 50 warnings).
    np._set_promotion_state(state)

    with pytest.raises(OverflowError):
        ufunc(np.int64(0), 2**63)  # 2**63 too large for int64

    if state == "weak_and_warn":
        with pytest.warns(UserWarning,
                match="result dtype changed.*float64.*uint64"):
            with pytest.raises(OverflowError):
                ufunc(np.uint64(0), 2**64)
    else:
        with pytest.raises(OverflowError):
            ufunc(np.uint64(0), 2**64)  # 2**64 cannot be represented by uint64

    # However, 2**63 can be represented by the uint64 (and that is used):
    if state == "weak_and_warn":
        with pytest.warns(UserWarning,
                match="result dtype changed.*float64.*uint64"):
            res = ufunc(np.uint64(1), 2**63)
    else:
        res = ufunc(np.uint64(1), 2**63)

    assert res.dtype == np.uint64
    assert res == ufunc(1, 2**63, dtype=object)

    # The following paths fail to warn correctly about the change:
    with pytest.raises(OverflowError):
        ufunc(np.int64(1), 2**63)  # np.array(2**63) would go to uint

    with pytest.raises(OverflowError):
        ufunc(np.int64(1), 2**100)  # np.array(2**100) would go to object

    # This would go to object and thus a Python float, not a NumPy one:
    res = ufunc(1.0, 2**100)
    assert isinstance(res, np.float64)


def test_nep50_in_concat_and_choose():
    np._set_promotion_state("weak_and_warn")

    with pytest.warns(UserWarning, match="result dtype changed"):
        res = np.concatenate([np.float32(1), 1.], axis=None)
    assert res.dtype == "float32"

    with pytest.warns(UserWarning, match="result dtype changed"):
        res = np.choose(1, [np.float32(1), 1.])
    assert res.dtype == "float32"


@pytest.mark.parametrize("expected,dtypes,optional_dtypes", [
        (np.float32, [np.float32],
            [np.float16, 0.0, np.uint16, np.int16, np.int8, 0]),
        (np.complex64, [np.float32, 0j],
            [np.float16, 0.0, np.uint16, np.int16, np.int8, 0]),
        (np.float32, [np.int16, np.uint16, np.float16],
            [np.int8, np.uint8, np.float32, 0., 0]),
        (np.int32, [np.int16, np.uint16],
            [np.int8, np.uint8, 0, np.bool]),
        ])
@hypothesis.given(data=strategies.data())
def test_expected_promotion(expected, dtypes, optional_dtypes, data):
    np._set_promotion_state("weak")

    # Sample randomly while ensuring "dtypes" is always present:
    optional = data.draw(strategies.lists(
            strategies.sampled_from(dtypes + optional_dtypes)))
    all_dtypes = dtypes + optional
    dtypes_sample = data.draw(strategies.permutations(all_dtypes))

    res = np.result_type(*dtypes_sample)
    assert res == expected


@pytest.mark.parametrize("sctype",
        [np.int8, np.int16, np.int32, np.int64,
         np.uint8, np.uint16, np.uint32, np.uint64])
@pytest.mark.parametrize("other_val",
        [-2*100, -1, 0, 9, 10, 11, 2**63, 2*100])
@pytest.mark.parametrize("comp",
        [operator.eq, operator.ne, operator.le, operator.lt,
         operator.ge, operator.gt])
def test_integer_comparison(sctype, other_val, comp):
    np._set_promotion_state("weak")

    # Test that comparisons with integers (especially out-of-bound) ones
    # works correctly.
    val_obj = 10
    val = sctype(val_obj)
    # Check that the scalar behaves the same as the python int:
    assert comp(10, other_val) == comp(val, other_val)
    assert comp(val, other_val) == comp(10, other_val)
    # Except for the result type:
    assert type(comp(val, other_val)) is np.bool

    # Check that the integer array and object array behave the same:
    val_obj = np.array([10, 10], dtype=object)
    val = val_obj.astype(sctype)
    assert_array_equal(comp(val_obj, other_val), comp(val, other_val))
    assert_array_equal(comp(other_val, val_obj), comp(other_val, val))


@pytest.mark.parametrize("comp",
        [np.equal, np.not_equal, np.less_equal, np.less,
         np.greater_equal, np.greater])
def test_integer_integer_comparison(comp):
    np._set_promotion_state("weak")

    # Test that the NumPy comparison ufuncs work with large Python integers
    assert comp(2**200, -2**200) == comp(2**200, -2**200, dtype=object)


def create_with_scalar(sctype, value):
    return sctype(value)


def create_with_array(sctype, value):
    return np.array([value], dtype=sctype)


@pytest.mark.parametrize("sctype",
        [np.int8, np.int16, np.int32, np.int64,
         np.uint8, np.uint16, np.uint32, np.uint64])
@pytest.mark.parametrize("create", [create_with_scalar, create_with_array])
def test_oob_creation(sctype, create):
    iinfo = np.iinfo(sctype)

    with pytest.raises(OverflowError):
        create(sctype, iinfo.min - 1)

    with pytest.raises(OverflowError):
        create(sctype, iinfo.max + 1)

    with pytest.raises(OverflowError):
        create(sctype, str(iinfo.min - 1))

    with pytest.raises(OverflowError):
        create(sctype, str(iinfo.max + 1))

    assert create(sctype, iinfo.min) == iinfo.min
    assert create(sctype, iinfo.max) == iinfo.max


@pytest.mark.skipif(IS_WASM, reason="wasm doesn't have support for threads")
def test_thread_local_promotion_state():
    b = threading.Barrier(2)

    def legacy_no_warn():
        np._set_promotion_state("legacy")
        b.wait()
        assert np._get_promotion_state() == "legacy"

    def weak_warn():
        np._set_promotion_state("weak")
        b.wait()
        assert np._get_promotion_state() == "weak"

    task1 = threading.Thread(target=legacy_no_warn)
    task2 = threading.Thread(target=weak_warn)

    task1.start()
    task2.start()
    task1.join()
    task2.join()
