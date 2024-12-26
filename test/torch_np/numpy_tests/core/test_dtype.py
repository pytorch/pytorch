# Owner(s): ["module: dynamo"]

import functools
import operator
import pickle
import sys
import types
from itertools import permutations
from typing import Any
from unittest import skipIf as skipif

import pytest
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


skip = functools.partial(skipif, True)

if TEST_WITH_TORCHDYNAMO:
    import numpy as np
    from numpy.testing import assert_, assert_equal
else:
    import torch._numpy as np
    from torch._numpy.testing import assert_, assert_equal

import numpy


def assert_dtype_equal(a, b):
    assert_equal(a, b)
    assert_equal(
        hash(a), hash(b), "two equivalent types do not hash to the same value !"
    )


def assert_dtype_not_equal(a, b):
    assert_(a != b)
    assert_(hash(a) != hash(b), "two different types hash to the same value !")


@instantiate_parametrized_tests
class TestBuiltin(TestCase):
    @parametrize("t", [int, float, complex, np.int32])
    def test_run(self, t):
        """Only test hash runs at all."""
        dt = np.dtype(t)
        hash(dt)

    def test_equivalent_dtype_hashing(self):
        # Make sure equivalent dtypes with different type num hash equal
        intp = np.dtype(np.intp)
        if intp.itemsize == 4:
            left = intp
            right = np.dtype(np.int32)
        else:
            left = intp
            right = np.dtype(np.int64)
        assert_(left == right)
        assert_(hash(left) == hash(right))

    @xfailIfTorchDynamo  # TypeError -> InternalTorchDynamoError
    def test_invalid_types(self):
        # Make sure invalid type strings raise an error

        assert_raises(TypeError, np.dtype, "O3")
        assert_raises(TypeError, np.dtype, "O5")
        assert_raises(TypeError, np.dtype, "O7")
        assert_raises(TypeError, np.dtype, "b3")
        assert_raises(TypeError, np.dtype, "h4")
        assert_raises(TypeError, np.dtype, "I5")
        assert_raises(TypeError, np.dtype, "e3")
        assert_raises(TypeError, np.dtype, "f5")

        if np.dtype("l").itemsize == 8:
            assert_raises(TypeError, np.dtype, "l4")
            assert_raises(TypeError, np.dtype, "L4")
        else:
            assert_raises(TypeError, np.dtype, "l8")
            assert_raises(TypeError, np.dtype, "L8")

    # XXX: what is 'q'? on my 64-bit ubuntu maching it's int64, same as 'l'
    #       if np.dtype('q').itemsize == 8:
    #           assert_raises(TypeError, np.dtype, 'q4')
    #           assert_raises(TypeError, np.dtype, 'Q4')
    #       else:
    #           assert_raises(TypeError, np.dtype, 'q8')
    #           assert_raises(TypeError, np.dtype, 'Q8')

    def test_richcompare_invalid_dtype_equality(self):
        # Make sure objects that cannot be converted to valid
        # dtypes results in False/True when compared to valid dtypes.
        # Here 7 cannot be converted to dtype. No exceptions should be raised

        assert not np.dtype(np.int32) == 7, "dtype richcompare failed for =="
        assert np.dtype(np.int32) != 7, "dtype richcompare failed for !="

    @parametrize("operation", [operator.le, operator.lt, operator.ge, operator.gt])
    def test_richcompare_invalid_dtype_comparison(self, operation):
        # Make sure TypeError is raised for comparison operators
        # for invalid dtypes. Here 7 is an invalid dtype.

        with pytest.raises(TypeError):
            operation(np.dtype(np.int32), 7)

    @skipif(
        numpy.__version__ < "1.24",
        reason="older numpies emit DeprecatioWarnings instead",
    )
    @parametrize(
        "dtype",
        [
            "Bool",
            "Bytes0",
            "Complex32",
            "Complex64",
            "Datetime64",
            "Float16",
            "Float32",
            "Float64",
            "Int8",
            "Int16",
            "Int32",
            "Int64",
            "Object0",
            "Str0",
            "Timedelta64",
            "UInt8",
            "UInt16",
            "Uint32",
            "UInt32",
            "Uint64",
            "UInt64",
            "Void0",
            "Float128",
            "Complex128",
        ],
    )
    def test_numeric_style_types_are_invalid(self, dtype):
        with assert_raises(TypeError):
            np.dtype(dtype)


@skip(reason="dtype attributes not yet implemented")
class TestDtypeAttributeDeletion(TestCase):
    def test_dtype_non_writable_attributes_deletion(self):
        dt = np.dtype(np.double)
        attr = [
            "subdtype",
            "descr",
            "str",
            "name",
            "base",
            "shape",
            "isbuiltin",
            "isnative",
            "isalignedstruct",
            "fields",
            "metadata",
            "hasobject",
        ]

        for s in attr:
            assert_raises(AttributeError, delattr, dt, s)

    def test_dtype_writable_attributes_deletion(self):
        dt = np.dtype(np.double)
        attr = ["names"]
        for s in attr:
            assert_raises(AttributeError, delattr, dt, s)


@instantiate_parametrized_tests
class TestPickling(TestCase):
    def check_pickling(self, dtype):
        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            buf = pickle.dumps(dtype, proto)
            # The dtype pickling itself pickles `np.dtype` if it is pickled
            # as a singleton `dtype` should be stored in the buffer:
            assert b"_DType_reconstruct" not in buf
            assert b"dtype" in buf
            pickled = pickle.loads(buf)
            assert_equal(pickled, dtype)

            # XXX: out dtypes do not have .descr
            #         assert_equal(pickled.descr, dtype.descr)
            #         if dtype.metadata is not None:
            #             assert_equal(pickled.metadata, dtype.metadata)
            # Check the reconstructed dtype is functional

            x = np.zeros(3, dtype=dtype)
            y = np.zeros(3, dtype=pickled)
            assert_equal(x, y)
            assert_equal(x[0], y[0])

    @parametrize("t", [int, float, complex, np.int32, bool])
    def test_builtin(self, t):
        self.check_pickling(np.dtype(t))

    @parametrize(
        "DType",
        [
            subtest(type(np.dtype(t)), name=f"{np.dtype(t).name}_{i}")
            for i, t in enumerate(np.typecodes["All"])
        ]
        + [np.dtype],
    )
    def test_pickle_types(self, DType):
        # Check that DTypes (the classes/types) roundtrip when pickling
        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            roundtrip_DType = pickle.loads(pickle.dumps(DType, proto))
            assert roundtrip_DType is DType


@skip(reason="XXX: value-based promotions, we don't have.")
@instantiate_parametrized_tests
class TestPromotion(TestCase):
    """Test cases related to more complex DType promotions.  Further promotion
    tests are defined in `test_numeric.py`
    """

    @parametrize(
        "other, expected, expected_weak",
        [
            (2**16 - 1, np.complex64, None),
            (2**32 - 1, np.complex128, np.complex64),
            subtest((np.float16(2), np.complex64, None), name="float16_complex64_None"),
            subtest((np.float32(2), np.complex64, None), name="float32_complex64_None"),
            # repeat for complex scalars:
            subtest(
                (np.complex64(2), np.complex64, None), name="complex64_complex64_None"
            ),
        ],
    )
    def test_complex_other_value_based(
        self, weak_promotion, other, expected, expected_weak
    ):
        if weak_promotion and expected_weak is not None:
            expected = expected_weak

        # This would change if we modify the value based promotion
        min_complex = np.dtype(np.complex64)

        res = np.result_type(other, min_complex)
        assert res == expected
        # Check the same for a simple ufunc call that uses the same logic:
        res = np.minimum(other, np.ones(3, dtype=min_complex)).dtype
        assert res == expected

    @parametrize(
        "other, expected",
        [
            (np.bool_, np.complex128),
            (np.int64, np.complex128),
            (np.float16, np.complex64),
            (np.float32, np.complex64),
            (np.float64, np.complex128),
            (np.complex64, np.complex64),
            (np.complex128, np.complex128),
        ],
    )
    def test_complex_scalar_value_based(self, other, expected):
        # This would change if we modify the value based promotion
        complex_scalar = 1j

        res = np.result_type(other, complex_scalar)
        assert res == expected
        # Check the same for a simple ufunc call that uses the same logic:
        res = np.minimum(np.ones(3, dtype=other), complex_scalar).dtype
        assert res == expected

    @parametrize("val", [2, 2**32, 2**63, 2**64, 2 * 100])
    def test_python_integer_promotion(self, val):
        # If we only path scalars (mainly python ones!), the result must take
        # into account that the integer may be considered int32, int64, uint64,
        # or object depending on the input value.  So test those paths!
        expected_dtype = np.result_type(np.array(val).dtype, np.array(0).dtype)
        assert np.result_type(val, 0) == expected_dtype
        # For completeness sake, also check with a NumPy scalar as second arg:
        assert np.result_type(val, np.int8(0)) == expected_dtype

    @parametrize(
        "dtypes, expected",
        [
            # These promotions are not associative/commutative:
            ([np.int16, np.float16], np.float32),
            ([np.int8, np.float16], np.float32),
            ([np.uint8, np.int16, np.float16], np.float32),
            # The following promotions are not ambiguous, but cover code
            # paths of abstract promotion (no particular logic being tested)
            ([1, 1, np.float64], np.float64),
            ([1, 1.0, np.complex128], np.complex128),
            ([1, 1j, np.float64], np.complex128),
            ([1.0, 1.0, np.int64], np.float64),
            ([1.0, 1j, np.float64], np.complex128),
            ([1j, 1j, np.float64], np.complex128),
            ([1, True, np.bool_], np.int_),
        ],
    )
    def test_permutations_do_not_influence_result(self, dtypes, expected):
        # Tests that most permutations do not influence the result.  In the
        # above some uint and int combintations promote to a larger integer
        # type, which would then promote to a larger than necessary float.
        for perm in permutations(dtypes):
            assert np.result_type(*perm) == expected


class TestMisc(TestCase):
    def test_dtypes_are_true(self):
        # test for gh-6294
        assert bool(np.dtype("f8"))
        assert bool(np.dtype("i8"))

    @xpassIfTorchDynamo_np  # (reason="No keyword arg for dtype ctor.")
    def test_keyword_argument(self):
        # test for https://github.com/numpy/numpy/pull/16574#issuecomment-642660971
        assert np.dtype(dtype=np.float64) == np.dtype(np.float64)

    @skipif(sys.version_info >= (3, 9), reason="Requires python 3.9")
    def test_class_getitem_38(self) -> None:
        with pytest.raises(TypeError):
            np.dtype[Any]


class TestFromDTypeAttribute(TestCase):
    def test_simple(self):
        class dt:
            dtype = np.dtype("f8")

        assert np.dtype(dt) == np.float64
        assert np.dtype(dt()) == np.float64

    @skip(
        reason="We simply require the .name attribute, so this "
        "fails with an AttributeError."
    )
    def test_recursion(self):
        class dt:
            pass

        dt.dtype = dt
        with pytest.raises(RecursionError):
            np.dtype(dt)

        dt_instance = dt()
        dt_instance.dtype = dt
        with pytest.raises(RecursionError):
            np.dtype(dt_instance)


@skip(reason="Parameteric dtypes, our stuff is simpler.")
@skipif(sys.version_info < (3, 9), reason="Requires python 3.9")
@instantiate_parametrized_tests
class TestClassGetItem(TestCase):
    def test_dtype(self) -> None:
        alias = np.dtype[Any]
        assert isinstance(alias, types.GenericAlias)
        assert alias.__origin__ is np.dtype

    @parametrize("code", np.typecodes["All"])
    def test_dtype_subclass(self, code: str) -> None:
        cls = type(np.dtype(code))
        alias = cls[Any]
        assert isinstance(alias, types.GenericAlias)
        assert alias.__origin__ is cls

    @parametrize("arg_len", range(4))
    def test_subscript_tuple(self, arg_len: int) -> None:
        arg_tup = (Any,) * arg_len
        if arg_len == 1:
            assert np.dtype[arg_tup]
        else:
            with pytest.raises(TypeError):
                np.dtype[arg_tup]

    def test_subscript_scalar(self) -> None:
        assert np.dtype[Any]


if __name__ == "__main__":
    run_tests()
