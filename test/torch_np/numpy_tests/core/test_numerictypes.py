# Owner(s): ["module: dynamo"]

import functools
import itertools
import sys

from unittest import skipIf as skipif

from pytest import raises as assert_raises

from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    skipIfTorchDynamo,
    TEST_WITH_TORCHDYNAMO,
    TestCase,
    xpassIfTorchDynamo,
)

if TEST_WITH_TORCHDYNAMO:
    import numpy as np
    from numpy.testing import assert_
else:
    import torch._numpy as np
    from torch._numpy.testing import assert_


skip = functools.partial(skipif, True)


@xpassIfTorchDynamo  # (
#    reason="We do not disctinguish between scalar and array types."
#    " Thus, scalars can upcast arrays."
# )
class TestCommonType(TestCase):
    def test_scalar_loses1(self):
        res = np.find_common_type(["f4", "f4", "i2"], ["f8"])
        assert_(res == "f4")

    def test_scalar_loses2(self):
        res = np.find_common_type(["f4", "f4"], ["i8"])
        assert_(res == "f4")

    def test_scalar_wins(self):
        res = np.find_common_type(["f4", "f4", "i2"], ["c8"])
        assert_(res == "c8")

    def test_scalar_wins2(self):
        res = np.find_common_type(["u4", "i4", "i4"], ["f4"])
        assert_(res == "f8")

    def test_scalar_wins3(self):  # doesn't go up to 'f16' on purpose
        res = np.find_common_type(["u8", "i8", "i8"], ["f8"])
        assert_(res == "f8")


class TestIsSubDType(TestCase):
    # scalar types can be promoted into dtypes
    wrappers = [np.dtype, lambda x: x]

    def test_both_abstract(self):
        assert_(np.issubdtype(np.floating, np.inexact))
        assert_(not np.issubdtype(np.inexact, np.floating))

    def test_same(self):
        for cls in (np.float32, np.int32):
            for w1, w2 in itertools.product(self.wrappers, repeat=2):
                assert_(np.issubdtype(w1(cls), w2(cls)))

    def test_subclass(self):
        # note we cannot promote floating to a dtype, as it would turn into a
        # concrete type
        for w in self.wrappers:
            assert_(np.issubdtype(w(np.float32), np.floating))
            assert_(np.issubdtype(w(np.float64), np.floating))

    def test_subclass_backwards(self):
        for w in self.wrappers:
            assert_(not np.issubdtype(np.floating, w(np.float32)))
            assert_(not np.issubdtype(np.floating, w(np.float64)))

    def test_sibling_class(self):
        for w1, w2 in itertools.product(self.wrappers, repeat=2):
            assert_(not np.issubdtype(w1(np.float32), w2(np.float64)))
            assert_(not np.issubdtype(w1(np.float64), w2(np.float32)))

    def test_nondtype_nonscalartype(self):
        # See gh-14619 and gh-9505 which introduced the deprecation to fix
        # this. These tests are directly taken from gh-9505
        assert not np.issubdtype(np.float32, "float64")
        assert not np.issubdtype(np.float32, "f8")
        assert not np.issubdtype(np.int32, "int64")
        # for the following the correct spellings are
        # np.integer, np.floating, or np.complexfloating respectively:
        assert not np.issubdtype(np.int8, int)  # np.int8 is never np.int_
        assert not np.issubdtype(np.float32, float)
        assert not np.issubdtype(np.complex64, complex)
        assert not np.issubdtype(np.float32, "float")
        assert not np.issubdtype(np.float64, "f")

        # Test the same for the correct first datatype and abstract one
        # in the case of int, float, complex:
        assert np.issubdtype(np.float64, "float64")
        assert np.issubdtype(np.float64, "f8")
        assert np.issubdtype(np.int64, "int64")
        assert np.issubdtype(np.int8, np.integer)
        assert np.issubdtype(np.float32, np.floating)
        assert np.issubdtype(np.complex64, np.complexfloating)
        assert np.issubdtype(np.float64, "float")
        assert np.issubdtype(np.float32, "f")


@xpassIfTorchDynamo  # (
#    reason="We do not have (or need) np.core.numerictypes."
#    " Our type aliases are in _dtypes.py."
# )
class TestBitName(TestCase):
    def test_abstract(self):
        assert_raises(ValueError, np.core.numerictypes.bitname, np.floating)


@skip(reason="Docstrings for scalar types, not yet.")
@skipif(
    sys.flags.optimize > 1,
    reason="no docstrings present to inspect when PYTHONOPTIMIZE/Py_OptimizeFlag > 1",
)
class TestDocStrings(TestCase):
    def test_platform_dependent_aliases(self):
        if np.int64 is np.int_:
            assert_("int64" in np.int_.__doc__)
        elif np.int64 is np.longlong:
            assert_("int64" in np.longlong.__doc__)


@instantiate_parametrized_tests
class TestScalarTypeNames(TestCase):
    # gh-9799

    numeric_types = [
        np.byte,
        np.short,
        np.intc,
        np.int_,  # , np.longlong, NB: torch does not properly have longlong
        np.ubyte,
        np.half,
        np.single,
        np.double,
        np.csingle,
        np.cdouble,
    ]

    def test_names_are_unique(self):
        # none of the above may be aliases for each other
        assert len(set(self.numeric_types)) == len(self.numeric_types)

        # names must be unique
        names = [t.__name__ for t in self.numeric_types]
        assert len(set(names)) == len(names)

    @parametrize("t", numeric_types)
    def test_names_reflect_attributes(self, t):
        """Test that names correspond to where the type is under ``np.``"""
        assert getattr(np, t.__name__) is t

    @skipIfTorchDynamo()  # XXX: weird, some names are not OK
    @parametrize("t", numeric_types)
    def test_names_are_undersood_by_dtype(self, t):
        """Test the dtype constructor maps names back to the type"""
        assert np.dtype(t.__name__).type is t


if __name__ == "__main__":
    run_tests()
