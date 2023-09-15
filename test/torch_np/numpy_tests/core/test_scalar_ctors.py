# Owner(s): ["module: dynamo"]

"""
Test the scalar constructors, which also do type-coercion
"""
import pytest

import torch._numpy as np
from torch._numpy.testing import assert_almost_equal, assert_equal


class TestFromString:
    @pytest.mark.xfail(reason="XXX: floats from strings")
    def test_floating(self):
        # Ticket #640, floats from string
        fsingle = np.single("1.234")
        fdouble = np.double("1.234")
        assert_almost_equal(fsingle, 1.234)
        assert_almost_equal(fdouble, 1.234)

    @pytest.mark.xfail(reason="XXX: floats from strings")
    def test_floating_overflow(self):
        """Strings containing an unrepresentable float overflow"""
        fhalf = np.half("1e10000")
        assert_equal(fhalf, np.inf)
        fsingle = np.single("1e10000")
        assert_equal(fsingle, np.inf)
        fdouble = np.double("1e10000")
        assert_equal(fdouble, np.inf)

        fhalf = np.half("-1e10000")
        assert_equal(fhalf, -np.inf)
        fsingle = np.single("-1e10000")
        assert_equal(fsingle, -np.inf)
        fdouble = np.double("-1e10000")
        assert_equal(fdouble, -np.inf)

    def test_bool(self):
        with pytest.raises(TypeError):
            np.bool_(False, garbage=True)


class TestFromInt:
    def test_intp(self):
        # Ticket #99
        assert_equal(1024, np.intp(1024))

    def test_uint64_from_negative(self):
        # NumPy test was asserting a DeprecationWarning
        assert_equal(np.uint8(-2), np.uint8(254))


int_types = [np.byte, np.short, np.intc, np.int_, np.longlong]
uint_types = [np.ubyte]
float_types = [np.half, np.single, np.double]
cfloat_types = [np.csingle, np.cdouble]


class TestArrayFromScalar:
    """gh-15467"""

    def _do_test(self, t1, t2):
        x = t1(2)
        arr = np.array(x, dtype=t2)
        # type should be preserved exactly
        if t2 is None:
            assert arr.dtype.type is t1
        else:
            assert arr.dtype.type is t2

        arr1 = np.asarray(x, dtype=t2)
        if t2 is None:
            assert arr1.dtype.type is t1
        else:
            assert arr1.dtype.type is t2

    @pytest.mark.parametrize("t1", int_types + uint_types)
    @pytest.mark.parametrize("t2", int_types + uint_types + [None])
    def test_integers(self, t1, t2):
        return self._do_test(t1, t2)

    @pytest.mark.parametrize("t1", float_types)
    @pytest.mark.parametrize("t2", float_types + [None])
    def test_reals(self, t1, t2):
        return self._do_test(t1, t2)

    @pytest.mark.parametrize("t1", cfloat_types)
    @pytest.mark.parametrize("t2", cfloat_types + [None])
    def test_complex(self, t1, t2):
        return self._do_test(t1, t2)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
