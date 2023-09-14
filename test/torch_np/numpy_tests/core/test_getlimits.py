# Owner(s): ["module: dynamo"]

""" Test functions for limits module.

"""
import warnings

import pytest

import torch._numpy as np
from pytest import raises as assert_raises
from torch._numpy import double, finfo, half, iinfo, single
from torch._numpy.testing import assert_, assert_equal

# from numpy.core.getlimits import _discovered_machar, _float_ma

##################################################


@pytest.mark.skip(reason="torch.finfo is not a singleton. Why demanding it is?")
class TestPythonFloat:
    def test_singleton(self):
        ftype = finfo(float)
        ftype2 = finfo(float)
        assert_equal(id(ftype), id(ftype2))


@pytest.mark.skip(reason="torch.finfo is not a singleton. Why demanding it is?")
class TestHalf:
    def test_singleton(self):
        ftype = finfo(half)
        ftype2 = finfo(half)
        assert_equal(id(ftype), id(ftype2))


@pytest.mark.skip(reason="torch.finfo is not a singleton. Why demanding it is?")
class TestSingle:
    def test_singleton(self):
        ftype = finfo(single)
        ftype2 = finfo(single)
        assert_equal(id(ftype), id(ftype2))


@pytest.mark.skip(reason="torch.finfo is not a singleton. Why demanding it is?")
class TestDouble:
    def test_singleton(self):
        ftype = finfo(double)
        ftype2 = finfo(double)
        assert_equal(id(ftype), id(ftype2))


class TestFinfo:
    def test_basic(self):
        dts = list(
            zip(
                ["f2", "f4", "f8", "c8", "c16"],
                [np.float16, np.float32, np.float64, np.complex64, np.complex128],
            )
        )
        for dt1, dt2 in dts:
            for attr in (
                "bits",
                "eps",
                "max",
                "min",
                "resolution",
                "tiny",
                "smallest_normal",
            ):
                assert_equal(getattr(finfo(dt1), attr), getattr(finfo(dt2), attr), attr)
        with assert_raises((TypeError, ValueError)):
            finfo("i4")

    @pytest.mark.xfail(reason="These attributes are not implemented yet.")
    def test_basic_missing(self):
        dt = np.float32
        for attr in [
            "epsneg",
            "iexp",
            "machep",
            "maxexp",
            "minexp",
            "negep",
            "nexp",
            "nmant",
            "precision",
            "smallest_subnormal",
        ]:
            getattr(finfo(dt), attr)


class TestIinfo:
    def test_basic(self):
        dts = list(
            zip(
                ["i1", "i2", "i4", "i8", "u1", "u2", "u4", "u8"],
                [
                    np.int8,
                    np.int16,
                    np.int32,
                    np.int64,
                    np.uint8,
                ],
            )
        )
        for dt1, dt2 in dts:
            for attr in ("bits", "min", "max"):
                assert_equal(getattr(iinfo(dt1), attr), getattr(iinfo(dt2), attr), attr)
        with assert_raises((TypeError, ValueError)):
            iinfo("f4")

    def test_unsigned_max(self):
        types = np.sctypes["uint"]
        for T in types:
            max_calculated = T(0) - T(1)
            assert_equal(iinfo(T).max, max_calculated)


class TestRepr:
    def test_iinfo_repr(self):
        expected = "iinfo(min=-32768, max=32767, dtype=int16)"
        assert_equal(repr(np.iinfo(np.int16)), expected)

    def test_finfo_repr(self):
        repr_f32 = repr(np.finfo(np.float32))
        assert "finfo(resolution=1e-06, min=-3.40282e+38," in repr_f32
        assert "dtype=float32" in repr_f32


@pytest.mark.skip(reason="Instantiate {i,f}info from dtypes.")
def test_instances():
    iinfo(10)
    finfo(3.0)


@pytest.mark.skip(reason="MachAr no implemented (does it need to be)?")
def assert_ma_equal(discovered, ma_like):
    # Check MachAr-like objects same as calculated MachAr instances
    for key, value in discovered.__dict__.items():
        assert_equal(value, getattr(ma_like, key))
        if hasattr(value, "shape"):
            assert_equal(value.shape, getattr(ma_like, key).shape)
            assert_equal(value.dtype, getattr(ma_like, key).dtype)


@pytest.mark.skip(reason="MachAr no implemented (does it need to)?")
def test_known_types():
    # Test we are correctly compiling parameters for known types
    for ftype, ma_like in (
        (np.float16, _float_ma[16]),
        (np.float32, _float_ma[32]),
        (np.float64, _float_ma[64]),
    ):
        assert_ma_equal(_discovered_machar(ftype), ma_like)
    # Suppress warning for broken discovery of double double on PPC
    ld_ma = _discovered_machar(np.longdouble)
    bytes = np.dtype(np.longdouble).itemsize
    if (ld_ma.it, ld_ma.maxexp) == (63, 16384) and bytes in (12, 16):
        # 80-bit extended precision
        assert_ma_equal(ld_ma, _float_ma[80])
    elif (ld_ma.it, ld_ma.maxexp) == (112, 16384) and bytes == 16:
        # IEE 754 128-bit
        assert_ma_equal(ld_ma, _float_ma[128])


@pytest.mark.skip(reason="MachAr no implemented (does it need to be)?")
def test_subnormal_warning():
    """Test that the subnormal is zero warning is not being raised."""
    ld_ma = _discovered_machar(np.longdouble)
    bytes = np.dtype(np.longdouble).itemsize
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        if (ld_ma.it, ld_ma.maxexp) == (63, 16384) and bytes in (12, 16):
            # 80-bit extended precision
            ld_ma.smallest_subnormal
            assert len(w) == 0
        elif (ld_ma.it, ld_ma.maxexp) == (112, 16384) and bytes == 16:
            # IEE 754 128-bit
            ld_ma.smallest_subnormal
            assert len(w) == 0
        else:
            # Double double
            ld_ma.smallest_subnormal
            # This test may fail on some platforms
            assert len(w) == 0


@pytest.mark.xfail(reason="None of nmant, minexp, maxexp is implemented.")
def test_plausible_finfo():
    # Assert that finfo returns reasonable results for all types
    for ftype in np.sctypes["float"] + np.sctypes["complex"]:
        info = np.finfo(ftype)
        assert_(info.nmant > 1)
        assert_(info.minexp < -1)
        assert_(info.maxexp > 1)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
