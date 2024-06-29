# Owner(s): ["module: dynamo"]

""" Test functions for limits module.

"""
import functools
import warnings
from unittest import expectedFailure as xfail, skipIf

import numpy
from pytest import raises as assert_raises

from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    subtest,
    TEST_WITH_TORCHDYNAMO,
    TestCase,
    xpassIfTorchDynamo,
)


if TEST_WITH_TORCHDYNAMO:
    import numpy as np
    from numpy import double, finfo, half, iinfo, single
    from numpy.testing import assert_, assert_equal
else:
    import torch._numpy as np
    from torch._numpy import double, finfo, half, iinfo, single
    from torch._numpy.testing import assert_, assert_equal


skip = functools.partial(skipIf, True)

##################################################


@skip(reason="torch.finfo is not a singleton. Why demanding it is?")
class TestPythonFloat(TestCase):
    def test_singleton(self):
        ftype = finfo(float)
        ftype2 = finfo(float)
        assert_equal(id(ftype), id(ftype2))


@skip(reason="torch.finfo is not a singleton. Why demanding it is?")
class TestHalf(TestCase):
    def test_singleton(self):
        ftype = finfo(half)
        ftype2 = finfo(half)
        assert_equal(id(ftype), id(ftype2))


@skip(reason="torch.finfo is not a singleton. Why demanding it is?")
class TestSingle(TestCase):
    def test_singleton(self):
        ftype = finfo(single)
        ftype2 = finfo(single)
        assert_equal(id(ftype), id(ftype2))


@skip(reason="torch.finfo is not a singleton. Why demanding it is?")
class TestDouble(TestCase):
    def test_singleton(self):
        ftype = finfo(double)
        ftype2 = finfo(double)
        assert_equal(id(ftype), id(ftype2))


class TestFinfo(TestCase):
    @skipIf(numpy.__version__ < "1.23", reason=".smallest_normal is new")
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

    @skip  # (reason="Some of these attributes are not implemented vs NP versions")
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


@instantiate_parametrized_tests
class TestIinfo(TestCase):
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

    @parametrize(
        "T",
        [
            np.uint8,
            # xfail: unsupported add (uint[16,32,64])
            subtest(np.uint16, decorators=[xfail]),
            subtest(np.uint32, decorators=[xfail]),
            subtest(np.uint64, decorators=[xfail]),
        ],
    )
    def test_unsigned_max(self, T):
        max_calculated = T(0) - T(1)
        assert_equal(iinfo(T).max, max_calculated)


class TestRepr(TestCase):
    def test_iinfo_repr(self):
        expected = "iinfo(min=-32768, max=32767, dtype=int16)"
        assert_equal(repr(np.iinfo(np.int16)), expected)

    @skipIf(TEST_WITH_TORCHDYNAMO, reason="repr differs")
    def test_finfo_repr(self):
        repr_f32 = repr(np.finfo(np.float32))
        assert "finfo(resolution=1e-06, min=-3.40282e+38," in repr_f32
        assert "dtype=float32" in repr_f32


def assert_ma_equal(discovered, ma_like):
    # Check MachAr-like objects same as calculated MachAr instances
    for key, value in discovered.__dict__.items():
        assert_equal(value, getattr(ma_like, key))
        if hasattr(value, "shape"):
            assert_equal(value.shape, getattr(ma_like, key).shape)
            assert_equal(value.dtype, getattr(ma_like, key).dtype)


class TestMisc(TestCase):
    @skip(reason="Instantiate {i,f}info from dtypes.")
    def test_instances(self):
        iinfo(10)
        finfo(3.0)

    @skip(reason="MachAr no implemented (does it need to)?")
    def test_known_types(self):
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

    @skip(reason="MachAr no implemented (does it need to be)?")
    def test_subnormal_warning(self):
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

    @xpassIfTorchDynamo  # (reason="None of nmant, minexp, maxexp is implemented.")
    def test_plausible_finfo(self):
        # Assert that finfo returns reasonable results for all types
        for ftype in np.sctypes["float"] + np.sctypes["complex"]:
            info = np.finfo(ftype)
            assert_(info.nmant > 1)
            assert_(info.minexp < -1)
            assert_(info.maxexp > 1)


if __name__ == "__main__":
    run_tests()
