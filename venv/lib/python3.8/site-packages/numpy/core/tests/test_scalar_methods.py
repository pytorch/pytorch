"""
Test the scalar constructors, which also do type-coercion
"""
import fractions
import platform

import pytest
import numpy as np

from numpy.testing import assert_equal, assert_raises


class TestAsIntegerRatio:
    # derived in part from the cpython test "test_floatasratio"

    @pytest.mark.parametrize("ftype", [
        np.half, np.single, np.double, np.longdouble])
    @pytest.mark.parametrize("f, ratio", [
        (0.875, (7, 8)),
        (-0.875, (-7, 8)),
        (0.0, (0, 1)),
        (11.5, (23, 2)),
        ])
    def test_small(self, ftype, f, ratio):
        assert_equal(ftype(f).as_integer_ratio(), ratio)

    @pytest.mark.parametrize("ftype", [
        np.half, np.single, np.double, np.longdouble])
    def test_simple_fractions(self, ftype):
        R = fractions.Fraction
        assert_equal(R(0, 1),
                     R(*ftype(0.0).as_integer_ratio()))
        assert_equal(R(5, 2),
                     R(*ftype(2.5).as_integer_ratio()))
        assert_equal(R(1, 2),
                     R(*ftype(0.5).as_integer_ratio()))
        assert_equal(R(-2100, 1),
                     R(*ftype(-2100.0).as_integer_ratio()))

    @pytest.mark.parametrize("ftype", [
        np.half, np.single, np.double, np.longdouble])
    def test_errors(self, ftype):
        assert_raises(OverflowError, ftype('inf').as_integer_ratio)
        assert_raises(OverflowError, ftype('-inf').as_integer_ratio)
        assert_raises(ValueError, ftype('nan').as_integer_ratio)

    def test_against_known_values(self):
        R = fractions.Fraction
        assert_equal(R(1075, 512),
                     R(*np.half(2.1).as_integer_ratio()))
        assert_equal(R(-1075, 512),
                     R(*np.half(-2.1).as_integer_ratio()))
        assert_equal(R(4404019, 2097152),
                     R(*np.single(2.1).as_integer_ratio()))
        assert_equal(R(-4404019, 2097152),
                     R(*np.single(-2.1).as_integer_ratio()))
        assert_equal(R(4728779608739021, 2251799813685248),
                     R(*np.double(2.1).as_integer_ratio()))
        assert_equal(R(-4728779608739021, 2251799813685248),
                     R(*np.double(-2.1).as_integer_ratio()))
        # longdouble is platform dependent

    @pytest.mark.parametrize("ftype, frac_vals, exp_vals", [
        # dtype test cases generated using hypothesis
        # first five generated cases per dtype
        (np.half, [0.0, 0.01154830649280303, 0.31082276347447274,
                   0.527350517124794, 0.8308562335072596],
                  [0, 1, 0, -8, 12]),
        (np.single, [0.0, 0.09248576989263226, 0.8160498218131407,
                     0.17389442853722373, 0.7956044195067877],
                    [0, 12, 10, 17, -26]),
        (np.double, [0.0, 0.031066908499895136, 0.5214135908877832,
                     0.45780736035689296, 0.5906586745934036],
                    [0, -801, 51, 194, -653]),
        pytest.param(
            np.longdouble,
            [0.0, 0.20492557202724854, 0.4277180662199366, 0.9888085019891495,
             0.9620175814461964],
            [0, -7400, 14266, -7822, -8721],
            marks=[
                pytest.mark.skipif(
                    np.finfo(np.double) == np.finfo(np.longdouble),
                    reason="long double is same as double"),
                pytest.mark.skipif(
                    platform.machine().startswith("ppc"),
                    reason="IBM double double"),
            ]
        )
    ])
    def test_roundtrip(self, ftype, frac_vals, exp_vals):
        for frac, exp in zip(frac_vals, exp_vals):
            f = np.ldexp(frac, exp, dtype=ftype)
            n, d = f.as_integer_ratio()

            try:
                # workaround for gh-9968
                nf = np.longdouble(str(n))
                df = np.longdouble(str(d))
            except (OverflowError, RuntimeWarning):
                # the values may not fit in any float type
                pytest.skip("longdouble too small on this platform")

            assert_equal(nf / df, f, "{}/{}".format(n, d))
