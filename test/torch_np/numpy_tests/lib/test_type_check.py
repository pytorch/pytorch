# Owner(s): ["module: dynamo"]


import functools
from unittest import expectedFailure as xfail, skipIf as skipif

from pytest import raises as assert_raises

from torch.testing._internal.common_utils import (
    run_tests,
    TEST_WITH_TORCHDYNAMO,
    TestCase,
    xpassIfTorchDynamo_np,
)


if TEST_WITH_TORCHDYNAMO:
    import numpy as np
    from numpy import (
        common_type,
        iscomplex,
        iscomplexobj,
        isneginf,
        isposinf,
        isreal,
        isrealobj,
        nan_to_num,
        real_if_close,
    )
    from numpy.testing import assert_, assert_array_equal, assert_equal
else:
    import torch._numpy as np
    from torch._numpy import (
        common_type,
        iscomplex,
        iscomplexobj,
        isneginf,
        isposinf,
        isreal,
        isrealobj,
        nan_to_num,
        real_if_close,
    )
    from torch._numpy.testing import assert_, assert_array_equal, assert_equal


skip = functools.partial(skipif, True)


def assert_all(x):
    assert_(np.all(x), x)


@xpassIfTorchDynamo_np  # (reason="common_type not implemented")
class TestCommonType(TestCase):
    def test_basic(self):
        ai32 = np.array([[1, 2], [3, 4]], dtype=np.int32)
        af16 = np.array([[1, 2], [3, 4]], dtype=np.float16)
        af32 = np.array([[1, 2], [3, 4]], dtype=np.float32)
        af64 = np.array([[1, 2], [3, 4]], dtype=np.float64)
        acs = np.array([[1 + 5j, 2 + 6j], [3 + 7j, 4 + 8j]], dtype=np.csingle)
        acd = np.array([[1 + 5j, 2 + 6j], [3 + 7j, 4 + 8j]], dtype=np.cdouble)
        assert_(common_type(ai32) == np.float64)
        assert_(common_type(af16) == np.float16)
        assert_(common_type(af32) == np.float32)
        assert_(common_type(af64) == np.float64)
        assert_(common_type(acs) == np.csingle)
        assert_(common_type(acd) == np.cdouble)


@xfail  # (reason="not implemented")
class TestMintypecode(TestCase):
    def test_default_1(self):
        for itype in "1bcsuwil":
            assert_equal(mintypecode(itype), "d")
        assert_equal(mintypecode("f"), "f")
        assert_equal(mintypecode("d"), "d")
        assert_equal(mintypecode("F"), "F")
        assert_equal(mintypecode("D"), "D")

    def test_default_2(self):
        for itype in "1bcsuwil":
            assert_equal(mintypecode(itype + "f"), "f")
            assert_equal(mintypecode(itype + "d"), "d")
            assert_equal(mintypecode(itype + "F"), "F")
            assert_equal(mintypecode(itype + "D"), "D")
        assert_equal(mintypecode("ff"), "f")
        assert_equal(mintypecode("fd"), "d")
        assert_equal(mintypecode("fF"), "F")
        assert_equal(mintypecode("fD"), "D")
        assert_equal(mintypecode("df"), "d")
        assert_equal(mintypecode("dd"), "d")
        # assert_equal(mintypecode('dF',savespace=1),'F')
        assert_equal(mintypecode("dF"), "D")
        assert_equal(mintypecode("dD"), "D")
        assert_equal(mintypecode("Ff"), "F")
        # assert_equal(mintypecode('Fd',savespace=1),'F')
        assert_equal(mintypecode("Fd"), "D")
        assert_equal(mintypecode("FF"), "F")
        assert_equal(mintypecode("FD"), "D")
        assert_equal(mintypecode("Df"), "D")
        assert_equal(mintypecode("Dd"), "D")
        assert_equal(mintypecode("DF"), "D")
        assert_equal(mintypecode("DD"), "D")

    def test_default_3(self):
        assert_equal(mintypecode("fdF"), "D")
        # assert_equal(mintypecode('fdF',savespace=1),'F')
        assert_equal(mintypecode("fdD"), "D")
        assert_equal(mintypecode("fFD"), "D")
        assert_equal(mintypecode("dFD"), "D")

        assert_equal(mintypecode("ifd"), "d")
        assert_equal(mintypecode("ifF"), "F")
        assert_equal(mintypecode("ifD"), "D")
        assert_equal(mintypecode("idF"), "D")
        # assert_equal(mintypecode('idF',savespace=1),'F')
        assert_equal(mintypecode("idD"), "D")


@xpassIfTorchDynamo_np  # (reason="TODO: decide on if [1] is a scalar or not")
class TestIsscalar(TestCase):
    def test_basic(self):
        assert_(np.isscalar(3))
        assert_(not np.isscalar([3]))
        assert_(not np.isscalar((3,)))
        assert_(np.isscalar(3j))
        assert_(np.isscalar(4.0))


class TestReal(TestCase):
    def test_real(self):
        y = np.random.rand(
            10,
        )
        assert_array_equal(y, np.real(y))

        y = np.array(1)
        out = np.real(y)
        assert_array_equal(y, out)
        assert_(isinstance(out, np.ndarray))

        y = 1
        out = np.real(y)
        assert_equal(y, out)
        # assert_(not isinstance(out, np.ndarray))  # XXX: 0D tensor, not scalar

    def test_cmplx(self):
        y = np.random.rand(
            10,
        ) + 1j * np.random.rand(
            10,
        )
        assert_array_equal(y.real, np.real(y))

        y = np.array(1 + 1j)
        out = np.real(y)
        assert_array_equal(y.real, out)
        assert_(isinstance(out, np.ndarray))

        y = 1 + 1j
        out = np.real(y)
        assert_equal(1.0, out)
        # assert_(not isinstance(out, np.ndarray))  # XXX: 0D tensor, not scalar


class TestImag(TestCase):
    def test_real(self):
        y = np.random.rand(
            10,
        )
        assert_array_equal(0, np.imag(y))

        y = np.array(1)
        out = np.imag(y)
        assert_array_equal(0, out)
        assert_(isinstance(out, np.ndarray))

        y = 1
        out = np.imag(y)
        assert_equal(0, out)
        # assert_(not isinstance(out, np.ndarray))  # XXX: 0D tensor, not scalar

    def test_cmplx(self):
        y = np.random.rand(
            10,
        ) + 1j * np.random.rand(
            10,
        )
        assert_array_equal(y.imag, np.imag(y))

        y = np.array(1 + 1j)
        out = np.imag(y)
        assert_array_equal(y.imag, out)
        assert_(isinstance(out, np.ndarray))

        y = 1 + 1j
        out = np.imag(y)
        assert_equal(1.0, out)
        # assert_(not isinstance(out, np.ndarray))  # XXX: 0D tensor, not scalar


class TestIscomplex(TestCase):
    def test_fail(self):
        z = np.array([-1, 0, 1])
        res = iscomplex(z)
        assert_(not np.any(res, axis=0))

    def test_pass(self):
        z = np.array([-1j, 1, 0])
        res = iscomplex(z)
        assert_array_equal(res, [1, 0, 0])


class TestIsreal(TestCase):
    def test_pass(self):
        z = np.array([-1, 0, 1j])
        res = isreal(z)
        assert_array_equal(res, [1, 1, 0])

    def test_fail(self):
        z = np.array([-1j, 1, 0])
        res = isreal(z)
        assert_array_equal(res, [0, 1, 1])

    def test_isreal_real(self):
        z = np.array([-1, 0, 1])
        res = isreal(z)
        assert res.all()


class TestIscomplexobj(TestCase):
    def test_basic(self):
        z = np.array([-1, 0, 1])
        assert_(not iscomplexobj(z))
        z = np.array([-1j, 0, -1])
        assert_(iscomplexobj(z))

    def test_scalar(self):
        assert_(not iscomplexobj(1.0))
        assert_(iscomplexobj(1 + 0j))

    def test_list(self):
        assert_(iscomplexobj([3, 1 + 0j, True]))
        assert_(not iscomplexobj([3, 1, True]))


class TestIsrealobj(TestCase):
    def test_basic(self):
        z = np.array([-1, 0, 1])
        assert_(isrealobj(z))
        z = np.array([-1j, 0, -1])
        assert_(not isrealobj(z))


class TestIsnan(TestCase):
    def test_goodvalues(self):
        z = np.array((-1.0, 0.0, 1.0))
        res = np.isnan(z) == 0
        assert_all(np.all(res, axis=0))

    def test_posinf(self):
        assert_all(np.isnan(np.array((1.0,)) / 0.0) == 0)

    def test_neginf(self):
        assert_all(np.isnan(np.array((-1.0,)) / 0.0) == 0)

    def test_ind(self):
        assert_all(np.isnan(np.array((0.0,)) / 0.0) == 1)

    def test_integer(self):
        assert_all(np.isnan(1) == 0)

    def test_complex(self):
        assert_all(np.isnan(1 + 1j) == 0)

    def test_complex1(self):
        assert_all(np.isnan(np.array(0 + 0j) / 0.0) == 1)


class TestIsfinite(TestCase):
    # Fixme, wrong place, isfinite now ufunc

    def test_goodvalues(self):
        z = np.array((-1.0, 0.0, 1.0))
        res = np.isfinite(z) == 1
        assert_all(np.all(res, axis=0))

    def test_posinf(self):
        assert_all(np.isfinite(np.array((1.0,)) / 0.0) == 0)

    def test_neginf(self):
        assert_all(np.isfinite(np.array((-1.0,)) / 0.0) == 0)

    def test_ind(self):
        assert_all(np.isfinite(np.array((0.0,)) / 0.0) == 0)

    def test_integer(self):
        assert_all(np.isfinite(1) == 1)

    def test_complex(self):
        assert_all(np.isfinite(1 + 1j) == 1)

    def test_complex1(self):
        assert_all(np.isfinite(np.array(1 + 1j) / 0.0) == 0)


class TestIsinf(TestCase):
    # Fixme, wrong place, isinf now ufunc

    def test_goodvalues(self):
        z = np.array((-1.0, 0.0, 1.0))
        res = np.isinf(z) == 0
        assert_all(np.all(res, axis=0))

    def test_posinf(self):
        assert_all(np.isinf(np.array((1.0,)) / 0.0) == 1)

    def test_posinf_scalar(self):
        assert_all(
            np.isinf(
                np.array(
                    1.0,
                )
                / 0.0
            )
            == 1
        )

    def test_neginf(self):
        assert_all(np.isinf(np.array((-1.0,)) / 0.0) == 1)

    def test_neginf_scalar(self):
        assert_all(np.isinf(np.array(-1.0) / 0.0) == 1)

    def test_ind(self):
        assert_all(np.isinf(np.array((0.0,)) / 0.0) == 0)


class TestIsposinf(TestCase):
    def test_generic(self):
        vals = isposinf(np.array((-1.0, 0, 1)) / 0.0)
        assert_(vals[0] == 0)
        assert_(vals[1] == 0)
        assert_(vals[2] == 1)


class TestIsneginf(TestCase):
    def test_generic(self):
        vals = isneginf(np.array((-1.0, 0, 1)) / 0.0)
        assert_(vals[0] == 1)
        assert_(vals[1] == 0)
        assert_(vals[2] == 0)


# @xfail  #(reason="not implemented")
class TestNanToNum(TestCase):
    def test_generic(self):
        vals = nan_to_num(np.array((-1.0, 0, 1)) / 0.0)
        assert_all(vals[0] < -1e10) and assert_all(np.isfinite(vals[0]))
        assert_(vals[1] == 0)
        assert_all(vals[2] > 1e10) and assert_all(np.isfinite(vals[2]))
        assert isinstance(vals, np.ndarray)

        # perform the same tests but with nan, posinf and neginf keywords
        vals = nan_to_num(np.array((-1.0, 0, 1)) / 0.0, nan=10, posinf=20, neginf=30)
        assert_equal(vals, [30, 10, 20])
        assert_all(np.isfinite(vals[[0, 2]]))
        assert isinstance(vals, np.ndarray)

    def test_array(self):
        vals = nan_to_num([1])
        assert_array_equal(vals, np.array([1], int))
        assert isinstance(vals, np.ndarray)
        vals = nan_to_num([1], nan=10, posinf=20, neginf=30)
        assert_array_equal(vals, np.array([1], int))
        assert isinstance(vals, np.ndarray)

    @skip(reason="we return OD arrays not scalars")
    def test_integer(self):
        vals = nan_to_num(1)
        assert_all(vals == 1)
        assert isinstance(vals, np.int_)
        vals = nan_to_num(1, nan=10, posinf=20, neginf=30)
        assert_all(vals == 1)
        assert isinstance(vals, np.int_)

    @skip(reason="we return OD arrays not scalars")
    def test_float(self):
        vals = nan_to_num(1.0)
        assert_all(vals == 1.0)
        assert_equal(type(vals), np.float64)
        vals = nan_to_num(1.1, nan=10, posinf=20, neginf=30)
        assert_all(vals == 1.1)
        assert_equal(type(vals), np.float64)

    @skip(reason="we return OD arrays not scalars")
    def test_complex_good(self):
        vals = nan_to_num(1 + 1j)
        assert_all(vals == 1 + 1j)
        assert isinstance(vals, np.complex128)
        vals = nan_to_num(1 + 1j, nan=10, posinf=20, neginf=30)
        assert_all(vals == 1 + 1j)
        assert_equal(type(vals), np.complex128)

    @skip(reason="we return OD arrays not scalars")
    def test_complex_bad(self):
        v = 1 + 1j
        v += np.array(0 + 1.0j) / 0.0
        vals = nan_to_num(v)
        # !! This is actually (unexpectedly) zero
        assert_all(np.isfinite(vals))
        assert_equal(type(vals), np.complex128)

    @skip(reason="we return OD arrays not scalars")
    def test_complex_bad2(self):
        v = 1 + 1j
        v += np.array(-1 + 1.0j) / 0.0
        vals = nan_to_num(v)
        assert_all(np.isfinite(vals))
        assert_equal(type(vals), np.complex128)
        # Fixme
        # assert_all(vals.imag > 1e10)  and assert_all(np.isfinite(vals))
        # !! This is actually (unexpectedly) positive
        # !! inf.  Comment out for now, and see if it
        # !! changes
        # assert_all(vals.real < -1e10) and assert_all(np.isfinite(vals))

    def test_do_not_rewrite_previous_keyword(self):
        # This is done to test that when, for instance, nan=np.inf then these
        # values are not rewritten by posinf keyword to the posinf value.
        vals = nan_to_num(np.array((-1.0, 0, 1)) / 0.0, nan=np.inf, posinf=999)
        assert_all(np.isfinite(vals[[0, 2]]))
        assert_all(vals[0] < -1e10)
        assert_equal(vals[[1, 2]], [np.inf, 999])
        assert isinstance(vals, np.ndarray)


class TestRealIfClose(TestCase):
    def test_basic(self):
        a = np.random.rand(10)
        b = real_if_close(a + 1e-15j)
        assert_all(isrealobj(b))
        assert_array_equal(a, b)
        b = real_if_close(a + 1e-7j)
        assert_all(iscomplexobj(b))
        b = real_if_close(a + 1e-7j, tol=1e-6)
        assert_all(isrealobj(b))


@xfail  # (reason="not implemented")
class TestArrayConversion(TestCase):
    def test_asfarray(self):
        a = asfarray(np.array([1, 2, 3]))
        assert_equal(a.__class__, np.ndarray)
        assert_(np.issubdtype(a.dtype, np.floating))

        # previously this would infer dtypes from arrays, unlike every single
        # other numpy function
        assert_raises(TypeError, asfarray, np.array([1, 2, 3]), dtype=np.array(1.0))


if __name__ == "__main__":
    run_tests()
