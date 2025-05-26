# Owner(s): ["module: dynamo"]

"""Test examples for NEP 50."""

import itertools
from unittest import skipIf as skipif, SkipTest


try:
    import numpy as _np

    v = _np.__version__.split(".")
    HAVE_NUMPY = int(v[0]) >= 1 and int(v[1]) >= 24
except ImportError:
    HAVE_NUMPY = False

import torch._numpy as tnp
from torch._numpy import (  # noqa: F401
    array,
    bool_,
    complex128,
    complex64,
    float32,
    float64,
    inf,
    int16,
    int32,
    int64,
    uint8,
)
from torch._numpy.testing import assert_allclose
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


uint16 = uint8  # can be anything here, see below


# from numpy import array, uint8, uint16, int64, float32, float64, inf
# from numpy.testing import assert_allclose
# import numpy as np
# np._set_promotion_state('weak')

from pytest import raises as assert_raises


unchanged = None

# expression    old result   new_result
examples = {
    "uint8(1) + 2": (int64(3), uint8(3)),
    "array([1], uint8) + int64(1)": (array([2], uint8), array([2], int64)),
    "array([1], uint8) + array(1, int64)": (array([2], uint8), array([2], int64)),
    "array([1.], float32) + float64(1.)": (
        array([2.0], float32),
        array([2.0], float64),
    ),
    "array([1.], float32) + array(1., float64)": (
        array([2.0], float32),
        array([2.0], float64),
    ),
    "array([1], uint8) + 1": (array([2], uint8), unchanged),
    "array([1], uint8) + 200": (array([201], uint8), unchanged),
    "array([100], uint8) + 200": (array([44], uint8), unchanged),
    "array([1], uint8) + 300": (array([301], uint16), Exception),
    "uint8(1) + 300": (int64(301), Exception),
    "uint8(100) + 200": (int64(301), uint8(44)),  # and RuntimeWarning
    "float32(1) + 3e100": (float64(3e100), float32(inf)),  # and RuntimeWarning [T7]
    "array([1.0], float32) + 1e-14 == 1.0": (array([True]), unchanged),
    "array([0.1], float32) == float64(0.1)": (array([True]), array([False])),
    "array(1.0, float32) + 1e-14 == 1.0": (array(False), array(True)),
    "array([1.], float32) + 3": (array([4.0], float32), unchanged),
    "array([1.], float32) + int64(3)": (array([4.0], float32), array([4.0], float64)),
    "3j + array(3, complex64)": (array(3 + 3j, complex128), array(3 + 3j, complex64)),
    "float32(1) + 1j": (array(1 + 1j, complex128), array(1 + 1j, complex64)),
    "int32(1) + 5j": (array(1 + 5j, complex128), unchanged),
    # additional examples from the NEP text
    "int16(2) + 2": (int64(4), int16(4)),
    "int16(4) + 4j": (complex128(4 + 4j), unchanged),
    "float32(5) + 5j": (complex128(5 + 5j), complex64(5 + 5j)),
    "bool_(True) + 1": (int64(2), unchanged),
    "True + uint8(2)": (uint8(3), unchanged),
}


@skipif(not HAVE_NUMPY, reason="NumPy not found")
@instantiate_parametrized_tests
class TestNEP50Table(TestCase):
    @parametrize("example", examples)
    def test_nep50_exceptions(self, example):
        old, new = examples[example]

        if new == Exception:
            with assert_raises(OverflowError):
                eval(example)

        else:
            result = eval(example)

            if new is unchanged:
                new = old

            assert_allclose(result, new, atol=1e-16)
            assert result.dtype == new.dtype


# ### Directly compare to numpy ###

weaks = (True, 1, 2.0, 3j)
non_weaks = (
    tnp.asarray(True),
    tnp.uint8(1),
    tnp.int8(1),
    tnp.int32(1),
    tnp.int64(1),
    tnp.float32(1),
    tnp.float64(1),
    tnp.complex64(1),
    tnp.complex128(1),
)
if HAVE_NUMPY:
    dtypes = (
        None,
        _np.bool_,
        _np.uint8,
        _np.int8,
        _np.int32,
        _np.int64,
        _np.float32,
        _np.float64,
        _np.complex64,
        _np.complex128,
    )
else:
    dtypes = (None,)


# ufunc name: [array.dtype]
corners = {
    "true_divide": ["bool_", "uint8", "int8", "int16", "int32", "int64"],
    "divide": ["bool_", "uint8", "int8", "int16", "int32", "int64"],
    "arctan2": ["bool_", "uint8", "int8", "int16", "int32", "int64"],
    "copysign": ["bool_", "uint8", "int8", "int16", "int32", "int64"],
    "heaviside": ["bool_", "uint8", "int8", "int16", "int32", "int64"],
    "ldexp": ["bool_", "uint8", "int8", "int16", "int32", "int64"],
    "power": ["uint8"],
    "nextafter": ["float32"],
}


@skipif(not HAVE_NUMPY, reason="NumPy not found")
@instantiate_parametrized_tests
class TestCompareToNumpy(TestCase):
    @parametrize("scalar, array, dtype", itertools.product(weaks, non_weaks, dtypes))
    def test_direct_compare(self, scalar, array, dtype):
        # compare to NumPy w/ NEP 50.
        try:
            state = _np._get_promotion_state()
            _np._set_promotion_state("weak")

            if dtype is not None:
                kwargs = {"dtype": dtype}
            try:
                result_numpy = _np.add(scalar, array.tensor.numpy(), **kwargs)
            except Exception:
                return

            kwargs = {}
            if dtype is not None:
                kwargs = {"dtype": getattr(tnp, dtype.__name__)}
            result = tnp.add(scalar, array, **kwargs).tensor.numpy()
            assert result.dtype == result_numpy.dtype
            assert result == result_numpy

        finally:
            _np._set_promotion_state(state)

    @parametrize("name", tnp._ufuncs._binary)
    @parametrize("scalar, array", itertools.product(weaks, non_weaks))
    def test_compare_ufuncs(self, name, scalar, array):
        if name in corners and (
            array.dtype.name in corners[name]
            or tnp.asarray(scalar).dtype.name in corners[name]
        ):
            raise SkipTest(f"{name}(..., dtype=array.dtype)")

        try:
            state = _np._get_promotion_state()
            _np._set_promotion_state("weak")

            if name in ["matmul", "modf", "divmod", "ldexp"]:
                return
            ufunc = getattr(tnp, name)
            ufunc_numpy = getattr(_np, name)

            try:
                result = ufunc(scalar, array)
            except RuntimeError:
                # RuntimeError: "bitwise_xor_cpu" not implemented for 'ComplexDouble' etc
                result = None

            try:
                result_numpy = ufunc_numpy(scalar, array.tensor.numpy())
            except TypeError:
                # TypeError: ufunc 'hypot' not supported for the input types
                result_numpy = None

            if result is not None and result_numpy is not None:
                assert result.tensor.numpy().dtype == result_numpy.dtype

        finally:
            _np._set_promotion_state(state)


if __name__ == "__main__":
    run_tests()
