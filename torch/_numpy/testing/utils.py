"""
Utility function to facilitate testing.

"""
import contextlib
import gc
import operator
import os
import platform
import pprint
import re
import shutil
import sys
import warnings
from functools import wraps
from io import StringIO
from tempfile import mkdtemp, mkstemp
from warnings import WarningMessage

import torch._numpy as np
from torch._numpy import arange, asarray as asanyarray, empty, float32, intp, ndarray

__all__ = [
    "assert_equal",
    "assert_almost_equal",
    "assert_approx_equal",
    "assert_array_equal",
    "assert_array_less",
    "assert_string_equal",
    "assert_",
    "assert_array_almost_equal",
    "build_err_msg",
    "decorate_methods",
    "print_assert_equal",
    "verbose",
    "assert_",
    "assert_array_almost_equal_nulp",
    "assert_raises_regex",
    "assert_array_max_ulp",
    "assert_warns",
    "assert_no_warnings",
    "assert_allclose",
    "IgnoreException",
    "clear_and_catch_warnings",
    "temppath",
    "tempdir",
    "IS_PYPY",
    "HAS_REFCOUNT",
    "IS_WASM",
    "suppress_warnings",
    "assert_array_compare",
    "assert_no_gc_cycles",
    "break_cycles",
    "IS_PYSTON",
]


verbose = 0

IS_WASM = platform.machine() in ["wasm32", "wasm64"]
IS_PYPY = sys.implementation.name == "pypy"
IS_PYSTON = hasattr(sys, "pyston_version_info")
HAS_REFCOUNT = getattr(sys, "getrefcount", None) is not None and not IS_PYSTON


def assert_(val, msg=""):
    """
    Assert that works in release mode.
    Accepts callable msg to allow deferring evaluation until failure.

    The Python built-in ``assert`` does not work when executing code in
    optimized mode (the ``-O`` flag) - no byte-code is generated for it.

    For documentation on usage, refer to the Python documentation.

    """
    __tracebackhide__ = True  # Hide traceback for py.test
    if not val:
        try:
            smsg = msg()
        except TypeError:
            smsg = msg
        raise AssertionError(smsg)


def gisnan(x):
    return np.isnan(x)


def gisfinite(x):
    return np.isfinite(x)


def gisinf(x):
    return np.isinf(x)


def build_err_msg(
    arrays,
    err_msg,
    header="Items are not equal:",
    verbose=True,
    names=("ACTUAL", "DESIRED"),
    precision=8,
):
    msg = ["\n" + header]
    if err_msg:
        if err_msg.find("\n") == -1 and len(err_msg) < 79 - len(header):
            msg = [msg[0] + " " + err_msg]
        else:
            msg.append(err_msg)
    if verbose:
        for i, a in enumerate(arrays):
            if isinstance(a, ndarray):
                # precision argument is only needed if the objects are ndarrays
                # r_func = partial(array_repr, precision=precision)
                r_func = ndarray.__repr__
            else:
                r_func = repr

            try:
                r = r_func(a)
            except Exception as exc:
                r = f"[repr failed for <{type(a).__name__}>: {exc}]"
            if r.count("\n") > 3:
                r = "\n".join(r.splitlines()[:3])
                r += "..."
            msg.append(f" {names[i]}: {r}")
    return "\n".join(msg)


def assert_equal(actual, desired, err_msg="", verbose=True):
    """
    Raises an AssertionError if two objects are not equal.

    Given two objects (scalars, lists, tuples, dictionaries or numpy arrays),
    check that all elements of these objects are equal. An exception is raised
    at the first conflicting values.

    When one of `actual` and `desired` is a scalar and the other is array_like,
    the function checks that each element of the array_like object is equal to
    the scalar.

    This function handles NaN comparisons as if NaN was a "normal" number.
    That is, AssertionError is not raised if both objects have NaNs in the same
    positions.  This is in contrast to the IEEE standard on NaNs, which says
    that NaN compared to anything must return False.

    Parameters
    ----------
    actual : array_like
        The object to check.
    desired : array_like
        The expected object.
    err_msg : str, optional
        The error message to be printed in case of failure.
    verbose : bool, optional
        If True, the conflicting values are appended to the error message.

    Raises
    ------
    AssertionError
        If actual and desired are not equal.

    Examples
    --------
    >>> np.testing.assert_equal([4,5], [4,6])
    Traceback (most recent call last):
        ...
    AssertionError:
    Items are not equal:
    item=1
     ACTUAL: 5
     DESIRED: 6

    The following comparison does not raise an exception.  There are NaNs
    in the inputs, but they are in the same positions.

    >>> np.testing.assert_equal(np.array([1.0, 2.0, np.nan]), [1, 2, np.nan])

    """
    __tracebackhide__ = True  # Hide traceback for py.test

    num_nones = sum([actual is None, desired is None])
    if num_nones == 1:
        raise AssertionError(f"Not equal: {actual} != {desired}")
    elif num_nones == 2:
        return True
    # else, carry on

    if isinstance(actual, np.DType) or isinstance(desired, np.DType):
        result = actual == desired
        if not result:
            raise AssertionError(f"Not equal: {actual} != {desired}")
        else:
            return True

    if isinstance(desired, str) and isinstance(actual, str):
        assert actual == desired
        return

    if isinstance(desired, dict):
        if not isinstance(actual, dict):
            raise AssertionError(repr(type(actual)))
        assert_equal(len(actual), len(desired), err_msg, verbose)
        for k in desired.keys():
            if k not in actual:
                raise AssertionError(repr(k))
            assert_equal(actual[k], desired[k], f"key={k!r}\n{err_msg}", verbose)
        return
    if isinstance(desired, (list, tuple)) and isinstance(actual, (list, tuple)):
        assert_equal(len(actual), len(desired), err_msg, verbose)
        for k in range(len(desired)):
            assert_equal(actual[k], desired[k], f"item={k!r}\n{err_msg}", verbose)
        return

    from torch._numpy import imag, iscomplexobj, isscalar, ndarray, real, signbit

    if isinstance(actual, ndarray) or isinstance(desired, ndarray):
        return assert_array_equal(actual, desired, err_msg, verbose)
    msg = build_err_msg([actual, desired], err_msg, verbose=verbose)

    # Handle complex numbers: separate into real/imag to handle
    # nan/inf/negative zero correctly
    # XXX: catch ValueError for subclasses of ndarray where iscomplex fail
    try:
        usecomplex = iscomplexobj(actual) or iscomplexobj(desired)
    except (ValueError, TypeError):
        usecomplex = False

    if usecomplex:
        if iscomplexobj(actual):
            actualr = real(actual)
            actuali = imag(actual)
        else:
            actualr = actual
            actuali = 0
        if iscomplexobj(desired):
            desiredr = real(desired)
            desiredi = imag(desired)
        else:
            desiredr = desired
            desiredi = 0
        try:
            assert_equal(actualr, desiredr)
            assert_equal(actuali, desiredi)
        except AssertionError:
            raise AssertionError(msg)  # noqa: TRY200

    # isscalar test to check cases such as [np.nan] != np.nan
    if isscalar(desired) != isscalar(actual):
        raise AssertionError(msg)

    # Inf/nan/negative zero handling
    try:
        isdesnan = gisnan(desired)
        isactnan = gisnan(actual)
        if isdesnan and isactnan:
            return  # both nan, so equal

        # handle signed zero specially for floats
        array_actual = np.asarray(actual)
        array_desired = np.asarray(desired)

        if desired == 0 and actual == 0:
            if not signbit(desired) == signbit(actual):
                raise AssertionError(msg)

    except (TypeError, ValueError, NotImplementedError):
        pass

    try:
        # Explicitly use __eq__ for comparison, gh-2552
        if not (desired == actual):
            raise AssertionError(msg)

    except (DeprecationWarning, FutureWarning) as e:
        # this handles the case when the two types are not even comparable
        if "elementwise == comparison" in e.args[0]:
            raise AssertionError(msg)  # noqa: TRY200
        else:
            raise


def print_assert_equal(test_string, actual, desired):
    """
    Test if two objects are equal, and print an error message if test fails.

    The test is performed with ``actual == desired``.

    Parameters
    ----------
    test_string : str
        The message supplied to AssertionError.
    actual : object
        The object to test for equality against `desired`.
    desired : object
        The expected result.

    Examples
    --------
    >>> np.testing.print_assert_equal('Test XYZ of func xyz', [0, 1], [0, 1])  # doctest: +SKIP
    >>> np.testing.print_assert_equal('Test XYZ of func xyz', [0, 1], [0, 2])  # doctest: +SKIP
    Traceback (most recent call last):
    ...
    AssertionError: Test XYZ of func xyz failed
    ACTUAL:
    [0, 1]
    DESIRED:
    [0, 2]

    """
    __tracebackhide__ = True  # Hide traceback for py.test
    import pprint

    if not (actual == desired):
        msg = StringIO()
        msg.write(test_string)
        msg.write(" failed\nACTUAL: \n")
        pprint.pprint(actual, msg)
        msg.write("DESIRED: \n")
        pprint.pprint(desired, msg)
        raise AssertionError(msg.getvalue())


def assert_almost_equal(actual, desired, decimal=7, err_msg="", verbose=True):
    """
    Raises an AssertionError if two items are not equal up to desired
    precision.

    .. note:: It is recommended to use one of `assert_allclose`,
              `assert_array_almost_equal_nulp` or `assert_array_max_ulp`
              instead of this function for more consistent floating point
              comparisons.

    The test verifies that the elements of `actual` and `desired` satisfy.

        ``abs(desired-actual) < float64(1.5 * 10**(-decimal))``

    That is a looser test than originally documented, but agrees with what the
    actual implementation in `assert_array_almost_equal` did up to rounding
    vagaries. An exception is raised at conflicting values. For ndarrays this
    delegates to assert_array_almost_equal

    Parameters
    ----------
    actual : array_like
        The object to check.
    desired : array_like
        The expected object.
    decimal : int, optional
        Desired precision, default is 7.
    err_msg : str, optional
        The error message to be printed in case of failure.
    verbose : bool, optional
        If True, the conflicting values are appended to the error message.

    Raises
    ------
    AssertionError
      If actual and desired are not equal up to specified precision.

    See Also
    --------
    assert_allclose: Compare two array_like objects for equality with desired
                     relative and/or absolute precision.
    assert_array_almost_equal_nulp, assert_array_max_ulp, assert_equal

    Examples
    --------
    >>> from torch._numpy.testing import assert_almost_equal
    >>> assert_almost_equal(2.3333333333333, 2.33333334)
    >>> assert_almost_equal(2.3333333333333, 2.33333334, decimal=10)
    Traceback (most recent call last):
        ...
    AssertionError:
    Arrays are not almost equal to 10 decimals
     ACTUAL: 2.3333333333333
     DESIRED: 2.33333334

    >>> assert_almost_equal(np.array([1.0,2.3333333333333]),
    ...                     np.array([1.0,2.33333334]), decimal=9)
    Traceback (most recent call last):
        ...
    AssertionError:
    Arrays are not almost equal to 9 decimals
    <BLANKLINE>
    Mismatched elements: 1 / 2 (50%)
    Max absolute difference: 6.666699636781459e-09
    Max relative difference: 2.8571569790287484e-09
     x: torch.ndarray([1.0000, 2.3333], dtype=float64)
     y: torch.ndarray([1.0000, 2.3333], dtype=float64)

    """
    __tracebackhide__ = True  # Hide traceback for py.test
    from torch._numpy import imag, iscomplexobj, ndarray, real

    # Handle complex numbers: separate into real/imag to handle
    # nan/inf/negative zero correctly
    # XXX: catch ValueError for subclasses of ndarray where iscomplex fail
    try:
        usecomplex = iscomplexobj(actual) or iscomplexobj(desired)
    except ValueError:
        usecomplex = False

    def _build_err_msg():
        header = "Arrays are not almost equal to %d decimals" % decimal
        return build_err_msg([actual, desired], err_msg, verbose=verbose, header=header)

    if usecomplex:
        if iscomplexobj(actual):
            actualr = real(actual)
            actuali = imag(actual)
        else:
            actualr = actual
            actuali = 0
        if iscomplexobj(desired):
            desiredr = real(desired)
            desiredi = imag(desired)
        else:
            desiredr = desired
            desiredi = 0
        try:
            assert_almost_equal(actualr, desiredr, decimal=decimal)
            assert_almost_equal(actuali, desiredi, decimal=decimal)
        except AssertionError:
            raise AssertionError(_build_err_msg())  # noqa: TRY200

    if isinstance(actual, (ndarray, tuple, list)) or isinstance(
        desired, (ndarray, tuple, list)
    ):
        return assert_array_almost_equal(actual, desired, decimal, err_msg)
    try:
        # If one of desired/actual is not finite, handle it specially here:
        # check that both are nan if any is a nan, and test for equality
        # otherwise
        if not (gisfinite(desired) and gisfinite(actual)):
            if gisnan(desired) or gisnan(actual):
                if not (gisnan(desired) and gisnan(actual)):
                    raise AssertionError(_build_err_msg())
            else:
                if not desired == actual:
                    raise AssertionError(_build_err_msg())
            return
    except (NotImplementedError, TypeError):
        pass
    if abs(desired - actual) >= np.float64(1.5 * 10.0 ** (-decimal)):
        raise AssertionError(_build_err_msg())


def assert_approx_equal(actual, desired, significant=7, err_msg="", verbose=True):
    """
    Raises an AssertionError if two items are not equal up to significant
    digits.

    .. note:: It is recommended to use one of `assert_allclose`,
              `assert_array_almost_equal_nulp` or `assert_array_max_ulp`
              instead of this function for more consistent floating point
              comparisons.

    Given two numbers, check that they are approximately equal.
    Approximately equal is defined as the number of significant digits
    that agree.

    Parameters
    ----------
    actual : scalar
        The object to check.
    desired : scalar
        The expected object.
    significant : int, optional
        Desired precision, default is 7.
    err_msg : str, optional
        The error message to be printed in case of failure.
    verbose : bool, optional
        If True, the conflicting values are appended to the error message.

    Raises
    ------
    AssertionError
      If actual and desired are not equal up to specified precision.

    See Also
    --------
    assert_allclose: Compare two array_like objects for equality with desired
                     relative and/or absolute precision.
    assert_array_almost_equal_nulp, assert_array_max_ulp, assert_equal

    Examples
    --------
    >>> np.testing.assert_approx_equal(0.12345677777777e-20, 0.1234567e-20)  # doctest: +SKIP
    >>> np.testing.assert_approx_equal(0.12345670e-20, 0.12345671e-20,  # doctest: +SKIP
    ...                                significant=8)
    >>> np.testing.assert_approx_equal(0.12345670e-20, 0.12345672e-20,  # doctest: +SKIP
    ...                                significant=8)
    Traceback (most recent call last):
        ...
    AssertionError:
    Items are not equal to 8 significant digits:
     ACTUAL: 1.234567e-21
     DESIRED: 1.2345672e-21

    the evaluated condition that raises the exception is

    >>> abs(0.12345670e-20/1e-21 - 0.12345672e-20/1e-21) >= 10**-(8-1)
    True

    """
    __tracebackhide__ = True  # Hide traceback for py.test
    import numpy as np

    (actual, desired) = map(float, (actual, desired))
    if desired == actual:
        return
    # Normalized the numbers to be in range (-10.0,10.0)
    # scale = float(pow(10,math.floor(math.log10(0.5*(abs(desired)+abs(actual))))))
    scale = 0.5 * (np.abs(desired) + np.abs(actual))
    scale = np.power(10, np.floor(np.log10(scale)))
    try:
        sc_desired = desired / scale
    except ZeroDivisionError:
        sc_desired = 0.0
    try:
        sc_actual = actual / scale
    except ZeroDivisionError:
        sc_actual = 0.0
    msg = build_err_msg(
        [actual, desired],
        err_msg,
        header="Items are not equal to %d significant digits:" % significant,
        verbose=verbose,
    )
    try:
        # If one of desired/actual is not finite, handle it specially here:
        # check that both are nan if any is a nan, and test for equality
        # otherwise
        if not (gisfinite(desired) and gisfinite(actual)):
            if gisnan(desired) or gisnan(actual):
                if not (gisnan(desired) and gisnan(actual)):
                    raise AssertionError(msg)
            else:
                if not desired == actual:
                    raise AssertionError(msg)
            return
    except (TypeError, NotImplementedError):
        pass
    if np.abs(sc_desired - sc_actual) >= np.power(10.0, -(significant - 1)):
        raise AssertionError(msg)


def assert_array_compare(
    comparison,
    x,
    y,
    err_msg="",
    verbose=True,
    header="",
    precision=6,
    equal_nan=True,
    equal_inf=True,
    *,
    strict=False,
):
    __tracebackhide__ = True  # Hide traceback for py.test
    from torch._numpy import all, array, asarray, bool_, inf, isnan, max

    x = asarray(x)
    y = asarray(y)

    def array2string(a):
        return str(a)

    # original array for output formatting
    ox, oy = x, y

    def func_assert_same_pos(x, y, func=isnan, hasval="nan"):
        """Handling nan/inf.

        Combine results of running func on x and y, checking that they are True
        at the same locations.

        """
        __tracebackhide__ = True  # Hide traceback for py.test
        x_id = func(x)
        y_id = func(y)
        # We include work-arounds here to handle three types of slightly
        # pathological ndarray subclasses:
        # (1) all() on `masked` array scalars can return masked arrays, so we
        #     use != True
        # (2) __eq__ on some ndarray subclasses returns Python booleans
        #     instead of element-wise comparisons, so we cast to bool_() and
        #     use isinstance(..., bool) checks
        # (3) subclasses with bare-bones __array_function__ implementations may
        #     not implement np.all(), so favor using the .all() method
        # We are not committed to supporting such subclasses, but it's nice to
        # support them if possible.
        if (x_id == y_id).all().item() is not True:
            msg = build_err_msg(
                [x, y],
                err_msg + "\nx and y %s location mismatch:" % (hasval),
                verbose=verbose,
                header=header,
                names=("x", "y"),
                precision=precision,
            )
            raise AssertionError(msg)
        # If there is a scalar, then here we know the array has the same
        # flag as it everywhere, so we should return the scalar flag.
        if isinstance(x_id, bool) or x_id.ndim == 0:
            return bool_(x_id)
        elif isinstance(y_id, bool) or y_id.ndim == 0:
            return bool_(y_id)
        else:
            return y_id

    try:
        if strict:
            cond = x.shape == y.shape and x.dtype == y.dtype
        else:
            cond = (x.shape == () or y.shape == ()) or x.shape == y.shape
        if not cond:
            if x.shape != y.shape:
                reason = f"\n(shapes {x.shape}, {y.shape} mismatch)"
            else:
                reason = f"\n(dtypes {x.dtype}, {y.dtype} mismatch)"
            msg = build_err_msg(
                [x, y],
                err_msg + reason,
                verbose=verbose,
                header=header,
                names=("x", "y"),
                precision=precision,
            )
            raise AssertionError(msg)

        flagged = bool_(False)

        if equal_nan:
            flagged = func_assert_same_pos(x, y, func=isnan, hasval="nan")

        if equal_inf:
            flagged |= func_assert_same_pos(
                x, y, func=lambda xy: xy == +inf, hasval="+inf"
            )
            flagged |= func_assert_same_pos(
                x, y, func=lambda xy: xy == -inf, hasval="-inf"
            )

        if flagged.ndim > 0:
            x, y = x[~flagged], y[~flagged]
            # Only do the comparison if actual values are left
            if x.size == 0:
                return
        elif flagged:
            # no sense doing comparison if everything is flagged.
            return

        val = comparison(x, y)

        if isinstance(val, bool):
            cond = val
            reduced = array([val])
        else:
            reduced = val.ravel()
            cond = reduced.all()

        # The below comparison is a hack to ensure that fully masked
        # results, for which val.ravel().all() returns np.ma.masked,
        # do not trigger a failure (np.ma.masked != True evaluates as
        # np.ma.masked, which is falsy).
        if not cond:
            n_mismatch = reduced.size - int(reduced.sum(dtype=intp))
            n_elements = flagged.size if flagged.ndim != 0 else reduced.size
            percent_mismatch = 100 * n_mismatch / n_elements
            remarks = [
                f"Mismatched elements: {n_mismatch} / {n_elements} ({percent_mismatch:.3g}%)"
            ]

            # with errstate(all='ignore'):
            # ignore errors for non-numeric types
            with contextlib.suppress(TypeError, RuntimeError):
                error = abs(x - y)
                if np.issubdtype(x.dtype, np.unsignedinteger):
                    error2 = abs(y - x)
                    np.minimum(error, error2, out=error)
                max_abs_error = max(error)
                remarks.append(
                    "Max absolute difference: " + array2string(max_abs_error.item())
                )

                # note: this definition of relative error matches that one
                # used by assert_allclose (found in np.isclose)
                # Filter values where the divisor would be zero
                nonzero = bool_(y != 0)
                if all(~nonzero):
                    max_rel_error = array(inf)
                else:
                    max_rel_error = max(error[nonzero] / abs(y[nonzero]))
                remarks.append(
                    "Max relative difference: " + array2string(max_rel_error.item())
                )

            err_msg += "\n" + "\n".join(remarks)
            msg = build_err_msg(
                [ox, oy],
                err_msg,
                verbose=verbose,
                header=header,
                names=("x", "y"),
                precision=precision,
            )
            raise AssertionError(msg)
    except ValueError:
        import traceback

        efmt = traceback.format_exc()
        header = f"error during assertion:\n\n{efmt}\n\n{header}"

        msg = build_err_msg(
            [x, y],
            err_msg,
            verbose=verbose,
            header=header,
            names=("x", "y"),
            precision=precision,
        )
        raise ValueError(msg)  # noqa: TRY200


def assert_array_equal(x, y, err_msg="", verbose=True, *, strict=False):
    """
    Raises an AssertionError if two array_like objects are not equal.

    Given two array_like objects, check that the shape is equal and all
    elements of these objects are equal (but see the Notes for the special
    handling of a scalar). An exception is raised at shape mismatch or
    conflicting values. In contrast to the standard usage in numpy, NaNs
    are compared like numbers, no assertion is raised if both objects have
    NaNs in the same positions.

    The usual caution for verifying equality with floating point numbers is
    advised.

    Parameters
    ----------
    x : array_like
        The actual object to check.
    y : array_like
        The desired, expected object.
    err_msg : str, optional
        The error message to be printed in case of failure.
    verbose : bool, optional
        If True, the conflicting values are appended to the error message.
    strict : bool, optional
        If True, raise an AssertionError when either the shape or the data
        type of the array_like objects does not match. The special
        handling for scalars mentioned in the Notes section is disabled.

    Raises
    ------
    AssertionError
        If actual and desired objects are not equal.

    See Also
    --------
    assert_allclose: Compare two array_like objects for equality with desired
                     relative and/or absolute precision.
    assert_array_almost_equal_nulp, assert_array_max_ulp, assert_equal

    Notes
    -----
    When one of `x` and `y` is a scalar and the other is array_like, the
    function checks that each element of the array_like object is equal to
    the scalar. This behaviour can be disabled with the `strict` parameter.

    Examples
    --------
    The first assert does not raise an exception:

    >>> np.testing.assert_array_equal([1.0,2.33333,np.nan],
    ...                               [np.exp(0),2.33333, np.nan])

    Use `assert_allclose` or one of the nulp (number of floating point values)
    functions for these cases instead:

    >>> np.testing.assert_allclose([1.0,np.pi,np.nan],
    ...                            [1, np.sqrt(np.pi)**2, np.nan],
    ...                            rtol=1e-10, atol=0)

    As mentioned in the Notes section, `assert_array_equal` has special
    handling for scalars. Here the test checks that each value in `x` is 3:

    >>> x = np.full((2, 5), fill_value=3)
    >>> np.testing.assert_array_equal(x, 3)

    Use `strict` to raise an AssertionError when comparing a scalar with an
    array:

    >>> np.testing.assert_array_equal(x, 3, strict=True)
    Traceback (most recent call last):
        ...
    AssertionError:
    Arrays are not equal
    <BLANKLINE>
    (shapes (2, 5), () mismatch)
     x: torch.ndarray([[3, 3, 3, 3, 3],
            [3, 3, 3, 3, 3]])
     y: torch.ndarray(3)

    The `strict` parameter also ensures that the array data types match:

    >>> x = np.array([2, 2, 2])
    >>> y = np.array([2., 2., 2.], dtype=np.float32)
    >>> np.testing.assert_array_equal(x, y, strict=True)
    Traceback (most recent call last):
        ...
    AssertionError:
    Arrays are not equal
    <BLANKLINE>
    (dtypes dtype("int64"), dtype("float32") mismatch)
     x: torch.ndarray([2, 2, 2])
     y: torch.ndarray([2., 2., 2.])
    """
    __tracebackhide__ = True  # Hide traceback for py.test
    assert_array_compare(
        operator.__eq__,
        x,
        y,
        err_msg=err_msg,
        verbose=verbose,
        header="Arrays are not equal",
        strict=strict,
    )


def assert_array_almost_equal(x, y, decimal=6, err_msg="", verbose=True):
    """
    Raises an AssertionError if two objects are not equal up to desired
    precision.

    .. note:: It is recommended to use one of `assert_allclose`,
              `assert_array_almost_equal_nulp` or `assert_array_max_ulp`
              instead of this function for more consistent floating point
              comparisons.

    The test verifies identical shapes and that the elements of ``actual`` and
    ``desired`` satisfy.

        ``abs(desired-actual) < 1.5 * 10**(-decimal)``

    That is a looser test than originally documented, but agrees with what the
    actual implementation did up to rounding vagaries. An exception is raised
    at shape mismatch or conflicting values. In contrast to the standard usage
    in numpy, NaNs are compared like numbers, no assertion is raised if both
    objects have NaNs in the same positions.

    Parameters
    ----------
    x : array_like
        The actual object to check.
    y : array_like
        The desired, expected object.
    decimal : int, optional
        Desired precision, default is 6.
    err_msg : str, optional
      The error message to be printed in case of failure.
    verbose : bool, optional
        If True, the conflicting values are appended to the error message.

    Raises
    ------
    AssertionError
        If actual and desired are not equal up to specified precision.

    See Also
    --------
    assert_allclose: Compare two array_like objects for equality with desired
                     relative and/or absolute precision.
    assert_array_almost_equal_nulp, assert_array_max_ulp, assert_equal

    Examples
    --------
    the first assert does not raise an exception

    >>> np.testing.assert_array_almost_equal([1.0,2.333,np.nan],
    ...                                      [1.0,2.333,np.nan])

    >>> np.testing.assert_array_almost_equal([1.0,2.33333,np.nan],
    ...                                      [1.0,2.33339,np.nan], decimal=5)
    Traceback (most recent call last):
        ...
    AssertionError:
    Arrays are not almost equal to 5 decimals
    <BLANKLINE>
    Mismatched elements: 1 / 3 (33.3%)
    Max absolute difference: 5.999999999994898e-05
    Max relative difference: 2.5713661239633743e-05
     x: torch.ndarray([1.0000, 2.3333,    nan], dtype=float64)
     y: torch.ndarray([1.0000, 2.3334,    nan], dtype=float64)

    >>> np.testing.assert_array_almost_equal([1.0,2.33333,np.nan],
    ...                                      [1.0,2.33333, 5], decimal=5)
    Traceback (most recent call last):
        ...
    AssertionError:
    Arrays are not almost equal to 5 decimals
    <BLANKLINE>
    x and y nan location mismatch:
     x: torch.ndarray([1.0000, 2.3333,    nan], dtype=float64)
     y: torch.ndarray([1.0000, 2.3333, 5.0000], dtype=float64)

    """
    __tracebackhide__ = True  # Hide traceback for py.test
    from torch._numpy import any as npany, float_, issubdtype, number, result_type

    def compare(x, y):
        try:
            if npany(gisinf(x)) or npany(gisinf(y)):
                xinfid = gisinf(x)
                yinfid = gisinf(y)
                if not (xinfid == yinfid).all():
                    return False
                # if one item, x and y is +- inf
                if x.size == y.size == 1:
                    return x == y
                x = x[~xinfid]
                y = y[~yinfid]
        except (TypeError, NotImplementedError):
            pass

        # make sure y is an inexact type to avoid abs(MIN_INT); will cause
        # casting of x later.
        dtype = result_type(y, 1.0)
        y = asanyarray(y, dtype)
        z = abs(x - y)

        if not issubdtype(z.dtype, number):
            z = z.astype(float_)  # handle object arrays

        return z < 1.5 * 10.0 ** (-decimal)

    assert_array_compare(
        compare,
        x,
        y,
        err_msg=err_msg,
        verbose=verbose,
        header=("Arrays are not almost equal to %d decimals" % decimal),
        precision=decimal,
    )


def assert_array_less(x, y, err_msg="", verbose=True):
    """
    Raises an AssertionError if two array_like objects are not ordered by less
    than.

    Given two array_like objects, check that the shape is equal and all
    elements of the first object are strictly smaller than those of the
    second object. An exception is raised at shape mismatch or incorrectly
    ordered values. Shape mismatch does not raise if an object has zero
    dimension. In contrast to the standard usage in numpy, NaNs are
    compared, no assertion is raised if both objects have NaNs in the same
    positions.



    Parameters
    ----------
    x : array_like
      The smaller object to check.
    y : array_like
      The larger object to compare.
    err_msg : string
      The error message to be printed in case of failure.
    verbose : bool
        If True, the conflicting values are appended to the error message.

    Raises
    ------
    AssertionError
      If actual and desired objects are not equal.

    See Also
    --------
    assert_array_equal: tests objects for equality
    assert_array_almost_equal: test objects for equality up to precision



    Examples
    --------
    >>> np.testing.assert_array_less([1.0, 1.0, np.nan], [1.1, 2.0, np.nan])
    >>> np.testing.assert_array_less([1.0, 1.0, np.nan], [1, 2.0, np.nan])
    Traceback (most recent call last):
        ...
    AssertionError:
    Arrays are not less-ordered
    <BLANKLINE>
    Mismatched elements: 1 / 3 (33.3%)
    Max absolute difference: 1.0
    Max relative difference: 0.5
     x: torch.ndarray([1.,  1., nan], dtype=float64)
     y: torch.ndarray([1.,  2., nan], dtype=float64)

    >>> np.testing.assert_array_less([1.0, 4.0], 3)
    Traceback (most recent call last):
        ...
    AssertionError:
    Arrays are not less-ordered
    <BLANKLINE>
    Mismatched elements: 1 / 2 (50%)
    Max absolute difference: 2.0
    Max relative difference: 0.6666666666666666
     x: torch.ndarray([1., 4.], dtype=float64)
     y: torch.ndarray(3)

    >>> np.testing.assert_array_less([1.0, 2.0, 3.0], [4])
    Traceback (most recent call last):
        ...
    AssertionError:
    Arrays are not less-ordered
    <BLANKLINE>
    (shapes (3,), (1,) mismatch)
     x: torch.ndarray([1., 2., 3.], dtype=float64)
     y: torch.ndarray([4])

    """
    __tracebackhide__ = True  # Hide traceback for py.test
    assert_array_compare(
        operator.__lt__,
        x,
        y,
        err_msg=err_msg,
        verbose=verbose,
        header="Arrays are not less-ordered",
        equal_inf=False,
    )


def assert_string_equal(actual, desired):
    """
    Test if two strings are equal.

    If the given strings are equal, `assert_string_equal` does nothing.
    If they are not equal, an AssertionError is raised, and the diff
    between the strings is shown.

    Parameters
    ----------
    actual : str
        The string to test for equality against the expected string.
    desired : str
        The expected string.

    Examples
    --------
    >>> np.testing.assert_string_equal('abc', 'abc')  # doctest: +SKIP
    >>> np.testing.assert_string_equal('abc', 'abcd')  # doctest: +SKIP
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
    ...
    AssertionError: Differences in strings:
    - abc+ abcd?    +

    """
    # delay import of difflib to reduce startup time
    __tracebackhide__ = True  # Hide traceback for py.test
    import difflib

    if not isinstance(actual, str):
        raise AssertionError(repr(type(actual)))
    if not isinstance(desired, str):
        raise AssertionError(repr(type(desired)))
    if desired == actual:
        return

    diff = list(
        difflib.Differ().compare(actual.splitlines(True), desired.splitlines(True))
    )
    diff_list = []
    while diff:
        d1 = diff.pop(0)
        if d1.startswith("  "):
            continue
        if d1.startswith("- "):
            l = [d1]
            d2 = diff.pop(0)
            if d2.startswith("? "):
                l.append(d2)
                d2 = diff.pop(0)
            if not d2.startswith("+ "):
                raise AssertionError(repr(d2))
            l.append(d2)
            if diff:
                d3 = diff.pop(0)
                if d3.startswith("? "):
                    l.append(d3)
                else:
                    diff.insert(0, d3)
            if d2[2:] == d1[2:]:
                continue
            diff_list.extend(l)
            continue
        raise AssertionError(repr(d1))
    if not diff_list:
        return
    msg = f"Differences in strings:\n{''.join(diff_list).rstrip()}"
    if actual != desired:
        raise AssertionError(msg)


import unittest


class _Dummy(unittest.TestCase):
    def nop(self):
        pass


_d = _Dummy("nop")


def assert_raises_regex(exception_class, expected_regexp, *args, **kwargs):
    """
    assert_raises_regex(exception_class, expected_regexp, callable, *args,
                        **kwargs)
    assert_raises_regex(exception_class, expected_regexp)

    Fail unless an exception of class exception_class and with message that
    matches expected_regexp is thrown by callable when invoked with arguments
    args and keyword arguments kwargs.

    Alternatively, can be used as a context manager like `assert_raises`.

    Notes
    -----
    .. versionadded:: 1.9.0

    """
    __tracebackhide__ = True  # Hide traceback for py.test
    return _d.assertRaisesRegex(exception_class, expected_regexp, *args, **kwargs)


def decorate_methods(cls, decorator, testmatch=None):
    """
    Apply a decorator to all methods in a class matching a regular expression.

    The given decorator is applied to all public methods of `cls` that are
    matched by the regular expression `testmatch`
    (``testmatch.search(methodname)``). Methods that are private, i.e. start
    with an underscore, are ignored.

    Parameters
    ----------
    cls : class
        Class whose methods to decorate.
    decorator : function
        Decorator to apply to methods
    testmatch : compiled regexp or str, optional
        The regular expression. Default value is None, in which case the
        nose default (``re.compile(r'(?:^|[\\b_\\.%s-])[Tt]est' % os.sep)``)
        is used.
        If `testmatch` is a string, it is compiled to a regular expression
        first.

    """
    if testmatch is None:
        testmatch = re.compile(r"(?:^|[\\b_\\.%s-])[Tt]est" % os.sep)
    else:
        testmatch = re.compile(testmatch)
    cls_attr = cls.__dict__

    # delayed import to reduce startup time
    from inspect import isfunction

    methods = [_m for _m in cls_attr.values() if isfunction(_m)]
    for function in methods:
        try:
            if hasattr(function, "compat_func_name"):
                funcname = function.compat_func_name
            else:
                funcname = function.__name__
        except AttributeError:
            # not a function
            continue
        if testmatch.search(funcname) and not funcname.startswith("_"):
            setattr(cls, funcname, decorator(function))
    return


def _assert_valid_refcount(op):
    """
    Check that ufuncs don't mishandle refcount of object `1`.
    Used in a few regression tests.
    """
    if not HAS_REFCOUNT:
        return True

    import gc

    import numpy as np

    b = np.arange(100 * 100).reshape(100, 100)
    c = b
    i = 1

    gc.disable()
    try:
        rc = sys.getrefcount(i)
        for j in range(15):
            d = op(b, c)
        assert_(sys.getrefcount(i) >= rc)
    finally:
        gc.enable()
    del d  # for pyflakes


def assert_allclose(
    actual,
    desired,
    rtol=1e-7,
    atol=0,
    equal_nan=True,
    err_msg="",
    verbose=True,
    check_dtype=False,
):
    """
    Raises an AssertionError if two objects are not equal up to desired
    tolerance.

    Given two array_like objects, check that their shapes and all elements
    are equal (but see the Notes for the special handling of a scalar). An
    exception is raised if the shapes mismatch or any values conflict. In
    contrast to the standard usage in numpy, NaNs are compared like numbers,
    no assertion is raised if both objects have NaNs in the same positions.

    The test is equivalent to ``allclose(actual, desired, rtol, atol)`` (note
    that ``allclose`` has different default values). It compares the difference
    between `actual` and `desired` to ``atol + rtol * abs(desired)``.

    .. versionadded:: 1.5.0

    Parameters
    ----------
    actual : array_like
        Array obtained.
    desired : array_like
        Array desired.
    rtol : float, optional
        Relative tolerance.
    atol : float, optional
        Absolute tolerance.
    equal_nan : bool, optional.
        If True, NaNs will compare equal.
    err_msg : str, optional
        The error message to be printed in case of failure.
    verbose : bool, optional
        If True, the conflicting values are appended to the error message.

    Raises
    ------
    AssertionError
        If actual and desired are not equal up to specified precision.

    See Also
    --------
    assert_array_almost_equal_nulp, assert_array_max_ulp

    Notes
    -----
    When one of `actual` and `desired` is a scalar and the other is
    array_like, the function checks that each element of the array_like
    object is equal to the scalar.

    Examples
    --------
    >>> x = [1e-5, 1e-3, 1e-1]
    >>> y = np.arccos(np.cos(x))
    >>> np.testing.assert_allclose(x, y, rtol=1e-5, atol=0)

    """
    __tracebackhide__ = True  # Hide traceback for py.test

    def compare(x, y):
        return np.isclose(x, y, rtol=rtol, atol=atol, equal_nan=equal_nan)

    actual, desired = asanyarray(actual), asanyarray(desired)
    header = f"Not equal to tolerance rtol={rtol:g}, atol={atol:g}"

    if check_dtype:
        assert actual.dtype == desired.dtype

    assert_array_compare(
        compare,
        actual,
        desired,
        err_msg=str(err_msg),
        verbose=verbose,
        header=header,
        equal_nan=equal_nan,
    )


def assert_array_almost_equal_nulp(x, y, nulp=1):
    """
    Compare two arrays relatively to their spacing.

    This is a relatively robust method to compare two arrays whose amplitude
    is variable.

    Parameters
    ----------
    x, y : array_like
        Input arrays.
    nulp : int, optional
        The maximum number of unit in the last place for tolerance (see Notes).
        Default is 1.

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If the spacing between `x` and `y` for one or more elements is larger
        than `nulp`.

    See Also
    --------
    assert_array_max_ulp : Check that all items of arrays differ in at most
        N Units in the Last Place.
    spacing : Return the distance between x and the nearest adjacent number.

    Notes
    -----
    An assertion is raised if the following condition is not met::

        abs(x - y) <= nulp * spacing(maximum(abs(x), abs(y)))

    Examples
    --------
    >>> x = np.array([1., 1e-10, 1e-20])
    >>> eps = np.finfo(x.dtype).eps
    >>> np.testing.assert_array_almost_equal_nulp(x, x*eps/2 + x)  # doctest: +SKIP

    >>> np.testing.assert_array_almost_equal_nulp(x, x*eps + x)  # doctest: +SKIP
    Traceback (most recent call last):
      ...
    AssertionError: X and Y are not equal to 1 ULP (max is 2)

    """
    __tracebackhide__ = True  # Hide traceback for py.test
    import numpy as np

    ax = np.abs(x)
    ay = np.abs(y)
    ref = nulp * np.spacing(np.where(ax > ay, ax, ay))
    if not np.all(np.abs(x - y) <= ref):
        if np.iscomplexobj(x) or np.iscomplexobj(y):
            msg = "X and Y are not equal to %d ULP" % nulp
        else:
            max_nulp = np.max(nulp_diff(x, y))
            msg = "X and Y are not equal to %d ULP (max is %g)" % (nulp, max_nulp)
        raise AssertionError(msg)


def assert_array_max_ulp(a, b, maxulp=1, dtype=None):
    """
    Check that all items of arrays differ in at most N Units in the Last Place.

    Parameters
    ----------
    a, b : array_like
        Input arrays to be compared.
    maxulp : int, optional
        The maximum number of units in the last place that elements of `a` and
        `b` can differ. Default is 1.
    dtype : dtype, optional
        Data-type to convert `a` and `b` to if given. Default is None.

    Returns
    -------
    ret : ndarray
        Array containing number of representable floating point numbers between
        items in `a` and `b`.

    Raises
    ------
    AssertionError
        If one or more elements differ by more than `maxulp`.

    Notes
    -----
    For computing the ULP difference, this API does not differentiate between
    various representations of NAN (ULP difference between 0x7fc00000 and 0xffc00000
    is zero).

    See Also
    --------
    assert_array_almost_equal_nulp : Compare two arrays relatively to their
        spacing.

    Examples
    --------
    >>> a = np.linspace(0., 1., 100)
    >>> res = np.testing.assert_array_max_ulp(a, np.arcsin(np.sin(a)))  # doctest: +SKIP

    """
    __tracebackhide__ = True  # Hide traceback for py.test
    import numpy as np

    ret = nulp_diff(a, b, dtype)
    if not np.all(ret <= maxulp):
        raise AssertionError(
            f"Arrays are not almost equal up to {maxulp:g} "
            f"ULP (max difference is {np.max(ret):g} ULP)"
        )
    return ret


def nulp_diff(x, y, dtype=None):
    """For each item in x and y, return the number of representable floating
    points between them.

    Parameters
    ----------
    x : array_like
        first input array
    y : array_like
        second input array
    dtype : dtype, optional
        Data-type to convert `x` and `y` to if given. Default is None.

    Returns
    -------
    nulp : array_like
        number of representable floating point numbers between each item in x
        and y.

    Notes
    -----
    For computing the ULP difference, this API does not differentiate between
    various representations of NAN (ULP difference between 0x7fc00000 and 0xffc00000
    is zero).

    Examples
    --------
    # By definition, epsilon is the smallest number such as 1 + eps != 1, so
    # there should be exactly one ULP between 1 and 1 + eps
    >>> nulp_diff(1, 1 + np.finfo(x.dtype).eps)  # doctest: +SKIP
    1.0
    """
    import numpy as np

    if dtype:
        x = np.asarray(x, dtype=dtype)
        y = np.asarray(y, dtype=dtype)
    else:
        x = np.asarray(x)
        y = np.asarray(y)

    t = np.common_type(x, y)
    if np.iscomplexobj(x) or np.iscomplexobj(y):
        raise NotImplementedError("_nulp not implemented for complex array")

    x = np.array([x], dtype=t)
    y = np.array([y], dtype=t)

    x[np.isnan(x)] = np.nan
    y[np.isnan(y)] = np.nan

    if not x.shape == y.shape:
        raise ValueError(f"x and y do not have the same shape: {x.shape} - {y.shape}")

    def _diff(rx, ry, vdt):
        diff = np.asarray(rx - ry, dtype=vdt)
        return np.abs(diff)

    rx = integer_repr(x)
    ry = integer_repr(y)
    return _diff(rx, ry, t)


def _integer_repr(x, vdt, comp):
    # Reinterpret binary representation of the float as sign-magnitude:
    # take into account two-complement representation
    # See also
    # https://randomascii.wordpress.com/2012/02/25/comparing-floating-point-numbers-2012-edition/
    rx = x.view(vdt)
    if not (rx.size == 1):
        rx[rx < 0] = comp - rx[rx < 0]
    else:
        if rx < 0:
            rx = comp - rx

    return rx


def integer_repr(x):
    """Return the signed-magnitude interpretation of the binary representation
    of x."""
    import numpy as np

    if x.dtype == np.float16:
        return _integer_repr(x, np.int16, np.int16(-(2**15)))
    elif x.dtype == np.float32:
        return _integer_repr(x, np.int32, np.int32(-(2**31)))
    elif x.dtype == np.float64:
        return _integer_repr(x, np.int64, np.int64(-(2**63)))
    else:
        raise ValueError(f"Unsupported dtype {x.dtype}")


@contextlib.contextmanager
def _assert_warns_context(warning_class, name=None):
    __tracebackhide__ = True  # Hide traceback for py.test
    with suppress_warnings() as sup:
        l = sup.record(warning_class)
        yield
        if not len(l) > 0:
            name_str = f" when calling {name}" if name is not None else ""
            raise AssertionError("No warning raised" + name_str)


def assert_warns(warning_class, *args, **kwargs):
    """
    Fail unless the given callable throws the specified warning.

    A warning of class warning_class should be thrown by the callable when
    invoked with arguments args and keyword arguments kwargs.
    If a different type of warning is thrown, it will not be caught.

    If called with all arguments other than the warning class omitted, may be
    used as a context manager:

        with assert_warns(SomeWarning):
            do_something()

    The ability to be used as a context manager is new in NumPy v1.11.0.

    .. versionadded:: 1.4.0

    Parameters
    ----------
    warning_class : class
        The class defining the warning that `func` is expected to throw.
    func : callable, optional
        Callable to test
    *args : Arguments
        Arguments for `func`.
    **kwargs : Kwargs
        Keyword arguments for `func`.

    Returns
    -------
    The value returned by `func`.

    Examples
    --------
    >>> import warnings
    >>> def deprecated_func(num):
    ...     warnings.warn("Please upgrade", DeprecationWarning)
    ...     return num*num
    >>> with np.testing.assert_warns(DeprecationWarning):
    ...     assert deprecated_func(4) == 16
    >>> # or passing a func
    >>> ret = np.testing.assert_warns(DeprecationWarning, deprecated_func, 4)
    >>> assert ret == 16
    """
    if not args:
        return _assert_warns_context(warning_class)

    func = args[0]
    args = args[1:]
    with _assert_warns_context(warning_class, name=func.__name__):
        return func(*args, **kwargs)


@contextlib.contextmanager
def _assert_no_warnings_context(name=None):
    __tracebackhide__ = True  # Hide traceback for py.test
    with warnings.catch_warnings(record=True) as l:
        warnings.simplefilter("always")
        yield
        if len(l) > 0:
            name_str = f" when calling {name}" if name is not None else ""
            raise AssertionError(f"Got warnings{name_str}: {l}")


def assert_no_warnings(*args, **kwargs):
    """
    Fail if the given callable produces any warnings.

    If called with all arguments omitted, may be used as a context manager:

        with assert_no_warnings():
            do_something()

    The ability to be used as a context manager is new in NumPy v1.11.0.

    .. versionadded:: 1.7.0

    Parameters
    ----------
    func : callable
        The callable to test.
    \\*args : Arguments
        Arguments passed to `func`.
    \\*\\*kwargs : Kwargs
        Keyword arguments passed to `func`.

    Returns
    -------
    The value returned by `func`.

    """
    if not args:
        return _assert_no_warnings_context()

    func = args[0]
    args = args[1:]
    with _assert_no_warnings_context(name=func.__name__):
        return func(*args, **kwargs)


def _gen_alignment_data(dtype=float32, type="binary", max_size=24):
    """
    generator producing data with different alignment and offsets
    to test simd vectorization

    Parameters
    ----------
    dtype : dtype
        data type to produce
    type : string
        'unary': create data for unary operations, creates one input
                 and output array
        'binary': create data for unary operations, creates two input
                 and output array
    max_size : integer
        maximum size of data to produce

    Returns
    -------
    if type is 'unary' yields one output, one input array and a message
    containing information on the data
    if type is 'binary' yields one output array, two input array and a message
    containing information on the data

    """
    ufmt = "unary offset=(%d, %d), size=%d, dtype=%r, %s"
    bfmt = "binary offset=(%d, %d, %d), size=%d, dtype=%r, %s"
    for o in range(3):
        for s in range(o + 2, max(o + 3, max_size)):
            if type == "unary":

                def inp():
                    return arange(s, dtype=dtype)[o:]

                out = empty((s,), dtype=dtype)[o:]
                yield out, inp(), ufmt % (o, o, s, dtype, "out of place")
                d = inp()
                yield d, d, ufmt % (o, o, s, dtype, "in place")
                yield out[1:], inp()[:-1], ufmt % (
                    o + 1,
                    o,
                    s - 1,
                    dtype,
                    "out of place",
                )
                yield out[:-1], inp()[1:], ufmt % (
                    o,
                    o + 1,
                    s - 1,
                    dtype,
                    "out of place",
                )
                yield inp()[:-1], inp()[1:], ufmt % (o, o + 1, s - 1, dtype, "aliased")
                yield inp()[1:], inp()[:-1], ufmt % (o + 1, o, s - 1, dtype, "aliased")
            if type == "binary":

                def inp1():
                    return arange(s, dtype=dtype)[o:]

                inp2 = inp1
                out = empty((s,), dtype=dtype)[o:]
                yield out, inp1(), inp2(), bfmt % (o, o, o, s, dtype, "out of place")
                d = inp1()
                yield d, d, inp2(), bfmt % (o, o, o, s, dtype, "in place1")
                d = inp2()
                yield d, inp1(), d, bfmt % (o, o, o, s, dtype, "in place2")
                yield out[1:], inp1()[:-1], inp2()[:-1], bfmt % (
                    o + 1,
                    o,
                    o,
                    s - 1,
                    dtype,
                    "out of place",
                )
                yield out[:-1], inp1()[1:], inp2()[:-1], bfmt % (
                    o,
                    o + 1,
                    o,
                    s - 1,
                    dtype,
                    "out of place",
                )
                yield out[:-1], inp1()[:-1], inp2()[1:], bfmt % (
                    o,
                    o,
                    o + 1,
                    s - 1,
                    dtype,
                    "out of place",
                )
                yield inp1()[1:], inp1()[:-1], inp2()[:-1], bfmt % (
                    o + 1,
                    o,
                    o,
                    s - 1,
                    dtype,
                    "aliased",
                )
                yield inp1()[:-1], inp1()[1:], inp2()[:-1], bfmt % (
                    o,
                    o + 1,
                    o,
                    s - 1,
                    dtype,
                    "aliased",
                )
                yield inp1()[:-1], inp1()[:-1], inp2()[1:], bfmt % (
                    o,
                    o,
                    o + 1,
                    s - 1,
                    dtype,
                    "aliased",
                )


class IgnoreException(Exception):
    "Ignoring this exception due to disabled feature"


@contextlib.contextmanager
def tempdir(*args, **kwargs):
    """Context manager to provide a temporary test folder.

    All arguments are passed as this to the underlying tempfile.mkdtemp
    function.

    """
    tmpdir = mkdtemp(*args, **kwargs)
    try:
        yield tmpdir
    finally:
        shutil.rmtree(tmpdir)


@contextlib.contextmanager
def temppath(*args, **kwargs):
    """Context manager for temporary files.

    Context manager that returns the path to a closed temporary file. Its
    parameters are the same as for tempfile.mkstemp and are passed directly
    to that function. The underlying file is removed when the context is
    exited, so it should be closed at that time.

    Windows does not allow a temporary file to be opened if it is already
    open, so the underlying file must be closed after opening before it
    can be opened again.

    """
    fd, path = mkstemp(*args, **kwargs)
    os.close(fd)
    try:
        yield path
    finally:
        os.remove(path)


class clear_and_catch_warnings(warnings.catch_warnings):
    """Context manager that resets warning registry for catching warnings

    Warnings can be slippery, because, whenever a warning is triggered, Python
    adds a ``__warningregistry__`` member to the *calling* module.  This makes
    it impossible to retrigger the warning in this module, whatever you put in
    the warnings filters.  This context manager accepts a sequence of `modules`
    as a keyword argument to its constructor and:

    * stores and removes any ``__warningregistry__`` entries in given `modules`
      on entry;
    * resets ``__warningregistry__`` to its previous state on exit.

    This makes it possible to trigger any warning afresh inside the context
    manager without disturbing the state of warnings outside.

    For compatibility with Python 3.0, please consider all arguments to be
    keyword-only.

    Parameters
    ----------
    record : bool, optional
        Specifies whether warnings should be captured by a custom
        implementation of ``warnings.showwarning()`` and be appended to a list
        returned by the context manager. Otherwise None is returned by the
        context manager. The objects appended to the list are arguments whose
        attributes mirror the arguments to ``showwarning()``.
    modules : sequence, optional
        Sequence of modules for which to reset warnings registry on entry and
        restore on exit. To work correctly, all 'ignore' filters should
        filter by one of these modules.

    Examples
    --------
    >>> import warnings
    >>> with np.testing.clear_and_catch_warnings(  # doctest: +SKIP
    ...         modules=[np.core.fromnumeric]):
    ...     warnings.simplefilter('always')
    ...     warnings.filterwarnings('ignore', module='np.core.fromnumeric')
    ...     # do something that raises a warning but ignore those in
    ...     # np.core.fromnumeric
    """

    class_modules = ()

    def __init__(self, record=False, modules=()):
        self.modules = set(modules).union(self.class_modules)
        self._warnreg_copies = {}
        super().__init__(record=record)

    def __enter__(self):
        for mod in self.modules:
            if hasattr(mod, "__warningregistry__"):
                mod_reg = mod.__warningregistry__
                self._warnreg_copies[mod] = mod_reg.copy()
                mod_reg.clear()
        return super().__enter__()

    def __exit__(self, *exc_info):
        super().__exit__(*exc_info)
        for mod in self.modules:
            if hasattr(mod, "__warningregistry__"):
                mod.__warningregistry__.clear()
            if mod in self._warnreg_copies:
                mod.__warningregistry__.update(self._warnreg_copies[mod])


class suppress_warnings:
    """
    Context manager and decorator doing much the same as
    ``warnings.catch_warnings``.

    However, it also provides a filter mechanism to work around
    https://bugs.python.org/issue4180.

    This bug causes Python before 3.4 to not reliably show warnings again
    after they have been ignored once (even within catch_warnings). It
    means that no "ignore" filter can be used easily, since following
    tests might need to see the warning. Additionally it allows easier
    specificity for testing warnings and can be nested.

    Parameters
    ----------
    forwarding_rule : str, optional
        One of "always", "once", "module", or "location". Analogous to
        the usual warnings module filter mode, it is useful to reduce
        noise mostly on the outmost level. Unsuppressed and unrecorded
        warnings will be forwarded based on this rule. Defaults to "always".
        "location" is equivalent to the warnings "default", match by exact
        location the warning warning originated from.

    Notes
    -----
    Filters added inside the context manager will be discarded again
    when leaving it. Upon entering all filters defined outside a
    context will be applied automatically.

    When a recording filter is added, matching warnings are stored in the
    ``log`` attribute as well as in the list returned by ``record``.

    If filters are added and the ``module`` keyword is given, the
    warning registry of this module will additionally be cleared when
    applying it, entering the context, or exiting it. This could cause
    warnings to appear a second time after leaving the context if they
    were configured to be printed once (default) and were already
    printed before the context was entered.

    Nesting this context manager will work as expected when the
    forwarding rule is "always" (default). Unfiltered and unrecorded
    warnings will be passed out and be matched by the outer level.
    On the outmost level they will be printed (or caught by another
    warnings context). The forwarding rule argument can modify this
    behaviour.

    Like ``catch_warnings`` this context manager is not threadsafe.

    Examples
    --------

    With a context manager::

        with np.testing.suppress_warnings() as sup:
            sup.filter(DeprecationWarning, "Some text")
            sup.filter(module=np.ma.core)
            log = sup.record(FutureWarning, "Does this occur?")
            command_giving_warnings()
            # The FutureWarning was given once, the filtered warnings were
            # ignored. All other warnings abide outside settings (may be
            # printed/error)
            assert_(len(log) == 1)
            assert_(len(sup.log) == 1)  # also stored in log attribute

    Or as a decorator::

        sup = np.testing.suppress_warnings()
        sup.filter(module=np.ma.core)  # module must match exactly
        @sup
        def some_function():
            # do something which causes a warning in np.ma.core
            pass
    """

    def __init__(self, forwarding_rule="always"):
        self._entered = False

        # Suppressions are either instance or defined inside one with block:
        self._suppressions = []

        if forwarding_rule not in {"always", "module", "once", "location"}:
            raise ValueError("unsupported forwarding rule.")
        self._forwarding_rule = forwarding_rule

    def _clear_registries(self):
        if hasattr(warnings, "_filters_mutated"):
            # clearing the registry should not be necessary on new pythons,
            # instead the filters should be mutated.
            warnings._filters_mutated()
            return
        # Simply clear the registry, this should normally be harmless,
        # note that on new pythons it would be invalidated anyway.
        for module in self._tmp_modules:
            if hasattr(module, "__warningregistry__"):
                module.__warningregistry__.clear()

    def _filter(self, category=Warning, message="", module=None, record=False):
        if record:
            record = []  # The log where to store warnings
        else:
            record = None
        if self._entered:
            if module is None:
                warnings.filterwarnings("always", category=category, message=message)
            else:
                module_regex = module.__name__.replace(".", r"\.") + "$"
                warnings.filterwarnings(
                    "always", category=category, message=message, module=module_regex
                )
                self._tmp_modules.add(module)
                self._clear_registries()

            self._tmp_suppressions.append(
                (category, message, re.compile(message, re.I), module, record)
            )
        else:
            self._suppressions.append(
                (category, message, re.compile(message, re.I), module, record)
            )

        return record

    def filter(self, category=Warning, message="", module=None):
        """
        Add a new suppressing filter or apply it if the state is entered.

        Parameters
        ----------
        category : class, optional
            Warning class to filter
        message : string, optional
            Regular expression matching the warning message.
        module : module, optional
            Module to filter for. Note that the module (and its file)
            must match exactly and cannot be a submodule. This may make
            it unreliable for external modules.

        Notes
        -----
        When added within a context, filters are only added inside
        the context and will be forgotten when the context is exited.
        """
        self._filter(category=category, message=message, module=module, record=False)

    def record(self, category=Warning, message="", module=None):
        """
        Append a new recording filter or apply it if the state is entered.

        All warnings matching will be appended to the ``log`` attribute.

        Parameters
        ----------
        category : class, optional
            Warning class to filter
        message : string, optional
            Regular expression matching the warning message.
        module : module, optional
            Module to filter for. Note that the module (and its file)
            must match exactly and cannot be a submodule. This may make
            it unreliable for external modules.

        Returns
        -------
        log : list
            A list which will be filled with all matched warnings.

        Notes
        -----
        When added within a context, filters are only added inside
        the context and will be forgotten when the context is exited.
        """
        return self._filter(
            category=category, message=message, module=module, record=True
        )

    def __enter__(self):
        if self._entered:
            raise RuntimeError("cannot enter suppress_warnings twice.")

        self._orig_show = warnings.showwarning
        self._filters = warnings.filters
        warnings.filters = self._filters[:]

        self._entered = True
        self._tmp_suppressions = []
        self._tmp_modules = set()
        self._forwarded = set()

        self.log = []  # reset global log (no need to keep same list)

        for cat, mess, _, mod, log in self._suppressions:
            if log is not None:
                del log[:]  # clear the log
            if mod is None:
                warnings.filterwarnings("always", category=cat, message=mess)
            else:
                module_regex = mod.__name__.replace(".", r"\.") + "$"
                warnings.filterwarnings(
                    "always", category=cat, message=mess, module=module_regex
                )
                self._tmp_modules.add(mod)
        warnings.showwarning = self._showwarning
        self._clear_registries()

        return self

    def __exit__(self, *exc_info):
        warnings.showwarning = self._orig_show
        warnings.filters = self._filters
        self._clear_registries()
        self._entered = False
        del self._orig_show
        del self._filters

    def _showwarning(
        self, message, category, filename, lineno, *args, use_warnmsg=None, **kwargs
    ):
        for cat, _, pattern, mod, rec in (self._suppressions + self._tmp_suppressions)[
            ::-1
        ]:
            if issubclass(category, cat) and pattern.match(message.args[0]) is not None:
                if mod is None:
                    # Message and category match, either recorded or ignored
                    if rec is not None:
                        msg = WarningMessage(
                            message, category, filename, lineno, **kwargs
                        )
                        self.log.append(msg)
                        rec.append(msg)
                    return
                # Use startswith, because warnings strips the c or o from
                # .pyc/.pyo files.
                elif mod.__file__.startswith(filename):
                    # The message and module (filename) match
                    if rec is not None:
                        msg = WarningMessage(
                            message, category, filename, lineno, **kwargs
                        )
                        self.log.append(msg)
                        rec.append(msg)
                    return

        # There is no filter in place, so pass to the outside handler
        # unless we should only pass it once
        if self._forwarding_rule == "always":
            if use_warnmsg is None:
                self._orig_show(message, category, filename, lineno, *args, **kwargs)
            else:
                self._orig_showmsg(use_warnmsg)
            return

        if self._forwarding_rule == "once":
            signature = (message.args, category)
        elif self._forwarding_rule == "module":
            signature = (message.args, category, filename)
        elif self._forwarding_rule == "location":
            signature = (message.args, category, filename, lineno)

        if signature in self._forwarded:
            return
        self._forwarded.add(signature)
        if use_warnmsg is None:
            self._orig_show(message, category, filename, lineno, *args, **kwargs)
        else:
            self._orig_showmsg(use_warnmsg)

    def __call__(self, func):
        """
        Function decorator to apply certain suppressions to a whole
        function.
        """

        @wraps(func)
        def new_func(*args, **kwargs):
            with self:
                return func(*args, **kwargs)

        return new_func


@contextlib.contextmanager
def _assert_no_gc_cycles_context(name=None):
    __tracebackhide__ = True  # Hide traceback for py.test

    # not meaningful to test if there is no refcounting
    if not HAS_REFCOUNT:
        yield
        return

    assert_(gc.isenabled())
    gc.disable()
    gc_debug = gc.get_debug()
    try:
        for i in range(100):
            if gc.collect() == 0:
                break
        else:
            raise RuntimeError(
                "Unable to fully collect garbage - perhaps a __del__ method "
                "is creating more reference cycles?"
            )

        gc.set_debug(gc.DEBUG_SAVEALL)
        yield
        # gc.collect returns the number of unreachable objects in cycles that
        # were found -- we are checking that no cycles were created in the context
        n_objects_in_cycles = gc.collect()
        objects_in_cycles = gc.garbage[:]
    finally:
        del gc.garbage[:]
        gc.set_debug(gc_debug)
        gc.enable()

    if n_objects_in_cycles:
        name_str = f" when calling {name}" if name is not None else ""
        raise AssertionError(
            "Reference cycles were found{}: {} objects were collected, "
            "of which {} are shown below:{}".format(
                name_str,
                n_objects_in_cycles,
                len(objects_in_cycles),
                "".join(
                    "\n  {} object with id={}:\n    {}".format(
                        type(o).__name__,
                        id(o),
                        pprint.pformat(o).replace("\n", "\n    "),
                    )
                    for o in objects_in_cycles
                ),
            )
        )


def assert_no_gc_cycles(*args, **kwargs):
    """
    Fail if the given callable produces any reference cycles.

    If called with all arguments omitted, may be used as a context manager:

        with assert_no_gc_cycles():
            do_something()

    .. versionadded:: 1.15.0

    Parameters
    ----------
    func : callable
        The callable to test.
    \\*args : Arguments
        Arguments passed to `func`.
    \\*\\*kwargs : Kwargs
        Keyword arguments passed to `func`.

    Returns
    -------
    Nothing. The result is deliberately discarded to ensure that all cycles
    are found.

    """
    if not args:
        return _assert_no_gc_cycles_context()

    func = args[0]
    args = args[1:]
    with _assert_no_gc_cycles_context(name=func.__name__):
        func(*args, **kwargs)


def break_cycles():
    """
    Break reference cycles by calling gc.collect
    Objects can call other objects' methods (for instance, another object's
     __del__) inside their own __del__. On PyPy, the interpreter only runs
    between calls to gc.collect, so multiple calls are needed to completely
    release all cycles.
    """

    gc.collect()
    if IS_PYPY:
        # a few more, just to make sure all the finalizers are called
        gc.collect()
        gc.collect()
        gc.collect()
        gc.collect()


def requires_memory(free_bytes):
    """Decorator to skip a test if not enough memory is available"""
    import pytest

    def decorator(func):
        @wraps(func)
        def wrapper(*a, **kw):
            msg = check_free_memory(free_bytes)
            if msg is not None:
                pytest.skip(msg)

            try:
                return func(*a, **kw)
            except MemoryError:
                # Probably ran out of memory regardless: don't regard as failure
                pytest.xfail("MemoryError raised")

        return wrapper

    return decorator


def check_free_memory(free_bytes):
    """
    Check whether `free_bytes` amount of memory is currently free.
    Returns: None if enough memory available, otherwise error message
    """
    env_var = "NPY_AVAILABLE_MEM"
    env_value = os.environ.get(env_var)
    if env_value is not None:
        try:
            mem_free = _parse_size(env_value)
        except ValueError as exc:
            raise ValueError(  # noqa: TRY200
                f"Invalid environment variable {env_var}: {exc}"
            )

        msg = (
            f"{free_bytes/1e9} GB memory required, but environment variable "
            f"NPY_AVAILABLE_MEM={env_value} set"
        )
    else:
        mem_free = _get_mem_available()

        if mem_free is None:
            msg = (
                "Could not determine available memory; set NPY_AVAILABLE_MEM "
                "environment variable (e.g. NPY_AVAILABLE_MEM=16GB) to run "
                "the test."
            )
            mem_free = -1
        else:
            msg = (
                f"{free_bytes/1e9} GB memory required, but {mem_free/1e9} GB available"
            )

    return msg if mem_free < free_bytes else None


def _parse_size(size_str):
    """Convert memory size strings ('12 GB' etc.) to float"""
    suffixes = {
        "": 1,
        "b": 1,
        "k": 1000,
        "m": 1000**2,
        "g": 1000**3,
        "t": 1000**4,
        "kb": 1000,
        "mb": 1000**2,
        "gb": 1000**3,
        "tb": 1000**4,
        "kib": 1024,
        "mib": 1024**2,
        "gib": 1024**3,
        "tib": 1024**4,
    }

    size_re = re.compile(
        r"^\s*(\d+|\d+\.\d+)\s*({})\s*$".format("|".join(suffixes.keys())), re.I
    )

    m = size_re.match(size_str.lower())
    if not m or m.group(2) not in suffixes:
        raise ValueError(f"value {size_str!r} not a valid size")
    return int(float(m.group(1)) * suffixes[m.group(2)])


def _get_mem_available():
    """Return available memory in bytes, or None if unknown."""
    try:
        import psutil

        return psutil.virtual_memory().available
    except (ImportError, AttributeError):
        pass

    if sys.platform.startswith("linux"):
        info = {}
        with open("/proc/meminfo") as f:
            for line in f:
                p = line.split()
                info[p[0].strip(":").lower()] = int(p[1]) * 1024

        if "memavailable" in info:
            # Linux >= 3.14
            return info["memavailable"]
        else:
            return info["memfree"] + info["cached"]

    return None


def _no_tracing(func):
    """
    Decorator to temporarily turn off tracing for the duration of a test.
    Needed in tests that check refcounting, otherwise the tracing itself
    influences the refcounts
    """
    if not hasattr(sys, "gettrace"):
        return func
    else:

        @wraps(func)
        def wrapper(*args, **kwargs):
            original_trace = sys.gettrace()
            try:
                sys.settrace(None)
                return func(*args, **kwargs)
            finally:
                sys.settrace(original_trace)

        return wrapper


def _get_glibc_version():
    try:
        ver = os.confstr("CS_GNU_LIBC_VERSION").rsplit(" ")[1]
    except Exception as inst:
        ver = "0.0"

    return ver


_glibcver = _get_glibc_version()


def _glibc_older_than(x):
    return _glibcver != "0.0" and _glibcver < x
