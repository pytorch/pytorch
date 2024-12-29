# mypy: allow-untyped-defs
from __future__ import annotations

from collections.abc import Collection
from collections.abc import Sized
from decimal import Decimal
import math
from numbers import Complex
import pprint
import re
from types import TracebackType
from typing import Any
from typing import Callable
from typing import cast
from typing import ContextManager
from typing import final
from typing import Mapping
from typing import overload
from typing import Pattern
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar

import _pytest._code
from _pytest.outcomes import fail


if TYPE_CHECKING:
    from numpy import ndarray


def _compare_approx(
    full_object: object,
    message_data: Sequence[tuple[str, str, str]],
    number_of_elements: int,
    different_ids: Sequence[object],
    max_abs_diff: float,
    max_rel_diff: float,
) -> list[str]:
    message_list = list(message_data)
    message_list.insert(0, ("Index", "Obtained", "Expected"))
    max_sizes = [0, 0, 0]
    for index, obtained, expected in message_list:
        max_sizes[0] = max(max_sizes[0], len(index))
        max_sizes[1] = max(max_sizes[1], len(obtained))
        max_sizes[2] = max(max_sizes[2], len(expected))
    explanation = [
        f"comparison failed. Mismatched elements: {len(different_ids)} / {number_of_elements}:",
        f"Max absolute difference: {max_abs_diff}",
        f"Max relative difference: {max_rel_diff}",
    ] + [
        f"{indexes:<{max_sizes[0]}} | {obtained:<{max_sizes[1]}} | {expected:<{max_sizes[2]}}"
        for indexes, obtained, expected in message_list
    ]
    return explanation


# builtin pytest.approx helper


class ApproxBase:
    """Provide shared utilities for making approximate comparisons between
    numbers or sequences of numbers."""

    # Tell numpy to use our `__eq__` operator instead of its.
    __array_ufunc__ = None
    __array_priority__ = 100

    def __init__(self, expected, rel=None, abs=None, nan_ok: bool = False) -> None:
        __tracebackhide__ = True
        self.expected = expected
        self.abs = abs
        self.rel = rel
        self.nan_ok = nan_ok
        self._check_type()

    def __repr__(self) -> str:
        raise NotImplementedError

    def _repr_compare(self, other_side: Any) -> list[str]:
        return [
            "comparison failed",
            f"Obtained: {other_side}",
            f"Expected: {self}",
        ]

    def __eq__(self, actual) -> bool:
        return all(
            a == self._approx_scalar(x) for a, x in self._yield_comparisons(actual)
        )

    def __bool__(self):
        __tracebackhide__ = True
        raise AssertionError(
            "approx() is not supported in a boolean context.\nDid you mean: `assert a == approx(b)`?"
        )

    # Ignore type because of https://github.com/python/mypy/issues/4266.
    __hash__ = None  # type: ignore

    def __ne__(self, actual) -> bool:
        return not (actual == self)

    def _approx_scalar(self, x) -> ApproxScalar:
        if isinstance(x, Decimal):
            return ApproxDecimal(x, rel=self.rel, abs=self.abs, nan_ok=self.nan_ok)
        return ApproxScalar(x, rel=self.rel, abs=self.abs, nan_ok=self.nan_ok)

    def _yield_comparisons(self, actual):
        """Yield all the pairs of numbers to be compared.

        This is used to implement the `__eq__` method.
        """
        raise NotImplementedError

    def _check_type(self) -> None:
        """Raise a TypeError if the expected value is not a valid type."""
        # This is only a concern if the expected value is a sequence.  In every
        # other case, the approx() function ensures that the expected value has
        # a numeric type.  For this reason, the default is to do nothing.  The
        # classes that deal with sequences should reimplement this method to
        # raise if there are any non-numeric elements in the sequence.


def _recursive_sequence_map(f, x):
    """Recursively map a function over a sequence of arbitrary depth"""
    if isinstance(x, (list, tuple)):
        seq_type = type(x)
        return seq_type(_recursive_sequence_map(f, xi) for xi in x)
    elif _is_sequence_like(x):
        return [_recursive_sequence_map(f, xi) for xi in x]
    else:
        return f(x)


class ApproxNumpy(ApproxBase):
    """Perform approximate comparisons where the expected value is numpy array."""

    def __repr__(self) -> str:
        list_scalars = _recursive_sequence_map(
            self._approx_scalar, self.expected.tolist()
        )
        return f"approx({list_scalars!r})"

    def _repr_compare(self, other_side: ndarray | list[Any]) -> list[str]:
        import itertools
        import math

        def get_value_from_nested_list(
            nested_list: list[Any], nd_index: tuple[Any, ...]
        ) -> Any:
            """
            Helper function to get the value out of a nested list, given an n-dimensional index.
            This mimics numpy's indexing, but for raw nested python lists.
            """
            value: Any = nested_list
            for i in nd_index:
                value = value[i]
            return value

        np_array_shape = self.expected.shape
        approx_side_as_seq = _recursive_sequence_map(
            self._approx_scalar, self.expected.tolist()
        )

        # convert other_side to numpy array to ensure shape attribute is available
        other_side_as_array = _as_numpy_array(other_side)
        assert other_side_as_array is not None

        if np_array_shape != other_side_as_array.shape:
            return [
                "Impossible to compare arrays with different shapes.",
                f"Shapes: {np_array_shape} and {other_side_as_array.shape}",
            ]

        number_of_elements = self.expected.size
        max_abs_diff = -math.inf
        max_rel_diff = -math.inf
        different_ids = []
        for index in itertools.product(*(range(i) for i in np_array_shape)):
            approx_value = get_value_from_nested_list(approx_side_as_seq, index)
            other_value = get_value_from_nested_list(other_side_as_array, index)
            if approx_value != other_value:
                abs_diff = abs(approx_value.expected - other_value)
                max_abs_diff = max(max_abs_diff, abs_diff)
                if other_value == 0.0:
                    max_rel_diff = math.inf
                else:
                    max_rel_diff = max(max_rel_diff, abs_diff / abs(other_value))
                different_ids.append(index)

        message_data = [
            (
                str(index),
                str(get_value_from_nested_list(other_side_as_array, index)),
                str(get_value_from_nested_list(approx_side_as_seq, index)),
            )
            for index in different_ids
        ]
        return _compare_approx(
            self.expected,
            message_data,
            number_of_elements,
            different_ids,
            max_abs_diff,
            max_rel_diff,
        )

    def __eq__(self, actual) -> bool:
        import numpy as np

        # self.expected is supposed to always be an array here.

        if not np.isscalar(actual):
            try:
                actual = np.asarray(actual)
            except Exception as e:
                raise TypeError(f"cannot compare '{actual}' to numpy.ndarray") from e

        if not np.isscalar(actual) and actual.shape != self.expected.shape:
            return False

        return super().__eq__(actual)

    def _yield_comparisons(self, actual):
        import numpy as np

        # `actual` can either be a numpy array or a scalar, it is treated in
        # `__eq__` before being passed to `ApproxBase.__eq__`, which is the
        # only method that calls this one.

        if np.isscalar(actual):
            for i in np.ndindex(self.expected.shape):
                yield actual, self.expected[i].item()
        else:
            for i in np.ndindex(self.expected.shape):
                yield actual[i].item(), self.expected[i].item()


class ApproxMapping(ApproxBase):
    """Perform approximate comparisons where the expected value is a mapping
    with numeric values (the keys can be anything)."""

    def __repr__(self) -> str:
        return f"approx({({k: self._approx_scalar(v) for k, v in self.expected.items()})!r})"

    def _repr_compare(self, other_side: Mapping[object, float]) -> list[str]:
        import math

        approx_side_as_map = {
            k: self._approx_scalar(v) for k, v in self.expected.items()
        }

        number_of_elements = len(approx_side_as_map)
        max_abs_diff = -math.inf
        max_rel_diff = -math.inf
        different_ids = []
        for (approx_key, approx_value), other_value in zip(
            approx_side_as_map.items(), other_side.values()
        ):
            if approx_value != other_value:
                if approx_value.expected is not None and other_value is not None:
                    max_abs_diff = max(
                        max_abs_diff, abs(approx_value.expected - other_value)
                    )
                    if approx_value.expected == 0.0:
                        max_rel_diff = math.inf
                    else:
                        max_rel_diff = max(
                            max_rel_diff,
                            abs(
                                (approx_value.expected - other_value)
                                / approx_value.expected
                            ),
                        )
                different_ids.append(approx_key)

        message_data = [
            (str(key), str(other_side[key]), str(approx_side_as_map[key]))
            for key in different_ids
        ]

        return _compare_approx(
            self.expected,
            message_data,
            number_of_elements,
            different_ids,
            max_abs_diff,
            max_rel_diff,
        )

    def __eq__(self, actual) -> bool:
        try:
            if set(actual.keys()) != set(self.expected.keys()):
                return False
        except AttributeError:
            return False

        return super().__eq__(actual)

    def _yield_comparisons(self, actual):
        for k in self.expected.keys():
            yield actual[k], self.expected[k]

    def _check_type(self) -> None:
        __tracebackhide__ = True
        for key, value in self.expected.items():
            if isinstance(value, type(self.expected)):
                msg = "pytest.approx() does not support nested dictionaries: key={!r} value={!r}\n  full mapping={}"
                raise TypeError(msg.format(key, value, pprint.pformat(self.expected)))


class ApproxSequenceLike(ApproxBase):
    """Perform approximate comparisons where the expected value is a sequence of numbers."""

    def __repr__(self) -> str:
        seq_type = type(self.expected)
        if seq_type not in (tuple, list):
            seq_type = list
        return f"approx({seq_type(self._approx_scalar(x) for x in self.expected)!r})"

    def _repr_compare(self, other_side: Sequence[float]) -> list[str]:
        import math

        if len(self.expected) != len(other_side):
            return [
                "Impossible to compare lists with different sizes.",
                f"Lengths: {len(self.expected)} and {len(other_side)}",
            ]

        approx_side_as_map = _recursive_sequence_map(self._approx_scalar, self.expected)

        number_of_elements = len(approx_side_as_map)
        max_abs_diff = -math.inf
        max_rel_diff = -math.inf
        different_ids = []
        for i, (approx_value, other_value) in enumerate(
            zip(approx_side_as_map, other_side)
        ):
            if approx_value != other_value:
                abs_diff = abs(approx_value.expected - other_value)
                max_abs_diff = max(max_abs_diff, abs_diff)
                if other_value == 0.0:
                    max_rel_diff = math.inf
                else:
                    max_rel_diff = max(max_rel_diff, abs_diff / abs(other_value))
                different_ids.append(i)

        message_data = [
            (str(i), str(other_side[i]), str(approx_side_as_map[i]))
            for i in different_ids
        ]

        return _compare_approx(
            self.expected,
            message_data,
            number_of_elements,
            different_ids,
            max_abs_diff,
            max_rel_diff,
        )

    def __eq__(self, actual) -> bool:
        try:
            if len(actual) != len(self.expected):
                return False
        except TypeError:
            return False
        return super().__eq__(actual)

    def _yield_comparisons(self, actual):
        return zip(actual, self.expected)

    def _check_type(self) -> None:
        __tracebackhide__ = True
        for index, x in enumerate(self.expected):
            if isinstance(x, type(self.expected)):
                msg = "pytest.approx() does not support nested data structures: {!r} at index {}\n  full sequence: {}"
                raise TypeError(msg.format(x, index, pprint.pformat(self.expected)))


class ApproxScalar(ApproxBase):
    """Perform approximate comparisons where the expected value is a single number."""

    # Using Real should be better than this Union, but not possible yet:
    # https://github.com/python/typeshed/pull/3108
    DEFAULT_ABSOLUTE_TOLERANCE: float | Decimal = 1e-12
    DEFAULT_RELATIVE_TOLERANCE: float | Decimal = 1e-6

    def __repr__(self) -> str:
        """Return a string communicating both the expected value and the
        tolerance for the comparison being made.

        For example, ``1.0 ± 1e-6``, ``(3+4j) ± 5e-6 ∠ ±180°``.
        """
        # Don't show a tolerance for values that aren't compared using
        # tolerances, i.e. non-numerics and infinities. Need to call abs to
        # handle complex numbers, e.g. (inf + 1j).
        if (not isinstance(self.expected, (Complex, Decimal))) or math.isinf(
            abs(self.expected)
        ):
            return str(self.expected)

        # If a sensible tolerance can't be calculated, self.tolerance will
        # raise a ValueError.  In this case, display '???'.
        try:
            vetted_tolerance = f"{self.tolerance:.1e}"
            if (
                isinstance(self.expected, Complex)
                and self.expected.imag
                and not math.isinf(self.tolerance)
            ):
                vetted_tolerance += " ∠ ±180°"
        except ValueError:
            vetted_tolerance = "???"

        return f"{self.expected} ± {vetted_tolerance}"

    def __eq__(self, actual) -> bool:
        """Return whether the given value is equal to the expected value
        within the pre-specified tolerance."""
        asarray = _as_numpy_array(actual)
        if asarray is not None:
            # Call ``__eq__()`` manually to prevent infinite-recursion with
            # numpy<1.13.  See #3748.
            return all(self.__eq__(a) for a in asarray.flat)

        # Short-circuit exact equality.
        if actual == self.expected:
            return True

        # If either type is non-numeric, fall back to strict equality.
        # NB: we need Complex, rather than just Number, to ensure that __abs__,
        # __sub__, and __float__ are defined.
        if not (
            isinstance(self.expected, (Complex, Decimal))
            and isinstance(actual, (Complex, Decimal))
        ):
            return False

        # Allow the user to control whether NaNs are considered equal to each
        # other or not.  The abs() calls are for compatibility with complex
        # numbers.
        if math.isnan(abs(self.expected)):
            return self.nan_ok and math.isnan(abs(actual))

        # Infinity shouldn't be approximately equal to anything but itself, but
        # if there's a relative tolerance, it will be infinite and infinity
        # will seem approximately equal to everything.  The equal-to-itself
        # case would have been short circuited above, so here we can just
        # return false if the expected value is infinite.  The abs() call is
        # for compatibility with complex numbers.
        if math.isinf(abs(self.expected)):
            return False

        # Return true if the two numbers are within the tolerance.
        result: bool = abs(self.expected - actual) <= self.tolerance
        return result

    # Ignore type because of https://github.com/python/mypy/issues/4266.
    __hash__ = None  # type: ignore

    @property
    def tolerance(self):
        """Return the tolerance for the comparison.

        This could be either an absolute tolerance or a relative tolerance,
        depending on what the user specified or which would be larger.
        """

        def set_default(x, default):
            return x if x is not None else default

        # Figure out what the absolute tolerance should be.  ``self.abs`` is
        # either None or a value specified by the user.
        absolute_tolerance = set_default(self.abs, self.DEFAULT_ABSOLUTE_TOLERANCE)

        if absolute_tolerance < 0:
            raise ValueError(
                f"absolute tolerance can't be negative: {absolute_tolerance}"
            )
        if math.isnan(absolute_tolerance):
            raise ValueError("absolute tolerance can't be NaN.")

        # If the user specified an absolute tolerance but not a relative one,
        # just return the absolute tolerance.
        if self.rel is None:
            if self.abs is not None:
                return absolute_tolerance

        # Figure out what the relative tolerance should be.  ``self.rel`` is
        # either None or a value specified by the user.  This is done after
        # we've made sure the user didn't ask for an absolute tolerance only,
        # because we don't want to raise errors about the relative tolerance if
        # we aren't even going to use it.
        relative_tolerance = set_default(
            self.rel, self.DEFAULT_RELATIVE_TOLERANCE
        ) * abs(self.expected)

        if relative_tolerance < 0:
            raise ValueError(
                f"relative tolerance can't be negative: {relative_tolerance}"
            )
        if math.isnan(relative_tolerance):
            raise ValueError("relative tolerance can't be NaN.")

        # Return the larger of the relative and absolute tolerances.
        return max(relative_tolerance, absolute_tolerance)


class ApproxDecimal(ApproxScalar):
    """Perform approximate comparisons where the expected value is a Decimal."""

    DEFAULT_ABSOLUTE_TOLERANCE = Decimal("1e-12")
    DEFAULT_RELATIVE_TOLERANCE = Decimal("1e-6")


def approx(expected, rel=None, abs=None, nan_ok: bool = False) -> ApproxBase:
    """Assert that two numbers (or two ordered sequences of numbers) are equal to each other
    within some tolerance.

    Due to the :doc:`python:tutorial/floatingpoint`, numbers that we
    would intuitively expect to be equal are not always so::

        >>> 0.1 + 0.2 == 0.3
        False

    This problem is commonly encountered when writing tests, e.g. when making
    sure that floating-point values are what you expect them to be.  One way to
    deal with this problem is to assert that two floating-point numbers are
    equal to within some appropriate tolerance::

        >>> abs((0.1 + 0.2) - 0.3) < 1e-6
        True

    However, comparisons like this are tedious to write and difficult to
    understand.  Furthermore, absolute comparisons like the one above are
    usually discouraged because there's no tolerance that works well for all
    situations.  ``1e-6`` is good for numbers around ``1``, but too small for
    very big numbers and too big for very small ones.  It's better to express
    the tolerance as a fraction of the expected value, but relative comparisons
    like that are even more difficult to write correctly and concisely.

    The ``approx`` class performs floating-point comparisons using a syntax
    that's as intuitive as possible::

        >>> from pytest import approx
        >>> 0.1 + 0.2 == approx(0.3)
        True

    The same syntax also works for ordered sequences of numbers::

        >>> (0.1 + 0.2, 0.2 + 0.4) == approx((0.3, 0.6))
        True

    ``numpy`` arrays::

        >>> import numpy as np                                                          # doctest: +SKIP
        >>> np.array([0.1, 0.2]) + np.array([0.2, 0.4]) == approx(np.array([0.3, 0.6])) # doctest: +SKIP
        True

    And for a ``numpy`` array against a scalar::

        >>> import numpy as np                                         # doctest: +SKIP
        >>> np.array([0.1, 0.2]) + np.array([0.2, 0.1]) == approx(0.3) # doctest: +SKIP
        True

    Only ordered sequences are supported, because ``approx`` needs
    to infer the relative position of the sequences without ambiguity. This means
    ``sets`` and other unordered sequences are not supported.

    Finally, dictionary *values* can also be compared::

        >>> {'a': 0.1 + 0.2, 'b': 0.2 + 0.4} == approx({'a': 0.3, 'b': 0.6})
        True

    The comparison will be true if both mappings have the same keys and their
    respective values match the expected tolerances.

    **Tolerances**

    By default, ``approx`` considers numbers within a relative tolerance of
    ``1e-6`` (i.e. one part in a million) of its expected value to be equal.
    This treatment would lead to surprising results if the expected value was
    ``0.0``, because nothing but ``0.0`` itself is relatively close to ``0.0``.
    To handle this case less surprisingly, ``approx`` also considers numbers
    within an absolute tolerance of ``1e-12`` of its expected value to be
    equal.  Infinity and NaN are special cases.  Infinity is only considered
    equal to itself, regardless of the relative tolerance.  NaN is not
    considered equal to anything by default, but you can make it be equal to
    itself by setting the ``nan_ok`` argument to True.  (This is meant to
    facilitate comparing arrays that use NaN to mean "no data".)

    Both the relative and absolute tolerances can be changed by passing
    arguments to the ``approx`` constructor::

        >>> 1.0001 == approx(1)
        False
        >>> 1.0001 == approx(1, rel=1e-3)
        True
        >>> 1.0001 == approx(1, abs=1e-3)
        True

    If you specify ``abs`` but not ``rel``, the comparison will not consider
    the relative tolerance at all.  In other words, two numbers that are within
    the default relative tolerance of ``1e-6`` will still be considered unequal
    if they exceed the specified absolute tolerance.  If you specify both
    ``abs`` and ``rel``, the numbers will be considered equal if either
    tolerance is met::

        >>> 1 + 1e-8 == approx(1)
        True
        >>> 1 + 1e-8 == approx(1, abs=1e-12)
        False
        >>> 1 + 1e-8 == approx(1, rel=1e-6, abs=1e-12)
        True

    You can also use ``approx`` to compare nonnumeric types, or dicts and
    sequences containing nonnumeric types, in which case it falls back to
    strict equality. This can be useful for comparing dicts and sequences that
    can contain optional values::

        >>> {"required": 1.0000005, "optional": None} == approx({"required": 1, "optional": None})
        True
        >>> [None, 1.0000005] == approx([None,1])
        True
        >>> ["foo", 1.0000005] == approx([None,1])
        False

    If you're thinking about using ``approx``, then you might want to know how
    it compares to other good ways of comparing floating-point numbers.  All of
    these algorithms are based on relative and absolute tolerances and should
    agree for the most part, but they do have meaningful differences:

    - ``math.isclose(a, b, rel_tol=1e-9, abs_tol=0.0)``:  True if the relative
      tolerance is met w.r.t. either ``a`` or ``b`` or if the absolute
      tolerance is met.  Because the relative tolerance is calculated w.r.t.
      both ``a`` and ``b``, this test is symmetric (i.e.  neither ``a`` nor
      ``b`` is a "reference value").  You have to specify an absolute tolerance
      if you want to compare to ``0.0`` because there is no tolerance by
      default.  More information: :py:func:`math.isclose`.

    - ``numpy.isclose(a, b, rtol=1e-5, atol=1e-8)``: True if the difference
      between ``a`` and ``b`` is less that the sum of the relative tolerance
      w.r.t. ``b`` and the absolute tolerance.  Because the relative tolerance
      is only calculated w.r.t. ``b``, this test is asymmetric and you can
      think of ``b`` as the reference value.  Support for comparing sequences
      is provided by :py:func:`numpy.allclose`.  More information:
      :std:doc:`numpy:reference/generated/numpy.isclose`.

    - ``unittest.TestCase.assertAlmostEqual(a, b)``: True if ``a`` and ``b``
      are within an absolute tolerance of ``1e-7``.  No relative tolerance is
      considered , so this function is not appropriate for very large or very
      small numbers.  Also, it's only available in subclasses of ``unittest.TestCase``
      and it's ugly because it doesn't follow PEP8.  More information:
      :py:meth:`unittest.TestCase.assertAlmostEqual`.

    - ``a == pytest.approx(b, rel=1e-6, abs=1e-12)``: True if the relative
      tolerance is met w.r.t. ``b`` or if the absolute tolerance is met.
      Because the relative tolerance is only calculated w.r.t. ``b``, this test
      is asymmetric and you can think of ``b`` as the reference value.  In the
      special case that you explicitly specify an absolute tolerance but not a
      relative tolerance, only the absolute tolerance is considered.

    .. note::

        ``approx`` can handle numpy arrays, but we recommend the
        specialised test helpers in :std:doc:`numpy:reference/routines.testing`
        if you need support for comparisons, NaNs, or ULP-based tolerances.

        To match strings using regex, you can use
        `Matches <https://github.com/asottile/re-assert#re_assertmatchespattern-str-args-kwargs>`_
        from the
        `re_assert package <https://github.com/asottile/re-assert>`_.

    .. warning::

       .. versionchanged:: 3.2

       In order to avoid inconsistent behavior, :py:exc:`TypeError` is
       raised for ``>``, ``>=``, ``<`` and ``<=`` comparisons.
       The example below illustrates the problem::

           assert approx(0.1) > 0.1 + 1e-10  # calls approx(0.1).__gt__(0.1 + 1e-10)
           assert 0.1 + 1e-10 > approx(0.1)  # calls approx(0.1).__lt__(0.1 + 1e-10)

       In the second example one expects ``approx(0.1).__le__(0.1 + 1e-10)``
       to be called. But instead, ``approx(0.1).__lt__(0.1 + 1e-10)`` is used to
       comparison. This is because the call hierarchy of rich comparisons
       follows a fixed behavior. More information: :py:meth:`object.__ge__`

    .. versionchanged:: 3.7.1
       ``approx`` raises ``TypeError`` when it encounters a dict value or
       sequence element of nonnumeric type.

    .. versionchanged:: 6.1.0
       ``approx`` falls back to strict equality for nonnumeric types instead
       of raising ``TypeError``.
    """
    # Delegate the comparison to a class that knows how to deal with the type
    # of the expected value (e.g. int, float, list, dict, numpy.array, etc).
    #
    # The primary responsibility of these classes is to implement ``__eq__()``
    # and ``__repr__()``.  The former is used to actually check if some
    # "actual" value is equivalent to the given expected value within the
    # allowed tolerance.  The latter is used to show the user the expected
    # value and tolerance, in the case that a test failed.
    #
    # The actual logic for making approximate comparisons can be found in
    # ApproxScalar, which is used to compare individual numbers.  All of the
    # other Approx classes eventually delegate to this class.  The ApproxBase
    # class provides some convenient methods and overloads, but isn't really
    # essential.

    __tracebackhide__ = True

    if isinstance(expected, Decimal):
        cls: type[ApproxBase] = ApproxDecimal
    elif isinstance(expected, Mapping):
        cls = ApproxMapping
    elif _is_numpy_array(expected):
        expected = _as_numpy_array(expected)
        cls = ApproxNumpy
    elif _is_sequence_like(expected):
        cls = ApproxSequenceLike
    elif isinstance(expected, Collection) and not isinstance(expected, (str, bytes)):
        msg = f"pytest.approx() only supports ordered sequences, but got: {expected!r}"
        raise TypeError(msg)
    else:
        cls = ApproxScalar

    return cls(expected, rel, abs, nan_ok)


def _is_sequence_like(expected: object) -> bool:
    return (
        hasattr(expected, "__getitem__")
        and isinstance(expected, Sized)
        and not isinstance(expected, (str, bytes))
    )


def _is_numpy_array(obj: object) -> bool:
    """
    Return true if the given object is implicitly convertible to ndarray,
    and numpy is already imported.
    """
    return _as_numpy_array(obj) is not None


def _as_numpy_array(obj: object) -> ndarray | None:
    """
    Return an ndarray if the given object is implicitly convertible to ndarray,
    and numpy is already imported, otherwise None.
    """
    import sys

    np: Any = sys.modules.get("numpy")
    if np is not None:
        # avoid infinite recursion on numpy scalars, which have __array__
        if np.isscalar(obj):
            return None
        elif isinstance(obj, np.ndarray):
            return obj
        elif hasattr(obj, "__array__") or hasattr("obj", "__array_interface__"):
            return np.asarray(obj)
    return None


# builtin pytest.raises helper

E = TypeVar("E", bound=BaseException)


@overload
def raises(
    expected_exception: type[E] | tuple[type[E], ...],
    *,
    match: str | Pattern[str] | None = ...,
) -> RaisesContext[E]: ...


@overload
def raises(
    expected_exception: type[E] | tuple[type[E], ...],
    func: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> _pytest._code.ExceptionInfo[E]: ...


def raises(
    expected_exception: type[E] | tuple[type[E], ...], *args: Any, **kwargs: Any
) -> RaisesContext[E] | _pytest._code.ExceptionInfo[E]:
    r"""Assert that a code block/function call raises an exception type, or one of its subclasses.

    :param expected_exception:
        The expected exception type, or a tuple if one of multiple possible
        exception types are expected. Note that subclasses of the passed exceptions
        will also match.

    :kwparam str | re.Pattern[str] | None match:
        If specified, a string containing a regular expression,
        or a regular expression object, that is tested against the string
        representation of the exception and its :pep:`678` `__notes__`
        using :func:`re.search`.

        To match a literal string that may contain :ref:`special characters
        <re-syntax>`, the pattern can first be escaped with :func:`re.escape`.

        (This is only used when ``pytest.raises`` is used as a context manager,
        and passed through to the function otherwise.
        When using ``pytest.raises`` as a function, you can use:
        ``pytest.raises(Exc, func, match="passed on").match("my pattern")``.)

    Use ``pytest.raises`` as a context manager, which will capture the exception of the given
    type, or any of its subclasses::

        >>> import pytest
        >>> with pytest.raises(ZeroDivisionError):
        ...    1/0

    If the code block does not raise the expected exception (:class:`ZeroDivisionError` in the example
    above), or no exception at all, the check will fail instead.

    You can also use the keyword argument ``match`` to assert that the
    exception matches a text or regex::

        >>> with pytest.raises(ValueError, match='must be 0 or None'):
        ...     raise ValueError("value must be 0 or None")

        >>> with pytest.raises(ValueError, match=r'must be \d+$'):
        ...     raise ValueError("value must be 42")

    The ``match`` argument searches the formatted exception string, which includes any
    `PEP-678 <https://peps.python.org/pep-0678/>`__ ``__notes__``:

        >>> with pytest.raises(ValueError, match=r"had a note added"):  # doctest: +SKIP
        ...     e = ValueError("value must be 42")
        ...     e.add_note("had a note added")
        ...     raise e

    The context manager produces an :class:`ExceptionInfo` object which can be used to inspect the
    details of the captured exception::

        >>> with pytest.raises(ValueError) as exc_info:
        ...     raise ValueError("value must be 42")
        >>> assert exc_info.type is ValueError
        >>> assert exc_info.value.args[0] == "value must be 42"

    .. warning::

       Given that ``pytest.raises`` matches subclasses, be wary of using it to match :class:`Exception` like this::

           with pytest.raises(Exception):  # Careful, this will catch ANY exception raised.
               some_function()

       Because :class:`Exception` is the base class of almost all exceptions, it is easy for this to hide
       real bugs, where the user wrote this expecting a specific exception, but some other exception is being
       raised due to a bug introduced during a refactoring.

       Avoid using ``pytest.raises`` to catch :class:`Exception` unless certain that you really want to catch
       **any** exception raised.

    .. note::

       When using ``pytest.raises`` as a context manager, it's worthwhile to
       note that normal context manager rules apply and that the exception
       raised *must* be the final line in the scope of the context manager.
       Lines of code after that, within the scope of the context manager will
       not be executed. For example::

           >>> value = 15
           >>> with pytest.raises(ValueError) as exc_info:
           ...     if value > 10:
           ...         raise ValueError("value must be <= 10")
           ...     assert exc_info.type is ValueError  # This will not execute.

       Instead, the following approach must be taken (note the difference in
       scope)::

           >>> with pytest.raises(ValueError) as exc_info:
           ...     if value > 10:
           ...         raise ValueError("value must be <= 10")
           ...
           >>> assert exc_info.type is ValueError

    **Using with** ``pytest.mark.parametrize``

    When using :ref:`pytest.mark.parametrize ref`
    it is possible to parametrize tests such that
    some runs raise an exception and others do not.

    See :ref:`parametrizing_conditional_raising` for an example.

    .. seealso::

        :ref:`assertraises` for more examples and detailed discussion.

    **Legacy form**

    It is possible to specify a callable by passing a to-be-called lambda::

        >>> raises(ZeroDivisionError, lambda: 1/0)
        <ExceptionInfo ...>

    or you can specify an arbitrary callable with arguments::

        >>> def f(x): return 1/x
        ...
        >>> raises(ZeroDivisionError, f, 0)
        <ExceptionInfo ...>
        >>> raises(ZeroDivisionError, f, x=0)
        <ExceptionInfo ...>

    The form above is fully supported but discouraged for new code because the
    context manager form is regarded as more readable and less error-prone.

    .. note::
        Similar to caught exception objects in Python, explicitly clearing
        local references to returned ``ExceptionInfo`` objects can
        help the Python interpreter speed up its garbage collection.

        Clearing those references breaks a reference cycle
        (``ExceptionInfo`` --> caught exception --> frame stack raising
        the exception --> current frame stack --> local variables -->
        ``ExceptionInfo``) which makes Python keep all objects referenced
        from that cycle (including all local variables in the current
        frame) alive until the next cyclic garbage collection run.
        More detailed information can be found in the official Python
        documentation for :ref:`the try statement <python:try>`.
    """
    __tracebackhide__ = True

    if not expected_exception:
        raise ValueError(
            f"Expected an exception type or a tuple of exception types, but got `{expected_exception!r}`. "
            f"Raising exceptions is already understood as failing the test, so you don't need "
            f"any special code to say 'this should never raise an exception'."
        )
    if isinstance(expected_exception, type):
        expected_exceptions: tuple[type[E], ...] = (expected_exception,)
    else:
        expected_exceptions = expected_exception
    for exc in expected_exceptions:
        if not isinstance(exc, type) or not issubclass(exc, BaseException):
            msg = "expected exception must be a BaseException type, not {}"  # type: ignore[unreachable]
            not_a = exc.__name__ if isinstance(exc, type) else type(exc).__name__
            raise TypeError(msg.format(not_a))

    message = f"DID NOT RAISE {expected_exception}"

    if not args:
        match: str | Pattern[str] | None = kwargs.pop("match", None)
        if kwargs:
            msg = "Unexpected keyword arguments passed to pytest.raises: "
            msg += ", ".join(sorted(kwargs))
            msg += "\nUse context-manager form instead?"
            raise TypeError(msg)
        return RaisesContext(expected_exception, message, match)
    else:
        func = args[0]
        if not callable(func):
            raise TypeError(f"{func!r} object (type: {type(func)}) must be callable")
        try:
            func(*args[1:], **kwargs)
        except expected_exception as e:
            return _pytest._code.ExceptionInfo.from_exception(e)
    fail(message)


# This doesn't work with mypy for now. Use fail.Exception instead.
raises.Exception = fail.Exception  # type: ignore


@final
class RaisesContext(ContextManager[_pytest._code.ExceptionInfo[E]]):
    def __init__(
        self,
        expected_exception: type[E] | tuple[type[E], ...],
        message: str,
        match_expr: str | Pattern[str] | None = None,
    ) -> None:
        self.expected_exception = expected_exception
        self.message = message
        self.match_expr = match_expr
        self.excinfo: _pytest._code.ExceptionInfo[E] | None = None
        if self.match_expr is not None:
            re_error = None
            try:
                re.compile(self.match_expr)
            except re.error as e:
                re_error = e
            if re_error is not None:
                fail(f"Invalid regex pattern provided to 'match': {re_error}")

    def __enter__(self) -> _pytest._code.ExceptionInfo[E]:
        self.excinfo = _pytest._code.ExceptionInfo.for_later()
        return self.excinfo

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool:
        __tracebackhide__ = True
        if exc_type is None:
            fail(self.message)
        assert self.excinfo is not None
        if not issubclass(exc_type, self.expected_exception):
            return False
        # Cast to narrow the exception type now that it's verified.
        exc_info = cast(Tuple[Type[E], E, TracebackType], (exc_type, exc_val, exc_tb))
        self.excinfo.fill_unfilled(exc_info)
        if self.match_expr is not None:
            self.excinfo.match(self.match_expr)
        return True
