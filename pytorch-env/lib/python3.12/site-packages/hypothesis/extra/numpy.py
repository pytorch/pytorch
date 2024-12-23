# This file is part of Hypothesis, which may be found at
# https://github.com/HypothesisWorks/hypothesis/
#
# Copyright the Hypothesis Authors.
# Individual contributors are listed in AUTHORS.rst and the git log.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

import importlib
import math
import types
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal, Optional, TypeVar, Union, cast, overload

import numpy as np

from hypothesis import strategies as st
from hypothesis._settings import note_deprecation
from hypothesis.errors import HypothesisException, InvalidArgument
from hypothesis.extra._array_helpers import (
    NDIM_MAX,
    BasicIndex,
    BasicIndexStrategy,
    BroadcastableShapes,
    Shape,
    array_shapes,
    broadcastable_shapes,
    check_argument,
    check_valid_dims,
    mutually_broadcastable_shapes as _mutually_broadcastable_shapes,
    order_check,
    valid_tuple_axes as _valid_tuple_axes,
)
from hypothesis.internal.conjecture import utils as cu
from hypothesis.internal.coverage import check_function
from hypothesis.internal.reflection import proxies
from hypothesis.internal.validation import check_type
from hypothesis.strategies._internal.lazy import unwrap_strategies
from hypothesis.strategies._internal.numbers import Real
from hypothesis.strategies._internal.strategies import (
    Ex,
    MappedStrategy,
    T,
    check_strategy,
)
from hypothesis.strategies._internal.utils import defines_strategy


def _try_import(mod_name: str, attr_name: str) -> Any:
    assert "." not in attr_name
    try:
        mod = importlib.import_module(mod_name)
        return getattr(mod, attr_name, None)
    except ImportError:
        return None


if TYPE_CHECKING:
    from numpy.typing import DTypeLike, NDArray
else:
    NDArray = _try_import("numpy.typing", "NDArray")

ArrayLike = _try_import("numpy.typing", "ArrayLike")
_NestedSequence = _try_import("numpy._typing._nested_sequence", "_NestedSequence")
_SupportsArray = _try_import("numpy._typing._array_like", "_SupportsArray")

__all__ = [
    "BroadcastableShapes",
    "from_dtype",
    "arrays",
    "array_shapes",
    "scalar_dtypes",
    "boolean_dtypes",
    "unsigned_integer_dtypes",
    "integer_dtypes",
    "floating_dtypes",
    "complex_number_dtypes",
    "datetime64_dtypes",
    "timedelta64_dtypes",
    "byte_string_dtypes",
    "unicode_string_dtypes",
    "array_dtypes",
    "nested_dtypes",
    "valid_tuple_axes",
    "broadcastable_shapes",
    "mutually_broadcastable_shapes",
    "basic_indices",
    "integer_array_indices",
]

TIME_RESOLUTIONS = tuple("Y  M  D  h  m  s  ms  us  ns  ps  fs  as".split())

# See https://github.com/HypothesisWorks/hypothesis/pull/3394 and linked discussion.
NP_FIXED_UNICODE = tuple(int(x) for x in np.__version__.split(".")[:2]) >= (1, 19)


@defines_strategy(force_reusable_values=True)
def from_dtype(
    dtype: np.dtype,
    *,
    alphabet: Optional[st.SearchStrategy[str]] = None,
    min_size: int = 0,
    max_size: Optional[int] = None,
    min_value: Union[int, float, None] = None,
    max_value: Union[int, float, None] = None,
    allow_nan: Optional[bool] = None,
    allow_infinity: Optional[bool] = None,
    allow_subnormal: Optional[bool] = None,
    exclude_min: Optional[bool] = None,
    exclude_max: Optional[bool] = None,
    min_magnitude: Real = 0,
    max_magnitude: Optional[Real] = None,
) -> st.SearchStrategy[Any]:
    """Creates a strategy which can generate any value of the given dtype.

    Compatible parameters are passed to the inferred strategy function while
    inapplicable ones are ignored.
    This allows you, for example, to customise the min and max values,
    control the length or contents of strings, or exclude non-finite
    numbers. This is particularly useful when kwargs are passed through from
    :func:`arrays` which allow a variety of numeric dtypes, as it seamlessly
    handles the ``width`` or representable bounds for you.
    """
    check_type(np.dtype, dtype, "dtype")
    kwargs = {k: v for k, v in locals().items() if k != "dtype" and v is not None}

    # Compound datatypes, eg 'f4,f4,f4'
    if dtype.names is not None and dtype.fields is not None:
        # mapping np.void.type over a strategy is nonsense, so return now.
        subs = [from_dtype(dtype.fields[name][0], **kwargs) for name in dtype.names]
        return st.tuples(*subs)

    # Subarray datatypes, eg '(2, 3)i4'
    if dtype.subdtype is not None:
        subtype, shape = dtype.subdtype
        return arrays(subtype, shape, elements=kwargs)

    def compat_kw(*args, **kw):
        """Update default args to the strategy with user-supplied keyword args."""
        assert {"min_value", "max_value", "max_size"}.issuperset(kw)
        for key in set(kwargs).intersection(kw):
            msg = f"dtype {dtype!r} requires {key}={kwargs[key]!r} to be %s {kw[key]!r}"
            if kw[key] is not None:
                if key.startswith("min_") and kw[key] > kwargs[key]:
                    raise InvalidArgument(msg % ("at least",))
                elif key.startswith("max_") and kw[key] < kwargs[key]:
                    raise InvalidArgument(msg % ("at most",))
        kw.update({k: v for k, v in kwargs.items() if k in args or k in kw})
        return kw

    # Scalar datatypes
    if dtype.kind == "b":
        result: st.SearchStrategy[Any] = st.booleans()
    elif dtype.kind == "f":
        result = st.floats(
            width=cast(Literal[16, 32, 64], min(8 * dtype.itemsize, 64)),
            **compat_kw(
                "min_value",
                "max_value",
                "allow_nan",
                "allow_infinity",
                "allow_subnormal",
                "exclude_min",
                "exclude_max",
            ),
        )
    elif dtype.kind == "c":
        result = st.complex_numbers(
            width=cast(
                Literal[32, 64, 128], min(8 * dtype.itemsize, 128)
            ),  # convert from bytes to bits
            **compat_kw(
                "min_magnitude",
                "max_magnitude",
                "allow_nan",
                "allow_infinity",
                "allow_subnormal",
            ),
        )
    elif dtype.kind in ("S", "a"):
        # Numpy strings are null-terminated; only allow round-trippable values.
        # `itemsize == 0` means 'fixed length determined at array creation'
        max_size = dtype.itemsize or None
        result = st.binary(**compat_kw("min_size", max_size=max_size)).filter(
            lambda b: b[-1:] != b"\0"
        )
    elif dtype.kind == "u":
        kw = compat_kw(min_value=0, max_value=2 ** (8 * dtype.itemsize) - 1)
        result = st.integers(**kw)
    elif dtype.kind == "i":
        overflow = 2 ** (8 * dtype.itemsize - 1)
        result = st.integers(**compat_kw(min_value=-overflow, max_value=overflow - 1))
    elif dtype.kind == "U":
        # Encoded in UTF-32 (four bytes/codepoint) and null-terminated
        max_size = (dtype.itemsize or 0) // 4 or None
        if NP_FIXED_UNICODE and "alphabet" not in kwargs:
            kwargs["alphabet"] = st.characters()
        result = st.text(**compat_kw("alphabet", "min_size", max_size=max_size)).filter(
            lambda b: b[-1:] != "\0"
        )
    elif dtype.kind in ("m", "M"):
        if "[" in dtype.str:
            res = st.just(dtype.str.split("[")[-1][:-1])
        else:
            # Note that this case isn't valid to pass to arrays(), but we support
            # it here because we'd have to guard against equivalents in arrays()
            # regardless and drawing scalars is a valid use-case.
            res = st.sampled_from(TIME_RESOLUTIONS)
        if allow_nan is not False:
            elems = st.integers(-(2**63), 2**63 - 1) | st.just("NaT")
        else:  # NEP-7 defines the NaT value as integer -(2**63)
            elems = st.integers(-(2**63) + 1, 2**63 - 1)
        result = st.builds(dtype.type, elems, res)
    else:
        raise InvalidArgument(f"No strategy inference for {dtype}")
    return result.map(dtype.type)


class ArrayStrategy(st.SearchStrategy):
    def __init__(self, element_strategy, shape, dtype, fill, unique):
        self.shape = tuple(shape)
        self.fill = fill
        self.array_size = int(np.prod(shape))
        self.dtype = dtype
        self.element_strategy = element_strategy
        self.unique = unique
        self._check_elements = dtype.kind not in ("O", "V")

    def __repr__(self):
        return (
            f"ArrayStrategy({self.element_strategy!r}, shape={self.shape}, "
            f"dtype={self.dtype!r}, fill={self.fill!r}, unique={self.unique!r})"
        )

    def set_element(self, val, result, idx, *, fill=False):
        try:
            result[idx] = val
        except TypeError as err:
            raise InvalidArgument(
                f"Could not add element={val!r} of {val.dtype!r} to array of "
                f"{result.dtype!r} - possible mismatch of time units in dtypes?"
            ) from err
        try:
            elem_changed = self._check_elements and val != result[idx] and val == val
        except Exception as err:  # pragma: no cover
            # This branch only exists to help debug weird behaviour in Numpy,
            # such as the string problems we had a while back.
            raise HypothesisException(
                "Internal error when checking element=%r of %r to array of %r"
                % (val, val.dtype, result.dtype)
            ) from err
        if elem_changed:
            strategy = self.fill if fill else self.element_strategy
            if self.dtype.kind == "f":  # pragma: no cover
                # This logic doesn't trigger in our coverage tests under Numpy 1.24+,
                # with built-in checks for overflow, but we keep it for good error
                # messages and compatibility with older versions of Numpy.
                try:
                    is_subnormal = 0 < abs(val) < np.finfo(self.dtype).tiny
                except Exception:
                    # val may be a non-float that does not support the
                    # operations __lt__ and __abs__
                    is_subnormal = False
                if is_subnormal:
                    raise InvalidArgument(
                        f"Generated subnormal float {val} from strategy "
                        f"{strategy} resulted in {result[idx]!r}, probably "
                        "as a result of NumPy being built with flush-to-zero "
                        "compiler options. Consider passing "
                        "allow_subnormal=False."
                    )
            raise InvalidArgument(
                "Generated array element %r from %r cannot be represented as "
                "dtype %r - instead it becomes %r (type %r).  Consider using a more "
                "precise strategy, for example passing the `width` argument to "
                "`floats()`."
                % (val, strategy, self.dtype, result[idx], type(result[idx]))
            )

    def do_draw(self, data):
        if 0 in self.shape:
            return np.zeros(dtype=self.dtype, shape=self.shape)

        # Because Numpy allocates memory for strings at array creation, if we have
        # an unsized string dtype we'll fill an object array and then cast it back.
        unsized_string_dtype = (
            self.dtype.kind in ("S", "a", "U") and self.dtype.itemsize == 0
        )

        # This could legitimately be a np.empty, but the performance gains for
        # that would be so marginal that there's really not much point risking
        # undefined behaviour shenanigans.
        result = np.zeros(
            shape=self.array_size, dtype=object if unsized_string_dtype else self.dtype
        )

        if self.fill.is_empty:
            # We have no fill value (either because the user explicitly
            # disabled it or because the default behaviour was used and our
            # elements strategy does not produce reusable values), so we must
            # generate a fully dense array with a freshly drawn value for each
            # entry.
            if self.unique:
                elems = st.lists(
                    self.element_strategy,
                    min_size=self.array_size,
                    max_size=self.array_size,
                    unique=True,
                )
                for i, v in enumerate(data.draw(elems)):
                    self.set_element(v, result, i)
            else:
                for i in range(len(result)):
                    self.set_element(data.draw(self.element_strategy), result, i)
        else:
            # We draw numpy arrays as "sparse with an offset". We draw a
            # collection of index assignments within the array and assign
            # fresh values from our elements strategy to those indices. If at
            # the end we have not assigned every element then we draw a single
            # value from our fill strategy and use that to populate the
            # remaining positions with that strategy.

            elements = cu.many(
                data,
                min_size=0,
                max_size=self.array_size,
                # sqrt isn't chosen for any particularly principled reason. It
                # just grows reasonably quickly but sublinearly, and for small
                # arrays it represents a decent fraction of the array size.
                average_size=min(
                    0.9 * self.array_size,  # ensure small arrays sometimes use fill
                    max(10, math.sqrt(self.array_size)),  # ...but *only* sometimes
                ),
            )

            needs_fill = np.full(self.array_size, True)
            seen = set()

            while elements.more():
                i = data.draw_integer(0, self.array_size - 1)
                if not needs_fill[i]:
                    elements.reject()
                    continue
                self.set_element(data.draw(self.element_strategy), result, i)
                if self.unique:
                    if result[i] in seen:
                        elements.reject()
                        continue
                    else:
                        seen.add(result[i])
                needs_fill[i] = False
            if needs_fill.any():
                # We didn't fill all of the indices in the early loop, so we
                # put a fill value into the rest.

                # We have to do this hilarious little song and dance to work
                # around numpy's special handling of iterable values. If the
                # value here were e.g. a tuple then neither array creation
                # nor putmask would do the right thing. But by creating an
                # array of size one and then assigning the fill value as a
                # single element, we both get an array with the right value in
                # it and putmask will do the right thing by repeating the
                # values of the array across the mask.
                one_element = np.zeros(
                    shape=1, dtype=object if unsized_string_dtype else self.dtype
                )
                self.set_element(data.draw(self.fill), one_element, 0, fill=True)
                if unsized_string_dtype:
                    one_element = one_element.astype(self.dtype)
                fill_value = one_element[0]
                if self.unique:
                    try:
                        is_nan = np.isnan(fill_value)
                    except TypeError:
                        is_nan = False

                    if not is_nan:
                        raise InvalidArgument(
                            f"Cannot fill unique array with non-NaN value {fill_value!r}"
                        )

                np.putmask(result, needs_fill, one_element)

        if unsized_string_dtype:
            out = result.astype(self.dtype)
            mismatch = out != result
            if mismatch.any():
                raise InvalidArgument(
                    "Array elements %r cannot be represented as dtype %r - instead "
                    "they become %r.  Use a more precise strategy, e.g. without "
                    "trailing null bytes, as this will be an error future versions."
                    % (result[mismatch], self.dtype, out[mismatch])
                )
            result = out

        result = result.reshape(self.shape).copy()

        assert result.base is None

        return result


def fill_for(elements, unique, fill, name=""):
    if fill is None:
        if unique or not elements.has_reusable_values:
            fill = st.nothing()
        else:
            fill = elements
    else:
        check_strategy(fill, f"{name}.fill" if name else "fill")
    return fill


D = TypeVar("D", bound="DTypeLike")
G = TypeVar("G", bound="np.generic")


@overload
@defines_strategy(force_reusable_values=True)
def arrays(
    dtype: Union["np.dtype[G]", st.SearchStrategy["np.dtype[G]"]],
    shape: Union[int, st.SearchStrategy[int], Shape, st.SearchStrategy[Shape]],
    *,
    elements: Optional[Union[st.SearchStrategy[Any], Mapping[str, Any]]] = None,
    fill: Optional[st.SearchStrategy[Any]] = None,
    unique: bool = False,
) -> "st.SearchStrategy[NDArray[G]]": ...


@overload
@defines_strategy(force_reusable_values=True)
def arrays(
    dtype: Union[D, st.SearchStrategy[D]],
    shape: Union[int, st.SearchStrategy[int], Shape, st.SearchStrategy[Shape]],
    *,
    elements: Optional[Union[st.SearchStrategy[Any], Mapping[str, Any]]] = None,
    fill: Optional[st.SearchStrategy[Any]] = None,
    unique: bool = False,
) -> "st.SearchStrategy[NDArray[Any]]": ...


@defines_strategy(force_reusable_values=True)
def arrays(
    dtype: Union[D, st.SearchStrategy[D]],
    shape: Union[int, st.SearchStrategy[int], Shape, st.SearchStrategy[Shape]],
    *,
    elements: Optional[Union[st.SearchStrategy[Any], Mapping[str, Any]]] = None,
    fill: Optional[st.SearchStrategy[Any]] = None,
    unique: bool = False,
) -> "st.SearchStrategy[NDArray[Any]]":
    r"""Returns a strategy for generating :class:`numpy:numpy.ndarray`\ s.

    * ``dtype`` may be any valid input to :class:`~numpy:numpy.dtype`
      (this includes :class:`~numpy:numpy.dtype` objects), or a strategy that
      generates such values.
    * ``shape`` may be an integer >= 0, a tuple of such integers, or a
      strategy that generates such values.
    * ``elements`` is a strategy for generating values to put in the array.
      If it is None a suitable value will be inferred based on the dtype,
      which may give any legal value (including eg NaN for floats).
      If a mapping, it will be passed as ``**kwargs`` to ``from_dtype()``
    * ``fill`` is a strategy that may be used to generate a single background
      value for the array. If None, a suitable default will be inferred
      based on the other arguments. If set to
      :func:`~hypothesis.strategies.nothing` then filling
      behaviour will be disabled entirely and every element will be generated
      independently.
    * ``unique`` specifies if the elements of the array should all be
      distinct from one another. Note that in this case multiple NaN values
      may still be allowed. If fill is also set, the only valid values for
      it to return are NaN values (anything for which :obj:`numpy:numpy.isnan`
      returns True. So e.g. for complex numbers ``nan+1j`` is also a valid fill).
      Note that if ``unique`` is set to ``True`` the generated values must be
      hashable.

    Arrays of specified ``dtype`` and ``shape`` are generated for example
    like this:

    .. code-block:: pycon

      >>> import numpy as np
      >>> arrays(np.int8, (2, 3)).example()
      array([[-8,  6,  3],
             [-6,  4,  6]], dtype=int8)
      >>> arrays(np.float, 3, elements=st.floats(0, 1)).example()
      array([ 0.88974794,  0.77387938,  0.1977879 ])

    Array values are generated in two parts:

    1. Some subset of the coordinates of the array are populated with a value
       drawn from the elements strategy (or its inferred form).
    2. If any coordinates were not assigned in the previous step, a single
       value is drawn from the ``fill`` strategy and is assigned to all remaining
       places.

    You can set :func:`fill=nothing() <hypothesis.strategies.nothing>` to
    disable this behaviour and draw a value for every element.

    If ``fill=None``, then it will attempt to infer the correct behaviour
    automatically. If ``unique`` is ``True``, no filling will occur by default.
    Otherwise, if it looks safe to reuse the values of elements across
    multiple coordinates (this will be the case for any inferred strategy, and
    for most of the builtins, but is not the case for mutable values or
    strategies built with flatmap, map, composite, etc) then it will use the
    elements strategy as the fill, else it will default to having no fill.

    Having a fill helps Hypothesis craft high quality examples, but its
    main importance is when the array generated is large: Hypothesis is
    primarily designed around testing small examples. If you have arrays with
    hundreds or more elements, having a fill value is essential if you want
    your tests to run in reasonable time.
    """
    # Our dtype argument might be a union, e.g. `np.float64 | np.complex64`; we handle
    # that by turning it into a strategy up-front.
    if type(dtype) in (getattr(types, "UnionType", object()), Union):
        dtype = st.one_of(*(from_dtype(np.dtype(d)) for d in dtype.__args__))  # type: ignore

    # We support passing strategies as arguments for convenience, or at least
    # for legacy reasons, but don't want to pay the perf cost of a composite
    # strategy (i.e. repeated argument handling and validation) when it's not
    # needed.  So we get the best of both worlds by recursing with flatmap,
    # but only when it's actually needed.
    if isinstance(dtype, st.SearchStrategy):
        return dtype.flatmap(
            lambda d: arrays(d, shape, elements=elements, fill=fill, unique=unique)
        )
    if isinstance(shape, st.SearchStrategy):
        return shape.flatmap(
            lambda s: arrays(dtype, s, elements=elements, fill=fill, unique=unique)
        )
    # From here on, we're only dealing with values and it's relatively simple.
    dtype = np.dtype(dtype)  # type: ignore[arg-type,assignment]
    assert isinstance(dtype, np.dtype)  # help mypy out a bit...
    if elements is None or isinstance(elements, Mapping):
        if dtype.kind in ("m", "M") and "[" not in dtype.str:
            # For datetime and timedelta dtypes, we have a tricky situation -
            # because they *may or may not* specify a unit as part of the dtype.
            # If not, we flatmap over the various resolutions so that array
            # elements have consistent units but units may vary between arrays.
            return (
                st.sampled_from(TIME_RESOLUTIONS)
                .map((dtype.str + "[{}]").format)
                .flatmap(lambda d: arrays(d, shape=shape, fill=fill, unique=unique))
            )
        elements = from_dtype(dtype, **(elements or {}))
    check_strategy(elements, "elements")
    # If there's a redundant cast to the requested dtype, remove it.  This unlocks
    # optimizations such as fast unique sampled_from, and saves some time directly too.
    unwrapped = unwrap_strategies(elements)
    if isinstance(unwrapped, MappedStrategy) and unwrapped.pack == dtype.type:
        elements = unwrapped.mapped_strategy
        if getattr(unwrapped, "force_has_reusable_values", False):
            elements.force_has_reusable_values = True  # type: ignore
    if isinstance(shape, int):
        shape = (shape,)
    shape = tuple(shape)
    check_argument(
        all(isinstance(s, int) for s in shape),
        "Array shape must be integer in each dimension, provided shape was {}",
        shape,
    )
    fill = fill_for(elements=elements, unique=unique, fill=fill)
    return ArrayStrategy(elements, shape, dtype, fill, unique)


@defines_strategy()
def scalar_dtypes() -> st.SearchStrategy[np.dtype]:
    """Return a strategy that can return any non-flexible scalar dtype."""
    return st.one_of(
        boolean_dtypes(),
        integer_dtypes(),
        unsigned_integer_dtypes(),
        floating_dtypes(),
        complex_number_dtypes(),
        datetime64_dtypes(),
        timedelta64_dtypes(),
    )


def defines_dtype_strategy(strat: T) -> T:
    @defines_strategy()
    @proxies(strat)
    def inner(*args, **kwargs):
        return strat(*args, **kwargs).map(np.dtype)

    return inner


@defines_dtype_strategy
def boolean_dtypes() -> st.SearchStrategy["np.dtype[np.bool_]"]:
    return st.just("?")  # type: ignore[arg-type]


def dtype_factory(kind, sizes, valid_sizes, endianness):
    # Utility function, shared logic for most integer and string types
    valid_endian = ("?", "<", "=", ">")
    check_argument(
        endianness in valid_endian,
        "Unknown endianness: was {}, must be in {}",
        endianness,
        valid_endian,
    )
    if valid_sizes is not None:
        if isinstance(sizes, int):
            sizes = (sizes,)
        check_argument(sizes, "Dtype must have at least one possible size.")
        check_argument(
            all(s in valid_sizes for s in sizes),
            "Invalid sizes: was {} must be an item or sequence in {}",
            sizes,
            valid_sizes,
        )
        if all(isinstance(s, int) for s in sizes):
            sizes = sorted({s // 8 for s in sizes})
    strat = st.sampled_from(sizes)
    if "{}" not in kind:
        kind += "{}"
    if endianness == "?":
        return strat.map(("<" + kind).format) | strat.map((">" + kind).format)
    return strat.map((endianness + kind).format)


@overload
@defines_dtype_strategy
def unsigned_integer_dtypes(
    *,
    endianness: str = "?",
    sizes: Literal[8],
) -> st.SearchStrategy["np.dtype[np.uint8]"]: ...


@overload
@defines_dtype_strategy
def unsigned_integer_dtypes(
    *,
    endianness: str = "?",
    sizes: Literal[16],
) -> st.SearchStrategy["np.dtype[np.uint16]"]: ...


@overload
@defines_dtype_strategy
def unsigned_integer_dtypes(
    *,
    endianness: str = "?",
    sizes: Literal[32],
) -> st.SearchStrategy["np.dtype[np.uint32]"]: ...


@overload
@defines_dtype_strategy
def unsigned_integer_dtypes(
    *,
    endianness: str = "?",
    sizes: Literal[64],
) -> st.SearchStrategy["np.dtype[np.uint64]"]: ...


@overload
@defines_dtype_strategy
def unsigned_integer_dtypes(
    *,
    endianness: str = "?",
    sizes: Sequence[Literal[8, 16, 32, 64]] = (8, 16, 32, 64),
) -> st.SearchStrategy["np.dtype[np.unsignedinteger[Any]]"]: ...


@defines_dtype_strategy
def unsigned_integer_dtypes(
    *,
    endianness: str = "?",
    sizes: Union[Literal[8, 16, 32, 64], Sequence[Literal[8, 16, 32, 64]]] = (
        8,
        16,
        32,
        64,
    ),
) -> st.SearchStrategy["np.dtype[np.unsignedinteger[Any]]"]:
    """Return a strategy for unsigned integer dtypes.

    endianness may be ``<`` for little-endian, ``>`` for big-endian,
    ``=`` for native byte order, or ``?`` to allow either byte order.
    This argument only applies to dtypes of more than one byte.

    sizes must be a collection of integer sizes in bits.  The default
    (8, 16, 32, 64) covers the full range of sizes.
    """
    return dtype_factory("u", sizes, (8, 16, 32, 64), endianness)


@overload
@defines_dtype_strategy
def integer_dtypes(
    *,
    endianness: str = "?",
    sizes: Literal[8],
) -> st.SearchStrategy["np.dtype[np.int8]"]: ...


@overload
@defines_dtype_strategy
def integer_dtypes(
    *,
    endianness: str = "?",
    sizes: Literal[16],
) -> st.SearchStrategy["np.dtype[np.int16]"]: ...


@overload
@defines_dtype_strategy
def integer_dtypes(
    *,
    endianness: str = "?",
    sizes: Literal[32],
) -> st.SearchStrategy["np.dtype[np.int32]"]: ...


@overload
@defines_dtype_strategy
def integer_dtypes(
    *,
    endianness: str = "?",
    sizes: Literal[64],
) -> st.SearchStrategy["np.dtype[np.int64]"]: ...


@overload
@defines_dtype_strategy
def integer_dtypes(
    *,
    endianness: str = "?",
    sizes: Sequence[Literal[8, 16, 32, 64]] = (8, 16, 32, 64),
) -> st.SearchStrategy["np.dtype[np.signedinteger[Any]]"]: ...


@defines_dtype_strategy
def integer_dtypes(
    *,
    endianness: str = "?",
    sizes: Union[Literal[8, 16, 32, 64], Sequence[Literal[8, 16, 32, 64]]] = (
        8,
        16,
        32,
        64,
    ),
) -> st.SearchStrategy["np.dtype[np.signedinteger[Any]]"]:
    """Return a strategy for signed integer dtypes.

    endianness and sizes are treated as for
    :func:`unsigned_integer_dtypes`.
    """
    return dtype_factory("i", sizes, (8, 16, 32, 64), endianness)


@overload
@defines_dtype_strategy
def floating_dtypes(
    *,
    endianness: str = "?",
    sizes: Literal[16],
) -> st.SearchStrategy["np.dtype[np.float16]"]: ...


@overload
@defines_dtype_strategy
def floating_dtypes(
    *,
    endianness: str = "?",
    sizes: Literal[32],
) -> st.SearchStrategy["np.dtype[np.float32]"]: ...


@overload
@defines_dtype_strategy
def floating_dtypes(
    *,
    endianness: str = "?",
    sizes: Literal[64],
) -> st.SearchStrategy["np.dtype[np.float64]"]: ...


@overload
@defines_dtype_strategy
def floating_dtypes(
    *,
    endianness: str = "?",
    sizes: Literal[128],
) -> st.SearchStrategy["np.dtype[np.float128]"]: ...


@overload
@defines_dtype_strategy
def floating_dtypes(
    *,
    endianness: str = "?",
    sizes: Sequence[Literal[16, 32, 64, 96, 128]] = (16, 32, 64),
) -> st.SearchStrategy["np.dtype[np.floating[Any]]"]: ...


@defines_dtype_strategy
def floating_dtypes(
    *,
    endianness: str = "?",
    sizes: Union[
        Literal[16, 32, 64, 96, 128], Sequence[Literal[16, 32, 64, 96, 128]]
    ] = (16, 32, 64),
) -> st.SearchStrategy["np.dtype[np.floating[Any]]"]:
    """Return a strategy for floating-point dtypes.

    sizes is the size in bits of floating-point number.  Some machines support
    96- or 128-bit floats, but these are not generated by default.

    Larger floats (96 and 128 bit real parts) are not supported on all
    platforms and therefore disabled by default.  To generate these dtypes,
    include these values in the sizes argument.
    """
    return dtype_factory("f", sizes, (16, 32, 64, 96, 128), endianness)


@overload
@defines_dtype_strategy
def complex_number_dtypes(
    *,
    endianness: str = "?",
    sizes: Literal[64],
) -> st.SearchStrategy["np.dtype[np.complex64]"]: ...


@overload
@defines_dtype_strategy
def complex_number_dtypes(
    *,
    endianness: str = "?",
    sizes: Literal[128],
) -> st.SearchStrategy["np.dtype[np.complex128]"]: ...


@overload
@defines_dtype_strategy
def complex_number_dtypes(
    *,
    endianness: str = "?",
    sizes: Literal[256],
) -> st.SearchStrategy["np.dtype[np.complex256]"]: ...


@overload
@defines_dtype_strategy
def complex_number_dtypes(
    *,
    endianness: str = "?",
    sizes: Sequence[Literal[64, 128, 192, 256]] = (64, 128),
) -> st.SearchStrategy["np.dtype[np.complexfloating[Any, Any]]"]: ...


@defines_dtype_strategy
def complex_number_dtypes(
    *,
    endianness: str = "?",
    sizes: Union[Literal[64, 128, 192, 256], Sequence[Literal[64, 128, 192, 256]]] = (
        64,
        128,
    ),
) -> st.SearchStrategy["np.dtype[np.complexfloating[Any, Any]]"]:
    """Return a strategy for complex-number dtypes.

    sizes is the total size in bits of a complex number, which consists
    of two floats.  Complex halves (a 16-bit real part) are not supported
    by numpy and will not be generated by this strategy.
    """
    return dtype_factory("c", sizes, (64, 128, 192, 256), endianness)


@check_function
def validate_time_slice(max_period, min_period):
    check_argument(
        max_period in TIME_RESOLUTIONS,
        "max_period {} must be a valid resolution in {}",
        max_period,
        TIME_RESOLUTIONS,
    )
    check_argument(
        min_period in TIME_RESOLUTIONS,
        "min_period {} must be a valid resolution in {}",
        min_period,
        TIME_RESOLUTIONS,
    )
    start = TIME_RESOLUTIONS.index(max_period)
    end = TIME_RESOLUTIONS.index(min_period) + 1
    check_argument(
        start < end,
        "max_period {} must be earlier in sequence {} than min_period {}",
        max_period,
        TIME_RESOLUTIONS,
        min_period,
    )
    return TIME_RESOLUTIONS[start:end]


@defines_dtype_strategy
def datetime64_dtypes(
    *, max_period: str = "Y", min_period: str = "ns", endianness: str = "?"
) -> st.SearchStrategy["np.dtype[np.datetime64]"]:
    """Return a strategy for datetime64 dtypes, with various precisions from
    year to attosecond."""
    return dtype_factory(
        "datetime64[{}]",
        validate_time_slice(max_period, min_period),
        TIME_RESOLUTIONS,
        endianness,
    )


@defines_dtype_strategy
def timedelta64_dtypes(
    *, max_period: str = "Y", min_period: str = "ns", endianness: str = "?"
) -> st.SearchStrategy["np.dtype[np.timedelta64]"]:
    """Return a strategy for timedelta64 dtypes, with various precisions from
    year to attosecond."""
    return dtype_factory(
        "timedelta64[{}]",
        validate_time_slice(max_period, min_period),
        TIME_RESOLUTIONS,
        endianness,
    )


@defines_dtype_strategy
def byte_string_dtypes(
    *, endianness: str = "?", min_len: int = 1, max_len: int = 16
) -> st.SearchStrategy["np.dtype[np.bytes_]"]:
    """Return a strategy for generating bytestring dtypes, of various lengths
    and byteorder.

    While Hypothesis' string strategies can generate empty strings, string
    dtypes with length 0 indicate that size is still to be determined, so
    the minimum length for string dtypes is 1.
    """
    order_check("len", 1, min_len, max_len)
    return dtype_factory("S", list(range(min_len, max_len + 1)), None, endianness)


@defines_dtype_strategy
def unicode_string_dtypes(
    *, endianness: str = "?", min_len: int = 1, max_len: int = 16
) -> st.SearchStrategy["np.dtype[np.str_]"]:
    """Return a strategy for generating unicode string dtypes, of various
    lengths and byteorder.

    While Hypothesis' string strategies can generate empty strings, string
    dtypes with length 0 indicate that size is still to be determined, so
    the minimum length for string dtypes is 1.
    """
    order_check("len", 1, min_len, max_len)
    return dtype_factory("U", list(range(min_len, max_len + 1)), None, endianness)


def _no_title_is_name_of_a_titled_field(ls):
    seen = set()
    for title_and_name, *_ in ls:
        if isinstance(title_and_name, tuple):
            if seen.intersection(title_and_name):  # pragma: no cover
                # Our per-element filters below make this as rare as possible,
                # so it's not always covered.
                return False
            seen.update(title_and_name)
    return True


@defines_dtype_strategy
def array_dtypes(
    subtype_strategy: st.SearchStrategy[np.dtype] = scalar_dtypes(),
    *,
    min_size: int = 1,
    max_size: int = 5,
    allow_subarrays: bool = False,
) -> st.SearchStrategy[np.dtype]:
    """Return a strategy for generating array (compound) dtypes, with members
    drawn from the given subtype strategy."""
    order_check("size", 0, min_size, max_size)
    # The empty string is replaced by f{idx}; see #1963 for details.  Much easier to
    # insist that field names be unique and just boost f{idx} strings manually.
    field_names = st.integers(0, 127).map("f{}".format) | st.text(min_size=1)
    name_titles = st.one_of(
        field_names,
        st.tuples(field_names, field_names).filter(lambda ns: ns[0] != ns[1]),
    )
    elements: st.SearchStrategy[tuple] = st.tuples(name_titles, subtype_strategy)
    if allow_subarrays:
        elements |= st.tuples(
            name_titles, subtype_strategy, array_shapes(max_dims=2, max_side=2)
        )
    return st.lists(  # type: ignore[return-value]
        elements=elements,
        min_size=min_size,
        max_size=max_size,
        unique_by=(
            # Deduplicate by both name and title for efficiency before filtering.
            # (Field names must be unique, as must titles, and no intersections)
            lambda d: d[0] if isinstance(d[0], str) else d[0][0],
            lambda d: d[0] if isinstance(d[0], str) else d[0][1],
        ),
    ).filter(_no_title_is_name_of_a_titled_field)


@defines_strategy()
def nested_dtypes(
    subtype_strategy: st.SearchStrategy[np.dtype] = scalar_dtypes(),
    *,
    max_leaves: int = 10,
    max_itemsize: Optional[int] = None,
) -> st.SearchStrategy[np.dtype]:
    """Return the most-general dtype strategy.

    Elements drawn from this strategy may be simple (from the
    subtype_strategy), or several such values drawn from
    :func:`array_dtypes` with ``allow_subarrays=True``. Subdtypes in an
    array dtype may be nested to any depth, subject to the max_leaves
    argument.
    """
    return st.recursive(
        subtype_strategy,
        lambda x: array_dtypes(x, allow_subarrays=True),
        max_leaves=max_leaves,
    ).filter(lambda d: max_itemsize is None or d.itemsize <= max_itemsize)


@proxies(_valid_tuple_axes)
def valid_tuple_axes(*args, **kwargs):
    return _valid_tuple_axes(*args, **kwargs)


valid_tuple_axes.__doc__ = f"""
    Return a strategy for generating permissible tuple-values for the
    ``axis`` argument for a numpy sequential function (e.g.
    :func:`numpy:numpy.sum`), given an array of the specified
    dimensionality.

    {_valid_tuple_axes.__doc__}
    """


@proxies(_mutually_broadcastable_shapes)
def mutually_broadcastable_shapes(*args, **kwargs):
    return _mutually_broadcastable_shapes(*args, **kwargs)


mutually_broadcastable_shapes.__doc__ = f"""
    {_mutually_broadcastable_shapes.__doc__}

    **Use with Generalised Universal Function signatures**

    A :doc:`universal function <numpy:reference/ufuncs>` (or ufunc for short) is a function
    that operates on ndarrays in an element-by-element fashion, supporting array
    broadcasting, type casting, and several other standard features.
    A :doc:`generalised ufunc <numpy:reference/c-api/generalized-ufuncs>` operates on
    sub-arrays rather than elements, based on the "signature" of the function.
    Compare e.g. :obj:`numpy.add() <numpy:numpy.add>` (ufunc) to
    :obj:`numpy.matmul() <numpy:numpy.matmul>` (gufunc).

    To generate shapes for a gufunc, you can pass the ``signature`` argument instead of
    ``num_shapes``.  This must be a gufunc signature string; which you can write by
    hand or access as e.g. ``np.matmul.signature`` on generalised ufuncs.

    In this case, the ``side`` arguments are applied to the 'core dimensions' as well,
    ignoring any frozen dimensions.  ``base_shape``  and the ``dims`` arguments are
    applied to the 'loop dimensions', and if necessary, the dimensionality of each
    shape is silently capped to respect the 32-dimension limit.

    The generated ``result_shape`` is the real result shape of applying the gufunc
    to arrays of the generated ``input_shapes``, even where this is different to
    broadcasting the loop dimensions.

    gufunc-compatible shapes shrink their loop dimensions as above, towards omitting
    optional core dimensions, and smaller-size core dimensions.

    .. code-block:: pycon

        >>> # np.matmul.signature == "(m?,n),(n,p?)->(m?,p?)"
        >>> for _ in range(3):
        ...     mutually_broadcastable_shapes(signature=np.matmul.signature).example()
        BroadcastableShapes(input_shapes=((2,), (2,)), result_shape=())
        BroadcastableShapes(input_shapes=((3, 4, 2), (1, 2)), result_shape=(3, 4))
        BroadcastableShapes(input_shapes=((4, 2), (1, 2, 3)), result_shape=(4, 3))

    """


@defines_strategy()
def basic_indices(
    shape: Shape,
    *,
    min_dims: int = 0,
    max_dims: Optional[int] = None,
    allow_newaxis: bool = False,
    allow_ellipsis: bool = True,
) -> st.SearchStrategy[BasicIndex]:
    """Return a strategy for :doc:`basic indexes <numpy:reference/routines.indexing>` of
    arrays with the specified shape, which may include dimensions of size zero.

    It generates tuples containing some mix of integers, :obj:`python:slice`
    objects, ``...`` (an ``Ellipsis``), and ``None``. When a length-one tuple
    would be generated, this strategy may instead return the element which will
    index the first axis, e.g. ``5`` instead of ``(5,)``.

    * ``shape`` is the shape of the array that will be indexed, as a tuple of
      positive integers. This must be at least two-dimensional for a tuple to be
      a valid index; for one-dimensional arrays use
      :func:`~hypothesis.strategies.slices` instead.
    * ``min_dims`` is the minimum dimensionality of the resulting array from use
      of the generated index. When ``min_dims == 0``, scalars and zero-dimensional
      arrays are both allowed.
    * ``max_dims`` is the the maximum dimensionality of the resulting array,
      defaulting to ``len(shape) if not allow_newaxis else
      max(len(shape), min_dims) + 2``.
    * ``allow_newaxis`` specifies whether ``None`` is allowed in the index.
    * ``allow_ellipsis`` specifies whether ``...`` is allowed in the index.
    """
    # Arguments to exclude scalars, zero-dim arrays, and dims of size zero were
    # all considered and rejected.  We want users to explicitly consider those
    # cases if they're dealing in general indexers, and while it's fiddly we can
    # back-compatibly add them later (hence using kwonlyargs).
    check_type(tuple, shape, "shape")
    check_argument(
        all(isinstance(x, int) and x >= 0 for x in shape),
        f"{shape=}, but all dimensions must be non-negative integers.",
    )
    check_type(bool, allow_ellipsis, "allow_ellipsis")
    check_type(bool, allow_newaxis, "allow_newaxis")
    check_type(int, min_dims, "min_dims")
    if min_dims > len(shape) and not allow_newaxis:
        note_deprecation(
            f"min_dims={min_dims} is larger than len(shape)={len(shape)}, "
            "but allow_newaxis=False makes it impossible for an indexing "
            "operation to add dimensions.",
            since="2021-09-15",
            has_codemod=False,
        )
    check_valid_dims(min_dims, "min_dims")

    if max_dims is None:
        if allow_newaxis:
            max_dims = min(max(len(shape), min_dims) + 2, NDIM_MAX)
        else:
            max_dims = min(len(shape), NDIM_MAX)
    else:
        check_type(int, max_dims, "max_dims")
        if max_dims > len(shape) and not allow_newaxis:
            note_deprecation(
                f"max_dims={max_dims} is larger than len(shape)={len(shape)}, "
                "but allow_newaxis=False makes it impossible for an indexing "
                "operation to add dimensions.",
                since="2021-09-15",
                has_codemod=False,
            )
    check_valid_dims(max_dims, "max_dims")

    order_check("dims", 0, min_dims, max_dims)

    return BasicIndexStrategy(
        shape,
        min_dims=min_dims,
        max_dims=max_dims,
        allow_ellipsis=allow_ellipsis,
        allow_newaxis=allow_newaxis,
        allow_fewer_indices_than_dims=True,
    )


I = TypeVar("I", bound=np.integer)


@overload
@defines_strategy()
def integer_array_indices(
    shape: Shape,
    *,
    result_shape: st.SearchStrategy[Shape] = array_shapes(),
) -> "st.SearchStrategy[tuple[NDArray[np.signedinteger[Any]], ...]]": ...


@overload
@defines_strategy()
def integer_array_indices(
    shape: Shape,
    *,
    result_shape: st.SearchStrategy[Shape] = array_shapes(),
    dtype: "np.dtype[I]",
) -> "st.SearchStrategy[tuple[NDArray[I], ...]]": ...


@defines_strategy()
def integer_array_indices(
    shape: Shape,
    *,
    result_shape: st.SearchStrategy[Shape] = array_shapes(),
    dtype: "np.dtype[I] | np.dtype[np.signedinteger[Any]]" = np.dtype(int),
) -> "st.SearchStrategy[tuple[NDArray[I], ...]]":
    """Return a search strategy for tuples of integer-arrays that, when used
    to index into an array of shape ``shape``, given an array whose shape
    was drawn from ``result_shape``.

    Examples from this strategy shrink towards the tuple of index-arrays::

        len(shape) * (np.zeros(drawn_result_shape, dtype), )

    * ``shape`` a tuple of integers that indicates the shape of the array,
      whose indices are being generated.
    * ``result_shape`` a strategy for generating tuples of integers, which
      describe the shape of the resulting index arrays. The default is
      :func:`~hypothesis.extra.numpy.array_shapes`.  The shape drawn from
      this strategy determines the shape of the array that will be produced
      when the corresponding example from ``integer_array_indices`` is used
      as an index.
    * ``dtype`` the integer data type of the generated index-arrays. Negative
      integer indices can be generated if a signed integer type is specified.

    Recall that an array can be indexed using a tuple of integer-arrays to
    access its members in an arbitrary order, producing an array with an
    arbitrary shape. For example:

    .. code-block:: pycon

        >>> from numpy import array
        >>> x = array([-0, -1, -2, -3, -4])
        >>> ind = (array([[4, 0], [0, 1]]),)  # a tuple containing a 2D integer-array
        >>> x[ind]  # the resulting array is commensurate with the indexing array(s)
        array([[-4,  0],
               [0, -1]])

    Note that this strategy does not accommodate all variations of so-called
    'advanced indexing', as prescribed by NumPy's nomenclature.  Combinations
    of basic and advanced indexes are too complex to usefully define in a
    standard strategy; we leave application-specific strategies to the user.
    Advanced-boolean indexing can be defined as ``arrays(shape=..., dtype=bool)``,
    and is similarly left to the user.
    """
    check_type(tuple, shape, "shape")
    check_argument(
        shape and all(isinstance(x, int) and x > 0 for x in shape),
        f"{shape=} must be a non-empty tuple of integers > 0",
    )
    check_strategy(result_shape, "result_shape")
    check_argument(
        np.issubdtype(dtype, np.integer), f"{dtype=} must be an integer dtype"
    )
    signed = np.issubdtype(dtype, np.signedinteger)

    def array_for(index_shape, size):
        return arrays(
            dtype=dtype,
            shape=index_shape,
            elements=st.integers(-size if signed else 0, size - 1),
        )

    return result_shape.flatmap(
        lambda index_shape: st.tuples(*(array_for(index_shape, size) for size in shape))
    )


def _unpack_generic(thing):
    # get_origin and get_args fail on python<3.9 because (some of) the
    # relevant types do not inherit from _GenericAlias.  So just pick the
    # value out directly.
    real_thing = getattr(thing, "__origin__", None)
    if real_thing is not None:
        return (real_thing, getattr(thing, "__args__", ()))
    else:
        return (thing, ())


def _unpack_dtype(dtype):
    dtype_args = getattr(dtype, "__args__", ())
    if dtype_args and type(dtype) not in (getattr(types, "UnionType", object()), Union):
        assert len(dtype_args) == 1
        if isinstance(dtype_args[0], TypeVar):
            # numpy.dtype[+ScalarType]
            assert dtype_args[0].__bound__ == np.generic
            dtype = Any
        else:
            # plain dtype
            dtype = dtype_args[0]
    return dtype


def _dtype_from_args(args):
    if len(args) <= 1:
        # Zero args: ndarray, _SupportsArray
        # One arg: ndarray[type], _SupportsArray[type]
        dtype = _unpack_dtype(args[0]) if args else Any
    else:
        # Two args: ndarray[shape, type], NDArray[*]
        assert len(args) == 2
        dtype = _unpack_dtype(args[1])

    if dtype is Any:
        return scalar_dtypes()
    elif type(dtype) in (getattr(types, "UnionType", object()), Union):
        return dtype
    return np.dtype(dtype)


def _from_type(thing: type[Ex]) -> Optional[st.SearchStrategy[Ex]]:
    """Called by st.from_type to try to infer a strategy for thing using numpy.

    If we can infer a numpy-specific strategy for thing, we return that; otherwise,
    we return None.
    """

    base_strats = st.one_of(
        [
            st.booleans(),
            st.integers(),
            st.floats(),
            st.complex_numbers(),
            st.text(),
            st.binary(),
        ]
    )
    # don't mix strings and non-ascii bytestrings (ex: ['', b'\x80']). See
    # https://github.com/numpy/numpy/issues/23899.
    base_strats_ascii = st.one_of(
        [
            st.booleans(),
            st.integers(),
            st.floats(),
            st.complex_numbers(),
            st.text(),
            st.binary().filter(bytes.isascii),
        ]
    )

    if thing == np.dtype:
        # Note: Parameterized dtypes and DTypeLike are not supported.
        return st.one_of(
            scalar_dtypes(),
            byte_string_dtypes(),
            unicode_string_dtypes(),
            array_dtypes(),
            nested_dtypes(),
        )

    if thing == ArrayLike:
        # We override the default type resolution to ensure the "coercible to
        # array" contract is honoured. See
        # https://github.com/HypothesisWorks/hypothesis/pull/3670#issuecomment-1578140422.
        # The actual type is (as of np 1.24), with
        # scalars:=[bool, int, float, complex, str, bytes]:
        # Union[
        #     _SupportsArray,
        #     _NestedSequence[_SupportsArray],
        #     *scalars,
        #     _NestedSequence[Union[*scalars]]
        # ]
        return st.one_of(
            # *scalars
            base_strats,
            # The two recursive strategies below cover the following cases:
            # - _SupportsArray (using plain ndarrays)
            # - _NestedSequence[Union[*scalars]] (but excluding non-ascii binary)
            # - _NestedSequence[_SupportsArray] (but with a single leaf element
            # .  to avoid the issue of unequally sized leaves)
            st.recursive(st.lists(base_strats_ascii), extend=st.tuples),
            st.recursive(st.from_type(np.ndarray), extend=st.tuples),
        )

    if isinstance(thing, type) and issubclass(thing, np.generic):
        dtype = np.dtype(thing)
        return from_dtype(dtype) if dtype.kind not in "OV" else None

    real_thing, args = _unpack_generic(thing)

    if real_thing == _NestedSequence:
        # We have to override the default resolution to ensure sequences are of
        # equal length. Actually they are still not, if the arg specialization
        # returns arbitrary-shaped sequences or arrays - hence the even more special
        # resolution of ArrayLike, above.
        assert len(args) <= 1
        base_strat = st.from_type(args[0]) if args else base_strats
        return st.one_of(
            st.lists(base_strat),
            st.recursive(st.tuples(), st.tuples),
            st.recursive(st.tuples(base_strat), st.tuples),
            st.recursive(st.tuples(base_strat, base_strat), st.tuples),
        )

    if real_thing in [np.ndarray, _SupportsArray]:
        dtype = _dtype_from_args(args)
        return arrays(dtype, array_shapes(max_dims=2))  # type: ignore[return-value]

    # We didn't find a type to resolve, continue
    return None
