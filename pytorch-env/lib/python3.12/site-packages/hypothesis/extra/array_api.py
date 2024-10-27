# This file is part of Hypothesis, which may be found at
# https://github.com/HypothesisWorks/hypothesis/
#
# Copyright the Hypothesis Authors.
# Individual contributors are listed in AUTHORS.rst and the git log.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

import math
import sys
from collections.abc import Iterable, Iterator, Mapping, Sequence
from numbers import Real
from types import SimpleNamespace
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    NamedTuple,
    Optional,
    TypeVar,
    Union,
    get_args,
)
from warnings import warn
from weakref import WeakValueDictionary

from hypothesis import strategies as st
from hypothesis.errors import HypothesisWarning, InvalidArgument
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
from hypothesis.internal.floats import next_down
from hypothesis.internal.reflection import proxies
from hypothesis.internal.validation import (
    check_type,
    check_valid_bound,
    check_valid_integer,
    check_valid_interval,
)
from hypothesis.strategies._internal.strategies import check_strategy
from hypothesis.strategies._internal.utils import defines_strategy

if TYPE_CHECKING:
    from typing import TypeAlias

__all__ = [
    "make_strategies_namespace",
]


RELEASED_VERSIONS = ("2021.12", "2022.12", "2023.12")
NOMINAL_VERSIONS = (*RELEASED_VERSIONS, "draft")
assert sorted(NOMINAL_VERSIONS) == list(NOMINAL_VERSIONS)  # sanity check
NominalVersion = Literal["2021.12", "2022.12", "2023.12", "draft"]
assert get_args(NominalVersion) == NOMINAL_VERSIONS  # sanity check


INT_NAMES = ("int8", "int16", "int32", "int64")
UINT_NAMES = ("uint8", "uint16", "uint32", "uint64")
ALL_INT_NAMES = INT_NAMES + UINT_NAMES
FLOAT_NAMES = ("float32", "float64")
REAL_NAMES = ALL_INT_NAMES + FLOAT_NAMES
COMPLEX_NAMES = ("complex64", "complex128")
NUMERIC_NAMES = REAL_NAMES + COMPLEX_NAMES
DTYPE_NAMES = ("bool", *NUMERIC_NAMES)

DataType = TypeVar("DataType")


@check_function
def check_xp_attributes(xp: Any, attributes: list[str]) -> None:
    missing_attrs = [attr for attr in attributes if not hasattr(xp, attr)]
    if len(missing_attrs) > 0:
        f_attrs = ", ".join(missing_attrs)
        raise InvalidArgument(
            f"Array module {xp.__name__} does not have required attributes: {f_attrs}"
        )


def partition_attributes_and_stubs(
    xp: Any, attributes: Iterable[str]
) -> tuple[list[Any], list[str]]:
    non_stubs = []
    stubs = []
    for attr in attributes:
        try:
            non_stubs.append(getattr(xp, attr))
        except AttributeError:
            stubs.append(attr)

    return non_stubs, stubs


def warn_on_missing_dtypes(xp: Any, stubs: list[str]) -> None:
    f_stubs = ", ".join(stubs)
    warn(
        f"Array module {xp.__name__} does not have the following "
        f"dtypes in its namespace: {f_stubs}",
        HypothesisWarning,
        stacklevel=3,
    )


def find_castable_builtin_for_dtype(
    xp: Any, api_version: NominalVersion, dtype: DataType
) -> type[Union[bool, int, float, complex]]:
    """Returns builtin type which can have values that are castable to the given
    dtype, according to :xp-ref:`type promotion rules <type_promotion.html>`.

    For floating dtypes we always return ``float``, even though ``int`` is also castable.
    """
    stubs = []

    try:
        bool_dtype = xp.bool
        if dtype == bool_dtype:
            return bool
    except AttributeError:
        stubs.append("bool")

    int_dtypes, int_stubs = partition_attributes_and_stubs(xp, ALL_INT_NAMES)
    if dtype in int_dtypes:
        return int

    float_dtypes, float_stubs = partition_attributes_and_stubs(xp, FLOAT_NAMES)
    # None equals NumPy's xp.float64 object, so we specifically skip it here to
    # ensure that InvalidArgument is still raised. xp.float64 is in fact an
    # alias of np.dtype('float64'), and its equality with None is meant to be
    # deprecated at some point. See https://github.com/numpy/numpy/issues/18434
    if dtype is not None and dtype in float_dtypes:
        return float

    stubs.extend(int_stubs)
    stubs.extend(float_stubs)

    if api_version > "2021.12":
        complex_dtypes, complex_stubs = partition_attributes_and_stubs(
            xp, COMPLEX_NAMES
        )
        if dtype in complex_dtypes:
            return complex
        stubs.extend(complex_stubs)

    if len(stubs) > 0:
        warn_on_missing_dtypes(xp, stubs)
    raise InvalidArgument(f"dtype={dtype} not recognised in {xp.__name__}")


@check_function
def dtype_from_name(xp: Any, name: str) -> Any:
    if name in DTYPE_NAMES:
        try:
            return getattr(xp, name)
        except AttributeError as e:
            raise InvalidArgument(
                f"Array module {xp.__name__} does not have dtype {name} in its namespace"
            ) from e
    else:
        f_valid_dtypes = ", ".join(DTYPE_NAMES)
        raise InvalidArgument(
            f"{name} is not a valid Array API data type (pick from: {f_valid_dtypes})"
        )


def _from_dtype(
    xp: Any,
    api_version: NominalVersion,
    dtype: Union[DataType, str],
    *,
    min_value: Optional[Union[int, float]] = None,
    max_value: Optional[Union[int, float]] = None,
    allow_nan: Optional[bool] = None,
    allow_infinity: Optional[bool] = None,
    allow_subnormal: Optional[bool] = None,
    exclude_min: Optional[bool] = None,
    exclude_max: Optional[bool] = None,
) -> st.SearchStrategy[Union[bool, int, float, complex]]:
    """Return a strategy for any value of the given dtype.

    Values generated are of the Python scalar which is
    :xp-ref:`promotable <type_promotion.html>` to ``dtype``, where the values do
    not exceed its bounds.

    * ``dtype`` may be a dtype object or the string name of a
      :xp-ref:`valid dtype <data_types.html>`.

    Compatible ``**kwargs`` are passed to the inferred strategy function for
    integers and floats.  This allows you to customise the min and max values,
    and exclude non-finite numbers. This is particularly useful when kwargs are
    passed through from :func:`arrays()`, as it seamlessly handles the ``width``
    or other representable bounds for you.
    """
    # TODO: for next released xp version, add note for complex dtype support
    check_xp_attributes(xp, ["iinfo", "finfo"])

    if isinstance(dtype, str):
        dtype = dtype_from_name(xp, dtype)
    builtin = find_castable_builtin_for_dtype(xp, api_version, dtype)

    def check_valid_minmax(prefix, val, info_obj):
        name = f"{prefix}_value"
        check_valid_bound(val, name)
        check_argument(
            val >= info_obj.min,
            f"dtype={dtype} requires {name}={val} to be at least {info_obj.min}",
        )
        check_argument(
            val <= info_obj.max,
            f"dtype={dtype} requires {name}={val} to be at most {info_obj.max}",
        )

    if builtin is bool:
        return st.booleans()
    elif builtin is int:
        iinfo = xp.iinfo(dtype)
        if min_value is None:
            min_value = iinfo.min
        if max_value is None:
            max_value = iinfo.max
        check_valid_integer(min_value, "min_value")
        check_valid_integer(max_value, "max_value")
        assert isinstance(min_value, int)
        assert isinstance(max_value, int)
        check_valid_minmax("min", min_value, iinfo)
        check_valid_minmax("max", max_value, iinfo)
        check_valid_interval(min_value, max_value, "min_value", "max_value")
        return st.integers(min_value=min_value, max_value=max_value)
    elif builtin is float:
        finfo = xp.finfo(dtype)
        kw = {}

        # Whilst we know the boundary values of float dtypes from finfo, we do
        # not assign them to the floats() strategy by default - passing min/max
        # values will modify test case reduction behaviour so that simple bugs
        # may become harder for users to identify. We plan to improve floats()
        # behaviour in https://github.com/HypothesisWorks/hypothesis/issues/2907.
        # Setting width should manage boundary values for us anyway.
        if min_value is not None:
            check_valid_bound(min_value, "min_value")
            assert isinstance(min_value, Real)
            check_valid_minmax("min", min_value, finfo)
            kw["min_value"] = min_value
        if max_value is not None:
            check_valid_bound(max_value, "max_value")
            assert isinstance(max_value, Real)
            check_valid_minmax("max", max_value, finfo)
            if min_value is not None:
                check_valid_interval(min_value, max_value, "min_value", "max_value")
            kw["max_value"] = max_value

        # We infer whether an array module will flush subnormals to zero, as may
        # be the case when libraries are built with compiler options that
        # violate IEEE-754 (e.g. -ffast-math and -ftz=true). Note we do this for
        # the specific dtype, as compilers may end up flushing subnormals for
        # one float but supporting subnormals for the other.
        #
        # By default, floats() will generate subnormals if they are in the
        # inferred values range. If we have detected that xp flushes to zero for
        # the passed dtype, we ensure from_dtype() will not generate subnormals
        # by default.
        if allow_subnormal is not None:
            kw["allow_subnormal"] = allow_subnormal
        else:
            subnormal = next_down(float(finfo.smallest_normal), width=finfo.bits)
            ftz = bool(xp.asarray(subnormal, dtype=dtype) == 0)
            if ftz:
                kw["allow_subnormal"] = False

        if allow_nan is not None:
            kw["allow_nan"] = allow_nan
        if allow_infinity is not None:
            kw["allow_infinity"] = allow_infinity
        if exclude_min is not None:
            kw["exclude_min"] = exclude_min
        if exclude_max is not None:
            kw["exclude_max"] = exclude_max

        return st.floats(width=finfo.bits, **kw)
    else:
        finfo = xp.finfo(dtype)
        # See above comment on FTZ inference. We explicitly infer with a
        # complex array, in case complex arrays have different FTZ behaviour
        # than arrays of the respective composite float.
        if allow_subnormal is None:
            subnormal = next_down(float(finfo.smallest_normal), width=finfo.bits)
            x = xp.asarray(complex(subnormal, subnormal), dtype=dtype)
            builtin_x = complex(x)
            allow_subnormal = builtin_x.real != 0 and builtin_x.imag != 0
        return st.complex_numbers(
            allow_nan=allow_nan,
            allow_infinity=allow_infinity,
            allow_subnormal=allow_subnormal,
            width=finfo.bits * 2,
        )


class ArrayStrategy(st.SearchStrategy):
    def __init__(
        self, *, xp, api_version, elements_strategy, dtype, shape, fill, unique
    ):
        self.xp = xp
        self.elements_strategy = elements_strategy
        self.dtype = dtype
        self.shape = shape
        self.fill = fill
        self.unique = unique
        self.array_size = math.prod(shape)
        self.builtin = find_castable_builtin_for_dtype(xp, api_version, dtype)
        self.finfo = None if self.builtin is not float else xp.finfo(self.dtype)

    def check_set_value(self, val, val_0d, strategy):
        if val == val and self.builtin(val_0d) != val:
            if self.builtin is float:
                assert self.finfo is not None  # for mypy
                try:
                    is_subnormal = 0 < abs(val) < self.finfo.smallest_normal
                except Exception:
                    # val may be a non-float that does not support the
                    # operations __lt__ and __abs__
                    is_subnormal = False
                if is_subnormal:
                    raise InvalidArgument(
                        f"Generated subnormal float {val} from strategy "
                        f"{strategy} resulted in {val_0d!r}, probably "
                        f"as a result of array module {self.xp.__name__} "
                        "being built with flush-to-zero compiler options. "
                        "Consider passing allow_subnormal=False."
                    )
            raise InvalidArgument(
                f"Generated array element {val!r} from strategy {strategy} "
                f"cannot be represented with dtype {self.dtype}. "
                f"Array module {self.xp.__name__} instead "
                f"represents the element as {val_0d}. "
                "Consider using a more precise elements strategy, "
                "for example passing the width argument to floats()."
            )

    def do_draw(self, data):
        if 0 in self.shape:
            return self.xp.zeros(self.shape, dtype=self.dtype)

        if self.fill.is_empty:
            # We have no fill value (either because the user explicitly
            # disabled it or because the default behaviour was used and our
            # elements strategy does not produce reusable values), so we must
            # generate a fully dense array with a freshly drawn value for each
            # entry.
            elems = data.draw(
                st.lists(
                    self.elements_strategy,
                    min_size=self.array_size,
                    max_size=self.array_size,
                    unique=self.unique,
                )
            )
            try:
                result = self.xp.asarray(elems, dtype=self.dtype)
            except Exception as e:
                if len(elems) <= 6:
                    f_elems = str(elems)
                else:
                    f_elems = f"[{elems[0]}, {elems[1]}, ..., {elems[-2]}, {elems[-1]}]"
                types = tuple(
                    sorted({type(e) for e in elems}, key=lambda t: t.__name__)
                )
                f_types = f"type {types[0]}" if len(types) == 1 else f"types {types}"
                raise InvalidArgument(
                    f"Generated elements {f_elems} from strategy "
                    f"{self.elements_strategy} could not be converted "
                    f"to array of dtype {self.dtype}. "
                    f"Consider if elements of {f_types} "
                    f"are compatible with {self.dtype}."
                ) from e
            for i in range(self.array_size):
                self.check_set_value(elems[i], result[i], self.elements_strategy)
        else:
            # We draw arrays as "sparse with an offset". We assume not every
            # element will be assigned and so first draw a single value from our
            # fill strategy to create a full array. We then draw a collection of
            # index assignments within the array and assign fresh values from
            # our elements strategy to those indices.

            fill_val = data.draw(self.fill)
            result_obj = [fill_val for _ in range(self.array_size)]
            fill_mask = [True for _ in range(self.array_size)]

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

            assigned = set()
            seen = set()

            while elements.more():
                i = data.draw_integer(0, self.array_size - 1)
                if i in assigned:
                    elements.reject("chose an array index we've already used")
                    continue
                val = data.draw(self.elements_strategy)
                if self.unique:
                    if val in seen:
                        elements.reject("chose an element we've already used")
                        continue
                    else:
                        seen.add(val)
                result_obj[i] = val
                assigned.add(i)
                fill_mask[i] = False

            try:
                result = self.xp.asarray(result_obj, dtype=self.dtype)
            except Exception as e:
                f_expr = f"xp.asarray({result_obj}, dtype={self.dtype})"
                raise InvalidArgument(f"Could not create array via {f_expr}") from e

            for i, val in enumerate(result_obj):
                val_0d = result[i]
                if fill_mask[i] and self.unique:
                    if not self.xp.isnan(val_0d):
                        raise InvalidArgument(
                            f"Array module {self.xp.__name__} did not recognise fill "
                            f"value {fill_val!r} as NaN - instead got {val_0d!r}. "
                            "Cannot fill unique array with non-NaN values."
                        )
                else:
                    self.check_set_value(val, val_0d, self.elements_strategy)

        return self.xp.reshape(result, self.shape)


def _arrays(
    xp: Any,
    api_version: NominalVersion,
    dtype: Union[DataType, str, st.SearchStrategy[DataType], st.SearchStrategy[str]],
    shape: Union[int, Shape, st.SearchStrategy[Shape]],
    *,
    elements: Optional[Union[Mapping[str, Any], st.SearchStrategy]] = None,
    fill: Optional[st.SearchStrategy[Any]] = None,
    unique: bool = False,
) -> st.SearchStrategy:
    """Returns a strategy for :xp-ref:`arrays <array_object.html>`.

    * ``dtype`` may be a :xp-ref:`valid dtype <data_types.html>` object or name,
      or a strategy that generates such values.
    * ``shape`` may be an integer >= 0, a tuple of such integers, or a strategy
      that generates such values.
    * ``elements`` is a strategy for values to put in the array. If ``None``
      then a suitable value will be inferred based on the dtype, which may give
      any legal value (including e.g. NaN for floats). If a mapping, it will be
      passed as ``**kwargs`` to :func:`from_dtype()` when inferring based on the dtype.
    * ``fill`` is a strategy that may be used to generate a single background
      value for the array. If ``None``, a suitable default will be inferred
      based on the other arguments. If set to
      :func:`~hypothesis.strategies.nothing` then filling behaviour will be
      disabled entirely and every element will be generated independently.
    * ``unique`` specifies if the elements of the array should all be distinct
      from one another; if fill is also set, the only valid values for fill to
      return are NaN values.

    Arrays of specified ``dtype`` and ``shape`` are generated for example
    like this:

    .. code-block:: pycon

      >>> from numpy import array_api as xp
      >>> xps.arrays(xp, xp.int8, (2, 3)).example()
      Array([[-8,  6,  3],
             [-6,  4,  6]], dtype=int8)

    Specifying element boundaries by a :obj:`python:dict` of the kwargs to pass
    to :func:`from_dtype` will ensure ``dtype`` bounds will be respected.

    .. code-block:: pycon

      >>> xps.arrays(xp, xp.int8, 3, elements={"min_value": 10}).example()
      Array([125, 13, 79], dtype=int8)

    Refer to :doc:`What you can generate and how <data>` for passing
    your own elements strategy.

    .. code-block:: pycon

      >>> xps.arrays(xp, xp.float32, 3, elements=floats(0, 1, width=32)).example()
      Array([ 0.88974794,  0.77387938,  0.1977879 ], dtype=float32)

    Array values are generated in two parts:

    1. A single value is drawn from the fill strategy and is used to create a
       filled array.
    2. Some subset of the coordinates of the array are populated with a value
       drawn from the elements strategy (or its inferred form).

    You can set ``fill`` to :func:`~hypothesis.strategies.nothing` if you want
    to disable this behaviour and draw a value for every element.

    By default ``arrays`` will attempt to infer the correct fill behaviour: if
    ``unique`` is also ``True``, no filling will occur. Otherwise, if it looks
    safe to reuse the values of elements across multiple coordinates (this will
    be the case for any inferred strategy, and for most of the builtins, but is
    not the case for mutable values or strategies built with flatmap, map,
    composite, etc.) then it will use the elements strategy as the fill, else it
    will default to having no fill.

    Having a fill helps Hypothesis craft high quality examples, but its
    main importance is when the array generated is large: Hypothesis is
    primarily designed around testing small examples. If you have arrays with
    hundreds or more elements, having a fill value is essential if you want
    your tests to run in reasonable time.
    """
    check_xp_attributes(
        xp, ["finfo", "asarray", "zeros", "all", "isnan", "isfinite", "reshape"]
    )

    if isinstance(dtype, st.SearchStrategy):
        return dtype.flatmap(
            lambda d: _arrays(
                xp, api_version, d, shape, elements=elements, fill=fill, unique=unique
            )
        )
    elif isinstance(dtype, str):
        dtype = dtype_from_name(xp, dtype)

    if isinstance(shape, st.SearchStrategy):
        return shape.flatmap(
            lambda s: _arrays(
                xp, api_version, dtype, s, elements=elements, fill=fill, unique=unique
            )
        )
    elif isinstance(shape, int):
        shape = (shape,)
    elif not isinstance(shape, tuple):
        raise InvalidArgument(f"shape={shape} is not a valid shape or strategy")
    check_argument(
        all(isinstance(x, int) and x >= 0 for x in shape),
        f"{shape=}, but all dimensions must be non-negative integers.",
    )

    if elements is None:
        elements = _from_dtype(xp, api_version, dtype)
    elif isinstance(elements, Mapping):
        elements = _from_dtype(xp, api_version, dtype, **elements)
    check_strategy(elements, "elements")

    if fill is None:
        assert isinstance(elements, st.SearchStrategy)  # for mypy
        if unique or not elements.has_reusable_values:
            fill = st.nothing()
        else:
            fill = elements
    check_strategy(fill, "fill")

    return ArrayStrategy(
        xp=xp,
        api_version=api_version,
        elements_strategy=elements,
        dtype=dtype,
        shape=shape,
        fill=fill,
        unique=unique,
    )


@check_function
def check_dtypes(xp: Any, dtypes: list[DataType], stubs: list[str]) -> None:
    if len(dtypes) == 0:
        assert len(stubs) > 0, "No dtypes passed but stubs is empty"
        f_stubs = ", ".join(stubs)
        raise InvalidArgument(
            f"Array module {xp.__name__} does not have the following "
            f"required dtypes in its namespace: {f_stubs}"
        )
    elif len(stubs) > 0:
        warn_on_missing_dtypes(xp, stubs)


def _scalar_dtypes(xp: Any, api_version: NominalVersion) -> st.SearchStrategy[DataType]:
    """Return a strategy for all :xp-ref:`valid dtype <data_types.html>` objects."""
    return st.one_of(_boolean_dtypes(xp), _numeric_dtypes(xp, api_version))


def _boolean_dtypes(xp: Any) -> st.SearchStrategy[DataType]:
    """Return a strategy for just the boolean dtype object."""
    try:
        return st.just(xp.bool)
    except AttributeError:
        raise InvalidArgument(
            f"Array module {xp.__name__} does not have a bool dtype in its namespace"
        ) from None


def _real_dtypes(xp: Any) -> st.SearchStrategy[DataType]:
    """Return a strategy for all real-valued dtype objects."""
    return st.one_of(
        _integer_dtypes(xp),
        _unsigned_integer_dtypes(xp),
        _floating_dtypes(xp),
    )


def _numeric_dtypes(
    xp: Any, api_version: NominalVersion
) -> st.SearchStrategy[DataType]:
    """Return a strategy for all numeric dtype objects."""
    strat: st.SearchStrategy[DataType] = _real_dtypes(xp)
    if api_version > "2021.12":
        strat |= _complex_dtypes(xp)
    return strat


@check_function
def check_valid_sizes(
    category: str, sizes: Sequence[int], valid_sizes: Sequence[int]
) -> None:
    check_argument(len(sizes) > 0, "No sizes passed")

    invalid_sizes = [s for s in sizes if s not in valid_sizes]
    f_valid_sizes = ", ".join(str(s) for s in valid_sizes)
    f_invalid_sizes = ", ".join(str(s) for s in invalid_sizes)
    check_argument(
        len(invalid_sizes) == 0,
        f"The following sizes are not valid for {category} dtypes: "
        f"{f_invalid_sizes} (valid sizes: {f_valid_sizes})",
    )


def numeric_dtype_names(base_name: str, sizes: Sequence[int]) -> Iterator[str]:
    for size in sizes:
        yield f"{base_name}{size}"


IntSize: "TypeAlias" = Literal[8, 16, 32, 64]
FltSize: "TypeAlias" = Literal[32, 64]
CpxSize: "TypeAlias" = Literal[64, 128]


def _integer_dtypes(
    xp: Any, *, sizes: Union[IntSize, Sequence[IntSize]] = (8, 16, 32, 64)
) -> st.SearchStrategy[DataType]:
    """Return a strategy for signed integer dtype objects.

    ``sizes`` contains the signed integer sizes in bits, defaulting to
    ``(8, 16, 32, 64)`` which covers all valid sizes.
    """
    if isinstance(sizes, int):
        sizes = (sizes,)
    check_valid_sizes("int", sizes, (8, 16, 32, 64))
    dtypes, stubs = partition_attributes_and_stubs(
        xp, numeric_dtype_names("int", sizes)
    )
    check_dtypes(xp, dtypes, stubs)
    return st.sampled_from(dtypes)


def _unsigned_integer_dtypes(
    xp: Any, *, sizes: Union[IntSize, Sequence[IntSize]] = (8, 16, 32, 64)
) -> st.SearchStrategy[DataType]:
    """Return a strategy for unsigned integer dtype objects.

    ``sizes`` contains the unsigned integer sizes in bits, defaulting to
    ``(8, 16, 32, 64)`` which covers all valid sizes.
    """
    if isinstance(sizes, int):
        sizes = (sizes,)
    check_valid_sizes("int", sizes, (8, 16, 32, 64))

    dtypes, stubs = partition_attributes_and_stubs(
        xp, numeric_dtype_names("uint", sizes)
    )
    check_dtypes(xp, dtypes, stubs)

    return st.sampled_from(dtypes)


def _floating_dtypes(
    xp: Any, *, sizes: Union[FltSize, Sequence[FltSize]] = (32, 64)
) -> st.SearchStrategy[DataType]:
    """Return a strategy for real-valued floating-point dtype objects.

    ``sizes`` contains the floating-point sizes in bits, defaulting to
    ``(32, 64)`` which covers all valid sizes.
    """
    if isinstance(sizes, int):
        sizes = (sizes,)
    check_valid_sizes("int", sizes, (32, 64))
    dtypes, stubs = partition_attributes_and_stubs(
        xp, numeric_dtype_names("float", sizes)
    )
    check_dtypes(xp, dtypes, stubs)
    return st.sampled_from(dtypes)


def _complex_dtypes(
    xp: Any, *, sizes: Union[CpxSize, Sequence[CpxSize]] = (64, 128)
) -> st.SearchStrategy[DataType]:
    """Return a strategy for complex dtype objects.

    ``sizes`` contains the complex sizes in bits, defaulting to ``(64, 128)``
    which covers all valid sizes.
    """
    if isinstance(sizes, int):
        sizes = (sizes,)
    check_valid_sizes("complex", sizes, (64, 128))
    dtypes, stubs = partition_attributes_and_stubs(
        xp, numeric_dtype_names("complex", sizes)
    )
    check_dtypes(xp, dtypes, stubs)
    return st.sampled_from(dtypes)


@proxies(_valid_tuple_axes)
def valid_tuple_axes(*args, **kwargs):
    return _valid_tuple_axes(*args, **kwargs)


valid_tuple_axes.__doc__ = f"""
    Return a strategy for permissible tuple-values for the ``axis``
    argument in Array API sequential methods e.g. ``sum``, given the specified
    dimensionality.

    {_valid_tuple_axes.__doc__}
    """


@defines_strategy()
def mutually_broadcastable_shapes(
    num_shapes: int,
    *,
    base_shape: Shape = (),
    min_dims: int = 0,
    max_dims: Optional[int] = None,
    min_side: int = 1,
    max_side: Optional[int] = None,
) -> st.SearchStrategy[BroadcastableShapes]:
    return _mutually_broadcastable_shapes(
        num_shapes=num_shapes,
        base_shape=base_shape,
        min_dims=min_dims,
        max_dims=max_dims,
        min_side=min_side,
        max_side=max_side,
    )


mutually_broadcastable_shapes.__doc__ = _mutually_broadcastable_shapes.__doc__


@defines_strategy()
def indices(
    shape: Shape,
    *,
    min_dims: int = 0,
    max_dims: Optional[int] = None,
    allow_newaxis: bool = False,
    allow_ellipsis: bool = True,
) -> st.SearchStrategy[BasicIndex]:
    """Return a strategy for :xp-ref:`valid indices <indexing.html>` of
    arrays with the specified shape, which may include dimensions of size zero.

    It generates tuples containing some mix of integers, :obj:`python:slice`
    objects, ``...`` (an ``Ellipsis``), and ``None``. When a length-one tuple
    would be generated, this strategy may instead return the element which will
    index the first axis, e.g. ``5`` instead of ``(5,)``.

    * ``shape`` is the shape of the array that will be indexed, as a tuple of
      integers >= 0. This must be at least two-dimensional for a tuple to be a
      valid index;  for one-dimensional arrays use
      :func:`~hypothesis.strategies.slices` instead.
    * ``min_dims`` is the minimum dimensionality of the resulting array from use
      of the generated index.
    * ``max_dims`` is the the maximum dimensionality of the resulting array,
      defaulting to ``len(shape) if not allow_newaxis else
      max(len(shape), min_dims) + 2``.
    * ``allow_ellipsis`` specifies whether ``None`` is allowed in the index.
    * ``allow_ellipsis`` specifies whether ``...`` is allowed in the index.
    """
    check_type(tuple, shape, "shape")
    check_argument(
        all(isinstance(x, int) and x >= 0 for x in shape),
        f"{shape=}, but all dimensions must be non-negative integers.",
    )
    check_type(bool, allow_newaxis, "allow_newaxis")
    check_type(bool, allow_ellipsis, "allow_ellipsis")
    check_type(int, min_dims, "min_dims")
    if not allow_newaxis:
        check_argument(
            min_dims <= len(shape),
            f"min_dims={min_dims} is larger than len(shape)={len(shape)}, "
            "but it is impossible for an indexing operation to add dimensions ",
            "when allow_newaxis=False.",
        )
    check_valid_dims(min_dims, "min_dims")

    if max_dims is None:
        if allow_newaxis:
            max_dims = min(max(len(shape), min_dims) + 2, NDIM_MAX)
        else:
            max_dims = min(len(shape), NDIM_MAX)
    check_type(int, max_dims, "max_dims")
    assert isinstance(max_dims, int)
    if not allow_newaxis:
        check_argument(
            max_dims <= len(shape),
            f"max_dims={max_dims} is larger than len(shape)={len(shape)}, "
            "but it is impossible for an indexing operation to add dimensions ",
            "when allow_newaxis=False.",
        )
    check_valid_dims(max_dims, "max_dims")

    order_check("dims", 0, min_dims, max_dims)

    return BasicIndexStrategy(
        shape,
        min_dims=min_dims,
        max_dims=max_dims,
        allow_ellipsis=allow_ellipsis,
        allow_newaxis=allow_newaxis,
        allow_fewer_indices_than_dims=False,
    )


# Cache for make_strategies_namespace()
_args_to_xps: WeakValueDictionary = WeakValueDictionary()


def make_strategies_namespace(
    xp: Any, *, api_version: Optional[NominalVersion] = None
) -> SimpleNamespace:
    """Creates a strategies namespace for the given array module.

    * ``xp`` is the Array API library to automatically pass to the namespaced methods.
    * ``api_version`` is the version of the Array API which the returned
      strategies namespace should conform to. If ``None``, the latest API
      version which ``xp`` supports will be inferred from ``xp.__array_api_version__``.
      If a version string in the ``YYYY.MM`` format, the strategies namespace
      will conform to that version if supported.

    A :obj:`python:types.SimpleNamespace` is returned which contains all the
    strategy methods in this module but without requiring the ``xp`` argument.
    Creating and using a strategies namespace for NumPy's Array API
    implementation would go like this:

    .. code-block:: pycon

      >>> xp.__array_api_version__  # xp is your desired array library
      '2021.12'
      >>> xps = make_strategies_namespace(xp)
      >>> xps.api_version
      '2021.12'
      >>> x = xps.arrays(xp.int8, (2, 3)).example()
      >>> x
      Array([[-8,  6,  3],
             [-6,  4,  6]], dtype=int8)
      >>> x.__array_namespace__() is xp
      True

    """
    not_available_msg = (
        "If the standard version you want is not available, please ensure "
        "you're using the latest version of Hypothesis, then open an issue if "
        "one doesn't already exist."
    )
    if api_version is None:
        check_argument(
            hasattr(xp, "__array_api_version__"),
            f"Array module {xp.__name__} has no attribute __array_api_version__, "
            "which is required when inferring api_version. If you believe "
            f"{xp.__name__} is indeed an Array API module, try explicitly "
            "passing an api_version.",
        )
        check_argument(
            isinstance(xp.__array_api_version__, str)
            and xp.__array_api_version__ in RELEASED_VERSIONS,
            f"{xp.__array_api_version__=}, but it must "
            f"be a valid version string {RELEASED_VERSIONS}. {not_available_msg}",
        )
        api_version = xp.__array_api_version__
        inferred_version = True
    else:
        check_argument(
            isinstance(api_version, str) and api_version in NOMINAL_VERSIONS,
            f"{api_version=}, but it must be None, or a valid version "
            f"string in {RELEASED_VERSIONS}. {not_available_msg}",
        )
        inferred_version = False
    try:
        array = xp.zeros(1)
        array.__array_namespace__()
    except Exception:
        warn(
            f"Could not determine whether module {xp.__name__} is an Array API library",
            HypothesisWarning,
            stacklevel=2,
        )

    try:
        namespace = _args_to_xps[(xp, api_version)]
    except (KeyError, TypeError):
        pass
    else:
        return namespace

    @defines_strategy(force_reusable_values=True)
    def from_dtype(
        dtype: Union[DataType, str],
        *,
        min_value: Optional[Union[int, float]] = None,
        max_value: Optional[Union[int, float]] = None,
        allow_nan: Optional[bool] = None,
        allow_infinity: Optional[bool] = None,
        allow_subnormal: Optional[bool] = None,
        exclude_min: Optional[bool] = None,
        exclude_max: Optional[bool] = None,
    ) -> st.SearchStrategy[Union[bool, int, float, complex]]:
        return _from_dtype(
            xp,
            api_version,
            dtype,
            min_value=min_value,
            max_value=max_value,
            allow_nan=allow_nan,
            allow_infinity=allow_infinity,
            allow_subnormal=allow_subnormal,
            exclude_min=exclude_min,
            exclude_max=exclude_max,
        )

    @defines_strategy(force_reusable_values=True)
    def arrays(
        dtype: Union[
            DataType, str, st.SearchStrategy[DataType], st.SearchStrategy[str]
        ],
        shape: Union[int, Shape, st.SearchStrategy[Shape]],
        *,
        elements: Optional[Union[Mapping[str, Any], st.SearchStrategy]] = None,
        fill: Optional[st.SearchStrategy[Any]] = None,
        unique: bool = False,
    ) -> st.SearchStrategy:
        return _arrays(
            xp,
            api_version,
            dtype,
            shape,
            elements=elements,
            fill=fill,
            unique=unique,
        )

    @defines_strategy()
    def scalar_dtypes() -> st.SearchStrategy[DataType]:
        return _scalar_dtypes(xp, api_version)

    @defines_strategy()
    def boolean_dtypes() -> st.SearchStrategy[DataType]:
        return _boolean_dtypes(xp)

    @defines_strategy()
    def real_dtypes() -> st.SearchStrategy[DataType]:
        return _real_dtypes(xp)

    @defines_strategy()
    def numeric_dtypes() -> st.SearchStrategy[DataType]:
        return _numeric_dtypes(xp, api_version)

    @defines_strategy()
    def integer_dtypes(
        *, sizes: Union[IntSize, Sequence[IntSize]] = (8, 16, 32, 64)
    ) -> st.SearchStrategy[DataType]:
        return _integer_dtypes(xp, sizes=sizes)

    @defines_strategy()
    def unsigned_integer_dtypes(
        *, sizes: Union[IntSize, Sequence[IntSize]] = (8, 16, 32, 64)
    ) -> st.SearchStrategy[DataType]:
        return _unsigned_integer_dtypes(xp, sizes=sizes)

    @defines_strategy()
    def floating_dtypes(
        *, sizes: Union[FltSize, Sequence[FltSize]] = (32, 64)
    ) -> st.SearchStrategy[DataType]:
        return _floating_dtypes(xp, sizes=sizes)

    from_dtype.__doc__ = _from_dtype.__doc__
    arrays.__doc__ = _arrays.__doc__
    scalar_dtypes.__doc__ = _scalar_dtypes.__doc__
    boolean_dtypes.__doc__ = _boolean_dtypes.__doc__
    real_dtypes.__doc__ = _real_dtypes.__doc__
    numeric_dtypes.__doc__ = _numeric_dtypes.__doc__
    integer_dtypes.__doc__ = _integer_dtypes.__doc__
    unsigned_integer_dtypes.__doc__ = _unsigned_integer_dtypes.__doc__
    floating_dtypes.__doc__ = _floating_dtypes.__doc__

    class StrategiesNamespace(SimpleNamespace):
        def __init__(self, **kwargs):
            for attr in ["name", "api_version"]:
                if attr not in kwargs:
                    raise ValueError(f"'{attr}' kwarg required")
            super().__init__(**kwargs)

        @property
        def complex_dtypes(self):
            try:
                return self.__dict__["complex_dtypes"]
            except KeyError as e:
                raise AttributeError(
                    "You attempted to access 'complex_dtypes', but it is not "
                    f"available for api_version='{self.api_version}' of "
                    f"xp={self.name}."
                ) from e

        def __repr__(self):
            f_args = self.name
            if not inferred_version:
                f_args += f", api_version='{self.api_version}'"
            return f"make_strategies_namespace({f_args})"

    kwargs = dict(
        name=xp.__name__,
        api_version=api_version,
        from_dtype=from_dtype,
        arrays=arrays,
        array_shapes=array_shapes,
        scalar_dtypes=scalar_dtypes,
        boolean_dtypes=boolean_dtypes,
        real_dtypes=real_dtypes,
        numeric_dtypes=numeric_dtypes,
        integer_dtypes=integer_dtypes,
        unsigned_integer_dtypes=unsigned_integer_dtypes,
        floating_dtypes=floating_dtypes,
        valid_tuple_axes=valid_tuple_axes,
        broadcastable_shapes=broadcastable_shapes,
        mutually_broadcastable_shapes=mutually_broadcastable_shapes,
        indices=indices,
    )

    if api_version > "2021.12":

        @defines_strategy()
        def complex_dtypes(
            *, sizes: Union[CpxSize, Sequence[CpxSize]] = (64, 128)
        ) -> st.SearchStrategy[DataType]:
            return _complex_dtypes(xp, sizes=sizes)

        complex_dtypes.__doc__ = _complex_dtypes.__doc__
        kwargs["complex_dtypes"] = complex_dtypes

    namespace = StrategiesNamespace(**kwargs)
    try:
        _args_to_xps[(xp, api_version)] = namespace
    except TypeError:
        pass

    return namespace


try:
    import numpy as np
except ImportError:
    if "sphinx" in sys.modules:
        # This is pretty awkward, but also the best way available
        from unittest.mock import Mock

        np = Mock()
    else:
        np = None  # type: ignore[assignment]
if np is not None:

    class FloatInfo(NamedTuple):
        bits: int
        eps: float
        max: float
        min: float
        smallest_normal: float

    def mock_finfo(dtype: DataType) -> FloatInfo:
        """Returns a finfo object compliant with the Array API

        Ensures all attributes are Python scalars and not NumPy scalars. This
        lets us ignore corner cases with how NumPy scalars operate, such as
        NumPy floats breaking our next_down() util.

        Also ensures the finfo obj has the smallest_normal attribute. NumPy only
        introduced it in v1.21.1, so we just use the equivalent tiny attribute
        to keep mocking with older versions working.
        """
        _finfo = np.finfo(dtype)  # type: ignore[call-overload]
        return FloatInfo(
            int(_finfo.bits),
            float(_finfo.eps),
            float(_finfo.max),
            float(_finfo.min),
            float(_finfo.tiny),
        )

    mock_xp = SimpleNamespace(
        __name__="mock",
        __array_api_version__="2022.12",
        # Data types
        int8=np.int8,
        int16=np.int16,
        int32=np.int32,
        int64=np.int64,
        uint8=np.uint8,
        uint16=np.uint16,
        uint32=np.uint32,
        uint64=np.uint64,
        float32=np.float32,
        float64=np.float64,
        complex64=np.complex64,
        complex128=np.complex128,
        bool=np.bool_,
        # Constants
        nan=np.nan,
        # Data type functions
        astype=lambda x, d: x.astype(d),
        iinfo=np.iinfo,
        finfo=mock_finfo,
        broadcast_arrays=np.broadcast_arrays,
        # Creation functions
        arange=np.arange,
        asarray=np.asarray,
        empty=np.empty,
        zeros=np.zeros,
        ones=np.ones,
        # Manipulation functions
        reshape=np.reshape,
        # Element-wise functions
        isnan=np.isnan,
        isfinite=np.isfinite,
        logical_or=np.logical_or,
        # Statistical functions
        sum=np.sum,
        # Searching functions
        nonzero=np.nonzero,
        # Sorting functions
        sort=np.sort,
        # Set functions
        unique_values=np.unique,
        # Utility functions
        any=np.any,
        all=np.all,
    )
