# ruff: noqa: ANN401
from collections.abc import Callable, Iterable, Sequence
from typing import (
    Any,
    Concatenate,
    ParamSpec,
    Protocol,
    SupportsIndex,
    SupportsInt,
    TypeAlias,
    TypeVar,
    overload,
    type_check_only,
)
from typing import Literal as L

from _typeshed import Incomplete
from typing_extensions import TypeIs, deprecated

import numpy as np
from numpy import (
    _OrderKACF,
    bool_,
    complex128,
    complexfloating,
    datetime64,
    float64,
    floating,
    generic,
    integer,
    intp,
    object_,
    timedelta64,
    vectorize,
)
from numpy._core.multiarray import bincount
from numpy._globals import _NoValueType
from numpy._typing import (
    ArrayLike,
    DTypeLike,
    NDArray,
    _ArrayLike,
    _ArrayLikeBool_co,
    _ArrayLikeComplex_co,
    _ArrayLikeDT64_co,
    _ArrayLikeFloat_co,
    _ArrayLikeInt_co,
    _ArrayLikeNumber_co,
    _ArrayLikeObject_co,
    _ArrayLikeTD64_co,
    _ComplexLike_co,
    _DTypeLike,
    _FloatLike_co,
    _NestedSequence,
    _NumberLike_co,
    _ScalarLike_co,
    _ShapeLike,
)

__all__ = [
    "select",
    "piecewise",
    "trim_zeros",
    "copy",
    "iterable",
    "percentile",
    "diff",
    "gradient",
    "angle",
    "unwrap",
    "sort_complex",
    "flip",
    "rot90",
    "extract",
    "place",
    "vectorize",
    "asarray_chkfinite",
    "average",
    "bincount",
    "digitize",
    "cov",
    "corrcoef",
    "median",
    "sinc",
    "hamming",
    "hanning",
    "bartlett",
    "blackman",
    "kaiser",
    "trapezoid",
    "trapz",
    "i0",
    "meshgrid",
    "delete",
    "insert",
    "append",
    "interp",
    "quantile",
]

_T = TypeVar("_T")
_T_co = TypeVar("_T_co", covariant=True)
# The `{}ss` suffix refers to the Python 3.12 syntax: `**P`
_Pss = ParamSpec("_Pss")
_ScalarT = TypeVar("_ScalarT", bound=generic)
_ScalarT1 = TypeVar("_ScalarT1", bound=generic)
_ScalarT2 = TypeVar("_ScalarT2", bound=generic)
_ArrayT = TypeVar("_ArrayT", bound=np.ndarray)

_2Tuple: TypeAlias = tuple[_T, _T]
_MeshgridIdx: TypeAlias = L['ij', 'xy']

@type_check_only
class _TrimZerosSequence(Protocol[_T_co]):
    def __len__(self, /) -> int: ...
    @overload
    def __getitem__(self, key: int, /) -> object: ...
    @overload
    def __getitem__(self, key: slice, /) -> _T_co: ...

###

@overload
def rot90(
    m: _ArrayLike[_ScalarT],
    k: int = ...,
    axes: tuple[int, int] = ...,
) -> NDArray[_ScalarT]: ...
@overload
def rot90(
    m: ArrayLike,
    k: int = ...,
    axes: tuple[int, int] = ...,
) -> NDArray[Any]: ...

@overload
def flip(m: _ScalarT, axis: None = ...) -> _ScalarT: ...
@overload
def flip(m: _ScalarLike_co, axis: None = ...) -> Any: ...
@overload
def flip(m: _ArrayLike[_ScalarT], axis: _ShapeLike | None = ...) -> NDArray[_ScalarT]: ...
@overload
def flip(m: ArrayLike, axis: _ShapeLike | None = ...) -> NDArray[Any]: ...

def iterable(y: object) -> TypeIs[Iterable[Any]]: ...

@overload
def average(
    a: _ArrayLikeFloat_co,
    axis: None = None,
    weights: _ArrayLikeFloat_co | None = None,
    returned: L[False] = False,
    *,
    keepdims: L[False] | _NoValueType = ...,
) -> floating: ...
@overload
def average(
    a: _ArrayLikeFloat_co,
    axis: None = None,
    weights: _ArrayLikeFloat_co | None = None,
    *,
    returned: L[True],
    keepdims: L[False] | _NoValueType = ...,
) -> _2Tuple[floating]: ...
@overload
def average(
    a: _ArrayLikeComplex_co,
    axis: None = None,
    weights: _ArrayLikeComplex_co | None = None,
    returned: L[False] = False,
    *,
    keepdims: L[False] | _NoValueType = ...,
) -> complexfloating: ...
@overload
def average(
    a: _ArrayLikeComplex_co,
    axis: None = None,
    weights: _ArrayLikeComplex_co | None = None,
    *,
    returned: L[True],
    keepdims: L[False] | _NoValueType = ...,
) -> _2Tuple[complexfloating]: ...
@overload
def average(
    a: _ArrayLikeComplex_co | _ArrayLikeObject_co,
    axis: _ShapeLike | None = None,
    weights: object | None = None,
    *,
    returned: L[True],
    keepdims: bool | bool_ | _NoValueType = ...,
) -> _2Tuple[Incomplete]: ...
@overload
def average(
    a: _ArrayLikeComplex_co | _ArrayLikeObject_co,
    axis: _ShapeLike | None = None,
    weights: object | None = None,
    returned: bool | bool_ = False,
    *,
    keepdims: bool | bool_ | _NoValueType = ...,
) -> Incomplete: ...

@overload
def asarray_chkfinite(
    a: _ArrayLike[_ScalarT],
    dtype: None = ...,
    order: _OrderKACF = ...,
) -> NDArray[_ScalarT]: ...
@overload
def asarray_chkfinite(
    a: object,
    dtype: None = ...,
    order: _OrderKACF = ...,
) -> NDArray[Any]: ...
@overload
def asarray_chkfinite(
    a: Any,
    dtype: _DTypeLike[_ScalarT],
    order: _OrderKACF = ...,
) -> NDArray[_ScalarT]: ...
@overload
def asarray_chkfinite(
    a: Any,
    dtype: DTypeLike,
    order: _OrderKACF = ...,
) -> NDArray[Any]: ...

@overload
def piecewise(
    x: _ArrayLike[_ScalarT],
    condlist: _ArrayLike[bool_] | Sequence[_ArrayLikeBool_co],
    funclist: Sequence[
        Callable[Concatenate[NDArray[_ScalarT], _Pss], NDArray[_ScalarT | Any]]
        | _ScalarT | object
    ],
    *args: _Pss.args,
    **kw: _Pss.kwargs,
) -> NDArray[_ScalarT]: ...
@overload
def piecewise(
    x: ArrayLike,
    condlist: _ArrayLike[bool_] | Sequence[_ArrayLikeBool_co],
    funclist: Sequence[
        Callable[Concatenate[NDArray[Any], _Pss], NDArray[Any]]
        | object
    ],
    *args: _Pss.args,
    **kw: _Pss.kwargs,
) -> NDArray[Any]: ...

def select(
    condlist: Sequence[ArrayLike],
    choicelist: Sequence[ArrayLike],
    default: ArrayLike = ...,
) -> NDArray[Any]: ...

@overload
def copy(
    a: _ArrayT,
    order: _OrderKACF,
    subok: L[True],
) -> _ArrayT: ...
@overload
def copy(
    a: _ArrayT,
    order: _OrderKACF = ...,
    *,
    subok: L[True],
) -> _ArrayT: ...
@overload
def copy(
    a: _ArrayLike[_ScalarT],
    order: _OrderKACF = ...,
    subok: L[False] = ...,
) -> NDArray[_ScalarT]: ...
@overload
def copy(
    a: ArrayLike,
    order: _OrderKACF = ...,
    subok: L[False] = ...,
) -> NDArray[Any]: ...

def gradient(
    f: ArrayLike,
    *varargs: ArrayLike,
    axis: _ShapeLike | None = ...,
    edge_order: L[1, 2] = ...,
) -> Any: ...

@overload
def diff(  # type: ignore[overload-overlap]
    a: _T,
    n: L[0],
    axis: SupportsIndex = -1,
    prepend: ArrayLike | _NoValueType = ...,  # = _NoValue
    append: ArrayLike | _NoValueType = ...,  # = _NoValue
) -> _T: ...
@overload
def diff(
    a: ArrayLike,
    n: int = 1,
    axis: SupportsIndex = -1,
    prepend: ArrayLike | _NoValueType = ...,  # = _NoValue
    append: ArrayLike | _NoValueType = ...,  # = _NoValue
) -> NDArray[Incomplete]: ...

@overload  # float scalar
def interp(
    x: _FloatLike_co,
    xp: _ArrayLikeFloat_co,
    fp: _ArrayLikeFloat_co,
    left: _FloatLike_co | None = None,
    right: _FloatLike_co | None = None,
    period: _FloatLike_co | None = None,
) -> float64: ...
@overload  # float array
def interp(
    x: NDArray[floating | integer | np.bool] | _NestedSequence[_FloatLike_co],
    xp: _ArrayLikeFloat_co,
    fp: _ArrayLikeFloat_co,
    left: _FloatLike_co | None = None,
    right: _FloatLike_co | None = None,
    period: _FloatLike_co | None = None,
) -> NDArray[float64]: ...
@overload  # float scalar or array
def interp(
    x: _ArrayLikeFloat_co,
    xp: _ArrayLikeFloat_co,
    fp: _ArrayLikeFloat_co,
    left: _FloatLike_co | None = None,
    right: _FloatLike_co | None = None,
    period: _FloatLike_co | None = None,
) -> NDArray[float64] | float64: ...
@overload  # complex scalar
def interp(
    x: _FloatLike_co,
    xp: _ArrayLikeFloat_co,
    fp: _ArrayLike[complexfloating],
    left: _NumberLike_co | None = None,
    right: _NumberLike_co | None = None,
    period: _FloatLike_co | None = None,
) -> complex128: ...
@overload  # complex or float scalar
def interp(
    x: _FloatLike_co,
    xp: _ArrayLikeFloat_co,
    fp: Sequence[complex | complexfloating],
    left: _NumberLike_co | None = None,
    right: _NumberLike_co | None = None,
    period: _FloatLike_co | None = None,
) -> complex128 | float64: ...
@overload  # complex array
def interp(
    x: NDArray[floating | integer | np.bool] | _NestedSequence[_FloatLike_co],
    xp: _ArrayLikeFloat_co,
    fp: _ArrayLike[complexfloating],
    left: _NumberLike_co | None = None,
    right: _NumberLike_co | None = None,
    period: _FloatLike_co | None = None,
) -> NDArray[complex128]: ...
@overload  # complex or float array
def interp(
    x: NDArray[floating | integer | np.bool] | _NestedSequence[_FloatLike_co],
    xp: _ArrayLikeFloat_co,
    fp: Sequence[complex | complexfloating],
    left: _NumberLike_co | None = None,
    right: _NumberLike_co | None = None,
    period: _FloatLike_co | None = None,
) -> NDArray[complex128 | float64]: ...
@overload  # complex scalar or array
def interp(
    x: _ArrayLikeFloat_co,
    xp: _ArrayLikeFloat_co,
    fp: _ArrayLike[complexfloating],
    left: _NumberLike_co | None = None,
    right: _NumberLike_co | None = None,
    period: _FloatLike_co | None = None,
) -> NDArray[complex128] | complex128: ...
@overload  # complex or float scalar or array
def interp(
    x: _ArrayLikeFloat_co,
    xp: _ArrayLikeFloat_co,
    fp: _ArrayLikeNumber_co,
    left: _NumberLike_co | None = None,
    right: _NumberLike_co | None = None,
    period: _FloatLike_co | None = None,
) -> NDArray[complex128 | float64] | complex128 | float64: ...

@overload
def angle(z: _ComplexLike_co, deg: bool = ...) -> floating: ...
@overload
def angle(z: object_, deg: bool = ...) -> Any: ...
@overload
def angle(z: _ArrayLikeComplex_co, deg: bool = ...) -> NDArray[floating]: ...
@overload
def angle(z: _ArrayLikeObject_co, deg: bool = ...) -> NDArray[object_]: ...

@overload
def unwrap(
    p: _ArrayLikeFloat_co,
    discont: float | None = ...,
    axis: int = ...,
    *,
    period: float = ...,
) -> NDArray[floating]: ...
@overload
def unwrap(
    p: _ArrayLikeObject_co,
    discont: float | None = ...,
    axis: int = ...,
    *,
    period: float = ...,
) -> NDArray[object_]: ...

def sort_complex(a: ArrayLike) -> NDArray[complexfloating]: ...

def trim_zeros(
    filt: _TrimZerosSequence[_T],
    trim: L["f", "b", "fb", "bf"] = "fb",
    axis: _ShapeLike | None = None,
) -> _T: ...

@overload
def extract(condition: ArrayLike, arr: _ArrayLike[_ScalarT]) -> NDArray[_ScalarT]: ...
@overload
def extract(condition: ArrayLike, arr: ArrayLike) -> NDArray[Any]: ...

def place(arr: NDArray[Any], mask: ArrayLike, vals: Any) -> None: ...

@overload
def cov(
    m: _ArrayLikeFloat_co,
    y: _ArrayLikeFloat_co | None = ...,
    rowvar: bool = ...,
    bias: bool = ...,
    ddof: SupportsIndex | SupportsInt | None = ...,
    fweights: ArrayLike | None = ...,
    aweights: ArrayLike | None = ...,
    *,
    dtype: None = ...,
) -> NDArray[floating]: ...
@overload
def cov(
    m: _ArrayLikeComplex_co,
    y: _ArrayLikeComplex_co | None = ...,
    rowvar: bool = ...,
    bias: bool = ...,
    ddof: SupportsIndex | SupportsInt | None = ...,
    fweights: ArrayLike | None = ...,
    aweights: ArrayLike | None = ...,
    *,
    dtype: None = ...,
) -> NDArray[complexfloating]: ...
@overload
def cov(
    m: _ArrayLikeComplex_co,
    y: _ArrayLikeComplex_co | None = ...,
    rowvar: bool = ...,
    bias: bool = ...,
    ddof: SupportsIndex | SupportsInt | None = ...,
    fweights: ArrayLike | None = ...,
    aweights: ArrayLike | None = ...,
    *,
    dtype: _DTypeLike[_ScalarT],
) -> NDArray[_ScalarT]: ...
@overload
def cov(
    m: _ArrayLikeComplex_co,
    y: _ArrayLikeComplex_co | None = ...,
    rowvar: bool = ...,
    bias: bool = ...,
    ddof: SupportsIndex | SupportsInt | None = ...,
    fweights: ArrayLike | None = ...,
    aweights: ArrayLike | None = ...,
    *,
    dtype: DTypeLike,
) -> NDArray[Any]: ...

# NOTE `bias` and `ddof` are deprecated and ignored
@overload
def corrcoef(
    x: _ArrayLikeFloat_co,
    y: _ArrayLikeFloat_co | None = None,
    rowvar: bool = True,
    bias: _NoValueType = ...,
    ddof: _NoValueType = ...,
    *,
    dtype: None = None,
) -> NDArray[floating]: ...
@overload
def corrcoef(
    x: _ArrayLikeComplex_co,
    y: _ArrayLikeComplex_co | None = None,
    rowvar: bool = True,
    bias: _NoValueType = ...,
    ddof: _NoValueType = ...,
    *,
    dtype: None = None,
) -> NDArray[complexfloating]: ...
@overload
def corrcoef(
    x: _ArrayLikeComplex_co,
    y: _ArrayLikeComplex_co | None = None,
    rowvar: bool = True,
    bias: _NoValueType = ...,
    ddof: _NoValueType = ...,
    *,
    dtype: _DTypeLike[_ScalarT],
) -> NDArray[_ScalarT]: ...
@overload
def corrcoef(
    x: _ArrayLikeComplex_co,
    y: _ArrayLikeComplex_co | None = None,
    rowvar: bool = True,
    bias: _NoValueType = ...,
    ddof: _NoValueType = ...,
    *,
    dtype: DTypeLike | None = None,
) -> NDArray[Any]: ...

def blackman(M: _FloatLike_co) -> NDArray[floating]: ...

def bartlett(M: _FloatLike_co) -> NDArray[floating]: ...

def hanning(M: _FloatLike_co) -> NDArray[floating]: ...

def hamming(M: _FloatLike_co) -> NDArray[floating]: ...

def i0(x: _ArrayLikeFloat_co) -> NDArray[floating]: ...

def kaiser(
    M: _FloatLike_co,
    beta: _FloatLike_co,
) -> NDArray[floating]: ...

@overload
def sinc(x: _FloatLike_co) -> floating: ...
@overload
def sinc(x: _ComplexLike_co) -> complexfloating: ...
@overload
def sinc(x: _ArrayLikeFloat_co) -> NDArray[floating]: ...
@overload
def sinc(x: _ArrayLikeComplex_co) -> NDArray[complexfloating]: ...

@overload
def median(
    a: _ArrayLikeFloat_co,
    axis: None = ...,
    out: None = ...,
    overwrite_input: bool = ...,
    keepdims: L[False] = ...,
) -> floating: ...
@overload
def median(
    a: _ArrayLikeComplex_co,
    axis: None = ...,
    out: None = ...,
    overwrite_input: bool = ...,
    keepdims: L[False] = ...,
) -> complexfloating: ...
@overload
def median(
    a: _ArrayLikeTD64_co,
    axis: None = ...,
    out: None = ...,
    overwrite_input: bool = ...,
    keepdims: L[False] = ...,
) -> timedelta64: ...
@overload
def median(
    a: _ArrayLikeObject_co,
    axis: None = ...,
    out: None = ...,
    overwrite_input: bool = ...,
    keepdims: L[False] = ...,
) -> Any: ...
@overload
def median(
    a: _ArrayLikeFloat_co | _ArrayLikeComplex_co | _ArrayLikeTD64_co | _ArrayLikeObject_co,
    axis: _ShapeLike | None = ...,
    out: None = ...,
    overwrite_input: bool = ...,
    keepdims: bool = ...,
) -> Any: ...
@overload
def median(
    a: _ArrayLikeFloat_co | _ArrayLikeComplex_co | _ArrayLikeTD64_co | _ArrayLikeObject_co,
    axis: _ShapeLike | None,
    out: _ArrayT,
    overwrite_input: bool = ...,
    keepdims: bool = ...,
) -> _ArrayT: ...
@overload
def median(
    a: _ArrayLikeFloat_co | _ArrayLikeComplex_co | _ArrayLikeTD64_co | _ArrayLikeObject_co,
    axis: _ShapeLike | None = ...,
    *,
    out: _ArrayT,
    overwrite_input: bool = ...,
    keepdims: bool = ...,
) -> _ArrayT: ...

_MethodKind = L[
    "inverted_cdf",
    "averaged_inverted_cdf",
    "closest_observation",
    "interpolated_inverted_cdf",
    "hazen",
    "weibull",
    "linear",
    "median_unbiased",
    "normal_unbiased",
    "lower",
    "higher",
    "midpoint",
    "nearest",
]

@overload
def percentile(
    a: _ArrayLikeFloat_co,
    q: _FloatLike_co,
    axis: None = ...,
    out: None = ...,
    overwrite_input: bool = ...,
    method: _MethodKind = ...,
    keepdims: L[False] = ...,
    *,
    weights: _ArrayLikeFloat_co | None = None,
    interpolation: None = None,  # deprecated
) -> floating: ...
@overload
def percentile(
    a: _ArrayLikeComplex_co,
    q: _FloatLike_co,
    axis: None = ...,
    out: None = ...,
    overwrite_input: bool = ...,
    method: _MethodKind = ...,
    keepdims: L[False] = ...,
    *,
    weights: _ArrayLikeFloat_co | None = None,
    interpolation: None = None,  # deprecated
) -> complexfloating: ...
@overload
def percentile(
    a: _ArrayLikeTD64_co,
    q: _FloatLike_co,
    axis: None = ...,
    out: None = ...,
    overwrite_input: bool = ...,
    method: _MethodKind = ...,
    keepdims: L[False] = ...,
    *,
    weights: _ArrayLikeFloat_co | None = None,
    interpolation: None = None,  # deprecated
) -> timedelta64: ...
@overload
def percentile(
    a: _ArrayLikeDT64_co,
    q: _FloatLike_co,
    axis: None = ...,
    out: None = ...,
    overwrite_input: bool = ...,
    method: _MethodKind = ...,
    keepdims: L[False] = ...,
    *,
    weights: _ArrayLikeFloat_co | None = None,
    interpolation: None = None,  # deprecated
) -> datetime64: ...
@overload
def percentile(
    a: _ArrayLikeObject_co,
    q: _FloatLike_co,
    axis: None = ...,
    out: None = ...,
    overwrite_input: bool = ...,
    method: _MethodKind = ...,
    keepdims: L[False] = ...,
    *,
    weights: _ArrayLikeFloat_co | None = None,
    interpolation: None = None,  # deprecated
) -> Any: ...
@overload
def percentile(
    a: _ArrayLikeFloat_co,
    q: _ArrayLikeFloat_co,
    axis: None = ...,
    out: None = ...,
    overwrite_input: bool = ...,
    method: _MethodKind = ...,
    keepdims: L[False] = ...,
    *,
    weights: _ArrayLikeFloat_co | None = None,
    interpolation: None = None,  # deprecated
) -> NDArray[floating]: ...
@overload
def percentile(
    a: _ArrayLikeComplex_co,
    q: _ArrayLikeFloat_co,
    axis: None = ...,
    out: None = ...,
    overwrite_input: bool = ...,
    method: _MethodKind = ...,
    keepdims: L[False] = ...,
    *,
    weights: _ArrayLikeFloat_co | None = None,
    interpolation: None = None,  # deprecated
) -> NDArray[complexfloating]: ...
@overload
def percentile(
    a: _ArrayLikeTD64_co,
    q: _ArrayLikeFloat_co,
    axis: None = ...,
    out: None = ...,
    overwrite_input: bool = ...,
    method: _MethodKind = ...,
    keepdims: L[False] = ...,
    *,
    weights: _ArrayLikeFloat_co | None = None,
    interpolation: None = None,  # deprecated
) -> NDArray[timedelta64]: ...
@overload
def percentile(
    a: _ArrayLikeDT64_co,
    q: _ArrayLikeFloat_co,
    axis: None = ...,
    out: None = ...,
    overwrite_input: bool = ...,
    method: _MethodKind = ...,
    keepdims: L[False] = ...,
    *,
    weights: _ArrayLikeFloat_co | None = None,
    interpolation: None = None,  # deprecated
) -> NDArray[datetime64]: ...
@overload
def percentile(
    a: _ArrayLikeObject_co,
    q: _ArrayLikeFloat_co,
    axis: None = ...,
    out: None = ...,
    overwrite_input: bool = ...,
    method: _MethodKind = ...,
    keepdims: L[False] = ...,
    *,
    weights: _ArrayLikeFloat_co | None = None,
    interpolation: None = None,  # deprecated
) -> NDArray[object_]: ...
@overload
def percentile(
    a: _ArrayLikeComplex_co | _ArrayLikeTD64_co | _ArrayLikeDT64_co | _ArrayLikeObject_co,
    q: _ArrayLikeFloat_co,
    axis: _ShapeLike | None = ...,
    out: None = ...,
    overwrite_input: bool = ...,
    method: _MethodKind = ...,
    keepdims: bool = ...,
    *,
    weights: _ArrayLikeFloat_co | None = None,
    interpolation: None = None,  # deprecated
) -> Any: ...
@overload
def percentile(
    a: _ArrayLikeComplex_co | _ArrayLikeTD64_co | _ArrayLikeDT64_co | _ArrayLikeObject_co,
    q: _ArrayLikeFloat_co,
    axis: _ShapeLike | None,
    out: _ArrayT,
    overwrite_input: bool = ...,
    method: _MethodKind = ...,
    keepdims: bool = ...,
    *,
    weights: _ArrayLikeFloat_co | None = None,
    interpolation: None = None,  # deprecated
) -> _ArrayT: ...
@overload
def percentile(
    a: _ArrayLikeComplex_co | _ArrayLikeTD64_co | _ArrayLikeDT64_co | _ArrayLikeObject_co,
    q: _ArrayLikeFloat_co,
    axis: _ShapeLike | None = ...,
    *,
    out: _ArrayT,
    overwrite_input: bool = False,
    method: _MethodKind = "linear",
    keepdims: bool = False,
    weights: _ArrayLikeFloat_co | None = None,
    interpolation: None = None,  # deprecated
) -> _ArrayT: ...

# NOTE: keep in sync with `percentile`
@overload
def quantile(
    a: _ArrayLikeFloat_co,
    q: _FloatLike_co,
    axis: None = None,
    out: None = None,
    overwrite_input: bool = False,
    method: _MethodKind = "linear",
    keepdims: L[False] = False,
    *,
    weights: _ArrayLikeFloat_co | None = None,
    interpolation: None = None,  # deprecated
) -> floating: ...
@overload
def quantile(
    a: _ArrayLikeComplex_co,
    q: _FloatLike_co,
    axis: None = None,
    out: None = None,
    overwrite_input: bool = False,
    method: _MethodKind = "linear",
    keepdims: L[False] = False,
    *,
    weights: _ArrayLikeFloat_co | None = None,
    interpolation: None = None,  # deprecated
) -> complexfloating: ...
@overload
def quantile(
    a: _ArrayLikeTD64_co,
    q: _FloatLike_co,
    axis: None = None,
    out: None = None,
    overwrite_input: bool = False,
    method: _MethodKind = "linear",
    keepdims: L[False] = False,
    *,
    weights: _ArrayLikeFloat_co | None = None,
    interpolation: None = None,  # deprecated
) -> timedelta64: ...
@overload
def quantile(
    a: _ArrayLikeDT64_co,
    q: _FloatLike_co,
    axis: None = None,
    out: None = None,
    overwrite_input: bool = False,
    method: _MethodKind = "linear",
    keepdims: L[False] = False,
    *,
    weights: _ArrayLikeFloat_co | None = None,
    interpolation: None = None,  # deprecated
) -> datetime64: ...
@overload
def quantile(
    a: _ArrayLikeObject_co,
    q: _FloatLike_co,
    axis: None = None,
    out: None = None,
    overwrite_input: bool = False,
    method: _MethodKind = "linear",
    keepdims: L[False] = False,
    *,
    weights: _ArrayLikeFloat_co | None = None,
    interpolation: None = None,  # deprecated
) -> Any: ...
@overload
def quantile(
    a: _ArrayLikeFloat_co,
    q: _ArrayLikeFloat_co,
    axis: None = None,
    out: None = None,
    overwrite_input: bool = False,
    method: _MethodKind = "linear",
    keepdims: L[False] = False,
    *,
    weights: _ArrayLikeFloat_co | None = None,
    interpolation: None = None,  # deprecated
) -> NDArray[floating]: ...
@overload
def quantile(
    a: _ArrayLikeComplex_co,
    q: _ArrayLikeFloat_co,
    axis: None = None,
    out: None = None,
    overwrite_input: bool = False,
    method: _MethodKind = "linear",
    keepdims: L[False] = False,
    *,
    weights: _ArrayLikeFloat_co | None = None,
    interpolation: None = None,  # deprecated
) -> NDArray[complexfloating]: ...
@overload
def quantile(
    a: _ArrayLikeTD64_co,
    q: _ArrayLikeFloat_co,
    axis: None = None,
    out: None = None,
    overwrite_input: bool = False,
    method: _MethodKind = "linear",
    keepdims: L[False] = False,
    *,
    weights: _ArrayLikeFloat_co | None = None,
    interpolation: None = None,  # deprecated
) -> NDArray[timedelta64]: ...
@overload
def quantile(
    a: _ArrayLikeDT64_co,
    q: _ArrayLikeFloat_co,
    axis: None = None,
    out: None = None,
    overwrite_input: bool = False,
    method: _MethodKind = "linear",
    keepdims: L[False] = False,
    *,
    weights: _ArrayLikeFloat_co | None = None,
    interpolation: None = None,  # deprecated
) -> NDArray[datetime64]: ...
@overload
def quantile(
    a: _ArrayLikeObject_co,
    q: _ArrayLikeFloat_co,
    axis: None = None,
    out: None = None,
    overwrite_input: bool = False,
    method: _MethodKind = "linear",
    keepdims: L[False] = False,
    *,
    weights: _ArrayLikeFloat_co | None = None,
    interpolation: None = None,  # deprecated
) -> NDArray[object_]: ...
@overload
def quantile(
    a: _ArrayLikeComplex_co | _ArrayLikeTD64_co | _ArrayLikeDT64_co | _ArrayLikeObject_co,
    q: _ArrayLikeFloat_co,
    axis: _ShapeLike | None = None,
    out: None = None,
    overwrite_input: bool = False,
    method: _MethodKind = "linear",
    keepdims: bool = False,
    *,
    weights: _ArrayLikeFloat_co | None = None,
    interpolation: None = None,  # deprecated
) -> Any: ...
@overload
def quantile(
    a: _ArrayLikeComplex_co | _ArrayLikeTD64_co | _ArrayLikeDT64_co | _ArrayLikeObject_co,
    q: _ArrayLikeFloat_co,
    axis: _ShapeLike | None,
    out: _ArrayT,
    overwrite_input: bool = False,
    method: _MethodKind = "linear",
    keepdims: bool = False,
    *,
    weights: _ArrayLikeFloat_co | None = None,
    interpolation: None = None,  # deprecated
) -> _ArrayT: ...
@overload
def quantile(
    a: _ArrayLikeComplex_co | _ArrayLikeTD64_co | _ArrayLikeDT64_co | _ArrayLikeObject_co,
    q: _ArrayLikeFloat_co,
    axis: _ShapeLike | None = None,
    *,
    out: _ArrayT,
    overwrite_input: bool = False,
    method: _MethodKind = "linear",
    keepdims: bool = False,
    weights: _ArrayLikeFloat_co | None = None,
    interpolation: None = None,  # deprecated
) -> _ArrayT: ...

_ScalarT_fm = TypeVar(
    "_ScalarT_fm",
    bound=floating | complexfloating | timedelta64,
)

class _SupportsRMulFloat(Protocol[_T_co]):
    def __rmul__(self, other: float, /) -> _T_co: ...

@overload
def trapezoid(  # type: ignore[overload-overlap]
    y: Sequence[_FloatLike_co],
    x: Sequence[_FloatLike_co] | None = ...,
    dx: float = ...,
    axis: SupportsIndex = ...,
) -> float64: ...
@overload
def trapezoid(
    y: Sequence[_ComplexLike_co],
    x: Sequence[_ComplexLike_co] | None = ...,
    dx: float = ...,
    axis: SupportsIndex = ...,
) -> complex128: ...
@overload
def trapezoid(
    y: _ArrayLike[bool_ | integer],
    x: _ArrayLike[bool_ | integer] | None = ...,
    dx: float = ...,
    axis: SupportsIndex = ...,
) -> float64 | NDArray[float64]: ...
@overload
def trapezoid(  # type: ignore[overload-overlap]
    y: _ArrayLikeObject_co,
    x: _ArrayLikeFloat_co | _ArrayLikeObject_co | None = ...,
    dx: float = ...,
    axis: SupportsIndex = ...,
) -> float | NDArray[object_]: ...
@overload
def trapezoid(
    y: _ArrayLike[_ScalarT_fm],
    x: _ArrayLike[_ScalarT_fm] | _ArrayLikeInt_co | None = ...,
    dx: float = ...,
    axis: SupportsIndex = ...,
) -> _ScalarT_fm | NDArray[_ScalarT_fm]: ...
@overload
def trapezoid(
    y: Sequence[_SupportsRMulFloat[_T]],
    x: Sequence[_SupportsRMulFloat[_T] | _T] | None = ...,
    dx: float = ...,
    axis: SupportsIndex = ...,
) -> _T: ...
@overload
def trapezoid(
    y: _ArrayLikeComplex_co | _ArrayLikeTD64_co | _ArrayLikeObject_co,
    x: _ArrayLikeComplex_co | _ArrayLikeTD64_co | _ArrayLikeObject_co | None = ...,
    dx: float = ...,
    axis: SupportsIndex = ...,
) -> (
    floating | complexfloating | timedelta64
    | NDArray[floating | complexfloating | timedelta64 | object_]
): ...

@deprecated("Use 'trapezoid' instead")
def trapz(y: ArrayLike, x: ArrayLike | None = None, dx: float = 1.0, axis: int = -1) -> generic | NDArray[generic]: ...

@overload
def meshgrid(
    *,
    copy: bool = ...,
    sparse: bool = ...,
    indexing: _MeshgridIdx = ...,
) -> tuple[()]: ...
@overload
def meshgrid(
    x1: _ArrayLike[_ScalarT],
    /,
    *,
    copy: bool = ...,
    sparse: bool = ...,
    indexing: _MeshgridIdx = ...,
) -> tuple[NDArray[_ScalarT]]: ...
@overload
def meshgrid(
    x1: ArrayLike,
    /,
    *,
    copy: bool = ...,
    sparse: bool = ...,
    indexing: _MeshgridIdx = ...,
) -> tuple[NDArray[Any]]: ...
@overload
def meshgrid(
    x1: _ArrayLike[_ScalarT1],
    x2: _ArrayLike[_ScalarT2],
    /,
    *,
    copy: bool = ...,
    sparse: bool = ...,
    indexing: _MeshgridIdx = ...,
) -> tuple[NDArray[_ScalarT1], NDArray[_ScalarT2]]: ...
@overload
def meshgrid(
    x1: ArrayLike,
    x2: _ArrayLike[_ScalarT],
    /,
    *,
    copy: bool = ...,
    sparse: bool = ...,
    indexing: _MeshgridIdx = ...,
) -> tuple[NDArray[Any], NDArray[_ScalarT]]: ...
@overload
def meshgrid(
    x1: _ArrayLike[_ScalarT],
    x2: ArrayLike,
    /,
    *,
    copy: bool = ...,
    sparse: bool = ...,
    indexing: _MeshgridIdx = ...,
) -> tuple[NDArray[_ScalarT], NDArray[Any]]: ...
@overload
def meshgrid(
    x1: ArrayLike,
    x2: ArrayLike,
    /,
    *,
    copy: bool = ...,
    sparse: bool = ...,
    indexing: _MeshgridIdx = ...,
) -> tuple[NDArray[Any], NDArray[Any]]: ...
@overload
def meshgrid(
    x1: ArrayLike,
    x2: ArrayLike,
    x3: ArrayLike,
    /,
    *,
    copy: bool = ...,
    sparse: bool = ...,
    indexing: _MeshgridIdx = ...,
) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any]]: ...
@overload
def meshgrid(
    x1: ArrayLike,
    x2: ArrayLike,
    x3: ArrayLike,
    x4: ArrayLike,
    /,
    *,
    copy: bool = ...,
    sparse: bool = ...,
    indexing: _MeshgridIdx = ...,
) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any], NDArray[Any]]: ...
@overload
def meshgrid(
    *xi: ArrayLike,
    copy: bool = ...,
    sparse: bool = ...,
    indexing: _MeshgridIdx = ...,
) -> tuple[NDArray[Any], ...]: ...

@overload
def delete(
    arr: _ArrayLike[_ScalarT],
    obj: slice | _ArrayLikeInt_co,
    axis: SupportsIndex | None = ...,
) -> NDArray[_ScalarT]: ...
@overload
def delete(
    arr: ArrayLike,
    obj: slice | _ArrayLikeInt_co,
    axis: SupportsIndex | None = ...,
) -> NDArray[Any]: ...

@overload
def insert(
    arr: _ArrayLike[_ScalarT],
    obj: slice | _ArrayLikeInt_co,
    values: ArrayLike,
    axis: SupportsIndex | None = ...,
) -> NDArray[_ScalarT]: ...
@overload
def insert(
    arr: ArrayLike,
    obj: slice | _ArrayLikeInt_co,
    values: ArrayLike,
    axis: SupportsIndex | None = ...,
) -> NDArray[Any]: ...

def append(
    arr: ArrayLike,
    values: ArrayLike,
    axis: SupportsIndex | None = ...,
) -> NDArray[Any]: ...

@overload
def digitize(
    x: _FloatLike_co,
    bins: _ArrayLikeFloat_co,
    right: bool = ...,
) -> intp: ...
@overload
def digitize(
    x: _ArrayLikeFloat_co,
    bins: _ArrayLikeFloat_co,
    right: bool = ...,
) -> NDArray[intp]: ...
