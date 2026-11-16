from __future__ import annotations

import builtins
import math
import operator
from collections.abc import Sequence
from typing import ParamSpec, TYPE_CHECKING, TypeVar

import torch

from . import _dtypes, _dtypes_impl, _funcs, _ufuncs, _util
from ._normalizations import (
    ArrayLike,
    normalize_array_like,
    normalizer,
    NotImplementedType,
)


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator
    from typing import ClassVar, Protocol, TypeAlias
    from typing_extensions import CapsuleType

    from ._normalizations import Scalar

    _ScalarOrNestedList: TypeAlias = Scalar | list["_ScalarOrNestedList"]

    class _SupportsDLPack(Protocol):
        def __dlpack__(self, *, stream: int | None = None) -> CapsuleType: ...
        def __dlpack_device__(self) -> tuple[int, int]: ...


_P = ParamSpec("_P")
_R = TypeVar("_R")


# NB: `_funcs` and `_ufuncs` populate their module namespaces dynamically (via
# `vars()[name] = ...` loops), so the type checker cannot see attributes such as
# `_funcs.reshape` or `_ufuncs.equal`. Accesses of those attributes below carry a
# `# pyrefly: ignore[missing-attribute]` for this reason.


newaxis = None

FLAGS = [
    "C_CONTIGUOUS",
    "F_CONTIGUOUS",
    "OWNDATA",
    "WRITEABLE",
    "ALIGNED",
    "WRITEBACKIFCOPY",
    "FNC",
    "FORC",
    "BEHAVED",
    "CARRAY",
    "FARRAY",
]

SHORTHAND_TO_FLAGS = {
    "C": "C_CONTIGUOUS",
    "F": "F_CONTIGUOUS",
    "O": "OWNDATA",
    "W": "WRITEABLE",
    "A": "ALIGNED",
    "X": "WRITEBACKIFCOPY",
    "B": "BEHAVED",
    "CA": "CARRAY",
    "FA": "FARRAY",
}


class Flags:
    def __init__(self, flag_to_value: dict[str, bool]) -> None:
        invalid_keys = [k for k in flag_to_value if k not in FLAGS]
        if invalid_keys:
            raise AssertionError(f"Invalid flag keys: {invalid_keys}")
        self._flag_to_value = flag_to_value

    def __getattr__(self, attr: str) -> bool:
        if attr.islower() and attr.upper() in FLAGS:
            return self[attr.upper()]
        else:
            raise AttributeError(f"No flag attribute '{attr}'")

    def __getitem__(self, key: str) -> bool:
        if key in SHORTHAND_TO_FLAGS:
            key = SHORTHAND_TO_FLAGS[key]
        if key in FLAGS:
            try:
                return self._flag_to_value[key]
            except KeyError as e:
                raise NotImplementedError(f"{key=}") from e
        else:
            raise KeyError(f"No flag key '{key}'")

    def __setattr__(self, attr: str, value: object) -> None:
        if attr.islower() and attr.upper() in FLAGS:
            self[attr.upper()] = value
        else:
            super().__setattr__(attr, value)

    def __setitem__(self, key: str, value: object) -> None:
        if key in FLAGS or key in SHORTHAND_TO_FLAGS:
            raise NotImplementedError("Modifying flags is not implemented")
        else:
            raise KeyError(f"No flag key '{key}'")


def create_method(fn: Callable[_P, _R], name: str | None = None) -> Callable[_P, _R]:
    name = name or fn.__name__

    def f(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        return fn(*args, **kwargs)

    f.__name__ = name
    f.__qualname__ = f"ndarray.{name}"
    return f


# Map ndarray.name_method -> np.name_func
# If name_func == None, it means that name_method == name_func
methods: dict[str, str | None] = {
    "clip": None,
    "nonzero": None,
    "repeat": None,
    "round": None,
    "squeeze": None,
    "swapaxes": None,
    "ravel": None,
    # linalg
    "diagonal": None,
    "dot": None,
    "trace": None,
    # sorting
    "argsort": None,
    "searchsorted": None,
    # reductions
    "argmax": None,
    "argmin": None,
    "any": None,
    "all": None,
    "max": None,
    "min": None,
    "ptp": None,
    "sum": None,
    "prod": None,
    "mean": None,
    "var": None,
    "std": None,
    # scans
    "cumsum": None,
    "cumprod": None,
    # advanced indexing
    "take": None,
    "choose": None,
}

dunder: dict[str, str | None] = {
    "abs": "absolute",
    "invert": None,
    "pos": "positive",
    "neg": "negative",
    "gt": "greater",
    "lt": "less",
    "ge": "greater_equal",
    "le": "less_equal",
}

# dunder methods with right-looking and in-place variants
ri_dunder: dict[str, str | None] = {
    "add": None,
    "sub": "subtract",
    "mul": "multiply",
    "truediv": "divide",
    "floordiv": "floor_divide",
    "pow": "power",
    "mod": "remainder",
    "and": "bitwise_and",
    "or": "bitwise_or",
    "xor": "bitwise_xor",
    "lshift": "left_shift",
    "rshift": "right_shift",
    "matmul": None,
}


def _upcast_int_indices(index: object) -> object:
    if isinstance(index, torch.Tensor):
        if index.dtype in (torch.int8, torch.int16, torch.int32, torch.uint8):
            return index.to(torch.int64)
    elif isinstance(index, tuple):
        return tuple(_upcast_int_indices(i) for i in index)
    return index


def _has_advanced_indexing(index: Iterable[object]) -> bool:
    """Check if there's any advanced indexing"""
    return any(
        isinstance(idx, (Sequence, bool))
        or (isinstance(idx, torch.Tensor) and (idx.dtype == torch.bool or idx.ndim > 0))
        for idx in index
    )


def _numpy_compatible_indexing(index: object) -> tuple[object, ...]:
    """Convert scalar indices to lists when advanced indexing is present for NumPy compatibility."""
    if not isinstance(index, tuple):
        index = (index,)

    # Check if there's any advanced indexing (sequences, booleans, or tensors)
    has_advanced = _has_advanced_indexing(index)

    if not has_advanced:
        return index

    # Convert integer scalar indices to single-element lists when advanced indexing is present
    # Note: Do NOT convert boolean scalars (True/False) as they have special meaning in NumPy
    converted: list[object] = []
    for idx in index:
        if isinstance(idx, int) and not isinstance(idx, bool):
            # Integer scalars should be converted to lists
            converted.append([idx])
        elif (
            isinstance(idx, torch.Tensor)
            and idx.ndim == 0
            and not torch.is_floating_point(idx)
            and idx.dtype != torch.bool
        ):
            # Zero-dimensional tensors holding integers should be treated the same as integer scalars
            converted.append([idx])
        else:
            # Everything else (booleans, lists, slices, etc.) stays as is
            converted.append(idx)

    return tuple(converted)


def _get_bool_depth(s: object) -> tuple[bool, int]:
    """Returns the depth of a boolean sequence/tensor"""
    if isinstance(s, bool):
        return True, 0
    if isinstance(s, torch.Tensor) and s.dtype == torch.bool:
        return True, s.ndim
    if not (isinstance(s, Sequence) and s and s[0] != s):
        return False, 0
    is_bool, depth = _get_bool_depth(s[0])
    return is_bool, depth + 1


def _numpy_empty_ellipsis_patch(
    index: object, tensor_ndim: int
) -> tuple[
    tuple[object, ...],
    Callable[[ndarray], ndarray],
    Callable[[object], object],
]:
    """
    Patch for NumPy-compatible ellipsis behavior when ellipsis doesn't match any dimensions.

    In NumPy, when an ellipsis (...) doesn't actually match any dimensions of the input array,
    it still acts as a separator between advanced indices. PyTorch doesn't have this behavior.

    This function detects when we have:
    1. Advanced indexing on both sides of an ellipsis
    2. The ellipsis doesn't actually match any dimensions
    """
    if not isinstance(index, tuple):
        index = (index,)

    # Find ellipsis position
    ellipsis_pos = None
    for i, idx in enumerate(index):
        if idx is Ellipsis:
            ellipsis_pos = i
            break

    # If no ellipsis, no patch needed
    if ellipsis_pos is None:
        return index, lambda x: x, lambda x: x

    # Count non-ellipsis dimensions consumed by the index
    consumed_dims = 0
    for idx in index:
        is_bool, depth = _get_bool_depth(idx)
        if is_bool:
            consumed_dims += depth
        elif idx is Ellipsis or idx is None:
            continue
        else:
            consumed_dims += 1

    # Calculate how many dimensions the ellipsis should match
    ellipsis_dims = tensor_ndim - consumed_dims

    # Check if ellipsis doesn't match any dimensions
    if ellipsis_dims == 0:
        # Check if we have advanced indexing on both sides of ellipsis
        left_advanced = _has_advanced_indexing(index[:ellipsis_pos])
        right_advanced = _has_advanced_indexing(index[ellipsis_pos + 1 :])

        if left_advanced and right_advanced:
            # This is the case where NumPy and PyTorch differ
            # We need to ensure the advanced indices are treated as separated
            new_index = index[:ellipsis_pos] + (None,) + index[ellipsis_pos + 1 :]
            end_ndims = 1 + sum(
                1 for idx in index[ellipsis_pos + 1 :] if isinstance(idx, slice)
            )

            def squeeze_fn(x: ndarray) -> ndarray:
                return x.squeeze(-end_ndims)

            def unsqueeze_fn(x: object) -> object:
                if isinstance(x, torch.Tensor) and x.ndim >= end_ndims:
                    return x.unsqueeze(-end_ndims)
                return x

            return new_index, squeeze_fn, unsqueeze_fn

    return index, lambda x: x, lambda x: x


# Used to indicate that a parameter is unspecified (as opposed to explicitly
# `None`)
class _Unspecified:
    unspecified: ClassVar[_Unspecified]


_Unspecified.unspecified = _Unspecified()

###############################################################
#                      ndarray class                          #
###############################################################


class ndarray:
    def __init__(self, t: torch.Tensor | None = None) -> None:
        if t is None:
            self.tensor = torch.Tensor()
        elif isinstance(t, torch.Tensor):
            self.tensor = t
        else:
            raise ValueError(
                "ndarray constructor is not recommended; prefer "
                "either array(...) or zeros/empty(...)"
            )

    if TYPE_CHECKING:
        # These methods are created dynamically in the class body below (see the
        # `methods`/`dunder` loops with vars() assignments). Declare the ones
        # referenced within this module so the type checker can see them.
        def ravel(self) -> ndarray: ...
        def squeeze(self, axis: int | tuple[int, ...] | None = ...) -> ndarray: ...
        def __invert__(self) -> ndarray: ...

    # Register NumPy functions as methods
    for method, name in methods.items():
        fn = getattr(_funcs, name or method)
        vars()[method] = create_method(fn, method)

    # Regular methods but coming from ufuncs
    # pyrefly: ignore[missing-attribute]
    conj = create_method(_ufuncs.conjugate, "conj")
    # pyrefly: ignore[missing-attribute]
    conjugate = create_method(_ufuncs.conjugate)

    for method, name in dunder.items():
        fn = getattr(_ufuncs, name or method)
        method = f"__{method}__"
        vars()[method] = create_method(fn, method)

    for method, name in ri_dunder.items():
        fn = getattr(_ufuncs, name or method)
        plain = f"__{method}__"
        vars()[plain] = create_method(fn, plain)
        rvar = f"__r{method}__"
        vars()[rvar] = create_method(lambda self, other, fn=fn: fn(other, self), rvar)
        ivar = f"__i{method}__"
        vars()[ivar] = create_method(
            lambda self, other, fn=fn: fn(self, other, out=self), ivar
        )

    # There's no __idivmod__
    __divmod__ = create_method(_ufuncs.divmod, "__divmod__")
    __rdivmod__ = create_method(
        lambda self, other: _ufuncs.divmod(other, self), "__rdivmod__"
    )

    # prevent loop variables leaking into the ndarray class namespace.
    # The loop variables are always bound because the dicts above are non-empty
    # literals; pyrefly cannot prove that after the dict annotations.
    del ivar, rvar, name, plain, fn, method  # pyrefly: ignore[unbound-name]

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.tensor.shape)

    @property
    def size(self) -> int:
        return self.tensor.numel()

    @property
    def ndim(self) -> int:
        return self.tensor.ndim

    @property
    def dtype(self) -> _dtypes.DType:
        return _dtypes.dtype(self.tensor.dtype)

    @property
    def strides(self) -> tuple[int, ...]:
        elsize = self.tensor.element_size()
        return tuple(stride * elsize for stride in self.tensor.stride())

    @property
    def itemsize(self) -> int:
        return self.tensor.element_size()

    @property
    def flags(self) -> Flags:
        # Note contiguous in torch is assumed C-style
        return Flags(
            {
                "C_CONTIGUOUS": self.tensor.is_contiguous(),
                "F_CONTIGUOUS": self.T.tensor.is_contiguous(),
                "OWNDATA": self.tensor._base is None,
                "WRITEABLE": True,  # pytorch does not have readonly tensors
            }
        )

    @property
    def data(self) -> int:
        return self.tensor.data_ptr()

    @property
    def nbytes(self) -> int:
        return self.tensor.storage().nbytes()

    @property
    def T(self) -> ndarray:
        return self.transpose()

    @property
    def real(self) -> ndarray:
        return _funcs.real(self)  # pyrefly: ignore[missing-attribute]

    @real.setter
    def real(self, value: object) -> None:
        self.tensor.real = asarray(value).tensor

    @property
    def imag(self) -> ndarray:
        return _funcs.imag(self)  # pyrefly: ignore[missing-attribute]

    @imag.setter
    def imag(self, value: object) -> None:
        self.tensor.imag = asarray(value).tensor

    @property
    def flat(self) -> ndarray:
        return self.ravel()

    # ctors
    def astype(
        self,
        dtype: object,
        order: str = "K",
        casting: str = "unsafe",
        subok: bool = True,
        copy: bool = True,
    ) -> ndarray:
        if order != "K":
            raise NotImplementedError(f"astype(..., order={order} is not implemented.")
        if casting != "unsafe":
            raise NotImplementedError(
                f"astype(..., casting={casting} is not implemented."
            )
        if not subok:
            raise NotImplementedError(f"astype(..., subok={subok} is not implemented.")
        if not copy:
            raise NotImplementedError(f"astype(..., copy={copy} is not implemented.")
        torch_dtype = _dtypes.dtype(dtype).torch_dtype
        t = self.tensor.to(torch_dtype)
        return ndarray(t)

    @normalizer
    def copy(self: ArrayLike, order: NotImplementedType = "C") -> torch.Tensor:
        return self.clone()

    @normalizer
    def flatten(self: ArrayLike, order: NotImplementedType = "C") -> torch.Tensor:
        return torch.flatten(self)

    def resize(self, *new_shape: int | tuple[int, ...], refcheck: bool = False) -> None:
        # NB: differs from np.resize: fills with zeros instead of making repeated copies of input.
        if refcheck:
            raise NotImplementedError(
                f"resize(..., refcheck={refcheck} is not implemented."
            )
        if new_shape in [(), (None,)]:
            return

        # support both x.resize((2, 2)) and x.resize(2, 2)
        if len(new_shape) == 1 and isinstance(new_shape[0], (tuple, list)):
            shape: tuple[int, ...] = tuple(new_shape[0])
        else:
            # varargs form: each dim must be an int (numpy raises TypeError otherwise)
            dims: list[int] = []
            for s in new_shape:
                if not isinstance(s, int):
                    raise TypeError(
                        f"'{type(s).__name__}' object cannot be interpreted as an integer"
                    )
                dims.append(s)
            shape = tuple(dims)

        if builtins.any(x < 0 for x in shape):
            raise ValueError("all elements of `new_shape` must be non-negative")

        new_numel, old_numel = math.prod(shape), self.tensor.numel()

        self.tensor.resize_(shape)

        if new_numel >= old_numel:
            # zero-fill new elements
            if not self.tensor.is_contiguous():
                raise AssertionError("tensor must be contiguous for resize with growth")
            b = self.tensor.flatten()  # does not copy
            b[old_numel:].zero_()

    def view(
        self,
        dtype: object = _Unspecified.unspecified,
        type: object = _Unspecified.unspecified,
    ) -> ndarray:
        if dtype is _Unspecified.unspecified:
            dtype = self.dtype
        if type is not _Unspecified.unspecified:
            raise NotImplementedError(f"view(..., type={type} is not implemented.")
        torch_dtype = _dtypes.dtype(dtype).torch_dtype
        tview = self.tensor.view(torch_dtype)
        return ndarray(tview)

    @normalizer
    def fill(self, value: ArrayLike) -> None:
        # Both PyTorch and NumPy accept 0D arrays/tensors and scalars, and
        # error out on D > 0 arrays
        self.tensor.fill_(value)

    def tolist(self) -> _ScalarOrNestedList:
        return self.tensor.tolist()

    def __iter__(self) -> Iterator[ndarray]:
        return (ndarray(x) for x in self.tensor.__iter__())

    def __str__(self) -> str:
        return (
            str(self.tensor)
            .replace("tensor", "torch.ndarray")
            .replace("dtype=torch.", "dtype=")
        )

    __repr__ = create_method(__str__)  # pyrefly: ignore[bad-override]

    def __eq__(self, other: object) -> ndarray:  # pyrefly: ignore[bad-override]
        try:
            return _ufuncs.equal(self, other)  # pyrefly: ignore[missing-attribute]
        except (RuntimeError, TypeError):
            # Failed to convert other to array: definitely not equal.
            # torch.full accepts the python `bool` type as `dtype` at runtime;
            # the stub is too strict.
            # pyrefly: ignore[no-matching-overload]
            falsy = torch.full(self.shape, fill_value=False, dtype=bool)
            return asarray(falsy)

    def __ne__(self, other: object) -> ndarray:  # pyrefly: ignore[bad-override]
        return ~(self == other)

    def __index__(self) -> int:
        try:
            # item() may return a float, in which case operator.index raises
            # TypeError (caught below); this is the intended behavior.
            # pyrefly: ignore[bad-argument-type]
            return operator.index(self.tensor.item())
        except Exception as exc:
            raise TypeError(
                "only integer scalar arrays can be converted to a scalar index"
            ) from exc

    def __bool__(self) -> bool:
        return bool(self.tensor)

    def __int__(self) -> int:
        return int(self.tensor)

    def __float__(self) -> float:
        return float(self.tensor)

    def __complex__(self) -> complex:
        return complex(self.tensor)

    def is_integer(self) -> bool:
        try:
            v = self.tensor.item()
            result = int(v) == v
        except Exception:
            result = False
        return result

    def __len__(self) -> int:
        return self.tensor.shape[0]

    def __contains__(self, x: object) -> bool:
        return self.tensor.__contains__(x)

    def transpose(self, *axes: int | tuple[int, ...]) -> ndarray:
        # np.transpose(arr, axis=None) but arr.transpose(*axes)
        return _funcs.transpose(self, axes)  # pyrefly: ignore[missing-attribute]

    def reshape(self, *shape: int | tuple[int, ...], order: str = "C") -> ndarray:
        # arr.reshape(shape) and arr.reshape(*shape)
        # pyrefly: ignore[missing-attribute]
        return _funcs.reshape(self, shape, order=order)

    def sort(
        self, axis: int = -1, kind: str | None = None, order: object = None
    ) -> None:
        # ndarray.sort works in-place
        # pyrefly: ignore[missing-attribute]
        _funcs.copyto(self, _funcs.sort(self, axis, kind, order))

    def item(self, *args: int) -> Scalar:
        # Mimic NumPy's implementation with three special cases (no arguments,
        # a flat index and a multi-index):
        # https://github.com/numpy/numpy/blob/main/numpy/_core/src/multiarray/methods.c#L702
        if args == ():
            return self.tensor.item()
        elif len(args) == 1:
            # int argument
            return self.ravel()[args[0]].tensor.item()
        else:
            return self.__getitem__(args).tensor.item()

    def __getitem__(self, index: object) -> ndarray:
        tensor = self.tensor

        def neg_step(i: int, s: object) -> object:
            if not (isinstance(s, slice) and s.step is not None and s.step < 0):
                return s

            nonlocal tensor
            tensor = torch.flip(tensor, (i,))

            # Account for the fact that a slice includes the start but not the end
            if not (isinstance(s.start, int) or s.start is None):
                raise AssertionError(
                    f"slice start must be int or None, got {type(s.start).__name__}"
                )
            if not (isinstance(s.stop, int) or s.stop is None):
                raise AssertionError(
                    f"slice stop must be int or None, got {type(s.stop).__name__}"
                )
            start = s.stop + 1 if s.stop else None
            stop = s.start + 1 if s.start else None

            return slice(start, stop, -s.step)

        if isinstance(index, Sequence):
            # type(index) is a concrete list/tuple at runtime; pyrefly only sees
            # the abstract Sequence base.
            # pyrefly: ignore[bad-instantiation, bad-argument-count]
            index = type(index)(neg_step(i, s) for i, s in enumerate(index))
        else:
            index = neg_step(0, index)
        index = _util.ndarrays_to_tensors(index)
        index = _upcast_int_indices(index)
        # Apply NumPy-compatible indexing conversion
        index = _numpy_compatible_indexing(index)
        # Apply NumPy-compatible empty ellipsis behavior
        index, maybe_squeeze, _ = _numpy_empty_ellipsis_patch(index, tensor.ndim)
        # index is an arbitrary normalized numpy-style index object.
        # pyrefly: ignore[bad-argument-type]
        return maybe_squeeze(ndarray(tensor.__getitem__(index)))

    def __setitem__(self, index: object, value: object) -> None:
        index = _util.ndarrays_to_tensors(index)
        index = _upcast_int_indices(index)
        # Apply NumPy-compatible indexing conversion
        index = _numpy_compatible_indexing(index)
        # Apply NumPy-compatible empty ellipsis behavior
        index, _, maybe_unsqueeze = _numpy_empty_ellipsis_patch(index, self.tensor.ndim)

        if not _dtypes_impl.is_scalar(value):
            value = normalize_array_like(value)
            value = _util.cast_if_needed(value, self.tensor.dtype)

        # index/value are arbitrary normalized numpy-style objects.
        # pyrefly: ignore[bad-argument-type]
        return self.tensor.__setitem__(index, maybe_unsqueeze(value))

    take = _funcs.take  # pyrefly: ignore[missing-attribute]
    put = _funcs.put  # pyrefly: ignore[missing-attribute]

    def __dlpack__(self, *, stream: int | None = None) -> CapsuleType:
        return self.tensor.__dlpack__(stream=stream)

    def __dlpack_device__(self) -> tuple[int, int]:
        return self.tensor.__dlpack_device__()


def _tolist(obj: Iterable[object]) -> list[object]:
    """Recursively convert tensors into lists."""
    a1: list[object] = []
    for elem in obj:
        if isinstance(elem, (list, tuple)):
            elem = _tolist(elem)
        if isinstance(elem, ndarray):
            a1.append(elem.tensor.tolist())
        else:
            a1.append(elem)
    return a1


# This is the ideally the only place which talks to ndarray directly.
# The rest goes through asarray (preferred) or array.


def array(
    obj: object,
    dtype: object = None,
    *,
    copy: bool = True,
    order: str = "K",
    subok: bool = False,
    ndmin: int = 0,
    like: object = None,
) -> ndarray:
    if subok is not False:
        raise NotImplementedError("'subok' parameter is not supported.")
    if like is not None:
        raise NotImplementedError("'like' parameter is not supported.")
    if order != "K":
        raise NotImplementedError

    # a happy path
    if (
        isinstance(obj, ndarray)
        and copy is False
        and dtype is None
        and ndmin <= obj.ndim
    ):
        return obj

    if isinstance(obj, (list, tuple)):
        # FIXME and they have the same dtype, device, etc
        if obj and all(isinstance(x, torch.Tensor) for x in obj):
            # list of arrays: *under torch.Dynamo* these are FakeTensors
            obj = torch.stack(obj)
        else:
            # XXX: remove tolist
            # lists of ndarrays: [1, [2, 3], ndarray(4)] convert to lists of lists
            obj = _tolist(obj)

    # is obj an ndarray already?
    if isinstance(obj, ndarray):
        obj = obj.tensor

    # is a specific dtype requested?
    torch_dtype = None
    if dtype is not None:
        torch_dtype = _dtypes.dtype(dtype).torch_dtype

    tensor = _util._coerce_to_tensor(obj, torch_dtype, copy, ndmin)
    return ndarray(tensor)


def asarray(
    a: object, dtype: object = None, order: str = "K", *, like: object = None
) -> ndarray:
    return array(a, dtype=dtype, order=order, like=like, copy=False, ndmin=0)


def ascontiguousarray(
    a: object, dtype: object = None, *, like: object = None
) -> ndarray:
    arr = asarray(a, dtype=dtype, like=like)
    if not arr.tensor.is_contiguous():
        arr.tensor = arr.tensor.contiguous()
    return arr


def from_dlpack(x: CapsuleType | _SupportsDLPack, /) -> ndarray:
    t = torch.from_dlpack(x)
    return ndarray(t)


def _extract_dtype(entry: object) -> _dtypes.DType:
    try:
        dty = _dtypes.dtype(entry)
    except Exception:
        dty = asarray(entry).dtype
    return dty


def can_cast(from_: object, to: object, casting: str = "safe") -> bool:
    from_ = _extract_dtype(from_)
    to_ = _extract_dtype(to)

    return _dtypes_impl.can_cast_impl(from_.torch_dtype, to_.torch_dtype, casting)


def result_type(*arrays_and_dtypes: object) -> _dtypes.DType:
    tensors: list[torch.Tensor] = []
    for entry in arrays_and_dtypes:
        try:
            t = asarray(entry).tensor
        except (RuntimeError, ValueError, TypeError):
            dty = _dtypes.dtype(entry)
            t = torch.empty(1, dtype=dty.torch_dtype)
        tensors.append(t)

    torch_dtype = _dtypes_impl.result_type_impl(*tensors)
    return _dtypes.dtype(torch_dtype)
