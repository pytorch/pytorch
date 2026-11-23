"""Wrapper to mimic (parts of) np.random API surface.

NumPy has strict guarantees on reproducibility etc; here we don't give any.

Q: default dtype is float64 in numpy

"""

from __future__ import annotations

import functools
from math import sqrt
from typing import ParamSpec, TYPE_CHECKING, TypeVar

import torch

from . import _dtypes_impl
from ._normalizations import array_or_scalar, ArrayLike, normalizer


if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import TypeAlias

    from ._ndarray import ndarray

    # array_or_scalar returns either a python scalar or a wrapped ndarray.
    _ScalarOrArray: TypeAlias = int | float | complex | bool | ndarray


_P = ParamSpec("_P")
_R = TypeVar("_R")


__all__ = [
    "seed",
    "random_sample",
    "sample",
    "random",
    "rand",
    "randn",
    "normal",
    "choice",
    "randint",
    "shuffle",
    "uniform",
]


def use_numpy_random() -> bool:
    # local import to avoid ref cycles
    import torch._dynamo.config as config

    return config.use_numpy_random_stream


def deco_stream(func: Callable[_P, _R]) -> Callable[_P, _R]:
    @functools.wraps(func)
    def inner(*args: _P.args, **kwds: _P.kwargs) -> _R:
        if not use_numpy_random():
            return func(*args, **kwds)
        else:
            import numpy

            from ._ndarray import ndarray

            f = getattr(numpy.random, func.__name__)

            # numpy funcs accept numpy ndarrays, unwrap
            np_args = tuple(
                arg.tensor.numpy() if isinstance(arg, ndarray) else arg for arg in args
            )
            np_kwds = {
                key: val.tensor.numpy() if isinstance(val, ndarray) else val
                for key, val in kwds.items()
            }

            value = f(*np_args, **np_kwds)

            # `value` can be either numpy.ndarray or python scalar (or None)
            if isinstance(value, numpy.ndarray):
                value = ndarray(torch.as_tensor(value))

            return value  # pyrefly: ignore[bad-return]

    return inner


@deco_stream
def seed(seed: int | None = None) -> None:
    if seed is not None:
        torch.random.manual_seed(seed)


@deco_stream
def random_sample(size: int | tuple[int, ...] | None = None) -> _ScalarOrArray:
    if size is None:
        size = ()
    dtype = _dtypes_impl.default_dtypes().float_dtype
    values = torch.empty(size, dtype=dtype).uniform_()
    return array_or_scalar(values, return_scalar=size == ())


def rand(*size: int) -> _ScalarOrArray:
    arg: tuple[int, ...] | None = size
    if arg == ():
        arg = None
    return random_sample(arg)


sample = random_sample
random = random_sample


@deco_stream
def uniform(
    low: float = 0.0, high: float = 1.0, size: int | tuple[int, ...] | None = None
) -> _ScalarOrArray:
    if size is None:
        size = ()
    dtype = _dtypes_impl.default_dtypes().float_dtype
    values = torch.empty(size, dtype=dtype).uniform_(low, high)
    return array_or_scalar(values, return_scalar=size == ())


@deco_stream
def randn(*size: int) -> _ScalarOrArray:
    dtype = _dtypes_impl.default_dtypes().float_dtype
    values = torch.randn(size, dtype=dtype)
    return array_or_scalar(values, return_scalar=size == ())


@deco_stream
def normal(
    loc: float = 0.0, scale: float = 1.0, size: int | tuple[int, ...] | None = None
) -> _ScalarOrArray:
    if size is None:
        size = ()
    dtype = _dtypes_impl.default_dtypes().float_dtype
    values = torch.empty(size, dtype=dtype).normal_(loc, scale)
    return array_or_scalar(values, return_scalar=size == ())


@deco_stream
def shuffle(x: torch.Tensor | ndarray) -> None:
    # no @normalizer because we do not cast e.g. lists to tensors
    from ._ndarray import ndarray

    if isinstance(x, torch.Tensor):
        tensor = x
    elif isinstance(x, ndarray):
        tensor = x.tensor
    else:
        raise NotImplementedError("We do not random.shuffle lists in-place")

    perm = torch.randperm(tensor.shape[0])
    xp = tensor[perm]
    tensor.copy_(xp)


@deco_stream
def randint(
    low: int, high: int | None = None, size: int | tuple[int, ...] | None = None
) -> _ScalarOrArray:
    if size is None:
        size = ()
    if not isinstance(size, (tuple, list)):
        size = (size,)
    if high is None:
        low, high = 0, low
    values = torch.randint(low, high, size=size)
    return array_or_scalar(values, int, return_scalar=size == ())


@deco_stream
@normalizer
def choice(
    a: ArrayLike,
    size: int | tuple[int, ...] | None = None,
    replace: bool = True,
    p: ArrayLike | None = None,
) -> torch.Tensor:
    # https://stackoverflow.com/questions/59461811/random-choice-with-pytorch
    if a.numel() == 1:
        # torch.arange accepts a 1-element tensor as `end`; the stub does not.
        a = torch.arange(a)  # pyrefly: ignore[no-matching-overload]

    # TODO: check a.dtype is integer -- cf np.random.choice(3.4) which raises

    # number of draws
    if size is None:
        num_el = 1
    elif isinstance(size, int):
        num_el = size
    else:
        num_el = 1
        for el in size:
            num_el *= el

    # prepare the probabilities
    if p is None:
        p = torch.ones_like(a) / a.shape[0]

    # cf https://github.com/numpy/numpy/blob/main/numpy/random/mtrand.pyx#L973
    atol = sqrt(torch.finfo(p.dtype).eps)
    if abs(p.sum() - 1.0) > atol:
        raise ValueError("probabilities do not sum to 1.")

    # actually sample
    indices = torch.multinomial(p, num_el, replacement=replace)

    if isinstance(size, (tuple, list)):
        indices = indices.reshape(size)

    samples = a[indices]

    return samples
