from __future__ import annotations

from typing import Generic, TypeVar

import torch

__all__ = ['Await']

W = TypeVar("W")

class _PyAwaitMeta(type(torch._C._Await), type(Generic)):  # type: ignore[misc, no-redef]
    pass

class _Await(torch._C._Await, Generic[W], metaclass=_PyAwaitMeta):
    r"""
    Wrapper around a ``torch._C.Await`` which encapsulates delayed execution
    of a callable. All manipulations happen with functions ``torch.jit._awaitable``,
    ``torch.jit._awaitable_wait``, ``torch.jit._awaitable_nowait``.

    Torch scriptable manipulations:
    ``torch.jit._awaitable(func, *args)``
    Creates ``Await[W]`` object, where W is return type of func.

    Returns:
    ``torch.jit._awaitable_wait(Await[W])``
    Returns the result of the function, specified at ``_awaitable``,  with specified arguments.

    Returns:
        The result of type ``W`` of the function call. The result is owned by ``Await[W]``
        and returned on all following ``_awaitable_wait`` calls.


    ``torch.jit._awaitable_nowait(W)``
    Returns:
        Trivial ``Await[W]`` with specified result.


    Only in eager mode:
    ``fn() -> Callable[Tuple[Any], W]``
    Returns:
        Specified at ``_awaitable`` python function ``func``.

    ``args() -> Tuple[Any]``
    Returns:
        Specified at ``_awaitable`` python args.

    ``is_nowait() -> _bool``
    Returns:
        ``True`` if this object was created via ``_awaitable_nowait`` call (trivial `Await[W]`).

    In eager mode ``Await[W]`` can be used as ``W`` i.e. attributes of W can be called on ``Await[W]``,
    ``_awaitable_wait()`` call will be transparently added.
    """
