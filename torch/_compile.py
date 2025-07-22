"""
APIs related to torch.compile which lazily import torch._dynamo to avoid
circular dependencies.
"""

import functools
from typing import Callable, Literal, Optional, overload, TypeVar, Union
from typing_extensions import ParamSpec


_T = TypeVar("_T")
_P = ParamSpec("_P")


@overload
def _disable_dynamo(
    fn: Callable[_P, _T], recursive: bool = True
) -> Callable[_P, _T]: ...


@overload
def _disable_dynamo(
    fn: Literal[None] = None, recursive: bool = True
) -> Callable[[Callable[_P, _T]], Callable[_P, _T]]: ...


def _disable_dynamo(
    fn: Optional[Callable[_P, _T]] = None, recursive: bool = True
) -> Union[Callable[_P, _T], Callable[[Callable[_P, _T]], Callable[_P, _T]]]:
    """
    This API should be only used inside torch, external users should still use
    torch._dynamo.disable. The main goal of this API is to avoid circular
    imports issues that is common while using _dynamo.disable inside torch
    itself.

    This API avoids it by lazily importing torch._dynamo from the import time to
    the invocation of the decorated function.
    """
    if fn is not None:

        @functools.wraps(fn)
        def inner(*args: _P.args, **kwargs: _P.kwargs) -> _T:
            # cache this on the first invocation to avoid adding too much overhead.
            disable_fn = getattr(fn, "__dynamo_disable", None)
            if disable_fn is None:
                import torch._dynamo

                # We can safely turn off functools.wraps here because the inner
                # already wraps fn in the outer scope.
                disable_fn = torch._dynamo.disable(fn, recursive, wrapping=False)
                fn.__dynamo_disable = disable_fn  # type: ignore[attr-defined]

            return disable_fn(*args, **kwargs)

        return inner
    else:
        # decorator usage like @_disable_dynamo(recursive=False). The resulting
        # object expects the original decorated function as the arg.
        return functools.partial(_disable_dynamo, recursive=recursive)
