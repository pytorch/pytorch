"""
APIs related to torch.compile which lazily import torch._dynamo to avoid
circular dependencies.
"""

import functools
from collections.abc import Callable
from typing import overload, TypeVar
from typing_extensions import ParamSpec


_T = TypeVar("_T")
_P = ParamSpec("_P")


@overload
def _disable_dynamo(
    fn: Callable[_P, _T],
    recursive: bool = True,
    *,
    library_fallback_reason: str | None = None,
) -> Callable[_P, _T]: ...


@overload
def _disable_dynamo(
    fn: None = None,
    recursive: bool = True,
    *,
    library_fallback_reason: str | None = None,
) -> Callable[[Callable[_P, _T]], Callable[_P, _T]]: ...


def _disable_dynamo(
    fn: Callable[_P, _T] | None = None,
    recursive: bool = True,
    *,
    library_fallback_reason: str | None = None,
) -> Callable[_P, _T] | Callable[[Callable[_P, _T]], Callable[_P, _T]]:
    """
    This API should be only used inside torch, external users should still use
    torch._dynamo.disable. The main goal of this API is to avoid circular
    imports issues that is common while using _dynamo.disable inside torch
    itself.

    This API avoids it by lazily importing torch._dynamo from the import time to
    the invocation of the decorated function.

    ``library_fallback_reason`` is for a function of PyTorch that is disabled on purpose, as
    a fallback for something that cannot be traced. It is a message for the user, and has to
    say what avoids the fallback: a call of the function in a region that torch._dynamo
    traces is reported as a graph break with that message, as a limitation of PyTorch that
    the user cannot remove, instead of as a ``torch.compiler.disable`` call in their code.
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

        if library_fallback_reason is not None:
            # What torch._dynamo.disable sets on the function it returns, which
            # torch._dynamo reads when the function is called in a traced region, without
            # importing torch._dynamo here.
            inner._torchdynamo_disable = True  # type: ignore[attr-defined]
            inner._torchdynamo_disable_msg = library_fallback_reason  # type: ignore[attr-defined]
            inner._torchdynamo_library_fallback = True  # type: ignore[attr-defined]
        return inner
    else:
        # decorator usage like @_disable_dynamo(recursive=False). The resulting
        # object expects the original decorated function as the arg.
        return functools.partial(
            _disable_dynamo,
            recursive=recursive,
            library_fallback_reason=library_fallback_reason,
        )


def _assume_constant_result(fn: Callable[_P, _T]) -> Callable[_P, _T]:
    """
    What torch.compiler.assume_constant_result does, for a function of torch that cannot
    import torch._dynamo: it sets the attribute that torch._dynamo reads when the function
    is called in a traced region, and it calls the function while tracing instead of
    putting the call into the graph.
    """
    fn._dynamo_marked_constant = True  # type: ignore[attr-defined]
    return fn
