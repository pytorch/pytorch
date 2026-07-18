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

# (call_with_disable, is_exporting), resolved lazily on the first disabled call
# to avoid importing torch at module load (see module docstring).
_fast_disable_hooks: tuple[Callable[..., object], Callable[[], bool]] | None = None


@overload
def _disable_dynamo(
    fn: Callable[_P, _T], recursive: bool = True
) -> Callable[_P, _T]: ...


@overload
def _disable_dynamo(
    fn: None = None, recursive: bool = True
) -> Callable[[Callable[_P, _T]], Callable[_P, _T]]: ...


def _disable_dynamo(
    fn: Callable[_P, _T] | None = None, recursive: bool = True
) -> Callable[_P, _T] | Callable[[Callable[_P, _T]], Callable[_P, _T]]:
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
            # Fast path: a fully-recursive disable outside of export is just
            # "run fn with the eval-frame handler off", which call_with_disable
            # does entirely in C (forwarding args via vectorcall), avoiding the
            # DisableContext wrapper. Export needs the DisableContext wrapper for
            # its fx_traceback annotation, and recursive=False needs its
            # non-recursive skip semantics, so both fall through below. The C
            # entry point and is_exporting are resolved once and cached to keep
            # this path cheap without importing torch at module load time.
            global _fast_disable_hooks
            if _fast_disable_hooks is None:
                import torch

                _fast_disable_hooks = (
                    torch._C._dynamo.eval_frame.call_with_disable,
                    torch.compiler.is_exporting,
                )
            call_with_disable, is_exporting = _fast_disable_hooks
            if recursive and not is_exporting():
                return call_with_disable(fn, *args, **kwargs)

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
