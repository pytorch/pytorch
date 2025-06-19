from __future__ import annotations

import atexit
import functools
import inspect
import logging
import sys
from typing import Callable, Optional, TypeVar, Union, TYPE_CHECKING

import torch
from torch._prims_common import IntLike
from torch.fx.experimental.sym_node import SymTypes

if TYPE_CHECKING:
    from torch.types import IntLikeType

__all__ = [
    "log_lru_cache_stats",
    "lru_cache", 
    "uninteresting_files",
    "has_hint",
    "_nested_int_aware_sort",
]

_T = TypeVar("_T")

log = logging.getLogger(__name__)

Scalar = Union[torch.SymInt, torch.SymFloat, torch.SymBool, int, float, bool]


def log_lru_cache_stats(wrapped_f: functools._lru_cache_wrapper[object]) -> None:
    log.debug(
        "lru_cache_stats %s: %s", wrapped_f.__name__, wrapped_f.cumulative_cache_info()  # type: ignore[attr-defined]
    )


# Wrapper on lru_cache that reports statistics at process end
def lru_cache(
    maxsize: Optional[int],
) -> Callable[[Callable[..., _T]], functools._lru_cache_wrapper[_T]]:
    def inner(f: Callable[..., _T]) -> functools._lru_cache_wrapper[_T]:
        wrapped_f = functools.lru_cache(maxsize)(f)
        old_cache_clear = wrapped_f.cache_clear
        prev_hits = 0
        prev_misses = 0

        # TODO: There's a ref-cycle here (wrapped_f -> cumulative_cache_info
        # -> wrapped_f) but cannot be solved with weakref as wrapped_f is not
        # weakref'able on some versions of Python

        def cumulative_cache_info() -> functools._CacheInfo:
            cur = wrapped_f.cache_info()
            return functools._CacheInfo(
                prev_hits + cur.hits,
                prev_misses + cur.misses,
                cur.maxsize,
                cur.currsize,
            )

        def new_cache_clear() -> None:
            nonlocal prev_hits, prev_misses
            cur = wrapped_f.cache_info()
            prev_hits += cur.hits
            prev_misses += cur.misses
            old_cache_clear()

        wrapped_f.cache_clear = new_cache_clear  # type: ignore[attr-defined, method-assign]
        wrapped_f.cumulative_cache_info = cumulative_cache_info  # type: ignore[attr-defined, method-assign]
        if log.isEnabledFor(logging.DEBUG):
            atexit.register(log_lru_cache_stats, wrapped_f)  # type: ignore[arg-type]
        return wrapped_f

    return inner


# These are modules that contain generic code for interacting with ShapeEnv
# which are unlikely to identify a particular interesting guard statement
@lru_cache(None)
def uninteresting_files() -> set[str]:
    import torch._compile
    import torch._dynamo.eval_frame
    import torch._inductor.sizevars
    import torch._library.custom_ops
    import torch._library.fake_impl
    import torch._logging
    import torch._subclasses.fake_tensor
    import torch._subclasses.meta_utils
    import torch.fx.experimental.recording
    import torch.fx.experimental.sym_node

    mods = [
        sys.modules[__name__],
        torch.fx.experimental.recording,
        torch.fx.experimental.sym_node,
        torch.fx.interpreter,
        torch,
        torch._compile,
        torch._dynamo.eval_frame,
        torch._inductor.sizevars,
        torch._library.custom_ops,
        torch._library.fake_impl,
        torch._subclasses.meta_utils,
        torch._subclasses.fake_tensor,
        torch._logging._internal,
        torch._logging.structured,
    ]
    import torch._dynamo.guards

    return (
        {inspect.getfile(m) for m in mods}
        | torch._dynamo.guards.uninteresting_files()
        | {"<string>"}
    )


def has_hint(a: Scalar) -> bool:
    if isinstance(a, SymTypes):
        return a.node.has_hint()
    return True


def _nested_int_aware_sort(
    tup: tuple[IntLikeType, int],  # type: ignore[name-defined]
) -> tuple[int, IntLikeType, int]:  # type: ignore[name-defined]
    from torch.fx.experimental.symbolic_shapes import is_nested_int
    return (
        # Order nested ints by their coefficients.
        # 1 here to order nested ints after non-nested-ints.
        (1, tup[0].node.nested_int_coeff(), tup[1])
        if is_nested_int(tup[0])
        else (0, *tup)
    )