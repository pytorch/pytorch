from collections.abc import Callable, Generator
from contextlib import contextmanager
from typing import Any


_UNSAFE_EXPORT_CALLABLES: set[Callable[..., Any]] = set()


def _is_unsafe_callable_registered(fn: Callable[..., Any]) -> bool:
    return fn in _UNSAFE_EXPORT_CALLABLES


@contextmanager
def unsafe_allow_callable_serialization(
    *callables: Callable[..., Any],
) -> Generator[None, None, None]:
    """Context manager that temporarily registers callables for export serialization.

    Registers the given callables on entry and removes them on exit. Callables
    that were already registered before entering the context are preserved after
    exit.

    Args:
        *callables: One or more callable objects to temporarily allow for serialization.

    Example::

        from torch._export.serde import unsafe_allow_callable_serialization
        from torch._functorch.predispatch import _jvp_increment_nesting

        with unsafe_allow_callable_serialization(_jvp_increment_nesting):
            torch.export.save(ep, buffer)
    """
    already_registered = {c for c in callables if c in _UNSAFE_EXPORT_CALLABLES}
    _UNSAFE_EXPORT_CALLABLES.update(callables)
    try:
        yield
    finally:
        newly_added = set(callables) - already_registered
        _UNSAFE_EXPORT_CALLABLES.difference_update(newly_added)
