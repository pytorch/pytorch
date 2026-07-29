"""Explicit guard-dispatch API.

Dynamo turns a function into a table of ``(guards, compiled_code)`` entries and,
on each call, runs the first entry whose guards pass, otherwise it traces a new
one. That dispatch loop runs in C++ (``torch/csrc/dynamo``: ``extra_state.cpp``
walks the cache and ``RootGuardManager::check_nopybind`` evaluates guards),
driven by the PEP 523 frame hook. This module gives that model an explicit
Python surface: ``Guards`` pairs accumulated guards with their compiled matcher,
and ``FuncDispatch`` documents the dispatch loop.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Callable

    from torch._guards import GuardsSet
    from torch._guards.manager import GuardManagerWrapper


class Guards:
    """Accumulated guards paired with their compiled runtime matcher.

    This is the guard half of a dispatch-table entry: ``guards_set`` holds the
    ``Guard`` objects accumulated while tracing, and ``guard_manager`` is the
    compiled ``GuardManagerWrapper`` (wrapping a C++ ``RootGuardManager``) that
    checks them at runtime.
    """

    def __init__(
        self,
        guards_set: GuardsSet | None = None,
        guard_manager: GuardManagerWrapper | None = None,
    ) -> None:
        self.guards_set = guards_set
        self.guard_manager = guard_manager

    def matches(self, f_locals: dict[str, object]) -> bool:
        """Return whether the guards pass for the given frame locals.

        Delegates to the compiled matcher. Note this is the debug-grade Python
        check; ``torch.compile``'s production hot path evaluates guards in C++
        (``RootGuardManager.check_nopybind``) directly over the frame's
        ``localsplus``.
        """
        if self.guard_manager is None:
            return False
        return self.guard_manager.check(f_locals)


class FuncDispatch:
    """Reference model of Dynamo's function dispatch.

    A compiled function is a table of ``(guards, compiled_fn)`` entries; each call
    runs the first entry whose guards pass, otherwise it traces a new entry. The
    production dispatch loop lives in C++ (``torch/csrc/dynamo``) and is driven by
    the PEP 523 frame hook; this class expresses the same model in Python and is
    NOT wired into ``torch.compile``.

    The trace-on-miss step needs ``torch._dynamo.run_and_trace`` (the seam that
    traces a function and returns its compiled form plus a populated ``Guards``),
    which is not yet provided; until then ``__call__`` raises on a cache miss.
    Populate ``dispatch_table`` directly to exercise the match loop.
    """

    # Set by functools.update_wrapper in __init__.
    __wrapped__: Callable[..., object]

    def __init__(self, fn: Callable[..., object]) -> None:
        functools.update_wrapper(self, fn)
        self.dispatch_table: list[tuple[Guards, Callable[..., object]]] = []

    def __call__(self, *args: object, **kwargs: object) -> object:
        f_locals = self._bind_f_locals(args, kwargs)
        for guards, fn in self.dispatch_table:
            if guards.matches(f_locals):
                return fn(*args, **kwargs)
        raise NotImplementedError(
            "FuncDispatch trace-on-miss is not wired: torch._dynamo.run_and_trace "
            "is not yet provided. FuncDispatch documents Dynamo's dispatch model; "
            "the production dispatcher is the C++ guard cache."
        )

    def _bind_f_locals(
        self, args: tuple[object, ...], kwargs: dict[str, object]
    ) -> dict[str, object]:
        import inspect

        bound = inspect.signature(self.__wrapped__).bind(*args, **kwargs)
        bound.apply_defaults()
        return dict(bound.arguments)
