"""Process-wide compiler exclusion for precompiled workers."""

from __future__ import annotations

import contextlib
import os
import threading
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Iterator


_ENV = "TORCH_PRECOMPILE_NO_COMPILATION"
_lock = threading.Lock()
_depth = 0
_previous_env: str | None = None


def is_compilation_forbidden() -> bool:
    return _depth > 0 or os.environ.get(_ENV) == "1"


@contextlib.contextmanager
def no_compilation() -> Iterator[None]:
    """Forbid compiler work on every thread of this process.

    This is a worker-lifetime policy, unlike precompile.serving(), which is
    deliberately thread-local. Overlapping owners keep the policy active until
    the last owner exits; callers must drain dispatched work before that exit.
    While active, TORCH_PRECOMPILE_NO_COMPILATION=1 is exported so child
    processes started under the policy inherit it; setting exactly "1" before
    startup enables the policy for the whole process. The environment is only
    written when the outermost owner enters or exits.
    """
    global _depth, _previous_env
    with _lock:
        if _depth == 0:
            _previous_env = os.environ.get(_ENV)
            os.environ[_ENV] = "1"
        _depth += 1
    try:
        yield
    finally:
        with _lock:
            _depth -= 1
            if _depth == 0:
                if _previous_env is None:
                    os.environ.pop(_ENV, None)
                else:
                    os.environ[_ENV] = _previous_env
                _previous_env = None


def check_compilation_allowed(operation: str, detail: str | None = None) -> None:
    if is_compilation_forbidden():
        from torch._dynamo.utils import counters
        from torch._precompile import PrecompileError

        counters["precompile"]["forbidden_compilation"] += 1
        counters["precompile"][operation] += 1
        raise PrecompileError(
            f"precompile.no_compilation() forbids {operation}; the supplied "
            "artifact or its kernel/autotune cache does not cover this execution."
            + (f"\n{detail}" if detail else "")
        )
