"""Process-wide compiler exclusion for precompiled training workers."""

from __future__ import annotations

import contextlib
import os
import threading
from collections.abc import Iterator


_ENV = "TORCH_PRECOMPILE_NO_COMPILATION"
_POLICY_ENVS = (_ENV,)
_lock = threading.Lock()
_depth = 0
_previous_env: dict[str, str | None] = {}


def is_compilation_forbidden() -> bool:
    return _depth > 0 or os.environ.get(_ENV) == "1"


@contextlib.contextmanager
def no_compilation() -> Iterator[None]:
    """Forbid compiler work on every thread and inherit the policy in workers.

    This is a worker-lifetime policy, unlike precompile.serving(), which is
    deliberately thread-local. Overlapping owners keep the policy active until
    the last owner exits; callers must drain dispatched work before that exit.
    """
    global _depth, _previous_env
    with _lock:
        if _depth == 0:
            _previous_env = {name: os.environ.get(name) for name in _POLICY_ENVS}
            for name in _POLICY_ENVS:
                os.environ[name] = "1"
        _depth += 1
    try:
        yield
    finally:
        with _lock:
            _depth -= 1
            if _depth == 0:
                for name, prior in _previous_env.items():
                    if prior is None:
                        os.environ.pop(name, None)
                    else:
                        os.environ[name] = prior
                _previous_env = {}


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
