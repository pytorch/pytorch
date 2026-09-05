"""Selection among an override's named implementations.

An override module may hold more than one implementation of the same op -- a
DSL kernel and the op's own eager body, say, or two kernels with different
tradeoffs. `cond` stays eligibility-only, so which one runs is a separate
question from whether the override applies at all, and this module is where
that question is answered.

The active variant is per qualified op (`"ns::op"`), so two ops overridden by
the same module are selected independently. Names are defined by the override
module in a table it also registers from, so declaring a variant and being
selectable are the same act; `PASSTHROUGH` is the one reserved name, meaning
"delegate to the op's own implementation", which keeps the override installed
and its `cond` in play while the kernels stand aside.

`TORCH_NATIVE_OVERRIDE_VARIANTS="ns::op=name;ns2::op2=name2"` seeds the
selection at import. It is an initializer, not a control surface: benchmark
harnesses run one process per configuration, and reading it once keeps a
mid-run `set_variant` from being silently overridden.
"""

from __future__ import annotations

import contextlib
import os
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Iterator


__all__ = ["PASSTHROUGH", "get_variant", "set_variant", "variant"]

PASSTHROUGH = "passthrough"

_ENV_VAR = "TORCH_NATIVE_OVERRIDE_VARIANTS"


def _parse_env(raw: str | None) -> dict[str, str]:
    """`"ns::op=name;ns2::op2=other"` -> mapping. Malformed entries are
    dropped rather than raised on: this runs during `import torch`, where a
    typo in an environment variable must not make torch unimportable."""
    selection: dict[str, str] = {}
    for entry in (raw or "").split(";"):
        entry = entry.strip()
        if not entry:
            continue
        op, sep, name = entry.partition("=")
        if sep and "::" in op and name.strip():
            selection[op.strip()] = name.strip()
    return selection


_active: dict[str, str] = _parse_env(os.environ.get(_ENV_VAR))


def get_variant(qualified_op: str, default: str) -> str:
    """The variant selected for `qualified_op`, or `default`."""
    return _active.get(qualified_op, default)


def set_variant(qualified_op: str, name: str | None) -> None:
    """Select `name` for `qualified_op`; `None` restores the module default.

    The name is not validated here -- the override module owns its table, and
    an unknown name surfaces there, where the valid ones can be named in the
    error.
    """
    if name is None:
        _active.pop(qualified_op, None)
    else:
        _active[qualified_op] = name


@contextlib.contextmanager
def variant(qualified_op: str, name: str | None) -> Iterator[None]:
    """Scope a selection to a block, restoring what was set before."""
    had = qualified_op in _active
    previous = _active.get(qualified_op)
    set_variant(qualified_op, name)
    try:
        yield
    finally:
        set_variant(qualified_op, previous if had else None)
