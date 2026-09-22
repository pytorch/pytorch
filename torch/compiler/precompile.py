"""Public ``torch.compiler.precompile`` surface.

Prototype API: capture ``fn`` ahead of time from the caller's own calls and lower
it to a self-contained Python source artifact plus an acceleration cache, then
reload it in a fresh process with :func:`load`. This module exports the types a
capture takes and returns: the ``MakeFxTracer`` configuration, the ``Capture``
handle, the ``PrecompiledRunnable`` a load returns and the ``PrecompileSummary``
report. See Note [precompile programming model]
in ``torch/_precompile.py`` for the contract. Signatures, error types and the
artifact format may change between releases without a deprecation cycle.

Distinct from ``torch._dynamo.config.caching_precompile`` (a ``torch.compile``
guard-serialization caching mode), despite the shared word.
"""

import typing

from torch._precompile import (
    Capture,
    load,
    MakeFxTracer,
    PrecompiledRunnable,
    PrecompileError,  # noqa: F401
)
from torch.compiler._precompile_types import PrecompileSummary


# These types are defined in torch._precompile / torch.compiler._precompile_types
# (import layering; and @dataclass dereferences sys.modules[cls.__module__] while
# decorating, so a class body cannot name a module that is still being imported).
# Declare this module their home so introspection (pickle, test_public_bindings,
# Sphinx) resolves them under torch.compiler.precompile, where they are re-exported.
# load is re-homed where it is defined: a function's annotations resolve through
# its own globals, not its __module__.
for _t in (Capture, MakeFxTracer, PrecompiledRunnable, PrecompileSummary):
    # torch._precompile uses ``from __future__ import annotations``, and
    # typing.get_type_hints resolves a class's string annotations through its
    # __module__. MakeFxTracer's only annotation today (``dict | None``) would resolve
    # from builtins anywhere; resolving against the DEFINING module before the
    # re-homing keeps that true for any annotation added later (a ``Callable`` field
    # would otherwise fail to resolve in this module's namespace).
    _t.__annotations__ = typing.get_type_hints(_t)
    _t.__module__ = "torch.compiler.precompile"
del _t
del typing  # not part of the public surface

# PrecompileError is intentionally NOT in __all__: its home is torch.compiler
# (torch.compiler.PrecompileError, for the conventional ``except`` spelling), so
# its __module__ is "torch.compiler". It is re-exported here only so
# ``torch.compiler.precompile.PrecompileError`` also resolves.
__all__ = [
    "load",
    "Capture",
    "MakeFxTracer",
    "PrecompiledRunnable",
    "PrecompileSummary",
]
