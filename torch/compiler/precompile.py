"""Public ``torch.compiler.precompile`` surface.

Prototype API: capture ``fn`` ahead of time from the caller's own calls and lower
it to a self-contained Python source artifact plus an acceleration cache, then
reload it in a fresh process. This module exports the types a capture takes and
returns: the ``MakeFxTracer`` configuration, the ``Capture`` handle, the
``PrecompiledRunnable`` a load returns, and the ``PrecompileSummary`` report; the
entry points that produce and consume them follow in later commits. See Note
[precompile programming model] in ``torch/_precompile.py`` for the contract.
Signatures, error types and the artifact format may change between releases
without a deprecation cycle.

Distinct from ``torch._dynamo.config.caching_precompile`` (a ``torch.compile``
guard-serialization caching mode), despite the shared word.
"""

import typing

from torch._precompile import (
    Capture,
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
# Their attributes are documented in the class docstring: Sphinx reads a bare
# attribute docstring from the source of sys.modules[cls.__module__], now this file.
for _t in (Capture, MakeFxTracer, PrecompiledRunnable, PrecompileSummary):
    # torch._precompile uses ``from __future__ import annotations``, and
    # typing.get_type_hints resolves a class's string annotations through its
    # __module__. Resolving against the DEFINING module before the re-homing keeps
    # every class-level annotation resolvable (Capture has none, so for it this is
    # the __module__ assignment alone). The strings a dataclass snapshotted at
    # decoration (``Field.type``, ``__init__.__annotations__``) are left as they
    # are: nothing here reads them, and ``__annotations__`` is what get_type_hints
    # and the public-members test consult.
    _t.__annotations__ = typing.get_type_hints(_t)
    _t.__module__ = "torch.compiler.precompile"
del _t
del typing  # not part of the public surface

# PrecompileError is intentionally NOT in __all__: its home is torch.compiler
# (torch.compiler.PrecompileError, for the conventional ``except`` spelling), so
# its __module__ is "torch.compiler". It is re-exported here only so
# ``torch.compiler.precompile.PrecompileError`` also resolves.
__all__ = ["Capture", "MakeFxTracer", "PrecompiledRunnable", "PrecompileSummary"]
