"""Public ``torch.compiler.precompile`` surface.

Prototype API: capture ``fn`` ahead of time and lower it to a self-contained
Python source artifact, then reload it in a fresh process. See :func:`capture` and
:func:`load`, and Note [precompile programming model] in
``torch/_precompile.py`` for the contract. Signatures, error types and the
artifact format may change between releases without a deprecation cycle.

Distinct from ``torch._dynamo.config.caching_precompile`` (a ``torch.compile``
guard-serialization caching mode), despite the shared word.
"""

import typing

from torch._precompile import (
    Capture,
    capture,
    DynamoTracer,
    load,
    MakeFxTracer,
    PrecompileError,  # noqa: F401
)
from torch.compiler._precompile_types import (
    FrameInvariants,
    GuardFact,
    PrecompileSummary,
)


# These types are defined in torch._precompile / a private module (for
# import-layering reasons, and because dataclass decoration resolves annotations
# against the defining module). Declare this module their home so introspection
# (test_public_bindings, Sphinx) resolves them under torch.compiler.precompile,
# where they are re-exported.
for _t in (
    MakeFxTracer,
    DynamoTracer,
    PrecompileSummary,
    FrameInvariants,
    GuardFact,
):
    # Resolve the string annotations against the DEFINING module's globals before
    # re-homing, so typing.get_type_hints (which resolves a class's annotations
    # through its __module__) does not later fail to find names like Callable in
    # this module's namespace.
    _t.__annotations__ = typing.get_type_hints(_t)
    _t.__module__ = "torch.compiler.precompile"
del _t
del typing  # not part of the public surface

# PrecompileError is intentionally NOT in __all__: its home is torch.compiler
# (torch.compiler.PrecompileError, for the conventional ``except`` spelling), so
# its __module__ is "torch.compiler". It is re-exported here only so
# ``torch.compiler.precompile.PrecompileError`` also resolves.
__all__ = [
    "capture",
    "load",
    "Capture",
    "MakeFxTracer",
    "DynamoTracer",
    "PrecompileSummary",
    "FrameInvariants",
    "GuardFact",
]
