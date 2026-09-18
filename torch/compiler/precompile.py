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

# ruff: noqa: PLC0414  # the `X as X` re-exports below are deliberate
from torch._precompile import (
    Capture,
    capture,
    DynamoTracer,
    load,
    MakeFxTracer,
    PrecompiledCallable as PrecompiledCallable,
    PrecompiledRunnable as PrecompiledRunnable,
    PrecompileError as PrecompileError,
)
from torch.compiler._precompile_types import (
    FrameInvariants,
    GuardFact,
    PrecompileSummary,
)


# The two tracers come from torch._precompile, which is under `from __future__ import
# annotations`, so their annotations are still strings that only resolve against THAT
# module's globals: resolve them BEFORE the re-homing below points
# typing.get_type_hints (which resolves through a class's __module__) at this module
# instead. The three _precompile_types classes need nothing -- that module has no
# future-annotations import, so their annotations are already objects. Only
# __annotations__ is rewritten here; dataclasses.fields() keeps the original strings,
# since it reads __dataclass_fields__, frozen at decoration.
for _t in (MakeFxTracer, DynamoTracer):
    _t.__annotations__ = typing.get_type_hints(_t)


# These types are defined in private modules (for import-layering reasons, and because
# dataclass decoration resolves annotations against the defining module). Declare this
# module their home so introspection (test_public_bindings, Sphinx) resolves them under
# torch.compiler.precompile, where they are re-exported; that rewrites their __module__
# process-globally, and the public spelling is the only one either module documents.
for _t in (
    MakeFxTracer,
    DynamoTracer,
    PrecompileSummary,
    FrameInvariants,
    GuardFact,
):
    _t.__module__ = "torch.compiler.precompile"


del _t
del typing  # not part of the public surface


# PrecompileError and the two loaded-artifact handles are intentionally NOT in
# __all__: their home is torch.compiler (torch.compiler.PrecompileError, for the
# conventional ``except`` spelling; the handles for an ``isinstance`` check), so their
# __module__ is "torch.compiler". They are re-exported here only so the
# ``torch.compiler.precompile.<name>`` spelling also resolves, each an explicit ``as``
# alias so a type checker under no_implicit_reexport sees the re-export.
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
