"""Public ``torch.compiler.precompile`` surface.

Prototype API: capture ``fn`` ahead of time and lower it to a self-contained
Python source artifact, then reload it in a fresh process. See :func:`capture` and
:func:`load`, and Note [precompile programming model] in
``torch/_precompile.py`` for the contract. Signatures, error types and the
artifact format may change between releases without a deprecation cycle.

Distinct from ``torch._dynamo.config.caching_precompile`` (a ``torch.compile``
guard-serialization caching mode), despite the shared word.
"""

# ruff: noqa: PLC0414  # the `X as X` re-exports below are deliberate
from torch._precompile import (
    Capture,
    capture,
    load,
    MakeFxTracer,
    PrecompiledRunnable as PrecompiledRunnable,
    PrecompileError as PrecompileError,
)


# MakeFxTracer is defined in a private module (for import-layering reasons, and because
# dataclass decoration resolves annotations against the defining module). Declare this
# module its home so introspection (test_public_bindings, Sphinx) resolves it under
# torch.compiler.precompile, where it is re-exported; that rewrites its __module__
# process-globally, and the public spelling is the only one either module documents.
MakeFxTracer.__module__ = "torch.compiler.precompile"


# PrecompileError and the loaded-artifact handle are intentionally NOT in __all__:
# their home is torch.compiler (torch.compiler.PrecompileError, for the conventional
# ``except`` spelling; the handle for an ``isinstance`` check), so their ``__module__``
# is "torch.compiler". They are re-exported here only so the
# ``torch.compiler.precompile.<name>`` spelling also resolves, each an explicit ``as``
# alias so a type checker under no_implicit_reexport sees the re-export.
__all__ = [
    "capture",
    "load",
    "Capture",
    "MakeFxTracer",
]
