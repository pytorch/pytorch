# Copyright (c) 2026, Han Guo, Tri Dao.
"""Fused GEMM epilogues.

Layer map (see AI/epilogue_transform_reorg.md):

* :mod:`quack.epilogue.ops` — the EpiOp vocabulary: per-tensor resource
  lifecycle, value ports, host schema, cache identity. Writing a new op here
  (or in a domain module) is the extension path.
* :mod:`quack.epilogue.mixin` — ComposableEpiMixin: composes ``_epi_ops``
  into the kernel epilogue hooks (hand-written mixins subclass this).
* :mod:`quack.epilogue.frontend` — ``@gemm_epilogue``: author an epilogue as
  a plain fn over the accumulator; EpiMod mints kernel classes and owns the
  torch-facing call surfaces. :mod:`quack.epilogue.visit` is its device half.
* :mod:`quack.epilogue.math` — the value vocabulary (Pair/F2/pexp...).
* :mod:`quack.epilogue.library` — the library of ready mods; domain content
  lives in sibling modules (rotary, scaled_exp, head_rmsnorm).

The frontend re-exports below are lazy (PEP 562): kernel classes import the
ops/mixin submodules directly without pulling the frontend (which imports
every per-arch kernel class for minting). math is a leaf and re-exports
eagerly.
"""

from torch._vendor.quack.epilogue.math import (  # noqa: F401
    F2,
    F16Lanes,
    Pair,
    pack,
    pexp,
    pexp2,
    unpack,
)

_FRONTEND = ("EpiMod", "EpiPlan", "StaticEpi", "epilogue_from_class", "gemm_epilogue")

__all__ = [*_FRONTEND, "F2", "F16Lanes", "Pair", "pack", "pexp", "pexp2", "unpack"]


def __getattr__(name):
    if name in _FRONTEND:
        from torch._vendor.quack.epilogue import frontend

        return getattr(frontend, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
