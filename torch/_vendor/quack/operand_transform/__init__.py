# Copyright (c) 2026, Tri Dao.
"""Per-operand transforms for GEMM mainloops (RS A-operand produces today;
TransformB smem rewrites later). Ported from the transformA branch onto the
interleaved RS mainloop's copy_block seam.

Transforms are a first-class axis of the generic GEMM host layer: pass any
handle (a W4 format name, a DecodeFormat, or an ``@a_transform`` mod) as
``transform_a=`` to :func:`quack.gemm_runtime.host.build_gemm_epi_plan` — or
compose with fn epilogues via ``EpiMod.__call__/plan/gemm(transform_a=...)``
— and it
rides the same jit/disk cache, async compile, and EpiOp argument machinery as
every epilogue variant (see quack.operand_transform.host for the picklable
reference + the W4 blob/strip geometry). ``quack.gemm_w4.gemm_w4a16`` is
thin sugar over that path.

Layering note: the fn frontend (a_transform / w4_transform) sits ABOVE the
host layer (it imports quack.gemm_runtime.identity for the semantic keyer
and transform-ref machinery), while transform.py and kinds.py are
KERNEL-SIDE code imported BY GemmSm90. The frontend re-exports below are
lazy (PEP 562) so the kernel can import this package without a cycle.
"""

from torch._vendor.quack.operand_transform.transform import (  # noqa: F401
    AuxKTileStrip,
    AuxOperandA,
    TransformA,
    TransformADropout,
    TransformAOperand,
    TransformAValue,
    TransformAW4,
)

_FRONTEND = ("ATransformMod", "PackedInput", "a_transform", "dropout_a", "w4_transform")
_HOST = ("transform_a_operand",)

__all__ = [
    "AuxKTileStrip",
    "AuxOperandA",
    "TransformA",
    "TransformADropout",
    "TransformAOperand",
    "TransformAValue",
    "TransformAW4",
    *_FRONTEND,
    *_HOST,
]


def __getattr__(name):
    if name in _FRONTEND:
        from torch._vendor.quack.operand_transform import frontend

        return getattr(frontend, name)
    if name in _HOST:
        from torch._vendor.quack.operand_transform import host

        return getattr(host, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
