# mypy: allow-untyped-defs
"""Gluon (Triton's low-level frontend) template support for Inductor.

Gluon kernels are ``@gluon.jit`` functions that expose explicit tile layouts,
shared-memory descriptors and target intrinsics (CDNA4 ``mfma``, TMA/MMA), which
lets an expert hand-tune past what Triton's automatic codegen achieves.

``GluonJITFunction`` subclasses ``triton.runtime.jit.JITFunction`` and launches
identically to a Triton kernel, and ``tl.*`` pointwise ops are valid inside a
``@gluon.jit`` body (layouts are inferred from the operands). So a Gluon
template can reuse the entire Triton template path -- ``TritonTemplateKernel``
codegen (``def_kernel``/``modification``/``store_output``), ``TritonScheduling``
and ``async_compile.triton`` -- and only the emitted imports and the jit
decorator differ.
"""

import math
from functools import lru_cache

import torch
from torch._inductor.codegen.simd import constant_repr
from torch._inductor.codegen.triton import triton_compute_type, TritonKernelOverrides
from torch._inductor.select_algorithm import TritonTemplate, TritonTemplateKernel


# Imports every generated Gluon kernel needs, regardless of target: the frontend
# plus the target-neutral layout classes from ``gluon.language._layouts``.
#
# Anything under ``gluon.language.amd.*`` or ``gluon.language.nvidia.*`` is a
# target intrinsic and belongs in a target's ``target_imports`` instead -- the
# families do not share primitives (CDNA4 has mfma + buffer_load_to_shared,
# gfx1250 has wmma + tensor descriptors + mbarrier, NVIDIA has mma + TMA), so a
# kernel body is written per target and only the scaffolding is shared.
GLUON_BASE_IMPORTS = """
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language._layouts import (
    DistributedLinearLayout,
    DotOperandLayout,
    PaddedSharedLayout,
)
"""


class GluonKernelOverrides(TritonKernelOverrides):
    """Triton op lowerings adjusted for Gluon's explicit-layout tensors.

    Gluon tensor constructors (``gl.full``/``gl.zeros``) require an explicit
    layout, which generated subgraph code has no way to pick: the right layout
    depends on the tile the constant is combined with. Emit *typed scalars*
    instead -- they broadcast against a tile of any layout.

    This mirrors ``TritonOverrides._shaped_constant``, including its ``-0.0``
    bit-pattern round-trip, so a fix to either belongs in both. The divergence is
    forced rather than chosen: the base materializes scalars as ``tl.full([], v,
    t)``, and that does not compile inside a ``@gluon.jit`` body even though an
    empty shape needs no layout. ``tl.cast`` is what does. The base's note about
    ``tl.full`` being what stops subnormal fp32 from promoting to fp64 does not
    reproduce on Triton 3.8 -- ``tl.full``, ``tl.cast`` and a bare literal all
    give bit-identical fp32 results there -- so there is no known cost to it.
    """

    @classmethod
    def constant(cls, value, dtype):
        type_ = torch._prims_common.dtype_to_type(dtype)
        triton_type = triton_compute_type(dtype)

        if value == 0 and math.copysign(1.0, value) < 0:
            # -0.0 does not survive a scalar literal (-0.0 == 0 in Python), so
            # round-trip the IEEE 754 bit pattern.
            if triton_type == "tl.float32":
                return f"tl.cast(0x80000000, tl.uint32).to({triton_type}, bitcast=True)"
            if triton_type == "tl.float64":
                return (
                    f"tl.cast(0x8000000000000000, tl.uint64)"
                    f".to({triton_type}, bitcast=True)"
                )

        triton_val = constant_repr(type_(value))
        if value < 0 and not dtype.is_signed:
            signed_type = f"tl.{triton_type[4:]}"
            return f"tl.cast({triton_val}, {signed_type}).to({triton_type})"
        return f"tl.cast({triton_val}, {triton_type})"


class GluonTemplateKernel(TritonTemplateKernel):
    """TritonTemplateKernel that emits a ``@gluon.jit`` kernel.

    Target-neutral: subclass and set ``target_imports`` to the intrinsics a
    target's kernel bodies use. ``gen_common_triton_imports`` is cached per
    ``cls``, so each target subclass gets its own cache entry.

    Everything else (signature generation, subgraph ``modification()`` rendering
    for score_mod/mask_mod, ``store_output``, argument plumbing and the kernel
    call) is inherited unchanged.
    """

    overrides = GluonKernelOverrides  # type: ignore[assignment]

    # Set by target subclasses; see GLUON_BASE_IMPORTS.
    target_imports: str = ""

    @classmethod
    @lru_cache(None)
    def gen_common_triton_imports(cls) -> str:
        return (
            super().gen_common_triton_imports()
            + GLUON_BASE_IMPORTS
            + cls.target_imports
        )

    def jit_lines(self) -> str:
        # Keep @triton_heuristics.template(...) -- the autotuner drives a
        # GluonJITFunction the same way it drives a JITFunction -- and swap only
        # the frontend decorator.
        return super().jit_lines().replace("@triton.jit", "@gluon.jit")


class GluonTemplate(TritonTemplate):
    kernel_type = GluonTemplateKernel
