# Copyright (c) 2026, Tri Dao.
"""Block-scaled operand container and format descriptors.

Design doc: AI/blockscaled_api.md. In short:

- ``BlockScaledFormat`` is a frozen descriptor (storage dtype, element bits, scale
  dtype, scale vector size, ...) and the single source of truth for format
  properties; nothing may re-derive them from tensor dtypes.
- ``BlockScaledOperand`` is a plain frozen dataclass holding ``qdata`` (quantized
  values), ``scale`` (128x4-blocked scale factors), the format, and (NVFP4 only)
  an optional per-tensor scale. It is deliberately NOT a torch.Tensor subclass:
  quack GEMM entry points are its only consumers, and the simplest container that
  keeps (qdata, scale, format, pts) atomic wins. It exposes an explicit, honest
  surface (``shape``/``dtype``/``mT``/``to``/``dequantize``) with no aten
  interception and no dequantize fallback - torch ops on it fail loudly with a
  TypeError.
- Transpose (``.mT`` / ``.T`` / ``transpose``) is a qdata stride-swap view;
  ``scale`` is carried unchanged - it blocks along K in either orientation. The
  blocked scale atom is not view-transposable.
- ``quant_dim`` records which logical dim the scale vector runs along (-1 or
  -2) - the analogue of CUTLASS's ``UMMA::Major`` on its SF atoms. Storage alone
  cannot express it: a square fp8 canonical operand and its ``.mT`` view are
  byte-identical. Packed formats (fp4, fp6) pin it to the packed (unit-stride)
  dim; 8-bit formats default to the last dim (quantize's convention); transpose
  flips it. Consumed as a GEMM operand, the quantized axis must be the
  contraction axis (the interface enforces this per operand slot).

This module must stay import-light (torch only at module level): it is imported
by ``quack.gemm_interface`` at module level and by the kernel-layer modules
lazily.
"""

import copy as _copy
from dataclasses import dataclass, replace
from typing import Optional, Tuple

import torch
import torch.utils._pytree as _pytree

from torch._vendor.quack.blockscaled.quantize import (
    QUANTIZERS,
    check_blocked_scale_atom,
    pack_scale_2d_to_blocked_contig,
)

__all__ = [
    "BlockScaledFormat",
    "BlockScaledOperand",
    "BLOCKSCALED_FORMAT_REGISTRY",
    "mma_kind_for_pair",
    "MXFP8_E4M3",
    "MXFP8_E5M2",
    "MXFP4",
    "MXFP4_BYTE",
    "NVFP4",
    "MXFP6_E2M3",
    "MXFP6_E3M2",
    "MXFP6_E2M3_PACKED",
    "MXFP6_E3M2_PACKED",
    "MXFP6_E2M3_BYTE",
    "MXFP6_E3M2_BYTE",
]


@dataclass(frozen=True)
class BlockScaledFormat:
    """Descriptor for a block-scaled quantization format.

    Frozen (hashable, picklable) so instances can serve as dynamo guard context and
    kernel-cache key material; only ``name`` crosses torch.library op schemas.
    """

    name: str
    qdata_dtype: torch.dtype
    # CuTe-DSL MMA element type (may differ from storage: fp6). None marks a format
    # with no DSL element type at all — host-side complete (quantize/dequantize/
    # serialize) but consumable by a kernel only once that kernel declares its own
    # (copy dtype, MMA dtype, convert) triple. See AI/blockscaled_api.md section 9.
    cutlass_dtype_name: Optional[str]
    elem_bits: int
    # Keep this field in its original positional slot. Besides preserving the
    # public constructor/pickle contract, it describes legacy container storage
    # exactly (fp4x2: 2, byte-container fp4/fp6: 1). Fractional dense packing is
    # selected explicitly by storage_layout below.
    elems_per_container: int
    scale_dtype: torch.dtype
    sf_vec_size: int  # logical K elements per scale factor (== min logical-K divisibility)
    has_per_tensor_scale: bool = False
    # None is intentionally the legacy default: old pickles do not carry this
    # field and must retain elems_per_container semantics. packed_lsb_v1 is a
    # dense little-endian bit stream (currently canonical packed fp6).
    storage_layout: Optional[str] = None

    def __setstate__(self, state):
        """Load both current and pre-storage-layout dataclass pickles.

        ``origin/main`` pickles contain ``elems_per_container`` and therefore
        retain container semantics when ``storage_layout`` is absent. The
        short-lived packed-FP6 feature schema contains neither field; derive
        its packing and rename unversioned FP6 to the explicit packed name.
        Materialize every field on the instance so generated dataclass helpers
        such as hash/repr/replace remain safe after load.
        """
        if not isinstance(state, dict):
            raise TypeError(f"invalid BlockScaledFormat pickle state: {type(state)}")
        values = dict(state)
        missing_epc = "elems_per_container" not in values
        missing_layout = "storage_layout" not in values
        if missing_epc:
            storage_bits = values["qdata_dtype"].itemsize * 8
            elem_bits = values["elem_bits"]
            if values.get("storage_layout") == "packed_lsb_v1" or (
                missing_layout and storage_bits % elem_bits
            ):
                values["elems_per_container"] = 1
                values.setdefault("storage_layout", "packed_lsb_v1")
            elif storage_bits % elem_bits == 0:
                values["elems_per_container"] = storage_bits // elem_bits
            else:
                raise ValueError("cannot infer elems_per_container from serialized format state")
        values.setdefault("has_per_tensor_scale", False)
        values.setdefault("storage_layout", None)
        if values["storage_layout"] == "packed_lsb_v1":
            values["name"] = {
                "mxfp6_e2m3": "mxfp6_e2m3_packed",
                "mxfp6_e3m2": "mxfp6_e3m2_packed",
            }.get(values["name"], values["name"])
        for name in (
            "name",
            "qdata_dtype",
            "cutlass_dtype_name",
            "elem_bits",
            "elems_per_container",
            "scale_dtype",
            "sf_vec_size",
            "has_per_tensor_scale",
            "storage_layout",
        ):
            if name not in values:
                raise ValueError(f"BlockScaledFormat pickle is missing {name!r}")
            object.__setattr__(self, name, values[name])
        self.__post_init__()

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name:
            raise TypeError("format name must be a non-empty str")
        if not isinstance(self.qdata_dtype, torch.dtype):
            raise TypeError(f"qdata_dtype must be a torch.dtype, got {self.qdata_dtype!r}")
        if self.cutlass_dtype_name is not None and not isinstance(self.cutlass_dtype_name, str):
            raise TypeError("cutlass_dtype_name must be a str or None")
        if (
            not isinstance(self.elem_bits, int)
            or isinstance(self.elem_bits, bool)
            or self.elem_bits <= 0
        ):
            raise TypeError(f"elem_bits must be a positive int, got {self.elem_bits!r}")
        if (
            not isinstance(self.elems_per_container, int)
            or isinstance(self.elems_per_container, bool)
            or self.elems_per_container <= 0
        ):
            raise TypeError(
                f"elems_per_container must be a positive int, got {self.elems_per_container!r}"
            )
        if not isinstance(self.scale_dtype, torch.dtype):
            raise TypeError(f"scale_dtype must be a torch.dtype, got {self.scale_dtype!r}")
        if (
            not isinstance(self.sf_vec_size, int)
            or isinstance(self.sf_vec_size, bool)
            or self.sf_vec_size <= 0
        ):
            raise TypeError(f"sf_vec_size must be a positive int, got {self.sf_vec_size!r}")
        if not isinstance(self.has_per_tensor_scale, bool):
            raise TypeError("has_per_tensor_scale must be bool")
        if self.storage_layout not in (None, "container_v1", "packed_lsb_v1"):
            raise ValueError(
                "storage_layout must be None, 'container_v1', or 'packed_lsb_v1', "
                f"got {self.storage_layout!r}"
            )
        if self.storage_layout != "packed_lsb_v1":
            used_bits = self.elem_bits * self.elems_per_container
            if used_bits > self.storage_elem_bits:
                raise ValueError(
                    f"{self.name}: {self.elems_per_container} x {self.elem_bits}-bit values "
                    f"do not fit in a {self.storage_elem_bits}-bit storage element"
                )

    def to_cutlass_dtype(self):
        """The CuTe-DSL element type for the MMA instruction (not the storage/copy
        dtype: packed MXFP6 stores raw uint8 bytes but computes as Float6*)."""
        if self.cutlass_dtype_name is None:
            raise ValueError(f"{self.name} has no CuTe-DSL element type")
        import cutlass  # lazy: keep this module importable without the DSL

        return getattr(cutlass, self.cutlass_dtype_name)

    # -- logical <-> storage K mapping ----------------------------------------
    # qdata shapes carry the STORAGE K extent. Legacy/container layouts use the
    # original integer elems_per_container contract; versioned dense bitstreams
    # derive the fractional ratio from elem_bits and the storage element width.

    @property
    def storage_elem_bits(self) -> int:
        """Bits per qdata storage element (the torch dtype's width)."""
        return self.qdata_dtype.itemsize * 8

    @property
    def is_packed(self) -> bool:
        """True when the storage K extent differs from logical K."""
        layout = getattr(self, "storage_layout", None)
        return layout == "packed_lsb_v1" or self.elems_per_container > 1

    @property
    def is_byte_container(self) -> bool:
        """True for deprecated one-code-per-storage-element sub-byte formats."""
        return not self.is_packed and self.elem_bits < self.storage_elem_bits

    def storage_k(self, logical_k: int) -> int:
        """Storage K extent of a qdata row holding ``logical_k`` elements
        (fp8: identity; fp4x2: K/2; packed fp6 in uint8: 3*K/4)."""
        if getattr(self, "storage_layout", None) == "packed_lsb_v1":
            bits = logical_k * self.elem_bits
            if bits % self.storage_elem_bits:
                raise ValueError(
                    f"{self.name}: logical K={logical_k} does not fill whole storage "
                    f"elements ({self.elem_bits}-bit codes in {self.storage_elem_bits}-bit units)"
                )
            return bits // self.storage_elem_bits
        if logical_k % self.elems_per_container:
            raise ValueError(
                f"{self.name}: logical K={logical_k} is not divisible by "
                f"elems_per_container={self.elems_per_container}"
            )
        return logical_k // self.elems_per_container

    def logical_k(self, storage_k: int) -> int:
        """Logical K extent of a qdata row of ``storage_k`` storage elements.
        Exact inverse of :meth:`storage_k`: rows hold whole packed groups
        (fp6: whole 3-byte groups of 4 codes)."""
        if getattr(self, "storage_layout", None) == "packed_lsb_v1":
            bits = storage_k * self.storage_elem_bits
            if bits % self.elem_bits:
                raise ValueError(
                    f"{self.name}: storage K={storage_k} is not whole packed groups "
                    f"({self.storage_elem_bits}-bit units of {self.elem_bits}-bit codes)"
                )
            return bits // self.elem_bits
        return storage_k * self.elems_per_container

    @classmethod
    def from_name(cls, name: str) -> "BlockScaledFormat":
        key = _LEGACY_FORMAT_NAMES.get(name, name)
        try:
            return BLOCKSCALED_FORMAT_REGISTRY[key]
        except KeyError:
            raise ValueError(
                f"unknown blockscaled format {name!r}; known: {sorted(BLOCKSCALED_FORMAT_REGISTRY)}"
            ) from None

    @classmethod
    def from_cutlass_dtypes(cls, ab_dtype, sf_dtype, sf_vec_size: int) -> "BlockScaledFormat":
        """Identify the format from CuTe-DSL dtypes (replaces the old
        ``_blockscaled_format_of`` if-ladder). NVFP4 vs MXFP4 disambiguates on the
        scale dtype + vec size; fp6 disambiguates on the element type."""
        from torch._vendor.quack.cute_dsl_utils import torch2cute_dtype_map  # lazy: needs the DSL

        for fmt in BLOCKSCALED_FORMAT_REGISTRY.values():
            if (
                fmt.cutlass_dtype_name is not None  # DSL-typeless formats can't match
                and not fmt.is_byte_container  # dtype triples select kernel-ready storage
                and fmt.to_cutlass_dtype() == ab_dtype
                and torch2cute_dtype_map[fmt.scale_dtype] == sf_dtype
                and fmt.sf_vec_size == sf_vec_size
            ):
                return fmt
        raise ValueError(
            f"no blockscaled format matches (ab={ab_dtype}, sf={sf_dtype}, vec={sf_vec_size})"
        )


MXFP8_E4M3 = BlockScaledFormat(
    "mxfp8_e4m3", torch.float8_e4m3fn, "Float8E4M3FN", 8, 1, torch.float8_e8m0fnu, 32
)
MXFP8_E5M2 = BlockScaledFormat(
    "mxfp8_e5m2", torch.float8_e5m2, "Float8E5M2", 8, 1, torch.float8_e8m0fnu, 32
)
MXFP4 = BlockScaledFormat(
    "mxfp4", torch.float4_e2m1fn_x2, "Float4E2M1FN", 4, 2, torch.float8_e8m0fnu, 32
)
NVFP4 = BlockScaledFormat(
    "nvfp4",
    torch.float4_e2m1fn_x2,
    "Float4E2M1FN",
    4,
    2,
    torch.float8_e4m3fn,
    16,
    has_per_tensor_scale=True,
)
# Origin/main's unversioned MXFP6 formats store one 6-bit code per uint8. Keep
# those names and descriptors byte-for-byte compatible with existing callers,
# checkpoints, and quantizer outputs. They are host-side compatibility formats:
# SM100 GEMM consumes the explicitly versioned packed formats below.
MXFP6_E2M3 = BlockScaledFormat(
    "mxfp6_e2m3", torch.uint8, "Float6E2M3FN", 6, 1, torch.float8_e8m0fnu, 32
)
MXFP6_E3M2 = BlockScaledFormat(
    "mxfp6_e3m2", torch.uint8, "Float6E3M2FN", 6, 1, torch.float8_e8m0fnu, 32
)

# Packed MXFP6 qdata is 6-bit storage in uint8 bytes: each row is the K 6-bit
# codes as a little-endian bit stream (element i occupies bits [6i, 6i+6)), so
# 4 codes pack into 3 bytes and the storage K extent is 3*K/4 (torch has no fp6
# dtype, so the bytes are raw uint8). This is the CUTLASS SubbyteReference
# layout the CU_TENSOR_MAP_DATA_TYPE_16U6 unpack tensormap consumes: TMA
# expands packed gmem into 8-bit SMEM containers for tcgen05 kind::mxf8f6f4
# (fp8's SMEM footprint, but 3/4 of its gmem/L2 bandwidth).
MXFP6_E2M3_PACKED = BlockScaledFormat(
    "mxfp6_e2m3_packed",
    torch.uint8,
    "Float6E2M3FN",
    6,
    1,
    torch.float8_e8m0fnu,
    32,
    storage_layout="packed_lsb_v1",
)
MXFP6_E3M2_PACKED = BlockScaledFormat(
    "mxfp6_e3m2_packed",
    torch.uint8,
    "Float6E3M2FN",
    6,
    1,
    torch.float8_e8m0fnu,
    32,
    storage_layout="packed_lsb_v1",
)

# Deprecated host-side compatibility formats. They remain quantizable,
# dequantizable, and serializable, but mma_kind_for_pair rejects them: SM100's
# mixed path consumes densely packed gmem via TMA unpack, not byte containers.
MXFP4_BYTE = BlockScaledFormat(
    "mxfp4_byte",
    torch.uint8,
    "Float4E2M1FN",
    4,
    1,
    torch.float8_e8m0fnu,
    32,
)
# Names added while packed FP6 was under development remain import aliases,
# but the public descriptor names are the origin/main unversioned names.
MXFP6_E2M3_BYTE = MXFP6_E2M3
MXFP6_E3M2_BYTE = MXFP6_E3M2

BLOCKSCALED_FORMAT_REGISTRY = {
    fmt.name: fmt
    for fmt in (
        MXFP8_E4M3,
        MXFP8_E5M2,
        MXFP4,
        NVFP4,
        MXFP6_E2M3,
        MXFP6_E3M2,
        MXFP6_E2M3_PACKED,
        MXFP6_E3M2_PACKED,
        MXFP4_BYTE,
    )
}

# Worked examples of formats the descriptor already expresses but that are NOT
# registered (registry rows land with their consumer kernels; see
# AI/blockscaled_api.md sections 9-10, and the executable versions in
# tests/test_blockscaled_operand.py::test_future_recipe_examples). The scale
# recipe axis mirrors cuBLASLt's cublasLtMatmulMatrixScale_t / torch's
# _ScalingType, factored into fields instead of an enum:
#
#   # DeepSeek-style 1x128 fp32-scale fp8 (SM90/SM100 blockwise promotion):
#   BlockScaledFormat("fp8_e4m3_1x128", torch.float8_e4m3fn, "Float8E4M3FN",
#                     8, 1, torch.float32, 128)
#   # kscale: bf16 elements, per-row fp32 scale every 128 of K (one-sided;
#   # quantize() would raise - scales are upstream-structural, from_parts only):
#   BlockScaledFormat("bf16_1x128", torch.bfloat16, "BFloat16",
#                     16, 1, torch.float32, 128)
#   # W4A16 int4-g128 (AWQ/GPTQ-style; uint4b8's fixed offset is element
#   # decode, not a descriptor field; bf16 scales are just data):
#   BlockScaledFormat("int4_1x128", torch.uint8, "Int4",
#                     4, 2, torch.bfloat16, 128)
#   # A hypothetical e3m4: no CuTe-DSL element type -> host-side only until a
#   # kernel declares its own (copy dtype, MMA dtype, convert) triple:
#   BlockScaledFormat("mxfp8_e3m4", torch.uint8, None,
#                     8, 1, torch.float8_e8m0fnu, 32)
#
# NVFP4 weight-only (W4A16) needs NO new row: the same NVFP4 format pairs with
# a plain bf16 activation tensor and dispatches to an upconvert kernel per
# arch. Still-missing axes (added with their first consumer): sf_block_mn (2D
# 128x128 grids), a full-extent sf_vec_size sentinel (rowwise / cuBLASLt
# OUTER_VEC), per-operand scale layouts ("linear"), and per-L batch scales.

# Legacy short names used by blockscaled_quantize / BLOCKSCALED_FORMATS (the
# inverse is derived - this dict is the single source of the aliasing).
_LEGACY_FORMAT_NAMES = {
    "mxfp8": "mxfp8_e4m3",
    "mxfp6_e2m3_byte": "mxfp6_e2m3",
    "mxfp6_e3m2_byte": "mxfp6_e3m2",
}
# Only mxfp8 has a preferred legacy generator name. The ``*_byte`` spellings
# above are lookup-only aliases and must not replace origin/main's stable
# unversioned keys in BLOCKSCALED_FORMATS.
_CANONICAL_TO_LEGACY = {"mxfp8_e4m3": "mxfp8"}


def legacy_format_name(fmt: "BlockScaledFormat") -> str:
    """The pre-descriptor short name ("mxfp8") where one exists, else fmt.name."""
    return _CANONICAL_TO_LEGACY.get(fmt.name, fmt.name)


def mma_kind_for_pair(fmt_a: "BlockScaledFormat", fmt_b: "BlockScaledFormat") -> str:
    """The tcgen05 MMA kind for an (A, B) format pair, or raise ValueError if the
    pair is not representable on the hardware. This is hardware LEGALITY only:
    whether a legal pair is implemented is enforced per-architecture by the
    gemm_smXXX kernel classes (each SM version supports different mx dtype
    combinations; see GemmSm100.is_valid_dtypes_and_scale_factor_vec_size,
    asserted in its blockscaled setup path).

    PTX rules: ``kind::mxf4nvf4`` requires fp4 on both operands and covers both
    scale configs - scale_vec::2X (vec 32, e8m0 = mxfp4; PTX also spells this
    instantiation ``kind::mxf4``) and scale_vec::4X (vec 16 = nvfp4). It is ONE
    MMA atom parameterized by (sf_dtype, sf_vec_size), mirroring CUTLASS C++'s
    ``SM100_MMA_MXF4_SS<..., VS>``; the scale config is instruction-wide, so
    the two operands' formats must match. ``kind::mxf8f6f4`` admits any mix of
    fp8/fp6/fp4 element types (e8m0 scales, vec 32).

    The kind rules are encoded in three places that must stay in sync (this
    function cannot be shared: the kernel layer keys on cutlass dtypes and must
    not import torch-level format descriptors): here (format names), instruction-K
    in ``GemmSm100._blockscaled_mma_inst_k`` (storage dtypes), and the per-arch
    subset in ``GemmSm100.is_valid_dtypes_and_scale_factor_vec_size`` (MMA
    dtypes). ``test_mma_kind_mirrors_kernel_inst_k`` pins the first mirror.
    """
    for fmt in (fmt_a, fmt_b):
        if fmt.is_byte_container:
            raise ValueError(
                f"{fmt.name} uses deprecated byte-per-element sub-byte storage and is "
                "host-side compatibility only; call operand.to_packed() before GEMM"
            )
    if fmt_a.name == "nvfp4" or fmt_b.name == "nvfp4":
        if fmt_a.name == fmt_b.name:
            return "mxf4nvf4"
        raise ValueError(
            f"nvfp4 (e4m3 scales, vec 16) cannot pair with "
            f"{fmt_b.name if fmt_a.name == 'nvfp4' else fmt_a.name}: "
            f"the mxf4nvf4 MMA kind's scale config is instruction-wide, so it "
            f"requires nvfp4 on both operands"
        )
    if fmt_a.name == "mxfp4" and fmt_b.name == "mxfp4":
        return "mxf4nvf4"
    # Gate BOTH axes explicitly rather than falling through: a format whose
    # elements no tcgen05 kind can consume (no DSL type, or a non-fp8/6/4 one),
    # or whose scale RECIPE the SF hardware cannot represent (kind::mxf8f6f4 is
    # e8m0 / vec-32 only — fp32 scales, vec 128, 2D grids are software recipes),
    # must fail HERE with the reason, not at the per-arch gate later. Software
    # kinds (blockwise promotion: fp32-scale fp8, one-sided bf16, int4) are a
    # follow-up with their own kind names — see AI/blockscaled_api.md section 9.
    for fmt in (fmt_a, fmt_b):
        if _mma_element_class(fmt) is None:
            raise ValueError(
                f"{fmt.name} has no tcgen05 MMA element type "
                f"(cutlass_dtype_name={fmt.cutlass_dtype_name}): no hardware "
                f"blockscaled MMA kind exists for this format"
            )
        if fmt.scale_dtype != torch.float8_e8m0fnu or fmt.sf_vec_size != 32:
            raise ValueError(
                f"{fmt.name} (scales {fmt.scale_dtype}, vec {fmt.sf_vec_size}) has no "
                f"hardware MMA kind: kind::mxf8f6f4 requires e8m0 scales with vec size "
                f"32; software-scaled recipes are a follow-up (AI/blockscaled_api.md "
                f"section 9)"
            )
    # Everything else - fp8 x fp8 and any mix involving a packed sub-byte
    # operand - runs kind::mxf8f6f4 (all e8m0 scales / vec 32): sub-byte
    # operands are loaded from packed gmem via TMA unpack into 8-bit smem
    # containers (the CUTLASS *_unpacksmem_t convention).
    return "mxf8f6f4"


def _mma_element_class(fmt: "BlockScaledFormat") -> Optional[str]:
    """The tcgen05 element class ("f8" | "f6" | "f4") of a format's MMA type, or
    None when the format has no hardware-representable element type. Derived from
    the DSL type name: PTX kind legality is a property of the MMA element type."""
    name = fmt.cutlass_dtype_name
    for prefix, cls in (("Float8", "f8"), ("Float6", "f6"), ("Float4", "f4")):
        if name is not None and name.startswith(prefix):
            return cls
    return None


def _coerce_format(format) -> BlockScaledFormat:
    if isinstance(format, BlockScaledFormat):
        interim_targets = {
            ("mxfp6_e2m3", "packed_lsb_v1"): MXFP6_E2M3_PACKED,
            ("mxfp6_e3m2", "packed_lsb_v1"): MXFP6_E3M2_PACKED,
        }
        target = interim_targets.get((format.name, getattr(format, "storage_layout", None)))
        if target is not None:
            if replace(format, name=target.name) != target:
                raise ValueError(
                    f"ambiguous interim packed FP6 descriptor {format.name}: fields do not "
                    f"match {target.name}"
                )
            return target
        return format
    if isinstance(format, str):
        return BlockScaledFormat.from_name(format)
    raise TypeError(f"format must be a BlockScaledFormat or str, got {type(format)}")


def _normalize_quant_dim(dim: int, ndim: int) -> int:
    """Normalize a quantized-dim argument to -1 or -2 (the only two directions a
    block-scaled operand can have - the batch dim is never quantized)."""
    d = dim if dim < 0 else dim - ndim
    if d not in (-1, -2):
        raise ValueError(f"quant_dim must be one of the last two dims, got {dim}")
    return d


def _packed_dim(qdata: torch.Tensor, fmt: BlockScaledFormat) -> int:
    """The qdata dim holding multiple logical elements per storage element.

    Convention: the unit-stride dim. Packing is always along the quantization
    (K) axis, and packed (fp4/fp6) formats are required K-major by the kernel, so
    unit-stride == packed == K. Scanned from the last dim so fully-contiguous
    qdata resolves to the innermost dim. Size-1 dims report arbitrary strides
    (a (Kp, 1) K-major operand can present stride 1 on BOTH dims), so dims with
    extent > 1 take precedence; if every unit-stride dim is size-1 the tensor is
    degenerate and any choice is equivalent - the innermost wins.
    """
    fallback = None
    for d in range(qdata.ndim - 1, -1, -1):
        if qdata.stride(d) == 1:
            if qdata.shape[d] != 1:
                return d
            fallback = d if fallback is None else fallback
    if fallback is not None:
        return fallback
    raise ValueError(
        f"qdata has no unit-stride dim: shape={tuple(qdata.shape)}, stride={qdata.stride()}"
    )


@dataclass(frozen=True, eq=False)  # eq=False: tensor fields do not compare with ==
class BlockScaledOperand:
    """Block-scaled quantized operand for quack GEMMs.

    A plain container - not a torch.Tensor. ``shape``/``dtype`` report the
    *logical* view (unpacked shape, original high-precision dtype); storage truth
    is ``qdata``. torch ops do not accept it (loud TypeError); the supported
    surface is quack.gemm* plus the explicit methods here.

    Construction validates mode-independent invariants only. The qdata-shape <->
    scale-shape relation is intentionally NOT checked here: varlen operands use
    padded scale buffers whose shape depends on cu_seqlens; the coupling is
    validated at GEMM dispatch time (``validate_blockscaled_sf``).
    """

    qdata: torch.Tensor
    scale: torch.Tensor
    format: BlockScaledFormat
    per_tensor_scale: Optional[torch.Tensor] = None  # scalar fp32, NVFP4 only
    orig_dtype: torch.dtype = torch.bfloat16
    # The logical dim the scale vector (sf_vec_size-sized blocks) runs along,
    # normalized to -1 or -2. The direct analogue of CUTLASS's UMMA::Major on
    # SfKMajorAtom/SfMNMajorAtom (sm100_blockscaled_layout.hpp) - the two atoms
    # are mode-swaps of one physical 512 B block, so a quant_dim=-2 operand is
    # byte-identical to the quant_dim=-1 operand of the transposed data. Resolved
    # in __post_init__ (fp4 packing pins it to the packed dim); None -> -1.
    quant_dim: Optional[int] = None

    def __post_init__(self):
        fmt = _coerce_format(self.format)
        object.__setattr__(self, "format", fmt)
        if self.qdata.dtype != fmt.qdata_dtype:
            raise TypeError(f"{fmt.name} qdata must be {fmt.qdata_dtype}, got {self.qdata.dtype}")
        check_blocked_scale_atom(self.scale, "scale")
        # Canonicalize the scale dtype from the format. A uint8 view is accepted
        # (scales round-tripped through collectives / the custom-op boundary arrive
        # as uint8) and re-viewed; anything else is a construction error. This kills
        # the old "uint8 implies e8m0" sniffing trap where an NVFP4 e4m3 scale seen
        # as uint8 would silently run as a vec-32 MX format.
        if self.scale.dtype == torch.uint8:
            object.__setattr__(self, "scale", self.scale.view(fmt.scale_dtype))
        elif self.scale.dtype != fmt.scale_dtype:
            raise TypeError(f"{fmt.name} scale must be {fmt.scale_dtype}, got {self.scale.dtype}")
        if self.per_tensor_scale is not None:
            if not fmt.has_per_tensor_scale:
                raise ValueError(f"{fmt.name} does not take a per_tensor_scale")
            if self.per_tensor_scale.numel() != 1 or self.per_tensor_scale.dtype != torch.float32:
                raise ValueError(
                    f"per_tensor_scale must be a scalar fp32 tensor, got "
                    f"{self.per_tensor_scale.dtype} with {self.per_tensor_scale.numel()} elements"
                )
            # Canonical (1,) shape: the GEMM's alpha folding then never reshapes.
            object.__setattr__(self, "per_tensor_scale", self.per_tensor_scale.reshape(1))
        if self.qdata.device != self.scale.device:
            raise ValueError(f"qdata on {self.qdata.device} but scale on {self.scale.device}")
        if self.per_tensor_scale is not None and self.qdata.device != self.per_tensor_scale.device:
            raise ValueError(
                f"qdata on {self.qdata.device} but per_tensor_scale on "
                f"{self.per_tensor_scale.device}"
            )
        requested = (
            None if self.quant_dim is None else _normalize_quant_dim(self.quant_dim, self.ndim)
        )
        if fmt.is_packed:
            # fp4/fp6 packing pins the quantized axis to the packed (unit-stride) dim.
            derived = -1 if _packed_dim(self.qdata, fmt) == self.qdata.ndim - 1 else -2
            if requested is not None and requested != derived:
                raise ValueError(
                    f"quant_dim={self.quant_dim} conflicts with the {fmt.name} packed dim "
                    f"(packing runs along the quantized axis, here dim {derived})"
                )
            object.__setattr__(self, "quant_dim", derived)
        else:
            object.__setattr__(self, "quant_dim", -1 if requested is None else requested)

    # -- logical metadata (honest, computed - no advertised-metadata machinery) --

    @property
    def shape(self) -> Tuple[int, ...]:
        """Logical (unpacked) shape; the packed dim (fp4x2: K/2, packed fp6:
        3*K/4 bytes) maps back to logical K, which ``quant_dim`` already
        tracks."""
        fmt = self.format
        if not fmt.is_packed:
            return tuple(self.qdata.shape)
        k_dim = self._mn_k_dims()[1]
        return tuple(fmt.logical_k(s) if d == k_dim else s for d, s in enumerate(self.qdata.shape))

    @property
    def ndim(self) -> int:
        return self.qdata.ndim

    @property
    def dtype(self) -> torch.dtype:
        """The original high-precision dtype (storage truth is ``qdata.dtype``)."""
        return self.orig_dtype

    @property
    def device(self) -> torch.device:
        return self.qdata.device

    def _mn_k_dims(self) -> Tuple[int, int]:
        d0, d1 = self.ndim - 2, self.ndim - 1
        return (d0, d1) if self.quant_dim == -1 else (d1, d0)

    # -- views: qdata stride-swap, scale carried unchanged ------------------

    def _view(self, qdata: torch.Tensor, quant_dim: int) -> "BlockScaledOperand":
        """Unvalidated view constructor: a stride swap of valid qdata cannot
        invalidate any invariant __post_init__ established on ``self`` (and
        ``.mT`` runs once per GEMM call - skip the re-validation)."""
        obj = object.__new__(BlockScaledOperand)
        for name, value in (
            ("qdata", qdata),
            ("scale", self.scale),
            ("format", self.format),
            ("per_tensor_scale", self.per_tensor_scale),
            ("orig_dtype", self.orig_dtype),
            ("quant_dim", quant_dim),
        ):
            object.__setattr__(obj, name, value)
        return obj

    @property
    def mT(self) -> "BlockScaledOperand":
        """Swap the last two logical dims. The blocked scale atom is not
        view-transposable; scale blocks along K in either orientation."""
        return self._view(self.qdata.transpose(-2, -1), -3 - self.quant_dim)  # -1 <-> -2

    @property
    def T(self) -> "BlockScaledOperand":
        if self.ndim != 2:
            raise ValueError(f"BlockScaledOperand.T requires 2-D, got {self.ndim}-D")
        return self.mT

    def transpose(self, dim0: int, dim1: int) -> "BlockScaledOperand":
        dim0, dim1 = dim0 % self.ndim, dim1 % self.ndim
        if {dim0, dim1} != {self.ndim - 2, self.ndim - 1}:
            raise ValueError(
                f"BlockScaledOperand.transpose({dim0}, {dim1}): only the last two dims may "
                f"swap (scale blocks along K; other permutations are not representable)"
            )
        return self.mT

    # -- constructors / conversions ------------------------------------------

    @classmethod
    def from_parts(
        cls,
        qdata: torch.Tensor,
        scale: torch.Tensor,
        format,
        *,
        per_tensor_scale: Optional[torch.Tensor] = None,
        orig_dtype: torch.dtype = torch.bfloat16,
        quant_dim: int = -1,
    ) -> "BlockScaledOperand":
        """Wrap already-quantized storage (checkpoints, external quantizers).

        ``quant_dim`` is the logical dim the scales block along (default: last,
        the quantize convention). To express the (K, N) B-operand orientation,
        either construct the (N, K) operand and take its ``.mT`` view, or pass
        ``quant_dim=-2`` for data already laid out as (K, N).
        """
        return _construct(
            qdata, scale, _coerce_format(format), per_tensor_scale, orig_dtype, quant_dim
        )

    @classmethod
    def quantize(
        cls,
        x: torch.Tensor,
        format,
        *,
        dim: int = -1,
        per_tensor_scale: Optional[torch.Tensor] = None,
    ) -> "BlockScaledOperand":
        """Quantize a 2-D/3-D bf16/fp32 tensor along ``dim`` (one of the last two
        dims; default: last).

        ``dim=-2`` quantizes along the second-to-last dim via a transpose copy of
        ``x`` (the cost CUTLASS/torchao pay for their dim-1 casts too) and returns
        the ``quant_dim=-2`` view - useful to build a (K, N) B operand directly.

        The NVFP4 ``per_tensor_scale`` is stored on the result and folded into
        GEMM alpha automatically at dispatch (the tuple-era "caller folds
        pts_A*pts_B into alpha" contract does not apply to container operands).
        """
        fmt = _coerce_format(format)
        if _normalize_quant_dim(dim, x.ndim) == -2:
            xt = x.transpose(-1, -2).contiguous()
            return cls.quantize(xt, fmt, per_tensor_scale=per_tensor_scale).mT
        quantizer_name = fmt.name
        if fmt.is_byte_container:
            quantizer_name = {
                "Float4E2M1FN": "mxfp4_byte",
                "Float6E2M3FN": "mxfp6_e2m3",
                "Float6E3M2FN": "mxfp6_e3m2",
            }.get(fmt.cutlass_dtype_name, fmt.name)
        quantizer = QUANTIZERS.get(quantizer_name)
        if quantizer is None:
            # Every registered format has a quantizer; this is the path for
            # custom (unregistered) descriptors.
            raise NotImplementedError(
                f"no in-repo quantizer for {fmt.name}; construct pre-quantized data via from_parts"
            )
        if per_tensor_scale is not None and not fmt.has_per_tensor_scale:
            raise ValueError(f"{fmt.name} does not take a per_tensor_scale")
        assert x.ndim in (2, 3), f"expected (M, K) or (L, M, K), got shape {tuple(x.shape)}"
        assert x.shape[-1] % fmt.sf_vec_size == 0, (
            f"K ({x.shape[-1]}) must be divisible by {fmt.sf_vec_size} for {fmt.name}"
        )
        # Under dynamo, call the raw quantizer: the torch.compile'd wrapper would
        # nest compilation; the raw fn traces into the enclosing graph.
        fn = quantizer[0] if torch.compiler.is_compiling() else quantizer[1]
        batched = x.ndim == 3
        l, mn, k = x.shape if batched else (1, *x.shape)
        x_flat = x.reshape(l * mn, k)
        pts = None
        if fmt.has_per_tensor_scale:
            q, sc, pts_out = fn(x_flat, fmt.sf_vec_size, per_tensor_scale)
            pts = pts_out.to(torch.float32) if per_tensor_scale is not None else None
        else:
            q, sc = fn(x_flat, fmt.sf_vec_size)
        # Quantizers emit the descriptor's storage: fp4 as uint8 nibble pairs
        # (re-viewed to fp4x2), unversioned fp6 as byte codes, packed fp6 as a
        # uint8 bit stream, and fp8 directly in its storage dtype.
        if q.dtype != fmt.qdata_dtype:
            q = q.view(fmt.qdata_dtype)
        q = q.reshape(*x.shape[:-1], -1)
        sf = pack_scale_2d_to_blocked_contig(sc.view(l, mn, k // fmt.sf_vec_size))
        if not batched:
            sf = sf.squeeze(0)
        return _construct(q, sf, fmt, pts, x.dtype, -1)

    def to_packed(self) -> "BlockScaledOperand":
        """Migrate a deprecated byte-container operand to canonical packed storage.

        The element codes and scale factors are preserved bit-exactly; no
        dequantize/requantize round trip is involved. Canonical packed operands
        are returned unchanged.
        """
        fmt = self.format

        def check_recipe(target):
            source_recipe = (
                fmt.cutlass_dtype_name,
                fmt.elem_bits,
                fmt.scale_dtype,
                fmt.sf_vec_size,
                fmt.has_per_tensor_scale,
            )
            target_recipe = (
                target.cutlass_dtype_name,
                target.elem_bits,
                target.scale_dtype,
                target.sf_vec_size,
                target.has_per_tensor_scale,
            )
            if source_recipe != target_recipe:
                raise ValueError(
                    f"cannot repack {fmt.name} as {target.name}: its element/scale recipe "
                    "does not match the canonical packed format"
                )

        if not fmt.is_byte_container:
            # Packed FP6 briefly shipped with the unversioned byte-format name.
            # Relabel those descriptors so schema-boundary name resolution does
            # not reinterpret their 3K/4 storage as one-byte-per-code storage.
            interim_targets = {
                ("mxfp6_e2m3", "packed_lsb_v1"): MXFP6_E2M3_PACKED,
                ("mxfp6_e3m2", "packed_lsb_v1"): MXFP6_E3M2_PACKED,
            }
            target = interim_targets.get((fmt.name, getattr(fmt, "storage_layout", None)))
            if target is not None:
                check_recipe(target)
                return _construct(
                    self.qdata,
                    self.scale,
                    target,
                    self.per_tensor_scale,
                    self.orig_dtype,
                    self.quant_dim,
                )
            return self
        from torch._vendor.quack.blockscaled.quantize import _pack_uint4, pack_uint6

        targets = {
            "Float4E2M1FN": MXFP4,
            "Float6E2M3FN": MXFP6_E2M3_PACKED,
            "Float6E3M2FN": MXFP6_E3M2_PACKED,
        }
        target = targets.get(fmt.cutlass_dtype_name)
        if target is None:
            raise NotImplementedError(f"no canonical packed format for {fmt.name}")
        check_recipe(target)
        mn_dim, k_dim = self._mn_k_dims()
        qdata = self.qdata if k_dim == self.ndim - 1 else self.qdata.transpose(mn_dim, k_dim)
        if target is MXFP4:
            qdata = _pack_uint4(qdata).view(target.qdata_dtype)
        else:
            qdata = pack_uint6(qdata)
        if k_dim != self.ndim - 1:
            qdata = qdata.transpose(mn_dim, k_dim)
        return _construct(
            qdata,
            self.scale,
            target,
            self.per_tensor_scale,
            self.orig_dtype,
            self.quant_dim,
        )

    def dequantize(self, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        """Explicit dequantization to a plain high-precision tensor.

        The scale buffer must have the canonical dense shape for ``qdata``.
        Varlen GEMMs use padded scale offsets derived from ``cu_seqlens``;
        those buffers cannot be interpreted without metadata the operand does
        not carry and therefore raise instead of returning incorrect values.
        """
        fmt = self.format
        from torch._vendor.quack.blockscaled.quantize import dequant_operand, unpack_scale_blocked_to_2d

        mn_dim, k_dim = self._mn_k_dims()
        qdata = self.qdata if k_dim == self.ndim - 1 else self.qdata.transpose(mn_dim, k_dim)
        batched = self.ndim == 3
        q3 = qdata if batched else qdata.unsqueeze(0)  # (l, mn, k_packed)
        vals = dequant_operand(q3, fmt)  # (l, mn, k) fp32
        l, mn, k = vals.shape
        scale = self.scale if self.scale.ndim == 6 else self.scale.unsqueeze(0)
        sf_k = (k + fmt.sf_vec_size - 1) // fmt.sf_vec_size
        expected_scale_shape = (l, (mn + 127) // 128, (sf_k + 3) // 4, 32, 4, 4)
        if tuple(scale.shape) != expected_scale_shape:
            raise ValueError(
                "dequantize requires dense scale factors matching qdata; "
                f"got {tuple(scale.shape)}, expected {expected_scale_shape}. "
                "Padded varlen scale buffers require sequence offsets, which "
                "BlockScaledOperand does not carry"
            )
        sf2d = unpack_scale_blocked_to_2d(scale, mn, sf_k).float()
        scale_values = sf2d.view(l, mn, sf_k).repeat_interleave(fmt.sf_vec_size, dim=-1)
        out = vals * scale_values[..., :k]
        if self.per_tensor_scale is not None:
            out = out * self.per_tensor_scale
        if not batched:
            out = out.squeeze(0)
        if k_dim != self.ndim - 1:
            out = out.transpose(mn_dim, k_dim)
        return out.to(dtype if dtype is not None else self.dtype)

    def to(self, device, non_blocking: bool = False) -> "BlockScaledOperand":
        """Device move only. Implicit dtype conversion of quantized storage is not
        supported; call ``dequantize(dtype)`` explicitly."""
        if isinstance(device, torch.dtype):
            raise TypeError(
                f"BlockScaledOperand.to({device}): dtype conversion of quantized storage "
                f"is not supported; call .dequantize(dtype) explicitly"
            )
        move = lambda t: t.to(device, non_blocking=non_blocking) if t is not None else None
        return replace(
            self,
            qdata=move(self.qdata),
            scale=move(self.scale),
            per_tensor_scale=move(self.per_tensor_scale),
        )

    def clone(self) -> "BlockScaledOperand":
        c = lambda t: t.clone() if t is not None else None
        return replace(
            self,
            qdata=c(self.qdata),
            scale=c(self.scale),
            per_tensor_scale=c(self.per_tensor_scale),
        )

    def __deepcopy__(self, memo):
        d = lambda t: _copy.deepcopy(t, memo) if t is not None else None
        return replace(
            self,
            qdata=d(self.qdata),
            scale=d(self.scale),
            per_tensor_scale=d(self.per_tensor_scale),
        )

    def __repr__(self):
        pts = "" if self.per_tensor_scale is None else f", per_tensor_scale={self.per_tensor_scale}"
        return (
            f"BlockScaledOperand(format={self.format.name}, shape={self.shape}, "
            f"quant_dim={self.quant_dim}, dtype={self.dtype}, "
            f"qdata={self.qdata.dtype}{tuple(self.qdata.shape)}, "
            f"scale={tuple(self.scale.shape)}{pts}, device={self.device})"
        )


def _construct(qdata, scale, fmt, per_tensor_scale, orig_dtype, quant_dim):
    return BlockScaledOperand(qdata, scale, fmt, per_tensor_scale, orig_dtype, quant_dim)


# pytree registration: lets the container cross torch.compile / functional-transform
# boundaries as a structure of tensor leaves (same mechanics as the legacy tuple).
_PYTREE_CONTEXT_VERSION = 1


def _dtype_to_context(dtype: torch.dtype) -> str:
    name = str(dtype)
    if not name.startswith("torch."):
        raise TypeError(f"cannot serialize torch dtype {dtype!r}")
    return name.removeprefix("torch.")


def _dtype_from_context(name) -> torch.dtype:
    # Old three-tuple contexts carried the dtype object directly.
    if isinstance(name, torch.dtype):
        return name
    if not isinstance(name, str):
        raise TypeError(f"invalid serialized torch dtype {name!r}")
    dtype = getattr(torch, name.removeprefix("torch."), None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"unknown serialized torch dtype {name!r}")
    return dtype


def _format_to_context(fmt: BlockScaledFormat) -> dict:
    """Encode every descriptor field using JSON scalar values only."""
    return {
        "name": fmt.name,
        "qdata_dtype": _dtype_to_context(fmt.qdata_dtype),
        "cutlass_dtype_name": fmt.cutlass_dtype_name,
        "elem_bits": fmt.elem_bits,
        "elems_per_container": fmt.elems_per_container,
        "scale_dtype": _dtype_to_context(fmt.scale_dtype),
        "sf_vec_size": fmt.sf_vec_size,
        "has_per_tensor_scale": fmt.has_per_tensor_scale,
        "storage_layout": getattr(fmt, "storage_layout", None),
    }


def _format_from_context(context: dict) -> BlockScaledFormat:
    if not isinstance(context, dict):
        raise TypeError(f"invalid serialized BlockScaledFormat context {context!r}")
    fmt = BlockScaledFormat(
        name=context["name"],
        qdata_dtype=_dtype_from_context(context["qdata_dtype"]),
        cutlass_dtype_name=context["cutlass_dtype_name"],
        elem_bits=context["elem_bits"],
        elems_per_container=context["elems_per_container"],
        scale_dtype=_dtype_from_context(context["scale_dtype"]),
        sf_vec_size=context["sf_vec_size"],
        has_per_tensor_scale=context["has_per_tensor_scale"],
        storage_layout=context.get("storage_layout"),
    )
    registered = BLOCKSCALED_FORMAT_REGISTRY.get(fmt.name)
    return registered if registered == fmt else fmt


def _flatten(op: BlockScaledOperand):
    children = (op.qdata, op.scale, op.per_tensor_scale)
    ctx = {
        "version": _PYTREE_CONTEXT_VERSION,
        "format": _format_to_context(op.format),
        "orig_dtype": _dtype_to_context(op.orig_dtype),
        "quant_dim": op.quant_dim,
    }
    return children, ctx


def _unflatten(children, ctx):
    qdata, scale, pts = children
    if isinstance(ctx, dict):
        version = ctx.get("version")
        if version != _PYTREE_CONTEXT_VERSION:
            raise ValueError(f"unsupported BlockScaledOperand pytree context version {version!r}")
        fmt = _format_from_context(ctx["format"])
        orig_dtype = _dtype_from_context(ctx["orig_dtype"])
        quant_dim = ctx["quant_dim"]
    else:
        # Origin/main emitted (format_name, torch.dtype, quant_dim). Also accept
        # the descriptor-valued three-tuple used briefly during packed-FP6
        # development so in-memory TreeSpecs remain reconstructible.
        if not isinstance(ctx, (tuple, list)) or len(ctx) != 3:
            raise TypeError(f"invalid legacy BlockScaledOperand pytree context {ctx!r}")
        fmt_or_name, orig_dtype, quant_dim = ctx
        fmt = (
            fmt_or_name
            if isinstance(fmt_or_name, BlockScaledFormat)
            else BlockScaledFormat.from_name(fmt_or_name)
        )
        orig_dtype = _dtype_from_context(orig_dtype)
    return BlockScaledOperand(qdata, scale, fmt, pts, orig_dtype, quant_dim)


def _dump_pytree_context(ctx):
    return ctx


def _load_pytree_context(ctx):
    return ctx


_pytree.register_pytree_node(
    BlockScaledOperand,
    _flatten,
    _unflatten,
    serialized_type_name="torch._vendor.quack.blockscaled.operand.BlockScaledOperand",
    to_dumpable_context=_dump_pytree_context,
    from_dumpable_context=_load_pytree_context,
)

# weights_only torch.load of pickled containers needs the classes allowlisted.
torch.serialization.add_safe_globals([BlockScaledOperand, BlockScaledFormat])
