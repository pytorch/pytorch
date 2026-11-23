# Copyright (c) 2026, Tri Dao.
"""Blockscaled (MXFP8 / MXFP4 / MXFP6 / NVFP4) GEMM support.

- :mod:`quack.blockscaled.quantize` — pure-PyTorch quantizers (ported from
  torchao) with torch.compile'd fast paths.
- :mod:`quack.blockscaled.utils` — scale-factor packing/unpacking, operand
  builders for tests/benchmarks, and the kernel-level compile path.
- :mod:`quack.blockscaled.quantize_utils` — the device-side (CuTe-DSL)
  quantize core shared by the SFD epilogue and fused-kernel quantized
  outputs (bit-exact cuBLAS/CUTLASS SF semantics).

The GEMM entry points live in :mod:`quack.gemm_interface`; pass
:class:`BlockScaledOperand` operands (``(data, scale_factor)`` tuples are
rejected with a TypeError). Design doc: ``AI/blockscaled_api.md``.
"""

from torch._vendor.quack.blockscaled.operand import (  # noqa: F401
    BLOCKSCALED_FORMAT_REGISTRY,
    MXFP4,
    MXFP4_BYTE,
    MXFP6_E2M3,
    MXFP6_E2M3_BYTE,
    MXFP6_E2M3_PACKED,
    MXFP6_E3M2,
    MXFP6_E3M2_BYTE,
    MXFP6_E3M2_PACKED,
    MXFP8_E4M3,
    MXFP8_E5M2,
    NVFP4,
    BlockScaledFormat,
    BlockScaledOperand,
    mma_kind_for_pair,
)
from torch._vendor.quack.blockscaled.quantize import (  # noqa: F401
    nvfp4_per_tensor_scale,
    to_blocked,
    to_mx,
    to_mx_compiled,
    to_mx_dim0,
    to_mx_dim0_compiled,
    to_mxfp4,
    to_mxfp4_byte,
    to_mxfp4_byte_compiled,
    to_mxfp4_compiled,
    to_mxfp6_e2m3,
    to_mxfp6_e2m3_byte,
    to_mxfp6_e2m3_byte_compiled,
    to_mxfp6_e2m3_compiled,
    to_mxfp6_e2m3_packed,
    to_mxfp6_e2m3_packed_compiled,
    to_mxfp6_e3m2,
    to_mxfp6_e3m2_byte,
    to_mxfp6_e3m2_byte_compiled,
    to_mxfp6_e3m2_compiled,
    to_mxfp6_e3m2_packed,
    to_mxfp6_e3m2_packed_compiled,
    to_nvfp4,
    to_nvfp4_compiled,
)
from torch._vendor.quack.blockscaled.utils import (  # noqa: F401
    BLOCKSCALED_FORMATS,
    blockscaled_gemm_reference,
    blockscaled_quantize,
    blockscaled_quantize_dim0,
    dequant_operand,
    pack_scale_2d_to_blocked_contig,
    scale_blocked_for_cublas,
    unpack_scale_blocked_to_2d,
)
