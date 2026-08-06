"""Packed low-precision conversions for FlexGEMM quantization epilogues."""

import functools
import hashlib
from pathlib import Path

import cutlass
import cutlass.cute as cute
from cutlass.cutlass_dsl import dsl_user_op


@functools.cache
def quant_intrinsics_cache_key() -> str:
    """Include this module's source in generated epilogue cache keys."""
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


@dsl_user_op
def nvfp4_pack_intrinsic(source, *, loc=None, ip=None):
    """Round paired Float32 values with SM100's packed E2M1 conversion."""
    packed = source.to(cutlass.Float4E2M1FN).bitcast(cutlass.Uint8)
    return packed.reshape((cute.size(packed.shape), 1, 1))
