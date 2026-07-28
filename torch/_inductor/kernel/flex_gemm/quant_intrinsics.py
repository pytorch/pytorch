"""Packed low-precision conversions for FlexGEMM quantization epilogues.

CuTeDSL's public E8M0 conversion neither exposes FLOOR rounding nor handles every
TensorSSA fragment width needed by FlexGEMM. NVFP4 uses its native packed E2M1
conversion directly; pre-rounding through pointwise selects produces equivalent
finite values with substantially worse code generation.
"""

import functools
import hashlib
import math
from pathlib import Path

import cutlass
import cutlass.cute as cute
from cutlass._mlir import ir
from cutlass._mlir.dialects import arith, llvm, vector
from cutlass.cutlass_dsl import dsl_user_op, T


@functools.cache
def quant_intrinsics_cache_key() -> str:
    """Include this module's source in generated epilogue cache keys."""
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


@dsl_user_op
def nvfp4_pack_intrinsic(source, *, loc=None, ip=None):
    """Round paired Float32 values with SM100's packed E2M1 conversion."""
    packed = source.to(cutlass.Float4E2M1FN).bitcast(cutlass.Uint8)
    return packed.reshape((cute.size(packed.shape), 1, 1))


def extract_vector_element(value: ir.Value, index: int) -> ir.Value:
    """Extract one statically indexed element from an MLIR vector."""
    return vector.extract(value, [], [index])


def convert_packed_e8m0_pair_to_f32(packed: ir.Value) -> tuple[ir.Value, ir.Value]:
    """Reconstruct two E8M0 values from packed BF16 values."""
    pair_type = ir.VectorType.get([2], cutlass.BFloat16.mlir_type)
    pair = llvm.bitcast(pair_type, packed)
    return tuple(
        arith.extf(
            cutlass.Float32.mlir_type,
            extract_vector_element(pair, index),
        )
        for index in range(2)
    )


def prepare_floor_e8m0_input(
    value: ir.Value, floor_inf_value: float | None
) -> ir.Value:
    """Replace infinity when FLOOR encoding must preserve exponent extraction."""
    value = cutlass.Float32(value)
    if floor_inf_value is None:
        return value.ir_value()
    is_inf = value == float("inf")
    return arith.select(
        is_inf.ir_value(),
        cutlass.Float32(floor_inf_value).ir_value(),
        value.ir_value(),
    )


def convert_e8m0_pair(
    low: ir.Value,
    high: ir.Value,
    rounding: str,
    floor_inf_value: float | None,
) -> tuple[ir.Value, ir.Value]:
    """Convert two Float32 values to E8M0 and reconstruct them as Float32."""
    instruction = {
        "floor": "cvt.rz.ue8m0x2.f32",
        "rceil": "cvt.rp.satfinite.ue8m0x2.f32",
    }[rounding]
    if rounding == "floor":
        low = prepare_floor_e8m0_input(low, floor_inf_value)
        high = prepare_floor_e8m0_input(high, floor_inf_value)
    packed = llvm.inline_asm(
        T.i32(),
        [low, high],
        (
            f"{{ .reg .b16 scale; {instruction} scale, $2, $1; "
            "cvt.rn.bf16x2.ue8m0x2 $0, scale; }"
        ),
        "=r,f,f",
        has_side_effects=False,
        is_align_stack=False,
    )
    return convert_packed_e8m0_pair_to_f32(packed)


def map_converted_pairs(
    source: cute.TensorSSA,
    rounding: str,
    floor_inf_value: float | None,
) -> cute.TensorSSA:
    """Apply the packed E8M0 conversion to every TensorSSA pair."""
    source_value = source.ir_value()
    count = math.prod(ir.VectorType(source_value.type).shape)
    converted = []
    for index in range(0, count, 2):
        low = extract_vector_element(source_value, index)
        high = (
            extract_vector_element(source_value, index + 1)
            if index + 1 < count
            else arith.constant(cutlass.Float32.mlir_type, 0.0)
        )
        converted.extend(convert_e8m0_pair(low, high, rounding, floor_inf_value))
    result_type = ir.VectorType.get([count], cutlass.Float32.mlir_type)
    return cute.TensorSSA(
        vector.from_elements(result_type, converted[:count]),
        source.shape,
        cutlass.Float32,
    )


@dsl_user_op
def mx_e8m0_scale_intrinsic(
    source,
    max_value: float,
    rounding: str,
    *,
    loc=None,
    ip=None,
):
    """Encode an MX scale with exact packed SM100 E8M0 conversion.

    CuTeDSL's public conversion does not support every fragment width or expose
    FLOOR/RZ. The MLIR boundary emits TorchAO-compatible saturating RCEIL or
    exponent-compatible FLOOR conversion per pair, then rebuilds the original
    TensorSSA.
    """
    max_power = math.floor(math.log2(max_value))
    scaled = source / max_value if rounding == "rceil" else source * 2.0**-max_power
    floor_inf_value = (
        2.0 ** (128 - max_power) if rounding == "floor" and max_power > 0 else None
    )

    if isinstance(scaled, cute.TensorSSA):
        return map_converted_pairs(scaled, rounding, floor_inf_value)
    return cutlass.Float32(
        convert_e8m0_pair(
            scaled.ir_value(),
            cutlass.Float32(0.0).ir_value(),
            rounding,
            floor_inf_value,
        )[0]
    )
