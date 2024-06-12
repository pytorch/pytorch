# mypy: allow-untyped-defs
import torch
from . import lowering

quantized = torch.ops.quantized
_quantized = torch.ops._quantized
aten = torch.ops.aten


def register_quantized_ops():
    lowering.add_needs_realized_inputs(
        [
            quantized.max_pool2d,
            _quantized.wrapped_fbgemm_pack_gemm_matrix_fp16,
            _quantized.wrapped_fbgemm_linear_fp16_weight,
        ]
    )

    lowering.make_fallback(quantized.max_pool2d)
    lowering.make_fallback(_quantized.wrapped_fbgemm_pack_gemm_matrix_fp16)
    lowering.make_fallback(_quantized.wrapped_fbgemm_linear_fp16_weight)


def register_woq_mm_ops():
    lowering.add_needs_realized_inputs(
        [
            aten._weight_int8pack_mm,
        ]
    )

    lowering.make_fallback(aten._weight_int8pack_mm)
