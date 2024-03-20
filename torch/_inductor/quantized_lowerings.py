import torch
from . import lowering

quantized = torch.ops.quantized
aten = torch.ops.aten


def register_quantized_ops():
    lowering.add_needs_realized_inputs(
        [
            quantized.max_pool2d,
        ]
    )

    lowering.make_fallback(quantized.max_pool2d)


def register_woq_mm_ops():
    lowering.add_needs_realized_inputs(
        [
            aten._weight_int8pack_mm,
        ]
    )

    lowering.make_fallback(aten._weight_int8pack_mm)
