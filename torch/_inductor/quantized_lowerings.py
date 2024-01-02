import torch


def register_quantized_ops():
    from . import lowering

    quantized = torch.ops.quantized

    lowering.add_needs_realized_inputs(
        [
            quantized.max_pool2d,
        ]
    )

    lowering.make_fallback(quantized.max_pool2d)
