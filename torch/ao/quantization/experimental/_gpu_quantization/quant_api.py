"""
Quantization API stuff which is not specific to SmoothQuant

Note: this is throwaway code for fast results on Blueberry, this is not
intended to be the actual long term quantization API for server GPUs.
"""

import torch
from torch.ao.quantization.experimental._gpu_quantization.dynamic_quant import (
    DynamicallyPerAxisQuantizedLinear,
)
from torch.ao.quantization.experimental._gpu_quantization.subclass import (
    DynamicallyQuantizedLinearWeight,
)
from torch.ao.quantization.experimental._gpu_quantization.weight_only import (
    WeightOnlyInt8QuantLinear,
)

__all__ = [
    "replace_with_custom_fn_if_matches_filter",
    "apply_weight_only_int8_quant",
    "apply_dynamic_quant",
    "change_linear_weights_to_dqtensors",
]


def replace_with_custom_fn_if_matches_filter(
    model, replacement_fn, filter_fn, cur_fqn=""
) -> None:
    """
    For each `child` in `model`, replaces it with `replacement_fn(child)`
    if `filter_fn(child)` is `True`
    """
    name_to_child = dict(model.named_children())
    for name, child in name_to_child.items():
        if cur_fqn == "":
            new_fqn = name
        else:
            new_fqn = f"{cur_fqn}.{name}"
        if filter_fn(child, new_fqn):
            new_child = replacement_fn(child)
            setattr(model, name, new_child)
        else:
            replace_with_custom_fn_if_matches_filter(
                child, replacement_fn, filter_fn, new_fqn
            )


def apply_weight_only_int8_quant(model):
    replace_with_custom_fn_if_matches_filter(
        model,
        WeightOnlyInt8QuantLinear.from_float,
        lambda mod, fqn: isinstance(mod, torch.nn.Linear),
    )


def apply_dynamic_quant(model, use_fused_int_mm=0):
    replace_with_custom_fn_if_matches_filter(
        model,
        lambda mod: DynamicallyPerAxisQuantizedLinear.from_float(mod, use_fused_int_mm),
        lambda mod, fqn: isinstance(mod, torch.nn.Linear),
    )


def change_linear_weights_to_dqtensors(model):
    def insert_subclass(lin):
        lin.weight = torch.nn.Parameter(
            DynamicallyQuantizedLinearWeight.from_float(lin.weight), requires_grad=False
        )
        return lin

    replace_with_custom_fn_if_matches_filter(
        model, insert_subclass, lambda mod, fqn: isinstance(mod, torch.nn.Linear)
    )
