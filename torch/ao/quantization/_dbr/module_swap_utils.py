from typing import Dict, Callable, Any

import torch

from torch.nn.intrinsic import _FusedModule
from ..utils import (
    activation_is_int8_quantized,
    op_is_int8_dynamically_quantized,
)
from torch.ao.quantization import swap_module

def _swap_child_modules(
    module: torch.nn.Module,
    static_mappings: Dict[Callable, Any],
    dynamic_mappings: Dict[Callable, Any],
) -> None:
    """
    For each direct child of `module`, swaps it using `static_mappings`
    if the qconfig for that child is using int8 static quantization,
    and the module type is in the mapping.

    Recursively calls itself on each child.
    """

    reassign = {}
    for name, mod in module.named_children():
        # both fused modules and observed custom modules are
        # swapped as one unit
        if not isinstance(mod, _FusedModule):
            _swap_child_modules(mod, static_mappings, dynamic_mappings)

        qconfig = getattr(mod, 'qconfig', None)
        if not qconfig:
            continue
        activation_int8_quantized = activation_is_int8_quantized(qconfig)
        op_int8_dynamically_quantized = op_is_int8_dynamically_quantized(qconfig)
        if activation_int8_quantized:
            if not type(mod) in static_mappings:
                continue
            reassign[name] = swap_module(mod, static_mappings, {})
        elif op_int8_dynamically_quantized:
            if not type(mod) in dynamic_mappings:
                continue
            reassign[name] = swap_module(mod, dynamic_mappings, {})
        # TODO(future PR): add support for other dtypes

    for key, value in reassign.items():
        module._modules[key] = value
