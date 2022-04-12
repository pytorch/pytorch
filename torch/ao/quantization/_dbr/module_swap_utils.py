from typing import Dict, Callable, Any, Optional

import torch

from torch.nn.intrinsic import _FusedModule
from ..utils import (
    activation_is_int8_quantized,
    activation_is_int32_quantized,
    op_is_int8_dynamically_quantized,
)
from torch.ao.quantization import swap_module
from torch.ao.quantization.quantization_mappings import (
    DEFAULT_REFERENCE_STATIC_QUANT_MODULE_MAPPINGS,
)

def _swap_child_modules(
    module: torch.nn.Module,
    static_mappings: Dict[Callable, Any],
    dynamic_mappings: Dict[Callable, Any],
    parent_fqn: Optional[str] = None,
) -> None:
    """
    For each direct child of `module`, swaps it using `static_mappings`
    if the qconfig for that child is using int8 static quantization,
    and the module type is in the mapping.

    Recursively calls itself on each child.
    """

    qstate = getattr(module, '_auto_quant_state', None)

    reassign = {}
    for local_fqn, mod in module.named_children():
        if parent_fqn is None:
            global_fqn = local_fqn
        else:
            global_fqn = f"{parent_fqn}.{local_fqn}"
        # both fused modules and observed custom modules are
        # swapped as one unit
        if not isinstance(mod, _FusedModule):
            _swap_child_modules(
                mod, static_mappings, dynamic_mappings, global_fqn)

        qconfig = getattr(mod, 'qconfig', None)
        if not qconfig:
            continue
        activation_int8_quantized = activation_is_int8_quantized(qconfig)
        op_int8_dynamically_quantized = op_is_int8_dynamically_quantized(qconfig)
        activation_int32_quantized = activation_is_int32_quantized(qconfig)

        # Get the output observer from qstate and attach it to the module,
        # to match the API for Eager mode module swaps
        if qstate is not None:
            output_obs = qstate.get_output_observer_from_fqn(global_fqn)
            if output_obs is not None:
                mod.activation_post_process = output_obs

        if activation_int8_quantized:
            if not type(mod) in static_mappings:
                continue
            reassign[local_fqn] = swap_module(mod, static_mappings, {})
        elif op_int8_dynamically_quantized:
            if not type(mod) in dynamic_mappings:
                continue
            reassign[local_fqn] = swap_module(mod, dynamic_mappings, {})
        elif activation_int32_quantized:
            # For now, only apply reference logic to modules quantized to
            # int32. Do it automatically.
            # TODO(future PR): extend this logic to more dtypes, and add
            # the is_reference API flag instead of doing this automatically.
            # Note: swap modules only does the swap if the mapping for this
            # module exists.
            reassign[local_fqn] = swap_module(
                mod, DEFAULT_REFERENCE_STATIC_QUANT_MODULE_MAPPINGS, {})

        # TODO(future PR): add support for other dtypes

    for key, value in reassign.items():
        module._modules[key] = value
