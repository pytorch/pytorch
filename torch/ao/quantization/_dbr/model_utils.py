"""
Contains model level utilities which can be aware of the AutoQuantizationState
type.
"""

import torch
import torch.nn.functional as F
toq = torch.ops.quantized
from .quantization_state import AutoQuantizationState
from torch.quantization import (
    ObserverBase,
    FakeQuantizeBase,
)

def pack_weights_for_functionals(
    module: torch.nn.Module,
) -> None:
    """
    Packs weights for functionals seen while tracing.
    Note: weight packing for modules is handled by eager mode quantization
    flow.
    """
    if hasattr(module, '_auto_quant_state'):
        qstate: AutoQuantizationState = module._auto_quant_state  # type: ignore[assignment]
        # find any ops which need packing
        for idx, seen_op_info in qstate.idx_to_seen_op_infos.items():
            packable_args_len = len(seen_op_info.packable_tensor_idx_to_name) + \
                len(seen_op_info.packable_nontensor_idx_to_arg)
            if packable_args_len == 0:
                continue

            if seen_op_info.type == F.conv2d:
                # fetch all the info needed for packed params
                weight = getattr(module, seen_op_info.packable_tensor_idx_to_name[1])
                bias = getattr(module, seen_op_info.packable_tensor_idx_to_name[2])
                stride = seen_op_info.packable_nontensor_idx_to_arg[3]
                padding = seen_op_info.packable_nontensor_idx_to_arg[4]
                dilation = seen_op_info.packable_nontensor_idx_to_arg[5]
                groups = seen_op_info.packable_nontensor_idx_to_arg[6]

                # quantize the weight
                # TODO: create weight observers from qconfig.weight
                weight_tensor_id = seen_op_info.input_tensor_infos[1].id
                weight_obs = qstate.tensor_id_to_observer[str(weight_tensor_id)]
                assert isinstance(weight_obs, (ObserverBase, FakeQuantizeBase))
                scale, zp = weight_obs.calculate_qparams()
                qweight = torch.quantize_per_tensor(weight, scale, zp, torch.qint8)

                # create the packed params
                packed_params = toq.conv2d_prepack(
                    qweight, bias, stride, padding, dilation, groups)

                # attach to module
                name_idx = 0
                prefix = "_packed_params_"
                name_candidate = f"{prefix}{name_idx}"
                while hasattr(module, name_candidate):
                    name_idx += 1
                    name_candidate = f"{prefix}{name_idx}"
                setattr(module, name_candidate, packed_params)
                qstate.idx_to_packed_weight_name[idx] = name_candidate
                # TODO: delete the original weights

            elif seen_op_info.type == F.linear:
                # fetch all the info needed for packed params
                weight = getattr(module, seen_op_info.packable_tensor_idx_to_name[1])
                bias = getattr(module, seen_op_info.packable_tensor_kwarg_name_to_name['bias'])

                # quantize the weight
                # TODO: create weight observers from qconfig.weight
                weight_tensor_id = seen_op_info.input_tensor_infos[1].id
                weight_obs = qstate.tensor_id_to_observer[str(weight_tensor_id)]
                assert isinstance(weight_obs, (ObserverBase, FakeQuantizeBase))
                scale, zp = weight_obs.calculate_qparams()
                qweight = torch.quantize_per_tensor(weight, scale, zp, torch.qint8)

                # create the packed params
                packed_params = toq.linear_prepack(qweight, bias)

                # attach to module
                name_idx = 0
                prefix = "_packed_params_"
                name_candidate = f"{prefix}{name_idx}"
                while hasattr(module, name_candidate):
                    name_idx += 1
                    name_candidate = f"{prefix}{name_idx}"
                setattr(module, name_candidate, packed_params)
                qstate.idx_to_packed_weight_name[idx] = name_candidate
                # TODO: delete the original weights

    for _, child in module.named_children():
        pack_weights_for_functionals(child)

def attach_scale_zp_values_to_model(
    module: torch.nn.Module,
) -> None:
    """
    Calculates the scale and zero_point from each observer and attaches
    these values to the parent module. This is done to avoid recalculating
    these values at inference.
    """
    if hasattr(module, '_auto_quant_state'):
        qstate: AutoQuantizationState = module._auto_quant_state  # type: ignore[assignment]
        for tensor_id, observer in qstate.tensor_id_to_observer.items():
            scale, zp = observer.calculate_qparams()
            # tensor_id_to_observer is a ModuleDict which has to have string keys
            # tensor_id_to_scale_zp is a normal dict which can have int keys
            qstate.tensor_id_to_scale_zp[int(tensor_id)] = (scale, zp)
        qstate.tensor_id_to_observer.clear()

    for _, child in module.named_children():
        attach_scale_zp_values_to_model(child)

def attach_op_convert_info_to_model(
    module: torch.nn.Module,
) -> None:
    """
    Calculates the info needed to convert each op and attaches
    it to the parent module. This is done to avoid recalculating these values
    at inference.
    """
    if hasattr(module, '_auto_quant_state'):
        qstate: AutoQuantizationState = module._auto_quant_state  # type: ignore[assignment]
        for _, seen_op_info in qstate.idx_to_seen_op_infos.items():
            qstate.idx_to_op_convert_info[seen_op_info.idx] = \
                qstate.calculate_op_convert_info(seen_op_info)

    for _, child in module.named_children():
        attach_op_convert_info_to_model(child)

def _populate_descendant_boolean_attr(
    module: torch.nn.Module,
    cur_mod_attr_name: str,
    cur_or_any_descendant_mod_attr_name: str,
) -> bool:
    """
    Assumes that `cur_mod_attr_name` is a boolean module attribute
    on `AutoQuantizationState`, and `cur_or_any_descendant_mod_attr_name`
    is the version of `cur_mod_attr_name` which applies to the current
    module and any of its descendants.

    Populates `cur_or_any_descendant_mod_attr_name` for `mod` and all of
    its children, based on the values of `cur_mod_attr_name`.
    """
    cur_value = False
    for k, v in module.named_children():
        child_value = _populate_descendant_boolean_attr(
            v, cur_mod_attr_name, cur_or_any_descendant_mod_attr_name)
        cur_value = cur_value or child_value

    if hasattr(module, '_auto_quant_state'):
        qstate: AutoQuantizationState = module._auto_quant_state  # type: ignore[assignment]
        cur_value = cur_value or getattr(qstate, cur_mod_attr_name)
        setattr(qstate, cur_or_any_descendant_mod_attr_name, cur_value)

    return cur_value

def attach_descendant_usage_info_to_model(
    module: torch.nn.Module,
) -> None:
    """
    Populates the `self_or_any_descendant_needs_dtype_transform_on_outputs`,
    `any_descendant_needs_arg_dequants` and `any_descendant_needs_op_hooks`
    flags on the model.
    """
    _populate_descendant_boolean_attr(
        module, 'needs_dtype_transform_on_outputs',
        'self_or_any_descendant_needs_dtype_transform_on_outputs')
    _populate_descendant_boolean_attr(
        module, 'any_child_needs_arg_dequants',
        'any_descendant_needs_arg_dequants')
    _populate_descendant_boolean_attr(
        module, 'any_child_needs_op_hooks',
        'any_descendant_needs_op_hooks')
