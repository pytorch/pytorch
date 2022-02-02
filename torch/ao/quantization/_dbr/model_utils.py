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
from typing import Optional

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
                assert seen_op_info.packable_tensor_idx_to_name[1] is not None
                weight = getattr(module, seen_op_info.packable_tensor_idx_to_name[1])
                assert seen_op_info.packable_tensor_idx_to_name[2] is not None
                bias = getattr(module, seen_op_info.packable_tensor_idx_to_name[2])
                stride = seen_op_info.packable_nontensor_idx_to_arg[3]
                padding = seen_op_info.packable_nontensor_idx_to_arg[4]
                dilation = seen_op_info.packable_nontensor_idx_to_arg[5]
                groups = seen_op_info.packable_nontensor_idx_to_arg[6]

                # quantize the weight
                # TODO: create weight observers from qconfig.weight
                assert seen_op_info.input_tensor_infos[1] is not None
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
                def get_tensor_param_name(idx: int, name: str) -> Optional[str]:
                    param_name = seen_op_info.packable_tensor_idx_to_name.get(idx, None)
                    if param_name is not None:
                        return param_name
                    return seen_op_info.packable_tensor_kwarg_name_to_name.get(name, None)

                weight_name = get_tensor_param_name(1, 'weight')
                assert weight_name is not None
                weight = getattr(module, weight_name)

                bias_name = get_tensor_param_name(2, 'bias')
                bias = getattr(module, bias_name) if bias_name is not None else None

                # quantize the weight
                # TODO: create weight observers from qconfig.weight
                assert seen_op_info.input_tensor_infos[1] is not None
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
            activation_int8_quantized = \
                observer.dtype in [torch.quint8, torch.qint8]
            if activation_int8_quantized:
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

def attach_output_convert_info_to_model(
    module: torch.nn.Module,
) -> None:
    """
    Calculates the info needed to perform the module outputs hook
    and attaches it to the parent module. This is done to avoid recalculating
    these values at inference.
    """
    if hasattr(module, '_auto_quant_state'):
        qstate: AutoQuantizationState = module._auto_quant_state  # type: ignore[assignment]
        qstate.set_needs_dtype_transform_on_outputs()

    for _, child in module.named_children():
        attach_output_convert_info_to_model(child)
