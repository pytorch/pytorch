import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.ao.nn.quantized.dynamic as nnqd
import torch.ao.nn.quantized as nnq
import torch.nn.intrinsic.qat as nniqat
import torch.ao.nn.qat as nnqat
import torch.nn.intrinsic as nni
import torch.nn.intrinsic.quantized as nniq
toq = torch.ops.quantized
from torch.fx import GraphModule
from torch.fx.graph import Node

from .utils import (
    get_target_type_str,
    getattr_from_fqn,
    return_first_non_observer_node,
)

from .ns_types import (
    NSSingleResultValuesType,
    NSSingleResultType,
)

from typing import List, Optional, Dict, Callable

def mod_weight_detach(mod: nn.Module) -> torch.Tensor:
    return mod.weight.detach()  # type: ignore[operator]

def mod_0_weight_detach(mod: nn.Module) -> torch.Tensor:
    return mod[0].weight.detach()  # type: ignore[index]

def mod_weight_bias_0(mod: nn.Module) -> torch.Tensor:
    return mod._weight_bias()[0]  # type: ignore[operator]

def get_lstm_weight(mod: nn.Module) -> List[torch.Tensor]:
    res = []
    for idx, param_name in enumerate(mod._flat_weights_names):  # type: ignore[arg-type]
        if 'weight_ih_l' in param_name or 'weight_hh_l' in param_name:
            param_value = mod._flat_weights[idx].detach()  # type: ignore[index]
            res.append(param_value)
    return res

def get_qlstm_weight(mod: nn.Module) -> List[torch.Tensor]:
    res = []
    for weight_value in mod._all_weight_values:  # type: ignore[union-attr]
        res.append(weight_value.param.__getstate__()[0][4][0].__getstate__()[0][0])
        res.append(weight_value.param.__getstate__()[0][4][1].__getstate__()[0][0])
    return res

def get_conv_mod_weight(mod: nn.Module) -> torch.Tensor:
    if (
        isinstance(mod, nn.Conv1d) or
        isinstance(mod, nn.Conv2d) or
        isinstance(mod, nn.Conv3d)
    ):
        return mod.weight.detach()
    elif (
        isinstance(mod, nni.ConvReLU1d) or
        isinstance(mod, nni.ConvReLU2d) or
        isinstance(mod, nni.ConvReLU3d)
    ):
        return mod[0].weight.detach()
    else:
        return mod._weight_bias()[0]  # type: ignore[operator]

def get_linear_mod_weight(mod: nn.Module) -> torch.Tensor:
    if isinstance(mod, nn.Linear):
        return mod.weight.detach()
    elif isinstance(mod, nni.LinearReLU):
        return mod[0].weight.detach()
    else:
        return mod._weight_bias()[0]  # type: ignore[operator]

def get_lstm_mod_weights(mod: nn.Module) -> List[torch.Tensor]:
    # TODO(future PR): make more generic, handle everything
    if isinstance(mod, nn.LSTM):
        res = []
        for idx, param_name in enumerate(mod._flat_weights_names):
            if 'weight_ih_l' in param_name or 'weight_hh_l' in param_name:
                param_value = mod._flat_weights[idx].detach()
                res.append(param_value)
        return res
    else:
        assert isinstance(mod, nnqd.LSTM), f"type {type(res)} not handled yet"
        res = []
        for weight_value in mod._all_weight_values:
            res.append(weight_value.param.__getstate__()[0][4][0].__getstate__()[0][0])
            res.append(weight_value.param.__getstate__()[0][4][1].__getstate__()[0][0])
        return res

def get_conv_fun_weight(node: Node, gm: GraphModule) -> torch.Tensor:
    # traverse backwards from the weight arg, accounting for any observers
    weight_arg_node = node.args[1]
    assert isinstance(weight_arg_node, Node)
    weight_node = return_first_non_observer_node(weight_arg_node, gm)
    assert isinstance(weight_node, Node)
    assert weight_node.op == 'get_attr'
    weight = getattr_from_fqn(gm, weight_node.target)  # type: ignore[arg-type]
    return weight.detach()

def get_qconv_fun_weight(node: Node, gm: GraphModule) -> torch.Tensor:
    # qconv state is arg 1
    qconv_state_node = node.args[1]
    assert isinstance(qconv_state_node, Node)
    assert qconv_state_node.op == 'get_attr'
    qconv_state_obj = getattr_from_fqn(gm, qconv_state_node.target)  # type: ignore[arg-type]
    return qconv_state_obj.weight()

def get_linear_fun_weight(node: Node, gm: GraphModule) -> torch.Tensor:
    # traverse backwards from the weight arg, accounting for any observers
    # supported patterns:
    # weight -> obs -> linear
    # weight -> to(torch.float16) -> dequantize -> linear
    linear_second_arg = node.args[1]
    assert isinstance(linear_second_arg, Node)

    if linear_second_arg.op == 'call_module':
        # weight -> obs -> linear
        weight_arg_node = node.args[1]
        assert isinstance(weight_arg_node, Node)
        weight_node = weight_arg_node.args[0]
        assert isinstance(weight_node, Node)
        assert weight_node.op == 'get_attr'
        weight = getattr_from_fqn(gm, weight_node.target)  # type: ignore[arg-type]
        return weight.detach()
    elif linear_second_arg.op == 'call_method':
        # weight -> to(torch.float16) -> dequantize -> linear
        assert linear_second_arg.op == 'call_method'
        dequant_node = node.args[1]
        assert isinstance(dequant_node, Node)
        to_fp16_node = dequant_node.args[0]
        assert isinstance(to_fp16_node, Node)
        # extract the dtype, so we can cast to it before returning
        target_dtype = to_fp16_node.args[1]
        weight_node = to_fp16_node.args[0]
        assert isinstance(weight_node, Node)
        assert weight_node.op == 'get_attr'
        weight = getattr_from_fqn(gm, weight_node.target)  # type: ignore[arg-type]
        # return the weight with fp16 cast
        return weight.detach().to(target_dtype)
    else:
        assert linear_second_arg.op == 'get_attr'
        weight = getattr_from_fqn(gm, linear_second_arg.target)  # type: ignore[arg-type]
        return weight.detach()

def get_qlinear_fun_weight(node: Node, gm: GraphModule) -> torch.Tensor:
    # packed weight is arg 1
    packed_weight_node = node.args[1]
    assert isinstance(packed_weight_node, Node)
    assert packed_weight_node.op == 'get_attr'
    packed_weight = getattr_from_fqn(gm, packed_weight_node.target)  # type: ignore[arg-type]
    # TODO(future PR): why does packed_weight.unpack() not work?
    (weight, _bias), _name = packed_weight.__getstate__()
    return weight

def get_op_to_type_to_weight_extraction_fn() -> Dict[str, Dict[Callable, Callable]]:

    op_to_type_to_weight_extraction_fn: Dict[str, Dict[Callable, Callable]] = {
        'call_module': {
            # Conv1d
            nn.Conv1d: mod_weight_detach,
            nni.ConvReLU1d: mod_0_weight_detach,
            nnq.Conv1d: mod_weight_bias_0,
            nnqat.Conv1d: mod_weight_detach,
            nniqat.ConvBn1d: mod_weight_detach,
            nniqat.ConvBnReLU1d: mod_weight_detach,
            nniqat.ConvReLU1d: mod_weight_detach,
            nniq.ConvReLU1d: mod_weight_bias_0,
            # Conv2d
            nn.Conv2d: mod_weight_detach,
            nni.ConvReLU2d: mod_0_weight_detach,
            nnq.Conv2d: mod_weight_bias_0,
            nnqat.Conv2d: mod_weight_detach,
            nniqat.ConvBn2d: mod_weight_detach,
            nniqat.ConvBnReLU2d: mod_weight_detach,
            nniqat.ConvReLU2d: mod_weight_detach,
            nniq.ConvReLU2d: mod_weight_bias_0,
            # Conv3d
            nn.Conv3d: mod_weight_detach,
            nni.ConvReLU3d: mod_0_weight_detach,
            nnq.Conv3d: mod_weight_bias_0,
            nnqat.Conv3d: mod_weight_detach,
            nniqat.ConvBn3d: mod_weight_detach,
            nniqat.ConvBnReLU3d: mod_weight_detach,
            nniqat.ConvReLU3d: mod_weight_detach,
            nniq.ConvReLU3d: mod_weight_bias_0,
            # Linear
            nn.Linear: mod_weight_detach,
            nnq.Linear: mod_weight_bias_0,
            nni.LinearReLU: mod_0_weight_detach,
            nniq.LinearReLU: mod_weight_bias_0,
            nnqat.Linear: mod_weight_detach,
            nnqd.Linear: mod_weight_bias_0,
            nniqat.LinearReLU: mod_weight_detach,
            nniqat.LinearBn1d: mod_weight_detach,
            nn.modules.linear.NonDynamicallyQuantizableLinear: mod_weight_detach,
            # LSTM
            nn.LSTM: get_lstm_weight,
            nnqd.LSTM: get_qlstm_weight,
        },
        'call_function': {
            # Conv
            F.conv1d: get_conv_fun_weight,
            F.conv2d: get_conv_fun_weight,
            F.conv3d: get_conv_fun_weight,
            toq.conv1d: get_qconv_fun_weight,
            toq.conv2d: get_qconv_fun_weight,
            toq.conv3d: get_qconv_fun_weight,
            toq.conv1d_relu: get_qconv_fun_weight,
            toq.conv2d_relu: get_qconv_fun_weight,
            toq.conv3d_relu: get_qconv_fun_weight,
            # Linear
            F.linear: get_linear_fun_weight,
            toq.linear: get_qlinear_fun_weight,
            toq.linear_relu: get_qlinear_fun_weight,
        },
    }

    return op_to_type_to_weight_extraction_fn

def extract_weight_from_node(
    node: Node,
    gm: GraphModule,
    op_to_type_to_weight_extraction_fn: Optional[Dict[str, Dict[Callable, Callable]]] = None,
) -> Optional[NSSingleResultType]:
    res_type = NSSingleResultValuesType.WEIGHT.value

    # Not all graphmodules have _node_name_to_scope, so only fill it
    # out if it exists.
    fqn = None
    if hasattr(gm, '_node_name_to_scope'):
        fqn = gm._node_name_to_scope[node.name][0]  # type: ignore[index]

    if op_to_type_to_weight_extraction_fn is None:
        op_to_type_to_weight_extraction_fn = get_op_to_type_to_weight_extraction_fn()

    ref_node_type = get_target_type_str(node, gm)
    # for extracting weights, these are always the same
    prev_node_type = ref_node_type

    if node.op == 'call_function':
        function_mapping = op_to_type_to_weight_extraction_fn['call_function']
        for target_fn_type, weight_extraction_fn in function_mapping.items():
            if node.target == target_fn_type:
                weight = weight_extraction_fn(node, gm)
                return {
                    'type': res_type,
                    'values': [weight],
                    'prev_node_name': node.name,
                    'prev_node_target_type': prev_node_type,
                    'ref_node_name': node.name,
                    'ref_node_target_type': ref_node_type,
                    'index_within_arg': 0,
                    'index_of_arg': 0,
                    'fqn': fqn,
                }

    elif node.op == 'call_module':
        # for call_module, we need to look up the modules to do the type check
        assert isinstance(node.target, str)
        mod = getattr_from_fqn(gm, node.target)
        module_mapping = op_to_type_to_weight_extraction_fn['call_module']
        for target_mod_type, weight_extraction_fn in module_mapping.items():
            if type(mod) == target_mod_type:
                weight = weight_extraction_fn(mod)
                return {
                    'type': res_type,
                    'values': [weight],
                    'prev_node_name': node.name,
                    'prev_node_target_type': prev_node_type,
                    'ref_node_name': node.name,
                    'ref_node_target_type': ref_node_type,
                    'index_within_arg': 0,
                    'index_of_arg': 0,
                    'fqn': fqn,
                }

    return None
