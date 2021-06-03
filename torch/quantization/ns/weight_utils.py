import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.quantized.dynamic as nnqd
import torch.nn.intrinsic as nni
toq = torch.ops.quantized
from torch.fx import GraphModule
from torch.fx.graph import Node

from .utils import getattr_from_fqn, return_first_non_observer_node

from .ns_types import (
    NSSingleResultValuesType,
    NSSingleResultType,
    NSNodeTargetType,
)

from typing import List, Optional, Set, Tuple

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
    # TODO(future PR): docblock
    # TODO(future PR): handle non standard weights (i.e. after reshape, etc)
    if node.target in (F.conv1d, F.conv2d, F.conv3d):
        # traverse backwards from the weight arg, accounting for any observers
        weight_arg_node = node.args[1]
        assert isinstance(weight_arg_node, Node)
        weight_node = return_first_non_observer_node(weight_arg_node, gm)
        assert isinstance(weight_node, Node)
        assert weight_node.op == 'get_attr'
        weight = getattr_from_fqn(gm, weight_node.target)  # type: ignore[arg-type]
        return weight.detach()
    else:
        assert node.target in (
            toq.conv1d, toq.conv2d, toq.conv3d, toq.conv1d_relu,
            toq.conv2d_relu, toq.conv3d_relu)
        # qconv state is arg 1
        qconv_state_node = node.args[1]
        assert isinstance(qconv_state_node, Node)
        assert qconv_state_node.op == 'get_attr'
        qconv_state_obj = getattr_from_fqn(gm, qconv_state_node.target)  # type: ignore[arg-type]
        return qconv_state_obj.weight()

def get_linear_fun_weight(node: Node, gm: GraphModule) -> torch.Tensor:
    # TODO(future PR): better docblock, with example FX IR
    if node.target in (F.linear,):
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
        else:
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
        assert node.target in (toq.linear, toq.linear_relu)
        # packed weight is arg 1
        packed_weight_node = node.args[1]
        assert isinstance(packed_weight_node, Node)
        assert packed_weight_node.op == 'get_attr'
        packed_weight = getattr_from_fqn(gm, packed_weight_node.target)  # type: ignore[arg-type]
        # TODO(future PR): why does packed_weight.unpack() not work?
        # TODO(future PR): discuss if we even need to unpack, or if the
        #   caller can handle the unpacking
        (weight, _bias), _name = packed_weight.__getstate__()
        return weight

def extract_weight_from_node(
    node: Node,
    gm: GraphModule,
    type_a_related_to_b: Set[Tuple[NSNodeTargetType, NSNodeTargetType]],
) -> Optional[NSSingleResultType]:
    res_type = NSSingleResultValuesType.WEIGHT.value
    if node.op == 'call_function':

        related_to_linear = node.target in (F.linear,) or \
            (node.target, F.linear) in type_a_related_to_b
        related_to_conv1d = node.target in (F.conv1d,) or \
            (node.target, F.conv1d) in type_a_related_to_b
        related_to_conv2d = node.target in (F.conv2d,) or \
            (node.target, F.conv2d) in type_a_related_to_b
        related_to_conv3d = node.target in (F.conv3d,) or \
            (node.target, F.conv3d) in type_a_related_to_b

        if related_to_linear:
            weight = get_linear_fun_weight(node, gm)
            return {
                'type': res_type,
                'values': [weight],
                'prev_node_name': node.name,
                'prev_node_target_type': str(node.target),
                'ref_node_name': node.name,
                'index_within_arg': 0,
                'index_of_arg': 0,
            }
        elif (related_to_conv1d or related_to_conv2d or related_to_conv3d):
            weight = get_conv_fun_weight(node, gm)
            return {
                'type': res_type,
                'values': [weight],
                'prev_node_name': node.name,
                'prev_node_target_type': str(node.target),
                'ref_node_name': node.name,
                'index_within_arg': 0,
                'index_of_arg': 0,
            }

    elif node.op == 'call_module':
        # for call_module, we need to look up the modules to do the type check
        assert isinstance(node.target, str)
        mod = getattr_from_fqn(gm, node.target)

        # check that A is one the modules we need
        # assume B is related (this is done by graph matcher)
        related_to_conv1d_mod = isinstance(mod, nn.Conv1d) or \
            (type(mod), nn.Conv1d) in type_a_related_to_b
        related_to_conv2d_mod = isinstance(mod, nn.Conv2d) or \
            (type(mod), nn.Conv2d) in type_a_related_to_b
        related_to_conv3d_mod = isinstance(mod, nn.Conv3d) or \
            (type(mod), nn.Conv3d) in type_a_related_to_b
        related_to_linear_mod = isinstance(mod, nn.Linear) or \
            (type(mod), nn.Linear) in type_a_related_to_b
        related_to_lstm_mod = isinstance(mod, nn.LSTM) or \
            (type(mod), nn.LSTM) in type_a_related_to_b

        if related_to_conv1d_mod or related_to_conv2d_mod or related_to_conv3d_mod:
            weights = [get_conv_mod_weight(mod)]
            return {
                'type': res_type,
                'values': weights,
                'prev_node_name': node.name,
                'prev_node_target_type': str(type(mod)),
                'ref_node_name': node.name,
                'index_within_arg': 0,
                'index_of_arg': 0,
            }
        elif related_to_lstm_mod:
            weights = get_lstm_mod_weights(mod)
            return {
                'type': res_type,
                'values': weights,
                'prev_node_name': node.name,
                'prev_node_target_type': str(type(mod)),
                'ref_node_name': node.name,
                'index_within_arg': 0,
                'index_of_arg': 0,
            }
        elif related_to_linear_mod:
            weights = [get_linear_mod_weight(mod)]
            return {
                'type': res_type,
                'values': weights,
                'prev_node_name': node.name,
                'prev_node_target_type': str(type(mod)),
                'ref_node_name': node.name,
                'index_within_arg': 0,
                'index_of_arg': 0,
            }

    return None
