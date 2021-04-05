import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.quantized.dynamic as nnqd
import torch.nn.intrinsic as nni
toq = torch.ops.quantized
from torch.fx import GraphModule
from torch.fx.graph import Node

from .utils import getattr_from_fqn

from typing import List

def get_conv_mod_weight(mod: nn.Module) -> torch.Tensor:
    # TODO(future PR): handle QAT variants
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
        return mod._weight_bias()[0]  # type: ignore

def get_linear_mod_weight(mod: nn.Module) -> torch.Tensor:
    # TODO(future PR): make more generic, handle everything
    if isinstance(mod, nn.Linear):
        return mod.weight.detach()
    else:
        return mod._weight_bias()[0]  # type: ignore

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
            weight = getattr_from_fqn(gm, weight_node.target)  # type: ignore
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
            weight = getattr_from_fqn(gm, weight_node.target)  # type: ignore
            # return the weight with fp16 cast
            return weight.detach().to(target_dtype)

    else:
        assert node.target in (toq.linear, toq.linear_relu)
        # packed weight is arg 1
        packed_weight_node = node.args[1]
        assert isinstance(packed_weight_node, Node)
        assert packed_weight_node.op == 'get_attr'
        packed_weight = getattr_from_fqn(gm, packed_weight_node.target)  # type: ignore
        # TODO(future PR): why does packed_weight.unpack() not work?
        # TODO(future PR): discuss if we even need to unpack, or if the
        #   caller can handle the unpacking
        (weight, _bias), _name = packed_weight.__getstate__()
        return weight
