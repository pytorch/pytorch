import operator

import torch
import torch.nn as nn
import torch.nn.functional as F
toq = torch.ops.quantized

import torch.nn.quantized as nnq
import torch.nn.quantized.dynamic as nnqd
import torch.nn.qat as nnqat
import torch.nn.intrinsic.quantized as nniq
import torch.nn.intrinsic.qat as nniqat
import torch.nn.intrinsic as nni

from torch.fx import GraphModule
from torch.fx.graph import Node

from .utils import getattr_from_fqn
from torch.quantization.fx.pattern_utils import get_default_quant_patterns

from typing import Dict, Tuple, Set, Callable, Any, Union

def get_base_name_to_sets_of_related_ops() -> Dict[str, Set[Callable]]:
    base_name_to_sets_of_related_ops: Dict[str, Set[Callable]] = {
        # conv modules
        'torch.nn.Conv1d': set([
            nn.Conv1d,
            nnq.Conv1d,
            nniqat.ConvBn1d,
            nniqat.ConvBnReLU1d,
            nniq.ConvReLU1d,
            nni.ConvReLU1d,
        ]),
        'torch.nn.Conv2d': set([
            nn.Conv2d,
            nnq.Conv2d,
            nnqat.Conv2d,
            nniqat.ConvBn2d,
            nniqat.ConvBnReLU2d,
            nniqat.ConvReLU2d,
            nniq.ConvReLU2d,
            nni.ConvReLU2d,
        ]),
        'torch.nn.Conv3d': set([
            nn.Conv3d,
            nnq.Conv3d,
            nnqat.Conv3d,
            nniqat.ConvBn3d,
            nniqat.ConvBnReLU3d,
            nniqat.ConvReLU3d,
            nniq.ConvReLU3d,
            nni.ConvReLU3d,
        ]),
        # conv functionals
        'torch.nn.functional.conv1d': set([
            F.conv1d,
            toq.conv1d,
            toq.conv1d_relu,
        ]),
        'torch.nn.functional.conv2d': set([
            F.conv2d,
            toq.conv2d,
            toq.conv2d_relu,
        ]),
        'torch.nn.functional.conv3d': set([
            F.conv3d,
            toq.conv3d,
            toq.conv3d_relu,
        ]),
        # linear modules
        'torch.nn.Linear': set([
            nn.Linear,
            nnq.Linear,
            nni.LinearReLU,
            nniq.LinearReLU,
            nnqat.Linear,
            nnqd.Linear,
            nniqat.LinearReLU,
            nn.modules.linear._LinearWithBias,
        ]),
        # linear functionals
        'torch.nn.functional.linear': set([
            F.linear,
            toq.linear,
            toq.linear_relu,
        ]),
        # LSTM
        'torch.nn.LSTM': set([
            nn.LSTM,
            nnqd.LSTM,
        ]),
        # add
        'torch.add': set([
            torch.add,
            toq.add,
            operator.add,  # x + y
        ]),
        # cat
        'torch.cat': set([
            torch.cat,
            toq.cat,
        ]),
        # mul
        'torch.mul': set([
            torch.mul,
            toq.mul,
            operator.mul,
        ]),
        # relu
        'torch.relu': set([
            F.relu,
        ]),
        # maxpool2d
        'torch.nn.MaxPool2d': set([
            nn.MaxPool2d,
        ]),
        # sigmoid
        'torch.sigmoid': set([
            torch.sigmoid,
        ]),
        # BatchNorm
        'torch.nn.BatchNorm2d': set([
            nn.BatchNorm2d,
            nnq.BatchNorm2d,
        ]),
        'torch.nn.BatchNorm3d': set([
            nn.BatchNorm3d,
            nnq.BatchNorm3d,
        ]),
        # ConvTranspose
        'torch.nn.ConvTranspose1d': set([
            nn.ConvTranspose1d,
            nnq.ConvTranspose1d,
        ]),
        'torch.nn.ConvTranspose2d': set([
            nn.ConvTranspose2d,
            nnq.ConvTranspose2d,
        ]),
        # ELU
        'torch.nn.ELU': set([
            nn.ELU,
            nnq.ELU,
        ]),
        # Embedding
        'torch.nn.Embedding': set([
            nn.Embedding,
            nnq.Embedding,
        ]),
        # EmbeddingBag
        'torch.nn.EmbeddingBag': set([
            nn.EmbeddingBag,
            nnq.EmbeddingBag,
        ]),
        # GroupNorm
        'torch.nn.GroupNorm': set([
            nn.GroupNorm,
            nnq.GroupNorm,
        ]),
        # Hardswish
        'torch.nn.Hardswish': set([
            nn.Hardswish,
            nnq.Hardswish,
        ]),
        # InstanceNorm
        'torch.nn.InstanceNorm1d': set([
            nn.InstanceNorm1d,
            nnq.InstanceNorm1d,
        ]),
        'torch.nn.InstanceNorm2d': set([
            nn.InstanceNorm2d,
            nnq.InstanceNorm2d,
        ]),
        'torch.nn.InstanceNorm3d': set([
            nn.InstanceNorm3d,
            nnq.InstanceNorm3d,
        ]),
        # LayerNorm
        'torch.nn.LayerNorm': set([
            nn.LayerNorm,
            nnq.LayerNorm,
        ]),
        # LeakyReLU
        'torch.nn.LeakyReLU': set([
            nn.LeakyReLU,
            nnq.LeakyReLU,
        ]),
        # ReLU6
        'torch.nn.ReLU6': set([
            nn.ReLU6,
            nnq.ReLU6,
        ]),
        # BNReLU2d
        'torch.nn.intrinsic.BNReLU2d': set([
            nni.BNReLU2d,
            nniq.BNReLU2d,
        ]),
        'torch.nn.intrinsic.BNReLU3d': set([
            nni.BNReLU3d,
            nniq.BNReLU3d,
        ]),
        # F.elu
        'torch.nn.functional.elu': set([
            F.elu,
            toq.elu,
        ]),
        # F.hardswish
        'torch.nn.functional.hardswish': set([
            F.hardswish,
            toq.hardswish,
        ]),
        # F.instance_norm
        'torch.nn.functional.instance_norm': set([
            F.instance_norm,
            toq.instance_norm,
        ]),
        # F.layer_norm
        'torch.nn.functional.layer_norm': set([
            F.layer_norm,
            toq.layer_norm,
        ]),
        # F.leaky_relu
        'torch.nn.functional.leaky_relu': set([
            F.leaky_relu,
            toq.leaky_relu,
        ]),
    }
    return base_name_to_sets_of_related_ops


def get_type_a_related_to_b(
    base_name_to_sets_of_related_ops: Dict[str, Set[Callable]],
) -> Set[Tuple[Callable, Callable]]:
    # TODO(future PR): allow customizations
    # TODO(future PR): reuse existing quantization mappings
    # TODO(future PR): add the rest of modules and ops here
    type_a_related_to_b: Set[Tuple[Callable, Callable]] = set()

    for base_name, s in base_name_to_sets_of_related_ops.items():
        s_list = list(s)
        # add every bidirectional pair
        for idx_0 in range(0, len(s_list) - 1):
            for idx_1 in range(idx_0 + 1, len(s_list)):
                type_a_related_to_b.add((s_list[idx_0], s_list[idx_1]))
                type_a_related_to_b.add((s_list[idx_1], s_list[idx_0]))

    return type_a_related_to_b


NSFusionElType = Union[
    Callable,  # call_function or call_module type, example: F.linear or nn.Conv2d
    str,  # call_method name, example: "dequantize"
    Tuple[str, Any],  # call_method name and first argument, example: ("to", torch.float16)
]
NSFusionType = Union[
    Tuple[NSFusionElType, NSFusionElType],
    Tuple[NSFusionElType, NSFusionElType, NSFusionElType, NSFusionElType],
]

def get_reversed_fusions() -> Set[Tuple[NSFusionType, int]]:
    """
    Set of potential fusions, in reverse order.  The order is reversed
    to match how fusion patterns are defined in quantization code.

    Fusion format:
    ((fusion_op_0, fusion_op_1), base_op_idx)

    Where base_op_idx is the idx of the op we should use to match other related
    ops. Note: base_op_idx is specified in non-reverse order, i.e. a base_op_idx
    of 0 represents the first op in regular (non-reverse) order, 1 represents the
    second op, etc.
    """
    results: Set[Tuple[NSFusionType, int]] = set([])

    # Possible syntaxes:
    # * single op: torch.nn.Conv2d
    # * multiple ops: (torch.nn.ReLU, torch.nn.Conv2d)
    # For fusions, we only care about patterns composed of multiple ops.
    # TODO(future PR): allow customizations from default patterns.
    all_quant_patterns = get_default_quant_patterns()
    default_base_op_idx = 0
    for quant_pattern, _quant_handler in all_quant_patterns.items():
        # this only takes patterns of multiple ops
        if isinstance(quant_pattern, tuple):
            results.add((quant_pattern, default_base_op_idx))  # type: ignore

    # After this point, results countains values such as
    # [..., ((torch.nn.Relu, torch.nn.Conv2d), 0), ...]

    # Patterns for matching fp16 emulation are not specified in the quantization
    # fusion mappings.  For now, define them here.
    fp16_em_base_op_idx = 1
    patterns_to_add = [
        # linear-relu fp16 emulation:
        # fp16_to_fp32 -> linear -> relu -> fp32_to_fp16
        ((("to", torch.float16), F.relu, F.linear, "dequantize"), fp16_em_base_op_idx,),
    ]
    for p in patterns_to_add:
        results.add(p)

    return results


def end_node_matches_reversed_fusion(
    end_node: Node,
    reversed_fusion: NSFusionType,
    gm: GraphModule,
) -> bool:
    """
    Returns true if a pattern ending with `end_node` matches
    the fusion pattern.
    """
    cur_node = end_node
    for fusion_idx in range(len(reversed_fusion)):
        cur_fusion_el = reversed_fusion[fusion_idx]

        if cur_node.op == 'call_function':
            fusion_el_is_fun = (not isinstance(cur_fusion_el, str)) and \
                (not isinstance(cur_fusion_el, type))
            if fusion_el_is_fun:
                if cur_node.target != cur_fusion_el:
                    return False
                if len(cur_node.args) > 0 and isinstance(cur_node.args[0], Node):
                    cur_node = cur_node.args[0]
                else:
                    return False
            else:
                return False

        elif cur_node.op == 'call_module':
            fusion_el_is_mod = isinstance(cur_fusion_el, type)
            if fusion_el_is_mod:
                assert isinstance(cur_node.target, str)
                target_mod = getattr_from_fqn(gm, cur_node.target)
                if not isinstance(cur_fusion_el, type):
                    return False
                if not isinstance(target_mod, cur_fusion_el):
                    return False
                if len(cur_node.args) > 0 and isinstance(cur_node.args[0], Node):
                    cur_node = cur_node.args[0]
                else:
                    return False
            else:
                return False

        elif cur_node.op == 'call_method':
            fusion_el_is_meth_with_second_arg = \
                isinstance(cur_fusion_el, tuple) and len(cur_fusion_el) == 2
            fusion_el_is_meth_without_args = isinstance(cur_fusion_el, str)
            if fusion_el_is_meth_without_args or fusion_el_is_meth_with_second_arg:
                if fusion_el_is_meth_without_args:
                    if cur_node.target != cur_fusion_el:
                        return False
                else:
                    assert isinstance(cur_fusion_el, tuple)
                    if cur_node.target != cur_fusion_el[0]:
                        return False
                    elif len(cur_node.args) < 2:
                        return False
                    elif cur_node.args[1] != cur_fusion_el[1]:
                        return False

                if len(cur_node.args) > 0 and isinstance(cur_node.args[0], Node):
                    cur_node = cur_node.args[0]
                else:
                    return False
            else:
                return False
        else:
            return False

    return True
