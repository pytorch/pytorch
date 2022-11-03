import torch
import torch.nn as nn
import torch.nn.functional as F
toq = torch.ops.quantized

from torch.fx import GraphModule
from torch.fx.graph import Node

from torch.ao.quantization.utils import _getattr_from_fqn
from .ns_types import NSNodeTargetType
from torch.ao.quantization.fx.backend_config_utils import get_native_quant_patterns
from torch.ao.quantization import (
    ObserverBase,
    FakeQuantizeBase,
)

from typing import Dict, Tuple, Set, Callable, Any, Union, List


def get_type_a_related_to_b(
    base_name_to_sets_of_related_ops: Dict[str, Set[NSNodeTargetType]],
) -> Set[Tuple[NSNodeTargetType, NSNodeTargetType]]:
    # TODO(future PR): allow customizations
    # TODO(future PR): reuse existing quantization mappings
    # TODO(future PR): add the rest of modules and ops here
    type_a_related_to_b: Set[Tuple[NSNodeTargetType, NSNodeTargetType]] = set()

    for base_name, s in base_name_to_sets_of_related_ops.items():
        s_list = list(s)
        # add every bidirectional pair
        for idx_0 in range(0, len(s_list)):
            for idx_1 in range(idx_0, len(s_list)):
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

def get_reversed_fusions() -> List[Tuple[NSFusionType, int]]:
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
    results: List[Tuple[NSFusionType, int]] = []

    # Possible syntaxes:
    # * single op: torch.nn.Conv2d
    # * multiple ops: (torch.nn.ReLU, torch.nn.Conv2d)
    # For fusions, we only care about patterns composed of multiple ops.
    # TODO(future PR): allow customizations from default patterns.
    all_quant_patterns = get_native_quant_patterns()

    default_base_op_idx = 0
    for quant_pattern, _quant_handler in all_quant_patterns.items():
        # TODO: this is a temporary hack to flatten the patterns from quantization so
        # that it works with the ns matcher function, maybe we should use `is_match`
        # in torch.ao.quantization.fx.match_utils to match the patterns
        if isinstance(quant_pattern, tuple) and len(quant_pattern) == 2 and \
           isinstance(quant_pattern[1], tuple) and len(quant_pattern[1]) == 2:
            # flatten the pattern with form (nn.ReLU, (nn.BatchNorm2d, nn.Conv2d))
            quant_pattern = (quant_pattern[0], quant_pattern[1][0], quant_pattern[1][1])

        # Only patterns of multiple ops are fusions, ignore
        # patterns which contain a single ops (they get matched
        # without caring about fusions).
        if isinstance(quant_pattern, tuple):
            results.append((quant_pattern, default_base_op_idx))  # type: ignore[arg-type]

        # For each pattern, add additional patterns with observers and
        # fake quants at the end.
        # TODO(future PR): if needed, implement matching for a node
        #   having multiple output observers.
        for cls in (ObserverBase, FakeQuantizeBase):
            if isinstance(quant_pattern, tuple):
                new_pattern = (cls, *quant_pattern)
            else:
                new_pattern = (cls, quant_pattern)
            results.append((new_pattern, default_base_op_idx))  # type: ignore[arg-type]


    # After this point, results countains values such as
    # [..., ((torch.nn.Relu, torch.nn.Conv2d), 0), ...]

    # Patterns for matching fp16 emulation are not specified in the quantization
    # fusion mappings.  For now, define them here.
    fp16_em_base_op_idx = 1
    patterns_to_add = [
        # linear-relu fp16 emulation:
        # fp16_to_fp32 -> linear -> relu -> fp32_to_fp16
        ((("to", torch.float16), F.relu, F.linear, "dequantize"), fp16_em_base_op_idx,),
        # Conv-BN fusion (this happens outside of quantization patterns,
        # which is why it is defined separately here).
        ((nn.BatchNorm1d, nn.Conv1d), default_base_op_idx),
        ((nn.BatchNorm2d, nn.Conv2d), default_base_op_idx),
        ((nn.BatchNorm3d, nn.Conv3d), default_base_op_idx),
        ((nn.ReLU, nn.BatchNorm1d, nn.Conv1d), default_base_op_idx),
        ((nn.ReLU, nn.BatchNorm2d, nn.Conv2d), default_base_op_idx),
        ((nn.ReLU, nn.BatchNorm3d, nn.Conv3d), default_base_op_idx),
    ]
    for p in patterns_to_add:
        results.append(p)  # type: ignore[arg-type]
        results.append(((ObserverBase, *p[0]), p[1]))  # type: ignore[arg-type]
        results.append(((FakeQuantizeBase, *p[0]), p[1]))  # type: ignore[arg-type]

    return results


def end_node_matches_reversed_fusion(
    end_node: Node,
    reversed_fusion: NSFusionType,
    gm: GraphModule,
    seen_nodes: Set[Node],
) -> bool:
    """
    Returns true if a pattern ending with `end_node` matches
    the fusion pattern.
    """
    cur_node = end_node
    for fusion_idx in range(len(reversed_fusion)):
        # each node can only belong to one matched pattern
        if cur_node in seen_nodes:
            return False

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
                target_mod = _getattr_from_fqn(gm, cur_node.target)
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
