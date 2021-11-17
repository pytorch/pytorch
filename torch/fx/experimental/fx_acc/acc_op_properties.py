from collections import defaultdict
from enum import Flag, auto
from typing import Callable, DefaultDict, Set

import torch
import torch.fx

class AccOpProperty(Flag):
    pointwise = auto()
    quantized = auto()
    unary = auto()

acc_op_properties: DefaultDict[Callable, Set[AccOpProperty]] = defaultdict(set)
acc_ops_with_property: DefaultDict[AccOpProperty, Set[Callable]] = defaultdict(set)


def register_acc_op_properties(*properties: AccOpProperty):
    """
    Attach properties to acc_op to inform optimization
    """
    def decorator(acc_op: Callable):
        acc_op_properties[acc_op] |= set(properties)
        for prop in properties:
            acc_ops_with_property[prop].add(acc_op)
        return acc_op
    return decorator


def add_optimization_properties_to_meta(mod: torch.fx.GraphModule) -> None:
    """
    Add acc_op properties to Node.meta to inform optimization
    """
    for node in mod.graph.nodes:
        node.meta['acc_op_properties'] = acc_op_properties[node.target]
