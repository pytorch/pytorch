from collections import defaultdict
from enum import Flag, auto
from typing import Callable, DefaultDict, Set

import torch
import torch.fx

class AccOpProperty(Flag):
    """
    A collection of static properties for acc_ops.

    * pointwise - op commutes with data restructuring ops such as reshape,
        transpose, permute. e.g. op(reshape(x)) == reshape(op(x)).
        Alternatively, for tensor x = (x1, x2, ...), there exists a scalar
        function f such that op(x) = (f(x1), f(x2), ...).
    * quantized - op expects quantized inputs and return quantized outputs
    * unary - op has exactly one graph dependent input. e.g. relu,
        dequantize, sum
    """
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
