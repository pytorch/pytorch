import torch.fx as fx
from torch.nn.utils.fusion import fuse_conv_bn_eval
from typing import Type, Dict, Any, Tuple, Iterable
import torch

def _parent_name(target : str) -> Tuple[str, str]:
    """
    Splits a qualname into parent path and last atom.
    For example, `foo.bar.baz` -> (`foo.bar`, `baz`)
    """
    *parent, name = target.rsplit('.', 1)
    return parent[0] if parent else '', name

# Works for length 2 patterns with 2 modules
def matches_module_pattern(pattern: Iterable[Type], node: fx.Node, modules: Dict[str, Any]):
    if len(node.args) == 0:
        return False
    nodes: Tuple[Any, fx.Node] = (node.args[0], node)
    for expected_type, current_node in zip(pattern, nodes):
        if not isinstance(current_node, fx.Node):
            return False
        if current_node.op != 'call_module':
            return False
        if not isinstance(current_node.target, str):
            return False
        if current_node.target not in modules:
            return False
        if type(modules[current_node.target]) is not expected_type:
            return False
    return True


def fuse(model: torch.nn.Module) -> torch.nn.Module:
    patterns = [(torch.nn.Conv1d, torch.nn.BatchNorm1d), (torch.nn.Conv2d, torch.nn.BatchNorm2d), (torch.nn.Conv3d, torch.nn.BatchNorm3d)]
    modules = dict(model.named_modules())
    for pattern in patterns:
        for node in model.graph.nodes:
            if matches_module_pattern(pattern, node, modules):
                if len(node.args[0].users) > 1:  # Output of conv is used by other nodes
                    continue
                fused_conv = fuse_conv_bn_eval(modules[node.args[0].target], modules[node.target])
                parent_name, name = _parent_name(node.args[0].target)
                setattr(modules[parent_name], name, fused_conv)
                node.replace_all_uses_with(node.args[0])
                model.graph.erase_node(node)
    return fx.GraphModule(model, model.graph)
