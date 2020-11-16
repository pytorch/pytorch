from torch.fx import (
    GraphModule,
)
from torch.nn.utils.fusion import fuse_conv_bn_eval
import torch

# turn foo.bar -> ['foo', 'bar']
def _parent_name(target):
    r = target.rsplit('.', 1)
    if len(r) == 1:
        return '', r[0]
    else:
        return r[0], r[1]

# Works for length 2 patterns with 2 modules
def matches_module_pattern(pattern, node, modules):
    if len(node.args) == 0:
        return False
    nodes = [node.args[0], node]
    for pattern_node, current_node in zip(pattern, nodes):
        if current_node.op != 'call_module':
            return False
        if not isinstance(current_node, torch.fx.Node):
            return False
        if current_node.target not in modules:
            return False
        if type(modules[current_node.target]) is not pattern_node:
            return False
    return True


def fuse(model):
    patterns = [(torch.nn.Conv2d, torch.nn.BatchNorm2d), (torch.nn.Conv1d, torch.nn.BatchNorm1d), (torch.nn.Conv3d, torch.nn.BatchNorm3d)]
    for pattern in patterns:
        for node in model.graph.nodes:
            modules = dict(model.named_modules())
            if matches_module_pattern(pattern, node, modules):
                if len(node.args[0].users) > 1:  # Output of conv is used by other nodes
                    continue
                fused_conv = fuse_conv_bn_eval(modules[node.args[0].target], modules[node.target])
                parent_name, name = _parent_name(node.args[0].target)
                setattr(dict(model.named_modules())[parent_name], name, fused_conv)
                node.replace_all_uses_with(node.args[0])
                model.graph.erase_node(node)
    return GraphModule(model, model.graph)
