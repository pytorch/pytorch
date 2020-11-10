from torch.fx import (
    GraphModule,
    map_arg
)
import copy
import torch.nn as nn
from torch.fx.graph import Graph
import torch
import sys

def fuse(model):
    patterns = [(torch.nn.Conv2d, torch.nn.BatchNorm2d), (torch.nn.Conv1d,torch.nn.BatchNorm1d), (torch.nn.Conv3d, torch.nn.BatchNorm3d)]
    for pattern in patterns:
        for node in model.graph.nodes:
            modules = dict(model.named_modules())
            current = node.target
            if len(node.args) == 0:
                continue
            if not isinstance(node.args[0], torch.fx.Node):
                continue
            prv = node.args[0].target
            if node.op != 'call_module' or node.args[0].op != 'call_module': continue
            if current not in modules or prv not in modules: continue
            if (type(modules[current]) is pattern[1]) and (type(modules[prv]) is pattern[0]):
                if len(node.args[0].users) > 1: continue
                fused_conv = fuse_conv_bn_eval(modules[node.args[0].target], modules[node.target])
                new_name = node.args[0].target.replace('.', "_")
                setattr(model, new_name, fused_conv)
                with model.graph.inserting_before(node.args[0]):
                    new_node = model.graph.call_module(new_name, node.args[0].args)
                node.replace_all_uses_with(new_node)
                t_args = node.args[0]
                node.args[0].replace_all_uses_with(new_node)
                model.graph.erase_node(t_args)
                model.graph.erase_node(node)
    return GraphModule(model, model.graph)

def fuse_conv_bn_eval(conv, bn):
    assert(not (conv.training or bn.training)), "Fusion only for eval!"

    fused_conv = copy.deepcopy(conv)

    fused_conv.weight, fused_conv.bias = \
        fuse_conv_bn_weights(fused_conv.weight, fused_conv.bias,
                             bn.running_mean, bn.running_var, bn.eps, bn.weight, bn.bias)

    # fused_conv.bias = None
    return fused_conv

def fuse_conv_bn_weights(conv_w, conv_b, bn_rm, bn_rv, bn_eps, bn_w, bn_b):
    if conv_b is None:
        conv_b = torch.zeros_like(bn_rm)
    if bn_w is None:
        bn_w = torch.ones_like(bn_rm)
    if bn_b is None:
        bn_b = torch.zeros_like(bn_rm)
    bn_var_rsqrt = torch.rsqrt(bn_rv + bn_eps)

    conv_w = conv_w * (bn_w * bn_var_rsqrt).reshape([-1] + [1] * (len(conv_w.shape) - 1))
    conv_b = (conv_b - bn_rm) * bn_var_rsqrt * bn_w + bn_b

    return torch.nn.Parameter(conv_w), torch.nn.Parameter(conv_b)
