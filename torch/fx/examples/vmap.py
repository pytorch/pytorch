import torch
import torch.nn as nn
import torch.fx as fx
from torch.fx import Proxy

import types
import torch.fx
from torch.fx.passes.shape_prop import ShapeProp


def lazify(transform):
    def lazy_transform(model, in_axes):
        if not hasattr(model, 'transforms'):
            model.transforms = []
            model.in_axes = []
        model.transforms.append(transform)
        model.in_axes.append(in_axes)

        def apply_transforms(model, *args, **kwargs):
            orig_args = args
            for in_axes in model.in_axes:
                orig_args = [a.select(bdim, 0) if bdim is not None else a for a, bdim in zip(orig_args, in_axes)]
            old_transforms = model.transforms
            old_axes = model.in_axes
            model = model.transforms[0](model, model.in_axes[0], orig_args)
            model.transforms = old_transforms[1:]
            model.in_axes = old_axes[1:]
            return model

        def new_model_f(model, *args, **kwargs):
            while len(model.transforms) > 0:
                model = apply_transforms(model, *args, **kwargs)

            cur_module = model
            return cur_module(*args, **kwargs)
        model.forward = types.MethodType(new_model_f, model)
        return model
    return lazy_transform


def move_bdim_to_front(x, result_ndim=None):
    x_dim = len(x.shape)
    x_bdim = x.node.bdim
    if x_bdim is None:
        x = torch.unsqueeze(x, 0)
    else:
        x = torch.movedim(x, x_bdim, 0)
    if result_ndim is None:
        return x
    diff = result_ndim - x_dim - (x_bdim is None)
    for _ in range(diff):
        x = torch.unsqueeze(x, 1)
    return x

batching_rules = {}
def gen_binary_op_batching_rule(op):
    def binary_op_batching_rule(a, b):
        a_ndim = len(a.shape)
        b_ndim = len(b.shape)
        result_ndim = max(a_ndim, b_ndim)
        a = move_bdim_to_front(a, result_ndim)
        b = move_bdim_to_front(b, result_ndim)
        res = op(a, b)
        return res, 0
    return binary_op_batching_rule

def unsqueeze_batching_rule(x, dim):
    x = move_bdim_to_front(x)
    if dim >= 0:
        return torch.unsqueeze(x, dim + 1), 0
    else:
        return torch.unsqueeze(x, dim), 0

def movedim_batching_rule(x, from_dim, to_dim):
    x = move_bdim_to_front(x)
    return torch.movedim(x, from_dim + 1, to_dim + 1), 0

batching_rules[torch.mul] = gen_binary_op_batching_rule(torch.mul)
batching_rules[torch.unsqueeze] = unsqueeze_batching_rule
batching_rules[torch.movedim] = movedim_batching_rule


def gen_batching_rule_function(target, new_graph, *args):
    def lift_shape(i):
        res = Proxy(i)
        res.shape = i.shape
        return res
    proxy_args = [lift_shape(i) if isinstance(i, fx.Node) else i for i in args]
    out, bdim = batching_rules[target](*proxy_args)
    out_node = out.node
    out_node.bdim = bdim
    return out_node

def vmap(model: torch.nn.Module, in_axes, inp_args) -> torch.nn.Module:
    in_axes = iter(in_axes)
    fx_model = fx.symbolic_trace(model)
    ShapeProp(fx_model).propagate(*inp_args)
    new_graph: fx.Graph = fx.Graph()

    def lookup_env(l):
        return fx.node.map_aggregate(l, lambda x: env[x.name] if isinstance(x, fx.Node) else x)
    env = {}
    for node in fx_model.graph.nodes:
        if node.op == 'placeholder':
            new_node = new_graph.placeholder(node.name)
            new_node.bdim = next(in_axes)
            new_node.shape = node.shape
            env[node.name] = new_node
        elif node.op == 'call_function':
            new_args = lookup_env(node.args)
            if any([x.bdim is not None for x in new_args if isinstance(x, fx.Node)]):
                new_node = gen_batching_rule_function(node.target, new_graph, *new_args)
            else:
                new_node = new_graph.node_copy(node, lambda x: env[x.name])
                new_node.bdim = None
            new_node.shape = node.shape
            env[node.name] = new_node
        elif node.op == 'output':
            new_graph.output(env[node.args[0].name])
        else:
            raise RuntimeError("Not yet implemented")


    res = fx.GraphModule(fx_model, new_graph)
    print(res.code)
    res.graph.lint()
    return res

x = torch.randn(3, 5)
y = torch.randn(2)
class M(nn.Module):
    def forward(self, a, b):
        return torch.mul(a, b)

vmap_lazy = lazify(vmap)

model = vmap_lazy(vmap_lazy(M(), in_axes=(0, None)), in_axes=(0, None))
print(model(x, y).shape)
