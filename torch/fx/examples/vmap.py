# This example is provided only for explanatory and educational purposes.
import torch
import torch.nn as nn
import torch.fx as fx
from torch.fx import Proxy

import torch.fx
from torch.fx.passes.shape_prop import ShapeProp

# Batching Rules
# ---------------
# Giving a full overview of how vmap is implemented is outside the scope of
# this example, but we will provide a brief description of batching rules. The
# general idea behind a batching rule is that given the input arguments were
# originally meant to operate upon a non-batched version, what do we do
# when one of the inputs now has a batch dimension?
# One simple example to illustrate this is `torch.movedim(x, from_dim,
# to_dim)`, which moves a dimension from `from_dim` to `to_dim`. For example,
# if `x.shape = (1,2,3,4)`, then torch.movedim(x, 0, 2) would result in a shape
# of `(2,3,1,4)`.
# However, let's say that we introduce a batch dimension, so `x.shape =
# (B,1,2,3,4)`. Now, we can't simply execute the same `torch.movedim(x,0,2)`,
# as there is an extra batch dimension in the front. Instead, we must execute
# `torch.movedim(x,1,3)`. This procedure (and some other stuff to make sure the
# batch dimension is always at the front) is what's done in
# `movedim_batching_rule`.
#
# There is one final thing to note about these batching rules - they're almost
# entirely written in normal PyTorch, with the exception of `bdim` attribute
# that's needed for tracking the batch dimension. That is because in order to
# use these batching rules, we will be tracing them by passing in `Proxy`
# objects that will track the operations performed on them and append them to
# the graph.

def move_bdim_to_front(x, result_ndim=None):
    x_dim = len(x.shape)
    x_bdim = x.bdim
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

def movedim_batching_rule(x, from_dim, to_dim):
    x = move_bdim_to_front(x)
    return torch.movedim(x, from_dim + 1, to_dim + 1), 0

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


batching_rules[torch.mul] = gen_binary_op_batching_rule(torch.mul)
batching_rules[torch.unsqueeze] = unsqueeze_batching_rule
batching_rules[torch.movedim] = movedim_batching_rule


# In order to apply a batching rule, we will simply pass in `Proxy` objects as
# inputs to the functions. As the batching rules need some extra information
# such as the batch dimension and shape, we will do some bookkeeping here.
def gen_batching_rule_function(target, *args):
    def lift_shape(i):
        res = Proxy(i)
        res.shape = i.shape
        res.bdim = i.bdim
        return res
    proxy_args = [lift_shape(i) if isinstance(i, fx.Node) else i for i in args]
    out, bdim = batching_rules[target](*proxy_args)
    out_node = out.node
    out_node.bdim = bdim
    return out_node

def vmap(model: torch.nn.Module, in_axes, example_args) -> torch.nn.Module:
    """vmap
    Given a model with inputs, vmap will return a function that works on
    batched versions of those inputs. Which inputs will be batched is
    determined by in_axes. In addition, as vmap requires shape (actually
    rank) information, we will pass in example_args (example inputs for the
    original module).
    """
    in_axes = iter(in_axes)
    fx_model = fx.symbolic_trace(model)
    # Here we run a shape propagation pass in order to annotate the graph with shape information.
    ShapeProp(fx_model).propagate(*example_args)
    # As vmap rewrites the whole graph, it's easiest to create an entirely new
    # graph and append to that.
    new_graph: fx.Graph = fx.Graph()

    # We will create an environment to map the new nodes created to the
    # corresponding old nodes.
    def lookup_env(l):
        return fx.node.map_aggregate(l, lambda x: env[x.name] if isinstance(x, fx.Node) else x)
    env = {}
    for node in fx_model.graph.nodes:
        if node.op == 'placeholder':
            # If the node is an input placeholder, we simply copy it over and
            # annotate it with the batch dimension from `in_axes`.
            new_node = new_graph.placeholder(node.name)
            new_node.bdim = next(in_axes)
            new_node.shape = node.shape
            env[node.name] = new_node
        elif node.op == 'output':
            new_graph.output(env[node.args[0].name])
        elif node.op == 'call_function':
            new_args = lookup_env(node.args)
            # If any of the inputs to the function has a new batch dimension,
            # we will need to use our batching rules. Otherwise, we will simply
            # copy the node over.
            if any([x.bdim is not None for x in new_args if isinstance(x, fx.Node)]):
                new_node = gen_batching_rule_function(node.target, *new_args)
            else:
                new_node = new_graph.node_copy(node, lambda x: env[x.name])
                new_node.bdim = None
            new_node.shape = node.shape
            env[node.name] = new_node
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

# Although this function actually takes in many shapes (due to broadcasting
# rules and such), pretend that M() operates only on scalars.
# The first thing we do is to turn this into a vector scalar multiplication. To
# do so, we will batch along the first dimension to turn it into a vector.
# We provide example_args to specify the original shapes of the function.
model = vmap(M(), in_axes=(None, 0), example_args=(x[0][0], y[0]))

# Now, our shape signature is ((), (M,)) -> (M,). This is computing the
# outer product of 2 vectors.
print(model(x[0][0], y) .shape)  # ((), (2,)) -> (2,)

# Now, we want to turn this from a scalar vector multiplication into a vector
# vector multiplication. That is, we would like to have the shape signature of
# ((N,), (M,)) -> (N, M). To do so, we will batch along the second argument.
# This is also known as an outer product.

model = vmap(model, in_axes=(0, None), example_args=(x[0][0], y))

print(model(x[0], y).shape)  # ((5,), (2,)) -> (5,2)


# We can continue to add an arbitary number of batch dimensions to our input.
# If we add another batch dimension to the first input we now get a batched
# outer product computation. ((B, N), (M,)) -> (B, N, M)

model = vmap(model, in_axes=(0, None), example_args=(x[0], y))
print(model(x, y).shape)  # ((3, 5), (2,)) -> (3, 5, 2)
