import torch
from torch import Tensor
import warnings
from itertools import count
from typing import Optional

import torch
import functools
import torch.utils._pytree as pytree
from torch._C import DispatchKey
from torch._dispatch.python import suspend_functionalization
from torch._higher_order_ops.utils import (
    _from_fun,
    _maybe_reenter_make_fx,
    clone_outputs_aliasing_inputs,
    prepare_fw_with_masks,
    reenter_make_fx,
)
from torch._ops import HigherOrderOperator
from torch._subclasses import FakeTensorMode
from torch._subclasses.functional_tensor import disable_functional_mode
from torch.fx.experimental.proxy_tensor import (
    disable_proxy_modes_tracing,
    ProxyTorchDispatchMode,
    track_tensor_tree,
)
from torch.fx.graph_module import GraphModule
from functorch import make_fx


import torch

class InvokeSubgraphHOP(HigherOrderOperator):
    def __init__(self) -> None:
        super().__init__("invoke_subgraph")

    def __call__(
        self,
        subgraph: GraphModule,
        operands,
    ):
        return super().__call__(subgraph, operands)


invoke_subgraph = InvokeSubgraphHOP()


@invoke_subgraph.py_impl(DispatchKey.CompositeExplicitAutograd)
def invoke_subgraph_composite_explicit_autograd(subgraph, operands):
    from torch.utils._python_dispatch import _get_current_dispatch_mode

    mode = _get_current_dispatch_mode()
    assert mode is None, "Mode should never be enabled for CPU/CUDA key"
    return subgraph(*operands)

def get_fwd_bwd(fn):
    """Returns a forward_fn and a backward_fn without actually running or tracing `fn`.
    The forward_fn's execution does not require us to trace out the backward graph
    or make assumptions about the backward gradients (tangents).
    """
    def forward(x):
        with torch.enable_grad():
            y = fn(x)
            saved_values = [(x, y)]
            node = y.grad_fn
            # Grab all of the saved values, return them from the fwd.
            while str(type(node)) != "<class 'AccumulateGrad'>":
                saved_values.append(node.saved)
                assert len(node.next_functions) == 1
                node = node.next_functions[0][0]
            return y.clone(), saved_values

    def backward(grad_y, saved):
        x_saved, y_saved = saved[0]
        saved = saved[1:]
        node = y_saved.grad_fn
        idx = 0
        # take the saved values, load them into their respective autograd Nodes.
        while str(type(node)) != "<class 'AccumulateGrad'>":
            node.saved = saved[idx]
            assert len(node.next_functions) == 1
            node = node.next_functions[0][0]
            idx += 1

        return torch.autograd.grad(y_saved, x_saved, grad_y)

    return forward, backward

@invoke_subgraph.py_impl(DispatchKey.Autograd)
def invoke_subgraph_autograd(subgraph, operands):
    class InvokeSubgraphFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            with torch._C._AutoDispatchBelowAutograd():
                fwd_fn, bwd_fn = get_fwd_bwd(subgraph)
                y, saved = invoke_subgraph(fwd_fn, operands)
                ctx.saved = saved
                ctx.bwd_fn = bwd_fn
                return y

        @staticmethod
        def backward(ctx, grad):
            result = invoke_subgraph(ctx.bwd_fn, (grad, ctx.saved))
            return result

    if torch.is_grad_enabled():
        return InvokeSubgraphFunction.apply(*operands)
    else:
        with torch._C._AutoDispatchBelowAutograd():
            return invoke_subgraph(subgraph, operands)


def trace_invoke_subgraph(
    proxy_mode: ProxyTorchDispatchMode, subgraph, operands
):
    example_out = invoke_subgraph(subgraph, operands)
    graph = reenter_make_fx(subgraph)(*operands)
    assert isinstance(proxy_mode.tracer, torch.fx.Tracer)
    qualname = proxy_mode.tracer.get_fresh_qualname("repeated_subgraph")
    proxy_mode.tracer.root.register_module(qualname, graph)

    node_args = (graph, operands)
    proxy_args = pytree.tree_map(proxy_mode.tracer.unwrap_proxy, node_args)
    out_proxy = proxy_mode.tracer.create_proxy(
        "call_function", invoke_subgraph, proxy_args, {}
    )
    return track_tensor_tree(
        example_out, out_proxy, constant=None, tracer=proxy_mode.tracer
    )


@invoke_subgraph.py_impl(ProxyTorchDispatchMode)
def invole_subgraph_proxy_torch_dispatch_mode(
    proxy_mode, subgraph, operands
):
    return trace_invoke_subgraph(proxy_mode, subgraph, operands)


class MySquare(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.saved = x
        return x ** 2

    @staticmethod
    def backward(ctx, grad):
        return grad * 2 * ctx.saved

"""
User code begins now
"""

def f(x):
    x = MySquare.apply(x)
    x = MySquare.apply(x)
    y = MySquare.apply(x)
    return y

# correctness test
x = torch.randn(3, requires_grad=True)
y = invoke_subgraph(f, (x,))
grad_x, = torch.autograd.grad(y, x, torch.ones_like(y))
expected, = torch.autograd.grad(f(x), x, torch.ones_like(y))
assert torch.allclose(grad_x, expected)

# make_fx on the forward graph *does not* require materialization of the backward graph,
# nor does it require us to make decisions about the tangents (grad_outputs).

x = torch.randn(3, requires_grad=True)

def isf(x):
    y = invoke_subgraph(f, (x,))
    return y

gm = make_fx(isf)(x)
print(gm.code)
"""
def forward(self, x_1):
    repeated_subgraph0 = self.repeated_subgraph0
    invoke_subgraph = torch.ops.higher_order.invoke_subgraph(repeated_subgraph0, (x_1,));  repeated_subgraph0 = x_1 = None
    getitem = invoke_subgraph[0]
    return getitem

# self.repeated_subgraph0
def forward(self, x_1):
    pow_1 = torch.ops.aten.pow.Tensor_Scalar(x_1, 2)
    pow_2 = torch.ops.aten.pow.Tensor_Scalar(pow_1, 2)
    pow_3 = torch.ops.aten.pow.Tensor_Scalar(pow_2, 2)
    clone = torch.ops.aten.clone.default(pow_3)
    return (clone, [(x_1, pow_3), pow_2, pow_1, x_1])
"""
