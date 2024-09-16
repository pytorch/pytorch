# mypy: allow-untyped-decorators
# mypy: allow-untyped-defs

import torch
import torch.utils._pytree as pytree
from torch._C import DispatchKey
from torch._dispatch.python import suspend_functionalization
from torch._higher_order_ops.utils import _from_fun, create_fw_bw_graph, reenter_make_fx
from torch._ops import HigherOrderOperator
from torch._subclasses import FakeTensorMode
from torch._subclasses.functional_tensor import disable_functional_mode
from torch.fx.experimental.proxy_tensor import (
    disable_proxy_modes_tracing,
    ProxyTorchDispatchMode,
    track_tensor_tree,
)
from torch.fx.graph_module import GraphModule


class InvokeSubgraphHOP(HigherOrderOperator):
    def __init__(self) -> None:
        super().__init__("invoke_subgraph")

    # No need of kwargs
    def __call__(self, subgraph: GraphModule, *args):
        return super().__call__(subgraph, *args)


invoke_subgraph = InvokeSubgraphHOP()


# TODO(anijain2305) - Just for testing - We can remove this .. I think
@invoke_subgraph.py_impl(DispatchKey.CPU)
def invoke_subgraph_cpu(subgraph, *args):
    # print(subgraph)
    return subgraph(*args)


class InvokeSubgraphAutogradOp(torch.autograd.Function):
    """
    This autograd function op is to stash the backward graph in the ctx while
    running forward.
    """

    @staticmethod
    def forward(ctx, fw_graph, bw_graph, *args):
        ctx._fw_graph = fw_graph
        ctx._bw_graph = bw_graph

        # Save the args for the backward graph.
        # TODO(anijain2305) - Is this the right thing to do?
        ctx.save_for_backward(*args)

        # TODO(anijain2305) - Learn what is this ctx manager
        with torch._C._AutoDispatchBelowAutograd():
            return invoke_subgraph(
                fw_graph,
                *args,
            )

    @staticmethod
    def backward(ctx, *grad_outs):
        bw_graph = ctx._bw_graph
        # print(bw_graph)
        args = ctx.saved_tensors
        grads = invoke_subgraph(bw_graph, *(grad_outs + args))
        return None, None, *grads


def create_fw_bw_graph_local(subgraph, *args):
    """
    This needs to call make_fx with functionalization, partitioner and decomps.
    """

    # See Note [HOP create fw_bw graph] in create_fw_bw_graph in utils.py

    # TODO(anijain2305) - Not sure if we should disable functionalization
    with suspend_functionalization(), disable_functional_mode():
        with disable_proxy_modes_tracing():
            # args are functional tensors, generate some examplle tensors
            fw_inputs = pytree.tree_map(_from_fun, args)

            fw_outputs = pytree.tree_map(_from_fun, subgraph(*fw_inputs))
            if any(
                not isinstance(out, torch.Tensor)
                for out in fw_outputs
                if out is not None
            ):
                raise RuntimeError(
                    "Expect outputs of true_fn to only contains tensors or None. "
                    f"Got types {[type(out) for out in fw_outputs]}."
                )

            fw_graph, joint_graph = create_fw_bw_graph(
                subgraph, False, fw_inputs, fw_outputs
            )
            return fw_graph, joint_graph


@invoke_subgraph.py_impl(DispatchKey.Autograd)
def invoke_subgraph_autograd(subgraph, *args):
    fw_graph, bw_graph = create_fw_bw_graph_local(subgraph, *args)

    return InvokeSubgraphAutogradOp.apply(fw_graph, bw_graph, *args)


@invoke_subgraph.py_functionalize_impl
def invoke_subgraph_func(ctx, subgraph, *args):
    unwrapped_args = ctx.unwrap_tensors(args)
    with ctx.redispatch_to_next() as m:
        # TODO(anijain2305) - What to do for mutation?
        functionalized_subgraph = ctx.functionalize(subgraph)

        # TODO(anijain2305) - Why is the next line not - functionalized_subgraph(*unwrapped_args)?
        out = invoke_subgraph(functionalized_subgraph, *unwrapped_args)
    return ctx.wrap_tensors(out)


@invoke_subgraph.py_impl(FakeTensorMode)
def invoke_subgraph_fake_tensor_mode(mode, subgraph, *args):
    with mode:
        return subgraph(*args)


def trace_invoke_subgraph(proxy_mode: ProxyTorchDispatchMode, subgraph, *args):
    example_out = subgraph(*args)

    graph = reenter_make_fx(subgraph)(*args)
    assert isinstance(proxy_mode.tracer, torch.fx.Tracer)
    qualname = proxy_mode.tracer.get_fresh_qualname("repeated_subgraph")
    proxy_mode.tracer.root.register_module(qualname, graph)

    node_args = (graph, *args)
    proxy_args = pytree.tree_map(proxy_mode.tracer.unwrap_proxy, node_args)
    out_proxy = proxy_mode.tracer.create_proxy(
        "call_function", invoke_subgraph, proxy_args, {}
    )
    return track_tensor_tree(
        example_out, out_proxy, constant=None, tracer=proxy_mode.tracer
    )


@invoke_subgraph.py_impl(ProxyTorchDispatchMode)
def invole_subgraph_proxy_torch_dispatch_mode(proxy_mode, subgraph, *args):
    return trace_invoke_subgraph(proxy_mode, subgraph, *args)
