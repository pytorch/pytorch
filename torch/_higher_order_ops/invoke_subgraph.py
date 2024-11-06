# mypy: allow-untyped-decorators
# mypy: allow-untyped-defs


from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.utils._pytree as pytree
from torch._dispatch.python import suspend_functionalization
from torch._higher_order_ops.utils import (
    _from_fun,
    _maybe_reenter_make_fx,
    clone_outputs_aliasing_inputs,
    get_dummy_aot_autograd_config,
    prepare_fw_with_masks,
)
from torch._subclasses.functional_tensor import disable_functional_mode
from torch.fx.experimental.proxy_tensor import disable_proxy_modes_tracing
from torch.fx.graph_module import GraphModule

from .prim_hop_base import FunctionWithNoFreeVars, PrimHOPBase


invoke_subgraph_counter = 0


class InvokeSubgraphHOP(PrimHOPBase):
    def __init__(self) -> None:
        super().__init__("invoke_subgraph")

    def __call__(
        self,
        subgraph: Union[GraphModule, Callable, FunctionWithNoFreeVars],
        operands: Union[
            List[Union[torch.Tensor, torch.SymInt]],
            Tuple[Union[torch.Tensor, torch.SymInt]],
        ],
        *,
        # identifier is setup by upper part of the stack. This helps us in
        # identifying two invoke_subgraph calls have same subgraph.
        identifier: Optional[str],
    ):
        assert identifier is None or isinstance(
            identifier, str
        ), "identifier must be a None or a string"

        assert isinstance(
            operands, (list, tuple)
        ), f"invoke_subgraph operands must be a list or tuple of tensors and SymInts {operands}"
        assert all(
            isinstance(o, (torch.Tensor, torch.SymInt)) for o in operands
        ), f"invoke_subgraph operands must be a list of tensors and SymInts {operands}"

        return super().__call__(subgraph, operands, identifier=identifier)

    # TODO: I've wiped out the Autograd cache. We should
    # figure out what we want to do with it...
    def _forward_kwargs(self, *_, **kwargs):
        return {"identifier": f"___forward_{kwargs['identifier']}"}

    def _backward_kwargs(self, *_, **kwargs):
        return {"identifier": f"___backward_{kwargs['identifier']}"}

    def _trace_subgraph(self, proxy_mode, subgraph, operands, *, identifier):
        # Check if we have already traced the subgraph.
        graph = None
        invoke_subgraph_cache = get_invoke_subgraph_cache()
        if invoke_subgraph_cache:
            graph = invoke_subgraph_cache.get_proxy_dispatch_entry(identifier)
        if graph is None:
            graph = super()._trace_subgraph(proxy_mode, subgraph, operands)
        if invoke_subgraph_cache:
            invoke_subgraph_cache.add_proxy_dispatch_entry(identifier, graph)
        return graph

    def _dynamo_call_function_hook(self, tx, body_gmod, kwargs):
        from torch._dynamo.variables.constant import ConstantVariable
        from torch._dynamo.variables.higher_order_ops import (
            add_subgraph,
            hash_graph_and_inputs,
        )

        fake_inputs = [
            node.meta["example_value"]
            for node in body_gmod.graph.nodes
            if node.op == "placeholder"
        ]

        key = hash_graph_and_inputs(tx, body_gmod, fake_inputs)
        invoke_subgraph_cache = (
            tx.output.tracing_context.hop_dispatch_set_cache.get_cache(invoke_subgraph)
        )
        if invoke_subgraph_cache:
            if identifier := invoke_subgraph_cache.get_dynamo_identifier(key):
                kwargs["identifier"] = ConstantVariable.create(identifier)
                return identifier, kwargs

        body_name = add_subgraph(tx, "invoke_subgraph", body_gmod)
        if invoke_subgraph_cache:
            invoke_subgraph_cache.add_dynamo_identifier(key, body_name)

        kwargs["identifier"] = ConstantVariable.create(body_name)

        return body_name, kwargs


invoke_subgraph = InvokeSubgraphHOP()


def get_invoke_subgraph_cache():
    cache = None
    if tracing_ctx := torch._guards.TracingContext.try_get():
        cache = tracing_ctx.hop_dispatch_set_cache.get_cache(invoke_subgraph)
    return cache


def trace_joint_graph(fn, fw_inputs, fw_outputs):
    """
    Naively trace out a joint graph. This simplifies the reconstruction of joint
    graph in the min-cut partitioner later on.
    """
    from torch._functorch.aot_autograd import create_joint

    dummy_aot_config = get_dummy_aot_autograd_config()

    # This joint_fn is inserted as the backward graph as is. This simplifies the
    # min-cut partitioner work later on.
    #   Input signature - (*primals, *tangents)
    #   Output signature - (*grads, *fw_outs)
    # The output signature is deliberately kept grads first and fw_outs second.
    # Having grads first makes the min-cut partitioner HOP graph stitching
    # easier.
    def joint_fn(*primals_and_tangents):
        primals = primals_and_tangents[: len(fw_inputs)]
        tangents = primals_and_tangents[len(fw_inputs) :]

        fw_outs, grads = create_joint(
            prepare_fw_with_masks(fn), aot_config=dummy_aot_config
        )(primals, tangents)

        maybe_clone = clone_outputs_aliasing_inputs(primals_and_tangents)

        # return signature is deliberately kept (*grads, *fw_outs). This
        # simplifies partitioning work later on.
        return pytree.tree_map(maybe_clone, grads + list(fw_outs))

    primals = list(fw_inputs)
    # This assumes that the tangent strides match fw_outputs strides. Check the
    # InvokeSubgraphAutogradOp backward op for the contiguous call.
    tangents = [_from_fun(out) for out in fw_outputs]

    joint_operands = primals + tangents

    return _maybe_reenter_make_fx(joint_fn)(*joint_operands)


def create_fw_bw_graph(subgraph, operands):
    with suspend_functionalization(), disable_functional_mode():
        with disable_proxy_modes_tracing():
            # args are functional tensors, generate some example tensors
            fw_inputs = pytree.tree_map(_from_fun, operands)

            fw_outputs = pytree.tree_map(_from_fun, subgraph(*fw_inputs))
            if any(
                not isinstance(out, torch.Tensor)
                for out in fw_outputs
                if out is not None
            ):
                raise RuntimeError(
                    "Expect outputs of invoke_subgraph to only contains tensors or None. "
                    f"Got types {[type(out) for out in fw_outputs]}."
                )

            # Trace the forward subgraph
            fw_graph = _maybe_reenter_make_fx(subgraph)(*fw_inputs)

            # Trace the joint graph and assign it to the bwd graph
            bw_graph = trace_joint_graph(
                subgraph,
                fw_inputs,
                fw_outputs,
            )
            return fw_graph, bw_graph, len(fw_outputs)


class InvokeSubgraphAutogradOp(torch.autograd.Function):
    """
    This autograd function op is to stash the backward graph in the ctx while
    running forward.
    """

    @staticmethod
    def forward(ctx, fw_graph, bw_graph, identifier, num_fw_outs, *operands):
        ctx._fw_graph = fw_graph
        ctx._bw_graph = bw_graph
        ctx._identifier = identifier
        ctx._num_fw_outs = num_fw_outs

        with torch._C._AutoDispatchBelowAutograd():
            out = invoke_subgraph(
                fw_graph,
                operands,
                identifier=f"___forward_{identifier}",
            )

        ctx.save_for_backward(*operands)
        return out

    @staticmethod
    def backward(ctx, *grad_outs):
        bw_graph = ctx._bw_graph
        identifier = ctx._identifier
        primals = ctx.saved_tensors
        num_fw_outs = ctx._num_fw_outs

        # While tracing we made the assumption that tangents are contiguous. So,
        # force the grad_outs to be contiguous.
        contiguous_grad_outs = tuple([o.contiguous() for o in grad_outs])

        # bw_graph is a joint graph with signature (*primals_and_tangents) and
        # returns (*grads_and_fw_outs). To get the grads, we use the num_fw_outs
        # to extract the grads.
        primals_and_tangents = primals + contiguous_grad_outs
        grads = invoke_subgraph(
            bw_graph,
            primals_and_tangents,
            identifier=f"___backward_{identifier}",
        )[:-num_fw_outs]
        return None, None, None, None, *grads
