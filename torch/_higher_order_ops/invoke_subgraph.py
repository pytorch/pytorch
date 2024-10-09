# mypy: allow-untyped-decorators
# mypy: allow-untyped-defs


import torch
import torch.utils._pytree as pytree
from torch._C import DispatchKey
from torch._dispatch.python import suspend_functionalization
from torch._higher_order_ops.utils import (
    _from_fun,
    _has_potential_branch_input_alias,
    _has_potential_branch_input_mutation,
    _maybe_reenter_make_fx,
    clone_outputs_aliasing_inputs,
    get_dummy_aot_autograd_config,
    prepare_fw_with_masks,
    reenter_make_fx,
    UnsupportedAliasMutationException,
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


class InvokeSubgraphHOP(HigherOrderOperator):
    def __init__(self) -> None:
        super().__init__("invoke_subgraph")

    # identifier is setup by upper part of the stack. This helps us in
    # identifying two invoke_subgraph calls have same subgraph.
    def __call__(
        self,
        subgraph: GraphModule,
        identifier: str,
        operands,
    ):
        return super().__call__(subgraph, identifier, operands)


invoke_subgraph = InvokeSubgraphHOP()


def trace_joint_graph(fn, fw_inputs, fw_outputs):
    """
    Naively trace out a joint graph. This simplifies the reconstruction of joint
    graph in the min-cut partitioner later on.
    """
    from torch._functorch.aot_autograd import create_joint

    dummy_aot_config = get_dummy_aot_autograd_config()

    def joint_fn(*primals_and_tangents):
        primals = primals_and_tangents[: len(fw_inputs)]
        tangents = primals_and_tangents[len(fw_inputs) :]

        fw_outs, grads = create_joint(
            prepare_fw_with_masks(fn), aot_config=dummy_aot_config
        )(primals, tangents)

        maybe_clone = clone_outputs_aliasing_inputs(primals_and_tangents)

        return pytree.tree_map(maybe_clone, list(fw_outs) + grads)

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
                f"___forward_{identifier}",
                operands,
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
        # returns (*fw_outs_and_grads). To get the grads, we use the num_fw_outs
        # to extract the grads.
        primals_and_tangents = primals + contiguous_grad_outs
        grads = invoke_subgraph(
            bw_graph, f"___backward_{identifier}", primals_and_tangents
        )[num_fw_outs:]
        return None, None, None, None, *grads


@invoke_subgraph.py_impl(DispatchKey.CompositeExplicitAutograd)
def invoke_subgraph_composite_explicit_autograd(subgraph, identifier, operands):
    from torch.utils._python_dispatch import _get_current_dispatch_mode

    mode = _get_current_dispatch_mode()
    assert mode is None, "Mode should never be enabled for CPU/CUDA key"
    return subgraph(*operands)


@invoke_subgraph.py_impl(DispatchKey.Autograd)
def invoke_subgraph_autograd(subgraph, identifier, operands):
    if not torch.is_grad_enabled():
        with torch._C._AutoDispatchBelowAutograd():
            return invoke_subgraph(subgraph, identifier, operands)

    # A shortcut for the case where all inputs don't require gradient,
    # we skip tracing the forward and backward graph.
    if pytree.tree_all_only(
        torch.Tensor,
        lambda t: not t.requires_grad,  # type: ignore[union-attr]
        operands,
    ):
        with torch._C._AutoDispatchBelowAutograd():
            return invoke_subgraph(subgraph, identifier, operands)

    fw_graph, bw_graph, num_fw_outs = create_fw_bw_graph(subgraph, operands)
    # TODO(anijain2305) - Implement caching of autograd function op.
    return InvokeSubgraphAutogradOp.apply(
        fw_graph, bw_graph, identifier, num_fw_outs, *operands
    )


@invoke_subgraph.py_functionalize_impl
def invoke_subgraph_func(ctx, subgraph, identifier, operands):
    unwrapped_operands = ctx.unwrap_tensors(operands)
    with ctx.redispatch_to_next() as m:
        # TODO(anijain2305) - Long term, it might be a bit restrictive to ban
        # mutation/aliasing. Investigate if there is a way to support this.
        # TODO(anijain2305) - Short term, improve compilation time by
        # skipping this check if the identifier has been seen before.
        pre_dispatch = hasattr(ctx, "mode") and ctx.mode.pre_dispatch
        if _has_potential_branch_input_mutation(
            subgraph, unwrapped_operands, pre_dispatch=pre_dispatch
        ):
            raise UnsupportedAliasMutationException(
                "One of invoke_subgraph hop might be modifying the input!"
            )
        if _has_potential_branch_input_alias(
            subgraph, unwrapped_operands, pre_dispatch=pre_dispatch
        ):
            raise UnsupportedAliasMutationException(
                "One of invoke_subgraph hop might be aliasing the input!"
            )

        functionalized_subgraph = ctx.functionalize(subgraph)
        out = invoke_subgraph(functionalized_subgraph, identifier, unwrapped_operands)
    return ctx.wrap_tensors(out)


@invoke_subgraph.py_impl(FakeTensorMode)
def invoke_subgraph_fake_tensor_mode(mode, subgraph, identifier, operands):
    # TODO(anijain2305) - Implement fake tensor caching.
    return subgraph(*operands)


@invoke_subgraph.py_impl(ProxyTorchDispatchMode)
def invoke_subgraph_proxy_torch_dispatch_mode(
    proxy_mode: ProxyTorchDispatchMode, subgraph, identifier, operands
):
    # TODO(anijain2305) - Implement proxy tensor caching.
    example_out = invoke_subgraph(subgraph, identifier, operands)
    graph = reenter_make_fx(subgraph)(*operands)
    assert isinstance(proxy_mode.tracer, torch.fx.Tracer)
    qualname = proxy_mode.tracer.get_fresh_qualname("repeated_subgraph")
    proxy_mode.tracer.root.register_module(qualname, graph)

    node_args = (graph, identifier, operands)
    proxy_args = pytree.tree_map(proxy_mode.tracer.unwrap_proxy, node_args)
    out_proxy = proxy_mode.tracer.create_proxy(
        "call_function", invoke_subgraph, proxy_args, {}
    )
    return track_tensor_tree(
        example_out, out_proxy, constant=None, tracer=proxy_mode.tracer
    )
