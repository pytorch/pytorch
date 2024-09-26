# mypy: allow-untyped-decorators
# mypy: allow-untyped-defs

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


invoke_subgraph_cache = {}

counter = count(0)

"""
Problems

- Functionalization needs to happen before partitioning - so probably
Autograd key. But also need support for inference, so probably functional
tensor as well
Somehow call aot_autograd

- Do we need DispatchKey.CompositeExplicitAutograd for invoke_subgraph?

- Decomps are not picked up by the time Autograd key is run. make_renter_fx
is not doing the right thing.

- We need to pass on the partitioner info from the top level to the subgraph.

- Unclear to me - how is input and output mutation handled? It seems to me,
we might need to raise an exception which tells Dynamo to restart and retrace
without any invoke_subgraph ops. It might be possible to do this at global
level.
--- Needs Dynamo work

- How to cache or de-dupe the invoke_subgraph such that it is traced only
once? It seems that we need some kind of fx-graph cache, at both Dynamo and
AOTDispatcher level. For AOTDispatcher, this key will be Dynamo subgraph and
we have to check if this subgraph has already been traced, if yes, use the
same lifted (must be present as an attr) graph module. For Dynamo, its unclear
to me. I am not sure if we require this in Dynamo, but I would definitely
prefer to de-dupe in Dynamo for cleanliness (and possibly ease of
implementation)

- What happens in inductor? And same de-dupe logic is required in inductor as
well.
"""

"""
Functional tensor holding a fake tensor

The order is
- Autograd
- Functionalize
- Decomps + ProxyTensorDispatchMode
- Fake propagation


torch.compile
- Parititioning - After everything has been done

With invoke_subgraph

- Autograd Key - its not functionalized, its not decomposed (because of renter_make_fx not expected) - and then I partition
This is a problem because of dce.

This was not a problem for flex and cond because they do not parition.
- Autograd - They create joint graph, no partioning - so no dec problem
- Funcitnoalize - they do this
- Decomps in ProxyTensorDispatchMode
- Fake tensor

For invoke
- Autograd - we need to functioanlize, decompose, decomps, fake (everything needs to happen here)

(Everything else is for inference mode)
- Functionalize - for inference, it needs to functionalize
- Decomps in ProxyTensorDispatchMode -
- Fake tensor
"""


class InvokeSubgraphHOP(HigherOrderOperator):
    def __init__(self) -> None:
        super().__init__("invoke_subgraph")

    # Identifier is for connecting forward and backward graphs
    def __call__(
        self,
        subgraph: GraphModule,
        identifier: str,
        graph_hash: Optional[int],
        operands,
    ):
        return super().__call__(subgraph, identifier, graph_hash, operands)


invoke_subgraph = InvokeSubgraphHOP()


def create_fw_bw_graph(fn, use_output_and_grad_bw, fw_inputs, fw_outputs):
    from torch._functorch.aot_autograd import AOTConfig, create_joint

    # Note:[HOP create fw_bw graph] We create "clean" environments for make_fx by suspending all dispatch keys
    # between Autograd and Python key. Currently, we only suspend functionalization but more can be
    # added when required. Will encounter two problems if we don't suspend functionalization:
    #
    # 1. make_fx fails to capture operations on input: the inputs are wrapped as _to_functional_tensor_wrapper,
    # but they will be unwrapped before entering ProxyTorchDispatchMode as part of the dispatching.
    # However, it's the outside wrapper that tracer creates proxies for. This casuses tracer fail to
    # fetch the proxy for the inputs and fail to capture any operations on them.
    #
    # 2. make_fx fails to capture output: the outputs after ProxyTorchDispatchMode are further
    # wrapped as FunctionalTensorWrapper in Functionalize key after return. However, the tracer
    # only associates the inner tensor with proxy in ProxyTorchDispatchMode. Therefore,
    # when creating the output node, it fails to associate the wrapped tensor with its proxy.
    # Instead, it will create _tensor_constant as output.

    dummy_aot_config = AOTConfig(
        fw_compiler=None,  # type: ignore[arg-type]
        bw_compiler=None,  # type: ignore[arg-type]
        partition_fn=None,  # type: ignore[arg-type]
        decompositions={},
        num_params_buffers=0,
        aot_id=0,
        keep_inference_input_mutations=False,
    )

    example_grad = [_from_fun(out) for out in fw_outputs]
    num_grads = len(example_grad)
    fw_graph = _maybe_reenter_make_fx(fn)(*fw_inputs)

    def joint_fn(*joint_operands_grads):
        if use_output_and_grad_bw:
            grads = joint_operands_grads[0]
            inputs = joint_operands_grads[1][-1:]
        else:
            grads = joint_operands_grads[:num_grads]
            inputs = joint_operands_grads[num_grads:]

        joint = create_joint(prepare_fw_with_masks(fn), aot_config=dummy_aot_config)
        _, grads = joint(
            list(inputs),
            [grad for grad in grads if grad is not None and grad.requires_grad],
        )

        # In order to keep map functional for backward graph,
        # we clone outputs that are aliasing inputs
        maybe_clone = clone_outputs_aliasing_inputs(joint_operands_grads)

        return pytree.tree_map(maybe_clone, grads)

    if use_output_and_grad_bw:
        example_xs_out = list(fw_inputs) + list(fw_outputs)
        joint_graph = _maybe_reenter_make_fx(joint_fn)(
            (list(example_grad), list(example_xs_out))
        )
    else:
        example_xs_out = list(fw_inputs)
        joint_graph = _maybe_reenter_make_fx(joint_fn)(
            *(list(example_grad) + list(example_xs_out))
        )

    return fw_graph, joint_graph


# def create_fw_bw_graph(
#     fn, use_output_and_grad_bw, fw_inputs, fw_outputs, partition_fn=None
# ):
#     from torch._functorch.aot_autograd import AOTConfig, create_joint

#     dummy_aot_config = AOTConfig(
#         fw_compiler=None,  # type: ignore[arg-type]
#         bw_compiler=None,  # type: ignore[arg-type]
#         partition_fn=None,  # type: ignore[arg-type]
#         decompositions={},
#         num_params_buffers=0,
#         aot_id=0,
#         keep_inference_input_mutations=False,
#     )

#     # TODO(anijain2305, bdhirsh) - This needs to be updated when we have aot
#     # autograd respecting strides for grad outs
#     example_grad = [_from_fun(out) for out in fw_outputs]
#     num_grads = len(example_grad)

#     # Partitioner needs primals and tangents. So, dont change the input signature.
#     def joint_fn(primals, tangents):
#         joint = create_joint(prepare_fw_with_masks(fn), aot_config=dummy_aot_config)
#         outs, grads = joint(
#             list(primals),
#             [grad for grad in tangents if grad is not None and grad.requires_grad],
#         )

#         # In order to keep map functional for backward graph,
#         # we clone outputs that are aliasing inputs
#         # TODO(anijain2305) - What is this?
#         maybe_clone = clone_outputs_aliasing_inputs(primals + tangents)

#         return pytree.tree_map(maybe_clone, list(outs) + list(grads))

#     example_xs_out = list(fw_inputs)
#     joint_graph = _maybe_reenter_make_fx(joint_fn)(list(fw_inputs), list(example_grad))

#     fw_graph, bw_graph = partition_fn(
#         joint_graph,
#         list(fw_inputs) + list(example_grad),
#         num_fwd_outputs=len(fw_outputs),
#     )
#     return fw_graph, bw_graph


# def get_cache_key(gm, args):
#     from torch._functorch._aot_autograd.autograd_cache import autograd_cache_key
#     from torch._functorch._aot_autograd.schemas import AOTConfig
#     from torch._dynamo.higher_order_ops import canonicalize


#     config = AOTConfig(
#         fw_compiler=None,
#         bw_compiler=None,
#         inference_compiler=None,
#         partition_fn=None,
#         decompositions={},
#         num_params_buffers=0,
#         aot_id=0,
#         keep_inference_input_mutations=False,
#         dynamic_shapes=True,
#         aot_autograd_arg_pos_to_source=None,
#         is_export=False,
#         no_tangents=False,
#         enable_log=False,
#     )


#     canonicalized_gmod = canonicalize(body_gmod, tx.output.nn_modules)

#     key, lines = autograd_cache_key(canonicalized_gmod, fake_inputs, config, {})
#     return key


@invoke_subgraph.py_impl(DispatchKey.CompositeExplicitAutograd)
def invoke_subgraph_composite_explicit_autograd(
    subgraph, identifier, graph_hash, operands
):
    from torch.utils._python_dispatch import _get_current_dispatch_mode

    mode = _get_current_dispatch_mode()
    assert mode is None, "Mode should never be enabled for CPU/CUDA key"
    return subgraph(*operands)


class InvokeSubgraphAutogradOp(torch.autograd.Function):
    """
    This autograd function op is to stash the backward graph in the ctx while
    running forward.
    """

    @staticmethod
    def forward(ctx, fw_graph, bw_graph, identifier, graph_hash, *operands):
        """
        num_fwd_outputs is the number of outputs of the forward graph. Rest of
        the outputs are saved for the backward graph.
        """
        ctx._fw_graph = fw_graph
        ctx._bw_graph = bw_graph
        ctx.identifier = identifier
        ctx.graph_hash = graph_hash

        # TODO(anijain2305) - Learn what is this ctx manager
        with torch._C._AutoDispatchBelowAutograd():
            out = invoke_subgraph(
                fw_graph,
                identifier,
                f"{graph_hash}_forward",
                operands,
            )

        ctx.save_for_backward(*operands)
        return out

    @staticmethod
    def backward(ctx, *grad_outs):
        bw_graph = ctx._bw_graph
        identifier = ctx.identifier
        saved_tensors = ctx.saved_tensors
        graph_hash = ctx.graph_hash
        grads = invoke_subgraph(
            bw_graph, identifier, f"{graph_hash}_backward", (grad_outs + saved_tensors)
        )
        return None, None, None, None, *grads


def create_fw_bw_graph_local(subgraph, operands):
    """
    This needs to call make_fx with functionalization, partitioner and decomps.
    """
    # TODO(anijain2305) - Need to get this from the top - min_cut_rematerialization_partition

    # See Note [HOP create fw_bw graph] in create_fw_bw_graph in utils.py
    # TODO(anijain2305) - Not sure if we should disable functionalization
    with suspend_functionalization(), disable_functional_mode():
        with disable_proxy_modes_tracing():
            # args are functional tensors, generate some examplle tensors
            fw_inputs = pytree.tree_map(_from_fun, operands)

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

            # TODO(anijain2305) - Need to get this from the top - min_cut_rematerialization_partition
            fw_graph, bw_graph = create_fw_bw_graph(
                subgraph,
                False,
                fw_inputs,
                fw_outputs,
                # partition_fn=min_cut_rematerialization_partition,
            )
            return fw_graph, bw_graph


invoke_subgraph_autograd_cached = {}

@invoke_subgraph.py_impl(DispatchKey.Autograd)
def invoke_subgraph_autograd(subgraph, identifier, graph_hash, operands):
    # All of these imports need to be here in order to avoid circular dependencies
    # import functools
    # from torch._dispatch.python import suspend_functionalization
    # from torch._subclasses.functional_tensor import disable_functional_mode
    # from torch.fx.experimental.proxy_tensor import disable_proxy_modes_tracing

    # def fw_compile(gm, *exmaple_args):
    #     def run(*run_args):
    #             return invoke_subgraph(gm, *run_args)
    #     return run

    # from torch._functorch.aot_autograd import aot_module_simplified
    # from torch._functorch.compilers import nop
    # with suspend_functionalization(), disable_functional_mode():
    #     with disable_proxy_modes_tracing():
    #         out = aot_module_simplified(subgraph, args, fw_compile)
    #         return out(*args)

    # A shortcut for the case where all inputs don't require gradient,
    # we skip tracing the forward and backward graph.
    if pytree.tree_all_only(
        torch.Tensor,
        lambda t: not t.requires_grad,  # type: ignore[union-attr]
        operands,
    ):
        with torch._C._AutoDispatchBelowAutograd():
            return invoke_subgraph(subgraph, identifier, graph_hash, operands)

    # Very bad hack to get around the failures - check test_linear
    if identifier == "_partitioned":
        with torch._C._AutoDispatchBelowAutograd():
            return invoke_subgraph(subgraph, identifier, graph_hash, operands)

    global invoke_subgraph_autograd_cached
    if graph_hash in invoke_subgraph_autograd_cached:
        return invoke_subgraph_autograd_cached[graph_hash](
            *operands
        )

    fw_graph, bw_graph = create_fw_bw_graph_local(subgraph, operands)
    global counter
    new_identifier = identifier
    if identifier == "start":
        new_identifier = f"subgraph_{next(counter)}"

    # TODO(anijain2305) - For some reason, if I pass operands as a tuple to
    # Autograd.Function, it does not pick up the backward pass.
    def cached_fn(*args):
        return InvokeSubgraphAutogradOp.apply(fw_graph, bw_graph, new_identifier, graph_hash, *args)
    invoke_subgraph_autograd_cached[graph_hash] = cached_fn
    return cached_fn(*operands)


@invoke_subgraph.py_functionalize_impl
def invoke_subgraph_func(ctx, subgraph, identifier, graph_hash, operands):
    unwrapped_operands = ctx.unwrap_tensors(operands)
    with ctx.redispatch_to_next() as m:
        # TODO(anijain2305) - What to do for mutation?
        functionalized_subgraph = ctx.functionalize(subgraph)

        out = invoke_subgraph(
            functionalized_subgraph, identifier, graph_hash, unwrapped_operands
        )
    return ctx.wrap_tensors(out)


invoke_subgraph_fake_prop_cache = {}

@invoke_subgraph.py_impl(FakeTensorMode)
def invoke_subgraph_fake_tensor_mode(mode, subgraph, identifier, graph_hash, operands):
    if graph_hash in invoke_subgraph_fake_prop_cache:
        return invoke_subgraph_fake_prop_cache[graph_hash]
    with mode:
        out =  subgraph(*operands)
        invoke_subgraph_fake_prop_cache[graph_hash] = out
        return out


invoke_subgraph_cache = {}


def trace_invoke_subgraph(
    proxy_mode: ProxyTorchDispatchMode, subgraph, identifier, graph_hash, operands
):
    example_out = subgraph(*operands)
    if graph_hash in invoke_subgraph_cache:
        graph = invoke_subgraph_cache[graph_hash]
    else:
        graph = reenter_make_fx(subgraph)(*operands)
        assert isinstance(proxy_mode.tracer, torch.fx.Tracer)
        qualname = proxy_mode.tracer.get_fresh_qualname("repeated_subgraph")
        proxy_mode.tracer.root.register_module(qualname, graph)
        invoke_subgraph_cache[graph_hash] = graph

    node_args = (graph, identifier, graph_hash, operands)
    proxy_args = pytree.tree_map(proxy_mode.tracer.unwrap_proxy, node_args)
    out_proxy = proxy_mode.tracer.create_proxy(
        "call_function", invoke_subgraph, proxy_args, {}
    )
    return track_tensor_tree(
        example_out, out_proxy, constant=None, tracer=proxy_mode.tracer
    )


@invoke_subgraph.py_impl(ProxyTorchDispatchMode)
def invole_subgraph_proxy_torch_dispatch_mode(
    proxy_mode, subgraph, identifier, graph_hash, operands
):
    return trace_invoke_subgraph(proxy_mode, subgraph, identifier, graph_hash, operands)
