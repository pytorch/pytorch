# mypy: allow-untyped-decorators
# mypy: allow-untyped-defs

import torch
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


class InvokeSubgraphHOP(HigherOrderOperator):
    def __init__(self) -> None:
        super().__init__("invoke_subgraph")

    # No need of kwargs
    def __call__(self, subgraph: GraphModule, *args):
        return super().__call__(subgraph, *args)


invoke_subgraph = InvokeSubgraphHOP()


# TODO(anijain2305) - COPIED FROM UTILS FOR PARTITIONER
def create_fw_bw_graph(
    fn, use_output_and_grad_bw, fw_inputs, fw_outputs, partition_fn=None
):
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
        inputs = joint_operands_grads[:len(fw_inputs)]
        example_grads = joint_operands_grads[len(fw_inputs) :]

        joint = create_joint(prepare_fw_with_masks(fn), aot_config=dummy_aot_config)
        outs, grads = joint(
            list(inputs),
            [grad for grad in example_grads if grad is not None and grad.requires_grad],
        )

        # In order to keep map functional for backward graph,
        # we clone outputs that are aliasing inputs
        maybe_clone = clone_outputs_aliasing_inputs(joint_operands_grads)

        return pytree.tree_map(maybe_clone, list(outs) + list(grads))

    example_xs_out = list(fw_inputs)
    joint_graph = _maybe_reenter_make_fx(joint_fn)(
        *(list(fw_inputs) + list(example_grad))
    )

    if partition_fn:
        # breakpoint()
        fw_graph, bw_graph = partition_fn(
            joint_graph,
            list(fw_inputs) + list(example_grad),
            num_fwd_outputs=len(fw_outputs),
        )
        # breakpoint()
        print(fw_graph)
        print(bw_graph)
        return fw_graph, bw_graph

    return fw_graph, joint_graph


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
    # TODO(anijain2305) - Need to get this from the top - min_cut_rematerialization_partition
    from functorch.compile import min_cut_rematerialization_partition

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

            # TODO(anijain2305) - Need to get this from the top - min_cut_rematerialization_partition
            fw_graph, joint_graph = create_fw_bw_graph(
                subgraph,
                False,
                fw_inputs,
                fw_outputs,
                partition_fn=min_cut_rematerialization_partition,
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
