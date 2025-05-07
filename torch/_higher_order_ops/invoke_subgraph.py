# mypy: allow-untyped-defs


import contextlib
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Optional, Union

import torch
import torch.utils._pytree as pytree
from torch._C import DispatchKey
from torch._dispatch.python import suspend_functionalization
from torch._higher_order_ops.utils import (
    _from_fun,
    _maybe_reenter_make_fx,
    _set_compilation_env,
    clone_outputs_aliasing_inputs,
    FunctionalizeCtxWrapper,
    get_dummy_aot_autograd_config,
    prepare_fw_with_masks,
    reenter_make_fx,
    register_fake,
    save_tensors_and_symints_for_backward,
    saved_tensors_and_symints,
)
from torch._ops import HigherOrderOperator
from torch._subclasses.functional_tensor import disable_functional_mode
from torch.fx.experimental.proxy_tensor import (
    _temp_remove_metadata_torch_function_mode,
    _temp_remove_pre_dispatch_torch_function_mode,
    disable_proxy_modes_tracing,
    ProxyTorchDispatchMode,
    track_tensor_tree,
)
from torch.fx.graph_module import GraphModule
from torch.fx.passes.runtime_assert import insert_deferred_runtime_asserts


invoke_subgraph_counter = 0


# During the tracing of the joint graph, we construct this information. This is
# used to filter out grad_outs/tangents in the `backward` method of
# InvokeSubgraphAutogradOp.
@dataclass
class OutputMetadata:
    num_fw_outs: Optional[int] = None
    indexes_with_none: set[int] = field(default_factory=set)
    indexes_with_no_grad: set[int] = field(default_factory=set)


class InvokeSubgraphHOP(HigherOrderOperator):
    def __init__(self) -> None:
        super().__init__("invoke_subgraph")
        # This is used by the fake tensor cache key validator to extract the
        # subgraph and iterate over the nodes to find if all nodes are fake
        # tensor cacheable.
        self.subgraph_indexes = [
            0,
        ]

    # identifier is setup by upper part of the stack. This helps us in
    # identifying two invoke_subgraph calls have same subgraph.
    def __call__(
        self,
        subgraph: Union[GraphModule, FunctionalizeCtxWrapper],
        identifier: Optional[str],
        *operands,
    ):
        assert identifier is None or isinstance(
            identifier, str
        ), "identifier must be a None or a string"

        assert all(
            isinstance(o, (torch.Tensor, int, torch.SymInt)) for o in operands
        ), f"invoke_subgraph operands must be a list of tensors/ints/SymInts {operands}"

        return super().__call__(subgraph, identifier, *operands)


invoke_subgraph = InvokeSubgraphHOP()


def invoke_subgraph_placeholder(func, *args, **kwargs):
    if torch.compiler.is_dynamo_compiling():
        # This is just a placeholder for Dynamo to replace with invoke_subgraph
        raise RuntimeError("invoke_subgraph should not be called directly in Dynamo")

    if torch.compiler.is_compiling():
        # For non-strict export tracing, we still want to go through Dynamo
        from torch._dynamo.backends.debugging import (
            make_eager_backend_with_torch_function_mode,
        )

        def _invoke_subgraph_placeholder_wrapper(func, args):
            return invoke_subgraph_placeholder(func, *args)

        with _set_compilation_env(), torch._dynamo.utils.disable_cache_limit(), _temp_remove_pre_dispatch_torch_function_mode():
            with _temp_remove_metadata_torch_function_mode() as metadata_mode:
                if metadata_mode:
                    backend = make_eager_backend_with_torch_function_mode(metadata_mode)
                else:
                    backend = "eager"

                return torch.compile(
                    _invoke_subgraph_placeholder_wrapper,
                    backend=backend,
                    fullgraph=True,
                )(func, args)

    return func(*args, **kwargs)


def mark_compile_region(fn=None):
    """
    This wrapper instructs torch.compile to compile the wrapped region once and
    reuse the compiled artifact, instead of the usual way of aggressively
    inlining the function.

    Under the hood, it tells TorchDynamo to use InvokeSubgraph HOP for the
    region. For PyTorch eager, this is a no-op.
    """

    def wrap(func):
        def inner(*args, **kwargs):
            return invoke_subgraph_placeholder(func, *args, **kwargs)

        return inner

    if fn:
        return wrap(fn)
    else:
        return wrap


def get_invoke_subgraph_cache():
    cache = None
    if tracing_ctx := torch._guards.TracingContext.try_get():
        cache = tracing_ctx.hop_dispatch_set_cache.get_cache(invoke_subgraph)
    return cache


# TODO (@anijain2305) - Delete this function when base_hop uses invoke_subgraph infra
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
        return pytree.tree_map(maybe_clone, tuple(grads + list(fw_outs)))

    primals = list(fw_inputs)
    # This assumes that the tangent strides match fw_outputs strides. Check the
    # InvokeSubgraphAutogradOp backward op for the contiguous call.
    tangents = [_from_fun(out) for out in fw_outputs]

    joint_operands = primals + tangents

    return _maybe_reenter_make_fx(joint_fn)(*joint_operands)


# TODO (@anijain2305) - Delete this function when base_hop uses invoke_subgraph infra
def create_fw_bw_graph(subgraph, operands, grad_outputs=None):
    with suspend_functionalization(), disable_functional_mode():
        with disable_proxy_modes_tracing():
            # args are functional tensors, generate some example tensors
            fw_inputs = pytree.tree_map(_from_fun, operands)

            from torch._guards import detect_fake_mode

            fake_mode = detect_fake_mode(fw_inputs)
            context = (
                nullcontext()
                if fake_mode is None or fake_mode.shape_env is None
                else fake_mode.shape_env.ignore_fresh_unbacked_symbols()
            )

            with context:
                fw_outs = pytree.tree_map(_from_fun, subgraph(*fw_inputs))

            num_fw_outs = len(fw_outs)

            # Collect the indexes of none in the output to check that the grad
            # is None at the corresponding index in the backward. This check is
            # performed in the autograd.Function - InvokeSubgraphAutogradOp.
            # Also collect the indexes of no_grad in the output to filter out
            # the grad_outs in the `backward` method.
            output_metadata = OutputMetadata()

            output_metadata.num_fw_outs = num_fw_outs
            for idx, fw_out in enumerate(fw_outs):
                if fw_out is None:
                    output_metadata.indexes_with_none.add(idx)
                elif not fw_out.requires_grad:
                    output_metadata.indexes_with_no_grad.add(idx)

            if grad_outputs is None:
                # Infer grad_outputs to be the same properties as the fw_outputs
                # if they're not passed in
                # Although fw_outs are equivalent to grad_outputs for tracing
                # purposes, we have to carefully handle the None and fw_out that do
                # not have require_grad. At those indexes, we will have None in the
                # backward graph.
                grad_outputs = fw_outs
                grad_outputs = [grad for grad in grad_outputs if grad is not None]
                grad_outputs = [grad for grad in grad_outputs if grad.requires_grad]

                # Force grad_out to be contiguous. This is because at runtime,
                # grad_out could have different strides than fw_outs. So, we
                # force the grad_outs to be contiguous for both tracing and
                # runtime.
                grad_outputs = [grad.contiguous() for grad in grad_outputs]

            if any(
                not isinstance(out, torch.Tensor)
                for out in grad_outputs
                if out is not None
            ):
                raise RuntimeError(
                    "Expect outputs of invoke_subgraph to only contains tensors or None. "
                    f"Got types {[type(out) for out in grad_outputs]}."
                )

            # Trace the forward subgraph
            fw_graph = _maybe_reenter_make_fx(subgraph)(*fw_inputs)

            # Trace the joint graph and assign it to the bwd graph
            bw_graph = trace_joint_graph(
                subgraph,
                fw_inputs,
                grad_outputs,
            )
            return fw_graph, bw_graph, output_metadata


def get_output_metadata(subgraph, *operands):
    with suspend_functionalization(), disable_functional_mode():
        with disable_proxy_modes_tracing():
            # args are functional tensors, generate some example tensors
            fw_inputs = pytree.tree_map(_from_fun, operands)

            from torch._guards import detect_fake_mode

            fake_mode = detect_fake_mode(fw_inputs)
            context = (
                nullcontext()
                if fake_mode is None or fake_mode.shape_env is None
                else fake_mode.shape_env.ignore_fresh_unbacked_symbols()
            )

            with context:
                fw_outs = pytree.tree_map(_from_fun, subgraph(*fw_inputs))

            num_fw_outs = len(fw_outs)

            # Collect the indexes of none in the output to check that the grad
            # is None at the corresponding index in the backward. This check is
            # performed in the autograd.Function - InvokeSubgraphAutogradOp.
            # Also collect the indexes of no_grad in the output to filter out
            # the grad_outs in the `backward` method.
            output_metadata = OutputMetadata()

            output_metadata.num_fw_outs = num_fw_outs
            for idx, fw_out in enumerate(fw_outs):
                if fw_out is None:
                    output_metadata.indexes_with_none.add(idx)
                elif not fw_out.requires_grad:
                    output_metadata.indexes_with_no_grad.add(idx)
            return output_metadata


def trace_joint_graph_as_bwd(
    subgraph, num_primals, joint_operands, include_key_set, exclude_key_set
):
    """
    Naively trace out a joint graph. This simplifies the reconstruction of joint
    graph in the min-cut partitioner later on.
    """
    from torch._functorch.aot_autograd import create_joint

    dummy_aot_config = get_dummy_aot_autograd_config()

    if isinstance(subgraph, torch.fx.GraphModule):

        def graph_with_interpreter(*args):
            # Running graph with interpreter is needed for propagating the stack_trace
            with torch.fx.traceback.preserve_node_meta():
                return torch.fx.Interpreter(subgraph).run(*args)

        fn = graph_with_interpreter
    else:
        fn = subgraph

    # This joint_fn is inserted as the backward graph as is. This simplifies the
    # min-cut partitioner work later on.
    #   Input signature - (*primals, *tangents)
    #   Output signature - (*grads, *fw_outs)
    # The output signature is deliberately kept grads first and fw_outs second.
    # Having grads first makes the min-cut partitioner HOP graph stitching
    # easier.
    def joint_fn(*primals_and_tangents):
        primals = primals_and_tangents[:num_primals]
        tangents = primals_and_tangents[num_primals:]

        fw_outs, grads = create_joint(
            prepare_fw_with_masks(fn), aot_config=dummy_aot_config
        )(primals, tangents)

        maybe_clone = clone_outputs_aliasing_inputs(primals_and_tangents)

        # return signature is deliberately kept (*grads, *fw_outs). This
        # simplifies partitioning work later on.
        return pytree.tree_map(maybe_clone, tuple(grads + list(fw_outs)))

    with suspend_functionalization(), disable_functional_mode():
        with disable_proxy_modes_tracing():
            joint_operands = [_from_fun(arg) for arg in joint_operands]
            with contextlib.ExitStack() as stack:
                stack.enter_context(
                    torch._C._ForceDispatchKeyGuard(include_key_set, exclude_key_set),
                )
                with torch.enable_grad():
                    return _maybe_reenter_make_fx(joint_fn)(*joint_operands)


class InvokeSubgraphAutogradOp(torch.autograd.Function):
    """
    Saves the subgraph, i.e. original callable, in the forward method. And then
    traces out a joint graph in the backward. This delaying of tracing in
    backward, also called as lazy backward, ensures that the assumptions about
    the grad_out strides and tensor-subclass-ness are already accounted for.
    """

    @staticmethod
    def forward(
        ctx,
        subgraph,
        identifier,
        output_metadata,
        *operands,
    ):
        # We want to delay the backward graph construction until the backward.
        # So in forward, we just run the fw callable as is. And save all the
        # information necessary to construct the backward graph in the ctx.
        ctx._subgraph = subgraph
        ctx._identifier = identifier
        ctx._output_metadata = output_metadata
        # We snapshot the dispatch keys in forward for materializing the
        # the bw_graph in backward.
        ctx._fw_include_key_set = torch._C._dispatch_tls_local_include_set()
        ctx._fw_exclude_key_set = torch._C._dispatch_tls_local_exclude_set()

        save_tensors_and_symints_for_backward(ctx, operands)

        with torch._C._AutoDispatchBelowAutograd():
            out = invoke_subgraph(
                subgraph,
                f"fw_{identifier}",
                *operands,
            )

        # Check that None is at expected indexes.
        for idx, o in enumerate(out):
            if o is None:
                assert idx in output_metadata.indexes_with_none

        return out

    @staticmethod
    def backward(
        ctx,
        *grad_outs,
    ):
        from torch._dynamo.utils import dynamo_timed

        subgraph = ctx._subgraph
        identifier = ctx._identifier
        output_metadata = ctx._output_metadata
        primals = saved_tensors_and_symints(ctx)

        # Filter out grads that are None or do not require_grad. This was
        # the assumption we made during the tracing of joint_graph.
        filtered_grad_outs = []
        for idx, o in enumerate(grad_outs):
            if o is None:
                assert idx in output_metadata.indexes_with_none
            elif idx in output_metadata.indexes_with_no_grad:
                # Deliberately skip over the grad_outs which we know should be
                # None because the corresponding fwd_out does not require_grad.
                pass
            else:
                filtered_grad_outs.append(o)
        filtered_grad_outs = tuple(filtered_grad_outs)

        # Important note - Even though the forward graph can be same for
        # different invoke_subgraphs, the backward graph can be different
        # because the tangent strides can be different. So, here we cache on
        # tangent_metadata in addition to identifier
        from torch._guards import detect_fake_mode
        from torch._subclasses._fake_tensor_utils import _CacheKeyState
        from torch._subclasses.fake_tensor import extract_tensor_metadata

        fake_mode = detect_fake_mode(primals + filtered_grad_outs)
        state = _CacheKeyState(fake_mode.shape_env)

        tangent_metadata: list[object] = []
        for tangent in filtered_grad_outs:
            metadata = extract_tensor_metadata(tangent)
            metadata._flatten_into(tangent_metadata, fake_mode, state)
        tangent_metadata = tuple(tangent_metadata)

        # bw_graph is a joint graph with signature (*primals_and_tangents) and
        # returns (*grads_and_fw_outs). To get the grads, we use the num_fw_outs
        # to extract the grads.
        primals_and_tangents = primals + filtered_grad_outs

        # Check if we have already traced the bwd subgraph.
        bw_graph = None
        suffix = None
        invoke_subgraph_cache = get_invoke_subgraph_cache()
        cache_hit = False
        if invoke_subgraph_cache:
            bw_graph, suffix = invoke_subgraph_cache.get_lazy_bwd_entry(
                identifier, tangent_metadata
            )
            cache_hit = bw_graph is not None

        if bw_graph is None:
            assert suffix is None
            with dynamo_timed(
                "invoke_subgraph_trace_joint_graph", log_pt2_compile_event=True
            ):
                bw_graph = trace_joint_graph_as_bwd(
                    subgraph,
                    len(primals),
                    primals_and_tangents,
                    ctx._fw_include_key_set,
                    ctx._fw_exclude_key_set,
                )

        if invoke_subgraph_cache and not cache_hit:
            suffix = invoke_subgraph_cache.add_lazy_bwd_entry(
                identifier, tangent_metadata, bw_graph
            )

        grads = invoke_subgraph(
            bw_graph, f"bw_{identifier}_{suffix}", *primals_and_tangents
        )[: -output_metadata.num_fw_outs]
        return None, None, None, *grads


@invoke_subgraph.py_autograd_impl
def _(subgraph, identifier, *operands):
    # Check if we have already traced the subgraph.
    invoke_subgraph_cache = get_invoke_subgraph_cache()
    if invoke_subgraph_cache:
        if saved_autograd_fn := invoke_subgraph_cache.get_autograd_key_entry(
            identifier
        ):
            return saved_autograd_fn(*operands)

    output_metadata = get_output_metadata(subgraph, *operands)

    def autograd_fn_callable(*args):
        return InvokeSubgraphAutogradOp.apply(
            subgraph, identifier, output_metadata, *args
        )

    # Save the autograd_fn_callable in the dispatch set cache.
    if invoke_subgraph_cache:
        invoke_subgraph_cache.add_autograd_key_entry(identifier, autograd_fn_callable)

    return autograd_fn_callable(*operands)


@invoke_subgraph.py_impl(DispatchKey.CompositeExplicitAutograd)
def _(subgraph, identifier, *operands):
    from torch.utils._python_dispatch import _get_current_dispatch_mode

    mode = _get_current_dispatch_mode()
    assert mode is None, "Mode should never be enabled for CPU/CUDA key"
    return subgraph(*operands)


@invoke_subgraph.py_functionalize_impl
def _(ctx, subgraph, identifier, *operands):
    unwrapped_operands = ctx.unwrap_tensors(operands)
    with ctx.redispatch_to_next():
        # NB: There is an assumption that subgraph does not mutate inputs and
        # there is no aliasing. Its Dynamo responsibility to prevent formation
        # of invoke_subgraph ops if input aliasing/mutation is detected.
        functionalized_subgraph = FunctionalizeCtxWrapper(ctx, subgraph)
        out = invoke_subgraph(functionalized_subgraph, identifier, *unwrapped_operands)
    return ctx.wrap_tensors(out)


# Register the hop fake fn. This will be called in the fake_tensor _dispatch_impl.
@register_fake(invoke_subgraph)
def _(subgraph, identifier, *operands):
    from torch._dynamo.utils import dynamo_timed

    with dynamo_timed("invoke_subgraph_fake_tensor", log_pt2_compile_event=True):
        return subgraph(*operands)


@invoke_subgraph.py_impl(ProxyTorchDispatchMode)
def _(proxy_mode: ProxyTorchDispatchMode, subgraph, identifier, *operands):
    # Check if we have already traced the subgraph.
    graph = None
    invoke_subgraph_cache = get_invoke_subgraph_cache()
    if invoke_subgraph_cache:
        graph = invoke_subgraph_cache.get_proxy_dispatch_entry(identifier)

    if graph is None:
        from torch._dynamo.utils import dynamo_timed

        with dynamo_timed("invoke_subgraph_proxy_tensor", log_pt2_compile_event=True):
            graph = reenter_make_fx(subgraph)(*operands)

        from torch._guards import detect_fake_mode

        fake_mode = detect_fake_mode(operands)
        insert_deferred_runtime_asserts(
            graph,
            fake_mode.shape_env,
            "invoke_subgraph_proxy_torch_dispatch_mode",
            export=True,
        )
        graph.recompile()

        assert isinstance(proxy_mode.tracer, torch.fx.Tracer)
        qualname = proxy_mode.tracer.get_fresh_qualname("repeated_subgraph")
        proxy_mode.tracer.root.register_module(qualname, graph)
        if invoke_subgraph_cache:
            invoke_subgraph_cache.add_proxy_dispatch_entry(identifier, graph)

    node_args = (graph, identifier, *operands)
    proxy_args = pytree.tree_map(proxy_mode.tracer.unwrap_proxy, node_args)  # type: ignore[union-attr]
    out_proxy = proxy_mode.tracer.create_proxy(
        "call_function", invoke_subgraph, proxy_args, {}
    )

    example_out = invoke_subgraph(graph, identifier, *operands)
    return track_tensor_tree(
        example_out, out_proxy, constant=None, tracer=proxy_mode.tracer
    )
