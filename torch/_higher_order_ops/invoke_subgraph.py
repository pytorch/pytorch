# mypy: allow-untyped-defs

import contextlib
import copy
import functools
from collections import defaultdict
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any, Optional, Union

import torch
import torch.utils._pytree as pytree
from torch._C import DispatchKey
from torch._dispatch.python import suspend_functionalization
from torch._higher_order_ops.utils import (
    _from_fun,
    _maybe_reenter_make_fx,
    clone_outputs_aliasing_inputs,
    FunctionalizeCtxWrapper,
    get_dummy_aot_autograd_config,
    HopInstance,
    prepare_fw_with_masks,
    redirect_to_mode,
    reenter_make_fx,
    register_fake,
    save_values_for_backward,
    saved_values,
)
from torch._library.fake_class_registry import FakeScriptObject
from torch._library.opaque_object import is_opaque_type
from torch._ops import HigherOrderOperator
from torch._subclasses.functional_tensor import disable_functional_mode
from torch.fx.experimental.proxy_tensor import (
    disable_proxy_modes_tracing,
    ProxyTorchDispatchMode,
    track_tensor_tree,
)
from torch.fx.graph_module import GraphModule
from torch.fx.passes.runtime_assert import insert_deferred_runtime_asserts
from torch.utils._debug_mode import DebugMode
from torch.utils.checkpoint import _CachedTorchDispatchMode, _CachingTorchDispatchMode


invoke_subgraph_counter = 0


# During the tracing of the joint graph, we construct this information. This is
# used to filter out grad_outs/tangents in the `backward` method of
# InvokeSubgraphAutogradOp.
@dataclass
class OutputMetadata:
    num_fw_outs: Optional[int] = None
    indexes_with_symint: set[int] = field(default_factory=set)
    indexes_with_no_grad: set[int] = field(default_factory=set)


# This config will be stored in invoke_subgraph HOP node.meta["custom"]["nested_region_config"]
# as well as the subgraph's gm.meta["nested_region_config"].
@dataclass
class NestedCompileRegionOptions:
    # A Callable that takes (gm, example_inputs, decompositions=None, **kwargs) as inputs.
    # Returns AOTCompiledArtifact
    fw_compiler: Optional[Callable] = None
    bw_compiler: Optional[Callable] = None

    # Note: [InvokeSubgraphHOP Partitioner]
    # If not None, add "partitioner" to HOP node meta.
    # If Callable, directly assign the callable, but the callable cannot be pickled
    # If str, the options are "default_partition" and "min_cut_rematerialization_partition".
    # The HOP joint graph will be partitioned using the corresponding functions in
    # torch/_functorch/partitioners.py
    partitioner: Optional[Callable | str] = None

    # If it's None, we'll inherit the parent call's decompositions.
    # Otherwise, the nested region will use this decompositions.
    decompositions: Optional[dict[str, Any]] = None


def _extract_nested_region_config(fn):
    """
    Extract the NestedCompileRegionOptions from the HOP subgraph gm.meta["nested_region_config"]
    """
    gm_to_compile = None
    if isinstance(fn, torch.fx.GraphModule):
        gm_to_compile = fn
    elif isinstance(fn, FunctionalizeCtxWrapper):
        gm_to_compile = fn.subgraph

    if (
        isinstance(gm_to_compile, torch.fx.GraphModule)
        and hasattr(gm_to_compile, "meta")
        and "nested_region_config" in gm_to_compile.meta
    ):
        if isinstance(
            gm_to_compile.meta["nested_region_config"], NestedCompileRegionOptions
        ):
            return gm_to_compile.meta["nested_region_config"].decompositions
    return None


class InvokeSubgraphHOP(HigherOrderOperator):
    def __init__(self) -> None:
        # Invoke subgraph does not have any state, it is just a wrapper over a
        # subgraph, so we can safely cache the HOP.
        super().__init__("invoke_subgraph", cacheable=True)
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
        assert identifier is None or isinstance(identifier, str), (
            "identifier must be a None or a string"
        )

        assert all(
            isinstance(
                o, (torch.Tensor, int, torch.SymInt, torch.Generator, FakeScriptObject)
            )
            or is_opaque_type(type(o))
            for o in operands
            if o is not None
        ), (
            f"invoke_subgraph operands must be a list of tensors/ints/SymInts/Generator/opaque objects {operands}"
        )

        # pyrefly: ignore [missing-attribute]
        return super().__call__(subgraph, identifier, *operands)

    # pyrefly: ignore [bad-override]
    def gen_schema(self, subgraph, identifier, *operands):
        from torch._higher_order_ops.schema import HopSchemaGenerator
        from torch._higher_order_ops.utils import (
            check_input_alias_and_mutation_return_outputs,
            materialize_as_graph,
        )

        subgraph_decomp_table = _extract_nested_region_config(subgraph)
        gm: torch.fx.GraphModule = materialize_as_graph(
            subgraph, operands, subgraph_decomp_table=subgraph_decomp_table
        )

        schema_gen = HopSchemaGenerator(self)
        schema_gen.add_arg("subgraph", gm)
        schema_gen.add_arg("identifier", identifier)
        (
            _,
            _,
            _,
            mutated_inputs,
            outputs,
        ) = check_input_alias_and_mutation_return_outputs(gm)
        for idx, arg in enumerate(operands):
            schema_gen.add_arg(f"arg{idx}", arg, is_mutated=idx in mutated_inputs)
        for out in outputs:
            schema_gen.add_output(out)

        return schema_gen.gen_schema()


invoke_subgraph = InvokeSubgraphHOP()


# Registers dispatches for SAC
redirect_to_mode(invoke_subgraph, _CachingTorchDispatchMode)
redirect_to_mode(invoke_subgraph, _CachedTorchDispatchMode)


def invoke_subgraph_placeholder(func, *args, **kwargs):
    if torch.compiler.is_dynamo_compiling():
        # This is just a placeholder for Dynamo to replace with invoke_subgraph
        raise RuntimeError("invoke_subgraph should not be called directly in Dynamo")

    if torch.compiler.is_compiling():
        # For non-strict export tracing, we still want to go through Dynamo

        def _invoke_subgraph_placeholder_wrapper(func, args):
            return invoke_subgraph_placeholder(func, *args)

        from torch._higher_order_ops.utils import setup_compilation_env

        with setup_compilation_env() as backend:
            return torch.compile(
                _invoke_subgraph_placeholder_wrapper,
                backend=backend,
                fullgraph=True,
            )(func, args)

    return func(*args, **kwargs)


def mark_compile_region(fn=None, options: Optional[NestedCompileRegionOptions] = None):
    """
    This wrapper instructs torch.compile to compile the wrapped region once and
    reuse the compiled artifact, instead of the usual way of aggressively
    inlining the function.

    Under the hood, it tells TorchDynamo to use InvokeSubgraph HOP for the
    region. For PyTorch eager, this is a no-op.

    Args:
        fn: The function to wrap
        options: Optional config to use for compiling the subgraph.
            Warning: this is an experimental feature under development and
            not ready for use yet.
    """

    def wrap(func):
        def inner(*args, **kwargs):
            # Get the innermost function to avoid nested compile regions
            inner_func = func
            while hasattr(inner_func, "__marked_compile_region_fn__"):
                inner_func = inner_func.__marked_compile_region_fn__
            return invoke_subgraph_placeholder(inner_func, *args, **kwargs)

        inner.__marked_compile_region_fn__ = func  # type: ignore[attr-defined]
        func.__marked_compile_region_config__ = options  # type: ignore[attr-defined]

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
                if isinstance(fw_out, torch.SymInt):
                    output_metadata.indexes_with_symint.add(idx)
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
    """
    Extract metadata about the subgraph outputs WITHOUT executing the subgraph.
    This avoids running side-effectful operations twice (once here, once in forward).
    We analyze the graph structure statically to extract metadata.
    """
    # Unwrap FunctionalizeCtxWrapper if present
    if isinstance(subgraph, FunctionalizeCtxWrapper):
        subgraph = subgraph.subgraph

    # If not a GraphModule, fall back to execution-based metadata extraction
    if not isinstance(subgraph, torch.fx.GraphModule):
        return _get_output_metadata_by_execution(subgraph, *operands)

    output_metadata = OutputMetadata()

    # Extract output arguments from the output node
    # The output node has args=(output_values,) where output_values is a tuple/list
    output_node = next(reversed(subgraph.graph.find_nodes(op="output")))
    output_metadata.num_fw_outs = len(output_node.args[0])

    for idx, output_arg in enumerate(output_node.args[0]):
        if not isinstance(output_arg, torch.fx.Node):
            if isinstance(output_arg, int):
                output_metadata.indexes_with_symint.add(idx)
            output_metadata.indexes_with_no_grad.add(idx)
            continue

        # Check node metadata for type information
        if output_arg.meta.get("val") is None:
            # If we don't have complete metadata for all outputs, fall back to execution
            # This is important for correctness (e.g., detecting SymInts) even though it
            # runs side-effectful operations
            return _get_output_metadata_by_execution(subgraph, *operands)

        val = output_arg.meta["val"]
        if isinstance(val, torch.SymInt):
            output_metadata.indexes_with_symint.add(idx)
            output_metadata.indexes_with_no_grad.add(idx)
        elif isinstance(val, torch.Tensor):
            # Check if tensor requires grad from metadata
            if hasattr(val, "requires_grad") and not val.requires_grad:
                output_metadata.indexes_with_no_grad.add(idx)
        else:
            # Non-tensor, non-symint (shouldn't happen but be safe)
            output_metadata.indexes_with_no_grad.add(idx)

    return output_metadata


def _get_output_metadata_by_execution(subgraph, *operands):
    """
    Fallback: Extract metadata by executing the subgraph.
    This should only be used when static analysis fails.
    WARNING: This will run side-effectful operations!
    """

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

            output_metadata = OutputMetadata()
            output_metadata.num_fw_outs = num_fw_outs

            for idx, fw_out in enumerate(fw_outs):
                if isinstance(fw_out, torch.SymInt):
                    output_metadata.indexes_with_symint.add(idx)
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
                subgraph_decomp_table = _extract_nested_region_config(subgraph)
                with torch.enable_grad():
                    return _maybe_reenter_make_fx(
                        joint_fn, subgraph_decomp_table=subgraph_decomp_table
                    )(*joint_operands)


class InvokeSubgraphAutogradOp(torch.autograd.Function):
    """
    Saves the subgraph, i.e. original callable, in the forward method. And then
    traces out a joint graph in the backward. This delaying of tracing in
    backward, also called as lazy backward, ensures that the assumptions about
    the grad_out strides and tensor-subclass-ness are already accounted for.
    """

    @staticmethod
    # pyrefly: ignore [bad-override]
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

        save_values_for_backward(ctx, operands)

        with torch._C._AutoDispatchBelowAutograd():
            out = invoke_subgraph(
                subgraph,
                f"fw_{identifier}",
                *operands,
            )

        # Check that int (coming from symint) is at expected indexes.
        for idx, o in enumerate(out):
            if isinstance(o, int):
                assert idx in output_metadata.indexes_with_symint

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
        primals = saved_values(ctx)

        # Filter out grads that are None or do not require_grad. This was
        # the assumption we made during the tracing of joint_graph.
        filtered_grad_outs = []
        for idx, o in enumerate(grad_outs):
            if o is None:
                assert idx in output_metadata.indexes_with_symint
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
        assert fake_mode is not None, "fake_mode should be enabled for HOPs"
        state = _CacheKeyState(fake_mode.shape_env)

        tangent_metadata: list[object] = []
        for tangent in filtered_grad_outs:
            metadata = extract_tensor_metadata(tangent)
            metadata._flatten_into(tangent_metadata, fake_mode, state)

        # Add aliasing information to tangent_metadata
        # Two tangents are aliased if they are the same tensor object (using id())
        # We create a tuple of tuples where each inner tuple contains indices of aliased tensors
        # e.g. ((0, 1),) would mean there is one aliasing group, and the first and second tangents are aliased
        # e.g. () would mean there is no aliasing between tangents
        tensor_to_indices: dict[int, list[int]] = defaultdict(list)
        for i, tangent in enumerate(filtered_grad_outs):
            if isinstance(tangent, torch.Tensor):
                tensor_to_indices[id(tangent)].append(i)

        aliasing_groups = tuple(
            sorted(
                tuple(indices)
                for indices in tensor_to_indices.values()
                if len(indices) > 1
            )
        )
        tangent_metadata.append(aliasing_groups)

        # pyrefly: ignore [bad-assignment]
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
                if (
                    hasattr(subgraph, "meta")
                    and "nested_region_config" in subgraph.meta
                ):
                    bw_graph.meta["nested_region_config"] = subgraph.meta[
                        "nested_region_config"
                    ]

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


@invoke_subgraph.py_impl(DebugMode)
def _(debug_mode, subgraph, identifier, *operands):
    # record HOP call
    call = torch.utils._debug_mode._OpCall(
        invoke_subgraph,
        (identifier, *operands),
        kwargs={},
        call_depth=debug_mode.call_depth + 1,
        stack=debug_mode.record_stack_trace,
    )
    debug_mode._record_call(call)

    debug_mode.call_depth += 1
    debug_mode._handle_annotate(f"[enter InvokeSubgraph HOP] {identifier}")

    # If the HOP is dispatched from DebugMode, we should enable debug_mode
    # for the subgraph call.
    with debug_mode:
        if getattr(subgraph, "_boxed_call", False):
            result = subgraph(list(operands))
        else:
            result = subgraph(*operands)
    debug_mode._handle_annotate(f"[exit InvokeSubgraph HOP] {identifier}")
    debug_mode.call_depth -= 1
    # record output of HOP
    debug_mode._record_call_output(call, result)
    return result


@invoke_subgraph.py_impl(DispatchKey.CompositeExplicitAutograd)
def _(subgraph, identifier, *operands):
    from torch.utils._python_dispatch import _get_current_dispatch_mode

    mode = _get_current_dispatch_mode()
    assert mode is None, "Mode should never be enabled for CPU/CUDA key"

    if getattr(subgraph, "_boxed_call", False):
        return subgraph(list(operands))
    else:
        return subgraph(*operands)


@invoke_subgraph.py_functionalize_impl
def _(ctx, subgraph, identifier, *operands):
    from torch._higher_order_ops.auto_functionalize import (
        can_auto_functionalize,
        do_auto_functionalize_v2,
    )

    # (in the functionalization metadata phase) Capture tokens before
    tokens_before = dict(ctx.mode._tokens)

    # Check if this subgraph has effects stored in the cache
    invoke_subgraph_cache = get_invoke_subgraph_cache()
    effects = None
    if invoke_subgraph_cache:
        effects = invoke_subgraph_cache.get_effects(identifier)

    if effects:
        assert len(effects) == 1, "Multiple effects within a subgraph NYI"
        tokens = ctx.mode._tokens
        effects = next(iter(effects))
        token_input = tokens[effects]

        operands = (token_input, *operands)

        def wrap_subgraph(subgraph):
            def wrapped_subgraph(token, *args):
                res = subgraph(*args)
                return ctx.unwrap_tensors(ctx.mode._tokens[effects]), *res

            return wrapped_subgraph

        subgraph = wrap_subgraph(subgraph)

    unwrapped_operands = ctx.unwrap_tensors(operands)

    hop_instance = HopInstance.create(invoke_subgraph, subgraph, identifier, *operands)
    if can_auto_functionalize(hop_instance):
        # NOTE: [auto_functionalize x invoke_subgraph caching]
        # We call auto_functionalized_v2 to support input mutation of invoke_subgraph.
        # See NOTE [Support input mutation of hops] for the overall design.
        #
        # invoke_subgraph is special because of its identifier based caching mechanism.
        # In invoke_subgraph's functionalization key implementation, we create a new
        # identifier because the subgraph is replaced by FunctionWithNoFreeVars in a
        # functional + epilogue form.
        assert isinstance(identifier, str), identifier
        return do_auto_functionalize_v2(
            ctx.mode,
            hop_instance,
            (subgraph, "auto_functionalized_" + identifier, *operands),
            {},
        )

    with ctx.redispatch_to_next():
        # NB: There is an assumption that subgraph does not mutate inputs and
        # there is no aliasing. It's Dynamo's responsibility to prevent formation
        # of invoke_subgraph ops if input aliasing/mutation is detected.
        functionalized_subgraph = FunctionalizeCtxWrapper(ctx, subgraph)
        out = invoke_subgraph(functionalized_subgraph, identifier, *unwrapped_operands)

    if effects:
        (new_token, *out) = out
        ctx.mode._tokens[effects] = new_token

    # (in the functionalization metadata phase) Capture tokens after and see if
    # there are any differences (there are new effects or the token value for an
    # effect type has changed)
    tokens_after = dict(ctx.mode._tokens)
    discovered_effects = set()
    for effect_type, token in tokens_after.items():
        if effect_type not in tokens_before or tokens_before[effect_type] is not token:
            discovered_effects.add(effect_type)

    if discovered_effects:
        assert ctx.mode._allow_token_discovery, (
            f"Number of tokens changed by {len(discovered_effects)} when tracing subgraph {subgraph}."
        )
        # Store discovered effects in the cache by identifier
        if invoke_subgraph_cache:
            invoke_subgraph_cache.add_effects(identifier, discovered_effects)

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
            subgraph_decomp_table = _extract_nested_region_config(subgraph)
            graph = reenter_make_fx(
                subgraph, subgraph_decomp_table=subgraph_decomp_table
            )(*operands)

        from torch._guards import detect_fake_mode

        fake_mode = detect_fake_mode(operands)
        # Only insert deferred runtime asserts when we have dynamic shapes.
        # When shape_env is None (static shapes), there are no deferred asserts to insert.
        if fake_mode is not None and fake_mode.shape_env is not None:
            insert_deferred_runtime_asserts(
                graph,
                fake_mode.shape_env,
                "invoke_subgraph_proxy_torch_dispatch_mode",
                export=True,
            )
            graph.recompile()

        assert isinstance(proxy_mode.tracer, torch.fx.Tracer)
        if invoke_subgraph_cache:
            invoke_subgraph_cache.add_proxy_dispatch_entry(identifier, graph)

    node_args = (graph, identifier, *operands)

    def _unwrap_proxy(arg):
        if isinstance(arg, torch.fx.GraphModule):
            # NOTE: [invoke_subgraph proxy_mode x auto_functionalize]
            # Previously, we assumed that `invoke_subgraph` would always be traced with the same tracer.
            # This allowed us to cache modules by their identifiers, assuming they were already registered.
            #
            # However, this assumption no longer holds when we auto-functionalize `invoke_subgraph`.
            # auto_functionalize functionalizes the subgraph and wrap it with `FunctionWithNoFreeVars`.
            # In the proxy mode implementation of `auto_functionalized_v2`, we need to materialize `FunctionWithNoFreeVars`
            # input as a graph module. To do this, we re-trace the `invoke_subgraph` hop, which starts a new sub-tracer
            # (see NOTE [materialize callable inputs as graph]). # When the new sub-tracer traces the `invoke_subgraph`
            # with a previously cached identifier, the corresponding graph module might not
            # exist as a submodule in the new tracer's root. Therefore, we register it as a submodule below.
            #
            # The alternative is to give a new identifier when we re-trace the invoke_subgraph but this will increase
            # the compilation time, which defeats the purpose of caching.
            registered_before = False
            for (
                _,
                submod,
            ) in proxy_mode.tracer.root.named_modules():  # type: ignore[union-attr]
                if arg is submod:
                    registered_before = True

            if not registered_before:
                qualname = proxy_mode.tracer.get_fresh_qualname("repeated_subgraph")  # type: ignore[union-attr]
                proxy_mode.tracer.root.register_module(qualname, arg)  # type: ignore[union-attr]
        return proxy_mode.tracer.unwrap_proxy(arg)  # type: ignore[union-attr]

    proxy_args = pytree.tree_map(_unwrap_proxy, node_args)  # type: ignore[union-attr]
    out_proxy = proxy_mode.tracer.create_proxy(
        "call_function", invoke_subgraph, proxy_args, {}
    )

    example_out = invoke_subgraph(graph, identifier, *operands)
    return track_tensor_tree(
        example_out, out_proxy, constant=None, tracer=proxy_mode.tracer
    )


def invoke_subgraph_inductor_compile(
    gm, example_inputs, inductor_config_patches=None, **kwargs
):
    from torch._functorch._aot_autograd.runtime_wrappers import (
        SerializableCompiledFunction,
    )
    from torch._functorch._aot_autograd.utils import simple_wraps
    from torch._inductor import config
    from torch._inductor.compile_fx import compile_fx_inner
    from torch._inductor.standalone_compile import AOTCompiledArtifact

    # Used for testing only, should only be changed via _testing_capture_invoke_subgraph_inductor_compile_gms()
    if (
        torch._dynamo.testing._testing_invoke_subgraph_inductor_compile_captured_gms
        is not None
    ):
        torch._dynamo.testing._testing_invoke_subgraph_inductor_compile_captured_gms.append(
            copy.deepcopy(gm)
        )

    if inductor_config_patches is None:
        inductor_config_patches = {}
    compile_fn = config.patch(inductor_config_patches)(compile_fx_inner)
    compiled_fn_inner = compile_fn(gm, example_inputs)
    assert compiled_fn_inner._boxed_call

    # Follow boxed calling convention
    @simple_wraps(compiled_fn_inner)
    def forward(*runtime_args: tuple[Any]):
        full_args = []
        full_args.extend(runtime_args)
        return compiled_fn_inner(full_args)

    # Just for convenience
    forward.zero_grad = gm.zero_grad  # type: ignore[attr-defined]
    forward.named_parameters = gm.named_parameters  # type: ignore[attr-defined]
    forward.named_buffers = gm.named_buffers  # type: ignore[attr-defined]

    # TODO: Do we need the post compile passes in _aot_stage2b_compile_forward_or_inference?
    # TODO: add a real serialize function for SerializableCompiledFunction like _cache_inference_info
    forward.serialize = SerializableCompiledFunction(forward, lambda: None)  # type: ignore[attr-defined]
    return AOTCompiledArtifact(forward)


def get_invoke_subgraph_compile_options(
    inductor_config_patches=None,
    decompositions=None,
    partitioner="min_cut_rematerialization_partition",
):
    if inductor_config_patches is None:
        inductor_config_patches = {"triton.autotune_at_compile_time": True}
    inductor_compile = functools.partial(
        invoke_subgraph_inductor_compile,
        inductor_config_patches=inductor_config_patches,
    )

    if inductor_config_patches:
        from torch._inductor import config as inductor_config

        # Validate that all config keys exist
        for key in inductor_config_patches:
            if not hasattr(inductor_config, key):
                raise ValueError(
                    f"Invalid inductor config key '{key}' in get_invoke_subgraph_compile_options. "
                    f"Available config keys can be found in torch._inductor.config"
                )

    return NestedCompileRegionOptions(
        fw_compiler=inductor_compile,
        bw_compiler=inductor_compile,
        partitioner=partitioner,
        decompositions=decompositions,
    )
