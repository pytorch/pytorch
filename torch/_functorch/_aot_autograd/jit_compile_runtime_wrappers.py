"""
Functions in this module do most of the "work" of AOTAutograd.
An aot_dispatch_* function:
- Takes in the input flat_fn, flat_args, and some metadata
- Runs a set of pre compile wrappers (e.g. argument deduping)
- Runs the actual compiler
- Wraps the returned callable in a set of post compile wrappers
- Returns the wrapped callable and metadata.
"""

import logging
from contextlib import nullcontext

from typing import Any, Callable, List, Optional, Sequence, Tuple

import torch
import torch.utils.dlpack
from torch import Tensor
from torch._dynamo.utils import lazy_format_graph_code
from torch._guards import CompileContext, TracingContext
from torch._logging import getArtifactLogger, trace_structured
from torch.fx.experimental._backward_state import BackwardState
from torch.fx.experimental.proxy_tensor import is_sym_node
from torch.fx.experimental.symbolic_shapes import fx_placeholder_vals
from .. import config
from .dispatch_and_compile_graph import (
    aot_dispatch_autograd_graph,
    aot_dispatch_base_graph,
)
from .logging_utils import track_graph_compiling

from .runtime_wrappers import (
    AOTDedupeWrapper,
    AOTDispatchAutograd,
    AOTDispatchSubclassWrapper,
    AOTSyntheticBaseWrapper,
    AutogradLazyBackwardCompileInfo,
    CompilerWrapper,
    DebugAssertWrapper,
    FakifiedOutWrapper,
    FunctionalizedRngRuntimeWrapper,
    post_compile,
    pre_compile,
    RuntimeWrapper,
)
from .schemas import AOTConfig, MutationType, ViewAndMutationMeta
from .subclass_utils import compute_inner_mutated_inp_indices_from_subclass_meta

from .utils import _get_symint_hints, make_boxed_func, strict_zip, unlift_tokens

zip = strict_zip

log = logging.getLogger(__name__)
aot_joint_log = getArtifactLogger(__name__, "aot_joint_graph")
aot_graphs_log = getArtifactLogger(__name__, "aot_graphs")

aten = torch.ops.aten

# Returns a Callable and a ViewAndMutationMeta.
# Currently, only export needs the ViewAndMutationMeta after this function.
DispatchReturn = Tuple[Callable, ViewAndMutationMeta]


def _create_wrappers_for_dispatch(needs_autograd: bool) -> List[CompilerWrapper]:
    """
    Wrappers that run on every dispatch function
    """
    return [AOTDedupeWrapper(), AOTSyntheticBaseWrapper(trace_joint=needs_autograd)]


# Export's dispatching logic is unique in a few ways: it only needs the "graph"
# bits of aot_autograd, and doesn't need to do any specific wrapping.
def aot_dispatch_export(
    flat_fn: Callable,
    flat_args: List[Any],
    aot_config: AOTConfig,
    *,
    fw_metadata: ViewAndMutationMeta,
    needs_autograd: bool,
) -> DispatchReturn:
    wrappers = _create_wrappers_for_dispatch(needs_autograd)
    flat_fn, flat_args, fw_metadata = pre_compile(
        wrappers,
        flat_fn,
        flat_args,
        aot_config,
        fw_metadata=fw_metadata,
    )
    if needs_autograd and not aot_config.pre_dispatch:
        graph, _, _ = aot_dispatch_autograd_graph(
            flat_fn, flat_args, aot_config, fw_metadata=fw_metadata
        )
    else:
        graph, _, _ = aot_dispatch_base_graph(
            flat_fn, flat_args, aot_config, fw_metadata=fw_metadata
        )

    # NB: the wrappers that run in pre_compile for export are
    # either a no-op, because they're not needed, or will raise a runtime error,
    # since they don't support export.
    # We still run these wrappers to make sure that they're not needed pre compile,
    # but we technically don't need to run them post compile at all here.
    compiled_fn, fw_metadata = post_compile(
        wrappers, graph, aot_config, runtime_metadata=fw_metadata
    )

    # Therefore, since no wrapperes run, we don't get back a callable - we get back the raw fx graph
    # (either a joint or an inference-only graph)
    assert isinstance(compiled_fn, torch.fx.GraphModule)
    return compiled_fn, fw_metadata


def aot_dispatch_base(
    flat_fn,
    flat_args: List[Any],
    aot_config: AOTConfig,
    *,
    fw_metadata: ViewAndMutationMeta,
) -> DispatchReturn:
    """
    Handles functions that don't need autograd. Runs wrappers and compiles with fw_compiler.
    """
    wrappers = _create_wrappers_for_dispatch(needs_autograd=False)
    flat_fn, flat_args, fw_metadata = pre_compile(
        wrappers, flat_fn, flat_args, aot_config, fw_metadata=fw_metadata
    )

    fw_module, updated_flat_args, maybe_subclass_meta = aot_dispatch_base_graph(  # type: ignore[misc]
        flat_fn, flat_args, aot_config, fw_metadata=fw_metadata
    )

    fakified_out_wrapper = FakifiedOutWrapper()
    (
        fw_module,
        updated_flat_args,
        fw_metadata,
    ) = fakified_out_wrapper.pre_compile(
        fw_module, updated_flat_args, aot_config, fw_metadata=fw_metadata
    )
    functionalized_rng_wrapper = FunctionalizedRngRuntimeWrapper()
    (
        fw_module,
        updated_flat_args,
        fw_metadata,
    ) = functionalized_rng_wrapper.pre_compile(
        fw_module, updated_flat_args, aot_config, fw_metadata=fw_metadata
    )

    disable_amp = torch._C._is_any_autocast_enabled()
    context = torch._C._DisableAutocast if disable_amp else nullcontext

    with context(), track_graph_compiling(aot_config, "inference"):
        compiler = (
            aot_config.inference_compiler
            if aot_config.inference_compiler is not None
            else aot_config.fw_compiler
        )

        if tracing_context := torch._guards.TracingContext.try_get():
            tracing_context.fw_metadata = (
                fw_metadata
                if maybe_subclass_meta is None
                else maybe_subclass_meta.fw_metadata
            )

        with TracingContext.report_output_strides() as fwd_output_strides:
            compiled_fw = compiler(fw_module, updated_flat_args)

        if fakified_out_wrapper.needs_post_compile:
            fakified_out_wrapper.set_fwd_output_strides(fwd_output_strides)

    # However, RuntimeWrapper does not expect the rng offsets in the
    # output. So, we have to create another wrapper and take out the offset. As
    # a result, we have to account for not boxed_call compilers as well.
    if not hasattr(compiled_fw, "_boxed_call"):
        compiled_fw = make_boxed_func(compiled_fw)

    # Create a wrapper to set up the rng functionalize and fakified out bits
    compiled_fw = functionalized_rng_wrapper.post_compile(
        compiled_fw, aot_config, runtime_metadata=fw_metadata
    )
    compiled_fw = fakified_out_wrapper.post_compile(
        compiled_fw,
        aot_config,
        runtime_metadata=fw_metadata,
    )
    # Why do we need to pass in num_fw_outs_saved_for_bw?
    # See Note: [Partitioner handling for Subclasses, Part 2]
    compiled_fw_func = AOTDispatchSubclassWrapper(
        trace_joint=False,
        # TODO: once we use pre_compile this will be flat_fn at the top of this function
        fw_only=None,
        maybe_subclass_meta=maybe_subclass_meta,
        num_fw_outs_saved_for_bw=None,
    ).post_compile(
        compiled_fw,
        aot_config,  # not used
        runtime_metadata=fw_metadata,
    )

    if not hasattr(compiled_fw_func, "_boxed_call"):
        compiled_fw_func = make_boxed_func(compiled_fw_func)

    compiled_fn = RuntimeWrapper(
        indices_of_inps_to_detach=[],
        trace_joint=False,
        disable_amp=disable_amp,
    ).post_compile(
        compiled_fw_func,
        aot_config,
        runtime_metadata=fw_metadata,
    )

    compiled_fn = post_compile(
        wrappers, compiled_fn, aot_config, runtime_metadata=fw_metadata
    )
    return compiled_fn


def aot_dispatch_autograd(
    flat_fn,
    flat_args: List[Any],
    aot_config: AOTConfig,
    *,
    fw_metadata: ViewAndMutationMeta,
) -> DispatchReturn:
    """
    Autograd logic. Generates a joint graph, partitions it, manipulates the input with various wrappers,
    and returns a wrapped torch.autograd.Function with a forward and backward.
    """
    wrappers = _create_wrappers_for_dispatch(needs_autograd=True)
    flat_fn, flat_args, fw_metadata = pre_compile(
        wrappers,
        flat_fn,
        flat_args,
        aot_config,
        fw_metadata=fw_metadata,
    )

    fw_metadata.deterministic = torch.are_deterministic_algorithms_enabled()
    fx_g, joint_inputs, maybe_subclass_meta = aot_dispatch_autograd_graph(
        flat_fn, flat_args, aot_config, fw_metadata=fw_metadata
    )

    # Copied from aot_dispatch_autograd_graph.
    disable_amp = torch._C._is_any_autocast_enabled()

    if aot_config.enable_log:
        aot_joint_log.info(
            "%s", lazy_format_graph_code("Joint graph", fx_g, aot_config.aot_id)
        )
        trace_structured(
            "aot_joint_graph",
            payload_fn=lambda: fx_g.print_readable(print_output=False),
        )

    with torch.no_grad():
        inner_meta = (
            fw_metadata
            if maybe_subclass_meta is None
            else maybe_subclass_meta.fw_metadata
        )
        with track_graph_compiling(aot_config, "joint"):
            # See Note: [Partitioner handling for Subclasses, Part 1]
            # See Note: [Recomputing subclass mutation handling]
            mutated_inp_runtime_indices = (
                compute_inner_mutated_inp_indices_from_subclass_meta(
                    fw_metadata, inner_meta
                )
            )
            num_tokens = len(fw_metadata.tokens)
            num_mutated_inp_runtime_indices = len(mutated_inp_runtime_indices)
            num_inner_fwd_outputs = (
                num_mutated_inp_runtime_indices
                + inner_meta.num_outputs
                + inner_meta.num_intermediate_bases
                + inner_meta.num_outputs_rng_offset
                + num_tokens  # See Note [Side-Effectful Tokens in AOTAutograd]
            )
            fw_module, bw_module = aot_config.partition_fn(
                fx_g, joint_inputs, num_fwd_outputs=num_inner_fwd_outputs
            )

            # See Note [Side-Effectful Tokens in AOTAutograd]
            if num_tokens != 0 and config.unlift_effect_tokens:
                unlift_tokens(fw_module, fw_metadata)
                num_inner_fwd_outputs -= num_tokens
                joint_inputs = (joint_inputs[0][num_tokens:], joint_inputs[1])

            fw_outs = next(iter(fw_module.graph.find_nodes(op="output"))).args[0]
            # we only need to bookkeep the symints that are saved for bw, not any symints
            # the user forward might have returned in its own output
            fw_outs_saved_for_bw = fw_outs[num_inner_fwd_outputs:]
            num_fw_outs_saved_for_bw = len(fw_outs_saved_for_bw)
            symint_outs_saved_for_bw = [
                n for n in fw_outs_saved_for_bw if is_sym_node(n)
            ]
            fw_metadata.num_symints_saved_for_bw = len(symint_outs_saved_for_bw)
            inner_meta.num_symints_saved_for_bw = len(symint_outs_saved_for_bw)
            num_symints_saved_for_bw = len(symint_outs_saved_for_bw)

        # Note [Detaching inputs that never need gradients]
        # See https://github.com/pytorch/pytorch/issues/97745
        # Suppose we have a function like this that we want to compile:
        #
        # def f(x, y):
        #     return torch.mul(x, y.detach())
        #
        # What gradients should we compute for x and y?
        # By default, AOTAutograd will compute a gradient for **every** input that requires gradients,
        # and so we'll compute:
        #    x_grad_input = y
        #    y_grad_input = None
        # Does this preserve the semantics of eager mode?
        # Unfortunately, no.
        # Doing the above will cause autograd to **continue** to backprop the autograd tape
        # that was generated from constructing y.
        #
        # This is **different** from what would have happened in eager mode.
        # In eager mode, if we backprop through the output of this function, autograd will only traverse
        # the bit of the autograd tape corresponding to "x".
        # In particular, if a user had previously backpropped through y's autograd tape,
        # And then they try to backprop through the output of the above function,
        # then we'll hit the dreaded "Trying to backward through the graph a second time" error.
        #
        # You might think: If autograd sees that a gradient is None, shouldn't it stop early,
        # instead of continuing the backprop through the ancestors of that node in the graph?
        #
        # Autograd has two passes:
        # (1) a first pass that traverses the autograd graph and figures out which nodes need to be executed
        # (2) a second pass that actually goes ahead and executes each node when it becomes ready,
        #     propagating gradients
        # By the time we're executing a node and we see that it produces a None, the set of nodes to execute
        # is already locked-in.
        #
        # The fix: instead, we can recognize statically that the graph we're compiling will never contribute
        # gradients to y, and prevent autograd from trying to traverse y's autograd tape at all.
        # We can do this by manually detach'ing y before sending it through the `CompiledFunction`.
        #
        # Note that this solution is not bulletproof.
        # It's possible to construct a case where eager may or may not have have tried to autograd through y,
        # depending on the actual grad_outputs that were passed in during the backward.
        # There is no easy fix for this: the simplest fix would be to run with `retain_graph=True`,
        # allowing autograd to re-use the graph.
        #
        # An example of this case is:
        # def f(x):
        #     return x.detach() * 2, x * 3
        # If we were to only backprop through outs[0], in eager, we would stop
        # If we backward only on the first output, we shouldn't send a grad through x.
        # But the custom autograd function doesn't know that: it will materialize zero grads for x * 3
        # and we will end up with a zero grad at x.
        # If we later backprop through the second output, this will also require backprop'ing through x.
        # Meaning we'll need to use `retain_graph=True` to be able to backprop through x the second time.
        _indices_of_inps_to_detach: List[int] = []

        # reversed() since we expect output at end of graph
        bw_output = next(reversed(bw_module.graph.find_nodes(op="output")))
        bw_outs: Sequence[torch.fx.Node] = bw_output.args[0]  # type: ignore[assignment]

        # TODO: we should apply the below "detach inputs if their gradients are statically known to be None"
        # optimization even if we have subclass inputs/outputs (we do not handle this today).
        # Computing which our our inputs get None gradients is a bit more complicated,
        # if any of our inputs are subclasses. Why?
        # (a) we need to make sure that we call .detach() on the input subclasses, since autograd sees subclasses.
        # (b) The grad_outputs that we AOT computed in our backward graph are the desugared tensor tensors,
        #     so we need to figure out which subclass fw inputs they map to.
        if maybe_subclass_meta is None:
            assert (
                len(bw_outs)
                == len(fw_metadata.input_info) + inner_meta.num_outputs_rng_offset
            )
            bw_outs_no_rng = bw_outs
            if inner_meta.num_outputs_rng_offset > 0:
                bw_outs_no_rng = bw_outs[: -inner_meta.num_outputs_rng_offset]
            assert len(bw_outs_no_rng) == len(fw_metadata.input_info)

            for i, (bw_out) in enumerate(bw_outs_no_rng):
                # If our input experiences a metadata mutation inside the graph (e.g. set_()),
                # we *must* not detach, otherwise it will be the detach'd input that gets the metadata mutation
                metadata_mutation_in_graph = (
                    fw_metadata.input_info[i].mutation_type
                    == MutationType.MUTATED_IN_GRAPH
                    and fw_metadata.input_info[i].mutates_storage_metadata
                )
                if bw_out is None and not metadata_mutation_in_graph:
                    _indices_of_inps_to_detach.append(i)

        if aot_config.enable_log:
            aot_graphs_log.info(
                "%s",
                lazy_format_graph_code("Forward graph", fw_module, aot_config.aot_id),
            )
            aot_graphs_log.info(
                "%s",
                lazy_format_graph_code("Backward graph", bw_module, aot_config.aot_id),
            )
            trace_structured(
                "aot_forward_graph",
                payload_fn=lambda: fw_module.print_readable(print_output=False),
            )
            trace_structured(
                "aot_backward_graph",
                payload_fn=lambda: bw_module.print_readable(print_output=False),
            )

        with track_graph_compiling(aot_config, "forward"):
            # flat_args at this point might still be subclasses-
            # make sure to pass the unwrapped fake tensors into the compiler!
            adjusted_flat_args = joint_inputs[0]

            fakified_out_wrapper = FakifiedOutWrapper()
            (
                fw_module,
                adjusted_flat_args,
                fw_metadata,
            ) = fakified_out_wrapper.pre_compile(
                fw_module, adjusted_flat_args, aot_config, fw_metadata=fw_metadata
            )

            functionalized_rng_wrapper = FunctionalizedRngRuntimeWrapper(
                return_new_outs=False
            )
            (
                fw_module,
                adjusted_flat_args,
                fw_metadata,
            ) = functionalized_rng_wrapper.pre_compile(
                fw_module, adjusted_flat_args, aot_config, fw_metadata=fw_metadata
            )
            if tracing_context := torch._guards.TracingContext.try_get():
                tracing_context.fw_metadata = inner_meta

            with TracingContext.report_output_strides() as fwd_output_strides:
                compiled_fw_func = aot_config.fw_compiler(fw_module, adjusted_flat_args)

            if not hasattr(compiled_fw_func, "_boxed_call"):
                compiled_fw_func = make_boxed_func(compiled_fw_func)

            if fakified_out_wrapper.needs_post_compile:
                fakified_out_wrapper.set_fwd_output_strides(fwd_output_strides)

            compiled_fw_func = AOTDispatchSubclassWrapper(
                fw_only=None,
                trace_joint=False,
                maybe_subclass_meta=maybe_subclass_meta,
                num_fw_outs_saved_for_bw=num_fw_outs_saved_for_bw,
            ).post_compile(
                compiled_fw_func,
                aot_config,  # not used
                runtime_metadata=fw_metadata,
            )

            compiled_fw_func = functionalized_rng_wrapper.post_compile(
                compiled_fw_func, aot_config, runtime_metadata=fw_metadata
            )
            compiled_fw_func = fakified_out_wrapper.post_compile(
                compiled_fw_func,
                aot_config,
                runtime_metadata=fw_metadata,
            )

        # NB: It's important to compile backwards ahead of time, as this may
        # add extra guards which we need to apply to the Dynamo cache at
        # forwards
        with track_graph_compiling(aot_config, "backward"):
            placeholder_list = fx_placeholder_vals(bw_module)

            forward_saved_for_backwards_strides = None
            if fwd_output_strides is not None:
                forward_saved_for_backwards_strides = fwd_output_strides[
                    inner_meta.tensors_saved_for_backwards_slice
                ]

            # saved activations can have different stride to eager if
            # the compiler does layout optimization. We should restride the
            # tensor passed in for compiling the backward graph using the
            # saved tensor's stride.
            for i in range(len(placeholder_list)):
                ph_arg = placeholder_list[i]
                if not isinstance(ph_arg, torch.Tensor):
                    continue

                if forward_saved_for_backwards_strides is None:
                    continue

                real_stride = None
                # Per all_args calling convention
                j = i - num_symints_saved_for_bw
                if 0 <= j < len(forward_saved_for_backwards_strides):
                    real_stride = forward_saved_for_backwards_strides[j]
                if real_stride is None:
                    continue

                # Comparing ph_arg.stride() with real_stride directly may
                # cause dynamic dimensions in ph_arg being specialized to static
                # value. Using the hints to avoid that.
                if _get_symint_hints(ph_arg.stride()) != real_stride:
                    # Note that here we use the stride of the real tensor to
                    # restride a FakeTensor. This does not cause trouble
                    # for dynamic shape since this code path only get
                    # executed if layout optimization is enabled. And we
                    # disable layout optimization for dynamic shape right
                    # now.
                    #
                    # A solution that decide stride order based on real
                    # tensor's stride and then apply that stride order to
                    # the FakeTensor does not work smoothly since some
                    # tensor's layout is not 'dense'. E.g. mixnet_l has a
                    # tensor with size [8, 64, 112, 112] and strides
                    # (2408448, 1, 21504, 192). The solution mentioned will
                    # decide a stride of (802816, 1, 7168, 64) for this
                    # tensor which is wrong.
                    placeholder_list[i] = ph_arg.as_strided(ph_arg.size(), real_stride)

            compiled_bw_func = None
            if num_symints_saved_for_bw > 0:
                context = torch._C._DisableAutocast if disable_amp else nullcontext
                with context():
                    try:
                        compiled_bw_func = aot_config.bw_compiler(
                            bw_module, placeholder_list
                        )
                    except Exception:
                        log.warning(
                            "failed to eagerly compile backwards for dynamic, suppressing in case backwards not needed",
                            exc_info=True,
                        )
            # Compiled autograd will run the bw_module in the backward pass,
            # so recompilation need happen anyway if the backward pass is ever
            # called.
            #
            # The reason we do the GraphModule recompilation here is because
            # the lazy recompilation will cause issue in the backward pass
            # with compiled autograd.
            #
            # Do the _LazyGraphModule.force_recompile here rather than when
            # bw_module is first generated by the partitioner because the bw_module.recompile
            # may be called in some code path later and cause the _LazyGraphModule.forward
            # becomes the lazy version again. One example is when dynamic shape is enabled
            # upfront, the bw_compiler will be called above which can cause extra
            # graph module recompilation on bw_module.
            if torch._dynamo.compiled_autograd.compiled_autograd_enabled_count:
                from torch.fx._lazy_graph_module import _LazyGraphModule

                _LazyGraphModule.force_recompile(bw_module)

    saved_context = TracingContext.try_get()
    saved_compile_context = CompileContext.try_get()

    backward_state_indices = [
        idx for idx, x in enumerate(flat_args) if isinstance(x, BackwardState)
    ]
    assert len(backward_state_indices) <= 1

    lazy_backward_info = AutogradLazyBackwardCompileInfo(
        bw_module,
        placeholder_list,
        saved_context,
        saved_compile_context,
    )

    compiled_fn = AOTDispatchAutograd.post_compile(
        compiled_fw_func,
        compiled_bw_func,
        maybe_subclass_meta,
        num_symints_saved_for_bw,
        backward_state_indices,
        disable_amp,
        _indices_of_inps_to_detach,
        lazy_backward_info,
        aot_config,
        fw_metadata=fw_metadata,
    )

    if config.debug_assert:
        flat_requires_grad: List[Optional[bool]] = [
            a.requires_grad if isinstance(a, Tensor) else None for a in flat_args
        ]
        compiled_fn = DebugAssertWrapper(
            flat_requires_grad=flat_requires_grad
        ).post_compile(compiled_fn, aot_config, runtime_metadata=fw_metadata)

    compiled_fn = post_compile(
        wrappers,
        compiled_fn,
        aot_config,
        runtime_metadata=fw_metadata,
    )
    return compiled_fn
