"""
These are the runtime wrappers that are associated with JIT-compiling.

This includes the forward-only and joint JIT runtime wrappers.

This module depends heavily on the runtime wrapper building blocks defined
in `runtime_wrappers`.
"""

import logging
from contextlib import nullcontext
from functools import wraps
from typing import Any, List, Optional

import torch
import torch.utils.dlpack
from torch import Tensor
from torch._dynamo.utils import lazy_format_graph_code
from torch._guards import detect_fake_mode, tracing, TracingContext
from torch._logging import getArtifactLogger, trace_structured
from torch._prims_common import CUDARngStateHelper
from torch._subclasses import FakeTensor
from torch.fx.experimental._backward_state import BackwardState
from torch.fx.experimental.proxy_tensor import is_sym_node
from torch.fx.experimental.symbolic_shapes import fx_placeholder_vals
from .. import config
from .dispatch_and_compile_graph import (
    aot_dispatch_autograd_graph,
    aot_dispatch_base_graph,
)
from .logging_utils import describe_input, format_guard_bug_msg, track_graph_compiling

from .runtime_wrappers import (
    aot_dispatch_subclass_wrapper,
    create_runtime_wrapper,
    functionalized_rng_runtime_epilogue,
)
from .schemas import (
    AOTConfig,
    MutationType,
    OutputType,
    SubclassMeta,
    TensorAlias,
    ViewAndMutationMeta,
)
from .subclass_utils import (
    compute_inner_mutated_inp_indices_from_subclass_meta,
    unwrap_tensor_subclasses,
    wrap_tensor_subclasses,
)

from .utils import (
    _get_symint_hints,
    call_func_at_runtime_with_args,
    make_boxed_func,
    normalize_as_list,
    strict_zip,
)

zip = strict_zip

log = logging.getLogger(__name__)
aot_joint_log = getArtifactLogger(__name__, "aot_joint_graph")
aot_graphs_log = getArtifactLogger(__name__, "aot_graphs")

aten = torch.ops.aten


def _compute_output_meta_with_inductor_strides(fw_module, fwd_output_strides):
    out = [n.meta["val"] for n in (list(fw_module.graph.nodes)[-1].args[0])]
    # will only be set for inductor
    if not fwd_output_strides:
        return out
    with TracingContext.get().fake_mode.shape_env.suppress_guards():
        for i in range(len(out)):
            if not isinstance(out[i], Tensor):
                continue
            if all(s1 == s2 for s1, s2 in zip(out[i].stride(), fwd_output_strides[i])):
                continue
            out[i] = out[i].as_strided(out[i].shape, fwd_output_strides[i])
    return out


def aot_dispatch_base(
    flat_fn,
    flat_args: List[Tensor],
    aot_config: AOTConfig,
    *,
    fw_metadata: ViewAndMutationMeta,
):
    fw_module, updated_flat_args, maybe_subclass_meta = aot_dispatch_base_graph(  # type: ignore[misc]
        flat_fn, flat_args, aot_config, fw_metadata=fw_metadata
    )

    disable_amp = torch._C._is_any_autocast_enabled()
    context = torch._C._DisableAutocast if disable_amp else nullcontext
    fakified_out = None

    with context(), track_graph_compiling(aot_config, "inference"):
        compiler = (
            aot_config.inference_compiler
            if aot_config.inference_compiler is not None
            else aot_config.fw_compiler
        )
        if config.functionalize_rng_ops:
            # Add the seed and offset as example inputs to pass to the compiler
            fake_mode = detect_fake_mode()
            seed, offset = CUDARngStateHelper.get_torch_state_as_tuple(fake_mode)
            updated_flat_args.extend([seed, offset])

        if tracing_context := torch._guards.TracingContext.try_get():
            tracing_context.fw_metadata = (
                fw_metadata
                if maybe_subclass_meta is None
                else maybe_subclass_meta.fw_metadata
            )

        with TracingContext.report_output_strides() as fwd_output_strides:
            compiled_fw = compiler(fw_module, updated_flat_args)

        # see note: [Returning Fake Tensors on First AOT Autograd Call]
        if tracing_context and tracing_context.fakify_first_call:
            fakified_out = _compute_output_meta_with_inductor_strides(
                fw_module, fwd_output_strides
            )

    # However, create_runtime_wrapper does not expect the rng offsets in the
    # output. So, we have to create another wrapper and take out the offset. As
    # a result, we have to account for not boxed_call compilers as well.
    if not hasattr(compiled_fw, "_boxed_call"):
        compiled_fw = make_boxed_func(compiled_fw)

    # Create a wrapper to set up the rng functionalize bits
    @wraps(compiled_fw)
    def rng_functionalization_wrapper(args):
        # see note: [Returning Fake Tensors on First AOT Autograd Call]
        nonlocal fakified_out
        if fakified_out is not None:
            out = fakified_out
            fakified_out = None
            return out

        # args is a list because compiled_fw is boxed_call
        if fw_metadata.is_rng_op_functionalized:
            # Add the seed and offset to args
            seed, offset = CUDARngStateHelper.get_torch_state_as_tuple()
            args.extend([seed, offset])
            out = compiled_fw(args)
            out = functionalized_rng_runtime_epilogue(fw_metadata, out)
            return out
        else:
            return compiled_fw(args)

    if maybe_subclass_meta is not None:
        compiled_fw_func = aot_dispatch_subclass_wrapper(
            rng_functionalization_wrapper,
            subclass_metas=fw_metadata.subclass_fw_graph_out_meta,
            num_fw_outs_saved_for_bw=None,
        )
    else:
        compiled_fw_func = rng_functionalization_wrapper

    if not hasattr(compiled_fw_func, "_boxed_call"):
        compiled_fw_func = make_boxed_func(compiled_fw_func)

    compiled_fn = create_runtime_wrapper(
        compiled_fw_func,
        runtime_metadata=fw_metadata,
        indices_of_inps_to_detach=[],
        trace_joint=False,
        keep_input_mutations=aot_config.keep_inference_input_mutations,
        disable_amp=disable_amp,
    )

    return compiled_fn


def aot_dispatch_autograd(
    flat_fn,
    flat_args: List[Any],
    aot_config: AOTConfig,
    *,
    fw_metadata: ViewAndMutationMeta,
):
    fw_metadata.deterministic = torch.are_deterministic_algorithms_enabled()
    fx_g, joint_inputs, maybe_subclass_meta = aot_dispatch_autograd_graph(  # type: ignore[misc]
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
            payload_fn=lambda: fx_g.print_readable(print_output=False),  # type: ignore[union-attr]
        )

    fakify_first_call = False
    fakified_out = None

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
            num_mutated_inp_runtime_indices = len(mutated_inp_runtime_indices)
            num_inner_fwd_outputs = (
                num_mutated_inp_runtime_indices
                + inner_meta.num_outputs
                + inner_meta.num_intermediate_bases
                + inner_meta.num_outputs_rng_offset
                + len(
                    fw_metadata.tokens
                )  # See Note [Side-Effectful Tokens in AOTAutograd]
            )
            fw_module, bw_module = aot_config.partition_fn(
                fx_g, joint_inputs, num_fwd_outputs=num_inner_fwd_outputs
            )

            fw_outs = next(n for n in fw_module.graph.nodes if n.op == "output").args[0]
            # we only need to bookkeep the symints that are saved for bw, not any symints
            # the user forward might have returned in its own output
            fw_outs_saved_for_bw = fw_outs[num_inner_fwd_outputs:]
            num_fw_outs_saved_for_bw = len(fw_outs_saved_for_bw)
            symint_outs_saved_for_bw = [
                n for n in fw_outs_saved_for_bw if is_sym_node(n)
            ]
            fw_metadata.num_symints_saved_for_bw = len(symint_outs_saved_for_bw)
            inner_meta.num_symints_saved_for_bw = len(symint_outs_saved_for_bw)
            _num_symints_saved_for_bw = len(symint_outs_saved_for_bw)

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
        _indices_of_inps_to_detach = []
        bw_outs = next(n for n in bw_module.graph.nodes if n.op == "output").args[0]

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
            for i, (bw_out) in enumerate(bw_outs):
                if bw_out is None:
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
            if config.functionalize_rng_ops:
                # Update example inputs for the fw_compiler
                fake_mode = detect_fake_mode()
                seed, offset = CUDARngStateHelper.get_torch_state_as_tuple(fake_mode)
                adjusted_flat_args.extend([seed, offset])
                # We are not clearing flat_args here because
                # 1) There is a check in the debug compiler at the end
                # 2) It does not matter as these are fake tensors

            if tracing_context := torch._guards.TracingContext.try_get():
                tracing_context.fw_metadata = inner_meta

            with TracingContext.report_output_strides() as fwd_output_strides:
                compiled_fw_func = aot_config.fw_compiler(fw_module, adjusted_flat_args)
            if not hasattr(compiled_fw_func, "_boxed_call"):
                compiled_fw_func = make_boxed_func(compiled_fw_func)

            # see note: [Returning Fake Tensors on First AOT Autograd Call]
            if tracing_context and tracing_context.fakify_first_call:
                fakified_out = _compute_output_meta_with_inductor_strides(
                    fw_module, fwd_output_strides
                )
                fakify_first_call = True

            if maybe_subclass_meta is not None:
                # Why do we need to pass in num_fw_outs_saved_for_bw?
                # See Note: [Partitioner handling for Subclasses, Part 2]
                compiled_fw_func = aot_dispatch_subclass_wrapper(
                    compiled_fw_func,
                    subclass_metas=fw_metadata.subclass_fw_graph_out_meta,
                    num_fw_outs_saved_for_bw=num_fw_outs_saved_for_bw,
                )
                if not hasattr(compiled_fw_func, "_boxed_call"):
                    compiled_fw_func = make_boxed_func(compiled_fw_func)

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
                j = i - len(symint_outs_saved_for_bw)
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
            if len(symint_outs_saved_for_bw):
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

    backward_state_indices = [
        idx for idx, x in enumerate(flat_args) if isinstance(x, BackwardState)
    ]
    assert len(backward_state_indices) <= 1

    class CompiledFunction(torch.autograd.Function):
        compiled_fw = compiled_fw_func
        compiled_bw = compiled_bw_func
        metadata: ViewAndMutationMeta = fw_metadata  # type: ignore[assignment]
        maybe_subclass_metadata: Optional[SubclassMeta] = maybe_subclass_meta
        num_symints_saved_for_bw = _num_symints_saved_for_bw
        _compiled_autograd_should_lift = False
        _fakify_first_call = fakify_first_call

        @staticmethod
        def _compiled_autograd_key(ctx):
            return (ctx._autograd_function_id, *ctx.symints)

        @staticmethod
        def forward(ctx, *deduped_flat_tensor_args):
            args = deduped_flat_tensor_args
            if backward_state_indices:
                bw_state = args[backward_state_indices[0]]
                assert isinstance(bw_state, BackwardState)
                ctx._compiled_autograd_backward_state = bw_state

            marked_dirty_inps = []
            for i in fw_metadata.mutated_graph_handled_indices_seen_by_autograd:
                arg = deduped_flat_tensor_args[i]
                if not (arg.requires_grad and arg.is_leaf):  # would error
                    ctx.mark_dirty(arg)
                marked_dirty_inps.append(arg)

            if not CompiledFunction._fakify_first_call:
                if CompiledFunction.metadata.is_rng_op_functionalized:
                    # Add the seed and offset to args
                    seed, offset = CUDARngStateHelper.get_torch_state_as_tuple()
                    args = (*args, seed, offset)
                # There is a pretty complicated calling convention around what the compiled fw returns.
                # The full list of outputs and their relative order is:
                # (*tokens, *mutated_inputs, *fw_outs, *fw_intermediate_bases, *saved_tensors, *saved_symints)
                # - Note that in the synthetic bases case, mutated_inputs will correspond to an updated version
                #   of the original view, and not the synthetic base

                fw_outs = call_func_at_runtime_with_args(
                    CompiledFunction.compiled_fw,
                    args,
                    disable_amp=disable_amp,
                )
            else:
                nonlocal fakified_out
                assert fakified_out is not None
                CompiledFunction._fakify_first_call = False
                fw_outs = fakified_out
                fakified_out = None

            num_outputs = CompiledFunction.metadata.num_outputs
            num_outputs_aliased = CompiledFunction.metadata.num_outputs_aliased
            num_mutated_runtime_inps = (
                CompiledFunction.metadata.num_mutated_inp_runtime_indices
            )
            num_tokens = len(CompiledFunction.metadata.tokens)
            num_forward_returns = CompiledFunction.metadata.num_forward_returns
            num_forward = CompiledFunction.metadata.num_forward

            # Partitioners must put symint arguments at the end separate from tensor arguments
            tensors_saved_for_backwards = fw_outs[
                CompiledFunction.metadata.tensors_saved_for_backwards_slice
            ]
            assert all(isinstance(x, torch.Tensor) for x in tensors_saved_for_backwards)
            # See Note [Detaching saved tensors in AOTAutograd]
            ctx.save_for_backward(
                *(
                    x.detach() if x._is_view() else x
                    for x in tensors_saved_for_backwards
                )
            )
            symint_outs = fw_outs[
                CompiledFunction.metadata.symints_saved_for_backwards_slice
            ]
            assert all(
                isinstance(x, (int, float, torch.SymInt, torch.SymFloat))
                for x in symint_outs
            ), str([type(x) for x in symint_outs])
            ctx.symints = symint_outs

            raw_returns = fw_outs[0 : num_forward_returns + num_tokens]

            # Wrap all autograd.Function.forward() outputs that are aliases
            # so that autograd.Function doesn't treat them as tensors
            if num_mutated_runtime_inps > 0:
                for i, idx in enumerate(
                    CompiledFunction.metadata.mutated_inp_runtime_indices
                ):
                    # We could make this faster by only looping over inputs with metadata-only mutations
                    # (instead of looping over inputs with either data or metadata mutations), but there shouldn't be many.
                    info = CompiledFunction.metadata.input_info[idx]
                    if info.mutates_metadata and not info.mutates_data:
                        raw_returns[i] = TensorAlias(raw_returns[i])

                if config.debug_assert:
                    user_mutated_inputs_raw = raw_returns[0:num_mutated_runtime_inps]
                    mut_inp_infos = [
                        x
                        for x in CompiledFunction.metadata.input_info
                        if x.mutates_data or x.mutates_metadata
                    ]
                    assert len(user_mutated_inputs_raw) == len(mut_inp_infos)

            if CompiledFunction.metadata.num_unsafe_view_outputs > 0:
                for idx in CompiledFunction.metadata.unsafe_view_out_indices:
                    raw_return_idx = num_mutated_runtime_inps + idx
                    o = raw_returns[raw_return_idx]
                    raw_returns[raw_return_idx] = torch.ops.aten._unsafe_view(
                        o, o.shape
                    )

            if num_outputs_aliased > 0:
                for idx in CompiledFunction.metadata.aliased_out_indices:
                    raw_return_idx = num_mutated_runtime_inps + idx
                    raw_returns[raw_return_idx] = TensorAlias(
                        raw_returns[raw_return_idx]
                    )

                if config.debug_assert:
                    intermediates_raw = raw_returns[
                        num_mutated_runtime_inps + num_outputs :
                    ]
                    assert not any(
                        isinstance(x, TensorAlias) for x in intermediates_raw
                    )

            # invariant: intermediate bases always require gradients, so we don't have to
            # consider marking them as non-differentiable.
            raw_returns_not_including_intermediate_bases = raw_returns[
                : num_mutated_runtime_inps + num_outputs
            ]
            raw_returns_meta = [
                x
                for x in CompiledFunction.metadata.input_info
                if x.mutation_type == MutationType.MUTATED_OUT_GRAPH
            ] + CompiledFunction.metadata.output_info

            fw_outs_not_requiring_grad = [
                x
                for (i, x) in enumerate(raw_returns_not_including_intermediate_bases)
                if isinstance(x, torch.Tensor) and not raw_returns_meta[i].requires_grad
            ]
            ctx.mark_non_differentiable(*fw_outs_not_requiring_grad)
            ctx._materialize_non_diff_grads = False

            functionalized_rng_runtime_epilogue(
                CompiledFunction.metadata,
                fw_outs[num_forward_returns:num_forward],
                return_new_outs=False,
            )
            return tuple(raw_returns) + tuple(marked_dirty_inps)

        @staticmethod
        def backward(ctx, *flat_args):
            # Calling convention: we expect a grad_out passed to the backward:
            # - for every output of the fw that does *not* alias an input or graph intermediate
            # - for every updated_input generated by the fw that does *not* alias an input (aka only data-mutations)
            # - for every graph intermediate that we need to use to generate an output later.
            # The other outputs in the autograd.Function.forward that do *not* show up in the backward include:
            # - outputs that alias inputs or graph intermediates
            # - updated inputs due to metadata-only mutations.
            # We need to return them in the forward, but ensure that they all do not get gradients in the backward,
            # and we filter them out here before passing the remaining grad_outputs into the compiled backward.
            num_intermediate_bases = CompiledFunction.metadata.num_intermediate_bases
            num_graph_handled_inputs = (
                CompiledFunction.metadata.num_mutated_graph_handled_indices_seen_by_autograd
            )
            num_mutated_runtime_inps = (
                CompiledFunction.metadata.num_mutated_inp_runtime_indices
            )
            expected_grad_outs = (
                CompiledFunction.metadata.num_outputs
                + num_mutated_runtime_inps
                + num_intermediate_bases
            )
            deterministic = CompiledFunction.metadata.deterministic
            global_deterministic = torch.are_deterministic_algorithms_enabled()
            if deterministic is not None:
                torch._check(
                    not (not deterministic and global_deterministic),
                    lambda: (
                        "This compiled backward function is being run with "
                        "torch.use_deterministic_algorithms(True), "
                        "but it was previously generated during the forward function while "
                        "torch.use_deterministic_algorithms(False) was set."
                    ),
                )

            if num_graph_handled_inputs > 0:
                flat_args = flat_args[:-num_graph_handled_inputs]
            assert len(flat_args) == expected_grad_outs
            out_info = CompiledFunction.metadata.output_info

            inp_tangents, out_tangents, intermediate_base_tangents = (
                flat_args[0:num_mutated_runtime_inps],
                flat_args[
                    num_mutated_runtime_inps : num_mutated_runtime_inps
                    + CompiledFunction.metadata.num_outputs
                ],
                flat_args[
                    num_mutated_runtime_inps + CompiledFunction.metadata.num_outputs :
                ],
            )
            # input_info contains info on *every* input,
            # But in the backward(), we are only given grad outputs for every mutated input
            # We then need to filter out the grad outputs that correspond to metadata-only mutations or don't require grad
            input_info = CompiledFunction.metadata.input_info
            inp_tangents_filtered = [
                x
                for x, info_idx in zip(
                    inp_tangents, CompiledFunction.metadata.mutated_inp_runtime_indices
                )
                if input_info[info_idx].mutates_data
                and input_info[info_idx].requires_grad
            ]
            # We also need to filter out grad outputs that correspond to outputs aliasing inputs/intermediates
            out_tangents_filtered = [
                x
                for x, info in zip(out_tangents, out_info)
                if info.output_type
                in [
                    OutputType.non_alias,
                    OutputType.unsafe_view_alias,
                    OutputType.custom_function_view,
                ]
                and issubclass(info.raw_type, torch.Tensor)
                and info.requires_grad
            ]
            # intermediate bases always require gradients, and always participate in the backward graph.
            flat_bw_args_with_grads = [
                *inp_tangents_filtered,
                *out_tangents_filtered,
                *intermediate_base_tangents,
            ]
            num_flat_bw_args_with_grads = len(flat_bw_args_with_grads)

            # sanity asserts
            # metadata_only_inps = [
            #     x for x, info_idx in zip(inp_tangents, mutated_inp_indices)
            #     if not input_info[info_idx].mutates_data
            # ]
            # aliased_outputs = [
            #     x for x, info in zip(out_tangents, out_info) if info.output_type != OutputType.non_alias]
            # assert all(x is None for x in metadata_only_inps)
            # assert all(x is None for x in aliased_outputs)

            rng_args = []
            if CompiledFunction.metadata.is_rng_op_functionalized:
                # Add the seed and offset to args
                rng_args = CUDARngStateHelper.get_torch_state_as_tuple()

            all_args = [
                *ctx.symints,
                *ctx.saved_tensors,
                *flat_bw_args_with_grads,
                *rng_args,
            ]
            del flat_bw_args_with_grads

            tangents_start_idx = (
                len(all_args) - num_flat_bw_args_with_grads - len(rng_args)
            )
            tangents_end_idx = len(all_args) - len(rng_args)

            # Note: [AOTAutograd Backward Guards]
            # During AOTDispatch, we eagerly create and trace out a joint fw-bw graph.
            # Doing so requires us to "guess" about some of the metadata of our grad_outputs.
            #
            # In particular: if an output to the forward is a plain tensor or a subclass,
            # its corresponding grad_output in the backward **may or may not** be
            # a plain tensor or a subclass. The main cases are:
            # (1) If an output is a plain tensor, its grad_out will also be a plain tensor,
            #     *unless* the output is used in some subclass compute later in the forward graph,
            #     which will cause its grad_output to become a subclass
            # (2) If an output is a subclass, its grad_out will also be a subclass,
            #     *unless* the output of the forward did not actually participate in the gradient computation,
            #     in which case autograd will insert a plain tensor of zeros for the grad_output.
            #     We could avoid this case with `torch.autograd.Function.set_materialize_grads`,
            #     although this is not turned on today in AOTAutgrad and would require more work.
            #
            # Today, we make a guess on subclass-ness based on the above examples,
            # and hard-error in the backward if we guessed wrong.
            #
            # In the future, we should add backward guards that would allow us to
            # properly handle this case instead of erroring: we would need to retrace the backward graph,
            # since we might produce an entirely different trace if our grad_outputs are subclass or not.
            assert (
                len(CompiledFunction.metadata.output_types)
                == num_flat_bw_args_with_grads
            )
            grad_output_types = [
                type(x) for x in all_args[-num_flat_bw_args_with_grads:]
            ]
            # In general, we can add more asserts/guards here for when we partitioned
            # with incorrect assumptions about the grad_outputs.
            # Normalize FakeTensor -> torch.Tensor
            # - during tracing our types are FakeTensor
            # - at runtime in the backward our types are torch.Tensor...
            # - unless we're running compiled backward, in which case they are also FakeTensor
            grad_output_types_ = [
                torch.Tensor if x is FakeTensor else x for x in grad_output_types
            ]
            assert (
                grad_output_types_ == CompiledFunction.metadata.output_types
            ), f"""\
We incorrectly attempted to compile the backward with incorrect subclass metadata.
If you run into this error, please file an issue.
Expected grad_output types: {str(CompiledFunction.metadata.output_types)}
Got grad_output types: {str(grad_output_types)}"""

            # TODO: figure out how to refactor the backward properly so I can use aot_dispatch_subclass_wrapper() here.
            if CompiledFunction.maybe_subclass_metadata is not None:
                # Get the number of tangents after unwrapping
                len_tangents = len(
                    unwrap_tensor_subclasses(
                        all_args[tangents_start_idx:tangents_end_idx],
                        is_joint_structure=False,
                    )
                )
                all_args = unwrap_tensor_subclasses(all_args, is_joint_structure=False)
                tangents_start_idx = len(all_args) - len_tangents - len(rng_args)
                tangents_end_idx = tangents_start_idx + len_tangents

            # Make the tangents contiguous. Note that we must do this after subclass desugaring
            # because inputs to inductor have to be contiguous
            all_args = [
                t.contiguous()
                if (
                    (tangents_start_idx <= i < tangents_end_idx)
                    and (not t.is_contiguous())
                )
                else t
                for i, t in enumerate(all_args)
            ]

            def call_compiled_backward():
                if ctx._is_compiled_autograd_tracing():
                    # For compiled autograd, run raw FX graph so that it can be inlined into the larger graph
                    symints = ctx._get_compiled_autograd_symints()
                    assert len(symints) == len(ctx.symints)
                    all_args[: len(symints)] = symints
                    if backward_state_indices:
                        assert ctx._compiled_autograd_backward_state.proxy is not None
                        all_args.append(ctx._compiled_autograd_backward_state)
                    context = torch._C._DisableAutocast if disable_amp else nullcontext
                    with context():
                        out = normalize_as_list(bw_module(*all_args))
                    out = functionalized_rng_runtime_epilogue(
                        CompiledFunction.metadata, out
                    )
                    return tuple(out)
                assert (
                    not backward_state_indices
                ), "BackwardState requires CompiledAutograd"
                ctx.maybe_clear_saved_tensors()
                if CompiledFunction.compiled_bw is None:
                    context = torch._C._DisableAutocast if disable_amp else nullcontext
                    with tracing(saved_context), context(), track_graph_compiling(
                        aot_config, "backward"
                    ):
                        CompiledFunction.compiled_bw = aot_config.bw_compiler(
                            bw_module, placeholder_list
                        )

                out = call_func_at_runtime_with_args(
                    CompiledFunction.compiled_bw,
                    all_args,
                    steal_args=True,
                    disable_amp=disable_amp,
                )

                out = functionalized_rng_runtime_epilogue(
                    CompiledFunction.metadata, out
                )
                return tuple(out)

            if torch.is_grad_enabled() and any(
                t.requires_grad for t in all_args if isinstance(t, torch.Tensor)
            ):
                # Ensure that the graph is connected, and error if double backward is performed.
                # See comment for why once_differentiable is not sufficient:
                # https://github.com/pytorch/pytorch/pull/92348/files#r1072962107
                class CompiledFunctionBackward(torch.autograd.Function):
                    # CompiledFunctionBackward is not yet supported in dynamo skipfiles
                    _compiled_autograd_should_lift = False

                    @staticmethod
                    def forward(ctx, *unused_args):
                        outs = call_compiled_backward()
                        # TODO: figure out how to refactor the backward properly so I can use aot_dispatch_subclass_wrapper() here.
                        if CompiledFunction.maybe_subclass_metadata is not None:
                            assert (
                                CompiledFunction.maybe_subclass_metadata.grad_input_metas
                                is not None
                            )
                            outs_wrapped = wrap_tensor_subclasses(
                                outs,
                                subclass_metas=CompiledFunction.maybe_subclass_metadata.grad_input_metas,
                            )
                            return outs_wrapped
                        return outs

                    @staticmethod
                    def backward(ctx, *args):
                        raise RuntimeError(
                            "torch.compile with aot_autograd does not currently support double backward"
                        )

                CompiledFunctionBackward._compiled_autograd_key = (  # type: ignore[method-assign]
                    CompiledFunction._compiled_autograd_key
                )

                # Pass args even though they're unused, so that the graph is built
                out = CompiledFunctionBackward.apply(*all_args)
            else:
                out = call_compiled_backward()

            # TODO: figure out how to refactor the backward properly so I can use aot_dispatch_subclass_wrapper() here.
            if CompiledFunction.maybe_subclass_metadata is not None:
                assert (
                    CompiledFunction.maybe_subclass_metadata.grad_input_metas
                    is not None
                )
                outs_wrapped = wrap_tensor_subclasses(
                    out,
                    subclass_metas=CompiledFunction.maybe_subclass_metadata.grad_input_metas,
                )
                return outs_wrapped
            return out

    compiled_function = create_runtime_wrapper(
        CompiledFunction.apply,
        runtime_metadata=fw_metadata,
        indices_of_inps_to_detach=_indices_of_inps_to_detach,
        trace_joint=True,
        keep_input_mutations=aot_config.keep_inference_input_mutations,
        disable_amp=disable_amp,
    )

    if not config.debug_assert:
        return compiled_function

    flat_requires_grad = [
        a.requires_grad if isinstance(a, Tensor) else None for a in flat_args
    ]

    @wraps(compiled_function)
    def debug_compiled_function(*args):
        # TODO: Check aliasing relationships
        # TODO: Check strides for metadata mutation
        # (NB: ideally, this logic is factored out of this function and
        # you move these debug checks there)

        # Check requires grad.  Bad case is when we compiled with
        # requires_grad = False, but input requires_grad = True
        # (vice versa is OK; we compute a gradient and then throw
        # it away when it hits the input.)
        for i, a in enumerate(args):
            can_require_grad = flat_requires_grad[i]
            if can_require_grad is None:
                assert not isinstance(a, Tensor)
            elif not can_require_grad:
                assert not a.requires_grad, format_guard_bug_msg(
                    aot_config,
                    f"{describe_input(i, aot_config)} would not require grad",
                )

        return compiled_function(*args)

    return debug_compiled_function
