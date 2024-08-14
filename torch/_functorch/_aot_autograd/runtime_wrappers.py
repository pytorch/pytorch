# mypy: allow-untyped-defs
"""
This module defines runtime wrappers, which, based on previous analysis attempts to:
1. process the inputs and outputs
2. apply mutations
3. handle functionalized randomness
4. deduplicate inputs and consolidate views into their bases (see input_output_analysis)
"""
import builtins
import collections
import pprint
from contextlib import nullcontext
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Callable, cast, Dict, List, Optional, Tuple, Union

import torch
import torch.utils.dlpack
from torch import Tensor
from torch._guards import (
    compile_context,
    CompileContext,
    detect_fake_mode,
    DuplicateInputs,
    tracing,
    TracingContext,
)
from torch._prims_common import CUDARngStateHelper
from torch._subclasses import FakeTensor
from torch.fx.experimental._backward_state import BackwardState
from torch.multiprocessing.reductions import StorageWeakRef
from torch.utils._python_dispatch import is_traceable_wrapper_subclass

from .. import config
from .collect_metadata_analysis import run_functionalized_fw_and_collect_metadata
from .functional_utils import gen_alias_from_base
from .input_output_analysis import (
    compute_overlapping_inputs,
    create_synthetic_base_metadata,
    remove_dupe_metadata,
)
from .logging_utils import describe_input, format_guard_bug_msg, track_graph_compiling
from .schemas import (
    AOTConfig,
    InputAliasInfo,
    MutationType,
    OutputType,
    SubclassCreationMeta,
    SubclassMeta,
    TensorAlias,
    ViewAndMutationMeta,
)
from .subclass_utils import (
    get_types_for_subclass,
    requires_subclass_dispatch,
    unwrap_tensor_subclasses,
    wrap_tensor_subclasses,
)
from .traced_function_transforms import aot_dispatch_subclass
from .utils import (
    call_func_at_runtime_with_args,
    make_boxed_func,
    normalize_as_list,
    partial_flatten_asdict,
    strict_zip,
)


zip = strict_zip


class CompilerWrapper:
    """
    A wrapper around the inputs and outputs to the compiler_fn. We separate these into two parts:

    1. The prologue, which edits the input to the compiler_fn(flat_fn, flat_args, etc)
    2. The epilogue, which edits the outputs of the compiler_fn (compiled_fn, real arguments)

    Each wrapper below should be implemented as a CompilerWrapper, so that we can facilitate
    caching on the compiled output, and re-wrapping the output via epilogues.
    Extra metadata that is needed to compute pre or post compile can be passed in via attributes.
    """

    def pre_compile(
        self,
        flat_fn,
        flat_args: List[Tensor],
        aot_config: AOTConfig,
        *,
        fw_metadata: ViewAndMutationMeta,
    ) -> Tuple[Callable, List[Tensor], ViewAndMutationMeta]:
        """
        Process the inputs to the compiler_fn. You can pass in extra metadata via kwargs.
        Args:
        flat_fn: The function to compile
        flat_args: Metadata from example inputs of the function to compile
        aot_config: AOTConfig passed in at compile time
        fw_metadata: ViewAndMutationMeta generated from flat_fn and flat_args
        """
        return flat_fn, flat_args, fw_metadata

    def post_compile(self, compiled_fn, aot_config, *, runtime_metadata) -> Callable:
        """
        Given an output of the compiler, wrap it with information received from prologue.
        Args:
        compiled_fn: Callable after calling compiler_fn
        aot_config: AOTConfig after calling prologue
        runtime_metadata: ViewAndMutationMeta after calling all wrappers's pre_compile steps.
        Example:

        def wrapped_compiled_fn(args):
            # do something with args, aot_config, fw_metadata
            return compiled_fn(args)

        return wrapped_compiled_fn
        """
        return compiled_fn


# The wrapper created by this function handles all of the runtime aliasing and mutation "epilogue" logic
# that needs to run after the compiled function.
#
# This function accepts a trace_joint flag, indicating whether or not we're generating the runtime
# epilogue for a forward-only inference graph, or for an autograd.Function.apply function.
# This is because there are some minor differences in how we treat these cases at runtime:
# - resize_() is currently handled in the inference case, but not fully handled in the autograd case.
# - the autograd cases inserts TensorAlias wrapper objects for outputs that alias inputs
@dataclass
class RuntimeWrapper(CompilerWrapper):
    indices_of_inps_to_detach: List[int]
    trace_joint: bool
    disable_amp: bool

    def post_compile(
        self,
        compiled_fn,
        aot_config: AOTConfig,
        *,
        runtime_metadata: ViewAndMutationMeta,
    ):
        return _create_runtime_wrapper(
            compiled_fn,
            runtime_metadata=runtime_metadata,
            indices_of_inps_to_detach=self.indices_of_inps_to_detach,
            trace_joint=self.trace_joint,
            keep_input_mutations=aot_config.keep_inference_input_mutations,
            disable_amp=self.disable_amp,
        )


class NoopAliasHandler:
    def __init__(self, info, runtime_metadata, trace_joint):
        pass

    def __call__(self, orig_inputs, fw_outs, out):
        return out


def _unwrap_tensoralias(x):
    assert isinstance(x, TensorAlias)
    return x.alias


def _identity(x):
    return x


class AliasOfInputHandler:
    def __init__(self, info, runtime_metadata, trace_joint):
        self.base_idx = info.base_idx
        self.unwrap_out = _unwrap_tensoralias if trace_joint else _identity
        self.requires_grad = info.requires_grad
        self.functional_tensor = info.functional_tensor
        self.replay_views = config.view_replay_for_aliased_outputs

    def __call__(self, orig_inputs, fw_outs, out):
        aliased_base_tensor = orig_inputs[self.base_idx]
        return gen_alias_from_base(
            aliased_base_tensor,
            self.unwrap_out(out),
            self.requires_grad,
            self.functional_tensor,
            replay_views=self.replay_views,
        )


class IsInputHandler:
    def __init__(self, info, runtime_metadata, trace_joint):
        self.base_idx = info.base_idx
        self.unwrap_out = _unwrap_tensoralias if trace_joint else _identity

    def __call__(self, orig_inputs, fw_outs, out):
        aliased_base_tensor = orig_inputs[self.base_idx]
        return aliased_base_tensor


class AliasOfIntermediateHandler:
    def __init__(self, info, runtime_metadata, trace_joint):
        if info.output_type in (
            OutputType.alias_of_intermediate,
            OutputType.alias_of_intermediate_save_as_output,
        ):
            num_user_outputs = len(runtime_metadata.output_info)
            self.base_idx = info.base_idx + num_user_outputs
        else:
            self.base_idx = info.base_idx

        self.unwrap_out = _unwrap_tensoralias if trace_joint else _identity
        self.requires_grad = info.requires_grad
        self.functional_tensor = info.functional_tensor
        self.replay_views = config.view_replay_for_aliased_outputs

    def __call__(self, orig_inputs, fw_outs, out):
        aliased_base_tensor = fw_outs[self.base_idx]
        return gen_alias_from_base(
            aliased_base_tensor,
            self.unwrap_out(out),
            self.requires_grad,
            self.functional_tensor,
            replay_views=self.replay_views,
        )


_HANDLER_MAP = {
    OutputType.non_alias: NoopAliasHandler,
    OutputType.unsafe_view_alias: NoopAliasHandler,
    OutputType.custom_function_view: NoopAliasHandler,
    OutputType.alias_of_input: AliasOfInputHandler,
    OutputType.is_input: IsInputHandler,
    OutputType.alias_of_intermediate: AliasOfIntermediateHandler,
    OutputType.alias_of_intermediate_save_as_output: AliasOfIntermediateHandler,
    OutputType.alias_of_intermediate_base_is_user_output: AliasOfIntermediateHandler,
}


def make_output_handler(info, runtime_metadata, trace_joint):
    handler_type = _HANDLER_MAP[info.output_type]
    return handler_type(info, runtime_metadata, trace_joint)


def _create_runtime_wrapper(
    compiled_fn,
    *,
    runtime_metadata: ViewAndMutationMeta,
    indices_of_inps_to_detach: List[int],
    trace_joint: bool,
    keep_input_mutations: bool,
    disable_amp: bool,
):
    if not hasattr(compiled_fn, "_boxed_call"):
        compiled_fn = make_boxed_func(compiled_fn)

    # Note [Inputs needed in runtime epilogue after list clearing]
    # In Python functions, you can't free the input arguments of a function within the scope of that function. A workaround is to
    # wrap the input arguments in a list, and clear the list from within the function.
    # Here, this is implemented as `call_func_at_runtime_with_args(..., steal_args=True)`.
    #
    # This is needed for Compiled Autograd since some of the inputs (activations) should be freed early.
    # However, we cannot blindly clear the entire list, because AOTAutograd may need access to some of the graph inputs
    # **after** the compiled function has finished running. There are two main cases:
    #   (1) Input mutations: If there are an input mutations that we must run outside of the graph, we need access to the input.
    #   (2) Output aliasing: Outputs that aliases graph inputs generally must be regenerated outside of the `autograd.Function`,
    #       and doing so requires us accessing the corresponding input after the compiled artifact has run.
    epilogue_args_idx = []
    epilogue_args_idx.extend(runtime_metadata.mutated_inp_runtime_indices)
    for info in runtime_metadata.output_info:
        if (
            info.output_type == OutputType.alias_of_input
            or info.output_type == OutputType.is_input
        ):
            assert isinstance(info.base_idx, int)
            epilogue_args_idx.append(info.base_idx)

    if config.unlift_effect_tokens:
        assert len(runtime_metadata.tokens) == 0

    replay_views = config.view_replay_for_aliased_outputs
    if runtime_metadata.num_outputs_aliased > 0:
        output_handlers = tuple(
            make_output_handler(info, runtime_metadata, trace_joint)
            for info in runtime_metadata.output_info
        )

    def runtime_wrapper(args: List[Any]):
        # stash a ref to each input tensor we plan to use after the compiled function
        orig_inputs = {i: args[i] for i in epilogue_args_idx}

        if keep_input_mutations:
            mutated_args = (
                args[i]
                for i in runtime_metadata.mutated_graph_handled_indices_seen_by_autograd
            )
            torch.autograd.graph.increment_version(mutated_args)

        if trace_joint:
            args_ = list(args)
            # See Note [Detaching inputs that never need gradients]
            for idx in indices_of_inps_to_detach:
                if isinstance(args_[idx], torch.Tensor):
                    args_[idx] = args_[idx].detach()

            # It's possible to have trace_joint inside user specified with no_grad() region,
            # if there is a nested with enable_grad(), that forces some outputs to require gradients.
            # Therefore, we unconditionally turn on enable_grad() for compiled_fn execution.
            with torch.autograd._force_original_view_tracking(
                True
            ), torch.enable_grad():
                all_outs = call_func_at_runtime_with_args(
                    compiled_fn, args_, disable_amp=disable_amp, steal_args=True
                )
        else:
            # When we have an inference graph, we run with grad disabled.
            # It's possible to get an inference graph with inputs that require grad,
            # in which case we want to make sure autograd is disabled
            # (since e.g., inductor will generate aten.addmm.out calls which autograd will complain on)
            # NOTE: We use _set_grad_enabled directly to reduce runtime overhead
            grad_enabled = torch.is_grad_enabled()
            try:
                if grad_enabled:
                    torch._C._set_grad_enabled(False)
                all_outs = call_func_at_runtime_with_args(
                    compiled_fn, args, disable_amp=disable_amp, steal_args=True
                )
            finally:
                if grad_enabled:
                    torch._C._set_grad_enabled(True)
        del args

        num_mutated_runtime_inps = runtime_metadata.num_mutated_inp_runtime_indices
        num_intermediate_bases = runtime_metadata.num_intermediate_bases

        assert (
            len(all_outs)
            == num_mutated_runtime_inps
            + runtime_metadata.num_outputs
            + num_intermediate_bases
        )

        # Step 3: After running the compiled fw, apply updates to mutated inputs
        num_mutations_to_apply = runtime_metadata.num_mutated_inp_runtime_indices
        if num_mutations_to_apply > 0:
            updated_inputs = all_outs[:num_mutations_to_apply]
            fw_outs = all_outs[num_mutations_to_apply:]

            for i, inpt_idx in enumerate(runtime_metadata.mutated_inp_runtime_indices):
                meta = runtime_metadata.input_info[inpt_idx]
                if not meta.mutates_data and not meta.mutates_metadata:
                    continue
                original_inpt = orig_inputs[inpt_idx]
                updated_inpt = updated_inputs[i]
                if meta.mutates_storage_metadata:
                    # See Note [set_() Input Mutations in AOTAutograd]
                    # mutates_storage_metadata means our input saw a x.set_(y) call.
                    # What if x **also** saw a data and/or a metadata mutation?
                    # (1) If the [meta]data mutation occurred after the set_(),
                    #     then there is no need to copy_() the data.
                    #     When we perform x.set_(x_updated), we are guaranteed that
                    #     x_updated already has the final version of the data/metadata
                    # (2) If a data mutation occurred before the set_().
                    #     This case seems very difficult to support.
                    #     TODO: discuss on the PR and decide if we want to tr to
                    #     either support it, or detect and ban it.
                    if trace_joint:
                        assert isinstance(updated_inpt, TensorAlias)
                        updated_inpt = updated_inpt.alias
                    with torch.no_grad():
                        original_inpt.set_(updated_inpt)
                    continue
                if meta.mutates_metadata and not meta.mutates_data:
                    if trace_joint:
                        assert isinstance(updated_inpt, TensorAlias)
                        updated_inpt = updated_inpt.alias
                    # We need to grab the size/stride/storage_offset from the compiled forward,
                    # and use that to mutate the metadata of the input
                    original_inpt.as_strided_(
                        updated_inpt.size(),
                        updated_inpt.stride(),
                        updated_inpt.storage_offset(),
                    )
                else:
                    if meta.mutates_data and meta.mutates_metadata:
                        original_inpt.as_strided_(
                            updated_inpt.size(),
                            updated_inpt.stride(),
                            updated_inpt.storage_offset(),
                        )
                    else:
                        assert meta.mutates_data
                    if meta.is_leaf and original_inpt.requires_grad:
                        # We can hit this situation in this case:
                        #   def f(x):
                        #       x.detach().mul_(2)
                        #       return x + 1
                        # AOTAutograd will see a mutation in the above case, and try to
                        # apply a copy_() here, in the epilogue.
                        # But if x required gradients, and is a leaf, then autograd
                        # will yell at us for trying to mutate it.
                        # However, it's only possible to end up in this scenario (like the above)
                        # if all of the mutations to the leaf input were non-autograd-tracking mutations
                        # (aka mutations under no_grad(), or on detached views).
                        # In that case, we fully want to hide the mutation from autograd, so detaching is ok.
                        original_inpt.detach().copy_(updated_inpt)
                    else:
                        original_inpt.copy_(updated_inpt)
        else:
            fw_outs = all_outs

        # Step 4: Manually regenerate any outputs that are aliased to inputs, instead of
        # compiling them.
        if runtime_metadata.num_outputs_aliased > 0:
            # The compiled forward also returned intermediate bases. We don't want to return them to the user.
            expect_num_outputs = (
                len(output_handlers) + runtime_metadata.num_intermediate_bases
            )
            assert len(fw_outs) == expect_num_outputs
            ret_outs = [
                handler(orig_inputs, fw_outs, out)
                for out, handler in builtins.zip(fw_outs, output_handlers)
            ]
        else:
            ret_outs = fw_outs

        if runtime_metadata.dynamic_outputs:
            for t, o in zip(ret_outs, runtime_metadata.output_info):
                if o.dynamic_dims is None:
                    continue
                if hasattr(t, "_dynamo_weak_dynamic_indices"):
                    t._dynamo_weak_dynamic_indices |= o.dynamic_dims
                else:
                    t._dynamo_weak_dynamic_indices = o.dynamic_dims.copy()
        if runtime_metadata.grad_enabled_mutation is not None:
            torch._C._set_grad_enabled(runtime_metadata.grad_enabled_mutation)
        return ret_outs

    return runtime_wrapper


@dataclass
class FunctionalizedRngRuntimeWrapper(CompilerWrapper):
    # TODO: I would love to get rid of this argument, but it's
    # Wrapped pretty tightly around our aot_dispatch_autograd logic.
    # Specifically, tensors_saved_for_backwards_slice's value is both used for calculating indices
    # for setting placeholder strides(which is done before runtime, before this wrapper runs)
    # and for saving tensors for backward (which is done during runtime, after this wrapper runs)
    # So in aot_dispatch_autograd, this wrapper can't edit the set of outs without making one
    # of those two indices incorrect.
    return_new_outs: bool = True

    def pre_compile(
        self,
        flat_fn,
        flat_args,
        aot_config,
        *,
        fw_metadata,
    ) -> Tuple[Callable, List[Tensor], ViewAndMutationMeta]:
        if config.functionalize_rng_ops:
            # Update example inputs for the fw_compiler
            fake_mode = detect_fake_mode()
            seed, offset = CUDARngStateHelper.get_torch_state_as_tuple(fake_mode)
            flat_args.extend([seed, offset])
            # We are not clearing flat_args here because
            # 1) There is a check in the debug compiler at the end
            # 2) It does not matter as these are fake tensors
        return flat_fn, flat_args, fw_metadata

    def post_compile(
        self,
        compiled_fn,
        aot_config: AOTConfig,
        *,
        runtime_metadata: ViewAndMutationMeta,
    ):
        @wraps(compiled_fn)
        def wrapper(runtime_args: List[Any]):
            if runtime_metadata.is_rng_op_functionalized:
                # Add the seed and offset to args
                seed, offset = CUDARngStateHelper.get_torch_state_as_tuple()
                runtime_args.extend([seed, offset])
                out = compiled_fn(runtime_args)
                out = self._functionalized_rng_runtime_epilogue(
                    runtime_metadata,
                    out,
                    # TODO: this won't be right for the backward when we convert the call_compiled_backward to use the wrapper
                    runtime_metadata.num_forward_returns,
                )
                return out
            return compiled_fn(runtime_args)

        return wrapper

    # Calling convention: If we are running functionalized RNG, then outs consists
    # of (user_outs, rng_offset)
    def _functionalized_rng_runtime_epilogue(
        self,
        metadata: ViewAndMutationMeta,
        outs,
        offset_index,
    ):
        if metadata.is_rng_op_functionalized:
            assert metadata.num_outputs_rng_offset == 1
            new_rng_offset = outs[offset_index]
            CUDARngStateHelper.set_new_offset(new_rng_offset)
            if self.return_new_outs:
                user_outs = outs[:offset_index] + outs[offset_index + 1 :]
                return user_outs
            else:
                return outs

        return outs


@dataclass
class FakifiedOutWrapper(CompilerWrapper):
    out_metas: List[torch.Tensor] = field(default_factory=list)
    # TracingContext.fwd_output_strides
    # Generated from actually doing compile
    fwd_output_strides: Optional[List[List[int]]] = None
    needs_post_compile: bool = True

    def pre_compile(
        self,
        fw_module,  # Must be fw_module from aot_dispatch_*_graph
        flat_args,
        aot_config,
        *,
        fw_metadata,
    ) -> Tuple[Callable, List[Tensor], ViewAndMutationMeta]:
        tracing_context = torch._guards.TracingContext.try_get()
        if tracing_context and tracing_context.fakify_first_call:
            self.out_metas = [
                n.meta["val"] for n in (list(fw_module.graph.nodes)[-1].args[0])
            ]
        else:
            self.needs_post_compile = False
        return fw_module, flat_args, fw_metadata

    def _compute_output_meta_with_inductor_strides(self):
        out = self.out_metas
        fwd_output_strides = self.fwd_output_strides
        if not fwd_output_strides:
            return out

        from torch.fx.experimental.symbolic_shapes import statically_known_true

        for i in range(len(out)):
            if not isinstance(out[i], Tensor):
                continue
            if all(
                statically_known_true(s1 == s2)
                for s1, s2 in zip(out[i].stride(), fwd_output_strides[i])
            ):
                continue
            out[i] = out[i].as_strided(out[i].shape, fwd_output_strides[i])
        return out

    # To be called post compile
    def set_fwd_output_strides(self, fwd_output_strides):
        self.fwd_output_strides = fwd_output_strides

    def post_compile(
        self,
        compiled_fn,
        aot_config: AOTConfig,
        *,
        runtime_metadata: ViewAndMutationMeta,
    ):
        if self.needs_post_compile:
            assert self.fwd_output_strides is not None
            fakified_out = self._compute_output_meta_with_inductor_strides()

            @wraps(compiled_fn)
            def wrapper(runtime_args):
                nonlocal fakified_out
                if fakified_out is not None:
                    out = fakified_out
                    fakified_out = None
                    return out
                return compiled_fn(runtime_args)

            return wrapper
        # If we don't need to fakify, we can just return the original compiled function
        return compiled_fn


# This wrapper handles the AOTDispatch runtime logic for tensor subclasses.
# At runtime, we have a compiled function that knows how to operate on the domain of DenseTensor -> DenseTensor,
# But the user might have passed us some tensor subclass inputs (or expect some subclass tensor outputs).
# This function handles the wrapping and unwrapping of tensor subclasses at runtime.
@dataclass
class AOTDispatchSubclassWrapper(CompilerWrapper):
    trace_joint: bool
    fw_only: Optional[Callable]  # Not cached, only used in pre_compile
    maybe_subclass_meta: Optional[SubclassMeta]
    num_fw_outs_saved_for_bw: Optional[int]

    def pre_compile(
        self,
        flat_fn,
        flat_args: List[Tensor],
        aot_config: AOTConfig,
        *,
        fw_metadata: ViewAndMutationMeta,
    ):
        (new_flat_fn, new_flat_args, subclass_meta) = aot_dispatch_subclass(
            flat_fn,
            flat_args,
            is_joint_structure=self.trace_joint,
            meta=fw_metadata,
            fw_only=self.fw_only,  # type: ignore[arg-type]
        )
        self.maybe_subclass_meta = subclass_meta
        return new_flat_fn, new_flat_args, fw_metadata

    def post_compile(
        self,
        compiled_fn,
        _aot_config: AOTConfig,
        *,
        runtime_metadata: ViewAndMutationMeta,
    ):
        if self.maybe_subclass_meta is None:
            return compiled_fn

        subclass_metas = runtime_metadata.subclass_fw_graph_out_meta

        @wraps(compiled_fn)
        def inner_fn(args: List[Any]):
            unwrapped_args = unwrap_tensor_subclasses(
                args, is_joint_structure=self.trace_joint
            )
            args.clear()
            # expectation: runtime_fn is a boxed fn
            unwrapped_outs = compiled_fn(unwrapped_args)
            wrapped_outs = wrap_tensor_subclasses(
                unwrapped_outs,
                subclass_metas=subclass_metas,
                num_fw_outs_saved_for_bw=self.num_fw_outs_saved_for_bw,
                is_runtime=True,
            )
            return wrapped_outs

        # box it
        inner_fn._boxed_call = True  # type: ignore[attr-defined]
        return inner_fn


@dataclass
class EffectTokensWrapper(CompilerWrapper):
    def post_compile(
        self,
        compiled_fn,
        _aot_config,
        *,
        runtime_metadata: ViewAndMutationMeta,
    ):
        num_tokens = len(runtime_metadata.tokens)

        @wraps(compiled_fn)
        def inner_fn(args: List[Any]):
            if num_tokens > 0:
                # Pass in effect tokens (See Note [Side-Effectful Tokens in AOTAutograd])
                old_args = args
                args = [*([None] * num_tokens), *args]
                old_args.clear()

            outs = compiled_fn(args)

            # Inductor cache DummyModule can return None
            if outs is None:
                return None
            # Toss out the effect tokens (See Note [Side-Effectful Tokens in AOTAutograd])
            return outs[num_tokens:]

        # box it
        inner_fn._boxed_call = True  # type: ignore[attr-defined]
        return inner_fn


# MOTIVATION:
#
# When tracing functions for future execution, one must be careful not to pass
# in the same input tensor multiple times (e.g., f(x, x), as this can result
# in graphs that are ONLY valid if you later pass a new tensor in exactly the
# same way (e.g., f(y, y)).  (NB: we really mean duplicate; two distinct
# tensors that alias each other is a different situation that is covered by
# aot_dispatch_deduplicated_autograd). Here are two examples:
#
# (1) Suppose you have a function:
#
#   def f(x, y):
#       return x + y
#
# If you make_fx(f)(x, x), you will trace out:
#
#   def f(x, y):
#       return y + y
#
# Oops!
#
# (2) For most tensors x and y, you can compute f's gradient with respect to
# these to inputs by saying torch.autograd.grad(f(x, y), (x, y)).  However,
# if x is y, you will trace out a program that gets incorrect gradients:
#
#   >>> x = torch.randn(1, requires_grad=True)
#   >>> torch.autograd.grad(x + x, (x, x))
#   (tensor([2.]), tensor([2.]))
#
# In other words, the gradient is double-counted.  Deduplicating the arguments
# gives you an appropriate gradient:
#
#   >>> y = torch.randn(1, requires_grad=True)
#   >>> torch.autograd.grad(x + y, (x, y))
#   (tensor([1.]), tensor([1.]))
#
# HOW TO DEDUPLICATE:
#
# There are a few strategies, in order of preference:
#
# 1. For every duplicate argument to the function, detach it into
#    a separate leaf tensor, so that it is no longer duplicated.
#
#       PRO: The resulting compiled graph works for any configuration
#       of duplicated arguments.
#
#       CON: It does not (naively) work if you mutate the metadata of inputs:
#
#           def f(x, y):
#               x.transpose_(0, 1)
#               y.transpose_(0, 2)
#
#           x = torch.randn(2, 3, 4)
#           f(x, x)
#
#       The ordering of the transposes inside f dictates whether or not
#       you get [4, 2, 3] or [3, 4, 2].  This means that you cannot precompute
#       what metadata mutations should get applied to each input; you need to
#       assume they aren't duplicates (what we do today) or preserve
#       the original metadata mutations exactly in order, so that they work
#       for any duplicate configuration.
#
#       CON: It does not (naively) work if you mutate the data of inputs.
#       In particular, leaf tensors that require grad cannot be mutated,
#       this makes it impossible to differentiate with respect to the original
#       base.
#
# 2. For every duplicate argument to the function, remove it, so it is
#    no longer part of the "true" signature:
#
#       PRO: Implemented naively, it still works for metadata/data mutation.
#
#       CON: The resulting compiled graph is duplicate-specialized: it only
#       works if future calls duplicate arguments in exactly the same way.
#       Horribly, Dynamo doesn't guard on this at the moment.  But even if
#       it did, you could still end up recompiling a bunch of each duplicate.
#
# Our strategy is to do (1) if we can, and do (2) otherwise, erroring if
# Dynamo's guards are not enough.  In practice, this seems to cover
# everything.
#
@dataclass
class AOTDedupeWrapper(CompilerWrapper):
    keep_arg_mask: List[bool] = field(default_factory=list)
    add_dupe_map: List[int] = field(default_factory=list)
    old_input_metadata: List[InputAliasInfo] = field(default_factory=list)
    needs_post_compile: bool = True

    # NB: Hot path, avoid set lookups here
    # TODO: Can avoid the zip here too, probably
    def remove_dupe_args(self, args):
        return [t for t, keep in zip(args, self.keep_arg_mask) if keep]

    def add_dupe_args(self, args):
        return [args[i] for i in self.add_dupe_map]

    def pre_compile(
        self,
        flat_fn,
        flat_args: List[Tensor],
        aot_config: AOTConfig,
        *,
        fw_metadata: ViewAndMutationMeta,
    ) -> Tuple[Callable, List[Tensor], ViewAndMutationMeta]:
        # Use information about whether or not flat_fn mutates its arguments
        # or not to handle dupe args

        # Strategy 1: For any input that is not mutated, we can leafify it if we
        # need to remove a duplicate.
        leaf_flat_args = []
        args_set = set()
        ok = True

        for i, a in enumerate(flat_args):
            if not isinstance(a, torch.Tensor):
                leaf_flat_args.append(a)
            elif a not in args_set:
                args_set.add(a)
                leaf_flat_args.append(a)
            elif (
                not fw_metadata.input_info[i].mutates_data
                and not fw_metadata.input_info[i].mutates_metadata
            ):
                leaf_flat_args.append(a.detach().requires_grad_(a.requires_grad))
            else:
                ok = False
                break

        if ok:
            self.needs_post_compile = False
            return flat_fn, leaf_flat_args, fw_metadata

        if requires_subclass_dispatch(leaf_flat_args, fw_metadata):
            raise RuntimeError(
                """\
        Encountered duplicate inputs that are mutated in the graph, but at least one input/output
        to the graph is a tensor subclass. This is not supported today. You can try to
        remove the aliasing yourself as a workaround, or otherwise file an issue on github."""
            )

        # export path: ban duplicate inputs for now, add later if requested.
        if aot_config.is_export:
            raise RuntimeError(
                f"""\
        Encountered duplicated inputs that are mutated in the graph you are trying to export.
        This functionality is currently not supported. If needed, please file a github issue.

        fw_metadata={str(fw_metadata)}
            """
            )

        # Strategy 2: Duplicate specialize.
        #
        # In Haskell types, suppose you have:
        #
        #   add_dupe_args :: DedupedArgs -> Args
        #   remove_dupe_args :: Args -> DedupedArgs
        #
        #   compiler_fn
        #       :: (DedupedArgs -> R) -> DedupedArgs -> AOTConfig -> (DedupedArgs -> R)
        #   deped_compiler_fn
        #       :: (Args -> R) -> Args -> AOTConfig -> (Args -> R)
        #
        # Then the code below can be written in point-free style as:
        #
        #   deduped_compiler_fn f a c =
        #       compiler_fn (f . add_dupe_args) (remove_dupe_args a) c . remove_dupe_args
        #
        # Suppose you have:
        #
        #   [a, b, a, c]
        #
        # We want:
        #
        #   remove_dupe_args([a, b, a, c]) == [a, b, c]
        #   add_dupe_args([a, b, c]) == [a, b, a, c]
        #
        # This is done via (respectively):
        #
        #   seen_args = {a: 0, b: 1, c: 2}
        #   enumerate(add_dupe_map) = [  # how to get args from the deduped list
        #       (0, 0),
        #       (1, 1),
        #       (2, 0),
        #       (3, 2),
        #   ]
        #   keep_arg_mask = [True, True, False, True]

        seen_args: Dict[Tensor, int] = {}
        # Implicitly map duped arg position (list index) to de-duped arg position
        keep_arg_mask: List[bool] = []
        add_dupe_map: List[int] = []
        duped_arg_len = len(flat_args)

        j = 0  # index into deduped_flat_args
        for t in flat_args:
            if isinstance(t, torch.Tensor):
                if t in seen_args:
                    keep_arg_mask.append(False)
                    add_dupe_map.append(seen_args[t])
                    continue
                seen_args[t] = j

            keep_arg_mask.append(True)
            add_dupe_map.append(j)
            j += 1
        assert (
            len(add_dupe_map) == duped_arg_len
        ), f"Expects add_dupe_map to have length {duped_arg_len} but got {len(add_dupe_map)}"

        self.keep_arg_mask = keep_arg_mask
        self.add_dupe_map = add_dupe_map

        deduped_flat_args = self.remove_dupe_args(flat_args)

        # Update our input metadata to remove duped input metadata.
        updated_fw_metadata = remove_dupe_metadata(
            fw_metadata, keep_arg_mask, add_dupe_map
        )

        if (
            tracing_context := TracingContext.try_get()
            and aot_config.aot_autograd_arg_pos_to_source
        ):
            # TODO(voz): This structure is 1:1, we could consider an alternate structure like
            # kept_pos:[dupe_arg_pos], however, add_dupe_map is 1:1 so we would need a new structure there,
            # which feels like needless complexity for a tiny bit of efficiency at this point.
            for dupe_arg_pos, (kept_pos, keep_arg) in enumerate(
                zip(add_dupe_map, keep_arg_mask)
            ):
                if not keep_arg:
                    dupe_arg_source = aot_config.aot_autograd_arg_pos_to_source[
                        dupe_arg_pos
                    ]
                    kept_arg_source = aot_config.aot_autograd_arg_pos_to_source[
                        kept_pos
                    ]
                    tracing_context.guards_context.aotautograd_guards.append(  # type: ignore[attr-defined]
                        DuplicateInputs(kept_arg_source, dupe_arg_source)
                    )

        @wraps(flat_fn)
        def wrapped_flat_fn(*args):
            return flat_fn(*self.add_dupe_args(args))

        if config.debug_assert:
            ref_fw_metadata = run_functionalized_fw_and_collect_metadata(
                wrapped_flat_fn,
                static_input_indices=aot_config.static_input_indices,
                keep_input_mutations=fw_metadata.keep_input_mutations,
                is_train=fw_metadata.is_train,
            )(*deduped_flat_args)
            assert (
                ref_fw_metadata == updated_fw_metadata
            ), f"ref_metadata={str(ref_fw_metadata)}, actual_metadata={str(updated_fw_metadata)}"

        return wrapped_flat_fn, deduped_flat_args, updated_fw_metadata

    def post_compile(
        self,
        compiled_fn,
        aot_config: AOTConfig,
        *,
        runtime_metadata: ViewAndMutationMeta,
    ):
        if not self.needs_post_compile:
            return compiled_fn

        @wraps(compiled_fn)
        def wrapped_compiled_fn(args: List[Any]):
            deduped_args = self.remove_dupe_args(args)
            args.clear()
            return compiled_fn(deduped_args)

        wrapped_compiled_fn._boxed_call = True  # type: ignore[attr-defined]

        # This can be uncommented when we properly guard for duplicates,
        # but right now we must not do it.
        # if not config.debug_assert:
        #     return wrapped_compiled_fn

        @wraps(wrapped_compiled_fn)
        def debugged_compiled_fn(args):
            # Test that the computed remove/add arg functions are an inverse
            new_args = self.add_dupe_args(self.remove_dupe_args(args))
            seen: Dict[Any, None] = {}
            for i, (x, y) in enumerate(zip(new_args, args)):
                seen[y] = None
                assert x is y, format_guard_bug_msg(
                    aot_config,
                    f"{describe_input(i, aot_config)} would be a duplicate of "
                    f"{describe_input(self.add_dupe_map[i], aot_config)}",
                )
            # This is only an error if there is metadata mutation on both of
            # the duped arguments; in this case, we need to know what order
            # the metadata mutation applies in.  You'll get the correct result
            # otherwise, because a graph that assumes distinct inputs works if
            # you dupe the inputs (the gradient contributions from each input
            # will get summed up appropriately.)
            #
            # TODO: work out how to setup this assert correctly
            """
            assert len(seen) == unique_args, format_guard_bug_msg(aot_config,
                f"there would be {unique_args} distinct arguments"
            )
            """
            return wrapped_compiled_fn(args)

        debugged_compiled_fn._boxed_call = True  # type: ignore[attr-defined]

        return debugged_compiled_fn


# This layer handles the situation where you have two inputs that alias each other,
# and one of the inputs is mutated.
# We need to take special care to ensure that the mutation is applied to the other aliases in the graph.
#
# pre-condition: AOTDedupWrapper has already run.
# (This function will in theory work if there are duplicate args.
# However, the synthetic base code path is a bit sub-optimal, and running with dupe'd inputs
# would cause us to hit that path more frequently).
@dataclass
class AOTSyntheticBaseWrapper(CompilerWrapper):
    # Currently, the only reason we need to plumb this bool is because
    # the synthetic base code prohibits more cases in the autograd case than the inference case.
    trace_joint: bool  # TODO: refactor trace_joint
    needs_post_compile: bool = True
    aliased_arg_idx_with_metadata_mutations: List[int] = field(default_factory=list)

    def pre_compile(
        self,
        flat_fn,
        flat_args: List[Any],
        aot_config: AOTConfig,
        *,
        fw_metadata: ViewAndMutationMeta,
    ) -> Tuple[Callable, List[Tensor], ViewAndMutationMeta]:
        is_inference = not self.trace_joint
        flat_args_with_synthetic_bases, synthetic_base_info = merge_view_inputs(
            flat_args,
            fw_metadata.input_info,
            is_inference=is_inference,
        )

        # Happy path: we don't need synthetic bases
        if synthetic_base_info is None:
            self.needs_post_compile = False
            return flat_fn, flat_args, fw_metadata

        # export path: ban synthetic bases for now, add later if requested.
        if requires_subclass_dispatch(flat_args, fw_metadata):
            raise RuntimeError(
                """\
        Encountered aliased inputs that are mutated in the graph, but at least one input/output
        to the graph is a tensor subclass. This is not supported today. You can try to
        remove the aliasing yourself as a workaround, or otherwise file an issue on github."""
            )

        if aot_config.is_export:
            raise RuntimeError(
                f"""\
        Encountered aliased inputs that are mutated in the graph you are trying to export.
        This functionality is currently not supported. If needed, please file a github issue.

        synthetic_base_info={str(synthetic_base_info)}

        fw_metadata={str(fw_metadata)}
                """
            )

        assert len(fw_metadata.input_info) == len(synthetic_base_info)

        # Update our forward metadata to take synthetic bases into account
        (
            fw_metadata_updated,
            aliased_arg_idx_with_metadata_mutations,
        ) = create_synthetic_base_metadata(
            fw_metadata, synthetic_base_info, flat_args, flat_args_with_synthetic_bases
        )
        # Save old input args for post-compile
        self.old_input_info = fw_metadata.input_info

        self.aliased_arg_idx_with_metadata_mutations = (
            aliased_arg_idx_with_metadata_mutations
        )

        num_aliased_args_with_metadata_mutations = len(
            aliased_arg_idx_with_metadata_mutations
        )

        replay_views = config.view_replay_for_aliased_outputs

        def _unpack_synthetic_bases(primals: Tuple[Any, ...]) -> List[Any]:
            f_args_inner = []
            for inner_idx_or_tuple in synthetic_base_info:
                if isinstance(inner_idx_or_tuple, int):
                    f_args_inner.append(primals[inner_idx_or_tuple])
                else:
                    inner_base_idx, view_tensor = inner_idx_or_tuple
                    base = primals[inner_base_idx]
                    view_arg = gen_alias_from_base(
                        base,
                        view_tensor,
                        view_tensor.requires_grad,
                        replay_views=replay_views,
                    )
                    f_args_inner.append(view_arg)
            return f_args_inner

        @wraps(flat_fn)
        def wrapped_flat_fn(*args):
            unpacked_args = _unpack_synthetic_bases(args)
            # This is a bit subtle. The goal of this entire function (aot_dispatch_synthetic_bases)
            # is to relieve the downstream logic from having to reason about mutations on inputs that alias
            # each other, by replacing aliased inputs with a synthetic base.
            # One area where this breaks down a bit however is if one of those aliased inputs
            # experienced a metadata mutation.
            # We are now obligated to reapply the metadata mutation directly to the user's input;
            # it isn't enough to apply mutations back to the synthetic base in the downstream logic.
            #
            # The way we handle this is by pretending that those aliased inputs that experience metadata mutations
            # are additional outputs in the user's forward function.
            # The downstream logic will just treat these as "user outputs that alias inputs".
            # However, we will manually grab them at runtime here, use them to reapply the metadata mutation
            # to the user inputs, and not return them to the user.
            aliased_args_with_metadata_mutations = [
                x
                for i, x in enumerate(unpacked_args)
                if i in self.aliased_arg_idx_with_metadata_mutations
            ]
            if len(aliased_args_with_metadata_mutations) > 0:
                return *(flat_fn(*unpacked_args)), *aliased_args_with_metadata_mutations
            else:
                return flat_fn(*unpacked_args)

        if config.debug_assert:
            ref_fw_metadata = run_functionalized_fw_and_collect_metadata(
                wrapped_flat_fn,
                static_input_indices=aot_config.static_input_indices,
                keep_input_mutations=fw_metadata.keep_input_mutations,
                is_train=fw_metadata.is_train,
            )(*flat_args_with_synthetic_bases)
            assert ref_fw_metadata == fw_metadata_updated, (
                f"ref_metadata={pprint.pformat(partial_flatten_asdict(ref_fw_metadata))}, "
                f"\nactual_metadata={pprint.pformat(partial_flatten_asdict(fw_metadata_updated))}"
            )
        return (
            wrapped_flat_fn,
            flat_args_with_synthetic_bases,
            fw_metadata_updated,
        )

    def post_compile(
        self,
        compiled_fn,
        aot_config: AOTConfig,
        *,
        runtime_metadata: ViewAndMutationMeta,
    ):
        if not self.needs_post_compile:
            return compiled_fn

        is_inference = not self.trace_joint

        @wraps(compiled_fn)
        def wrapped_compiled_fn(args):
            args_with_synthetic_bases, synthetic_base_info = merge_view_inputs(
                args, self.old_input_info, is_inference=is_inference
            )
            assert synthetic_base_info is not None
            aliased_args_w_metadata_mutations = [
                args[i] for i in self.aliased_arg_idx_with_metadata_mutations
            ]
            num_aliased_args_with_metadata_mutations = len(
                aliased_args_w_metadata_mutations
            )
            args.clear()
            outs = compiled_fn(args_with_synthetic_bases)
            if num_aliased_args_with_metadata_mutations > 0:
                # This code does not handle **all** input metadata mutations.
                # Instead, it only handles metadata mutations on inputs that were converted into synthetic bases
                # (which only happens if at least one aliased input experienced a data mutation).
                # e.g:
                # def f(a, b):
                #     a.mul_(2)
                #     b.t_(1, 0)
                # f(x.view(2, 2), x.view(2, 2))
                mutated_metadata_inps = outs[-num_aliased_args_with_metadata_mutations:]
                user_outs = outs[:-num_aliased_args_with_metadata_mutations]
                for inp, mutated_inp in zip(
                    aliased_args_w_metadata_mutations, mutated_metadata_inps
                ):
                    inp.as_strided_(
                        mutated_inp.size(),
                        mutated_inp.stride(),
                        mutated_inp.storage_offset(),
                    )
                return user_outs
            return outs

        return wrapped_compiled_fn


# Note [Handling mutations on an input that aliases other inputs]
# The easiest example to show-case this edge case is here:
#
# def f(a, b):
#     a.mul_(2)
#     out = a + b
#     return out
# b = torch.ones(...)
# a = b.view(-1)
# f(a, b)
#
# In this situation, if a and b happened to be aliased, we need to trace something different!
# Suppose we had b = a.view(-1)
# (In this case, that means that `a._base is b`)
#
# We need to ensure that the aliasing relationship between a and b is preserved.
# We do that detecting the specific situation above (mutate an input that aliases another input),
# and when we do that, we create a synthetic base argument. Then inside of the traced forward,
# we regenerate a and b off of that base.
# The complete example of the transformed function looks like this:
#
# // The traced forward takes in a synthetic base, and regenerates the aliased inputs as views
# // We could consider getting view-replay support here to minimize as_strided_scatter ops in the graph
# def traced_forward(base):
#     a = base.as_strided(...)
#     b = base.as_strided(...)
#     a_updated = a.mul(2)
#     base_updated = torch.as_strided_scatter(base, a_updated, ...)
#     b_updated = base_updated.as_strided(...)
#     out = a_updated + b_updated
#     return a_updated, out
#
# def compiled_fn(a, b):
#     // we detect that a is the "differentiable base" here
#     base = a
#     // In other situations, we might do either:
#     // (1) a and b are both views off of some larger differentiable base
#     //     assert a._base is b._base and a._base is not None
#     //     base = a._base
#     // (2) a and b both don't require gradients. Create a base from the storage
#     //     assert a._base is None and b._base is None
#     //     base = torch.Tensor(a.storage())
#     a_updated, out = traced_forward(base)
#     a.copy_(a_updated)
#     return out
#
# This function:
# (1) Merges input views into a synthetic base argument, when any of those input views are mutated
# (2) Returns metadata telling the autograd.Function how to modify their arguments properly,
#     to respect the new calling convention.
#
# The calling convention is as follows.
# Any inputs that were originally views of one another get yanked, and replaced with a synthetic base.
# The argument list ordering goes [base1, ..., baseN], [arg1, ..., argN],
# Where the ordering of the bases is determined from the ordering of the original view args.
# baseA will come before baseB if the earliest original argument coming from baseA
# showed up earlier in the argument list than the earliest original argument coming from baseB.
#
# Example, given some tensors a, b, c, d
# call site:
#   f(a, c.view(-1), b.view(-1), b, c, d)
# Modified argument list:
#   c_base comes first because the first c view came earlier in arg list than the first b view
#   a and d still show up in the modified arg list, but b and c don't- they're regenerated from their bases
#   b_base = torch.Tensor(b.storage())
#   c_base = torch.Tensor(c.storage())
#   f(c_base, b_base, a, d)
def merge_view_inputs(
    fwd_inputs: List[Any],
    mutated_input_info: List[InputAliasInfo],
    *,
    # The autograd case currently has more restrictions than the inference case.
    is_inference: bool,
) -> Tuple[List[Any], Optional[List[Union[int, Tuple[int, torch.Tensor]]]]]:
    def _are_differentiable_views(view1, view2):
        if view1 is view2:
            return True
        if view1._base is None and view2._base is None:
            return False
        if view1._base is view2._base or view1._base is view2 or view1 is view2._base:
            return True
        return False

    def _same_dtype_views(view1, view2):
        if view1.dtype != view2.dtype:
            return False
        if view1._base is not None and view1.dtype != view1._base.dtype:
            return False
        if view2._base is not None and view2.dtype != view2._base.dtype:
            return False
        return True

    assert len(fwd_inputs) == len(mutated_input_info)
    if not [info for info in mutated_input_info if info.mutates_data]:
        # Return early when there are no mutations.
        return fwd_inputs, None

    storage_ref_to_idx: Dict[StorageWeakRef, List[int]] = collections.defaultdict(list)
    base_args = []
    other_args = []
    for i, inpt in enumerate(fwd_inputs):
        if isinstance(inpt, Tensor):
            storage_ref = StorageWeakRef(inpt.untyped_storage())
            storage_ref_to_idx[storage_ref].append(i)
        else:
            other_args.append(inpt)
    # Note [Synthetic Base Info Metadata]
    # This list contains metadata that tells you what the i'th argument in the inner calling convention should be.
    # It's either:
    # - another int (corresponding to the index in the argument list of the element from the outer calling convention)
    # - idx, view_tensor, where we can generate the new output with view_tensor._view_func(old_args[idx])
    #   idx corresponds to which synthetic base from the outer calling context to view
    inner_calling_convention_meta: Dict[int, Union[int, Tuple[int, torch.Tensor]]] = {}
    for aliased_input_indices in storage_ref_to_idx.values():
        if len(aliased_input_indices) <= 1 or not any(
            # We only care about mutations that affect all aliases,
            # so metadata mutations on an input doesn't require us to do synthetic base handling.
            mutated_input_info[inpt_idx].mutates_data
            for inpt_idx in aliased_input_indices
        ):
            for curr_idx in aliased_input_indices:
                other_args.append(fwd_inputs[curr_idx])
            continue

        # Here, we attempt to do a more complicated check to detect false aliasing
        # (e.g. if all the tensors have the same storage, but don't actually overlap)
        # In theory, we could have a large group of tensors that all share storages, where only *some* of them
        # have overlapping memory.
        # I don't bother with that case for now: here, we only bail out earlier if we detect that **every** pair
        # of tensors in the current group that shares a storage is non-overlapping.
        aliased_input_indices_no_false_sharing = compute_overlapping_inputs(
            fwd_inputs, aliased_input_indices
        )
        if len(aliased_input_indices_no_false_sharing) <= 1:
            for curr_idx in aliased_input_indices:
                other_args.append(fwd_inputs[curr_idx])
            continue

        # We detected an input that was mutated, AND aliases with another input.
        # we need to replace this set of aliased inputs with a single synthetic base.
        # For now, I'm banning a bunch of cases. We expect dynamo to properly detect these cases
        # and error out. We can fix them later.
        # These checks are transitive, so we don't need to check every pair.
        for idx1, idx2 in zip(
            aliased_input_indices, aliased_input_indices[1:], strict=False
        ):
            view1 = fwd_inputs[idx1]
            view2 = fwd_inputs[idx2]
            # The "inputs that are aliased but have different differentiable bases" case
            # is more complicated and hopefully pretty rare. Not currently handled.
            if not is_inference:
                assert _are_differentiable_views(
                    view1, view2
                ), "aot_autograd() does not yet handle non-differentiable view input mutations."
            # Regenerating views when reinterpreting complex / real tensors seems non-trivial,
            # not handling for now
            assert _same_dtype_views(
                view1, view2
            ), "aot_autograd() does not yet handle input mutations on views with different dtypes."
        non_none_bases = [
            fwd_inputs[i]._base
            for i in aliased_input_indices
            if fwd_inputs[i]._base is not None
        ]
        aliases_with_none_bases = [
            fwd_inputs[i] for i in aliased_input_indices if fwd_inputs[i]._base is None
        ]
        if len(non_none_bases) == 0:
            # Case where none of the aliases have a ._base
            # we generate a synthetic base without gradients, and generate views off of it
            # We hit this case when we have input tensors to the graph that share a storage,
            # but do not have a ._base field.
            # Wondering when we hit this case?
            # The _base field simply says that autograd knows about the aliasing relationship,
            # but sometimes we create tensors which are aliased out of the same storage but guaranteed
            # to be disjoint. In these cases, we will skip setting up the _base relationship
            # for performance reasons (because the fact that the tensors share the same storage
            # is unobservable unless you (1) do naughty things with resize_/as_strided
            # or (2) look at the storage--as we are doing here.)
            # One particular example of this is optimizer steps on the LSTM module:
            # LSTM parameters are packed into a contiguous storage for efficiency reasons when
            # calling cuDNN kernels, so when these parameters get passed to the optimizer we will
            # find they share the same storage, but do not have _base set since they are all disjoint.
            #
            # NOTE: There is one case where this is unsafe:
            # torch.Tensor(storage) will ALWAYS create a 1D tensor, which is not necessarily
            # the same shape as the "actual" base that the tensor came from.
            # For the most part this is fine, because we always use as_strided()
            # to generate the original aliased inputs again.
            # If we were to use view-replay though, this could cause the aliased views
            # to have incorrect sizes.
            example_idx = aliased_input_indices[0]
            example_alias = fwd_inputs[example_idx]
            # Note that this function is re-used at both trace time and runtime.
            # At trace time, we're under a FakeMode so synthetic_base becomes a FakeTensor.
            synthetic_base = torch.empty(
                (0,), dtype=example_alias.dtype, device=example_alias.device
            )
            # We don't actually have a convenient way of going from storage -> tensor,
            # So using set_() here (we suffer some minor overhead, but this case is rare).
            synthetic_base.set_(example_alias.untyped_storage())
        else:
            # Case where all of the aliases require gradients, and have the same _base.
            synthetic_base = non_none_bases[0]
            for other_base in non_none_bases[1:]:
                assert (
                    other_base is synthetic_base
                ), "aot_autograd() does not yet handle non-differentiable view input mutations."
            for alias in aliases_with_none_bases:
                assert (
                    alias is synthetic_base
                ), "aot_autograd() does not yet handle non-differentiable view input mutations."
        base_args.append(synthetic_base)
        for curr_view_idx in aliased_input_indices:
            curr_view = fwd_inputs[curr_view_idx]
            base_idx = len(base_args) - 1
            # We store just enough info here so that we can regenerate the view later.
            # Regeneration: curr_view._view_func(args[base_idx])
            inner_calling_convention_meta[curr_view_idx] = (base_idx, curr_view)
    if len(base_args) == 0:
        assert len(other_args) == len(fwd_inputs)
        # If no synthetic bases are necessary, just return the original inputs.
        return fwd_inputs, None
    else:
        # Otherwise, return:
        # (1) The new args according to the updated calling convention: (synthetic_bases, other_args)
        # (2) Metadata telling functionalization how to generate the inner argument list given the outer calling convention.
        #     We post-process it into a list, where meta[i] tells you info about the i'th argument in the inner calling convention.
        args_to_functionalization = base_args + other_args
        arg_to_old_idx_map = {arg: i for (i, arg) in enumerate(fwd_inputs)}
        for i, other_arg in enumerate(other_args):
            new_idx = len(base_args) + i
            old_idx = arg_to_old_idx_map[other_arg]
            inner_calling_convention_meta[old_idx] = new_idx
        # post process into a list
        post_processed_calling_convention_meta: List[
            Union[int, Tuple[int, torch.Tensor]]
        ] = [-1 for _ in range(len(inner_calling_convention_meta))]
        for k, v in inner_calling_convention_meta.items():
            post_processed_calling_convention_meta[k] = v
        # Quick assert: every argument in the inner calling convention should be accounted for.
        for x in post_processed_calling_convention_meta:
            assert x != -1
        return args_to_functionalization, post_processed_calling_convention_meta


@dataclass
class AutogradLazyBackwardCompileInfo:
    bw_module: Callable
    placeholder_list: List[Any]
    saved_context: Optional[TracingContext]
    saved_compile_context: Optional[CompileContext]


# This is wrapped in a class just for namespacing purposes
# No need to make it into an actual CompilerWrapper because it doesn't fit the abstract as cleanly
class AOTDispatchAutograd:
    @staticmethod
    def _force_contiguous(x):
        if not isinstance(x, torch.Tensor):
            return x
        x = x.contiguous()
        if not is_traceable_wrapper_subclass(x):
            return x
        for attr in x.__tensor_flatten__()[0]:  # type: ignore[attr-defined]
            elem = getattr(x, attr)
            if not elem.is_contiguous():
                setattr(x, attr, elem.contiguous())
        return x

    # See Note [Tangents must be contiguous, Part 2]
    @staticmethod
    def coerce_runtime_tangent(x, metadata):
        if not isinstance(x, torch.Tensor):
            return x
        if not is_traceable_wrapper_subclass(x):
            return x
        assert metadata is not None
        (_, expected_tangent_metadata) = metadata
        _, runtime_tangent_metadata = x.__tensor_flatten__()  # type: ignore[attr-defined]
        if runtime_tangent_metadata == expected_tangent_metadata:
            return x
        if not hasattr(x, "__coerce_same_metadata_as_tangent__"):
            raise RuntimeError(
                f"""
During the backward, we encountered a tensor subclass where we guessed its
metadata incorrectly.

Expected metadata: {str(expected_tangent_metadata)}

Runtime metadata: {str(runtime_tangent_metadata)}

shape: {str(cast(torch.Tensor, x).shape)}
To fix this, your tensor subclass must implement the dunder method __force_to_same_metadata__.
"""
            )
        return x.__coerce_same_metadata_as_tangent__(expected_tangent_metadata)  # type: ignore[attr-defined]

    @staticmethod
    def post_compile(
        compiled_fw_func,  # fw_module after compilation + wrappers
        compiled_bw_func,  # bw_module after compilation + wrappers
        maybe_subclass_meta: Optional[SubclassMeta],
        num_symints_saved_for_bw_: int,
        backward_state_indices: List[int],
        disable_amp: bool,
        indices_of_inps_to_detach: List[int],
        lazy_backward_info: Optional[AutogradLazyBackwardCompileInfo],
        aot_config: AOTConfig,
        *,
        fw_metadata: ViewAndMutationMeta,  # runtime metadata
        try_save_cache_entry: Optional[Callable],  # Save cache entry after compilation
    ):
        class CompiledFunction(torch.autograd.Function):
            compiled_fw = compiled_fw_func
            compiled_bw = compiled_bw_func
            metadata: ViewAndMutationMeta = fw_metadata  # type: ignore[assignment]
            maybe_subclass_metadata: Optional[SubclassMeta] = maybe_subclass_meta
            num_symints_saved_for_bw = num_symints_saved_for_bw_
            _compiled_autograd_should_lift = False

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

                # There is a pretty complicated calling convention around what the compiled fw returns.
                # The full list of outputs and their relative order is:
                # (*tokens, *mutated_inputs, *fw_outs, *fw_intermediate_bases, *saved_tensors, *saved_symints)
                # - Note that in the synthetic bases case, mutated_inputs will correspond to an updated version
                #   of the original view, and not the synthetic base
                # - Note that donated buffer logic requires (*saved_tensors, *saved_symints) showing up last
                #   in the fw output order.
                fw_outs = call_func_at_runtime_with_args(
                    CompiledFunction.compiled_fw,
                    args,
                    disable_amp=disable_amp,
                )

                num_outputs = CompiledFunction.metadata.num_outputs
                num_outputs_aliased = CompiledFunction.metadata.num_outputs_aliased
                num_mutated_runtime_inps = (
                    CompiledFunction.metadata.num_mutated_inp_runtime_indices
                )
                num_forward_returns = CompiledFunction.metadata.num_forward_returns

                # Partitioners must put symint arguments at the end separate from tensor arguments
                tensors_saved_for_backwards = fw_outs[
                    CompiledFunction.metadata.tensors_saved_for_backwards_slice
                ]
                assert all(
                    isinstance(x, torch.Tensor) for x in tensors_saved_for_backwards
                )
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

                raw_returns = fw_outs[0:num_forward_returns]

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
                            raw_return_idx = i
                            raw_returns[raw_return_idx] = TensorAlias(
                                raw_returns[raw_return_idx]
                            )

                    if config.debug_assert:
                        user_mutated_inputs_raw = raw_returns[
                            0:num_mutated_runtime_inps
                        ]
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
                    for (i, x) in enumerate(
                        raw_returns_not_including_intermediate_bases
                    )
                    if isinstance(x, torch.Tensor)
                    and not raw_returns_meta[i].requires_grad
                ]
                ctx.mark_non_differentiable(*fw_outs_not_requiring_grad)
                ctx._materialize_non_diff_grads = False
                return tuple(raw_returns)

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
                num_intermediate_bases = (
                    CompiledFunction.metadata.num_intermediate_bases
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

                assert len(flat_args) == expected_grad_outs
                out_info = CompiledFunction.metadata.output_info

                inp_tangents, out_tangents, intermediate_base_tangents = (
                    flat_args[:num_mutated_runtime_inps],
                    flat_args[
                        num_mutated_runtime_inps : num_mutated_runtime_inps
                        + CompiledFunction.metadata.num_outputs
                    ],
                    flat_args[
                        num_mutated_runtime_inps
                        + CompiledFunction.metadata.num_outputs :
                    ],
                )
                # input_info contains info on *every* input,
                # But in the backward(), we are only given grad outputs for every mutated input
                # We then need to filter out the grad outputs that correspond to metadata-only mutations or don't require grad
                input_info = CompiledFunction.metadata.input_info
                inp_tangents_filtered = [
                    x
                    for x, info_idx in zip(
                        inp_tangents,
                        CompiledFunction.metadata.mutated_inp_runtime_indices,
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
                # TODO: replace this with FunctionalizedRngRuntimeWrapper
                rng_args = []
                if CompiledFunction.metadata.is_rng_op_functionalized:
                    # Add the seed and offset to args
                    rng_args = CUDARngStateHelper.get_torch_state_as_tuple()

                # - note: donated buffer logic requires (*ctx.symints, *ctx.saved_tensors) showing up first
                #   in the bw output order.
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

                # TODO: figure out how to refactor the backward properly
                # so I can use aot_dispatch_subclass_wrapper() here.
                if CompiledFunction.maybe_subclass_metadata is not None:
                    tangents = all_args[tangents_start_idx:tangents_end_idx]

                    def get_types_for_tangents(tangents):
                        infos = []
                        idx = 0
                        for a in tangents:
                            if isinstance(a, Tensor) and is_traceable_wrapper_subclass(
                                a
                            ):
                                infos.append(get_types_for_subclass(a))
                            else:
                                infos.append(idx)
                            idx += 1
                        return infos

                    runtime_subclass_info = get_types_for_tangents(tangents)

                    if len(runtime_subclass_info) != len(
                        CompiledFunction.metadata.subclass_tangent_meta
                    ):
                        raise RuntimeError(
                            "The grad inputs should be same number as forward output tangents"
                        )
                    for a, b in zip(
                        runtime_subclass_info,
                        CompiledFunction.metadata.subclass_tangent_meta,
                    ):
                        # Types should match between runtime and traced tangents.
                        # TODO (tmanlaibaatar) Should actually call coerce_runtime_tangent
                        if isinstance(a, List) and (
                            isinstance(b, SubclassCreationMeta) and b.subclass_type
                        ):
                            if not a == b.subclass_type:
                                raise RuntimeError(
                                    "The grad inputs should be same tensor subclass type as forward output"
                                )

                    # Get the number of tangents after unwrapping
                    len_tangents = len(
                        unwrap_tensor_subclasses(
                            tangents,
                            is_joint_structure=False,
                        )
                    )
                    assert CompiledFunction.metadata.traced_tangent_metas is not None
                    all_args = [
                        (
                            AOTDispatchAutograd.coerce_runtime_tangent(
                                t,
                                CompiledFunction.metadata.traced_tangent_metas[
                                    i - tangents_start_idx
                                ],
                            )
                            if tangents_start_idx <= i < tangents_end_idx
                            else t
                        )
                        for i, t in enumerate(all_args)
                    ]
                    all_args = unwrap_tensor_subclasses(
                        all_args, is_joint_structure=False
                    )
                    tangents_start_idx = len(all_args) - len_tangents - len(rng_args)
                    tangents_end_idx = tangents_start_idx + len_tangents

                # Make the tangents contiguous. Note that we must do this after subclass desugaring
                # because inputs to inductor have to be contiguous
                all_args = [
                    (
                        AOTDispatchAutograd._force_contiguous(t)
                        if (tangents_start_idx <= i < tangents_end_idx)
                        else t
                    )
                    for i, t in enumerate(all_args)
                ]

                def call_compiled_backward():
                    if ctx._is_compiled_autograd_tracing():
                        if lazy_backward_info is None:
                            raise RuntimeError(
                                """This compiled backward function was saved by AOTAutogradCache, which does not support
                            compiled autograd. Please turn off AOTAutogradCache using `ENABLE_AOT_AUTOGRAD_CACHE=0` to continue."""
                            )
                        bw_module = lazy_backward_info.bw_module
                        # For compiled autograd, run raw FX graph so that it can be inlined into the larger graph
                        symints = ctx._get_compiled_autograd_symints()
                        assert len(symints) == len(ctx.symints)
                        all_args[: len(symints)] = symints
                        if backward_state_indices:
                            assert (
                                ctx._compiled_autograd_backward_state.proxy is not None
                            )
                            all_args.append(ctx._compiled_autograd_backward_state)
                        context = (
                            torch._C._DisableAutocast if disable_amp else nullcontext
                        )
                        with context():
                            out = normalize_as_list(bw_module(*all_args))
                        # TODO: replace with post_compile wrapper
                        out = FunctionalizedRngRuntimeWrapper()._functionalized_rng_runtime_epilogue(
                            CompiledFunction.metadata, out, offset_index=len(out) - 1
                        )
                        return tuple(out)
                    assert (
                        not backward_state_indices
                    ), "BackwardState requires CompiledAutograd"
                    ctx.maybe_clear_saved_tensors()

                    saved_tensors_use_once = (
                        not torch._C._autograd._get_current_graph_task_keep_graph()
                    )

                    if CompiledFunction.compiled_bw is None:
                        assert lazy_backward_info is not None

                        if not saved_tensors_use_once:
                            fw_metadata.bw_donated_idxs = []
                            # Update bw_donated_idxs if using lazy_backward_info from `aot_dispatch_autograd`
                            if (
                                hasattr(lazy_backward_info, "saved_context")
                                and hasattr(
                                    lazy_backward_info.saved_context, "fw_metadata"
                                )
                                and hasattr(
                                    lazy_backward_info.saved_context.fw_metadata,  # type: ignore[union-attr]
                                    "bw_donated_idxs",
                                )
                            ):
                                lazy_backward_info.saved_context.fw_metadata.bw_donated_idxs = (  # type: ignore[union-attr]
                                    []
                                )

                        bw_module = lazy_backward_info.bw_module
                        placeholder_list = lazy_backward_info.placeholder_list
                        saved_context = lazy_backward_info.saved_context
                        saved_compile_context = lazy_backward_info.saved_compile_context

                        context = (
                            torch._C._DisableAutocast if disable_amp else nullcontext
                        )
                        with tracing(saved_context), compile_context(
                            saved_compile_context
                        ), context(), track_graph_compiling(aot_config, "backward"):
                            CompiledFunction.compiled_bw = aot_config.bw_compiler(
                                bw_module, placeholder_list
                            )
                            # Maybe save cache entry
                            if try_save_cache_entry is not None:
                                try_save_cache_entry(
                                    CompiledFunction.compiled_bw, fw_metadata
                                )

                    if (
                        torch._functorch.config.donated_buffer
                        and not saved_tensors_use_once
                        and fw_metadata.bw_donated_idxs != []
                    ):
                        torch._check(
                            False,
                            lambda: (
                                "This backward function was compiled with non-empty donated "
                                "buffers which requires create_graph=False and retain_graph=False. "
                                "Please keep backward(create_graph=False, retain_graph=False) "
                                "across all backward() function calls, or set "
                                "torch._functorch.config.donated_buffer=False to disable "
                                "donated buffer."
                            ),
                        )

                    out = call_func_at_runtime_with_args(
                        CompiledFunction.compiled_bw,
                        all_args,
                        steal_args=True,
                        disable_amp=disable_amp,
                    )
                    # TODO: replace this with FunctionalizedRngRuntimeWrapper.post_compile
                    out = FunctionalizedRngRuntimeWrapper()._functionalized_rng_runtime_epilogue(
                        CompiledFunction.metadata, out, offset_index=len(out) - 1
                    )
                    return tuple(out)

                # Backward with forward inputs mutations is not supported in double backward.
                if (
                    torch.is_grad_enabled()
                    and CompiledFunction.metadata.indices_of_inputs_that_requires_grad_with_mutations_in_bw
                ):
                    raise RuntimeError(
                        "aot_autograd does not support input mutations with requires_grad in backward for create_graph=True"
                    )

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
                            # TODO: figure out how to refactor the backward properly
                            # so I can use aot_dispatch_subclass_wrapper() here.
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

        compiled_function = RuntimeWrapper(
            indices_of_inps_to_detach=indices_of_inps_to_detach,
            trace_joint=True,
            disable_amp=disable_amp,
        ).post_compile(
            CompiledFunction.apply,
            aot_config,
            runtime_metadata=fw_metadata,
        )

        return compiled_function


@dataclass
class DebugAssertWrapper(CompilerWrapper):
    flat_requires_grad: List[Optional[bool]] = field(default_factory=list)

    def post_compile(
        self,
        compiled_fn,
        aot_config: AOTConfig,
        *,
        runtime_metadata: ViewAndMutationMeta,
    ):
        @wraps(compiled_fn)
        def debug_compiled_function(args: List[Any]):
            # TODO: Check aliasing relationships
            # TODO: Check strides for metadata mutation
            # (NB: ideally, this logic is factored out of this function and
            # you move these debug checks there)

            # Check requires grad.  Bad case is when we compiled with
            # requires_grad = False, but input requires_grad = True
            # (vice versa is OK; we compute a gradient and then throw
            # it away when it hits the input.)
            for i, a in enumerate(args):
                can_require_grad = self.flat_requires_grad[i]
                if can_require_grad is None:
                    assert not isinstance(a, Tensor)
                elif not can_require_grad:
                    assert not a.requires_grad, format_guard_bug_msg(
                        aot_config,
                        f"{describe_input(i, aot_config)} would not require grad",
                    )

            return compiled_fn(args)

        return debug_compiled_function


def pre_compile(
    wrappers: List[CompilerWrapper],
    flat_fn: Callable,
    flat_args: List[Any],
    aot_config: AOTConfig,
    *,
    fw_metadata: ViewAndMutationMeta,
) -> Tuple[Callable, List[Tensor], ViewAndMutationMeta]:
    """
    Runs a sequence of wrappers on the given function and arguments.
    Mutates wrappers in place.
    """
    for wrapper in wrappers:
        flat_fn, flat_args, fw_metadata = wrapper.pre_compile(
            flat_fn, flat_args, aot_config, fw_metadata=fw_metadata
        )
    return flat_fn, flat_args, fw_metadata


def post_compile(
    wrappers: List[CompilerWrapper],
    compiled_fn: Callable,
    aot_config: AOTConfig,
    *,
    runtime_metadata: ViewAndMutationMeta,
) -> Tuple[Callable, ViewAndMutationMeta]:
    """
    Runs a sequence of wrappers on the given function. Should be called after pre_compile()
    """
    for wrapper in reversed(wrappers):
        compiled_fn = wrapper.post_compile(
            compiled_fn, aot_config, runtime_metadata=runtime_metadata
        )
    return compiled_fn, runtime_metadata


def make_runtime_safe(
    fw_metadata: ViewAndMutationMeta,
    maybe_subclass_meta: Optional[SubclassMeta],
):
    """
    Calls make_runtime_safe on all ViewAndMutationMetas.
    Modifies both arguments. Allows ViewAndMutationMetas to
    be safely cached in AOTAutogradCache.
    """
    fw_metadata.make_runtime_safe()
    if maybe_subclass_meta is not None:
        maybe_subclass_meta.fw_metadata.make_runtime_safe()
