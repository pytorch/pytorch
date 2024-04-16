"""
This module defines runtime wrappers, which, based on previous analysis attempts to:
1. process the inputs and outputs
2. apply mutations
3. handle functionalized randomness
4. deduplicate inputs and consolidate views into their bases (see input_output_analysis)
"""

import collections
import pprint
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.utils.dlpack
from torch import Tensor
from torch._guards import DuplicateInputs, TracingContext
from torch._prims_common import CUDARngStateHelper
from torch.multiprocessing.reductions import StorageWeakRef
from .. import config
from .collect_metadata_analysis import run_functionalized_fw_and_collect_metadata

from .functional_utils import gen_alias_from_base
from .input_output_analysis import (
    compute_overlapping_inputs,
    create_synthetic_base_metadata,
    remove_dupe_metadata,
)
from .logging_utils import describe_input, format_guard_bug_msg
from .schemas import (
    AOTConfig,
    InputAliasInfo,
    OutputType,
    SubclassCreationMeta,
    TensorAlias,
    ViewAndMutationMeta,
)
from .subclass_utils import (
    requires_subclass_dispatch,
    unwrap_tensor_subclasses,
    wrap_tensor_subclasses,
)

from .utils import (
    call_func_at_runtime_with_args,
    make_boxed_func,
    partial_flatten_asdict,
    strict_zip,
)


zip = strict_zip


# The wrapper created by this function handles all of the runtime aliasing and mutation "epilogue" logic
# that needs to run after the compiled function.
#
# This function accepts a trace_joint flag, indicating whether or not we're generating the runtime
# epilogue for a forward-only inference graph, or for an autograd.Function.apply function.
# This is because there are some minor differences in how we treat these cases at runtime:
# - resize_() is currently handled in the inference case, but not fully handled in the autograd case.
# - the autograd cases inserts TensorAlias wrapper objects for outputs that alias inputs
def create_runtime_wrapper(
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
    num_tokens = len(runtime_metadata.tokens)
    for info in runtime_metadata.output_info:
        if (
            info.output_type == OutputType.alias_of_input
            or info.output_type == OutputType.is_input
        ):
            assert isinstance(info.base_idx, int)
            epilogue_args_idx.append(info.base_idx + num_tokens)

    def runtime_wrapper(args: List[Any]):
        if config.unlift_effect_tokens:
            assert num_tokens == 0
        elif num_tokens > 0:
            # Pass in effect tokens (See Note [Side-Effectful Tokens in AOTAutograd])
            old_args = args
            args = [[None] * num_tokens, *args]
            old_args.clear()

        # stash a ref to each input tensor we plan to use after the compiled function
        orig_inputs = {i: args[i] for i in epilogue_args_idx}

        if trace_joint:
            args_ = list(args)
            # See Note [Detaching inputs that never need gradients]
            for idx in indices_of_inps_to_detach:
                if isinstance(args_[idx], torch.Tensor):
                    args_[idx] = args_[idx].detach()
            with torch.autograd._force_original_view_tracking(True):
                all_outs = call_func_at_runtime_with_args(
                    compiled_fn, args_, disable_amp=disable_amp, steal_args=True
                )
        else:
            # When we have an inference graph, we run with torch.no_grad.
            # It's possible to get an inference graph with inputs that require grad,
            # in which case we want to make sure autograd is disabled
            # (since e.g., inductor will generate aten.addmm.out calls which autograd will complain on)
            if torch.is_grad_enabled():
                with torch.no_grad():
                    all_outs = call_func_at_runtime_with_args(
                        compiled_fn, args, disable_amp=disable_amp, steal_args=True
                    )
            else:
                all_outs = call_func_at_runtime_with_args(
                    compiled_fn, args, disable_amp=disable_amp, steal_args=True
                )
        del args

        num_mutated_runtime_inps = runtime_metadata.num_mutated_inp_runtime_indices
        num_intermediate_bases = runtime_metadata.num_intermediate_bases

        if keep_input_mutations and trace_joint:
            num_input_mutations_handled_by_autograd = (
                runtime_metadata.num_mutated_graph_handled_indices_seen_by_autograd
            )
            # autograd.Function requires us to return the mutated inputs as extra outputs to the autograd.Function.forward
            if num_input_mutations_handled_by_autograd > 0:
                all_outs = all_outs[:-num_input_mutations_handled_by_autograd]

        assert (
            len(all_outs)
            == num_mutated_runtime_inps
            + runtime_metadata.num_outputs
            + num_intermediate_bases
            + num_tokens
        )

        # Toss out the effect tokens (See Note [Side-Effectful Tokens in AOTAutograd])
        all_outs = all_outs[num_tokens:]

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
            if runtime_metadata.num_intermediate_bases > 0:
                fw_outs_no_intermediate_bases = fw_outs[
                    : -runtime_metadata.num_intermediate_bases
                ]
                intermediate_bases = fw_outs[-runtime_metadata.num_intermediate_bases :]
            else:
                fw_outs_no_intermediate_bases = fw_outs
                intermediate_bases = []

            assert len(fw_outs_no_intermediate_bases) == len(
                runtime_metadata.output_info
            )
            fw_outs_including_aliases = []
            for i, (o, info) in enumerate(
                zip(fw_outs_no_intermediate_bases, runtime_metadata.output_info)
            ):
                if info.output_type in [
                    OutputType.non_alias,
                    OutputType.unsafe_view_alias,
                    OutputType.custom_function_view,
                ]:
                    fw_outs_including_aliases.append(o)
                    continue
                if trace_joint:
                    assert isinstance(o, TensorAlias)
                    o_ = o.alias
                else:
                    o_ = o

                o_grad = runtime_metadata.output_info[i].requires_grad
                if info.output_type == OutputType.alias_of_input:
                    aliased_base_tensor = orig_inputs[info.base_idx + num_tokens]  # type: ignore[index]
                    regenerated_out = gen_alias_from_base(
                        aliased_base_tensor, o_, o_grad, info.functional_tensor
                    )
                    fw_outs_including_aliases.append(regenerated_out)
                    continue
                elif info.output_type == OutputType.is_input:
                    aliased_base_tensor = orig_inputs[info.base_idx + num_tokens]  # type: ignore[index]
                    regenerated_out = aliased_base_tensor
                    fw_outs_including_aliases.append(regenerated_out)
                    continue
                elif info.output_type == OutputType.alias_of_intermediate:
                    base_tensor_list = intermediate_bases
                elif (
                    info.output_type == OutputType.alias_of_intermediate_save_as_output
                ):
                    base_tensor_list = intermediate_bases
                else:
                    assert (
                        info.output_type
                        == OutputType.alias_of_intermediate_base_is_user_output
                    )
                    base_tensor_list = fw_outs_no_intermediate_bases
                aliased_base_tensor = base_tensor_list[info.base_idx]
                # TODO: handle the custom autograd function case here.
                # We need a way to check whether a tensor came from a custom autograd fn from python,
                # AND a way to replay that custom view fn.
                regenerated_out = gen_alias_from_base(
                    aliased_base_tensor, o_, o_grad, info.functional_tensor
                )
                fw_outs_including_aliases.append(regenerated_out)
            ret_outs = fw_outs_including_aliases
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
            torch.set_grad_enabled(runtime_metadata.grad_enabled_mutation)
        return ret_outs

    return runtime_wrapper


# Calling convention: If we are running functionalized RNG, then outs consists
# of (user_outs, rng_offset)
def functionalized_rng_runtime_epilogue(
    metadata: ViewAndMutationMeta, outs, return_new_outs=True
):
    if metadata.is_rng_op_functionalized:
        assert metadata.num_outputs_rng_offset == 1
        new_rng_offset = outs[-1]
        CUDARngStateHelper.set_new_offset(new_rng_offset)
        if return_new_outs:
            user_outs = outs[:-1]
            return user_outs
        else:
            return None
    return outs


# This wrapper handles the AOTDispatch runtime logic for tensor subclasses.
# At runtime, we have a compiled function that knows how to operate on the domain of DenseTensor -> DenseTensor,
# But the user might have passed us some tensor subclass inputs (or expect some subclass tensor outputs).
# This function handles the wrapping and unwrapping of tensor subclasses at runtime.
def aot_dispatch_subclass_wrapper(
    runtime_fn: Callable,
    *,
    subclass_metas: List[Union[int, SubclassCreationMeta]],
    num_fw_outs_saved_for_bw: Optional[int],
) -> Callable:
    def inner_fn(args: List[Any]):
        unwrapped_args = unwrap_tensor_subclasses(args, is_joint_structure=False)
        args.clear()
        # expectation: runtime_fn is a boxed fn
        unwrapped_outs = runtime_fn(unwrapped_args)
        wrapped_outs = wrap_tensor_subclasses(
            unwrapped_outs,
            subclass_metas=subclass_metas,
            num_fw_outs_saved_for_bw=num_fw_outs_saved_for_bw,
            is_runtime=True,
        )
        return wrapped_outs

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
def aot_wrapper_dedupe(
    flat_fn,
    flat_args: List[Tensor],
    aot_config: AOTConfig,
    *,
    compiler_fn,
    fw_metadata,
):
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
        return compiler_fn(flat_fn, leaf_flat_args, aot_config, fw_metadata=fw_metadata)

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
    keep_arg_mask = []
    # Implicitly map duped arg position (list index) to de-duped arg position
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

    # NB: Hot path, avoid set lookups here
    # TODO: Can avoid the zip here too, probably
    def remove_dupe_args(args):
        return [t for t, keep in zip(args, keep_arg_mask) if keep]

    def add_dupe_args(args):
        return [args[add_dupe_map[i]] for i in range(duped_arg_len)]

    deduped_flat_args = remove_dupe_args(flat_args)

    # Update our input metadata to remove duped input metadata.
    updated_fw_metadata = remove_dupe_metadata(fw_metadata, keep_arg_mask, add_dupe_map)

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
                kept_arg_source = aot_config.aot_autograd_arg_pos_to_source[kept_pos]
                tracing_context.guards_context.aotautograd_guards.append(  # type: ignore[attr-defined]
                    DuplicateInputs(kept_arg_source, dupe_arg_source)
                )

    @wraps(flat_fn)
    def wrapped_flat_fn(*args):
        return flat_fn(*add_dupe_args(args))

    if config.debug_assert:
        ref_fw_metadata = run_functionalized_fw_and_collect_metadata(
            wrapped_flat_fn,
            keep_input_mutations=fw_metadata.keep_input_mutations,
            is_train=fw_metadata.is_train,
        )(*deduped_flat_args)
        assert (
            ref_fw_metadata == updated_fw_metadata
        ), f"ref_metadata={str(ref_fw_metadata)}, actual_metadata={str(updated_fw_metadata)}"

    compiled_fn = compiler_fn(
        wrapped_flat_fn, deduped_flat_args, aot_config, fw_metadata=updated_fw_metadata
    )

    @wraps(compiled_fn)
    def wrapped_compiled_fn(args: List[Any]):
        deduped_args = remove_dupe_args(args)
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
        new_args = add_dupe_args(remove_dupe_args(args))
        seen: Dict[Any, None] = {}
        for i, (x, y) in enumerate(zip(new_args, args)):
            seen[y] = None
            assert x is y, format_guard_bug_msg(
                aot_config,
                f"{describe_input(i, aot_config)} would be a duplicate of "
                f"{describe_input(add_dupe_map[i], aot_config)}",
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
# pre-condition: aot_wrapper_dedup has already run.
# (This function will in theory work if there are duplicate args.
# However, the synthetic base code path is a bit sub-optimal, and running with dupe'd inputs
# would cause us to hit that path more frequently).
def aot_wrapper_synthetic_base(
    flat_fn,
    flat_args: List[Tensor],
    aot_config: AOTConfig,
    *,
    fw_metadata: ViewAndMutationMeta,
    # Currently, the only reason we need to plumb this bool is because
    # the synthetic base code prohibits more cases in the autograd case than the inference case.
    needs_autograd: bool,
    compiler_fn,
):
    is_inference = not needs_autograd
    flat_args_with_synthetic_bases, synthetic_base_info = merge_view_inputs(
        flat_args,
        fw_metadata.input_info,
        is_inference=is_inference,
    )
    # Happy path: we don't need synthetic bases
    if synthetic_base_info is None:
        return compiler_fn(flat_fn, flat_args, aot_config, fw_metadata=fw_metadata)

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

    num_aliased_args_with_metadata_mutations = len(
        aliased_arg_idx_with_metadata_mutations
    )

    def _unpack_synthetic_bases(primals: Tuple[Any, ...]) -> List[Any]:
        f_args_inner = []
        for inner_idx_or_tuple in synthetic_base_info:
            if isinstance(inner_idx_or_tuple, int):
                f_args_inner.append(primals[inner_idx_or_tuple])
            else:
                inner_base_idx, view_tensor = inner_idx_or_tuple
                base = primals[inner_base_idx]
                view_arg = gen_alias_from_base(
                    base, view_tensor, view_tensor.requires_grad
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
            if i in aliased_arg_idx_with_metadata_mutations
        ]
        if len(aliased_args_with_metadata_mutations) > 0:
            return *(flat_fn(*unpacked_args)), *aliased_args_with_metadata_mutations
        else:
            return flat_fn(*unpacked_args)

    if config.debug_assert:
        ref_fw_metadata = run_functionalized_fw_and_collect_metadata(
            wrapped_flat_fn,
            keep_input_mutations=fw_metadata.keep_input_mutations,
            is_train=fw_metadata.is_train,
        )(*flat_args_with_synthetic_bases)
        assert ref_fw_metadata == fw_metadata_updated, (
            f"ref_metadata={pprint.pformat(partial_flatten_asdict(ref_fw_metadata))}, "
            f"\nactual_metadata={pprint.pformat(partial_flatten_asdict(fw_metadata_updated))}"
        )

    compiled_fn = compiler_fn(
        wrapped_flat_fn,
        flat_args_with_synthetic_bases,
        aot_config,
        fw_metadata=fw_metadata_updated,
    )

    @wraps(compiled_fn)
    def wrapped_compiled_fn(args):
        args_with_synthetic_bases, synthetic_base_info = merge_view_inputs(
            args, fw_metadata.input_info, is_inference=is_inference
        )
        assert synthetic_base_info is not None
        aliased_args_w_metadata_mutations = [
            args[i] for i in aliased_arg_idx_with_metadata_mutations
        ]
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
