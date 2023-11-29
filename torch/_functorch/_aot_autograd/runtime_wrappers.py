"""
This module defines runtime wrappers, which, based on previous analysis
attempts to process the inputs and outputs, apply mutations, functionalize randomness
and dispatch subclasses.
"""

from contextlib import nullcontext
from typing import Callable, List, Optional, Union
from unittest.mock import patch

import torch
from torch._decomp.decompositions_for_rng import PhiloxStateTracker
from torch._guards import detect_fake_mode
from torch._prims_common import CUDARngStateHelper

from .functional_utils import gen_alias_from_base
from .schemas import OutputType, SubclassCreationMeta, TensorAlias, ViewAndMutationMeta
from .subclass_utils import unwrap_tensor_subclasses, wrap_tensor_subclasses
from .utils import call_func_at_runtime_with_args, make_boxed_func


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

    def runtime_wrapper(*args):
        if trace_joint:
            args_ = list(args)
            # See Note [Detaching inputs that never need gradients]
            for idx in indices_of_inps_to_detach:
                if isinstance(args_[idx], torch.Tensor):
                    args_[idx] = args_[idx].detach()
            with torch.autograd._force_original_view_tracking(True):
                all_outs = call_func_at_runtime_with_args(
                    compiled_fn,
                    args_,
                    disable_amp=disable_amp,
                )
        else:
            # When we have an inference graph, we run with torch.no_grad.
            # It's possible to get an inference graph with inputs that require grad,
            # in which case we want to make sure autograd is disabled
            # (since e.g., inductor will generate aten.addmm.out calls which autograd will complain on)
            with torch.no_grad():
                all_outs = call_func_at_runtime_with_args(
                    compiled_fn,
                    args,
                    disable_amp=disable_amp,
                )

        num_mutated_runtime_inps = runtime_metadata.num_mutated_inp_runtime_indices
        num_intermediate_bases = runtime_metadata.num_intermediate_bases

        if keep_input_mutations and trace_joint:
            num_graph_handled = runtime_metadata.num_mutated_graph_handled_indices
            # autograd.Function requires us to return the mutated inputs as extra outputs to the autograd.Function.forward
            if num_graph_handled > 0:
                all_outs = all_outs[:-num_graph_handled]

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
                original_inpt = args[inpt_idx]
                updated_inpt = updated_inputs[i]
                if meta.mutates_storage_metadata:
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
                    aliased_base_tensor = args[info.base_idx]  # type: ignore[index]
                    regenerated_out = gen_alias_from_base(
                        aliased_base_tensor, o_, o_grad
                    )
                    fw_outs_including_aliases.append(regenerated_out)
                    continue
                elif info.output_type == OutputType.is_input:
                    aliased_base_tensor = args[info.base_idx]  # type: ignore[index]
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
                regenerated_out = gen_alias_from_base(aliased_base_tensor, o_, o_grad)
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


def create_functionalized_rng_ops_wrapper(func, args, trace_joint=True):
    # Functionalization of rng ops changes the calling convention of the joint graph.
    # It goes from (primals, tangents) to (seed, offset, primals, tangents)
    # At runtime, we pass on the current seed and offset. This is hidden from
    # the user.
    fake_mode = detect_fake_mode()
    if fake_mode is None:
        fake_mode = nullcontext()

    def override_get_rng_state(device: Union[int, str, torch.device] = "cuda"):
        out = PhiloxStateTracker.get_state_as_tensor()
        return out

    def override_set_rng_state(x, device: Union[int, str, torch.device] = "cuda"):
        PhiloxStateTracker.set_state_from_tensor(x)

    def append_rng_offsets(args):
        if trace_joint:
            # args signature before: Tuple(fwd_outputs), Tuple(bwd_outputs)
            # args signature after: Tuple(fwd_outputs, new_fwd_rng_offset), Tuple(bwd_offset, new_bwd_rng_offset)
            return (
                (*args[0], PhiloxStateTracker.get_updated_fwd_offset()),
                (*args[1], PhiloxStateTracker.get_updated_bwd_offset()),
            )
        else:
            # args signature before: Tuple(fwd_outputs)
            # args signature after: Tuple(fwd_outputs, new_fwd_rng_offset)
            return (*args, PhiloxStateTracker.get_updated_fwd_offset())

    def traced_joint(
        primals, tangents, fwd_seed, fwd_base_offset, bwd_seed, bwd_base_offset
    ):
        with patch("torch.cuda.get_rng_state", override_get_rng_state), patch(
            "torch.cuda.set_rng_state", override_set_rng_state
        ):
            return append_rng_offsets(func(primals, tangents))

    def traced_forward(*primals_fwd_seed_fwd_base_offset):
        # The signature is (*primals, seed, offset)
        with patch("torch.cuda.get_rng_state", override_get_rng_state), patch(
            "torch.cuda.set_rng_state", override_set_rng_state
        ):
            return append_rng_offsets(func(*primals_fwd_seed_fwd_base_offset[:-2]))

    if trace_joint:
        # Get the current seed and offset to setup tracing.
        fwd_seed, fwd_base_offset = CUDARngStateHelper.get_torch_state_as_tuple(
            fake_mode
        )
        bwd_seed, bwd_base_offset = CUDARngStateHelper.get_torch_state_as_tuple(
            fake_mode
        )
        PhiloxStateTracker.record_state(fwd_seed, fwd_base_offset, "forward")
        PhiloxStateTracker.record_state(bwd_seed, bwd_base_offset, "backward")
        return traced_joint, (
            *args,
            fwd_seed,
            fwd_base_offset,
            bwd_seed,
            bwd_base_offset,
        )
    else:
        # Get the current seed and offset to setup tracing.
        fwd_seed, fwd_base_offset = CUDARngStateHelper.get_torch_state_as_tuple(
            fake_mode
        )
        PhiloxStateTracker.record_state(fwd_seed, fwd_base_offset, "forward")
        return traced_forward, (*args, fwd_seed, fwd_base_offset)


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
    def inner_fn(args):
        unwrapped_args = unwrap_tensor_subclasses(args, is_joint_structure=False)
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
