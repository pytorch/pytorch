"""
This module is one of the analysis modules - it takes as input a function or graph
and some preexisting properties, and returns some data that is useful for deciding
how to further proceed with compilation or construct runtime wrappers.

In particular, the following analyses are provided:
1. refine the view and mutation metadata collected previously - removing duplicate
   inputs or mapping views to their bases.
2. Based on view base analysis, it may merge inputs of different views into a single
   input corresponding to a common base.
3. We also analyze the function signature for export graphs.
"""

import collections
import itertools
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.utils._pytree as pytree
from torch import Tensor
from torch._logging import getArtifactLogger
from torch._subclasses.functional_tensor import FunctionalTensor
from torch.fx.experimental.symbolic_shapes import is_concrete_int
from torch.multiprocessing.reductions import StorageWeakRef
from .functional_utils import _get_mutation_type
from .schemas import (
    BackwardSignature,
    GraphSignature,
    InputAliasInfo,
    OutputAliasInfo,
    OutputType,
    ViewAndMutationMeta,
)
from .utils import strict_zip

zip = strict_zip

aot_graphs_log = getArtifactLogger(__name__, "aot_graphs")


def remove_dupe_metadata(
    m: ViewAndMutationMeta,
    keep_arg_mask: List[bool],
    add_dupe_map: List[int],
) -> ViewAndMutationMeta:
    assert len(m.input_info) == len(keep_arg_mask)
    # Easy invariant: the first argument should never be a dupe (it will be kept)
    assert len(keep_arg_mask) > 0 and keep_arg_mask[0]

    # Filter dupe'd mutated inputs out of traced_tangents
    num_data_mutations = len([x for x in m.input_info if x.mutates_data])
    other_traced_tangents = m.traced_tangents[num_data_mutations:]
    inp_traced_tangents = m.traced_tangents[:num_data_mutations]
    filtered_inp_traced_tangents = [
        x
        for i, x in enumerate(inp_traced_tangents)
        if keep_arg_mask[m.mutated_inp_runtime_indices[i]]
    ]
    traced_tangents = filtered_inp_traced_tangents + other_traced_tangents

    return ViewAndMutationMeta(
        input_info=[x for i, x in enumerate(m.input_info) if keep_arg_mask[i]],
        # For outputs that are views of inputs, we store the index of the input that the output
        # was generated from. Need to update that index to account for removed dupes.
        output_info=[
            OutputAliasInfo(
                output_type=o.output_type,
                raw_type=o.raw_type,
                dynamic_dims=o.dynamic_dims,
                base_idx=None if o.base_idx is None else add_dupe_map[o.base_idx],
                requires_grad=o.requires_grad,
            )
            for o in m.output_info
        ],
        num_intermediate_bases=m.num_intermediate_bases,
        keep_input_mutations=m.keep_input_mutations,
        traced_tangents=traced_tangents,
        # We are guaranteed not to get here, since dupes are not supported today with subclass inputs.
        subclass_inp_meta=[],
        subclass_fw_graph_out_meta=[],
        subclass_tangent_meta=[],
        is_train=m.is_train,
    )


# Given our ViewAndMutation metadata, this fn constructs a new set of metadata,
# after adding synthetic base arguments to the function.
# Most of the work in this fn is slogging through all of the metadata corresponding to inputs,
# and updating it with our synthetic base calling convention.
#
# When config.debug_assert is set, we automatically regenerate the metadata
# and compare it to this output for sanity.
#
# In addition to the updated metadata, also return the list of input indices
# that will need to be updated in the synthetic base epilogue


# Given our ViewAndMutation metadata, this fn constructs a new set of metadata,
# after adding synthetic base arguments to the function.
# Most of the work in this fn is slogging through all of the metadata corresponding to inputs,
# and updating it with our synthetic base calling convention.
#
# When config.debug_assert is set, we automatically regenerate the metadata
# and compare it to this output for sanity.
#
# In addition to the updated metadata, also return the list of input indices
# that will need to be updated in the synthetic base epilogue
def create_synthetic_base_metadata(
    m: ViewAndMutationMeta,
    # Maps each outer argument idx to its inner idx (or, if this outer arg is generated from a
    # synthetic base, you get a tuple of (i, TensorMeta), telling you the base tensor idx, and view metadata)
    synthetic_base_info: List[Union[int, Tuple[int, torch.Tensor]]],
    outer_args: List[Any],
    inner_args: List[Any],
) -> Tuple[ViewAndMutationMeta, List[int]]:
    # maps inner arg indices to outer arg indices
    synthetic_base_to_indices: Dict[int, List[int]] = {}
    for inner_idx in range(len(inner_args)):
        outer_aliased_indices_of_current_base_arg = [
            outer_idx
            for outer_idx, inner_idx_or_tuple in enumerate(synthetic_base_info)
            if (isinstance(inner_idx_or_tuple, int) and inner_idx_or_tuple == inner_idx)
            or (
                isinstance(inner_idx_or_tuple, tuple)
                and inner_idx_or_tuple[0] == inner_idx
            )
        ]
        synthetic_base_to_indices[inner_idx] = outer_aliased_indices_of_current_base_arg

    # given the requires_grad info on mutated inputs,
    # generate the requires_grad info on those same mutated inputs, but after constructing synthetic bases.
    input_infos = []
    for outer_indices in synthetic_base_to_indices.values():
        # leaf-ness should be all-or-nothing for aliased tensor.
        # (aka if "a" and "b" are views, then a.is_leaf == b.is_leaf)
        any_leaf = any(m.input_info[x].is_leaf for x in outer_indices)
        all_leaf = all(m.input_info[x].is_leaf for x in outer_indices)
        assert any_leaf == all_leaf

        mutates_data = (
            True
            if len(outer_indices) > 1
            else m.input_info[outer_indices[0]].mutates_data
        )
        mutates_metadata = (
            False
            if len(outer_indices) > 1
            else m.input_info[outer_indices[0]].mutates_metadata
        )
        requires_grad = any(m.input_info[x].requires_grad for x in outer_indices)
        mutations_hidden_from_autograd = all(
            m.input_info[x].mutations_hidden_from_autograd for x in outer_indices
        )
        mutations_under_no_grad_or_inference_mode = all(
            m.input_info[x].mutations_under_no_grad_or_inference_mode
            for x in outer_indices
        )
        mutation_type = _get_mutation_type(
            m.keep_input_mutations,
            mutates_data,
            mutates_metadata,
            mutations_hidden_from_autograd,
            mutations_under_no_grad_or_inference_mode,
            requires_grad,
        )

        inpt_info = InputAliasInfo(
            # If len(outer_indices) > 1, then this input is a synthetic base.
            # The invariant is that to the rest of aot autograd, synthetic bases only show up if
            # one of their aliases gets a data mutation. And if any of their aliases get metadata
            # mutations, they will be hidden from the rest of aot autograd.
            mutates_data=mutates_data,
            mutates_metadata=mutates_metadata,
            mutations_hidden_from_autograd=all(
                m.input_info[x].mutations_hidden_from_autograd for x in outer_indices
            ),
            mutates_storage_metadata=False
            if len(outer_indices) > 1
            else m.input_info[outer_indices[0]].mutates_storage_metadata,
            mutations_under_no_grad_or_inference_mode=mutations_under_no_grad_or_inference_mode,
            is_leaf=any_leaf,
            requires_grad=requires_grad,
            mutation_type=mutation_type,
        )
        input_infos.append(inpt_info)

    # Find any inputs that fulfill the following criteria:
    # (1) They are part of a synthetic base (because they alias another input,
    #      and at least one input experiences a data mutation)
    # (2) They experience a metadata mutation
    outer_aliased_arg_idx_with_metadata_mutations = [
        outer_idx
        for outer_idx, inpt_info in enumerate(m.input_info)
        if inpt_info.mutates_metadata
        and not isinstance(synthetic_base_info[outer_idx], int)
    ]

    # grab the original requires grad info on the outputs, except the ones from the mutated inputs
    input_metadata_output_info = [
        OutputAliasInfo(
            output_type=OutputType.alias_of_input,
            raw_type=FunctionalTensor,
            dynamic_dims={
                i
                for i, s in enumerate(outer_args[outer_idx].shape)
                if not is_concrete_int(s)
            },
            base_idx=synthetic_base_info[outer_idx][0],  # type: ignore[index]
            requires_grad=outer_args[outer_idx].requires_grad,
        )
        for outer_idx in outer_aliased_arg_idx_with_metadata_mutations
    ]
    existing_output_infos = [
        OutputAliasInfo(
            output_type=o.output_type,
            raw_type=o.raw_type,
            dynamic_dims=o.dynamic_dims,
            # Map the input idx pre-synthetic-bases to the new idx post-synthetic-bases
            base_idx=None  # type: ignore[arg-type]
            if o.base_idx is None
            else synthetic_base_info[o.base_idx]
            if isinstance(synthetic_base_info[o.base_idx], int)
            else synthetic_base_info[o.base_idx][0],  # type: ignore[index]
            requires_grad=o.requires_grad,
        )
        for o in m.output_info
    ]

    inner_mutated_tangents = [
        x
        for inner_idx, x in enumerate(inner_args)
        if input_infos[inner_idx].mutates_data and input_infos[inner_idx].requires_grad
    ]

    output_info = existing_output_infos + input_metadata_output_info
    # Regenerate traced tangents to include mutated inputs including synthetic bases
    traced_tangents = (
        inner_mutated_tangents + m.traced_tangents[len(inner_mutated_tangents) :]
    )

    return (
        ViewAndMutationMeta(
            input_info=input_infos,
            output_info=output_info,
            num_intermediate_bases=m.num_intermediate_bases,
            keep_input_mutations=m.keep_input_mutations,
            traced_tangents=traced_tangents,
            # We are guaranteed not to get here, since synthetic_base codepaths are not supported today with subclass inputs.
            subclass_inp_meta=[],
            subclass_fw_graph_out_meta=[],
            subclass_tangent_meta=[],
            is_train=m.is_train,
        ),
        outer_aliased_arg_idx_with_metadata_mutations,
    )


# Assumption: x and y are known to share a storage, and we are trying to determine
# if their memory is actually completely disjoint, based on sizes/strides/storage_offset
def _tensors_definitely_do_not_overlap(x, y):
    if x is y:
        return False
    if x.numel() == 0 or y.numel() == 0:
        return True

    # Make x always on the left
    if x.storage_offset() > y.storage_offset():
        x, y = y, x
    # Short-circuit in the "obvious" overlapping case: both tensors are contiguous
    if x.is_contiguous() and y.is_contiguous():
        if x.storage_offset() + x.numel() > y.storage_offset():
            # definitely overlap
            return False
        else:
            # definitely no overlap
            return True

    if x.dim() == 2 and y.dim() == 2 and x.stride(1) == 1 and y.stride(1) == 1:
        # This cases is needed for the shampoo optimizer.
        # All tensors are 2d (non-contiguous), have the same outer stride, and have an inner stride of 1
        # (so rows are contiguous)
        if x.stride(0) == y.stride(0):
            offset_delta = y.storage_offset() - x.storage_offset()
            if offset_delta < x.size(1):
                # definitely overlaps (row 0 of y overlaps with row 0 of x)
                # Example:
                #   base = torch.arange(32).reshape(4, 8)
                #   x = base.narrow(1, 0, 4)
                #     x: size=(4, 4), stride=(8, 1), offset=0
                #   y = base.narrow(1, 3, 4)
                #     y: size=(4, 4), stride=(8, 1), offset=3
                return False
            x_total_elems_covered = x.stride(0) * (x.size(0) - 1) + x.size(1)
            if x_total_elems_covered <= offset_delta:
                # definitely does not overlap (last byte of x is before start of y)
                # Example:
                #   x: size=(4, 4), stride=(8, 1), offset=0 (last byte is 27)
                #   y: size=(4, 4), stride=(8, 1), offset=28 (start byte is 28)
                return True
            # At this point, we want to check if the 0th row of y
            # overlaps with **some** row of x.
            # We can check this by shifting y backward by the shared stride, repeatedly,
            # until the first row of y is before the first row of x.
            # Then we can check if these rows overlap.
            # We can accomplish this by modding our offset by the stride.
            offset_delta_mod = offset_delta % x.stride(0)
            # Example:
            # 0 1 2 3
            # 9 10 11 12
            # 18 19 20 21
            # 27 28 29 30
            #   x: size=(4, 4), stride=(9, 1), offset=0
            #   y: size=(4, 4), stride=(9, 1), offset=22 (this would not overlap)
            #   y: size=(4, 4), stride=(9, 1), offset=23 (this would not overlap)
            #   y: size=(4, 4), stride=(9, 1), offset=24 (this would overlap)
            #   y: size=(4, 4), stride=(9, 1), offset=25 (this would overlap)
            # If the interval [modded_offset, modded_offset + x_size] falls entirely
            # without
            if offset_delta_mod + y.size(1) <= x.stride(0):
                return True
            else:
                return False
    return False


def _compute_overlapping_inputs(fwd_inputs, aliased_input_indices):
    actual_aliased_indices = set()
    for j in range(len(aliased_input_indices)):
        for i in range(j):
            i_ = aliased_input_indices[i]
            j_ = aliased_input_indices[j]
            if not _tensors_definitely_do_not_overlap(fwd_inputs[i_], fwd_inputs[j_]):
                actual_aliased_indices.add(i_)
                actual_aliased_indices.add(j_)
    return actual_aliased_indices


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
        aliased_input_indices_no_false_sharing = _compute_overlapping_inputs(
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


def _graph_input_names(gm):
    return [node.name for node in gm.graph.nodes if node.op == "placeholder"]


def _graph_output_names(gm):
    output_node = next(iter(reversed(gm.graph.nodes)))
    assert output_node.op == "output" and len(output_node.args) == 1
    return_args = output_node.args[0]
    return [getattr(return_arg, "name", None) for return_arg in return_args]


def create_graph_signature(
    fx_g: torch.fx.GraphModule,
    fw_metadata: ViewAndMutationMeta,
    in_spec: pytree.TreeSpec,
    out_spec: pytree.TreeSpec,
    *,
    user_args_flat: List[Tensor],
    params_and_buffers_flat: List[Tensor],
    param_names: List[str],
    buffer_names: List[str],
    trace_joint: bool,
    num_user_fw_outs: Optional[int],
    loss_index: Optional[int],
) -> GraphSignature:
    # Retrieve graph input names
    graph_input_names = _graph_input_names(fx_g)
    # Retrieve graph output names
    graph_output_names = _graph_output_names(fx_g)

    num_params_buffers = len(param_names) + len(buffer_names)
    # We have enough restrictions on the graph (no de-duping, synthetic bases, etc),
    # Such that # graph inps = # user inps + # params + # buffers
    num_user_args = len(graph_input_names) - num_params_buffers

    if trace_joint:
        assert num_user_fw_outs is not None
        num_fw_outs = num_user_fw_outs + fw_metadata.num_mutated_inp_runtime_indices
        backward_output_names = graph_output_names[num_fw_outs:]

        grad_index = itertools.count(0)
        gradients_to_parameters = {
            backward_output_names[next(grad_index)]: param_names[i]
            for i, param in enumerate(params_and_buffers_flat)
            if param.requires_grad
        }

        gradients_to_user_inputs = {
            backward_output_names[next(grad_index)]: graph_input_names[
                i + len(params_and_buffers_flat)
            ]
            for i, user_input in enumerate(user_args_flat)
            if user_input.requires_grad
        }

        assert len(gradients_to_parameters) + len(gradients_to_user_inputs) == len(
            backward_output_names
        )

        # Check that we have fully accounted for all graph outputs
        backward_signature = BackwardSignature(
            gradients_to_parameters,
            gradients_to_user_inputs,
            graph_output_names[loss_index],
        )
    else:
        backward_signature = None
        num_user_fw_outs = (
            len(graph_output_names) - fw_metadata.num_mutated_inp_runtime_indices
        )

    return GraphSignature.from_tracing_metadata(
        in_spec=in_spec,
        out_spec=out_spec,
        graph_input_names=graph_input_names,
        graph_output_names=graph_output_names,
        view_mutation_metadata=fw_metadata,
        named_parameters=param_names,
        named_buffers=buffer_names,
        num_user_inputs=num_user_args,
        num_user_outputs=num_user_fw_outs,
        loss_index=loss_index,
        backward_signature=backward_signature,
    )
