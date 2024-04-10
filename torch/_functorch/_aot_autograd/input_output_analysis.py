"""
This module is one of the analysis modules - it takes as input a function or graph
and some preexisting properties, and returns some data that is useful for deciding
how to further proceed with compilation or construct runtime wrappers.

In particular, the following analyses are provided:
1. Refine the view and mutation metadata collected previously - removing duplicate
   inputs or mapping views to their bases.
2. We also analyze the function signature for export graphs.
"""

import itertools
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.utils._pytree as pytree
from torch import Tensor
from torch._subclasses.functional_tensor import FunctionalTensor
from torch.fx.experimental.symbolic_shapes import is_concrete_int
from .collect_metadata_analysis import coerce_tangent
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
        # See Note [Tangents must be contiguous]
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

        mutation_inductor_storage_resize = all(
            m.input_info[x].mutation_inductor_storage_resize for x in outer_indices
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
            mutation_inductor_storage_resize=mutation_inductor_storage_resize,
            is_leaf=any_leaf,
            requires_grad=requires_grad,
            keep_input_mutations=m.keep_input_mutations,
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
    existing_output_infos = []
    for o in m.output_info:
        new_base_idx = (
            None
            if o.base_idx is None
            else (
                synthetic_base_info[o.base_idx]
                if isinstance(synthetic_base_info[o.base_idx], int)
                else synthetic_base_info[o.base_idx][0]  # type: ignore[index]
            )
        )
        # If base_idx is changed for OutputType.is_input, we need to update the output type to reflect the change
        new_output_type = (
            OutputType.alias_of_input
            if o.output_type == OutputType.is_input and o.base_idx != new_base_idx
            else o.output_type
        )
        existing_output_infos.append(
            OutputAliasInfo(
                output_type=new_output_type,
                raw_type=o.raw_type,
                dynamic_dims=o.dynamic_dims,
                # Map the input idx pre-synthetic-bases to the new idx post-synthetic-bases
                base_idx=new_base_idx,  # type: ignore[arg-type]
                requires_grad=o.requires_grad,
            )
        )

    inner_mutated_tangents = [
        # See Note [Tangents must be contiguous]
        coerce_tangent(x)
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


def _get_last_mem_address(x):
    out = x.storage_offset()
    for size, stride in zip(x.size(), x.stride()):
        out += (size - 1) * stride
    return out


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

    # Short-circuit: if last memory address of x is < start of y, then not overlapping.
    x_last = _get_last_mem_address(x)
    if x_last < y.storage_offset():
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


def compute_overlapping_inputs(fwd_inputs, aliased_input_indices):
    actual_aliased_indices = set()
    for j in range(len(aliased_input_indices)):
        for i in range(j):
            i_ = aliased_input_indices[i]
            j_ = aliased_input_indices[j]
            if not _tensors_definitely_do_not_overlap(fwd_inputs[i_], fwd_inputs[j_]):
                actual_aliased_indices.add(i_)
                actual_aliased_indices.add(j_)
    return actual_aliased_indices


def _graph_input_names(gm):
    return [node.name for node in gm.graph.find_nodes(op="placeholder")]


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
    num_tokens = len(fw_metadata.tokens)
    # We have enough restrictions on the graph (no de-duping, synthetic bases, etc),
    # Such that # graph inps = # user inps + # params + # buffers
    num_user_args = len(graph_input_names) - num_params_buffers - num_tokens

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
            len(graph_output_names)
            - fw_metadata.num_mutated_inp_runtime_indices
            - num_tokens
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
