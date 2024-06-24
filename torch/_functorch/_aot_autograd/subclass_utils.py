# mypy: allow-untyped-defs
"""
This file contains utilities for tracing through __torch_dispatch__ based tensor subclasses and modes.
AOTAutograd's responsibility is to trace through all pytorch capabilities that live in the pytorch dispatcher,
and this includes tensor subclasses that implement __torch_dispatch__.
"""

from typing import Any, List, Optional, Tuple, Union

import torch.utils._pytree as pytree

from torch import Tensor
from torch.utils._python_dispatch import is_traceable_wrapper_subclass

from .schemas import MutationType, SubclassCreationMeta, ViewAndMutationMeta
from .utils import strict_zip

zip = strict_zip


def requires_subclass_dispatch(args, fw_metadata: ViewAndMutationMeta) -> bool:
    args_flattened = pytree.arg_tree_leaves(*args)
    any_subclass_args = any(
        is_traceable_wrapper_subclass(x)
        for x in args_flattened
        if isinstance(x, Tensor)
    )
    from torch._functorch._aot_autograd.schemas import SubclassCreationMeta

    any_subclass_outputs = any(
        type(x) is SubclassCreationMeta for x in fw_metadata.subclass_fw_graph_out_meta
    )
    # This tells us whether or not we need to perform any unwrapping/wrapping of tensor subclasses at runtime.
    return any_subclass_args or any_subclass_outputs


# Given a flat list of arguments, some of which may be tensor subclasses,
# computes metadata about "how to reconstruct the current list of subclasses,
# if we were given their flattened dense tensors instead"
def create_subclass_meta(
    curr_args: Union[List[Any], Tuple[Any, ...]],
) -> List[Union[int, SubclassCreationMeta]]:
    idx = 0
    infos: List[Union[int, SubclassCreationMeta]] = []
    for a in curr_args:
        if isinstance(a, Tensor) and is_traceable_wrapper_subclass(a):
            attrs, meta = a.__tensor_flatten__()  # type: ignore[attr-defined]
            start_idx = idx
            cnt = len(attrs)
            curr_cnt = cnt
            infos.append(
                SubclassCreationMeta(
                    flat_tensor_start_idx=start_idx,
                    arg_count=curr_cnt,
                    original_subclass=a,
                    meta=meta,
                    inner_keys=attrs,
                    outer_size=a.shape,
                    outer_stride=a.stride(),
                )
            )
        else:
            infos.append(idx)
            cnt = 1
        idx += cnt
    return infos


# Output structure:
# - List[Tensor] if tracing an inference graph
# - Tuple[List[Tensor], List[Tensor]] if tracing a joint graph.
# This function effectively concats each inner list of subclass tensors
# into a (potentially longer) list of inner tensors.
#
# This function takes in a pytree of arguments and unwraps any tensor subclasses.
# Annoyingly, we can't use pytrees to perform the unwrapping, because unwrapping returns
# a list of tensors that we would then need to concat together.
# Instead, we specialize the logic for the inference vs. joint graph case.
# NOTE: this function is hot, since we unwrap tensor subclass inputs at runtime
def unwrap_tensor_subclasses(wrapped_args, *, is_joint_structure: bool):
    def concat_inner_tensors_from_subclasses(xs):
        xs_inner = []
        for x in xs:
            if isinstance(x, Tensor) and is_traceable_wrapper_subclass(x):
                attrs, _ = x.__tensor_flatten__()  # type: ignore[attr-defined]
                xs_inner += [getattr(x, attr) for attr in attrs]
            else:
                xs_inner += [x]
        return xs_inner

    if is_joint_structure:
        assert isinstance(wrapped_args, tuple) and len(wrapped_args) == 2
        assert isinstance(wrapped_args[0], (tuple, list)) and isinstance(
            wrapped_args[1], (tuple, list)
        )
        unwrapped_args_fw = concat_inner_tensors_from_subclasses(wrapped_args[0])
        unwrapped_args_tangents = concat_inner_tensors_from_subclasses(wrapped_args[1])
        unwrapped_args = (unwrapped_args_fw, unwrapped_args_tangents)
    else:
        assert isinstance(wrapped_args, (list, tuple))
        unwrapped_args_fw = concat_inner_tensors_from_subclasses(wrapped_args)
        unwrapped_args = unwrapped_args_fw
    return unwrapped_args


# Turns a flattened list of tensor arguments into (maybe) subclass tensors.
# This function is used both at trace time and runtime, so we have an is_runtime flag telling us which context we're in.
def wrap_tensor_subclasses(
    unwrapped_args: Union[Tuple[Any, ...], List[Any]],
    *,
    subclass_metas: List[Union[int, SubclassCreationMeta]],
    num_fw_outs_saved_for_bw: Optional[int] = None,
    is_runtime: bool = False,
) -> Tuple[Any, ...]:
    wrapped_args = []
    num_args_tallied = 0
    for subclass_meta in subclass_metas:
        if isinstance(subclass_meta, int):
            wrapped_args.append(unwrapped_args[subclass_meta])
            num_args_tallied += 1
        else:
            assert isinstance(subclass_meta, SubclassCreationMeta)
            wrapped_args.append(
                subclass_meta.creation_fn(unwrapped_args, is_runtime=is_runtime)
            )
            num_args_tallied += subclass_meta.arg_count

    # Note: [Partitioner handling for Subclasses, Part 2]
    # At the beginning of AOTAutograd, we collect metadata on the inputs and outputs of the user fw,
    # to figure out which inputs/outputs are subclasses, and how to reconstruct the subclasses after flattening them.
    #
    # When this function is called at runtime in the forward,
    # we have been passed a list of (flattened) dense-tensor fw-outs, and need to reconstruct any subclass fw outs.
    #
    # One reasonable question that you should ask: when should the dense_tensor -> subclass_tensor wrapping happen?
    # Answer: we do it **inside of our compiled autograd.Function**.
    # This seems like morally the right place: autograd happens above subclass desugaring,
    # so autograd should see actual tensor subclasses at runtime, and not flattened dense tensors.
    #
    # This causes a tricky interaction though: when we run the min-cut partitioner to divvy up the joint graph
    # into a forward and backward graph, we end up with some activations that show up as extra outputs
    # in the compiled forward graph, that are **not** user outputs.
    # These activations are not visible to the user, and so there's no need for us to wrap them back into subclasses.
    #
    # On top of that, when we first computed subclass metadata (in `run_functionalized_fw_and_collect_metadata`),
    # we computed subclass metadata on every forward output, but this did **not** include activations
    # created by the partitioner.
    # as a result, `unwrapped_args` here will correspond to (*unwrapped_user_fw_outs, *activations),
    # but `subclass_metas` will only correspond to subclass metatadata on `user_fw_outs`.
    # We then need to make sure that we return (*wrapped_user_fw_outs, *activations).
    if num_fw_outs_saved_for_bw is not None:
        assert len(unwrapped_args) == num_args_tallied + num_fw_outs_saved_for_bw, (
            f"Expected the number actual unwrapped-subclass outputs {len(unwrapped_args)} to equal "
            f"the number of args calculated from subclasses ({num_args_tallied}) plus the number of "
            f"additional activations saved for the backward pass ({num_fw_outs_saved_for_bw})"
        )
        activations = unwrapped_args[num_args_tallied:]
        if isinstance(wrapped_args, tuple) and isinstance(activations, tuple):
            return wrapped_args + activations
        return tuple(list(wrapped_args) + list(activations))
    else:
        assert len(unwrapped_args) == num_args_tallied
        return tuple(wrapped_args)


# Given a bunch of "dense" tensor arguments, this function (potentially) wraps them into tensor subclasses.
# This function carefully handles the inference vs. joint cases:
# - when is_joint_structure is True, args is (primals, tangents)
# - when is_joint_structure is False, args is [*primals]
def wrap_tensor_subclasses_maybe_joint(
    unwrapped_args, *, is_joint_structure: bool, meta: ViewAndMutationMeta
) -> Union[Tuple[Any, ...], List[Any]]:
    # Since this function is re-used for both inference and joint graphs,
    if is_joint_structure:
        assert isinstance(unwrapped_args, tuple) and len(unwrapped_args) == 2
        assert isinstance(unwrapped_args[0], (tuple, list)) and isinstance(
            unwrapped_args[1], (tuple, list)
        )
        primals, tangents = unwrapped_args[0], unwrapped_args[1]
        wrapped_primals = wrap_tensor_subclasses(
            primals, subclass_metas=meta.subclass_inp_meta
        )
        wrapped_tangents = wrap_tensor_subclasses(
            tangents, subclass_metas=meta.subclass_tangent_meta
        )
        return (wrapped_primals, wrapped_tangents)
    else:
        wrapped_args = wrap_tensor_subclasses(
            unwrapped_args, subclass_metas=meta.subclass_inp_meta
        )
        return wrapped_args


# TODO: UNUSED. delete?
def create_metadata_for_subclass(meta: ViewAndMutationMeta) -> ViewAndMutationMeta:
    # input infos
    input_info = []
    for inp, subclass_meta in zip(meta.input_info, meta.subclass_inp_meta):
        num_inps = 1 if isinstance(subclass_meta, int) else subclass_meta.arg_count
        for _ in range(num_inps):
            input_info.append(inp)

    # output infos
    output_info = []
    subclass_out_meta_user_outs_only = meta.subclass_fw_graph_out_meta[
        meta.num_mutated_inp_runtime_indices :
    ]
    if meta.num_intermediate_bases > 0:
        subclass_out_meta_user_outs_only = subclass_out_meta_user_outs_only[
            : -meta.num_intermediate_bases
        ]
    # sanity assert
    assert len(meta.output_info) == len(subclass_out_meta_user_outs_only)
    # Assume that the information on the output is shared by all of its inner tensors.
    for out, subclass_meta in zip(meta.output_info, subclass_out_meta_user_outs_only):
        num_outs = 1 if isinstance(subclass_meta, int) else subclass_meta.arg_count
        for _ in range(num_outs):
            output_info.append(out)

    # A bit hacky, but we don't actually care about all of the metadata here.
    # This metadata is used **underneath** both autograd and subclass de-sugaring,
    # So all we really care about is stuff like:
    # - num inputs/outputs (needed by the partitioner)
    # - input mutations (**not** used today, since we don't handle input mutations inside the subclass,
    #   although we should handle this eventually)
    #   TODO: add a test case to assert we error when this happens, instead of getting silent correctness
    num_intermediate_bases = None
    keep_input_mutations = meta.keep_input_mutations
    traced_tangents = None
    subclass_inp_meta = None
    subclass_fw_graph_out_meta = None
    subclass_tangent_meta = None

    metadata = ViewAndMutationMeta(
        input_info=input_info,  # type: ignore[arg-type]
        output_info=output_info,  # type: ignore[arg-type]
        num_intermediate_bases=num_intermediate_bases,  # type: ignore[arg-type]
        keep_input_mutations=keep_input_mutations,  # type: ignore[arg-type]
        traced_tangents=traced_tangents,  # type: ignore[arg-type]
        subclass_inp_meta=subclass_inp_meta,  # type: ignore[arg-type]
        subclass_fw_graph_out_meta=subclass_fw_graph_out_meta,  # type: ignore[arg-type]
        subclass_tangent_meta=subclass_tangent_meta,  # type: ignore[arg-type]
    )
    return metadata


def compute_inner_mutated_inp_indices_from_subclass_meta(
    fw_metadata: ViewAndMutationMeta,
    inner_metadata: ViewAndMutationMeta,
) -> List[int]:
    # Note: [Recomputing subclass mutation handling]
    #
    # Generally, if a subclass requires grad, its components will not require grad.
    # But for the purposes of tracking returned tensors, we should treat those component
    # tensors as if they require grad.
    #
    # For example, if the subclass tensor requires grad and will be mutated in a way that
    # requires us to handle the mutation outside of the graph, we need to return it
    # from the forward graph. The inner_meta data won't consider the component tensors
    # as if they need to be returned, because they don't require grad; but really, we
    # should handle those tensors the same way we handle the subclass tensor itself; i.e.
    # if we'd include the subclass tensor as part of the outputs, then we should also
    # include the component tensors.
    #
    # To do this, we patch num_mutated_inp_runtime_indices below by expanding the inputs
    # from the outer subclass tensors and propagating

    updated_input_info = []
    inner_idx = 0
    if not fw_metadata.subclass_inp_meta:
        # Sometimes we don't have subclass info, e.g. synthetic_base codepaths
        return inner_metadata.mutated_inp_runtime_indices
    assert len(fw_metadata.subclass_inp_meta) == len(fw_metadata.input_info)
    for outer_idx, inp_meta in enumerate(fw_metadata.subclass_inp_meta):
        if isinstance(inp_meta, int):
            assert outer_idx < len(fw_metadata.input_info)
            if inner_metadata is not None:
                assert inner_idx < len(inner_metadata.input_info)
                assert (
                    inner_metadata.input_info[inner_idx]
                    == fw_metadata.input_info[outer_idx]
                )
            updated_input_info.append(fw_metadata.input_info[outer_idx])
            inner_idx += 1
        else:
            for _ in range(inp_meta.arg_count):
                updated_input_info.append(fw_metadata.input_info[outer_idx])
                inner_idx += 1
    if inner_metadata is not None:
        assert len(inner_metadata.input_info) == len(updated_input_info)

    return [
        i
        for i, inp in enumerate(updated_input_info)
        if inp.mutation_type == MutationType.MUTATED_OUT_GRAPH
    ]
