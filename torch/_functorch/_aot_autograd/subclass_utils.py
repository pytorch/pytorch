# mypy: allow-untyped-defs
"""
This file contains utilities for tracing through __torch_dispatch__ based tensor subclasses and modes.
AOTAutograd's responsibility is to trace through all pytorch capabilities that live in the pytorch dispatcher,
and this includes tensor subclasses that implement __torch_dispatch__.
"""

import typing
from typing import Any, Iterable, List, Optional, Tuple, Union

import torch.utils._pytree as pytree
from torch import SymInt, Tensor
from torch._subclasses.fake_tensor import get_plain_tensors
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


def create_subclass_metadata(a: object, start_idx: int, count_symints: bool):
    if not is_traceable_wrapper_subclass(a):
        return None, start_idx + 1

    inner_keys, metadata = a.__tensor_flatten__()
    new_start_idx = start_idx
    attrs = {}
    for key in inner_keys:
        new_subclass_meta, new_start_idx = create_subclass_metadata(
            getattr(a, key),
            new_start_idx,
            count_symints=False,  # only count symints in the outer class
        )
        attrs[key] = new_subclass_meta

    # It *must* be because is_traceable_wrapper_subclass() - but mypy is not smart.
    assert isinstance(a, Tensor)

    new_start_idx = (
        new_start_idx
        + count_symints * len(filter_symints(a.size()))
        + count_symints * len(filter_symints(a.stride()))
    )

    return (
        SubclassCreationMeta(
            flat_tensor_start_idx=start_idx,
            arg_count=new_start_idx - start_idx,
            included_subclass_symints=count_symints,
            attrs=attrs,
            meta=metadata,
            outer_size=a.size(),  # type: ignore[attr-defined, arg-type]
            outer_stride=a.stride(),  # type: ignore[arg-type]
            original_subclass=a,
        ),
        new_start_idx,
    )


# Given a real tensor subclass, returns a nested list of Plain tensor types
def get_types_for_subclass(tensor_subclass):
    if not is_traceable_wrapper_subclass(tensor_subclass):
        return ["Tensor"]
    inner_keys, _ = tensor_subclass.__tensor_flatten__()
    result = []
    for key in inner_keys:
        inner_tensor = getattr(tensor_subclass, key)
        result.extend(get_types_for_subclass(inner_tensor))
    return result


# Given a flat list of arguments, some of which may be tensor subclasses,
# computes metadata about "how to reconstruct the current list of subclasses,
# if we were given their flattened dense tensors instead"
def create_subclass_meta(
    curr_args: Union[List[Any], Tuple[Any, ...]], *, count_symints: bool = True
) -> List[Union[int, SubclassCreationMeta]]:
    idx = 0
    infos: List[Union[int, SubclassCreationMeta]] = []
    for a in curr_args:
        if is_traceable_wrapper_subclass(a):
            assert isinstance(a, Tensor)
            start_idx = idx
            subclass_meta, _ = create_subclass_metadata(
                a, start_idx, count_symints=count_symints
            )
            infos.append(subclass_meta)
            cnt = subclass_meta.arg_count
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


def filter_symints(lst: Iterable[Union[int, SymInt]]) -> List[SymInt]:
    return [s for s in lst if isinstance(s, SymInt) and not s.node.is_nested_int()]


def compute_symint_placeholders(lst: Iterable[Union[None, int, SymInt]]) -> List[bool]:
    # Non-nested symints are replaced with None values in `make_runtime_safe()`
    return [s is None for s in lst]


# The reason for "append_symints"
#
# * At compile time: we append extra symint args when unwrapping primals
# (but not tangents, because they should always share symints with primals).
# We also append extra symints when unwrapping the subclass outputs of the
# traced function, so we can return them as extra outputs
#
# * At runtime: we similarly append subclass sizes when we unwrap subclass
# primals (but not tangents) on entry to the forward.
#
# subclass_metas is needed at runtime to compute which indices are symints in
# the outer_size/outer_stride
def unwrap_tensor_subclasses(
    wrapped_args: List[Union[Tensor, int]],
    *,
    is_runtime: bool,
    append_symints: bool,
    subclass_metas: Optional[List[Union[int, SubclassCreationMeta]]] = None,
):
    xs_inner: List[Union[int, Tensor, SymInt]] = []

    for idx, x in enumerate(wrapped_args):
        if not is_traceable_wrapper_subclass(x):
            xs_inner.append(x)
            continue

        size = x.size()
        stride = x.stride()
        xs_inner.extend(get_plain_tensors(typing.cast(Tensor, x)))
        if append_symints:
            if is_runtime:
                assert subclass_metas is not None
                meta = subclass_metas[idx]
                assert isinstance(meta, SubclassCreationMeta)

                # outer_size
                symint_placeholders = compute_symint_placeholders(meta.outer_size)
                assert len(size) == len(symint_placeholders)
                xs_inner.extend(
                    [
                        r
                        for (r, is_symint) in zip(size, symint_placeholders)
                        if is_symint
                    ]
                )

                # outer_stride
                symint_placeholders = compute_symint_placeholders(meta.outer_stride)
                assert len(stride) == len(symint_placeholders)
                xs_inner.extend(
                    [
                        r
                        for (r, is_symint) in zip(stride, symint_placeholders)
                        if is_symint
                    ]
                )
            else:
                xs_inner.extend(filter_symints(size))
                xs_inner.extend(filter_symints(stride))
    return xs_inner


def remap_unwrapped_subclass_arg_indices(wrapped_args, static_input_indices):
    static_input_indices = set(static_input_indices)
    new_ind = 0
    remapped_static_indices = []
    for i, arg in enumerate(wrapped_args):
        num_indices = 1
        if is_traceable_wrapper_subclass(arg):
            num_indices = (
                len(get_plain_tensors(typing.cast(Tensor, arg)))
                + len(filter_symints(arg.size()))
                + len(filter_symints(arg.stride()))
            )

        for _ in range(num_indices):
            if i in static_input_indices:
                remapped_static_indices.append(new_ind)

            new_ind += 1

    return remapped_static_indices


# Turns a flattened list of tensor arguments into (maybe) subclass tensors.
# This function is used both at trace time and runtime, so we have an is_runtime flag telling us which context we're in.
def wrap_tensor_subclasses(
    unwrapped_args: Union[Tuple[Any, ...], List[Any]],
    *,
    subclass_metas: List[Union[int, SubclassCreationMeta]],
    num_fw_outs_saved_for_bw: Optional[int] = None,
    included_subclass_symints: bool = False,
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
            assert subclass_meta.included_subclass_symints == included_subclass_symints
            wrapped_args.append(
                subclass_meta.creation_fn(
                    unwrapped_args, is_runtime=is_runtime, is_nested=False
                )
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
        assert (
            len(unwrapped_args) == num_args_tallied
        ), f"Expected {len(unwrapped_args)} == {num_args_tallied}"
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
            primals,
            subclass_metas=meta.subclass_inp_meta,
            included_subclass_symints=True,
        )
        wrapped_tangents = wrap_tensor_subclasses(
            tangents,
            subclass_metas=meta.subclass_tangent_meta,
            included_subclass_symints=False,
        )
        return (wrapped_primals, wrapped_tangents)
    else:
        wrapped_args = wrap_tensor_subclasses(
            unwrapped_args,
            subclass_metas=meta.subclass_inp_meta,
            included_subclass_symints=True,
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
            assert inp_meta.original_subclass is not None
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
