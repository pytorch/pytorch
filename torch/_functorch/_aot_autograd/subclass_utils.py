# mypy: allow-untyped-defs
"""
This file contains utilities for tracing through __torch_dispatch__ based tensor subclasses and modes.
AOTAutograd's responsibility is to trace through all pytorch capabilities that live in the pytorch dispatcher,
and this includes tensor subclasses that implement __torch_dispatch__.
"""

import collections
import typing
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type, TypeVar, Union

import torch
import torch.utils._pytree as pytree
from torch import SymInt, Tensor
from torch._subclasses.fake_tensor import get_plain_tensors
from torch.utils._python_dispatch import is_traceable_wrapper_subclass

from .schemas import (
    MutationType,
    PlainTensorMeta,
    SubclassCreationMeta,
    ViewAndMutationMeta,
)
from .utils import strict_zip


zip = strict_zip

T = TypeVar("T", bound=torch.Tensor)


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


suggest_memory_format = torch._prims_common.suggest_memory_format


def maybe_suggest_memory_format(
    t, with_memory_format: bool
) -> Optional[torch.memory_format]:
    if not with_memory_format:
        return None

    return suggest_memory_format(t)


def get_subclass_typing_container(
    tensor_subclass: torch.Tensor,
) -> Dict[Type[torch.Tensor], List[Type[torch.Tensor]]]:
    """
    Given a subclass, returns a recursive dictionary mapping each
    inner tensors to its' subclass types.
    """

    def _get_types_for_subclass(tensor_subclass: torch.Tensor) -> None:
        if not is_traceable_wrapper_subclass(tensor_subclass):
            return
        tracker[type(tensor_subclass)].append(tensor_subclass)
        inner_keys, _ = tensor_subclass.__tensor_flatten__()
        for key in inner_keys:
            inner_tensor = getattr(tensor_subclass, key)
            _get_types_for_subclass(inner_tensor)

    tracker: Dict[Any, List[Any]] = collections.defaultdict(list)
    _get_types_for_subclass(tensor_subclass)
    return tracker


def create_subclass_metadata(
    a: Any, start_idx: int, count_symints: bool, with_memory_format: bool = False
):
    if not is_traceable_wrapper_subclass(a):
        idx = start_idx + 1
        return (
            PlainTensorMeta(
                idx,
                memory_format=maybe_suggest_memory_format(a, with_memory_format),
            ),
            idx,
        )

    inner_keys, metadata = a.__tensor_flatten__()
    new_start_idx = start_idx
    attrs = {}

    for key in inner_keys:
        new_subclass_meta, new_start_idx = create_subclass_metadata(
            getattr(a, key),
            new_start_idx,
            count_symints=count_symints,
            with_memory_format=with_memory_format,
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
            memory_format=maybe_suggest_memory_format(a, with_memory_format),
        ),
        new_start_idx,
    )


# Given a flat list of arguments, some of which may be tensor subclasses,
# computes metadata about "how to reconstruct the current list of subclasses,
# if we were given their flattened dense tensors instead"
def create_subclass_meta(
    curr_args: Union[List[Any], Tuple[Any, ...]],
    *,
    count_symints: bool = True,
    with_memory_format: bool = False,
) -> List[Union[PlainTensorMeta, SubclassCreationMeta]]:
    idx = 0
    infos: List[Union[PlainTensorMeta, SubclassCreationMeta]] = []
    for a in curr_args:
        if is_traceable_wrapper_subclass(a):
            assert isinstance(a, Tensor)
            start_idx = idx
            subclass_meta, _ = create_subclass_metadata(
                a,
                start_idx,
                count_symints=count_symints,
                with_memory_format=with_memory_format,
            )
            infos.append(subclass_meta)
            cnt = subclass_meta.arg_count
        else:
            infos.append(
                PlainTensorMeta(
                    idx,
                    memory_format=maybe_suggest_memory_format(a, with_memory_format),
                )
            )
            cnt = 1
        idx += cnt
    return infos


def filter_symints(lst: Iterable[Union[int, SymInt]]):
    # Capture all SymInts from the iterable.
    def symint_check(s: Union[int, SymInt]) -> bool:
        return isinstance(s, SymInt) and not s.node.is_nested_int()

    return [s for s in lst if symint_check(s)]


def compute_symint_placeholders(lst: Iterable[Union[None, int, SymInt]]) -> List[bool]:
    # Non-nested symints are replaced with None in `make_runtime_safe()`
    return [s is None for s in lst]


# This function takes in a pytree of arguments and unwraps any tensor
# subclasses.
#
# NOTE: The reason for "append_symints":
#
# * At compile time: we append extra symint args when unwrapping primals
# (but not tangents, because they should always share symints with primals).
# We also append extra symints when unwrapping the subclass outputs of the
# traced function, so we can return them as extra outputs
#
# * At runtime: we similarly append subclass sizes when we unwrap subclass
# primals (but not tangents) on entry to the forward. See the runtime version of
# this function below.
def unwrap_tensor_subclasses(
    wrapped_args: List[Union[Tensor, int]],
    *,
    append_symints: bool,
):
    def flatten_subclass(t: Union[Tensor, int], *, out=None):
        # unwrap a subclass into plain tensors and their size/stride if "append_symint"
        # is True
        if not is_traceable_wrapper_subclass(t):
            out.append(t)
            return

        attrs, _ = t.__tensor_flatten__()

        for attr in attrs:
            inner_tensor = getattr(t, attr)
            flatten_subclass(inner_tensor, out=out)

        if append_symints:
            out.extend(filter_symints(t.size()))
            out.extend(filter_symints(t.stride()))

    xs_inner: List[Union[int, Tensor, SymInt]] = []

    for x in wrapped_args:
        flatten_subclass(typing.cast(Tensor, x), out=xs_inner)

    return xs_inner


# subclass_metas is needed at runtime to compute which indices are symints in
# the outer_size/outer_stride
def runtime_unwrap_tensor_subclasses(
    wrapped_args: List[Union[Tensor, int]],
    *,
    append_symints: bool,
    subclass_metas: Optional[List[Union[PlainTensorMeta, SubclassCreationMeta]]] = None,
):
    def flatten_subclass(x: Tensor, meta: Optional[SubclassCreationMeta], *, out):
        if not is_traceable_wrapper_subclass(x):
            out.append(x)
            return out

        assert isinstance(x, Tensor)

        attrs, _ = x.__tensor_flatten__()

        for attr in attrs:
            inner_tensor = getattr(x, attr)
            inner_meta = meta.attrs.get(attr)
            flatten_subclass(inner_tensor, inner_meta, out=out)

        if append_symints:
            assert isinstance(meta, SubclassCreationMeta)
            # outer_size
            size = x.size()
            symint_placeholders = compute_symint_placeholders(meta.outer_size)
            assert len(size) == len(symint_placeholders)
            out.extend(
                [r for (r, is_symint) in zip(size, symint_placeholders) if is_symint]
            )

            # outer_stride
            stride = x.stride()
            symint_placeholders = compute_symint_placeholders(meta.outer_stride)
            assert len(stride) == len(symint_placeholders)
            out.extend(
                [r for (r, is_symint) in zip(stride, symint_placeholders) if is_symint]
            )
        return out

    xs_inner: List[Union[int, Tensor, SymInt]] = []

    if append_symints:
        assert subclass_metas is not None

    for idx, x in enumerate(wrapped_args):
        if not is_traceable_wrapper_subclass(x):
            xs_inner.append(x)
            continue

        if subclass_metas is None:
            get_plain_tensors(typing.cast(Tensor, x), out=xs_inner)
        else:
            meta = subclass_metas[idx]
            assert isinstance(meta, SubclassCreationMeta)
            flatten_subclass(typing.cast(Tensor, x), meta, out=xs_inner)

    return xs_inner


def unwrap_tensor_subclasses_with_indices_to_original(wrapped_args):
    ret_unwrapped = []
    ret_indices_to_original = []
    for i, a in enumerate(wrapped_args):
        a_unwrapped = unwrap_tensor_subclasses([a], append_symints=False)
        ret_unwrapped.extend(a_unwrapped)
        n = len(a_unwrapped)
        ret_indices_to_original.extend([i] * n)

    return ret_unwrapped, ret_indices_to_original


def remap_unwrapped_subclass_arg_indices(wrapped_args, static_input_indices):
    static_input_indices = set(static_input_indices)
    new_ind = 0
    remapped_static_indices = []
    for i, arg in enumerate(wrapped_args):
        num_indices = 1
        if is_traceable_wrapper_subclass(arg):
            num_indices = (
                len(get_plain_tensors(typing.cast(Tensor, arg), out=[]))
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
    subclass_metas: List[Union[PlainTensorMeta, SubclassCreationMeta]],
    num_fw_outs_saved_for_bw: Optional[int] = None,
    included_subclass_symints: bool = False,
    is_runtime: bool = False,
) -> Tuple[Any, ...]:
    wrapped_args = []
    num_args_tallied = 0
    for subclass_meta in subclass_metas:
        if isinstance(subclass_meta, PlainTensorMeta):
            wrapped_args.append(unwrapped_args[subclass_meta.unwrapped_idx])
            num_args_tallied += 1
        else:
            assert isinstance(subclass_meta, SubclassCreationMeta)
            assert subclass_meta.included_subclass_symints == included_subclass_symints
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
        if isinstance(inp_meta, PlainTensorMeta):
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
