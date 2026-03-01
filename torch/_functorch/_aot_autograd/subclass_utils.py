"""
This file contains utilities for tracing through __torch_dispatch__ based tensor subclasses and modes.
AOTAutograd's responsibility is to trace through all pytorch capabilities that live in the pytorch dispatcher,
and this includes tensor subclasses that implement __torch_dispatch__.
"""

import collections
import typing
from collections.abc import Callable, Iterable, Sequence
from typing import Any, TYPE_CHECKING, TypeGuard, TypeVar

import torch
import torch.utils._pytree as pytree
from torch import SymInt, Tensor
from torch._subclasses.fake_tensor import get_plain_tensors
from torch.types import IntLikeType
from torch.utils._python_dispatch import is_traceable_wrapper_subclass

from .descriptors import (
    AOTInput,
    AOTOutput,
    DummyAOTInput,
    SubclassGetAttrAOTInput,
    SubclassGetAttrAOTOutput,
    SubclassSizeAOTInput,
    SubclassSizeAOTOutput,
    SubclassStrideAOTInput,
    SubclassStrideAOTOutput,
)
from .schemas import (
    FakifiedFlatArgs,
    FxValue,
    MutationType,
    PlainTensorMeta,
    SubclassCreationMeta,
    ViewAndMutationMeta,
)
from .utils import strict_zip


if TYPE_CHECKING:
    from torch._library.opaque_object import OpaqueType


zip = strict_zip

T = TypeVar("T", bound=torch.Tensor)


def requires_subclass_dispatch(
    args: FakifiedFlatArgs, fw_metadata: ViewAndMutationMeta
) -> bool:
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
    return bool(any_subclass_args or any_subclass_outputs)


from .schemas import MemoryFormatMeta


def maybe_suggest_memory_format(
    t: Tensor, with_memory_format: bool
) -> MemoryFormatMeta | None:
    if not with_memory_format:
        return None

    return MemoryFormatMeta.from_tensor(t)


def get_subclass_typing_container(
    tensor_subclass: torch.Tensor,
) -> dict[type[torch.Tensor], list[type[torch.Tensor]]]:
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

    tracker: dict[Any, list[Any]] = collections.defaultdict(list)
    _get_types_for_subclass(tensor_subclass)
    return tracker


def create_subclass_metadata(
    a: Any, start_idx: int, count_symints: bool, with_memory_format: bool = False
) -> tuple[Any, int]:
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
    if not isinstance(a, Tensor):
        raise AssertionError(f"expected Tensor, got {type(a)}")

    new_start_idx = (
        new_start_idx
        + count_symints * len(enumerate_filter_symints(a.size()))
        + count_symints * len(enumerate_filter_symints(a.stride()))
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
    curr_args: list[Any] | tuple[Any, ...],
    *,
    count_symints: bool = True,
    with_memory_format: bool = False,
) -> list[PlainTensorMeta | SubclassCreationMeta]:
    idx = 0
    infos: list[PlainTensorMeta | SubclassCreationMeta] = []
    for a in curr_args:
        if is_traceable_wrapper_subclass(a):
            if not isinstance(a, Tensor):
                raise AssertionError(
                    f"expected Tensor for traceable wrapper subclass, got {type(a)}"
                )
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


def enumerate_filter_symints(lst: Iterable[IntLikeType]) -> list[tuple[int, SymInt]]:
    # Capture all SymInts from the iterable.
    def symint_check(s: IntLikeType) -> TypeGuard[SymInt]:
        return isinstance(s, SymInt) and not s.node.is_nested_int()

    return [(i, s) for i, s in enumerate(lst) if symint_check(s)]


def compute_symint_placeholders(lst: Iterable[None | int | SymInt]) -> list[bool]:
    # Non-nested symints are replaced with None in `make_runtime_safe()`
    return [s is None for s in lst]


# Intended to make it easier to define function that is
# either (AOTInput -> AOTInput) or (AOTOutput -> AOTOutput)
# but not the other combos
AOTDescriptor = TypeVar("AOTDescriptor", AOTInput, AOTOutput)


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
    wrapped_args: list[FxValue],
    wrapped_args_descs: list[AOTDescriptor],
    *,
    append_symints: bool,
) -> tuple[list[FxValue], list[AOTDescriptor]]:
    def flatten_subclass(
        t: FxValue,
        desc: AOTDescriptor,
        *,
        out: tuple[list[FxValue], list[AOTDescriptor]],
    ) -> None:
        # unwrap a subclass into plain tensors and their size/stride if "append_symint"
        # is True
        if not is_traceable_wrapper_subclass(t):
            out[0].append(t)
            out[1].append(desc)
            return

        attrs, _ = t.__tensor_flatten__()

        for attr in attrs:
            inner_tensor = getattr(t, attr)
            n_desc: Any = (
                SubclassGetAttrAOTInput(desc, attr)
                if isinstance(desc, AOTInput)
                # pyrefly: ignore [bad-argument-type]
                else SubclassGetAttrAOTOutput(desc, attr)
            )
            flatten_subclass(inner_tensor, n_desc, out=out)

        if append_symints:
            sizes = enumerate_filter_symints(t.size())
            strides = enumerate_filter_symints(t.stride())
            out[0].extend(s for _, s in sizes)
            out[0].extend(s for _, s in strides)
            if isinstance(desc, AOTInput):
                out[1].extend(SubclassSizeAOTInput(desc, i) for i, _ in sizes)  # type: ignore[misc]
                out[1].extend(SubclassStrideAOTInput(desc, i) for i, _ in strides)  # type: ignore[misc]
            else:
                out[1].extend(SubclassSizeAOTOutput(desc, i) for i, _ in sizes)  # type: ignore[misc]
                out[1].extend(SubclassStrideAOTOutput(desc, i) for i, _ in strides)  # type: ignore[misc]

    xs_inner: list[FxValue] = []
    descs_inner: list[AOTDescriptor] = []

    for x, desc in zip(wrapped_args, wrapped_args_descs):
        # pyrefly: ignore [bad-argument-type]
        flatten_subclass(typing.cast(Tensor, x), desc, out=(xs_inner, descs_inner))

    return xs_inner, descs_inner


# subclass_metas is needed at runtime to compute which indices are symints in
# the outer_size/outer_stride
def runtime_unwrap_tensor_subclasses(
    wrapped_args: list[Tensor | int],
    *,
    append_symints: bool,
    subclass_metas: list[PlainTensorMeta | SubclassCreationMeta] | None = None,
) -> list[Any]:
    def flatten_subclass(
        x: Tensor, meta: SubclassCreationMeta | None, *, out: list[Any]
    ) -> list[Any]:
        if not is_traceable_wrapper_subclass(x):
            out.append(x)
            return out

        if not isinstance(x, Tensor):
            raise AssertionError(f"expected Tensor, got {type(x)}")

        attrs, _ = x.__tensor_flatten__()

        for attr in attrs:
            inner_tensor = getattr(x, attr)
            # pyrefly: ignore [missing-attribute]
            inner_meta = meta.attrs.get(attr)
            flatten_subclass(inner_tensor, inner_meta, out=out)

        if append_symints:
            if not isinstance(meta, SubclassCreationMeta):
                raise AssertionError(f"expected SubclassCreationMeta, got {type(meta)}")
            # outer_size
            size = x.size()
            symint_placeholders = compute_symint_placeholders(meta.outer_size)
            if len(size) != len(symint_placeholders):
                raise AssertionError(
                    f"size length mismatch: {len(size)} != {len(symint_placeholders)}"
                )
            out.extend(
                [r for (r, is_symint) in zip(size, symint_placeholders) if is_symint]
            )

            # outer_stride
            stride = x.stride()
            symint_placeholders = compute_symint_placeholders(meta.outer_stride)
            if len(stride) != len(symint_placeholders):
                raise AssertionError(
                    f"stride length mismatch: {len(stride)} != {len(symint_placeholders)}"
                )
            out.extend(
                [r for (r, is_symint) in zip(stride, symint_placeholders) if is_symint]
            )
        return out

    xs_inner: list[int | Tensor | SymInt | OpaqueType] = []

    if append_symints:
        if subclass_metas is None:
            raise AssertionError(
                "subclass_metas must not be None when append_symints is True"
            )

    for idx, x in enumerate(wrapped_args):
        if not is_traceable_wrapper_subclass(x):
            xs_inner.append(x)
            continue

        if subclass_metas is None:
            get_plain_tensors(typing.cast(Tensor, x), out=xs_inner)
        else:
            meta = subclass_metas[idx]
            if not isinstance(meta, SubclassCreationMeta):
                raise AssertionError(f"expected SubclassCreationMeta, got {type(meta)}")
            flatten_subclass(typing.cast(Tensor, x), meta, out=xs_inner)

    return xs_inner


def unwrap_tensor_subclasses_with_indices_to_original(
    wrapped_args: list[Any],
) -> tuple[list[Any], list[int]]:
    ret_unwrapped = []
    ret_indices_to_original = []
    for i, a in enumerate(wrapped_args):
        a_unwrapped, _ = unwrap_tensor_subclasses(
            [a], [DummyAOTInput(9999)], append_symints=False
        )
        ret_unwrapped.extend(a_unwrapped)
        n = len(a_unwrapped)
        ret_indices_to_original.extend([i] * n)

    return ret_unwrapped, ret_indices_to_original


def remap_unwrapped_subclass_arg_indices(
    wrapped_args: list[Any], static_input_indices: list[int]
) -> list[int]:
    static_input_indices_set = set(static_input_indices)
    new_ind = 0
    remapped_static_indices = []
    for i, arg in enumerate(wrapped_args):
        num_indices = 1
        if is_traceable_wrapper_subclass(arg):
            num_indices = (
                len(get_plain_tensors(typing.cast(Tensor, arg), out=[]))
                + len(enumerate_filter_symints(arg.size()))
                + len(enumerate_filter_symints(arg.stride()))
            )

        for _ in range(num_indices):
            if i in static_input_indices_set:
                remapped_static_indices.append(new_ind)

            new_ind += 1

    return remapped_static_indices


# Turns a flattened list of tensor arguments into (maybe) subclass tensors.
# This function is used both at trace time and runtime, so we have an is_runtime flag telling us which context we're in.
def wrap_tensor_subclasses(
    unwrapped_args: Sequence[Any],
    *,
    subclass_metas: list[PlainTensorMeta | SubclassCreationMeta],
    num_fw_outs_saved_for_bw: int | None = None,
    included_subclass_symints: bool = False,
    is_runtime: bool = False,
    make_subclass_override: Callable[..., Any] | None = None,
) -> tuple[Any, ...]:
    # pyrefly: ignore [implicit-any]
    wrapped_args = []
    num_args_tallied = 0
    for subclass_meta in subclass_metas:
        if isinstance(subclass_meta, PlainTensorMeta):
            wrapped_args.append(unwrapped_args[subclass_meta.unwrapped_idx])
            num_args_tallied += 1
        else:
            if not isinstance(subclass_meta, SubclassCreationMeta):
                raise AssertionError(
                    f"expected SubclassCreationMeta, got {type(subclass_meta)}"
                )
            if subclass_meta.included_subclass_symints != included_subclass_symints:
                raise AssertionError(
                    f"included_subclass_symints mismatch: {subclass_meta.included_subclass_symints} != {included_subclass_symints}"
                )

            if make_subclass_override:
                wrapped_args.append(
                    make_subclass_override(subclass_meta, is_runtime, unwrapped_args)
                )
            else:
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
    # but `subclass_metas` will only correspond to subclass metadata on `user_fw_outs`.
    # We then need to make sure that we return (*wrapped_user_fw_outs, *activations).
    if num_fw_outs_saved_for_bw is not None:
        if len(unwrapped_args) != num_args_tallied + num_fw_outs_saved_for_bw:
            raise AssertionError(
                f"Expected the number actual unwrapped-subclass outputs {len(unwrapped_args)} to equal "
                f"the number of args calculated from subclasses ({num_args_tallied}) plus the number of "
                f"additional activations saved for the backward pass ({num_fw_outs_saved_for_bw})"
            )
        activations = unwrapped_args[num_args_tallied:]
        if isinstance(wrapped_args, tuple) and isinstance(activations, tuple):
            return wrapped_args + activations
        return tuple(list(wrapped_args) + list(activations))
    else:
        if len(unwrapped_args) != num_args_tallied:
            raise AssertionError(
                f"Expected {len(unwrapped_args)} == {num_args_tallied}"
            )
        return tuple(wrapped_args)


# Given a bunch of "dense" tensor arguments, this function (potentially) wraps them into tensor subclasses.
# This function carefully handles the inference vs. joint cases:
# - when is_joint_structure is True, args is (primals, tangents)
# - when is_joint_structure is False, args is [*primals]
def wrap_tensor_subclasses_maybe_joint(
    unwrapped_args: Sequence[Any],
    *,
    is_joint_structure: bool,
    meta: ViewAndMutationMeta,
) -> tuple[Any, ...]:
    # Since this function is reused for both inference and joint graphs,
    if is_joint_structure:
        if not (isinstance(unwrapped_args, tuple) and len(unwrapped_args) == 2):
            unwrapped_len = (
                len(unwrapped_args)
                if isinstance(unwrapped_args, (tuple, list))
                else "N/A"
            )
            raise AssertionError(
                f"expected tuple of length 2 for joint structure, "
                f"got {type(unwrapped_args)} with length {unwrapped_len}"
            )
        if not (
            isinstance(unwrapped_args[0], (tuple, list))
            and isinstance(unwrapped_args[1], (tuple, list))
        ):
            raise AssertionError(
                f"expected primals and tangents to be tuple or list, got {type(unwrapped_args[0])} and {type(unwrapped_args[1])}"
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
) -> list[int]:
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
    if len(fw_metadata.subclass_inp_meta) != len(fw_metadata.input_info):
        raise AssertionError(
            f"subclass_inp_meta length ({len(fw_metadata.subclass_inp_meta)}) != input_info length ({len(fw_metadata.input_info)})"
        )
    for outer_idx, inp_meta in enumerate(fw_metadata.subclass_inp_meta):
        if isinstance(inp_meta, PlainTensorMeta):
            if outer_idx >= len(fw_metadata.input_info):
                raise AssertionError(
                    f"outer_idx ({outer_idx}) >= len(fw_metadata.input_info) ({len(fw_metadata.input_info)})"
                )
            if inner_metadata is not None:
                if inner_idx >= len(inner_metadata.input_info):
                    raise AssertionError(
                        f"inner_idx ({inner_idx}) >= len(inner_metadata.input_info) ({len(inner_metadata.input_info)})"
                    )
                if (
                    inner_metadata.input_info[inner_idx]
                    != fw_metadata.input_info[outer_idx]
                ):
                    raise AssertionError(
                        f"input_info mismatch at inner_idx={inner_idx}, outer_idx={outer_idx}: "
                        f"{inner_metadata.input_info[inner_idx]} != {fw_metadata.input_info[outer_idx]}"
                    )
            updated_input_info.append(fw_metadata.input_info[outer_idx])
            inner_idx += 1
        else:
            if inp_meta.original_subclass is None:
                raise AssertionError(
                    "inp_meta.original_subclass must not be None for SubclassCreationMeta"
                )
            for _ in range(inp_meta.arg_count):
                updated_input_info.append(fw_metadata.input_info[outer_idx])
                inner_idx += 1
    if inner_metadata is not None:
        if len(inner_metadata.input_info) != len(updated_input_info):
            raise AssertionError(
                f"inner_metadata.input_info length ({len(inner_metadata.input_info)}) "
                f"!= updated_input_info length ({len(updated_input_info)})"
            )

    return [
        i
        for i, inp in enumerate(updated_input_info)
        if inp.mutation_type == MutationType.MUTATED_OUT_GRAPH
    ]
