from __future__ import annotations


"""
This module is one of the analysis modules - it takes as input a function or graph
and some preexisting properties, and returns some data that is useful for deciding
how to further proceed with compilation or construct runtime wrappers.

In particular, the analysis here constructs view and mutation metadata from running
a functionalized version of the graph under compilation.
"""

import collections
import contextlib
import logging
import weakref
from typing import Any, NamedTuple, TYPE_CHECKING

import torch
import torch.utils._pytree as pytree
from torch import Tensor
from torch._guards import detect_fake_mode
from torch._library.opaque_object import is_custom_class
from torch._logging import getArtifactLogger
from torch._subclasses.functional_tensor import FunctionalTensor, FunctionalTensorMode
from torch._subclasses.meta_utils import safe_is_leaf
from torch.fx.experimental.proxy_tensor import disable_autocast_cache
from torch.fx.experimental.symbolic_shapes import is_concrete_int
from torch.multiprocessing.reductions import StorageWeakRef
from torch.overrides import TorchFunctionMode
from torch.utils._python_dispatch import (
    is_traceable_wrapper_subclass,
    transform_subclass,
)

from .descriptors import (
    AOTInput,
    AOTOutput,
    InputMutationAOTOutput,
    IntermediateBaseAOTOutput,
    PlainAOTOutput,
    TangentAOTInput,
)
from .functional_utils import (
    are_all_mutations_hidden_from_autograd,
    are_all_mutations_under_no_grad_or_inference_mode,
    from_fun,
    has_data_mutation,
    has_metadata_mutation,
    has_same_metadata,
    MetadataKey,
    to_fun,
    ViewMetaSequence,
    was_inductor_storage_resized,
    was_shallow_copy_data,
)
from .schemas import (
    InputAliasInfo,
    MemoryFormatMeta,
    MutationType,
    OutputAliasInfo,
    OutputType,
    ViewAndMutationMeta,
)
from .subclass_utils import create_subclass_meta
from .utils import _get_autocast_states, KNOWN_TYPES, simple_wraps, strict_zip


if TYPE_CHECKING:
    from collections.abc import Callable

zip = strict_zip

log = logging.getLogger(__name__)
static_input_logger = getArtifactLogger("torch._dynamo", "cudagraph_static_inputs")


class _ViewReplayState(NamedTuple):
    tensor_ref: weakref.ReferenceType[Tensor]
    effective_input_versions: tuple[int | None, ...]
    creation_error_input_indices: tuple[int, ...]
    has_active_multi_output_restriction: bool
    requires_grad: bool
    has_delayed_requires_grad_transition: bool
    non_grad_input_regrad_view_count: int | None
    lineage_grad_root_node: Any | None
    view_meta_grad_enabled: tuple[bool | None, ...]
    view_meta_sequence: tuple[Any, ...]
    storage_ref: StorageWeakRef
    base_idx: int | None


class _InputVersionSnapshotMode(TorchFunctionMode):
    def __init__(self, inputs: list[Any]) -> None:
        super().__init__()
        self.inputs = inputs
        self.initial_grad_enabled = torch.is_grad_enabled()
        self.output_states: dict[int, _ViewReplayState] = {}
        self.creation_error_conditions: dict[int, set[int]] = collections.defaultdict(
            set
        )
        self.input_tensor_id_to_idx = {
            id(inp): i for i, inp in enumerate(inputs) if isinstance(inp, Tensor)
        }
        self.input_storage_to_indices: dict[StorageWeakRef, list[int]] = (
            collections.defaultdict(list)
        )
        for i, inp in enumerate(inputs):
            if isinstance(inp, FunctionalTensor):
                self.input_storage_to_indices[
                    StorageWeakRef(inp.elem.untyped_storage())
                ].append(i)

    def input_versions(self) -> tuple[int | None, ...]:
        return tuple(
            inp._version if isinstance(inp, Tensor) and not inp.is_inference() else None
            for inp in self.inputs
        )

    def __torch_function__(
        self,
        func: Callable[..., Any],
        types: tuple[type, ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        kwargs = kwargs or {}
        tracked_inputs: list[tuple[FunctionalTensor, _ViewReplayState]] = []
        seen_input_ids: set[int] = set()
        for inp in pytree.tree_leaves((args, kwargs)):
            if not isinstance(inp, FunctionalTensor) or id(inp) in seen_input_ids:
                continue
            state = self.output_states.get(id(inp))
            if state is not None and state.tensor_ref() is inp:
                tracked_inputs.append((inp, state))
                seen_input_ids.add(id(inp))

        input_versions_before = self.input_versions() if tracked_inputs else None
        out = func(*args, **kwargs)
        flat_outputs = pytree.tree_leaves(out)
        current_grad_enabled = torch.is_grad_enabled()
        if current_grad_enabled and input_versions_before is not None:
            for tracked_input, state in tracked_inputs:
                if (
                    not state.has_active_multi_output_restriction
                    or state.base_idx is None
                ):
                    continue
                pending_input_indices = tuple(
                    i
                    for i, (effective_version, current_version) in enumerate(
                        zip(
                            state.effective_input_versions,
                            input_versions_before,
                            strict=True,
                        )
                    )
                    if effective_version is not None
                    and current_version != effective_version
                )
                if not pending_input_indices or not any(
                    isinstance(output, Tensor)
                    and output is not tracked_input
                    and output.grad_fn is not None
                    and _autograd_output_depends_on_input(output.grad_fn, tracked_input)
                    for output in flat_outputs
                ):
                    continue
                self.creation_error_conditions[state.base_idx].update(
                    pending_input_indices
                )

        functional_outputs = [
            output for output in flat_outputs if isinstance(output, FunctionalTensor)
        ]
        if functional_outputs:
            output_creation_versions = self.input_versions()
            for output in functional_outputs:
                key = id(output)
                prior = self.output_states.get(key)
                same_object_state = (
                    prior
                    if prior is not None and prior.tensor_ref() is output
                    else None
                )

                view_meta_sequence = tuple(
                    torch._C._functionalization.get_view_meta_sequence(output.elem)
                )
                if not view_meta_sequence:
                    # Storage-changing operations can leave the wrapper's
                    # multi-output tag set after clearing its view recipe. The
                    # new storage no longer belongs to that lineage.
                    self.output_states.pop(key, None)
                    continue
                is_multi_output_view = any(
                    view_meta.is_multi_output for view_meta in view_meta_sequence
                )

                output_storage = StorageWeakRef(output.elem.untyped_storage())
                parent_matches: list[tuple[int, _ViewReplayState]] = []
                for _, state in tracked_inputs:
                    if state.storage_ref != output_storage:
                        continue
                    parent_sequence = state.view_meta_sequence
                    if len(parent_sequence) > len(view_meta_sequence):
                        continue
                    if parent_sequence == view_meta_sequence[: len(parent_sequence)]:
                        parent_matches.append((len(parent_sequence), state))

                parent_state = None
                if parent_matches:
                    longest_prefix = max(length for length, _ in parent_matches)
                    longest_matches = [
                        state
                        for length, state in parent_matches
                        if length == longest_prefix
                    ]
                    first_state = longest_matches[0]
                    if any(
                        state.effective_input_versions
                        != first_state.effective_input_versions
                        or state.creation_error_input_indices
                        != first_state.creation_error_input_indices
                        or state.view_meta_grad_enabled
                        != first_state.view_meta_grad_enabled
                        or state.has_active_multi_output_restriction
                        != first_state.has_active_multi_output_restriction
                        or state.requires_grad != first_state.requires_grad
                        or state.has_delayed_requires_grad_transition
                        != first_state.has_delayed_requires_grad_transition
                        or state.non_grad_input_regrad_view_count
                        != first_state.non_grad_input_regrad_view_count
                        or state.lineage_grad_root_node
                        is not first_state.lineage_grad_root_node
                        for state in longest_matches[1:]
                    ):
                        raise AssertionError(
                            "multi-output view has ambiguous parent creation state"
                        )
                    parent_state = first_state

                lineage_state = same_object_state or parent_state
                base_idx = lineage_state.base_idx if lineage_state is not None else None
                if base_idx is None and output._base is not None:
                    base_idx = self.input_tensor_id_to_idx.get(id(output._base))
                if base_idx is None:
                    storage_input_indices = self.input_storage_to_indices.get(
                        output_storage, []
                    )
                    unique_input_ids = {
                        id(self.inputs[i]): i for i in storage_input_indices
                    }
                    if len(unique_input_ids) == 1:
                        base_idx = next(iter(unique_input_ids.values()))
                    elif len(unique_input_ids) > 1:
                        raise AssertionError(
                            "multi-output view lineage has ambiguous input storage"
                        )

                grad_enabled = (
                    current_grad_enabled
                    if current_grad_enabled != self.initial_grad_enabled
                    else None
                )
                if (
                    lineage_state is not None
                    and len(lineage_state.view_meta_sequence) <= len(view_meta_sequence)
                    and lineage_state.view_meta_sequence
                    == view_meta_sequence[: len(lineage_state.view_meta_sequence)]
                ):
                    view_meta_grad_enabled = lineage_state.view_meta_grad_enabled + (
                        grad_enabled,
                    ) * (
                        len(view_meta_sequence) - len(lineage_state.view_meta_sequence)
                    )
                else:
                    view_meta_grad_enabled = (grad_enabled,) * len(view_meta_sequence)

                has_active_multi_output_restriction = (
                    is_multi_output_view
                    and output._is_view()
                    and output.requires_grad
                    and not output.is_inference()
                    and torch._C._autograd._get_creation_meta(output)
                    != torch._C._autograd.CreationMeta.DEFAULT
                )

                if same_object_state is not None:
                    # Data mutations and in-place metadata operations return
                    # the same FunctionalTensor. An active autograd restriction
                    # keeps its original snapshot so the mutation makes it
                    # stale. Inactive state starts a fresh interval, as do
                    # transitions into or out of the restricted state.
                    effective_input_versions = (
                        same_object_state.effective_input_versions
                        if same_object_state.has_active_multi_output_restriction
                        and has_active_multi_output_restriction
                        else output_creation_versions
                    )
                    detach_indices = [
                        i
                        for i, view_meta in enumerate(view_meta_sequence)
                        if type(view_meta).__name__
                        in ("detach_ViewMeta", "detach__ViewMeta")
                    ]
                    has_delayed_requires_grad_transition = (
                        same_object_state.has_delayed_requires_grad_transition
                        or (
                            output.requires_grad != same_object_state.requires_grad
                            and bool(detach_indices)
                            and len(view_meta_sequence) > detach_indices[-1] + 1
                        )
                    )
                    non_grad_input_regrad_view_count = (
                        same_object_state.non_grad_input_regrad_view_count
                    )
                    lineage_grad_root_node = same_object_state.lineage_grad_root_node
                    if (
                        not same_object_state.requires_grad
                        and output.requires_grad
                        and base_idx is not None
                        and not self.inputs[base_idx].requires_grad
                        and not detach_indices
                    ):
                        non_grad_input_regrad_view_count = len(view_meta_sequence)
                    if not same_object_state.requires_grad and output.requires_grad:
                        with FunctionalTensorMode():
                            lineage_grad_root_node = (
                                torch.autograd.graph.get_gradient_edge(output).node
                            )
                    self.output_states[key] = _ViewReplayState(
                        tensor_ref=weakref.ref(output),
                        effective_input_versions=effective_input_versions,
                        creation_error_input_indices=(
                            same_object_state.creation_error_input_indices
                        ),
                        has_active_multi_output_restriction=(
                            has_active_multi_output_restriction
                        ),
                        requires_grad=output.requires_grad,
                        has_delayed_requires_grad_transition=(
                            has_delayed_requires_grad_transition
                        ),
                        non_grad_input_regrad_view_count=(
                            non_grad_input_regrad_view_count
                        ),
                        lineage_grad_root_node=lineage_grad_root_node,
                        view_meta_grad_enabled=view_meta_grad_enabled,
                        view_meta_sequence=view_meta_sequence,
                        storage_ref=output_storage,
                        base_idx=base_idx,
                    )
                    continue

                creation_error_input_indices: tuple[int, ...] = ()
                if parent_state is not None:
                    creation_error_input_indices = (
                        parent_state.creation_error_input_indices
                    )
                    if (
                        parent_state.has_active_multi_output_restriction
                        and has_active_multi_output_restriction
                        and current_grad_enabled
                    ):
                        new_errors = tuple(
                            i
                            for i, (parent_version, creation_version) in enumerate(
                                zip(
                                    parent_state.effective_input_versions,
                                    output_creation_versions,
                                    strict=True,
                                )
                            )
                            if parent_version is not None
                            and creation_version != parent_version
                        )
                        creation_error_input_indices = tuple(
                            dict.fromkeys((*creation_error_input_indices, *new_errors))
                        )
                if creation_error_input_indices and base_idx is not None:
                    self.creation_error_conditions[base_idx].update(
                        creation_error_input_indices
                    )

                self.output_states[key] = _ViewReplayState(
                    tensor_ref=weakref.ref(output),
                    effective_input_versions=output_creation_versions,
                    creation_error_input_indices=creation_error_input_indices,
                    has_active_multi_output_restriction=(
                        has_active_multi_output_restriction
                    ),
                    requires_grad=output.requires_grad,
                    has_delayed_requires_grad_transition=(
                        parent_state.has_delayed_requires_grad_transition
                        if parent_state is not None
                        else False
                    ),
                    non_grad_input_regrad_view_count=(
                        parent_state.non_grad_input_regrad_view_count
                        if parent_state is not None
                        else None
                    ),
                    lineage_grad_root_node=(
                        parent_state.lineage_grad_root_node
                        if parent_state is not None
                        else None
                    ),
                    view_meta_grad_enabled=view_meta_grad_enabled,
                    view_meta_sequence=view_meta_sequence,
                    storage_ref=output_storage,
                    base_idx=base_idx,
                )
        return out


def _multi_output_view_node_and_index(
    output: Tensor,
    grad_fn: Any,
    view_meta_sequence: ViewMetaSequence,
) -> tuple[Any, int] | None:
    """Find the last multi-output view node represented in a view chain.

    ViewMeta sequences run from the input base to the output, while autograd
    edges run in the opposite direction. Walking one unique autograd edge for
    every ViewMeta after the last multi-output op identifies the shared node
    even when each sibling has additional, distinct suffix views.
    """
    multi_output_indices = [
        i
        for i, view_meta in enumerate(view_meta_sequence.sequence)
        if view_meta.is_multi_output
    ]
    if not multi_output_indices:
        return None

    view_meta_index = multi_output_indices[-1]
    node = grad_fn
    autograd_output_index = int(output.output_nr)
    for _ in range(len(view_meta_sequence.sequence) - view_meta_index - 1):
        next_edges = [edge for edge in node.next_functions if edge[0] is not None]
        if len(next_edges) != 1:
            log.debug(
                "Cannot batch multi-output view replay: expected one autograd "
                "edge while walking suffix views, got %s",
                len(next_edges),
            )
            return None
        node, autograd_output_index = next_edges[0]

    view_meta_output_index = int(view_meta_sequence.sequence[view_meta_index].out_index)
    if autograd_output_index != view_meta_output_index:
        log.debug(
            "Cannot batch multi-output view replay: autograd output index %s "
            "does not match ViewMeta output index %s",
            autograd_output_index,
            view_meta_output_index,
        )
        return None
    return node, view_meta_output_index


def _autograd_graph_reaches_input(grad_fn: Any, input_tensor: Tensor) -> bool:
    target = torch.autograd.graph.get_gradient_edge(input_tensor).node
    return _autograd_graph_reaches_node(grad_fn, target)


def _autograd_graph_reaches_node(grad_fn: Any, target: Any) -> bool:
    pending = [grad_fn]
    seen: set[Any] = set()
    while pending:
        node = pending.pop()
        if node is target:
            return True
        if node is None or node in seen:
            continue
        seen.add(node)
        pending.extend(edge[0] for edge in node.next_functions)
    return False


def _autograd_output_depends_on_input(grad_fn: Any, input_tensor: Tensor) -> bool:
    try:
        return _autograd_graph_reaches_input(grad_fn, input_tensor)
    except AssertionError:
        # A view made under no_grad can require grad while having neither a
        # grad_fn nor an AccumulateGrad edge. A differentiable operation that
        # directly consumes it represents that missing edge with None.
        return any(edge[0] is None for edge in grad_fn.next_functions)


# Note [Tangents memory format]
# We assume tangents memory format to be similar to corresponding output's memory_format.
# The idea is that we are technically making a guess about the strides of our tangents,
# while we trace out the joint.
# If runtime specified tangents will not have the same memory format as predicted traced tangents,
# we coerce them at runtime to traced tangents memory format.


# Coercing and collecting traced tangents memory format in one recursive traversal
def coerce_tangent_and_suggest_memory_format(
    x: Tensor,
) -> tuple[Any, MemoryFormatMeta | list[Any] | None, bool]:
    updated = False
    if not isinstance(x, Tensor):
        return x, None, updated

    out = x.detach()

    is_subclass = is_traceable_wrapper_subclass(out)

    memory_format = MemoryFormatMeta.from_tensor(out)

    # pyrefly: ignore [missing-attribute]
    if memory_format.memory_format is not None:
        was = out
        # pyrefly: ignore [bad-argument-type]
        out = out.contiguous(memory_format=memory_format.memory_format)
        updated = was is not out

    # For subclass we keep memory format of outer strides at the beginning of the list
    out_memory_format = [memory_format] if is_subclass else memory_format

    # Note [Tangents memory format, Part 2]
    # In the same way that "what strides do we assigns to our tangents" is a question
    # that we can not answer (and therefore have to guess) as we trace the backward ahead-of-time,
    # The same applies to any tensor subclass metadata, when we have tangents that are subclasses.
    # To handle this situation, we have two new methods that a tensor subclass can implement:
    # (1) __coerce_tangent_metadata__(self)
    #     Given a subclass with "non-standard" metadata, turn it into a new subclass with "normal" metadata.
    #     The main example here is a DTensor with the "_Partial" placement.
    #     If we have a forward output with a _Partial placement, and corresponding tangent
    #     with a Replicate/Shard placement, we have no way to convert the tangent "back" to a _Partial placement.
    #     This method lets us avoid the problem entirely by allowing subclasses to ensure that we can never
    #     have a tangent with "problematic" metadata, that we cannot convert to.
    # (1) __coerce_same_metadata_as_tangent__(self, metadata)
    #     Given a subclass, and a target differing metadata,
    #     convert self to have the same metadata as the target.
    #     With DTensor being the main example, we can use this to convert a DTensor with a Replicate()
    #     placement into one with a Shard() placement, in the case that we "guessed wrong",
    #     and traced tangents with a Shard() placement at compile time.
    #
    if is_subclass and hasattr(out, "__coerce_tangent_metadata__"):
        out = out.__coerce_tangent_metadata__()  # type: ignore[attr-defined]

    if is_subclass:
        # pyrefly: ignore [missing-attribute]
        attrs = out.__tensor_flatten__()[0]

        for attr in attrs:
            elem = getattr(out, attr)
            (
                new_elem,
                new_elem_memory_format,
                elem_updated,
            ) = coerce_tangent_and_suggest_memory_format(elem)
            # pyrefly: ignore [missing-attribute]
            out_memory_format.append(new_elem_memory_format)
            if elem_updated:
                setattr(out, attr, new_elem)

    return out, out_memory_format, updated


# This is a version of functionalization that is specifically designed
# for the AOTAutograd use case.
#
# Unlike functorch's variant, this doesn't use the functorch level system,
# instead it directly uses PyTorch's conventional dispatcher to hit the
# functionalization key.  In particular, this means that FunctionalTensorWrapper
# can have autograd data stored directly on it.
#
# In typical AOTAutograd usage, the dispatch key order will look like:
#
#   Autograd - Functionalization ~~~~> Proxy Mode - Fake Tensor
#       outer tensor                        inner tensor
#
# Returns:
# - ViewAndMutationMeta, telling us metadata about the inputs and outputs, and
#   The list of outputs from the forward, but **only** the outputs that we need
#   to pass in as tangents into the backward.
#   Specifically, aliased outputs from the forward get regenerated, and don't participate
#   in the compiled backward function.
def run_functionalized_fw_and_collect_metadata(
    f: Callable[..., Any],
    *,
    flat_args_descs: list[AOTInput],
    keep_input_mutations: bool,
    capture_multi_output_view_invalidations: bool = True,
    # Note: this is guaranteed to be set when running under dynamo
    static_input_indices: list[int] | None = None,
    pre_dispatch: bool = False,
) -> Callable[..., ViewAndMutationMeta]:
    memo: dict[Tensor, Tensor] = {}

    # TODO: see if we can rewrite this to be more accurate using
    # overload
    def _to_fun(t: object) -> object:
        if isinstance(t, Tensor):
            if t in memo:
                return memo[t]
            r = to_fun(t)
            memo[t] = r
            return r
        else:
            return t

    @simple_wraps(f)
    def inner(*flat_args: Any) -> ViewAndMutationMeta:
        # This function is meant to be run with the forward, which expects a flat list of tensor/symint/other args.
        if not all(
            isinstance(a, tuple(KNOWN_TYPES)) or is_custom_class(type(a))
            for a in flat_args
        ):
            raise AssertionError("all flat_args must be KNOWN_TYPES or opaque types")

        input_info: list[InputAliasInfo] = []
        output_info: list[OutputAliasInfo] = []

        prior_grad_enabled = torch.is_grad_enabled()
        prior_autocast_states = _get_autocast_states()

        # See Note [Disabling Functionalize TLS Above Python Functionalization]
        disable_above = torch._C._ExcludeDispatchKeyGuard(
            torch._C.DispatchKeySet(torch._C.DispatchKey.Functionalize)
        )

        # It doesn't matter if we run this under predispatch or not because it is
        # only for figuring out metadata
        mode = FunctionalTensorMode(
            _allow_token_discovery=True,
            _keep_input_mutations=keep_input_mutations,
        )
        suppress_pending = contextlib.nullcontext()
        fake_mode = detect_fake_mode()
        if fake_mode and (shape_env := fake_mode.shape_env):
            suppress_pending = shape_env.ignore_fresh_unbacked_symbols()
        with disable_above, mode, suppress_pending, disable_autocast_cache():
            # precondition: The passed in function already handles unflattening inputs + flattening outputs
            flat_f_args = pytree.tree_map(_to_fun, flat_args)
            flat_f_args_descs = flat_args_descs
            mutation_order_mode = (
                _InputVersionSnapshotMode(flat_f_args)
                if capture_multi_output_view_invalidations
                else contextlib.nullcontext()
            )
            with mutation_order_mode:
                flat_f_outs = f(*flat_f_args)

            # Assert that f does NOT have an AOTOutputs in it, easy mistake to
            # make!  You need to drop the second output before calling this
            # function
            if pytree.tree_any(lambda x: isinstance(x, AOTOutput), flat_f_outs):
                raise AssertionError(
                    f"{f} returned AOTOutput when it shouldn't. Did you remember to wrap the "
                    "function with without_output_descs before passing it here?"
                )

            # NB: this is just to setup the input descriptors, we will
            # recreate these descriptors (with the same convention!) when we
            # actually do the trace
            flat_f_outs_descs = [PlainAOTOutput(i) for i in range(len(flat_f_outs))]

            # We didn't do any tracing, so we don't need to process the
            # unbacked symbols, they will just disappear into the ether.
            # Also, prevent memoization from applying.
            if fake_mode:
                fake_mode.epoch += 1
                fake_mode.reset_nt_tensor_id_counter()

        if prior_autocast_states != _get_autocast_states():
            raise RuntimeError(
                "AOTAutograd does not support tracing graphs that mutate the autocast state. "
                "Dynamo will only insert autocast context managers (e.g. with torch.autocast(..)) into the graph, "
                "which will unwind all of their mutations to autocast state before the graph exits. "
                "If you encounter this error while using torch.compile, please file a bug."
            )

        # Inspect the state of the input tensor functional wrapper to detect input mutation info
        # If inp[i] has a metadata-only mutation, then maybe_inputs_with_mutated_metadata[i] contains the updated version
        for arg, f_arg in zip(flat_args, flat_f_args):
            mutates_data = has_data_mutation(f_arg)
            mutates_metadata = has_metadata_mutation(
                f_arg, arg, check_only_storage_mutation=False
            )
            mutates_storage_metadata = has_metadata_mutation(
                f_arg, arg, check_only_storage_mutation=True
            )
            mutations_hidden_from_autograd = are_all_mutations_hidden_from_autograd(
                f_arg
            )
            mutations_under_no_grad_or_inference_mode = (
                mutates_data
                and are_all_mutations_under_no_grad_or_inference_mode(f_arg)
            )
            mutation_inductor_storage_resize = was_inductor_storage_resized(f_arg)

            if mutates_storage_metadata:
                mutates_data = False

            requires_grad = isinstance(f_arg, torch.Tensor) and f_arg.requires_grad

            input_info.append(
                InputAliasInfo(
                    is_leaf=isinstance(arg, Tensor) and safe_is_leaf(arg),
                    mutates_data=mutates_data,
                    mutates_metadata=mutates_metadata,
                    mutations_hidden_from_autograd=mutations_hidden_from_autograd,
                    mutates_storage_metadata=mutates_storage_metadata,
                    mutation_is_shallow_copy_data=was_shallow_copy_data(f_arg),
                    mutations_under_no_grad_or_inference_mode=mutations_under_no_grad_or_inference_mode,
                    mutation_inductor_storage_resize=mutation_inductor_storage_resize,
                    requires_grad=requires_grad,
                    keep_input_mutations=keep_input_mutations,
                )
            )

        # If a function involves creating a tensor, and returning a view of it, such that its _base is the intermediate,
        # We need to make sure our graph returns the _base as a graph output, and we manually recreate the view
        # to return to the user. Why? The backend compiler is free to (incorrectly) not set requires_grad
        # on the base tensor, but we are obligated to properly set requires-gradness on the real output.

        inp_storage_refs = {
            StorageWeakRef(inpt.untyped_storage()): idx
            for idx, inpt in enumerate(flat_f_args)
            if isinstance(inpt, Tensor)
        }

        # We need inp tensor id's to be able to tell if an outputs **are** inputs.
        inp_tensor_ids = {id(inpt) for inpt in flat_f_args if isinstance(inpt, Tensor)}
        inp_tensor_id_to_idx = {
            id(inpt): idx
            for idx, inpt in enumerate(flat_f_args)
            if isinstance(inpt, Tensor)
        }
        # We need output tensor id's to tell if any output._base` attributes **are** other outputs.
        # (This is also a dict because we need to know that output's index, so we can regenerate
        # the alias from it).
        out_tensor_ids = {id(o): i for i, o in enumerate(flat_f_outs)}

        # Keep track of which outputs alias other outputs
        out_tensor_alias_counts: collections.defaultdict[StorageWeakRef | None, int] = (
            collections.defaultdict(int)
        )
        # This tells us, for a given group of outputs that alias each other,
        # whether they e.g. all came from an unbind call
        num_aliased_tensors_that_are_multi_output_views: collections.defaultdict[
            StorageWeakRef | None, int
        ] = collections.defaultdict(int)
        multi_output_view_tensor_ids: set[int] = set()

        out_storage_to_metadata_key_to_tensors: collections.defaultdict[
            StorageWeakRef | None,
            collections.defaultdict[MetadataKey, set[torch.Tensor]],
        ] = collections.defaultdict(lambda: collections.defaultdict(set))

        curr_storage = None
        for o in flat_f_outs:
            if isinstance(o, torch.Tensor):
                curr_storage = StorageWeakRef(o.untyped_storage())
                out_tensor_alias_counts[curr_storage] += 1
                # Note: [AOTAutograd: differentiable outputs that alias each other from a multi-output view call]
                # This is an optimization on top of the "alias of intermediates" logic,
                # which you can read more about under Note [AOT Autograd: outputs aliasing inputs or intermediates!]
                #
                # Before describing the optimization: this is important for AOTAutograd to have good
                # perf around, multi-output views. HOWEVER:
                # - There is a more generic change to AOTAutograd that we'd like to make, that subsumes this case,
                #   around using pre-dispatch tracing to partition out a graph so we can faithfully replay all
                #   views without having to regenerate them at runtime.
                # - It's loosely described in this doc (more details will be added soon):
                #   https://docs.google.com/document/d/1DlfFq8TKbuAn2zyJxLfoW-X1qkkm5PLdHFtySo03QAk/edit
                # - Once that change lands, we should just rip out this "optimization", since:
                #   (1) It will be fully unnecessary
                #   (2) Although it is only a few lines of code, it is a bit difficult to reason about
                #       its correctness with the autograd engine in all cases.
                #
                #
                # What is this optimization? Consider the below case:
                # def f(x):
                #     intermediate = x.mul(2)
                #     # x and intermediate here require grad
                #     o1, o2, ... o10 = intermediate.unbind(-1)
                #     return intermediate, o1, o2, ... o10
                # Now, the "intermediate base" handling in AOTAutograd implies that we must do the following:
                #   (1) return "intermediate as an extra output of the compiled graph
                #   (2) regenerate each aliased output off of "intermediate", **outside** of the autograd.Function.
                # The reason AOTAutograd ordinarily does this is for safety: the autograd engine needs to know
                # that o1 through o10 are all aliased, and if we blindly return o1 through o10 from the autograd.Function,
                # this information will be hidden.
                # In particular, mutating one alias might require autograd to update autograd metadata on the other aliases
                # (like their grad_fn, for example, when the autograd engine needs to do view-replay).
                #
                # However, intermediate_base logic can be bad for backward performance (we sometimes generate
                # as_strided calls during the intermediate base logic, which can have a slow backward formula).
                # Is it possible to find a set of conditions where it is **safe** to hide the output aliasing from autograd?
                #
                # For a set of outputs of the graph that alias each other, o_1...o_k, consider:
                # (1) They came from the same multi-output view op, e.g. o_1, ..., o_k = intermediate.unbind(0)
                # (2) If there are any other aliases of o_1 through o_k (in the example above, intermediate),
                #     **at most** 1 can escape from the graph (e.g. there is not some other graph input/output
                #     o_other, that aliases these outputs)
                # (3) o_1...o_k all require_grad, they all share the same ._base, and their ._base requires grad.
                #     This condition is important because it's what causes slowness in the intermediate_base
                #     codepath of aot_autograd. Ordinarily, o_1...o_k would all get a grad_fn, and
                #     aot_autograd's view-replay might give each output an AsStridedBackward as its grad_fn.
                #     "K" AsStridedBackward calls will be *much* slower than a single UnbindBackward.
                # In this setup, is it possible to mutate one of the outputs o_i in a way that would affect the autograd meta
                # of the other aliases?
                #
                # Claim: No! Consider a few example (which I'm pretty sure cover all cases of mutation w.r.t. autograd):
                # (a) What happens if we mutate any of o_1 through o_k directly?
                #     Autograd raises an error:
                #     "RuntimeError: Output 0 of UnbindBackward0 is a view and is being modified inplace. This view is
                #      the output of a function that returns multiple views. Such functions do not allow the output
                #      views to be modified inplace. You should replace the inplace operation by an out-of-place one."
                # (b) What if we take a view of o_k and mutate it, o_k.view(o_k.shape).mul_(2)?
                #     Autograd raises the same error- the "multi-output-view"ness of an alias propagates to future views.
                # (c) What if we mutate o_k under no_grad?
                #     Autograd raises the same error
                # (d) What if we detach and mutate, e.g. o_k.detach().mul_(2)?
                #     Autograd allows this, *but* autograd updates all alias's grad_fn's to be error functions when accessed.
                #     Autograd raises the same error
                # (e) What if we try to mutate another alias of o_1...o_k, that was **not** created from a multi-output view?
                #     We promised that there is at most **one** such alias, e.g. intermediate in the example above.
                #     You can mutate intermediate, but in eager mode this will change the grad_fn of o_1...o_k
                #     to be error fn's.
                #     Since intermediate was the *only* non-multi-output-alias, there are no other aliases
                #     of `intermediate` around that were produced by the compiled fn and have a valid grad_fn.
                #
                # Coming back to this optimization:
                # Given that it is not possible for mutating one of these aliases to affect the autograd metadata of another alias
                # without causing an error in eager mode, we will simple hide the aliasing from autograd during torch.compile
                # if all of the above conditions are met.
                #
                # This optimization only applies to views of intermediates. For
                # views of graph inputs, hiding the aliases gives all siblings a
                # shared CompiledFunctionBackward whose saved tensors are freed
                # after the first backward. Those aliases instead use grouped
                # runtime view replay so independent backwards match eager mode.
                # This has the slight downside that it's possible to write some "bad" code that autograd will raise an error on
                # in eager but fail to during torch.compile, but it has the benefit that this code has much better performance.
                # NOTE: if and when we eventually update AOTAutograd to do the "view graph slicing" defined here:
                # https://docs.google.com/document/d/1DlfFq8TKbuAn2zyJxLfoW-X1qkkm5PLdHFtySo03QAk/edit,
                # then this optimization will probably matter less and might be ok to remove.
                is_cur_tensor_multi_out_view = isinstance(
                    o, FunctionalTensor
                ) and torch._functionalize_is_multi_output_view(  # type: ignore[attr-defined]
                    o.elem
                )
                if is_cur_tensor_multi_out_view and isinstance(
                    mutation_order_mode, _InputVersionSnapshotMode
                ):
                    state = mutation_order_mode.output_states.get(id(o))
                    is_cur_tensor_multi_out_view = (
                        state is not None
                        and state.tensor_ref() is o
                        and any(
                            view_meta.is_multi_output
                            for view_meta in state.view_meta_sequence
                        )
                    )
                if is_cur_tensor_multi_out_view:
                    num_aliased_tensors_that_are_multi_output_views[curr_storage] += 1
                    multi_output_view_tensor_ids.add(id(o))
                if o.requires_grad:
                    out_storage_to_metadata_key_to_tensors[curr_storage][
                        MetadataKey.make(o)
                    ].add(o)

        # maps the id of an intermediate base to its index in the output of the compiled forward
        intermediate_base_tensor_id_to_output_idx: dict[int, int] = {}
        intermediate_bases: list[torch.Tensor] = []
        intermediate_bases_descs: list[AOTInput] = []
        input_multi_output_view_groups: dict[tuple[int, Any], int] = {}
        detached_output_roots: dict[int, tuple[Any, int]] = {}
        output_grad_fns: list[Any] = []
        # Why Do We Care If Storage Changed?
        # It's important to understand the implications of storage changes in complex scenarios. Take this example:
        #
        # def f(x):
        #     x_storage = x.untyped_storage()
        #     non_leaf_tensor = torch.ones(4, requires_grad=True).clone()
        #
        #     # Using no_grad() and _unsafe_preserve_version_counter to simulate the .data = operation
        #     with torch.no_grad(), torch.autograd._unsafe_preserve_version_counter(x):
        #         x.set_(non_leaf_tensor.untyped_storage())
        #
        #     out = x.view(-1)
        #
        #     # Restoring x to its original storage, again simulating .data = operation
        #     with torch.no_grad(), torch.autograd._unsafe_preserve_version_counter(x):
        #         x.set_(x_storage)
        #
        #     return out
        #
        # In this scenario, 'x' and 'out' have different shapes and are stored at different memory addresses, aka no aliasing.
        # However, due to how set_() and more specifically, set is functionalized, is defined to preserve eager semantics,
        # the autograd engine mistakenly assumes that 'x' and 'out' are aliased, treating 'x' as 'out._base'.
        # This misinterpretation leads to an 'alias_of_input' flag, causing an unnecessary as_strided() call to be generated,
        # which could lead to issues later in the code.
        for o, desc in zip(flat_f_outs, flat_f_outs_descs):
            functional_tensor_storage_changed = isinstance(
                o, FunctionalTensor
            ) and torch._functionalize_was_storage_changed(  # type: ignore[attr-defined]
                o.elem
            )
            curr_storage = (
                None
                if not isinstance(o, torch.Tensor)
                else StorageWeakRef(o.untyped_storage())
            )
            multi_output_view_base_idx = (
                inp_storage_refs.get(curr_storage)
                if curr_storage is not None and id(o) in multi_output_view_tensor_ids
                else None
            )
            outs_with_identical_metadata_that_require_grad: list[torch.Tensor] = (
                []
                if not isinstance(o, Tensor)
                else [
                    curr
                    for curr in out_storage_to_metadata_key_to_tensors[curr_storage][
                        MetadataKey.make(o)
                    ]
                    if o is not curr
                ]
            )

            # See Note [Accessing .grad_fn on FunctionalTensor]
            # In-place operations on views will trigger a lazy rebase of the autograd graph;
            # this runs during access to the .grad_fn. The rebase logic will invoke view ops
            # on FunctionalTensors, so we must enable a FunctionalTensorMode here to ensure
            # these op calls succeed.
            grad_fn = None
            creation_meta = None
            multi_output_view_was_invalidated = False
            multi_output_view_invalidating_input_indices: tuple[int, ...] = ()
            view_replay_grad_enabled: tuple[bool | None, ...] = ()
            replay_from_detached_base = False
            replay_detached_view_meta_sequence = False
            has_explicit_detach = False
            has_delayed_requires_grad_transition = False
            non_grad_input_regrad_view_count = None
            lineage_grad_root_node = None
            if isinstance(o, Tensor):
                if o._is_view() and not o.is_inference():
                    creation_meta = torch._C._autograd._get_creation_meta(o)
                original_attr_version = None
                if (
                    capture_multi_output_view_invalidations
                    and id(o) in multi_output_view_tensor_ids
                    and o._is_view()
                    and creation_meta != torch._C._autograd.CreationMeta.DEFAULT
                ):
                    original_attr_version = torch._C._autograd._get_view_attr_version(o)
                    multi_output_view_was_invalidated = (
                        original_attr_version != o._version
                    )
                    if multi_output_view_was_invalidated:
                        # Inspecting grad_fn normally raises for this stale
                        # multi-output view. Temporarily restore its creation
                        # version so metadata collection can recover the shared
                        # node, then put the eager error state back.
                        torch._C._autograd._unsafe_set_view_attr_version(o, o._version)
                try:
                    with FunctionalTensorMode():
                        grad_fn = o.grad_fn
                finally:
                    if multi_output_view_was_invalidated:
                        if original_attr_version is None:
                            raise AssertionError("expected an original view version")
                        torch._C._autograd._unsafe_set_view_attr_version(
                            o, original_attr_version
                        )
                if (
                    capture_multi_output_view_invalidations
                    and id(o) in multi_output_view_tensor_ids
                ):
                    if not isinstance(mutation_order_mode, _InputVersionSnapshotMode):
                        raise AssertionError("expected mutation-order metadata")
                    creation = mutation_order_mode.output_states.get(id(o))
                    if creation is None or creation.tensor_ref() is not o:
                        raise AssertionError(
                            "missing input versions for multi-output view"
                        )
                    if creation.base_idx is not None:
                        multi_output_view_base_idx = creation.base_idx
                    view_replay_grad_enabled = creation.view_meta_grad_enabled
                    has_explicit_detach = any(
                        type(view_meta).__name__
                        in ("detach_ViewMeta", "detach__ViewMeta")
                        for view_meta in creation.view_meta_sequence
                    )
                    has_delayed_requires_grad_transition = (
                        creation.has_delayed_requires_grad_transition
                    )
                    non_grad_input_regrad_view_count = (
                        creation.non_grad_input_regrad_view_count
                    )
                    lineage_grad_root_node = creation.lineage_grad_root_node
                    # detach() descendants can retain functionalization's
                    # multi-output-view tag without carrying differentiable
                    # view restrictions. A direct detach is not a view, while
                    # a view after detach has DEFAULT creation metadata.
                    if (
                        o._is_view()
                        and o.requires_grad
                        and not o.is_inference()
                        and creation_meta != torch._C._autograd.CreationMeta.DEFAULT
                    ):
                        final_input_versions = mutation_order_mode.input_versions()
                        multi_output_view_invalidating_input_indices = tuple(
                            i
                            for i, (creation_version, final_version) in enumerate(
                                zip(
                                    creation.effective_input_versions,
                                    final_input_versions,
                                    strict=True,
                                )
                            )
                            if creation_version is not None
                            and final_version != creation_version
                        )

            is_result_of_custom_autograd_fn = False
            # Need to check for both custom cpp (CppFunction) and python (BackwardCFunction)
            # autograd fns
            if type(grad_fn).__name__ == "CppFunction":
                is_result_of_custom_autograd_fn = True
            if isinstance(grad_fn, torch.autograd.function.BackwardCFunction):
                is_result_of_custom_autograd_fn = True

            if not isinstance(o, Tensor):
                output_type = OutputType.non_alias
                base_idx = None
            elif (
                curr_storage in inp_storage_refs
                and grad_fn is not None
                and is_result_of_custom_autograd_fn
            ):
                output_type = OutputType.custom_function_view
                base_idx = None
            elif (
                curr_storage in inp_storage_refs
                and not functional_tensor_storage_changed
            ):
                # pyrefly: ignore [bad-index, index-error]
                base_idx = inp_storage_refs[curr_storage]
                is_input_tensor = id(o) in inp_tensor_ids
                # Preserve input aliasing even for multi-output views (e.g.
                # unbind/split). Otherwise pure view functions get a shared
                # CompiledFunctionBackward instead of replaying views from the
                # input, and individual backwards over outputs incorrectly
                # re-enter that node after its saved tensors are freed.
                if is_input_tensor:
                    output_type = OutputType.is_input
                elif (
                    id(o) in multi_output_view_tensor_ids
                    and has_explicit_detach
                    and multi_output_view_base_idx is not None
                ):
                    # Functionalization keeps view ancestry across detach, but
                    # replaying that recipe would reconnect the severed
                    # autograd history. Rebuild the final geometry from a
                    # detached runtime base instead.
                    output_type = OutputType.alias_of_input
                    base_idx = multi_output_view_base_idx
                    replay_from_detached_base = True
                    replay_detached_view_meta_sequence = (
                        o._is_view()
                        and creation_meta
                        == torch._C._autograd.CreationMeta.MULTI_OUTPUT_NODE
                    )
                elif (
                    id(o) in multi_output_view_tensor_ids
                    and o._is_view()
                    and creation_meta == torch._C._autograd.CreationMeta.DEFAULT
                    and multi_output_view_base_idx is not None
                ):
                    # A view after detach keeps functionalization's ancestry
                    # tag but has ordinary DEFAULT autograd view semantics.
                    output_type = OutputType.alias_of_input
                    base_idx = multi_output_view_base_idx
                    replay_from_detached_base = True
                elif (
                    id(o) in multi_output_view_tensor_ids
                    and grad_fn is None
                    and o.requires_grad
                    and not o._is_view()
                    and multi_output_view_base_idx is not None
                ):
                    # detach().requires_grad_() starts a new leaf even though
                    # functionalization retains the multi-output ancestry tag.
                    # Replaying it from the input would reconnect that severed
                    # autograd history.
                    output_type = OutputType.alias_of_input
                    base_idx = multi_output_view_base_idx
                    replay_from_detached_base = True
                elif (
                    id(o) in multi_output_view_tensor_ids
                    and grad_fn is not None
                    and creation_meta
                    == torch._C._autograd.CreationMeta.MULTI_OUTPUT_NODE
                    and o._base is not None
                    and id(o._base) in inp_tensor_id_to_idx
                    and flat_f_args[inp_tensor_id_to_idx[id(o._base)]].requires_grad
                    and _autograd_graph_reaches_input(
                        grad_fn, flat_f_args[inp_tensor_id_to_idx[id(o._base)]]
                    )
                ):
                    # Storage identity is insufficient when multiple inputs
                    # share storage but have independent autograd histories.
                    # Use the differentiable view base that actually owns this
                    # output's backward edge.
                    base_idx = inp_tensor_id_to_idx[id(o._base)]
                    output_type = OutputType.alias_of_input
                elif id(o) in multi_output_view_tensor_ids and grad_fn is not None:
                    if (
                        creation_meta
                        == torch._C._autograd.CreationMeta.MULTI_OUTPUT_NODE
                        and multi_output_view_base_idx is not None
                    ):
                        output_type = OutputType.alias_of_input
                        base_idx = multi_output_view_base_idx
                        if not flat_f_args[base_idx].requires_grad:
                            # requires_grad_() started a new leaf from a
                            # non-grad input. Replay on a detached runtime base
                            # so a later view does not acquire an input edge.
                            replay_from_detached_base = True
                            replay_detached_view_meta_sequence = True
                        # Otherwise a no_grad view severed the graph without a
                        # detach. Per-ViewMeta grad modes can replay that
                        # boundary directly on the original input, preserving
                        # eager's exposed ._base identity.
                    else:
                        output_type = OutputType.non_alias
                        base_idx = None
                else:
                    output_type = OutputType.alias_of_input
            elif functional_tensor_storage_changed and id(o) in inp_tensor_ids:
                # When there is a set_() on an input, we cannot rely on checking storages
                # to detect if we are returning an input (since the inputs storage is different)
                if curr_storage is None:
                    raise AssertionError("curr_storage must not be None")
                base_idx = inp_storage_refs[curr_storage]
                output_type = OutputType.is_input

            # We only need to handle the intermediate base case when both
            # the intermediate base and the output require gradients.
            # See Note [AOT Autograd: outputs aliasing inputs or intermediates!]
            elif o._base is not None and o.requires_grad and o._base.requires_grad:
                num_aliased_outs = out_tensor_alias_counts[curr_storage]
                num_multi_output_view_outs = (
                    num_aliased_tensors_that_are_multi_output_views[curr_storage]
                )
                num_aliased_outs_that_are_not_multi_output_views = (
                    num_aliased_outs - num_multi_output_view_outs
                )
                # Note: [AOTAutograd: differentiable outputs that alias each other from a multi-output view call]
                if (
                    out_tensor_alias_counts[curr_storage] == 1
                    or num_aliased_outs_that_are_not_multi_output_views <= 1
                ):
                    # Note [Intermediate Bases Optimization]
                    # Normally if we have an output that aliases an intermediate,
                    # we need to add the extra "intermediate base" logic further down
                    # to prevent autograd from yelling at us if the user later tries to
                    # mutate that output.
                    # However, the common case here is if we have an output that aliases an intermediate,
                    # but doesn't alias any other outputs.
                    # In that case, autograd shouldn't have to worry about the aliasing at all
                    # (if that output is mutated, there are no other live aliases for autograd to worry about).
                    # The "intermediate bases" can hurt inductor perf by forcing more variables to become outputs.
                    # So as an optimization, we won't do intermediate base handling in this case.
                    # Instead, we'll hide the aliasing from autograd using aten._unsafe_view().
                    if (
                        out_tensor_alias_counts[curr_storage] != 1
                        and num_aliased_outs_that_are_not_multi_output_views <= 1
                    ):
                        log.debug(
                            "Encountered AOTAutograd case: differentiable outputs that alias each other \
from a multi-output view call"
                        )
                    output_type = OutputType.unsafe_view_alias
                    base_idx = None
                else:
                    # First, check if o's ._base is an existing output
                    maybe_existing_out_idx = out_tensor_ids.get(id(o._base))
                    if maybe_existing_out_idx is not None:
                        # Special case where the output is an alias of a graph intermediate, but that intermediate
                        # is itself also a user output.
                        output_type = (
                            OutputType.alias_of_intermediate_base_is_user_output
                        )
                        base_idx = maybe_existing_out_idx
                    else:
                        # Next, check if o's ._base is an intermediate base that we already returned
                        maybe_existing_base_output_idx = (
                            intermediate_base_tensor_id_to_output_idx.get(id(o._base))
                        )
                        if maybe_existing_base_output_idx is not None:
                            output_type = OutputType.alias_of_intermediate
                            base_idx = maybe_existing_base_output_idx
                        else:
                            # Otherwise, take o._base and explicitly return it as an output in the compiled graph
                            new_out_idx = len(intermediate_bases)
                            base_idx = new_out_idx
                            # Indicate to the logic later on (when we trace the joint)
                            # that this particular output should get it's ._base appended to the forward graph outputs
                            output_type = (
                                OutputType.alias_of_intermediate_save_as_output
                            )
                            intermediate_base_tensor_id_to_output_idx[id(o._base)] = (
                                new_out_idx
                            )
                            intermediate_bases.append(o._base)
                            # NB: The desc we picked here is guaranteed to be
                            # synchronized with the one in
                            # graph_capture_wrappers.py because we
                            # SPECIFICALLY notated this output as
                            # alias_of_intermediate_save_as_output
                            intermediate_bases_descs.append(
                                TangentAOTInput(IntermediateBaseAOTOutput(desc))
                            )
            elif (
                # See https://github.com/pytorch/pytorch/issues/100348 for this case.
                # This protects against the specific case where a user fn returns (output, output.detach())
                out_tensor_alias_counts[curr_storage] > 1
                and len(outs_with_identical_metadata_that_require_grad) > 0
                and not o.requires_grad
            ):
                # In theory we could use any of these tensors to regenerate the aliased outputs from,
                # since they all alias each other and have identical metadata
                out_alias = outs_with_identical_metadata_that_require_grad[0]
                existing_out_idx = out_tensor_ids[id(out_alias)]
                output_type = OutputType.alias_of_intermediate_base_is_user_output
                base_idx = existing_out_idx
            elif (
                o._base is not None
                and not is_traceable_wrapper_subclass(o)
                and has_same_metadata(o, o._base)
                and id(o._base) in out_tensor_ids
            ):
                # o is a no-op view of another user output, but not
                # differentiable, so none of the branches above fired. Left as
                # non_alias, the backend is free to collapse the view and return
                # one object for both outputs while eager returns two; an eager
                # resize_() on the alias across a graph break then corrupts the
                # base. Regenerate the view at runtime instead.
                # See https://github.com/pytorch/pytorch/issues/191449
                #
                # Two shapes of this bug are knowingly still broken here and are
                # tracked in #191449, which stays open for them: t.detach() leaves ._base as None so
                # it never reaches this arm, and traceable wrapper subclasses
                # are excluded because their view_meta_sequence is not captured
                # and the as_strided fallback in gen_alias_from_base is not
                # supported by e.g. DTensor. For both, the issue's own repro
                # still returns a base of shape (12,) where eager gives (1,).
                #
                # o._base.requires_grad would imply o.requires_grad
                # (DifferentiableViewMeta), which the branch above already
                # handles, so o._base is never a saved intermediate base and
                # this index is always in user-output space.
                output_type = OutputType.alias_of_intermediate_base_is_user_output
                base_idx = out_tensor_ids[id(o._base)]
            else:
                output_type = OutputType.non_alias
                base_idx = None

            if isinstance(o, torch.Tensor):
                dynamic_dims = {
                    i for i, s in enumerate(o.shape) if not is_concrete_int(s)
                }
            else:
                dynamic_dims = None

            # Save the current FunctionalTensor output.
            #
            # This will be used at runtime for reconstructing output views from
            # their respective base tensors.
            #
            # The FunctionalTensor will be saved if one of the 2 conditions below
            # is true:
            view_meta_sequence = None
            if (
                # 1. If the output_type is either of:
                #    (i) alias_of_intermediate;
                #    (ii) alias_of_intermediate_save_as_output; or
                #    (iii) alias_of_intermediate_base_is_user_output.
                #
                # No need to worry about in-place view operations here, since
                # this functionalization step eliminates mutations.
                #
                # i.e. we have access to the actual base tensor, before the
                # in-place operation was applied.
                output_type
                in (
                    OutputType.alias_of_intermediate,
                    OutputType.alias_of_intermediate_save_as_output,
                    OutputType.alias_of_intermediate_base_is_user_output,
                )
            ) or (
                # 2. If the output_type is alias_of_input, and no in-place view
                #    operation was run on the input (base tensor).
                #
                # In this case, we need to check for metadata mutation because
                # the runtime explicitly reconstructs the inputs, before actually
                # reconstructing the outputs. Due to in-place view operations, the
                # fully reconstructed input may not be this output base tensor
                # anymore.
                output_type == OutputType.alias_of_input
                and base_idx is not None
                and not input_info[base_idx].mutates_metadata
            ):
                if isinstance(o, FunctionalTensor):
                    view_meta_sequence = ViewMetaSequence(o)
            requires_structured_view_replay = (
                replay_from_detached_base or len(set(view_replay_grad_enabled)) > 1
            )
            if replay_from_detached_base and has_delayed_requires_grad_transition:
                raise AssertionError(
                    "aot_autograd() does not yet handle delayed requires_grad_() "
                    "transitions in detached multi-output-view lineages"
                )
            if (
                non_grad_input_regrad_view_count is not None
                and view_meta_sequence is not None
                and len(view_meta_sequence.sequence) > non_grad_input_regrad_view_count
            ):
                raise AssertionError(
                    "aot_autograd() does not yet handle view operations after "
                    "requires_grad_() in a non-grad-input multi-output-view lineage"
                )
            if (
                output_type == OutputType.alias_of_input
                and id(o) in multi_output_view_tensor_ids
                and requires_structured_view_replay
                and (
                    view_meta_sequence is None
                    or any(
                        view_meta.has_symbolic_inputs
                        for view_meta in view_meta_sequence.sequence
                    )
                )
            ):
                raise AssertionError(
                    "aot_autograd() does not yet handle symbolic or unavailable "
                    "ViewMeta replay for detached or mixed-grad-mode "
                    "multi-output views"
                )
            starts_new_grad_history_from_non_grad_input = (
                isinstance(o, Tensor)
                and output_type is OutputType.alias_of_input
                and id(o) in multi_output_view_tensor_ids
                and o.requires_grad
                and base_idx is not None
                and not flat_f_args[base_idx].requires_grad
            )
            if (
                (
                    replay_from_detached_base
                    or starts_new_grad_history_from_non_grad_input
                )
                and isinstance(o, Tensor)
                and lineage_grad_root_node is not None
            ):
                # Reconstructing aliases independently only changes observable
                # autograd behavior when they share the same gradient root.
                # Purely non-differentiable detached siblings have no root to
                # split and remain safe to return.
                grad_root_id = id(lineage_grad_root_node)
                if grad_root_id in detached_output_roots:
                    raise AssertionError(
                        "aot_autograd() does not yet handle multiple returned "
                        "aliases sharing a detached multi-output-view lineage"
                    )
                detached_output_roots[grad_root_id] = (
                    lineage_grad_root_node,
                    len(output_info),
                )
            requires_grad = isinstance(o, torch.Tensor) and o.requires_grad
            multi_output_view_group = None
            multi_output_view_index = None
            if (
                output_type == OutputType.alias_of_input
                and id(o) in multi_output_view_tensor_ids
                and not replay_from_detached_base
                and grad_fn is not None
                and view_meta_sequence is not None
                and view_meta_sequence.sequence
            ):
                multi_output_view = _multi_output_view_node_and_index(
                    o, grad_fn, view_meta_sequence
                )
                if multi_output_view is not None:
                    multi_output_node, multi_output_view_index = multi_output_view
                    if base_idx is None:
                        raise AssertionError(
                            "multi-output input alias must have an input base"
                        )
                    group_key = (base_idx, multi_output_node)
                    if group_key not in input_multi_output_view_groups:
                        input_multi_output_view_groups[group_key] = len(
                            input_multi_output_view_groups
                        )
                    multi_output_view_group = input_multi_output_view_groups[group_key]

            if (
                id(o) in multi_output_view_tensor_ids
                and multi_output_view_base_idx is None
                and output_type in (OutputType.alias_of_input, OutputType.is_input)
            ):
                multi_output_view_base_idx = base_idx
            out_info = OutputAliasInfo(
                output_type=output_type,
                raw_type=type(o),
                base_idx=base_idx,
                dynamic_dims=dynamic_dims,
                requires_grad=requires_grad,
                # A view created under no_grad() inherits requires_grad from
                # its base but has no grad_fn and does not participate in
                # differentiation.
                requires_grad_for_backward=requires_grad
                and (o._base is None or grad_fn is not None)
                and not replay_from_detached_base,
                is_conj=isinstance(o, Tensor) and o.is_conj(),
                is_neg=isinstance(o, Tensor) and o.is_neg(),
                is_view=isinstance(o, Tensor) and o._is_view(),
                is_multi_output_view=id(o) in multi_output_view_tensor_ids,
                view_replay_grad_enabled=view_replay_grad_enabled,
                multi_output_view_base_idx=multi_output_view_base_idx,
                replay_from_detached_base=replay_from_detached_base,
                replay_detached_view_meta_sequence=(replay_detached_view_meta_sequence),
                multi_output_view_was_invalidated=multi_output_view_was_invalidated,
                multi_output_view_invalidating_input_indices=(
                    multi_output_view_invalidating_input_indices
                ),
                view_meta_sequence=view_meta_sequence,
                multi_output_view_group=multi_output_view_group,
                multi_output_view_index=multi_output_view_index,
            )
            output_info.append(out_info)
            output_grad_fns.append(grad_fn)

        for grad_root_node, alias_idx in detached_output_roots.values():
            for output_idx, (info, grad_fn) in enumerate(
                zip(output_info, output_grad_fns, strict=True)
            ):
                if (
                    output_idx == alias_idx
                    or info.replay_from_detached_base
                    or grad_fn is None
                ):
                    continue
                if _autograd_graph_reaches_node(grad_fn, grad_root_node):
                    raise AssertionError(
                        "aot_autograd() does not yet handle returned "
                        "differentiable outputs that share a detached "
                        "multi-output-view lineage"
                    )

        # See Note [AOT Autograd: Views to avoid tangents aliasing inputs]
        def view_avoid_dupes_with_primals(t: object) -> object:
            if isinstance(t, Tensor) and is_traceable_wrapper_subclass(t):
                return transform_subclass(
                    t, lambda _, inner_t: view_avoid_dupes_with_primals(inner_t)
                )
            if isinstance(t, Tensor):
                return t.view(t.shape)
            return t

        # This analysis function returns *only* the outputs that are meant to be tangents to the backwards.
        # Anything that aliases (inputs returned in the fw due to metadata mutations, or outputs that alias inputs/intermediates)
        # are *regenerated* later, and not used directly in the autograd graph
        def _plain_fake_tensor_like_subclass(x: Any) -> torch.Tensor:
            # pyrefly: ignore [bad-context-manager]
            with detect_fake_mode():
                return torch.empty(
                    x.shape, dtype=x.dtype, device=x.device, layout=x.layout
                )

        def _is_subclass_mutated_input_tangent_always_subclass(inp: object) -> bool:
            return (
                isinstance(inp, torch.nested._internal.nested_tensor.NestedTensor)
                or torch._functorch.config.disable_guess_zero_tangent_for_mutated_input_subclass
            )

        f_input_tangents_pairs = [
            # Note: [AOTAutograd Tangent Subclassness for mutated inputs]
            # Generally when creating tangents to trace with, we assume that tangents will have
            # the same subclass-ness as their forward outs
            # however: for tangents that correspond to input mutations, in practice it is more likely
            # that these tangents will be plain tensors of zeros at runtime, so we tweak our guess
            # to assume that these tangents should always be plaint tensors.
            # Example:
            #  def f(x):
            #      x.mul_(2)
            #      return x + 1
            #  out = f(x)
            #  out.sum().backward()
            # In the above code, we will have a tangent "x_updated_tangent",
            # which will be a plain tensor of zeros, *unless* x is used in some compute after executing f
            #
            # However, there are exceptions to this logic. If a view is created from mutated input and is used in backward,
            # The tangent for this subclass input will be a subclass tensor.
            # Example:
            #  def f(a, b):
            #      a.mul_(2)
            #      b.mul_(3)
            #      return b.view(b.shape), a + b
            # a_out, b_out = f(..., Subclass)
            # (a * b).sum().backward()
            #
            # We can not deduce it easily now, so introducing a debug config to be able to turn off this for specific cases.
            # NJT guarantees to have its tangent as NJT, because it has dedicated integration in Autograd
            # See torch/csrc/autograd/python_function.cpp, use_zeros_like.
            (
                (
                    _plain_fake_tensor_like_subclass(inp)
                    if is_traceable_wrapper_subclass(inp)
                    and not _is_subclass_mutated_input_tangent_always_subclass(inp)
                    else inp
                ),
                TangentAOTInput(InputMutationAOTOutput(inp_desc)),
            )
            for inp, inp_desc, info in zip(flat_f_args, flat_f_args_descs, input_info)
            if info.mutation_type == MutationType.MUTATED_OUT_GRAPH
            and info.mutates_data
            and info.requires_grad
        ]
        f_input_tangents, f_input_tangents_descs = (
            [x[0] for x in f_input_tangents_pairs],
            [x[1] for x in f_input_tangents_pairs],
        )

        f_output_tangents_pairs = [
            (o, TangentAOTInput(desc))
            for o, info, desc in zip(flat_f_outs, output_info, flat_f_outs_descs)
            if info.output_type
            in [
                OutputType.non_alias,
                OutputType.unsafe_view_alias,
                OutputType.custom_function_view,
            ]
            and issubclass(info.raw_type, torch.Tensor)
            and info.requires_grad_for_backward
        ]
        f_output_tangents, f_output_tangents_descs = (
            [x[0] for x in f_output_tangents_pairs],
            [x[1] for x in f_output_tangents_pairs],
        )

        # intermediate bases are also included in the backward graph
        f_tangents = f_input_tangents + f_output_tangents + intermediate_bases
        f_tangents_descs = (
            f_input_tangents_descs + f_output_tangents_descs + intermediate_bases_descs
        )

        # TODO: I'm pretty sure you don't need a tree_map here
        traced_tangents = pytree.tree_map(from_fun, f_tangents)
        traced_tangents = pytree.tree_map(
            view_avoid_dupes_with_primals, traced_tangents
        )
        traced_tangents = [
            coerce_tangent_and_suggest_memory_format(tt)[0]
            for i, tt in enumerate(traced_tangents)
        ]
        # NB: update this if the maps above ever change structure.
        # Also, it might be helpful to add coercion information to the tangent desc!
        traced_tangents_descs = f_tangents_descs

        nonlocal static_input_indices
        static_input_indices = static_input_indices or []
        if torch._dynamo.compiled_autograd.in_compiled_autograd_region:
            passed_indices = set(static_input_indices)
            static_input_indices = [
                i
                for i, arg in enumerate(flat_args)
                if (isinstance(arg, torch.nn.Parameter) or i in passed_indices)
            ]

        static_input_logger.debug(
            "static input indices metadata analysis: %s", static_input_indices
        )

        f_mutated_inputs = [
            inp
            for inp, info in zip(flat_f_args, input_info)
            if info.mutation_type == MutationType.MUTATED_OUT_GRAPH
        ]
        # Build the full list of forward graph outputs so the subclass wrapping
        # code knows exactly which graph outputs to wrap back into subclasses.
        # Including intermediate_bases unconditionally is safe: they are only
        # populated when outputs require grad (line ~539), so they are naturally
        # empty during pure inference.  In the "downgrade from training to
        # inference" path, num_intermediate_bases > 0 is already gated behind
        # `assert not req_subclass_dispatch` (aot_autograd.py), so the subclass
        # wrapping code that consumes subclass_fw_graph_out_meta never sees them.
        f_fw_graph_outs = [*f_mutated_inputs, *flat_f_outs, *intermediate_bases]
        fw_graph_outs = pytree.tree_map(from_fun, f_fw_graph_outs)

        grad_enabled_mutation = None
        if torch.is_grad_enabled() != prior_grad_enabled:
            grad_enabled_mutation = torch.is_grad_enabled()
            torch.set_grad_enabled(
                prior_grad_enabled
            )  # Restore the prior state after tracing it
            log.debug(
                (
                    "grad_mode mutation encountered in graph. "
                    "Will emit mutation epilogue, to set grad_mode=%s"
                ),
                grad_enabled_mutation,
            )

        subclass_inp_meta = create_subclass_meta(flat_args)
        subclass_fw_graph_out_meta = create_subclass_meta(fw_graph_outs)
        subclass_tangent_meta = create_subclass_meta(
            traced_tangents, count_symints=False, with_memory_format=True
        )

        metadata = ViewAndMutationMeta(
            input_info=input_info,
            output_info=output_info,
            num_intermediate_bases=len(intermediate_bases),
            keep_input_mutations=keep_input_mutations,
            traced_tangents=traced_tangents,
            traced_tangents_descs=traced_tangents_descs,
            subclass_inp_meta=subclass_inp_meta,
            subclass_fw_graph_out_meta=subclass_fw_graph_out_meta,
            subclass_tangent_meta=subclass_tangent_meta,
            multi_output_view_creation_error_conditions=(
                tuple(
                    (base_idx, tuple(sorted(indices)))
                    for base_idx, indices in mutation_order_mode.creation_error_conditions.items()
                )
                if isinstance(mutation_order_mode, _InputVersionSnapshotMode)
                else ()
            ),
            grad_enabled_mutation=grad_enabled_mutation,
            static_input_indices=static_input_indices,
            tokens=mode._tokens,
        )
        return metadata

    return inner
