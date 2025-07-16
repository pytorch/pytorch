# mypy: allow-untyped-defs
"""
The various dataclasses, Enums, namedtuples etc used in AOTAutograd. This includes
input/output types, metadata, config, function signatures etc.
"""

import collections
import functools
import itertools
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, NewType, Optional, Union

import torch
import torch.utils._pytree as pytree
from torch._guards import Source
from torch._ops import OpOverload
from torch._subclasses import FakeTensor
from torch._subclasses.fake_tensor import is_fake
from torch.utils._python_dispatch import is_traceable_wrapper_subclass

from .. import config
from .functional_utils import _check_if_mutation_can_be_in_graph, ViewMetaSequence
from .utils import strict_zip


zip = strict_zip


OutputType = Enum(
    "OutputType",
    (
        # output is not an alias
        "non_alias",
        # output aliases an input
        "alias_of_input",
        # output **is** an input tensor
        "is_input",
        # output has a ._base tensor, which is a graph intermediate.
        # We need to return its ._base as a graph output,
        # so its requires_grad info is populated correctly.
        # Instructs the runtime code to regenerate the current output
        # from a base tensor, graph_intermediates[base_idx]
        "alias_of_intermediate_save_as_output",
        # Same as above; but we don't need to explicitly add its ._base
        # as a graph output, because it already **is** a graph output.
        "alias_of_intermediate",
        # Same as above; but the output's ._base is **already** a user output.
        # Instructs the runtime code to regenerate the current output from
        # a base tensor, user_outputs[base_idx]
        "alias_of_intermediate_base_is_user_output",
        # See Note [Intermediate Bases Optimization]
        "unsafe_view_alias",
        # output is an alias, but has a custom autograd.Function backward.
        # In this case, we don't want to do view-replay, since we won't be able to replay the custom function.
        # Instead, we'll treat this output "normally", and trace its backward into the graph.
        "custom_function_view",
    ),
)


# This class stores info about every user output.
@dataclass(frozen=True)
class OutputAliasInfo:
    # Tells us if this output is:
    # (1) a regular (non-aliased) output
    # (2) an alias of a forward input
    # (3) **is** a forward input (special case of "alias_of_input")
    # (4) an alias of an intermediate (aka an alias of an output of the inner traced forward)
    # (5) an alias of an intermediate, that explicitly requires returning the intermediate
    #     as a graph output
    # (6) an alias of an intermediate, where that intermediate is also a user output
    output_type: OutputType
    # The raw type of the output (torch.Tensor, SymInt, etc)
    raw_type: type
    # If (1) above, then
    # - base_idx is None
    # If (2) or (3) above, then
    # - Tells us that the base of this alias is user_fwd_input[base_idx]
    #   (This is an index into the inputs *before* we make synthetic bases)
    # If (4) or (5) above, then
    # - Tells us that the base of this alias is output_graph_intermediates[base_idx]
    #   here, this refers to the index of the *direct* traced
    # If (6) above, then:
    # - Tells us that the base of this alias is output_user_fwds[base_idx]
    #   here, this refers to the index of the *direct* traced
    base_idx: Optional[int]
    # If it is a Tensor, what the dynamic dims are (otherwise is None)
    dynamic_dims: Optional[set[int]]
    # requires_grad
    requires_grad: bool
    # Sequence of ViewMeta objects.
    #
    # Provides us the means to re-run view functions on other tensors.
    #
    # We need to wrap the actual list of ViewMeta with this class so that
    # we compare the ViewMeta elements appropriately, i.e. their type and
    # the elements returned by the `as_tuple()` call.
    view_meta_sequence: Optional[ViewMetaSequence] = None


class MutationType(Enum):
    NOT_MUTATED = 1
    MUTATED_IN_GRAPH = 2
    MUTATED_OUT_GRAPH = 3


# This class tells us info about user inputs.
@dataclass(frozen=True)
class InputAliasInfo:
    is_leaf: bool
    mutates_data: bool
    mutates_metadata: bool
    mutations_hidden_from_autograd: bool
    mutations_under_no_grad_or_inference_mode: bool
    mutation_inductor_storage_resize: bool
    mutates_storage_metadata: bool
    requires_grad: bool
    keep_input_mutations: bool

    def __post_init__(self):
        if self.mutates_storage_metadata:
            # For convenience, we guarantee that this is always true.
            # In practice, If we call .set_(), then at runtime there is no need
            # to additionally fix  up the tensor metadata, since our runtime
            # call to inp.set_(updated_inp) will already have the right metadata
            assert self.mutates_metadata

    @functools.cached_property
    def mutation_type(self) -> MutationType:
        if (
            (not self.mutates_data)
            and (not self.mutates_metadata)
            and not (self.mutation_inductor_storage_resize)
        ):
            return MutationType.NOT_MUTATED

        if _check_if_mutation_can_be_in_graph(
            self.keep_input_mutations,
            self.mutates_data,
            self.mutates_metadata,
            self.mutations_hidden_from_autograd,
            self.mutations_under_no_grad_or_inference_mode,
            self.mutates_storage_metadata,
            self.mutation_inductor_storage_resize,
            self.requires_grad,
        ):
            return MutationType.MUTATED_IN_GRAPH

        return MutationType.MUTATED_OUT_GRAPH


@dataclass
class MemoryFormatMeta:
    # For static shapes we assume tangents have the same strideness as outputs
    size: Optional[Sequence[int]] = None
    stride: Optional[Sequence[int]] = None

    # For dynamic shapes we assume the same memory format: contiguous, channels_last etc.
    memory_format: Optional[torch.memory_format] = None

    @staticmethod
    def from_tensor(t: torch.Tensor) -> Optional["MemoryFormatMeta"]:
        # We only memorize expected memory format for
        # 1. Traceable wrapper subclasses
        # We can not create restrided subclass tensor, as torch.empty_strided works only with dense tensors.
        # 2. Dynamic shape tensors
        # Support for symbolic shapes is not implemented yet.
        use_memory_format: bool = (
            not torch._functorch.config.guess_tangent_strides_as_outputs
            or is_traceable_wrapper_subclass(t)
        )
        if not use_memory_format:
            is_static_shape = True
            for s in itertools.chain(t.shape, t.stride()):
                if not isinstance(s, int):
                    is_static_shape = False
                    break

            use_memory_format = not is_static_shape

        if use_memory_format:
            return MemoryFormatMeta(
                memory_format=torch._prims_common.suggest_memory_format(t),
            )

        return MemoryFormatMeta(
            size=t.size(),
            stride=t.stride(),
        )


@dataclass
class PlainTensorMeta:
    unwrapped_idx: int
    memory_format: Optional[MemoryFormatMeta] = None


@dataclass
class SubclassCreationMeta:
    """
    Used for AOTDispatch.
    This dataclass gives us the information we need to reconstruct a tensor subclass
    from our flat inputs.
    Why is this important? The graph that we'd like to trace out contains flat tensor inputs,
    But the user's original model may have subclass inputs and outputs.
    So we need to wrap/unwrap subclasses as necessary to translate between the user's
    view (subclass inps/outs), and the backend compiler's view (graph with no subclass args).

    Complications arise mostly from the fact that a subclass can hold more than one inner tensor;
    So for a given subclass input/output, we need to carefully track which indices map
    to the subclass tensor in the corresponding "dense-tensor-only" graph.
    """

    # In the inner graph that only takes in dense tensor inputs,
    # this maps to the first index of "tensors that should go in this subclass wrapper"
    flat_tensor_start_idx: int
    # arg_count is inclusive of the arg_counts of any
    # inner tensor subclasses: If I have a TwoTensor and
    # both of its inner elements are TwoTensors, then the
    # arg_count of the outer-most subclass will be 4
    arg_count: int
    # Mark where or not symints were included. This flag is only used in one assertion
    # in "wrap_tensor_subclasses"
    included_subclass_symints: bool
    # meta and attrs are produced by the subclass's __tensor_flatten__.
    # We need to keep them around along with outer_size / outer_stride to plumb them
    # into __tensor_unflatten__
    attrs: dict[str, Union["SubclassCreationMeta", PlainTensorMeta]]
    outer_size: Iterable[Union[None, int, torch.SymInt]]
    outer_stride: Iterable[Union[None, int, torch.SymInt]]
    meta: Any
    # Stores the original subclass itself.
    # This is needed because we need the autograd metadata on the original subclass
    # (this is guaranteed to be a wrapper subclass that holds a fake tensor,
    #  so holding onto this at runtime shouldn't leak memory)
    # This field is nulled out after calling make_runtime_safe()
    original_subclass: Optional[torch.Tensor]

    # Used at runtime to determine the subclass type, so we don't need to save the original subclass
    original_subclass_type: Optional[type] = None
    memory_format: Optional[MemoryFormatMeta] = None

    def compute_outer_size_and_stride(
        self,
        all_args,
        *,
        curr_start_idx: int,
    ):
        from .subclass_utils import compute_symint_placeholders

        def compute(outer, start_idx):
            placeholders = compute_symint_placeholders(outer)
            has_symbolic = any(placeholders)

            if has_symbolic:
                start = curr_start_idx
                end = start_idx + sum(placeholders)
                it_args = iter(all_args[start:end])
                it_placeholders = iter(placeholders)
                return pytree.tree_map_only(
                    lambda _: next(it_placeholders), lambda _: next(it_args), outer
                ), start + len(placeholders)
            else:
                return outer, start_idx

        outer_size, next_idx = compute(self.outer_size, curr_start_idx)
        outer_stride, _ = compute(self.outer_stride, next_idx)
        return outer_size, outer_stride

    def creation_fn(
        self,
        all_args,
        *,
        is_runtime: bool,
    ):
        inner_tensors = {}

        curr_start_idx = self.flat_tensor_start_idx
        for attr, creation_meta in self.attrs.items():
            if isinstance(creation_meta, PlainTensorMeta):
                subclass = all_args[curr_start_idx]
                curr_start_idx += 1
            else:
                subclass = creation_meta.creation_fn(
                    all_args,
                    is_runtime=is_runtime,
                )
                curr_start_idx += creation_meta.arg_count
            inner_tensors[attr] = subclass

        if is_runtime:
            assert self.original_subclass_type is not None
            original_subclass_type = self.original_subclass_type
        else:
            original_subclass_type = type(self.original_subclass)

        if is_runtime:
            outer_size, outer_stride = self.compute_outer_size_and_stride(
                all_args,
                curr_start_idx=curr_start_idx,
            )
        else:
            outer_size, outer_stride = self.outer_size, self.outer_stride

        rebuilt = original_subclass_type.__tensor_unflatten__(  # type: ignore[attr-defined]
            inner_tensors, self.meta, outer_size, outer_stride
        )

        if not is_runtime:
            # After wrapping up the inner dense tensors into a subclass, we need to make sure that our new wrapper
            # has correct autograd metadata, since we'll be tracing through the autograd engine with the subclass.
            # We don't trace through the autograd engine at runtime though, so no need
            # to compute this extra metadata then!
            torch._mirror_autograd_meta_to(self.original_subclass, rebuilt)  # type: ignore[attr-defined]

        return rebuilt

    def make_runtime_safe(self):
        def _make_size_runtime_safe(x: Union[None, int, torch.SymInt]) -> Optional[int]:
            dummy = -1
            if isinstance(x, torch.SymInt):
                # Replace nested ints by a dummy value (-1) as NJT ignores
                # the outer_size/outer_stride at runtime.
                return dummy if x.node.is_nested_int() else None
            return x

        assert self.original_subclass is not None
        self.original_subclass_type = type(self.original_subclass)
        self.original_subclass = None

        # Note: NJT outer_size in AOTDispatcher
        # `_make_size_runtime_safe` replaces any nested int with a dummy value (-1)
        # to prevent serializing a SymInt at runtime. Internally, nested tensor __tensor_unflatten__
        # is designed to safely ignore this dummy value.
        # For more details, see: https://github.com/pytorch/pytorch/blob/5141ade8e30c64e873e14dcc8de233da45d15025/torch/nested/_internal/nested_tensor.py#L266-L299  # noqa: B950
        self.outer_size = tuple(map(_make_size_runtime_safe, self.outer_size))
        self.outer_stride = tuple(map(_make_size_runtime_safe, self.outer_stride))

        # Recurse on nested subclass info
        for creation_meta in self.attrs.values():
            if isinstance(creation_meta, SubclassCreationMeta):
                creation_meta.make_runtime_safe()

    def __post_init__(self):
        # sanity assert to make sure we don't leak memory
        assert is_fake(self.original_subclass)


# This class encapsulates all aliasing + mutation info we need about the forward graph
# See a more detailed overview of the edge case handling at
# https://docs.google.com/document/d/19UoIh_SVrMy_b2Sx5ZaeOJttm6P0Qmyss2rdBuyfoic/edit
# NOTE: This class is saved in AOTAutogradCache, If you are adding elements, make sure
# they are covered by warm cache tests.
@dataclass(eq=False)
class ViewAndMutationMeta:
    # length = # user inputs
    # This gives us info about every input, and what sort of mutation happened to it (if any)
    input_info: list[InputAliasInfo]

    # length = # user outputs
    # This gives us info about every output (mostly around whether it aliases other tensors)
    output_info: list[OutputAliasInfo]

    # length = the number of intermediate bases appended as outputs to the end of the forward graph.
    # Note: this is not necessarily the same thing as:
    #   len([x for x in output_info if x.output_type == OutputType.alias_of_intermediate])
    # Because outputs might share a ._base, or an output's ._base might itself be
    # another user output (in both cases, we won't redundantly append bases to the end of the graph)
    num_intermediate_bases: int

    # For inference only: instructs us to keep data-only input mutations directly in the graph
    keep_input_mutations: bool

    # length = (# inputs w data mutations) + (# user outputs that are non_aliasing tensors)
    #        + (# intermediate bases)
    # These are the FakeTensor (or potential SymInt) outputs that we traced from our
    # metadata pass of the user's forward function.
    # Their only use today is to pass them as a best-guess for tangents when tracing the joint.
    # Stashing them as part of our "metadata" makes it simpler if we want to run our analysis
    # pass once, and reuse the output throughout AOTAutograd
    traced_tangents: list[Any]

    # Each of these is a list telling us about subclasses for the inputs/outputs/grad_outs
    # They are used throughout AOTDispatch to tell us how to generate a list of subclass tensors,
    # Given a (potentially larger) list of plain torch tensors.

    # Taking subclass_inp_meta as an example:
    #   subclass_inp_meta[i] = j (an int) tells us:
    #     "The i'th user input is not a subclass, and corresponds to inputs[j] of the plain-tensor graph."
    #   subclass_inp_meta[i] = SubclassCreationMeta(flat_tensor_start_idx=3, arg_count=2)
    #     "The i'th user input is subclass holding two inner tensors, which are
    #      inputs[3] and inputs[4] of the plain-tensor graph".

    # length = # user inputs
    subclass_inp_meta: list[Union[PlainTensorMeta, SubclassCreationMeta]]
    # So, the full set of outputs to the forward graph looks something like:
    # (*mutated_inps, *user_outs, *intermediate_bases, *saved_for_bw_tensors)
    # where the first 3 of those 4 can be subclasses
    # (but not saved_for_bw tensors, since these are internal to the compiler
    # and not user visible, so there's no point in wrapping/unwrapping them at runtime).
    # This list contains subclass information on all of the fw graph outputs
    # except for saved_for_bw_tensors.
    subclass_fw_graph_out_meta: list[Union[PlainTensorMeta, SubclassCreationMeta]]
    # length = # backward graph inputs
    subclass_tangent_meta: list[Union[PlainTensorMeta, SubclassCreationMeta]]
    # TODO: we should kill this
    # (need to default it to not break internal)
    is_train: bool = False

    # length = (# inputs w data mutations) + (# user outputs that are non_aliasing tensors)
    #        + (# intermediate bases)
    # At runtime, we don't keep the traced_tangents around since they're not serializable.
    # Instead, we keep any necessary subclass metadata necessary about each traced_tangent.
    # This list is generated after calling make_runtime_safe().
    traced_tangent_metas: Optional[list[Any]] = None

    num_symints_saved_for_bw: Optional[int] = None

    # The grad_enabled mutation that will be emitted in the runtime_wrapper epilogue
    # NOTE: AOTAutograd will assume that the ambient `is_grad_enabled` is the grad mode
    # that is intended to be in effect prior to running the graph, in keeping with
    # equivalence to eager mode. It is the responsibility of upstream graph acquisition
    # to reset the grad mode to its pre-graph value prior to calling aot_autograd.
    grad_enabled_mutation: Optional[bool] = None

    # Keeps track of whether `torch.use_deterministic_algorithms` was turned on
    # when the forward was run. If deterministic mode was turned off during the
    # forward, but is turned on during the backward call, then an error is
    # raised
    deterministic: Optional[bool] = None

    # Keeps track of which input indices store parameters (which we will treat as static)
    static_input_indices: list[int] = field(default_factory=list)

    # Map of effect type (ex. _EffectType.ORDERED) to token.  If there are
    # side-effectful operators, FunctionalTensorMode will populate this
    # dictionary telling us how many tokens we will need during tracing.
    tokens: dict[Any, torch.Tensor] = field(default_factory=dict)

    # Only filled in if/when we trace the joint function
    # If an input requires grad and is mutated in the backward, it is only safe to keep the mutation
    # in the graph if gradients are disabled while the backward runs
    # (grad mode is disabled by default when users run the backward, but can be turned on with create_graph=True)
    # At runtime during the backward, we use this list of indices to error properly if we find out
    # that it was not safe to include a backward mutation in the graph.
    indices_of_inputs_that_requires_grad_with_mutations_in_bw: list[int] = field(
        default_factory=list
    )

    # Indexes of saved tensors which are donated buffer.
    # Donated buffer means the tensor is not alias of any forward user input, forward user output,
    # and backward output.
    bw_donated_idxs: Optional[list[int]] = None

    # Number of tokens used in backward, appended at the end of backward outputs.
    # Filled after tracing joint function.
    num_backward_tokens: int = 0

    # Number of rng states that will get thread into the forward and backward for
    # cudagraph compatible run_and_save_rng
    num_graphsafe_rng_states: int = 0

    graphsafe_rng_state_index: Optional[int] = None

    def __post_init__(self):
        # pre-compute the indices of the inputs that are mutated.
        # When keep_input_mutations is set, we don't need to worry about our epilogue
        # handling data-only mutations, because we keep them directly in the graph.
        mutated_inp_runtime_indices = [
            i
            for i, m in enumerate(self.input_info)
            if (m.mutation_type == MutationType.MUTATED_OUT_GRAPH)
        ]

        mutated_graph_handled_indices = [
            i
            for i, m in enumerate(self.input_info)
            if m.mutation_type == MutationType.MUTATED_IN_GRAPH
        ]
        self.mutated_graph_handled_indices = mutated_graph_handled_indices
        self.num_mutated_graph_handled_indices = len(self.mutated_graph_handled_indices)

        mutated_graph_handled_indices_seen_by_autograd = [
            i
            for i in mutated_graph_handled_indices
            if not self.input_info[i].mutations_hidden_from_autograd
        ]

        self.mutated_graph_handled_indices_seen_by_autograd = (
            mutated_graph_handled_indices_seen_by_autograd
        )
        self.num_mutated_graph_handled_indices_seen_by_autograd = len(
            self.mutated_graph_handled_indices_seen_by_autograd
        )

        aliased_out_indices = [
            i
            for i, m in enumerate(self.output_info)
            if m.output_type
            not in [
                OutputType.non_alias,
                OutputType.unsafe_view_alias,
                OutputType.custom_function_view,
            ]
        ]
        unsafe_view_out_indices = [
            i
            for i, m in enumerate(self.output_info)
            if m.output_type is OutputType.unsafe_view_alias
        ]

        # This is pre-computed in post_init for perf.
        # It contains the index of every element
        # of input_info that corresponds to a mutation (data or metadata or both)
        self.mutated_inp_runtime_indices = mutated_inp_runtime_indices
        self.num_mutated_inp_runtime_indices = len(self.mutated_inp_runtime_indices)

        # This is pre-computed for perf.
        # It contains the index of every element
        # of output_info that corresponds to an alias (either of an input or intermediate)
        self.aliased_out_indices = aliased_out_indices
        self.unsafe_view_out_indices = unsafe_view_out_indices
        self.num_outputs = len(self.output_info)
        self.num_outputs_non_aliased = len(
            [
                x
                for x in self.output_info
                if x.output_type
                in [
                    OutputType.non_alias,
                    OutputType.unsafe_view_alias,
                    OutputType.custom_function_view,
                ]
            ]
        )
        self.num_outputs_aliased_to_inputs = len(
            [
                x
                for x in self.output_info
                if x.output_type
                in [
                    OutputType.alias_of_input,
                    OutputType.is_input,
                ]
            ]
        )
        self.num_unsafe_view_outputs = len(self.unsafe_view_out_indices)
        self.num_outputs_aliased_to_intermediates = len(
            [
                x
                for x in self.output_info
                if x.output_type
                in [
                    OutputType.alias_of_intermediate,
                    OutputType.alias_of_intermediate_save_as_output,
                    OutputType.alias_of_intermediate_base_is_user_output,
                ]
            ]
        )
        self.num_outputs_aliased = (
            self.num_outputs_aliased_to_inputs
            + self.num_outputs_aliased_to_intermediates
        )

        # Record dynamic outputs of the Dynamo traced forward graph
        # Mark them as dynamic at the end of the runtime wrapper
        self.dynamic_outputs = any(o.dynamic_dims for o in self.output_info)

        # Record the indices of dynamic outputs in the partitioned forward graph
        # Mark them as dynamic in the runtime wrapper
        # activation index -> dynamic dims indices
        self.dynamic_saved_tensors_idxs: dict[int, set[int]] = {}

        # See Note: [AOTAutograd Backward Guards]
        # This is pre-computed for fast asserts on the types of our grad_outputs in the backward.
        # Eventually, we should kill this and replace with real backward guards.
        # (we want to precompute the "runtime" types, so replace FakeTensor with torch.Tensor)
        self.output_types = [
            torch.Tensor if isinstance(x, FakeTensor) else type(x)
            for x in self.traced_tangents
        ]

        self.is_rng_op_functionalized = config.functionalize_rng_ops
        # All of the above metadata is collected by tracing the fw function.
        # However, extra outputs for rng offsets behave differently. Both fwd
        # and bwd graphs have their own outputs for the total consumed offsets.
        # Unlike mutated inputs, we don't have to worry about sending the right
        # set of tensors between fwd and bwd. Fwd and bwd offsets are
        # independent and simpler to handle. Therefore, we track them
        # separately.
        self.num_outputs_rng_offset = 1 if self.is_rng_op_functionalized else 0

        # Our forward() returns both (tokens, mutated_inputs, outputs, output_intermediate_bases, saved_tensors, saved_symints)
        # Tokens will be split out before mutations/view handling and we do not count them here.
        self.num_forward_returns = (
            self.num_mutated_inp_runtime_indices
            + self.num_outputs
            + self.num_intermediate_bases
        )
        # In case of functionalization of rng ops, the fw_module returns one
        # additional output for rng offset. This rng offset is used right
        # away to advance the rng state, and is not passed on to the raw
        # outputs. However, we need to know the exact boundary to identify
        # which tensors to be saved for the bwd graph.  num_forward captures
        # this information.
        self.num_forward = self.num_forward_returns + self.num_outputs_rng_offset

    def make_runtime_safe(self):
        """
        There are various fields in ViewAndMutationMeta that aren't serializable. This function is called after all tracing
        is completed to simplify certain fields in the metadata so that they can be safely cached.

        Doing so may lose information (in the case of traced_tangents), but none of the information is needed at runtime.
        """
        # TODO: This function is only a best effort: there are other fields that may not be cache safe
        # (i.e., there's no guarantee that tensor_flatten() returns a serializable result), or that
        # SubclassCreationMeta is cache safe.
        assert self.traced_tangent_metas is None

        def extract_metadata(t):
            if isinstance(t, torch.Tensor) and is_traceable_wrapper_subclass(t):
                (inner_tensors, flatten_spec) = t.__tensor_flatten__()  # type: ignore[attr-defined]
                # Technically, we only need the flatten_spec, not the inner tensors.
                # However, some Tensor subclasses (like TwoTensor) may have flatten_spec = None.
                # And we want to be able to assert that this metadata is non-None,
                # to distinguish between "this was a tensor subclass with no metadata" vs.
                # "this wasn't a tensor subclass at all".
                return (inner_tensors, flatten_spec)
            else:
                return None

        self.traced_tangent_metas = [extract_metadata(t) for t in self.traced_tangents]
        # Clear traced tangents at runtime
        self.traced_tangents = []
        for inp_meta in self.subclass_inp_meta:
            if isinstance(inp_meta, SubclassCreationMeta):
                inp_meta.make_runtime_safe()
        for inp_meta in self.subclass_fw_graph_out_meta:
            if isinstance(inp_meta, SubclassCreationMeta):
                inp_meta.make_runtime_safe()
        for inp_meta in self.subclass_tangent_meta:
            if isinstance(inp_meta, SubclassCreationMeta):
                inp_meta.make_runtime_safe()

    @property
    def tensors_saved_for_backwards_slice(self):
        assert self.num_symints_saved_for_bw is not None
        if self.num_symints_saved_for_bw > 0:
            return slice(self.num_forward, -self.num_symints_saved_for_bw)
        else:
            return slice(self.num_forward, None)

    @property
    def symints_saved_for_backwards_slice(self):
        assert self.num_symints_saved_for_bw is not None
        if self.num_symints_saved_for_bw > 0:
            return slice(-self.num_symints_saved_for_bw, None)
        else:
            return slice(0, 0)  # empty slice

    def __eq__(self, other):
        if not isinstance(other, ViewAndMutationMeta):
            return NotImplemented
        return (
            self.input_info == other.input_info
            and self.output_info == other.output_info
            and self.num_intermediate_bases == other.num_intermediate_bases
            and self.keep_input_mutations == other.keep_input_mutations
            and self.is_rng_op_functionalized == other.is_rng_op_functionalized
            and self.num_outputs_rng_offset == other.num_outputs_rng_offset
            and len(self.traced_tangents) == len(other.traced_tangents)
            and all(
                x.shape == y.shape and x.dtype == y.dtype
                for x, y in zip(self.traced_tangents, other.traced_tangents)
            )
            and self.num_backward_tokens == other.num_backward_tokens
        )


@dataclass(eq=False)
class SubclassMeta:
    # A copy of all forward metadata, but computed on the *dense* tensor forward (after desugaring subclasses)
    # So for example, if the user had a model containing two `TwoTensor` inputs,
    # Then `SubclassMeta.fw_metadata.input_infos` would have length 4 here.
    fw_metadata: ViewAndMutationMeta

    # Note: [Computing Subclass Metadata about grad_inputs]
    # Given a list of flattened, plain tensor grad_inputs, this tells us how to reconstruct the grad_input subclasses
    #
    # You might think: why not just assume that all grad_inputs will have the same subclass-ness as the original inputs?
    # (AOTAutograd generally assumes other properties, e.g. that grad_outputs are contiguous)
    #
    # This doesn't really work though. take this example:
    #
    # def f(DoubleTensor, DenseTensor):
    #     return DoubleTensor  * DenseTensor
    #
    # In the above example, the .grad field of *both* DoubleTensor and DenseTensor will be a DoubleTensor.
    # When we trace out a joint fw-bw graph, we'll end up returning two subclasses for the two grad_inputs.
    # This means that our backward graph will return 4 outputs (two dense tensors for each DoubleTensor grad_input)
    # and we need to properly store the metadata that tells us how to turn these 4 outputs back into DoubleTensors.
    #
    # Note that this info **cannot** easily be figured out from ViewAndMutationMeta.
    # We can only compute this info by tracing the entire joint and examining the grad_inputs that we computed.
    #
    # See Note: [AOTAutograd Backward Guards]
    # This will also eventually require us to install backward guards,
    # in case we made incorrect assumptions about the subclass-ness of our grad_outputs
    #
    # Optional field because we don't compute for inference graphs
    grad_input_metas: Optional[list[Union[PlainTensorMeta, SubclassCreationMeta]]] = (
        None
    )

    def __init__(self) -> None:
        # The fields in this class get set after its construction.
        pass


# This class exists because:
# - the autograd.Function.forward() in aot autograd returns outputs that might alias inputs
# - we only care about the metadata on those aliases, so we can regenerate them.
#   We do not want them to participate in the autograd.Function.
# We do that by wrapping them in an opaque class, so the autograd.Function
# does not know to treat them as tensors.
@dataclass(frozen=True)
class TensorAlias:
    alias: torch.Tensor


@dataclass
class BackwardSignature:
    """
    Provides information about the backward section of an exported
    joint forward-backward graph.
    For a particular fx GraphModule, this class contains information on:
    (1) A mapping from each gradient (backwards output) to the parameter
        it corresponds to (forward input)
    (2) A mapping from each gradient (backwards output) to the user input
        it corresponds to (forward input)
    (3) Which of the forward outputs corresponds to the loss, that we backprop on.

    Each string name is the `node.name` of the corresponding node in the fx graph.
    """

    gradients_to_parameters: dict[str, str]
    gradients_to_user_inputs: dict[str, str]
    loss_output: str


GraphOutputName = NewType("GraphOutputName", str)
GraphInputName = NewType("GraphInputName", str)
FQN = NewType("FQN", str)


@dataclass
class GraphSignature:
    """
    Provides information about an exported module.
    For a particular fx GraphModule, this class contains information on:
    (1) Which graph inputs are parameters, buffers, or user inputs
    (2) (for params/buffers) a mapping from the name of each graph argument
        to its parameter/buffer FQN in the original nn.Module.
    (3) If there are input mutations, these are represented as extra outputs
        in the fx GraphModule. We provide a mapping from these
        extra output names to the names of the actual inputs.
    (4) The pytree metadata on how to flatten/unflatten inputs and outputs.
        The corresponding FX GraphModule only accepts and returns
        pytree-flattened inputs/outputs.
    (5) (Optionally) if the FX is a joint forward-backward graph, we provide
        a signature on the backward section of the joint graph.
    """

    parameters: list[FQN]
    buffers: list[FQN]

    user_inputs: list[GraphInputName]
    user_outputs: list[GraphOutputName]
    inputs_to_parameters: dict[GraphInputName, FQN]
    inputs_to_buffers: dict[GraphInputName, FQN]

    # If the user's module mutates a buffer,
    # it's represented in the graph as an extra graph output.
    # This dict is a mapping from
    # "graph outputs that correspond to updated buffers"
    # to the FQN names of those mutated buffers.
    buffers_to_mutate: dict[GraphOutputName, FQN]
    user_inputs_to_mutate: dict[GraphOutputName, GraphInputName]

    in_spec: pytree.TreeSpec
    out_spec: pytree.TreeSpec

    backward_signature: Optional[BackwardSignature]

    input_tokens: list[GraphInputName]
    output_tokens: list[GraphOutputName]

    @classmethod
    def from_tracing_metadata(
        cls,
        *,
        in_spec: pytree.TreeSpec,
        out_spec: pytree.TreeSpec,
        graph_input_names: list[str],
        graph_output_names: list[str],
        view_mutation_metadata: ViewAndMutationMeta,
        named_parameters: list[str],
        named_buffers: list[str],
        num_user_inputs: int,
        num_user_outputs: int,
        loss_index: Optional[int],
        backward_signature: Optional[BackwardSignature],
    ) -> "GraphSignature":
        graph_inputs = graph_input_names
        graph_outputs = graph_output_names
        parameters = list(named_parameters)
        buffers = list(named_buffers)
        num_tokens = len(view_mutation_metadata.tokens)

        # Calling convention assumptions:
        # (1) graph inputs = (input_tokens, params, buffers, user_inputs)
        # (2) graph outputs = (output_tokens, mutated_inputs, user_outs, param_gradients)
        # (If we are capturing an inference graph, this convention is identical
        #  except that param_gradients is empty)
        # See Note [Side-Effectful Tokens in AOTAutograd] for information on tokens

        # Address input calling conventions:
        start, stop = 0, num_tokens
        input_tokens = graph_inputs[start:stop]

        start, stop = stop, stop + len(parameters)
        inputs_to_parameters = dict(zip(graph_inputs[start:stop], parameters))

        start, stop = stop, stop + len(buffers)
        inputs_to_buffers = dict(
            zip(
                graph_inputs[start:stop],
                buffers,
            )
        )

        start, stop = stop, stop + num_user_inputs
        user_inputs = graph_inputs[start:stop]

        # We should've gone through all the inputs now
        assert len(graph_inputs) - stop == 0

        # Address output calling conventions:
        start, stop = 0, num_tokens
        output_tokens = graph_outputs[start:stop]

        names = [*input_tokens, *parameters, *buffers, *user_inputs]
        mutations = []
        for idx, input_info in enumerate(view_mutation_metadata.input_info):
            if input_info.mutates_data:
                # Only buffers can be mutated, not parameters
                assert idx >= len(parameters)
                mutations.append(names[idx + num_tokens])

        assert len(mutations) == view_mutation_metadata.num_mutated_inp_runtime_indices

        start, stop = (
            stop,
            stop + view_mutation_metadata.num_mutated_inp_runtime_indices,
        )
        outputs_to_mutations = dict(zip(graph_outputs[start:stop], mutations))

        user_inputs_to_mutate = {}
        buffers_to_mutate = {}
        for output_name, mutation_name in outputs_to_mutations.items():
            if mutation_name in user_inputs:
                user_inputs_to_mutate[output_name] = mutation_name
            else:
                assert mutation_name in buffers
                buffers_to_mutate[output_name] = mutation_name

        start, stop = stop, stop + num_user_outputs
        user_outputs = graph_outputs[start:stop]

        unused_outputs = len(graph_outputs) - stop
        if backward_signature is not None:
            unused_outputs -= len(backward_signature.gradients_to_parameters) + len(
                backward_signature.gradients_to_user_inputs
            )
        assert unused_outputs == 0

        return GraphSignature(
            parameters=parameters,  # type: ignore[arg-type]
            buffers=buffers,  # type: ignore[arg-type]
            user_inputs=user_inputs,  # type: ignore[arg-type]
            user_outputs=user_outputs,  # type: ignore[arg-type]
            inputs_to_buffers=inputs_to_buffers,  # type: ignore[arg-type]
            inputs_to_parameters=inputs_to_parameters,  # type: ignore[arg-type]
            user_inputs_to_mutate=user_inputs_to_mutate,
            buffers_to_mutate=buffers_to_mutate,  # type: ignore[arg-type]
            in_spec=in_spec,
            out_spec=out_spec,
            backward_signature=backward_signature,
            input_tokens=input_tokens,  # type: ignore[arg-type]
            output_tokens=output_tokens,  # type: ignore[arg-type]
        )


@dataclass
class AOTAutogradCacheInfo:
    cache_key: str
    start_time_ns: int
    forward_symints: list[torch.SymInt]


@dataclass
class AOTConfig:
    """
    Configuration for AOTDispatcher
    """

    fw_compiler: Callable
    bw_compiler: Callable
    partition_fn: Callable
    decompositions: dict[OpOverload, Callable]
    num_params_buffers: int
    aot_id: int
    keep_inference_input_mutations: bool
    is_export: bool = False
    no_tangents: bool = False
    dynamic_shapes: bool = False
    aot_autograd_arg_pos_to_source: Optional[list[Source]] = None
    static_input_indices: Optional[list[int]] = None
    inference_compiler: Optional[Callable] = None
    enable_log: bool = True
    # this is always false outside of export.
    pre_dispatch: bool = False
    # Key to use for AOTAutogradCache
    cache_info: Optional[AOTAutogradCacheInfo] = None
    # If we should ignore the shape_env in the ambient tracing_context.
    # The net effect is that if dynamic shapes are on, we end up
    # specializing on example_inputs.
    # Used only by standalone_compile.
    ignore_shape_env: bool = False
    precompile_backend_id: Optional[str] = None

    def __post_init__(self):
        if self.pre_dispatch:
            assert self.is_export, "Can only have pre_dispatch IR for export."


SubclassTracingInfo = collections.namedtuple(
    "SubclassTracingInfo",
    ["plain_tensor_trace_fn", "plain_tensor_args", "maybe_subclass_meta"],
)
