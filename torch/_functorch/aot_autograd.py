import collections
import dataclasses
import itertools
import logging
import warnings
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from enum import Enum
from functools import partial, wraps
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from functorch import make_fx

import torch
import torch.fx.traceback as fx_traceback
import torch.nn as nn
import torch.utils._pytree as pytree
import torch.utils.dlpack
from torch import Tensor
from torch._dispatch.python import enable_python_dispatcher
from torch._dynamo.utils import dynamo_timed
from torch._subclasses import CrossRefFakeMode, FakeTensor, FakeTensorMode
from torch.fx import immutable_collections, Interpreter
from torch.fx.experimental.proxy_tensor import is_sym_node, py_sym_types
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from torch.multiprocessing.reductions import StorageWeakRef
from torch.nn.utils import stateless
from . import config
from .partitioners import default_partition
from torch._guards import TracingContext, DuplicateInputs

log = logging.getLogger(__name__)

MutationType = Enum(
    "MutationType", ("none", "metadata_only", "data", "data_and_metadata")
)
OutputType = Enum(
    "OutputType", (
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
    )
)

pytree._register_pytree_node(
    immutable_collections.immutable_list,
    lambda x: (list(x), None),
    lambda x, c: immutable_collections.immutable_list(x),
)
pytree._register_pytree_node(
    immutable_collections.immutable_dict,
    lambda x: (list(x.values()), list(x.keys())),
    lambda x, c: immutable_collections.immutable_dict(
        {key: value for key, value in zip(c, x)}
    ),
)

aten = torch.ops.aten

# This global counter increments every time we compile a graph with
# AOTAutograd.  You can use this to correlate runtime error messages
# with compile time (e.g., if you get an error at runtime saying
# compiled graph 3 failed, you can set a breakpoint at compile time
# for this graph number to investigate further at compile time.)
#
# NB: this is different from get_aot_compilation_context, which tracks
# each underlying graph that is compiled.  In contrast, AOT_COUNTER
# corresponds to top-level invocations of aot_module/aot_function;
# one counter is allocated per entire compiled block (but this block
# may involve compiling multiple subgraphs; e.g., for forwards/backwards)
AOT_COUNTER = itertools.count()

KNOWN_TYPES = tuple(
    [torch.Tensor, int, str, float, bool, type(None)] + list(py_sym_types)
)


@contextmanager
def preserve_rng_state():
    rng_state = torch.clone(torch.random.get_rng_state())
    if torch.cuda.is_available():
        cuda_rng_state = torch.clone(torch.cuda.get_rng_state())
    try:
        yield
    finally:
        torch.random.set_rng_state(rng_state)
        if torch.cuda.is_available():
            torch.cuda.set_rng_state(cuda_rng_state)


# Set up hooks so that during backward the fx's stack_trace is properly set
callback_set = False


def setup_stacktrace_preservation_hooks(roots: List):
    def iter_graph(roots):
        if not roots:
            return
        seen = set()
        q = collections.deque()
        for node in roots:
            if node is not None:
                seen.add(node)
                q.append(node)

        while q:
            node = q.popleft()
            for fn, _idx in node.next_functions:
                if fn in seen or fn is None:
                    continue
                seen.add(fn)
                q.append(fn)

            yield node

    def get_callback(saved_stack_):
        def callback():
            global callback_set
            fx_traceback.set_stack_trace(saved_stack_)
            callback_set = False

        return callback

    def get_prehook(stack_):
        def prehook(grad_output):
            global callback_set

            if not callback_set:
                torch.autograd.variable.Variable._execution_engine.queue_callback(
                    get_callback(fx_traceback.format_stack())
                )
                callback_set = True

            fx_traceback.set_stack_trace(stack_)

        return prehook

    def get_posthook(special_stack_):
        def posthook(grad_input, grad_output):
            fx_traceback.set_stack_trace(special_stack_)

        return posthook

    for node in iter_graph(roots):
        forward_node_stack = node.metadata.get("traceback_", [])
        node.register_prehook(get_prehook(forward_node_stack))

        special_stack = forward_node_stack.copy()
        special_stack.append(
            "Gradient addition node due to multiple use of tensor around:"
        )
        node.register_hook(get_posthook(special_stack))


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# AOT Autograd contains a pretty non-trivial amount of logic to handle edge cases around aliasing and mutation
# that are external to the graph (they show up as side effects in some way when you run the graph).
#
# Take a look at `test_aotdispatch.py TestAOTAutograd.test_input_mutation*` tests for some examples functions
# and what they're compiled graphs looks like.
# Below is a very long comment detailing several edge cases, and showing how AOT Autograd handles them.
#
# Note [AOT Autograd: input data mutations]
#
# If we compile a function that mutates inputs, then those input mutations are real side effects
# that a user expects to see after running the compiled graph.
# However, the graph that we want to send to a backend needs to be *entirely* functional.
# The way we reconcile this difference is that we remove the mutations completely from the graph that we compile
# but we update the graph to return (updated_inputs, user_outputs).
# In the epilogue that runs after the compiled graph is executed, we copy the updated inputs back to the originals.
#
# Example: original user code:
# def f(x):
#     x.mul_(2)
#     out = x.mul(3)
#     return out
#
# After AOT Autograd compiles, we end up with a:
# (a) compiled graph
# (b) autograd.Function.forward() method, that executes the compiled graph
# (c) wrapper function, that calls the autograd.Function.forward() and performs the epilogue
#
# The output of (a, b, c) are all written below.
#
# def compiled_forward_graph(x):
#     x_updated = x.mul(2)
#     out = x_updated.mul(3)
#     return x_updated, out
#
# # x_updated gets a gradient in the compiled backward
# def compiled_backward_graph(grad_x_updated, grad_out):
#     grad_x = ...
#     return grad_x
#
# def autograd.Function.forward(x):
#     x_updated, out = compiled_forward_graph(x)
#     return x_updated, out
#
# def compiled_wrapper(x):
#     x_updated, out = autograd.Function.apply(x)
#     x.copy_(x_updated)
#     return out
#
# Another important thing to note is that updated inputs (due to data mutations) *do* participate
# in the compiled backward graph! Since the compiled forward graph gets N extra outputs
# (due to updated inputs showing up as graph outputs),
# The compiled backward gets an additional N inputs.
# That way, during the x.copy_(x_updated) bit in the epilogue, gradients will flow from the updated input
# back to the original input.


# Note [AOT Autograd: input metadata mutations]
#
# For the same reason as input mutations, we also don't put input metadata mutations in the graph.
# Instead, we return the updated version of the input (a view), and mutate the input's metadata outside of the graph
#
# Example: original user code:
# def f(x):
#     x.t_()
#     out = x.mul(3)
#     return out
#
# AOT Autograd output (compiled graph, autograd.Function.forward(), wrapper function):
# def compiled_forward_graph(x):
#     x_updated = x.t()
#     out = x_updated.mul(3)
#     return x_updated, out
#
# # x_updated does *not* get a gradient in the compiled backward
# def compiled_backward_graph(grad_out):
#     grad_x = ...
#     return grad_x
#
# def autograd.Function.forward(x):
#     x_updated, out = compiled_forward_graph(x)
#     return x_updated, out
#
# def compiled_wrapper(x):
#     x_updated, out = autograd.Function.apply(x)
#     x.as_strided_(x_updated)
#     return out


# Note [AOT Autograd: outputs aliasing inputs or intermediates!]
#
# AOT Autograd needs special handling for outputs that alias graph inputs or intermediates!
# Why?
# (1) autograd.Function.forward() has a limitation, where views that returned in the forward cannot later be mutated.
# (2) views don't need to be compiled in the graph anyway - it's cheap to generate them outside of the compiled graph,
#     in an epilogue.
# For outputs that alias inputs, we do the following:
# (a) *still* return the aliased output as a graph output
# (b) In the AOT Autograd wrapper/epilogue, we don't return that aliased output. Instead, we use it to regenerate the output.
#
# For outputs that alias *intermediates*, we do the following:
# (a) Return the output in the compiled forward, **and** return it's ._base (a graph intermediates) as an output in the forward
# (b) Use (output, graph_intermediate) to regenerate the alias, and return that to the user (instead of the compiled fw output).
# You might wonder why we return the aliased output directly in the graph (and making the graph compute it),
# only to not return it and instead generate a fresh alias off of the intermediate,
# instead of (say) just storing metadata about the size/stride of the output somewhere to generate the alias. There are two reasons:
# (1) Getting the actual alias tensor allows us to use view-replay to generate the alias, instead of an as_strided() call
# (2) Inductor (and other backends) are free to change the memory format of graph outputs, if it results in better performance.
#     This can result in problems if a user later tries to .view() that output expecting it to have one set of strides,
#     when it has a different set of strides.
#     By including the view op directly in the graph, inductor takes that into account when deciding what memory format
#     the graph intermediate should be.
#
# Another important thing to note is how our traced backward() graph handles aliases.
# (this applies to outputs aliasing inputs, outputs aliasing intermediates,
#  *and* updated inputs returned in the compiled forward due to metadata-only mutations).
# Any outputs that alias (either inputs or intermediates) do NOT participate in the compiled backward graph
# It would be wasteful to include them in the compiled backward(), because we regenerate them eagerly
# at the end of the forward.
#
# Example: original user code:
# def f(x):
#     out1 = x.t()
#     intermediate = x.mul(2)
#     out2 = intermediate.view(-1)
#     return out1, out2
#
# AOT Autograd output (compiled graph, autograd.Function.forward(), wrapper function):
# def compiled_forward_graph(x):
#     out1 = x.t()
#     intermediate = x.mul(2)
#     out2 = intermediate.view(-1)
#     # the compiled graph also returns the intermediate
#     return out1, out2, intermediate
#
# # intermediate gets a gradient in the compiled backward.
# # both output aliases (out1 and out2) do not.
# def compiled_backward_graph(grad_intermediate):
#     grad_x = ...
#     return grad_x
#
# def autograd.Function.forward(x):
#     out1, out2, intermediate = compiled_forward_graph(x)
#     return out1, out2, intermediate
#
# def compiled_wrapper(x):
#     out1, out2, intermediate = autograd.Function.apply(x)
#     # regenerate out1 from the input
#     out1_regenerated = out1._view_func(x)
#     # regenerate out1 from the intermediate
#     out2_regenerated = out2._view_func(intermediate)
#     return out1_regenerated, out2_regenerated


# Note [AOT Autograd: mutations to inputs that alias other inputs]
#
# Another edge case that is (only partially) handled today is when an input is mutated, but itself aliases another input.
# AOT Autograd needs to **ensure** that functionalization knows that the two inputs are aliased to each other.
# That way, when the aliased input is accessed later in the graph, functionalization knows to "update" the alias
# given the mutation that occurred.
#
# This is handled by updating the calling convention: we create a "synthetic base" that becomes a new input
# in the compiled function, and we regenerate the original (aliased) inputs directly off of the base
# inside of the compiled function.
#
# See merge_view_inputs() for more detailed info.
#
# Example: original user code:
# def f(x, x_view):
#     x.mul_(2)
#     out = x * x_view
#     return out
# f(x, x.view(-1))
#
# AOT Autograd output (compiled graph, autograd.Function.forward(), wrapper function):
# def compiled_forward_graph(base)
#     x = generate_x(base)
#     x_view = generate_x_view(base)
#     x_updated = x.mul(2)
#     x_view_updated = x_updated.view(-1)
#     out = x_updated * x_view_udpated
#     return x_updated, out
#
# # The calling convention change from (aliases) -> (base) happens
# # *outside* of the autograd.Function.forward().
# # That means the forward() only has 1 input (base),
# # and the backward() only has 1 output (grad_base)
# def compiled_backward_graph(grad_out):
#     grad_base = ...
#     return grad_base
#
# def autograd.Function.forward(base):
#     x_updated, out = compiled_forward_graph(base)
#     return x_updated, out
#
# # The compiled wrapper is where we create synthetic bases.
# # The info on which inputs are mutated is also tracked *before* synthetic base creation.
# def compiled_wrapper(x, x_view):
#     base = merge_view_inputs(x, x_view)
#     x_updated, out = autograd.Function.apply(base)
#     # x and x_view are aliased in eager mode, so this mutation to x will automatically affect x_view.
#     x.copy_(x_updated)
#     return out
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


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


# This class tells us info about user inputs.
@dataclass(frozen=True)
class InputAliasInfo:
    is_leaf: bool
    mutates_data: bool
    mutates_metadata: bool


# This class encapsulates all aliasing + mutation info we need about the forward graph
# See a more detailed overview of the edge case handling at
# https://docs.google.com/document/d/19UoIh_SVrMy_b2Sx5ZaeOJttm6P0Qmyss2rdBuyfoic/edit
@dataclass()
class ViewAndMutationMeta:
    # length = # user inputs
    # This gives us info about every input, and what sort of mutation happened to it (if any)
    input_info: List[InputAliasInfo]

    # length = # user outputs
    # This gives us info about every output (mostly around whether it aliases other tensors)
    output_info: List[OutputAliasInfo]

    # length = # mutated inps + # user outputs
    # For every output *and* mutated input returned from the forward,
    # tells us whether or not the output should require gradients or not
    requires_grad_info: List[bool]

    # length = the number of intermediate bases appended as outputs to the end of the forward graph.
    # Note: this is not necessarily the same thing as:
    #   len([x for x in output_info if x.output_type == OutputType.alias_of_intermediate])
    # Because outputs might share a ._base, or an output's ._base might itself be
    # another user output (in both cases, we won't redundantly append bases to the end of the graph)
    num_intermediate_bases: int

    # For inference only: instructs us to keep data-only input mutations directly in the graph
    keep_input_mutations: int

    def __post_init__(self):
        # pre-compute the indices of the inputs that are mutated.
        # When keep_input_mutations is set, we don't need to worry about our epilogue
        # handling data-only mutations, because we keep them directly in the graph.
        mutated_inp_indices = [
            i for i, m in enumerate(self.input_info) if m.mutates_metadata or (not self.keep_input_mutations and m.mutates_data)
        ]
        aliased_out_indices = [
            i
            for i, m in enumerate(self.output_info)
            if m.output_type != OutputType.non_alias
        ]

        # This is pre-computed in post_init for perf.
        # It contains the index of every element
        # of input_info that corresponds to a mutation (data or metadata or both)
        self.mutated_inp_indices = mutated_inp_indices
        # This is pre-computed for perf.
        # It contains the index of every element
        # of output_info that corresponds to an alias (either of an input or intermediate)
        self.aliased_out_indices = aliased_out_indices


# This class exists because:
# - the autograd.Function.forward() in aot autograd returns outputs that might alias inputs
# - we only care about the metadata on those aliases, so we can regenerate them.
#   We do not want them to participate in the autograd.Function.
# We do that by wrapping them in an opaque class, so the autograd.Function
# does not know to treat them as tensors.
@dataclass(frozen=True)
class TensorAlias:
    alias: torch.Tensor


def has_same_metadata(t1, t2):
    return (
        t1.size() == t2.size()
        and t1.stride() == t2.stride()
        and t1.storage_offset() == t2.storage_offset()
    )


def gen_alias_from_base(aliased_base_tensor, target_meta_tensor, target_requires_grad):
    # Try to do view-replay if possible.
    # fall back to .as_strided() if we can't.
    if target_meta_tensor._base is not None:
        # The base that we want to replay our view off of might have a different shape than the view's original base.
        b = target_meta_tensor._base
        abt = aliased_base_tensor
        # Don't unnecessarily call as_strided if nothing changed; as_strided's
        # backward is poorly implemented and slow
        if abt is not b and (
            abt.size() != b.size() or
            abt.stride() != b.stride() or
            abt.storage_offset() != b.storage_offset()
        ):
            reshaped_base_tensor = aliased_base_tensor.as_strided(
                b.size(), b.stride(), b.storage_offset()
            )
        else:
            reshaped_base_tensor = aliased_base_tensor
        out = target_meta_tensor._view_func(reshaped_base_tensor)
        # This shape mismatch can happen due to a bug in inplace/view handling in autograd.
        # Try putting a breakpoint here and running
        # `test/functorch/test_aotdispatch TestAOTAutograd.test_output_all_alias_types`
        # Also, https://github.com/pytorch/pytorch/issues/49825
        #
        # As a stopgap, we'll fall back to as_strided.
        if out is not None and out.shape == target_meta_tensor.shape:
            if aliased_base_tensor.requires_grad and not target_requires_grad:
                out = out.detach()
            elif not aliased_base_tensor.requires_grad and target_requires_grad:
                out.requires_grad_(True)
            return out
    size = target_meta_tensor.size()
    stride = target_meta_tensor.stride()
    storage_offset = target_meta_tensor.storage_offset()
    if aliased_base_tensor.is_complex() and not target_meta_tensor.is_complex():
        aliased_out = torch.view_as_real(aliased_base_tensor).as_strided(
            size, stride, storage_offset
        )
    elif not aliased_base_tensor.is_complex() and target_meta_tensor.is_complex():
        aliased_out = torch.view_as_complex(aliased_base_tensor).as_strided(
            size, stride, storage_offset
        )
    else:
        aliased_out = aliased_base_tensor.as_strided(size, stride, storage_offset)
    # For outputs aliasing inputs, we need to check if the requires-gradness has changed.
    if aliased_base_tensor.requires_grad and not target_requires_grad:
        aliased_out = aliased_out.detach()
    elif not aliased_base_tensor.requires_grad and target_requires_grad:
        aliased_out.requires_grad_(True)
    return aliased_out

def to_fun(t):
    if isinstance(t, Tensor):
        return torch._to_functional_tensor(t, mirror_autograd_meta=True)
    else:
        return t

def from_fun(t):
    if not isinstance(t, Tensor) or not torch._is_functional_tensor(t):
        return t
    torch._sync(t)
    return torch._from_functional_tensor(t)


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
# - ViewAndMutationMeta, telling us metadata about the inputs and outputs
# - The list of outputs from the forward, but **only** the outputs that we need
#   to pass in as tangents into the backward.
#   Specifically, aliased outputs from the forward get regenerated, and don't participate
#   in the compiled backward function.
def run_functionalized_fw_and_collect_metadata(
    f,
    *,
    keep_input_mutations: bool
) -> Tuple[ViewAndMutationMeta, List[Any]]:
    memo = {}

    def to_fun(t):
        if isinstance(t, Tensor):
            if t in memo:
                return memo[t]
            r = torch._to_functional_tensor(t, mirror_autograd_meta=True)
            memo[t] = r
            return r
        else:
            return t

    def from_fun(t):
        if not isinstance(t, Tensor) or not torch._is_functional_tensor(t):
            return t
        torch._sync(t)
        return torch._from_functional_tensor(t)

    @wraps(f)
    def inner(*flat_args):
        # This function is meant to be run with the forward, which expects a flat list of tensor/symint/other args.
        assert all(isinstance(a, KNOWN_TYPES) for a in flat_args)

        input_info: List[InputAliasInfo] = []
        output_info: List[OutputAliasInfo] = []
        input_requires_grad_info: List[bool] = []
        output_requires_grad_info: List[bool] = []

        flat_f_args = pytree.tree_map(to_fun, flat_args)

        torch._enable_functionalization(reapply_views=True)
        try:
            # precondition: The passed in function already handles unflattening inputs + flattening outputs
            flat_f_outs = f(*flat_f_args)
        finally:
            torch._disable_functionalization()

        # Inspect the state of the input tensor functional wrapper to detect input mutation info
        # If inp[i] has a metadata-only mutation, then maybe_inputs_with_mutated_metadata[i] contains the updated version
        for (i, (arg, f_arg)) in enumerate(zip(flat_args, flat_f_args)):
            if not isinstance(arg, Tensor):
                new_arg = arg
            else:
                torch._sync(f_arg)
                new_arg = torch._from_functional_tensor(f_arg)
            if arg is not new_arg:
                if StorageWeakRef(arg.untyped_storage()) == StorageWeakRef(new_arg.untyped_storage()):
                    mutates_data = False
                    mutates_metadata = True
                else:
                    mutates_data = True
                    mutates_metadata = not has_same_metadata(arg, new_arg)
                # Only track requires_grad info on *mutated* inputs,
                # because they show up in the autograd.Function.forward as outputs
                input_requires_grad_info.append(
                    isinstance(f_arg, torch.Tensor) and f_arg.requires_grad
                )
            else:
                mutates_data = False
                mutates_metadata = False

            input_info.append(InputAliasInfo(
                is_leaf=isinstance(arg, torch.Tensor) and arg.is_leaf,
                mutates_data=mutates_data,
                mutates_metadata=mutates_metadata
            ))

        # If a function involves creating a tensor, and returning a view of it, such that its _base is the intermediiate,
        # We need to make sure our graph returns the _base as a graph output, and we manually recreate the view
        # to return to the user. Why? The backend compiler is free to (incorrectly) not set requires_grad
        # on the base tensor, but we are obligated to properly set requires-gradness on the real output.

        num_mutated_inps = len(
            [x for x in input_info if x.mutates_data or x.mutates_metadata]
        )
        inp_storage_refs = {
            StorageWeakRef(inpt.untyped_storage()): idx
            for idx, inpt in enumerate(flat_f_args)
            if isinstance(inpt, torch.Tensor)
        }

        # We need inp tensor id's to be able to tell if an outputs **are** inputs.
        inp_tensor_ids = {
            id(inpt) for inpt in flat_f_args if isinstance(inpt, torch.Tensor)
        }
        # We need output tensor id's to tell if any output._base` attributes **are** other outputs.
        # (This is also a dict because we need to know that output's index, so we can regenerate
        # the alias from it).
        out_tensor_ids = {id(o): i for i, o in enumerate(flat_f_outs)}
        # maps the id of an intermediate base to its index in the output of the compiled forward
        intermediate_base_tensor_id_to_output_idx: Dict[int, int] = {}
        intermediate_bases: List[torch.Tensor] = []
        for o in flat_f_outs:
            if (
                isinstance(o, torch.Tensor)
                and StorageWeakRef(o.untyped_storage()) in inp_storage_refs
            ):
                base_idx = inp_storage_refs[StorageWeakRef(o.untyped_storage())]
                is_input_tensor = id(o) in inp_tensor_ids
                if is_input_tensor:
                    output_type = OutputType.is_input
                else:
                    output_type = OutputType.alias_of_input

            # We only need to handle the intermediate base case when both
            # the intermediate base and the output require gradients.
            # See Note [AOT Autograd: outputs aliasing inputs or intermediates!]
            elif (
                isinstance(o, torch.Tensor)
                and o._base is not None
                and o.requires_grad
                and o._base.requires_grad
            ):
                # First, check if o's ._base is an existing output
                maybe_existing_out_idx = out_tensor_ids.get(id(o._base), None)
                if maybe_existing_out_idx is not None:
                    # Special case where the output is an alias of a graph intermediate, but that intermediate
                    # is itself also a user output.
                    output_type = OutputType.alias_of_intermediate_base_is_user_output
                    base_idx = maybe_existing_out_idx
                else:
                    # Next, check if o's ._base is an intermediate base that we already returned
                    maybe_existing_base_output_idx = intermediate_base_tensor_id_to_output_idx.get(
                        id(o._base), None
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
                        output_type = OutputType.alias_of_intermediate_save_as_output
                        intermediate_base_tensor_id_to_output_idx[id(o._base)] = new_out_idx
                        intermediate_bases.append(o._base)
            else:
                output_type = OutputType.non_alias
                base_idx = None

            out_info = OutputAliasInfo(
                output_type=output_type,
                base_idx=base_idx,
            )
            output_info.append(out_info)
            output_requires_grad_info.append(
                isinstance(o, torch.Tensor) and o.requires_grad
            )

        # Our autograd.Function.forward returns both mutated inputs and outputs,
        # so we need grad info on all of them.
        requires_grad_info = input_requires_grad_info + output_requires_grad_info
        assert len(requires_grad_info) == len(output_info) + len(
            [x for x in input_info if x.mutates_data or x.mutates_metadata]
        )

        # This analysis function returns *only* the outputs that are meant to be tangents to the backwards.
        # Anything that aliases (inputs returned in the fw due to metadata mutations, or outputs that alias inputs/intermediates)
        # are *regenerated* later, and not used directly in the autograd graph
        f_input_tangents = [
            inp
            for inp, info in zip(flat_f_args, input_info)
            if info.mutates_data
        ]
        f_output_tangents = [
            o
            for o, info in zip(flat_f_outs, output_info)
            if info.output_type == OutputType.non_alias
        ]
        # intermediate bases are also included in the backward graph
        f_tangents = f_input_tangents + f_output_tangents + intermediate_bases

        metadata = ViewAndMutationMeta(
            input_info=input_info,
            requires_grad_info=requires_grad_info,
            output_info=output_info,
            num_intermediate_bases=len(intermediate_bases),
            keep_input_mutations=keep_input_mutations,
        )
        return metadata, pytree.tree_map(from_fun, f_tangents)

    return inner


def unpack_synthetic_bases(
    primals: List[Any],
    synthetic_base_info: Optional[List[Union[int, Tuple[int, torch.Tensor]]]],
) -> List[Any]:
    # This is only not None if our graph mutates a graph input that aliases another graph input.
    if synthetic_base_info is None:
        return primals

    f_args_inner = []
    for outer_idx_or_tuple in synthetic_base_info:
        if isinstance(outer_idx_or_tuple, int):
            f_args_inner.append(primals[outer_idx_or_tuple])
        else:
            outer_base_idx, view_tensor = outer_idx_or_tuple
            outer_base = primals[outer_base_idx]
            view_arg = gen_alias_from_base(
                outer_base, view_tensor, view_tensor.requires_grad
            )
            f_args_inner.append(view_arg)
    return f_args_inner

# This class contains all the metadata we care about for the current function we're compiling.
# This data is needed both at trace time and at runtime.
@dataclass
class CompiledRuntimeMetadata:
    # This type / object should be cleaned up
    # See Note [Synthetic Base Info Metadata]
    synthetic_base_info: Optional[List[Union[int, Tuple[int, torch.Tensor]]]]
    fw_metadata: ViewAndMutationMeta

    def __post_init__(self):
        self.num_outputs = len(self.fw_metadata.output_info)
        self.num_outputs_non_aliased = len(
            [x for x in self.fw_metadata.output_info if x.output_type == OutputType.non_alias]
        )
        self.num_outputs_aliased_to_inputs = len(
            [
                x
                for x in self.fw_metadata.output_info
                if x.output_type in [
                    OutputType.alias_of_input,
                    OutputType.is_input,
                ]
            ]
        )
        self.num_outputs_aliased_to_intermediates = len(
            [
                x
                for x in self.fw_metadata.output_info
                if x.output_type in [
                    OutputType.alias_of_intermediate,
                    OutputType.alias_of_intermediate_save_as_output,
                    OutputType.alias_of_intermediate_base_is_user_output,
                ]
            ]
        )
        self.num_outputs_aliased = (
            self.num_outputs_aliased_to_inputs + self.num_outputs_aliased_to_intermediates
        )
        self.num_mutated_data_inputs = len(
            [x for x in self.fw_metadata.input_info if x.mutates_data]
        )
        self.num_mutated_metadata_inputs = len(
            [
                x
                for x in self.fw_metadata.input_info
                if x.mutates_metadata
            ]
        )
        self.num_mutated_metadata_only_inputs = len(
            [
                x
                for x in self.fw_metadata.input_info
                if not x.mutates_data and x.mutates_metadata
            ]
        )
        self.num_mutated_inputs = self.num_mutated_data_inputs + self.num_mutated_metadata_only_inputs

# This function takes in a tensor t, and returns one of t, t.view(), or t.clone().
# When tracing the joint forward + backward, for any inputs in the graph that are mutated,
# we need to clone them first (and similarly for metadata-only mutations, we need to view them first).
# The idea is that when we trace the backward, we need to pass in the *original* primals
# to autograd.grad(), before they were mutated.
# Note: when we have synthetic base inputs, we need to clone them *before* creating views off of them.
# This means that "idx" here represents the index of the (potentially) synthetic base.
# What we need to do is:
# (1) map the current (post-synthetic-base calling convention) input argument index
#     to int index pre-synthetic-base-calling-convention.
# (2) There could be multiple, if this index corresponds to a synthetic base
#     that has multiple input aliases.
# (3) If any of those corresponding inputs get metadata mutations, then we clone the base.
def maybe_to_fresh_input(idx, t, meta):
    if not isinstance(t, Tensor):
        return t

    if meta.synthetic_base_info is None:
        outer_aliased_indices_of_current_base_arg = [idx]
    else:
        outer_aliased_indices_of_current_base_arg = [
            # For every argument index in the outer calling convention (before synthetic bases)
            # find its index in the inner calling convention.
            # if it matches the index of our current arg (idx), track the outer argument's index (i)
            i
            for i, outer_idx_or_tuple in enumerate(meta.synthetic_base_info)
            if (isinstance(outer_idx_or_tuple, int) and outer_idx_or_tuple == idx)
            or (
                isinstance(outer_idx_or_tuple, tuple)
                and outer_idx_or_tuple[0] == idx
            )
        ]
    if any(
        meta.fw_metadata.input_info[i].mutates_data
        for i in outer_aliased_indices_of_current_base_arg
    ):
        # Make sure the primal we pass to autograd.grad()
        # sees the tensor before the mutation
        return t.clone()
    if any(
        meta.fw_metadata.input_info[i].mutates_metadata and not meta.fw_metadata.input_info[i].mutates_data
        for i in outer_aliased_indices_of_current_base_arg
    ):
        # Make sure the primal we pass to autograd.grad()
        # sees the tensor before the metadata mutation
        return t.view(t.shape)
    return t

# This function takes in a forward fn, runs it, and (optionally) runs autograd to compute the joint.
# When maybe_tangents is None, we only run the forward. Otherwise we run the "joint" forward + backward.
# Preconditions:
# - fn corresponds to the flattened user fw function, with duplicate inputs removed
# - functionalization is turned on (and inputs are wrapped in functional tensors)
# - Synthetic bases have been *removed* (we've taken views on them corresponding to the user argument views).
# - primals_after_cloning are what we run our forward function on. It is identical to primals_before_cloning,
#   except that every input we know will be mutated in the forward has been cloned.
#   We run our forward on primals_after_cloning (potentially mutating some inputs), and then compute our gradients
#   w.r.t. primals_before_cloning (so we properly capture the mutation in our gradient computation).
# Importantly, due functionalization + some autograd.Function constraints, this function can return EXTRA outputs
# compared to what the original user forward returns.
#
# If we are only running the forward (and not computing the joint):
# - Our function will return (updated_inputs, fw_outs)
#
# If we are running the forward + backward (computing the joint):
# - Our function will return (updated_inputs, fw_outs, intermediate_bases), (gradients)
#
# Finally, if keep_input_mutations is set, then we will explicitly *not* return updated inputs, for any inputs
# that experienced data-only mutations.
# Instead, we are relying on the logic in create_forward_or_joint_functionalized to manually perform the input mutations,
# keeping them directly in the traced graph.
def forward_or_joint(
    fn: Callable,
    primals_before_cloning: List[Any],
    primals_after_cloning: List[Any],
    maybe_tangents: Optional[List[Any]],
    meta: CompiledRuntimeMetadata,
    keep_input_mutations: bool,
) -> Any:
    outs = fn(*primals_after_cloning)
    assert len(meta.fw_metadata.output_info) == len(outs)

    # The compiled fw will return mutated input tensors, *including* metadata-only mutation.
    # However, if keep_input_mutations is set, the compiled fw only needs to return metadata-mutated inputs.
    # (because data-only input mutations are handled directly in the compiled graph)
    if keep_input_mutations:
        mutated_inputs_to_return = [
            x
            for (i, x) in enumerate(primals_after_cloning)
            if meta.fw_metadata.input_info[i].mutates_metadata
        ]
    else:
        mutated_inputs_to_return = [
            x
            for (i, x) in enumerate(primals_after_cloning)
            if meta.fw_metadata.input_info[i].mutates_data or meta.fw_metadata.input_info[i].mutates_metadata
        ]

    # Case 1: We are just tracing the forward; not the joint forward + backward.
    if maybe_tangents is None:
        return *mutated_inputs_to_return, *outs
    else:
        tangents = maybe_tangents

    # Case 2: We are tracing the joint forward backward.
    # This also requires us to:
    # - update the graph to return intermediate bases
    # - Figure out what grad_outputs to pass into the backward
    # - (this includes intermediate bases in the forward, and forward inputs that had data mutations)
    # - actually call autograd.grad to trace the backward.
    intermediate_bases = []
    for o, info in zip(outs, meta.fw_metadata.output_info):
        if info.output_type == OutputType.alias_of_intermediate_save_as_output:
            intermediate_bases.append(o._base)

    assert meta.fw_metadata.num_intermediate_bases == len(intermediate_bases)

    # Pass any (non-aliased) outputs in as tangents, since they'll be returned as outputs in the fw
    # For outputs that are aliases of intermediates, we will have returned the output's _base as an output in the graph instead,
    # which we *should* send to grad()
    outputs_for_grad = [
        x
        for (i, x) in enumerate(outs)
        if meta.fw_metadata.output_info[i].output_type == OutputType.non_alias
    ]
    # Pass any (non-aliased) mutated inputs in as tangents, since they'll be returned as outputs in the fw
    # Important: the traced joint fw/bw will return updated inputs with data mutations,
    # but *not* with metadata mutations.
    # Instead, we shunt the updated metadata around externally
    # and update the input's metadata outside of the autograd.Function
    mutated_inputs_for_grad = [
        x
        for (i, x) in enumerate(primals_after_cloning)
        if meta.fw_metadata.input_info[i].mutates_data
    ]
    # The tensors that we include in the backward graph are:
    # - inputs that recieve *data* mutations (not metadata-only; those are recomputed later)
    # - outputs that are not aliased (aliased outputs are recomputed later)
    # - intermediate ._base tensors of aliased outputs (we use those later to recompute the aliased outputs)
    fw_outs_to_grad = mutated_inputs_for_grad + outputs_for_grad + intermediate_bases
    assert len(tangents) == len(fw_outs_to_grad)

    # the compiled forward should return (mutated_inputs, user_outs, intermediate_bases)
    fw_outs_to_return = *mutated_inputs_to_return, *outs, *intermediate_bases

    # Take care to grab and sync the updated inputs from primals_after_cloning (the inputs we actually mutate!)
    # and not primals_before_cloning (the preserved inputs, pre-mutation, that we pass to grad())
    for i, arg in enumerate(primals_after_cloning):
        if not isinstance(arg, Tensor):
            continue
        torch._sync(arg)

    # Get the inputs that need gradients
    grad_primals = []
    inputs_needs_grads = []
    # Note that we're not using primals_before_cloning here,
    # being carefully not to pass any mutated inputs into autograd.grad()
    for p in primals_before_cloning:
        is_grad_tensor = isinstance(p, Tensor) and p.requires_grad
        inputs_needs_grads.append(is_grad_tensor)
        if is_grad_tensor:
            grad_primals.append(p)

    # Get the outputs that need gradients
    needed_outs = []
    needed_tangents = []
    for out, tangent in zip(fw_outs_to_grad, tangents):
        if isinstance(out, Tensor) and out.requires_grad:
            # A bit sketchy, but fixes e.g. test_aot_autograd_exhaustive_matmul_cpu_float32
            # The issue is that we are sensitive to decomps that don't accurately maintain
            # their output's _base.shape compared to eager mode, and this helps mitigate a bit.
            needed_outs.append(
                out if out.shape == tangent.shape else out.view(tangent.shape)
            )
            needed_tangents.append(tangent.requires_grad_(True))

    setup_stacktrace_preservation_hooks([out.grad_fn for out in needed_outs])

    backward_out = []
    # Call the backwards pass
    if grad_primals:
        with fx_traceback.preserve_node_meta():
            backward_out = torch.autograd.grad(
                needed_outs,
                grad_primals,
                grad_outputs=needed_tangents,
                allow_unused=True,
            )
    backward_out_iter = iter(backward_out)
    return fw_outs_to_return, [
        next(backward_out_iter) if i else None for i in inputs_needs_grads
    ]

# This function expands synthetic base arguments into the original aliased inputs that the user passed in.
# Preconditions:
# - fn corresponds to the flattened user fw function, with duplicate inputs removed
# - functionalization is turned on (and inputs are wrapped in functional tensors)
# - both primals args **include** synthetic bases.
#   "primals_after_cloning" just corresponds to "primals_before_cloning", but with some inputs (optionally) cloned.
#   "primals_before_cloning" is unused, and is only needed so we can pass the correct leaf tensors into autograd.
def flat_fn_with_synthetic_bases_expanded(
    fn: Callable,
    primals_before_cloning: List[Any],
    primals_after_cloning: List[Any],
    maybe_tangents: Optional[List[Any]],
    meta: CompiledRuntimeMetadata,
    keep_input_mutations: bool
):
    # This is where we handle the calling convention around synthetic bases.
    # We need to make sure that we convert any synthetic base arguments into views
    # *after* we clone inputs for autograd (see below), to preserve the view relationship.
    primals = unpack_synthetic_bases(primals_after_cloning, meta.synthetic_base_info)
    assert len(meta.fw_metadata.input_info) == len(primals)
    outs = forward_or_joint(fn, primals_before_cloning, primals, maybe_tangents, meta, keep_input_mutations)
    return outs

# This function adds extra clone() calls on any inputs in the forward that get mutated.
# It *only* does this if we plan on performing autograd on fn.
# The idea here is that when computing grdients w.r.t. inputs, we need to compute our gradients
# w.r.t. the inputs *before* they were mutated!
# Preconditions:
# - fn corresponds to the flattened user fw function, with duplicate inputs removed
# - primals **includes** synthetic bases. Importantly, if a synthetic base is mutated,
#   we need to clone it *before* taking views off of it (if we clone the views they won't be views anymore)
# - functionalization is turned on (and inputs are wrapped in functional tensors)
def flat_fn_no_input_mutations(
    fn: Callable,
    primals: List[Any],
    maybe_tangents: Optional[List[Any]],
    meta: CompiledRuntimeMetadata,
    keep_input_mutations: bool
):
    # When tracing the joint fwd + bwd, making sure to clone any inputs that are mutated first.
    # We need to ensure that the inputs we pass to autograd.grad() are the *original*
    # inputs, and not their mutated values.
    if maybe_tangents is not None:
        primals_after_cloning = [
            maybe_to_fresh_input(i, t, meta) for i, t in enumerate(primals)
        ]
    else:
        primals_after_cloning = primals
    outs = flat_fn_with_synthetic_bases_expanded(fn, primals, primals_after_cloning, maybe_tangents, meta, keep_input_mutations)
    return outs

# This creates the final function that we want to trace using make_fx(),
# in both aot_dispatch_autograd and aot_dispatch_base.
# Preconditions:
# - fn corresponds to the user's fw function
# - fn arguments have been flattened, duplicate arguments have been handled
# - In the returned function, the "primals" arguments *includes* synthetic bases.
# This function does the work of functionalizing the input function,
# and performing copy_() calls at the end of the function if `keep_input_mutations` is set.
# The function returned has signature that is either:
# (1) "traced_fn(primals: List[Any])" if trace_joint is False
# (2) "traced_fn(primals: List[Any], tangents: List[Any])" if trace_joint is True
def create_forward_or_joint_functionalized(
    fn,
    *,
    meta: CompiledRuntimeMetadata,
    trace_joint: bool,
    keep_input_mutations: bool
):

    def functionalized_f_helper(primals, maybe_tangents=None):
        # Convention: this function is used to trace both the joint, and just the forward (for inference).
        # When trace_joint is set, tangents should be passed in.
        assert (maybe_tangents is not None) == trace_joint
        # Wrap inputs into functional wrappers
        f_primals = pytree.tree_map(to_fun, primals)
        f_tangents = None if maybe_tangents is None else pytree.tree_map(to_fun, maybe_tangents)
        torch._enable_functionalization(reapply_views=True)
        try:
            # Run the joint
            f_outs = flat_fn_no_input_mutations(fn, f_primals, f_tangents, meta, keep_input_mutations)
        finally:
            torch._disable_functionalization()

        if keep_input_mutations:
            # Note: This is a bit annoying. There's a layering issue here, where:
            # (1) functionalization needs to operate on **synthetic base** inputs, before unpacking them into the "real" inputs.
            # (2) For keep_input_mutations, we support tracing a call to copy_() directly on mutated inputs.
            #     However, we **only** want to support this for inputs that have data-only (and no metadata) mutations,
            #     because inductor (and backends in generally) would prefer not to see these (e.g. as_strided_(), resize_()).
            #     This makes it pretty difficult for this logic to operate on synthetic bases.
            # (3) In addition, there are cases where it's significantly cheaper to perform the copy on the individual
            #     (unpacked) input aliases, instead of the synthetic base.
            # The result is that ideally this function shouldn't have to worry about synthetic bases
            # (unpacking them happens underneath this function),
            # but we actually do need to unpack the synthetic bases when performing the copy_'s to keep input mutations around.
            # Example case where this could be important:
            #
            #     def f(x, y):
            #         x.mul_(2)
            #         y.mul_(3)
            #         return x, y
            #    a = torch.ones(1'000'000)
            #    x, y = out(a[0:9], a[1:10])
            #
            # It would be much better to add copy_() calls into the graph for the two tiny slices, instead of materializing
            # a giant "updated synthetic base" and copying into a's entire storage.
            primals_unpacked = unpack_synthetic_bases(primals, meta.synthetic_base_info)
            f_primals_unpacked = unpack_synthetic_bases(f_primals, meta.synthetic_base_info)
            assert len(meta.fw_metadata.input_info) == len(f_primals_unpacked)
            for i, (inpt_old, inpt_f) in enumerate(zip(primals_unpacked, f_primals_unpacked)):
                if not isinstance(inpt_f, torch.Tensor):
                    continue
                torch._sync(inpt_f)
                inpt_new = torch._from_functional_tensor(inpt_f)
                if meta.fw_metadata.input_info[i].mutates_data and not meta.fw_metadata.input_info[i].mutates_metadata:
                    # We found an input that had a (data-only) mutation.
                    # Since keep_input_mutations is set, we need to faithfully apply a copy_()
                    # so the compiler will see the input mutation in the graph.
                    assert inpt_new is not inpt_old
                    assert has_same_metadata(inpt_new, inpt_old)
                    inpt_old.copy_(inpt_new)

        return pytree.tree_map(from_fun, f_outs)

    # the joint needs have args named "primals" and "tangents",
    # which are hardcoded into the partitioning logic.
    def traced_joint(primals, tangents):
        return functionalized_f_helper(primals, tangents)

    def traced_forward(*primals):
        return functionalized_f_helper(primals)

    if trace_joint:
        return traced_joint
    else:
        return traced_forward


def normalize_as_list(x):
    if isinstance(x, tuple):
        return list(x)
    elif isinstance(x, list):
        return x
    return [x]


aot_autograd_decompositions = {}


# This is a list since looking forward, we can have this arbitrarily nested.
graph_being_compiled: List[str] = []
# TODO: It would be nice to reset the numbering every time aot_id goes
# up, but this is annoying to do right now (because we don't know if
# an aot_id will come back from the dead), so right now this also happens
# to be a globally unique number too (at the cost of wobbling if you change
# how the graphs compile)
nth_graph: int = 0
model_name: str = "model"


def set_model_name(name):
    global model_name
    model_name = name


def get_aot_compilation_context() -> Tuple[List[str], str, int]:
    return list(graph_being_compiled), model_name, nth_graph


def get_aot_graph_name() -> str:
    """
    Returns the name of the graph being compiled.
    """
    global model_name, graph_being_compiled, nth_graph
    return f"{model_name}__{'_'.join(graph_being_compiled)}_{nth_graph}"


get_graph_being_compiled = get_aot_graph_name


@contextmanager
def track_graph_compiling(aot_config, graph_name):
    global graph_being_compiled
    # TODO: Don't shove the aot_id in here; set it in the context
    graph_being_compiled = [f"{aot_config.aot_id}_{graph_name}"]
    yield
    global nth_graph
    nth_graph += 1
    graph_being_compiled = []


def make_boxed_func(f):
    def g(args):
        return f(*args)

    g._boxed_call = True
    return g


def make_boxed_compiler(compiler):
    @wraps(compiler)
    def f(fx_g, inps):
        out_f = compiler(fx_g, inps)
        fx_g = make_boxed_func(out_f)
        return fx_g

    return f


def call_func_with_args(f, args, steal_args=False, disable_amp=False):
    if not steal_args:
        args = list(args)
    assert isinstance(args, list)

    if disable_amp:
        guard = torch._C._DisableAutocast()
    try:
        if hasattr(f, "_boxed_call"):
            out = normalize_as_list(f(args))
        else:
            # TODO: Please remove soon
            # https://github.com/pytorch/pytorch/pull/83137#issuecomment-1211320670
            warnings.warn(
                "Your compiler for AOTAutograd is returning a a function that doesn't take boxed arguments. "
                "Please wrap it with functorch.compile.make_boxed_func or handle the boxed arguments yourself. "
                "See https://github.com/pytorch/pytorch/pull/83137#issuecomment-1211320670 for rationale."
            )
            out = normalize_as_list(f(*args))
    finally:
        if disable_amp:
            del guard
    return out

@dataclasses.dataclass
class AOTConfig:
    """
    Configuration for AOTDispatcher
    """

    fw_compiler: Callable
    bw_compiler: Callable
    partition_fn: Callable
    decompositions: Dict[Callable, Callable]
    num_params_buffers: int
    aot_id: int
    keep_inference_input_mutations: bool

def aot_dispatch_base(flat_fn, flat_args: List[Tensor], aot_config: AOTConfig):
    with enable_python_dispatcher():
        _fw_metadata, _out = run_functionalized_fw_and_collect_metadata(
            flat_fn,
            keep_input_mutations=aot_config.keep_inference_input_mutations,
        )(
            *flat_args
        )

    _input_info = _fw_metadata.input_info

    flat_args_with_views_handled, _synthetic_base_info = merge_view_inputs(
        flat_args, _input_info, is_inference=True
    )
    metadata_ = CompiledRuntimeMetadata(
        synthetic_base_info=_synthetic_base_info,
        fw_metadata=_fw_metadata,
    )
    # aot_dispatch_base requires functionalization, but doesn't need to handle as many cases as the autograd case.
    # The cases that aot_dispatch_base doesn't need to handle include:
    # - outputs that are aliases of graph intermediates
    # - outputs that are aliases of graph inputs
    # While cases that it does need to handle include:
    # - input mutations (including when inputs are aliases of each other)
    # - input metadata mutations
    trace_fn = create_forward_or_joint_functionalized(
        flat_fn,
        meta=metadata_,
        trace_joint=False,
        keep_input_mutations=aot_config.keep_inference_input_mutations
    )

    with enable_python_dispatcher():
        fw_module = make_fx(trace_fn, aot_config.decompositions)(*flat_args_with_views_handled)

    if not aot_config.keep_inference_input_mutations:
        # As long as we opted to remove input mutations, then
        # there should be *NO* mutating ops in the graph at this point.
        assert_functional_graph(fw_module.graph)
        fw_module.graph.eliminate_dead_code()
        fw_module.recompile()

    if config.debug_graphs:
        log.debug(f"====== Forward (only) graph {aot_config.aot_id} ======")
        log.debug(fw_module.print_readable(print_output=False))

    disable_amp = torch._C._is_any_autocast_enabled()
    context = disable_autocast_manager if disable_amp else nullcontext

    with context(), track_graph_compiling(aot_config, "inference"):
        compiled_fw = aot_config.fw_compiler(fw_module, flat_args_with_views_handled)

    compiled_fn = create_runtime_wrapper(
        compiled_fw,
        runtime_metadata=metadata_,
        trace_joint=False,
        keep_input_mutations=aot_config.keep_inference_input_mutations
    )

    return compiled_fn


def assert_functional_graph(fx_g: torch.fx.Graph):
    for n in fx_g.nodes:
        if isinstance(n.target, torch._ops.OpOverload):
            assert not n.target._schema.is_mutable, \
                f'aot_autograd expected to have an entirely functional graph, but found {n.format_node()}'


@contextmanager
def disable_autocast_manager():
    guard = torch._C._DisableAutocast()
    try:
        yield
    finally:
        del guard


def are_differentiable_views(view1, view2):
    if view1 is view2:
        return True
    if view1._base is None and view2._base is None:
        return False
    if view1._base is view2._base or view1._base is view2 or view1 is view2._base:
        return True
    return False


def same_dtype_views(view1, view2):
    if view1.dtype != view2.dtype:
        return False
    if view1._base is not None and view1.dtype != view1._base.dtype:
        return False
    if view2._base is not None and view2.dtype != view2._base.dtype:
        return False
    return True


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
    fwd_inputs: List[Any], mutated_input_info: List[InputAliasInfo],
    *,
    # The autograd case currently has more restrictions than the inference case.
    is_inference: bool,
) -> Tuple[List[Any], Optional[List[Union[int, Tuple[int, torch.Tensor]]]]]:
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
        # We detected an input that was mutated, AND aliases with another input.
        # we need to replace this set of aliased inputs with a single synthetic base.
        # For now, I'm banning a bunch of cases. We expect dynamo to properly detect these cases
        # and error out. We can fix them later.
        # These checks are transitive, so we don't need to check every pair.
        for idx1, idx2 in zip(aliased_input_indices, aliased_input_indices[1:]):
            view1 = fwd_inputs[idx1]
            view2 = fwd_inputs[idx2]
            # The "inputs that are aliased but have different differentiable bases" case
            # is more complicated and hopefully pretty rare. Not currently handled.
            if not is_inference:
                assert are_differentiable_views(
                    view1, view2
                ), "aot_autograd() does not yet handle non-differentiable view input mutations."
            # Regenerating views when reinterpreting complex / real tensors seems non-trivial,
            # not handling for now
            assert same_dtype_views(
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
            # Note that this function is re-used at both trace time and rutnime.
            # At trace time, we're under a FakeMode so synthetic_base becomes a FakeTensor.
            synthetic_base = torch.empty((0,), dtype=example_alias.dtype, device=example_alias.device)
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
        post_processed_calling_convention_meta: List[Union[int, Callable]] = [
            -1 for _ in range(len(inner_calling_convention_meta))
        ]
        for k, v in inner_calling_convention_meta.items():
            post_processed_calling_convention_meta[k] = v
        # Quick assert: every argument in the inner calling convention should be accounted for.
        for x in post_processed_calling_convention_meta:
            assert x != -1
        return args_to_functionalization, post_processed_calling_convention_meta


def format_guard_bug_msg(aot_config, expected):
    return (
        f"At compilation time, graph {aot_config.aot_id} was compiled under the "
        f"assumption that {expected}, but at runtime this was not the case.  "
        "This indicates a guard bug in AOTAutograd or Dynamo, please file a bug to PyTorch."
    )


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
    flat_fn, flat_args: List[Tensor], aot_config: AOTConfig, *, compiler_fn
):
    # Get information about whether or not flat_fn mutates its arguments
    # or not
    try:
        with enable_python_dispatcher():
            fw_metadata, _out = run_functionalized_fw_and_collect_metadata(
                flat_fn,
                # For the purpose of checking for dupes that are mutated,
                # we always want our metadata to correctly reflect input mutations
                keep_input_mutations=False,
            )(
                *flat_args
            )
    except RuntimeError as e:
        log.warning(
            "Failed to collect metadata on function, produced code may be suboptimal.  "
            "Known situations this can occur are inference mode only compilation involving "
            "resize_ or prims (!schema.hasAnyAliasInfo() INTERNAL ASSERT FAILED); "
            "if your situation looks different please file a bug to PyTorch.",
            exc_info=True,
        )
        # Analysis failed, fall back to duplicate specialize
        # TODO: Known analysis problems:
        #   - resize_: TestInductorOpInfoCPU.test_comprehensive_resize__cpu_bool
        #   - prims: test_tmp_not_defined_issue1_cpu
        pass
    else:
        # Strategy 1: For any input that is not mutated, we can leafify it if we
        # need to remove a duplicate.
        leaf_flat_args = []
        args_set = set()
        ok = True

        for i, a in enumerate(flat_args):
            if a not in args_set:
                args_set.add(a)
                leaf_flat_args.append(a)
            elif not fw_metadata.input_info[i].mutates_data and not fw_metadata.input_info[i].mutates_metadata:
                leaf_flat_args.append(a.detach().requires_grad_(a.requires_grad))
            else:
                ok = False
                break

        if ok:
            return compiler_fn(flat_fn, leaf_flat_args, aot_config)

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
    #   add_dupe_map = {  # how to get args from the deduped list
    #       0: 0,
    #       1: 1,
    #       2: 0,
    #       3: 2,
    #   }
    #   keep_arg_mask = [True, True, False, True]

    seen_args = {}
    keep_arg_mask = []
    add_dupe_map = {}
    duped_arg_len = len(flat_args)

    j = 0  # index into deduped_flat_args
    for i, t in enumerate(flat_args):
        if t in seen_args:
            keep_arg_mask.append(False)
            add_dupe_map[i] = seen_args[t]
            continue
        keep_arg_mask.append(True)
        seen_args[t] = j
        add_dupe_map[i] = j
        j += 1

    unique_args = j

    # NB: Hot path, avoid set lookups here
    # TODO: Can avoid the zip here too, probably
    def remove_dupe_args(args):
        return [t for t, keep in zip(args, keep_arg_mask) if keep]

    def add_dupe_args(args):
        return [args[add_dupe_map[i]] for i in range(duped_arg_len)]

    deduped_flat_args = remove_dupe_args(flat_args)

    tracing_context = TracingContext.get()
    if tracing_context:
        # TODO(voz): This structure is 1:1, we could consider an alternate structure like
        # kept_pos:[dupe_arg_pos], however, add_dupe_map is 1:1 so we would need a new structure there,
        # which feels like needless complexity for a tiny bit of efficiency at this point.
        for dupe_arg_pos, kept_pos in add_dupe_map.items():
            dupe_arg_dict = flat_args[dupe_arg_pos].__dict__
            kept_arg_dict = flat_args[kept_pos].__dict__
            if 'graph_arg_pos' in dupe_arg_dict and 'graph_arg_pos' in kept_arg_dict:
                d_positions = dupe_arg_dict['graph_arg_pos']
                k_positions = kept_arg_dict['graph_arg_pos']
                assert(d_positions == k_positions)
                if len(d_positions) > 1:
                    for i in range(1, len(d_positions)):
                        pos = d_positions[i]
                        pre_pos = d_positions[i - 1]
                        tracing_context.guards_context.aotautograd_guards.append(DuplicateInputs(pre_pos, pos))

    @wraps(flat_fn)
    def wrapped_flat_fn(*args):
        return flat_fn(*add_dupe_args(args))

    compiled_fn = compiler_fn(wrapped_flat_fn, deduped_flat_args, aot_config)

    if not hasattr(compiled_fn, "_boxed_call"):
        compiled_fn = make_boxed_func(compiled_fn)

    @wraps(compiled_fn)
    def wrapped_compiled_fn(args):
        deduped_args = remove_dupe_args(args)
        args.clear()
        return compiled_fn(deduped_args)

    wrapped_compiled_fn._boxed_call = True

    # This can be uncommented when we properly guard for duplicates,
    # but right now we must not do it.
    # if not config.debug_assert:
    #     return wrapped_compiled_fn

    @wraps(wrapped_compiled_fn)
    def debugged_compiled_fn(args):
        # Test that the computed remove/add arg functions are an inverse
        new_args = add_dupe_args(remove_dupe_args(args))
        seen = {}
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

    debugged_compiled_fn._boxed_call = True

    return debugged_compiled_fn


def describe_input(i, aot_config):
    if i < aot_config.num_params_buffers:
        return f"parameter/buffer {i}"
    else:
        return f"input {i - aot_config.num_params_buffers}"

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
    runtime_metadata: CompiledRuntimeMetadata,
    trace_joint: bool,
    keep_input_mutations: bool,
):
    if not hasattr(compiled_fn, "_boxed_call"):
        compiled_fn = make_boxed_func(compiled_fn)

    def runtime_wrapper(*args):
        # Step 2: remove aliased inputs that are mutated, replace with synthetic bases
        # Only happens if our graph mutates an input that aliases another input.
        if runtime_metadata.synthetic_base_info is not None:
            # Given: the original args, including at least one pair of inputs that are aliased
            # and get subsequently mutated.
            # Generate: the updated args, including (potentially multiple) synthetic bases
            # that replace the views. The input views are regenerated manually in the compiled function.
            # TODO: think harder about what happens if (a view of) one of these mutated input views is ALSO returned
            new_inputs, metadata = merge_view_inputs(
                args, runtime_metadata.fw_metadata.input_info, is_inference=not trace_joint,
            )
            # We're just re-running the original-args-to-synthetic-base transformation
            # that we ran during compilation.
            # This returns metadata that we use during tracing to recover the input views,
            # which we don't actually need at runtime.
            assert metadata is not None
            args_with_synthetic_bases = new_inputs
        else:
            args_with_synthetic_bases = args

        with torch.autograd._force_original_view_tracking(True):
            all_outs = call_func_with_args(
                compiled_fn,
                args_with_synthetic_bases,
                disable_amp=True,
            )

        num_mutated_inps = runtime_metadata.num_mutated_inputs
        num_metadata_mutated_inps = runtime_metadata.num_mutated_metadata_inputs
        num_intermediate_bases = runtime_metadata.fw_metadata.num_intermediate_bases

        if keep_input_mutations:
            assert (
                len(all_outs)
                == num_metadata_mutated_inps + runtime_metadata.num_outputs + num_intermediate_bases
            )
            assert (
                len(runtime_metadata.fw_metadata.mutated_inp_indices) == num_metadata_mutated_inps
            )
        else:
            assert (
                len(all_outs)
                == num_mutated_inps + runtime_metadata.num_outputs + num_intermediate_bases
            )
            assert (
                len(runtime_metadata.fw_metadata.mutated_inp_indices) == num_mutated_inps
            )
        # Step 3: After running the compiled fw, apply updates to mutated inputs
        num_mutations_to_apply = len(runtime_metadata.fw_metadata.mutated_inp_indices)
        if num_mutations_to_apply > 0:
            updated_inputs = all_outs[: num_mutations_to_apply]
            fw_outs = all_outs[num_mutations_to_apply :]

            for i, inpt_idx in enumerate(
                runtime_metadata.fw_metadata.mutated_inp_indices
            ):
                meta = runtime_metadata.fw_metadata.input_info[inpt_idx]
                if not meta.mutates_data and not meta.mutates_metadata:
                    continue
                original_inpt = args[inpt_idx]
                updated_inpt = updated_inputs[i]
                # TODO: add better resize_() support for autograd case.
                # Check for the case when an input has been resized.
                # Note: One important thing to check for is user code that calls inpt.storage().resize_().
                # We can't trace operations on storage into the graph, so we should get dynamo to graph break.
                # TODO: handle resize_() on inputs to a larger size.
                # This is actually non-trivial to detect, so we should probably just handle it
                # (or make dynamo detect).
                # We can't just check of original_inpt.storage_size != updated_inpt.storage_size,
                # Because the original_inpt might be a view of some larger tensor,
                # and updated_inpt is always densely packed.
                if not trace_joint and original_inpt.storage().size() != updated_inpt.storage().size():
                    original_inpt.resize_(updated_inpt.size())
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
            if runtime_metadata.fw_metadata.num_intermediate_bases > 0:
                fw_outs_no_intermediate_bases = fw_outs[
                    : -runtime_metadata.fw_metadata.num_intermediate_bases
                ]
                intermediate_bases = fw_outs[-runtime_metadata.fw_metadata.num_intermediate_bases:]
            else:
                fw_outs_no_intermediate_bases = fw_outs
                intermediate_bases = []
            assert len(fw_outs_no_intermediate_bases) == len(runtime_metadata.fw_metadata.output_info)

            fw_outs_including_aliases = []
            for i, (o, info) in enumerate(zip(
                fw_outs_no_intermediate_bases, runtime_metadata.fw_metadata.output_info
            )):
                if info.output_type == OutputType.non_alias:
                    fw_outs_including_aliases.append(o)
                    continue
                if trace_joint:
                    assert isinstance(o, TensorAlias)
                    o_ = o.alias
                else:
                    o_ = o
                o_grad = runtime_metadata.fw_metadata.requires_grad_info[runtime_metadata.num_mutated_inputs + i]
                if info.output_type == OutputType.alias_of_input:
                    aliased_base_tensor = args[info.base_idx]
                    regenerated_out = gen_alias_from_base(aliased_base_tensor, o_, o_grad)
                    fw_outs_including_aliases.append(regenerated_out)
                    continue
                elif info.output_type == OutputType.is_input:
                    aliased_base_tensor = args[info.base_idx]
                    regenerated_out = aliased_base_tensor
                    fw_outs_including_aliases.append(regenerated_out)
                    continue
                elif info.output_type == OutputType.alias_of_intermediate:
                    base_tensor_list = intermediate_bases
                elif info.output_type == OutputType.alias_of_intermediate_save_as_output:
                    base_tensor_list = intermediate_bases
                else:
                    assert info.output_type == OutputType.alias_of_intermediate_base_is_user_output
                    base_tensor_list = fw_outs_no_intermediate_bases
                aliased_base_tensor = base_tensor_list[info.base_idx]
                # TODO: handle the custom autograd function case here.
                # We need a way to check whether a tensor came from a custom autograd fn from python,
                # AND a way to replay that custom view fn.
                regenerated_out = gen_alias_from_base(aliased_base_tensor, o_, o_grad)
                fw_outs_including_aliases.append(regenerated_out)
            return fw_outs_including_aliases
        else:
            return fw_outs
    return runtime_wrapper

# Has the precondition that there
# are no duplicate arguments in flat_args (e.g., the same Tensor
# object never shows up twice.  However, two tensor inputs MAY alias
# the same storage, so long as they have separate TensorImpls.)
def aot_dispatch_autograd(flat_fn, flat_args: List[Any], aot_config: AOTConfig):

    with enable_python_dispatcher():
        _fw_metadata, out = run_functionalized_fw_and_collect_metadata(
            flat_fn,
            # Note: in the non-inference path, we are currently not passing input mutations into the graph directly.
            # This is mainly difficult due to the partitioner, but we are leaving (a bit of) perf on the table.
            keep_input_mutations=False,
        )(
            *flat_args
        )


    # out here corresponds to the set of outputs in the traced forward that should get grad_outputs in the traced backward.
    # It includes outputs of the original forward, *and* any updated inputs due to input mutations.
    # However, it does *not* include any outputs that are aliases of inputs or intermediates, or any metadata-only input mutations.
    out = pytree.tree_map(
        lambda x: x.detach().contiguous() if isinstance(x, Tensor) else x,
        out,
    )

    # merge_view_inputs() is used again at runtime to create synthetic bases out of aliased inputs.
    # This code only executes at runtime if we have graph inputs that alias each other, and one of those inputs
    # gets its data mutated.
    # When that happens, we replace the aliased inputs with a synthetic base, and in the traced forward
    # we later generate the input views
    flat_args_with_views_handled, _synthetic_base_info = merge_view_inputs(
        flat_args, _fw_metadata.input_info, is_inference=False,
    )

    # pre-compute, so we can bail out quickly in the hotpath
    metadata_ = CompiledRuntimeMetadata(
        synthetic_base_info=_synthetic_base_info,
        fw_metadata=_fw_metadata,
    )

    assert len(_fw_metadata.requires_grad_info) == metadata_.num_mutated_inputs + metadata_.num_outputs

    joint_forward_backward = create_forward_or_joint_functionalized(
        flat_fn,
        meta=metadata_,
        trace_joint=True,
        # For now in the autograd case, we NEVER keep input mutations (we could eventually fix this for slightly better perf
        # in some cases, but it's annoying to fix the partitioner)
        keep_input_mutations=False,
    )

    joint_inputs = (flat_args_with_views_handled, out)

    disable_amp = torch._C._is_any_autocast_enabled()

    if config.use_functionalize:
        with enable_python_dispatcher():
            flattened_joints, _ = pytree.tree_flatten(joint_inputs)
            fx_g = make_fx(joint_forward_backward, aot_config.decompositions)(
                *joint_inputs
            )

        # There should be *NO* mutating ops in the graph at this point.
        assert_functional_graph(fx_g.graph)
        # Redudant with the check above, but worth having in case tracing introduced
        # a fake tensor. Unlikely.
        # See Note: [Fake Modules and AOTAutograd]
        torch._dynamo.utils.assert_no_fake_params_or_buffers(fx_g)
        fx_g.graph.eliminate_dead_code()
        fx_g.recompile()
    else:
        # joint_forward_backward() now always runs with functionalization, and factoring it out
        # to make that toggleable is a bit painful.
        # aot autograd without functionalization is wrong anyway, so we error.
        raise AssertionError(
            "Graph partitioning without functionalization is not sound, we may introduce errors"
        )

    if config.debug_joint:
        log.debug(f"====== Joint graph {aot_config.aot_id} ======")
        log.debug(fx_g.print_readable(print_output=False))

    with torch.no_grad():
        with track_graph_compiling(aot_config, "joint"):
            num_inner_fwd_outputs = metadata_.num_mutated_inputs + metadata_.num_outputs + _fw_metadata.num_intermediate_bases
            fw_module, bw_module = aot_config.partition_fn(
                fx_g, joint_inputs, num_fwd_outputs=num_inner_fwd_outputs
            )
            fw_outs = [n for n in fw_module.graph.nodes if n.op == "output"][0].args[0]
            # we only need to bookkeep the symints that are saved for bw, not any symints
            # the user forward might have returned in its own output
            fw_outs_saved_for_bw = fw_outs[num_inner_fwd_outputs:]
            symint_outs_saved_for_bw = [
                n for n in fw_outs_saved_for_bw if is_sym_node(n)
            ]
            _num_symints_saved_for_bw = len(symint_outs_saved_for_bw)

        if config.debug_graphs:
            log.debug(f"====== Forward graph {aot_config.aot_id} ======")
            log.debug(fw_module.print_readable(print_output=False))
            log.debug(f"====== Backward graph {aot_config.aot_id} ======")
            log.debug(bw_module.print_readable(print_output=False))

        with track_graph_compiling(aot_config, "forward"):
            compiled_fw_func = aot_config.fw_compiler(
                fw_module, flat_args_with_views_handled
            )

    class CompiledFunction(torch.autograd.Function):
        compiled_fw = compiled_fw_func
        compiled_bw = None
        metadata = metadata_
        num_symints_saved_for_bw = _num_symints_saved_for_bw

        @staticmethod
        def forward(ctx, *deduped_flat_tensor_args):

            # There is a pretty complicated calling convention around what the compiled fw returns.
            # The full list of outputs and their relative order is:
            # (*mutated_inputs, *fw_outs, *fw_intermediate_bases, *saved_tensors, *saved_symints)
            # - Note that in the synthetic bases case, mutated_inputs will correspond to an updated version
            #   of the original view, and not the synthetic base
            fw_outs = call_func_with_args(
                CompiledFunction.compiled_fw,
                deduped_flat_tensor_args,
                disable_amp=disable_amp,
            )

            num_outputs = CompiledFunction.metadata.num_outputs
            num_outputs_aliased_to_inputs = (
                CompiledFunction.metadata.num_outputs_aliased_to_inputs
            )
            num_outputs_aliased_to_intermediates = (
                CompiledFunction.metadata.num_outputs_aliased_to_intermediates
            )
            num_outputs_aliased = CompiledFunction.metadata.num_outputs_aliased
            num_intermediate_bases = CompiledFunction.metadata.fw_metadata.num_intermediate_bases
            num_symints_saved_for_bw = CompiledFunction.num_symints_saved_for_bw
            num_mutated_inputs = CompiledFunction.metadata.num_mutated_inputs
            num_mutated_metadata_only_inputs = (
                CompiledFunction.metadata.num_mutated_metadata_only_inputs
            )
            # Our forward() returns both (mutated_inputs, outputs, output_intermediate_bases, saved_tensors, saved_symints)
            num_forward_returns = num_mutated_inputs + num_outputs + num_intermediate_bases

            assert num_forward_returns == len(
                CompiledFunction.metadata.fw_metadata.requires_grad_info
            ) + num_intermediate_bases

            # Partitioners must put symint arguments at the end separate from tensor arguments
            if num_symints_saved_for_bw > 0:
                tensors_saved_for_backwards = fw_outs[
                    num_forward_returns:-num_symints_saved_for_bw
                ]
                assert all(
                    [isinstance(x, torch.Tensor) for x in tensors_saved_for_backwards]
                )
                # See Note [Detaching saved tensors in AOTAutograd]
                ctx.save_for_backward(*map(lambda x: x.detach() if x._is_view() else x, tensors_saved_for_backwards))
                symint_outs = fw_outs[-num_symints_saved_for_bw:]
                assert all(
                    [
                        isinstance(x, (int, float, torch.SymInt, torch.SymFloat))
                        for x in symint_outs
                    ]
                )
                ctx.symints = symint_outs
            else:
                tensors_saved_for_backwards = fw_outs[num_forward_returns:]
                # See Note [Detaching saved tensors in AOTAutograd]
                ctx.save_for_backward(*map(lambda x: x.detach() if x._is_view() else x, tensors_saved_for_backwards))
                ctx.symints = []

            raw_returns = fw_outs[0:num_forward_returns]

            # Wrap all autograd.Function.forward() outputs that are aliases
            # so that autograd.Function doesn't treat them as tensors
            if num_mutated_metadata_only_inputs > 0:
                for i, idx in enumerate(
                    CompiledFunction.metadata.fw_metadata.mutated_inp_indices
                ):
                    # We could make this faster by only looping over inputs with metadata-only mutations
                    # (instead of looping over inputs with either data or metadata mutations), but there shouldn't be many.
                    info = CompiledFunction.metadata.fw_metadata.input_info[idx]
                    if info.mutates_metadata and not info.mutates_data:
                        raw_returns[i] = TensorAlias(raw_returns[i])

                if config.debug_assert:
                    user_mutated_inputs_raw = raw_returns[0:num_mutated_inputs]
                    mut_inp_infos = [
                        x for x in CompiledFunction.metadata.fw_metadata.input_info if x.mutates_data or x.mutates_metadata
                    ]
                    assert len(user_mutated_inputs_raw) == len(mut_inp_infos)

            if num_outputs_aliased > 0:
                for idx in CompiledFunction.metadata.fw_metadata.aliased_out_indices:
                    raw_return_idx = num_mutated_inputs + idx
                    raw_returns[raw_return_idx] = TensorAlias(raw_returns[raw_return_idx])

                if config.debug_assert:
                    intermediates_raw = raw_returns[num_mutated_inputs + num_outputs:]
                    assert not any(isinstance(x, TensorAlias) for x in intermediates_raw)

            # invariant: intermediate bases always require gradients, so we don't have to
            # consider marking them as non-differentiable.
            raw_returns_not_including_intermediate_bases = raw_returns[:num_mutated_inputs + num_outputs]
            fw_outs_not_requiring_grad = [
                x
                for (i, x) in enumerate(raw_returns_not_including_intermediate_bases)
                if isinstance(x, torch.Tensor)
                and not CompiledFunction.metadata.fw_metadata.requires_grad_info[i]
            ]
            ctx.mark_non_differentiable(*fw_outs_not_requiring_grad)

            return tuple(raw_returns)

        @staticmethod
        def backward(ctx, *flat_args):
            # Calling convention: we expect a grad_out passed to the backward:
            # - for every output of the fw that does *not* alias an input or graph intermediate
            # - for every updated_input generated by the fw that does *not* alias an input (aka only data-mutations)
            # - for every graph intermediate that we need to use to generate an output later.
            # The other outputs in the autograd.Function.forward that do *not* show up in the backward include:
            # - outputs that alias inputs or graph intermediates
            # - updated inputs due to metadata-only mutations.
            # We need to return them in the forward, but ensure that they all do not get gradients in the backward,
            # and we filter them out here before passing the remaining grad_outputs into the compiled backward.
            num_mutated_inps = CompiledFunction.metadata.num_mutated_inputs
            num_intermediate_bases = CompiledFunction.metadata.fw_metadata.num_intermediate_bases
            expected_grad_outs = (
                CompiledFunction.metadata.num_outputs + num_mutated_inps + num_intermediate_bases
            )

            assert len(flat_args) == expected_grad_outs
            if (
                CompiledFunction.metadata.num_mutated_metadata_only_inputs > 0
                or CompiledFunction.metadata.num_outputs_aliased > 0
            ):
                inp_tangents, out_tangents, intermediate_base_tangents = (
                    flat_args[0:num_mutated_inps],
                    flat_args[num_mutated_inps:num_mutated_inps + CompiledFunction.metadata.num_outputs],
                    flat_args[num_mutated_inps + CompiledFunction.metadata.num_outputs:],
                )
                # input_info contains info on *every* input,
                # But in the backward(), we are only given grad outputs for every mutated input.
                # We then need to filter out the grad outputs that correspond to metadata-only mutations.
                mutated_inp_indices = CompiledFunction.metadata.fw_metadata.mutated_inp_indices
                input_info = CompiledFunction.metadata.fw_metadata.input_info
                assert len(inp_tangents) == len(mutated_inp_indices)
                inp_tangents_filtered = [
                    x
                    for x, info_idx in zip(inp_tangents, mutated_inp_indices)
                    if input_info[info_idx].mutates_data
                ]
                # We also need to filter out grad outputs that correspond to outputs aliasing inputs/intermediates
                out_info = CompiledFunction.metadata.fw_metadata.output_info
                out_tangents_filtered = [
                    x
                    for x, info in zip(out_tangents, out_info)
                    if info.output_type == OutputType.non_alias
                ]
                # intermediate bases always require gradients, and always participate in the backward graph.
                flat_bw_args = itertools.chain(inp_tangents_filtered, out_tangents_filtered, intermediate_base_tangents)

                # sanity asserts
                # metadata_only_inps = [
                #     x for x, info_idx in zip(inp_tangents, mutated_inp_indices)
                #     if not input_info[info_idx].mutates_data
                # ]
                # aliased_outputs = [
                #     x for x, info in zip(out_tangents, out_info) if info.output_type != OutputType.non_alias]
                # assert all(x is None for x in metadata_only_inps)
                # assert all(x is None for x in aliased_outputs)
            else:
                flat_bw_args = flat_args

            contiguous_args = [
                t.contiguous() if torch.is_tensor(t) else t for t in flat_bw_args
            ]

            all_args = (
                list(ctx.symints) + list(ctx.saved_tensors) + list(contiguous_args)
            )
            del contiguous_args

            def call_compiled_backward():
                if CompiledFunction.compiled_bw is None:
                    # TODO - pass in fake tensors ?
                    context = disable_autocast_manager if disable_amp else nullcontext
                    with context(), track_graph_compiling(aot_config, "backward"):
                        CompiledFunction.compiled_bw = aot_config.bw_compiler(
                            bw_module, all_args
                        )

                ctx.maybe_clear_saved_tensors()
                out = call_func_with_args(
                    CompiledFunction.compiled_bw,
                    all_args,
                    steal_args=True,
                    disable_amp=disable_amp,
                )

                return tuple(out)

            if torch.is_grad_enabled() and any(t.requires_grad for t in all_args if isinstance(t, torch.Tensor)):
                # Ensure that the graph is connected, and error if double backward is performed.
                # See comment for why once_differentiable is not sufficient:
                # https://github.com/pytorch/pytorch/pull/92348/files#r1072962107
                class CompiledFunctionBackward(torch.autograd.Function):
                    @staticmethod
                    def forward(ctx, *unused_args):
                        return call_compiled_backward()

                    @staticmethod
                    def backward(ctx, *args):
                        raise RuntimeError("torch.compile with aot_autograd does not currently support double backward")
                # Pass args even though they're unused, so that the graph is built
                out = CompiledFunctionBackward.apply(*all_args)
            else:
                out = call_compiled_backward()
            return out

    compiled_function = create_runtime_wrapper(
        CompiledFunction.apply,
        runtime_metadata=metadata_,
        trace_joint=True,
        keep_input_mutations=False,
    )

    if not config.debug_assert:
        return compiled_function

    flat_requires_grad = [
        a.requires_grad if isinstance(a, Tensor) else None for a in flat_args
    ]

    @wraps(compiled_function)
    def debug_compiled_function(*args):
        # TODO: Check aliasing relationships
        # TODO: Check strides for metadata mutation
        # (NB: ideally, this logic is factored out of this function and
        # you move these debug checks there)

        # Check requires grad.  Bad case is when we compiled with
        # requires_grad = False, but input requires_grad = True
        # (vice versa is OK; we compute a gradient and then throw
        # it away when it hits the input.)
        for i, a in enumerate(args):
            can_require_grad = flat_requires_grad[i]
            if can_require_grad is None:
                assert not isinstance(a, Tensor)
            elif not can_require_grad:
                assert not a.requires_grad, format_guard_bug_msg(
                    aot_config,
                    f"{describe_input(i, aot_config)} would not require grad",
                )

        return compiled_function(*args)

    return debug_compiled_function


@dynamo_timed
def create_aot_dispatcher_function(
    flat_fn, flat_args: List[Any], aot_config: AOTConfig
):
    """
    Traces the forward and backward graphs of the attr:`flat_fn` to generate a
    joint graph. The joint graph is an Fx graph with Aten ops. Please refer to
    the tracing mechanism to understand the graph capturing details.

    The joint graph is then passed through attr:`partition_fn` to isolate the
    forward and backward portions, which are then respectively compiled via the
    provided attr:`fw_compiler` and attr:`bw_compiler`.

    The resulting compiled forward and backward graphs are then wrapped up in a
    ``torch.autograd.Function`` object.

    The calling convention here is that the first aot_config.num_params_buffers
    inputs in flat_args are parameters and buffers, and the rest are inputs.

    We use this to assume that parameters/buffer's shapes don't change.
    """

    # This is the main entry point.
    # TODO: Chillee argues that dynamo itself should pass in fake tensors to
    # the list of arguments when compiling; at the moment we do not do this

    if aot_config.decompositions is None:
        aot_config.decompositions = {}

    aot_config.decompositions = {
        **aot_autograd_decompositions,
        **aot_config.decompositions,
    }

    log.setLevel(config.log_level)

    # NB: don't bother setting allow_fallback_kernels; this should not actually
    # be configurable in fake tensor, we should automatically do the right
    # thing
    if config.debug_fake_cross_ref:
        # This is a little messy but TorchDynamo directly changes `use_fake_tensor`
        # so it's not enough for user to change the config manually
        # TODO: have TorchDynamo read in `use_fake_tensor` from os environ /
        # coordinate flags
        config.use_fake_tensor = False

    # Check flat_args to see if they're already fake.  If so, use that fake
    # mode instead.

    for x in flat_args:
        if isinstance(x, FakeTensor):
            fake_mode = x.fake_mode
            shape_env = fake_mode.shape_env
            break
    else:
        shape_env = ShapeEnv() if config.use_dynamic_shapes else None
        fake_mode = (
            FakeTensorMode(shape_env=shape_env)
            if config.use_fake_tensor
            else nullcontext()
        )

    cross_ref = CrossRefFakeMode() if config.debug_fake_cross_ref else nullcontext()
    python_dispatcher_mode = (
        enable_python_dispatcher() if shape_env is not None else nullcontext()
    )

    with torch.autograd.set_multithreading_enabled(
        False
    ), preserve_rng_state(), cross_ref, fake_mode, python_dispatcher_mode:

        def process_inputs(flat_args):
            if config.use_fake_tensor or isinstance(fake_mode, FakeTensorMode):

                def convert(idx, x):
                    if not isinstance(x, torch.Tensor):
                        return x
                    if isinstance(x, FakeTensor):
                        assert x.fake_mode is fake_mode
                        return x
                    if (
                        idx < aot_config.num_params_buffers
                        and config.static_weight_shapes
                    ):
                        return fake_mode.from_tensor(x, static_shapes=True)
                    return fake_mode.from_tensor(x, static_shapes=False)

                return [convert(idx, x) for idx, x in enumerate(flat_args)]
            else:
                return flat_args

        fake_flat_args = process_inputs(flat_args)

        needs_autograd = (
            any([x.requires_grad for x in fake_flat_args if isinstance(x, Tensor)])
            and torch.is_grad_enabled()
        )
        # crappy version of dispatcher
        # TODO: Do this properly
        if needs_autograd:
            compiler_fn = aot_dispatch_autograd
        else:
            compiler_fn = aot_dispatch_base

        compiler_fn = partial(aot_wrapper_dedupe, compiler_fn=compiler_fn)
        # You can put more passes here

        compiled_fn = compiler_fn(flat_fn, fake_flat_args, aot_config)

        if not hasattr(compiled_fn, "_boxed_call"):
            compiled_fn = make_boxed_func(compiled_fn)

        return compiled_fn


# Inspired by autodidax (thanks!)
class PytreeThunk:
    spec = None
    # These are some kinda dumb microoptimizations that save about 3-4 us of overhead.
    is_simple = (
        None  # if the output spec is a tuple/list, we won't bother unflattening it.
    )
    is_really_simple = None  # if the output spec is a LeafSpec

    def set(self, spec):
        assert self.spec is None or self.spec == spec
        self.spec = spec
        if type(self.spec) in [tuple, list] and all(
            isinstance(i, pytree.LeafSpec) for i in spec.children_specs
        ):
            self.is_simple = True
        if isinstance(self.spec, pytree.LeafSpec):
            self.is_really_simple = True

    def unflatten(self, x):
        if self.is_really_simple:
            return x[0]
        if self.is_simple:
            return x
        return pytree.tree_unflatten(x, self.spec)


def aot_function(
    fn: Callable,
    fw_compiler: Callable,
    bw_compiler: Optional[Callable] = None,
    partition_fn: Callable = default_partition,
    decompositions: Optional[Dict] = None,
    num_params_buffers: int = 0,
    hasher_type=None,  # deprecated
    static_argnums: Optional[Tuple[int]] = None,  # deprecated
    keep_inference_input_mutations: bool = False
) -> Callable:
    """
    Traces the forward and backward graph of :attr:`fn` using torch dispatch
    mechanism, and then compiles the generated forward and backward graphs
    through :attr:`fw_compiler` and :attr:`bw_compiler`.

    :func:`aot_function` traces the forward and backward graph ahead of time,
    and generates a joint forward and backward graph.  :attr:`partition_fn` is
    then used to separate out forward and backward graphs. The partitioner
    function can be used to perform optimizations such as recomputation. One can
    set `decompositions` dictionary to decompose the operators into a sequence
    of core or simpler operators supported by the backend compilers.

    :func:`aot_function` uses a compilation cache, based on input tensor
    properties, to detect when there is a need of recompilation.

    .. warning::
        This API is experimental and likely to change.

    Args:
        fn (Callable): A Python function that takes one ore more arguments. Must
            return one or more Tensors.
        fw_compiler (Callable): A Python function that accepts an Fx graph with
            Aten ops and input args, and returns a Callable that semantically is
            equivalent to the input Fx graph.
        bw_compiler (Optional[Callable]): A Python function that accepts an
            Fx graph with Aten ops and input args, and returns a Callable that
            semantically is equivalent to the input Fx graph.  Default: None
            (when None, it defaults to the :attr:`fw_compiler`)
        partition_fn (Callable): A Python function that takes a joint forward
            and backward graph, and partitions it into separate forward and
            backward graphs.
        decompositions (Dict): A dictionary to define the decomposition of
            larger Aten ops into simpler or core Aten ops.

    Returns:
        Returns a ``Callable`` that retains the eager behavior of the original
        :attr:`fn`, but with forward and backward graph compiled via
        :attr:`fw_compile` and :attr:`bw_compile`.

    A simple example usage of :func:`aot_function` is as follows. This example
    will print the forward and backward graphs of the function ``fn``

        >>> fn = lambda x : x.sin().cos()
        >>> def print_compile_fn(fx_module, args):
        >>>     print(fx_module)
        >>>     return fx_module
        >>> aot_fn = aot_function(fn, print_compile_fn)
        >>> x = torch.randn(4, 5, requires_grad=True)
        >>> aot_fn(x)
    """
    if static_argnums is not None:
        raise RuntimeError(
            "static_argnums has been deprecated - manually wrap your function or use torchdynamo."
        )

    if bw_compiler is None:
        bw_compiler = fw_compiler
    aot_config = AOTConfig(
        fw_compiler=fw_compiler,
        bw_compiler=bw_compiler,
        partition_fn=partition_fn,
        decompositions=decompositions,
        num_params_buffers=num_params_buffers,
        aot_id=next(AOT_COUNTER),
        keep_inference_input_mutations=keep_inference_input_mutations
    )
    cached_res = None

    @wraps(fn)
    def returned_function(*args, **kwargs):
        nonlocal cached_res
        # Now flatten the tensor args
        flat_args, _ = pytree.tree_flatten((args, kwargs))

        # Compile the function and save it in the cache
        if cached_res is None:
            # Save the args_spec for flat_tensor_args to unflatten while tracing
            _, tensor_args_spec = pytree.tree_flatten((args, kwargs))
            out_spec = PytreeThunk()

            def flat_fn(*flat_args):
                # The input are flattened tensor args. Prepare the args in the
                # order that original function expects. Add static args as well.
                # They will appear as tensor constants in the traced graph.
                nonlocal out_spec
                args, kwargs = pytree.tree_unflatten(flat_args, tensor_args_spec)
                tree_out = fn(*args, **kwargs)
                flat_out, spec = pytree.tree_flatten(tree_out)
                for i in flat_out:
                    is_known_type = False
                    for j in KNOWN_TYPES:
                        if isinstance(i, j):
                            is_known_type = True
                            break
                    if not is_known_type:
                        raise RuntimeError(
                            f"Found {type(i)} in output, which is not a known type. "
                            "If this type holds tensors, you need to register a pytree for it. "
                            "See https://github.com/pytorch/functorch/issues/475 for a brief "
                            "explanation why. If you don't need to register a pytree, please "
                            "leave a comment explaining your use case and we'll make this more "
                            "ergonomic to deal with"
                        )
                out_spec.set(spec)
                return flat_out

            compiled_fn = create_aot_dispatcher_function(
                flat_fn,
                flat_args,
                aot_config,
            )
            cached_res = (compiled_fn, out_spec)

        cached_fn, out_spec = cached_res
        out = cached_fn(flat_args)
        return out_spec.unflatten(out)

    return returned_function


def aot_module(mod: nn.Module, *args, **kwargs) -> nn.Module:
    """
    Traces the forward and backward graph of :attr:`mod` using torch dispatch
    tracing mechanism. It is wrapper function, that underneath uses
    :func:`aot_function` to perform tracing and compilation.

    :func:`aot_module` lifts the parameters and buffers of ``nn.Module`` as inputs
    to a new callable which is then compiled through :func:`aot_function`.

    .. warning::
        This API is experimental and likely to change.

    Args:
        mod (Callable): A ``nn.Module`` module.
        args : args to be passed to :func:`aot_function`
        kwargs : kwargs to be passed to :func:`aot_function`

    Returns:
        Returns a ``nn.Module`` that retains the eager behavior of the original
        :attr:`mod`, but with forward and backward graph compiled.

    """
    # See Note: [Fake Modules and AOTAutograd]
    torch._dynamo.utils.assert_no_fake_params_or_buffers(mod)

    def functional_call(named_params, named_buffers, *args, **kwargs):
        params_and_buffers = {**named_params, **named_buffers}
        return torch.func.functional_call(mod, params_and_buffers, args, kwargs)

    named_params = dict(mod.named_parameters(remove_duplicate=False))
    named_buffers = dict(mod.named_buffers(remove_duplicate=False))
    num_params_buffers = len(named_params) + len(named_buffers)
    compiled_f = aot_function(
        functional_call, num_params_buffers=num_params_buffers, *args, **kwargs
    )

    class AOTModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.orig_module = mod

        def forward(self, *args, **kwargs):
            return compiled_f(
                named_params,
                named_buffers,
                *args,
                **kwargs,
            )

    return AOTModule()


def aot_module_simplified(
    mod: nn.Module,
    args,
    fw_compiler: Callable,
    bw_compiler: Optional[Callable] = None,
    partition_fn: Callable = default_partition,
    decompositions: Optional[Dict] = None,
    hasher_type=None,
    static_argnums=None,
    keep_inference_input_mutations=False,
) -> nn.Module:
    """
    This is the simplified or low overhead version of aot_module. For frontends
    like TorchDynamo, the input functions/modules to AOT are static and have
    unpacked inputs/outputs. This gives us an opportunity to remove the
        (1) pytree overhead to parse inputs/outputs,
        (2) AOT Autograd cache,
        (3) Reading of params/buffers in every forward call

    :func:`aot_module_simplified` removes these overheads.
    """
    #########################################################

    # Redudant with dynamo, but worth having in case this gets invoked elsewhere.

    # Note [Fake Modules and AOTAutograd]
    #
    # A simple heuristic for when to use fake versus real tensors is that fake tensors are for compile time
    # (when we don't want to actually run the compute, but we do want to know about metadata),
    # and real tensors are for runtime (when we actually want to do the compute.) However, in AOTAutograd,
    # modules are the exception: we always pass AOTAutograd modules with real tensors.
    # This is because AOTAutograd will produce a compiled function which needs to directly access any
    # parameters the compiled function may need, but these parameters will NOT be passed in by the caller (aka Dynamo).
    # So at compile time, the compiled function we produce must close over any parameters, and those parameters must be
    # real parameters, and we cannot do this unless at compile time we get a module with real tensors.

    # Even if Dynamo did pass all parameters explicitly at runtime, which would eliminate the need to close over
    # the parameters, it would still be profitable to pass real tensor parameters to the compiler at compile time,
    # because some compilation strategies like CUDA graphs want to burn in the pointer addresses where the parameter data live,
    # and of course we can't do that unless we give the backend a real tensor.
    torch._dynamo.utils.assert_no_fake_params_or_buffers(mod)

    params = {
        **dict(mod.named_parameters(remove_duplicate=False)),
        **dict(mod.named_buffers(remove_duplicate=False)),
    }
    params_flat, params_spec = pytree.tree_flatten(params)
    params_flat = tuple(params_flat)
    params_len = len(params_flat)

    def functional_call(*args, **kwargs):
        with stateless._reparametrize_module(
            mod, pytree.tree_unflatten(args[:params_len], params_spec)
        ):
            if isinstance(mod, torch.fx.GraphModule):
                with fx_traceback.preserve_node_meta(), warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore", "Anomaly Detection has been enabled."
                    )
                    with torch.autograd.detect_anomaly(check_nan=False):
                        out = Interpreter(mod).run(*args[params_len:], **kwargs)
            else:
                out = mod(*args[params_len:], **kwargs)

        if not isinstance(out, (tuple, list)):
            raise RuntimeError(
                "Graph output must be a tuple(). This is so that we can avoid "
                "pytree processing of the ouputs. Please change the module to "
                "have tuple outputs or use aot_module instead."
            )
        return out

    assert static_argnums is None
    if bw_compiler is None:
        bw_compiler = fw_compiler
    aot_config = AOTConfig(
        fw_compiler=fw_compiler,
        bw_compiler=bw_compiler,
        partition_fn=partition_fn,
        decompositions=decompositions,
        num_params_buffers=params_len,
        aot_id=next(AOT_COUNTER),
        keep_inference_input_mutations=keep_inference_input_mutations,
    )

    full_args = []
    full_args.extend(params_flat)
    full_args.extend(args)

    compiled_fn = create_aot_dispatcher_function(
        functional_call,
        full_args,
        aot_config,
    )

    # TODO: There is something deeply wrong here; compiled_fn running with
    # the boxed calling convention, but aot_module_simplified somehow
    # historically returned a function that was not the boxed calling
    # convention.  This should get fixed...
    def forward(*runtime_args):
        full_args = []
        full_args.extend(params_flat)
        full_args.extend(runtime_args)
        return compiled_fn(full_args)

    # Just for convenience
    forward.zero_grad = mod.zero_grad
    forward.named_parameters = mod.named_parameters
    forward.named_buffers = mod.named_buffers

    return forward


compiled_function = aot_function
compiled_module = aot_module
