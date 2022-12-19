import collections
import dataclasses
import warnings
import itertools
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from enum import Enum
from functools import wraps, partial
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from torch.fx.experimental.proxy_tensor import is_sym_node
import logging

import torch
import torch.fx.traceback as fx_traceback
import torch.nn as nn
import torch.utils._pytree as pytree
import torch.utils.dlpack
from torch import Tensor
from torch._dynamo.utils import dynamo_timed
from torch._subclasses import FakeTensorMode, CrossRefFakeMode, FakeTensor
from torch.fx import immutable_collections, Interpreter
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from torch.multiprocessing.reductions import StorageWeakRef
from torch.nn.utils import stateless

from functorch import make_fx
from torch._dispatch.python import enable_python_dispatcher
from . import config
from .named_members_polyfill import _named_buffers, _named_parameters
from .partitioners import default_partition
from torch._guards import TracingContext, DuplicateInputs

log = logging.getLogger(__name__)

MutationType = Enum("MutationType", ("none", "metadata_only", "data"))
OutputType = Enum(
    "OutputType", ("non_alias", "alias_of_input", "alias_of_intermediate")
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

KNOWN_TYPES = [torch.Tensor, int, str, float, bool, torch.SymInt, torch.SymFloat]


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


# This class tells us about a user's forward output that is an alias.
# It can be an alias of either a user forward input, of of a graph intermediate.
@dataclass(frozen=True)
class OutputAliasInfo:
    # Tells us if this output is:
    # (1) a regular (non-aliased) output
    # (2) an alias of a forward input
    # (2) an alias of an intermediate (aka an alias of an output of the inner traced forward)
    output_type: OutputType
    # If (1) above, then
    # - Tells us that the base of this alias is user_fwd_input[base_idx]
    #   (This is an index into the inputs *before* we make synthetic bases)
    # If (2) above, then
    # - Tells us that the base of this alias is traced_fwd_outputs[base_idx]
    #   here, this refers to the index of the *direct* traced
    base_idx: int
    # sizes, strides and storage offset of the aliased output are all returned as actual (sym)ints
    # in the compiled forward. These indices tell us where in the forward outputs to grab them.
    sizes_idx: Optional[int]
    strides_idx: Optional[int]
    storage_offset_idx: Optional[int]
    # We store the actual output alias that we traced in the forward (should be a fake tensor)
    # to grab any other non-symbolic properties on the output alias, like requires_grad.
    # It's optional here, for cases where the user directly returns an input as an output.
    # If output_type == non_alias, then these fields are also always None.
    tensor_meta: Optional[Tensor]


# This class tells us about how to perform a metadata mutation on forward inputs.
# it only applies to forward inputs that experience metadata-only mutations
@dataclass(frozen=True)
class InputAliasInfo:
    # This object gives us information about how to perform a metadata-mutation
    # on original_fwd_inputs[base_idx]
    #   (This is an index into the inputs *before* we make synthetic bases)
    base_idx: int
    # sizes, strides and storage offset of the aliased output are all returned as actual (sym)ints
    # in the compiled forward. These indices tell us where in the forward outputs to grab them.
    sizes_idx: int
    strides_idx: int
    storage_offset_idx: int
    # We store the actual output alias that we traced in the forward (should be a fake tensor)
    # to grab any other non-symbolic properties on the output alias, like requires_grad.
    tensor_meta: Tensor


# This class encapsulates all aliasing + mutation info we need about the forward graph
# See a more detailed overview of the edge case handling at
# https://docs.google.com/document/d/19UoIh_SVrMy_b2Sx5ZaeOJttm6P0Qmyss2rdBuyfoic/edit
@dataclass(frozen=True)
class ViewAndMutationMeta:
    # length: # user forward inputs
    # For every input, tells us whether the input:
    # (a) is not mutated
    # (b) only metadata is mutated
    # (c) data (and maybe metadta) is mutated
    mutated_input_info: List[MutationType]
    # length: (# inputs of the user forward)
    # metadata_mutation_input_info[i] is not None <====> mutated_input_info[i] == MutationType.metadata_only
    # We stash the updated FakeTensor that we traced with in the forward in here,
    # that way we can use it to replay the metadata mutation
    metadata_mutation_input_info: List[Optional[InputAliasInfo]]
    # length: # outputs in the compiled forward (not including output alias symints). Equal to:
    # length: (# inputs w data mutations) + (# outputs that don't alias inputs)
    # For every output *and* mutated input returned from the forward,
    # tells us whether or not the output should require gradients or not
    requires_grad_out_info: List[bool]
    # length: # fw outputs
    aliased_output_info: List[OutputAliasInfo]


def gen_alias_from_base(
    aliased_base_tensor, size, stride, storage_offset, target_meta_tensor
):
    # handle R2C and C2R
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
    if aliased_base_tensor.requires_grad and not target_meta_tensor.requires_grad:
        aliased_out = aliased_out.detach()
    elif not aliased_base_tensor.requires_grad and target_meta_tensor.requires_grad:
        aliased_out.requires_grad_(True)
    return aliased_out


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
# TODO: Provide a faster version of this that assumes flat arguments
# (so no pytree necessary)
def run_functionalized_fw_and_collect_metadata(f):
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
    def inner(*args):
        # This function is meant to be run with the forward, which expects a flat list of tensor/symint/other args.
        assert all(a is None or isinstance(a, torch.Tensor) or type(a) in KNOWN_TYPES for a in args)

        collect_mutated_input_info: List[MutationType] = []
        collect_requires_grad_out_info: List[bool] = []
        collect_aliased_output_info: List[OutputAliasInfo] = []
        collect_metadata_mutation_input_info: List[Optional[InputAliasInfo]] = []

        f_args = pytree.tree_map(to_fun, args)

        torch._enable_functionalization(reapply_views=True)
        try:
            outs = f(*f_args)
        finally:
            torch._disable_functionalization()

        flat_args, _ = pytree.tree_flatten(args)
        flat_f_args, _ = pytree.tree_flatten(f_args)
        flat_outs, _ = pytree.tree_flatten(outs)

        # Inspect the state of the input tensor functional wrapper to detect input mutation info
        inputs_with_mutated_data = []
        # If inp[i] has a metadata-only mutation, then maybe_inputs_with_mutated_metadata[i] contains the updated version
        maybe_inputs_with_mutated_metadata: List[Optional[torch.Tensor]] = []
        for (i, (arg, f_arg)) in enumerate(zip(flat_args, flat_f_args)):
            if not isinstance(arg, Tensor):
                new_arg = arg
            else:
                torch._sync(f_arg)
                new_arg = torch._from_functional_tensor(f_arg)
            if arg is not new_arg:
                # Note [Input mutation handling in aot autograd]
                # We use functionalization to detect two types in input mutations:
                # (1) metadata-only input mutations, like input.t_()
                # (2) data input mutations, like input.add_(1)
                #     inputs that have both data and metadata mutated get lumped into (2).
                #
                # Why do we distinguish these two cases? aot autograd needs to handle them very differently.
                # For data mutations, we return the updated inputs *directly* in the compiled forward graph.
                # e.g.
                # def f(x):
                #     x.mul_(2)
                #     out = x.mul(3)
                #     return out
                #
                # // This function gets compiled and dumped inside of an autograd.Function.forward()
                # def traced_forward(x):
                #     x_updated = x.mul(2)
                #     out = x_updated.mul(3)
                #     return x_updated, out
                #
                # // The returned function will call the compiled forward, and apply input mutations afterwards
                # def compiled_fn(x):
                #    x_updated, out = traced_forward(x)
                #    x.copy_(x_updated)
                #    return out
                #
                # For input metadata mutations, though, we cannot return the "updated input" in the forward graph,
                # Because it is an alias of an input. autograd.Function.forward can't handle arbitrary outputs that alias inputs.
                # Instead, we stash the "updated input metadata" during tracing
                # e.g.
                # def f(x):
                #     x.t_()
                #     out = x.mul(3)
                #     return out
                #
                # // This function gets compiled and dumped inside of an autograd.Function.forward()
                # // (We don't return x_updated. Just return the original fw out)
                # def traced_forward(x):
                #     x_updated = x.t()
                #     out = x_updated.mul(3)
                #     return out
                #
                # // The returned function will call the compiled forward, and apply input mutations afterwards
                # def compiled_fn(x):
                #    out = traced_forward(x)
                #    _x_updated_metadata = CompiledFunction.fw_metadata.metadata_mutation_input_info[0]
                #    x.as_strided_(_x_updated_metadata.size(), _x_updated_metadata.stride(), _x_updated_metadata.storage_offset())
                #    return out
                if StorageWeakRef(arg._storage()) == StorageWeakRef(new_arg._storage()):
                    # We can use the storage aliasing of the inputs and updated inputs
                    # to detect when an input was actually updated, or just inplace-viewed.
                    collect_mutated_input_info.append(MutationType.metadata_only)
                else:
                    collect_mutated_input_info.append(MutationType.data)
                    # Only return mutated inputs that mutate *data*, not metadata
                    # Note [Input mutation handling in aot autograd]
                    inputs_with_mutated_data.append(new_arg)
                    # For every mutated input, we ALSO need to return info on
                    # whether than mutated input requires gradients. Why?
                    # Our custom autograd.Function.forward returns updated inputs as outputs,
                    collect_requires_grad_out_info.append(f_arg.requires_grad)
            else:
                collect_mutated_input_info.append(MutationType.none)

            maybe_inputs_with_mutated_metadata.append(
                new_arg
                if collect_mutated_input_info[-1] == MutationType.metadata_only
                else None
            )

        def collect_grad_info(t):
            # Collect info on which output tensors require gradients,
            # so we can mark them properly in the returned autograd.Function.
            # We only collect requires_grad info on real forward outputs, and not on inputs.
            collect_requires_grad_out_info.append(
                isinstance(t, torch.Tensor) and t.requires_grad
            )

        # Note [output alias handling in aot autograd]
        # Given a function to compile where one of its outputs aliases an input,
        # we need to remove that output from the compiled graph and generate it off to the side.
        # e.g.
        # def f(x):
        #     return x.view(-1)
        #
        # Why? Two reasons:
        # (1) If your autograd.Function returns a view on an input in the forward, autograd.Function
        #     will not allow you to mutate it (This original came from arbitrary user code where the user might want to mutate)
        # (2) There's no reason to compile views anyway. We can just regenerate the view of the input off to the side,
        #
        # Another interesting case is when you have both mutation and aliasing:
        # def f(x):
        #     x.mul_(2)
        #     return x.view(-1)
        #
        # You could imagine that this output is now *safe* to compile and return in the autograd.Function,
        # because after functionalization runs, it will technically not alias an input:
        # def f_functionalized(x):
        #     x_updated = x.mul(2)
        #     return x_updated, x_updated.view(-1)
        #
        # However, this is still wrong: we can't return x_updated.view(-1) to the user. We are on the hook to return:
        # def traced_forward(x):
        #     x_updated = x.mul(2)
        #     return x_updated
        #
        # def compiled_fn(x)
        #     x_updated = traced_forward(x)
        #     x.copy_(x_updated)
        #     return x.view(-1)
        #
        # Why can't we return x_updated.view(-1) to the user?
        # It can have different metadata from x.view(-1)! Specifically, the input x could be a non-memory-dense tensor,
        # But the intermediate created by our graph, x_updated, will always be memory-dense.
        def filter_and_record_aliased_outs(outputs):
            # NOTE: this dict will clobber keys if we have multiple inputs that alias.
            # Let's say inpA and inpB alias, and the user generated an output using out = inpA.view(...)
            # For now, since we're not handling the case with multiple _base's sharing a storage,
            # it is actually fine to arbitrarily pick which input to regenerate the aliased output from.
            # e.g. out_new = inpB.as_strided(out.size(), out.stride(), out.storage_offset())
            #
            # This will be more complicated when you have multiple _base tensors aliasing the same
            # underlying storage, when we eventually handle that.
            # We'll need to ensure that we generate the view off of the right base.
            inp_storage_refs = {
                StorageWeakRef(inpt._storage()): idx for idx, inpt in enumerate(flat_f_args) if isinstance(inpt, torch.Tensor)}
            inp_tensor_ids = {id(inpt) for inpt in flat_f_args if isinstance(inpt, torch.Tensor)}
            inp_storage_refs_set = set(inp_storage_refs)

            non_aliased_input_outs = []
            # For a given output tensor that alias an input, tells us:
            # (1) the index of the input that we alias
            # (2) Whether or not the output is a view of the input, or if `output is input`
            #     (so we don't need to generate a view, and can return the input directly)
            # Note: if the function returns an output that *is* an input, we still cannot return it in the graph.
            # e.g.
            #   def f(x):
            #       x.add_(1)
            #       return x
            # Our compiled fw will return an "x_updated", but it is *not* ok to return that to the user.
            # We need to manually do x.copy_(x_updated), and return the original x to the user.
            # Why? for example, the metadata between x and x_updated might be different (e.g. _is_leaf())
            aliased_out_idx: Dict[torch.Tensor, Tuple[int, bool]] = {}

            for o in outputs:
                # Note: When detecting input/output aliasing, we NEED to do it using the outer FunctionalTensorWrapper objects.
                # In the case where we mutate an input *and* return a view of it, the outer wrappers will still alias,
                # but the inner tensors no longer alias.
                if isinstance(o, torch.Tensor) and StorageWeakRef(o._storage()) in inp_storage_refs:
                    aliased_inp_idx = inp_storage_refs[StorageWeakRef(o._storage())]
                    is_exact_input = id(o) in inp_tensor_ids
                    aliases_intermediate_and_not_input = False
                    aliased_out_idx[o] = (
                        aliased_inp_idx,
                        aliases_intermediate_and_not_input,
                        is_exact_input,
                    )
                else:
                    # Only return outputs that are not aliases of inputs.
                    non_aliased_input_outs.append(o)
            # If a function involves creating a tensor, and returning a view of it, such that its _base is the intermediiate,
            # We need to make sure our graph returns the _base as a graph output, and we manually recreate the view
            # to return to the user. Why? The backend compiler is free to (incorrectly) not set requires_grad
            # on the base tensor, but we are obligated to properly set requires-gradness on the real output.
            non_aliased_outs = []
            for i, o in enumerate(non_aliased_input_outs):
                non_aliased_outs.append(o)

            return non_aliased_outs, aliased_out_idx

        non_aliased_outs, aliased_out_to_inp_idx = filter_and_record_aliased_outs(outs)

        pytree.tree_map(collect_grad_info, non_aliased_outs)

        # Calling convention: the output is (mutated_input_values, original_outs)
        # We return all mutated inputs + outputs here, **except** for any mutated inputs or outputs
        # that alias original inputs.
        # See Note [Input mutation handling in aot autograd]
        mutated_inps_and_outs = inputs_with_mutated_data + list(non_aliased_outs)

        # Our compiled forward function will return:
        # (1) non-aliased updated inputs
        # (2) non-aliased fw outputs
        # (3) size/stride/storage_offset metadata for updated aliased inputs
        # (4) size/stride/storage_offset metadata for aliased outputs

        start_idx_for_aliased_output_metadata = 0

        # First, gather the metadata info on mutated inputs (this only applies to inputs with metadata-only mutations))
        for i, maybe_aliased_updated_inp in enumerate(
            maybe_inputs_with_mutated_metadata
        ):
            if maybe_aliased_updated_inp is None:
                collect_metadata_mutation_input_info.append(None)
                continue
            # Figure out where the sizes/strides/storage_offset are in the compiled fw output.
            sizes_idx = start_idx_for_aliased_output_metadata
            strides_idx = sizes_idx + len(maybe_aliased_updated_inp.size())
            storage_offset_idx = strides_idx + len(maybe_aliased_updated_inp.stride())
            # update our offset for the next tensor
            start_idx_for_aliased_output_metadata = storage_offset_idx + 1
            inp_info = InputAliasInfo(
                base_idx=i,
                sizes_idx=sizes_idx,
                strides_idx=strides_idx,
                storage_offset_idx=storage_offset_idx,
                tensor_meta=maybe_aliased_updated_inp,
            )
            collect_metadata_mutation_input_info.append(inp_info)

        # Next, gather the metadata info on the user's outputs that alias (either inputs or graph outputs)
        num_non_input_aliased_outputs = 0
        for o in outs:
            maybe_alias_info = (
                aliased_out_to_inp_idx.get(o, None)
                if isinstance(o, torch.Tensor)
                else None
            )
            if maybe_alias_info is None:
                output_type = OutputType.non_alias
                # Here, alias_idx will tell us which output from the inner forward this corresponds to.
                alias_idx = num_non_input_aliased_outputs
                sizes_idx = None
                strides_idx = None
                storage_offset_idx = None
                tensor_meta = None
            else:
                (
                    input_alias_idx,
                    is_alias_of_intermediate_not_input,
                    is_exact_input,
                ) = maybe_alias_info
                if is_exact_input:
                    assert not is_alias_of_intermediate_not_input
                    output_type = OutputType.alias_of_input
                    alias_idx = input_alias_idx
                    sizes_idx = None
                    strides_idx = None
                    storage_offset_idx = None
                    tensor_meta = None
                else:
                    if is_alias_of_intermediate_not_input:
                        output_type = OutputType.alias_of_intermediate
                        alias_idx = num_non_input_aliased_outputs
                    else:
                        output_type = OutputType.alias_of_input
                        alias_idx = input_alias_idx
                    tensor_meta = o
                    # Figure out where the sizes/strides/storage_offset are in the compiled fw output.
                    sizes_idx = start_idx_for_aliased_output_metadata
                    strides_idx = sizes_idx + len(tensor_meta.size())
                    storage_offset_idx = strides_idx + len(tensor_meta.stride())
                    # update our offset for the next tensor
                    start_idx_for_aliased_output_metadata = storage_offset_idx + 1

            if output_type != OutputType.alias_of_input:
                num_non_input_aliased_outputs += 1

            inp_info = OutputAliasInfo(
                output_type=output_type,
                base_idx=alias_idx,
                sizes_idx=sizes_idx,
                strides_idx=strides_idx,
                storage_offset_idx=storage_offset_idx,
                tensor_meta=tensor_meta,
            )
            collect_aliased_output_info.append(inp_info)

        # This is the total number of size/stride/storage_offset metadata outputs that we return in the forward,
        # used for regenerating aliases later.
        num_aliasing_metadata_outs = start_idx_for_aliased_output_metadata

        assert len(collect_metadata_mutation_input_info) == len(
            collect_mutated_input_info
        )

        assert len(
            [x for x in collect_metadata_mutation_input_info if x is not None]
        ) == len(
            [x for x in collect_mutated_input_info if x == MutationType.metadata_only]
        )
        assert len(collect_aliased_output_info) == len(outs)
        assert len(
            [
                x
                for x in collect_aliased_output_info
                if x.output_type != OutputType.alias_of_input
            ]
        ) == len(non_aliased_outs)

        # Our autograd.Function.forward returns both mutated inputs and outputs,
        # so we need grad info on all of them.
        assert len(collect_requires_grad_out_info) == len(mutated_inps_and_outs)

        metadata = ViewAndMutationMeta(
            mutated_input_info=collect_mutated_input_info,
            metadata_mutation_input_info=collect_metadata_mutation_input_info,
            requires_grad_out_info=collect_requires_grad_out_info,
            aliased_output_info=collect_aliased_output_info,
        )
        return (
            metadata,
            pytree.tree_map(from_fun, mutated_inps_and_outs),
            num_aliasing_metadata_outs,
        )

    return inner


# This creates a functionalized joint forwards-backwards function given both
# the primals (to run forwards) and tangents (to run backwards).
#
# It uses the metadata that was created earlier to figure out what all of the outputs to the autograd.Function.forward are:
# (1) Which inputs received data mutations (and need to be passed as outputs into autograd.grad())
# (2) Which outputs are aliases of inputs (and should *not* be passed as outputs into autograd.grad())
def create_joint_forward_backward_functionalized(
    fn,
    *,
    meta: ViewAndMutationMeta,
    synthetic_base_info: Optional[List[Union[int, Tuple[int, List[Any]]]]],
):
    # NOTE: when we have synthetic base inputs, we need to clone them *before* creating views off of them.
    # This means that "idx" here represents the index of the (potentially) synthetic base.
    # What we need to do is:
    # (1) map the current (post-synthetic-base calling convention) input argument index
    #     to int index pre-synthetic-base-calling-convention.
    # (2) There could be multiple, if this index corresponds to a synthetic base
    #     that has multiple input aliases.
    # (3) If any of those corresponding inputs get metadata mutations, then we clone the base.
    def maybe_to_fresh_input(idx, t):
        if not isinstance(t, Tensor):
            return t

        if synthetic_base_info is None:
            outer_aliased_indices_of_current_base_arg = [idx]
        else:
            outer_aliased_indices_of_current_base_arg = [
                # For every argument index in the outer calling convention (before synthetic bases)
                # find its index in the inner calling convention.
                # if it matches the index of our current arg (idx), track the outer argument's index (i)
                i
                for i, outer_idx_or_lambda in enumerate(synthetic_base_info)
                if (isinstance(outer_idx_or_lambda, int) and outer_idx_or_lambda == idx)
                or (
                    isinstance(outer_idx_or_lambda, tuple)
                    and outer_idx_or_lambda[0] == idx
                )
            ]
        if any(
            meta.mutated_input_info[i] == MutationType.data
            for i in outer_aliased_indices_of_current_base_arg
        ):
            # Make sure the primal we pass to autograd.grad()
            # seees the tensor before the mutation
            out = t.clone()
        elif any(
            meta.mutated_input_info[i] == MutationType.metadata_only
            for i in outer_aliased_indices_of_current_base_arg
        ):
            # Make sure the primal we pass to autograd.grad()
            # seees the tensor before the metadata mutation
            out = t.view(t.shape)
        else:
            out = t
        return out

    def unpack_synthetic_bases(primals: List[Any]) -> List[Any]:
        # This is only not None if our graph mutates a graph input that aliases another graph input.
        if synthetic_base_info is None:
            return primals

        f_args_inner = []
        for outer_idx_or_lambda in synthetic_base_info:
            if isinstance(outer_idx_or_lambda, int):
                f_args_inner.append(primals[outer_idx_or_lambda])
            else:
                outer_base_idx, strided_args = outer_idx_or_lambda
                outer_base = primals[outer_base_idx]
                # TODO: we could consider storing and executing view replay logic here,
                # instead of a general as_strided() call.
                # This could also improve perf, since today this will cause
                # more as_strided_scatter() ops in the graph.
                view_arg = outer_base.as_strided(*strided_args)
                f_args_inner.append(view_arg)
        return f_args_inner

    def joint_forward_backward(
        primals: List[Any], tangents: List[Any]
    ) -> Tuple[List[Any], List[Any]]:
        # Call the forward pass, making sure to clone any inputs that are mutated first.
        # We need to ensure that the inputs we pass to autograd.grad() are the *original*
        # inputs, and not their mutated values.
        primals_no_input_mutations = [
            maybe_to_fresh_input(i, t) for i, t in enumerate(primals)
        ]
        # This is also where we handle the calling convention around synthetic bases.
        # We need to make sure that we convert any synthetic base arguments into views
        # *after* we do the cloning above, to preserve the view relationship.
        primals_ = unpack_synthetic_bases(primals_no_input_mutations)
        assert len(meta.mutated_input_info) == len(primals_)
        all_outs = fn(*primals_)
        assert len(meta.aliased_output_info) == len(all_outs)

        # Pass any (non-aliased) outputs in as tangents, since they'll be returned as outputs in the fw
        # For outputs that are aliases of intermediates, we will have returned the output's _base as an output in the graph instead,
        # which we *should* send to grad()
        outputs_for_grad = [
            x
            # TODO: support ._base
            # x._base if meta.aliased_output_info[i].output_type == OutputType.alias_of_intermediate else x
            for (i, x) in enumerate(all_outs)
            if meta.aliased_output_info[i].output_type != OutputType.alias_of_input
        ]
        # Pass any (non-aliased) mutated inputs in as tangents, since they'll be returned as outputs in the fw
        # Important: the traced joint fw/bw will return updated inputs with data mutations,
        # but *not* with metadata mutations.
        # Instead, we shunt the updated metadata around externally
        # and update the input's metadata outside of the autograd.Function
        mutated_inputs_for_grad = [
            x
            for (i, x) in enumerate(primals_)
            if meta.mutated_input_info[i] == MutationType.data
        ]
        mutated_inputs_and_outs_to_grad = mutated_inputs_for_grad + outputs_for_grad

        metadata_mutated_inps = [
            x
            for (i, x) in enumerate(primals_)
            if meta.mutated_input_info[i] == MutationType.metadata_only
        ]
        # for user outputs that are aliases (either of inputs, or of graph intermediates)
        # figure out what metadata to return in the forward, which is needed to regenerate the output aliases
        aliased_outs = [
            x
            for (i, x) in enumerate(all_outs)
            if meta.aliased_output_info[i].output_type != OutputType.non_alias
            and meta.aliased_output_info[i].tensor_meta is not None
        ]
        output_metadata_for_fw = []
        for curr_alias in metadata_mutated_inps + aliased_outs:
            size_ = curr_alias.size()
            stride_ = curr_alias.stride()
            storage_offset_ = curr_alias.storage_offset()
            # FX IR doesn't know about tuples, so we flatten the metadata into individual ints/symints,
            # and index into the final output list later.
            output_metadata_for_fw += size_ + stride_ + (storage_offset_,)

        # Take care to grab and sync the updated inputs from primals_ (the inputs we actually mutate!)
        # and not primals (the preserved inputs, pre-mutation, that we pass to grad())
        for i, arg in enumerate(primals_):
            if not isinstance(arg, Tensor):
                continue
            torch._sync(arg)

        # Get the inputs that need gradients
        grad_primals = []
        inputs_needs_grads = []
        # Note that we're not using primals_ here, being carefully not to pass any mutated inputs into autograd.grad()
        for p in primals:
            is_grad_tensor = isinstance(p, Tensor) and p.requires_grad
            inputs_needs_grads.append(is_grad_tensor)
            if is_grad_tensor:
                grad_primals.append(p)

        # Get the outputs that need gradients
        assert len(tangents) == len(mutated_inputs_and_outs_to_grad)
        needed_outs = []
        needed_tangents = []
        for out, tangent in zip(mutated_inputs_and_outs_to_grad, tangents):
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
            with fx_traceback.override_stack_trace():
                backward_out = torch.autograd.grad(
                    needed_outs,
                    grad_primals,
                    grad_outputs=needed_tangents,
                    allow_unused=True,
                )
        backward_out_iter = iter(backward_out)
        all_fw_outs = mutated_inputs_and_outs_to_grad + output_metadata_for_fw
        return all_fw_outs, [
            next(backward_out_iter) if i else None for i in inputs_needs_grads
        ]

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

    def functionalized_joint(
        primals: List[Any], tangents: List[Any]
    ) -> Tuple[List[Any], List[Any]]:

        # Wrap inputs into functional wrappers
        f_primals, f_tangents = pytree.tree_map(to_fun, (primals, tangents))
        torch._enable_functionalization(reapply_views=True)
        try:
            # Run the joint
            outs = joint_forward_backward(f_primals, f_tangents)
        finally:
            torch._disable_functionalization()

        # Syncing of inputs/outputs was already done directly in the joint call
        return pytree.tree_map(from_fun, outs)

    return functionalized_joint


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


def aot_dispatch_base(flat_fn, flat_args: List[Tensor], aot_config: AOTConfig):
    fw_module = make_fx(flat_fn, aot_config.decompositions)(*flat_args)
    if config.debug_graphs:
        log.debug("====== Forward (only) graph {aot_config.aot_id} ======")
        log.debug(fw_module.print_readable(print_output=False))

    disable_amp = torch._C._is_any_autocast_enabled()
    context = disable_autocast_manager if disable_amp else nullcontext

    with context(), track_graph_compiling(aot_config, "inference"):
        compiled_fw = aot_config.fw_compiler(fw_module, flat_args)

    @wraps(compiled_fw)
    def new_fn(args):
        fw_outs = call_func_with_args(compiled_fw, args, disable_amp=disable_amp)
        return fw_outs
    new_fn._boxed_call = True

    return new_fn


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
#   b_base = torch.Tensor(b.storage())
#   c_base = torch.Tensor(c.storage())
#   f(c_base, b_base, a, d)
def merge_view_inputs(
    fwd_inputs: List[Any], mutated_input_info: List[MutationType]
) -> Tuple[List[Any], Optional[List[Union[int, Tuple[int, Tuple[Any]]]]]]:
    assert len(fwd_inputs) == len(mutated_input_info)
    storage_ref_to_idx: Dict[StorageWeakRef, List[int]] = collections.defaultdict(list)
    base_args = []
    other_args = []
    for i, inpt in enumerate(fwd_inputs):
        if isinstance(inpt, Tensor):
            storage_ref = StorageWeakRef(inpt._storage())
            storage_ref_to_idx[storage_ref].append(i)
        else:
            other_args.append(inpt)
    # This list contains metadata that tells you what the i'th argument in the inner calling convention should be.
    # It's either:
    # - another int (corresponding to the index in the argument list of the element from the outer calling convention)
    # - idx, *args, where we can generate the new output with old_args[idx].as_strided(*args)
    #   idx corresponds to which synthetic base from the outer calling context to view
    inner_calling_convention_meta: Dict[int, Union[int, Tuple[int, List[Any]]]] = {}
    for aliased_input_indices in storage_ref_to_idx.values():
        if len(aliased_input_indices) > 1 and any(
            # We only care about mutations that affect all aliases,
            # so metadata mutations on an input doesn't require us to do synthetic base handling.
            mutated_input_info[inpt_idx] == MutationType.data
            for inpt_idx in aliased_input_indices
        ):
            # We detected an input that was mutated, AND aliases with another input.
            # we need to replace this set of aliased inputs with a single synthetic base.
            # For now, I'm banning a bunch of cases. We expect dynamo to properly detect these cases
            # and error out. We can fix them later.
            for idx1, idx2 in zip(aliased_input_indices, aliased_input_indices[1:]):
                view1 = fwd_inputs[idx1]
                view2 = fwd_inputs[idx2]
                # The "inputs that are aliased but have different differentiable bases" case
                # is more complicated and hopefully pretty rare. Not currently handled.
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
                fwd_inputs[i]
                for i in aliased_input_indices
                if fwd_inputs[i]._base is None
            ]
            if len(non_none_bases) == 0:
                # Case where none of the aliases require gradients
                example_idx = aliased_input_indices[0]
                synthetic_base = torch.Tensor(fwd_inputs[example_idx]._storage())
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
                size_ = curr_view.size()
                stride_ = curr_view.stride()
                storage_offset_ = curr_view.storage_offset()
                # We store just enough info here so that we can regenerate the view later.
                # Regeneration: args[base_idx].as_strided(size_, stride_, storage_offset_)
                # If we want view replay instead of as_strided() calls, this will need to change.
                inner_calling_convention_meta[curr_view_idx] = (
                    base_idx,
                    (size_, stride_, storage_offset_),
                )
        else:
            for curr_idx in aliased_input_indices:
                other_args.append(fwd_inputs[curr_idx])
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
def aot_wrapper_dedupe(flat_fn, flat_args: List[Tensor], aot_config: AOTConfig, *, compiler_fn):
    # Get information about whether or not flat_fn mutates its arguments
    # or not
    try:
        with enable_python_dispatcher():
            fw_metadata, _out, _num_aliasing_metadata_outs = run_functionalized_fw_and_collect_metadata(
                flat_fn
            )(*flat_args)
    except RuntimeError as e:
        log.warning(
            "Failed to collect metadata on function, produced code may be suboptimal.  "
            "Known situations this can occur are inference mode only compilation involving "
            "resize_ or prims (!schema.hasAnyAliasInfo() INTERNAL ASSERT FAILED); "
            "if your situation looks different please file a bug to PyTorch.",
            exc_info=True
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
            elif fw_metadata.mutated_input_info[i] == MutationType.none:
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
            # Edge case, only happens for identity
            if dupe_arg_pos != kept_pos:
                tracing_context.guards_context.aotautograd_guards.append(DuplicateInputs(kept_pos, dupe_arg_pos))

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
                f"{describe_input(add_dupe_map[i], aot_config)}"
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


# Has the precondition that there
# are no duplicate arguments in flat_args (e.g., the same Tensor
# object never shows up twice.  However, two tensor inputs MAY alias
# the same storage, so long as they have separate TensorImpls.)
def aot_dispatch_autograd(flat_fn, flat_args: List[Tensor], aot_config: AOTConfig):

    with enable_python_dispatcher():
        (
            _fw_metadata,
            out,
            _num_aliasing_metadata_outs,
        ) = run_functionalized_fw_and_collect_metadata(flat_fn)(*flat_args)

    # pre-compute, so we can bail out quickly in the hotpath
    _num_outputs_aliased_to_inputs = len(
        [
            x
            for x in _fw_metadata.aliased_output_info
            if x.output_type == OutputType.alias_of_input
        ]
    )
    _num_outputs_aliased_to_intermediates = len(
        [
            x
            for x in _fw_metadata.aliased_output_info
            if x.output_type == OutputType.alias_of_intermediate
        ]
    )
    _num_mutated_data_inputs = len(
        [x for x in _fw_metadata.mutated_input_info if x == MutationType.data]
    )
    _num_mutated_metadata_only_inputs = len(
        [x for x in _fw_metadata.metadata_mutation_input_info if x is not None]
    )
    _num_mutated_inputs = _num_mutated_data_inputs + _num_mutated_metadata_only_inputs


    if isinstance(out, (list, tuple)):
        _num_non_aliased_outs = len(out[_num_mutated_data_inputs:])
    else:
        _num_non_aliased_outs = 1
    assert (
        len(_fw_metadata.requires_grad_out_info)
        == _num_mutated_data_inputs + _num_non_aliased_outs
    )

    # out here corresponds to the set of outputs that should be returned by the traced forward call.
    # It includes outputs of the original forward, *and* any updated inputs due to input mutations.
    # However, it does *not* include any outputs that are aliases of inputs, or any metadata-only input mutations.
    out = pytree.tree_map(
        lambda x: x.detach().contiguous() if isinstance(x, Tensor) else x,
        out,
    )

    # This code only executes if we have graph inputs that alias each other, and one of those inputs
    # gets its data mutated.
    # When that happens, we replace the aliased inputs with a synthetic base, and in the traced forward
    # we later generate the input views
    flat_args_with_views_handled, _synthetic_base_info = merge_view_inputs(
        flat_args, _fw_metadata.mutated_input_info
    )

    joint_forward_backward = create_joint_forward_backward_functionalized(
        flat_fn,
        meta=_fw_metadata,
        synthetic_base_info=_synthetic_base_info,
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
            num_inner_fwd_outputs = (
                _num_mutated_data_inputs
                + _num_non_aliased_outs
                + _num_aliasing_metadata_outs
            )
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
            log.debug("====== Forward graph {aot_config.aot_id} ======")
            log.debug(fw_module.print_readable(print_output=False))

        with track_graph_compiling(aot_config, "forward"):
            compiled_fw_func = aot_config.fw_compiler(
                fw_module, flat_args_with_views_handled
            )

    class CompiledFunction(torch.autograd.Function):
        compiled_fw = compiled_fw_func
        compiled_bw = None
        # Corresponds to number of outs (not including updated inputs returns as outs),
        # *and* not including outs that are aliases of inputs
        num_non_aliased_outs = _num_non_aliased_outs
        num_symints_saved_for_bw = _num_symints_saved_for_bw
        # Corresponds to number of inputs that are mutated (both metadata only, and data)
        num_mutated_inputs = _num_mutated_inputs
        # Corresponds to number of inputs that only have their metadata mutated
        num_mutated_data_inputs = _num_mutated_data_inputs
        # Corresponds to number of inputs that get their metadata (but not data) mutated
        # We don't return these in the compiled fw, and instead we stash enough info
        # to replay the metadata mutations later.
        num_mutated_metadata_only_inputs = _num_mutated_metadata_only_inputs
        # Corresponds to number of outputs in the original fw that are aliases of inputs
        # (These are all not returned by the compiled forward, and instead they are manually
        # created in the epilogue)
        num_outputs_aliased_to_inputs = _num_outputs_aliased_to_inputs
        # Corresponds to the number of user outputs that alias intermediates (aka graph outputs).
        num_outputs_aliased_to_intermediates = _num_outputs_aliased_to_intermediates
        # For every output that aliases and input, and every input that gets only its metadata mutated,
        # we return that tensor's size/stride/storage_offset directly at the end of the compiled forward,
        # as a big list of ints.
        # The number is tracked here.
        num_aliasing_metadata_outs = _num_aliasing_metadata_outs
        synthetic_base_info = _synthetic_base_info
        fw_metadata = _fw_metadata

        @staticmethod
        def forward(ctx, *deduped_flat_tensor_args):

            # There is a pretty complicated calling convention around what the compiled fw returns.
            # The full list of outputs and their relative order is:
            # (*mutated_data_inputs, *non_aliased_fw_outs, *saved_tensors, *saved_symints)
            # - Note that in the synthetic bases case, mutated_inputs will correspond to an updated version
            #   of the original view, and not the synthetic base
            fw_outs = call_func_with_args(
                CompiledFunction.compiled_fw,
                deduped_flat_tensor_args,
                disable_amp=disable_amp,
            )

            num_non_aliased_outs = CompiledFunction.num_non_aliased_outs
            num_aliasing_metadata_outs = CompiledFunction.num_aliasing_metadata_outs
            num_symints_saved_for_bw = CompiledFunction.num_symints_saved_for_bw
            num_mutated_data_inputs = CompiledFunction.num_mutated_data_inputs
            # Our forward() returns both (mutated_inputs, outputs, output_alias_meta, saved_tensors, saved_symints)
            num_forward_returns = (
                num_mutated_data_inputs
                + num_non_aliased_outs
                + num_aliasing_metadata_outs
            )
            num_forward_returns_not_including_alias_meta = (
                num_mutated_data_inputs + num_non_aliased_outs
            )

            # Partitioners must put symint arguments at the end separate from tensor arguments
            if num_symints_saved_for_bw > 0:
                tensors_saved_for_backwards = fw_outs[
                    num_forward_returns:-num_symints_saved_for_bw
                ]
                assert all(
                    [isinstance(x, torch.Tensor) for x in tensors_saved_for_backwards]
                )
                ctx.save_for_backward(*tensors_saved_for_backwards)
                symint_outs = fw_outs[-num_symints_saved_for_bw:]
                assert all(
                    [
                        isinstance(x, (int, float, torch.SymInt, torch.SymFloat))
                        for x in symint_outs
                    ]
                )
                ctx.symints = symint_outs
            else:
                ctx.save_for_backward(*fw_outs[num_forward_returns:])
                ctx.symints = []

            fw_outs_not_requiring_grad = [
                x
                for (i, x) in enumerate(
                    fw_outs[:num_forward_returns_not_including_alias_meta]
                )
                if isinstance(x, torch.Tensor)
                and not CompiledFunction.fw_metadata.requires_grad_out_info[i]
            ]
            fw_out_ids_requiring_grad = [
                id(x)
                for (i, x) in enumerate(
                    fw_outs[:num_forward_returns_not_including_alias_meta]
                )
                if isinstance(x, torch.Tensor)
                and CompiledFunction.fw_metadata.requires_grad_out_info[i]
            ]

            ctx.mark_non_differentiable(*fw_outs_not_requiring_grad)

            return tuple(fw_outs[0:num_forward_returns])

        @staticmethod
        def backward(ctx, *all_flat_args):
            # Calling convention: we expect a grad_out passed to the backward:
            # - for every output of the fw that does *not* alias an input
            # - for every updated_input generated by the fw that does *not* alias an input
            # - for every size/stride metadata value for aliased outputs.
            #   These are returned by the forward, but we just drop them in the backward.
            #   We need to return them in the forward, but unfortunately there's no way to specify
            #   in autograd.Function that certain non-tensor forward outputs shouldn't show up in the backward.
            expected_grad_outs = (
                CompiledFunction.num_non_aliased_outs
                + CompiledFunction.num_mutated_data_inputs
            )
            if CompiledFunction.num_aliasing_metadata_outs > 0:
                flat_args = all_flat_args[
                    : -CompiledFunction.num_aliasing_metadata_outs
                ]
                metadata_args = all_flat_args[
                    -CompiledFunction.num_aliasing_metadata_outs :
                ]
                # metadata args are all ints/symints, which autograd will send Nones for as grad_outputs in the bw
                assert all([x is None for x in metadata_args])
                # delete
                # for out_idx, (base_sizes, base_strides, base_storage_offset) in CompiledFunctions.fw_out_base_metadata.items():

            else:
                flat_args = all_flat_args

            assert len(flat_args) == expected_grad_outs
            contiguous_args = [
                t.contiguous() if torch.is_tensor(t) else t for t in flat_args
            ]
            all_args = (
                list(ctx.symints) + list(ctx.saved_tensors) + list(contiguous_args)
            )
            del contiguous_args
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

    @wraps(CompiledFunction.apply)
    def compiled_function(*args):
        # Step 2: remove aliased inputs that are mutated, replace with synthetic bases
        # Only happens if our graph mutates an input that aliases another input.
        if CompiledFunction.synthetic_base_info is not None:
            # Given: the original args, including at least one pair of inputs that are aliased
            # and get subsequently mutated.
            # Generate: the updated args, including (potentially multiple) synthetic bases
            # that replace the views. The input views are regenerated manually in the compiled function.
            # TODO: think harder about what happens if (a view of) one of these mutated input views is ALSO returned
            new_inputs, metadata = merge_view_inputs(
                args, CompiledFunction.fw_metadata.mutated_input_info
            )
            # We're just re-running the original-args-to-synthetic-base transformation
            # that we ran during compilation.
            # This returns metadata that we use during tracing to recover the input views,
            # which we don't actually need at runtime.
            assert metadata is not None
            args_with_synthetic_bases = new_inputs
        else:
            args_with_synthetic_bases = args

        all_outs = CompiledFunction.apply(*args_with_synthetic_bases)
        if CompiledFunction.num_aliasing_metadata_outs > 0:
            outs = all_outs[: -CompiledFunction.num_aliasing_metadata_outs]
            aliasing_metadata_outs = all_outs[
                -CompiledFunction.num_aliasing_metadata_outs :
            ]
        else:
            outs = all_outs
            aliasing_metadata_outs = []

        assert (
            len(all_outs)
            == CompiledFunction.num_mutated_data_inputs
            + CompiledFunction.num_non_aliased_outs
            + CompiledFunction.num_aliasing_metadata_outs
        )

        # Step 3: After running the compiled fw, apply updates to mutated inputs
        if CompiledFunction.num_mutated_inputs > 0:
            # Calling convention: (mutated_inputs, real_outs, aliasing_metadata)

            if CompiledFunction.num_mutated_data_inputs > 0:
                updated_inputs = outs[: CompiledFunction.num_mutated_data_inputs]
                fw_outs = outs[CompiledFunction.num_mutated_data_inputs :]
            else:
                updated_inputs = []
                fw_outs = outs

            curr_mutated_inpt_idx = 0
            for inpt_idx, (mutation_type, metadata_mutation_info) in enumerate(
                zip(
                    # TODO: I should merge these two pieces of state
                    CompiledFunction.fw_metadata.mutated_input_info,
                    CompiledFunction.fw_metadata.metadata_mutation_input_info,
                )
            ):
                if mutation_type == MutationType.none:
                    continue
                original_inpt = args[inpt_idx]
                if mutation_type == MutationType.metadata_only:
                    # We need to grab the size/stride/storage_offset from the compiled forward,
                    # and use that to mutate the metadata of the input
                    expected_meta = (
                        CompiledFunction.fw_metadata.metadata_mutation_input_info[
                            inpt_idx
                        ]
                    )
                    assert expected_meta is not None
                    fake_meta = expected_meta.tensor_meta
                    size_len = len(fake_meta.size())
                    stride_len = len(fake_meta.stride())
                    size_ = aliasing_metadata_outs[
                        expected_meta.sizes_idx : expected_meta.sizes_idx + size_len
                    ]
                    stride_ = aliasing_metadata_outs[
                        expected_meta.strides_idx : expected_meta.strides_idx
                        + stride_len
                    ]
                    storage_offset_ = aliasing_metadata_outs[
                        expected_meta.storage_offset_idx
                    ]
                    original_inpt.as_strided_(size_, stride_, storage_offset_)
                else:
                    updated_inpt = updated_inputs[curr_mutated_inpt_idx]
                    curr_mutated_inpt_idx += 1
                    # TODO: handle resize_() on inputs to a larger size.
                    # This is actually non-trivial to detect, so we should probably just handle it
                    # (or make dynamo detect).
                    # We can't just check of original_inpt.storage_size != updated_inpt.storage_size,
                    # Because the original_inpt might be a view of some larger tensor,
                    # and updated_inpt is always densely packed.
                    if (
                        original_inpt.size() != updated_inpt.size()
                        or original_inpt.stride() != updated_inpt.stride()
                        or original_inpt.storage_offset()
                        != updated_inpt.storage_offset()
                    ):
                        # Functionalization can't easily tell us if an input had BOTH its metadata actual data mutated.
                        # So we check if metadata needs to be mutated here manually.
                        original_inpt.as_strided_(
                            updated_inpt.size(),
                            updated_inpt.stride(),
                            updated_inpt.storage_offset(),
                        )
                    original_inpt.copy_(updated_inpt)
        else:
            fw_outs = outs

        # Step 4: Manually regenerate any outputs that are aliased to inputs, instead of
        # compiling them.
        if (
            CompiledFunction.num_outputs_aliased_to_inputs > 0
            or CompiledFunction.num_outputs_aliased_to_intermediates > 0
        ):
            assert CompiledFunction.num_outputs_aliased_to_inputs + len(fw_outs) == len(
                CompiledFunction.fw_metadata.aliased_output_info
            )
            fw_outs_including_aliases = []
            for (
                aliased_out_metadata
            ) in CompiledFunction.fw_metadata.aliased_output_info:
                if aliased_out_metadata.output_type == OutputType.non_alias:
                    fw_outs_including_aliases.append(
                        fw_outs[aliased_out_metadata.base_idx]
                    )
                else:
                    if aliased_out_metadata.output_type == OutputType.alias_of_input:
                        aliased_base_tensor = args[aliased_out_metadata.base_idx]
                    else:
                        assert (
                            aliased_out_metadata.output_type
                            == OutputType.alias_of_intermediate
                        )
                        aliased_base_tensor = fw_outs[aliased_out_metadata.base_idx]
                    # Note: here, we manually regenerate the output, using an as_strided() call,
                    # OR if the aliased output came from a custom autograd.function, we replay it.
                    # The as_strided() in the normal case is good for perf (this is hot-path code,
                    # and we're consolidating potential chains of views into a single view op).
                    # But we might need to figure out view replaying for e.g. XLA.
                    # TODO: handle the custom autograd function case here.
                    # We need a way to check whether a tensor came from a custom autograd fn from python,
                    # AND a way to replay that custom view fn.
                    fake_meta = aliased_out_metadata.tensor_meta
                    if fake_meta is None:
                        # This handles the specific case where the user returns an output that *was* an input. Don't create a view.
                        fw_outs_including_aliases.append(aliased_base_tensor)
                    else:
                        # We need to grab the size/stride/storage_offset from the compiled forward,
                        # and use that to create a view off of the right input
                        fake_meta = aliased_out_metadata.tensor_meta
                        size_len = len(fake_meta.size())
                        stride_len = len(fake_meta.stride())
                        size_ = aliasing_metadata_outs[
                            aliased_out_metadata.sizes_idx : aliased_out_metadata.sizes_idx
                            + size_len
                        ]
                        stride_ = aliasing_metadata_outs[
                            aliased_out_metadata.strides_idx : aliased_out_metadata.strides_idx
                            + stride_len
                        ]
                        storage_offset_ = aliasing_metadata_outs[
                            aliased_out_metadata.storage_offset_idx
                        ]
                        # Create the output alias
                        aliased_out = gen_alias_from_base(
                            aliased_base_tensor,
                            size_,
                            stride_,
                            storage_offset_,
                            fake_meta,
                        )
                        fw_outs_including_aliases.append(aliased_out)

            for inner_out, user_out in zip(fw_outs, fw_outs_including_aliases):
                # Sanity check assert
                assert type(inner_out) == type(user_out)
            return fw_outs_including_aliases
        else:
            return fw_outs

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
    flat_fn, flat_args: List[Tensor], aot_config: AOTConfig
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

    if config.use_dynamic_shapes:
        assert config.use_fake_tensor, "Dynamic shapes only works with fake tensor"

    # Check flat_args to see if they're already fake.  If so, use that fake
    # mode instead.

    for x in flat_args:
        if isinstance(x, FakeTensor):
            fake_mode = x.fake_mode
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
        enable_python_dispatcher() if config.use_dynamic_shapes else nullcontext()
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

        fake_flat_tensor_args = process_inputs(flat_args)

        needs_autograd = (
            any(
                [
                    x.requires_grad
                    for x in fake_flat_tensor_args
                    if isinstance(x, Tensor)
                ]
            )
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

        compiled_fn = compiler_fn(flat_fn, fake_flat_tensor_args, aot_config)

        if not hasattr(compiled_fn, '_boxed_call'):
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
        return stateless.functional_call(mod, params_and_buffers, args, kwargs)

    named_params = dict(_named_parameters(mod, remove_duplicate=False))
    named_buffers = dict(_named_buffers(mod, remove_duplicate=False))
    num_params_buffers = len(named_params) + len(named_buffers)
    compiled_f = aot_function(
        functional_call, num_params_buffers=num_params_buffers, *args, **kwargs
    )

    class AOTModule(nn.Module):
        def __init__(self):
            super(AOTModule, self).__init__()
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
        **dict(_named_parameters(mod, remove_duplicate=False)),
        **dict(_named_buffers(mod, remove_duplicate=False)),
    }
    params_flat, params_spec = pytree.tree_flatten(params)
    params_flat = tuple(params_flat)
    params_len = len(params_flat)

    def functional_call(*args, **kwargs):
        with stateless._reparametrize_module(
            mod, pytree.tree_unflatten(args[:params_len], params_spec)
        ):
            if isinstance(mod, torch.fx.GraphModule):
                with fx_traceback.override_stack_trace(), warnings.catch_warnings():
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

    return forward


compiled_function = aot_function
compiled_module = aot_module
