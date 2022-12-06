import collections
import dataclasses
import warnings
import itertools
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from torch.fx.experimental.proxy_tensor import is_sym_node

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

MutationType = Enum("MutationType", ("none", "metadata_only", "data"))
OutputType = Enum("OutputType", ("non_alias", "alias_of_input", "alias_of_intermediate"))

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
    # (3) an alias of an intermediate (aka an alias of an output of the inner traced forward)
    output_type: OutputType
    # If (1) above, then
    # - Tells us that this output corresponds directly to traced_fw_outputs[base_idx]
    # If (2) above, then
    # - Tells us that the base of this alias is user_fwd_input[base_idx]
    #   (This is an index into the inputs *before* we make synthetic bases)
    # If (3) above, then
    # - Tells us that the base of this alias is traced_fwd_outputs[base_idx]
    #   here, this refers to the index of the *direct* traced
    output_type: OutputType
    base_idx: int
    # Only important when output_type == OutputType.alias_of_input
    is_input_tensor: bool

# This class tells us about how to perform a metadata mutation on forward inputs.
# it only applies to forward inputs that experience metadata-only mutations
@dataclass(frozen=True)
class InputAliasInfo:
    mutation_type: MutationType

# This class encapsulates all aliasing + mutation info we need about the forward graph
# See a more detailed overview of the edge case handling at
# https://docs.google.com/document/d/19UoIh_SVrMy_b2Sx5ZaeOJttm6P0Qmyss2rdBuyfoic/edit
@dataclass(frozen=True)
class ViewAndMutationMeta:
    input_info: List[InputAliasInfo]
    output_info: List[OutputAliasInfo]
    # length: # outputs in the compiled forward. Equal to:
    # len(input_info) + len(output_info)
    # For every output *and* mutated input returned from the forward,
    # tells us whether or not the output should require gradients or not
    requires_grad_info: List[bool]

# This class exists because:
# - the autograd.Function.forward() in aot autograd returns outputs that might alias inputs
# - we only care about the metadata on those aliases, so we can regenerate them.
#   We do not want them to participate in the autograd.Function.
# We do that by wrapping them in an opaque class, so the autograd.Function
# does not know to treat them as tensors.
@dataclass(frozen=True)
class TensorAlias:
    alias: torch.Tensor


def gen_alias_from_base(aliased_base_tensor, size, stride, storage_offset, target_meta_tensor):
    # handle R2C and C2R
    if aliased_base_tensor.is_complex() and not target_meta_tensor.is_complex():
        aliased_out = torch.view_as_real(aliased_base_tensor).as_strided(size, stride, storage_offset)
    elif not aliased_base_tensor.is_complex() and target_meta_tensor.is_complex():
        aliased_out = torch.view_as_complex(aliased_base_tensor).as_strided(size, stride, storage_offset)
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

    @wraps(f)
    def inner(*args):
        # This function is meant to be run with the forward, which expects a flat list of tensor/symint/other args.
        assert all(isinstance(a, torch.Tensor) or type(a) in KNOWN_TYPES for a in args)

        input_info: List[InputAliasInfo] = []
        output_info: List[OutputAliasInfo] = []
        input_requires_grad_info: List[bool] = []
        output_requires_grad_info: List[bool] = []

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
        # If inp[i] has a metadata-only mutation, then maybe_inputs_with_mutated_metadata[i] contains the updated version
        for (i, (arg, f_arg)) in enumerate(zip(flat_args, flat_f_args)):
            if isinstance(arg, Tensor):
                torch._sync(f_arg)
                new_arg = torch._from_functional_tensor(f_arg)
            else:
                new_arg = arg

            if arg is not new_arg:
                if StorageWeakRef(arg.storage()) == StorageWeakRef(new_arg.storage()):
                    mutation_type = MutationType.metadata_only
                else:
                    mutation_type = MutationType.data
                # Only track requires_grad info on *mutated* inputs,
                # because they show up in the autograd.Function.forward as outputs
                input_requires_grad_info.append(isinstance(f_arg, torch.Tensor) and f_arg.requires_grad)
            else:
                mutation_type = MutationType.none

            input_info.append(InputAliasInfo(mutation_type=mutation_type))

        # If a function involves creating a tensor, and returning a view of it, such that its _base is the intermediiate,
        # We need to make sure our graph returns the _base as a graph output, and we manually recreate the view
        # to return to the user. Why? The backend compiler is free to (incorrectly) not set requires_grad
        # on the base tensor, but we are obligated to properly set requires-gradness on the real output.

        inp_storage_refs = {StorageWeakRef(inpt.storage()): idx for idx, inpt in enumerate(flat_f_args)}
        inp_tensor_ids = {id(inpt) for inpt in flat_f_args if isinstance(inpt, torch.Tensor)}
        for o in outs:
            is_input_tensor = False
            if isinstance(o, torch.Tensor) and StorageWeakRef(o.storage()) in inp_storage_refs:
                output_type = OutputType.alias_of_input
                base_idx = inp_storage_refs[StorageWeakRef(o.storage())]
                is_exact_input = id(o) in inp_tensor_ids
            else:
                # TODO: check for aliases of intermediates here
                output_type = OutputType.non_alias
                base_idx = None

            out_info = OutputAliasInfo(
                output_type=output_type,
                base_idx=base_idx,
                is_input_tensor=is_input_tensor,
            )
            output_info.append(out_info)
            output_requires_grad_info.append(isinstance(o, torch.Tensor) and o.requires_grad)

        # Our autograd.Function.forward returns both mutated inputs and outputs,
        # so we need grad info on all of them.
        requires_grad_info = input_requires_grad_info + output_requires_grad_info
        assert len(requires_grad_info) == len(output_info) + len([x for x in input_info if x.mutation_type != MutationType.none])

        # This analysis function returns *only* the outputs that are meant to be tangents to the backwards.
        # Anything that aliases (inputs returned in the fw due to metadata mutations, or outputs that alias inputs/intermediates)
        # are *regenerated* later, and not used directly in the autograd graph
        input_tangents = [inp for inp, info in zip(flat_f_args, input_info) if info.mutation_type == MutationType.data]
        output_tangents = [o for o, info in zip(flat_f_args, output_info) if info.output_type == OutputType.non_alias]
        tangents = input_tangents + output_tangents

        metadata = ViewAndMutationMeta(
            input_info=input_info,
            requires_grad_info=requires_grad_info,
            output_info=output_info,
        )
        return metadata, pytree.tree_map(from_fun, tangents)
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
                i for i, outer_idx_or_lambda in enumerate(synthetic_base_info)
                if (isinstance(outer_idx_or_lambda, int) and outer_idx_or_lambda == idx)
                or (isinstance(outer_idx_or_lambda, tuple) and outer_idx_or_lambda[0] == idx)
            ]
        if any(meta.input_info[i] == MutationType.data for i in outer_aliased_indices_of_current_base_arg):
            # Make sure the primal we pass to autograd.grad()
            # seees the tensor before the mutation
            out = t.clone()
        elif any(meta.input_info[i] == MutationType.metadata_only for i in outer_aliased_indices_of_current_base_arg):
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
        primals_no_input_mutations = [maybe_to_fresh_input(i, t) for i, t in enumerate(primals)]
        # This is also where we handle the calling convention around synthetic bases.
        # We need to make sure that we convert any synthetic base arguments into views
        # *after* we do the cloning above, to preserve the view relationship.
        primals_ = unpack_synthetic_bases(primals_no_input_mutations)
        assert len(meta.input_info) == len(primals_)
        all_outs = fn(*primals_)
        assert len(meta.output_info) == len(all_outs)

        # Pass any (non-aliased) outputs in as tangents, since they'll be returned as outputs in the fw
        # For outputs that are aliases of intermediates, we will have returned the output's _base as an output in the graph instead,
        # which we *should* send to grad()
        outputs_for_grad = [
            x
            # TODO: support ._base
            # x._base if meta.output_info[i].output_type == OutputType.alias_of_intermediate else x
            for (i, x) in enumerate(all_outs) if meta.output_info[i].output_type == OutputType.non_alias
        ]
        # Pass any (non-aliased) mutated inputs in as tangents, since they'll be returned as outputs in the fw
        # Important: the traced joint fw/bw will return updated inputs with data mutations,
        # but *not* with metadata mutations.
        # Instead, we shunt the updated metadata around externally
        # and update the input's metadata outside of the autograd.Function
        mutated_inputs_for_grad = [x for (i, x) in enumerate(primals_) if meta.input_info[i].mutation_type == MutationType.data]
        mutated_inputs_and_outs_to_grad = mutated_inputs_for_grad + outputs_for_grad

        mutated_inputs_to_return = [x for (i, x) in enumerate(primals_) if meta.input_info[i].mutation_type != MutationType.none]
        fw_outs_to_return = mutated_inputs_to_return + all_outs

        metadata_mutated_inps = [
            x for (i, x) in enumerate(primals_) if meta.input_info[i].mutation_type == MutationType.metadata_only]
        # TODO: add intermediate bases as outputs here

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
                needed_outs.append(out if out.shape == tangent.shape else out.view(tangent.shape))
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
        return fw_outs_to_return, [
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
        print("====== Forward (only) graph {aot_config.aot_id} ======")
        fw_module.print_readable()


    disable_amp = torch._C._is_any_autocast_enabled()
    context = disable_autocast_manager if disable_amp else nullcontext

    with context(), track_graph_compiling(aot_config, "inference"):
        compiled_fw = aot_config.fw_compiler(fw_module, flat_args)

    @wraps(compiled_fw)
    def new_fn(args):
        fw_outs = call_func_with_args(compiled_fw, args, disable_amp=disable_amp)
        return fw_outs

    return new_fn


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
    fwd_inputs: List[Any],
    mutated_input_info: List[MutationType]
) -> Tuple[List[Any], Optional[List[Union[int, Tuple[int, Tuple[Any]]]]]]:
    assert len(fwd_inputs) == len(mutated_input_info)
    storage_ref_to_idx: Dict[StorageWeakRef, List[int]] = collections.defaultdict(list)
    for i, inpt in enumerate(fwd_inputs):
        if isinstance(inpt, Tensor):
            storage_ref = StorageWeakRef(inpt.storage())
            storage_ref_to_idx[storage_ref].append(i)
    base_args = []
    other_args = []
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
            mutated_input_info[inpt_idx] == MutationType.data for inpt_idx in aliased_input_indices
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
                assert are_differentiable_views(view1, view2), \
                    "aot_autograd() does not yet handle non-differentiable view input mutations."
                # Regenerating views when reinterpreting complex / real tensors seems non-trivial,
                # not handling for now
                assert same_dtype_views(view1, view2), \
                    "aot_autograd() does not yet handle input mutations on views with different dtypes."
            non_none_bases = [fwd_inputs[i]._base for i in aliased_input_indices if fwd_inputs[i]._base is not None]
            aliases_with_none_bases = [fwd_inputs[i] for i in aliased_input_indices if fwd_inputs[i]._base is None]
            if len(non_none_bases) == 0:
                # Case where none of the aliases require gradients
                example_idx = aliased_input_indices[0]
                synthetic_base = torch.Tensor(fwd_inputs[example_idx].storage())
            else:
                # Case where all of the aliases require gradients, and have the same _base.
                synthetic_base = non_none_bases[0]
                for other_base in non_none_bases[1:]:
                    assert other_base is synthetic_base, \
                        "aot_autograd() does not yet handle non-differentiable view input mutations."
                for alias in aliases_with_none_bases:
                    assert alias is synthetic_base, "aot_autograd() does not yet handle non-differentiable view input mutations."
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
                inner_calling_convention_meta[curr_view_idx] = (base_idx, (size_, stride_, storage_offset_))
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
        post_processed_calling_convention_meta: List[Union[int, Callable]] = [-1 for _ in range(len(inner_calling_convention_meta))]
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


# Wraps aot_dispatch_deduplicated_autograd, ensuring that duplicate arguments
# are dropped from the inner compilation function.
#
# In Haskell types, suppose you have:
#
#   add_dupe_args :: DedupedArgs -> Args
#   remove_dupe_args :: Args -> DedupedArgs
#
#   aot_dispatch_deduplicated_autograd
#       :: (DedupedArgs -> R) -> DedupedArgs -> AOTConfig -> (DedupedArgs -> R)
#   aot_dispatch_autograd
#       :: (Args -> R) -> Args -> AOTConfig -> (Args -> R)
#
# Then the code below can be written in point-free style as:
#
#   aot_dispatch_deduplicate_autograd f a c =
#       aot_dispatch_autograd (f . add_dupe_args) (remove_dupe_args a) c . remove_dupe_args
#
def aot_dispatch_autograd(flat_fn, flat_args: List[Tensor], aot_config: AOTConfig):
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
    #   seen_args = {2}  # what to drop
    #   add_dupe_map = {  # how to get args from the deduped list
    #       0: 0,
    #       1: 1,
    #       2: 0,
    #       3: 2,
    #   }
    #   keep_arg_mask = [True, True, False, True]

    seen_args = {}
    keep_arg_mask = []
    dropped_args = False
    add_dupe_map = {}
    duped_arg_len = len(flat_args)

    j = 0  # index into deduped_flat_args
    for i, t in enumerate(flat_args):
        if t in seen_args:
            keep_arg_mask.append(False)
            dropped_args = True
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

    def maybe_wrap_debug(f):
        if not config.debug_assert:
            return f

        @wraps(f)
        def debug_wrapper(*args):
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
            """
            assert len(seen) == unique_args, format_guard_bug_msg(aot_config,
                f"there would be {unique_args} distinct arguments"
            )
            """
            return f(*args)

        return debug_wrapper

    # Fastpath
    if not dropped_args:
        return maybe_wrap_debug(aot_dispatch_deduplicated_autograd(flat_fn, flat_args, aot_config))

    deduped_flat_args = remove_dupe_args(flat_args)

    @wraps(flat_fn)
    def wrapped_flat_fn(*args):
        return flat_fn(*add_dupe_args(args))

    compiled_fn = aot_dispatch_deduplicated_autograd(wrapped_flat_fn, deduped_flat_args, aot_config)

    @wraps(compiled_fn)
    def wrapped_compiled_fn(*args):
        return compiled_fn(*remove_dupe_args(args))

    return maybe_wrap_debug(wrapped_compiled_fn)


def describe_input(i, aot_config):
    if i < aot_config.num_params_buffers:
        return f"parameter/buffer {i}"
    else:
        return f"input {i - aot_config.num_params_buffers}"


# Like aot_dispatch_autograd, but with the precondition that there
# are no duplicate arguments in flat_args (e.g., the same Tensor
# object never shows up twice.  However, two tensor inputs MAY alias
# the same storage, so long as they have separate TensorImpls.)
def aot_dispatch_deduplicated_autograd(flat_fn, flat_args: List[Tensor], aot_config: AOTConfig):

    with enable_python_dispatcher():
        _fw_metadata, out = run_functionalized_fw_and_collect_metadata(
            flat_fn
        )(*flat_args)

    # pre-compute, so we can bail out quickly in the hotpath
    _num_outputs = len(_fw_metadata.output_info)
    _num_outputs_non_aliased = len([
        x for x in _fw_metadata.output_info if x.output_type == OutputType.non_alias])
    _num_outputs_aliased_to_inputs = len([
        x for x in _fw_metadata.output_info if x.output_type == OutputType.alias_of_input])
    _num_outputs_aliased_to_intermediates = len([
        x for x in _fw_metadata.output_info if x.output_type == OutputType.alias_of_intermediate])
    _num_outputs_aliased = _num_outputs_aliased_to_inputs + _num_outputs_aliased_to_intermediates

    _num_mutated_data_inputs = len([x for x in _fw_metadata.input_info if x.mutation_type == MutationType.data])
    _num_mutated_metadata_only_inputs = len([x for x in _fw_metadata.input_info if x.mutation_type == MutationType.metadata_only])
    _num_mutated_inputs = _num_mutated_data_inputs + _num_mutated_metadata_only_inputs
    # TODO
    _num_intermediate_bases = 0

    assert len(_fw_metadata.requires_grad_info) == _num_mutated_inputs + _num_outputs

    # out here corresponds to the set of outputs in the traced forward that should get grad_outputs in the traced backward.
    # It includes outputs of the original forward, *and* any updated inputs due to input mutations.
    # However, it does *not* include any outputs that are aliases of inputs or intermediates, or any metadata-only input mutations.
    out = pytree.tree_map(
        lambda x: x.detach().contiguous() if isinstance(x, Tensor) else x,
        out,
    )

    # This code only executes if we have graph inputs that alias each other, and one of those inputs
    # gets its data mutated.
    # When that happens, we replace the aliased inputs with a synthetic base, and in the traced forward
    # we later generate the input views
    flat_args_with_views_handled, _synthetic_base_info = merge_view_inputs(
        flat_args, _fw_metadata.input_info)

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
            fx_g = make_fx(
                joint_forward_backward, aot_config.decompositions
            )(*joint_inputs)

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
        raise AssertionError("Graph partitioning without functionalization is not sound, we may introduce errors")

    if config.debug_joint:
        print(f"====== Joint graph {aot_config.aot_id} ======")
        fx_g.print_readable()

    with torch.no_grad():
        with track_graph_compiling(aot_config, "joint"):
            num_inner_fwd_outputs = _num_mutated_inputs + _num_outputs
            fw_module, bw_module = aot_config.partition_fn(
                fx_g, joint_inputs, num_fwd_outputs=num_inner_fwd_outputs)
            fw_outs = [n for n in fw_module.graph.nodes if n.op == "output"][0].args[0]
            # we only need to bookkeep the symints that are saved for bw, not any symints
            # the user forward might have returned in its own output
            fw_outs_saved_for_bw = fw_outs[num_inner_fwd_outputs:]
            symint_outs_saved_for_bw = [n for n in fw_outs_saved_for_bw if is_sym_node(n)]
            _num_symints_saved_for_bw = len(symint_outs_saved_for_bw)

        if config.debug_graphs:
            print("====== Forward graph {aot_config.aot_id} ======")
            fw_module.print_readable()
            print("====== Backward graph {aot_config.aot_id} ======")
            bw_module.print_readable()

        with track_graph_compiling(aot_config, "forward"):
            compiled_fw_func = aot_config.fw_compiler(fw_module, flat_args_with_views_handled)

    class CompiledFunction(torch.autograd.Function):
        compiled_fw = compiled_fw_func
        compiled_bw = None
        num_outputs = _num_outputs
        num_outputs_aliased_to_inputs = _num_outputs_aliased_to_inputs
        num_outputs_aliased_to_intermediates = _num_outputs_aliased_to_intermediates
        num_outputs_aliased = _num_outputs_aliased
        num_symints_saved_for_bw = _num_symints_saved_for_bw
        num_mutated_inputs = _num_mutated_inputs
        num_mutated_data_inputs = _num_mutated_data_inputs
        num_mutated_metadata_only_inputs = _num_mutated_metadata_only_inputs
        num_intermediate_bases = _num_intermediate_bases
        synthetic_base_info = _synthetic_base_info
        fw_metadata = _fw_metadata

        @staticmethod
        def forward(ctx, *deduped_flat_tensor_args):

            # There is a pretty complicated calling convention around what the compiled fw returns.
            # The full list of outputs and their relative order is:
            # (*mutated_inputs, *fw_outs, *fw_intermediate_bases, *saved_tensors, *saved_symints)
            # - Note that in the synthetic bases case, mutated_inputs will correspond to an updated version
            #   of the original view, and not the synthetic base
            fw_outs = call_func_with_args(
                CompiledFunction.compiled_fw, deduped_flat_tensor_args, disable_amp=disable_amp
            )

            num_outputs = CompiledFunction.num_outputs
            num_outputs_aliased_to_inputs = CompiledFunction.num_outputs_aliased_to_inputs
            num_outputs_aliased_to_intermediates = CompiledFunction.num_outputs_aliased_to_intermediates
            num_intermediate_bases = CompiledFunction.num_intermediate_bases
            num_symints_saved_for_bw = CompiledFunction.num_symints_saved_for_bw
            num_mutated_inputs = CompiledFunction.num_mutated_inputs
            # Our forward() returns both (mutated_inputs, outputs, output_intermediate_bases, saved_tensors, saved_symints)
            num_forward_returns_not_including_intermediate_bases = num_mutated_inputs + num_outputs \
                + num_outputs_aliased_to_inputs + num_intermediate_bases
            num_forward_returns = num_forward_returns_not_including_intermediate_bases + num_intermediate_bases

            assert num_forward_returns == len(CompiledFunction.fw_metadata.requires_grad_info)

            # Partitioners must put symint arguments at the end separate from tensor arguments
            if num_symints_saved_for_bw > 0:
                tensors_saved_for_backwards = fw_outs[num_forward_returns:-num_symints_saved_for_bw]
                assert all([isinstance(x, torch.Tensor) for x in tensors_saved_for_backwards])
                ctx.save_for_backward(*tensors_saved_for_backwards)
                symint_outs = fw_outs[-num_symints_saved_for_bw:]
                assert all([isinstance(x, (int, float, torch.SymInt, torch.SymFloat)) for x in symint_outs])
                ctx.symints = symint_outs
            else:
                ctx.save_for_backward(*fw_outs[num_forward_returns:])
                ctx.symints = []

            # TODO: we need to be careful so that we don't return a tensor's ._base as an output,
            # if it is *already* an output somewhere else in the forward (either bc there are multiple aliased outs,
            # or because the ._base itself is also an output for the user)
            # Should the bases be marked as not requiring gradients? Seems safer not to do that, so I'm not for now.
            raw_returns = tuple(fw_outs[0:num_forward_returns])
            user_mutated_inputs_raw, user_fw_outputs_raw = raw_returns[0:num_mutated_inputs], raw_returns[num_mutated_inputs:]

            assert len(user_mutated_inputs_raw) == len([
                x for x in CompiledFunction.fw_metadata.input_info if x.mutation_type != MutationType.none])
            assert len(user_fw_outputs_raw) == len(CompiledFunction.fw_metadata.output_info)

            # Wrap all autograd.Function.forward() outputs that are aliases
            # so that autograd.Function doesn't treat them as tensors
            user_mutated_inputs = [
                TensorAlias(x) if info.mutation_type == MutationType.metadata_only else x
                for x, info in zip(user_mutated_inputs_raw, CompiledFunction.fw_metadata.input_info)
            ]
            user_fw_outputs = [
                TensorAlias(x) if info.output_type != OutputType.non_alias else x
                for x, info in zip(user_fw_outputs_raw, CompiledFunction.fw_metadata.output_info)
            ]
            fw_outs_to_return = *user_mutated_inputs, *user_fw_outputs
            fw_outs_not_requiring_grad = [
                x for (i, x) in enumerate(fw_outs_to_return)
                if isinstance(x, torch.Tensor) and not CompiledFunction.fw_metadata.requires_grad_info[i]
            ]
            ctx.mark_non_differentiable(*fw_outs_not_requiring_grad)

            return fw_outs_to_return

        @staticmethod
        def backward(ctx, *flat_args):
            # Calling convention: we expect a grad_out passed to the backward:
            # - for every output of the fw that does *not* alias an input
            # - for every updated_input generated by the fw that does *not* alias an input
            # - for every size/stride metadata value for aliased outputs.
            #   These are returned by the forward, but we just drop them in the backward.
            #   We need to return them in the forward, but unfortunately there's no way to specify
            #   in autograd.Function that certain non-tensor forward outputs shouldn't show up in the backward.
            expected_grad_outs = CompiledFunction.num_outputs + CompiledFunction.num_mutated_inputs \
                + CompiledFunction.num_intermediate_bases
            num_mutated_inps = CompiledFunction.num_mutated_inputs

            assert len(flat_args) == expected_grad_outs
            if CompiledFunction.num_mutated_metadata_only_inputs > 0 or CompiledFunction.num_outputs_aliased > 0:
                inp_tangents, out_tangents = flat_args[0:num_mutated_inps], flat_args[num_mutated_inps:]
                # input_info contains info on *every* input,
                # But in the backward(), we are only given grad outputs for every mutated input.
                # We then need to filter out the grad outputs that correspond to metadata-only mutations.
                mutated_inp_info = [x for x in CompiledFunction.fw_metadata.input_info if x.mutation_type != MutationType.none]
                assert len(inp_tangents) == len(mutated_inp_info)
                inp_tangents_filtered = [x for x, info in zip(inp_tangents, mutated_inp_info) if info.mutation_type == MutationType.data]
                inp_tangents_aliased = [x for x, info in zip(inp_tangents, mutated_inp_info) if info.mutation_type != MutationType.data]
                # We also need to filter out grad outputs that correspond to outputs aliasing inputs/intermediates
                out_info = CompiledFunction.fw_metadata.output_info
                out_tangents_filtered = [x for x, info in zip(out_tangents, out_info) if info.output_type == OutputType.non_alias]
                out_tangents_aliased = [x for x, info in zip(out_tangents, out_info) if info.output_type != OutputType.non_alias]
                flat_bw_args = inp_tangents_filtered + out_tangents_filtered
                # sanity asserts
                assert all(x is None for x in inp_tangents_aliased)
                assert all(x is None for x in out_tangents_aliased)
            else:
                flat_bw_args = flat_args

            contiguous_args = [t.contiguous() if torch.is_tensor(t) else t for t in flat_bw_args]
            all_args = list(ctx.symints) + list(ctx.saved_tensors) + list(contiguous_args)
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
                CompiledFunction.compiled_bw, all_args, steal_args=True, disable_amp=disable_amp
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
            new_inputs, metadata = merge_view_inputs(args, CompiledFunction.fw_metadata.input_info)
            # We're just re-running the original-args-to-synthetic-base transformation
            # that we ran during compilation.
            # This returns metadata that we use during tracing to recover the input views,
            # which we don't actually need at runtime.
            assert metadata is not None
            args_with_synthetic_bases = new_inputs
        else:
            args_with_synthetic_bases = args

        all_outs = CompiledFunction.apply(*args_with_synthetic_bases)

        assert len(all_outs) == CompiledFunction.num_mutated_inputs + CompiledFunction.num_outputs \
            + CompiledFunction.num_intermediate_bases

        if CompiledFunction.num_intermediate_bases > 0:
            outs = all_outs[:-CompiledFunction.num_intermediate_bases]
            intermediate_bases = all_outs[-CompiledFunction.num_intermediate_bases:]
        else:
            outs = all_outs
            intermediate_bases = []

        # Step 3: After running the compiled fw, apply updates to mutated inputs
        if CompiledFunction.num_mutated_inputs > 0:
            assert len([x for x in CompiledFunction.fw_metadata.input_info if x.mutation_type != MutationType.none]) \
                == CompiledFunction.num_mutated_inputs

            updated_inputs = outs[:CompiledFunction.num_mutated_inputs]
            fw_outs = outs[CompiledFunction.num_mutated_inputs:]

            for inpt_idx, meta in enumerate(CompiledFunction.fw_metadata.input_info):
                if meta.mutation_type == MutationType.none:
                    continue
                original_inpt = args[inpt_idx]
                updated_inpt = updated_inputs[inpt_idx]
                if meta.mutation_type == MutationType.metadata_only:
                    assert isinstance(updated_inpt, TensorAlias)
                    updated_inpt = updated_inpt.alias
                    # We need to grab the size/stride/storage_offset from the compiled forward,
                    # and use that to mutate the metadata of the input
                    original_inpt.as_strided_(updated_inpt.size(), updated_inpt.stride(), updated_inpt.storage_offset())
                else:
                    # TODO: handle resize_() on inputs to a larger size.
                    # This is actually non-trivial to detect, so we should probably just handle it
                    # (or make dynamo detect).
                    # We can't just check of original_inpt.storage_size != updated_inpt.storage_size,
                    # Because the original_inpt might be a view of some larger tensor,
                    # and updated_inpt is always densely packed.
                    if original_inpt.size() != updated_inpt.size() \
                            or original_inpt.stride() != updated_inpt.stride() \
                            or original_inpt.storage_offset() != updated_inpt.storage_offset():
                        # Functionalization can't easily tell us if an input had BOTH its metadata actual data mutated.
                        # So we check if metadata needs to be mutated here manually.
                        original_inpt.as_strided_(updated_inpt.size(), updated_inpt.stride(), updated_inpt.storage_offset())
                    original_inpt.copy_(updated_inpt)
        else:
            fw_outs = outs

        # Step 4: Manually regenerate any outputs that are aliased to inputs, instead of
        # compiling them.
        if CompiledFunction.num_outputs_aliased > 0:
            assert len(fw_outs) == len(CompiledFunction.fw_metadata.output_info)
            fw_outs_including_aliases = []
            for o, info in zip(fw_outs, CompiledFunction.fw_metadata.output_info):
                if info.output_type == OutputType.non_alias:
                    fw_outs_including_aliases.append(o)
                assert isinstance(o, TensorAlias)
                o = o.alias
                if info.output_type == OutputType.alias_of_input:
                    aliased_base_tensor = args[info.base_idx]
                    if info.output_is_input:
                        # Special case for when an output *is* an input (just return the input, don't generate a view)
                        regenerated_out = aliased_base_tensor
                    else:
                        regenerated_out = gen_alias_from_base(aliased_base_tensor, o.size(), o.stride(), o.storage_offset(), o)
                    fw_outs_including_aliases.append(regenerated_out)
                else:
                    assert info.output_type == OutputType.alias_of_intermediate
                    aliased_base_tensor = intermediate_bases[info.base_idx]
                    assert o._base is not None and o._base is aliased_base_tensor
                    # TODO: handle the custom autograd function case here.
                    # We need a way to check whether a tensor came from a custom autograd fn from python,
                    # AND a way to replay that custom view fn.
                    regenerated_out = gen_alias_from_base(aliased_base_tensor, o.size(), o.stride(), o.storage_offset(), o)
                    fw_outs_including_aliases.append(regenerated_out)
            return fw_outs_including_aliases
        else:
            return fw_outs

    if not config.debug_assert:
        return compiled_function

    flat_requires_grad = [a.requires_grad if isinstance(a, Tensor) else None for a in flat_args]

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
                    f"{describe_input(i, aot_config)} would not require grad"
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
        fake_mode = FakeTensorMode(shape_env=shape_env) if config.use_fake_tensor else nullcontext()

    cross_ref = CrossRefFakeMode() if config.debug_fake_cross_ref else nullcontext()
    python_dispatcher_mode = enable_python_dispatcher() if config.use_dynamic_shapes else nullcontext()

    with torch.autograd.set_multithreading_enabled(False), preserve_rng_state(), cross_ref, fake_mode, python_dispatcher_mode:

        def process_inputs(flat_args):
            if config.use_fake_tensor or isinstance(fake_mode, FakeTensorMode):
                def convert(idx, x):
                    if not isinstance(x, torch.Tensor):
                        return x
                    if isinstance(x, FakeTensor):
                        assert x.fake_mode is fake_mode
                        return x
                    if idx < aot_config.num_params_buffers and config.static_weight_shapes:
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
            return make_boxed_func(
                aot_dispatch_autograd(flat_fn, fake_flat_tensor_args, aot_config)
            )
        else:
            return aot_dispatch_base(flat_fn, fake_flat_tensor_args, aot_config)


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
        raise RuntimeError("static_argnums has been deprecated - manually wrap your function or use torchdynamo.")

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
                args, kwargs = pytree.tree_unflatten(
                    flat_args, tensor_args_spec
                )
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
    compiled_f = aot_function(functional_call, num_params_buffers=num_params_buffers, *args, **kwargs)

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
    static_argnums=None
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
