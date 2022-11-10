import collections
import dataclasses
import warnings
from contextlib import contextmanager, nullcontext
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple, Set
from torch.fx.experimental.proxy_tensor import is_sym_node

import torch
import torch.fx.traceback as fx_traceback
import torch.nn as nn
import torch.utils._pytree as pytree
import torch.utils.dlpack
from torch import Tensor
from torch._subclasses import FakeTensorMode, CrossRefFakeMode
from torch.fx import immutable_collections, Interpreter
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from torch.multiprocessing.reductions import StorageWeakRef
from torch.nn.utils import stateless

from functorch import make_fx
from torch._dispatch.python import enable_python_dispatcher
from . import config
from .named_members_polyfill import _named_buffers, _named_parameters
from .partitioners import default_partition

try:
    from torchdynamo import disable as disable_torchdynamo
except ImportError:

    def disable_torchdynamo(x):
        return x


try:
    from torchdynamo.utils import dynamo_timed
except ImportError:

    def dynamo_timed(x):
        return x


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


# This is a version of functionalization that is specifically designed
# for the AOTAutograd use case.  It might be generally applicable though
# (if so, move it out of this file), so I've tried to give it a name that
# describes what it does.
#
# Given a function f, it produces a new function g that:
#
#   - Detaches all inputs before running f; the inner function
#     does not directly participate in any pre-existing autograd.
#     preserve_requires_grad is provided as a convenience to set the
#     requires_grad on the new detached leaves in sync with the originals.
#     (NB: In principle, you could backward through the pure operations
#     produced by functionalization; this is not used for AOTAutograd
#     and we have not tested it.)
#
#   - Functionalizes all operations on f, under the assumption that the passed
#     in function f must be "observationally pure"; that is, it cannot perform any
#     mutations (inplace data or view operations) on the passed in inputs, nor is
#     it allowed to directly close over tensors that aren't passed via its
#     arguments.  See
#     https://docs.google.com/document/d/19UoIh_SVrMy_b2Sx5ZaeOJttm6P0Qmyss2rdBuyfoic/edit
#     for discussion how how to implement the more complicated case.
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
def detach_and_functionalize_pure(f, preserve_requires_grad=True, *, prewrapped_input_map=None, collect_mutated_input_info: Optional[List[bool]]=None, collect_requires_grad_out_info=None, maybe_inner_calling_convention_meta=None):
    @wraps(f)
    def inner(*args, **kwargs):
        def to_fun(t):
            if isinstance(t, Tensor):
                if preserve_requires_grad:
                    r = torch._to_functional_tensor(t, mirror_autograd_meta=True)
                else:
                    r = torch._to_functional_tensor(t)
                return r
            else:
                return t

        # Ideally we would always do the wrapping to functional tensors directly in this function.
        # We can't do that though, because we need to preserve .is_leaf metadata on the wrapped inputs,
        # so that input mutations are ok during tracing.
        # Doing that requires running an extra operation to mock up autograd data, which we don't
        # want to end up in the graph!
        # TODO: delete and re-think
        if prewrapped_input_map is not None:
            f_args, f_kwargs = pytree.tree_map(lambda x: prewrapped_input_map[x], (args, kwargs))
        else:
            f_args, f_kwargs = pytree.tree_map(to_fun, (args, kwargs))

        if maybe_inner_calling_convention_meta is not None:
            # Expectation, we were passed a flat list of tensors
            assert kwargs is None
            for a in args:
                # TODO: I don't think this is true. expectation is that args is a list[primals, tangents]
                assert isinstance(a, torch.Tensor)

            # Map the outer calling convention to the inner calling convention
            f_args_inner = len(maybe_inner_calling_convention_meta)
            for i in range(len(maybe_inner_calling_convention_meta)):
                idx_or_lambda = maybe_inner_calling_convention_meta[i]
                if isinstance(idx_or_lambda, int):
                    f_args_inner[i] = f_args_inner[idx_or_lambda]
                else:
                    assert isinstance(idx_or_lambda, Callable)
                    f_args_inner[i] = idx_or_lambda()
        else:
            f_args_inner = f_args

        torch._enable_functionalization(reapply_views=True)
        try:
            outs = f(*f_args_inner, **f_kwargs)
        finally:
            torch._disable_functionalization()

        # Detect input mutation and error if found
        flat_args, _ = pytree.tree_flatten((args, kwargs))
        flat_f_args, _ = pytree.tree_flatten((f_args, f_kwargs))

        # This is just for sanity checking, can be skipped
        mutated_inputs = []
        for (i, (arg, f_arg)) in enumerate(zip(flat_args, flat_f_args)):
            if not isinstance(arg, Tensor):
                continue
            torch._sync(f_arg)
            new_arg = torch._from_functional_tensor(f_arg)
            if arg is not new_arg:
                if collect_mutated_input_info is not None:
                    collect_mutated_input_info[i] = True
                # Calling convention: the output is (mutated_input_values, original_outs)
                mutated_inputs.append(new_arg)

        def from_fun(t):
            # Collect info on which output tensors require gradients,
            # so we can mark them properly in the returned autograd.Function
            nonlocal collect_requires_grad_out_info
            if collect_requires_grad_out_info is not None:
                collect_requires_grad_out_info.append(isinstance(t, torch.Tensor) and not t.requires_grad)

            if not isinstance(t, Tensor) or not torch._is_functional_tensor(t):
                return t
            torch._sync(t)
            return torch._from_functional_tensor(t)

        mutated_inps_and_outs = mutated_inputs + list(outs)
        return pytree.tree_map(from_fun, tuple(mutated_inps_and_outs))
    return inner


# This creates a joint forwards-backwards function given both
# the primals (to run forwards) and tangents (to run backwards).
#
# It has a precondition which is that the passed in function
# must be observationally pure; it is not permitted to mutate
# the primals or tangents.
def create_joint_forward_backward_pure(fn):
    def joint_forward_backward(
        primals: List[Any], tangents: List[Any]
    ) -> Tuple[List[Any], List[Any]]:
        # Call the forward pass
        outs = fn(*primals)
        # Get the inputs that need gradients
        grad_primals = []
        inputs_needs_grads = []
        for p in primals:
            is_grad_tensor = isinstance(p, Tensor) and p.requires_grad
            inputs_needs_grads.append(is_grad_tensor)
            if is_grad_tensor:
                grad_primals.append(p)

        # Get the outputs that need gradients
        assert len(tangents) == len(outs)
        needed_outs = []
        needed_tangents = []
        for out, tangent in zip(outs, tangents):
            if isinstance(out, Tensor) and out.requires_grad:
                needed_outs.append(out)
                needed_tangents.append(tangent)

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
        return outs, [
            next(backward_out_iter) if i else None for i in inputs_needs_grads
        ]

    return joint_forward_backward


def normalize_as_list(x):
    if isinstance(x, tuple):
        return list(x)
    elif isinstance(x, list):
        return x
    return [x]


aot_autograd_decompositions = {}


# This is a list since looking forward, we can have this arbitrarily nested.
graph_being_compiled: List[str] = []
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
    return f"{model_name}_{'_'.join(graph_being_compiled)}_{nth_graph}"


get_graph_being_compiled = get_aot_graph_name


@contextmanager
def track_graph_compiling(graph_name, increment_index=False):
    global graph_being_compiled
    graph_being_compiled = [graph_name]
    yield
    if increment_index:
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


def aot_dispatch_base(flat_fn, flat_args: List[Tensor], aot_config: AOTConfig):
    fw_module = make_fx(flat_fn, aot_config.decompositions)(*flat_args)
    if config.debug_graphs:
        print("====== Forward (only) graph ======")
        fw_module.print_readable()


    disable_amp = torch._C._is_any_autocast_enabled()
    context = disable_autocast_manager if disable_amp else nullcontext

    with context(), track_graph_compiling("inference"):
        compiled_fw = aot_config.fw_compiler(fw_module, flat_args)

    @wraps(compiled_fw)
    def new_fn(args):
        fw_outs = call_func_with_args(compiled_fw, args, disable_amp=disable_amp)
        return fw_outs

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
def merge_view_inputs(fwd_inputs, mutated_input_info: List[bool]):
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
    # - or a lambda corresponding to an input view arg, that functionalization will use to generate the argument from a synthetic base.
    inner_calling_convention_meta: Dict[int, Union[int, Callable]] = [None for _ in range(len(fwd_inputs))]
    for aliased_input_indices in storage_ref_to_idx.values():
        if len(aliased_input_indices) > 1 and any(mutated_input_info[inpt_idx] for inpt_idx in aliased_input_indices):
            # We detected an input that was mutated, AND aliases with another input.
            # we need to replace this set of aliased inputs with a single synthetic base.
            # For now, I'm banning a bunch of cases. We expect dynamo to properly detect these cases
            # and error out. We can fix them later.
            for idx1, idx2 in zip(aliased_input_indices, aliased_input_indices[1:]):
                view1 = fwd_inputs[idx1]
                view2 = fwd_inputs[idx2]
                assert are_differentiable_views(view1, view2), "aot_dispatch_autograd() does not yet handle non-differentiable view input mutations."
                # Regenerating views when reinterpreting complex / real tensors seems non-trivial,
                # not handling for now
                assert same_dtype_views(view1, view2), "aot_dispatch_autograd() does not yet handle input mutations on views with different dtypes."
            # Create the synthetic base.
            storage = fwd_inputs[aliased_input_indices[0]].storage()
            synthetic_base = torch.Tensor(storage)
            base_args.append(synthetic_base)
            for curr_view_idx in aliased_input_indices:
                curr_view = fwd_inputs[curr_view_idx]
                inner_calling_convention_meta[curr_view_idx] = lambda: synthetic_base.as_strided(curr_view.sizes(), curr_view.strides(), curr_view.storage_offset())
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
        assert False, "Not ready yet"
        args_to_functionalization = base_args + other_args
        arg_to_old_idx_map = {arg: i for (i, arg) in enumerate(fwd_inputs)}
        for i, other_arg in enumerate(other_args):
            new_idx = len(base_args) + i
            old_idx = arg_to_old_idx_map[other_arg]
            inner_calling_convention_meta[old_idx] = new_idx
        return args_to_functionalization, inner_calling_convention_meta



def aot_dispatch_autograd(flat_fn, flat_args: List[Tensor], aot_config: AOTConfig):
    # Deduplicate inputs.  Suppose you have:
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
    #
    # Whether to use flat_args or deduped_flat_args?  flat_fn takes flat_args,
    # and the autograd.Function must take deduped_flat_args; everything
    # else is just getting the types right.

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

    # NB: Hot path, avoid set lookups here
    def remove_dupe_args(args):
        if not dropped_args:
            return args
        return [t for t, keep in zip(args, keep_arg_mask) if keep]

    def add_dupe_args(args):
        if not dropped_args:
            return args
        return [args[add_dupe_map[i]] for i in range(duped_arg_len)]

    def functionalize_arg(a):
        # Wraps input tensors into FunctionalTensorWrappers.
        # we can't do this inside of detach_and_functionalize_pure directly,
        # because we need to do stuff like .clone() in order to get accurate autograd metadata (.is_leaf),
        # and we don't want that to show up in the trace
        if isinstance(a, torch.Tensor):
            func_arg = torch._to_functional_tensor(a)
            func_arg.requires_grad = a.requires_grad
            # We also need to fake up the is_leaf()-ness of our wrappers.
            # Even though we're running with functionalization,
            # autograd runs first. So autograd will see a mutation on
            # a leaf and complain.
            # TODO: is this fine? Bad if we're not using fake tensor,
            # but we (mostly) should be at this point.
            if a.requires_grad and not a.is_leaf:
                func_arg = func_arg.clone()
            return func_arg
        return a

    deduped_flat_args = remove_dupe_args(flat_args)

    joint_forward_backward = create_joint_forward_backward_pure(lambda *args: flat_fn(*add_dupe_args(args)))

    # Run the forward to get proper metadata on the output, which we use to trace the backward.
    # We also run the forward with functionalization turned on, for two reasons:
    # - We don't want this first run to actually mutate any inputs, which we re-use for the full trace later.
    # - We collect info on which inputs are mutated in this pass, which we use
    #   to properly handle mutations on aliased inputs in the trace later.
    _mutated_input_info = [False for _ in range(len(deduped_flat_args))]
    # requires_grad gets set on the wrappers and not on the actual inner tensors,
    # so we need to record this info inside of functionalization instead of being able to
    # directly check the outputs
    _flat_outs_not_requiring_grad = []
    out = detach_and_functionalize_pure(
        lambda *args: flat_fn(*(add_dupe_args(args))),
        collect_mutated_input_info=_mutated_input_info,
        collect_requires_grad_out_info=_flat_outs_not_requiring_grad
    )(*deduped_flat_args)

    _num_mutated_inputs = len([x for x in _mutated_input_info if x])
    # Note using detach_and_functionalize_pure() is useful here because it lets us getting everything in one go:
    # (1) Which inputs were mutated
    # (2) Which outputs require/don't require grad
    # (3) The actual outputs
    # The downside is that:
    # (1) The actual outputs that we get don't have correct grad info. We end up
    #     storing gradient info on the wrapper and not the actual out tensor.
    #     This is ok, because we don't care about that gradient info here
    #     (see the .detach().contiguous() below).
    # (2) We end up getting back (updated_inpts, outputs). We don't care about the mutated inputs here.
    out = out[_num_mutated_inputs:]
    out = pytree.tree_map(
        lambda x: x.detach().contiguous() if isinstance(x, Tensor) else x,
        out,
    )

    if isinstance(out, (list, tuple)):
        _num_outs = len(out)
    else:
        _num_outs = 1

    deduped_flat_args_with_views_handled, _maybe_inner_calling_convention_meta = merge_view_inputs(deduped_flat_args, _mutated_input_info)

    joint_inputs = (deduped_flat_args_with_views_handled, out)

    disable_amp = torch._C._is_any_autocast_enabled()

    if config.use_functionalize:
        with enable_python_dispatcher():
            flattened_joints, _ = pytree.tree_flatten(joint_inputs)
            fx_g = make_fx(
                detach_and_functionalize_pure(
                    joint_forward_backward,
                    maybe_inner_calling_convention_meta=_maybe_inner_calling_convention_meta,
                    # TODO: delete
                    # prewrapped_input_map={a: functionalize_arg(a) for a in flattened_joints},
                ),
                aot_config.decompositions
                # Need to .clone() when wrapping inputs, don't want that to show up in the graph so we do it outside
            )(*joint_inputs)
        # There should be *NO* mutating ops in the graph at this point.
        assert_functional_graph(fx_g.graph)
        fx_g.graph.eliminate_dead_code()
        fx_g.recompile()
    else:
        warnings.warn("graph partitioning without functionalization is not sound, we may introduce errors")
        fx_g = make_fx(joint_forward_backward, aot_config.decompositions)(*joint_inputs)

    if config.debug_joint:
        print("====== Joint graph ======")
        fx_g.print_readable()

    with torch.no_grad():
        with track_graph_compiling("joint"):
            fw_module, bw_module = aot_config.partition_fn(fx_g, joint_inputs, num_fwd_outputs=_num_mutated_inputs + _num_outs)
            fw_outs = [n for n in fw_module.graph.nodes if n.op == "output"][0].args[0]
            # we only need to bookkeep the symints that are saved for bw, not any symints
            # the user forward might have returned in its own output
            fw_outs = fw_outs[_num_outs:]
            symint_outs = [n for n in fw_outs if is_sym_node(n)]
            _num_symints = len(symint_outs)

        if config.debug_graphs:
            print("====== Forward graph ======")
            fw_module.print_readable()
            print("====== Backward graph ======")
            bw_module.print_readable()

        with track_graph_compiling("forward"):
            compiled_fw_func = aot_config.fw_compiler(fw_module, deduped_flat_args_with_views_handled)

    class CompiledFunction(torch.autograd.Function):
        compiled_fw = compiled_fw_func
        compiled_bw = None
        num_outs = _num_outs
        num_symints = _num_symints
        flat_outs_not_requiring_grad = _flat_outs_not_requiring_grad
        mutated_input_info = _mutated_input_info
        num_mutated_inputs = _num_mutated_inputs
        maybe_inner_calling_convention_meta = _maybe_inner_calling_convention_meta

        @staticmethod
        @disable_torchdynamo
        def forward(ctx, *deduped_flat_tensor_args):
            if CompiledFunction.maybe_inner_calling_convention_meta is not None:
                # Given: the original args, including at least one pair of inputs that are aliased
                # and get subsequently mutated.
                # Generate: the updated args, including (potentially multiple) synthetic bases
                # that replace the views. The input views are regenerated manually in the compiled function.
                # TODO: think harder about what happens if (a view of) one of these mutated input views is ALSO returned
                new_inputs, metadata = merge_view_inputs(deduped_flat_tensor_args, CompiledFunction.mutated_input_info)
                # We're just re-running the original-args-to-synthetic-base transformation
                # that we ran during compilation.
                # This returns metadata that we use during tracing to recover the input views,
                # which we don't actually need at runtime.
                assert metadata is not None
                deduped_flat_tensor_args = new_inputs

            # There is a pretty complicated calling convention around what the compiled fw returns.
            # The full list of outputs and their relative order is:
            # (*mutated_inputs, *fw_outs, *saved_tensors, *saved_symints)
            fw_outs = call_func_with_args(
                CompiledFunction.compiled_fw, deduped_flat_tensor_args, disable_amp=disable_amp
            )
            num_outs = CompiledFunction.num_outs
            num_symints = CompiledFunction.num_symints
            num_mutated_inputs = CompiledFunction.num_mutated_inputs
            # Our forward() returns both outputs and mutated inputs,
            num_forward_returns = num_mutated_inputs + num_outs

            # Partitioners must put symint arguments at the end separate from tensor arguments
            if num_symints > 0:
                ctx.save_for_backward(*fw_outs[num_forward_returns:-num_symints])
                ctx.symints = fw_outs[-num_symints:]
            else:
                ctx.save_for_backward(*fw_outs[num_forward_returns:])
                ctx.symints = []

            # We can't blindly mark the updated inputs that we return as not having gradients,
            # as you can have a function that mutates an input and returns it.
            # So instead, we let the mutated inputs get passed into the backward, and filter them out.
            fw_outs_not_requiring_grad = [
                x for (i, x) in enumerate(fw_outs[num_mutated_inputs:num_outs]) if CompiledFunction.flat_outs_not_requiring_grad[i]
            ]
            ctx.mark_non_differentiable(*fw_outs_not_requiring_grad)

            return tuple(fw_outs[0:num_forward_returns])

        @staticmethod
        @disable_torchdynamo
        def backward(ctx, *flat_args):
            # TODO: I'm using ctx.mark_non_differentiable() on the updated inputs that we return
            # in the forward. That should prevent grad_outputs for them from showing up in the backwrd?
            flat_args_ignore_mutated_inputs = flat_args[CompiledFunction.num_mutated_inputs:]
            assert len(flat_args_ignore_mutated_inputs) == CompiledFunction.num_outs
            contiguous_args = [t.contiguous() if torch.is_tensor(t) else t for t in flat_args_ignore_mutated_inputs]
            all_args = list(ctx.symints) + list(ctx.saved_tensors) + list(contiguous_args)
            del contiguous_args
            if CompiledFunction.compiled_bw is None:
                # TODO - pass in fake tensors ?
                context = disable_autocast_manager if disable_amp else nullcontext
                with context(), track_graph_compiling("backward", True):
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
        if CompiledFunction.num_mutated_inputs > 0:
            # TODO: test that duped arg cases works with input mutations
            no_dupe_args = remove_dupe_args(args)
            outs = CompiledFunction.apply(*no_dupe_args)
            assert len(outs) == CompiledFunction.num_mutated_inputs + CompiledFunction.num_outs
            # Calling convention: mutated inputs show up first.
            updated_inputs = outs[:CompiledFunction.num_mutated_inputs]
            fw_outs = outs[CompiledFunction.num_mutated_inputs:]
            curr_mutated_inpt_idx = 0
            for inpt_idx, was_mutated in enumerate(CompiledFunction.mutated_input_info):
                if was_mutated:
                    original_inpt = no_dupe_args[inpt_idx]
                    updated_inpt = updated_inputs[curr_mutated_inpt_idx]
                    curr_mutated_inpt_idx += 1
                    # Resize storage if necessary
                    if original_inpt.storage().size() != updated_inpt.storage().size():
                        # TODO: add tests for resizing inputs (smaller or larger)
                        original_inpt.resize_(updated_inpt.size())
                    # Mutate metadata if necessary
                    if original_inpt.size() != updated_inpt.size() or original_inpt.stride() != updated_inpt.stride() or original_inpt.storage_offset() != updated_inpt.storage_offset():
                        original_inpt.as_strided_(updated_inpt.size(), updated_inpt.stride(), updated_inpt.storage_offset())
                    # Copy data over
                    original_inpt.copy_(updated_inpt)
            return fw_outs
        else:
            return CompiledFunction.apply(*remove_dupe_args(args))

    return compiled_function


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

    shape_env = ShapeEnv() if config.use_dynamic_shapes else None
    fake_mode = FakeTensorMode(shape_env=shape_env) if config.use_fake_tensor else nullcontext()
    cross_ref = CrossRefFakeMode() if config.debug_fake_cross_ref else nullcontext()
    python_dispatcher_mode = enable_python_dispatcher() if config.use_dynamic_shapes else nullcontext()

    with torch.autograd.set_multithreading_enabled(False), preserve_rng_state(), cross_ref, fake_mode, python_dispatcher_mode:

        def process_inputs(flat_args):
            if config.use_fake_tensor:
                def convert(idx, x):
                    if not isinstance(x, torch.Tensor):
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

KNOWN_TYPES = [torch.Tensor, int, str, float, bool, torch.SymInt, torch.SymFloat]


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


def aot_module_simplified(mod: nn.Module, *top_args, **top_kwargs) -> nn.Module:
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

    def aot_function_simplified(
        fn: Callable,
        fw_compiler: Callable,
        bw_compiler: Optional[Callable] = None,
        partition_fn: Callable = default_partition,
        decompositions: Optional[Dict] = None,
        hasher_type=None,
        static_argnums=None,
    ) -> Callable:
        assert static_argnums is None
        if bw_compiler is None:
            bw_compiler = fw_compiler
        aot_config = AOTConfig(
            fw_compiler=fw_compiler,
            bw_compiler=bw_compiler,
            partition_fn=partition_fn,
            decompositions=decompositions,
            num_params_buffers=params_len,
        )

        compiled_fn = None

        @wraps(fn)
        def new_func(*args):
            nonlocal compiled_fn
            if compiled_fn is None:
                compiled_fn = create_aot_dispatcher_function(
                    fn,
                    args,
                    aot_config,
                )
            return compiled_fn(args)

        return new_func

    compiled_f = aot_function_simplified(functional_call, *top_args, **top_kwargs)

    if top_kwargs:

        def forward(*args, **kwargs):
            return compiled_f(
                *params_flat,
                *args,
                **kwargs,
            )

    else:

        def forward(*args):
            return compiled_f(
                *params_flat,
                *args,
            )

    forward.zero_grad = mod.zero_grad
    forward.named_parameters = mod.named_parameters
    return forward


compiled_function = aot_function
compiled_module = aot_module
