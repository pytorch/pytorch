import itertools
import logging
import warnings
import pprint
from contextlib import nullcontext
from functools import partial, wraps
from typing import Any, Callable, Dict, List, Optional, Tuple
from unittest.mock import patch

from torch.fx.experimental.proxy_tensor import make_fx

import torch
import torch.fx.traceback as fx_traceback
import torch.nn as nn
import torch.utils._pytree as pytree
import torch.utils.dlpack
from torch import Tensor
from torch._dispatch.python import enable_python_dispatcher
from torch._dynamo import compiled_autograd
from torch._dynamo.utils import dynamo_timed, lazy_format_graph_code, preserve_rng_state
from torch._guards import detect_fake_mode, tracing
from torch._prims_common import CUDARngStateHelper
from torch._logging import getArtifactLogger
from torch._subclasses import FakeTensor, FakeTensorMode
from torch.fx import Interpreter
from torch.fx.experimental.proxy_tensor import is_sym_node
from torch.fx.experimental.symbolic_shapes import (
    ShapeEnv, fx_placeholder_vals
)
from torch.nn.utils import stateless
from torch.utils._python_dispatch import is_traceable_wrapper_subclass
from torch._decomp.decompositions_for_rng import PhiloxStateTracker, rng_decompositions
from . import config
from .partitioners import default_partition
from torch._guards import TracingContext, DuplicateInputs

from ._aot_autograd.utils import (  # noqa: F401
    strict_zip,
    _get_symint_hints,
    KNOWN_TYPES,
    partial_flatten_asdict,
    normalize_as_list,
    _get_autocast_states,
    make_boxed_func,
    make_boxed_compiler,
    call_func_at_runtime_with_args,
    create_tree_flattened_fn,
    maybe_to_fresh_input,
)
from ._aot_autograd.logging_utils import (  # noqa: F401
    graph_being_compiled,
    nth_graph,
    model_name,
    set_model_name,
    get_aot_compilation_context,
    get_aot_graph_name,
    get_graph_being_compiled,
    track_graph_compiling,
    callback_set,
    setup_stacktrace_preservation_hooks,
    describe_input,
    format_guard_bug_msg,
)
from ._aot_autograd.functional_utils import (  # noqa: F401
    is_fun,
    to_fun,
    from_fun,
    sync_functional_tensor,
    has_metadata_mutation,
    has_data_mutation,
    are_all_mutations_hidden_from_autograd,
    are_all_mutations_under_no_grad_or_inference_mode,
    gen_alias_from_base,
    assert_functional_graph,
    _get_mutation_type,
    _check_if_mutation_can_be_in_graph,
)
from ._aot_autograd.schemas import (  # noqa: F401
    OutputType,
    OutputAliasInfo,
    MutationType,
    InputAliasInfo,
    SubclassCreationMeta,
    ViewAndMutationMeta,
    SubclassMeta,
    TensorAlias,
    BackwardSignature,
    GraphOutputName,
    GraphInputName,
    FQN,
    GraphSignature,
    AOTConfig,
    SubclassTracingInfo,
)
from ._aot_autograd.subclass_utils import (  # noqa: F401
    requires_subclass_dispatch,
    create_subclass_meta,
    unwrap_tensor_subclasses,
    wrap_tensor_subclasses,
    wrap_tensor_subclasses_maybe_joint,
    create_metadata_for_subclass,
)
from ._aot_autograd.collect_metadata_analysis import (  # noqa: F401
    run_functionalized_fw_and_collect_metadata,
)
from ._aot_autograd.input_output_analysis import (  # noqa: F401
    remove_dupe_metadata,
    create_synthetic_base_metadata,
    _tensors_definitely_do_not_overlap,
    _compute_overlapping_inputs,
    merge_view_inputs,
    create_graph_signature,
)
from ._aot_autograd.runtime_wrappers import (  # noqa: F401
    create_runtime_wrapper,
    functionalized_rng_runtime_epilogue,
    aot_dispatch_subclass_wrapper,
)
from ._aot_autograd.traced_function_transforms import (  # noqa: F401
    fn_input_mutations_to_outputs,
    fn_prepped_for_autograd,
    create_joint,
    create_functionalized_fn,
    create_functionalized_rng_ops_wrapper,
)

zip = strict_zip

log = logging.getLogger(__name__)
aot_joint_log = getArtifactLogger(__name__, "aot_joint_graph")
aot_graphs_log = getArtifactLogger(__name__, "aot_graphs")

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
# This logic is fully encapsulated in aot_wrapper_synthetic_base()
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
#     out = x_updated * x_view_updated
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


# Note [AOT Autograd: Views to avoid tangents aliasing inputs]
#
# We view every forward output when creating out tangent tensors to handle the problematic
# case in which a subclass does extra aliasing between graph outputs/inputs in a way that
# is not visible above the sublass.
#
# Ordinarily, when constructing the joint function that we want to trace in AOTAutograd,
# we're guaranteed that the tangent tensors that we pass
# into the joint are distinct tensors from the primals. This is because when
# decide which forward outputs to create tangents for, we only create tangents
# for forward outputs that are not aliases of inputs (See Note
# [AOT Autograd: outputs aliasing inputs or intermediates!]).
#
# However, when wrapper tensor subclasses enter the picture, it is possible
# to have an output of the forward that is a subclass that is not an
# input / alias of an input, but one of its inner tensors is an alias!
# NestedTensor is an example: Performing an out-of-place pointwise op on a
# NestedTensor constructs a fresh NestedTensor that holds onto the input's
# offsets tensor directly.
#
# Having tangent tensors that are the same as the (primal) forward inputs,
# can cause problems during tracing as make_fx() will specialize on our
# duplicate inputs: If we passed in the same tensor for primals_1 and
# tangents_1 during tracing, make_fx() will happily sub out all usages of
# tangents_1 with primals_1 in the graph, which is not what we want.
#
# To work around this, we view every forward output when creating out tangent
# tensors so that tangents can never be the same as forward inputs even if
# forward inputs alias forward outputs.
#
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



def create_graph(f, args, *, aot_config: AOTConfig) -> torch.fx.GraphModule:
    with enable_python_dispatcher():
        fx_g = make_fx(f, decomposition_table=aot_config.decompositions)(*args)

    return fx_g


aot_autograd_decompositions = {}


def aot_dispatch_base_graph(
    flat_fn,
    flat_args: List[Tensor],
    aot_config: AOTConfig,
    *,
    fw_metadata: ViewAndMutationMeta
) -> Tuple[Callable, List[Any], Optional[SubclassMeta]]:
    # aot_dispatch_base requires functionalization, but doesn't need to handle as many cases as the autograd case.
    # The cases that aot_dispatch_base doesn't need to handle include:
    # - outputs that are aliases of graph intermediates
    # - outputs that are aliases of graph inputs
    # While cases that it does need to handle include:
    # - input mutations (including when inputs are aliases of each other)
    # - input metadata mutations
    fn_to_trace = fn_input_mutations_to_outputs(
        flat_fn,
        fw_metadata,
        keep_data_input_mutations=aot_config.keep_inference_input_mutations,
    )

    fn_to_trace, updated_flat_args = create_functionalized_fn(
        fn_to_trace, flat_args, meta=fw_metadata, aot_config=aot_config, trace_joint=False)

    fn_to_trace, updated_flat_args_subclasses_desugared, maybe_subclass_meta = aot_dispatch_subclass(
        fn_to_trace, updated_flat_args, is_joint_structure=False, meta=fw_metadata, fw_only=flat_fn
    )

    fw_module = create_graph(
        fn_to_trace,
        updated_flat_args_subclasses_desugared,
        aot_config=aot_config,
    )

    # As long as we opted to remove input mutations, then
    # there should be *NO* mutating ops in the graph at this point.
    copy_count = assert_functional_graph(fw_module.graph, allow_input_mutations=aot_config.keep_inference_input_mutations)

    fw_module.graph.eliminate_dead_code()
    fw_module.recompile()

    copy_count2 = assert_functional_graph(fw_module.graph, allow_input_mutations=aot_config.keep_inference_input_mutations)

    assert copy_count == copy_count2

    if aot_config.enable_log:
        aot_graphs_log.info("%s", lazy_format_graph_code("Forward graph", fw_module, aot_config.aot_id))

    # TODO: should factor this into a separate function for export that always only returns just the graph.
    if aot_config.is_export:
        assert maybe_subclass_meta is None, "aot_export_module does not support tensor subclass inputs for now."
        return fw_module
    return fw_module, list(updated_flat_args_subclasses_desugared), maybe_subclass_meta

def aot_dispatch_base(flat_fn, flat_args: List[Tensor], aot_config: AOTConfig, *, fw_metadata: ViewAndMutationMeta):
    fw_module, updated_flat_args, maybe_subclass_meta = aot_dispatch_base_graph(
        flat_fn, flat_args, aot_config, fw_metadata=fw_metadata)

    disable_amp = torch._C._is_any_autocast_enabled()
    context = torch._C._DisableAutocast if disable_amp else nullcontext

    with context(), track_graph_compiling(aot_config, "inference"):
        compiler = aot_config.inference_compiler if aot_config.inference_compiler is not None else aot_config.fw_compiler
        if config.functionalize_rng_ops:
            # Add the seed and offset as example inputs to pass to the compiler
            fake_mode = detect_fake_mode()
            seed, offset = CUDARngStateHelper.get_torch_state_as_tuple(fake_mode)
            updated_flat_args.extend([seed, offset])

        if tracing_context := torch._guards.TracingContext.try_get():
            tracing_context.fw_metadata = fw_metadata \
                if maybe_subclass_meta is None else maybe_subclass_meta.fw_metadata
        compiled_fw = compiler(fw_module, updated_flat_args)

    # This boxed_call handling happens inside create_runtime_wrapper as well.
    # However, create_runtime_wrapper does not expect the rng offsets in the
    # output. So, we have to create another wrapper and take out the offset. As
    # a result, we have to account for not boxed_call compilers as well.
    if not hasattr(compiled_fw, "_boxed_call"):
        compiled_fw = make_boxed_func(compiled_fw)

    # Create a wrapper to set up the rng functionalize bits
    @wraps(compiled_fw)
    def rng_functionalization_wrapper(args):
        # args is a list because compiled_fw is boxed_call
        if fw_metadata.is_rng_op_functionalized:
            # Add the seed and offset to args
            seed, offset = CUDARngStateHelper.get_torch_state_as_tuple()
            args.extend([seed, offset])
            out = compiled_fw(args)
            out = functionalized_rng_runtime_epilogue(fw_metadata, out)
            return out
        else:
            return compiled_fw(args)

    if maybe_subclass_meta is not None:
        compiled_fw_func = aot_dispatch_subclass_wrapper(
            rng_functionalization_wrapper, subclass_metas=fw_metadata.subclass_fw_graph_out_meta, num_fw_outs_saved_for_bw=None)
    else:
        compiled_fw_func = rng_functionalization_wrapper

    if not hasattr(compiled_fw_func, "_boxed_call"):
        compiled_fw_func = make_boxed_func(compiled_fw_func)

    compiled_fn = create_runtime_wrapper(
        compiled_fw_func,
        runtime_metadata=fw_metadata,
        indices_of_inps_to_detach=[],
        trace_joint=False,
        keep_input_mutations=aot_config.keep_inference_input_mutations,
        disable_amp=disable_amp
    )

    return compiled_fn


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
    flat_fn,
    flat_args: List[Tensor],
    aot_config: AOTConfig,
    *,
    compiler_fn,
    fw_metadata,
):
    # Use information about whether or not flat_fn mutates its arguments
    # or not to handle dupe args

    # Strategy 1: For any input that is not mutated, we can leafify it if we
    # need to remove a duplicate.
    leaf_flat_args = []
    args_set = set()
    ok = True

    for i, a in enumerate(flat_args):
        if not isinstance(a, torch.Tensor):
            leaf_flat_args.append(a)
        elif a not in args_set:
            args_set.add(a)
            leaf_flat_args.append(a)
        elif not fw_metadata.input_info[i].mutates_data and not fw_metadata.input_info[i].mutates_metadata:
            leaf_flat_args.append(a.detach().requires_grad_(a.requires_grad))
        else:
            ok = False
            break

    if ok:
        return compiler_fn(flat_fn, leaf_flat_args, aot_config, fw_metadata=fw_metadata)

    if requires_subclass_dispatch(leaf_flat_args, fw_metadata):
        raise RuntimeError("""\
Encountered duplicate inputs that are mutated in the graph, but at least one input/output
to the graph is a tensor subclass. This is not supported today. You can try to
remove the aliasing yourself as a workaround, or otherwise file an issue on github.""")

    # export path: ban duplicate inputs for now, add later if requested.
    if aot_config.is_export:
        raise RuntimeError(f"""\
Encountered duplicated inputs that are mutated in the graph you are trying to export.
This functionality is currently not supported. If needed, please file a github issue.

fw_metadata={str(fw_metadata)}
        """)

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
    #   enumerate(add_dupe_map) = [  # how to get args from the deduped list
    #       (0, 0),
    #       (1, 1),
    #       (2, 0),
    #       (3, 2),
    #   ]
    #   keep_arg_mask = [True, True, False, True]

    seen_args = {}
    keep_arg_mask = []
    # Implicitly map duped arg position (list index) to de-duped arg position
    add_dupe_map: List[int] = []
    duped_arg_len = len(flat_args)

    j = 0  # index into deduped_flat_args
    for t in flat_args:
        if isinstance(t, torch.Tensor):
            if t in seen_args:
                keep_arg_mask.append(False)
                add_dupe_map.append(seen_args[t])
                continue
            seen_args[t] = j

        keep_arg_mask.append(True)
        add_dupe_map.append(j)
        j += 1
    assert len(add_dupe_map) == duped_arg_len, (
        f"Expects add_dupe_map to have length {duped_arg_len} but got {len(add_dupe_map)}"
    )

    # NB: Hot path, avoid set lookups here
    # TODO: Can avoid the zip here too, probably
    def remove_dupe_args(args):
        return [t for t, keep in zip(args, keep_arg_mask) if keep]

    def add_dupe_args(args):
        return [args[add_dupe_map[i]] for i in range(duped_arg_len)]

    deduped_flat_args = remove_dupe_args(flat_args)

    # Update our input metadata to remove duped input metadata.
    updated_fw_metadata = remove_dupe_metadata(fw_metadata, keep_arg_mask, add_dupe_map)

    if tracing_context := TracingContext.try_get() and aot_config.aot_autograd_arg_pos_to_source:
        # TODO(voz): This structure is 1:1, we could consider an alternate structure like
        # kept_pos:[dupe_arg_pos], however, add_dupe_map is 1:1 so we would need a new structure there,
        # which feels like needless complexity for a tiny bit of efficiency at this point.
        for dupe_arg_pos, (kept_pos, keep_arg) in enumerate(zip(add_dupe_map, keep_arg_mask)):
            if not keep_arg:
                dupe_arg_source = aot_config.aot_autograd_arg_pos_to_source[dupe_arg_pos]
                kept_arg_source = aot_config.aot_autograd_arg_pos_to_source[kept_pos]
                tracing_context.guards_context.aotautograd_guards.append(DuplicateInputs(kept_arg_source, dupe_arg_source))

    @wraps(flat_fn)
    def wrapped_flat_fn(*args):
        return flat_fn(*add_dupe_args(args))

    if config.debug_assert:
        ref_fw_metadata = run_functionalized_fw_and_collect_metadata(
            wrapped_flat_fn,
            keep_input_mutations=fw_metadata.keep_input_mutations,
            is_train=fw_metadata.is_train,
        )(*deduped_flat_args)
        assert ref_fw_metadata == updated_fw_metadata, \
            f'ref_metadata={str(ref_fw_metadata)}, actual_metadata={str(updated_fw_metadata)}'

    compiled_fn = compiler_fn(wrapped_flat_fn, deduped_flat_args, aot_config, fw_metadata=updated_fw_metadata)

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

# This layer handles the situation where you have two inputs that alias each other,
# and one of the inputs is mutated.
# We need to take special care to ensure that the mutation is applied to the other aliases in the graph.
#
# pre-condition: aot_wrapper_dedup has already run.
# (This function will in theory work if there are duplicate args.
# However, the synthetic base code path is a bit sub-optimal, and running with dupe'd inputs
# would cause us to hit that path more frequently).
def aot_wrapper_synthetic_base(
    flat_fn,
    flat_args: List[Tensor],
    aot_config: AOTConfig,
    *,
    fw_metadata: ViewAndMutationMeta,
    # Currently, the only reason we need to plumb this bool is because
    # the synthetic base code prohibits more cases in the autograd case than the inference case.
    needs_autograd: bool,
    compiler_fn,
):
    is_inference = not needs_autograd
    flat_args_with_synthetic_bases, synthetic_base_info = merge_view_inputs(
        flat_args, fw_metadata.input_info, is_inference=is_inference,
    )
    # Happy path: we don't need synthetic bases
    if synthetic_base_info is None:
        return compiler_fn(flat_fn, flat_args, aot_config, fw_metadata=fw_metadata)

    # export path: ban synthetic bases for now, add later if requested.
    if requires_subclass_dispatch(flat_args, fw_metadata):
        raise RuntimeError("""\
Encountered aliased inputs that are mutated in the graph, but at least one input/output
to the graph is a tensor subclass. This is not supported today. You can try to
remove the aliasing yourself as a workaround, or otherwise file an issue on github.""")

    if aot_config.is_export:
        raise RuntimeError(f"""\
Encountered aliased inputs that are mutated in the graph you are trying to export.
This functionality is currently not supported. If needed, please file a github issue.

synthetic_base_info={str(synthetic_base_info)}

fw_metadata={str(fw_metadata)}
        """)

    assert len(fw_metadata.input_info) == len(synthetic_base_info)

    # Update our forward metadata to take synthetic bases into account
    fw_metadata_updated, aliased_arg_idx_with_metadata_mutations = \
        create_synthetic_base_metadata(fw_metadata, synthetic_base_info, flat_args, flat_args_with_synthetic_bases)

    num_aliased_args_with_metadata_mutations = len(aliased_arg_idx_with_metadata_mutations)

    def unpack_synthetic_bases(primals: List[Any]) -> List[Any]:
        f_args_inner = []
        for inner_idx_or_tuple in synthetic_base_info:
            if isinstance(inner_idx_or_tuple, int):
                f_args_inner.append(primals[inner_idx_or_tuple])
            else:
                inner_base_idx, view_tensor = inner_idx_or_tuple
                base = primals[inner_base_idx]
                view_arg = gen_alias_from_base(
                    base, view_tensor, view_tensor.requires_grad
                )
                f_args_inner.append(view_arg)
        return f_args_inner

    @wraps(flat_fn)
    def wrapped_flat_fn(*args):
        unpacked_args = unpack_synthetic_bases(args)
        # This is a bit subtle. The goal of this entire function (aot_dispatch_synthetic_bases)
        # is to relieve the downstream logic from having to reason about mutations on inputs that alias
        # each other, by replacing aliased inputs with a synthetic base.
        # One area where this breaks down a bit however is if one of those aliased inputs
        # experienced a metadata mutation.
        # We are now obligated to reapply the metadata mutation directly to the user's input;
        # it isn't enough to apply mutations back to the synthetic base in the downstream logic.
        #
        # The way we handle this is by pretending that those aliased inputs that experience metadata mutations
        # are additional outputs in the user's forward function.
        # The downstream logic will just treat these as "user outputs that alias inputs".
        # However, we will manually grab them at runtime here, use them to reapply the metadata mutation
        # to the user inputs, and not return them to the user.
        aliased_args_with_metadata_mutations = [
            x for i, x in enumerate(unpacked_args) if i in aliased_arg_idx_with_metadata_mutations]
        if len(aliased_args_with_metadata_mutations) > 0:
            return *(flat_fn(*unpacked_args)), *aliased_args_with_metadata_mutations
        else:
            return flat_fn(*unpacked_args)

    if config.debug_assert:
        ref_fw_metadata = run_functionalized_fw_and_collect_metadata(
            wrapped_flat_fn,
            keep_input_mutations=fw_metadata.keep_input_mutations,
            is_train=fw_metadata.is_train,
        )(*flat_args_with_synthetic_bases)
        assert ref_fw_metadata == fw_metadata_updated, (
            f'ref_metadata={pprint.pformat(partial_flatten_asdict(ref_fw_metadata))}, '
            f'\nactual_metadata={pprint.pformat(partial_flatten_asdict(fw_metadata_updated))}'
        )

    compiled_fn = compiler_fn(wrapped_flat_fn, flat_args_with_synthetic_bases, aot_config, fw_metadata=fw_metadata_updated)

    if not hasattr(compiled_fn, "_boxed_call"):
        compiled_fn = make_boxed_func(compiled_fn)

    @wraps(compiled_fn)
    def wrapped_compiled_fn(args):
        args_with_synthetic_bases, synthetic_base_info = merge_view_inputs(
            args, fw_metadata.input_info, is_inference=is_inference
        )
        assert synthetic_base_info is not None
        aliased_args_w_metadata_mutations = [args[i] for i in aliased_arg_idx_with_metadata_mutations]
        args.clear()
        outs = compiled_fn(args_with_synthetic_bases)
        if num_aliased_args_with_metadata_mutations > 0:
            # This code does not handle **all** input metadata mutations.
            # Instead, it only handles metadata mutations on inputs that were converted into synthetic bases
            # (which only happens if at least one aliased input experienced a data mutation).
            # e.g:
            # def f(a, b):
            #     a.mul_(2)
            #     b.t_(1, 0)
            # f(x.view(2, 2), x.view(2, 2))
            mutated_metadata_inps = outs[-num_aliased_args_with_metadata_mutations:]
            user_outs = outs[:-num_aliased_args_with_metadata_mutations]
            for inp, mutated_inp in zip(aliased_args_w_metadata_mutations, mutated_metadata_inps):
                inp.as_strided_(mutated_inp.size(), mutated_inp.stride(), mutated_inp.storage_offset())
            return user_outs
        return outs

    return wrapped_compiled_fn


# Given a function operating on Subclass -> Subclass, returns an function that operates on Tensor -> Tensor
# Also returns:
# - the new set of arguments to pass into this function (now that tensor subclasses have been eliminated)
# - the updated ViewAndMutationMeta for this dense -> dense function.
# The other important arguments are:
# - flat_fn_maybe_joint: when is_joint_structure=True, this is the joint fw-bw function.
#                        when is_joint_structure=False, this is just the forward function.
# - fw_only: this is *always* the forward-only function.
#   Why do we need this? We need to collect updated ViewAndMutationMeta on our new dense -> dense functions.
#   In particular, we need this to tell the partitioner how many dense forward outputs there are.
def aot_dispatch_subclass(
    flat_fn_maybe_joint,
    args: List[Any],
    *,
    is_joint_structure: bool,
    meta: ViewAndMutationMeta,
    fw_only: Callable,
) -> "SubclassTracingInfo":
    # Skip logic if we don't need to trace through any subclasses
    req_subclass_dispatch = requires_subclass_dispatch(args, meta)
    if not req_subclass_dispatch:
        return SubclassTracingInfo(
            plain_tensor_trace_fn=flat_fn_maybe_joint,
            plain_tensor_args=args,
            maybe_subclass_meta=None,
        )

    # TODO: add subclass guards (later PR).

    # What's going on here? We need to compute subclass metadata about the outputs of the joint (grad_inputs).
    # Annoying: we don't know the grad input metas until we're in the middle of tracing the joint,
    # so we set it later, while we're tracing the joint (see inner_fn() below).
    # Another option would be to run our run_functionalized_fw_and_collect_metadata() function
    # directly on the joint, but this would hurt compile time (adding yet another pass through the joint).
    subclass_meta = SubclassMeta()

    def inner_fn(fn, args, *, use_trace_joint: bool):
        # Step 1: wrap tensor inputs into subclasses if necessary
        all_args = wrap_tensor_subclasses_maybe_joint(args, is_joint_structure=use_trace_joint, meta=meta)

        # Step 2: call the inner function, with our (maybe subclass) inputs
        wrapped_outs = fn(*all_args)

        if use_trace_joint:
            # See Note: [Computing Subclass Metadata about grad_inputs]
            # We also stash subclass info on our grad_inputs, if we're tracing the joint.
            nonlocal subclass_meta
            assert isinstance(wrapped_outs, tuple) and len(wrapped_outs) == 2
            # Don't need fw outs since we already have subclass metadata on them
            grad_inputs = wrapped_outs[1]
            subclass_meta.grad_input_metas = create_subclass_meta(grad_inputs)

        # Step 3: Unwrap any subclass outputs back into dense tensors
        unwrapped_outs = unwrap_tensor_subclasses(wrapped_outs, is_joint_structure=use_trace_joint)
        return unwrapped_outs

    def joint_fn(primals, tangents):
        return inner_fn(flat_fn_maybe_joint, (primals, tangents), use_trace_joint=True)

    def fw_fn(*primals):
        return inner_fn(flat_fn_maybe_joint, primals, use_trace_joint=False)

    def metadata_fn(*primals):
        return inner_fn(fw_only, primals, use_trace_joint=False)

    args_unwrapped = unwrap_tensor_subclasses(args, is_joint_structure=is_joint_structure)

    if is_joint_structure:
        primals_unwrapped = args_unwrapped[0]
        fn_to_trace = joint_fn
    else:
        primals_unwrapped = args_unwrapped
        fn_to_trace = fw_fn

    # Note: [Partitioner handling for Subclasses, Part 1]
    # The way the partitioner works is that:
    # (1) we pass is a single graph containing the joint fw/bw,
    #     where the # of graph outputs corresponds to # fw_outputs + # grad_inputs
    # (2) The partitioner accepts an arguments, num_fwd_outputs,
    #     and assumes that the first "num_fwd_outputs" graph outputs correspond
    #     to outputs of the forward graph.
    # How do tensor subclasses enter the picture?
    # the num_fwd_outputs in the final graph is actually non-trivial to compute,
    # because it can be influenced by input mutations and intermediate bases.
    # So we compute it by inspecting the current ViewAndMutationMeta object.
    # However, the original ViewAndMutationMeta that we computed was created
    # on the subclass -> subclass graph,
    # which can have a different number of outputs than the dense -> dense graph.
    # That's why we createa a fresh metadata object on the dense -> dense function here,
    # and plumb it back up to the partitioner.
    # See Note: [Partitioner handling for Subclasses, Part 2] for more info.
    meta_updated = run_functionalized_fw_and_collect_metadata(
        metadata_fn,
        keep_input_mutations=meta.keep_input_mutations,
        is_train=meta.is_train,
        requires_subclass_dispatch=True,
    )(*primals_unwrapped)

    subclass_meta.fw_metadata = meta_updated

    return SubclassTracingInfo(
        plain_tensor_trace_fn=fn_to_trace,
        plain_tensor_args=args_unwrapped,
        maybe_subclass_meta=subclass_meta,
    )


# Has the precondition that there
# are no duplicate arguments in flat_args (e.g., the same Tensor
# object never shows up twice.  However, two tensor inputs MAY alias
# the same storage, so long as they have separate TensorImpls.)
def aot_dispatch_autograd_graph(flat_fn, flat_args: List[Any], aot_config: AOTConfig, *, fw_metadata: ViewAndMutationMeta):
    # traced_tangents corresponds to the set of outputs in the traced forward that should get grad_outputs in the traced backward.
    # It includes outputs of the original forward, *and* any updated inputs due to input mutations.
    # However, it does *not* include any outputs that are aliases of inputs or intermediates, or any metadata-only input mutations.
    traced_tangents = pytree.tree_map(
        lambda x: x.detach().contiguous() if isinstance(x, Tensor) else x,
        fw_metadata.traced_tangents,
    )

    joint_inputs = (flat_args, traced_tangents)

    fn_prepared_for_autograd = fn_prepped_for_autograd(
        flat_fn,
        fw_metadata,
    )
    joint_fn_to_trace = create_joint(fn_prepared_for_autograd, aot_config=aot_config)

    joint_fn_to_trace, updated_joint_inputs = create_functionalized_fn(
        joint_fn_to_trace,
        joint_inputs,
        meta=fw_metadata,
        aot_config=aot_config,
        trace_joint=True,
    )

    subclass_tracing_info = aot_dispatch_subclass(
        joint_fn_to_trace, updated_joint_inputs, is_joint_structure=True, meta=fw_metadata, fw_only=flat_fn
    )

    joint_fn_to_trace = subclass_tracing_info.plain_tensor_trace_fn
    updated_joint_inputs = subclass_tracing_info.plain_tensor_args
    maybe_subclass_meta = subclass_tracing_info.maybe_subclass_meta

    fx_g = create_graph(joint_fn_to_trace, updated_joint_inputs, aot_config=aot_config)

    # There should be *NO* mutating ops in the graph at this point.
    assert_functional_graph(fx_g.graph, allow_input_mutations=aot_config.keep_inference_input_mutations)

    # Redundant with the check above, but worth having in case tracing introduced
    # a fake tensor. Unlikely.
    # See Note: [Fake Modules and AOTAutograd]
    torch._dynamo.utils.assert_no_fake_params_or_buffers(fx_g)
    fx_g.graph.eliminate_dead_code()
    fx_g.recompile()
    # TODO: in AOTAutograd, we create metadata like _indices_of_inps_to_detach to detect
    # when we need to manually detach() some inputs in the forward.
    # Higher order ops might eventually need to do the same.
    if aot_config.is_export:
        assert maybe_subclass_meta is None, "aot_export_module does not support tensor subclass inputs for now."
        return fx_g
    return fx_g, updated_joint_inputs, maybe_subclass_meta

def aot_dispatch_autograd(flat_fn, flat_args: List[Any], aot_config: AOTConfig, *, fw_metadata: ViewAndMutationMeta):
    fx_g, joint_inputs, maybe_subclass_meta = aot_dispatch_autograd_graph(flat_fn, flat_args, aot_config, fw_metadata=fw_metadata)

    # Copied from aot_dispatch_autograd_graph.
    traced_tangents = pytree.tree_map(
        lambda x: x.detach().contiguous() if isinstance(x, Tensor) else x,
        fw_metadata.traced_tangents,
    )
    disable_amp = torch._C._is_any_autocast_enabled()

    if aot_config.enable_log:
        aot_joint_log.info("%s", lazy_format_graph_code("Joint graph", fx_g, aot_config.aot_id))

    with torch.no_grad():
        inner_meta = fw_metadata if maybe_subclass_meta is None else maybe_subclass_meta.fw_metadata
        with track_graph_compiling(aot_config, "joint"):
            # See Note: [Partitioner handling for Subclasses, Part 1]
            num_inner_fwd_outputs = (
                inner_meta.num_mutated_inp_runtime_indices
                + inner_meta.num_outputs
                + inner_meta.num_intermediate_bases
                + inner_meta.num_outputs_rng_offset
            )
            fw_module, bw_module = aot_config.partition_fn(
                fx_g, joint_inputs, num_fwd_outputs=num_inner_fwd_outputs
            )
            fw_outs = next(n for n in fw_module.graph.nodes if n.op == "output").args[0]
            # we only need to bookkeep the symints that are saved for bw, not any symints
            # the user forward might have returned in its own output
            fw_outs_saved_for_bw = fw_outs[num_inner_fwd_outputs:]
            num_fw_outs_saved_for_bw = len(fw_outs_saved_for_bw)
            symint_outs_saved_for_bw = [
                n for n in fw_outs_saved_for_bw if is_sym_node(n)
            ]
            fw_metadata.num_symints_saved_for_bw = len(symint_outs_saved_for_bw)
            inner_meta.num_symints_saved_for_bw = len(symint_outs_saved_for_bw)
            _num_symints_saved_for_bw = len(symint_outs_saved_for_bw)

        # Note [Detaching inputs that never need gradients]
        # See https://github.com/pytorch/pytorch/issues/97745
        # Suppose we have a function like this that we want to compile:
        #
        # def f(x, y):
        #     return torch.mul(x, y.detach())
        #
        # What gradients should we compute for x and y?
        # By default, AOTAutograd will compute a gradient for **every** input that requires gradients,
        # and so we'll compute:
        #    x_grad_input = y
        #    y_grad_input = None
        # Does this preserve the semantics of eager mode?
        # Unfortunately, no.
        # Doing the above will cause autograd to **continue** to backprop the autograd tape
        # that was generated from constructing y.
        #
        # This is **different** from what would have happened in eager mode.
        # In eager mode, if we backprop through the output of this function, autograd will only traverse
        # the bit of the autograd tape corresponding to "x".
        # In particular, if a user had previously backpropped through y's autograd tape,
        # And then they try to backprop through the output of the above function,
        # then we'll hit the dreaded "Trying to backward through the graph a second time" error.
        #
        # You might think: If autograd sees that a gradient is None, shouldn't it stop early,
        # instead of continuing the backprop through the ancestors of that node in the graph?
        #
        # Autograd has two passes:
        # (1) a first pass that traverses the autograd graph and figures out which nodes need to be executed
        # (2) a second pass that actually goes ahead and executes each node when it becomes ready,
        #     propagating gradients
        # By the time we're executing a node and we see that it produces a None, the set of nodes to execute
        # is already locked-in.
        #
        # The fix: instead, we can recognize statically that the graph we're compiling will never contribute
        # gradients to y, and prevent autograd from trying to traverse y's autograd tape at all.
        # We can do this by manually detach'ing y before sending it through the `CompiledFunction`.
        #
        # Note that this solution is not bulletproof.
        # It's possible to construct a case where eager may or may not have have tried to autograd through y,
        # depending on the actual grad_outputs that were passed in during the backward.
        # There is no easy fix for this: the simplest fix would be to run with `retain_graph=True`,
        # allowing autograd to re-use the graph.
        #
        # An example of this case is:
        # def f(x):
        #     return x.detach() * 2, x * 3
        # If we were to only backprop through outs[0], in eager, we would stop
        # If we backward only on the first output, we shouldn't send a grad through x.
        # But the custom autograd function doesn't know that: it will materialize zero grads for x * 3
        # and we will end up with a zero grad at x.
        # If we later backprop through the second output, this will also require backprop'ing through x.
        # Meaning we'll need to use `retain_graph=True` to be able to backprop through x the second time.
        _indices_of_inps_to_detach = []
        bw_outs = next(n for n in bw_module.graph.nodes if n.op == "output").args[0]

        # TODO: we should apply the below "detach inputs if their gradients are statically known to be None"
        # optimization even if we have subclass inputs/outputs (we do not handle this today).
        # Computing which our our inputs get None gradients is a bit more complicated,
        # if any of our inputs are subclasses. Why?
        # (a) we need to make sure that we call .detach() on the input subclasses, since autograd sees subclasses.
        # (b) The grad_outputs that we AOT computed in our backward graph are the desugared tensor tensors,
        #     so we need to figure out which subclass fw inputs they map to.
        if maybe_subclass_meta is None:
            assert len(bw_outs) == len(fw_metadata.input_info) + inner_meta.num_outputs_rng_offset
            for i, (bw_out) in enumerate(bw_outs):
                if bw_out is None:
                    _indices_of_inps_to_detach.append(i)

        if aot_config.enable_log:
            aot_graphs_log.info("%s", lazy_format_graph_code("Forward graph", fw_module, aot_config.aot_id))
            aot_graphs_log.info("%s", lazy_format_graph_code("Backward graph", bw_module, aot_config.aot_id))

        with track_graph_compiling(aot_config, "forward"):
            # flat_args at this point might still be subclasses-
            # make sure to pass the unwrapped fake tensors into the compiler!
            adjusted_flat_args = joint_inputs[0]
            if config.functionalize_rng_ops:
                # Update example inputs for the fw_compiler
                fake_mode = detect_fake_mode()
                seed, offset = CUDARngStateHelper.get_torch_state_as_tuple(fake_mode)
                adjusted_flat_args.extend([seed, offset])
                # We are not clearing flat_args here because
                # 1) There is a check in the debug compiler at the end
                # 2) It does not matter as these are fake tensors

            if tracing_context := torch._guards.TracingContext.try_get():
                tracing_context.fw_metadata = inner_meta

            with TracingContext.report_output_strides() as fwd_output_strides:
                compiled_fw_func = aot_config.fw_compiler(
                    fw_module, adjusted_flat_args
                )
            if not hasattr(compiled_fw_func, "_boxed_call"):
                compiled_fw_func = make_boxed_func(compiled_fw_func)

            if maybe_subclass_meta is not None:
                # Why do we need to pass in num_fw_outs_saved_for_bw?
                # See Note: [Partitioner handling for Subclasses, Part 2]
                compiled_fw_func = aot_dispatch_subclass_wrapper(
                    compiled_fw_func,
                    subclass_metas=fw_metadata.subclass_fw_graph_out_meta,
                    num_fw_outs_saved_for_bw=num_fw_outs_saved_for_bw
                )
                if not hasattr(compiled_fw_func, "_boxed_call"):
                    compiled_fw_func = make_boxed_func(compiled_fw_func)

        # NB: It's important to compile backwards ahead of time, as this may
        # add extra guards which we need to apply to the Dynamo cache at
        # forwards
        with track_graph_compiling(aot_config, "backward"):
            placeholder_list = fx_placeholder_vals(bw_module)

            forward_saved_for_backwards_strides = None
            if fwd_output_strides is not None:
                forward_saved_for_backwards_strides = fwd_output_strides[inner_meta.tensors_saved_for_backwards_slice]

            # saved activations can have different stride to eager if
            # the compiler does layout optimization. We should restride the
            # tensor passed in for compiling the backward graph using the
            # saved tensor's stride.
            for i in range(len(placeholder_list)):
                ph_arg = placeholder_list[i]
                if not isinstance(ph_arg, torch.Tensor):
                    continue

                if forward_saved_for_backwards_strides is None:
                    continue

                real_stride = None
                # Per all_args calling convention
                j = i - len(symint_outs_saved_for_bw)
                if 0 <= j < len(forward_saved_for_backwards_strides):
                    real_stride = forward_saved_for_backwards_strides[j]
                if real_stride is None:
                    continue

                # Comparing ph_arg.stride() with real_stride directly may
                # cause dynamic dimensions in ph_arg being specialized to static
                # value. Using the hints to avoid that.
                if _get_symint_hints(ph_arg.stride()) != real_stride:
                    # Note that here we use the stride of the real tensor to
                    # restride a FakeTensor. This does not cause trouble
                    # for dynamic shape since this code path only get
                    # executed if layout optimization is enabled. And we
                    # disable layout optimization for dynamic shape right
                    # now.
                    #
                    # A solution that decide stride order based on real
                    # tensor's stride and then apply that stride order to
                    # the FakeTensor does not work smoothly since some
                    # tensor's layout is not 'dense'. E.g. mixnet_l has a
                    # tensor with size [8, 64, 112, 112] and strides
                    # (2408448, 1, 21504, 192). The solution mentioned will
                    # decide a stride of (802816, 1, 7168, 64) for this
                    # tensor which is wrong.
                    placeholder_list[i] = ph_arg.as_strided(ph_arg.size(), real_stride)

            compiled_bw_func = None
            if len(symint_outs_saved_for_bw):
                context = torch._C._DisableAutocast if disable_amp else nullcontext
                with context():
                    try:
                        compiled_bw_func = aot_config.bw_compiler(
                            bw_module, placeholder_list
                        )
                    except Exception:
                        log.warning(
                            "failed to eagerly compile backwards for dynamic, suppressing in case backwards not needed",
                            exc_info=True
                        )

    saved_context = TracingContext.try_get()

    class CompiledFunction(torch.autograd.Function):
        compiled_fw = compiled_fw_func
        compiled_bw = compiled_bw_func
        metadata = fw_metadata
        maybe_subclass_metadata: Optional[SubclassMeta] = maybe_subclass_meta
        num_symints_saved_for_bw = _num_symints_saved_for_bw

        @staticmethod
        def _compiled_autograd_key(ctx):
            return (aot_config.aot_id, *ctx.symints)

        @staticmethod
        def forward(ctx, *deduped_flat_tensor_args):
            args = deduped_flat_tensor_args

            marked_dirty_inps = []
            for i in fw_metadata.mutated_graph_handled_indices:
                ctx.mark_dirty(deduped_flat_tensor_args[i])
                marked_dirty_inps.append(deduped_flat_tensor_args[i])

            if CompiledFunction.metadata.is_rng_op_functionalized:
                # Add the seed and offset to args
                seed, offset = CUDARngStateHelper.get_torch_state_as_tuple()
                args = (*args, seed, offset)
            # There is a pretty complicated calling convention around what the compiled fw returns.
            # The full list of outputs and their relative order is:
            # (*mutated_inputs, *fw_outs, *fw_intermediate_bases, *saved_tensors, *saved_symints)
            # - Note that in the synthetic bases case, mutated_inputs will correspond to an updated version
            #   of the original view, and not the synthetic base
            fw_outs = call_func_at_runtime_with_args(
                CompiledFunction.compiled_fw,
                args,
                disable_amp=disable_amp,
            )

            num_outputs = CompiledFunction.metadata.num_outputs
            num_outputs_aliased = CompiledFunction.metadata.num_outputs_aliased
            num_intermediate_bases = CompiledFunction.metadata.num_intermediate_bases
            num_symints_saved_for_bw = CompiledFunction.num_symints_saved_for_bw
            num_mutated_runtime_inps = CompiledFunction.metadata.num_mutated_inp_runtime_indices
            num_forward_returns = CompiledFunction.metadata.num_forward_returns
            num_forward = CompiledFunction.metadata.num_forward

            # Partitioners must put symint arguments at the end separate from tensor arguments
            tensors_saved_for_backwards = fw_outs[
                CompiledFunction.metadata.tensors_saved_for_backwards_slice
            ]
            assert all(
                isinstance(x, torch.Tensor) for x in tensors_saved_for_backwards
            )
            # See Note [Detaching saved tensors in AOTAutograd]
            ctx.save_for_backward(*(x.detach() if x._is_view() else x for x in tensors_saved_for_backwards))
            symint_outs = fw_outs[CompiledFunction.metadata.symints_saved_for_backwards_slice]
            assert all(
                isinstance(x, (int, float, torch.SymInt, torch.SymFloat))
                for x in symint_outs
            ), str([type(x) for x in symint_outs])
            ctx.symints = symint_outs

            raw_returns = fw_outs[0:num_forward_returns]

            # Wrap all autograd.Function.forward() outputs that are aliases
            # so that autograd.Function doesn't treat them as tensors
            if num_mutated_runtime_inps > 0:
                for i, idx in enumerate(
                    CompiledFunction.metadata.mutated_inp_runtime_indices
                ):
                    # We could make this faster by only looping over inputs with metadata-only mutations
                    # (instead of looping over inputs with either data or metadata mutations), but there shouldn't be many.
                    info = CompiledFunction.metadata.input_info[idx]
                    if info.mutates_metadata and not info.mutates_data:
                        raw_returns[i] = TensorAlias(raw_returns[i])

                if config.debug_assert:
                    user_mutated_inputs_raw = raw_returns[0:num_mutated_runtime_inps]
                    mut_inp_infos = [
                        x for x in CompiledFunction.metadata.input_info if x.mutates_data or x.mutates_metadata
                    ]
                    assert len(user_mutated_inputs_raw) == len(mut_inp_infos)

            if CompiledFunction.metadata.num_unsafe_view_outputs > 0:
                for idx in CompiledFunction.metadata.unsafe_view_out_indices:
                    raw_return_idx = num_mutated_runtime_inps + idx
                    o = raw_returns[raw_return_idx]
                    raw_returns[raw_return_idx] = torch.ops.aten._unsafe_view(o, o.shape)

            if num_outputs_aliased > 0:
                for idx in CompiledFunction.metadata.aliased_out_indices:
                    raw_return_idx = num_mutated_runtime_inps + idx
                    raw_returns[raw_return_idx] = TensorAlias(raw_returns[raw_return_idx])

                if config.debug_assert:
                    intermediates_raw = raw_returns[num_mutated_runtime_inps + num_outputs:]
                    assert not any(isinstance(x, TensorAlias) for x in intermediates_raw)

            # invariant: intermediate bases always require gradients, so we don't have to
            # consider marking them as non-differentiable.
            raw_returns_not_including_intermediate_bases = raw_returns[:num_mutated_runtime_inps + num_outputs]
            raw_returns_meta = (
                [
                    x for x in CompiledFunction.metadata.input_info
                    if x.mutation_type == MutationType.MUTATED_OUT_GRAPH
                ] + CompiledFunction.metadata.output_info
            )

            fw_outs_not_requiring_grad = [
                x
                for (i, x) in enumerate(raw_returns_not_including_intermediate_bases)
                if isinstance(x, torch.Tensor)
                and not raw_returns_meta[i].requires_grad
            ]
            ctx.mark_non_differentiable(*fw_outs_not_requiring_grad)
            ctx._materialize_non_diff_grads = False

            functionalized_rng_runtime_epilogue(
                CompiledFunction.metadata,
                fw_outs[num_forward_returns:num_forward],
                return_new_outs=False
            )
            return tuple(raw_returns) + tuple(marked_dirty_inps)

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
            num_intermediate_bases = CompiledFunction.metadata.num_intermediate_bases
            num_graph_handled_inputs = CompiledFunction.metadata.num_mutated_graph_handled_indices
            num_mutated_runtime_inps = CompiledFunction.metadata.num_mutated_inp_runtime_indices
            expected_grad_outs = (
                CompiledFunction.metadata.num_outputs + num_mutated_runtime_inps + num_intermediate_bases
            )

            if num_graph_handled_inputs > 0:
                flat_args = flat_args[:-num_graph_handled_inputs]
            assert len(flat_args) == expected_grad_outs
            out_info = CompiledFunction.metadata.output_info

            inp_tangents, out_tangents, intermediate_base_tangents = (
                flat_args[0:num_mutated_runtime_inps],
                flat_args[num_mutated_runtime_inps:num_mutated_runtime_inps + CompiledFunction.metadata.num_outputs],
                flat_args[num_mutated_runtime_inps + CompiledFunction.metadata.num_outputs:],
            )
            # input_info contains info on *every* input,
            # But in the backward(), we are only given grad outputs for every mutated input
            # We then need to filter out the grad outputs that correspond to metadata-only mutations or don't require grad
            input_info = CompiledFunction.metadata.input_info
            inp_tangents_filtered = [
                x
                for x, info_idx in zip(inp_tangents, CompiledFunction.metadata.mutated_inp_runtime_indices)
                if input_info[info_idx].mutates_data and input_info[info_idx].requires_grad
            ]
            # We also need to filter out grad outputs that correspond to outputs aliasing inputs/intermediates
            out_tangents_filtered = [
                x
                for x, info in zip(out_tangents, out_info)
                if info.output_type in [OutputType.non_alias, OutputType.unsafe_view_alias, OutputType.custom_function_view]
                and issubclass(info.raw_type, torch.Tensor)
                and info.requires_grad
            ]
            # intermediate bases always require gradients, and always participate in the backward graph.
            flat_bw_args_with_grads = [*inp_tangents_filtered, *out_tangents_filtered, *intermediate_base_tangents]
            num_flat_bw_args_with_grads = len(flat_bw_args_with_grads)

            # sanity asserts
            # metadata_only_inps = [
            #     x for x, info_idx in zip(inp_tangents, mutated_inp_indices)
            #     if not input_info[info_idx].mutates_data
            # ]
            # aliased_outputs = [
            #     x for x, info in zip(out_tangents, out_info) if info.output_type != OutputType.non_alias]
            # assert all(x is None for x in metadata_only_inps)
            # assert all(x is None for x in aliased_outputs)

            rng_args = []
            if CompiledFunction.metadata.is_rng_op_functionalized:
                # Add the seed and offset to args
                rng_args = CUDARngStateHelper.get_torch_state_as_tuple()

            all_args = [
                *ctx.symints,
                *ctx.saved_tensors,
                *flat_bw_args_with_grads,
                *rng_args
            ]
            del flat_bw_args_with_grads

            tangents_start_idx = len(all_args) - num_flat_bw_args_with_grads - len(rng_args)
            tangents_end_idx = len(all_args) - len(rng_args)

            # Note: [AOTAutograd Backward Guards]
            # During AOTDispatch, we eagerly create and trace out a joint fw-bw graph.
            # Doing so requires us to "guess" about some of the metadata of our grad_outputs.
            #
            # In particular: if an output to the forward is a plain tensor or a subclass,
            # its corresponding grad_output in the backward **may or may not** be
            # a plain tensor or a subclass. The main cases are:
            # (1) If an output is a plain tensor, its grad_out will also be a plain tensor,
            #     *unless* the output is used in some subclass compute later in the forward graph,
            #     which will cause its grad_output to become a subclass
            # (2) If an output is a subclass, its grad_out will also be a subclass,
            #     *unless* the output of the forward did not actually participate in the gradient computation,
            #     in which case autograd will insert a plain tensor of zeros for the grad_output.
            #     We could avoid this case with `torch.autograd.Function.set_materialize_grads`,
            #     although this is not turned on today in AOTAutgrad and would require more work.
            #
            # Today, we make a guess on subclass-ness based on the above examples,
            # and hard-error in the backward if we guessed wrong.
            #
            # In the future, we should add backward guards that would allow us to
            # properly handle this case instead of erroring: we would need to retrace the backward graph,
            # since we might produce an entirely different trace if our grad_outputs are subclass or not.
            assert len(CompiledFunction.metadata.output_types) == num_flat_bw_args_with_grads
            grad_output_types = [type(x) for x in all_args[-num_flat_bw_args_with_grads:]]
            # In general, we can add more asserts/guards here for when we partitioned
            # with incorrect assumptions about the grad_outputs.
            # Normalize FakeTensor -> torch.Tensor
            # - during tracing our types are FakeTensor
            # - at runtime in the backward our types are torch.Tensor...
            # - unless we're running compiled backward, in which case they are also FakeTensor
            grad_output_types_ = [torch.Tensor if x is FakeTensor else x for x in grad_output_types]
            assert grad_output_types_ == CompiledFunction.metadata.output_types, f"""\
We incorrectly attempted to compile the backward with incorrect subclass metadata.
If you run into this error, please file an issue.
Expected grad_output types: {str(CompiledFunction.metadata.output_types)}
Got grad_output types: {str(grad_output_types)}"""

            # TODO: figure out how to refactor the backward properly so I can use aot_dispatch_subclass_wrapper() here.
            if CompiledFunction.maybe_subclass_metadata is not None:
                # Get the number of tangents after unwrapping
                len_tangents = len(unwrap_tensor_subclasses(
                    all_args[tangents_start_idx: tangents_end_idx], is_joint_structure=False
                ))
                all_args = unwrap_tensor_subclasses(all_args, is_joint_structure=False)
                tangents_start_idx = len(all_args) - len_tangents - len(rng_args)
                tangents_end_idx = tangents_start_idx + len_tangents

            # Make the tangents contiguous. Note that we must do this after subclass desugaring
            # because inputs to inductor have to be contiguous
            all_args = [
                t.contiguous() if tangents_start_idx <= i < tangents_end_idx else t
                for i, t in enumerate(all_args)
            ]

            def call_compiled_backward():
                if ctx._is_compiled_autograd_tracing():
                    # For compiled autograd, run raw FX graph so that it can be inlined into the larger graph
                    symints = ctx._get_compiled_autograd_symints()
                    assert len(symints) == len(ctx.symints)
                    all_args[:len(symints)] = symints
                    context = torch._C._DisableAutocast if disable_amp else nullcontext
                    with context():
                        out = normalize_as_list(bw_module(*all_args))
                    out = functionalized_rng_runtime_epilogue(CompiledFunction.metadata, out)
                    return tuple(out)
                ctx.maybe_clear_saved_tensors()
                if CompiledFunction.compiled_bw is None:
                    context = torch._C._DisableAutocast if disable_amp else nullcontext
                    with tracing(saved_context), context(), track_graph_compiling(aot_config, "backward"):
                        CompiledFunction.compiled_bw = aot_config.bw_compiler(
                            bw_module, placeholder_list
                        )

                out = call_func_at_runtime_with_args(
                    CompiledFunction.compiled_bw,
                    all_args,
                    steal_args=True,
                    disable_amp=disable_amp,
                )

                out = functionalized_rng_runtime_epilogue(CompiledFunction.metadata, out)
                return tuple(out)

            if torch.is_grad_enabled() and any(t.requires_grad for t in all_args if isinstance(t, torch.Tensor)):
                # Ensure that the graph is connected, and error if double backward is performed.
                # See comment for why once_differentiable is not sufficient:
                # https://github.com/pytorch/pytorch/pull/92348/files#r1072962107
                class CompiledFunctionBackward(torch.autograd.Function):
                    @staticmethod
                    def forward(ctx, *unused_args):
                        outs = call_compiled_backward()
                        # TODO: figure out how to refactor the backward properly so I can use aot_dispatch_subclass_wrapper() here.
                        if CompiledFunction.maybe_subclass_metadata is not None:
                            outs_wrapped = wrap_tensor_subclasses(
                                outs, subclass_metas=CompiledFunction.maybe_subclass_metadata.grad_input_metas)
                            return outs_wrapped
                        return outs

                    @staticmethod
                    def backward(ctx, *args):
                        raise RuntimeError("torch.compile with aot_autograd does not currently support double backward")

                CompiledFunctionBackward._compiled_autograd_key = CompiledFunction._compiled_autograd_key

                # Pass args even though they're unused, so that the graph is built
                out = CompiledFunctionBackward.apply(*all_args)
            else:
                out = call_compiled_backward()

            # TODO: figure out how to refactor the backward properly so I can use aot_dispatch_subclass_wrapper() here.
            if CompiledFunction.maybe_subclass_metadata is not None:
                outs_wrapped = wrap_tensor_subclasses(
                    out, subclass_metas=CompiledFunction.maybe_subclass_metadata.grad_input_metas)
                return outs_wrapped
            return out

    compiled_function = create_runtime_wrapper(
        CompiledFunction.apply,
        runtime_metadata=fw_metadata,
        indices_of_inps_to_detach=_indices_of_inps_to_detach,
        trace_joint=True,
        keep_input_mutations=aot_config.keep_inference_input_mutations,
        disable_amp=disable_amp
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

    Note: this function is used both by aot_function and aot_export (controlled by aot_config.is_export)
        When aot_config.is_export is True, we return an FX graph + metadata
        When aot_config.is_export is False, we return an ordinary runtime function
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

    if config.functionalize_rng_ops:
        # Update the decompositions with functionalized random decompositions
        aot_config.decompositions = {
            **rng_decompositions,
            **aot_config.decompositions,
        }

    # Check flat_args to see if they're already fake.  If so, use that fake
    # mode instead.

    fake_mode = detect_fake_mode(flat_args)
    if fake_mode is None:
        shape_env = ShapeEnv() if aot_config.dynamic_shapes else None
        fake_mode = FakeTensorMode(shape_env=shape_env)
    else:
        shape_env = fake_mode.shape_env

    python_dispatcher_mode = (
        enable_python_dispatcher() if shape_env is not None else nullcontext()
    )

    with torch.autograd.set_multithreading_enabled(
        False
    ), preserve_rng_state(), fake_mode, python_dispatcher_mode, PhiloxStateTracker():

        def process_inputs(flat_args):
            def convert(idx, x):
                if shape_env is not None:
                    from torch._dynamo.source import ConstantSource
                    if isinstance(x, int):
                        source = ConstantSource(f"sym_{idx}")
                        return shape_env.create_symintnode(
                            shape_env.create_symbol(x, source),
                            hint=x,
                            source=source
                        )
                if not isinstance(x, torch.Tensor):
                    return x
                if isinstance(x, FakeTensor):
                    assert x.fake_mode is fake_mode
                    return x
                if is_traceable_wrapper_subclass(x):
                    attrs, _ = x.__tensor_flatten__()
                    if all(isinstance(getattr(x, attr), FakeTensor) for attr in attrs):
                        assert all(getattr(x, attr).fake_mode is fake_mode for attr in attrs)
                        return x


                # see note [Tensor Fakification and Symbol Caching]
                symbolic_context = None
                source = None
                if tracing_context := torch._guards.TracingContext.try_get():
                    if x in tracing_context.tensor_to_context:
                        symbolic_context = tracing_context.tensor_to_context[x]
                        source = symbolic_context.tensor_source
                if (
                    idx < aot_config.num_params_buffers
                    and config.static_weight_shapes
                    and not symbolic_context
                ):
                    # TODO: Ensure that this codepath is never exercised from
                    # Dynamo
                    return fake_mode.from_tensor(x, static_shapes=True)

                return fake_mode.from_tensor(
                    x, static_shapes=False, symbolic_context=symbolic_context, source=source
                )

            return [convert(idx, x) for idx, x in enumerate(flat_args)]

        fake_flat_args = process_inputs(flat_args)

        needs_autograd = (
            any(x.requires_grad for x in fake_flat_args if isinstance(x, Tensor))
            and torch.is_grad_enabled()
        )

        with enable_python_dispatcher():
            # Patch set_rng_state as set_rng_state with fake tensors is
            # nonsensical. This does not affect the collection of metadata.
            with patch("torch.cuda.set_rng_state", lambda *args: None):
                fw_metadata = run_functionalized_fw_and_collect_metadata(
                    flat_fn,
                    keep_input_mutations=aot_config.keep_inference_input_mutations,
                    is_train=needs_autograd,
                )(*fake_flat_args)

                req_subclass_dispatch = requires_subclass_dispatch(fake_flat_args, fw_metadata)

                if needs_autograd and not any(x.requires_grad for x in fw_metadata.output_info):
                    # We realized that none of the outputs require grad,
                    # so we actually have an inference graph.
                    needs_autograd = False
                    # A bit silly: right now in the subclass codepath, our ViewAndMutationMeta
                    # changes depending on whether we pass in is_train / keep_input_mutations,
                    # so we're forced to recompute the metadata.
                    # TODO: refactor the subclass path of run_functionalized_fw_and_collect_metadata
                    # so that this is unnecessary.
                    if req_subclass_dispatch:
                        fw_metadata = run_functionalized_fw_and_collect_metadata(
                            flat_fn,
                            keep_input_mutations=aot_config.keep_inference_input_mutations and not needs_autograd,
                            is_train=needs_autograd,
                        )(*fake_flat_args)
                    else:
                        fw_metadata = ViewAndMutationMeta(
                            input_info=fw_metadata.input_info,
                            output_info=fw_metadata.output_info,
                            num_intermediate_bases=fw_metadata.num_intermediate_bases,
                            keep_input_mutations=aot_config.keep_inference_input_mutations and not needs_autograd,
                            traced_tangents=fw_metadata.traced_tangents,
                            subclass_inp_meta=fw_metadata.subclass_inp_meta,
                            subclass_fw_graph_out_meta=fw_metadata.subclass_fw_graph_out_meta,
                            subclass_tangent_meta=fw_metadata.subclass_tangent_meta,
                            is_train=needs_autograd,
                        )


        if fw_metadata.num_intermediate_bases > 0:
            assert not req_subclass_dispatch, f"""\
torch.compile is currently being used with tensor subclass inputs:
{','.join([str(type(x)) for x in fake_flat_args])}. We are attempting to a compile a graph with two graph outputs
that alias one another, which is currently unsupported in the subclass use case. If you run into this,
please file a github issue"""

        if aot_config.is_export:
            # aot_export: ban input metadata mutations for now to keep shared code paths simpler.
            # Keeping .resize_() in the graph will require some work
            # Allowing it but keeping the graph functional will require some calling convention changes.
            if len([x for x in fw_metadata.input_info if x.mutates_metadata]) != 0:
                raise RuntimeError(f"""\
Found an input that received a metadata mutation, through e.g. a call to `.resize_()` or `.transpose_()`.
This is currently banned in the aot_export workflow. If you need this functionality, please file a github issue.

fw_metadata={str(fw_metadata)}""")
            # In export, banning data mutations on inputs that require grad for now.
            # This should be rare, and is tricky to get right. When we trace the backward,
            # we currently trace with autograd.grad instead of .backward(), which makes it difficult
            # to ensure that we run autograd all the way through the input **before** it saw the mutation.
            if len([x for x in fw_metadata.input_info if x.requires_grad and x.mutates_data]) != 0:
                raise RuntimeError(f"""\
Found a graph input that requires gradients, and received a mutation.
This is currently banned in the aot_export workflow. If you need this functionality, please file a github issue.

fw_metadata={str(fw_metadata)}""")
            if req_subclass_dispatch:
                raise RuntimeError("""\
aot_export is not currently supported with traceable tensor subclass.
If you need this feature, please comment on <CREATE_ISSUE_LINK>""")

            # Need to decide on a strategy for functionalized RNG: toggling via global config seems bad,
            # and turning it on will require a non-trivial calling convention change for any export runtime.
            if config.functionalize_rng_ops:
                raise RuntimeError("""\
Functionalized RNG is not currently supported in the aot_export workflow. Please file a github issue,
or otherwise set torch._functorch.config.functionalize_rng_ops = False.""")

        # crappy version of dispatcher
        # TODO: Do this properly
        if needs_autograd:
            # For now, aot_dispatch_autograd knows to explicitly return a graph
            # when run with export, and an opaque callable otherwise.
            # In theory we could factor these out, but I wanted to let the dust
            # settle on how functionalized rng fits into export first.
            compiler_fn = aot_dispatch_autograd_graph if aot_config.is_export else aot_dispatch_autograd
        else:
            # aot_dispatch_base_graph contains only the "graph bits", while aot_dispatch_base
            # includes some extra work around handling a runtime epilogue.
            compiler_fn = aot_dispatch_base_graph if aot_config.is_export else aot_dispatch_base

        compiler_fn = partial(aot_wrapper_synthetic_base, compiler_fn=compiler_fn, needs_autograd=needs_autograd)
        compiler_fn = partial(aot_wrapper_dedupe, compiler_fn=compiler_fn)
        # You can put more passes here

        compiled_fn = compiler_fn(flat_fn, fake_flat_args, aot_config, fw_metadata=fw_metadata)
        if aot_config.is_export:
            mutated_user_inp_locs = [
                idx - aot_config.num_params_buffers
                for idx in fw_metadata.mutated_inp_runtime_indices
                if idx >= aot_config.num_params_buffers
            ]
            if len(mutated_user_inp_locs) > 0:
                raise RuntimeError(f"""
Found following user inputs located at {mutated_user_inp_locs} are mutated. This is currently banned in the aot_export workflow.
If you need this functionality, please file a github issue.

fw_metadata={str(fw_metadata)}""")

            # During export, we don't get back a callable - we get back the raw fx graph
            # (either a joint or an inference-only graph)
            assert isinstance(compiled_fn, torch.fx.GraphModule)
            return compiled_fn, fw_metadata

        if not hasattr(compiled_fn, "_boxed_call"):
            compiled_fn = make_boxed_func(compiled_fn)

        return compiled_fn



def create_functional_call(mod, params_spec, params_len):
    # Redundant with dynamo, but worth having in case this gets invoked elsewhere.
    # https://github.com/pytorch/pytorch/issues/103569

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
                "pytree processing of the outputs. Please change the module to "
                "have tuple outputs or use aot_module instead."
            )
        return out
    return functional_call


def aot_function(
    fn: Callable,
    fw_compiler: Callable,
    bw_compiler: Optional[Callable] = None,
    partition_fn: Callable = default_partition,
    decompositions: Optional[Dict] = None,
    num_params_buffers: int = 0,
    keep_inference_input_mutations: bool = False,
    inference_compiler: Optional[Callable] = None,
    *,
    # Whether or not to trace with dynamic shapes
    dynamic=False,
    enable_log=True,
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
        inference_compiler (Optional[Callable]): A Python function that accepts an
            Fx graph with Aten ops and input args, and returns a Callable that
            semantically is equivalent to the input Fx graph. inference_compiler is invoked
            if no autograd is needed. Default: None
            (when None, it defaults to the :attr:`fw_compiler`)
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

    if bw_compiler is None:
        bw_compiler = fw_compiler
    if inference_compiler is None:
        inference_compiler = fw_compiler
    aot_config = AOTConfig(
        fw_compiler=fw_compiler,
        bw_compiler=bw_compiler,
        inference_compiler=inference_compiler,
        partition_fn=partition_fn,
        decompositions=decompositions,
        num_params_buffers=num_params_buffers,
        aot_id=next(AOT_COUNTER),
        keep_inference_input_mutations=keep_inference_input_mutations,
        dynamic_shapes=dynamic,
        aot_autograd_arg_pos_to_source=None,
        is_export=False,
        no_tangents=False,
        enable_log=enable_log,
    )
    cached_res = None

    @wraps(fn)
    def returned_function(*args, **kwargs):
        nonlocal cached_res
        # Now flatten the tensor args
        flat_args = pytree.arg_tree_leaves(*args, **kwargs)

        # Compile the function and save it in the cache
        if cached_res is None:
            flat_fn, out_spec = create_tree_flattened_fn(fn, args, kwargs)

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
        functional_call, *args, num_params_buffers=num_params_buffers, **kwargs
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
    keep_inference_input_mutations=False,
    inference_compiler: Optional[Callable] = None,
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
    params = {
        **dict(mod.named_parameters(remove_duplicate=False)),
        **dict(mod.named_buffers(remove_duplicate=False)),
    }
    params_flat, params_spec = pytree.tree_flatten(params)
    params_flat = list(params_flat)
    params_len = len(params_flat)

    functional_call = create_functional_call(mod, params_spec, params_len)

    if bw_compiler is None:
        bw_compiler = fw_compiler
    if inference_compiler is None:
        inference_compiler = fw_compiler

    seen_sources = set()

    full_args = []
    # First, the params
    full_args.extend(params_flat)

    if tracing_context := torch._guards.TracingContext.try_get():
        tracing_context.params_flat = params_flat

    aot_autograd_arg_pos_to_source = None
    # Then, the params 1:1 mapped sources, if relevant.
    if hasattr(mod, "_param_name_to_source"):
        aot_autograd_arg_pos_to_source = []
        # We now know this came from dynamo, and (1) we care about guards,
        # so setting up aot_autograd_arg_pos_to_source for downstream dedup guards
        # can now be done safely. (2) Dynamo logic protects the 1:1 sizing below.
        for name in params.keys():
            assert name in mod._param_name_to_source, f"{name} not found."
            source = mod._param_name_to_source[name]
            assert source not in seen_sources, source
            seen_sources.add(source)
            aot_autograd_arg_pos_to_source.append(source)

    # Next, the input args
    full_args.extend(args)

    if hasattr(mod, "graph"):
        # Non dynamo entrypoints can get to here...
        for i, node in enumerate(mod.graph.nodes):
            if node.op == "placeholder":
                if hasattr(node, "_dynamo_source"):
                    # ... but not here!
                    if aot_autograd_arg_pos_to_source is None:
                        aot_autograd_arg_pos_to_source = []
                    source = node._dynamo_source
                    assert source not in seen_sources, source
                    seen_sources.add(source)
                    aot_autograd_arg_pos_to_source.append(source)

    if aot_autograd_arg_pos_to_source is not None:
        assert len(full_args) == len(aot_autograd_arg_pos_to_source)

    dynamic_shapes = False
    for x in full_args:
        if isinstance(x, FakeTensor):
            dynamic_shapes = x.fake_mode.shape_env is not None
            break

    aot_config = AOTConfig(
        fw_compiler=fw_compiler,
        bw_compiler=bw_compiler,
        inference_compiler=inference_compiler,
        partition_fn=partition_fn,
        decompositions=decompositions,
        num_params_buffers=params_len,
        aot_id=next(AOT_COUNTER),
        keep_inference_input_mutations=keep_inference_input_mutations,
        dynamic_shapes=dynamic_shapes,
        aot_autograd_arg_pos_to_source=aot_autograd_arg_pos_to_source,
        is_export=False,
        no_tangents=False,
    )

    with compiled_autograd.disable():
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

def aot_export_module(
    mod: nn.Module,
    args,
    *,
    decompositions: Optional[Dict] = None,
    # If true, we'll return a joint forward-backward graph,
    # As well as metadata on the loss + gradients in the backward.
    trace_joint: bool,
    # If trace_joint is True, we expect your module to return a scalar loss.
    # Your module can return multiple outputs, so you must specify which output the loss is.
    output_loss_index: Optional[int] = None,
) -> Tuple[torch.fx.GraphModule, GraphSignature]:
    """
    This function takes in a module, and returns:
    (1) an FX graph that can be exported
    (2) some metadata about the graph

    If `trace_joint=True` we will return a joint graph of the forward + backward.

    The traced FX graph will have the following properties compared to the original module:
    (1) Inputs and outputs to the module will be pytree-flattened
    (2) Parameters and buffers on the module will be lifted into graph inputs,
        graph_inputs = (*parameters, *buffers, *user_inputs)
    (3) The graph will be fully functionalized
    (4) Any input mutations will be converted into additional outputs in the graph,
        meaning whoever calls this graph is responsible for applying the mutations
        back to the original inputs.
    (5) If is_joint is provided the graph will return parameter gradients in addition to user outputs.
        The graph output will look like:
        graph_outputs = (*updated_inputs, *user_outputs, *param_gradients)

    There are also several restrictions on what modules can use this API. In particular:
    (1) If trace_joint is specified, we expect the loss function to be **fused**
        into the module forward. One of the outputs to the forward must be a scalar loss,
        which is specified with `output_loss_index`.
        All other outputs to the forward are presumed to not require gradients.
    (2) This API cannot capture optimizers (although in theory we could build an API for this).
    (3) Metadata mutations on params/buffers/inputs are banned.
    (4) Data mutations on anything that requires gradients are banned (parameters)
    (5) If an input is mutated, it is not allowed to alias any other inputs.
    (6) Parameters must not be duplicated.
    """
    named_parameters = dict(mod.named_parameters(remove_duplicate=False))
    named_buffers = dict(mod.named_buffers(remove_duplicate=False))
    params_and_buffers = {
        **dict(named_parameters),
        **dict(named_buffers),
    }
    params_and_buffers_flat, params_spec = pytree.tree_flatten(params_and_buffers)
    params_and_buffers_flat = tuple(params_and_buffers_flat)
    params_len = len(params_and_buffers_flat)

    functional_call = create_functional_call(mod, params_spec, params_len)

    num_fw_outs = None

    if trace_joint:
        # This helper effectively just adds some extra asserts about what the backward will look like:
        # Outputs must include a scalar loss, that we compute gradients w.r.t.
        # We don't compute gradients w.r.t. anything else: so just in case we detach()
        # and other output tensors.
        def fn_to_trace(*args):
            nonlocal num_fw_outs
            out = functional_call(*args)
            if output_loss_index is None:
                raise RuntimeError("""\
If trace_joint=Trueit is required that one of your forward outputs must be a scalar loss.
You must specify the which (index) output is the loss with output_loss_index.""")
            if isinstance(out, (torch.Tensor)):
                out = (out,)
            if not isinstance(out, (tuple, list)):
                raise RuntimeError(f"Expected forward output to be either a tensor or a list/tuple of tensors. found {type(out)}")

            for i, o in enumerate(out):
                # We only want to create a backward graph w.r.t. the loss that the user passed in.
                # This implies that every other output should not require gradients.
                # Instead of making this an error (and forcing the user to detach all other outputs
                # of their forward),
                # we'll automatically detach them here.
                if o.requires_grad and i != output_loss_index:
                    raise RuntimeError(f"""\
Found an output of the forward that requires gradients, that was not the scalar loss.
We require all outputs to the forward that are not the scalar loss to not require gradient,
because we will only compute a backward graph against the scalar loss.
You can fix this by calling .detach() on each of your forward outputs that is not the loss.
You specified that output index {output_loss_index} is the loss, but we found that
the output at index {i} requires gradients.""")
            out_loss = out[output_loss_index]
            num_fw_outs = len(out)
            if not out_loss.requires_grad:
                raise RuntimeError(f"""\
The output at index {output_loss_index} was marked as the loss, but it does not require gradients""")
            if out_loss.numel() != 1:
                raise RuntimeError(f"""\
We require the output marked as the loss (at index {output_loss_index}) to be a scalar, but it has shape {out_loss.shape}""")
            return out
        ctx = nullcontext
    else:
        # Run under no_grad, so our tracing machinery only traces an inference graph.
        ctx = torch.no_grad
        fn_to_trace = functional_call

    full_args = []
    # First, the params
    # NB: It is REQUIRED that parameters come first, Inductor infers "fixed"
    # parameters by looking at the difference in parameter count outside
    # and inside AOTAutograd, and assumes the prefix of arguments are fixed
    # arguments
    full_args.extend(params_and_buffers_flat)
    # Next, the input args
    full_args.extend(args)

    with ctx():
        fx_g, metadata, in_spec, out_spec = _aot_export_function(
            fn_to_trace,
            full_args,
            decompositions=decompositions,
            num_params_buffers=params_len,
            no_tangents=True,
        )
    if trace_joint:
        def flattened_joint(*args):
            # The idea here is that the joint graph that AOTAutograd creates has some strict properties:
            # (1) It accepts two arguments (primals, tangents), and pytree_flattens them
            # (2) It returns a tuple of (fw_outs, gradients)
            # This is a very useful convention for anyone who wants to partition the joint graph
            # into a separate forward and backward graph.
            # However,
            # (1) for people exporting a single joint graph, it would be preferable not to have
            #     any pytrees in the graph.
            # (2) We are guaranteed in the aot_export_module case that the forward outputs a loss,
            #     and there are therefore no tangents that are needed to run the joint graph.
            # (3) AOTAutograd creates a grad_input for every input in the forward,
            #     including None's for inputs that are not grad-requiring tensors.
            #     we don't want these in our export graph.
            #     and there are therefore no tangents that are needed to run the joint graph.
            # This function "fixes" both of the above by removing any tangent inputs,
            # and removing pytrees from the original FX graph.
            fake_tangents = [None for _ in range(metadata.num_outputs + metadata.num_mutated_inp_runtime_indices)]
            fw_outs, gradients = fx_g(args, fake_tangents)
            assert len(gradients) == len(args)
            output_gradients = []
            for i, (a, grad) in enumerate(zip(args, gradients)):
                if isinstance(a, torch.Tensor) and a.requires_grad:
                    assert grad is not None, """\
Found a parameter that did not receive a gradient.
"This is most likely a bug, but if this needs to be supported please comment on this Github issue:
https://github.com/pytorch/pytorch/issues/101192
"""
                    output_gradients.append(grad)
                else:
                    assert grad is None
            return *fw_outs, *output_gradients
        fx_g = make_fx(flattened_joint)(*full_args)

    user_args_flat = pytree.arg_tree_leaves(*args)
    return fx_g, create_graph_signature(
        fx_g,
        metadata,
        in_spec,
        out_spec,
        user_args_flat=user_args_flat,
        params_and_buffers_flat=params_and_buffers_flat,
        param_names=list(named_parameters.keys()),
        buffer_names=list(named_buffers.keys()),
        trace_joint=trace_joint,
        num_user_fw_outs=num_fw_outs,
        loss_index=output_loss_index,
    )

def aot_export_joint_simple(
    func: Callable,
    args,
    *,
    trace_joint: bool,
    # It looks like the main consequence of this API is that for dynamic shapes,
    # it will assume that parms/buffers are static.
    # With the new inferred dynamic shapes API, maybe this doesn't matter?
    num_params_buffers: int = 0,
    decompositions: Optional[Dict] = None,
) -> torch.fx.GraphModule:
    """
    A simplified version of export. Used by higher order operators.

    This function makes a high-level "no calling convention changes" guarantee:
    - If no inputs require grad (so we export an inference graph),
      there are *no* calling convention change between the exported graph, and "func".
    - If at least one input requires grad (so we trace out and export a joint fw-bw graph),
      Then if you were partition the graph into a separate forward and backward graph,
      The forward graph will have no calling convention changes compared to "func".

    The above also relies on some strong restrictions around which functions this API accepts:
    (1) `args` cannot contain any pytrees (they must have been pytree_flattened already)
    (2) `func` cannot mutate any inputs
    (3) The outputs of `func` cannot alias any inputs.

    Note: this function is only lightly tested today. It will probably be tested more heavily by higher order ops.
    """
    if trace_joint:
        ctx = nullcontext
    else:
        # Run under no_grad, so our tracing machinery only traces an inference graph.
        ctx = torch.no_grad

    with ctx():
        fx_g, metadata, in_spec, out_spec = _aot_export_function(
            func,
            args,
            decompositions=decompositions,
        )
    # At this point, we can just directly return the (joint or inference graph) that we traced.
    # First though: a bunch of assertions to make sure that our graph doesn't require
    # any calling convention changes compared to the original function.
    # These restrictions are *in addition to* the general restrictions on export.

    # No input mutations
    if len([x for x in metadata.input_info if x.mutates_data or x.mutates_metadata]) != 0:
        raise RuntimeError(f"aot_export_joint_simple does not support input mutations. {str(metadata)}")
    # No output aliasing
    if len([x for x in metadata.output_info if x.output_type != OutputType.non_alias]) != 0:
        raise RuntimeError(f"aot_export_joint_simple does not support outputs that alias inputs. {str(metadata)}")
    # No pytrees
    if type(in_spec) == pytree.LeafSpec:
        raise RuntimeError(f"aot_export_joint_simple requires inputs to be a single list/tuple. in_spec={str(in_spec)}")
    if len([x for x in in_spec.children_specs if type(x) != pytree.LeafSpec]) != 0:
        raise RuntimeError(f"aot_export_joint_simple requires individual inputs not to be pytrees. in_spec={str(in_spec)}")
    if type(out_spec) == pytree.LeafSpec:
        raise RuntimeError(f"aot_export_joint_simple requires outputs to be a single list/tuple. out_spec={str(out_spec)}")
    if len([x for x in out_spec.children_specs if type(x) != pytree.LeafSpec]) != 0:
        raise RuntimeError(f"aot_export_joint_simple requires individual outputs not to be pytrees. out_spec={str(out_spec)}")
    # TODO: we might have to temporarily patch config.functionalize_rng
    # so that it doesn't run when we're exporting a higher order op.

    if config.debug_assert:
        # Smoke test that after partitioning, we can run the forward without any calling convention changes.
        fw_module, bw_module = aot_config.default_partition(
            fx_g, args, num_fwd_outputs=len(fw_metadata.output_infos)
        )
        # Attempt to run the fw_module with the original user inputs
        fake_mode = detect_fake_mode(args)
        if fake_mode is None:
            fake_mode = FakeTensorMode()
        with fake_mode:
            fw_module(*args)
    return fx_g

# Private for now because we aren't providing a contract on what to return
# for joint graphs (we could when there's a clearer use case)
# In the future, we may need to add more export API's that provide their own strong guarantees.
# This is meant as a general helper function for handling various export-y use cases.
def _aot_export_function(
    func: Callable,
    args,
    *,
    num_params_buffers: int = 0,
    decompositions: Optional[Dict] = None,
    # If we're exporting a joint graph and we don't want any tangent inputs in the graph
    # (because we are backpropping through a scalar 1 loss),
    # we need to explicitly specify not to include tangents in the graph.
    # It's not enough just to check that our tangent is a scalar, since we also
    # need to know if it is a 1 (no need to make it a graph input), or something else
    # (requiring it to be a graph input).
    # We don't know this info at trace time though, so we need to make it an explicit config.
    no_tangents: bool = False,
) -> Tuple[torch.fx.GraphModule, ViewAndMutationMeta, pytree.TreeSpec, pytree.TreeSpec]:
    dynamic_shapes = False
    for x in args:
        if isinstance(x, FakeTensor):
            dynamic_shapes = x.fake_mode.shape_env is not None
            break

    flat_fn, out_spec = create_tree_flattened_fn(func, args)
    flat_args, in_spec = pytree.tree_flatten(args)

    # The export use case doesn't care about several bits of AOTConfig
    # (1) compilers (we just export the graph)
    # (2) partitioners (export is only full graph, user can partition themselves)
    aot_config = AOTConfig(
        fw_compiler=None,
        bw_compiler=None,
        inference_compiler=None,
        partition_fn=None,
        decompositions=decompositions,
        num_params_buffers=num_params_buffers,
        aot_id=next(AOT_COUNTER),
        # For now there's no use case involving keeping input mutations in the graph
        # (which we can only do in the inference case anyway).
        # We can add this later if we need to.
        keep_inference_input_mutations=False,
        dynamic_shapes=dynamic_shapes,
        aot_autograd_arg_pos_to_source=None,
        is_export=True,
        no_tangents=no_tangents,
    )

    fx_g, meta = create_aot_dispatcher_function(
        flat_fn,
        flat_args,
        aot_config,
    )
    return fx_g, meta, in_spec, out_spec.spec


compiled_function = aot_function
compiled_module = aot_module
