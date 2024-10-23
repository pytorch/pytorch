# mypy: ignore-errors

import itertools
from contextlib import contextmanager, nullcontext
from functools import partial, wraps
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple
from unittest.mock import patch

import torch
import torch._dynamo.logging
import torch.nn as nn
import torch.utils._pytree as pytree
import torch.utils.dlpack
from torch import Tensor
from torch._decomp.decompositions_for_rng import PhiloxStateTracker, rng_decompositions
from torch._dispatch.python import enable_python_dispatcher
from torch._dynamo import compiled_autograd
from torch._dynamo.utils import (
    dynamo_timed,
    get_chromium_event_logger,
    preserve_rng_state,
)
from torch._guards import detect_fake_mode
from torch._inductor.utils import BoxedBool
from torch._subclasses import FakeTensor, FakeTensorMode
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from torch.utils._python_dispatch import is_traceable_wrapper_subclass


static_inputs_log = torch._logging.getArtifactLogger(
    __name__, "cudagraph_static_inputs"
)
from . import config
from ._aot_autograd.autograd_cache import (  # noqa: F401
    AOTAutogradCache,
    autograd_cache_key,
    should_use_local_autograd_cache,
    should_use_remote_autograd_cache,
)
from ._aot_autograd.collect_metadata_analysis import (  # noqa: F401
    run_functionalized_fw_and_collect_metadata,
)
from ._aot_autograd.functional_utils import (  # noqa: F401
    _check_if_mutation_can_be_in_graph,
    are_all_mutations_hidden_from_autograd,
    are_all_mutations_under_no_grad_or_inference_mode,
    assert_functional_graph,
    from_fun,
    gen_alias_from_base,
    has_data_mutation,
    has_metadata_mutation,
    is_fun,
    sync_functional_tensor,
    to_fun,
)
from ._aot_autograd.input_output_analysis import (  # noqa: F401
    _tensors_definitely_do_not_overlap,
    compute_overlapping_inputs,
    create_graph_signature,
    create_synthetic_base_metadata,
    remove_dupe_metadata,
)
from ._aot_autograd.jit_compile_runtime_wrappers import (  # noqa: F401
    aot_dispatch_autograd,
    aot_dispatch_base,
    aot_dispatch_export,
)
from ._aot_autograd.logging_utils import (  # noqa: F401
    callback_set,
    describe_input,
    format_guard_bug_msg,
    get_aot_compilation_context,
    get_aot_graph_name,
    get_graph_being_compiled,
    graph_being_compiled,
    model_name,
    nth_graph,
    set_model_name,
    setup_stacktrace_preservation_hooks,
    track_graph_compiling,
)
from ._aot_autograd.runtime_wrappers import (  # noqa: F401
    AOTDedupeWrapper,
    AOTSyntheticBaseWrapper,
)
from ._aot_autograd.schemas import (  # noqa: F401
    AOTConfig,
    BackwardSignature,
    FQN,
    GraphInputName,
    GraphOutputName,
    GraphSignature,
    InputAliasInfo,
    MutationType,
    OutputAliasInfo,
    OutputType,
    SubclassCreationMeta,
    SubclassMeta,
    TensorAlias,
    ViewAndMutationMeta,
)
from ._aot_autograd.subclass_utils import (  # noqa: F401
    create_metadata_for_subclass,
    requires_subclass_dispatch,
    unwrap_tensor_subclasses,
    unwrap_tensor_subclasses_with_indices_to_original,
    wrap_tensor_subclasses,
    wrap_tensor_subclasses_maybe_joint,
)
from ._aot_autograd.traced_function_transforms import (  # noqa: F401
    aot_dispatch_subclass,
    create_functional_call,
    create_functionalized_fn,
    create_functionalized_rng_ops_wrapper,
    create_joint,
    fn_input_mutations_to_outputs,
    fn_prepped_for_autograd,
)
from ._aot_autograd.utils import (  # noqa: F401
    _get_autocast_states,
    _get_symint_hints,
    call_func_at_runtime_with_args,
    create_tree_flattened_fn,
    KNOWN_TYPES,
    make_boxed_compiler,
    make_boxed_func,
    maybe_to_fresh_input,
    normalize_as_list,
    partial_flatten_asdict,
    root_module_when_exporting_non_strict,
    strict_zip,
)
from .partitioners import default_partition


zip = strict_zip

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

# Note [Side-Effectful Tokens in AOTAutograd]
#
# We allow some some side-effectful operators in
# the post-AOTAutograd (functional) graph, such as prints and torchbind operations.
# To ensure that these side-effects are compatible to future graph passes that
# assume that the graph is functional, we will thread "effect tokens" to show
# data dependence between these side-effectful operators. Practically speaking,
# effect tokens are just dummy values (torch.tensor([])). The graph would look
# like the following:
#
# def gm(self, token0, reader):
#    token1, frame = with_token(ordered_effect_op, (reader,), token0)
#    frame = frame * 2
#    token2, frame2 = with_token(ordered_effect_op, (reader,), token1)
#    frame2 = frame2 * 2
#    return token2, frame, frame2
#
# We will pass the token as an input to the graph, thread it through
# side-effectful operators using the `with_effects` high order operator, and then
# return the updated token as an output.
# So the signature of the graph input would look something like
# (*tokens, *params_buffers, *user_inputs), and the signature of the graph
# output would look something like (*tokens, *outputs).
#
# However, Inductor does not want the concept of tokens in the final generated
# code's input and output. Since changing the graph signature inside of inductor
# is difficult, after generating the forward graph, we will run a pass to
# remove the tokens from the inputgenerate the following graph for Inductor, where
# the tokens are created and sunk within the graph, rather than as inputs and
# outputs:
#
# def gm(self, reader):
#    token0 = torch.ops.prims._make_token()
#    token1, frame = with_token(ordered_effect_op, (reader,), token0)
#    frame = frame * 2
#    token2, frame2 = with_token(ordered_effect_op, (reader,), token1)
#    frame2 = frame2 * 2
#    sink_token = torch.ops.prims._sink_tokens([token2])
#    return frame, frame2

#
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


aot_autograd_decompositions = {}

FakifiedFlatArgs = NewType("FakifiedFlatArgs", List[Any])


def process_inputs(
    flat_args: List[Any],
    aot_config: AOTConfig,
    fake_mode: FakeTensorMode,
    shape_env: Optional[ShapeEnv],
) -> FakifiedFlatArgs:
    with fake_mode:

        def convert(idx, x):
            if shape_env is not None:
                from torch._dynamo.source import ConstantSource

                if isinstance(x, int):
                    # We always specialize on scalar values in export.
                    if aot_config.is_export:
                        return x
                    source = ConstantSource(f"sym_{idx}")
                    return shape_env.create_symintnode(
                        shape_env.create_symbol(x, source), hint=x, source=source
                    )
            if isinstance(x, torch.ScriptObject):
                return torch._library.fake_class_registry.maybe_to_fake_obj(
                    fake_mode, x
                )
            if not isinstance(x, torch.Tensor):
                return x
            if isinstance(x, FakeTensor):
                assert x.fake_mode is fake_mode
                return x
            if is_traceable_wrapper_subclass(x):
                attrs, _ = x.__tensor_flatten__()
                if all(isinstance(getattr(x, attr), FakeTensor) for attr in attrs):
                    assert all(
                        getattr(x, attr).fake_mode is fake_mode for attr in attrs
                    )
                    return x

            # see note [Tensor Fakification and Symbol Caching]
            symbolic_context = None
            source = None
            trace = True
            if tracing_context := torch._guards.TracingContext.try_get():
                if x in tracing_context.tensor_to_context:
                    symbolic_context = tracing_context.tensor_to_context[x]
                    source = symbolic_context.tensor_source
                    # We already fakeified this tensor in Dynamo, don't
                    # dump the trace for it again
                    trace = False
            if (
                idx < aot_config.num_params_buffers
                and config.static_weight_shapes
                and not symbolic_context
            ):
                # TODO: Ensure that this codepath is never exercised from
                # Dynamo
                return fake_mode.from_tensor(x, static_shapes=True)

            return fake_mode.from_tensor(
                x,
                static_shapes=False,
                symbolic_context=symbolic_context,
                source=source,
                trace=trace,
            )

        return FakifiedFlatArgs([convert(idx, x) for idx, x in enumerate(flat_args)])


def construct_fake_mode(
    flat_args: List[Any], aot_config: AOTConfig
) -> Tuple[FakeTensorMode, Optional[ShapeEnv]]:
    fake_mode = detect_fake_mode(flat_args)
    if fake_mode is None:
        shape_env = ShapeEnv() if aot_config.dynamic_shapes else None
        fake_mode = FakeTensorMode(shape_env=shape_env)
    else:
        shape_env = fake_mode.shape_env
    return (fake_mode, shape_env)


def create_aot_dispatcher_function(
    flat_fn,
    fake_flat_args: FakifiedFlatArgs,
    aot_config: AOTConfig,
    fake_mode: FakeTensorMode,
    shape_env: Optional[ShapeEnv],
) -> Tuple[Callable, ViewAndMutationMeta]:
    with dynamo_timed("create_aot_dispatcher_function"):
        return _create_aot_dispatcher_function(
            flat_fn, fake_flat_args, aot_config, fake_mode, shape_env
        )


def _create_aot_dispatcher_function(
    flat_fn,
    fake_flat_args: FakifiedFlatArgs,
    aot_config: AOTConfig,
    fake_mode: FakeTensorMode,
    shape_env: Optional[ShapeEnv],
) -> Tuple[Callable, ViewAndMutationMeta]:
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

    python_dispatcher_mode = (
        enable_python_dispatcher() if shape_env is not None else nullcontext()
    )

    # See NOTE: [Deferring tensor pack/unpack hooks until runtime]
    # If any saved tensor hooks are active, we **don't** want to trace them.
    # Instead, we'll let them run at runtime, around the custom autograd.Function
    # that we generate in torch.compile.
    with torch.autograd.set_multithreading_enabled(
        False
    ), preserve_rng_state(), (
        fake_mode
    ), (
        python_dispatcher_mode
    ), PhiloxStateTracker(), torch._dynamo.utils._disable_saved_tensors_hooks_during_tracing():
        from torch._library.fake_class_registry import (
            FakeScriptObject,
            maybe_to_fake_obj,
        )

        # Tracing may mutate the states the fake script object,
        # so we need to duplicate the fake script objects so that subsequent tracing
        # won't be affected.
        def _dup_fake_script_obj(fake_flat_args):
            return [
                maybe_to_fake_obj(detect_fake_mode(fake_flat_args), arg.real_obj)
                if isinstance(arg, FakeScriptObject)
                else arg
                for arg in fake_flat_args
            ]

        needs_autograd = any(
            x.requires_grad for x in fake_flat_args if isinstance(x, Tensor)
        )

        with enable_python_dispatcher():
            # Patch set_rng_state as set_rng_state with fake tensors is
            # nonsensical. This does not affect the collection of metadata.
            with patch("torch.cuda.set_rng_state", lambda *args: None):
                mod = root_module_when_exporting_non_strict(flat_fn)
                if mod is not None:
                    ctx = _detect_attribute_assignment(mod)
                else:
                    ctx = nullcontext()
                with ctx:
                    fw_metadata = run_functionalized_fw_and_collect_metadata(
                        flat_fn,
                        static_input_indices=aot_config.static_input_indices,
                        keep_input_mutations=aot_config.keep_inference_input_mutations,
                        is_train=needs_autograd,
                        pre_dispatch=aot_config.pre_dispatch,
                    )(*_dup_fake_script_obj(fake_flat_args))

                req_subclass_dispatch = requires_subclass_dispatch(
                    fake_flat_args, fw_metadata
                )

                get_chromium_event_logger().add_event_data(
                    "backend_compile",
                    requires_subclass_dispatch=req_subclass_dispatch,
                )

                output_and_mutation_safe = not any(
                    x.requires_grad
                    # view-type operations preserve requires_grad even in no_grad.
                    # Do not count aliases of inputs with requires_grad as reason to make a training graph,
                    # as AOTAutograd will perform view-replay to regenerate the view outputs at runtime,
                    # setting their grad_fn properly.
                    and not (
                        x.output_type
                        in (OutputType.alias_of_input, OutputType.is_input)
                        and fw_metadata.input_info[x.base_idx].requires_grad
                    )
                    for x in fw_metadata.output_info
                ) and not any(
                    x.requires_grad
                    and x.mutates_data
                    and not x.mutations_under_no_grad_or_inference_mode
                    and not x.mutations_hidden_from_autograd
                    for x in fw_metadata.input_info
                )

                if needs_autograd and output_and_mutation_safe:
                    # We realized that none of the outputs require grad,
                    # and none of the inputs that require grad are mutated.
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
                            keep_input_mutations=aot_config.keep_inference_input_mutations,
                            is_train=False,
                            pre_dispatch=aot_config.pre_dispatch,
                            static_input_indices=aot_config.static_input_indices,
                        )(*fake_flat_args)
                    else:
                        fw_metadata = ViewAndMutationMeta(
                            input_info=fw_metadata.input_info,
                            output_info=fw_metadata.output_info,
                            num_intermediate_bases=fw_metadata.num_intermediate_bases,
                            keep_input_mutations=aot_config.keep_inference_input_mutations,
                            traced_tangents=fw_metadata.traced_tangents,
                            traced_tangent_memory_formats=fw_metadata.traced_tangent_memory_formats,
                            subclass_inp_meta=fw_metadata.subclass_inp_meta,
                            subclass_fw_graph_out_meta=fw_metadata.subclass_fw_graph_out_meta,
                            subclass_tangent_meta=fw_metadata.subclass_tangent_meta,
                            is_train=False,
                            tokens=fw_metadata.tokens,
                            static_input_indices=fw_metadata.static_input_indices,
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
                raise RuntimeError(
                    f"""\
Found an input that received a metadata mutation, through e.g. a call to `.resize_()` or `.transpose_()`.
This is currently banned in the aot_export workflow. If you need this functionality, please file a github issue.

fw_metadata={str(fw_metadata)}"""
                )
            # In export, banning data mutations on inputs that require grad for now.
            # This should be rare, and is tricky to get right. When we trace the backward,
            # we currently trace with autograd.grad instead of .backward(), which makes it difficult
            # to ensure that we run autograd all the way through the input **before** it saw the mutation.
            if (
                len(
                    [
                        x
                        for x in fw_metadata.input_info
                        if x.requires_grad and x.mutates_data
                    ]
                )
                != 0
            ):
                raise RuntimeError(
                    f"""\
Found a graph input that requires gradients, and received a mutation.
This is currently banned in the aot_export workflow. If you need this functionality, please file a github issue.

fw_metadata={str(fw_metadata)}"""
                )
            if req_subclass_dispatch:
                raise RuntimeError(
                    """\
aot_export is not currently supported with traceable tensor subclass.
If you need this feature, please comment on <CREATE_ISSUE_LINK>"""
                )

            # Need to decide on a strategy for functionalized RNG: toggling via global config seems bad,
            # and turning it on will require a non-trivial calling convention change for any export runtime.
            if config.functionalize_rng_ops:
                raise RuntimeError(
                    """\
Functionalized RNG is not currently supported in the aot_export workflow. Please file a github issue,
or otherwise set torch._functorch.config.functionalize_rng_ops = False."""
                )

        def choose_dispatcher(needs_autograd, aot_config):
            """
            Pick a dispatcher based on the config rules.
            """
            if aot_config.is_export:
                # export uses just the "graph bits", whereas the other
                # two dispatchers include some extra work around handling a runtime epilogue
                get_chromium_event_logger().add_event_data(
                    "create_aot_dispatcher_function", dispatch_mode="export"
                )
                return partial(aot_dispatch_export, needs_autograd=needs_autograd)
            elif needs_autograd and not aot_config.pre_dispatch:
                get_chromium_event_logger().add_event_data(
                    "create_aot_dispatcher_function", dispatch_mode="autograd"
                )
                return aot_dispatch_autograd
            else:
                get_chromium_event_logger().add_event_data(
                    "create_aot_dispatcher_function", dispatch_mode="inference"
                )
                return aot_dispatch_base

        compiler_fn = choose_dispatcher(needs_autograd, aot_config)

        compiled_fn, fw_metadata = compiler_fn(
            flat_fn,
            _dup_fake_script_obj(fake_flat_args),
            aot_config,
            fw_metadata=fw_metadata,
        )
        return compiled_fn, fw_metadata


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
            (fake_mode, shape_env) = construct_fake_mode(flat_args, aot_config)
            fake_flat_args: FakifiedFlatArgs = process_inputs(
                flat_args, aot_config, fake_mode, shape_env
            )
            compiled_fn, _ = create_aot_dispatcher_function(
                flat_fn,
                fake_flat_args,
                aot_config,
                fake_mode,
                shape_env,
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
        def __init__(self) -> None:
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
    cudagraphs: Optional[BoxedBool] = None,
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

    if cudagraphs is None:
        cudagraphs = BoxedBool(torch._inductor.config.triton.cudagraphs)

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
        (
            tracing_context.params_flat_unwrap_subclasses,
            tracing_context.params_unwrapped_to_flat_index,
        ) = unwrap_tensor_subclasses_with_indices_to_original(params_flat)

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

    static_input_indices = []
    if hasattr(mod, "graph"):
        # Non dynamo entrypoints can get to here...
        for pos, node in enumerate(mod.graph.find_nodes(op="placeholder")):
            if hasattr(node, "_dynamo_source"):
                # ... but not here!
                if aot_autograd_arg_pos_to_source is None:
                    aot_autograd_arg_pos_to_source = []
                source = node._dynamo_source
                assert source not in seen_sources, source
                seen_sources.add(source)
                aot_autograd_arg_pos_to_source.append(source)
                source_name = source.name() if source else str(source)

                if "tensor_dict" in node.meta and node.meta["tensor_dict"].get(
                    "_dynamo_static_input_type", None
                ):
                    static_inputs_log.debug(
                        "Adding static input pos %s for source %s", pos, source_name
                    )
                    static_input_indices.append(pos)
                else:
                    static_inputs_log.debug(
                        "Non-static input pos %s for source %s", pos, source_name
                    )

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
        static_input_indices=static_input_indices,
        is_export=False,
        no_tangents=False,
        cache_info=None,
    )
    fake_mode, shape_env = construct_fake_mode(full_args, aot_config)
    fake_flat_args = process_inputs(full_args, aot_config, fake_mode, shape_env)

    def dispatch_and_compile():
        functional_call = create_functional_call(mod, params_spec, params_len)
        with compiled_autograd.disable():
            compiled_fn, _ = create_aot_dispatcher_function(
                functional_call,
                fake_flat_args,
                aot_config,
                fake_mode,
                shape_env,
            )
        return compiled_fn

    # Autograd cache stuff
    remote = should_use_remote_autograd_cache()
    local = should_use_local_autograd_cache()

    if local or remote:
        compiled_fn = AOTAutogradCache.load(
            dispatch_and_compile,
            mod,
            fake_flat_args,
            aot_config,
            cudagraphs,
            local,
            remote,
        )
    else:
        compiled_fn = dispatch_and_compile()

    if isinstance(mod, torch._dynamo.utils.GmWrapper):
        # This function is called by the flatten_graph_inputs wrapper, which boxes
        # the inputs so that they can be freed before the end of this scope.
        # For overhead reasons, this is not the default wrapper, see comment:
        # https://github.com/pytorch/pytorch/pull/122535/files#r1560096481
        def boxed_forward(runtime_args: List[Any]):
            flat_args = []
            flat_args.extend(params_flat)
            flat_args.extend(runtime_args)
            runtime_args.clear()
            return compiled_fn(flat_args)

        # Just for convenience
        boxed_forward.zero_grad = mod.zero_grad
        boxed_forward.named_parameters = mod.named_parameters
        boxed_forward.named_buffers = mod.named_buffers
        return boxed_forward

    # TODO: There is something deeply wrong here; compiled_fn running with
    # the boxed calling convention, but aot_module_simplified somehow
    # historically returned a function that was not the boxed calling
    # convention.  This should get fixed...
    # NB: GraphModule/nn.Module rely on the non-boxed calling convention here
    def forward(*runtime_args: Tuple[Any]):
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
    pre_dispatch: bool = False,
    # If None, will be infered from inputs and mod.graph.nodes if mod is a graph module, but the inferred result might be wrong.
    dynamic_shapes: Optional[bool] = None,
    kwargs=None,
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
    if pre_dispatch and trace_joint:
        raise RuntimeError("pre_dispatch is not supported when trace_joint is True.")
    named_parameters = dict(mod.named_parameters(remove_duplicate=False))
    named_buffers = dict(mod.named_buffers(remove_duplicate=False))

    params_and_buffers = {
        **dict(named_parameters),
        **dict(named_buffers),
    }
    params_and_buffers_flat, params_spec = pytree.tree_flatten(params_and_buffers)
    params_and_buffers_flat = tuple(params_and_buffers_flat)
    params_len = len(params_and_buffers_flat)

    kwargs = kwargs or {}

    functional_call = create_functional_call(
        mod, params_spec, params_len, store_orig_mod=True
    )

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
                raise RuntimeError(
                    """\
If trace_joint=Trueit is required that one of your forward outputs must be a scalar loss.
You must specify the which (index) output is the loss with output_loss_index."""
                )
            if isinstance(out, (torch.Tensor)):
                out = (out,)
            if not isinstance(out, (tuple, list)):
                raise RuntimeError(
                    f"Expected forward output to be either a tensor or a list/tuple of tensors. found {type(out)}"
                )

            for i, o in enumerate(out):
                # We only want to create a backward graph w.r.t. the loss that the user passed in.
                # This implies that every other output should not require gradients.
                # Instead of making this an error (and forcing the user to detach all other outputs
                # of their forward),
                # we'll automatically detach them here.
                if o.requires_grad and i != output_loss_index:
                    raise RuntimeError(
                        f"""\
Found an output of the forward that requires gradients, that was not the scalar loss.
We require all outputs to the forward that are not the scalar loss to not require gradient,
because we will only compute a backward graph against the scalar loss.
You can fix this by calling .detach() on each of your forward outputs that is not the loss.
You specified that output index {output_loss_index} is the loss, but we found that
the output at index {i} requires gradients."""
                    )
            out_loss = out[output_loss_index]
            num_fw_outs = len(out)
            if not out_loss.requires_grad:
                raise RuntimeError(
                    f"""\
The output at index {output_loss_index} was marked as the loss, but it does not require gradients"""
                )
            if out_loss.numel() != 1:
                raise RuntimeError(
                    f"""\
We require the output marked as the loss (at index {output_loss_index}) to be a scalar, but it has shape {out_loss.shape}"""
                )
            return out

        ctx = nullcontext
    else:
        # Run under no_grad, so our tracing machinery only traces an inference graph.
        # However if pre_dispatch=True, we want to correctly trace set_grad_enabled calls for training.
        ctx = nullcontext if pre_dispatch else torch.no_grad
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
            pre_dispatch=pre_dispatch,
            dynamic_shapes=dynamic_shapes,
            kwargs=kwargs,
        )
    if trace_joint:

        @wraps(functional_call)
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
            fake_tangents = [
                None
                for _ in range(
                    metadata.num_outputs + metadata.num_mutated_inp_runtime_indices
                )
            ]
            fw_outs, gradients = fx_g(args, fake_tangents)
            assert len(gradients) == len(args)
            output_gradients = []
            for i, (a, grad) in enumerate(zip(args, gradients)):
                if isinstance(a, torch.Tensor) and a.requires_grad:
                    assert (
                        grad is not None
                    ), """\
Found a parameter that did not receive a gradient.
"This is most likely a bug, but if this needs to be supported please comment on this Github issue:
https://github.com/pytorch/pytorch/issues/101192
"""
                    output_gradients.append(grad)
                else:
                    assert grad is None
            return *fw_outs, *output_gradients

        fx_g = make_fx(flattened_joint, record_module_stack=True)(*full_args)

    user_args_flat = pytree.arg_tree_leaves(*args, **kwargs)
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
        in_spec, _kw_in_spec = in_spec.children_specs
    # At this point, we can just directly return the (joint or inference graph) that we traced.
    # First though: a bunch of assertions to make sure that our graph doesn't require
    # any calling convention changes compared to the original function.
    # These restrictions are *in addition to* the general restrictions on export.

    # No input mutations
    if (
        len([x for x in metadata.input_info if x.mutates_data or x.mutates_metadata])
        != 0
    ):
        raise RuntimeError(
            f"aot_export_joint_simple does not support input mutations. {str(metadata)}"
        )
    # No output aliasing
    if (
        len([x for x in metadata.output_info if x.output_type != OutputType.non_alias])
        != 0
    ):
        raise RuntimeError(
            f"aot_export_joint_simple does not support outputs that alias inputs. {str(metadata)}"
        )
    # No pytrees
    if in_spec.is_leaf():
        raise RuntimeError(
            f"aot_export_joint_simple requires inputs to be a single list/tuple. in_spec={str(in_spec)}"
        )
    if not all(child.is_leaf() for child in in_spec.children_specs):
        raise RuntimeError(
            f"aot_export_joint_simple requires individual inputs not to be pytrees. in_spec={str(in_spec)}"
        )
    if out_spec.is_leaf():
        raise RuntimeError(
            f"aot_export_joint_simple requires outputs to be a single list/tuple. out_spec={str(out_spec)}"
        )
    if not all(child.is_leaf() for child in out_spec.children_specs):
        raise RuntimeError(
            f"aot_export_joint_simple requires individual outputs not to be pytrees. out_spec={str(out_spec)}"
        )
    # TODO: we might have to temporarily patch config.functionalize_rng
    # so that it doesn't run when we're exporting a higher order op.

    if config.debug_assert:
        # Smoke test that after partitioning, we can run the forward without any calling convention changes.
        fw_module, bw_module = aot_config.default_partition(  # noqa: F821
            fx_g, args, num_fwd_outputs=len(fw_metadata.output_infos)  # noqa: F821
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
    pre_dispatch: bool = False,
    # If None, `dynamic_shapes` will be infered from inputs, but the inferred result might be wrong.
    dynamic_shapes: Optional[bool] = None,
    kwargs=None,
) -> Tuple[torch.fx.GraphModule, ViewAndMutationMeta, pytree.TreeSpec, pytree.TreeSpec]:
    kwargs = kwargs or {}

    flat_fn, out_spec = create_tree_flattened_fn(func, args, kwargs)
    flat_args, in_spec = pytree.tree_flatten((args, kwargs))

    if dynamic_shapes is None:
        # Try to infer `dynamic_shapes from inputs and graph nodes
        fake_mode = detect_fake_mode(flat_args)
        if (
            fake_mode is None
            and hasattr(func, "_orig_mod")
            and isinstance(func._orig_mod, torch.fx.GraphModule)
        ):
            vals = [
                node.meta["val"]
                for node in func._orig_mod.graph.nodes
                if "val" in node.meta
            ]
            fake_mode = detect_fake_mode(vals)
        dynamic_shapes = fake_mode is not None and fake_mode.shape_env is not None

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
        pre_dispatch=pre_dispatch,
    )
    fake_mode, shape_env = construct_fake_mode(flat_args, aot_config)
    fake_flat_args = process_inputs(flat_args, aot_config, fake_mode, shape_env)

    fx_g, meta = create_aot_dispatcher_function(
        flat_fn,
        fake_flat_args,
        aot_config,
        fake_mode,
        shape_env,
    )
    return fx_g, meta, in_spec, out_spec.spec


@contextmanager
def _detect_attribute_assignment(mod: torch.nn.Module):
    # Do not allow assignment of tensor attributes during export unless
    # the attribute is registered as a buffer.

    NN_MODULE_STD_ATTRS = [
        "_backward_hooks",
        "_backward_pre_hooks",
        "_buffers",
        "_forward_hooks",
        "_forward_hooks_always_called",
        "_forward_hooks_with_kwargs",
        "_forward_pre_hooks",
        "_forward_pre_hooks_with_kwargs",
        "_is_full_backward_hook",
        "_load_state_dict_post_hooks",
        "_load_state_dict_pre_hooks",
        "_modules",
        "_non_persistent_buffers_set",
        "_parameters",
        "_state_dict_hooks",
        "_state_dict_pre_hooks",
        "training",
    ]
    NN_MODULE_LAZY_STD_ATTRS = [
        "_initialize_hook",
        "_load_hook",
    ]
    STD_ATTRS = {
        *NN_MODULE_STD_ATTRS,
        *NN_MODULE_LAZY_STD_ATTRS,
    }

    def _get_attributes(mod):
        # return any attributes of a module that are not standard attributes
        return {k: v for k, v in mod.__dict__.items() if k not in STD_ATTRS}

    def is_leaf(x):
        # Ideally is_leaf should not be needed when mapping, but it seems that
        # subclasses of a standard container X may sometimes map to X, which
        # destroys information and can cause future mapping to fail.
        known_subclasses_that_lose_info = (
            torch.Size,
            # add more here if needed
        )
        return isinstance(x, known_subclasses_that_lose_info)

    # save state of attributes before enter
    snapshot = pytree.tree_map(lambda x: x, _get_attributes(mod), is_leaf=is_leaf)
    try:
        yield
    finally:
        # after exit, compare state of attributes with snapshot
        # to detect which tensor attributes were assigned
        assigned_tensor_attributes = []

        def _collect_assigned_tensor_attributes(kp, v, _v):
            if _v is not v:
                attr, *rest = kp
                if isinstance(v, torch.Tensor):
                    assigned_tensor_attributes.append(
                        f"self.{attr.key}{pytree.keystr(rest)}"
                    )
                # TODO(avik): Assigning all other types are allowed right now.
                # Maybe in the future we want to limit this to primitive types?

        pytree.tree_map_with_path(
            _collect_assigned_tensor_attributes, snapshot, _get_attributes(mod)
        )
        # restore state of all attributes (including, e.g., of primitive types)
        mod.__dict__.update(snapshot)

        if assigned_tensor_attributes:
            if len(assigned_tensor_attributes) > 1:
                noun, verb = "attributes", "were"
            else:
                noun, verb = "attribute", "was"
            raise ValueError(
                f"The tensor {noun} {', '.join(assigned_tensor_attributes)} {verb} assigned during export. "
                "Such attributes must be registered as buffers using the `register_buffer` API "
                "(https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_buffer)."
            )


compiled_function = aot_function
compiled_module = aot_module
