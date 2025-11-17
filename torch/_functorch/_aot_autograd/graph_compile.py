# mypy: allow-untyped-defs
"""
Functions in this module do most of the "work" of AOTAutograd.
An aot_dispatch_* function:
- Takes in the input flat_fn, flat_args, and some metadata
- Runs a set of pre compile wrappers (e.g. argument deduping)
- Runs the actual compiler
- Wraps the returned callable in a set of post compile wrappers
- Returns the wrapped callable and metadata.
"""

import copy
import dataclasses
import itertools
import logging
import operator
import time
import traceback
from collections import defaultdict
from collections.abc import Callable
from contextlib import nullcontext
from typing import Any, Optional, TYPE_CHECKING, Union


if TYPE_CHECKING:
    from collections.abc import Sequence

import torch
import torch.utils._pytree as pytree
import torch.utils.dlpack
from torch import Tensor
from torch._dynamo.utils import (
    CompileEventLogger,
    detect_fake_mode,
    dynamo_timed,
    lazy_format_graph_code,
)
from torch._guards import CompileContext, TracingContext
from torch._logging import getArtifactLogger, trace_structured
from torch._subclasses import FakeTensor
from torch._subclasses.meta_utils import is_sparse_any
from torch.fx.experimental._backward_state import BackwardState
from torch.fx.experimental.proxy_tensor import is_sym_node
from torch.fx.experimental.symbolic_shapes import fx_placeholder_vals
from torch.fx.graph_module import GraphModule
from torch.fx.passes._tensorify_python_scalars import tensorify_python_scalars
from torch.multiprocessing.reductions import StorageWeakRef
from torch.types import py_sym_types
from torch.utils._python_dispatch import is_traceable_wrapper_subclass
from torchgen.utils import dataclass_repr

from .. import config
from .autograd_cache import (
    AOTAutogradCache,
    GenericAOTAutogradCacheEntry,
    serialize_graph_module,
    should_bundle_autograd_cache,
    should_use_remote_autograd_cache,
)
from .descriptors import AOTOutput, PlainAOTOutput
from .graph_capture import aot_dispatch_autograd_graph, aot_dispatch_base_graph
from .logging_utils import track_graph_compiling
from .runtime_wrappers import (
    AOTDedupeWrapper,
    AOTDispatchAutograd,
    AOTDispatchSubclassWrapper,
    AOTSyntheticBaseWrapper,
    AutogradLazyBackwardCompileInfo,
    CompilerWrapper,
    DebugAssertWrapper,
    EffectTokensWrapper,
    FakifiedOutWrapper,
    FunctionalizedRngRuntimeWrapper,
    make_runtime_safe,
    post_compile,
    pre_compile,
    RuntimeWrapper,
    SerializableCompiledFunction,
)
from .schemas import (
    AOTConfig,
    AOTGraphCapture,
    AOTState,
    FlatFn,
    FxValue,
    MutationType,
    ViewAndMutationMeta,
)
from .subclass_utils import compute_inner_mutated_inp_indices_from_subclass_meta
from .utils import (
    _get_symint_hints,
    contain_metadata_mutation_ops,
    get_cuda_generator_meta_val,
    make_boxed_func,
    simple_wraps,
    strict_zip,
    unlift_tokens,
)


zip = strict_zip

log = logging.getLogger(__name__)
aot_joint_log = getArtifactLogger(__name__, "aot_joint_graph")
aot_graphs_log = getArtifactLogger(__name__, "aot_graphs")

aten = torch.ops.aten

# Returns a Callable and a ViewAndMutationMeta.
# Currently, only export needs the ViewAndMutationMeta after this function.
# TODO: Refactor this
DispatchReturn = tuple[Callable, ViewAndMutationMeta]


def _create_wrappers_for_dispatch(needs_autograd: bool) -> list[CompilerWrapper]:
    """
    Wrappers that run on every dispatch function
    """
    return [AOTDedupeWrapper(), AOTSyntheticBaseWrapper(trace_joint=needs_autograd)]


def aot_stage1_graph_capture(
    aot_state: AOTState,
    orig_flat_fn: FlatFn,
) -> AOTGraphCapture:
    # NB: flat_fn at this point coincides with the initial info from forward
    # metadata collection returning a list[Tensor].  We are now going to
    # augment the output to return a tuple[list[Tensor], list[AOTOutput]] and
    # then preserve this convention through the rest of the passes.

    # TODO: We could test for consistency with fw_metadata, but this is not a
    # big deal
    @simple_wraps(orig_flat_fn)
    def orig_flat_fn2(*args: FxValue) -> tuple[list[FxValue], list[AOTOutput]]:
        out = orig_flat_fn(*args)
        out_descs: list[AOTOutput] = type(out)(  # type: ignore[assignment]
            PlainAOTOutput(i)  # type: ignore[misc]
            for i in range(len(out))  # type: ignore[misc]
        )
        return out, out_descs

    aot_config = aot_state.aot_config

    wrappers = _create_wrappers_for_dispatch(aot_state.needs_autograd)
    flat_fn, aot_state.flat_args, aot_state.flat_args_descs, aot_state.fw_metadata = (
        pre_compile(
            wrappers,
            orig_flat_fn2,
            aot_state.flat_args,
            aot_state.flat_args_descs,
            aot_config,
            fw_metadata=aot_state.fw_metadata,
        )
    )

    # NB: This is currently only used for backwards, where fwd/bwd
    # deterministic TLS can be different
    aot_state.fw_metadata.deterministic = torch.are_deterministic_algorithms_enabled()
    updated_flat_args: Union[list[Any], tuple[list[Any], list[Any]]]
    if aot_state.needs_autograd and not aot_config.pre_dispatch:
        # FYI: this being moved to trigger in export is new, seems fine!
        with dynamo_timed("aot_trace_joint_graph", log_pt2_compile_event=True):
            graph, updated_flat_args, updated_flat_args_descs, maybe_subclass_meta = (
                aot_dispatch_autograd_graph(
                    flat_fn,
                    aot_state.flat_args,
                    aot_state.flat_args_descs,
                    aot_config,
                    fw_metadata=aot_state.fw_metadata,
                )
            )
    else:
        graph, updated_flat_args, updated_flat_args_descs, maybe_subclass_meta = (
            aot_dispatch_base_graph(  # type: ignore[assignment]
                flat_fn,
                aot_state.flat_args,
                aot_state.flat_args_descs,
                aot_config,
                fw_metadata=aot_state.fw_metadata,
            )
        )

    return AOTGraphCapture(
        wrappers=wrappers,
        graph_module=graph,
        updated_flat_args=updated_flat_args,
        updated_flat_args_descs=updated_flat_args_descs,
        maybe_subclass_meta=maybe_subclass_meta,
    )


def aot_stage2_export(
    aot_state: AOTState, aot_graph_capture: AOTGraphCapture
) -> DispatchReturn:
    graph = aot_graph_capture.graph_module
    aot_config = aot_state.aot_config
    wrappers = aot_graph_capture.wrappers

    CompileEventLogger.try_add_pt2_compile("backend_compile", dispatch_mode="export")

    # NB: the wrappers that run in pre_compile for export are
    # either a no-op, because they're not needed, or will raise a runtime error,
    # since they don't support export.
    # We still run these wrappers to make sure that they're not needed pre compile,
    # but we technically don't need to run them post compile at all here.
    compiled_fn, aot_state.fw_metadata = post_compile(
        wrappers, graph, aot_config, runtime_metadata=aot_state.fw_metadata
    )

    # Therefore, since no wrapperes run, we don't get back a callable - we get back the raw fx graph
    # (either a joint or an inference-only graph)
    assert isinstance(compiled_fn, torch.fx.GraphModule)
    return compiled_fn, aot_state.fw_metadata


def sanitize_aot_config(input: AOTConfig) -> AOTConfig:
    return AOTConfig(
        fw_compiler=None,  # type: ignore[arg-type]
        bw_compiler=None,  # type: ignore[arg-type]
        partition_fn=None,  # type: ignore[arg-type]
        decompositions={},
        inference_compiler=None,
        num_params_buffers=input.num_params_buffers,
        aot_id=input.aot_id,
        keep_inference_input_mutations=input.keep_inference_input_mutations,
        is_export=input.is_export,
        no_tangents=input.no_tangents,
        aot_autograd_arg_pos_to_source=input.aot_autograd_arg_pos_to_source,
        dynamic_shapes=input.dynamic_shapes,
        enable_log=input.enable_log,
        static_input_indices=input.static_input_indices,
        pre_dispatch=input.pre_dispatch,
        cache_info=None,
        precompile_backend_id=input.precompile_backend_id,
    )


def aot_stage2_compile(
    aot_state: AOTState,
    aot_graph_capture: AOTGraphCapture,
    partition_fn: Callable,
    fw_compiler: Callable,
    bw_compiler: Optional[Callable] = None,
    inference_compiler: Optional[Callable] = None,
) -> DispatchReturn:
    if bw_compiler is None:
        bw_compiler = fw_compiler
    if inference_compiler is None:
        inference_compiler = fw_compiler
    # Update the AOTState with the provided compilers
    aot_state.aot_config.partition_fn = partition_fn
    aot_state.aot_config.fw_compiler = fw_compiler
    aot_state.aot_config.bw_compiler = bw_compiler
    aot_state.aot_config.inference_compiler = inference_compiler

    if aot_state.needs_autograd and not aot_state.aot_config.pre_dispatch:
        return aot_stage2_autograd(aot_state, aot_graph_capture)
    else:
        return aot_stage2_inference(aot_state, aot_graph_capture)


def aot_stage2_inference(
    aot_state: AOTState,
    aot_graph_capture: AOTGraphCapture,
) -> DispatchReturn:
    """
    Handles functions that don't need autograd. Runs wrappers and compiles with fw_compiler.
    """

    aot_config = aot_state.aot_config
    fw_metadata = aot_state.fw_metadata
    fw_module = aot_graph_capture.graph_module
    wrappers = aot_graph_capture.wrappers
    updated_flat_args = aot_graph_capture.updated_flat_args
    maybe_subclass_meta = aot_graph_capture.maybe_subclass_meta

    CompileEventLogger.try_add_pt2_compile("backend_compile", dispatch_mode="inference")

    # Save the forward_graph_str right after aot_dispatch_base_graph,
    # to save in the cache
    aot_forward_graph_str = None
    if aot_config.cache_info is not None:
        aot_forward_graph_str = fw_module.print_readable(
            print_output=False,
            include_stride=True,
            include_device=True,
            fast_sympy_print=True,
            expanded_def=True,
        )

    fakified_out_wrapper = FakifiedOutWrapper()
    fakified_out_wrapper.pre_compile(
        fw_module, updated_flat_args, aot_config, fw_metadata=fw_metadata
    )
    functionalized_rng_wrapper = FunctionalizedRngRuntimeWrapper()
    functionalized_rng_wrapper.pre_compile(
        fw_module, updated_flat_args, aot_config, fw_metadata=fw_metadata
    )
    assert isinstance(fw_module, GraphModule)

    if aot_config.enable_log:
        trace_structured(
            "artifact",
            metadata_fn=lambda: {
                "name": "torch._functorch.config",
                "encoding": "string",
            },
            payload_fn=lambda: torch._functorch.config.get_config_copy(),
        )

    disable_amp = torch._C._is_any_autocast_enabled()
    context = torch._C._DisableAutocast if disable_amp else nullcontext

    with context(), track_graph_compiling(aot_config, "inference"):
        compiler = (
            aot_config.inference_compiler
            if aot_config.inference_compiler is not None
            else aot_config.fw_compiler
        )

        if tracing_context := torch._guards.TracingContext.try_get():
            tracing_context.fw_metadata = (
                fw_metadata
                if maybe_subclass_meta is None
                else maybe_subclass_meta.fw_metadata
            )

        with TracingContext.report_output_strides() as fwd_output_strides:
            fake_mode = detect_fake_mode()
            if fake_mode is not None and fake_mode.shape_env is not None:
                tensorify_python_scalars(fw_module, fake_mode.shape_env, fake_mode)
            compiled_fw = compiler(fw_module, updated_flat_args)

        if fakified_out_wrapper.needs_post_compile:
            fakified_out_wrapper.set_fwd_output_strides(fwd_output_strides)

    make_runtime_safe(fw_metadata, maybe_subclass_meta)

    # However, RuntimeWrapper does not expect the rng offsets in the
    # output. So, we have to create another wrapper and take out the offset. As
    # a result, we have to account for not boxed_call compilers as well.
    if not getattr(compiled_fw, "_boxed_call", False):
        compiled_fw = make_boxed_func(compiled_fw)

    # Create a wrapper to set up the rng functionalize and fakified out bits
    compiled_fw = functionalized_rng_wrapper.post_compile(
        compiled_fw, aot_config, runtime_metadata=fw_metadata
    )
    cache_info = aot_config.cache_info

    def should_save_cache():
        if should_bundle_autograd_cache():
            return True
        else:
            return hasattr(compiled_fw, "_fx_graph_cache_key")

    if cache_info is not None:
        if should_save_cache():
            time_taken_ns = time.time_ns() - cache_info.start_time_ns
            guards_expr = AOTAutogradCache.generate_guards_expression(cache_info)
            entry = AOTAutogradCache.make_entry(
                compiled_fw_func=compiled_fw,  # type: ignore[arg-type]
                compiled_bw_func=None,
                aot_joint_graph_str=None,
                aot_forward_graph_str=aot_forward_graph_str,
                aot_backward_graph_str=None,
                runtime_metadata=fw_metadata,
                dispatch_wrappers=wrappers,
                maybe_subclass_meta=maybe_subclass_meta,
                num_fw_outs_saved_for_bw=None,
                indices_of_inps_to_detach=[],
                forward_time_taken_ns=time_taken_ns,
                backward_time_taken_ns=0,
                sanitized_aot_config=sanitize_aot_config(aot_config),
                guards_expr=guards_expr,
                backward_state_indices=None,
                num_symints_saved_for_bw=None,
                serialized_bw_module=None,
            )
            AOTAutogradCache.save(
                cache_info.cache_key, entry, remote=should_use_remote_autograd_cache()
            )
            compiled_fw = SerializableCompiledFunction(compiled_fw, lambda: entry)

    compiled_fw = fakified_out_wrapper.post_compile(
        compiled_fw,
        aot_config,
        runtime_metadata=fw_metadata,
    )

    compiled_fw = EffectTokensWrapper().post_compile(
        compiled_fw,
        aot_config,
        runtime_metadata=fw_metadata,
    )

    # Why do we need to pass in num_fw_outs_saved_for_bw?
    # See Note: [Partitioner handling for Subclasses, Part 2]
    compiled_fw = AOTDispatchSubclassWrapper(
        trace_joint=False,
        # TODO: once we use pre_compile this will be flat_fn at the top of this function
        fw_only=None,
        maybe_subclass_meta=maybe_subclass_meta,
        num_fw_outs_saved_for_bw=None,
    ).post_compile(
        compiled_fw,
        aot_config,  # not used
        runtime_metadata=fw_metadata,
    )

    if not getattr(compiled_fw, "_boxed_call", False):
        compiled_fw = make_boxed_func(compiled_fw)

    compiled_fn = RuntimeWrapper(
        indices_of_inps_to_detach=[],
        trace_joint=False,
        disable_amp=disable_amp,
    ).post_compile(
        compiled_fw,
        aot_config,
        runtime_metadata=fw_metadata,
    )

    compiled_fn = post_compile(
        wrappers, compiled_fn, aot_config, runtime_metadata=fw_metadata
    )
    return compiled_fn


def collect_fw_donated_buffer_idxs(
    fw_ins: list[Optional[FakeTensor]],
    user_fw_outs: list[Optional[FakeTensor]],
    bw_outs: list[Optional[FakeTensor]],
    saved_tensors: list[FakeTensor],
) -> list[int]:
    """
    Checks if the saved tensors are donated buffers, which means a saved tensor is not
    an alias of any tensors in fw_ins, user_fw_outs, and bw_outs.
    """

    storage_refs = set()

    for t in itertools.chain(fw_ins, user_fw_outs, bw_outs):
        # Only access storage if a tensor has storage (not sparse)
        if t is not None and isinstance(t, FakeTensor) and not is_sparse_any(t):
            storage_refs.add(StorageWeakRef(t.untyped_storage()))

    num_saved_tensor = len(saved_tensors)
    donated_buffer_idxs = []
    for i in range(num_saved_tensor):
        t = saved_tensors[i]
        if (
            t is not None
            and not is_sparse_any(t)
            and StorageWeakRef(t.untyped_storage()) not in storage_refs
        ):
            donated_buffer_idxs.append(i)

    return donated_buffer_idxs


def collect_bw_donated_buffer_idxs(
    fw_module: torch.fx.GraphModule,
    bw_module: torch.fx.GraphModule,
    fw_metadata: ViewAndMutationMeta,
) -> list[int]:
    """
    Collects backward donated buffer indexes from fw_module and bw_module.
    """

    # [Note: Metadata mutation in proxy tracing]
    # node.meta["val"] is a snapshot of the tensor value when tracing a graph,
    # instead of the final state after the graph has run. node.meta["val"] is
    # not updated even if later there is a metadata mutation op.
    # See: https://github.com/pytorch/pytorch/pull/141308#issuecomment-2495798947
    #
    # Currently, metadata mutation op happens only for sacrificial parameter
    # specifically the `set_` op. This motivates banning metadata mutation from
    # proxy tracing.
    #
    # Since node.meta["val"] is used to detect donated buffer, we return an empty
    # list if there exists metadata mutation op.
    if contain_metadata_mutation_ops(fw_module) or contain_metadata_mutation_ops(
        bw_module
    ):
        return []

    fw_ins = fw_module.graph.find_nodes(op="placeholder")
    bw_outs = next(reversed(bw_module.graph.find_nodes(op="output"))).args[0]
    fw_outs = next(reversed(fw_module.graph.find_nodes(op="output"))).args[0]

    fw_ins = [
        n.meta["val"] if (hasattr(n, "meta") and "val" in n.meta) else None
        for n in fw_ins
    ]
    fw_outs = [
        n.meta["val"] if (hasattr(n, "meta") and "val" in n.meta) else None
        for n in fw_outs
    ]
    bw_outs = [
        n.meta["val"] if (hasattr(n, "meta") and "val" in n.meta) else None
        for n in bw_outs
    ]

    user_fw_outs = fw_outs[: fw_metadata.num_forward]
    saved_tensors = fw_outs[fw_metadata.tensors_saved_for_backwards_slice]

    fw_donated_buffer = collect_fw_donated_buffer_idxs(
        fw_ins,
        user_fw_outs,
        bw_outs,
        # pyrefly: ignore  # bad-argument-type
        saved_tensors,
    )

    assert fw_metadata.num_symints_saved_for_bw is not None
    return [fw_metadata.num_symints_saved_for_bw + i for i in fw_donated_buffer]


@dataclasses.dataclass
class InvokeSubgraphHopGraphs:
    """
    A data structure to hold all the information needed to partition the
    `joint_hop_gm` and joint graph and the restitch the `new_fw_hop_gm` and
    `new_bw_hop_gm` into the bigger `joint_gm`.
    """

    # To avoid re-partitioning subgraphs
    partitioning_done: bool = False
    old_num_fw_outputs: Optional[int] = None
    old_num_fw_inputs: Optional[int] = None

    new_fw_hop_gm: Optional[torch.fx.GraphModule] = None
    new_bw_hop_gm: Optional[torch.fx.GraphModule] = None
    new_num_sym_nodes: Optional[int] = None
    new_num_saved_nodes: Optional[int] = None


def prepare_for_partitioner(mod, num_primals, num_fw_outputs):
    # min-cut partitioner requires the placeholders to have primals and
    # tangents string in the node.name. The signature of the joint graph is
    # (*primals, *tangents)

    # We also have to update the output signature which is right now
    # (*grads, *fw_outs) and we have to change to (*fw_outs, *grads) for the
    # partitioner to work.
    new_graph = torch.fx.Graph()
    env = {}

    primals_counter = itertools.count(0)
    tangents_counter = itertools.count(0)

    for idx, node in enumerate(mod.graph.nodes):
        if node.op == "placeholder":
            if idx < num_primals:
                env[node] = new_graph.placeholder(f"primals_{next(primals_counter)}")
            else:
                env[node] = new_graph.placeholder(f"tangents_{next(tangents_counter)}")
            env[node].meta = copy.copy(node.meta)
        elif node.op == "output":
            # Reverse the (*grads, *fw_outs) to (*fw_outs, *grads)
            # The reason for having the reversed signature in the first
            # place is to simplify step 3.
            old_outputs = node.args[0]
            new_outputs = (
                *old_outputs[-num_fw_outputs:],
                *old_outputs[:-num_fw_outputs],
            )
            new_outputs = [env[n] if n else None for n in new_outputs]
            new_graph.output(tuple(new_outputs))
        else:
            env[node] = new_graph.node_copy(node, lambda n: env[n])
            env[node].meta = copy.copy(node.meta)

    new_graph.lint()

    out = torch.fx.GraphModule(mod, new_graph)
    return out


def run_joint_graph_passes_on_hops(
    joint_gm: torch.fx.GraphModule,
    joint_inputs: Any,
    aot_config: AOTConfig,
) -> torch.fx.GraphModule:
    """
    This pass runs the joint graph passes on the HOP graph. In torch.compile, we
    typically have many passes which work on the joint graph and then end with a
    partitioner.


    The partitioner part is quite mechanical to handle. HOP have their own
    forward and backward graph. The process can be broken into following steps

    1) Get a `joint_hop_gm` from the `fw_hop_gm` and `bw_hop_gm`
    2) Run joint graph passes on the `joint_hop_gm` to get `new_fw_hop_gm` and `new_bw_hop_gm`
    3) Stitch the `new_fw_hop_gm` and `new_bw_hop_gm` back into the `joint_gm`.

    The terminology used in the code is
    `joint_graph/joint_gm` : Refers to the main graph. This may contain many HOPs which have their own `hop_graph`
    `fw_hop_graph/fw_hop_gm` : Refers to the forward graph associated with a HOP.
    `bw_hop_graph/bw_hop_gm` : Refers to the backward graph associated with a HOP.
    `joint_hop_graph/joint_hop_gm` : Refers to the subgraph associated with the HOP like invoke_subgraph.
    `new_fw_hop_graph/new_fw_hop_gm` : Refers to the forward graph after partitioning is applied to `joint_hop_gm`.
    `new_bw_hop_graph/new_bw_hop_gm` : Refers to the backward graph after partitioning is applied to `joint_hop_gm`.

    NB: This pass works for invoke_subgraph today because we took extra care in
    the Autograd.Dispatch key of invoke_subgraph to vastly simplify Step 1.
    """
    from torch._higher_order_ops import invoke_subgraph

    def num_outputs(mod):
        return len(mod.graph.find_nodes(op="output")[0].args[0])

    def num_inputs(mod):
        return len(mod.graph.find_nodes(op="placeholder"))

    new_hop_graphs: dict[str, InvokeSubgraphHopGraphs] = defaultdict(
        lambda: InvokeSubgraphHopGraphs()
    )

    # Step 1 - Get a `joint_hop_gm` from the `fw_hop_gm` and `bw_hop_gm` This is
    # easy to do for `invoke_subgraph` HOP. During the Autograd dispatch key
    # tracing, we have put the joint_hop_graph in the backward hop graph itself.
    # So to recover the joint_hop_gm, we just have to look at the backward
    # HOP graphs.
    # So we will merge step 1 and step 2 in this next section

    # Save the fw and bwd hop nodes. We will later in-place modify the graph
    # using these nodes.
    fw_hop_nodes = []
    bw_hop_nodes = []
    for node in joint_gm.graph.nodes:
        if (
            node.op == "call_function"
            and node.target is invoke_subgraph
            and isinstance(node.args[1], str)
        ):
            if node.args[1].startswith("fw"):
                fw_hop_nodes.append(node)
            elif node.args[1].startswith("bw"):
                bw_hop_nodes.append(node)

    if not bw_hop_nodes:
        return joint_gm

    assert len(fw_hop_nodes) == len(bw_hop_nodes)

    # Create a bw to hop node mapping. This helps us in identifying the bw and
    # fw subgraph pairs without relying on the identifier. This is important
    # because we can have different subgraphs for bwd for same subgraph in the
    # fwd because of differing strides in the backward.
    bw_to_fw_hop_node = dict(zip(list(reversed(bw_hop_nodes)), fw_hop_nodes))

    for node in bw_hop_nodes:
        identifier = node.args[1].removeprefix("bw")

        # If partitioning already done for this identifier, skip. This saves
        # redundant joint graph passes for same subgraphs.
        if new_hop_graphs[identifier].partitioning_done:
            continue

        # Collect some information from the forward hop graph
        fw_hop_node = bw_to_fw_hop_node[node]
        fw_hop_gm = getattr(joint_gm, fw_hop_node.args[0].target)
        assert isinstance(fw_hop_gm, torch.fx.GraphModule)
        num_fw_inputs = num_inputs(fw_hop_gm)
        num_fw_outputs = num_outputs(fw_hop_gm)
        new_hop_graphs[identifier].old_num_fw_inputs = num_fw_inputs
        new_hop_graphs[identifier].old_num_fw_outputs = num_fw_outputs

        # Step 1) - Get the `joint_hop_gm`. As mentioned earlier, the
        # backward graph is the joint graph.
        joint_hop_gm = getattr(joint_gm, node.args[0].target)
        assert isinstance(joint_hop_gm, torch.fx.GraphModule)

        # Prepare the graph for the partitioner
        joint_hop_gm = prepare_for_partitioner(
            joint_hop_gm, num_fw_inputs, num_fw_outputs
        )

        # TODO: invoke_subgraph should track which of its inputs static indices
        # so it can propagate them to the partitioner (and use in cudagraphs)
        static_lifetime_input_indices: list[int] = []
        # Step 2) and 3) - Run joint graph passes and partitioner
        new_fw_hop_gm, new_bw_hop_gm = aot_config.partition_fn(
            joint_hop_gm,
            [],
            num_fwd_outputs=num_fw_outputs,
            static_lifetime_input_indices=static_lifetime_input_indices,
        )

        # Save the new forward and backward graph modules
        new_hop_graphs[identifier].new_fw_hop_gm = new_fw_hop_gm
        new_hop_graphs[identifier].new_bw_hop_gm = new_bw_hop_gm

        # Save the number of symints and saved tensors
        new_fw_out_nodes = new_fw_hop_gm.graph.find_nodes(op="output")[0].args[0]
        extra_outputs = new_fw_out_nodes[num_fw_outputs:]
        symint_outputs = [n for n in extra_outputs if is_sym_node(n)]

        new_hop_graphs[identifier].new_num_sym_nodes = len(symint_outputs)
        new_hop_graphs[identifier].new_num_saved_nodes = len(extra_outputs) - len(
            symint_outputs
        )

        new_hop_graphs[identifier].partitioning_done = True

    # Step 3) Restitch the new fw and bw graphs back into the main graph.
    #
    # This is a very mechanical process. There are a quite a few pieces that we
    # need to connect together to make it work. Lets try to understand the
    # problem statement first.
    #
    # For the forward graph, the signature of the old_fw_hop_gm is
    #   inputs - (*primals)
    #   outputs - (*fw_outs)
    # Now the signature of the new_fw_hop_gm is
    #   inputs - (*primals)     -- This is same
    #   outputs - (*fw_outs, *saved_tensors)    - This is different
    # At a high level, this is an easy transformation, in the new graph we just
    # have to replace the old_fw_hop_gm with the new_fw_hop_gm. Everything else
    # falls into place, because the input signature (i.e. args) is same. And
    # even though output signature is different, fw_outs are still at the same
    # indexes as before. So the forward of the `joint_gm` works nicely.
    #
    # Now, lets look at the backward hop graph. Old signature
    #   inputs - (*primals, *tangents)
    #   outputs - (*grad_outs, *fw_outs)
    # New signature
    #   inputs - (*saved_tensors, *tangents) -- Different
    #   outputs - (*grad_outs)  -- Different
    # Here both input and output signature change. The output signature handling
    # is quite easy because the grads_out are sitting at the right place, so we
    # dont have to do anything.
    #
    # For the input signature, we have to collect the saved tensors from the
    # corresponding forward graph output. We collect all saved_tensors when we
    # see the forward graph, and save it into a map and then later use it during
    # the backward.

    # The stack of fw_nodes for invoke_subgraph HOP. There is an implicit
    # assumption about the graph structure, i.e., if we have hop1, hop2, hop3,
    # ... in the forward part of the joint graph, we will have .., hop3, hop2,
    # hop1 order for the backward. This structure allows us to just use a stack
    # to collect all the information that we need to pass from the forward hop
    # node to the corresponding backward node.

    already_added_new_hop_mods = set()

    def add_new_hop_gm(new_subgraph_mod, name):
        new_subgraph_attr_name = f"partitioned_{name}"
        if new_subgraph_attr_name in already_added_new_hop_mods:
            return new_subgraph_attr_name

        joint_gm.register_module(new_subgraph_attr_name, new_subgraph_mod)
        already_added_new_hop_mods.add(new_subgraph_attr_name)
        return new_subgraph_attr_name

    def propagate_meta_info(new_hop_gm, new_call_function_node, old_call_function_node):
        # Copy all the fields from the old call_function node. And then override
        # the `val` meta field with the outputs of new_hop_gm.
        new_call_function_node.meta = copy.copy(old_call_function_node.meta)

        output = new_hop_gm.graph.find_nodes(op="output")[0]
        out_example_vals = [n.meta["val"] if n else None for n in output.args[0]]
        new_call_function_node.meta["val"] = tuple(out_example_vals)

    for bw_node in reversed(bw_hop_nodes):
        identifier = bw_node.args[1].removeprefix("bw")

        # Make changes to the corresponding fw and bw node pair simultaneously.
        # The removes the need of any bookkeeping.

        # Fw node changes
        # Insert the new_fw_hop_gm. This is straightforward. Get the
        # new_fw_hop_gm, insert the hop_gm as a get_attr fw_node, and then
        # add a call_function fw_node. Additionally, also use getitem
        # call_functions to collect the saved_tensor nodes

        fw_node = bw_to_fw_hop_node[bw_node]
        new_fw_hop_gm = new_hop_graphs[identifier].new_fw_hop_gm
        assert new_fw_hop_gm is not None

        old_num_fw_outputs = new_hop_graphs[identifier].old_num_fw_outputs
        new_num_sym_nodes = new_hop_graphs[identifier].new_num_sym_nodes
        new_num_saved_nodes = new_hop_graphs[identifier].new_num_saved_nodes
        assert old_num_fw_outputs is not None
        assert new_num_sym_nodes is not None
        assert new_num_saved_nodes is not None
        total_outputs = old_num_fw_outputs + new_num_saved_nodes + new_num_sym_nodes

        extra_fw_outputs = []

        # Insert the new_fw_hop_gm into the joint_gm
        with joint_gm.graph.inserting_after(fw_node):
            new_fw_mod_attr_name = add_new_hop_gm(new_fw_hop_gm, f"fw{identifier}")
            new_fw_mod_attr = joint_gm.graph.get_attr(new_fw_mod_attr_name)

        # new_hop_fw_gm output signature is (*fw_outs, *saved_tensors)
        with joint_gm.graph.inserting_after(new_fw_mod_attr):
            new_fw_node = joint_gm.graph.call_function(
                the_function=invoke_subgraph,
                args=(
                    new_fw_mod_attr,
                    new_fw_mod_attr_name,
                    *fw_node.args[2:],
                ),
            )
            propagate_meta_info(new_fw_hop_gm, new_fw_node, fw_node)

        # old_num_fw_outputs = (*fw_outs)
        # new_num_fw_outputs = (*fw_outs, *saved_tensors, *sym_nodes)
        with joint_gm.graph.inserting_after(new_fw_node):
            for fw_out_idx in range(old_num_fw_outputs, total_outputs):
                saved_tensor_node = joint_gm.graph.call_function(
                    the_function=operator.getitem, args=(new_fw_node, fw_out_idx)
                )
                saved_tensor_node.meta = copy.copy(new_fw_node.meta)
                saved_tensor_node.meta["val"] = new_fw_node.meta["val"][fw_out_idx]
                extra_fw_outputs.append(saved_tensor_node)

        fw_node.replace_all_uses_with(new_fw_node)
        joint_gm.graph.erase_node(fw_node)

        # Bw node changes
        # Prepare the operands for the bwd graph
        # Old bw graph signature : (*primals, *tangents)
        # New signature will be : (*sym_nodes, *saved_tensors, *tangents)
        # We have already collected the saved_tensors in the forward hop processing.

        # extra_fw_outputs are in the order (*saved_nodes, *sym_nodes).
        # Partitioner has this quirk where the backward wants sym_nodes
        # first. So extract the sym and saved nodes.

        new_bw_hop_gm = new_hop_graphs[identifier].new_bw_hop_gm
        assert new_bw_hop_gm is not None

        saved_tensor_nodes = extra_fw_outputs[:new_num_saved_nodes]
        sym_nodes = extra_fw_outputs[new_num_saved_nodes:]

        num_primals = new_hop_graphs[identifier].old_num_fw_inputs
        assert num_primals is not None
        tangents = list(bw_node.args[2 + num_primals :])
        operands = sym_nodes + saved_tensor_nodes + tangents

        # Insert the new_bw_hop_gm into the joint_gm
        with joint_gm.graph.inserting_after(bw_node):
            new_bw_mod_attr_name = add_new_hop_gm(new_bw_hop_gm, bw_node.args[1])
            new_bw_mod_attr = joint_gm.graph.get_attr(new_bw_mod_attr_name)

        with joint_gm.graph.inserting_after(new_bw_mod_attr):
            new_bw_node = joint_gm.graph.call_function(
                the_function=invoke_subgraph,
                args=(
                    new_bw_mod_attr,
                    new_bw_mod_attr_name,
                    *operands,
                ),
            )
            propagate_meta_info(new_bw_hop_gm, new_bw_node, bw_node)
            # Since the partitioner is run after the graph passes, we have lost
            # the eager information and cannot faithfully extract the eager
            # inputs for the new partitioned backward graph. For the forward
            # graph, it was fine because the input signature remains same.
            new_bw_node.meta.pop("eager_input_vals", None)

        bw_node.replace_all_uses_with(new_bw_node)
        joint_gm.graph.erase_node(bw_node)

    joint_gm.graph.eliminate_dead_code()
    joint_gm.graph.lint()
    joint_gm.recompile()
    return joint_gm


def maybe_log_graph(
    gm,
    graph_name,
    aot_config,
    structured_log_prefix_fn,
    out_structured_logs: Optional[list[str]] = None,
):
    if not aot_config.enable_log:
        return
    aot_graphs_log.debug(
        "%s",
        lazy_format_graph_code(
            f"{graph_name}",
            gm,
            aot_config.aot_id,
            include_stride=True,
            include_device=True,
            colored=True,
        ),
    )

    def gm_str_fn() -> str:
        return gm.print_readable(
            print_output=False,
            include_stride=True,
            include_device=True,
            expanded_def=True,
        )

    if out_structured_logs is not None:
        out_structured_logs.append(f"{structured_log_prefix_fn()}:{gm_str_fn()}")
    else:
        trace_structured(
            f"{structured_log_prefix_fn()}",
            payload_fn=lambda: gm_str_fn(),
        )


def create_wrap_fn(fn, args):
    from torch.fx.experimental.proxy_tensor import maybe_enable_thunkify

    from .functional_utils import from_fun, has_data_mutation, to_fun

    def assert_no_mutation(t):
        assert not has_data_mutation(t), (
            "Saved tensors hooks with inputs mutations are not allowed"
        )

    @simple_wraps(fn)
    def _wrapper(*args):
        with maybe_enable_thunkify():
            disable_above = torch._C._ExcludeDispatchKeyGuard(
                torch._C.DispatchKeySet(torch._C.DispatchKey.Functionalize)
            )

            with disable_above:
                f_args = pytree.tree_map(to_fun, args)
                f_outs = fn(*f_args)
                pytree.tree_map(assert_no_mutation, f_args)
                return pytree.tree_map(from_fun, f_outs)

    return _wrapper, args


def prepare_hook_gm(aot_config, fn, args):
    from torch._functorch._aot_autograd.graph_capture import _create_graph

    fn, args = create_wrap_fn(fn, args)
    gm = _create_graph(fn, args, aot_config=aot_config)
    return gm


# Inline Autograd saved_tensors_hooks into epilogue of forward graph
# and prologue of backward graph.
# This changes forward graph outputs and inputs.
# Pack hook can return tensors, sym scalars, constants.
# All tensors to save for backward will be grouped together at front.
# Sym scalars grouped on another end. Constants are inlined in the graph.
def maybe_inline_graph_saved_tensors_hooks(
    fw_module,  # torch.fx.GraphModule
    bw_module,  # torch.fx.GraphModule
    num_inner_fwd_outputs,
    inner_meta,
    aot_config,
    static_input_indices,
):
    if torch._dynamo.compiled_autograd.in_compiled_autograd_region:
        return

    get_hooks = torch._functorch._aot_autograd.utils.top_saved_tensors_hooks
    are_inline_hooks = (
        torch._functorch._aot_autograd.utils.saved_tensors_hooks_are_inlineable
    )

    hooks = get_hooks()
    if not are_inline_hooks(hooks):
        return

    pack_hook_gm, unpack_hook_gm = hooks

    structured_logs: list[str] = []
    maybe_log_graph(
        fw_module,
        "Forward graph pre saved_tensors_hooks inlining",
        aot_config,
        lambda: "aot_forward_graph_pre_saved_tensors_hooks",
        structured_logs,
    )
    maybe_log_graph(
        bw_module,
        "Backward graph pre saved_tensors_hooks inlining",
        aot_config,
        lambda: "aot_backward_graph_pre_saved_tensors_hooks",
        structured_logs,
    )
    fw_g = fw_module.graph
    bw_g = bw_module.graph

    fw_g_names = {node.name for node in fw_g.nodes}
    bw_g_names = {node.name for node in bw_g.nodes}

    def _gen_unused_name(candidate: str):
        c = candidate
        i = 0
        while c in fw_g_names or c in bw_g_names:
            c = f"{candidate}_{i}"
            i = i + 1
        return c

    bw_g_inputs = bw_g.find_nodes(op="placeholder")

    fw_out_n = fw_g.output_node()
    fw_outs = fw_out_n.args[0]  # type: ignore[var-annotated]
    fw_outs_inner_set = set(fw_outs[:num_inner_fwd_outputs])
    fw_outs_saved_for_bw = fw_outs[num_inner_fwd_outputs:]
    fw_outs_packed_tensors = []  # type: ignore[var-annotated]
    fw_outs_packed_syms = []  # type: ignore[var-annotated]

    # The main use case for saved_tensors_hooks is activation quantization,
    # for memory usage optimization.
    # Desired behavior is to quantize saved activations to free the original saved tensor.
    # Saved nodes may include forward inputs, outputs, parameters.
    # They may be held by something else and will not be deallocated after quantization.
    # Donated buffers are intermediates in the graph invisible for the user,
    # this guarantees that they can be deallocated.
    # Using this as a default behavior to select saved nodes to apply hooks.
    # There is also a config to apply hooks for all saved nodes without any filtering.
    # The plan is to propagate meta about the source of the saved node to the user hook function.
    mode = torch._functorch.config.saved_tensors_hooks_filtering_mode
    allow_set = None
    exclude_set = None

    if mode == "donated":
        # collect_bw_donated_buffer_idxs requires inner_meta to have num_symints_saved_for_bw
        inner_meta.num_symints_saved_for_bw = len(
            [n for n in fw_outs_saved_for_bw if is_sym_node(n)]
        )
        bw_donated_idxs = collect_bw_donated_buffer_idxs(
            fw_module,
            bw_module,
            inner_meta,
        )
        fw_donated_idxs = [
            i - inner_meta.num_symints_saved_for_bw for i in bw_donated_idxs
        ]
        allow_set = {fw_outs_saved_for_bw[i].name for i in fw_donated_idxs}
    elif mode == "no_static":
        fw_g_inputs = fw_g.find_nodes(op="placeholder")
        exclude_set = {fw_g_inputs[i].name for i in static_input_indices}

    if (allow_set is not None) and (not allow_set):
        # This means we have empty whitelist,
        # No donated (intermediate) saved.
        # Do not do anything in this case
        return

    if aot_config.enable_log:
        structured_logs.append(f"fw_outs_saved_for_bw:{fw_outs_saved_for_bw}")
        structured_logs.append(f"mode:{mode}")
        structured_logs.append(f"allow_set:{allow_set}")
        structured_logs.append(f"exclude_set:{exclude_set}")

    for saved in fw_outs_saved_for_bw:
        if ((allow_set is not None) and (saved.name not in allow_set)) or (
            (exclude_set is not None) and (saved.name in exclude_set)
        ):
            if isinstance(saved.meta["val"], torch.Tensor):
                fw_outs_packed_tensors.append(saved)
            continue

        val = saved.meta["val"]
        if not isinstance(val, torch.Tensor):
            continue

        pack_out_val = pack_hook_gm(val)

        requires_sc_handling = any(
            is_traceable_wrapper_subclass(x) for x in pytree.tree_leaves(pack_out_val)
        )
        if requires_sc_handling:
            raise NotImplementedError(
                "Tensor subclasses in GraphModule saved tensors hooks are not supported"
                "You can workaround it by manually returning subclass's inner tensors"
                " in the pack hook, and reconstructing the subclass in the unpack hook"
            )

        pack_gm = prepare_hook_gm(aot_config, pack_hook_gm, (val,))
        pack_g = pack_gm.graph
        maybe_log_graph(
            pack_gm,
            f"saved_tensors_pack_hook {saved.name}",
            aot_config,
            lambda: f"aot_saved_tensors_hooks_pack {saved.name}",
            structured_logs,
        )
        pack_out_val = pack_gm(val)

        # Install pack hook graph as eiplogue of fw_module.
        # Saved tensor output becomes input of pack hook graph.
        # Replace saved tensor output with pack hook graph output.
        # Outputs symbolic scalars, tensors  are accumulated separately.
        # Then in forward outputs and backward inputs installed in order
        # sym_scalars, packed_saved_tensors.
        # Keeping all tensors together allows to preserve
        # the same identification at runtime,
        # updating only number of saved sym_scalars and tensors.
        pack_g_inputs = pack_g.find_nodes(op="placeholder")
        assert len(pack_g_inputs) == 1
        env = {pack_g_inputs[0]: saved}
        fw_pack_out_args = None
        with fw_g.inserting_before(fw_out_n):
            for node in pack_g.nodes:
                if node.op == "placeholder":
                    continue
                new_n = fw_g.node_copy(node, lambda n: env[n])
                fw_g_names.add(new_n.name)
                env[node] = new_n
                # Output node is temporarily copied to have remapped arguments.
                # Removed in the end.
                if node.op == "output":
                    fw_pack_out_args = new_n.args[0]
                    fw_g.erase_node(new_n)

        env.clear()
        assert fw_pack_out_args
        fw_outs_bw_ins_node_names = []
        for out_idx, _n in enumerate(pytree.tree_leaves(fw_pack_out_args)):
            if not isinstance(_n, torch.fx.Node):
                fw_outs_bw_ins_node_names.append("")
                continue

            # This happens when hook is noop and it is either user input or user output.
            # Do not do anything with this node.
            if _n.op == "placeholder" or _n in fw_outs_inner_set:
                # This means the hook returned input primals unchanged
                # Do not rename in this case.
                n = _n
                new_node_name = _n.name
                fw_outs_bw_ins_node_names.append(new_node_name)
            else:
                # We can not specify desired name in node_copy.
                # Copying node manually to set specific name,
                # to have matching fw_outs, bw_inputs names.
                new_node_name = _gen_unused_name(f"{saved.name}_hook_{out_idx}")
                with fw_g.inserting_before(_n):
                    n = fw_g.create_node(
                        _n.op,
                        _n.target,
                        _n.args,
                        _n.kwargs,
                        name=new_node_name,
                    )
                assert n.name == new_node_name
                fw_outs_bw_ins_node_names.append(new_node_name)
                n.meta = copy.copy(_n.meta)
                _n.replace_all_uses_with(n)
                fw_g.erase_node(_n)
            if isinstance(n.meta["val"], torch.Tensor):
                fw_outs_packed_tensors.append(n)
            elif is_sym_node(n):
                fw_outs_packed_syms.append(n)

        # Install unpack hook graph as a prologue of backward graph
        # Saved tensors inputs are replaced with packed tensors and packed sym scalars.
        # The saved tensors inputs usages in the graph are replaced with unpack hook graph outputs.
        unpack_gm = prepare_hook_gm(aot_config, unpack_hook_gm, (pack_out_val,))
        unpack_g = unpack_gm.graph
        maybe_log_graph(
            unpack_gm,
            f"saved_tensors_unpack_hook {saved.name}",
            aot_config,
            lambda: f"aot_saved_tensors_hooks_unpack {saved.name}",
            structured_logs,
        )

        def find_saved_in_bw_inputs(bw_inputs):
            for n in bw_inputs:
                if n.name == saved.name:
                    return n

        bw_g_input = find_saved_in_bw_inputs(bw_g_inputs)
        assert bw_g_input
        original_bw_g_input_users = list(bw_g_input.users.keys())
        bw_g_input_used_directly = False

        # Replace backward graph saved tensor input with copy of pack graph outputs
        # All non-Tensor, non-symscalars outputs are constanted.

        unpack_g_inputs = unpack_g.find_nodes(op="placeholder")
        env = {}
        for out_idx, (unp_in_n, out_n, val) in enumerate(
            zip(
                unpack_g_inputs,
                pytree.tree_leaves(fw_pack_out_args),
                pytree.tree_leaves(pack_out_val),
            )
        ):
            is_sym = isinstance(val, py_sym_types)
            if isinstance(val, torch.Tensor) or is_sym:
                # We want forward_outputs names to match backward_inputs,
                # Potentially backward may already have "{saved.name}_hook_{idx}",
                # In this case fx.Graph will add suffix.
                new_node_name = fw_outs_bw_ins_node_names[out_idx]
                if bw_g_input.name == new_node_name:
                    env[unp_in_n] = bw_g_input
                    bw_g_input_used_directly = True
                else:
                    # Backward calling convention: ctx_symints,ctx_saved_tensors
                    # Inserting packed sym scalars before first saved tensor input.
                    # Inserting packed tensors before last saved tensor input.
                    # Saved tensor inputs between them will be removed.
                    with (
                        bw_g.inserting_before(bw_g_inputs[0])
                        if is_sym
                        else bw_g.inserting_before(bw_g_input)
                    ):
                        new_n = bw_g.placeholder(new_node_name)
                        assert new_n.name == new_node_name
                    new_n.meta = copy.copy(out_n.meta)
                    env[unp_in_n] = new_n
            else:
                # Inline values of non-Tensor, non-SymScalars
                env[unp_in_n] = val

        # Inserting unpack hook after placeholders.
        bw_unpack_out_n = None
        with bw_g.inserting_before(bw_g_inputs[-1].next):
            for node in unpack_g.nodes:
                if node.op == "placeholder":
                    continue
                new_n = bw_g.node_copy(node, lambda n: env[n])
                bw_g_names.add(new_n.name)
                env[node] = new_n
                # Temporary insert output, to have remapped by node_copy args.
                # Removed in the end.
                if node.op == "output":
                    bw_unpack_out_n = new_n

        assert bw_unpack_out_n
        _leaves = pytree.tree_leaves(bw_unpack_out_n.args)
        assert len(_leaves) == 1
        unpack_saved_tensor_n = _leaves[0]

        if not bw_g_input_used_directly:
            bw_g_input.replace_all_uses_with(unpack_saved_tensor_n)
            bw_g.erase_node(bw_g_input)
        else:
            # Keep usages of bw_g_input in inserted unpacked hook graph.
            # Replace other usages of bw_g_input with unpack_saved_tensor_n.
            from torch._C import _fx_map_arg

            def maybe_replace_node(n):
                return unpack_saved_tensor_n if n == bw_g_input else n

            for use_node in original_bw_g_input_users:
                new_args = _fx_map_arg(use_node.args, maybe_replace_node)
                new_kwargs = _fx_map_arg(use_node.kwargs, maybe_replace_node)
                assert isinstance(new_args, tuple)
                assert isinstance(new_kwargs, dict)
                use_node._update_args_kwargs(new_args, new_kwargs)
        bw_g.erase_node(bw_unpack_out_n)

    # Changing forward graph outputs,
    # Inserting packed_tensors and packed_syms on the place of saved tensors.
    # Packed sym_scalars are together with saved symints
    symint_outs_saved_for_bw = [n for n in fw_outs_saved_for_bw if is_sym_node(n)]
    fw_new_outs = pytree.tree_leaves(
        (
            fw_outs[:num_inner_fwd_outputs],
            fw_outs_packed_tensors,
            fw_outs_packed_syms,
            symint_outs_saved_for_bw,
        )
    )
    fw_out_n.args = (tuple(fw_new_outs),)

    # Assert that saved tensors and symints in forward outputs are aligned with backward inputs
    _fw_n = num_inner_fwd_outputs
    _fw_num_t = len(fw_outs_packed_tensors)
    _fw_num_s = len(fw_outs_packed_syms) + len(symint_outs_saved_for_bw)
    fw_outs_saved_tensors = fw_new_outs[_fw_n : _fw_n + _fw_num_t]
    fw_outs_saved_syms = fw_new_outs[_fw_n + _fw_num_t :]
    bw_new_ins = list(bw_g.find_nodes(op="placeholder"))
    bw_ins_saved_syms = bw_new_ins[:_fw_num_s]
    bw_ins_saved_tensors = bw_new_ins[_fw_num_s : _fw_num_s + _fw_num_t]

    fw_t_names = [n.name for n in fw_outs_saved_tensors]
    bw_t_names = [n.name for n in bw_ins_saved_tensors]
    fw_s_names = [n.name for n in fw_outs_saved_syms]
    bw_s_names = [n.name for n in bw_ins_saved_syms]

    def _log_structured_logs():
        if not aot_config.enable_log:
            return

        trace_structured(
            "artifact",
            metadata_fn=lambda: {
                "name": "aot_saved_tensors_hooks_graphs",
                "encoding": "string",
            },
            payload_fn=lambda: "\n".join(structured_logs),
        )

    if aot_config.enable_log:
        structured_logs.append(
            f"fw_outs[:num_inner_fwd_outputs]:{fw_outs[:num_inner_fwd_outputs]}"
        )
        structured_logs.append(f"fw_outs_packed_tensors:{fw_outs_packed_tensors}")
        structured_logs.append(f"fw_t_names:{fw_t_names}")
        structured_logs.append(f"bw_t_names:{bw_t_names}")
        structured_logs.append(f"fw_s_names:{fw_s_names}")
        structured_logs.append(f"bw_s_names:{bw_s_names}")
        structured_logs.append(f"\nfw_g_pre_assert:{fw_g}")
        structured_logs.append(f"\nbw_g_pre_assert:{bw_g}")
        maybe_log_graph(
            fw_module,
            "Forward graph after transform pre-assert",
            aot_config,
            lambda: "aot_forward_graph_pre_assert_saved_tensors_hooks",
            structured_logs,
        )
        maybe_log_graph(
            bw_module,
            "Backward graph after transform pre-assert",
            aot_config,
            lambda: "aot_backward_graph_pre_assert_saved_tensors_hooks",
            structured_logs,
        )
        _log_structured_logs()

    assert fw_t_names == bw_t_names
    assert fw_s_names == bw_s_names

    fw_g.lint()
    bw_g.lint()
    fw_module.recompile()
    bw_module.recompile()


def aot_stage2_autograd(
    aot_state: AOTState,
    aot_graph_capture: AOTGraphCapture,
) -> DispatchReturn:
    """
    Autograd logic. Generates a joint graph, partitions it, manipulates the input with various wrappers,
    and returns a wrapped torch.autograd.Function with a forward and backward.
    """

    wrappers = aot_graph_capture.wrappers
    fx_g = aot_graph_capture.graph_module
    flat_args = aot_state.flat_args
    joint_inputs = aot_graph_capture.updated_flat_args
    maybe_subclass_meta = aot_graph_capture.maybe_subclass_meta
    aot_config = aot_state.aot_config
    fw_metadata = aot_state.fw_metadata

    CompileEventLogger.try_add_pt2_compile("backend_compile", dispatch_mode="autograd")

    # Copied from aot_dispatch_autograd_graph.
    disable_amp = torch._C._is_any_autocast_enabled()
    joint_graph_str = None
    if aot_config.enable_log:
        aot_joint_log.info(
            "%s",
            lazy_format_graph_code(
                "Joint graph",
                fx_g,
                aot_config.aot_id,
                include_stride=True,
                include_device=True,
                colored=True,
            ),
        )
        joint_graph_str = fx_g.print_readable(
            print_output=False,
            include_stride=True,
            include_device=True,
            expanded_def=True,
        )
        trace_structured(
            "aot_joint_graph",
            payload_fn=lambda: joint_graph_str,
        )

    with torch.no_grad():
        inner_meta = (
            fw_metadata
            if maybe_subclass_meta is None
            else maybe_subclass_meta.fw_metadata
        )
        context = torch._C._DisableAutocast if disable_amp else nullcontext
        with context(), track_graph_compiling(aot_config, "joint"):
            # See Note: [Partitioner handling for Subclasses, Part 1]
            # See Note: [Recomputing subclass mutation handling]
            mutated_inp_runtime_indices = (
                compute_inner_mutated_inp_indices_from_subclass_meta(
                    fw_metadata, inner_meta
                )
            )
            num_tokens = len(fw_metadata.tokens)
            num_mutated_inp_runtime_indices = len(mutated_inp_runtime_indices)
            num_inner_fwd_outputs = (
                num_mutated_inp_runtime_indices
                + inner_meta.num_outputs
                + inner_meta.num_intermediate_bases
                + inner_meta.num_outputs_rng_offset
                + num_tokens  # See Note [Side-Effectful Tokens in AOTAutograd]
            )
            fake_mode = detect_fake_mode()
            fx_g = run_joint_graph_passes_on_hops(fx_g, joint_inputs, aot_config)

            # TODO(anijain2305) - Add tensorify_python_scalars to the HOP graph passes.
            if fake_mode is not None and fake_mode.shape_env is not None:
                tensorify_python_scalars(fx_g, fake_mode.shape_env, fake_mode)

            static_lifetime_input_indices = fw_metadata.static_input_indices
            fw_module, bw_module = aot_config.partition_fn(
                fx_g,
                joint_inputs,
                num_fwd_outputs=num_inner_fwd_outputs,
                static_lifetime_input_indices=static_lifetime_input_indices,
            )
            rng_states = [
                n
                for n in fw_module.graph.find_nodes(op="placeholder")
                if "fwd_rng_state" in n.name
            ]
            fw_metadata.num_graphsafe_rng_states = len(rng_states)
            if rng_states:
                fw_metadata.graphsafe_rng_state_index = (
                    rng_states[0].meta["val"].device.index
                )

            # See Note [Side-Effectful Tokens in AOTAutograd]
            if config.unlift_effect_tokens and (
                num_tokens > 0 or fw_metadata.num_backward_tokens > 0
            ):
                unlift_tokens(fw_module, fw_metadata, aot_config, bw_module)

                num_inner_fwd_outputs -= num_tokens
                joint_inputs = (
                    joint_inputs[0][num_tokens:],
                    joint_inputs[1],
                )

            maybe_inline_graph_saved_tensors_hooks(
                fw_module,
                bw_module,
                num_inner_fwd_outputs,
                inner_meta,
                aot_config,
                fw_metadata.static_input_indices,
            )
            static_lifetime_input_indices = fw_metadata.static_input_indices

            fw_outs = next(iter(fw_module.graph.find_nodes(op="output"))).args[0]
            # we only need to bookkeep the symints that are saved for bw, not any symints
            # the user forward might have returned in its own output
            fw_outs_saved_for_bw = fw_outs[num_inner_fwd_outputs:]
            num_fw_outs_saved_for_bw = len(fw_outs_saved_for_bw)
            symint_outs_saved_for_bw = []
            for idx, node in enumerate(fw_outs_saved_for_bw):
                if is_sym_node(node):
                    symint_outs_saved_for_bw.append(node)
                elif (
                    isinstance(node, torch.fx.Node)
                    and "val" in getattr(node, "meta", {})
                    and isinstance(node.meta["val"], FakeTensor)
                ):
                    # record dynamic tensor activations
                    dynamic_dims: set[int] = {
                        dim
                        for dim, size in enumerate(node.meta["val"].shape)
                        if not isinstance(size, int)
                    }
                    if dynamic_dims:
                        fw_metadata.dynamic_saved_tensors_idxs[idx] = dynamic_dims

            fw_metadata.num_symints_saved_for_bw = len(symint_outs_saved_for_bw)
            inner_meta.num_symints_saved_for_bw = len(symint_outs_saved_for_bw)
            num_symints_saved_for_bw = len(symint_outs_saved_for_bw)
            if torch._functorch.config.donated_buffer:
                fw_metadata.bw_donated_idxs = collect_bw_donated_buffer_idxs(
                    fw_module,
                    bw_module,
                    inner_meta,
                )
                inner_meta.bw_donated_idxs = fw_metadata.bw_donated_idxs

        if aot_config.enable_log:
            trace_structured(
                "artifact",
                metadata_fn=lambda: {
                    "name": "torch._functorch.config",
                    "encoding": "string",
                },
                payload_fn=lambda: torch._functorch.config.get_config_copy(),
            )
            aot_graphs_log.info(
                "aot_config id: %s, fw_metadata=%s, inner_meta=%s",
                str(aot_config.aot_id),
                str(fw_metadata),
                str(inner_meta),
            )

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
        # allowing autograd to reuse the graph.
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
        _indices_of_inps_to_detach: list[int] = []

        # reversed() since we expect output at end of graph
        bw_output = next(reversed(bw_module.graph.find_nodes(op="output")))
        bw_outs: Sequence[torch.fx.Node] = bw_output.args[0]  # type: ignore[assignment]

        # TODO: we should apply the below "detach inputs if their gradients are statically known to be None"
        # optimization even if we have subclass inputs/outputs (we do not handle this today).
        # Computing which our our inputs get None gradients is a bit more complicated,
        # if any of our inputs are subclasses. Why?
        # (a) we need to make sure that we call .detach() on the input subclasses, since autograd sees subclasses.
        # (b) The grad_outputs that we AOT computed in our backward graph are the desugared tensor tensors,
        #     so we need to figure out which subclass fw inputs they map to.
        if maybe_subclass_meta is None:
            num_backward_tokens: int = inner_meta.num_backward_tokens
            assert (
                len(bw_outs)
                == len(fw_metadata.input_info)
                + inner_meta.num_outputs_rng_offset
                + num_backward_tokens
            )
            bw_outs_no_rng_no_tokens = bw_outs
            if (inner_meta.num_outputs_rng_offset + num_backward_tokens) > 0:
                bw_outs_no_rng_no_tokens = bw_outs[
                    : -(inner_meta.num_outputs_rng_offset + num_backward_tokens)
                ]
            assert len(bw_outs_no_rng_no_tokens) == len(fw_metadata.input_info)

            for i, (bw_out) in enumerate(bw_outs_no_rng_no_tokens):
                # If our input experiences a metadata mutation inside the graph (e.g. set_()),
                # we *must* not detach, otherwise it will be the detach'd input that gets the metadata mutation
                metadata_mutation_in_graph = (
                    fw_metadata.input_info[i].mutation_type
                    == MutationType.MUTATED_IN_GRAPH
                    and fw_metadata.input_info[i].mutates_storage_metadata
                )
                is_non_leaf = (
                    fw_metadata.input_info[i].requires_grad
                    and not fw_metadata.input_info[i].is_leaf
                )
                if bw_out is None and not metadata_mutation_in_graph and is_non_leaf:
                    _indices_of_inps_to_detach.append(i)

        fw_module_str = None
        bw_module_str = None
        if aot_config.enable_log:
            aot_graphs_log.info(
                "%s",
                lazy_format_graph_code(
                    "Forward graph",
                    fw_module,
                    aot_config.aot_id,
                    include_stride=True,
                    include_device=True,
                    colored=True,
                ),
            )
            aot_graphs_log.info(
                "%s",
                lazy_format_graph_code(
                    "Backward graph",
                    bw_module,
                    aot_config.aot_id,
                    include_stride=True,
                    include_device=True,
                    colored=True,
                ),
            )
            fw_module_str = fw_module.print_readable(
                print_output=False,
                include_stride=True,
                include_device=True,
                expanded_def=True,
            )
            bw_module_str = bw_module.print_readable(
                print_output=False,
                include_stride=True,
                include_device=True,
                expanded_def=True,
            )

            trace_structured(
                "artifact",
                metadata_fn=lambda: {
                    "name": "aot_forward_graph_fw_metadata",
                    "encoding": "string",
                },
                payload_fn=lambda: dataclass_repr(fw_metadata),
            )
            if maybe_subclass_meta is not None:
                trace_structured(
                    "artifact",
                    metadata_fn=lambda: {
                        "name": "aot_forward_graph_fw_subclass_metadata",
                        "encoding": "string",
                    },
                    payload_fn=lambda: dataclass_repr(maybe_subclass_meta),
                )

            trace_structured(
                "aot_forward_graph",
                payload_fn=lambda: fw_module_str,
            )
            trace_structured(
                "aot_backward_graph",
                payload_fn=lambda: bw_module_str,
            )

        # AMP is already traced out in joint graph. we do not wish to reapply it accidentally
        # in the compiler.
        with track_graph_compiling(aot_config, "forward"), torch._C._DisableAutocast():
            # flat_args at this point might still be subclasses-
            # make sure to pass the unwrapped fake tensors into the compiler!
            adjusted_flat_args = joint_inputs[0]

            fakified_out_wrapper = FakifiedOutWrapper()
            fakified_out_wrapper.pre_compile(
                fw_module, adjusted_flat_args, aot_config, fw_metadata=fw_metadata
            )

            functionalized_rng_wrapper = FunctionalizedRngRuntimeWrapper(
                return_new_outs=False
            )

            if rng_states:
                index = fw_metadata.graphsafe_rng_state_index
                assert index is not None
                rng_states = [
                    get_cuda_generator_meta_val(index)
                    for _ in range(fw_metadata.num_graphsafe_rng_states)
                ]
                adjusted_flat_args.extend(rng_states)  # type: ignore[arg-type]

            functionalized_rng_wrapper.pre_compile(
                fw_module, adjusted_flat_args, aot_config, fw_metadata=fw_metadata
            )
            if tracing_context := torch._guards.TracingContext.try_get():
                tracing_context.fw_metadata = inner_meta

            with TracingContext.report_output_strides() as fwd_output_strides:
                compiled_fw_func = aot_config.fw_compiler(fw_module, adjusted_flat_args)

            if not getattr(compiled_fw_func, "_boxed_call", False):
                compiled_fw_func = make_boxed_func(compiled_fw_func)

            if fakified_out_wrapper.needs_post_compile:
                fakified_out_wrapper.set_fwd_output_strides(fwd_output_strides)

            compiled_fw_func = EffectTokensWrapper().post_compile(
                compiled_fw_func,
                aot_config,
                runtime_metadata=fw_metadata,
            )

            compiled_fw_func = AOTDispatchSubclassWrapper(
                fw_only=None,
                trace_joint=False,
                maybe_subclass_meta=maybe_subclass_meta,
                num_fw_outs_saved_for_bw=num_fw_outs_saved_for_bw,
            ).post_compile(
                compiled_fw_func,
                aot_config,  # not used
                runtime_metadata=fw_metadata,
            )

            compiled_fw_func = functionalized_rng_wrapper.post_compile(
                compiled_fw_func, aot_config, runtime_metadata=fw_metadata
            )
            compiled_fw_func = fakified_out_wrapper.post_compile(
                compiled_fw_func,
                aot_config,
                runtime_metadata=fw_metadata,
            )

        # NB: It's important to compile backwards ahead of time, as this may
        # add extra guards which we need to apply to the Dynamo cache at
        # forwards
        with track_graph_compiling(aot_config, "backward"), torch._C._DisableAutocast():
            placeholder_list = fx_placeholder_vals(bw_module)

            forward_saved_for_backwards_strides = None
            if fwd_output_strides is not None:
                forward_saved_for_backwards_strides = fwd_output_strides[
                    inner_meta.tensors_saved_for_backwards_slice
                ]

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
                j = i - num_symints_saved_for_bw
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
                    # pyrefly: ignore  # bad-argument-type
                    placeholder_list[i] = ph_arg.as_strided(ph_arg.size(), real_stride)

            compiled_bw_func = None
            if (
                num_symints_saved_for_bw > 0
                or aot_config.force_non_lazy_backward_lowering
            ):
                try:
                    # See Note: [Backward graph lazy lowering]
                    with torch._subclasses.fake_tensor.unset_fake_temporarily():
                        # If bw_module contains lifted constants, they will be real tensors stored as
                        # GraphModule. Deepcopying tensors under fake mode is not supported and will
                        # raise when attempting to set storage.
                        bw_module_copy = copy.deepcopy(bw_module)
                    compiled_bw_func = aot_config.bw_compiler(
                        bw_module_copy, placeholder_list
                    )
                    del bw_module_copy
                except Exception as e:
                    if aot_config.force_non_lazy_backward_lowering:
                        raise
                    exc = e
                    trace_structured(
                        "artifact",
                        metadata_fn=lambda: {
                            "name": "eager_compile_backwards_failure",
                            "encoding": "string",
                        },
                        payload_fn=lambda: "\n".join(
                            traceback.format_exception(
                                type(exc), exc, exc.__traceback__
                            )
                        ),
                    )
                    log.warning(
                        "failed to eagerly compile backwards for dynamic, suppressing in case backwards not needed",
                        exc_info=True,
                    )
            # Compiled autograd will run the bw_module in the backward pass,
            # so recompilation need happen anyway if the backward pass is ever
            # called.
            #
            # The reason we do the GraphModule recompilation here is because
            # the lazy recompilation will cause issue in the backward pass
            # with compiled autograd.
            #
            # Do the _LazyGraphModule.force_recompile here rather than when
            # bw_module is first generated by the partitioner because the bw_module.recompile
            # may be called in some code path later and cause the _LazyGraphModule.forward
            # becomes the lazy version again. One example is when dynamic shape is enabled
            # upfront, the bw_compiler will be called above which can cause extra
            # graph module recompilation on bw_module.
            if torch._dynamo.compiled_autograd.in_compiled_autograd_region:
                from torch.fx._lazy_graph_module import _LazyGraphModule

                _LazyGraphModule.force_recompile(bw_module)

    saved_context = TracingContext.try_get()
    saved_compile_context = CompileContext.try_get()

    backward_state_indices = [
        idx for idx, x in enumerate(flat_args) if isinstance(x, BackwardState)
    ]
    assert len(backward_state_indices) <= 1

    lazy_backward_info = AutogradLazyBackwardCompileInfo(
        bw_module,
        placeholder_list,
        saved_context,
        saved_compile_context,
    )

    make_runtime_safe(fw_metadata, maybe_subclass_meta)

    try_save_cache_entry: Optional[Callable] = None
    entry: Optional[GenericAOTAutogradCacheEntry] = None

    if aot_config.cache_info is not None:
        forward_time_taken_ns = time.time_ns() - aot_config.cache_info.start_time_ns

        # NB: aot_config here is technically not needed as an argument: we could just
        # close over aot_config.cache_info, since aot_config never changes.
        # But closing over random variables is confusing IMO, so I'm leaving it.
        def try_save_cache_entry(  # noqa: F811
            compiled_bw_func: Callable,
            bw_module: torch.fx.GraphModule,
            _fw_metadata: ViewAndMutationMeta,
            aot_config: AOTConfig,
        ) -> Optional[GenericAOTAutogradCacheEntry]:
            cache_info = aot_config.cache_info

            def should_save_cache():
                if should_bundle_autograd_cache():
                    return True
                else:
                    return hasattr(compiled_fw_func, "_fx_graph_cache_key") and hasattr(
                        compiled_bw_func, "_fx_graph_cache_key"
                    )

            if cache_info is not None and should_save_cache():
                assert forward_time_taken_ns is not None
                # TODO: technically, AOTAutograd does a *little* bit of post processing work
                # in the backward that isn't measured here. But it's small enough that it's not worth
                # the complexity of threading a bunch of times through the code, so we
                # use the compiled_bw_func's inductor compile time instead.
                # It's possible this changes in the future, in which case we should
                # update backward_time_taken_ns to be more inclusive
                backward_time_taken_ns = getattr(compiled_bw_func, "_time_taken_ns", 0)

                aot_forward_graph_str: Optional[str] = fw_module_str
                aot_backward_graph_str: Optional[str] = bw_module_str
                aot_joint_graph_str: Optional[str] = joint_graph_str
                guards_expr = AOTAutogradCache.generate_guards_expression(cache_info)

                entry = AOTAutogradCache.make_entry(
                    compiled_fw_func,  # type: ignore[arg-type]
                    compiled_bw_func,  # type: ignore[arg-type]
                    aot_joint_graph_str,
                    aot_forward_graph_str,
                    aot_backward_graph_str,
                    _fw_metadata,
                    wrappers,
                    maybe_subclass_meta,
                    num_fw_outs_saved_for_bw,
                    _indices_of_inps_to_detach,
                    forward_time_taken_ns,
                    backward_time_taken_ns,
                    sanitized_aot_config=sanitize_aot_config(aot_config),
                    guards_expr=guards_expr,
                    backward_state_indices=backward_state_indices,
                    num_symints_saved_for_bw=num_symints_saved_for_bw,
                    serialized_bw_module=serialize_graph_module(bw_module),
                )
                remote = should_use_remote_autograd_cache()
                AOTAutogradCache.save(cache_info.cache_key, entry, remote)
                return entry
            return None

        if compiled_bw_func is not None:
            # If we already compiled the backward, we save its cache entry now
            entry = try_save_cache_entry(
                compiled_bw_func, bw_module, fw_metadata, aot_config
            )
            try_save_cache_entry = None

    compiled_fn = AOTDispatchAutograd.post_compile(
        compiled_fw_func,
        compiled_bw_func,
        maybe_subclass_meta,
        num_symints_saved_for_bw,
        backward_state_indices,
        disable_amp,
        _indices_of_inps_to_detach,
        lazy_backward_info,
        aot_config,
        fw_metadata=fw_metadata,
        try_save_cache_entry=try_save_cache_entry,
    )

    if entry is not None:
        compiled_fn = SerializableCompiledFunction(compiled_fn, lambda: entry)

    if config.debug_assert:
        flat_requires_grad: list[Optional[bool]] = [
            a.requires_grad if isinstance(a, Tensor) else None for a in flat_args
        ]
        compiled_fn = DebugAssertWrapper(
            flat_requires_grad=flat_requires_grad
        ).post_compile(compiled_fn, aot_config, runtime_metadata=fw_metadata)

    compiled_fn = post_compile(
        wrappers,
        compiled_fn,
        aot_config,
        runtime_metadata=fw_metadata,
    )
    return compiled_fn
