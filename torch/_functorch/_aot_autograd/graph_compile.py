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
import functools
import itertools
import logging
import operator
import threading
import time
import traceback
from collections import defaultdict
from collections.abc import Callable, Generator
from contextlib import contextmanager, nullcontext
from typing import Any

import torch
import torch.utils._pytree as pytree
import torch.utils.dlpack
from torch import Tensor
from torch._custom_class_base import CustomClassBase
from torch._dynamo.utils import (
    CompileEventLogger,
    detect_fake_mode,
    dynamo_timed,
    lazy_format_graph_code,
)
from torch._guards import CompileContext, TracingContext
from torch._library.fake_class_registry import FakeScriptObject
from torch._library.opaque_object import is_custom_class_obj
from torch._logging import getArtifactLogger, trace_structured
from torch._subclasses import FakeTensor
from torch._subclasses.fake_tensor import is_fake_tensor
from torch._subclasses.meta_utils import is_sparse_any
from torch.fx.experimental._backward_state import BackwardState
from torch.fx.experimental.proxy_tensor import is_sym_node
from torch.fx.experimental.symbolic_shapes import fx_placeholder_vals, guard_or_true
from torch.fx.graph_module import GraphModule
from torch.fx.passes._tensorify_python_scalars import tensorify_python_scalars
from torch.multiprocessing.reductions import StorageWeakRef
from torch.types import py_sym_types
from torch.utils._python_dispatch import is_traceable_wrapper_subclass
from torchgen.utils import dataclass_repr

from .. import config
from .aot_autograd_result import GenericAOTAutogradResult, serialize_graph_module
from .autograd_cache import (
    AOTAutogradCache,
    should_bundle_autograd_cache,
    should_use_remote_autograd_cache,
)
from .descriptors import AOTOutput, PlainAOTOutput
from .graph_capture import aot_dispatch_autograd_graph, aot_dispatch_base_graph
from .logging_utils import track_graph_compiling
from .runtime_wrappers import (
    AOTDedupeWrapper,
    AOTDispatchAutograd,
    AOTDispatchAutogradCompileSpec,
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
    SubclassMeta,
    ViewAndMutationMeta,
)
from .subclass_utils import compute_inner_mutated_inp_indices_from_subclass_meta
from .utils import (
    contain_metadata_mutation_ops,
    get_default_generator,
    make_boxed_func,
    simple_wraps,
    strict_zip,
    unlift_tokens,
)


def is_opaque_node(node: Any) -> bool:
    """Check if a node contains an opaque or non-tensor value (e.g., ProcessGroup)."""
    from torch._library.fake_class_registry import FakeScriptObject

    if not isinstance(node, torch.fx.Node):
        return False
    if "val" not in getattr(node, "meta", {}):
        return False
    val = node.meta["val"]
    if is_custom_class_obj(val):
        return True
    if isinstance(val, (torch.ScriptObject, FakeScriptObject)):
        return True
    return False


_thread_local = threading.local()


def _should_save_cache(*compiled_fns: Callable[..., Any]) -> bool:
    if should_bundle_autograd_cache():
        return True
    return all(
        getattr(fn, "_fx_graph_cache_key", None) is not None for fn in compiled_fns
    )


@contextmanager
def maybe_skip_decompose(aot_config: AOTConfig) -> Generator[AOTConfig, None, None]:
    if config.selective_decompose:
        yield dataclasses.replace(aot_config, decompositions={})
    else:
        yield aot_config


# Saved tensor hooks context
# Compiled saved tensor hooks are convenient way to inline some logic in the graphs
# for saved nodes from forward to backward. (E.g. activations quantization)
# In base implementation user does not have any additional information about saved value
# in the hook, except FakeTensor shape, dtype, device etc.
# _get_saved_tensor_hook_context gives additional graph information about that saved value,
# that can be used to make a decisions which pack/unpack to apply for particular saved value.
# This allows user to reuse saved tensors hooks api to apply selective pack/unpack in
# graph aware way.
# Alternative to this will be making user to write a custom pass that mucks with forward outputs,
# backward input metadata, which requires significantly more effort.
#
# As for now in context we expose forward graph, backward graph and current saved node,
# which contains node.meta with additional information about that fx.Node.
# Warning: This API may change without backward compatibility.
@contextmanager
def _saved_tensor_hook_context(state: dict[str, Any]) -> Generator[None, None, None]:
    previous_state = getattr(_thread_local, "state", None)
    try:
        _thread_local.state = state
        yield
    finally:
        # Clean up: restore previous state or remove attribute
        if previous_state is not None:
            _thread_local.state = previous_state
        else:
            if hasattr(_thread_local, "state"):
                delattr(_thread_local, "state")


def _get_saved_tensor_hook_context() -> dict[str, Any] | None:
    return getattr(_thread_local, "state", None)


zip = strict_zip

log = logging.getLogger(__name__)
aot_joint_log = getArtifactLogger(__name__, "aot_joint_graph")
aot_graphs_log = getArtifactLogger(__name__, "aot_graphs")

aten = torch.ops.aten

# Returns a Callable and a ViewAndMutationMeta.
# Currently, only export needs the ViewAndMutationMeta after this function.
# TODO: Refactor this
DispatchReturn = tuple[Callable[..., Any], ViewAndMutationMeta]


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
    if aot_config.disable_functionalization:
        # Effect tokens are introduced by FunctionalTensorMode.  The
        # disable_functionalization path intentionally traces without the
        # effect-token wrapper, so metadata-discovered tokens must not affect
        # graph signatures or forward/backward output partitioning.
        aot_state.fw_metadata.tokens = {}
        aot_state.fw_metadata.num_backward_tokens = 0

    # NB: This is currently only used for backwards, where fwd/bwd
    # deterministic TLS can be different
    aot_state.fw_metadata.deterministic = torch.are_deterministic_algorithms_enabled()
    updated_flat_args: list[Any] | tuple[list[Any], list[Any]]

    with maybe_skip_decompose(aot_config) as graph_capture_aot_config:
        # if config.selective_decompose, skip decomposition and apply selective_decompose
        # after we get the joint graph. See [Note: Selective Decomposition] for details.
        if aot_state.needs_autograd and not aot_config.pre_dispatch:
            # FYI: this being moved to trigger in export is new, seems fine!
            with dynamo_timed("aot_trace_joint_graph", log_pt2_compile_event=True):
                (
                    graph,
                    updated_flat_args,
                    updated_flat_args_descs,
                    maybe_subclass_meta,
                ) = aot_dispatch_autograd_graph(
                    flat_fn,
                    aot_state.flat_args,
                    aot_state.flat_args_descs,
                    graph_capture_aot_config,
                    fw_metadata=aot_state.fw_metadata,
                )
        else:
            graph, updated_flat_args, updated_flat_args_descs, maybe_subclass_meta = (
                aot_dispatch_base_graph(
                    flat_fn,
                    aot_state.flat_args,
                    aot_state.flat_args_descs,
                    graph_capture_aot_config,
                    fw_metadata=aot_state.fw_metadata,
                )
            )
    if config.selective_decompose:
        from torch.fx.experimental.proxy_tensor import selective_decompose
        from torch.fx.passes.regional_inductor import _needs_inductor_compile

        graph = selective_decompose(
            graph,
            *updated_flat_args,
            decomposition=aot_config.decompositions,
            should_decompose=_needs_inductor_compile,
            trace_joint_graph=aot_state.needs_autograd and not aot_config.pre_dispatch,
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
        wrappers,
        graph,  # pyrefly: ignore [bad-argument-type]
        aot_config,
        runtime_metadata=aot_state.fw_metadata,
    )

    # Therefore, since no wrapperes run, we don't get back a callable - we get back the raw fx graph
    # (either a joint or an inference-only graph)
    if not isinstance(compiled_fn, torch.fx.GraphModule):
        raise AssertionError(
            f"expected compiled_fn to be GraphModule, got {type(compiled_fn)}"
        )
    return compiled_fn, aot_state.fw_metadata


def _get_inner_meta(
    maybe_subclass_meta: SubclassMeta | None,
    fw_metadata: ViewAndMutationMeta,
) -> ViewAndMutationMeta:
    """
    Util to get view and mutation metadata.
    """
    return (
        fw_metadata if maybe_subclass_meta is None else maybe_subclass_meta.fw_metadata
    )


def _apply_tensorify_python_scalars(module: torch.fx.GraphModule) -> None:
    """
    Util to apply tensorify_python_scalars.
    """
    # TODO(anijain2305) - Add tensorify_python_scalars to the HOP graph passes.
    fake_mode = detect_fake_mode()
    if fake_mode is not None and fake_mode.shape_env is not None:
        tensorify_python_scalars(module, fake_mode.shape_env, fake_mode)


def aot_stage2_compile(
    aot_state: AOTState,
    aot_graph_capture: AOTGraphCapture,
    # pyrefly: ignore [implicit-any]
    partition_fn: Callable,
    # pyrefly: ignore [implicit-any]
    fw_compiler: Callable,
    # pyrefly: ignore [implicit-any]
    bw_compiler: Callable | None = None,
    # pyrefly: ignore [implicit-any]
    inference_compiler: Callable | None = None,
) -> DispatchReturn:
    if bw_compiler is None:
        bw_compiler = fw_compiler
    if inference_compiler is None:
        inference_compiler = fw_compiler

    if aot_state.needs_autograd and not aot_state.aot_config.pre_dispatch:
        return aot_stage2_autograd(
            aot_state,
            aot_graph_capture,
            partition_fn,
            fw_compiler,
            bw_compiler,
        )
    else:
        return aot_stage2_inference(
            aot_state,
            aot_graph_capture,
            partition_fn,
            inference_compiler,
        )


def _log_inference_graph(
    fw_module: torch.fx.GraphModule,
    aot_config: AOTConfig,
) -> str | None:
    """
    Log the inference graph to the structured logger.
    Return a str representation of the graph.
    """
    if aot_config.enable_log:
        trace_structured(
            "artifact",
            metadata_fn=lambda: {
                "name": "torch._functorch.config",
                "encoding": "string",
            },
            payload_fn=lambda: torch._functorch.config.get_serializable_config_copy(),
        )

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

    return aot_forward_graph_str


def _aot_stage2b_inference_compile(
    fw_module: torch.fx.GraphModule,
    updated_flat_args: list[Any],
    maybe_subclass_meta: SubclassMeta | None,
    fw_metadata: ViewAndMutationMeta,
    aot_config: AOTConfig,
    # pyrefly: ignore [implicit-any]
    inference_compiler: Callable,
    # pyrefly: ignore [implicit-any]
) -> Callable:
    return _aot_stage2b_compile_forward_or_inference(
        fw_module,
        updated_flat_args,  # type: ignore[arg-type]
        maybe_subclass_meta,
        fw_metadata,
        aot_config,
        inference_compiler,
        is_inference=True,
    )[1]


def aot_stage2_inference(
    aot_state: AOTState,
    aot_graph_capture: AOTGraphCapture,
    # pyrefly: ignore [implicit-any]
    partition_fn: Callable,
    # pyrefly: ignore [implicit-any]
    inference_compiler: Callable,
) -> DispatchReturn:
    """
    Handles functions that don't need autograd. Runs wrappers and compiles with
    the stage-2 inference compiler.
    """

    aot_config = aot_state.aot_config
    fw_metadata = aot_state.fw_metadata
    fw_module = aot_graph_capture.graph_module
    wrappers = aot_graph_capture.wrappers
    updated_flat_args = aot_graph_capture.updated_flat_args
    maybe_subclass_meta = aot_graph_capture.maybe_subclass_meta

    CompileEventLogger.try_add_pt2_compile("backend_compile", dispatch_mode="inference")
    aot_forward_graph_str = _log_inference_graph(fw_module, aot_config)

    if not isinstance(fw_module, GraphModule):
        raise AssertionError(
            f"expected fw_module to be GraphModule, got {type(fw_module)}"
        )
    _apply_tensorify_python_scalars(fw_module)

    # When trace_autograd_ops=True, the inference graph may contain fw/bw
    # invoke_subgraph pairs from traced torch.autograd.grad/backward calls.
    # Partition them before the remat pass so that remat duplicates the
    # already-partitioned fw subgraphs (which produce saved tensors for bw).
    fw_module = run_joint_graph_passes_on_hops(
        fw_module, None, aot_config, default_partition_fn=partition_fn
    )

    # Apply AC rematerialization after HOP partitioning. This must happen
    # after partitioning so remat duplicates the partitioned fw subgraphs
    # (not the original unpartitioned ones).
    if torch._functorch.config.remat_using_tags_for_fwd_loss_bwd_graph:
        from torch._functorch._activation_checkpointing.remat_using_tags_for_fwd_loss_bwd_graph_pass import (
            remat_using_tags_for_fwd_loss_bwd_graph,
        )

        fw_module = remat_using_tags_for_fwd_loss_bwd_graph(fw_module)

    if _has_invoke_subgraph_node(fw_module):
        trace_structured(
            "artifact",
            metadata_fn=lambda: {
                "name": "aot_inference_graph_after_hop_passes",
                "encoding": "string",
            },
            payload_fn=lambda: fw_module.print_readable(
                print_output=False,
                include_stride=True,
                include_device=True,
                expanded_def=True,
            ),
        )

    compiled_fw = _aot_stage2b_inference_compile(
        fw_module,
        updated_flat_args,  # type: ignore[arg-type]
        maybe_subclass_meta,
        fw_metadata,
        aot_config,
        inference_compiler,
    )

    entry = _cache_inference_info(
        aot_config,
        fw_metadata,
        maybe_subclass_meta,
        compiled_fw,
        aot_forward_graph_str,
        wrappers,
    )

    return _aot_stage2c_make_inference_function(
        aot_config,
        fw_metadata,
        compiled_fw,
        wrappers,
        entry,
    )


def _cache_inference_info(
    aot_config: AOTConfig,
    fw_metadata: ViewAndMutationMeta,
    maybe_subclass_meta: SubclassMeta | None,
    compiled_fw: Callable[..., Any],
    aot_forward_graph_str: str | None,
    wrappers: list[CompilerWrapper],
) -> GenericAOTAutogradResult[Any, Any] | None:
    make_runtime_safe(fw_metadata, maybe_subclass_meta)

    cache_info = aot_config.cache_info

    entry: GenericAOTAutogradResult[Any, Any] | None = None
    if cache_info is not None and _should_save_cache(compiled_fw):
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
            sanitized_aot_config=aot_config.to_cacheable(),
            guards_expr=guards_expr,
            backward_state_indices=None,
            num_symints_saved_for_bw=None,
            serialized_bw_module=None,
            min_cut_info_str=None,
        )
        AOTAutogradCache.save(
            cache_info.cache_key,
            entry,
            remote=should_use_remote_autograd_cache(),
        )

    return entry


def _aot_stage2c_make_inference_function(
    aot_config: AOTConfig,
    fw_metadata: ViewAndMutationMeta,
    compiled_fw: Callable[..., Any],
    wrappers: list[CompilerWrapper],
    entry: GenericAOTAutogradResult[Any, Any] | None,
) -> DispatchReturn:
    if entry is not None:
        compiled_fw = SerializableCompiledFunction(compiled_fw, lambda: entry)

    disable_amp = torch._C._is_any_autocast_enabled()
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
    fw_ins: list[FakeTensor | None],
    user_fw_outs: list[FakeTensor | None],
    bw_outs: list[FakeTensor | None],
    saved_tensors: list[FakeTensor | None],
) -> list[int]:
    """
    Checks if the saved tensors are donated buffers, which means a saved tensor is not
    an alias of any tensors in fw_ins, user_fw_outs, and bw_outs.
    """

    storage_refs = set()

    for t in itertools.chain(fw_ins, user_fw_outs, bw_outs):
        # Only access storage if a tensor has storage (not sparse)
        if t is not None and is_fake_tensor(t) and not is_sparse_any(t):
            storage_refs.add(StorageWeakRef(t.untyped_storage()))

    num_saved_tensor = len(saved_tensors)
    donated_buffer_idxs = []
    for i in range(num_saved_tensor):
        t = saved_tensors[i]
        if (
            t is not None
            and is_fake_tensor(t)
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
        saved_tensors,
    )

    if fw_metadata.num_symints_saved_for_bw is None:
        raise AssertionError("fw_metadata.num_symints_saved_for_bw must not be None")
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
    old_num_fw_outputs: int | None = None
    old_num_fw_inputs: int | None = None

    new_fw_hop_gm: torch.fx.GraphModule | None = None
    new_bw_hop_gm: torch.fx.GraphModule | None = None
    new_num_sym_nodes: int | None = None
    new_num_saved_nodes: int | None = None


def prepare_for_partitioner(
    mod: torch.fx.GraphModule, num_primals: int, num_fw_outputs: int
) -> torch.fx.GraphModule:
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


def _get_partition_fn(
    fw_hop_node: torch.fx.Node,
    aot_config: AOTConfig,
    default_partition_fn: Callable[
        ..., tuple[torch.fx.GraphModule, torch.fx.GraphModule]
    ]
    | None,
) -> tuple[bool, Callable[..., tuple[torch.fx.GraphModule, torch.fx.GraphModule]]]:
    """
    Return either `default_partition_fn` or a HOP specific partition function.

    If a HOP specific partition function is returned, used_hop_custom_partition is True.

    See Note [InvokeSubgraphHOP Partitioner]
    """
    used_hop_custom_partition = False

    # Check for HOP-specific partition function first. This is needed because
    # run_joint_graph_passes_on_hops can be called without an outer stage-2
    # partition function (e.g., in aot_stage1_graph_capture before the remat
    # pass).
    if (
        fw_hop_node.target == torch._higher_order_ops.invoke_subgraph
        and "custom" in fw_hop_node.meta
        and "nested_region_config" in fw_hop_node.meta["custom"]
    ):
        hop_partition_fn = fw_hop_node.meta["custom"][
            "nested_region_config"
        ].partitioner
        if hop_partition_fn is not None:
            if callable(hop_partition_fn):
                raw_partitioner = hop_partition_fn
            elif not isinstance(hop_partition_fn, str):
                raise AssertionError(
                    f"expected hop_partition_fn to be str, got {type(hop_partition_fn)}"
                )
            else:
                match hop_partition_fn:
                    case "default_partition":
                        raw_partitioner = (
                            torch._functorch.partitioners.default_partition
                        )
                    case "min_cut_rematerialization_partition":
                        raw_partitioner = torch._functorch.partitioners.min_cut_rematerialization_partition
                    case _:
                        raise ValueError(
                            f"Unknown HOP partitioner config: {hop_partition_fn}"
                        )

            # Route through Inductor's `partition_fn` so joint-graph passes
            # (e.g. scatter_upon_const_tensor) run on the HOP subgraph before
            # the user-selected raw partitioner.
            from torch._inductor.compile_fx import (
                partition_fn as _inductor_partition_fn,
            )

            return True, functools.partial(
                _inductor_partition_fn, partitioner_fn_override=raw_partitioner
            )

    # Fall back to the parent partitioner from aot_config. When the outer
    # compile is Inductor this is already `compile_fx.partition_fn` so
    # joint-graph passes run there too.
    if default_partition_fn is None:
        raise AssertionError("default_partition_fn must not be None")
    return used_hop_custom_partition, default_partition_fn


def _has_invoke_subgraph_node(gm: torch.fx.GraphModule):
    from torch._higher_order_ops import invoke_subgraph

    for node in gm.graph.nodes:
        if node.op == "call_function" and node.target is invoke_subgraph:
            return True
    return False


def _scan_sac_tags(joint_hop_gm: torch.fx.GraphModule) -> tuple[bool, bool]:
    """Inspect the SAC recompute tags on a checkpoint region's forward nodes.

    A region with a context_fn policy has its forward nodes (and their
    decompositions) tagged node.meta["recompute"] by the policy's caching
    dispatch mode during joint tracing (see trace_joint_graph_as_bwd), exactly as
    the tag path / eager SAC do. Returns (has_save, has_cpu_offload):
    has_save   -- any op is MUST_SAVE/PREFER_SAVE (the region is genuinely
                  selective and the partitioner should honor the tags);
    has_cpu_offload -- any op requests CPU offload, which this path can't emit and
                  the caller rejects.
    """
    from torch.utils.checkpoint import CheckpointPolicy

    _has_tag_is_forward = torch._functorch.partitioners._has_tag_is_forward
    save = {CheckpointPolicy.MUST_SAVE, CheckpointPolicy.PREFER_SAVE}
    offload = {CheckpointPolicy.MUST_CPU_OFFLOAD, CheckpointPolicy.PREFER_CPU_OFFLOAD}
    has_save = has_cpu_offload = False
    for node in joint_hop_gm.graph.nodes:
        if not _has_tag_is_forward(node):
            continue
        policy = node.meta.get("recompute")
        if policy in save:
            has_save = True
        elif policy in offload:
            has_cpu_offload = True
    return has_save, has_cpu_offload


def _force_save_rng(joint_hop_gm: torch.fx.GraphModule) -> tuple[bool, bool]:
    """Force-save RNG ops in a checkpoint region so the backward doesn't re-draw.

    RNG can't be recomputed naively: re-running it on the backward draws fresh
    values (e.g. a different dropout mask), so the backward would use a different
    mask than the forward -- silently wrong gradients. The standard partitioner
    avoids this by functionalizing recomputed RNG (run_and_save_rng_state on the
    forward / run_with_rng_state on the backward, sharing the drawn state). That
    functionalization makes the forward and backward RNG ops structurally
    different, which the shared-slice mechanism can't express (it needs fw/bw to
    run the *same* GraphModule), so here we instead tag the RNG op (and its
    getitems, e.g. native_dropout's mask) MUST_SAVE: drawn once on the forward,
    saved, and consumed by the backward -- no re-draw, no drift. This runs before
    default_partition, whose own RNG functionalization then sees nothing to do.

    Collectives are intentionally left alone: recomputing a collective in an AC
    region is standard, correct behavior (deterministic and symmetric across
    ranks -- the expected communication cost of recompute), matching what the
    standard partitioner does for user-annotated AC regions.

    We only force-save *forward* RNG in the top-level joint graph (where
    default_partition operates and where a forward op would otherwise be
    recomputed); a backward RNG op is part of grad computation, not recomputed, so
    it is left alone. RNG inside a nested subgraph module would be recomputed
    atomically, so the caller rejects that. Returns (has_top_level_rng,
    has_nested_rng).
    """
    from torch._prims.rng_prims import (
        graphsafe_run_with_rng_state,
        run_and_save_rng_state,
        run_dtensor_rng_op,
        run_with_rng_state,
    )
    from torch.utils.checkpoint import CheckpointPolicy

    rng_hops = (
        run_and_save_rng_state,
        run_with_rng_state,
        graphsafe_run_with_rng_state,
        run_dtensor_rng_op,
    )
    is_rng_op = torch._functorch.partitioners.is_rng_op
    _has_tag_is_forward = torch._functorch.partitioners._has_tag_is_forward

    def _is_rng(n: torch.fx.Node) -> bool:
        return n.op == "call_function" and (is_rng_op(n) or n.target in rng_hops)

    has_top = False
    for node in joint_hop_gm.graph.nodes:
        if _is_rng(node) and _has_tag_is_forward(node):
            has_top = True
            node.meta["recompute"] = CheckpointPolicy.MUST_SAVE
            for user in node.users:
                if user.op == "call_function" and user.target is operator.getitem:
                    user.meta["recompute"] = CheckpointPolicy.MUST_SAVE

    has_nested = any(
        _is_rng(n)
        for m in joint_hop_gm.modules()
        if isinstance(m, torch.fx.GraphModule) and m is not joint_hop_gm
        for n in m.graph.nodes
    )
    return has_top, has_nested


def _clear_sac_tags(joint_hop_gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
    """Strip SAC recompute tags so the partitioner saves everything.

    Used for the whole-region recompute-as-call path (vanilla AC, a policy that
    saves nothing, or the shared-slice fallback): with the tags removed
    default_partition saves all backward-needed activations, which
    _rewrite_bw_hop_to_recompute_fw then turns into a re-invoke of the forward.
    """
    for node in joint_hop_gm.graph.nodes:
        node.meta.pop("recompute", None)
        node.meta.pop("ac_graph_id", None)
    return joint_hop_gm


def _rewrite_bw_hop_to_recompute_fw(
    new_bw_hop_gm: torch.fx.GraphModule,
    new_fw_hop_gm: torch.fx.GraphModule,
    fw_identifier: str,
    num_primals: int,
    num_fw_outputs: int,
    num_saved: int,
    num_sym: int,
    num_tangents: int,
) -> None:
    """Turn a save-partitioned region backward into a recompute-as-call one.

    The region is partitioned save-all, so new_fw_hop_gm outputs
    (*fw_outs, *saved, *sym_nodes) and new_bw_hop_gm takes
    (*sym_nodes, *saved, *tangents) -> (*grads), where `saved` is the
    non-symint saved values (num_saved of them) and `sym_nodes` are the symints
    (num_sym of them, always last). This pass relies on the forward's extra
    outputs and the backward's leading inputs sharing that relative ordering.

    We rewrite new_bw_hop_gm in place so it instead takes (*primals, *tangents)
    and re-invokes new_fw_hop_gm(*primals) to regenerate (*saved, *sym_nodes)
    internally. Because the forward subgraph is invoked from both the forward
    graph and inside this backward subgraph, the two lower from the same
    GraphModule (structurally identical kernels), so a recomputed reduction
    matches the original forward (gh-186572), while only primals -- not
    activations -- cross the outer fwd/bwd boundary.
    """
    from torch._higher_order_ops import invoke_subgraph

    g = new_bw_hop_gm.graph
    placeholders = g.find_nodes(op="placeholder")
    # The bw signature must be exactly (*sym, *saved, *tangents). Extra placeholder
    # classes (opaque objects, backward_state, bwd_seed_offset) would be silently
    # misclassified as tangents, so fail loudly instead. bwd_seed_offset can't
    # appear here: a region with RNG is force-saved and routed as selective, so it
    # never reaches this whole-region recompute-as-call path.
    if len(placeholders) != num_sym + num_saved + num_tangents:
        raise AssertionError(
            "recompute-as-call expected bw placeholders "
            f"= {num_sym} sym + {num_saved} saved + {num_tangents} tangents, got "
            f"{len(placeholders)} (unsupported placeholder class in the region)"
        )
    old_sym = placeholders[:num_sym]
    old_saved = placeholders[num_sym : num_sym + num_saved]

    fw_placeholders = new_fw_hop_gm.graph.find_nodes(op="placeholder")
    if len(fw_placeholders) != num_primals:
        raise AssertionError(
            f"expected {num_primals} fw placeholders, got {len(fw_placeholders)}"
        )
    fw_out_vals = [
        n.meta.get("val") if n is not None else None
        for n in new_fw_hop_gm.graph.find_nodes(op="output")[0].args[0]
    ]
    if len(fw_out_vals) != num_fw_outputs + num_saved + num_sym:
        raise AssertionError(
            "recompute-as-call expected fw outputs "
            f"= {num_fw_outputs} fw_outs + {num_saved} saved + {num_sym} sym, got "
            f"{len(fw_out_vals)}"
        )

    new_bw_hop_gm.add_module("recompute_fw_subgraph", new_fw_hop_gm)

    # New primal placeholders go at the very front (before existing placeholders).
    new_primals = []
    with g.inserting_before(placeholders[0]):
        for i in range(num_primals):
            p = g.placeholder(f"recompute_primal_{i}")
            p.meta.update(copy.copy(fw_placeholders[i].meta))
            new_primals.append(p)

    # The re-invoke and getitems must come after all placeholders.
    body_start = next(n for n in g.nodes if n.op != "placeholder")
    with g.inserting_before(body_start):
        get_attr_node = g.get_attr("recompute_fw_subgraph")
        fw_call = g.call_function(
            invoke_subgraph, args=(get_attr_node, fw_identifier, *new_primals)
        )
        fw_call.meta["val"] = tuple(fw_out_vals)
        saved_gis = []
        for j in range(num_saved):
            gi = g.call_function(operator.getitem, args=(fw_call, num_fw_outputs + j))
            gi.meta["val"] = fw_out_vals[num_fw_outputs + j]
            saved_gis.append(gi)
        sym_gis = []
        for k in range(num_sym):
            idx = num_fw_outputs + num_saved + k
            gi = g.call_function(operator.getitem, args=(fw_call, idx))
            gi.meta["val"] = fw_out_vals[idx]
            sym_gis.append(gi)

    for old, new in zip(old_sym, sym_gis):
        old.replace_all_uses_with(new)
    for old, new in zip(old_saved, saved_gis):
        old.replace_all_uses_with(new)
    for old in [*old_sym, *old_saved]:
        g.erase_node(old)

    g.eliminate_dead_code()
    g.lint()
    new_bw_hop_gm.recompile()


def _splice_shared_slice_call(
    parent_gm: torch.fx.GraphModule,
    slice_gm: torch.fx.GraphModule,
    identifier: str,
    slice_out_vals: list[Any],
    operands: list[torch.fx.Node],
    replace: dict[str, torch.fx.Node],
    slice_index: dict[str, int],
    erase_names: set[str],
    insert_after: torch.fx.Node,
) -> None:
    """Replace slice nodes in ``parent_gm`` with a call to the shared slice.

    Inserts ``invoke_subgraph(slice_gm, identifier, *operands)`` after
    ``insert_after``, swaps each node in ``replace`` for the corresponding getitem
    of the call's output tuple, and erases every node named in ``erase_names``
    (``replace`` plus multi-output op nodes whose getitems were replaced). Both
    the forward and backward call the same ``slice_gm``, so the recomputed ops
    lower from one GraphModule and cannot drift (gh-186572).
    """
    from torch._higher_order_ops import invoke_subgraph

    g = parent_gm.graph
    attr_name = f"{identifier}_mod"
    parent_gm.register_module(attr_name, slice_gm)
    with g.inserting_after(insert_after):
        get_attr_node = g.get_attr(attr_name)
    with g.inserting_after(get_attr_node):
        call = g.call_function(
            invoke_subgraph, args=(get_attr_node, identifier, *operands)
        )
        call.meta["val"] = tuple(slice_out_vals)
    anchor = call
    for name, tgt in replace.items():
        idx = slice_index[name]
        with g.inserting_after(anchor):
            gi = g.call_function(operator.getitem, args=(call, idx))
        gi.meta["val"] = slice_out_vals[idx]
        tgt.replace_all_uses_with(gi)
        anchor = gi
    # Erase in reverse topological order so no node is erased before its users.
    for n in reversed([n for n in g.nodes if n.name in erase_names]):
        g.erase_node(n)


def _extract_sac_shared_slice(
    new_fw_hop_gm: torch.fx.GraphModule,
    new_bw_hop_gm: torch.fx.GraphModule,
    num_fw_outputs: int,
    identifier: str,
) -> bool:
    """Share the recomputed forward slice of a selectively-checkpointed region.

    ``default_partition`` (honoring the SAC recompute tags) already produced a
    correct split: ``new_fw`` saves the MUST_SAVE activations and ``new_bw``
    recomputes the rest inline. But those two copies of the recomputed forward
    ops compile independently and can drift (gh-186572). This pass lifts the
    recomputed (transient, i.e. not-saved) forward nodes into one shared slice
    subgraph and makes both ``new_fw`` and ``new_bw`` invoke it, so the slice
    lowers from a single GraphModule in forward and backward.

    Returns True on success, False if the region isn't expressible as a single
    shared slice (e.g. a saved value depends on a recomputed one, or a slice
    input isn't available in the backward), in which case the caller falls back
    to whole-region recompute-as-call.
    """
    fw_graph = new_fw_hop_gm.graph
    bw_graph = new_bw_hop_gm.graph
    bw_by_name = {n.name: n for n in bw_graph.nodes}
    fw_out_node = fw_graph.find_nodes(op="output")[0]
    fw_out_vals = list(fw_out_node.args[0])
    saved_set = {n for n in fw_out_vals[num_fw_outputs:] if n is not None}

    # The shared slice is only the forward ops the backward actually recomputes:
    # transient (not-saved) forward nodes whose recomputed copy (matched by name)
    # appears in new_bw. Transient ops that only feed the region output (never
    # recomputed in the backward, e.g. the final elementwise of a norm) stay
    # inline in new_fw. Pulling them into the slice would make the backward
    # compute values it does not need and, worse, let Inductor reuse a saved
    # input's buffer for those dead outputs -- corrupting the saved tensor.
    bw_body_names = {
        n.name for n in bw_graph.nodes if n.op not in ("placeholder", "output")
    }
    # get_attr (constants/params) are left inline in both fw and bw rather than
    # lifted into the slice: they carry no drift risk and have no meta['val'] to
    # return through the slice's flat output tuple.
    s_nodes = [
        n
        for n in fw_graph.nodes
        if n.op not in ("placeholder", "output", "get_attr")
        and n not in saved_set
        and n.name in bw_body_names
    ]
    if not s_nodes:
        # Nothing the backward needs is recomputed, so there is no cross-graph
        # copy to drift. Leave the partition as-is.
        return True
    s_set = set(s_nodes)

    # Slice inputs: inputs to slice nodes from outside the slice (primals or saved
    # values), in first-seen order for a stable signature.
    s_inputs: list[torch.fx.Node] = []
    seen: set[torch.fx.Node] = set()
    for n in s_nodes:
        for inp in n.all_input_nodes:
            if inp not in s_set and inp not in seen:
                seen.add(inp)
                s_inputs.append(inp)

    # Slice inputs must be recoverable in new_bw: only primals (placeholders) and
    # saved values cross the fw/bw boundary. A get_attr (constant/param/nested
    # module ref) consumed by a recomputed node is neither, so it lands here and
    # forces a bail -- the region then falls back to whole-region recompute (or is
    # rejected if it has force-saved RNG). Lifting such get_attrs into the slice is
    # future work; it is rare (a recomputed op reading a lifted constant).
    for inp in s_inputs:
        if inp.op != "placeholder" and inp not in saved_set:
            return False

    # The slice becomes one node in new_fw. That is only valid if every slice
    # input is defined before every consumer of a slice output; otherwise the
    # contracted graph would be cyclic (a saved/region value interleaves with the
    # recomputed slice). Detect via original node order and bail if so.
    node_index = {n: i for i, n in enumerate(fw_graph.nodes)}
    consumers = [
        n
        for n in fw_graph.nodes
        if n not in s_set and any(inp in s_set for inp in n.all_input_nodes)
    ]
    if consumers:
        first_consumer_idx = min(node_index[c] for c in consumers)
        last_s_input_idx = max((node_index[i] for i in s_inputs), default=-1)
        if last_s_input_idx >= first_consumer_idx:
            return False

    # Multi-output ops (tuple/list-valued) are kept in the slice body but never
    # returned directly (invoke_subgraph outputs must be flat tensors/symints);
    # their getitems carry the values.
    def _is_tuple_val(n: torch.fx.Node) -> bool:
        return isinstance(n.meta.get("val"), (tuple, list))

    all_s_names = {n.name for n in s_nodes}
    for n in s_nodes:
        if not _is_tuple_val(n):
            continue
        # Forward: every getitem of a multi-output producer must be in the slice,
        # else one shared slice can't express the region.
        if any(u not in s_set for u in n.users):
            return False
        # Backward: the recomputed producer may have MORE getitem users than the
        # forward (e.g. native_layer_norm / SDPA expose mean/rstd that only the
        # op's own backward consumes). Those getitems are absent from the
        # fw-derived slice, so erasing the producer in the backward would leave
        # them dangling -- bail to whole-region recompute rather than crash.
        bw_n = bw_by_name.get(n.name)
        if bw_n is not None and any(u.name not in all_s_names for u in bw_n.users):
            return False

    # Values the slice returns: every slice node except the tuple producers.
    s_out_nodes = [n for n in s_nodes if not _is_tuple_val(n)]

    # Build the shared slice subgraph S: (s_inputs) -> (s_out_nodes).
    s_graph = torch.fx.Graph()
    env: dict[torch.fx.Node, torch.fx.Node] = {}
    for inp in s_inputs:
        p = s_graph.placeholder(inp.name)
        p.meta = copy.copy(inp.meta)
        env[inp] = p
    for n in fw_graph.nodes:  # iterate in topological order
        if n in s_set:
            env[n] = s_graph.node_copy(n, lambda x: env[x])
            env[n].meta = copy.copy(n.meta)
    s_graph.output(tuple(env[n] for n in s_out_nodes))
    s_graph.lint()
    slice_gm = torch.fx.GraphModule(new_fw_hop_gm, s_graph)
    slice_out_vals = [n.meta.get("val") for n in s_out_nodes]
    slice_index = {n.name: i for i, n in enumerate(s_out_nodes)}
    # Forward and backward call the *same* slice_gm object (so it lowers from one
    # GraphModule -> no drift) but under distinct identifiers. A shared identifier
    # would let Inductor's invoke_subgraph cache reuse the forward-specialized
    # compilation for the backward call, whose saved-tensor operand can have a
    # different layout -- silently corrupting the backward (mirrors AC
    # recompute-as-call, which likewise uses a distinct backward identifier).
    fw_slice_identifier = f"sac_slice_fw_{identifier}"
    bw_slice_identifier = f"sac_slice_bw_{identifier}"

    # Validate that the backward is expressible BEFORE mutating new_fw, so a bail
    # returns with both graphs untouched (the caller then re-partitions a pristine
    # copy for whole-region recompute). new_bw's recomputed slice nodes all follow
    # its placeholders, so the call is inserted right after the last placeholder;
    # its slice inputs must be the corresponding saved/primal placeholders.
    bw_operands: list[torch.fx.Node] = []
    for inp in s_inputs:
        m = bw_by_name.get(inp.name)
        # A slice input is a primal/saved value, which is a placeholder in the
        # backward. If the name instead resolves to a recomputed body node the
        # invariant is broken (the call would reference a not-yet-defined value);
        # bail to whole-region recompute rather than emit an invalid graph.
        if m is None or m.op != "placeholder":
            return False
        bw_operands.append(m)
    bw_replace = {
        name: bw_by_name[name]
        for name in slice_index
        if name in bw_by_name and bw_by_name[name].op != "placeholder"
    }
    bw_last_ph = None
    for n in bw_graph.nodes:
        if n.op == "placeholder":
            bw_last_ph = n
    if bw_replace and bw_last_ph is None:
        return False

    # Rewrite new_fw: replace the slice nodes with the shared-slice call, inserted
    # after the last slice input (all inputs defined, before any consumer). With
    # no external inputs (e.g. the slice is a recomputed factory like aten.ones),
    # insert right after the last placeholder so the call precedes all consumers.
    if s_inputs:
        last_input = max(s_inputs, key=lambda n: node_index[n])
    else:
        fw_placeholders = fw_graph.find_nodes(op="placeholder")
        last_input = (
            fw_placeholders[-1] if fw_placeholders else next(iter(fw_graph.nodes))
        )
    _splice_shared_slice_call(
        new_fw_hop_gm,
        slice_gm,
        fw_slice_identifier,
        slice_out_vals,
        operands=s_inputs,
        replace={n.name: n for n in s_out_nodes},
        slice_index=slice_index,
        erase_names=all_s_names,
        insert_after=last_input,
    )

    # Rewrite new_bw in place using the operands/placeholders validated above.
    if bw_replace:
        # bw_last_ph is non-None here (the bail above returns when bw_replace and
        # it is None); re-assert to narrow the type.
        if bw_last_ph is None:
            raise AssertionError("bw_last_ph must not be None")
        bw_erase = {
            name
            for name in all_s_names
            if name in bw_by_name and bw_by_name[name].op != "placeholder"
        }
        _splice_shared_slice_call(
            new_bw_hop_gm,
            slice_gm,
            bw_slice_identifier,
            slice_out_vals,
            operands=bw_operands,
            replace=bw_replace,
            slice_index=slice_index,
            erase_names=bw_erase,
            insert_after=bw_last_ph,
        )

    new_fw_hop_gm.graph.eliminate_dead_code()
    new_fw_hop_gm.graph.lint()
    new_fw_hop_gm.recompile()
    new_bw_hop_gm.graph.eliminate_dead_code()
    new_bw_hop_gm.graph.lint()
    new_bw_hop_gm.recompile()
    return True


def run_joint_graph_passes_on_hops(
    joint_gm: torch.fx.GraphModule,
    joint_inputs: Any,
    aot_config: AOTConfig,
    *,
    default_partition_fn: Callable[
        ..., tuple[torch.fx.GraphModule, torch.fx.GraphModule]
    ]
    | None = None,
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

    NB: This pass only matches **top-level** invoke_subgraph HOP nodes in
    `joint_gm`. It does not recurse into subgraph modules, so when one
    `nested_compile_region` is invoked from inside another, the inner
    invoke_subgraph HOPs live inside the outer's subgraph module and are not
    paired here. Top-level HOPs are still paired correctly via `call_id`;
    recursive partitioning of nested regions is left to downstream passes.
    """
    from torch._higher_order_ops import invoke_subgraph
    from torch._higher_order_ops.invoke_subgraph import (
        get_backward_nested_region_config,
    )

    # Identifiers of checkpoint regions rewritten to recompute-as-call, so the
    # restitch loop can pass primals+tangents (not saved tensors) to them.
    recompute_ac_ids: set[str] = set()

    def num_outputs(mod: torch.fx.GraphModule) -> int:
        return len(mod.graph.find_nodes(op="output")[0].args[0])

    def num_inputs(mod: torch.fx.GraphModule) -> int:
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
    # pyrefly: ignore [implicit-any]
    fw_hop_nodes = []
    # pyrefly: ignore [implicit-any]
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

    # The fw and bw HOP counts are not necessarily equal. A fw HOP can have no
    # corresponding bw HOP when autograd never runs backward through that call
    # — e.g. the user does `x_d = x.detach().requires_grad_()` between two
    # regions, or some outputs are not used in the loss. Likewise, multiple fw
    # calls in the same compile region can share a single bw if autograd dedups
    # or DCEs intermediate bws.
    #
    # Pair fw and bw HOPs by `call_id` — a per-call counter stamped by
    # InvokeSubgraphAutogradOp's fw/bw on each FX node. This is unambiguous
    # regardless of how autograd dispatched the backward (single outer
    # `.backward()`, per-iter `.backward()` in a loop, interleaved regions).
    fws_by_call_id: dict[int, torch.fx.Node] = {}
    for fw in fw_hop_nodes:
        cid = fw.meta.get("custom", {}).get("call_id")
        if cid is None:
            continue
        if cid in fws_by_call_id:
            raise AssertionError(
                f"duplicate call_id={cid} on fw HOPs "
                f"{fws_by_call_id[cid].name!r} and {fw.name!r}"
            )
        fws_by_call_id[cid] = fw

    bw_to_fw_hop_node: dict[torch.fx.Node, torch.fx.Node] = {}
    paired_fws: set[torch.fx.Node] = set()
    for bw in bw_hop_nodes:
        cid = bw.meta.get("custom", {}).get("call_id")
        if cid is None:
            raise AssertionError(
                f"bw HOP {bw.args[1]!r} has no call_id in meta['custom']"
            )
        fw = fws_by_call_id.get(cid)
        if fw is None:
            raise AssertionError(
                f"could not find matching fw HOP for bw {bw.args[1]!r} "
                f"with call_id={cid} (fw call_ids: {sorted(fws_by_call_id)})"
            )
        bw_to_fw_hop_node[bw] = fw
        paired_fws.add(fw)

    # Extra fws share a compile region with a paired bw but have no bw of
    # their own (autograd may dedup or DCE the bw). They still must be
    # rewritten to call the new partitioned fw subgraph, otherwise the output
    # signatures diverge from the rewritten paired fws. Key by the bw
    # identifier so the key matches `new_hop_graphs`.
    fw_args_to_bw_identifier: dict[str, str] = {}
    for bw, fw in bw_to_fw_hop_node.items():
        fw_arg = fw.args[1]
        bw_arg = bw.args[1]
        if not (isinstance(fw_arg, str) and isinstance(bw_arg, str)):
            raise AssertionError(
                f"expected fw/bw invoke_subgraph HOP args[1] to be str identifiers, "
                f"got fw={type(fw_arg)}, bw={type(bw_arg)}"
            )
        fw_args_to_bw_identifier.setdefault(fw_arg, bw_arg.removeprefix("bw"))
    extra_fws_by_id: dict[str, list[torch.fx.Node]] = defaultdict(list)
    for fw in fw_hop_nodes:
        if fw in paired_fws:
            continue
        bw_ident_key = fw_args_to_bw_identifier.get(fw.args[1])
        if bw_ident_key is None:
            continue
        extra_fws_by_id[bw_ident_key].append(fw)

    for node in bw_hop_nodes:
        identifier = node.args[1].removeprefix("bw")

        # If partitioning already done for this identifier, skip. This saves
        # redundant joint graph passes for same subgraphs.
        if new_hop_graphs[identifier].partitioning_done:
            continue

        # Collect some information from the forward hop graph
        fw_hop_node = bw_to_fw_hop_node[node]
        fw_subgraph_attr = fw_hop_node.args[0]
        if not isinstance(fw_subgraph_attr, torch.fx.Node):
            raise AssertionError(
                f"expected fw invoke_subgraph HOP args[0] to be torch.fx.Node, "
                f"got {type(fw_subgraph_attr)}"
            )
        fw_subgraph_attr_target = fw_subgraph_attr.target
        if not isinstance(fw_subgraph_attr_target, str):
            raise AssertionError(
                f"expected fw invoke_subgraph HOP args[0].target to be str, "
                f"got {type(fw_subgraph_attr_target)}"
            )
        fw_hop_gm = getattr(joint_gm, fw_subgraph_attr_target)
        if not isinstance(fw_hop_gm, torch.fx.GraphModule):
            raise AssertionError(
                f"expected fw_hop_gm to be GraphModule, got {type(fw_hop_gm)}"
            )
        num_fw_inputs = num_inputs(fw_hop_gm)
        num_fw_outputs = num_outputs(fw_hop_gm)
        new_hop_graphs[identifier].old_num_fw_inputs = num_fw_inputs
        new_hop_graphs[identifier].old_num_fw_outputs = num_fw_outputs

        # Only rewrite checkpoint regions (marked by the front-door and
        # propagated to node meta by invoke_subgraph) to recompute-as-call; leave
        # nested_compile_region and other invoke_subgraph HOPs -- including any
        # with their own partitioner config -- untouched.
        # A checkpoint region (the front-door marked it) always recomputes via a
        # shared-subgraph call. Recompute is coupled to the front-door rather than
        # a separate flag, so enabling checkpoint_via_invoke_subgraph can't
        # silently turn checkpoint into save-everything.
        region_recompute = fw_hop_node.meta.get("custom", {}).get(
            "_checkpoint_region", False
        )

        # Step 1) - Get the `joint_hop_gm`. As mentioned earlier, the
        # backward graph is the joint graph.
        joint_hop_gm = getattr(joint_gm, node.args[0].target)
        if not isinstance(joint_hop_gm, torch.fx.GraphModule):
            raise AssertionError(
                f"expected joint_hop_gm to be GraphModule, got {type(joint_hop_gm)}"
            )

        # Prepare the graph for the partitioner
        joint_hop_gm = prepare_for_partitioner(
            joint_hop_gm, num_fw_inputs, num_fw_outputs
        )

        # Selective activation checkpointing (SAC): a checkpoint region carrying a
        # context_fn policy already had its forward nodes tagged
        # node.meta["recompute"] by the policy's caching dispatch mode during
        # joint tracing (see trace_joint_graph_as_bwd) -- the same mechanism the
        # tag path and eager SAC use, so decompositions and HOPs are tagged too.
        # Here we just read those tags: the partitioner honors them (save the
        # MUST_SAVE ops, recompute the rest) and the recomputed slice is shared.
        # `pristine_joint_hop_gm` is a tag-cleared copy for the whole-region
        # recompute fallback (region not expressible as one shared slice).
        region_is_selective = False
        # Force-saved RNG can't be safely re-invoked, so a region containing it
        # must go through the save+shared-slice path, never the whole-region
        # recompute-as-call fallback.
        region_has_forced_save_rng = False
        pristine_joint_hop_gm: torch.fx.GraphModule | None = None
        if region_recompute:
            has_save, has_cpu_offload = _scan_sac_tags(joint_hop_gm)
            if has_cpu_offload:
                # CPU offload needs forward->CPU / backward->GPU transfers this
                # path doesn't emit; reject rather than silently saving on device.
                raise NotImplementedError(
                    "checkpoint_via_invoke_subgraph does not support CPU-offload "
                    "checkpoint policies (MUST_CPU_OFFLOAD / PREFER_CPU_OFFLOAD); "
                    "use MUST_SAVE / PREFER_RECOMPUTE or disable the flag."
                )
            # (Effectful ops are rejected earlier, on the Dynamo-traced region body
            # in CheckpointHigherOrderVariable, before the effectful op is lifted
            # out of the graph the partitioner sees.)
            # RNG can't be recomputed on the backward re-invoke (it would re-draw);
            # force-save it (run once) rather than banning, matching the standard
            # partitioner. RNG inside a nested subgraph can't be force-saved here.
            has_top_rng, has_nested_rng = _force_save_rng(joint_hop_gm)
            if has_nested_rng:
                raise RuntimeError(
                    "checkpoint_via_invoke_subgraph does not support RNG inside a "
                    f"nested subgraph of the region (invoke_subgraph "
                    f"{fw_hop_node.name}); move it out or disable the flag."
                )
            region_has_forced_save_rng = has_top_rng
            # A region is routed through save + shared-slice (rather than
            # whole-region recompute-as-call) when it selectively saves ops (a
            # policy) or contains force-saved RNG.
            region_is_selective = has_save or has_top_rng
            if region_is_selective:
                # Keep a tag-cleared copy for the shared-slice fallback.
                pristine_joint_hop_gm = _clear_sac_tags(copy.deepcopy(joint_hop_gm))
            else:
                # Vanilla AC: clear any recompute tags so default_partition saves
                # all and the whole-region recompute-as-call path takes over.
                _clear_sac_tags(joint_hop_gm)

        # TODO: invoke_subgraph should track which of its inputs static indices
        # so it can propagate them to the partitioner (and use in cudagraphs)
        static_lifetime_input_indices: list[int] = []

        if region_recompute:
            # Partition checkpoint regions with default_partition so it respects
            # the save/recompute tags (MUST_SAVE from a policy or force-saved RNG).
            # Route through Inductor's partition_fn so joint-graph passes run first.
            from torch._inductor.compile_fx import (
                partition_fn as _inductor_partition_fn,
            )

            used_hop_custom_partition = False
            partition_fn = functools.partial(
                _inductor_partition_fn,
                partitioner_fn_override=torch._functorch.partitioners.default_partition,
            )
        else:
            used_hop_custom_partition, partition_fn = _get_partition_fn(
                fw_hop_node, aot_config, default_partition_fn
            )

        # Step 2) and 3) - Run joint graph passes and partitioner
        try:
            new_fw_hop_gm, new_bw_hop_gm = partition_fn(
                joint_hop_gm,
                [],
                num_fwd_outputs=num_fw_outputs,
                static_lifetime_input_indices=static_lifetime_input_indices,
            )
        except Exception as e:
            if used_hop_custom_partition:
                raise RuntimeError(
                    f"Error in custom partition function for invoke_subgraph node {fw_hop_node.name}: {e}"
                ) from e
            else:
                raise

        def _sym_saved_counts(fw_gm: torch.fx.GraphModule) -> tuple[int, int]:
            outs = fw_gm.graph.find_nodes(op="output")[0].args[0]
            extra = outs[num_fw_outputs:]
            n_sym = len([n for n in extra if is_sym_node(n)])
            return n_sym, len(extra) - n_sym

        if region_is_selective:
            # The partition already saves the MUST_SAVE ops and recomputes the
            # rest. Share the recomputed slice between fw and bw so it lowers from
            # one GraphModule and can't drift. If the region can't be expressed as
            # a single shared slice, fall back to whole-region recompute on the
            # untagged joint.
            if not _extract_sac_shared_slice(
                new_fw_hop_gm, new_bw_hop_gm, num_fw_outputs, identifier
            ):
                # Whole-region recompute-as-call would re-invoke the whole forward,
                # re-drawing force-saved RNG -- unsafe. Such a region can't fall
                # back, so reject rather than silently corrupt.
                if region_has_forced_save_rng:
                    raise RuntimeError(
                        "checkpoint_via_invoke_subgraph: region with RNG cannot be "
                        f"expressed as a shared recompute slice (invoke_subgraph "
                        f"{fw_hop_node.name}); restructure it or disable the flag."
                    )
                region_is_selective = False
                if pristine_joint_hop_gm is None:
                    raise AssertionError("pristine_joint_hop_gm must not be None")
                new_fw_hop_gm, new_bw_hop_gm = partition_fn(
                    pristine_joint_hop_gm,
                    [],
                    num_fwd_outputs=num_fw_outputs,
                    static_lifetime_input_indices=static_lifetime_input_indices,
                )

        # Save the new forward and backward graph modules
        new_hop_graphs[identifier].new_fw_hop_gm = new_fw_hop_gm
        new_hop_graphs[identifier].new_bw_hop_gm = new_bw_hop_gm

        # Save the number of symints and saved tensors
        new_num_sym_nodes, new_num_saved_nodes = _sym_saved_counts(new_fw_hop_gm)
        new_hop_graphs[identifier].new_num_sym_nodes = new_num_sym_nodes
        new_hop_graphs[identifier].new_num_saved_nodes = new_num_saved_nodes

        if region_recompute and not region_is_selective:
            recompute_ac_ids.add(identifier)
            # Rewrite the backward to re-invoke the forward subgraph instead of
            # consuming its saved activations (see helper docstring). The bw HOP
            # node's args are (subgraph, identifier, *primals, *tangents), so the
            # tangent count is what remains after the primals.
            num_tangents = len(node.args) - 2 - num_fw_inputs
            _rewrite_bw_hop_to_recompute_fw(
                new_bw_hop_gm,
                new_fw_hop_gm,
                f"recompute_fw_{identifier}",
                num_fw_inputs,
                num_fw_outputs,
                new_num_saved_nodes,
                new_num_sym_nodes,
                num_tangents,
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
    # don't have to do anything.
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

    def add_new_hop_gm(new_subgraph_mod: torch.fx.GraphModule, name: str) -> str:
        new_subgraph_attr_name = f"partitioned_{name}"
        if new_subgraph_attr_name in already_added_new_hop_mods:
            return new_subgraph_attr_name

        joint_gm.register_module(new_subgraph_attr_name, new_subgraph_mod)
        already_added_new_hop_mods.add(new_subgraph_attr_name)
        return new_subgraph_attr_name

    def propagate_meta_info(
        new_hop_gm: torch.fx.GraphModule,
        new_call_function_node: torch.fx.Node,
        old_call_function_node: torch.fx.Node,
    ) -> None:
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
        if new_fw_hop_gm is None:
            raise AssertionError(
                f"new_fw_hop_gm for identifier {identifier} must not be None"
            )

        old_num_fw_outputs = new_hop_graphs[identifier].old_num_fw_outputs
        new_num_sym_nodes = new_hop_graphs[identifier].new_num_sym_nodes
        new_num_saved_nodes = new_hop_graphs[identifier].new_num_saved_nodes
        if old_num_fw_outputs is None:
            raise AssertionError(
                f"old_num_fw_outputs for identifier {identifier} must not be None"
            )
        if new_num_sym_nodes is None:
            raise AssertionError(
                f"new_num_sym_nodes for identifier {identifier} must not be None"
            )
        if new_num_saved_nodes is None:
            raise AssertionError(
                f"new_num_saved_nodes for identifier {identifier} must not be None"
            )
        total_outputs = old_num_fw_outputs + new_num_saved_nodes + new_num_sym_nodes

        extra_fw_outputs = []

        # Insert the new_fw_hop_gm into the joint_gm
        fw_subgraph_attr = fw_node.args[0]
        if not isinstance(fw_subgraph_attr, torch.fx.Node):
            raise AssertionError(
                f"expected fw invoke_subgraph HOP args[0] to be torch.fx.Node, "
                f"got {type(fw_subgraph_attr)}"
            )
        with joint_gm.graph.inserting_after(fw_node):
            new_fw_mod_attr_name = add_new_hop_gm(new_fw_hop_gm, f"fw{identifier}")
            new_fw_mod_attr = joint_gm.graph.get_attr(new_fw_mod_attr_name)
            new_fw_mod_attr.meta = copy.copy(fw_subgraph_attr.meta)

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
        if new_bw_hop_gm is None:
            raise AssertionError(
                f"new_bw_hop_gm for identifier {identifier} must not be None"
            )

        num_primals = new_hop_graphs[identifier].old_num_fw_inputs
        if num_primals is None:
            raise AssertionError(
                f"num_primals for identifier {identifier} must not be None"
            )

        if identifier in recompute_ac_ids:
            # new_bw_hop_gm was rewritten to take (*primals, *tangents) and
            # re-invoke the forward subgraph internally to regenerate its
            # activations -- so pass primals+tangents straight through (exactly
            # bw_node's original operands). The forward's saved-tensor getitems
            # (extra_fw_outputs) go unused here and are cleaned up by DCE.
            operands = list(bw_node.args[2:])
        else:
            saved_tensor_nodes = extra_fw_outputs[:new_num_saved_nodes]
            sym_nodes = extra_fw_outputs[new_num_saved_nodes:]
            tangents = list(bw_node.args[2 + num_primals :])
            operands = sym_nodes + saved_tensor_nodes + tangents

        # Insert the new_bw_hop_gm into the joint_gm
        with joint_gm.graph.inserting_after(bw_node):
            new_bw_mod_attr_name = add_new_hop_gm(new_bw_hop_gm, bw_node.args[1])
            new_bw_mod_attr = joint_gm.graph.get_attr(new_bw_mod_attr_name)
            new_bw_mod_attr.meta = copy.copy(bw_node.args[0].meta)

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

            # When the region sets backward-specific inductor config, compile the
            # partitioned backward under it; the forward keeps its own config.
            fw_region_config = fw_node.meta.get("custom", {}).get(
                "nested_region_config"
            )
            # get_backward_nested_region_config returns fw_config unchanged when
            # the region has no distinct backward config, so identity tells us
            # whether to stamp.
            bw_region_config = get_backward_nested_region_config(fw_region_config)
            if bw_region_config is not fw_region_config:
                # Re-stamp on the fresh backward node's meta["custom"] (the source
                # of truth: unlike a GraphModule's meta it survives FX transforms).
                # Lowering picks it up via the subgraph-module mirror (_propagate_*)
                # or the ir.py node fallback.
                new_bw_node.meta["custom"] = {
                    **new_bw_node.meta.get("custom", {}),
                    "nested_region_config": bw_region_config,
                }

        bw_node.replace_all_uses_with(new_bw_node)
        joint_gm.graph.erase_node(bw_node)

    # Rewrite extra (unpaired) fws to call the new partitioned fw subgraph.
    # Their additional saved-tensor outputs become dead and are pruned by
    # eliminate_dead_code.
    for identifier, extras in extra_fws_by_id.items():
        if not new_hop_graphs[identifier].partitioning_done:
            continue
        new_fw_hop_gm = new_hop_graphs[identifier].new_fw_hop_gm
        if new_fw_hop_gm is None:
            continue
        for fw_node in extras:
            fw_subgraph_attr = fw_node.args[0]
            if not isinstance(fw_subgraph_attr, torch.fx.Node):
                raise AssertionError(
                    f"expected fw invoke_subgraph HOP args[0] to be torch.fx.Node, "
                    f"got {type(fw_subgraph_attr)}"
                )
            with joint_gm.graph.inserting_after(fw_node):
                new_attr_name = add_new_hop_gm(new_fw_hop_gm, f"fw{identifier}")
                new_attr = joint_gm.graph.get_attr(new_attr_name)
                new_attr.meta = copy.copy(fw_subgraph_attr.meta)
            with joint_gm.graph.inserting_after(new_attr):
                new_fw_node = joint_gm.graph.call_function(
                    the_function=invoke_subgraph,
                    args=(new_attr, new_attr_name, *fw_node.args[2:]),
                )
                propagate_meta_info(new_fw_hop_gm, new_fw_node, fw_node)
            fw_node.replace_all_uses_with(new_fw_node)
            joint_gm.graph.erase_node(fw_node)

    joint_gm.graph.eliminate_dead_code()
    joint_gm.graph.lint()
    joint_gm.recompile()
    return joint_gm


def maybe_log_graph(
    gm: torch.fx.GraphModule,
    graph_name: str,
    aot_config: AOTConfig,
    structured_log_prefix_fn: Callable[[], str],
    out_structured_logs: list[str] | None = None,
) -> None:
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


def create_wrap_fn(
    fn: Callable[..., Any], args: tuple[Any, ...]
) -> tuple[Callable[..., Any], tuple[Any, ...]]:
    from torch.fx.experimental.proxy_tensor import maybe_enable_thunkify

    from .functional_utils import from_fun, has_data_mutation, to_fun

    def assert_no_mutation(t: Any) -> None:
        if has_data_mutation(t):
            raise AssertionError(
                "Saved tensors hooks with inputs mutations are not allowed"
            )

    @simple_wraps(fn)
    def _wrapper(*args: Any) -> Any:
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


def prepare_hook_gm(
    aot_config: AOTConfig, fn: Callable[..., Any], args: tuple[Any, ...]
) -> torch.fx.GraphModule:
    from torch._functorch._aot_autograd.graph_capture import _create_graph

    fn, args = create_wrap_fn(fn, args)
    gm = _create_graph(fn, args, aot_config=aot_config)  # type: ignore[arg-type]
    return gm


# Inline Autograd saved_tensors_hooks into epilogue of forward graph
# and prologue of backward graph.
# This changes forward graph outputs and inputs.
# Pack hook can return tensors, sym scalars, constants.
# All tensors to save for backward will be grouped together at front.
# Sym scalars grouped on another end. Constants are inlined in the graph.
def maybe_inline_graph_saved_tensors_hooks(
    fw_module: torch.fx.GraphModule,
    bw_module: torch.fx.GraphModule,
    num_inner_fwd_outputs: int,
    inner_meta: ViewAndMutationMeta,
    aot_config: AOTConfig,
    static_input_indices: list[int],
) -> None:
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

    def _gen_unused_name(candidate: str) -> str:
        c = candidate
        i = 0
        while c in fw_g_names or c in bw_g_names:
            c = f"{candidate}_{i}"
            i = i + 1
        return c

    bw_g_inputs = bw_g.find_nodes(op="placeholder")

    fw_out_n = fw_g.output_node()
    fw_outs = fw_out_n.args[0]  # type: ignore[var-annotated]
    fw_outs_inner_set = set(fw_outs[:num_inner_fwd_outputs])  # type: ignore[index]
    fw_outs_saved_for_bw = fw_outs[num_inner_fwd_outputs:]  # type: ignore[index]
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
            [n for n in fw_outs_saved_for_bw if is_sym_node(n)]  # type: ignore[arg-type]
        )
        # Count tensors with no version counter check (used in tensors_saved_for_backwards_slice)
        inner_meta.num_tensors_saved_with_no_vc_check = len(
            [
                n
                # pyrefly: ignore [not-iterable]
                for n in fw_outs_saved_for_bw
                if isinstance(n, torch.fx.Node)
                and n.meta.get("saved_tensor_with_no_vc_check", False)
            ]
        )
        bw_donated_idxs = collect_bw_donated_buffer_idxs(
            fw_module,
            bw_module,
            inner_meta,
        )
        fw_donated_idxs = [
            i - inner_meta.num_symints_saved_for_bw for i in bw_donated_idxs
        ]
        allow_set = {fw_outs_saved_for_bw[i].name for i in fw_donated_idxs}  # type: ignore[union-attr]
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

    # pyrefly: ignore [not-iterable]
    for saved in fw_outs_saved_for_bw:
        if ((allow_set is not None) and (saved.name not in allow_set)) or (  # type: ignore[union-attr]
            (exclude_set is not None) and (saved.name in exclude_set)  # type: ignore[union-attr]
        ):
            if isinstance(saved.meta["val"], torch.Tensor):  # type: ignore[union-attr]
                fw_outs_packed_tensors.append(saved)
            continue

        val = saved.meta["val"]  # type: ignore[union-attr]
        if not isinstance(val, torch.Tensor):
            continue

        def _get_extra_info() -> dict[str, Any]:
            return {"_fw_graph": fw_g, "_bw_graph": bw_g, "_node": saved}

        with _saved_tensor_hook_context(_get_extra_info()):
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

        with _saved_tensor_hook_context(_get_extra_info()):
            pack_gm = prepare_hook_gm(aot_config, pack_hook_gm, (val,))
            pack_g = pack_gm.graph
            maybe_log_graph(
                pack_gm,
                f"saved_tensors_pack_hook {saved.name}",  # type: ignore[union-attr]
                aot_config,
                lambda: f"aot_saved_tensors_hooks_pack {saved.name}",  # type: ignore[union-attr]
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
        if len(pack_g_inputs) != 1:
            raise AssertionError(
                f"expected exactly 1 pack_g_input, got {len(pack_g_inputs)}"
            )
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
        if not fw_pack_out_args:
            raise AssertionError("fw_pack_out_args must not be empty")
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
                new_node_name = _gen_unused_name(f"{saved.name}_hook_{out_idx}")  # type: ignore[union-attr]
                with fw_g.inserting_before(_n):
                    n = fw_g.create_node(
                        _n.op,
                        _n.target,
                        _n.args,
                        _n.kwargs,
                        name=new_node_name,
                    )
                if n.name != new_node_name:
                    raise AssertionError(
                        f"expected n.name == {new_node_name}, got {n.name}"
                    )
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
        with _saved_tensor_hook_context(_get_extra_info()):
            unpack_gm = prepare_hook_gm(aot_config, unpack_hook_gm, (pack_out_val,))
            unpack_g = unpack_gm.graph
            maybe_log_graph(
                unpack_gm,
                f"saved_tensors_unpack_hook {saved.name}",  # type: ignore[union-attr]
                aot_config,
                lambda: f"aot_saved_tensors_hooks_unpack {saved.name}",  # type: ignore[union-attr]
                structured_logs,
            )

        def find_saved_in_bw_inputs(
            bw_inputs: list[torch.fx.Node],
        ) -> torch.fx.Node | None:
            for n in bw_inputs:
                if n.name == saved.name:  # type: ignore[union-attr]
                    return n

        bw_g_input = find_saved_in_bw_inputs(bw_g_inputs)
        if not bw_g_input:
            raise AssertionError(
                f"could not find saved tensor {saved.name} in bw_g_inputs"  # type: ignore[union-attr]
            )
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
                        if new_n.name != new_node_name:
                            raise AssertionError(
                                f"expected new_n.name == {new_node_name}, got {new_n.name}"
                            )
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

        if not bw_unpack_out_n:
            raise AssertionError("bw_unpack_out_n must not be None")
        _leaves = pytree.tree_leaves(bw_unpack_out_n.args)
        if len(_leaves) != 1:
            raise AssertionError(f"expected exactly 1 leaf, got {len(_leaves)}")
        unpack_saved_tensor_n = _leaves[0]

        if not bw_g_input_used_directly:
            bw_g_input.replace_all_uses_with(unpack_saved_tensor_n)
            bw_g.erase_node(bw_g_input)
        else:
            # Keep usages of bw_g_input in inserted unpacked hook graph.
            # Replace other usages of bw_g_input with unpack_saved_tensor_n.
            for use_node in original_bw_g_input_users:
                use_node._replace_input_with(bw_g_input, unpack_saved_tensor_n)
        bw_g.erase_node(bw_unpack_out_n)

    # Changing forward graph outputs,
    # Inserting packed_tensors and packed_syms on the place of saved tensors.
    # Packed sym_scalars are together with saved symints
    symint_outs_saved_for_bw = [n for n in fw_outs_saved_for_bw if is_sym_node(n)]  # type: ignore[arg-type]
    fw_new_outs = pytree.tree_leaves(
        (
            fw_outs[:num_inner_fwd_outputs],  # type: ignore[index]
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

    def _log_structured_logs() -> None:
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
            f"fw_outs[:num_inner_fwd_outputs]:{fw_outs[:num_inner_fwd_outputs]}"  # type: ignore[index]
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

    if fw_t_names != bw_t_names:
        raise AssertionError(
            f"expected fw_t_names == bw_t_names, got {fw_t_names} != {bw_t_names}"
        )
    if fw_s_names != bw_s_names:
        raise AssertionError(
            f"expected fw_s_names == bw_s_names, got {fw_s_names} != {bw_s_names}"
        )

    fw_g.lint()
    bw_g.lint()
    fw_module.recompile()
    bw_module.recompile()


def _log_joint_graph(
    fx_g: torch.fx.GraphModule,
    aot_config: AOTConfig,
) -> str | None:
    """
    Log the joint graph to the structured logger.
    Return a str representation of the graph.
    """
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
    return joint_graph_str


def _log_fw_bw_graphs(
    fw_module: torch.fx.GraphModule,
    bw_module: torch.fx.GraphModule,
    maybe_subclass_meta: SubclassMeta | None,
    fw_metadata: ViewAndMutationMeta,
    aot_config: AOTConfig,
) -> tuple[str | None, str | None]:
    """
    Log the fw and bw graphs to the structured logger.
    Return str representations of the graphs.
    """
    fw_module_str = None
    bw_module_str = None
    if aot_config.enable_log:
        trace_structured(
            "artifact",
            metadata_fn=lambda: {
                "name": "torch._functorch.config",
                "encoding": "string",
            },
            payload_fn=lambda: torch._functorch.config.get_serializable_config_copy(),
        )
        aot_graphs_log.info(
            "aot_config id: %s, fw_metadata=%s, inner_meta=%s",
            aot_config.aot_id,
            fw_metadata,
            _get_inner_meta(maybe_subclass_meta, fw_metadata),
        )

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
    return fw_module_str, bw_module_str


def _partition_joint_graph_into_fw_bw(
    fx_g: torch.fx.GraphModule,
    joint_inputs: list[Any] | tuple[list[Any], list[Any]],
    inner_meta: ViewAndMutationMeta,
    fw_metadata: ViewAndMutationMeta,
    aot_config: AOTConfig,
    # pyrefly: ignore [implicit-any]
    partition_fn: Callable,
) -> tuple[torch.fx.GraphModule, torch.fx.GraphModule, int]:
    # See Note: [Partitioner handling for Subclasses, Part 1]
    # See Note: [Recomputing subclass mutation handling]
    mutated_inp_runtime_indices = compute_inner_mutated_inp_indices_from_subclass_meta(
        fw_metadata, inner_meta
    )
    num_tokens = len(fw_metadata.tokens)
    num_inner_fwd_outputs = (
        len(mutated_inp_runtime_indices)
        + inner_meta.num_outputs
        + inner_meta.num_intermediate_bases
        + inner_meta.num_outputs_rng_offset
        + num_tokens  # See Note [Side-Effectful Tokens in AOTAutograd]
    )

    fx_g = run_joint_graph_passes_on_hops(
        fx_g, joint_inputs, aot_config, default_partition_fn=partition_fn
    )

    # apply joint_gm callback here
    if callable(torch._functorch.config.joint_custom_pass):
        # pyrefly: ignore [bad-assignment]
        fx_g = torch._functorch.config.joint_custom_pass(fx_g, joint_inputs)

    fw_module, bw_module = partition_fn(
        fx_g,
        joint_inputs,
        num_fwd_outputs=num_inner_fwd_outputs,
        static_lifetime_input_indices=fw_metadata.static_input_indices,
    )

    rng_states = [
        n
        for n in fw_module.graph.find_nodes(op="placeholder")
        if "fwd_rng_state" in n.name
    ]
    fw_metadata.num_graphsafe_rng_states = len(rng_states)
    if rng_states:
        rng_device = rng_states[0].meta["val"].device
        fw_metadata.graphsafe_rng_state_index = rng_device.index
        fw_metadata.graphsafe_rng_device = rng_device

    return fw_module, bw_module, num_inner_fwd_outputs


def _joint_inputs_for_forward(
    joint_inputs: list[Any] | tuple[list[Any], list[Any]],
) -> list[Any]:
    return joint_inputs[0] if isinstance(joint_inputs, tuple) else joint_inputs


def _maybe_unlift_partitioned_effect_tokens(
    fw_module: torch.fx.GraphModule,
    bw_module: torch.fx.GraphModule,
    joint_inputs: list[Any] | tuple[list[Any], list[Any]],
    fw_metadata: ViewAndMutationMeta,
    aot_config: AOTConfig,
    num_inner_fwd_outputs: int,
) -> tuple[int, list[Any] | tuple[list[Any], list[Any]]]:
    num_tokens = len(fw_metadata.tokens)

    # See Note [Side-Effectful Tokens in AOTAutograd]
    if config.unlift_effect_tokens and (
        num_tokens > 0 or fw_metadata.num_backward_tokens > 0
    ):
        unlift_tokens(fw_module, fw_metadata, aot_config, bw_module)
        num_inner_fwd_outputs -= num_tokens
        if isinstance(joint_inputs, tuple):
            joint_inputs = (
                _joint_inputs_for_forward(joint_inputs)[num_tokens:],
                joint_inputs[1],
            )
        else:
            joint_inputs = joint_inputs[num_tokens:]

    return num_inner_fwd_outputs, joint_inputs


def _categorize_saved_tensors_for_backward(
    fw_module: torch.fx.GraphModule,
    bw_module: torch.fx.GraphModule,
    inner_meta: ViewAndMutationMeta,
    fw_metadata: ViewAndMutationMeta,
    num_inner_fwd_outputs: int,
) -> tuple[int, int]:
    fw_outs = next(iter(fw_module.graph.find_nodes(op="output"))).args[0]
    # we only need to bookkeep the symints that are saved for bw, not any symints
    # the user forward might have returned in its own output
    fw_outs_saved_for_bw = fw_outs[num_inner_fwd_outputs:]
    num_fw_outs_saved_for_bw = len(fw_outs_saved_for_bw)

    num_symints_saved_for_bw = 0
    num_opaque_objects_saved_for_bw = 0
    saved_tensor_is_graph_input: list[bool] = []
    for idx, node in enumerate(fw_outs_saved_for_bw):
        if is_sym_node(node):
            num_symints_saved_for_bw += 1
        elif is_opaque_node(node):
            num_opaque_objects_saved_for_bw += 1
        elif isinstance(node, torch.fx.Node) and "val" in getattr(node, "meta", {}):
            if is_fake_tensor(node.meta["val"]):
                # If the saved_tensor is a view, a graph intermediate,
                # and returned from the autograd.Function output, we need to
                # detach() it to prevent a reference cycle. Record
                # if the saved_tensor is a graph input here to help.
                saved_tensor_is_graph_input.append(node.op == "placeholder")
                # record dynamic tensor activations
                dynamic_dims: set[int] = {
                    dim
                    for dim, size in enumerate(node.meta["val"].shape)
                    if not isinstance(size, int)
                }
                if dynamic_dims:
                    fw_metadata.dynamic_saved_tensors_idxs[idx] = dynamic_dims
            elif isinstance(node.meta["val"], (FakeScriptObject, CustomClassBase)):
                num_opaque_objects_saved_for_bw += 1
        else:
            saved_tensor_is_graph_input.append(False)

    fw_metadata.num_symints_saved_for_bw = num_symints_saved_for_bw
    fw_metadata.num_opaque_objects_saved_for_bw = num_opaque_objects_saved_for_bw
    num_tensors_saved_for_bw = (
        num_fw_outs_saved_for_bw
        - num_symints_saved_for_bw
        - num_opaque_objects_saved_for_bw
    )
    if len(saved_tensor_is_graph_input) != num_tensors_saved_for_bw:
        raise AssertionError(
            "expected one saved_tensor_is_graph_input entry per saved tensor, "
            f"got {len(saved_tensor_is_graph_input)} != {num_tensors_saved_for_bw}"
        )
    fw_metadata.saved_tensor_is_graph_input = saved_tensor_is_graph_input
    inner_meta.num_symints_saved_for_bw = num_symints_saved_for_bw
    inner_meta.num_opaque_objects_saved_for_bw = num_opaque_objects_saved_for_bw
    inner_meta.saved_tensor_is_graph_input = saved_tensor_is_graph_input

    # See Note [Activations with no version counter checks in eager]
    # Count tensors saved with no version counter check.
    # These are tensors that were stashed on ctx (e.g., ctx.x = x) rather than
    # via save_for_backward in an autograd.Function.
    # The partitioner sorts these to be at the end of saved_values.
    num_tensors_saved_with_no_vc_check = sum(
        1
        for node in fw_outs_saved_for_bw
        if isinstance(node, torch.fx.Node)
        and node.meta.get("saved_tensor_with_no_vc_check", False)
    )
    fw_metadata.num_tensors_saved_with_no_vc_check = num_tensors_saved_with_no_vc_check
    inner_meta.num_tensors_saved_with_no_vc_check = num_tensors_saved_with_no_vc_check

    if torch._functorch.config.donated_buffer:
        fw_metadata.bw_donated_idxs = collect_bw_donated_buffer_idxs(
            fw_module,
            bw_module,
            inner_meta,
        )
        inner_meta.bw_donated_idxs = fw_metadata.bw_donated_idxs

    return num_fw_outs_saved_for_bw, num_symints_saved_for_bw


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
# It's possible to construct a case where eager may or may not have tried to autograd through y,
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
def _compute_indices_of_inps_to_detach(
    bw_module: torch.fx.GraphModule,
    maybe_subclass_meta: SubclassMeta | None,
    inner_meta: ViewAndMutationMeta,
    fw_metadata: ViewAndMutationMeta,
) -> list[int]:
    # TODO: we should apply the below "detach inputs if their gradients are statically known to be None"
    # optimization even if we have subclass inputs/outputs (we do not handle this today).
    # Computing which our our inputs get None gradients is a bit more complicated,
    # if any of our inputs are subclasses. Why?
    # (a) we need to make sure that we call .detach() on the input subclasses, since autograd sees subclasses.
    # (b) The grad_outputs that we AOT computed in our backward graph are the desugared tensor tensors,
    #     so we need to figure out which subclass fw inputs they map to.
    if maybe_subclass_meta is not None:
        return []

    indices_of_inps_to_detach: list[int] = []

    # reversed() since we expect output at end of graph
    bw_output = next(reversed(bw_module.graph.find_nodes(op="output")))
    bw_outs = bw_output.args[0]

    num_backward_tokens = inner_meta.num_backward_tokens
    expected_bw_outs = (
        len(fw_metadata.input_info)
        + inner_meta.num_outputs_rng_offset
        + num_backward_tokens
    )
    if len(bw_outs) != expected_bw_outs:
        raise AssertionError(
            f"expected len(bw_outs) == {expected_bw_outs}, got {len(bw_outs)}"
        )

    bw_outs_no_rng_no_tokens = bw_outs
    if (inner_meta.num_outputs_rng_offset + num_backward_tokens) > 0:
        bw_outs_no_rng_no_tokens = bw_outs[
            : -(inner_meta.num_outputs_rng_offset + num_backward_tokens)
        ]
    if len(bw_outs_no_rng_no_tokens) != len(fw_metadata.input_info):
        raise AssertionError(
            f"expected len(bw_outs_no_rng_no_tokens) == {len(fw_metadata.input_info)}, "
            f"got {len(bw_outs_no_rng_no_tokens)}"
        )

    for i, bw_out in enumerate(bw_outs_no_rng_no_tokens):
        # If our input experiences a metadata mutation inside the graph (e.g. set_()),
        # we *must* not detach, otherwise it will be the detach'd input that gets the metadata mutation
        metadata_mutation_in_graph = (
            fw_metadata.input_info[i].mutation_type == MutationType.MUTATED_IN_GRAPH
            and fw_metadata.input_info[i].mutates_storage_metadata
        )
        is_non_leaf = (
            fw_metadata.input_info[i].requires_grad
            and not fw_metadata.input_info[i].is_leaf
        )
        if bw_out is None and not metadata_mutation_in_graph and is_non_leaf:
            indices_of_inps_to_detach.append(i)

    return indices_of_inps_to_detach


def _aot_stage2a_partition(
    fx_g: torch.fx.GraphModule,
    joint_inputs: list[Any] | tuple[list[Any], list[Any]],
    maybe_subclass_meta: SubclassMeta | None,
    fw_metadata: ViewAndMutationMeta,
    aot_config: AOTConfig,
    # pyrefly: ignore [implicit-any]
    partition_fn: Callable,
) -> tuple[torch.fx.GraphModule, torch.fx.GraphModule, int, int, list[int], list[Any]]:
    """
    Partition the joint graph into a forward graph and a backward graph. Returns:
    - the forward and backward graphs
    - the number of forward outputs and the number of symints saved for backward
    - indices of inputs to detach
    - adjusted inputs to forward
    """
    disable_amp = torch._C._is_any_autocast_enabled()
    inner_meta = _get_inner_meta(maybe_subclass_meta, fw_metadata)

    with torch.no_grad():
        context = torch._C._DisableAutocast if disable_amp else nullcontext
        with context(), track_graph_compiling(aot_config, "joint"):
            fw_module, bw_module, num_inner_fwd_outputs = (
                _partition_joint_graph_into_fw_bw(
                    fx_g,
                    joint_inputs,
                    inner_meta,
                    fw_metadata,
                    aot_config,
                    partition_fn,
                )
            )
            num_inner_fwd_outputs, joint_inputs = (
                _maybe_unlift_partitioned_effect_tokens(
                    fw_module,
                    bw_module,
                    joint_inputs,
                    fw_metadata,
                    aot_config,
                    num_inner_fwd_outputs,
                )
            )

            maybe_inline_graph_saved_tensors_hooks(
                fw_module,
                bw_module,
                num_inner_fwd_outputs,
                inner_meta,
                aot_config,
                fw_metadata.static_input_indices,
            )
            num_fw_outs_saved_for_bw, num_symints_saved_for_bw = (
                _categorize_saved_tensors_for_backward(
                    fw_module,
                    bw_module,
                    inner_meta,
                    fw_metadata,
                    num_inner_fwd_outputs,
                )
            )

        _indices_of_inps_to_detach = _compute_indices_of_inps_to_detach(
            bw_module,
            maybe_subclass_meta,
            inner_meta,
            fw_metadata,
        )

    return (
        fw_module,
        bw_module,
        num_fw_outs_saved_for_bw,
        num_symints_saved_for_bw,
        _indices_of_inps_to_detach,
        _joint_inputs_for_forward(joint_inputs),
    )


def _aot_stage2b_fw_compile(
    fw_module: torch.fx.GraphModule,
    adjusted_flat_args: list[Any],
    maybe_subclass_meta: SubclassMeta | None,
    fw_metadata: ViewAndMutationMeta,
    num_fw_outs_saved_for_bw: int,
    aot_config: AOTConfig,
    # pyrefly: ignore [implicit-any]
    fw_compiler: Callable,
    # pyrefly: ignore [implicit-any]
) -> tuple[list[tuple[int, ...] | None] | None, Callable]:
    return _aot_stage2b_compile_forward_or_inference(
        fw_module,
        adjusted_flat_args,
        maybe_subclass_meta,
        fw_metadata,
        aot_config,
        fw_compiler,
        is_inference=False,
        num_fw_outs_saved_for_bw=num_fw_outs_saved_for_bw,
    )


def _aot_stage2b_bw_compile(
    bw_module: torch.fx.GraphModule,
    maybe_subclass_meta: SubclassMeta | None,
    fw_metadata: ViewAndMutationMeta,
    fwd_output_strides: list[tuple[int, ...] | None] | None,
    num_symints_saved_for_bw: int,
    aot_config: AOTConfig,
    # pyrefly: ignore [implicit-any]
    bw_compiler: Callable,
    # pyrefly: ignore [implicit-any]
) -> tuple[AutogradLazyBackwardCompileInfo, Callable | None]:
    """
    Compile the backward graph. Returns:
    - the placeholder list for the backward graph
    - the compiled backward function
    """
    with torch.no_grad():
        # NB: It's important to compile backwards ahead of time, as this may
        # add extra guards which we need to apply to the Dynamo cache at
        # forwards
        with track_graph_compiling(aot_config, "backward"), torch._C._DisableAutocast():
            placeholder_list = fx_placeholder_vals(bw_module)

            forward_saved_for_backwards_strides = None
            if fwd_output_strides is not None:
                inner_meta = _get_inner_meta(maybe_subclass_meta, fw_metadata)
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
                # value. Using suppress_guards and guard_or_true to avoid that.

                stride_different = False
                fake_mode = detect_fake_mode()
                suppress_ctx = (
                    fake_mode.shape_env.suppress_guards()
                    if fake_mode is not None and fake_mode.shape_env is not None
                    else nullcontext()
                )

                # Inductor can choose different strides for activations than
                # what backward graph has. if we can't statically tell that
                # strides are the same, we assume they are not.
                with suppress_ctx:
                    for k in range(len(ph_arg.stride())):
                        # real_stride can't be symbolic.

                        if guard_or_true(ph_arg.stride()[k] != int(real_stride[k])):
                            stride_different = True
                            break

                if stride_different:
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

                    ph_size = ph_arg.size()

                    placeholder_list[i] = ph_arg.as_strided(ph_size, real_stride)
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
                    compiled_bw_func = bw_compiler(bw_module_copy, placeholder_list)
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

            lazy_backward_info = AutogradLazyBackwardCompileInfo(
                # pyrefly: ignore [bad-argument-type]
                bw_module,
                placeholder_list,
                saved_context,
                saved_compile_context,
            )

            return lazy_backward_info, compiled_bw_func


def aot_stage2_autograd(
    aot_state: AOTState,
    aot_graph_capture: AOTGraphCapture,
    # pyrefly: ignore [implicit-any]
    partition_fn: Callable,
    # pyrefly: ignore [implicit-any]
    fw_compiler: Callable,
    # pyrefly: ignore [implicit-any]
    bw_compiler: Callable,
) -> DispatchReturn:
    """
    Autograd logic. Generates a joint graph, partitions it, manipulates the input with various wrappers,
    and returns a wrapped torch.autograd.Function with a forward and backward.
    """

    fx_g = aot_graph_capture.graph_module
    maybe_subclass_meta = aot_graph_capture.maybe_subclass_meta
    fw_metadata = aot_state.fw_metadata
    aot_config = aot_state.aot_config

    CompileEventLogger.try_add_pt2_compile("backend_compile", dispatch_mode="autograd")
    joint_graph_str = _log_joint_graph(fx_g, aot_config)

    _apply_tensorify_python_scalars(fx_g)

    (
        fw_module,
        bw_module,
        num_fw_outs_saved_for_bw,
        num_symints_saved_for_bw,
        _indices_of_inps_to_detach,
        adjusted_flat_args,
    ) = _aot_stage2a_partition(
        fx_g,
        aot_graph_capture.updated_flat_args,
        maybe_subclass_meta,
        fw_metadata,
        aot_config,
        partition_fn,
    )

    min_cut_info_str = getattr(fx_g.graph, "_min_cut_info_str", None)

    fw_module_str, bw_module_str = _log_fw_bw_graphs(
        fw_module, bw_module, maybe_subclass_meta, fw_metadata, aot_config
    )

    fwd_output_strides, compiled_fw_func = _aot_stage2b_fw_compile(
        fw_module,
        adjusted_flat_args,
        maybe_subclass_meta,
        fw_metadata,
        num_fw_outs_saved_for_bw,
        aot_config,
        fw_compiler,
    )

    lazy_backward_info, compiled_bw_func = _aot_stage2b_bw_compile(
        bw_module,
        maybe_subclass_meta,
        fw_metadata,
        fwd_output_strides,
        num_symints_saved_for_bw,
        aot_config,
        bw_compiler,
    )

    try_save_cache_entry, entry = _cache_autograd_info(
        aot_config,
        aot_state.flat_args,
        compiled_fw_func,
        compiled_bw_func,
        fw_module_str,
        bw_module_str,
        joint_graph_str,
        aot_graph_capture.wrappers,
        maybe_subclass_meta,
        fw_metadata,
        num_fw_outs_saved_for_bw,
        _indices_of_inps_to_detach,
        num_symints_saved_for_bw,
        bw_module,
        min_cut_info_str,
    )

    return _aot_stage2c_make_autograd_function(
        aot_config,
        aot_state.flat_args,
        fw_metadata,
        maybe_subclass_meta,
        aot_graph_capture.wrappers,
        compiled_fw_func,
        compiled_bw_func,
        bw_compiler,
        lazy_backward_info,
        try_save_cache_entry,  # type: ignore[arg-type]
        entry,  # type: ignore[arg-type]
        _indices_of_inps_to_detach,
        num_symints_saved_for_bw,
    )


def _aot_stage2c_make_autograd_function(
    aot_config: AOTConfig,
    flat_args: list[Any],
    fw_metadata: ViewAndMutationMeta,
    maybe_subclass_meta: SubclassMeta | None,
    wrappers: list[CompilerWrapper],
    compiled_fw_func: Callable[..., Any],
    compiled_bw_func: Callable[..., Any] | None,
    # pyrefly: ignore [implicit-any]
    bw_compiler: Callable,
    lazy_backward_info: AutogradLazyBackwardCompileInfo | None,
    try_save_cache_entry: Callable[..., Any],
    entry: GenericAOTAutogradResult[Any, Any] | None,
    _indices_of_inps_to_detach: list[int],
    num_symints_saved_for_bw: int,
) -> DispatchReturn:
    backward_state_indices = [
        idx for idx, x in enumerate(flat_args) if isinstance(x, BackwardState)
    ]
    if len(backward_state_indices) > 1:
        raise AssertionError(
            f"expected at most 1 backward_state_index, got {len(backward_state_indices)}"
        )

    disable_amp = torch._C._is_any_autocast_enabled()
    compile_spec = AOTDispatchAutogradCompileSpec(
        compiled_fw_func=compiled_fw_func,
        compiled_bw_func=compiled_bw_func,
        maybe_subclass_meta=maybe_subclass_meta,
        num_symints_saved_for_bw=num_symints_saved_for_bw,
        backward_state_indices=backward_state_indices,
        disable_amp=disable_amp,
        indices_of_inps_to_detach=_indices_of_inps_to_detach,
        lazy_backward_info=lazy_backward_info,
        bw_compiler=bw_compiler,
        aot_config=aot_config,
        fw_metadata=fw_metadata,
        try_save_cache_entry=try_save_cache_entry,
    )
    compiled_fn = AOTDispatchAutograd.post_compile(compile_spec)

    if entry is not None:
        compiled_fn = SerializableCompiledFunction(compiled_fn, lambda: entry)

    if config.debug_assert:
        flat_requires_grad: list[bool | None] = [
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


def _cache_autograd_info(
    aot_config: AOTConfig,
    flat_args: list[Any],
    compiled_fw_func: Callable[..., Any],
    compiled_bw_func: Callable[..., Any] | None,
    fw_module_str: str | None,
    bw_module_str: str | None,
    joint_graph_str: str | None,
    wrappers: list[CompilerWrapper],
    maybe_subclass_meta: SubclassMeta | None,
    fw_metadata: ViewAndMutationMeta,
    num_fw_outs_saved_for_bw: int,
    _indices_of_inps_to_detach: list[int],
    num_symints_saved_for_bw: int,
    bw_module: torch.fx.GraphModule | None,
    min_cut_info_str: str | None,
) -> tuple[
    GenericAOTAutogradResult[Any, Any] | None,
    Callable[..., Any],
]:
    backward_state_indices = [
        idx for idx, x in enumerate(flat_args) if isinstance(x, BackwardState)
    ]
    if len(backward_state_indices) > 1:
        raise AssertionError(
            f"expected at most 1 backward_state_index, got {len(backward_state_indices)}"
        )

    make_runtime_safe(fw_metadata, maybe_subclass_meta)

    try_save_cache_entry: Callable[..., Any] | None = None
    entry: GenericAOTAutogradResult[Any, Any] | None = None

    if aot_config.cache_info is not None:
        forward_time_taken_ns = time.time_ns() - aot_config.cache_info.start_time_ns

        # NB: aot_config here is technically not needed as an argument: we could just
        # close over aot_config.cache_info, since aot_config never changes.
        # But closing over random variables is confusing IMO, so I'm leaving it.
        def try_save_cache_entry(
            compiled_bw_func: Callable[..., Any],
            bw_module: torch.fx.GraphModule,
            _fw_metadata: ViewAndMutationMeta,
            aot_config: AOTConfig,
        ) -> GenericAOTAutogradResult[Any, Any] | None:
            cache_info = aot_config.cache_info

            if cache_info is not None and _should_save_cache(
                compiled_fw_func, compiled_bw_func
            ):
                if forward_time_taken_ns is None:
                    raise AssertionError("forward_time_taken_ns must not be None")
                # TODO: technically, AOTAutograd does a *little* bit of post processing work
                # in the backward that isn't measured here. But it's small enough that it's not worth
                # the complexity of threading a bunch of times through the code, so we
                # use the compiled_bw_func's inductor compile time instead.
                # It's possible this changes in the future, in which case we should
                # update backward_time_taken_ns to be more inclusive
                backward_time_taken_ns = getattr(compiled_bw_func, "_time_taken_ns", 0)

                aot_forward_graph_str: str | None = fw_module_str
                aot_backward_graph_str: str | None = bw_module_str
                aot_joint_graph_str: str | None = joint_graph_str
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
                    sanitized_aot_config=aot_config.to_cacheable(),
                    guards_expr=guards_expr,
                    backward_state_indices=backward_state_indices,
                    num_symints_saved_for_bw=num_symints_saved_for_bw,
                    serialized_bw_module=serialize_graph_module(bw_module),
                    min_cut_info_str=min_cut_info_str,
                )
                AOTAutogradCache.save(
                    cache_info.cache_key,
                    entry,
                    remote=should_use_remote_autograd_cache(),
                )
                return entry
            return None

        if compiled_bw_func is not None:
            # If we already compiled the backward, we save its cache entry now
            if bw_module is None:
                raise AssertionError(
                    "bw_module must not be None when compiled_bw_func is not None"
                )
            entry = try_save_cache_entry(
                compiled_bw_func,
                bw_module,
                fw_metadata,
                aot_config,  # type: ignore[arg-type]
            )
            try_save_cache_entry = None

    return try_save_cache_entry, entry  # type: ignore[return-value]


def _aot_stage2b_compile_forward_or_inference(
    fw_module: torch.fx.GraphModule,
    adjusted_flat_args: list[Any],
    maybe_subclass_meta: SubclassMeta | None,
    fw_metadata: ViewAndMutationMeta,
    aot_config: AOTConfig,
    # pyrefly: ignore [implicit-any]
    compiler: Callable,
    *,
    is_inference: bool,
    num_fw_outs_saved_for_bw: int | None = None,
    # pyrefly: ignore [implicit-any]
) -> tuple[list[tuple[int, ...] | None] | None, Callable]:
    """
    Compile the forward or inference graph. Returns:
    - the output strides of the forward graph
    - the compiled forward/inference function

    Args:
        fw_module: The forward graph module to compile
        adjusted_flat_args: Flattened arguments after adjustments
        maybe_subclass_meta: Metadata for tensor subclasses
        fw_metadata: View and mutation metadata
        aot_config: AOT configuration
        is_inference: If True, compile for inference; if False, compile for forward (autograd)
        num_fw_outs_saved_for_bw: Number of forward outputs saved for backward (required if not is_inference)

    Before compiling, we run pre_compile for the following wrappers:
    - FakifiedOutWrapper
    - FunctionalizedRngRuntimeWrapper
    After compiling, we run post_compile for the following wrappers:
    - EffectTokensWrapper
    - AOTDispatchSubclassWrapper
    - FunctionalizedRngRuntimeWrapper
    - FakifiedOutWrapper
    """

    # Validation
    if not is_inference and num_fw_outs_saved_for_bw is None:
        raise ValueError(
            "num_fw_outs_saved_for_bw must be provided when is_inference=False"
        )

    # Determine grad context, autocast context, and tracking mode.
    if is_inference:
        grad_ctx: Any = nullcontext
        autocast_ctx: Any = (
            torch._C._DisableAutocast
            if torch._C._is_any_autocast_enabled()
            else nullcontext
        )
        tracking_mode: str = "inference"
    else:
        grad_ctx = torch.no_grad
        autocast_ctx = torch._C._DisableAutocast
        tracking_mode = "forward"

    with grad_ctx(), autocast_ctx(), track_graph_compiling(aot_config, tracking_mode):
        # Setup wrappers
        fakified_out_wrapper = FakifiedOutWrapper()
        fakified_out_wrapper.pre_compile(
            fw_module, adjusted_flat_args, aot_config, fw_metadata=fw_metadata
        )

        # Initialize RNG wrapper based on mode
        functionalized_rng_wrapper = FunctionalizedRngRuntimeWrapper(
            return_new_outs=is_inference
        )

        # Add RNG states for forward mode only
        if not is_inference and fw_metadata.num_graphsafe_rng_states > 0:
            index = fw_metadata.graphsafe_rng_state_index
            if index is None:
                raise AssertionError(
                    "fw_metadata.graphsafe_rng_state_index must not be None when num_graphsafe_rng_states > 0"
                )
            device = fw_metadata.graphsafe_rng_device
            if device is None:
                raise AssertionError(
                    "fw_metadata.graphsafe_rng_device must not be None when num_graphsafe_rng_states > 0"
                )
            rng_states = [
                get_default_generator(device).clone_state()
                for _ in range(fw_metadata.num_graphsafe_rng_states)
            ]
            adjusted_flat_args.extend(rng_states)  # type: ignore[arg-type]

        functionalized_rng_wrapper.pre_compile(
            fw_module, adjusted_flat_args, aot_config, fw_metadata=fw_metadata
        )

        # Set tracing context
        if tracing_context := torch._guards.TracingContext.try_get():
            tracing_context.fw_metadata = _get_inner_meta(
                maybe_subclass_meta, fw_metadata
            )

        if config.enable_complex_wrapper:
            from .complex_decomposition import decompose_complex_in_graph

            fw_module = decompose_complex_in_graph(
                fw_module, adjusted_flat_args, aot_config.decompositions
            )

        with TracingContext.report_output_strides() as fwd_output_strides:
            # pyrefly: ignore[not-callable]
            compiled_fw_func = compiler(fw_module, adjusted_flat_args)

        # Make boxed if needed
        if not getattr(compiled_fw_func, "_boxed_call", False):
            compiled_fw_func = make_boxed_func(compiled_fw_func)

        # Set forward output strides if needed
        if fakified_out_wrapper.needs_post_compile:
            fakified_out_wrapper.set_fwd_output_strides(fwd_output_strides)  # type: ignore[arg-type]

        # Apply post-compile wrappers
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
            aot_config,
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

        return fwd_output_strides, compiled_fw_func
