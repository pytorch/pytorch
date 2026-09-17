import copy
import logging
import threading
import weakref
from collections import defaultdict
from dataclasses import dataclass

import torch
from torch._inductor.standalone_compile import AOTCompiledArtifact
from torch.compiler._cache import CacheArtifactManager
from torch.fx._compatibility import compatibility
from torch.fx.passes.regional_inductor import (
    _disable_remat_for_regional_subcompile,
    _dummy_wrapper,
)


logger = logging.getLogger(__name__)

__all__ = ["regional_inductor_invoke_subgraph"]


class _RegionalCudagraphState:
    def __init__(self) -> None:
        # Devices a forward region was CUDA Graph captured on, and the subset of
        # those whose backward region is captured too.
        self.device_indices: set[int] = set()
        self.backward_device_indices: set[int] = set()


# Identity shared by one graph's forward and backward compile: the compile id
# (the lazy backward re-enters the forward's CompileContext, so this pairs them
# even without a TracingContext) plus the AOT graph scope (which keeps graphs a
# single compile splits, e.g. DDPOptimizer, apart). Values are weak: the strong
# reference is _RegionCompileContext, pinned on the compiled forward callable
# below, so an entry lives exactly as long as the forward it belongs to.
_RegionalScopeKey = tuple[object, tuple[str, ...] | None]

_regional_cudagraph_states: weakref.WeakValueDictionary[
    _RegionalScopeKey, _RegionalCudagraphState
] = weakref.WeakValueDictionary()
_regional_cudagraph_states_lock = threading.Lock()


_AOT_GRAPH_DIRECTIONS = ("forward", "backward", "inference")


@dataclass(frozen=True)
class _RegionCompileContext:
    """Everything the regions of one graph compile against."""

    # Forward/backward CUDA Graph handoff, shared by both directions.
    cudagraph_state: _RegionalCudagraphState
    # None when this compile cannot be paired with its other direction, in which
    # case the state is private and must not be published to the registry.
    scope_key: _RegionalScopeKey | None
    # None means the direction is unknown; fall back to the node's own tag.
    is_inference: bool | None


def _get_aot_graph_scope_and_direction() -> tuple[tuple[str, ...] | None, str | None]:
    """Scope shared by a graph's forward/backward pair, plus which one this is."""
    context = torch._guards.TracingContext.try_get()
    if context is None or not context.aot_graph_name:
        return None, None

    graph_name = list(context.aot_graph_name)
    for direction in _AOT_GRAPH_DIRECTIONS:
        if graph_name[-1].endswith(f"_{direction}"):
            graph_name[-1] = graph_name[-1].removesuffix(f"_{direction}")
            return tuple(graph_name), direction
    return tuple(graph_name), None


def _has_backward_tagged_node(gm: torch.fx.GraphModule) -> bool:
    return any(
        node.meta.get("partitioner_tag") == "is_backward"
        for module in gm.modules()
        if isinstance(module, torch.fx.GraphModule)
        for node in module.graph.nodes
    )


def _get_regional_scope_key() -> _RegionalScopeKey | None:
    compile_id = torch._guards.CompileContext.current_compile_id()
    scope, _ = _get_aot_graph_scope_and_direction()
    if compile_id is None and scope is None:
        # Nothing identifies this compile, so a shared key would collide across
        # unrelated live models. Stay unpaired instead.
        logger.debug(
            "no compile id or AOT graph name; regional cudagraph state is unpaired"
        )
        return None
    return (compile_id, scope)


def _get_regional_cudagraph_state(
    key: _RegionalScopeKey | None,
) -> _RegionalCudagraphState:
    if key is None:
        return _RegionalCudagraphState()
    with _regional_cudagraph_states_lock:
        state = _regional_cudagraph_states.get(key)
        if state is None:
            state = _RegionalCudagraphState()
            _regional_cudagraph_states[key] = state
        return state


def _compile_submod(
    gm: torch.fx.GraphModule,
    subgraph: str,
    subgraph_users: list[torch.fx.Node],
    region_ctx: _RegionCompileContext,
) -> torch.fx.GraphModule:
    """
    Compiles subgraph submodule in gm. subgraph is used by subgraph_users.
    subgraph_users must all be  torch.ops.higher_order.invoke_subgraph HOP.
    """

    submod = getattr(gm, subgraph)

    compile_config = None
    fake_inputs = []

    # We use the first user for compile configs and inputs
    sub_node = subgraph_users[0]
    if not _needs_inductor_compile(sub_node):
        raise AssertionError("sub_node does not need inductor compile")
    compile_config = sub_node.meta["custom"]["nested_region_config"]
    partitioner_tag = sub_node.meta.get("partitioner_tag")
    # The partitioner names a region's two halves partitioned_fw_X/partitioned_bw_X,
    # so stripping the prefix is what pairs them.
    cudagraph_state_subgraph = subgraph
    if partitioner_tag == "is_backward":
        compile_fn = compile_config.bw_compiler
        cudagraph_state_subgraph = subgraph.removeprefix("partitioned_bw")
    else:
        compile_fn = compile_config.fw_compiler
        if partitioner_tag == "is_forward":
            cudagraph_state_subgraph = subgraph.removeprefix("partitioned_fw")
    cudagraph_state_key = (
        None
        if region_ctx.scope_key is None
        else (*region_ctx.scope_key, cudagraph_state_subgraph)
    )

    for inp_node in sub_node.all_input_nodes[
        1:
    ]:  # exclude the graph module input to torch.ops.higher_order.invoke_subgraph
        if hasattr(inp_node, "meta") and "val" in inp_node.meta:
            fake_inputs.append(inp_node.meta["val"])
        else:
            raise RuntimeError(
                f"Partition is bad because non fake tensor value is seen {inp_node}"
            )

    # Log the options being used
    logger.info(
        "Compiling submodule %s with inductor options: %s",
        subgraph,
        compile_config,
    )

    def get_compiled_fn() -> AOTCompiledArtifact:
        context = torch._guards.TracingContext.get()
        if context.fake_mode is None:
            raise AssertionError("context.fake_mode is None")

        context = torch._guards.TracingContext(context.fake_mode)

        with (
            torch._guards.tracing(context),
            CacheArtifactManager.with_fresh_cache(),
            torch._functorch.config.patch("bundled_autograd_cache", True),
            _disable_remat_for_regional_subcompile(),
            torch.fx.traceback._set_regional_inductor_subgraph_name(subgraph),
        ):
            # compile_fx can mutate gm
            gm = copy.deepcopy(submod)

            compiled_fn = compile_fn(
                gm,
                fake_inputs,
                is_inference=(
                    partitioner_tag is None
                    if region_ctx.is_inference is None
                    else region_ctx.is_inference
                ),
                cudagraph_state_key=cudagraph_state_key,
                regional_cudagraph_state=region_ctx.cudagraph_state,
            )
            return compiled_fn

    compiled_fn = get_compiled_fn()
    if not isinstance(compiled_fn, AOTCompiledArtifact):
        raise AssertionError(f"Expected AOTCompiledArtifact, got {type(compiled_fn)}")

    # _dummy_wrapper is to make call_function happy
    compiled_submod = _dummy_wrapper(compiled_fn)
    for node in subgraph_users:
        with gm.graph.inserting_after(node):
            new_node = gm.graph.call_function(
                # exclude graph nodes input args
                compiled_submod,
                args=node.args[2:],
                kwargs=node.kwargs,
            )
            new_node.meta = node.meta
            node.replace_all_uses_with(new_node)
            gm.graph.erase_node(node)

    gm.recompile()
    return gm


def _needs_inductor_compile(node: torch.fx.Node) -> bool:
    # TODO: maybe we could change to check
    # node.meta.get("partitioner_tag") != "is_forward"
    # if the tag is relibable
    return bool(
        (
            node.op not in ("placeholder", "output")
            and hasattr(node, "meta")
            and node.meta.get("custom", None)
            and node.meta["custom"].get("nested_region_config", None)
            and node.meta["custom"]["nested_region_config"].fw_compiler
            and node.meta.get("partitioner_tag") != "is_backward"
        )
        or (
            node.op not in ("placeholder", "output")
            and hasattr(node, "meta")
            and node.meta.get("custom", None)
            and node.meta["custom"].get("nested_region_config", None)
            and node.meta["custom"]["nested_region_config"].bw_compiler
            and node.meta.get("partitioner_tag") == "is_backward"
        )
    )


def _compile_invoke_subgraph_nodes_with_inductor(
    gm: torch.fx.GraphModule,
    region_ctx: _RegionCompileContext,
) -> torch.fx.GraphModule:
    map_subgraph_to_nodes = defaultdict(list)
    subgraphs: set[str] = set()

    for node in gm.graph.find_nodes(
        op="call_function", target=torch.ops.higher_order.invoke_subgraph
    ):
        if not _needs_inductor_compile(node):
            continue
        if node.args[0].op != "get_attr":
            raise AssertionError(f"Expected get_attr, got {node.args[0].op}")
        subgraph_name = node.args[0].target
        if not isinstance(subgraph_name, str):
            raise AssertionError(f"Expected str, got {type(subgraph_name)}")
        subgraphs.add(subgraph_name)
        map_subgraph_to_nodes[subgraph_name].append(node)

    for subgraph in subgraphs:
        gm = _compile_submod(gm, subgraph, map_subgraph_to_nodes[subgraph], region_ctx)

    return gm


def _recursive_compile_invoke_subgraph_nodes(
    gm: torch.fx.GraphModule,
    region_ctx: _RegionCompileContext,
) -> torch.fx.GraphModule:
    for node in gm.graph.find_nodes(op="get_attr"):
        if _needs_inductor_compile(node):
            # If the get_attr itself is marked for compile, the outer graph will
            # take care of it. If we don't do that, we end up with nested
            # regional inductor compiles that do not work well.
            continue
        submod = getattr(gm, node.target)
        if isinstance(submod, torch.fx.GraphModule):
            _recursive_compile_invoke_subgraph_nodes(submod, region_ctx)

    return _compile_invoke_subgraph_nodes_with_inductor(gm, region_ctx)


@compatibility(is_backward_compatible=False)
def regional_inductor_invoke_subgraph(
    gm: torch.fx.GraphModule, *example_args: object
) -> torch.fx.GraphModule:
    """
    Compile invoke_subgraph nodes if they have custom compiler specified
    in node.meta["nested_region_config"].bw_compiler or fw_compiler
    """
    from torch._functorch._aot_autograd.utils import simple_wraps
    from torch._inductor import config
    from torch._inductor.compile_fx import _validate_nested_region_cudagraphs

    # AOTAutograd tells us the direction directly. Fall back to the node tags
    # only when there is no tracing context (aot_module_simplified without
    # Dynamo); an inference graph can carry is_backward-tagged regions from a
    # traced torch.autograd.grad call, so the tags alone are not conclusive.
    _, direction = _get_aot_graph_scope_and_direction()
    if direction is None:
        is_backward = _has_backward_tagged_node(gm)
        is_inference = None
    else:
        is_backward = direction == "backward"
        is_inference = direction == "inference"
    _validate_nested_region_cudagraphs(
        gm,
        enclosing_forward=config.triton.cudagraphs,
        enclosing_backward=config.triton.cudagraphs,
        validate_forward=not is_backward,
        validate_backward=is_backward,
    )
    scope_key = _get_regional_scope_key()
    region_ctx = _RegionCompileContext(
        cudagraph_state=_get_regional_cudagraph_state(scope_key),
        scope_key=scope_key,
        is_inference=is_inference,
    )
    regional_cudagraph_state = region_ctx.cudagraph_state

    # fuser utils create new nodes using create_proxy which retains the seq_nr
    # metadata and cause issues
    with torch.fx.traceback.preserve_node_meta(enable=False):
        compiled_gm = _recursive_compile_invoke_subgraph_nodes(gm, region_ctx)
        # TODO: might not need this boxed_nop after we switch to _RegionCompiler
        compiled_fn = torch._dynamo.backends.debugging.boxed_nop(
            compiled_gm, example_inputs=[]
        )
        if not is_backward:
            # Strong reference that keeps this compile's entry in the weak
            # registry alive until the matching backward compiles.
            compiled_fn._regional_cudagraph_state = (  # pyrefly: ignore [missing-attribute]
                region_ctx
            )
            # pyrefly: ignore [bad-return]
            return compiled_fn

        # A device whose backward region is captured transitions CUDA Graph Trees
        # from inside its own node. Doing it here as well would clear
        # running_forwards_with_pending_backwards before that node runs, letting
        # it start a new generation and free the forward pool holding the saved
        # activations it is about to read.
        transition_devices = (
            regional_cudagraph_state.device_indices
            - regional_cudagraph_state.backward_device_indices
        )
        if not transition_devices or not config.triton.cudagraph_trees:
            # pyrefly: ignore [bad-return]
            return compiled_fn

        @simple_wraps(compiled_fn)
        def backward(args: list[object]) -> object:
            from torch._inductor.cudagraph_trees import get_manager

            for device_index in transition_devices:
                manager = get_manager(device_index, create_if_none_exists=False)
                if manager is not None:
                    manager.set_to_running_backward()
            return compiled_fn(args)

        backward._boxed_call = True  # pyrefly: ignore [missing-attribute]
        # pyrefly: ignore [bad-return]
        return backward
