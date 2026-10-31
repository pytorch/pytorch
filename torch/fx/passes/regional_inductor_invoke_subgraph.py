import copy
import logging
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from torch._higher_order_ops.invoke_subgraph import invoke_subgraph_inductor_compile
from torch._inductor.standalone_compile import AOTCompiledArtifact
from torch.compiler._cache import CacheArtifactManager
from torch.fx._compatibility import compatibility
from torch.fx.passes.regional_inductor import (
    _disable_remat_for_regional_subcompile,
    _dummy_wrapper,
)


logger = logging.getLogger(__name__)

__all__ = ["regional_inductor_invoke_subgraph"]


# Identity shared by one graph's forward and backward compile: the compile id
# (the lazy backward re-enters the forward's CompileContext) plus the AOT graph
# scope (which keeps graphs a single compile splits, e.g. DDPOptimizer, apart).
_RegionalScopeKey = tuple[object, tuple[str, ...] | None]


_AOT_GRAPH_DIRECTIONS = ("forward", "backward", "inference")


@dataclass(frozen=True)
class _RegionCompileContext:
    """Everything the regions of one graph compile against."""

    # None when this compile cannot be paired with its other direction, in which
    # case each region keeps its CUDA Graph state private.
    scope_key: _RegionalScopeKey | None
    # The graph's direction from AOTAutograd's graph name; None when unknown,
    # in which case each node's partitioner tag decides.
    direction: str | None


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


def _get_cudagraph_pairing_name(identifier: str, is_backward: bool) -> str | None:
    """Name shared by a region's forward and backward invoke_subgraph calls.

    None for a call that has no counterpart in the other direction, e.g. a
    forward call recomputed in the backward.
    """
    # A first-level region is partitioned into partitioned_fw<X> and
    # partitioned_bw<X>. A region nested in an uncompiled one is not partitioned
    # and keeps InvokeSubgraphAutogradOp's identifiers, fw_<id> and bw_<id>_<n>,
    # where <n> numbers the backward variants of one forward.
    prefixes = ("partitioned_bw", "bw_") if is_backward else ("partitioned_fw", "fw_")
    if identifier.startswith(prefixes[0]):
        return identifier.removeprefix(prefixes[0])
    if identifier.startswith(prefixes[1]):
        name = identifier.removeprefix(prefixes[1])
        return name.rpartition("_")[0] if is_backward else name
    return None


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
    compile_fn = _get_region_compiler(sub_node, region_ctx.direction)
    if compile_fn is None:
        raise AssertionError("sub_node does not need inductor compile")
    compile_config = sub_node.meta["custom"]["nested_region_config"]
    partitioner_tag = sub_node.meta.get("partitioner_tag")
    identifier = sub_node.args[1]
    if not isinstance(identifier, str):
        raise AssertionError(f"Expected str identifier, got {type(identifier)}")
    uses_backward_config = _uses_backward_compiler(sub_node, region_ctx.direction)
    if region_ctx.direction is None:
        is_inference = partitioner_tag is None
        compiles_as_backward = uses_backward_config
    else:
        is_inference = region_ctx.direction == "inference"
        compiles_as_backward = region_ctx.direction == "backward"
    # A region pairs with its other direction only when it runs in the graph of
    # its own direction: not a forward call recomputed in the backward, nor a
    # backward traced into a forward or inference graph by torch.autograd.grad.
    pairing_name = (
        _get_cudagraph_pairing_name(identifier, uses_backward_config)
        if uses_backward_config == compiles_as_backward and not is_inference
        else None
    )
    cudagraph_state_key = (
        None
        if region_ctx.scope_key is None or pairing_name is None
        else (*region_ctx.scope_key, pairing_name)
    )
    # Only the built-in region compiler takes the regional CUDA Graph context; a
    # custom compiler keeps its (gm, example_inputs) call.
    region_kwargs: dict[str, object] = {}
    if getattr(compile_fn, "func", compile_fn) is invoke_subgraph_inductor_compile:
        region_kwargs = {
            # Inductor compiles a region in its graph's direction, whichever
            # config it uses.
            "is_backward": compiles_as_backward and not is_inference,
            "is_inference": is_inference,
            "cudagraph_state_key": cudagraph_state_key,
        }

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

            compiled_fn = compile_fn(gm, fake_inputs, **region_kwargs)
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


def _uses_backward_compiler(node: torch.fx.Node, direction: str | None) -> bool:
    if direction is None:
        return node.meta.get("partitioner_tag") == "is_backward"
    # Everything in a backward graph runs as part of the backward, including a
    # forward call recomputed there. Elsewhere a backward region can only come
    # from a traced torch.autograd.grad, and only its identifier says so.
    identifier = node.args[1]
    return direction == "backward" or (
        isinstance(identifier, str) and identifier.startswith(("partitioned_bw", "bw_"))
    )


def _get_region_compiler(
    node: torch.fx.Node, direction: str | None
) -> Callable[..., Any] | None:
    nested_config = node.meta.get("custom", {}).get("nested_region_config")
    if nested_config is None:
        return None
    if _uses_backward_compiler(node, direction):
        return nested_config.bw_compiler
    return nested_config.fw_compiler


def _compile_invoke_subgraph_nodes_with_inductor(
    gm: torch.fx.GraphModule,
    region_ctx: _RegionCompileContext,
) -> torch.fx.GraphModule:
    # Keyed in graph order, so regions compile in a deterministic order.
    map_subgraph_to_nodes: dict[str, list[torch.fx.Node]] = defaultdict(list)

    for node in gm.graph.find_nodes(
        op="call_function", target=torch.ops.higher_order.invoke_subgraph
    ):
        if _get_region_compiler(node, region_ctx.direction) is None:
            continue
        if node.args[0].op != "get_attr":
            raise AssertionError(f"Expected get_attr, got {node.args[0].op}")
        subgraph_name = node.args[0].target
        if not isinstance(subgraph_name, str):
            raise AssertionError(f"Expected str, got {type(subgraph_name)}")
        map_subgraph_to_nodes[subgraph_name].append(node)

    for subgraph, subgraph_users in map_subgraph_to_nodes.items():
        gm = _compile_submod(gm, subgraph, subgraph_users, region_ctx)

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
    from torch._inductor import config
    from torch._inductor.compile_fx import _validate_nested_region_cudagraphs

    # AOTAutograd names the graph with its direction. Fall back to the node tags
    # when it did not.
    _, direction = _get_aot_graph_scope_and_direction()
    is_backward = (
        _has_backward_tagged_node(gm) if direction is None else direction == "backward"
    )
    # A region nested in an uncompiled one is compiled on its own here, so only
    # compiled regions enclose the regions inside them.
    _validate_nested_region_cudagraphs(
        gm,
        enclosing_forward=config.triton.cudagraphs,
        enclosing_backward=config.triton.cudagraphs,
        validate_forward=not is_backward,
        validate_backward=is_backward,
        region_encloses=lambda node: _get_region_compiler(node, direction) is not None,
    )
    region_ctx = _RegionCompileContext(
        scope_key=_get_regional_scope_key(), direction=direction
    )

    # fuser utils create new nodes using create_proxy which retains the seq_nr
    # metadata and cause issues
    with torch.fx.traceback.preserve_node_meta(enable=False):
        compiled_gm = _recursive_compile_invoke_subgraph_nodes(gm, region_ctx)
        # TODO: might not need this boxed_nop after we switch to _RegionCompiler
        # pyrefly: ignore [bad-return]
        return torch._dynamo.backends.debugging.boxed_nop(
            compiled_gm, example_inputs=[]
        )
