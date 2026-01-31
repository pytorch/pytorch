# mypy: allow-untyped-defs

import copy
import logging
from collections import defaultdict

import torch
from torch._inductor.standalone_compile import AOTCompiledArtifact
from torch.compiler._cache import CacheArtifactManager
from torch.fx._compatibility import compatibility
from torch.fx.passes.regional_inductor import _dummy_wrapper


logger = logging.getLogger(__name__)

__all__ = ["regional_inductor_invoke_subgraph"]


def _compile_submod(
    gm: torch.fx.GraphModule, subgraph: str, subgraph_users: list[torch.fx.Node]
):
    """
    Compiles subgraph submodule in gm. subgraph is used by subgraph_users.
    subgraph_users must all be  torch.ops.higher_order.invoke_subgraph HOP.
    """

    submod = getattr(gm, subgraph)

    compile_config = None
    fake_inputs = []

    # We use the first user for compile configs and inputs
    sub_node = subgraph_users[0]
    assert _needs_inductor_compile(sub_node)
    compile_config = sub_node.meta["custom"]["nested_region_config"]
    if sub_node.meta.get("partitioner_tag") == "is_forward":
        compile_fn = compile_config.fw_compiler
    else:
        compile_fn = compile_config.bw_compiler

    for inp_node in sub_node.all_input_nodes[
        1:
    ]:  # exlucde the graph module input to torch.ops.higher_order.invoke_subgraph
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

    def get_compiled_fn():
        context = torch._guards.TracingContext.get()
        assert context.fake_mode is not None

        context = torch._guards.TracingContext(context.fake_mode)

        with (
            torch._guards.tracing(context),
            CacheArtifactManager.with_fresh_cache(),
            torch._functorch.config.patch("bundled_autograd_cache", True),
        ):
            # compile_fx can mutate gm
            gm = copy.deepcopy(submod)

            compiled_fn = compile_fn(gm, fake_inputs)
            return compiled_fn

    compiled_fn = get_compiled_fn()
    assert isinstance(compiled_fn, AOTCompiledArtifact)

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


def _needs_inductor_compile(node: torch.fx.Node):
    # TODO: maybe we could change to check
    # node.meta.get("partitioner_tag") != "is_forward"
    # if the tag is relibable
    return (
        node.op not in ("placeholder", "output")
        and hasattr(node, "meta")
        and node.meta.get("custom", None)
        and node.meta["custom"].get("nested_region_config", None)
        and node.meta["custom"]["nested_region_config"].fw_compiler
        and node.meta.get("partitioner_tag") != "is_backward"
    ) or (
        node.op not in ("placeholder", "output")
        and hasattr(node, "meta")
        and node.meta.get("custom", None)
        and node.meta["custom"].get("nested_region_config", None)
        and node.meta["custom"]["nested_region_config"].bw_compiler
        and node.meta.get("partitioner_tag") == "is_backward"
    )


def _compile_invoke_subgraph_nodes_with_inductor(gm):
    map_subgraph_to_nodes = defaultdict(list)
    subgraphs: set[str] = set()

    for node in gm.graph.find_nodes(
        op="call_function", target=torch.ops.higher_order.invoke_subgraph
    ):
        if not _needs_inductor_compile(node):
            continue
        assert node.args[0].op == "get_attr"
        subgraph_name = node.args[0].target
        assert isinstance(subgraph_name, str)
        subgraphs.add(subgraph_name)
        map_subgraph_to_nodes[subgraph_name].append(node)

    for subgraph in subgraphs:
        gm = _compile_submod(gm, subgraph, map_subgraph_to_nodes[subgraph])

    return gm


def _recursive_compile_invoke_subgraph_nodes(gm):
    for node in gm.graph.find_nodes(op="get_attr"):
        if _needs_inductor_compile(node):
            # If the get_attr itself is marked for compile, the outer graph will
            # take care of it. If we dont do that, we end up with nested
            # regional inductor compiles that do not work well.
            continue
        submod = getattr(gm, node.target)
        if isinstance(submod, torch.fx.GraphModule):
            _recursive_compile_invoke_subgraph_nodes(submod)

    return _compile_invoke_subgraph_nodes_with_inductor(gm)


@compatibility(is_backward_compatible=False)
def regional_inductor_invoke_subgraph(gm, *example_args):
    """
    Compile invoke_subgraph nodes if they have custom compiler specified
    in node.meta["nested_region_config"].bw_compiler or fw_compiler
    """
    # fuser utils create new nodes using create_proxy which retains the seq_nr
    # metadata and cause issues
    with torch.fx.traceback.preserve_node_meta(enable=False):
        compiled_gm = _recursive_compile_invoke_subgraph_nodes(gm)
        # TODO: might not need this boxed_nop after we switch to _RegionCompiler
        return torch._dynamo.backends.debugging.boxed_nop(
            compiled_gm, example_inputs=[]
        )
