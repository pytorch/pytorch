# mypy: allow-untyped-defs

import functools
import logging

import torch
from torch.fx._compatibility import compatibility


logger = logging.getLogger(__name__)

__all__ = ["regional_inductor"]


# standalone_inductor returns a callable class object - this does not sit well
# with Fx graph node op call_function which expects a function. So this is just
# a wrapper function to make Fx graph codegen happy.
def _dummy_wrapper(fn):
    @functools.wraps(fn)
    def inner(*args, **kwargs):
        return fn(*args, **kwargs)

    return inner


def _partition_by_supported_nodes(gm, supported_ops, prefix):
    from torch.fx.passes.infra.partitioner import CapabilityBasedPartitioner
    from torch.fx.passes.utils.fuser_utils import fuse_by_partitions

    partitioner = CapabilityBasedPartitioner(
        gm, supported_ops, allows_single_node_partition=True
    )

    candidate_partitions = partitioner.propose_partitions()
    partitioned_gm = fuse_by_partitions(
        partitioner.graph_module,
        [partition.nodes for partition in candidate_partitions],
        prefix=prefix,
        always_return_tuple=True,
    )

    return partitioned_gm


def _compile_submod(gm, prefix):
    for node in gm.graph.nodes:
        if node.op == "call_module" and node.target.startswith(prefix):
            fake_inputs = []
            for inp_node in node.all_input_nodes:
                if hasattr(inp_node, "meta") and "val" in inp_node.meta:
                    fake_inputs.append(inp_node.meta["val"])
                else:
                    raise RuntimeError(
                        f"Partition is bad because non fake tensor value is seen {inp_node}"
                    )

            submod = getattr(gm, node.target)

            # _dummy_wrapper is to make call_function happy
            compiled_submod = _dummy_wrapper(
                torch._inductor.standalone_compile(
                    submod, fake_inputs, dynamic_shapes="from_tracing_context"
                )
            )

            with gm.graph.inserting_after(node):
                new_node = gm.graph.call_function(
                    compiled_submod, args=node.args, kwargs=node.kwargs
                )
                new_node.meta = node.meta
                node.replace_all_uses_with(new_node)
                gm.graph.erase_node(node)
                del gm._modules[node.target]

    gm.recompile()
    return gm


def _needs_inductor_compile(node):
    return (
        node.op not in ("placeholder", "output")
        and hasattr(node, "meta")
        and node.meta.get("custom", None)
        and "compile_with_inductor" in node.meta["custom"]
    )


def _compile_fx_annotated_nodes_with_inductor(gm):
    from torch.fx.passes.operator_support import OperatorSupport

    found_marked_node = False
    for node in gm.graph.nodes:
        if _needs_inductor_compile(node):
            found_marked_node = True
            break

    if not found_marked_node:
        logger.info("No inductor marked nodes found")
        return gm

    class InductorMarkedNodes(OperatorSupport):
        def is_node_supported(self, submodules, node: torch.fx.Node) -> bool:
            return _needs_inductor_compile(node)

    marked_nodes = InductorMarkedNodes()
    gm = _partition_by_supported_nodes(gm, marked_nodes, "__marked_inductor_submod")
    gm = _compile_submod(gm, "__marked_inductor_submod")
    return gm


def _recursive_compile_fx_annotated_nodes_with_inductor(gm):
    for node in gm.graph.find_nodes(op="get_attr"):
        if _needs_inductor_compile(node):
            # If the get_attr itself is marked for compile, the outer graph will
            # take care of it. If we dont do that, we end up with nested
            # regional inductor compiles that do not work well.
            continue
        submod = getattr(gm, node.target)
        if isinstance(submod, torch.fx.GraphModule):
            _recursive_compile_fx_annotated_nodes_with_inductor(submod)

    return _compile_fx_annotated_nodes_with_inductor(gm)


@compatibility(is_backward_compatible=False)
def regional_inductor(gm, *example_args):
    """
    Scoops out inductor marked regions and compiles them with inductor.
    """
    # fuser utils create new nodes using create_proxy which retains the seq_nr
    # metadata and cause issues
    with torch.fx.traceback.preserve_node_meta(enable=False):
        return _recursive_compile_fx_annotated_nodes_with_inductor(gm)
