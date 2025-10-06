# mypy: allow-untyped-defs
import functools

import torch

# from torch._dynamo.backends.common import aot_autograd as auto_autograd_backend
from torch._guards import TracingContext
from torch.fx.passes.infra.partitioner import CapabilityBasedPartitioner
from torch.fx.passes.operator_support import OperatorSupport
from torch.fx.passes.utils.fuser_utils import fuse_by_partitions


def listify_inputs(fn):
    # Handles boxed arguments expectation from compile_fx_inner
    @functools.wraps(fn)
    def inner(*args):
        return fn(list(args))

    return inner


def partition_by_supported_nodes(gm, supported_ops, prefix):
    partitioner = CapabilityBasedPartitioner(
        gm, supported_ops, allows_single_node_partition=True
    )

    candidate_partitions = partitioner.propose_partitions()
    partitioned_gm = fuse_by_partitions(
        partitioner.graph_module,
        [candidate_partitions[i].nodes for i in range(len(candidate_partitions))],
        prefix=prefix,
        always_return_tuple=True,
    )

    return partitioned_gm


def compile_submod(gm, prefix):
    for node in gm.graph.nodes:
        if node.op == "call_module" and node.target.startswith(prefix):
            fake_inputs = []
            for inp_node in node.all_input_nodes:
                if hasattr(inp_node, "meta") and "val" in inp_node.meta:
                    fake_inputs.append(inp_node.meta["val"])

            submod = getattr(gm, node.target)
            # Ensure that it runs in eager
            submod(*fake_inputs)

            from torch._inductor.compile_fx import compile_fx_inner

            # [inductor-stateless-issue] - Calling compile_fx_inner is changing
            # the reported output strides.
            with TracingContext.report_output_strides():
                compiled_submod = listify_inputs(compile_fx_inner(submod, fake_inputs))

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


def has_marked_node_custom_metadata(node):
    return (
        node.op != "placeholder"
        and hasattr(node, "meta")
        and node.meta.get("custom", None)
        and "compile_with_inductor" in node.meta["custom"]
    )


def compile_fx_annotated_nodes_with_inductor(gm, *example_args):
    # fuser utils create new nodes using create_proxy which retains the seq_nr
    # metadata and cause issues
    with torch.fx.traceback.preserve_node_meta(enable=False):
        found_marked_node = False
        for node in gm.graph.nodes:
            if has_marked_node_custom_metadata(node):
                found_marked_node = True
                break

        if not found_marked_node:
            print("No inductor marked nodes found")
            return gm

        class InductorMarkedNodes(OperatorSupport):
            def is_node_supported(self, submodules, node: torch.fx.Node) -> bool:
                return has_marked_node_custom_metadata(node)

        marked_nodes = InductorMarkedNodes()
        gm = partition_by_supported_nodes(gm, marked_nodes, "__marked_inductor_submod")
        gm = compile_submod(gm, "__marked_inductor_submod")
        return gm
