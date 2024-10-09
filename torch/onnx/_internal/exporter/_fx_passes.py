# mypy: allow-untyped-defs
from __future__ import annotations

import torch
import torch.export
import torch.fx
from torch.onnx._internal.exporter import _decomp, _registration
from torch.onnx._internal.fx import diagnostics, passes


def decompose_with_registry(
    exported_program: torch.export.ExportedProgram, registry: _registration.ONNXRegistry
) -> torch.export.ExportedProgram:
    """Decompose the exported program with the given registry.

    This function is needed so it shows clearly on the profiler results.
    """
    onnx_registered_ops = set(_decomp.get_onnx_implemented_overloads(registry))
    decomp_table = _decomp.create_onnx_friendly_decomposition_table(onnx_registered_ops)
    return exported_program.run_decompositions(decomp_table)


def insert_type_promotion_nodes(
    graph_module: torch.fx.GraphModule,
) -> torch.fx.GraphModule:
    """Inplace pass to insert explicit type promotion nodes."""
    diagnostic_context = diagnostics.DiagnosticContext(
        "torch.onnx.export",
        torch.__version__,
    )
    return passes.InsertTypePromotion(diagnostic_context, graph_module).run()


def remove_assertion_nodes(graph_module: torch.fx.GraphModule) -> torch.fx.GraphModule:
    """Remove all assertion and check nodes from the FX graph"""
    aten_assertion_targets = {
        torch.ops.aten.sym_constrain_range_for_size.default,
        torch.ops.aten._assert_async.default,
        torch.ops.aten._assert_async.msg,
        torch.ops.aten._assert_scalar.default,
        torch.ops.aten._assert_tensor_metadata.default,
    }
    for node in graph_module.graph.nodes:
        if node.op == "call_function" and node.target in aten_assertion_targets:
            graph_module.graph.erase_node(node)
    graph_module.recompile()
    return graph_module


def insert_contiguous_between_transpose_and_view(
    exported_program: torch.export.ExportedProgram,
) -> torch.export.ExportedProgram:
    """Modifies the module inplace to insert a node 'contiguous' between a node 'transpose' followed by a node 'view'.

    The modification takes place inplace.

    Remove after issue https://github.com/pytorch/pytorch/issues/136543 is fixed.
    """
    modified = False
    graph = exported_program.graph_module.graph
    for node in graph.nodes:
        if (
            node.op != "call_function"
            or not hasattr(node.target, "name")
            or node.target.name() != "aten::transpose.int"
        ):
            continue
        insert = False
        for user in node.users:
            if (
                user.op == "call_function"
                and hasattr(node.target, "name")
                and user.target.name() == "aten::view"
            ):
                insert = True
                break
        if not insert:
            continue

        modified = True
        with graph.inserting_after(node):
            new_node = graph.call_function(
                torch.ops.aten.contiguous.default, args=(node,)
            )
            node.replace_all_uses_with(new_node)
            # new_node is replaced as well so we manually revert the replacement
            new_node.update_arg(0, node)
            node.users = {new_node: None}

    if not modified:
        # no rewrite was done.
        return exported_program

    graph.lint()
    return exported_program
