# mypy: allow-untyped-defs
from __future__ import annotations

import logging
from typing import Sequence

from onnxscript import ir


logger = logging.getLogger(__name__)


def rename_inputs(model: ir.Model, new_names: Sequence[str]) -> None:
    # TODO: Ensure the names do not have duplicates
    for input, new_name in zip(model.graph.inputs, new_names):
        input.metadata_props["pkg.torch.onnx.original_node_name"] = str(input.name)
        input.name = new_name


def rename_outputs(model: ir.Model, new_names: Sequence[str]) -> None:
    for output, new_name in zip(model.graph.outputs, new_names):
        output.metadata_props["pkg.torch.onnx.original_node_name"] = str(output.name)
        output.name = new_name


def add_opset_imports(model: ir.Model) -> None:
    """Collect all opsets used and add opset imports to the model and functions."""
    for node in ir.traversal.RecursiveGraphIterator(model.graph):
        domain = node.domain
        version = node.version if node.version is not None else 1
        if domain in model.opset_imports:
            # Heuristic to use the latest version seen
            version = max(version, model.opset_imports[domain])
        model.opset_imports[domain] = version

    for function in model.functions.values():
        for node in ir.traversal.RecursiveGraphIterator(function):
            domain = node.domain
            version = node.version if node.version is not None else 1
            if domain in function.opset_imports:
                # Heuristic to use the latest version seen
                version = max(version, function.opset_imports[domain])
            function.opset_imports[domain] = version
