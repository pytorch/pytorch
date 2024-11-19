# mypy: allow-untyped-defs
from __future__ import annotations

import logging
from typing import Mapping, Sequence

from onnxscript import ir

_MIN_ONNX_OPSET_VERSION = 18


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


def _get_opset_version(node: ir.Node, opset_imports: Mapping[str, int]) -> int:
    """Determine the appropriate opset version for a node."""
    domain = node.domain
    version = node.version if node.version is not None else 1
    if domain == "":
        return max(version, _MIN_ONNX_OPSET_VERSION)
    elif domain in opset_imports:
        # Heuristic to use the latest version seen
        return max(version, opset_imports[domain])
    return version


def add_opset_imports(model: ir.Model) -> None:
    """Collect all opsets used and add opset imports to the model and functions."""
    for node in ir.traversal.RecursiveGraphIterator(model.graph):
        domain = node.domain
        model.opset_imports[domain] = _get_opset_version(node, model.opset_imports)

    for function in model.functions.values():
        for node in ir.traversal.RecursiveGraphIterator(function):
            domain = node.domain
            function.opset_imports[domain] = _get_opset_version(
                node, function.opset_imports
            )
