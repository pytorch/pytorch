# mypy: allow-untyped-defs
from __future__ import annotations

import logging
from typing import Sequence

from torch.onnx._internal._lazy_import import onnxscript_apis, onnxscript_ir as ir


# The opset domain for ONNX operators
_ONNX_DOMAIN = ""

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


def add_torchlib_common_imports(model: ir.Model) -> None:
    """Hack to add torchlib common imports to the model."""

    try:
        # TODO(justinchuby): Remove this hack and improved onnxscript
        from onnxscript.function_libs.torch_lib.ops import common as common_ops

        model.opset_imports["pkg.onnxscript.torch_lib.common"] = 1
        rank_func = ir.serde.deserialize_function(common_ops.Rank.to_function_proto())
        is_scalar_func = ir.serde.deserialize_function(
            common_ops.IsScalar.to_function_proto()
        )
        model.functions[rank_func.identifier()] = rank_func
        model.functions[is_scalar_func.identifier()] = is_scalar_func
    except Exception:
        logger.exception("Failed to add torchlib common imports to the model.")


def _maybe_set_opset_version(
    opset_imports: dict[str, int], domain: str, version: int | None
) -> None:
    """Set the opset version for the domain."""
    if domain in opset_imports and opset_imports[domain] != 1:
        # Already set
        return
    if domain == _ONNX_DOMAIN:
        # Set the default opset version for ONNX operators
        opset_imports[domain] = onnxscript_apis.torchlib_opset_version()
        return
    if version is None:
        # We don't know the opset version, so set it to 1
        # This is valid for the custom function domains like "pkg.torch.__subgraph__"
        opset_imports[domain] = 1
        return
    # Set the known opset version for the domain
    opset_imports[domain] = version


def add_opset_imports(model: ir.Model) -> None:
    """Collect all opsets used and add opset imports to the model and functions."""
    for node in ir.traversal.RecursiveGraphIterator(model.graph):
        domain = node.domain
        _maybe_set_opset_version(model.opset_imports, domain, node.version)

    for function in model.functions.values():
        for node in ir.traversal.RecursiveGraphIterator(function):
            domain = node.domain
            _maybe_set_opset_version(function.opset_imports, domain, node.version)
        for domain, version in function.opset_imports.items():
            # Add all opsets used in the function to the model, because ONNX Runtime
            # does not handle adding the opset imports to the model after inlining during inference.
            # This should happen after all opsets are collected for the function from its nodes.
            _maybe_set_opset_version(model.opset_imports, domain, version)
