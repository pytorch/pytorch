# mypy: allow-untyped-defs
from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

from torch.onnx._internal._lazy_import import onnxscript_ir as ir
from torch.onnx._internal.exporter import _constants


if TYPE_CHECKING:
    from collections.abc import Sequence


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


def _all_values(model: ir.Model):
    """Yield all values in a model."""
    # Yield all values in the model
    yield from model.graph.inputs
    yield from model.graph.initializers.values()
    for node in ir.traversal.RecursiveGraphIterator(model.graph):
        yield from node.outputs
    # Yield all values in functions
    for function in model.functions.values():
        yield from function.inputs
        for node in ir.traversal.RecursiveGraphIterator(function):
            yield from node.outputs


def _replace_names(shape_expr: str, rename_mapping: dict[str, str]) -> str:
    """Replace all known names in a shape expression with new names."""
    for old_name, new_name in rename_mapping.items():
        shape_expr = re.sub(
            rf"(?<!\w){re.escape(old_name)}(?!\w)", new_name, shape_expr
        )
    return shape_expr


def rename_axis(model: ir.Model, rename_mapping: dict[str, str]) -> None:
    """Rename dynamic axes in a model according to the specified dynamic_axes names."""

    # NOTE: Mapping needs to be srted by length because the shape expression
    # could have multiple ways to be expressed, for example,
    # {"s1": sequence_length, "s11": "past_sequence_length", "s1 + s11": "masked_sequence_length"}
    # We prefer the replacement starts from the longest match.
    sorted_rename_mapping = dict(
        sorted(rename_mapping.items(), key=lambda item: len(item[0]), reverse=True)
    )

    for value in _all_values(model):
        if value.shape is None:
            continue
        new_shape = []
        changed = False
        for dim in value.shape:
            if not isinstance(dim, ir.SymbolicDim):
                new_shape.append(dim)
                continue
            dim_name = dim.value
            if dim_name in sorted_rename_mapping:
                new_shape.append(sorted_rename_mapping[dim_name])
                changed = True
            elif dim_name is not None:
                # For example: "2*s1", "s1+1", "s1-1", "s1*s2", "s1/s2"
                new_name = _replace_names(dim_name, sorted_rename_mapping)
                new_shape.append(new_name)
                if new_name != dim_name:
                    changed = True
            else:
                new_shape.append(None)
        if changed:
            value.shape = ir.Shape(new_shape)


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
        opset_imports[domain] = _constants.TORCHLIB_OPSET
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
