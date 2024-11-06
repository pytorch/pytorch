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
