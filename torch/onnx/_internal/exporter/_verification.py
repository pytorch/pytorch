from __future__ import annotations

import dataclasses
from typing import Any, Collection, TYPE_CHECKING

import torch
from torch.utils import _pytree as pytree


if TYPE_CHECKING:
    from torch.onnx._internal.exporter import _onnx_program


@dataclasses.dataclass
class VerificationInfo:
    name: str
    absolute_difference: float
    relative_difference: float
    expected_dtype: torch.dtype
    actual_dtype: torch.dtype
    # NOTE: We don't need to include shape because the expected shape is already known
    # and checked by the runtime


def _compare_tensors(
    expected: torch.Tensor,
    actual: torch.Tensor,
) -> tuple[float, float]:
    # Move tensors to the same device
    expected = expected.detach().cpu()
    actual = actual.detach().cpu()
    absolute_difference = torch.abs(expected - actual).max().item()
    eps = 1e-7
    relative_difference = (
        (torch.abs(expected - actual) / (torch.abs(expected) + eps)).max().item()
    )
    return absolute_difference, relative_difference


def verify_onnx_program(
    onnx_program: _onnx_program.ONNXProgram,
    args: tuple[Any, ...] | None = None,
    kwargs: dict[str, Any] | None = None,
) -> list[VerificationInfo]:
    exported_program = onnx_program.exported_program
    if args is None and kwargs is None:
        # User did not provide example inputs, use the default example inputs
        if exported_program.example_inputs is None:
            raise ValueError(
                "No example inputs provided and the exported_program does not contain example inputs. "
                "Please provide arguments to verify the ONNX program."
            )
        args, kwargs = exported_program.example_inputs
    if args is None:
        args = ()
    if kwargs is None:
        kwargs = {}
    torch_module = exported_program.module()
    torch_outputs, _ = pytree.tree_flatten(torch_module(*args, **kwargs))
    onnx_outputs = onnx_program(*args, **kwargs)
    results = []
    for torch_output, onnx_output, output_val in zip(
        torch_outputs, onnx_outputs, onnx_program.model.graph.outputs
    ):
        name = output_val.name
        absolute_difference, relative_difference = _compare_tensors(
            torch_output, onnx_output
        )
        results.append(
            VerificationInfo(
                name=name,
                absolute_difference=absolute_difference,
                relative_difference=relative_difference,
                expected_dtype=torch_output.dtype,
                actual_dtype=onnx_output.dtype,
            )
        )
    return results


def save_node_data_for_model_explorer(verification_infos: Collection[VerificationInfo]):
    # https://github.com/google-ai-edge/model-explorer/wiki/4.-API-Guide#create-custom-node-data
    # This API is unstable and may change in the future.
    from model_explorer import node_data_builder as ndb

    for field in ("absolute_difference", "relative_difference"):
        # Populate values for the main graph in a model.
        main_graph_results: dict[str, ndb.NodeDataResult] = {}
        for info in verification_infos:
            main_graph_results[f"[value] {info.name}"] = ndb.NodeDataResult(
                value=getattr(info, field)
            )

        thresholds: list[ndb.ThresholdItem] = [
            ndb.ThresholdItem(value=0.00001, bgColor="#388e3c"),
            ndb.ThresholdItem(value=0.0001, bgColor="#8bc34a"),
            ndb.ThresholdItem(value=0.001, bgColor="#c8e6c9"),
            ndb.ThresholdItem(value=0.01, bgColor="#ffa000"),
            ndb.ThresholdItem(value=1, bgColor="#ff5722"),
            ndb.ThresholdItem(value=100, bgColor="#d32f2f"),
        ]

        # Construct the data for the main graph.
        main_graph_data = ndb.GraphNodeData(
            results=main_graph_results, thresholds=thresholds
        )

        # Construct the data for the model.
        # "main_graph" is defined in _core.py
        model_data = ndb.ModelNodeData(graphsData={"main_graph": main_graph_data})

        model_data.save_to_file(f"{field}.json")
