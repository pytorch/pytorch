# mypy: allow-untyped-defs
from __future__ import annotations


__all__ = [
    "VerificationInfo",
    "verify_onnx_program",
]

import dataclasses
import math
from typing import Any, TYPE_CHECKING

import torch
from torch.utils import _pytree


if TYPE_CHECKING:
    from torch.onnx._internal.exporter import _onnx_program


@dataclasses.dataclass
class VerificationInfo:
    name: str
    max_abs_diff: float
    max_rel_diff: float
    abs_diff_hist: tuple[torch.Tensor, torch.Tensor]
    rel_diff_hist: tuple[torch.Tensor, torch.Tensor]
    expected_dtype: torch.dtype
    actual_dtype: torch.dtype
    # NOTE: We don't need to include shape because the expected shape is already known
    # and checked by the runtime


def _compare_tensors(
    expected: torch.Tensor,
    actual: torch.Tensor,
) -> tuple[float, float, torch.Tensor, torch.Tensor]:
    # Move tensors to the same device
    expected = expected.detach().cpu()
    actual = actual.detach().cpu()
    if expected.numel() == 0 or actual.numel() == 0:
        return math.inf, math.inf, torch.tensor(math.inf), torch.tensor(math.inf)
    if expected.dtype == torch.bool:
        expected = expected.to(torch.float32)
        actual = actual.to(torch.float32)
    abs_diff = torch.abs(expected - actual)
    eps = 1e-7
    normalizer = torch.abs(expected) + eps
    rel_diff = abs_diff / normalizer

    max_absolute_difference = abs_diff.max().item()
    max_relative_difference = rel_diff.max().item()

    return max_absolute_difference, max_relative_difference, abs_diff, rel_diff


def verify_onnx_program(
    onnx_program: _onnx_program.ONNXProgram,
    args: tuple[Any, ...] | None = None,
    kwargs: dict[str, Any] | None = None,
) -> list[VerificationInfo]:
    exported_program = onnx_program.exported_program
    if exported_program is None:
        raise ValueError(
            "The ONNX program does not contain an exported_program. "
            "Please provide an exported_program to verify the ONNX program."
        )
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
    torch_outputs, _ = _pytree.tree_flatten(torch_module(*args, **kwargs))
    onnx_outputs = onnx_program(*args, **kwargs)
    results = []
    for torch_output, onnx_output, output_val in zip(
        torch_outputs, onnx_outputs, onnx_program.model.graph.outputs
    ):
        name = output_val.name
        max_abs_diff, max_rel_diff, abs_diff, rel_diff = _compare_tensors(
            torch_output, onnx_output
        )
        abs_diff = abs_diff.flatten()
        rel_diff = rel_diff.flatten()
        bins = torch.tensor(
            [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10, 1000000],
            dtype=abs_diff.dtype,
        )
        abs_diff_hist = torch.histogram(abs_diff, bins=bins)
        rel_diff_hist = torch.histogram(rel_diff, bins=bins)
        results.append(
            VerificationInfo(
                name=str(name),
                max_abs_diff=max_abs_diff,
                max_rel_diff=max_rel_diff,
                abs_diff_hist=abs_diff_hist,
                rel_diff_hist=rel_diff_hist,
                expected_dtype=torch_output.dtype,
                actual_dtype=onnx_output.dtype,
            )
        )
    return results
