from __future__ import annotations


__all__ = [
    "VerificationInfo",
    "VerificationInterpreter",
    "verify_onnx_program",
]

import dataclasses
import logging
import math
from typing import Any, TYPE_CHECKING

import torch
from torch.utils import _pytree


if TYPE_CHECKING:
    from onnxscript import ir

    from torch.onnx._internal.exporter import _onnx_program


logger = logging.getLogger(__name__)


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

    @classmethod
    def from_tensors(
        cls,
        name: str,
        expected: torch.Tensor | int | float | bool,
        actual: torch.Tensor | int | float | bool,
    ) -> VerificationInfo:
        """Create a VerificationInfo object from two tensors.

        Args:
            name: The name of the value.
            expected: The expected tensor.
            actual: The actual tensor.

        Returns:
            VerificationInfo: The VerificationInfo object.
        """
        if not isinstance(expected, torch.Tensor):
            expected = torch.tensor(expected)
        if not isinstance(actual, torch.Tensor):
            actual = torch.tensor(actual)

        max_abs_diff, max_rel_diff, abs_diff, rel_diff = _compare_tensors(
            expected, actual
        )
        bins = torch.tensor(
            [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10, 1000000],
            dtype=torch.float,
        )
        abs_diff_hist = torch.histogram(abs_diff.float(), bins=bins)
        rel_diff_hist = torch.histogram(rel_diff.float(), bins=bins)
        return cls(
            name=name,
            max_abs_diff=max_abs_diff,
            max_rel_diff=max_rel_diff,
            abs_diff_hist=abs_diff_hist,
            rel_diff_hist=rel_diff_hist,
            expected_dtype=expected.dtype,
            actual_dtype=actual.dtype,
        )


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
    if torch.is_complex(expected):
        expected = torch.view_as_real(expected)
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
        results.append(
            VerificationInfo.from_tensors(
                name=str(name),
                expected=torch_output,
                actual=onnx_output,
            )
        )
    return results


def _create_value_mapping(graph: ir.Graph) -> dict[str, ir.Value]:
    """Return a dictionary mapping names to values in the graph.

    The mapping does not include values from subgraphs.

    Args:
        graph: The graph to extract the mapping from.

    Returns:
        A dictionary mapping names to values.
    """
    values = {}
    values.update(graph.initializers)
    # The names of the values can be None or "", which we need to exclude
    for input in graph.inputs:
        if not input.name:
            continue
        values[input.name] = input
    for node in graph:
        for value in node.outputs:
            if not value.name:
                continue
            values[value.name] = value
    return values


class VerificationInterpreter(torch.fx.Interpreter):
    """Interpreter for verifying converted ONNX model accuracy by comparing intermediate values.

    To compare models, first initialize the interpreter with an ONNX program.
    Then, call the :meth:`run` method with the input arguments to execute the model.
    The :meth:`run` method will execute the model and populate the
    :attr:`verification_infos` attribute with the verification information for each value.

    ::
        onnx_program = torch.onnx.export(model, args, dynamo=True)
        interpreter = VerificationInterpreter(onnx_program)
        interpreter.run(*args)
        verification_infos = interpreter.verification_infos
        for info in verification_infos:
            print("value name:", info.name, info)

    The verification information includes the maximum absolute difference, maximum relative
    difference, and histograms of absolute and relative differences between the expected
    and actual values. See :class:`VerificationInfo` for more details.

    Attributes:
        verification_infos: A list of verification information for each value.
            It is populated when the `run` method is called.
    """

    def __init__(self, onnx_program: torch.onnx.ONNXProgram) -> None:
        """Initialize the VerificationInterpreter with an ONNX program.

        Args:
            onnx_program: The ONNX program to verify.
        """
        if onnx_program.exported_program is None:
            raise ValueError(
                "The ONNX program does not contain an exported_program. "
                "Please provide an exported_program to verify the ONNX program."
            )
        super().__init__(onnx_program.exported_program.module())
        self._onnx_program = onnx_program
        self._onnx_values = _create_value_mapping(onnx_program.model.graph)
        self._args: list[Any] = []
        self.verification_infos: list[VerificationInfo] = []

    def run(
        self,
        *args: Any,
        initial_env: dict[torch.fx.Node, Any] | None = None,
        enable_io_processing: bool = True,
    ) -> Any:
        """Run the interpreter with the given input arguments.

        This method executes the model and populates the :attr:`verification_infos` attribute
        with the verification information for each value.

        Args:
            args: The input arguments for the model.
            initial_env: The initial environment for the interpreter.
            enable_io_processing: Whether to enable IO processing.

        Returns:
            Any: The result of executing the model.
        """
        self.verification_infos = []
        self.args = args
        return super().run(
            *args,
            initial_env=initial_env,
            enable_io_processing=enable_io_processing,
        )

    def run_node(self, n: torch.fx.Node) -> Any:
        result = super().run_node(n)
        if n.op != "call_function":
            return result
        node_name = n.name
        if node_name not in self._onnx_values:
            return result
        (onnx_result,) = self._onnx_program.compute_values([node_name], self.args)
        info = VerificationInfo.from_tensors(
            name=node_name,
            expected=result,
            actual=onnx_result,
        )
        self.verification_infos.append(info)
        if info.max_abs_diff > 0.01 or info.max_rel_diff > 0.1:
            logger.warning(
                "Verification info for node %s: max_abs_diff: %s, max_rel_diff: %s",
                node_name,
                info.max_abs_diff,
                info.max_rel_diff,
            )
        else:
            logger.info(
                "Verification info for node %s: max_abs_diff: %s, max_rel_diff: %s",
                node_name,
                info.max_abs_diff,
                info.max_rel_diff,
            )
        return result
