"""ONNX exporter exceptions."""
from __future__ import annotations

import textwrap
from typing import Optional

from torch import _C
from torch.onnx import _constants
from torch.onnx._internal import diagnostics

__all__ = [
    "OnnxExporterError",
    "OnnxExporterWarning",
    "CallHintViolationWarning",
    "CheckerError",
    "UnsupportedOperatorError",
    "SymbolicValueError",
]


class OnnxExporterWarning(UserWarning):
    """Base class for all warnings in the ONNX exporter."""

    pass


class CallHintViolationWarning(OnnxExporterWarning):
    """Warning raised when a type hint is violated during a function call."""

    pass


class OnnxExporterError(RuntimeError):
    """Errors raised by the ONNX exporter."""

    pass


class CheckerError(OnnxExporterError):
    """Raised when ONNX checker detects an invalid model."""

    pass


class UnsupportedOperatorError(OnnxExporterError):
    """Raised when an operator is unsupported by the exporter."""

    def __init__(
        self,
        domain: str,
        op_name: str,
        version: int,
        supported_version: Optional[int],
    ):
        if domain in {"", "aten", "prim", "quantized"}:
            msg = f"Exporting the operator '{domain}::{op_name}' to ONNX opset version {version} is not supported. "
            if supported_version is not None:
                msg += (
                    f"Support for this operator was added in version {supported_version}, "
                    "try exporting with this version."
                )
                diagnostics.context.diagnose(
                    diagnostics.rules.operator_supported_in_newer_opset_version,
                    diagnostics.levels.ERROR,
                    message_args=(
                        f"{domain}::{op_name}",
                        version,
                        supported_version,
                    ),
                )
            else:
                msg += "Please feel free to request support or submit a pull request on PyTorch GitHub: "
                msg += _constants.PYTORCH_GITHUB_ISSUES_URL
                diagnostics.context.diagnose(
                    diagnostics.rules.missing_standard_symbolic_function,
                    diagnostics.levels.ERROR,
                    message_args=(
                        f"{domain}::{op_name}",
                        version,
                        _constants.PYTORCH_GITHUB_ISSUES_URL,
                    ),
                )
        else:
            msg = (
                f"ONNX export failed on an operator with unrecognized namespace '{domain}::{op_name}'. "
                "If you are trying to export a custom operator, make sure you registered "
                "it with the right domain and version."
            )
            diagnostics.context.diagnose(
                diagnostics.rules.missing_custom_symbolic_function,
                diagnostics.levels.ERROR,
                message_args=(f"{domain}::{op_name}",),
            )
        super().__init__(msg)


class SymbolicValueError(OnnxExporterError):
    """Errors around TorchScript values and nodes."""

    def __init__(self, msg: str, value: _C.Value):
        message = (
            f"{msg}  [Caused by the value '{value}' (type '{value.type()}') in the "
            f"TorchScript graph. The containing node has kind '{value.node().kind()}'.] "
        )

        code_location = value.node().sourceRange()
        if code_location:
            message += f"\n    (node defined in {code_location})"

        try:
            # Add its input and output to the message.
            message += "\n\n"
            message += textwrap.indent(
                (
                    "Inputs:\n"
                    + (
                        "\n".join(
                            f"    #{i}: {input_}  (type '{input_.type()}')"
                            for i, input_ in enumerate(value.node().inputs())
                        )
                        or "    Empty"
                    )
                    + "\n"
                    + "Outputs:\n"
                    + (
                        "\n".join(
                            f"    #{i}: {output}  (type '{output.type()}')"
                            for i, output in enumerate(value.node().outputs())
                        )
                        or "    Empty"
                    )
                ),
                "    ",
            )
        except AttributeError:
            message += (
                " Failed to obtain its input and output for debugging. "
                "Please refer to the TorchScript graph for debugging information."
            )

        super().__init__(message)
