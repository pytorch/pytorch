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
    "CheckerError",
    "SymbolicValueError",
    "UnsupportedOperatorError",
]


class OnnxExporterWarning(UserWarning):
    """Base class for all warnings in the ONNX exporter."""

    pass


class OnnxExporterError(RuntimeError):
    """Errors raised by the ONNX exporter."""

    pass


class CheckerError(OnnxExporterError):
    """Raised when ONNX checker detects an invalid model."""

    pass


class UnsupportedOperatorError(OnnxExporterError):
    """Raised when an operator is unsupported by the exporter."""

    def __init__(self, name: str, version: int, supported_version: Optional[int]):
        if supported_version is not None:
            diagnostic_rule: diagnostics.infra.Rule = (
                diagnostics.rules.operator_supported_in_newer_opset_version
            )
            msg = diagnostic_rule.format_message(name, version, supported_version)
            diagnostics.diagnose(diagnostic_rule, diagnostics.levels.ERROR, msg)
        else:
            if name.startswith(("aten::", "prim::", "quantized::")):
                diagnostic_rule = diagnostics.rules.missing_standard_symbolic_function
                msg = diagnostic_rule.format_message(
                    name, version, _constants.PYTORCH_GITHUB_ISSUES_URL
                )
                diagnostics.diagnose(diagnostic_rule, diagnostics.levels.ERROR, msg)
            else:
                diagnostic_rule = diagnostics.rules.missing_custom_symbolic_function
                msg = diagnostic_rule.format_message(name)
                diagnostics.diagnose(diagnostic_rule, diagnostics.levels.ERROR, msg)
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
