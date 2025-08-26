"""ONNX exporter exceptions."""

from __future__ import annotations


__all__ = [
    "OnnxExporterWarning",
    "SymbolicValueError",
    "UnsupportedOperatorError",
]

import textwrap
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from torch import _C


class OnnxExporterWarning(UserWarning):
    """Warnings in the ONNX exporter."""


class OnnxExporterError(RuntimeError):
    """Errors raised by the ONNX exporter. This is the base class for all exporter errors."""


class UnsupportedOperatorError(OnnxExporterError):
    """Raised when an operator is unsupported by the exporter."""

    # NOTE: This is legacy and is only used by the torchscript exporter
    # Clean up when the torchscript exporter is removed
    def __init__(self, name: str, version: int, supported_version: int | None):
        if supported_version is not None:
            msg = (
                f"Exporting the operator '{name}' to ONNX opset version {version} "
                "is not supported. Support for this operator was added in version "
                f"{supported_version}, try exporting with this version"
            )
        elif name.startswith(("aten::", "prim::", "quantized::")):
            msg = (
                f"Exporting the operator '{name}' to ONNX opset version {version} "
                "is not supported"
            )
        else:
            msg = (
                "ONNX export failed on an operator with unrecognized namespace {op_name}. "
                "If you are trying to export a custom operator, make sure you registered it with "
                "the right domain and version."
            )

        super().__init__(msg)


class SymbolicValueError(OnnxExporterError):
    """Errors around TorchScript values and nodes."""

    # NOTE: This is legacy and is only used by the torchscript exporter
    # Clean up when the torchscript exporter is removed
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
