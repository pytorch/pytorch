"""ONNX exporter exceptions."""

from typing import Optional

from torch.onnx import _constants

__all__ = ["OnnxExporterError", "CheckerError", "UnsupportedOperatorError"]


class OnnxExporterError(RuntimeError):
    """Errors raised by the ONNX exporter."""

    pass


class CheckerError(OnnxExporterError):
    r"""Raised when ONNX checker detects an invalid model."""

    pass


class UnsupportedOperatorError(OnnxExporterError):
    """Raised when an operator is unsupported by the exporter."""

    def __init__(
        self, domain: str, op_name: str, version: int, supported_version: Optional[int]
    ):
        if domain in {"", "aten", "prim", "quantized"}:
            msg = f"Exporting the operator '{domain}::{op_name}' to ONNX opset version {version} is not supported. "
            if supported_version is not None:
                msg += (
                    f"Support for this operator was added in version {supported_version}, "
                    "try exporting with this version."
                )
            else:
                msg += "Please feel free to request support or submit a pull request on PyTorch GitHub: "
                msg += _constants.PYTORCH_GITHUB_ISSUES_URL
        else:
            msg = (
                f"ONNX export failed on an operator with unrecognized namespace '{domain}::{op_name}'. "
                "If you are trying to export a custom operator, make sure you registered "
                "it with the right domain and version."
            )
        super().__init__(msg)
