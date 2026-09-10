"""A set of tools to verify the correctness of ONNX models."""

__all__ = ["VerificationInfo", "verify_onnx_program"]

from torch.onnx._internal.exporter._verification import (
    VerificationInfo,
    verify_onnx_program,
)


VerificationInfo.__module__ = "torch.onnx.verification"
verify_onnx_program.__module__ = "torch.onnx.verification"
