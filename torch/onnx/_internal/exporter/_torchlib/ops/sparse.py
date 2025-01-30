"""torch.ops.aten operators under the `sparse` module."""

# mypy: disable-error-code="misc,arg-type,type-arg,valid-type,assignment,return-value,type-var,operator,no-untyped-def,index"
# ruff: noqa: TCH001,TCH002
# flake8: noqa

from __future__ import annotations

from onnxscript.onnx_types import TensorType


def aten_sparse_sampled_addmm(
    self: TensorType,
    mat1: TensorType,
    mat2: TensorType,
    beta: float = 1.0,
    alpha: float = 1.0,
) -> TensorType:
    """sparse_sampled_addmm(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1) -> Tensor"""

    raise NotImplementedError
