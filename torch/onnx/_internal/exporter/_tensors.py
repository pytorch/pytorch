"""Subclass of ir.Value that supports Python operators."""

# mypy: allow-untyped-defs
from __future__ import annotations

from typing import TYPE_CHECKING

from torch.onnx._internal._lazy_import import onnx_ir as ir


if TYPE_CHECKING:
    import onnxscript


class SymbolicTensor(ir.Value):
    """A subclass of ir.Value that supports Python operators."""

    def __init__(
        self,
        opset: onnxscript.values.Opset,
        name: str | None = None,
        shape: ir.Shape | None = None,
        type: ir.TypeProtocol | None = None,
        doc_string: str | None = None,
        const_value: ir.TensorProtocol | None = None,
    ) -> None:
        super().__init__(
            name=name,
            shape=shape,
            type=type,
            doc_string=doc_string,
            const_value=const_value,
        )
        self._opset = opset

    @property
    def rank(self) -> int | None:
        # pyrefly: ignore [missing-attribute]
        if self.shape is None:
            return None
        # pyrefly: ignore [bad-argument-type]
        return len(self.shape)

    # TODO: Implement indexing

    def __mod__(self, other):
        # pyrefly: ignore [missing-attribute]
        if self.dtype in {
            ir.DataType.FLOAT,
            ir.DataType.DOUBLE,
            ir.DataType.FLOAT16,
            ir.DataType.BFLOAT16,
        }:
            return self._opset.Mod(self, other, fmod=1)
        return self._opset.Mod(self, other)

    def __ne__(self, other):
        return self._opset.Not(self._opset.Equal(self, other))

    def __neg__(self):
        return self._opset.Neg(self)

    def __add__(self, other):
        return self._opset.Add(self, other)

    def __radd__(self, other):
        return self._opset.Add(other, self)

    def __rand__(self, other):
        return self._opset.And(other, self)

    def __mul__(self, other):
        return self._opset.Mul(self, other)

    def __rmul__(self, other):
        return self._opset.Mul(other, self)

    def __matmul__(self, other):
        return self._opset.MatMul(self, other)

    def __pow__(self, other):
        return self._opset.Pow(self, other)

    def __sub__(self, other):
        return self._opset.Sub(self, other)

    def __rsub__(self, other):
        return self._opset.Sub(other, self)

    def __truediv__(self, other):
        return self._opset.Div(self, other)

    def __lt__(self, other):
        return self._opset.Less(self, other)

    def __le__(self, other):
        return self._opset.LessOrEqual(self, other)

    def __ge__(self, other):
        return self._opset.GreaterOrEqual(self, other)

    def __gt__(self, other):
        return self._opset.Greater(self, other)
