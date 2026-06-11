# pyre-strict
"""Logical ops: and, or, xor (binary) + not (unary).

Inputs may be any numeric dtype OR bool -- semantics treat non-zero as
True. Output is always bool.
"""

from __future__ import annotations

import random

import torch

from torchfuzz.operators._dtypes import contiguous_stride, random_broadcast_shape
from torchfuzz.operators.base import Operator
from torchfuzz.tensor_fuzzer import Spec, TensorSpec


_LOGICAL_INPUT_DTYPES: tuple[torch.dtype, ...] = (
    torch.float16,
    torch.float32,
    torch.float64,
    torch.bfloat16,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.uint8,
    torch.uint16,
    torch.uint32,
    torch.uint64,
    torch.bool,
)


# ---------------------------------------------------------------------------
# Base class for binary logical ops
# ---------------------------------------------------------------------------


class LogicalBinaryOperatorBase(Operator):
    """Two-tensor logical -> bool. Inputs share a randomly-chosen dtype."""

    @property
    def torch_op_name(self) -> str:
        return self.name

    def can_produce(self, output_spec: Spec) -> bool:
        return isinstance(output_spec, TensorSpec) and output_spec.dtype == torch.bool

    def fuzz_inputs_specs(self, output_spec: Spec) -> list[Spec]:
        assert isinstance(output_spec, TensorSpec)  # noqa: S101
        dtype = random.choice(_LOGICAL_INPUT_DTYPES)
        spec = TensorSpec(
            size=output_spec.size,
            stride=output_spec.stride,
            dtype=dtype,
        )
        specs: list[Spec] = [spec, spec]
        if random.random() < 0.3:
            idx = random.randint(0, 1)
            bcast_size = random_broadcast_shape(tuple(output_spec.size))
            if bcast_size != tuple(output_spec.size):
                specs[idx] = TensorSpec(
                    size=bcast_size,
                    stride=contiguous_stride(bcast_size),
                    dtype=dtype,
                )
        return specs

    def codegen(
        self,
        output_name: str,
        input_names: list[str],
        output_spec: Spec,
    ) -> str:
        return (
            f"{output_name} = {self.torch_op_name}({input_names[0]}, {input_names[1]})"
        )


# ---------------------------------------------------------------------------
# Logical binary subclasses
# ---------------------------------------------------------------------------


class LogicalAndOperator(LogicalBinaryOperatorBase):
    def __init__(self) -> None:
        super().__init__("torch.logical_and")


class LogicalOrOperator(LogicalBinaryOperatorBase):
    def __init__(self) -> None:
        super().__init__("torch.logical_or")


class LogicalXorOperator(LogicalBinaryOperatorBase):
    def __init__(self) -> None:
        super().__init__("torch.logical_xor")


# ---------------------------------------------------------------------------
# Logical unary (only one op, so a base class is not justified)
# ---------------------------------------------------------------------------


class LogicalNotOperator(Operator):
    """Single-tensor logical -> bool."""

    def __init__(self) -> None:
        super().__init__("torch.logical_not")

    @property
    def torch_op_name(self) -> str:
        return self.name

    def can_produce(self, output_spec: Spec) -> bool:
        return isinstance(output_spec, TensorSpec) and output_spec.dtype == torch.bool

    def fuzz_inputs_specs(self, output_spec: Spec) -> list[Spec]:
        assert isinstance(output_spec, TensorSpec)  # noqa: S101
        dtype = random.choice(_LOGICAL_INPUT_DTYPES)
        return [
            TensorSpec(
                size=output_spec.size,
                stride=output_spec.stride,
                dtype=dtype,
            )
        ]

    def codegen(
        self,
        output_name: str,
        input_names: list[str],
        output_spec: Spec,
    ) -> str:
        return f"{output_name} = torch.logical_not({input_names[0]})"
