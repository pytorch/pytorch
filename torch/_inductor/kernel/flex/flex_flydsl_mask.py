# mypy: allow-untyped-defs

from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import Any

import torch
from torch.fx import Node

from ...virtualized import V


_BINARY_TARGETS = {
    operator.add: "add",
    torch.ops.aten.add.Tensor: "add",
    torch.ops.aten.add.Scalar: "add",
    operator.sub: "sub",
    torch.ops.aten.sub.Tensor: "sub",
    torch.ops.aten.sub.Scalar: "sub",
    operator.mul: "mul",
    torch.ops.aten.mul.Tensor: "mul",
    torch.ops.aten.mul.Scalar: "mul",
    operator.floordiv: "floordiv",
    torch.ops.aten.floor_divide.default: "floordiv",
    operator.mod: "remainder",
    torch.ops.aten.remainder.Tensor: "remainder",
    torch.ops.aten.remainder.Scalar: "remainder",
    operator.ge: "ge",
    torch.ops.aten.ge.Tensor: "ge",
    torch.ops.aten.ge.Scalar: "ge",
    operator.gt: "gt",
    torch.ops.aten.gt.Tensor: "gt",
    torch.ops.aten.gt.Scalar: "gt",
    operator.le: "le",
    torch.ops.aten.le.Tensor: "le",
    torch.ops.aten.le.Scalar: "le",
    operator.lt: "lt",
    torch.ops.aten.lt.Tensor: "lt",
    torch.ops.aten.lt.Scalar: "lt",
    operator.eq: "eq",
    torch.ops.aten.eq.Tensor: "eq",
    torch.ops.aten.eq.Scalar: "eq",
    operator.ne: "ne",
    torch.ops.aten.ne.Tensor: "ne",
    torch.ops.aten.ne.Scalar: "ne",
    operator.and_: "and",
    torch.ops.aten.bitwise_and.Tensor: "and",
    torch.ops.aten.bitwise_and.Scalar: "and",
    operator.or_: "or",
    torch.ops.aten.bitwise_or.Tensor: "or",
    torch.ops.aten.bitwise_or.Scalar: "or",
}

_UNARY_TARGETS = {
    operator.invert: "not",
    torch.ops.aten.bitwise_not.default: "not",
}

_ALPHA_TARGETS = (
    torch.ops.aten.add.Tensor,
    torch.ops.aten.add.Scalar,
    torch.ops.aten.sub.Tensor,
    torch.ops.aten.sub.Scalar,
)


@dataclass(frozen=True)
class FlyDSLMaskProgram:
    instructions: tuple[tuple[Any, ...], ...]
    output: int
    buffer_shapes: tuple[tuple[int, ...], ...]
    buffer_strides: tuple[tuple[int, ...], ...]

    @property
    def buffer_count(self) -> int:
        return len(self.buffer_shapes)


def lower_flydsl_mask_graph(
    graph_module,
    mask_mod_other_buffers,
    *,
    max_buffers: int = 4,
) -> tuple[FlyDSLMaskProgram | None, str]:
    """Lower a supported mask FX graph into FlyDSL mask bytecode."""
    nodes = list(graph_module.graph.nodes)
    placeholders = [node for node in nodes if node.op == "placeholder"]
    outputs = [node for node in nodes if node.op == "output"]
    if len(outputs) != 1 or len(placeholders) < 4:
        return None, "mask_mod must have four index placeholders and one output"

    captures = placeholders[4:]
    if len(captures) != len(mask_mod_other_buffers):
        return None, "mask_mod capture placeholders do not match captured buffers"
    if len(captures) > max_buffers:
        return None, f"mask_mod supports at most {max_buffers} captured buffers"

    try:
        buffer_shapes = tuple(
            tuple(V.graph.sizevars.guard_int(value) for value in buffer.get_size())
            for buffer in mask_mod_other_buffers
        )
        buffer_strides = tuple(
            tuple(V.graph.sizevars.guard_int(value) for value in buffer.get_stride())
            for buffer in mask_mod_other_buffers
        )
        buffer_dtypes = tuple(buffer.get_dtype() for buffer in mask_mod_other_buffers)
    except (AttributeError, TypeError, ValueError):
        return None, "mask_mod requires static captured-buffer shapes and strides"

    if any(dtype != torch.int32 for dtype in buffer_dtypes):
        return None, "mask_mod captured buffers must be int32"
    if any(
        not shape
        or len(shape) > 4
        or len(shape) != len(stride)
        or any(value <= 0 for value in stride)
        for shape, stride in zip(buffer_shapes, buffer_strides)
    ):
        return None, "mask_mod captured buffers require rank 1-4 positive strides"

    capture_index = {node: index for index, node in enumerate(captures)}
    value_ids: dict[Node, int] = {
        placeholders[0]: 0,
        placeholders[1]: 1,
        placeholders[2]: 2,
        placeholders[3]: 3,
    }
    constant_ids: dict[tuple[str, int], int] = {}
    instructions: list[tuple[Any, ...]] = []

    def append(instruction: tuple[Any, ...]) -> int:
        value_id = 4 + len(instructions)
        instructions.append(instruction)
        return value_id

    def value_id(value) -> int:
        if isinstance(value, Node):
            if value not in value_ids:
                raise NotImplementedError(
                    f"mask_mod value {value.name} is not scalar-lowered"
                )
            return value_ids[value]
        if isinstance(value, bool):
            key = ("bool", int(value))
            if key not in constant_ids:
                constant_ids[key] = append(("const_bool", bool(value)))
            return constant_ids[key]
        if isinstance(value, int):
            key = ("int", value)
            if key not in constant_ids:
                constant_ids[key] = append(("const_i32", value))
            return constant_ids[key]
        raise NotImplementedError(f"mask_mod scalar constant {value!r} is unsupported")

    try:
        for node in nodes:
            if node.op in ("placeholder", "output"):
                continue
            if node.op != "call_function":
                raise NotImplementedError(
                    f"mask_mod node kind {node.op!r} is unsupported"
                )

            target = node.target
            if target == torch.ops.aten.full.default:
                shape, fill = node.args[:2]
                if list(shape) != [] or not isinstance(fill, (bool, int)):
                    raise NotImplementedError(
                        "mask_mod aten.full supports scalar bool/int constants only"
                    )
                value_ids[node] = value_id(fill)
                continue

            if target == torch.ops.aten.index.Tensor:
                source, indices = node.args[:2]
                if source not in capture_index:
                    raise NotImplementedError(
                        "mask_mod indexing is supported only for captured buffers"
                    )
                buffer_index = capture_index[source]
                index_values = tuple(value_id(index) for index in indices)
                if len(index_values) != len(buffer_shapes[buffer_index]):
                    raise NotImplementedError(
                        "mask_mod captured-buffer index rank mismatch"
                    )
                value_ids[node] = append(("load_i32", buffer_index, index_values))
                continue

            binary_op = _BINARY_TARGETS.get(target)
            if binary_op is not None:
                lhs, rhs = node.args[:2]
                rhs_id = value_id(rhs)
                if target in _ALPHA_TARGETS:
                    alpha = node.kwargs.get("alpha", 1)
                    if not isinstance(alpha, int):
                        raise NotImplementedError(
                            "mask_mod aten.add/sub supports integer alpha only"
                        )
                    if alpha != 1:
                        rhs_id = append(("mul", rhs_id, value_id(int(alpha))))
                value_ids[node] = append((binary_op, value_id(lhs), rhs_id))
                continue

            unary_op = _UNARY_TARGETS.get(target)
            if unary_op is not None:
                value_ids[node] = append((unary_op, value_id(node.args[0])))
                continue

            raise NotImplementedError(f"mask_mod operation {target!r} is unsupported")

        output_value = outputs[0].args[0]
        if isinstance(output_value, (tuple, list)):
            if len(output_value) != 1:
                raise NotImplementedError("mask_mod must return one scalar")
            output_value = output_value[0]
        output_id = value_id(output_value)
    except NotImplementedError as error:
        return None, str(error)

    return (
        FlyDSLMaskProgram(
            instructions=tuple(instructions),
            output=output_id,
            buffer_shapes=buffer_shapes,
            buffer_strides=buffer_strides,
        ),
        "",
    )
