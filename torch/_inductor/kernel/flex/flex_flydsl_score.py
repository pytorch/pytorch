# mypy: allow-untyped-defs

from __future__ import annotations

import math
import operator
from dataclasses import dataclass
from typing import Any

import torch
from torch.fx import Node
from torch.utils._ordered_set import OrderedSet

from ...virtualized import V
from ..vendored_templates.flydsl.kernels.flex_attn_bwd_utils import (
    CAPTURE_GRAD_FULL,
    CAPTURE_GRAD_INVARIANT,
    CAPTURE_GRAD_KEY,
    CAPTURE_GRAD_QUERY,
    CAPTURE_GRAD_RELATIVE,
    CAPTURE_GRAD_UNUSED,
)


# Value kinds tracked while lowering. Integer arithmetic follows the mask
# evaluator's int32 semantics; every floating-point value is computed in FP32,
# matching the FP32 score registers used by the Triton template.
_F32 = "f32"
_I32 = "i32"
_BOOL = "bool"

_FLOAT_DTYPES = OrderedSet([torch.float32, torch.bfloat16, torch.float16])
_INTEGER_DTYPES = OrderedSet(
    [
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
        torch.uint16,
        torch.uint32,
        torch.uint64,
    ]
)

_ARITHMETIC_TARGETS = {
    operator.add: "add",
    torch.ops.aten.add.Tensor: "add",
    torch.ops.aten.add.Scalar: "add",
    operator.sub: "sub",
    torch.ops.aten.sub.Tensor: "sub",
    torch.ops.aten.sub.Scalar: "sub",
    operator.mul: "mul",
    torch.ops.aten.mul.Tensor: "mul",
    torch.ops.aten.mul.Scalar: "mul",
    torch.ops.aten.maximum.default: "max",
    torch.ops.aten.minimum.default: "min",
}

_ALPHA_TARGETS = (
    torch.ops.aten.add.Tensor,
    torch.ops.aten.add.Scalar,
    torch.ops.aten.sub.Tensor,
    torch.ops.aten.sub.Scalar,
)

_RSUB_TARGETS = (torch.ops.aten.rsub.Scalar, torch.ops.aten.rsub.Tensor)

_TRUE_DIVIDE_TARGETS = (
    operator.truediv,
    torch.ops.aten.div.Tensor,
    torch.ops.aten.div.Scalar,
)

_FLOOR_DIVIDE_TARGETS = (operator.floordiv, torch.ops.aten.floor_divide.default)

_REMAINDER_TARGETS = (
    operator.mod,
    torch.ops.aten.remainder.Tensor,
    torch.ops.aten.remainder.Scalar,
)

_COMPARISON_TARGETS = {
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
}

_LOGICAL_TARGETS = {
    operator.and_: "and",
    torch.ops.aten.bitwise_and.Tensor: "and",
    torch.ops.aten.bitwise_and.Scalar: "and",
    torch.ops.aten.logical_and.default: "and",
    operator.or_: "or",
    torch.ops.aten.bitwise_or.Tensor: "or",
    torch.ops.aten.bitwise_or.Scalar: "or",
    torch.ops.aten.logical_or.default: "or",
}

_NOT_TARGETS = (
    operator.invert,
    torch.ops.aten.bitwise_not.default,
    torch.ops.aten.logical_not.default,
)

_NEG_TARGETS = (operator.neg, torch.ops.aten.neg.default)
_ABS_TARGETS = (abs, operator.abs, torch.ops.aten.abs.default)

# Unary ops that always produce a floating-point result.
_FLOAT_UNARY_TARGETS = {
    torch.ops.aten.exp.default: "exp",
    torch.ops.aten.exp2.default: "exp2",
    torch.ops.aten.log.default: "log",
    torch.ops.aten.log2.default: "log2",
    torch.ops.aten.tanh.default: "tanh",
    torch.ops.aten.sigmoid.default: "sigmoid",
    torch.ops.aten.sqrt.default: "sqrt",
    torch.ops.aten.rsqrt.default: "rsqrt",
    torch.ops.aten.reciprocal.default: "recip",
    torch.ops.aten.sin.default: "sin",
    torch.ops.aten.cos.default: "cos",
}

_IDENTITY_TARGETS = (
    torch.ops.aten.clone.default,
    torch.ops.aten.alias.default,
    torch.ops.aten.lift_fresh_copy.default,
)

_CONVERT_TARGETS = (
    torch.ops.prims.convert_element_type.default,
    torch.ops.aten._to_copy.default,
)

_CONSTANT_TARGETS = (torch.ops.aten.full.default, torch.ops.aten.scalar_tensor.default)

_WHERE_TARGETS = (
    torch.ops.aten.where.self,
    torch.ops.aten.where.ScalarSelf,
    torch.ops.aten.where.ScalarOther,
    torch.ops.aten.where.Scalar,
)

_CLAMP_TARGETS = (
    torch.ops.aten.clamp.default,
    torch.ops.aten.clamp_min.default,
    torch.ops.aten.clamp_max.default,
)

_POW_TARGETS = (torch.ops.aten.pow.Tensor_Scalar, operator.pow)


@dataclass(frozen=True)
class FlyDSLScoreProgram:
    """Typed bytecode for one score_mod graph.

    Slots 0-4 hold ``score`` (FP32), ``batch``, ``head``, ``query_index`` and
    ``key_index`` (int32). Every instruction appends one slot.
    """

    instructions: tuple[tuple[Any, ...], ...]
    output: int
    buffer_shapes: tuple[tuple[int, ...], ...]
    buffer_strides: tuple[tuple[int, ...], ...]

    @property
    def buffer_count(self) -> int:
        return len(self.buffer_shapes)


def lower_flydsl_score_graph(
    graph_module,
    score_mod_other_buffers,
    *,
    max_buffers: int = 4,
) -> tuple[FlyDSLScoreProgram | None, str]:
    """Lower a supported score_mod FX graph into FlyDSL score bytecode."""
    nodes = list(graph_module.graph.nodes)
    placeholders = [node for node in nodes if node.op == "placeholder"]
    outputs = [node for node in nodes if node.op == "output"]
    if len(outputs) != 1 or len(placeholders) < 5:
        return None, "score_mod must have five input placeholders and one output"

    captures = placeholders[5:]
    if len(captures) != len(score_mod_other_buffers):
        return None, "score_mod capture placeholders do not match captured buffers"
    if len(captures) > max_buffers:
        return None, f"score_mod supports at most {max_buffers} captured buffers"

    try:
        buffer_shapes = tuple(
            tuple(V.graph.sizevars.guard_int(value) for value in buffer.get_size())
            for buffer in score_mod_other_buffers
        )
        buffer_strides = tuple(
            tuple(V.graph.sizevars.guard_int(value) for value in buffer.get_stride())
            for buffer in score_mod_other_buffers
        )
        buffer_dtypes = tuple(buffer.get_dtype() for buffer in score_mod_other_buffers)
    except (AttributeError, TypeError, ValueError):
        return None, "score_mod requires static captured-buffer shapes and strides"

    buffer_kinds = []
    for dtype in buffer_dtypes:
        if dtype in _FLOAT_DTYPES:
            buffer_kinds.append(_F32)
        elif dtype == torch.int32:
            buffer_kinds.append(_I32)
        else:
            return None, (
                "score_mod captured buffers must be float32, bfloat16, float16, "
                f"or int32, got {dtype}"
            )
    if any(
        len(shape) > 4
        or len(shape) != len(stride)
        or any(value < 0 for value in stride)
        for shape, stride in zip(buffer_shapes, buffer_strides)
    ):
        return None, "score_mod captured buffers require rank 0-4 non-negative strides"

    capture_index = {node: index for index, node in enumerate(captures)}
    value_ids: dict[Node, int] = {
        placeholder: index for index, placeholder in enumerate(placeholders[:5])
    }
    kinds: list[str] = [_F32, _I32, _I32, _I32, _I32]
    constant_ids: dict[tuple[str, Any], int] = {}
    conversion_ids: dict[tuple[str, int], int] = {}
    instructions: list[tuple[Any, ...]] = []

    def append(instruction: tuple[Any, ...], kind: str) -> int:
        value_id = 5 + len(instructions)
        instructions.append(instruction)
        kinds.append(kind)
        return value_id

    def constant(value, kind: str) -> int:
        if kind == _BOOL:
            key: tuple[str, Any] = (kind, bool(value))
            instruction: tuple[Any, ...] = ("const_bool", bool(value))
        elif kind == _I32:
            value = int(value)
            if not -(1 << 31) <= value < (1 << 31):
                raise NotImplementedError(
                    f"score_mod int constant {value} does not fit int32"
                )
            key = (kind, value)
            instruction = ("const_i32", value)
        else:
            value = float(value)
            if math.isnan(value):
                raise NotImplementedError("score_mod NaN constants are unsupported")
            # Hex text is exact, distinguishes signed zeros, and renders inf as a
            # string literal in the generated template module.
            key = (kind, value.hex())
            instruction = ("const_f32", value.hex())
        if key not in constant_ids:
            constant_ids[key] = append(instruction, kind)
        return constant_ids[key]

    def load_capture(source: Node, index_values: tuple[int, ...]) -> int:
        buffer_index = capture_index[source]
        if len(index_values) != len(buffer_shapes[buffer_index]):
            raise NotImplementedError("score_mod captured-buffer index rank mismatch")
        if any(kinds[index] == _F32 for index in index_values):
            raise NotImplementedError("score_mod captured-buffer index must be integer")
        index_values = tuple(as_kind(index, _I32) for index in index_values)
        kind = buffer_kinds[buffer_index]
        op = "load_f32" if kind == _F32 else "load_i32"
        return append((op, buffer_index, index_values), kind)

    def value_id(value) -> int:
        if isinstance(value, Node):
            if value in capture_index:
                # A rank-0 capture can be used directly as a scalar operand.
                if buffer_shapes[capture_index[value]]:
                    raise NotImplementedError(
                        "score_mod captured tensors must be indexed to scalars"
                    )
                if value not in value_ids:
                    value_ids[value] = load_capture(value, ())
                return value_ids[value]
            if value not in value_ids:
                raise NotImplementedError(
                    f"score_mod value {value.name} is not scalar-lowered"
                )
            return value_ids[value]
        if isinstance(value, bool):
            return constant(value, _BOOL)
        if isinstance(value, int):
            return constant(value, _I32)
        if isinstance(value, float):
            return constant(value, _F32)
        raise NotImplementedError(f"score_mod scalar constant {value!r} is unsupported")

    def small_integer_as_float(value: int, depth: int = 0) -> int | None:
        """Float form of an int built from b/h/q/kv and small constants.

        ``float(q - kv)`` becomes ``float(q) - float(kv)``: exact while all
        terms stay below 2**24, and the per-row and per-column conversions are
        shared across a tile's elements.
        """
        if value in (1, 2, 3, 4):
            return as_kind(value, _F32, distribute=False)
        if value < 5 or depth > 3:
            return None
        op, *operands = instructions[value - 5]
        if op == "const_i32" and abs(operands[0]) < (1 << 20):
            return constant(float(operands[0]), _F32)
        if op not in ("add", "sub"):
            return None
        converted = [small_integer_as_float(operand, depth + 1) for operand in operands]
        if None in converted:
            return None
        return append(("f" + op, *converted), _F32)

    def as_kind(value: int, kind: str, distribute: bool = True) -> int:
        current = kinds[value]
        if current == kind:
            return value
        key = (kind, value)
        if key in conversion_ids:
            return conversion_ids[key]
        if kind == _F32 and current == _I32 and distribute:
            converted = small_integer_as_float(value)
            if converted is not None:
                conversion_ids[key] = converted
                return converted
        if kind == _F32:
            op = "b2f" if current == _BOOL else "i2f"
        elif kind == _I32:
            op = "b2i" if current == _BOOL else "f2i"
        else:
            op = "f2b" if current == _F32 else "i2b"
        conversion_ids[key] = append((op, value), kind)
        return conversion_ids[key]

    def promote(*values: int) -> str:
        value_kinds = [kinds[value] for value in values]
        if _F32 in value_kinds:
            return _F32
        if _I32 in value_kinds:
            return _I32
        return _BOOL

    def arithmetic(op: str, lhs: int, rhs: int) -> int:
        kind = promote(lhs, rhs)
        if kind == _BOOL:
            # Torch promotes bool arithmetic to integers except for max/min.
            kind = _I32
        lhs, rhs = as_kind(lhs, kind), as_kind(rhs, kind)
        name = ("f" + op) if kind == _F32 else op
        return append((name, lhs, rhs), kind)

    def compare(op: str, lhs: int, rhs: int) -> int:
        kind = promote(lhs, rhs)
        if kind == _BOOL:
            kind = _I32
        lhs, rhs = as_kind(lhs, kind), as_kind(rhs, kind)
        name = ("f" + op) if kind == _F32 else op
        return append((name, lhs, rhs), _BOOL)

    def float_unary(op: str, operand: int) -> int:
        return append((op, as_kind(operand, _F32)), _F32)

    def select(condition: int, lhs: int, rhs: int) -> int:
        condition = as_kind(condition, _BOOL)
        kind = promote(lhs, rhs)
        return append(
            ("select", condition, as_kind(lhs, kind), as_kind(rhs, kind)), kind
        )

    def dtype_kind(dtype) -> str:
        if dtype == torch.bool:
            return _BOOL
        if dtype in _INTEGER_DTYPES:
            return _I32
        if dtype in _FLOAT_DTYPES or dtype == torch.float64:
            return _F32
        raise NotImplementedError(f"score_mod dtype {dtype} is unsupported")

    def scalar_kind(value, dtype) -> str:
        if dtype is not None:
            return dtype_kind(dtype)
        if isinstance(value, bool):
            return _BOOL
        if isinstance(value, int):
            return _I32
        return _F32

    def integer_power(base: int, exponent: int) -> int:
        result = None
        power = base
        remaining = exponent
        while remaining:
            if remaining & 1:
                result = power if result is None else arithmetic("mul", result, power)
            remaining >>= 1
            if remaining:
                power = arithmetic("mul", power, power)
        return result

    try:
        for node in nodes:
            if node.op in ("placeholder", "output"):
                continue
            if node.op != "call_function":
                raise NotImplementedError(
                    f"score_mod node kind {node.op!r} is unsupported"
                )

            target = node.target
            args = node.args
            kwargs = node.kwargs

            if target in _CONSTANT_TARGETS:
                if target == torch.ops.aten.full.default:
                    shape, fill = args[:2]
                    if list(shape) != []:
                        raise NotImplementedError(
                            "score_mod aten.full supports scalar constants only"
                        )
                else:
                    fill = args[0]
                if not isinstance(fill, (bool, int, float)):
                    raise NotImplementedError(
                        f"score_mod constant {fill!r} is unsupported"
                    )
                dtype = kwargs.get("dtype")
                if dtype is None:
                    dtype = getattr(node.meta.get("val"), "dtype", None)
                value_ids[node] = constant(fill, scalar_kind(fill, dtype))
                continue

            if target == torch.ops.aten.index.Tensor:
                source, indices = args[:2]
                if source not in capture_index:
                    raise NotImplementedError(
                        "score_mod indexing is supported only for captured buffers"
                    )
                if any(index is None for index in indices):
                    raise NotImplementedError(
                        "score_mod indexing requires one index per dimension"
                    )
                value_ids[node] = load_capture(
                    source, tuple(value_id(index) for index in indices)
                )
                continue

            if target in _IDENTITY_TARGETS:
                value_ids[node] = value_id(args[0])
                continue

            if target in _CONVERT_TARGETS:
                if target == torch.ops.prims.convert_element_type.default:
                    dtype = args[1]
                else:
                    dtype = kwargs.get("dtype")
                    if dtype is None:
                        dtype = getattr(node.meta.get("val"), "dtype", None)
                operand = value_id(args[0])
                if dtype is None:
                    value_ids[node] = operand
                else:
                    value_ids[node] = as_kind(operand, dtype_kind(dtype))
                continue

            arithmetic_op = _ARITHMETIC_TARGETS.get(target)
            if arithmetic_op is not None:
                lhs, rhs = value_id(args[0]), value_id(args[1])
                if target in _ALPHA_TARGETS:
                    alpha = kwargs.get("alpha", 1)
                    if not isinstance(alpha, (int, float)) or isinstance(alpha, bool):
                        raise NotImplementedError(
                            "score_mod aten.add/sub supports numeric alpha only"
                        )
                    if alpha != 1:
                        rhs = arithmetic("mul", rhs, value_id(alpha))
                value_ids[node] = arithmetic(arithmetic_op, lhs, rhs)
                continue

            if target in _RSUB_TARGETS:
                lhs, rhs = value_id(args[0]), value_id(args[1])
                alpha = kwargs.get("alpha", 1)
                if alpha != 1:
                    lhs = arithmetic("mul", lhs, value_id(alpha))
                value_ids[node] = arithmetic("sub", rhs, lhs)
                continue

            if target in _TRUE_DIVIDE_TARGETS or (
                target == torch.ops.aten.div.Tensor_mode
                and kwargs.get("rounding_mode") is None
            ):
                lhs = as_kind(value_id(args[0]), _F32)
                rhs = as_kind(value_id(args[1]), _F32)
                value_ids[node] = append(("fdiv", lhs, rhs), _F32)
                continue

            if target in _FLOOR_DIVIDE_TARGETS or (
                target == torch.ops.aten.div.Tensor_mode
                and kwargs.get("rounding_mode") == "floor"
            ):
                lhs, rhs = value_id(args[0]), value_id(args[1])
                if promote(lhs, rhs) == _F32:
                    raise NotImplementedError(
                        "score_mod floor division supports integers only"
                    )
                value_ids[node] = arithmetic("floordiv", lhs, rhs)
                continue

            if target in _REMAINDER_TARGETS:
                lhs, rhs = value_id(args[0]), value_id(args[1])
                if promote(lhs, rhs) == _F32:
                    raise NotImplementedError(
                        "score_mod remainder supports integers only"
                    )
                value_ids[node] = arithmetic("remainder", lhs, rhs)
                continue

            comparison_op = _COMPARISON_TARGETS.get(target)
            if comparison_op is not None:
                value_ids[node] = compare(
                    comparison_op, value_id(args[0]), value_id(args[1])
                )
                continue

            logical_op = _LOGICAL_TARGETS.get(target)
            if logical_op is not None:
                lhs, rhs = value_id(args[0]), value_id(args[1])
                if target in (
                    torch.ops.aten.logical_and.default,
                    torch.ops.aten.logical_or.default,
                ) or (kinds[lhs] == _BOOL and kinds[rhs] == _BOOL):
                    value_ids[node] = append(
                        (logical_op, as_kind(lhs, _BOOL), as_kind(rhs, _BOOL)), _BOOL
                    )
                    continue
                raise NotImplementedError(
                    "score_mod bitwise and/or supports bool operands only"
                )

            if target in _NOT_TARGETS:
                operand = value_id(args[0])
                if target != torch.ops.aten.logical_not.default and (
                    kinds[operand] != _BOOL
                ):
                    raise NotImplementedError(
                        "score_mod bitwise not supports bool operands only"
                    )
                value_ids[node] = append(("not", as_kind(operand, _BOOL)), _BOOL)
                continue

            if target in _NEG_TARGETS:
                operand = value_id(args[0])
                if kinds[operand] == _F32:
                    value_ids[node] = append(("fneg", operand), _F32)
                else:
                    zero = constant(0, _I32)
                    value_ids[node] = arithmetic("sub", zero, operand)
                continue

            if target in _ABS_TARGETS:
                operand = value_id(args[0])
                if kinds[operand] == _F32:
                    value_ids[node] = append(("fabs", operand), _F32)
                else:
                    operand = as_kind(operand, _I32)
                    zero = constant(0, _I32)
                    negative = compare("lt", operand, zero)
                    value_ids[node] = select(
                        negative, arithmetic("sub", zero, operand), operand
                    )
                continue

            unary_op = _FLOAT_UNARY_TARGETS.get(target)
            if unary_op is not None:
                value_ids[node] = float_unary(unary_op, value_id(args[0]))
                continue

            if target in _POW_TARGETS:
                base = value_id(args[0])
                exponent = args[1]
                if isinstance(exponent, bool) or not isinstance(exponent, (int, float)):
                    raise NotImplementedError(
                        "score_mod pow supports constant scalar exponents only"
                    )
                if float(exponent).is_integer() and 1 <= exponent <= 64:
                    if isinstance(exponent, float):
                        base = as_kind(base, _F32)
                    value_ids[node] = integer_power(base, int(exponent))
                elif exponent == 0.5:
                    value_ids[node] = float_unary("sqrt", base)
                elif exponent == -0.5:
                    value_ids[node] = float_unary("rsqrt", base)
                elif exponent == -1:
                    value_ids[node] = float_unary("recip", base)
                else:
                    raise NotImplementedError(
                        f"score_mod pow exponent {exponent!r} is unsupported"
                    )
                continue

            if target in _WHERE_TARGETS:
                condition, lhs, rhs = args[:3]
                value_ids[node] = select(
                    value_id(condition), value_id(lhs), value_id(rhs)
                )
                continue

            if target in _CLAMP_TARGETS:
                operand = value_id(args[0])
                if target == torch.ops.aten.clamp.default:
                    lower = args[1] if len(args) > 1 else kwargs.get("min")
                    upper = args[2] if len(args) > 2 else kwargs.get("max")
                elif target == torch.ops.aten.clamp_min.default:
                    lower, upper = args[1], None
                else:
                    lower, upper = None, args[1]
                if lower is not None:
                    operand = arithmetic("max", operand, value_id(lower))
                if upper is not None:
                    operand = arithmetic("min", operand, value_id(upper))
                value_ids[node] = operand
                continue

            raise NotImplementedError(f"score_mod operation {target!r} is unsupported")

        output_value = outputs[0].args[0]
        if isinstance(output_value, (tuple, list)):
            if len(output_value) != 1:
                raise NotImplementedError("score_mod must return one scalar")
            output_value = output_value[0]
        output_id = as_kind(value_id(output_value), _F32)
    except NotImplementedError as error:
        return None, str(error)

    return (
        FlyDSLScoreProgram(
            instructions=tuple(instructions),
            output=output_id,
            buffer_shapes=buffer_shapes,
            buffer_strides=buffer_strides,
        ),
        "",
    )


@dataclass(frozen=True)
class FlyDSLCaptureGrad:
    """How the backward accumulates the gradient of one trainable score capture.

    ``dims`` describes each buffer dimension's index: ``("b",)``, ``("h",)``,
    ``("q",)``, ``("kv",)`` or ``("rel", sign, offset)`` for
    ``sign * (q - kv) + offset``.
    """

    buffer_index: int
    kind: str
    load_slot: int
    dims: tuple[tuple[Any, ...], ...]


def _relative_index(instructions, value: int) -> tuple[int, int] | None:
    """Return ``(sign, offset)`` if ``value`` is ``sign * (q - kv) + offset``."""
    if value < 5:
        return None
    op, *operands = instructions[value - 5]
    if op == "sub" and tuple(operands) == (3, 4):
        return 1, 0
    if op == "sub" and tuple(operands) == (4, 3):
        return -1, 0
    if op not in ("add", "sub"):
        return None

    def constant(operand):
        if operand < 5:
            return None
        instruction = instructions[operand - 5]
        return instruction[1] if instruction[0] == "const_i32" else None

    lhs, rhs = operands
    lhs_relative, rhs_relative = (
        _relative_index(instructions, lhs),
        _relative_index(instructions, rhs),
    )
    lhs_constant, rhs_constant = constant(lhs), constant(rhs)
    if lhs_relative is not None and rhs_constant is not None:
        sign, offset = lhs_relative
        return sign, offset + (rhs_constant if op == "add" else -rhs_constant)
    if op == "add" and rhs_relative is not None and lhs_constant is not None:
        sign, offset = rhs_relative
        return sign, offset + lhs_constant
    if op == "sub" and rhs_relative is not None and lhs_constant is not None:
        sign, offset = rhs_relative
        return -sign, lhs_constant - offset
    return None


def classify_capture_gradient(
    program: FlyDSLScoreProgram,
    buffer_index: int,
    *,
    batch: int,
    heads: int,
    sequence_length: int,
) -> tuple[FlyDSLCaptureGrad | None, str]:
    """Classify a trainable capture so its gradient needs no atomics."""
    instructions = program.instructions
    loads = [
        slot
        for slot, instruction in enumerate(instructions, 5)
        if instruction[0] in ("load_f32", "load_i32") and instruction[1] == buffer_index
    ]
    if not loads:
        return FlyDSLCaptureGrad(buffer_index, CAPTURE_GRAD_UNUSED, -1, ()), ""
    if len(loads) != 1:
        return None, "a trainable score_mod capture must be indexed once"
    load_slot = loads[0]
    if instructions[load_slot - 5][0] != "load_f32":
        return None, "trainable score_mod captures must be floating point"

    shape = program.buffer_shapes[buffer_index]
    plain = {1: "b", 2: "h", 3: "q", 4: "kv"}
    extents = {"b": batch, "h": heads, "q": sequence_length, "kv": sequence_length}
    dims = []
    for dimension, value in enumerate(instructions[load_slot - 5][2]):
        if value in plain:
            name = plain[value]
            if shape[dimension] != extents[name]:
                return None, (
                    f"trainable score_mod capture dimension {dimension} indexed by "
                    f"{name} must have size {extents[name]}"
                )
            dims.append((name,))
            continue
        relative = _relative_index(instructions, value)
        if relative is None:
            return None, (
                "trainable score_mod capture indices must be b, h, q_idx, kv_idx, "
                "or +-(q_idx - kv_idx) + constant"
            )
        dims.append(("rel", *relative))

    names = [dim[0] for dim in dims]
    if len(set(names)) != len(names):
        return None, "trainable score_mod capture repeats an index variable"
    if "rel" in names and ("q" in names or "kv" in names):
        return None, "trainable score_mod capture mixes relative and absolute positions"
    if "q" in names and "kv" in names:
        if (batch > 1 and "b" not in names) or (heads > 1 and "h" not in names):
            return None, (
                "a trainable [q_idx, kv_idx] score_mod capture must also be "
                "indexed by every batch and head dimension larger than one"
            )
        if program.buffer_strides[buffer_index] != tuple(_contiguous_stride(shape)):
            return (
                None,
                "a trainable [q_idx, kv_idx] score_mod capture must be contiguous",
            )
        kind = CAPTURE_GRAD_FULL
    elif "rel" in names:
        kind = CAPTURE_GRAD_RELATIVE
    elif "kv" in names:
        kind = CAPTURE_GRAD_KEY
    elif "q" in names:
        kind = CAPTURE_GRAD_QUERY
    else:
        kind = CAPTURE_GRAD_INVARIANT
    return FlyDSLCaptureGrad(buffer_index, kind, load_slot, tuple(dims)), ""


def _contiguous_stride(shape) -> list[int]:
    stride = [1] * len(shape)
    for index in range(len(shape) - 2, -1, -1):
        stride[index] = stride[index + 1] * max(shape[index + 1], 1)
    return stride
