# mypy: allow-untyped-defs

from __future__ import annotations

import math
import operator
from dataclasses import dataclass
from typing import Any

import sympy

import torch
from torch.fx import Node
from torch.utils._ordered_set import OrderedSet

from ...virtualized import V


aten = torch.ops.aten
prims = torch.ops.prims


def _named(name, packet, *overloads, python_op=None):
    """Map ``python_op`` and ``overloads`` of aten op ``packet`` to ``name``."""
    targets = [getattr(getattr(aten, packet), overload) for overload in overloads]
    return dict.fromkeys([python_op, *targets] if python_op else targets, name)


_TENSOR_SCALAR = ("Tensor", "Scalar")
_ARITHMETIC_TARGETS = {
    **_named("add", "add", *_TENSOR_SCALAR, python_op=operator.add),
    **_named("sub", "sub", *_TENSOR_SCALAR, python_op=operator.sub),
    **_named("mul", "mul", *_TENSOR_SCALAR, python_op=operator.mul),
    **_named(
        "floordiv", "floor_divide", "default", "Scalar", python_op=operator.floordiv
    ),
    **_named("remainder", "remainder", *_TENSOR_SCALAR, python_op=operator.mod),
    **_named("fmod", "fmod", *_TENSOR_SCALAR),
}
_COMPARISON_TARGETS = {
    target: name
    for name in ("ge", "gt", "le", "lt", "eq", "ne")
    for target in _named(name, name, *_TENSOR_SCALAR, python_op=getattr(operator, name))
}
_BITWISE_TARGETS = {
    **_named("and", "bitwise_and", *_TENSOR_SCALAR, python_op=operator.and_),
    **_named("or", "bitwise_or", *_TENSOR_SCALAR, python_op=operator.or_),
    **_named("xor", "bitwise_xor", *_TENSOR_SCALAR, python_op=operator.xor),
}
_SHIFT_TARGETS = {
    **_named("shl", "__lshift__", "Scalar", "Tensor", python_op=operator.lshift),
    **_named("shl", "bitwise_left_shift", "Tensor", "Tensor_Scalar"),
    **_named("shr", "__rshift__", "Scalar", "Tensor", python_op=operator.rshift),
    **_named("shr", "bitwise_right_shift", "Tensor", "Tensor_Scalar"),
}

_LOGICAL_TARGETS = {
    aten.logical_and.default: "and",
    aten.logical_or.default: "or",
    aten.logical_xor.default: "xor",
}

_MINMAX_TARGETS = {
    aten.minimum.default: "min",
    aten.maximum.default: "max",
}

_UNARY_TARGETS = {
    operator.invert: "not",
    aten.bitwise_not.default: "not",
    operator.neg: "neg",
    aten.neg.default: "neg",
    operator.abs: "abs",
    aten.abs.default: "abs",
}

_ALPHA_TARGETS = (
    aten.add.Tensor,
    aten.add.Scalar,
    aten.sub.Tensor,
    aten.sub.Scalar,
)
_TRUE_DIVIDE_TARGETS = (operator.truediv, aten.div.Tensor, aten.div.Scalar)
_ROUNDING_DIVIDE_TARGETS = (aten.div.Tensor_mode, aten.div.Scalar_mode)
_CAST_TARGETS = (
    prims.convert_element_type.default,
    aten._to_copy.default,
    aten.to.dtype,
)
_IDENTITY_TARGETS = (aten.clone.default, aten.alias.default)
_WHERE_TARGETS = (
    aten.where.self,
    aten.where.ScalarSelf,
    aten.where.ScalarOther,
    aten.where.Scalar,
)
_CONSTANT_TARGETS = (aten.full.default, aten.scalar_tensor.default)

# Bytecode value kinds. Narrow integers live sign- or zero-extended in i32 and
# fp16/bf16 values are computed in f32, as in Inductor's Triton codegen.
_VALUE_KINDS = {
    torch.bool: "bool",
    torch.int8: "i32",
    torch.int16: "i32",
    torch.int32: "i32",
    torch.uint8: "i32",
    torch.int64: "i64",
    torch.float16: "f32",
    torch.bfloat16: "f32",
    torch.float32: "f32",
}
_ROUNDED_FLOAT_KINDS = {torch.float16: "fp16", torch.bfloat16: "bf16"}
_NARROW_INTEGER_BITS = {torch.int8: 8, torch.int16: 16, torch.uint8: 8}
_CONSTANT_DTYPES = {"bool": torch.bool, "i32": torch.int32, "i64": torch.int64}
_MAX_CAPTURE_ELEMENTS = 1 << 31
_MAX_CAPTURE_BYTES = 1 << 32


@dataclass(frozen=True)
class FlyDSLMaskProgram:
    instructions: tuple[tuple[Any, ...], ...]
    output: int
    buffer_shapes: tuple[tuple[int, ...], ...]
    buffer_strides: tuple[tuple[int, ...], ...]
    buffer_dtypes: tuple[str, ...] = ()

    @property
    def buffer_count(self) -> int:
        return len(self.buffer_shapes)


def _is_cpu_buffer(buffer) -> bool:
    device = getattr(buffer, "get_device", lambda: None)()
    return device is not None and device.type == "cpu"


def _is_size(buffer) -> bool:
    return isinstance(buffer, (int, sympy.Expr))


def mask_capture_tensors(mask_mod_other_buffers):
    """The captured tensors of a mask_mod, without the sizes passed with them."""
    return [buffer for buffer in mask_mod_other_buffers if not _is_size(buffer)]


def prepare_flydsl_mask_captures(mask_mod_other_buffers, device: torch.device):
    """Move 0-d mask captures to ``device`` as [1] tensors; FlyDSL kernel
    arguments need a dim."""
    from ...ir import StorageBox, TensorBox
    from ...lowering import to_device, view

    prepared = []
    for buffer in mask_mod_other_buffers:
        if isinstance(buffer, StorageBox):
            buffer = TensorBox(buffer)
        if isinstance(buffer, TensorBox) and len(buffer.get_size()) == 0:
            if _is_cpu_buffer(buffer):
                buffer = to_device(buffer, device)
            buffer = view(buffer, [1])
        prepared.append(buffer)
    return prepared


def _value_kind(dtype) -> str:
    kind = _VALUE_KINDS.get(dtype)
    if kind is None:
        raise NotImplementedError(f"mask_mod dtype {dtype} is unsupported")
    return kind


def _dtype_name(dtype) -> str:
    return str(dtype).removeprefix("torch.")


def _fits(source, target) -> bool:
    if source == torch.bool:
        return True
    if source.is_floating_point:
        return False
    source_info, target_info = torch.iinfo(source), torch.iinfo(target)
    return target_info.min <= source_info.min and source_info.max <= target_info.max


class _MaskLowering:
    def __init__(self, captures, buffer_shapes, buffer_dtypes):
        self.capture_index = {node: index for index, node in enumerate(captures)}
        self.buffer_shapes = buffer_shapes
        self.buffer_dtypes = buffer_dtypes
        self.instructions: list[tuple[Any, ...]] = []
        self.dtypes = [torch.int32] * 4
        self.nonnegative = OrderedSet([0, 1, 2, 3])
        self.values: dict[Node, int] = {}
        self.constants: dict[tuple[str, Any], int] = {}

    def append(self, instruction, dtype, nonnegative=False) -> int:
        value_id = 4 + len(self.instructions)
        self.instructions.append(instruction)
        self.dtypes.append(dtype)
        if nonnegative:
            self.nonnegative.add(value_id)
        return value_id

    def is_nonnegative(self, *value_ids) -> bool:
        return all(value_id in self.nonnegative for value_id in value_ids)

    def load(self, buffer_index, indices) -> int:
        dtype = self.buffer_dtypes[buffer_index]
        if dtype == torch.int32:
            instruction = ("load_i32", buffer_index, indices)
        else:
            instruction = ("load", buffer_index, indices, _dtype_name(dtype))
        return self.append(
            instruction, dtype, nonnegative=dtype in (torch.bool, torch.uint8)
        )

    def node(self, value: Any) -> int:
        if not isinstance(value, Node):
            raise NotImplementedError(f"mask_mod operand {value!r} is unsupported")
        buffer_index = self.capture_index.get(value)
        if (
            value not in self.values
            and buffer_index is not None
            and math.prod(self.buffer_shapes[buffer_index]) == 1
        ):
            indices = (self.constant(0, torch.int32),) * len(
                self.buffer_shapes[buffer_index]
            )
            self.values[value] = self.load(buffer_index, indices)
        if value not in self.values:
            raise NotImplementedError(
                f"mask_mod value {value.name} is not scalar-lowered"
            )
        return self.values[value]

    def dtype_of(self, value):
        if isinstance(value, Node):
            return self.dtypes[self.node(value)]
        if isinstance(value, (bool, int, float)):
            return value
        raise NotImplementedError(f"mask_mod scalar constant {value!r} is unsupported")

    def result_type(self, *operands):
        """Promote like PyTorch for 0-d tensors and Python scalars."""
        samples = []
        for operand in operands:
            dtype = self.dtype_of(operand)
            samples.append(
                torch.empty((), dtype=dtype)
                if isinstance(dtype, torch.dtype)
                else dtype
            )
        if not any(isinstance(sample, torch.Tensor) for sample in samples):
            raise NotImplementedError("mask_mod operations need a tensor operand")
        result = samples[0]
        for sample in samples[1:]:
            result = torch.empty((), dtype=torch.result_type(result, sample))
        _value_kind(result.dtype)
        return result.dtype

    def result_dtype(self, node: Node, *operands):
        dtype = getattr(node.meta.get("val"), "dtype", None)
        if dtype is None:
            return self.result_type(*operands)
        _value_kind(dtype)
        return dtype

    def constant(self, value, dtype) -> int:
        kind = _value_kind(dtype)
        if kind == "bool":
            key: tuple[str, Any] = ("bool", int(bool(value)))
            instruction: tuple[Any, ...] = ("const_bool", bool(value))
        elif kind == "f32":
            rounded = torch.tensor(float(value), dtype=dtype).item()
            if not math.isfinite(rounded):
                raise NotImplementedError(
                    f"mask_mod float constant {value} is unsupported"
                )
            key = ("f32", rounded)
            instruction = ("const_f32", rounded)
        else:
            if isinstance(value, float):
                raise NotImplementedError(
                    f"mask_mod float constant {value} needs a float operation"
                )
            info = torch.iinfo(dtype)
            if not info.min <= value <= info.max:
                raise NotImplementedError(
                    f"mask_mod int constant {value} does not fit {_dtype_name(dtype)}"
                )
            key = ("int", int(value)) if kind == "i32" else ("i64", int(value))
            instruction = ("const_i32" if kind == "i32" else "const_i64", int(value))
        if key not in self.constants:
            self.constants[key] = self.append(
                instruction,
                _CONSTANT_DTYPES.get(kind, torch.float32),
                nonnegative=kind != "f32" and value >= 0,
            )
        return self.constants[key]

    def wrap(self, value_id, dtype) -> int:
        """Reduce an i32 value to the range of a narrow integer dtype."""
        bits = _NARROW_INTEGER_BITS[dtype]
        if dtype == torch.uint8:
            mask = self.constant((1 << bits) - 1, torch.int32)
            return self.append(("and", value_id, mask), dtype, nonnegative=True)
        shift = self.constant(32 - bits, torch.int32)
        shifted = self.append(("shl", value_id, shift), torch.int32)
        return self.append(("shr", shifted, shift), dtype)

    def narrow(self, value_id, dtype) -> int:
        return self.wrap(value_id, dtype) if dtype in _NARROW_INTEGER_BITS else value_id

    def convert(self, value_id, dtype) -> int:
        source = self.dtypes[value_id]
        if source == dtype:
            return value_id
        kind = _value_kind(dtype)
        if dtype in _ROUNDED_FLOAT_KINDS:
            return self.append(("cast", value_id, _ROUNDED_FLOAT_KINDS[dtype]), dtype)
        if _value_kind(source) != kind:
            value_id = self.append(
                ("cast", value_id, kind),
                dtype,
                nonnegative=kind != "f32"
                and (source == torch.bool or self.is_nonnegative(value_id)),
            )
        if dtype in _NARROW_INTEGER_BITS and not _fits(source, dtype):
            value_id = self.wrap(value_id, dtype)
        return value_id

    def coerce(self, value, dtype) -> int:
        if isinstance(value, Node):
            return self.convert(self.node(value), dtype)
        if isinstance(value, (bool, int, float)):
            return self.constant(value, dtype)
        raise NotImplementedError(f"mask_mod scalar constant {value!r} is unsupported")

    def binary(self, op, lhs, rhs, dtype, result_dtype=None, nonnegative=None):
        rhs_id = self.coerce(rhs, dtype)
        lhs_id = self.coerce(lhs, dtype)
        if nonnegative is None:
            nonnegative = self.is_nonnegative(lhs_id, rhs_id)
        return self.append(
            (op, lhs_id, rhs_id),
            dtype if result_dtype is None else result_dtype,
            nonnegative=nonnegative,
        )

    def combine(self, op, lhs_id, rhs_id, dtype) -> int:
        """``op`` of two values; ``and``/``max`` are non-negative if either is."""
        if op in ("and", "max"):
            nonnegative = self.is_nonnegative(lhs_id) or self.is_nonnegative(rhs_id)
        else:
            nonnegative = self.is_nonnegative(lhs_id, rhs_id)
        return self.append((op, lhs_id, rhs_id), dtype, nonnegative=nonnegative)

    def where(self, condition_id, true_id, false_id) -> int:
        return self.append(
            ("where", condition_id, true_id, false_id),
            self.dtypes[true_id],
            nonnegative=self.is_nonnegative(true_id, false_id),
        )

    def arithmetic(self, node, op, lhs, rhs) -> int:
        dtype = self.result_dtype(node, lhs, rhs)
        kind = _value_kind(dtype)
        if kind == "bool":
            if op not in ("add", "mul"):
                raise NotImplementedError(
                    f"mask_mod {op} on bool values is unsupported"
                )
            return self.binary("or" if op == "add" else "and", lhs, rhs, dtype)
        alpha = node.kwargs.get("alpha", 1) if node.target in _ALPHA_TARGETS else 1
        if not isinstance(alpha, (int, float)) or (
            isinstance(alpha, float) and kind != "f32"
        ):
            raise NotImplementedError(
                "mask_mod aten.add/sub alpha must match the dtype"
            )
        rhs_id = self.coerce(rhs, dtype)
        if alpha != 1:
            rhs_id = self.append(
                ("mul", rhs_id, self.constant(alpha, dtype)),
                dtype,
                nonnegative=self.is_nonnegative(rhs_id) and alpha >= 0,
            )
        lhs_id = self.coerce(lhs, dtype)
        nonnegative = (
            self.is_nonnegative(rhs_id)
            if op == "remainder"
            else op in ("add", "mul", "floordiv")
            and self.is_nonnegative(lhs_id, rhs_id)
        )
        value_id = self.append((op, lhs_id, rhs_id), dtype, nonnegative=nonnegative)
        return value_id if op in ("remainder", "fmod") else self.narrow(value_id, dtype)

    def true_divide(self, node, lhs, rhs) -> int:
        dtype = self.result_dtype(node, lhs, rhs)
        if not dtype.is_floating_point:
            dtype = torch.get_default_dtype()
        _value_kind(dtype)
        return self.binary("div", lhs, rhs, dtype, nonnegative=False)

    def truncate_divide(self, node, lhs, rhs) -> int:
        dtype = self.result_dtype(node, lhs, rhs)
        if _value_kind(dtype) == "f32":
            quotient = self.binary("div", lhs, rhs, dtype, nonnegative=False)
            magnitude = self.append(("abs", quotient), dtype, nonnegative=True)
            floor = self.append(
                ("floordiv", magnitude, self.constant(1.0, dtype)),
                dtype,
                nonnegative=True,
            )
            negated = self.append(("neg", floor), dtype)
            negative = self.append(
                ("lt", quotient, self.constant(0.0, dtype)), torch.bool
            )
            return self.where(negative, negated, floor)
        if _value_kind(dtype) == "bool":
            raise NotImplementedError("mask_mod division of bool values is unsupported")
        rhs_id = self.coerce(rhs, dtype)
        lhs_id = self.coerce(lhs, dtype)
        remainder = self.append(("fmod", lhs_id, rhs_id), dtype)
        exact = self.append(("sub", lhs_id, remainder), dtype)
        quotient = self.append(
            ("floordiv", exact, rhs_id),
            dtype,
            nonnegative=self.is_nonnegative(lhs_id, rhs_id),
        )
        return self.narrow(quotient, dtype)

    def comparison(self, op, lhs, rhs) -> int:
        dtype = self.result_type(lhs, rhs)
        if dtype == torch.bool and op not in ("eq", "ne"):
            dtype = torch.int32
        return self.binary(op, lhs, rhs, dtype, result_dtype=torch.bool)

    def bitwise(self, node, op, lhs, rhs) -> int:
        dtype = self.result_dtype(node, lhs, rhs)
        if _value_kind(dtype) == "f32":
            raise NotImplementedError(f"mask_mod bitwise {op} needs integer operands")
        rhs_id = self.coerce(rhs, dtype)
        return self.combine(op, self.coerce(lhs, dtype), rhs_id, dtype)

    def shift(self, node, op, lhs, rhs) -> int:
        dtype = self.result_dtype(node, lhs, rhs)
        kind = _value_kind(dtype)
        if kind not in ("i32", "i64"):
            raise NotImplementedError(f"mask_mod {op} needs integer operands")
        width = 64 if kind == "i64" else 32
        rhs_id = self.coerce(rhs, dtype)
        lhs_id = self.coerce(lhs, dtype)
        nonnegative = op == "shr" and self.is_nonnegative(lhs_id)
        constant_amount = not isinstance(rhs, bool) and isinstance(rhs, int)
        if constant_amount and 0 <= rhs < width:
            value_id = self.append((op, lhs_id, rhs_id), dtype, nonnegative=nonnegative)
        else:
            # Out-of-range amounts follow eager PyTorch: << gives 0, >> fills the sign.
            in_range = self.append(
                (
                    "and",
                    self.append(("ge", rhs_id, self.constant(0, dtype)), torch.bool),
                    self.append(
                        ("lt", rhs_id, self.constant(width, dtype)), torch.bool
                    ),
                ),
                torch.bool,
            )
            if op == "shl":
                shifted = self.append(("shl", lhs_id, rhs_id), dtype)
                value_id = self.where(in_range, shifted, self.constant(0, dtype))
            else:
                amount = self.where(in_range, rhs_id, self.constant(width - 1, dtype))
                value_id = self.append(
                    ("shr", lhs_id, amount), dtype, nonnegative=nonnegative
                )
        return self.narrow(value_id, dtype) if op == "shl" else value_id

    def minmax(self, node, op, lhs, rhs) -> int:
        dtype = self.result_dtype(node, lhs, rhs)
        if dtype == torch.bool:
            return self.binary("and" if op == "min" else "or", lhs, rhs, dtype)
        rhs_id = self.coerce(rhs, dtype)
        return self.combine(op, self.coerce(lhs, dtype), rhs_id, dtype)

    def unary(self, op, value) -> int:
        value_id = self.node(value)
        dtype = self.dtypes[value_id]
        kind = _value_kind(dtype)
        if op == "not":
            if kind == "f32":
                raise NotImplementedError("mask_mod bitwise not needs integer operands")
            value_id = self.append(("not", value_id), dtype)
            return self.wrap(value_id, dtype) if dtype == torch.uint8 else value_id
        if kind == "bool":
            raise NotImplementedError(f"mask_mod {op} on bool values is unsupported")
        if op == "abs" and dtype == torch.uint8:
            return value_id
        value_id = self.append((op, value_id), dtype, nonnegative=op == "abs")
        return self.narrow(value_id, dtype)

    def clamp(self, node, value, lower, upper) -> int:
        dtype = self.result_dtype(
            node, value, *(bound for bound in (lower, upper) if bound is not None)
        )
        if dtype == torch.bool:
            raise NotImplementedError("mask_mod clamp on bool values is unsupported")
        value_id = self.coerce(value, dtype)
        for op, bound in (("max", lower), ("min", upper)):
            if bound is not None:
                value_id = self.combine(op, value_id, self.coerce(bound, dtype), dtype)
        return value_id

    def index(self, source, indices) -> int:
        if source not in self.capture_index:
            raise NotImplementedError(
                "mask_mod indexing is supported only for captured buffers"
            )
        buffer_index = self.capture_index[source]
        shape = self.buffer_shapes[buffer_index]
        if len(indices) != len(shape) or any(index is None for index in indices):
            raise NotImplementedError("mask_mod captured-buffer index rank mismatch")
        index_ids = []
        for index, size in zip(indices, shape):
            if isinstance(index, Node):
                index_id = self.node(index)
            elif isinstance(index, int) and not isinstance(index, bool):
                index_id = self.constant(index, torch.int32)
            else:
                raise NotImplementedError(
                    "mask_mod captured buffers need integer indices"
                )
            dtype = self.dtypes[index_id]
            if _value_kind(dtype) not in ("i32", "i64"):
                raise NotImplementedError(
                    "mask_mod captured buffers need integer indices"
                )
            if not self.is_nonnegative(index_id):
                # Negative indices wrap once, as in eager indexing.
                negative = self.append(
                    ("lt", index_id, self.constant(0, dtype)), torch.bool
                )
                wrapped = self.append(
                    ("add", index_id, self.constant(size, dtype)), dtype
                )
                index_id = self.where(negative, wrapped, index_id)
            if dtype == torch.int64:
                index_id = self.append(
                    ("cast", index_id, "i32"),
                    torch.int32,
                    nonnegative=self.is_nonnegative(index_id),
                )
            index_ids.append(index_id)
        return self.load(buffer_index, tuple(index_ids))

    def lower(self, node: Node) -> int:
        target = node.target
        args = node.args
        if target in _CONSTANT_TARGETS:
            if target == aten.full.default:
                shape, fill = args[:2]
                if not isinstance(shape, (tuple, list)) or shape:
                    raise NotImplementedError(
                        "mask_mod aten.full supports scalar constants only"
                    )
            else:
                (fill,) = args[:1]
            if not isinstance(fill, (bool, int, float)):
                raise NotImplementedError(
                    "mask_mod aten.full supports scalar constants only"
                )
            dtype = node.kwargs.get("dtype")
            if dtype is None:
                dtype = getattr(node.meta.get("val"), "dtype", None)
            if dtype is None:
                dtype = torch.tensor(fill).dtype
            return self.constant(fill, dtype)
        if target == aten.index.Tensor:
            return self.index(*args[:2])
        if target in _IDENTITY_TARGETS:
            return self.node(args[0])
        if target in _CAST_TARGETS:
            dtype = args[1] if len(args) > 1 else node.kwargs.get("dtype")
            value_id = self.node(args[0])
            return value_id if dtype is None else self.convert(value_id, dtype)
        if target in _ARITHMETIC_TARGETS:
            return self.arithmetic(node, _ARITHMETIC_TARGETS[target], *args[:2])
        if target in _TRUE_DIVIDE_TARGETS:
            return self.true_divide(node, *args[:2])
        if target in _ROUNDING_DIVIDE_TARGETS:
            rounding_mode = node.kwargs.get(
                "rounding_mode", args[2] if len(args) > 2 else None
            )
            if rounding_mode is None:
                return self.true_divide(node, *args[:2])
            if rounding_mode == "floor":
                return self.arithmetic(node, "floordiv", *args[:2])
            return self.truncate_divide(node, *args[:2])
        if target in _COMPARISON_TARGETS:
            return self.comparison(_COMPARISON_TARGETS[target], *args[:2])
        if target in _BITWISE_TARGETS:
            return self.bitwise(node, _BITWISE_TARGETS[target], *args[:2])
        if target in _SHIFT_TARGETS:
            return self.shift(node, _SHIFT_TARGETS[target], *args[:2])
        if target in _LOGICAL_TARGETS:
            return self.binary(_LOGICAL_TARGETS[target], args[0], args[1], torch.bool)
        if target == aten.logical_not.default:
            return self.append(("not", self.coerce(args[0], torch.bool)), torch.bool)
        if target in _MINMAX_TARGETS:
            return self.minmax(node, _MINMAX_TARGETS[target], *args[:2])
        if target in _UNARY_TARGETS:
            return self.unary(_UNARY_TARGETS[target], args[0])
        if target in _WHERE_TARGETS:
            condition, true_value, false_value = args[:3]
            dtype = self.result_dtype(node, true_value, false_value)
            condition_id = self.coerce(condition, torch.bool)
            true_id = self.coerce(true_value, dtype)
            return self.where(condition_id, true_id, self.coerce(false_value, dtype))
        if target == aten.clamp.default:
            bounds = [*args[1:3], None, None][:2]
            lower = node.kwargs.get("min", bounds[0])
            upper = node.kwargs.get("max", bounds[1])
            return self.clamp(node, args[0], lower, upper)
        if target == aten.clamp_min.default:
            return self.clamp(node, args[0], args[1], None)
        if target == aten.clamp_max.default:
            return self.clamp(node, args[0], None, args[1])
        raise NotImplementedError(f"mask_mod operation {target} is unsupported")


BLOCK_STATE_EMPTY, BLOCK_STATE_PARTIAL, BLOCK_STATE_FULL = 0, 1, 2


def add_block_states(
    program: FlyDSLMaskProgram | None,
    *,
    shape: tuple[int, ...],
    strides: tuple[int, ...],
    block_sizes: tuple[int, int],
    head_group: int,
) -> FlyDSLMaskProgram:
    """Keep a score only if its block is full, or partial and kept by ``program``
    (None keeps all). The int8 capture of ``shape`` holds a ``BLOCK_STATE_*``
    per (batch, head, Q block, KV block); query head h reads h // head_group.
    """
    base = program or FlyDSLMaskProgram((), 0, (), ())
    instructions = list(base.instructions)
    constants: dict[int, int] = {}

    def emit(*instruction) -> int:
        instructions.append(instruction)
        return 3 + len(instructions)

    def constant(value: int) -> int:
        if value not in constants:
            constants[value] = emit("const_i32", value)
        return constants[value]

    def block_index(position: int, block_size: int) -> int:
        # Positions are non-negative, so a shift divides by a power of two.
        if block_size & (block_size - 1):
            return emit("floordiv", position, constant(block_size))
        return emit("shr", position, constant(block_size.bit_length() - 1))

    batch = constant(0) if shape[0] == 1 else 0
    if shape[1] == 1:
        head = constant(0)
    elif head_group == 1:
        head = 1
    else:
        head = emit("floordiv", 1, constant(head_group))
    q_block = block_index(2, block_sizes[0])
    kv_block = block_index(3, block_sizes[1])
    state = emit("load", base.buffer_count, (batch, head, q_block, kv_block), "int8")
    if program is None:
        keep = emit("ne", state, constant(BLOCK_STATE_EMPTY))
    else:
        full = emit("eq", state, constant(BLOCK_STATE_FULL))
        partial = emit("eq", state, constant(BLOCK_STATE_PARTIAL))
        keep = emit("or", full, emit("and", partial, program.output))
    return FlyDSLMaskProgram(
        instructions=tuple(instructions),
        output=keep,
        buffer_shapes=(*base.buffer_shapes, tuple(shape)),
        buffer_strides=(*base.buffer_strides, tuple(strides)),
        buffer_dtypes=(*base.buffer_dtypes, "int8"),
    )


def lower_flydsl_mask_graph(
    graph_module,
    mask_mod_other_buffers,
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
    # The sizes of captured tensors with dynamic shapes arrive as symbols.
    scalars = [
        (node, buffer)
        for node, buffer in zip(captures, mask_mod_other_buffers)
        if _is_size(buffer)
    ]
    captures = [
        node
        for node, buffer in zip(captures, mask_mod_other_buffers)
        if not _is_size(buffer)
    ]
    mask_mod_other_buffers = mask_capture_tensors(mask_mod_other_buffers)

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

    if any(dtype not in _VALUE_KINDS for dtype in buffer_dtypes):
        return None, (
            "mask_mod captured buffers must be bool, int8, int16, int32, int64, "
            "uint8, float16, bfloat16 or float32"
        )
    if any(
        not shape
        or len(shape) > 4
        or len(shape) != len(stride)
        or any(value <= 0 for value in stride)
        for shape, stride in zip(buffer_shapes, buffer_strides)
    ):
        return None, "mask_mod captured buffers require rank 1-4 positive strides"
    # Kernels address captures with i32 element offsets through 32-bit buffer views.
    for shape, stride, dtype in zip(buffer_shapes, buffer_strides, buffer_dtypes):
        extent = 1 + sum((size - 1) * step for size, step in zip(shape, stride))
        if extent >= _MAX_CAPTURE_ELEMENTS or extent * dtype.itemsize >= (
            _MAX_CAPTURE_BYTES
        ):
            return None, "mask_mod captured buffers must be smaller than 4 GiB"
    if any(_is_cpu_buffer(buffer) for buffer in mask_mod_other_buffers):
        return None, "mask_mod captured buffers must live on the attention device"

    lowering = _MaskLowering(captures, buffer_shapes, buffer_dtypes)
    for placeholder in placeholders[:4]:
        lowering.values[placeholder] = len(lowering.values)
    try:
        for node, size in scalars:
            lowering.values[node] = lowering.constant(
                V.graph.sizevars.guard_int(size), torch.int32
            )
    except (TypeError, ValueError):
        return None, "mask_mod requires backed captured-buffer sizes"

    try:
        for node in nodes:
            if node.op in ("placeholder", "output"):
                continue
            if node.op != "call_function":
                raise NotImplementedError(
                    f"mask_mod node kind {node.op!r} is unsupported"
                )
            lowering.values[node] = lowering.lower(node)

        output_value = outputs[0].args[0]
        if isinstance(output_value, (tuple, list)):
            if len(output_value) != 1:
                raise NotImplementedError("mask_mod must return one scalar")
            output_value = output_value[0]
        output_id = lowering.coerce(output_value, torch.bool)
    except NotImplementedError as error:
        return None, str(error)

    return (
        FlyDSLMaskProgram(
            instructions=tuple(lowering.instructions),
            output=output_id,
            buffer_shapes=buffer_shapes,
            buffer_strides=buffer_strides,
            buffer_dtypes=tuple(_dtype_name(dtype) for dtype in buffer_dtypes),
        ),
        "",
    )
