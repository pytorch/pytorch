from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING, TypeAlias


if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

import torch
from torch.fx.immutable_collections import immutable_dict, immutable_list


Dimension: TypeAlias = int | torch.SymInt
NestedSize: TypeAlias = tuple[tuple[Dimension, ...], ...]
_SCALAR_TYPES = (
    bool,
    int,
    float,
    complex,
    str,
    bytes,
    torch.SymBool,
    torch.SymFloat,
    torch.SymInt,
    torch.dtype,
    torch.device,
    torch.layout,
    torch.memory_format,
)

_IR_PUBLIC_API = (
    "Block",
    "Branch",
    "DictSpec",
    "Graph",
    "Instruction",
    "Jump",
    "ListSpec",
    "Literal",
    "Location",
    "ObjectSpec",
    "OptionalSpec",
    "Return",
    "ScalarSpec",
    "Spec",
    "Successor",
    "TensorSpec",
    "TupleSpec",
    "ValidationError",
    "Value",
    "literal",
)
__all__ = list(_IR_PUBLIC_API)


class ValidationError(ValueError):
    """Raised when a :class:`Graph` violates CFG invariants."""


class Spec(ABC):
    """
    A typed, normalized description of a value.

    ``Spec`` intentionally replaces the ad hoc ``node.meta["val"]`` /
    ``node.meta["example_value"]`` pattern with a single explicit field.
    """

    @staticmethod
    def from_value(value: object) -> Spec:
        if isinstance(value, torch.Tensor):
            return TensorSpec.from_tensor(value)
        if value is None:
            return OptionalSpec(None)
        if isinstance(value, tuple):
            return TupleSpec(tuple(Spec.from_value(elem) for elem in value))
        if isinstance(value, list):
            return ListSpec(tuple(Spec.from_value(elem) for elem in value))
        if isinstance(value, dict):
            return DictSpec(
                tuple((str(key), Spec.from_value(elem)) for key, elem in value.items())
            )
        if isinstance(value, _SCALAR_TYPES):
            return ScalarSpec(type(value))
        return ObjectSpec(type(value))

    @abstractmethod
    def format(self) -> str:
        raise NotImplementedError

    def __str__(self) -> str:
        return self.format()


@dataclass(frozen=True, slots=True)
class TensorSpec(Spec):
    shape: tuple[Dimension, ...] | None
    dtype: torch.dtype
    device: torch.device
    stride: tuple[Dimension, ...] | None = None
    nested_size: NestedSize | None = None
    requires_grad: bool = False

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor) -> TensorSpec:
        shape: tuple[Dimension, ...] | None
        nested_size: NestedSize | None = None
        if tensor.is_nested:
            try:
                shape = tuple(tensor.shape)
            except RuntimeError:
                shape = None
                # Nested tensor sizes are still exposed through a private API
                # until nested tensor metadata is fully stabilized.
                nested_size = tuple(
                    tuple(int(dim) for dim in size)
                    for size in tensor._nested_tensor_size().tolist()
                )
        else:
            shape = tuple(tensor.shape)

        stride = (
            None if tensor.is_sparse or tensor.is_nested else tuple(tensor.stride())
        )
        return cls(
            shape=shape,
            dtype=tensor.dtype,
            device=tensor.device,
            stride=stride,
            nested_size=nested_size,
            requires_grad=tensor.requires_grad,
        )

    def format(self) -> str:
        dtype = str(self.dtype).removeprefix("torch.")
        pieces = [f"dtype={dtype}"]
        if self.shape is not None:
            shape = ", ".join(str(dim) for dim in self.shape)
            pieces.append(f"shape=({shape})")
        elif self.nested_size is not None:
            pieces.append(f"nested_size={self.nested_size}")
        else:
            pieces.append("shape=<unknown>")
        pieces.append(f"device={self.device}")
        if self.requires_grad:
            pieces.append("requires_grad=True")
        return f"Tensor[{', '.join(pieces)}]"


@dataclass(frozen=True, slots=True)
class ScalarSpec(Spec):
    python_type: type[Any]

    def format(self) -> str:
        return _qualified_name(self.python_type)


@dataclass(frozen=True, slots=True)
class TupleSpec(Spec):
    elements: tuple[Spec, ...]

    def format(self) -> str:
        inner = ", ".join(elem.format() for elem in self.elements)
        return f"tuple[{inner}]"


@dataclass(frozen=True, slots=True)
class ListSpec(Spec):
    elements: tuple[Spec, ...]

    def format(self) -> str:
        inner = ", ".join(elem.format() for elem in self.elements)
        return f"list[{inner}]"


@dataclass(frozen=True, slots=True)
class DictSpec(Spec):
    entries: tuple[tuple[str, Spec], ...]

    def format(self) -> str:
        inner = ", ".join(f"{key}: {spec.format()}" for key, spec in self.entries)
        return f"dict[{inner}]"


@dataclass(frozen=True, slots=True)
class OptionalSpec(Spec):
    element: Spec | None

    def format(self) -> str:
        if self.element is None:
            return "Optional[unknown]"
        return f"Optional[{self.element.format()}]"


@dataclass(frozen=True, slots=True)
class ObjectSpec(Spec):
    python_type: type[Any]

    def format(self) -> str:
        return _qualified_name(self.python_type)


@dataclass(frozen=True, slots=True)
class Literal:
    value: object
    spec: Spec | None = None

    def __post_init__(self) -> None:
        if self.spec is None:
            object.__setattr__(self, "spec", Spec.from_value(self.value))

    def __str__(self) -> str:
        return repr(self.value)


def literal(value: object, spec: Spec | None = None) -> Literal:
    return Literal(value=value, spec=spec)


@dataclass(frozen=True, slots=True)
class Value:
    name: str
    spec: Spec | None = None
    doc: str | None = None

    def __str__(self) -> str:
        return f"%{self.name}"


# Normalized operands are pytrees whose leaves are ``Value`` or ``Literal``.
Argument: TypeAlias = (
    Value
    | Literal
    | None
    | tuple["Argument", ...]
    | list["Argument"]
    | dict[str, "Argument"]
    | slice
)


@dataclass(frozen=True, slots=True)
class Location:
    file: str | None = None
    line: int | None = None
    function: str | None = None
    stack: str | None = None

    def format(self) -> str:
        if self.file is not None and self.line is not None:
            prefix = f"{self.file}:{self.line}"
            if self.function is not None:
                return f"{prefix} in {self.function}"
            return prefix
        if self.function is not None:
            return self.function
        if self.stack is not None:
            return self.stack.strip().splitlines()[-1]
        return "<unknown>"


@dataclass(frozen=True, slots=True)
class Instruction:
    name: str
    opcode: str
    target: object
    inputs: tuple[Argument, ...] = ()
    attributes: Mapping[str, Argument] = field(default_factory=immutable_dict)
    outputs: tuple[Value, ...] = ()
    location: Location | None = None
    doc: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "inputs",
            tuple(_normalize_argument(arg) for arg in self.inputs),
        )
        object.__setattr__(
            self,
            "attributes",
            immutable_dict(
                {key: _normalize_argument(arg) for key, arg in self.attributes.items()}
            ),
        )
        object.__setattr__(self, "outputs", tuple(self.outputs))

    def format(self) -> str:
        outputs = ", ".join(str(output) for output in self.outputs)
        rendered_inputs = ", ".join(_format_argument(arg) for arg in self.inputs)
        rendered_attributes = ", ".join(
            f"{key}={_format_argument(arg)}" for key, arg in self.attributes.items()
        )
        rendered_args = ", ".join(
            piece for piece in (rendered_inputs, rendered_attributes) if piece
        )
        target = _format_target(self.target)
        operation = f"{self.opcode}[target={target}]({rendered_args})"
        if outputs:
            return f"{outputs} = {operation}"
        return operation


@dataclass(frozen=True, slots=True)
class Successor:
    block: str
    arguments: tuple[Argument, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "arguments", tuple(_normalize_argument(arg) for arg in self.arguments)
        )

    def format(self) -> str:
        if not self.arguments:
            return self.block
        args = ", ".join(_format_argument(arg) for arg in self.arguments)
        return f"{self.block}({args})"


class _Terminator(ABC):
    def successors(self) -> tuple[Successor, ...]:
        return ()

    def references(self) -> tuple[Argument, ...]:
        return ()

    @abstractmethod
    def format(self) -> str:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class Jump(_Terminator):
    target: Successor

    def successors(self) -> tuple[Successor, ...]:
        return (self.target,)

    def references(self) -> tuple[Argument, ...]:
        return self.target.arguments

    def format(self) -> str:
        return f"jump {self.target.format()}"


@dataclass(frozen=True, slots=True)
class Branch(_Terminator):
    condition: Argument
    true_branch: Successor
    false_branch: Successor

    def __post_init__(self) -> None:
        object.__setattr__(self, "condition", _normalize_argument(self.condition))

    def successors(self) -> tuple[Successor, ...]:
        return (self.true_branch, self.false_branch)

    def references(self) -> tuple[Argument, ...]:
        return (
            self.condition,
            *self.true_branch.arguments,
            *self.false_branch.arguments,
        )

    def format(self) -> str:
        return (
            f"branch {_format_argument(self.condition)} -> "
            f"{self.true_branch.format()}, {self.false_branch.format()}"
        )


@dataclass(frozen=True, slots=True)
class Return(_Terminator):
    value: Argument

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", _normalize_argument(self.value))

    def references(self) -> tuple[Argument, ...]:
        return (self.value,)

    def format(self) -> str:
        return f"return {_format_argument(self.value)}"


@dataclass(frozen=True, slots=True)
class Block:
    name: str
    parameters: tuple[Value, ...] = ()
    instructions: tuple[Instruction, ...] = ()
    terminator: _Terminator = field(default_factory=lambda: Return(None))
    doc: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "parameters", tuple(self.parameters))
        object.__setattr__(self, "instructions", tuple(self.instructions))

    def format(self) -> str:
        header = ", ".join(_format_value(value) for value in self.parameters)
        lines = [f"block {self.name}({header}):"]
        for instruction in self.instructions:
            lines.append(f"  {instruction.format()}")
        lines.append(f"  {self.terminator.format()}")
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class Graph:
    """
    Immutable block-based CFG.

    Value names are globally unique across the whole graph, even across blocks,
    so textual rendering and validation can use them as stable identifiers.
    """

    name: str
    entry: str
    blocks: tuple[Block, ...]
    doc: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "blocks", tuple(self.blocks))
        self.validate()

    def block(self, name: str) -> Block:
        for block in self.blocks:
            if block.name == name:
                return block
        raise KeyError(f"Unknown block {name!r}")

    def validate(self) -> None:
        if not self.name:
            raise ValidationError("graph name must be non-empty")
        if not self.blocks:
            raise ValidationError("graph must contain at least one block")

        block_map: dict[str, Block] = {}
        for block in self.blocks:
            if not block.name:
                raise ValidationError("block name must be non-empty")
            if block.name in block_map:
                raise ValidationError(f"duplicate block name {block.name!r}")
            block_map[block.name] = block

        if self.entry not in block_map:
            raise ValidationError(f"entry block {self.entry!r} does not exist")

        values_by_name: dict[str, Value] = {}
        for block in self.blocks:
            local_scope: dict[str, Value] = {}
            for parameter in block.parameters:
                _register_value(
                    parameter,
                    where=f"block {block.name!r} parameter",
                    local_scope=local_scope,
                    global_scope=values_by_name,
                )

            for instruction in block.instructions:
                if not instruction.name:
                    raise ValidationError(
                        f"block {block.name!r} contains an instruction "
                        "with an empty name"
                    )
                for argument in instruction.inputs:
                    _validate_argument(
                        argument,
                        where=f"instruction {instruction.name!r}",
                        block_name=block.name,
                        local_scope=local_scope,
                        global_scope=values_by_name,
                    )
                for key, argument in instruction.attributes.items():
                    _validate_argument(
                        argument,
                        where=f"instruction {instruction.name!r} attribute {key!r}",
                        block_name=block.name,
                        local_scope=local_scope,
                        global_scope=values_by_name,
                    )
                for output in instruction.outputs:
                    _register_value(
                        output,
                        where=f"instruction {instruction.name!r}",
                        local_scope=local_scope,
                        global_scope=values_by_name,
                    )

            for argument in block.terminator.references():
                _validate_argument(
                    argument,
                    where=f"terminator of block {block.name!r}",
                    block_name=block.name,
                    local_scope=local_scope,
                    global_scope=values_by_name,
                )

            for successor in block.terminator.successors():
                if successor.block not in block_map:
                    raise ValidationError(
                        f"block {block.name!r} jumps to unknown block "
                        f"{successor.block!r}"
                    )
                target = block_map[successor.block]
                if len(successor.arguments) != len(target.parameters):
                    raise ValidationError(
                        f"jump to block {successor.block!r} expects "
                        f"{len(target.parameters)} arguments but received "
                        f"{len(successor.arguments)}"
                    )

    def format(self) -> str:
        sections = [f"graph {self.name}:"]
        for block in self.blocks:
            sections.append(block.format())
        return "\n\n".join(sections)

    def __str__(self) -> str:
        return self.format()


def _qualified_name(obj: type[Any]) -> str:
    module = getattr(obj, "__module__", None)
    qualname = getattr(obj, "__qualname__", None)
    if module is None or qualname is None or module == "builtins":
        return getattr(obj, "__name__", repr(obj))
    return f"{module}.{qualname}"


def _format_target(target: object) -> str:
    if isinstance(target, str):
        return target
    module = getattr(target, "__module__", None)
    qualname = getattr(target, "__qualname__", None)
    if module is not None and qualname is not None:
        return f"{module}.{qualname}"
    name = getattr(target, "__name__", None)
    if module is not None and name is not None:
        return f"{module}.{name}"
    return repr(target)


def _format_value(value: Value) -> str:
    if value.spec is None:
        return str(value)
    return f"{value}: {value.spec.format()}"


def _format_argument(argument: Argument) -> str:
    if isinstance(argument, Value):
        return str(argument)
    if isinstance(argument, Literal):
        return str(argument)
    if argument is None:
        return "None"
    if isinstance(argument, tuple):
        trailing_comma = "," if len(argument) == 1 else ""
        rendered = ", ".join(_format_argument(elem) for elem in argument)
        return f"({rendered}{trailing_comma})"
    if isinstance(argument, list):
        return f"[{', '.join(_format_argument(elem) for elem in argument)}]"
    if isinstance(argument, dict):
        rendered_items = ", ".join(
            f"{key}: {_format_argument(value)}" for key, value in argument.items()
        )
        return "{" + rendered_items + "}"
    if isinstance(argument, slice):
        return (
            "slice("
            f"{_format_argument(argument.start)}, "
            f"{_format_argument(argument.stop)}, "
            f"{_format_argument(argument.step)})"
        )
    raise TypeError(f"Unexpected argument node: {type(argument)!r}")


def _normalize_argument(argument: object) -> Argument:
    if isinstance(argument, (Value, Literal)) or argument is None:
        return argument
    if isinstance(argument, tuple):
        return tuple(_normalize_argument(elem) for elem in argument)
    if isinstance(argument, list):
        return immutable_list(_normalize_argument(elem) for elem in argument)
    if isinstance(argument, dict):
        return immutable_dict(
            {str(key): _normalize_argument(value) for key, value in argument.items()}
        )
    if isinstance(argument, slice):
        return slice(
            _normalize_argument(argument.start),
            _normalize_argument(argument.stop),
            _normalize_argument(argument.step),
        )
    return Literal(argument)


def _iter_values(argument: Argument) -> Iterable[Value]:
    if isinstance(argument, Value):
        yield argument
        return
    if isinstance(argument, Literal) or argument is None:
        return
    if isinstance(argument, (tuple, list)):
        for elem in argument:
            yield from _iter_values(elem)
        return
    if isinstance(argument, dict):
        for elem in argument.values():
            yield from _iter_values(elem)
        return
    if isinstance(argument, slice):
        yield from _iter_values(argument.start)
        yield from _iter_values(argument.stop)
        yield from _iter_values(argument.step)
        return
    raise TypeError(f"Unexpected argument node: {type(argument)!r}")


def _register_value(
    value: Value,
    *,
    where: str,
    local_scope: dict[str, Value],
    global_scope: dict[str, Value],
) -> None:
    if not value.name:
        raise ValidationError(f"{where} defines a value with an empty name")
    if value.name in local_scope:
        raise ValidationError(
            f"{where} redefines value {value.name!r} in the same block"
        )
    if value.name in global_scope:
        raise ValidationError(f"{where} collides with existing value {value.name!r}")
    local_scope[value.name] = value
    global_scope[value.name] = value


def _validate_argument(
    argument: Argument,
    *,
    where: str,
    block_name: str,
    local_scope: dict[str, Value],
    global_scope: dict[str, Value],
) -> None:
    for value in _iter_values(argument):
        if value.name not in local_scope:
            raise ValidationError(
                f"{where} in block {block_name!r} references undefined value "
                f"{value.name!r}"
            )
        if global_scope.get(value.name) != value:
            raise ValidationError(
                f"{where} in block {block_name!r} references a non-canonical "
                f"value object "
                f"for {value.name!r}"
            )
