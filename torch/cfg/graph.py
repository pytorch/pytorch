from __future__ import annotations

import enum
import inspect
import re
import types
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

import torch


if TYPE_CHECKING:
    from torch.fx import Graph as FxGraph
    from torch.fx import GraphModule as FxGraphModule
    from torch.fx import Node as FxNode


__all__ = [
    "BasicBlock",
    "BlockParameterSpec",
    "BranchTerminator",
    "ControlFlowGraph",
    "Graph",
    "Instruction",
    "InstructionKind",
    "JumpTerminator",
    "RaiseTerminator",
    "ReturnTerminator",
    "SourceLocation",
    "TensorSpec",
    "Value",
    "ValueInfo",
]


class _UnsetExampleValue:
    def __repr__(self) -> str:
        return "<unset>"


_UNSET_EXAMPLE_VALUE = _UnsetExampleValue()
_STACK_TRACE_FRAME = re.compile(r'File "(.+)", line (\d+), in (.+)')


@dataclass(frozen=True)
class SourceLocation:
    file: str
    line: int
    function: str | None = None


@dataclass(frozen=True)
class TensorSpec:
    dtype: torch.dtype
    shape: tuple[int | torch.SymInt | None, ...]
    device: torch.device | None = None
    requires_grad: bool | None = None
    stride: tuple[int | torch.SymInt, ...] | None = None
    layout: torch.layout | None = None
    is_quantized: bool | None = None

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor) -> TensorSpec:
        return cls(
            dtype=tensor.dtype,
            shape=tuple(tensor.shape),
            device=tensor.device,
            requires_grad=tensor.requires_grad,
            stride=tuple(tensor.stride()),
            layout=tensor.layout,
            is_quantized=tensor.is_quantized,
        )

    def __str__(self) -> str:
        parts = [
            f"dtype={self.dtype}",
            f"shape={_format_shape(self.shape)}",
        ]
        if self.device is not None:
            parts.append(f"device={self.device}")
        if self.requires_grad is not None:
            parts.append(f"requires_grad={self.requires_grad}")
        if self.layout is not None and self.layout is not torch.strided:
            parts.append(f"layout={self.layout}")
        if self.is_quantized:
            parts.append("quantized=True")
        return f"Tensor[{', '.join(parts)}]"


@dataclass(frozen=True)
class ValueInfo:
    python_type: Any | None = None
    tensor: TensorSpec | None = None
    example_value: Any = _UNSET_EXAMPLE_VALUE
    source: SourceLocation | None = None

    @property
    def has_example_value(self) -> bool:
        return self.example_value is not _UNSET_EXAMPLE_VALUE

    @classmethod
    def from_example(
        cls,
        example_value: Any,
        *,
        python_type: Any | None = None,
        source: SourceLocation | None = None,
    ) -> ValueInfo:
        inferred_python_type = python_type if python_type is not None else type(example_value)
        tensor = TensorSpec.from_tensor(example_value) if isinstance(example_value, torch.Tensor) else None
        return cls(
            python_type=inferred_python_type,
            tensor=tensor,
            example_value=example_value,
            source=source,
        )

    @classmethod
    def from_fx_node(cls, node: FxNode) -> ValueInfo:
        source = _source_from_stack_trace(node.meta.get("stack_trace"))
        if "val" in node.meta:
            return cls.from_example(node.meta["val"], python_type=node.type, source=source)
        if "example_value" in node.meta:
            return cls.from_example(
                node.meta["example_value"],
                python_type=node.type,
                source=source,
            )
        return cls(python_type=node.type, source=source)

    def summary(self) -> str:
        if self.tensor is not None:
            return str(self.tensor)
        if self.python_type is not None:
            return _describe_python_type(self.python_type)
        if self.has_example_value:
            return f"example={_short_repr(self.example_value)}"
        return "unknown"


@dataclass(frozen=True)
class BlockParameterSpec:
    name: str
    info: ValueInfo = field(default_factory=ValueInfo)


@dataclass(frozen=True, eq=False)
class Value:
    name: str
    info: ValueInfo = field(default_factory=ValueInfo)
    is_block_parameter: bool = False

    def __str__(self) -> str:
        return f"%{self.name}"

    def __repr__(self) -> str:
        kind = "parameter" if self.is_block_parameter else "value"
        return f"Value(name={self.name!r}, kind={kind})"


class InstructionKind(enum.Enum):
    GET_ATTR = "get_attr"
    CALL_FUNCTION = "call_function"
    CALL_METHOD = "call_method"
    CALL_MODULE = "call_module"


@dataclass(frozen=True, eq=False)
class Instruction:
    name: str
    kind: InstructionKind
    target: Any
    args: tuple[Any, ...] = ()
    kwargs: Mapping[str, Any] = field(
        default_factory=lambda: types.MappingProxyType({})
    )
    result: Value | None = None
    source: SourceLocation | None = None

    def __str__(self) -> str:
        lhs = f"{self.result} = " if self.result is not None else ""
        formatted_args = [_format_operand(arg) for arg in self.args]
        formatted_args.extend(
            f"{key}={_format_operand(value)}" for key, value in self.kwargs.items()
        )
        joined_args = ", ".join(formatted_args)
        return (
            f"{lhs}{self.kind.value}[target={_format_target(self.target)}]"
            f"({joined_args})"
        )


@dataclass(frozen=True)
class ReturnTerminator:
    value: Any

    def __str__(self) -> str:
        return f"return {_format_operand(self.value)}"


@dataclass(frozen=True)
class RaiseTerminator:
    error: Any

    def __str__(self) -> str:
        return f"raise {_format_operand(self.error)}"


@dataclass(frozen=True)
class JumpTerminator:
    target: BasicBlock
    arguments: tuple[Any, ...] = ()

    def __str__(self) -> str:
        return f"jump {self.target.name}({_format_operands(self.arguments)})"


@dataclass(frozen=True)
class BranchTerminator:
    condition: Any
    true_target: BasicBlock
    false_target: BasicBlock
    true_arguments: tuple[Any, ...] = ()
    false_arguments: tuple[Any, ...] = ()

    def __str__(self) -> str:
        return (
            f"branch {_format_operand(self.condition)}"
            f" -> {self.true_target.name}({_format_operands(self.true_arguments)})"
            f", {self.false_target.name}({_format_operands(self.false_arguments)})"
        )


Terminator = ReturnTerminator | RaiseTerminator | JumpTerminator | BranchTerminator


class BasicBlock:
    """
    A basic block is an ordered list of instructions with explicit block parameters
    and a single terminating control-flow operation.

    Values are block-local. If a successor needs a value, pass it explicitly via
    the terminator arguments into the successor block parameters.
    """

    def __init__(
        self,
        graph: Graph,
        name: str,
        parameters: Sequence[BlockParameterSpec],
    ) -> None:
        self._graph = graph
        self._name = name
        self._parameters: tuple[Value, ...] = tuple(
            graph._register_value(spec.name, spec.info, block=self, is_block_parameter=True)
            for spec in parameters
        )
        self._instructions: list[Instruction] = []
        self._terminator: Terminator | None = None

    @property
    def graph(self) -> Graph:
        return self._graph

    @property
    def name(self) -> str:
        return self._name

    @property
    def parameters(self) -> tuple[Value, ...]:
        return self._parameters

    @property
    def instructions(self) -> tuple[Instruction, ...]:
        return tuple(self._instructions)

    @property
    def terminator(self) -> Terminator | None:
        return self._terminator

    @property
    def successors(self) -> tuple[BasicBlock, ...]:
        terminator = self._terminator
        if isinstance(terminator, JumpTerminator):
            return (terminator.target,)
        if isinstance(terminator, BranchTerminator):
            return (terminator.true_target, terminator.false_target)
        return ()

    def get_attr(
        self,
        name: str,
        target: str,
        *,
        info: ValueInfo | None = None,
        source: SourceLocation | None = None,
    ) -> Value:
        return self._append_instruction(
            name=name,
            kind=InstructionKind.GET_ATTR,
            target=target,
            args=(),
            kwargs=None,
            info=info,
            source=source,
        )

    def call_function(
        self,
        name: str,
        target: Any,
        args: Sequence[Any] = (),
        kwargs: Mapping[str, Any] | None = None,
        *,
        info: ValueInfo | None = None,
        source: SourceLocation | None = None,
    ) -> Value:
        return self._append_instruction(
            name=name,
            kind=InstructionKind.CALL_FUNCTION,
            target=target,
            args=args,
            kwargs=kwargs,
            info=info,
            source=source,
        )

    def call_method(
        self,
        name: str,
        target: str,
        args: Sequence[Any] = (),
        kwargs: Mapping[str, Any] | None = None,
        *,
        info: ValueInfo | None = None,
        source: SourceLocation | None = None,
    ) -> Value:
        return self._append_instruction(
            name=name,
            kind=InstructionKind.CALL_METHOD,
            target=target,
            args=args,
            kwargs=kwargs,
            info=info,
            source=source,
        )

    def call_module(
        self,
        name: str,
        target: str,
        args: Sequence[Any] = (),
        kwargs: Mapping[str, Any] | None = None,
        *,
        info: ValueInfo | None = None,
        source: SourceLocation | None = None,
    ) -> Value:
        return self._append_instruction(
            name=name,
            kind=InstructionKind.CALL_MODULE,
            target=target,
            args=args,
            kwargs=kwargs,
            info=info,
            source=source,
        )

    def jump(self, target: BasicBlock, *arguments: Any) -> JumpTerminator:
        return self._set_terminator(JumpTerminator(target=target, arguments=tuple(arguments)))

    def branch(
        self,
        condition: Any,
        *,
        true_target: BasicBlock,
        false_target: BasicBlock,
        true_arguments: Sequence[Any] = (),
        false_arguments: Sequence[Any] = (),
    ) -> BranchTerminator:
        return self._set_terminator(
            BranchTerminator(
                condition=condition,
                true_target=true_target,
                false_target=false_target,
                true_arguments=tuple(true_arguments),
                false_arguments=tuple(false_arguments),
            )
        )

    def return_(self, value: Any) -> ReturnTerminator:
        return self._set_terminator(ReturnTerminator(value=value))

    def raise_(self, error: Any) -> RaiseTerminator:
        return self._set_terminator(RaiseTerminator(error=error))

    def _append_instruction(
        self,
        *,
        name: str,
        kind: InstructionKind,
        target: Any,
        args: Sequence[Any],
        kwargs: Mapping[str, Any] | None,
        info: ValueInfo | None,
        source: SourceLocation | None,
    ) -> Value:
        self._ensure_open()
        result = self._graph._register_value(name, info or ValueInfo(), block=self)
        instruction = Instruction(
            name=name,
            kind=kind,
            target=target,
            args=tuple(args),
            kwargs=types.MappingProxyType(dict(kwargs or {})),
            result=result,
            source=source,
        )
        self._instructions.append(instruction)
        return result

    def _set_terminator(self, terminator: Terminator):
        self._ensure_open()
        self._terminator = terminator
        return terminator

    def _ensure_open(self) -> None:
        if self._terminator is not None:
            raise ValueError(f"Block {self.name!r} already has a terminator")

    def __iter__(self) -> Iterator[Instruction]:
        return iter(self._instructions)

    def __repr__(self) -> str:
        return f"BasicBlock(name={self.name!r})"


class Graph:
    """
    A strict, block-structured SSA control-flow graph.

    Design rules:
    - Value facts live in `ValueInfo`, not in an open-ended metadata bag.
    - Control flow is explicit through `return`, `jump`, `branch`, and `raise`.
    - Values are block-local; successor blocks receive data through parameters.
    - Graphs are validated eagerly for duplicate names and block shape, and can be
      validated end-to-end with `validate()`.
    """

    def __init__(self, name: str = "graph") -> None:
        self.name = name
        self._blocks_by_name: dict[str, BasicBlock] = {}
        self._value_by_name: dict[str, Value] = {}
        self._value_owner_block: dict[int, BasicBlock] = {}
        self._entry: BasicBlock | None = None

    @property
    def blocks(self) -> tuple[BasicBlock, ...]:
        return tuple(self._blocks_by_name.values())

    @property
    def entry(self) -> BasicBlock:
        if self._entry is None:
            raise ValueError("Graph does not have an entry block yet")
        return self._entry

    def new_block(
        self,
        name: str,
        *,
        parameters: Sequence[str | BlockParameterSpec | tuple[str, ValueInfo]] = (),
        is_entry: bool = False,
    ) -> BasicBlock:
        if name in self._blocks_by_name:
            raise ValueError(f"Graph already contains a block named {name!r}")

        block = BasicBlock(self, name, _normalize_block_parameters(parameters))
        self._blocks_by_name[name] = block

        if self._entry is None:
            self._entry = block
        elif is_entry:
            raise ValueError(
                f"Graph {self.name!r} already has entry block {self._entry.name!r}"
            )
        return block

    def get_block(self, name: str) -> BasicBlock:
        try:
            return self._blocks_by_name[name]
        except KeyError as error:
            raise KeyError(f"Unknown block {name!r}") from error

    def validate(self) -> Graph:
        if self._entry is None:
            raise ValueError("Graph must define an entry block")

        if not self._blocks_by_name:
            raise ValueError("Graph must contain at least one block")

        for block in self.blocks:
            if block.terminator is None:
                raise ValueError(f"Block {block.name!r} does not end in a terminator")

            visible_values = {id(value) for value in block.parameters}
            for instruction in block.instructions:
                self._validate_local_operands(
                    block=block,
                    value_ids=visible_values,
                    operand=instruction.args,
                    context=f"instruction {instruction.name!r} args",
                )
                self._validate_local_operands(
                    block=block,
                    value_ids=visible_values,
                    operand=instruction.kwargs,
                    context=f"instruction {instruction.name!r} kwargs",
                )
                if instruction.result is not None:
                    visible_values.add(id(instruction.result))
            self._validate_terminator(block, visible_values)

        reachable = self._reachable_blocks()
        unreachable = [block.name for block in self.blocks if block not in reachable]
        if unreachable:
            raise ValueError(
                f"Graph {self.name!r} contains unreachable blocks: {', '.join(unreachable)}"
            )

        return self

    @classmethod
    def from_fx(
        cls,
        fx_graph: FxGraph | FxGraphModule,
        *,
        name: str | None = None,
    ) -> Graph:
        from torch.fx import GraphModule, Node

        if isinstance(fx_graph, GraphModule):
            graph = fx_graph.graph
            graph_name = name if name is not None else type(fx_graph).__name__
        else:
            graph = fx_graph
            graph_name = name if name is not None else "fx_graph"

        placeholders = [
            BlockParameterSpec(node.name, ValueInfo.from_fx_node(node))
            for node in graph.nodes
            if node.op == "placeholder"
        ]

        cfg = cls(graph_name)
        entry = cfg.new_block("entry", parameters=placeholders, is_entry=True)

        fx_to_cfg_value: dict[FxNode, Value] = {}
        placeholder_nodes = [node for node in graph.nodes if node.op == "placeholder"]
        for fx_node, parameter in zip(placeholder_nodes, entry.parameters):
            fx_to_cfg_value[fx_node] = parameter

        def map_operand(operand: Any) -> Any:
            if isinstance(operand, Node):
                return fx_to_cfg_value[operand]
            if isinstance(operand, tuple):
                return tuple(map_operand(item) for item in operand)
            if isinstance(operand, list):
                return [map_operand(item) for item in operand]
            if isinstance(operand, dict):
                return {key: map_operand(value) for key, value in operand.items()}
            if isinstance(operand, slice):
                return slice(
                    map_operand(operand.start),
                    map_operand(operand.stop),
                    map_operand(operand.step),
                )
            return operand

        for node in graph.nodes:
            if node.op == "placeholder":
                continue
            if node.op == "get_attr":
                fx_to_cfg_value[node] = entry.get_attr(
                    node.name,
                    node.target,
                    info=ValueInfo.from_fx_node(node),
                )
                continue
            if node.op == "call_function":
                fx_to_cfg_value[node] = entry.call_function(
                    node.name,
                    node.target,
                    args=tuple(map_operand(arg) for arg in node.args),
                    kwargs={key: map_operand(value) for key, value in node.kwargs.items()},
                    info=ValueInfo.from_fx_node(node),
                )
                continue
            if node.op == "call_method":
                fx_to_cfg_value[node] = entry.call_method(
                    node.name,
                    node.target,
                    args=tuple(map_operand(arg) for arg in node.args),
                    kwargs={key: map_operand(value) for key, value in node.kwargs.items()},
                    info=ValueInfo.from_fx_node(node),
                )
                continue
            if node.op == "call_module":
                fx_to_cfg_value[node] = entry.call_module(
                    node.name,
                    node.target,
                    args=tuple(map_operand(arg) for arg in node.args),
                    kwargs={key: map_operand(value) for key, value in node.kwargs.items()},
                    info=ValueInfo.from_fx_node(node),
                )
                continue
            if node.op == "output":
                entry.return_(map_operand(node.args[0]))
                continue
            raise ValueError(f"Unsupported FX node operation: {node.op!r}")

        return cfg.validate()

    def format(self) -> str:
        lines = [f"cfg {self.name} {{"]
        for block in self.blocks:
            parameters = ", ".join(_format_value_declaration(value) for value in block.parameters)
            lines.append(f"  block {block.name}({parameters}):")
            if not block.instructions:
                lines.append("    pass")
            else:
                for instruction in block.instructions:
                    lines.append(f"    {instruction}")
            if block.terminator is None:
                lines.append("    <missing terminator>")
            else:
                lines.append(f"    {block.terminator}")
        lines.append("}")
        return "\n".join(lines)

    def _register_value(
        self,
        name: str,
        info: ValueInfo,
        *,
        block: BasicBlock,
        is_block_parameter: bool = False,
    ) -> Value:
        if name in self._value_by_name:
            raise ValueError(f"Graph already contains a value named {name!r}")

        value = Value(name=name, info=info, is_block_parameter=is_block_parameter)
        self._value_by_name[name] = value
        self._value_owner_block[id(value)] = block
        return value

    def _reachable_blocks(self) -> set[BasicBlock]:
        worklist = [self.entry]
        visited: set[BasicBlock] = set()
        while worklist:
            block = worklist.pop()
            if block in visited:
                continue
            visited.add(block)
            worklist.extend(block.successors)
        return visited

    def _validate_local_operands(
        self,
        *,
        block: BasicBlock,
        value_ids: set[int],
        operand: Any,
        context: str,
    ) -> None:
        for value in _iter_values(operand):
            owner = self._value_owner_block.get(id(value))
            if owner is None:
                raise ValueError(f"{context} in block {block.name!r} references an unknown value {value}")
            if id(value) not in value_ids:
                raise ValueError(
                    f"{context} in block {block.name!r} uses {value} without receiving it "
                    f"as a block parameter or defining it earlier in the block"
                )

    def _validate_jump_like_target(
        self,
        *,
        source: BasicBlock,
        target: BasicBlock,
        arguments: Sequence[Any],
        visible_values: set[int],
        context: str,
    ) -> None:
        if self._blocks_by_name.get(target.name) is not target:
            raise ValueError(f"{context} in block {source.name!r} targets a block outside the graph")
        if len(arguments) != len(target.parameters):
            raise ValueError(
                f"{context} in block {source.name!r} passes {len(arguments)} arguments "
                f"to block {target.name!r}, which expects {len(target.parameters)}"
            )
        self._validate_local_operands(
            block=source,
            value_ids=visible_values,
            operand=tuple(arguments),
            context=context,
        )

    def _validate_terminator(self, block: BasicBlock, visible_values: set[int]) -> None:
        terminator = block.terminator
        if terminator is None:
            raise ValueError(f"Block {block.name!r} does not end in a terminator")
        if isinstance(terminator, ReturnTerminator):
            self._validate_local_operands(
                block=block,
                value_ids=visible_values,
                operand=terminator.value,
                context="return terminator",
            )
            return
        if isinstance(terminator, RaiseTerminator):
            self._validate_local_operands(
                block=block,
                value_ids=visible_values,
                operand=terminator.error,
                context="raise terminator",
            )
            return
        if isinstance(terminator, JumpTerminator):
            self._validate_jump_like_target(
                source=block,
                target=terminator.target,
                arguments=terminator.arguments,
                visible_values=visible_values,
                context="jump terminator",
            )
            return
        if isinstance(terminator, BranchTerminator):
            self._validate_local_operands(
                block=block,
                value_ids=visible_values,
                operand=terminator.condition,
                context="branch terminator condition",
            )
            self._validate_jump_like_target(
                source=block,
                target=terminator.true_target,
                arguments=terminator.true_arguments,
                visible_values=visible_values,
                context="branch true edge",
            )
            self._validate_jump_like_target(
                source=block,
                target=terminator.false_target,
                arguments=terminator.false_arguments,
                visible_values=visible_values,
                context="branch false edge",
            )
            return
        raise TypeError(f"Unsupported terminator type: {type(terminator)!r}")

    def __str__(self) -> str:
        return self.format()

    def __repr__(self) -> str:
        return f"Graph(name={self.name!r}, blocks={len(self._blocks_by_name)})"


ControlFlowGraph = Graph


def _normalize_block_parameters(
    parameters: Sequence[str | BlockParameterSpec | tuple[str, ValueInfo]],
) -> tuple[BlockParameterSpec, ...]:
    normalized: list[BlockParameterSpec] = []
    for parameter in parameters:
        if isinstance(parameter, BlockParameterSpec):
            normalized.append(parameter)
        elif isinstance(parameter, str):
            normalized.append(BlockParameterSpec(parameter))
        elif (
            isinstance(parameter, tuple)
            and len(parameter) == 2
            and isinstance(parameter[0], str)
            and isinstance(parameter[1], ValueInfo)
        ):
            normalized.append(BlockParameterSpec(parameter[0], parameter[1]))
        else:
            raise TypeError(
                "Block parameters must be strings, BlockParameterSpec, or "
                "(name, ValueInfo) tuples"
            )
    return tuple(normalized)


def _source_from_stack_trace(stack_trace: str | None) -> SourceLocation | None:
    if not stack_trace:
        return None
    for line in stack_trace.splitlines():
        match = _STACK_TRACE_FRAME.search(line.strip())
        if match is None:
            continue
        file_name, line_number, function = match.groups()
        return SourceLocation(file=file_name, line=int(line_number), function=function)
    return None


def _iter_values(operand: Any) -> Iterator[Value]:
    if isinstance(operand, Value):
        yield operand
        return
    if isinstance(operand, (tuple, list)):
        for item in operand:
            yield from _iter_values(item)
        return
    if isinstance(operand, dict):
        for item in operand.values():
            yield from _iter_values(item)
        return
    if isinstance(operand, slice):
        yield from _iter_values(operand.start)
        yield from _iter_values(operand.stop)
        yield from _iter_values(operand.step)


def _format_value_declaration(value: Value) -> str:
    return f"{value}: {value.info.summary()}"


def _format_shape(shape: Sequence[int | torch.SymInt | None]) -> str:
    formatted = ", ".join(str(dim) for dim in shape)
    if len(shape) == 1:
        formatted += ","
    return f"({formatted})"


def _format_target(target: Any) -> str:
    if isinstance(target, str):
        return target
    if hasattr(target, "__qualname__") and hasattr(target, "__module__"):
        return f"{target.__module__}.{target.__qualname__}"
    if hasattr(target, "__name__"):
        return target.__name__
    return _short_repr(target)


def _format_operand(operand: Any) -> str:
    if isinstance(operand, Value):
        return str(operand)
    if isinstance(operand, tuple):
        joined = ", ".join(_format_operand(item) for item in operand)
        if len(operand) == 1:
            joined += ","
        return f"({joined})"
    if isinstance(operand, list):
        return "[" + ", ".join(_format_operand(item) for item in operand) + "]"
    if isinstance(operand, dict):
        parts = ", ".join(
            f"{key}: {_format_operand(value)}" for key, value in operand.items()
        )
        return "{" + parts + "}"
    if isinstance(operand, slice):
        return (
            "slice("
            f"{_format_operand(operand.start)}, "
            f"{_format_operand(operand.stop)}, "
            f"{_format_operand(operand.step)})"
        )
    return _short_repr(operand)


def _format_operands(operands: Sequence[Any]) -> str:
    return ", ".join(_format_operand(operand) for operand in operands)


def _describe_python_type(python_type: Any) -> str:
    if inspect.isclass(python_type):
        if python_type.__module__ == "builtins":
            return python_type.__qualname__
        return f"{python_type.__module__}.{python_type.__qualname__}"
    return _short_repr(python_type)


def _short_repr(value: Any) -> str:
    text = repr(value)
    return text if len(text) <= 80 else text[:77] + "..."
