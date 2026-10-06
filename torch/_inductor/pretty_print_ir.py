# mypy: allow-untyped-defs
"""Readable Inductor loop IR for debugging.

The same computation has a different loop nest at each stage. For example, a
reduction over 16 elements whose output is [4, 1, 8]:

1. Post-lowering (ComputedBuffer): the loop nest exactly as lowered; size-1
   dims stay.

    for idx0 in range(4):
        for idx1 in range(1):
            for idx2 in range(8):
                for ridx in range(16):
                    body(idx0, idx1, idx2, ridx)

2. Pre-fusion (SchedulerNode): the body is rebuilt over snode.get_ranges().
   Size-1 dims are dropped, dims may be reordered by stride, and dims are
   merged where every access is contiguous across them, so the nest can still
   be multidimensional. snode._body takes one variable per remaining dim; the
   mapping back to the buffer's dims is folded into its index expressions.

    for idx0 in range(4):
        for idx1 in range(8):  # was idx2
            for ridx in range(16):
                snode._body(idx0, idx1, ridx)

3. Post-fusion (FusedSchedulerNode): fused.group = (numel, rnumel) is the
   product of the pointwise dims and of the reduction dims, so the kernel has
   one pointwise loop and one reduction loop. Each snode's variables are
   recovered from (idx, ridx) by SIMDKernel.map_kernel_groups_to_node_sizes;
   snodes placed outside the reduction loop are mapped with rnumel = 1. E.g.
   s1 is the reduction above and s2 is a pointwise epilogue on its output:

    for idx in range(4 * 8):
        for ridx in range(16):
            # s1map: (idx, ridx) -> (idx // 8, idx % 8, ridx)
            s1._body(*s1map(idx, ridx))
        # s2map: (idx,) -> (idx,)
        s2._body(*s2map(idx))
"""

from __future__ import annotations

import math
from collections.abc import ValuesView
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

import sympy

import torch
from torch._inductor.dtype_propagation import DtypePropagationOpsHandler
from torch._inductor.ops_handler import DefaultHandler
from torch._inductor.utils import sympy_index_symbol, sympy_product
from torch._inductor.virtualized import OpsValue, V
from torch.utils._ordered_set import OrderedSet
from torch.utils._sympy.printers import PythonPrinter


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Sequence


_INDEX_PRINTER = PythonPrinter()

_BINARY_OPERATORS = {
    "add": "+",
    "sub": "-",
    "mul": "*",
    "dot": "*",
    "truediv": "/",
    "floordiv": "//",
    "mod": "%",
    "pow": "**",
    "lshift": "<<",
    "rshift": ">>",
    "bitwise_and": "&",
    "bitwise_or": "|",
    "bitwise_xor": "^",
    "and_": "&",
    "or_": "|",
    "xor": "^",
    "logical_and": "and",
    "logical_or": "or",
    "eq": "==",
    "ne": "!=",
    "lt": "<",
    "gt": ">",
    "le": "<=",
    "ge": ">=",
}

_UNARY_OPERATORS = {
    "neg": "-",
    "logical_not": "not ",
    "bitwise_not": "~",
}

_DTYPE_NAMES = {
    torch.bool: "bool",
    torch.bfloat16: "bf16",
    torch.float16: "f16",
    torch.float32: "f32",
    torch.float64: "f64",
    torch.int8: "i8",
    torch.int16: "i16",
    torch.int32: "i32",
    torch.int64: "i64",
    torch.uint8: "u8",
    torch.uint16: "u16",
    torch.uint32: "u32",
    torch.uint64: "u64",
}

_REDUCTION_IDENTITIES = {
    "sum": "0",
    "dot": "0",
    "xor_sum": "0",
    "any": "False",
    "prod": "1",
}

_REDUCTION_COMBINES = {
    "sum": "{acc} + {value}",
    "dot": "{acc} + {value}",
    "xor_sum": "{acc} ^ {value}",
    "any": "{acc} | {value}",
    "prod": "{acc} * {value}",
    "max": "max({acc}, {value})",
    "fmax": "fmax({acc}, {value})",
    "min": "min({acc}, {value})",
}


class _Unsupported(Exception):
    pass


@dataclass(frozen=True)
class _Value:
    text: str
    dtype: torch.dtype | None


def _dtype_name(dtype: torch.dtype | None) -> str:
    if dtype is None:
        return "?"
    return _DTYPE_NAMES.get(dtype, str(dtype).removeprefix("torch."))


def _unwrap(value: Any) -> Any:
    if isinstance(value, OpsValue):
        return value.value
    if isinstance(value, tuple):
        return tuple(_unwrap(item) for item in value)
    if isinstance(value, list):
        return [_unwrap(item) for item in value]
    return value


def _render(value: object) -> str:
    value = _unwrap(value)
    if isinstance(value, _Value):
        return value.text
    if isinstance(value, sympy.Expr):
        return _INDEX_PRINTER.doprint(value)
    if isinstance(value, torch.dtype):
        return _dtype_name(value)
    if isinstance(value, float) and math.isinf(value):
        return "-inf" if value < 0 else "inf"
    if isinstance(value, str):
        return repr(value)
    if isinstance(value, tuple):
        suffix = "," if len(value) == 1 else ""
        return f"({', '.join(_render(item) for item in value)}{suffix})"
    if isinstance(value, list):
        return f"[{', '.join(_render(item) for item in value)}]"
    return repr(value)


def _indent(lines: Sequence[str], level: int = 1) -> list[str]:
    prefix = "    " * level
    return [f"{prefix}{line}" if line else "" for line in lines]


def _loop_nest(
    variables: Sequence[sympy.Symbol],
    ranges: Sequence[sympy.Expr],
    body: Sequence[str],
) -> list[str]:
    lines = list(body) or ["pass"]
    for variable, extent in reversed(tuple(zip(variables, ranges))):
        lines = [
            f"for {_render(variable)} in [0, {_render(sympy.sympify(extent))}):",
            *_indent(lines),
        ]
    return lines


class _PrettyOpsHandler(DefaultHandler):
    name = "PrettyLoopOpsHandler"

    def __init__(self, reduction_variables: Sequence[sympy.Symbol]) -> None:
        self.reduction_variables = tuple(reduction_variables)
        self.reads = OrderedSet[str]()
        self.body: list[str] = []
        self.initializers: list[str] = []
        self.finalizers: list[str] = []
        self._temporary_count = 0
        self._accumulator_count = 0
        self._emit_temporaries = True
        # The same dtype rules codegen applies to CSE variables; _Value has a .dtype.
        self._dtypes = DtypePropagationOpsHandler()

    def _bind(self, expression: str, dtype: torch.dtype | None) -> _Value:
        if not self._emit_temporaries:
            return _Value(expression, dtype)
        name = f"tmp{self._temporary_count}"
        self._temporary_count += 1
        self.body.append(f"{name}: {_dtype_name(dtype)} = {expression}")
        return _Value(name, dtype)

    def _default(self, name: str, args: tuple[Any, ...], kwargs: dict[str, Any]):
        args = tuple(_unwrap(arg) for arg in args)
        dtype = getattr(self._dtypes, name)(*args, **kwargs)

        if name in _BINARY_OPERATORS and len(args) == 2 and not kwargs:
            expression = (
                f"{_render(args[0])} {_BINARY_OPERATORS[name]} {_render(args[1])}"
            )
        elif name in _UNARY_OPERATORS and len(args) == 1 and not kwargs:
            expression = f"{_UNARY_OPERATORS[name]}{_render(args[0])}"
        else:
            if name == "maximum":
                name = "max"
            elif name == "minimum":
                name = "min"
            rendered = [_render(arg) for arg in args]
            rendered.extend(f"{key}={_render(value)}" for key, value in kwargs.items())
            expression = f"{name}({', '.join(rendered)})"
        return self._bind(expression, dtype)

    def constant(self, value: bool | float | int, dtype: torch.dtype):
        if dtype.is_floating_point and not isinstance(value, bool):
            value = float(value)
        return self._bind(_render(value), dtype)

    def load(self, name: str, index: sympy.Expr):
        name = _real_name(name)
        self.reads.add(name)
        return self._bind(
            f"{name}[{_render(index)}]",
            V.graph.get_dtype(name),
        )

    def load_seed(self, name: str, offset: object):
        self.reads.add(name)
        return self._bind(f"{name}[{_render(offset)}]", torch.int64)

    def store(self, name, index, value, mode=None):
        operator = "+=" if mode == "atomic_add" else "="
        index, value = _render(index), _render(value)
        self.body.append(f"{_real_name(name)}[{index}] {operator} {value}")

    def reduction(self, dtype, src_dtype, reduction_type, value):
        if reduction_type not in _REDUCTION_COMBINES:
            raise _Unsupported(f"reduction {reduction_type}")

        name = f"acc_{self._accumulator_count}"
        self._accumulator_count += 1
        if reduction_type in ("max", "fmax"):
            identity = (
                "-inf"
                if dtype.is_floating_point
                else f"dtype_min({_dtype_name(dtype)})"
            )
        elif reduction_type == "min":
            identity = (
                "inf" if dtype.is_floating_point else f"dtype_max({_dtype_name(dtype)})"
            )
        else:
            identity = _REDUCTION_IDENTITIES[reduction_type]
        self.initializers.append(f"{name}: {_dtype_name(dtype)} = {identity}")
        combine = _REDUCTION_COMBINES[reduction_type]
        self.body.append(f"{name} = {combine.format(acc=name, value=_render(value))}")
        return _Value(name, dtype)

    def store_reduction(self, name, index, value):
        index, value = _render(index), _render(value)
        self.finalizers.append(f"{_real_name(name)}[{index}] = {value}")

    def where(self, condition, input, other):
        dtype = self._dtypes.where(condition, input, other)
        return self._bind(
            f"where({_render(condition)}, {_render(input)}, {_render(other)})",
            dtype,
        )

    def masked(self, mask, body, other):
        old_emit_temporaries = self._emit_temporaries
        self._emit_temporaries = False
        try:
            value = _unwrap(body())
        finally:
            self._emit_temporaries = old_emit_temporaries
        dtype = value.dtype if isinstance(value, _Value) else None
        return self._bind(
            f"where({_render(mask)}, {_render(value)}, {_render(other)})",
            dtype,
        )

    def index_expr(self, expr: sympy.Expr, dtype: torch.dtype):
        return self._bind(_render(expr), dtype)

    def value_expr(self, expr: sympy.Expr, dtype: torch.dtype):
        return self._bind(f"{_dtype_name(dtype)}({_render(expr)})", dtype)

    def identity(self, value):
        return value

    def to_dtype(
        self,
        value,
        dtype: torch.dtype,
        src_dtype: torch.dtype | None = None,
        use_compute_types: bool = True,
    ):
        return self._bind(f"{_dtype_name(dtype)}({_render(value)})", dtype)

    def indirect_indexing(self, index, size, check=True, wrap_neg=True):
        # Callers do sympy index arithmetic on the result, so return a symbol
        # whose name is the rendered index.
        text = _render(index)
        if wrap_neg:
            text = f"wrap_neg({text})"
        elif not text.isidentifier():
            text = f"({text})"
        return sympy_index_symbol(text)

    def scan(self, dtypes, combine_fn, values):
        raise _Unsupported("scan")

    def sort(self, dtypes, values, stable, descending):
        raise _Unsupported("sort")


def _make_variables(prefix: str, ranges: Sequence[sympy.Expr]):
    return [sympy_index_symbol(f"{prefix}{index}") for index in range(len(ranges))]


def _real_name(name: str) -> str:
    """The buffer that owns `name`'s storage. A mutation output (e.g. with a
    MutationLayoutSHOULDREMOVE) has none: codegen writes it through to the buffer
    it mutates, as Scheduler.mutation_real_name records."""
    while (buffer := V.graph.try_get_buffer(name)) is not None and (
        mutated := buffer.get_mutation_names()
    ):
        name = mutated[0]
    return name


def _tensor_declaration(name: str, mutable: bool = False) -> str:
    value = V.graph.try_get_buffer(name)
    if value is None:
        value = V.graph.graph_inputs.get(name)
    if value is None or not value.has_tensor_output():
        tensor_type = "?[?]"
    else:
        shape = ", ".join(_render(sympy.sympify(size)) for size in value.get_size())
        tensor_type = f"{_dtype_name(value.get_dtype())}[{shape}]"
    return f"{name}: {'mutate ' if mutable else ''}{tensor_type}"


def _format_loop_body(
    name: str,
    written_names: OrderedSet[str],
    run: Callable[[list[sympy.Symbol], list[sympy.Symbol]], object],
    pointwise_ranges: Sequence[sympy.Expr],
    reduction_ranges: Sequence[sympy.Expr],
) -> str:
    pointwise_variables = _make_variables("p", pointwise_ranges)
    reduction_variables = _make_variables("r", reduction_ranges)
    handler = _PrettyOpsHandler(reduction_variables)

    try:
        with V.set_ops_handler(handler):
            run(pointwise_variables, reduction_variables)
    except _Unsupported as exc:
        return f"kernel {name}:\n    unimplemented {exc}"
    except Exception as exc:
        return f"kernel {name}:\n    unimplemented {type(exc).__name__}: {exc}"

    body = list(handler.body)
    if reduction_ranges:
        body = [
            # Setting up the accumulator with initial value
            *handler.initializers,
            # Main computation
            *_loop_nest(reduction_variables, reduction_ranges, body),
            # Storing the accumulator to the final buffer
            *handler.finalizers,
        ]
    body = _loop_nest(pointwise_variables, pointwise_ranges, body)

    signature = _signature("kernel", name, handler.reads, written_names)
    signature[-1] += ":"
    return "\n".join([*signature, *_indent(body)])


def _signature(
    keyword: str,
    name: str,
    read_names: Iterable[str],
    written_names: OrderedSet[str],
    comment: str | None = None,
) -> list[str]:
    # A mutation output is written into the buffer it mutates, so that buffer
    # is a `mutate` input instead, even if the loop body never loads it.
    real_names = {written: _real_name(written) for written in written_names}
    mutated_names = OrderedSet(
        real for written, real in real_names.items() if real != written
    )
    output_names = [written for written, real in real_names.items() if real == written]
    suffix = f"  # {comment}" if comment else ""
    inputs = [
        _tensor_declaration(read, mutable=read in mutated_names)
        for read in OrderedSet([*mutated_names, *read_names])
        if read not in written_names
    ]
    outputs = ", ".join(_tensor_declaration(output) for output in output_names)
    if len(output_names) > 1:
        outputs = f"({outputs})"
    arrow = f" -> {outputs}" if outputs else ""
    if not inputs:
        return [f"{keyword} {name}(){arrow}{suffix}"]
    return [
        f"{keyword} {name}({suffix}",
        *(
            f"    {value}{',' if index + 1 < len(inputs) else ''}"
            for index, value in enumerate(inputs)
        ),
        f"){arrow}",
    ]


def _format_extern_kernel(name: str, kernel) -> str:
    written_names = OrderedSet(
        output.get_name() for output in _multi_outputs(kernel) or kernel.get_outputs()
    )
    # Buffer.get_read_names shadows the operation's reads on extern kernels.
    read_names = [dep.name for dep in kernel.get_read_writes().reads]
    return "\n".join(
        _signature(
            "extern_kernel", name, read_names, written_names, _extern_callee(kernel)
        )
    )


def _multi_outputs(kernel) -> list[Any]:
    """The MultiOutput buffers a kernel with a MultiOutputLayout returns."""
    from torch._inductor import ir

    if not isinstance(getattr(kernel, "layout", None), ir.MultiOutputLayout):
        return []

    def flatten(value) -> Iterator[Any]:
        if isinstance(value, dict):
            value = value.values()
        if isinstance(value, (list, tuple, ValuesView)):
            for item in value:
                yield from flatten(item)
        elif isinstance(value, ir.MultiOutput):
            yield value

    return list(flatten(getattr(kernel, "outputs", None) or ()))


def _is_folded_multi_output(operation) -> bool:
    """MultiOutput selectors are shown as outputs of the kernel they index."""
    from torch._inductor import ir

    if not isinstance(operation, ir.MultiOutput) or not operation.inputs:
        return False
    return any(output is operation for output in _multi_outputs(operation.inputs[0]))


def _extern_callee(kernel) -> str | None:
    # e.g. extern_kernels.mm or torch.ops.aten.sort.stable; MultiOutput has no
    # kernel of its own, so fall back to the aten ops it was lowered from.
    if kernel.python_kernel_name:
        return kernel.python_kernel_name
    targets = sorted(OrderedSet(str(origin.target) for origin in kernel.origins))
    return ", ".join(targets) or None


def format_scheduler_node(node) -> str:
    from torch._inductor import ir
    from torch._inductor.scheduler import ExternKernelSchedulerNode, SchedulerNode

    name = node.get_name()
    if isinstance(node, ExternKernelSchedulerNode):
        return _format_extern_kernel(name, node.node)
    if not isinstance(node, SchedulerNode) or not isinstance(
        node.node, ir.ComputedBuffer
    ):
        return f"kernel {name}:\n    unimplemented {type(node).__name__}"
    if node._body is None:
        return f"kernel {name}:\n    unimplemented {type(node.node).__name__}"

    def run(pointwise_variables, reduction_variables):
        node._body(
            pointwise_variables, reduction_variables, allow_same_symbol_in_index=True
        )

    pointwise_ranges, reduction_ranges = node.get_ranges()
    written_names = OrderedSet(output.get_name() for output in node.get_outputs())
    return _format_loop_body(
        name, written_names, run, pointwise_ranges, reduction_ranges
    )


def _format_fused_node(node) -> str:
    from torch._inductor import ir
    from torch._inductor.codegen.simd import (
        DisableReduction,
        EnableReduction,
        SIMDKernel,
    )

    name = node.get_name()
    device, (numel, rnumel) = node.group
    backend = node.scheduler.get_backend(device)
    # CUDACombinedScheduling hands fused pointwise/reduction nodes to Triton.
    backend = getattr(backend, "_triton_scheduling", backend)
    if not hasattr(backend, "generate_node_schedule"):
        return f"kernel {name}:\n    unimplemented {type(backend).__name__}"

    # Nodes between DisableReduction and EnableReduction run once per p0,
    # outside any r0 loop; each other run of nodes shares one r0 loop.
    segments: list[tuple[bool, list[Any]]] = [(True, [])]
    for item in backend.generate_node_schedule(node.get_nodes(), numel, rnumel):
        if item is DisableReduction:
            segments.append((False, []))
        elif item is EnableReduction:
            segments.append((True, []))
        else:
            segments[-1][1].append(item)

    pointwise_variable = sympy_index_symbol("p0")
    reduction_variable = sympy_index_symbol("r0")
    handler = _PrettyOpsHandler([reduction_variable])
    variable_ranges = {pointwise_variable: numel, reduction_variable: rnumel}

    def set_ranges(*ranges):
        # Unflatten each kernel loop variable into the node's sizes, as a view would.
        return [
            [
                V.graph.sizevars.simplify_with_ranges(index, variable_ranges)
                for index in ir.View._dynamic_reshape_indexer(
                    sizes, [sympy_product(sizes)]
                )([variable])
            ]
            for variable, sizes in zip((pointwise_variable, reduction_variable), ranges)
        ]

    lines: list[str] = []
    try:
        with V.set_ops_handler(handler):
            for inside_reduction, snodes in segments:
                if not snodes:
                    continue
                groups = [numel, rnumel if inside_reduction else 1]
                for snode in snodes:
                    if handler.body:
                        handler.body.append("")
                    handler.body.append(f"# {snode.get_name()}")
                    snode._body(
                        *SIMDKernel.map_kernel_groups_to_node_sizes(
                            groups, snode.get_ranges(), set_ranges
                        ),
                        allow_same_symbol_in_index=True,
                    )
                body = handler.body
                if inside_reduction and rnumel != 1:
                    body = [
                        *handler.initializers,
                        *_loop_nest([reduction_variable], [rnumel], body),
                        *handler.finalizers,
                    ]
                lines.extend(body)
                if handler.finalizers:
                    lines.append("")
                handler.body, handler.initializers, handler.finalizers = [], [], []
    except _Unsupported as exc:
        return f"kernel {name}:\n    unimplemented {exc}"
    except Exception as exc:
        return f"kernel {name}:\n    unimplemented {type(exc).__name__}: {exc}"
    if lines and not lines[-1]:
        lines.pop()

    # Buffers only used inside the fused kernel are neither inputs nor outputs.
    operation_names = node.get_operation_names()
    buffer_names = node.get_buffer_names()
    written_names = OrderedSet(
        buffer.get_name()
        for buffer in node.get_outputs()
        if any(user.get_name() not in operation_names for user in buffer.users)
    )
    read_names = [read for read in handler.reads if read not in buffer_names]
    signature = _signature("kernel", name, read_names, written_names)
    signature[-1] += ":"
    body = _loop_nest([pointwise_variable], [numel], lines)
    return "\n".join([*signature, *_indent(body)])


def format_computed_buffer(buffer) -> str:
    from torch._inductor import ir

    name = buffer.get_operation_name()
    if isinstance(buffer, ir.ExternKernel):
        return _format_extern_kernel(name, buffer)
    if not isinstance(buffer, ir.ComputedBuffer):
        return f"kernel {name}:\n    unimplemented {type(buffer).__name__}"

    body = buffer.get_store_function()
    pointwise_ranges = buffer.get_pointwise_size()
    reduction_ranges = buffer.get_reduction_size()

    def run(pointwise_variables, reduction_variables):
        if reduction_ranges:
            body(pointwise_variables, reduction_variables)
        else:
            body(pointwise_variables)

    written_names = OrderedSet(output.get_name() for output in buffer.get_outputs())
    return _format_loop_body(
        name, written_names, run, pointwise_ranges, reduction_ranges
    )


def format_pre_fusion_ir(nodes: Sequence[Any]) -> str:
    """Render pre-fusion SchedulerNodes using their current loop bodies."""
    return "\n\n".join(
        format_scheduler_node(node)
        for node in nodes
        if not _is_folded_multi_output(getattr(node, "node", None))
    )


def format_post_lowering_ir(operations: Sequence[Any]) -> str:
    """Render lowered operations using their default, unsimplified loop bodies."""
    return "\n\n".join(
        format_computed_buffer(operation)
        for operation in operations
        if not _is_folded_multi_output(operation)
    )


def format_post_fusion_ir(nodes: Sequence[Any]) -> str:
    """Render each fused kernel as one loop nest shared by the nodes it fuses."""
    from torch._inductor.scheduler import FusedSchedulerNode

    return "\n\n".join(
        # Subclasses (foreach, mix-order, ...) have their own codegen.
        _format_fused_node(node)
        if type(node) is FusedSchedulerNode
        else format_scheduler_node(node)
        for node in nodes
        if not _is_folded_multi_output(getattr(node, "node", None))
    )
