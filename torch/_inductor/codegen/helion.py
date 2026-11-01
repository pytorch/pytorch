from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Any, cast, TYPE_CHECKING

import sympy

import torch
from torch._subclasses.fake_impls import _compute_stride
from torch.utils._ordered_set import OrderedSet
from torch.utils._sympy.functions import FloorDiv, ModularIndexing

from .. import config, metrics
from ..dependencies import MemoryDep
from ..ir import (
    FixedLayout,
    IRNode,
    MutableBox,
    MutationLayoutSHOULDREMOVE,
    ReinterpretView,
)
from ..ops_handler import DefaultHandler
from ..scheduler import SchedulerNode
from ..sizevars import SimplifyIndexing
from ..utils import IndentedBuffer
from ..virtualized import V
from .common import PythonPrinter
from .simd import SIMDKernel
from .simd_kernel_features import DisableReduction, EnableReduction, SIMDKernelFeatures
from .triton import TritonScheduling


if TYPE_CHECKING:
    from collections.abc import Callable


class HelionPythonPrinter(PythonPrinter):
    def _print_min_max(self, expr: sympy.Expr, name: str) -> str:
        values = []
        for arg in expr.args:
            value = self.doprint(arg, simplify=False)
            if not arg.free_symbols:
                dtype = torch.int64 if arg.is_integer else torch.float64
                value = f"hl.full([], {value}, dtype={dtype})"
            values.append(value)
        result = values[0]
        for value in values[1:]:
            result = f"torch.{name}({result}, {value})"
        return result

    def _print_Min(self, expr: sympy.Expr) -> str:
        return self._print_min_max(expr, "minimum")

    def _print_Max(self, expr: sympy.Expr) -> str:
        return self._print_min_max(expr, "maximum")


@dataclass(frozen=True)
class HelionValue:
    name: str
    depending_indices: tuple[str, ...]


@dataclass(frozen=True)
class HelionTensorArg:
    name: str
    operand: IRNode

    @property
    def source(self) -> str:
        return self.operand.get_name()


class HelionKernel(DefaultHandler):
    """Emit a source-only preview of MM/BMM schedules and pointwise epilogues."""

    def __init__(
        self, features: SIMDKernelFeatures, tiling: dict[str, sympy.Expr]
    ) -> None:
        if tuple(tiling) not in (("y", "x", "r0_"), ("z", "y", "x", "r0_")) or not all(
            isinstance(size, sympy.Integer) for size in tiling.values()
        ):
            raise NotImplementedError("Helion preview requires static MM/BMM tiling")
        if not config.triton.codegen_upcast_to_fp32:
            raise NotImplementedError("Helion preview requires FP32 compute upcasting")
        nodes = tuple(features.scheduler_nodes())
        self.matmul_inputs = {
            name: V.graph.helion_matmul_inputs[name]
            for node in nodes
            for name in node.get_buffer_names()
            if name in V.graph.helion_matmul_inputs
        }
        if len(self.matmul_inputs) != features.op_counts().get("dot", 0):
            raise NotImplementedError("Helion preview requires lowered MM/BMM operands")
        if any(
            operand.get_dtype() not in (torch.float16, torch.bfloat16, torch.float32)
            for operands in self.matmul_inputs.values()
            for operand in operands
        ):
            raise NotImplementedError("Helion preview supports FP16/BF16/FP32 matmul")
        self.features = features
        self.tiling = tiling
        self.outer_axes = tuple(tiling)[:-1]
        self.symbols = {
            axis: sympy.Symbol(f"helion_{axis}", integer=True) for axis in tiling
        }
        self.tiles = {"z": "tile_z", "y": "tile_y", "x": "tile_x", "r0_": "tile_r"}
        self.accumulators = {
            name: "acc" if number == 0 else f"acc{number}"
            for number, name in enumerate(self.matmul_inputs)
        }
        self.current_matmul: str | None = None
        self.inputs: OrderedSet[str] = OrderedSet()
        fused_names = OrderedSet(node.get_name() for node in nodes)
        self.outputs = OrderedSet(
            output.get_name()
            for node in nodes
            for output in node.get_outputs()
            if not isinstance(output.node.get_layout(), MutationLayoutSHOULDREMOVE)
            and not V.graph.scheduler.can_buffer_be_removed_through_fusion(
                output.get_name(), fused_names
            )
        )
        self.arguments: dict[tuple[Any, ...], HelionTensorArg] = {}
        self.store_forwarding: dict[tuple[str, sympy.Expr], HelionValue] = {}
        self.indirect_indices: dict[sympy.Symbol, HelionValue] = {}
        self.active_mask: HelionValue | None = None
        self.reduction_body = IndentedBuffer()
        self.epilogue = IndentedBuffer()
        self.body = self.reduction_body
        # A tile_r temporary cannot be reused outside the K loop.
        self.cse: dict[
            IndentedBuffer, dict[tuple[str, tuple[str, ...]], HelionValue]
        ] = {self.reduction_body: {}, self.epilogue: {}}
        self.counter = itertools.count()
        self.dots_emitted: OrderedSet[str] = OrderedSet()
        self.executable = False
        self.kernel_name: str | None = None

    @staticmethod
    def compute_dtype(dtype: torch.dtype) -> torch.dtype:
        if config.triton.codegen_upcast_to_fp32 and dtype in (
            torch.float16,
            torch.bfloat16,
        ):
            return torch.float32
        return dtype

    def emit(self, expression: str, depending_indices: tuple[str, ...]) -> HelionValue:
        cache = self.cse[self.body]
        key = (expression, depending_indices)
        if key not in cache:
            name = f"tmp{next(self.counter)}"
            self.body.writeline(f"{name} = {expression}")
            cache[key] = HelionValue(name, depending_indices)
        return cache[key]

    def align(
        self,
        value: HelionValue,
        depending_indices: tuple[str, ...],
        *,
        for_index: bool = False,
    ) -> str:
        # Pointwise ops broadcast trailing axes: bias[x] + acc[y, x] needs no
        # reshape, while row[y] needs row.reshape([tile_y, 1]). Tensor index
        # coordinates need the full rank, e.g. x.reshape([1, 1, tile_x]).
        if not value.depending_indices or value.depending_indices == depending_indices:
            return value.name
        dropped = tuple(
            axis for axis in value.depending_indices if axis not in depending_indices
        )
        if dropped:
            if any(self.tiling[axis] != 1 for axis in dropped):
                raise NotImplementedError("Helion preview cannot drop a tensor axis")
            name = value.name
            for dim in reversed(range(len(value.depending_indices))):
                if value.depending_indices[dim] in dropped:
                    name += f".squeeze({dim})"
            value = HelionValue(
                name,
                tuple(axis for axis in value.depending_indices if axis not in dropped),
            )
        order = tuple(
            axis for axis in depending_indices if axis in value.depending_indices
        )
        result = value.name
        if order != value.depending_indices:
            permutation = [value.depending_indices.index(axis) for axis in order]
            result += f".permute({', '.join(map(str, permutation))})"
        if len(order) < len(depending_indices) and (
            for_index or order != depending_indices[-len(order) :]
        ):
            shape = [
                self.tiles[axis] if axis in value.depending_indices else "1"
                for axis in depending_indices
            ]
            result += f".reshape([{', '.join(shape)}])"
        return result

    def simplify_index(self, index: sympy.Expr) -> sympy.Expr:
        # Expose quotient/remainder cancellation before inferring tensor axes:
        # x % B + B * (x // B) becomes x, so it no longer needs two indices.
        index = sympy.sympify(index).xreplace(
            {
                self.symbols[axis]: sympy.S.Zero
                for axis, size in self.tiling.items()
                if size == 1
            }
        )
        expanded = index.replace(
            ModularIndexing,
            lambda base, divisor, modulus: FloorDiv(base, divisor)
            - modulus * FloorDiv(base, divisor * modulus),
        )
        var_ranges = dict(zip(self.symbols.values(), self.tiling.values()))
        expanded = sympy.expand(expanded)
        simplified = V.graph.sizevars.simplify_with_ranges(expanded, var_ranges)
        # Keep genuine nonlinear maps compact for the fallback index decoder.
        return index if simplified.has(FloorDiv) else simplified

    def tensor_arg(
        self,
        operand: IRNode,
        alias: str | None = None,
        *,
        index: sympy.Expr | None = None,
    ) -> HelionTensorArg:
        r"""tensor_arg(operand, alias=None, *, index=None) -> HelionTensorArg

        Preserve storage views and recover logical tile axes for a tensor argument.
        """
        while isinstance(operand, MutableBox):
            operand = operand.data
        layout = operand.get_layout()
        if index is not None:
            # Drop constant singleton coordinates through a caller view:
            # input[g, 0, y, x] becomes input_squeezed[g, y, x]. Keeping the
            # unused dimension can change Helion's advanced-index result rank.
            active = [
                dim
                for dim, size in enumerate(layout.size)
                if size != 1 or any(self.tiling[axis] == 1 for axis in self.outer_axes)
            ]
            if len(active) != len(layout.size):
                layout = FixedLayout(
                    layout.device,
                    layout.dtype,
                    [layout.size[dim] for dim in active],
                    [layout.stride[dim] for dim in active],
                    layout.offset,
                    layout.is_pinned,
                )
                operand = ReinterpretView(data=operand, layout=layout)
            # Recover a view from the linear address, not the original rank.
            # For N*y + x, a contiguous (B, M, N) buffer can be passed as
            # reinterpret_tensor(buffer, (B*M, N), (N, 1), 0), then indexed
            # inside the kernel as buffer[tile_y, tile_x].
            symbols = index.free_symbols
            axes = tuple(
                axis
                for axis in self.outer_axes
                if self.symbols[axis] in symbols or self.tiling[axis] == 1
            )
            old_size, old_stride = list(layout.size), list(layout.stride)
            size = [self.tiling[axis] for axis in axes]
            expression = sympy.expand(index)
            stride = [expression.coeff(self.symbols[axis]) for axis in axes]
            expected = layout.offset + sum(
                self.symbols[axis] * s for axis, s in zip(axes, stride)
            )
            integers = (*old_size, *old_stride, *stride)
            if (
                all(isinstance(v, (int, sympy.Integer)) for v in integers)
                and math.prod(size) == math.prod(old_size)
                and (size != old_size or stride != old_stride)
                and V.graph.sizevars.statically_known_equals(index, expected)
            ):
                # Match storage chunks independently of the logical dimension order.
                old_order = sorted(range(len(old_size)), key=lambda d: -old_stride[d])
                new_order = sorted(range(len(size)), key=lambda d: -stride[d])
                view_stride = _compute_stride(
                    [int(old_size[d]) for d in old_order],
                    [int(old_stride[d]) for d in old_order],
                    [int(size[d]) for d in new_order],
                )
                if view_stride == [stride[d] for d in new_order]:
                    layout = FixedLayout(
                        layout.device,
                        layout.dtype,
                        size,
                        stride,
                        layout.offset,
                        layout.is_pinned,
                    )
                    operand = ReinterpretView(data=operand, layout=layout)
        key = (
            operand.get_name(),
            tuple(layout.size),
            tuple(layout.stride),
            layout.offset,
        )
        if key not in self.arguments:
            if alias is None:
                prefix = "out" if operand.get_name() in self.outputs else "in"
                number = sum(
                    arg.name.startswith(prefix) for arg in self.arguments.values()
                )
                alias = f"{prefix}{number}"
            self.arguments[key] = HelionTensorArg(alias, operand)
        return self.arguments[key]

    def tensor_indices(
        self, operand: IRNode, index: sympy.Expr
    ) -> tuple[list[str], tuple[str, ...]]:
        layout = operand.get_layout()
        # For shape (M, N), stride (N, 1), match N*y + x to [tile_y, tile_x].
        # A bias with shape (N,) and index x becomes bias[tile_x].
        candidates = [
            [
                axis
                for axis in self.outer_axes
                if V.graph.sizevars.statically_known_equals(size, self.tiling[axis])
            ]
            + ([None] if size == 1 else [])
            for size in layout.size
        ]
        for axes in itertools.product(*candidates):
            active = tuple(axis for axis in axes if axis is not None)
            if len(active) != len(OrderedSet(active)):
                continue
            expected = layout.offset + sum(
                self.symbols[axis] * stride
                for axis, stride in zip(axes, layout.stride)
                if axis is not None
            )
            if V.graph.sizevars.statically_known_equals(
                index, self.simplify_index(expected)
            ):
                return (
                    [self.tiles[axis] if axis is not None else "0" for axis in axes]
                    or ["()"],
                    active,
                )
        # A flattened MM row y can address a 3D output as [y//M, y%M, x].
        # Decode the existing layout; keep ordinary Tile loads above for TMA.
        order = sorted(range(len(layout.size)), key=lambda dim: layout.stride[dim])
        span = 1
        for dim in order:
            size, stride = layout.size[dim], layout.stride[dim]
            if not isinstance(size, (int, sympy.Integer)) or not isinstance(
                stride, (int, sympy.Integer)
            ):
                raise NotImplementedError("Helion indexing requires static layouts")
            if size == 1:
                continue
            if stride < span:
                raise NotImplementedError(
                    "Helion indexing requires nonoverlapping strides"
                )
            span += (size - 1) * stride
        offset = index - layout.offset
        coordinates = []
        var_ranges = dict(zip(self.symbols.values(), self.tiling.values()))
        for dim, (size, stride) in enumerate(zip(layout.size, layout.stride)):
            coordinate = FloorDiv(offset, stride)
            if dim != order[-1]:
                # Mod drops whole rows, including negative indirect row indices.
                coordinate = sympy.Mod(coordinate, size).replace(
                    sympy.Mod, lambda base, modulus: ModularIndexing(base, 1, modulus)
                )
            coordinates.append(
                sympy.S.Zero
                if size == 1
                else V.graph.sizevars.simplify_with_ranges(coordinate, var_ranges)
            )
        restored = sum(
            coord * stride for coord, stride in zip(coordinates, layout.stride)
        )
        if V.graph.sizevars.simplify_with_ranges(restored - offset, var_ranges) != 0:
            raise NotImplementedError(f"Cannot recover Helion tensor indices: {index}")
        symbols = OrderedSet().union(*(coord.free_symbols for coord in coordinates))
        axes = tuple(
            axis
            for axis in self.outer_axes
            if self.symbols[axis] in symbols
            or any(
                axis in self.indirect_indices[symbol].depending_indices
                for symbol in symbols
                if symbol in self.indirect_indices
            )
        )
        replacements = {
            # All explicit coordinates share one broadcast rank. For [z, y, x],
            # use z[:, None, None], y[None, :, None], and x[None, None, :].
            symbol: sympy.Symbol(
                self.align(
                    HelionValue(f"{self.tiles[axis]}.index", (axis,)),
                    axes,
                    for_index=True,
                )
            )
            for axis, symbol in self.symbols.items()
            if symbol in symbols
        }
        replacements.update(
            {
                symbol: sympy.Symbol(self.align(value, axes, for_index=True))
                for symbol, value in self.indirect_indices.items()
                if symbol in symbols
            }
        )
        if symbols - replacements.keys():
            raise NotImplementedError(f"Unsupported Helion indexing: {index}")
        printer = HelionPythonPrinter()
        return [
            printer.doprint(coord.xreplace(replacements), simplify=False)
            for coord in coordinates
        ], axes

    def indirect_indexing(
        self,
        value: HelionValue,
        size: sympy.Expr,
        check: bool = True,
        wrap_neg: bool = True,
    ) -> sympy.Symbol:
        # out.index_add_(0, indices, acc) becomes hl.atomic_add(out, [indices, x], acc).
        # Helion's tensor indexing handles negative-index wrapping.
        # For wrap_neg=False, this preview assumes valid nonnegative indices;
        # it does not yet emit Inductor's runtime index checks.
        if self.executable:
            raise NotImplementedError(
                "Executable Helion matmul does not support indirect indexing"
            )
        symbol = sympy.Symbol(
            f"helion_indirect{len(self.indirect_indices)}", integer=True
        )
        self.indirect_indices[symbol] = value
        return symbol

    def load(self, name: str, index: sympy.Expr) -> HelionValue:
        index = self.simplify_index(index)
        if (name, index) in self.store_forwarding:
            return self.store_forwarding[name, index]
        if self.current_matmul is not None:
            operands = self.matmul_inputs[self.current_matmul]
            number = tuple(self.matmul_inputs).index(self.current_matmul)
            batch = ("z",) if "z" in self.tiling else ()
            for operand, axes, alias in (
                (operands[0], (*batch, "y", "r0_"), "A"),
                (operands[1], (*batch, "r0_", "x"), "B"),
            ):
                if operand.get_name() != name:
                    continue
                expected = operand.make_indexer()([self.symbols[axis] for axis in axes])
                if V.graph.sizevars.statically_known_equals(
                    index, self.simplify_index(expected)
                ):
                    argument = self.tensor_arg(
                        operand, alias + (str(number) if number else "")
                    )
                    self.inputs.add(name)
                    coordinates = ", ".join(self.tiles[axis] for axis in axes)
                    dtype = self.compute_dtype(operand.get_dtype())
                    return self.emit(
                        f"{argument.name}[{coordinates}].to({dtype})", axes
                    )
            raise NotImplementedError("Helion preview does not support fused prologues")
        operand = V.graph.get_buffer(name)
        argument = self.tensor_arg(operand, index=index)
        coordinates, axes = self.tensor_indices(argument.operand, index)
        self.inputs.add(name)
        dtype = self.compute_dtype(operand.get_dtype())
        if self.active_mask is not None:
            mask = self.align(self.active_mask, axes, for_index=True)
            return self.emit(
                f"hl.load({argument.name}, [{', '.join(coordinates)}], "
                f"extra_mask={mask}).to({dtype})",
                axes,
            )
        return self.emit(f"{argument.name}[{', '.join(coordinates)}].to({dtype})", axes)

    def constant(self, value: bool | float | int, dtype: torch.dtype) -> HelionValue:
        dtype = self.compute_dtype(dtype)
        if isinstance(value, float) and not math.isfinite(value):
            literal = f"float({str(value)!r})"
        else:
            literal = repr(value)
        return self.emit(f"hl.full([], {literal}, dtype={dtype})", ())

    def index_expr(self, index: sympy.Expr, dtype: torch.dtype) -> HelionValue:
        # Tile indices are tensors. A symbolic max(0, y - x) must become a
        # tensor operation; Python max() cannot compare whole tile tensors.
        index = self.simplify_index(index)
        symbols = index.free_symbols
        if not symbols:
            value = int(index) if index.is_Integer else float(index)
            return self.constant(value, dtype)
        indirect_axes = OrderedSet(
            [
                axis
                for symbol in symbols
                if symbol in self.indirect_indices
                for axis in self.indirect_indices[symbol].depending_indices
            ]
        )
        axes = tuple(
            axis
            for axis in ("z", "y", "r0_", "x")
            if axis in self.symbols
            and (self.symbols[axis] in symbols or axis in indirect_axes)
        )
        replacements = {}
        for axis in axes:
            symbol = self.symbols[axis]
            if symbol not in symbols:
                continue
            coordinate = f"{self.tiles[axis]}.index"
            if len(axes) > 1:
                broadcast = ", ".join(":" if dim == axis else "None" for dim in axes)
                coordinate += f"[{broadcast}]"
            replacements[symbol] = sympy.Symbol(coordinate)
        replacements.update(
            {
                symbol: sympy.Symbol(self.align(value, axes, for_index=True))
                for symbol, value in self.indirect_indices.items()
                if symbol in symbols
            }
        )
        if symbols - replacements.keys():
            raise NotImplementedError(f"Unsupported Helion index expression: {index}")
        expression = HelionPythonPrinter().doprint(
            index.xreplace(replacements), simplify=False
        )
        return self.emit(f"({expression}).to({self.compute_dtype(dtype)})", axes)

    def masked(
        self,
        mask: HelionValue,
        body: Callable[[], HelionValue],
        other: bool | float | int,
    ) -> HelionValue:
        previous_mask = self.active_mask
        if previous_mask is not None:
            mask = self._default("bitwise_and", (previous_mask, mask), {})
        self.active_mask = mask
        try:
            result = body()
        finally:
            self.active_mask = previous_mask
        literal = (
            f"float({str(other)!r})"
            if isinstance(other, float) and not math.isfinite(other)
            else repr(other)
        )
        fallback = self.emit(
            f"torch.full_like({result.name}, {literal})", result.depending_indices
        )
        return self._default("where", (mask, result, fallback), {})

    def to_dtype(
        self,
        value: HelionValue,
        dtype: torch.dtype,
        src_dtype: torch.dtype | None = None,
        use_compute_types: bool = True,
    ) -> HelionValue:
        if use_compute_types:
            dtype = self.compute_dtype(dtype)
        return self.emit(f"{value.name}.to({dtype})", value.depending_indices)

    def dot(self, lhs: HelionValue, rhs: HelionValue) -> HelionValue:
        name = self.current_matmul
        if name is None or name in self.dots_emitted:
            raise NotImplementedError("Helion dot must belong to a lowered MM/BMM node")
        self.dots_emitted.add(name)
        batch = ("z",) if "z" in self.tiling else ()
        left = self.align(lhs, (*batch, "y", "r0_"))
        right = self.align(rhs, (*batch, "r0_", "x"))
        acc = self.accumulators[name]
        # Horizontal fusion needs one accumulator per MM, not one shared sum.
        op = "baddbmm" if batch else "addmm"
        self.body.writeline(f"{acc} = torch.{op}({acc}, {left}, {right})")
        return HelionValue(acc, self.outer_axes)

    def reduction(
        self,
        dtype: torch.dtype,
        src_dtype: torch.dtype,
        reduction_type: str,
        value: HelionValue,
    ) -> HelionValue:
        if reduction_type != "dot":
            raise NotImplementedError("Helion preview supports one dot reduction")
        return value

    def store(
        self,
        name: str,
        index: sympy.Expr,
        value: HelionValue,
        mode: str | None = None,
    ) -> None:
        index = self.simplify_index(index)
        if self.active_mask is not None:
            raise NotImplementedError("Helion preview does not support masked stores")
        if mode not in (None, "atomic_add"):
            raise NotImplementedError(f"Unsupported Helion store mode: {mode}")
        operand = V.graph.get_buffer(name)
        layout = operand.get_layout()
        if isinstance(layout, MutationLayoutSHOULDREMOVE):
            if mode is None:
                raise NotImplementedError(
                    "Helion preview supports atomic_add mutations"
                )
            operand = layout.target
            self.inputs.add(operand.get_name())
        if mode == "atomic_add" or name in self.outputs:
            # Physical stores run in the epilogue and may alias cached loads.
            self.cse[self.epilogue].clear()
        if mode == "atomic_add":
            if operand.get_dtype() == torch.bool:
                raise NotImplementedError(
                    "Helion preview does not support bool atomic stores"
                )
            argument = self.tensor_arg(operand, index=index)
            coordinates, axes = self.tensor_indices(argument.operand, index)
            stored = self.align(value, axes)
            self.epilogue.writeline(
                f"hl.atomic_add({argument.name}, [{', '.join(coordinates)}], "
                f"{stored}.to({operand.get_dtype()}))"
            )
            return
        self.store_forwarding[name, sympy.sympify(index)] = value
        if name in self.outputs:
            argument = self.tensor_arg(operand, index=index)
            coordinates, axes = self.tensor_indices(argument.operand, index)
            stored = self.align(value, axes)
            self.epilogue.writeline(
                f"{argument.name}[{', '.join(coordinates)}] = "
                f"{stored}.to({operand.get_dtype()})"
            )

    def store_reduction(self, name: str, index: sympy.Expr, value: HelionValue) -> None:
        self.store(name, index, value)

    binary_ops = {
        "add": "+",
        "sub": "-",
        "mul": "*",
        "truediv": "/",
        "eq": "==",
        "ne": "!=",
        "lt": "<",
        "le": "<=",
        "gt": ">",
        "ge": ">=",
        "bitwise_and": "&",
        "and_": "&",
    }
    torch_ops = OrderedSet(
        [
            "relu",
            "erf",
            "tanh",
            "neg",
            "sigmoid",
            "exp",
            "expm1",
            "abs",
            "log",
            "logical_not",
            "where",
            "minimum",
            "maximum",
            "logical_and",
        ]
    )

    def _default(self, name: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        binary = self.binary_ops
        if not kwargs and (name in binary or name in self.torch_ops):
            # Operand order must not set axis order: bias[x] + acc[y, x]
            # keeps [y, x], and batched values keep [z, y, x].
            axes = OrderedSet().union(*(value.depending_indices for value in args))
            depending_indices = tuple(
                axis for axis in ("z", "y", "r0_", "x") if axis in axes
            )
            values = [self.align(value, depending_indices) for value in args]
            # Torch pointwise ops are lowered by Helion; align the tile axes first.
            expression = (
                f"{values[0]} {binary[name]} {values[1]}"
                if name in binary
                else f"torch.{name}({', '.join(values)})"
            )
            return self.emit(expression, depending_indices)
        raise NotImplementedError(f"Unsupported Helion preview op: {name}")

    def codegen_kernel(self, *, executable: bool = False) -> str:
        """Replay the SIMD schedule and emit the kernel plus its metadata-view caller."""

        self.executable = executable

        def set_ranges(*ranges):
            # SIMD may split y into batch/row: y//M and y%M, without another loop.
            return [
                [
                    FloorDiv(symbol, math.prod(sizes[dim + 1 :]))
                    if dim == 0
                    else ModularIndexing(symbol, math.prod(sizes[dim + 1 :]), size)
                    for dim, size in enumerate(sizes)
                ]
                for symbol, sizes in zip(self.symbols.values(), ranges)
            ]

        var_ranges = dict(zip(self.symbols.values(), self.tiling.values()))
        for node in self.features.node_schedule:
            if node is DisableReduction:
                self.body = self.epilogue
            elif node is EnableReduction:
                self.body = self.reduction_body
            else:
                node = cast(SchedulerNode, node)
                self.current_matmul = next(
                    (
                        name
                        for name in node.get_buffer_names()
                        if name in self.matmul_inputs
                    ),
                    None,
                )
                groups = list(self.tiling.values())
                if self.body is self.epilogue:
                    groups[-1] = sympy.S.One
                indices = SIMDKernel.map_kernel_groups_to_node_sizes(
                    groups, node.get_ranges(), set_ranges
                )
                with V.set_ops_handler(SimplifyIndexing(self, var_ranges)):
                    node._body(*indices)
        code = IndentedBuffer()
        code.writelines(
            ["import torch", "import helion", "import helion.language as hl", ""]
        )
        tensor_args = sorted(
            self.arguments.values(), key=lambda arg: arg.source in self.outputs
        )
        if any(isinstance(arg.operand, ReinterpretView) for arg in tensor_args):
            code.writeline(
                "from torch._C._dynamo.guards import _reinterpret_tensor as reinterpret_tensor"
            )
            code.writeline("")
        if executable:
            code.writeline(
                "from torch._inductor.runtime.helion import HelionKernelWrapper"
            )
            backend = {"triton": "triton", "cutedsl": "cute"}[config.helion_backend]
            code.writeline(
                f'@helion.kernel(backend="{backend}", static_shapes=True, '
                f'autotune_effort="{config.helion_autotune_effort}", '
                "ref_mode=helion.RefMode.OFF)"
            )
        else:
            code.writeline("@helion.kernel(static_shapes=True)")
        self.kernel_name = "helion_" + "_".join(
            node.get_name() for node in self.features.scheduler_nodes()
        )
        name = self.kernel_name
        signature = ", ".join(f"{arg.name}: torch.Tensor" for arg in tensor_args)
        code.writeline(f"def {name}({signature}) -> None:")
        with code.indent():
            tiles = ", ".join(self.tiles[axis] for axis in self.outer_axes)
            sizes = ", ".join(str(self.tiling[axis]) for axis in self.outer_axes)
            code.writeline(f"for {tiles} in hl.tile([{sizes}]):")
            with code.indent():
                for acc in self.accumulators.values():
                    code.writeline(f"{acc} = hl.zeros([{tiles}], dtype=torch.float32)")
                code.writeline(f"for tile_r in hl.tile({self.tiling['r0_']}):")
                with code.indent():
                    code.splice(self.reduction_body)
                code.splice(self.epilogue)
        if executable:
            code.writeline(f"{name} = HelionKernelWrapper({name})")
        code.writeline("")
        signature = ", ".join(
            f"{arg}: torch.Tensor" for arg in (*self.inputs, *self.outputs)
        )
        code.writeline(f"def call_{name}({signature}) -> None:")
        with code.indent():
            # a.shape == (33, 9, 8) can be passed as an existing (33, 72) view.
            # Helion then loads A[tile_y, tile_r], without r//8 or r%8.
            arguments = ", ".join(
                arg.operand.codegen_reference(code)
                if isinstance(arg.operand, ReinterpretView)
                else arg.source
                for arg in tensor_args
            )
            code.writeline(f"{name}({arguments})")
        return code.getvalue()


class HelionScheduling(TritonScheduling):
    """Use Helion for native matmul schedules and Triton for other work."""

    @staticmethod
    def format_kernel_definition(name: str, source: str, caller: str) -> str:
        escaped = source.replace("\\", "\\\\").replace('"""', '\\"""')
        return (
            f'{name}_source = """\\\n{escaped}"""\n'
            f"{name} = PyCodeCache.load({name}_source).{caller}"
        )

    def can_fuse(self, node1, node2):
        if node1.is_native_matmul() or node2.is_native_matmul():
            supported = (
                OrderedSet(HelionKernel.binary_ops)
                | HelionKernel.torch_ops
                | OrderedSet(
                    [
                        "load",
                        "store",
                        "store_reduction",
                        "constant",
                        "to_dtype",
                        "dot",
                        "reduction",
                        "index_expr",
                        "masked",
                        "output",
                    ]
                )
            )
            for node in (*node1.get_nodes(), *node2.get_nodes()):
                if not isinstance(node, SchedulerNode):
                    return False
                if OrderedSet(node._body.op_counts) - supported:
                    return False
                if node.is_reduction() and not node.is_native_matmul():
                    return False
                if any(
                    isinstance(dep, MemoryDep) and dep.is_indirect()
                    for dep in node.read_writes.reads_and_writes()
                ):
                    return False
        return super().can_fuse(node1, node2)

    can_fuse_vertical = can_fuse
    can_fuse_horizontal = can_fuse

    def codegen_combo_kernel(self, node):
        if node.is_native_matmul():
            raise NotImplementedError(
                "Helion matmul cannot be emitted as a Triton combo kernel"
            )
        with config.patch(cuda_backend="triton"):
            return super().codegen_combo_kernel(node)

    def generate_combo_kernel_code(self, subkernel_nodes, *args, **kwargs):
        if any(node.is_native_matmul() for node in subkernel_nodes):
            raise NotImplementedError(
                "Helion matmul cannot be emitted as a Triton combo kernel"
            )
        with config.patch(cuda_backend="triton"):
            return super().generate_combo_kernel_code(subkernel_nodes, *args, **kwargs)

    def generate_kernel_code_from_nodes(self, nodes, *args, **kwargs):
        if any(node.is_native_matmul() for node in nodes):
            raise NotImplementedError(
                "Helion matmul does not support Triton fusion benchmarking"
            )
        with config.patch(cuda_backend="triton"):
            return super().generate_kernel_code_from_nodes(nodes, *args, **kwargs)

    def codegen_node_schedule(self, kernel_features: SIMDKernelFeatures):
        if not kernel_features.is_native_matmul():
            # The non-matmul half of this backend is ordinary Triton codegen.
            with config.patch(cuda_backend="triton"):
                return super().codegen_node_schedule(kernel_features)
        if V.graph.cpp_wrapper or V.graph.fx_wrapper:
            raise NotImplementedError(
                "Helion native matmul requires the Python wrapper"
            )
        matmul = next(
            V.graph.helion_matmul_inputs[name]
            for node in kernel_features.scheduler_nodes()
            for name in node.get_buffer_names()
            if name in V.graph.helion_matmul_inputs
        )
        lhs, rhs = matmul
        tiling = {
            "y": sympy.sympify(lhs.get_size()[-2]),
            "x": sympy.sympify(rhs.get_size()[-1]),
            "r0_": sympy.sympify(lhs.get_size()[-1]),
        }
        if len(lhs.get_size()) == 3:
            tiling = {"z": sympy.sympify(lhs.get_size()[0]), **tiling}
        kernel = HelionKernel(kernel_features, tiling)
        source = kernel.codegen_kernel(executable=True)
        torch._logging.trace_structured(
            "artifact",
            metadata_fn=lambda: {"name": "helion_code", "encoding": "string"},
            payload_fn=lambda: source,
        )
        V.debug.helion_code(source)
        wrapper = V.graph.wrapper_code
        if source not in wrapper.src_to_kernel:
            name = f"helion_{wrapper.next_kernel_suffix()}"
            wrapper.src_to_kernel[source] = name
            wrapper.add_import_once("from torch._inductor.codecache import PyCodeCache")
            wrapper.define_kernel(
                name,
                self.format_kernel_definition(
                    name, source, f"call_{kernel.kernel_name}"
                ),
                gpu=False,
                standalone=True,
            )
            # pyrefly: ignore [bad-assignment]
            metrics.generated_kernel_count += 1
        name = wrapper.src_to_kernel[source]
        nodes = list(kernel_features.scheduler_nodes())
        for node in nodes:
            for output in node.get_outputs():
                if (
                    output.get_name() not in kernel.outputs
                    and not output.get_mutations()
                    and not output.get_aliases()
                ):
                    V.graph.removed_buffers.add(output.get_name())
            node.mark_run()
        self.codegen_comment(nodes, name)
        with wrapper.kernel_profile_scope(name, nodes):
            wrapper.writeline(
                wrapper.wrap_kernel_call(name, [*kernel.inputs, *kernel.outputs])
            )
        self.free_buffers_in_scheduler()
