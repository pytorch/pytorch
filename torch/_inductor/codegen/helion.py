"""Inductor codegen for Helion kernels.

``config.helion.native_matmul`` lowers CUDA mm/bmm/addmm/baddbmm to native
matmul IR (``ops.dot`` plus ``reduction("dot")``), and ``HelionScheduling``
emits each native matmul schedule, i.e. its dots and fused pointwise epilogue,
as one Helion kernel. Every other schedule is still Triton. For bf16
``torch.relu(a @ b + bias)`` the body of the generated ``@helion.kernel`` is::

    for tile_y, tile_x in hl.tile([40, 136]):
        acc = hl.zeros([tile_y, tile_x], dtype=torch.float32)
        for tile_r in hl.tile(72):
            tmp0 = in0[tile_y, tile_r]
            tmp1 = in1[tile_r, tile_x]
            acc = torch.addmm(acc, tmp0, tmp1)
        tmp2 = in2[tile_x].to(torch.float32)
        tmp3 = acc + tmp2
        tmp4 = torch.relu(tmp3)
        out0[tile_y, tile_x] = tmp4.to(torch.bfloat16)

test_mm_source in test/inductor/test_codegen_helion.py shows the whole module.
The wrapper binds its kernel as e.g. ``helion_fused_add_mm_relu_0`` and calls
``helion_fused_add_mm_relu_0(arg0_1, arg1_1, arg2_1, buf1)``, passing
``reinterpret_tensor`` views where the kernel indexes one.

The structure follows the Triton backend. SIMDScheduling picks the tiling and
runs each node body through ``HelionKernel``, the CSE names each value
``tmpN``, and ``HelionOverrides`` spells each op in torch semantics, which is
what Helion device code is.
"""

from __future__ import annotations

import collections
import itertools
import math
from dataclasses import dataclass
from typing import Any, cast, TYPE_CHECKING

import sympy

import torch
from torch.utils._ordered_set import OrderedSet
from torch.utils._sympy.functions import FloorDiv, ModularIndexing
from torch.utils._sympy.value_ranges import bound_sympy, ValueRanges

from .. import config
from ..ir import FixedLayout, IRNode, MutationLayoutSHOULDREMOVE, ReinterpretView
from ..scheduler import SchedulerNode, WhyNoFuse
from ..utils import (
    get_fused_kernel_name,
    IndentedBuffer,
    sympy_product,
    sympy_subs,
    upcast_compute_type,
)
from ..virtualized import V
from .common import CSEVariable, OpOverrides, PythonPrinter
from .simd import SIMDKernel
from .triton import TritonScheduling


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from ..scheduler import BaseSchedulerNode

# Shape of a kernel value: a tile name per dim, or 1 for a broadcast dim.
TileShape = tuple[str | int, ...]
TILE_NAMES = {"z": "tile_z", "y": "tile_y", "x": "tile_x", "r0_": "tile_r"}
TILE_AXES = {name: axis for axis, name in TILE_NAMES.items()}
# Order of tile dims in a subscript, so that dot operands read as
# in0[tile_y, tile_r] and in1[tile_r, tile_x].
AXIS_ORDER = ("z", "y", "r0_", "x")


class HelionPrinter(PythonPrinter):
    """Print index expressions over tile index tensors.

    Python min/max cannot compare tiles, so ``Min(y, 5)`` prints as
    ``torch.minimum(tile_y.index[:, None], hl.full([], 5, dtype=torch.int64))``.
    """

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


class HelionOverrides(OpOverrides):
    """Spell Inductor ops as Helion device code, i.e. torch ops on tiles.

    Arithmetic, comparisons and bitwise ops are inherited as Python operators
    (``tmp0 + tmp1``, ``tmp0 < tmp1``, ``tmp0 & tmp1``); every other op a Helion
    kernel supports is spelled below, one function per op.
    """

    @staticmethod
    def constant(value: bool | float | int, dtype: torch.dtype) -> str:
        # hl.full([], 2.0, dtype=torch.float32): a 0-d tensor broadcasts against
        # any tile and, unlike a Python scalar, keeps its dtype explicit.
        literal = repr(value)
        if isinstance(value, float) and not math.isfinite(value):
            literal = f"float({literal!r})"
        return f"hl.full([], {literal}, dtype={upcast_compute_type(dtype)})"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def to_dtype(
        x: CSEVariable,
        dtype: torch.dtype,
        src_dtype: torch.dtype | None = None,
        use_compute_types: bool = True,
    ) -> CSEVariable | str:
        if use_compute_types:
            dtype = upcast_compute_type(dtype)
        # No cast for a value that already has the dtype, e.g. a bf16 operand
        # that the native matmul body casts back to bf16 before the dot.
        if x.dtype == dtype:
            return x
        return f"{x}.to({dtype})"

    @staticmethod
    def index_expr(expr: sympy.Expr, dtype: torch.dtype) -> CSEVariable:
        return cast(HelionKernel, V.kernel).index_expr(expr, dtype)

    @staticmethod
    def value_expr(expr: sympy.Expr, dtype: torch.dtype) -> CSEVariable:
        # SIMDScheduling rewrites index_expr ops that compute values (not
        # indices) to value_expr, which keeps the requested dtype.
        return cast(HelionKernel, V.kernel).tile_expr(expr, dtype)

    @staticmethod
    # pyrefly: ignore [bad-override]
    def masked(
        mask: CSEVariable, body: Callable[[], CSEVariable], other: bool | float | int
    ) -> CSEVariable:
        return cast(HelionKernel, V.kernel).masked(mask, body, other)

    @staticmethod
    # pyrefly: ignore [bad-override]
    def dot(a: CSEVariable, b: CSEVariable) -> CSEVariable:
        return cast(HelionKernel, V.kernel).dot(a, b)

    @staticmethod
    # pyrefly: ignore [bad-override]
    def where(cond: str, a: str, b: str) -> str:
        return f"torch.where({cond}, {a}, {b})"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def fma(x: str, y: str, z: str) -> str:
        # x * y + z rounded once: Helion's Triton backend lowers
        # torch.addcmul(z, x, y) to tl.fma(x, y, z), Inductor's ops.fma.
        return f"torch.addcmul({z}, {x}, {y})"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def minimum(a: str, b: str) -> str:
        return f"torch.minimum({a}, {b})"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def maximum(a: str, b: str) -> str:
        return f"torch.maximum({a}, {b})"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def logical_and(a: str, b: str) -> str:
        return f"torch.logical_and({a}, {b})"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def logical_or(a: str, b: str) -> str:
        return f"torch.logical_or({a}, {b})"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def logical_not(x: str) -> str:
        return f"torch.logical_not({x})"

    @staticmethod
    def relu(x: str) -> str:
        return f"torch.relu({x})"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def abs(x: str) -> str:
        return f"torch.abs({x})"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def exp(x: str) -> str:
        return f"torch.exp({x})"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def exp2(x: str) -> str:
        return f"torch.exp2({x})"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def expm1(x: str) -> str:
        return f"torch.expm1({x})"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def log(x: str) -> str:
        return f"torch.log({x})"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def log1p(x: str) -> str:
        return f"torch.log1p({x})"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def log2(x: str) -> str:
        return f"torch.log2({x})"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def sqrt(x: str) -> str:
        return f"torch.sqrt({x})"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def rsqrt(x: str) -> str:
        return f"torch.rsqrt({x})"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def sin(x: str) -> str:
        return f"torch.sin({x})"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def cos(x: str) -> str:
        return f"torch.cos({x})"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def tanh(x: str) -> str:
        return f"torch.tanh({x})"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def erf(x: str) -> str:
        return f"torch.erf({x})"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def sigmoid(x: str) -> str:
        return f"torch.sigmoid({x})"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def floor(x: str) -> str:
        return f"torch.floor({x})"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def ceil(x: str) -> str:
        return f"torch.ceil({x})"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def reciprocal(x: str) -> str:
        return f"torch.reciprocal({x})"


# Ops a Helion kernel may contain; a node using any other op stays out of it.
HELION_OPS = OrderedSet(
    [
        # Handled by the kernel itself.
        "load",
        "store",
        "store_reduction",
        "reduction",
        "output",
        # Python operators inherited from BasicMathOpsMixin and OpOverrides.
        "add",
        "sub",
        "mul",
        "truediv",
        "neg",
        "eq",
        "ne",
        "lt",
        "le",
        "gt",
        "ge",
        "and_",
        "or_",
        "xor",
        "bitwise_and",
        "bitwise_or",
        "bitwise_xor",
        "bitwise_not",
    ]
) | OrderedSet(name for name in vars(HelionOverrides) if not name.startswith("_"))


def is_helion_matmul(node: BaseSchedulerNode) -> bool:
    # is_native_matmul() asserts on template nodes.
    return not node.is_template() and node.is_native_matmul()


@dataclass(frozen=True)
class HelionTensorArg:
    """A kernel parameter: a buffer, or a view of one that the wrapper passes."""

    name: str
    # Buffer or ReinterpretView of a buffer.
    operand: IRNode
    output: bool


class HelionKernel(SIMDKernel[CSEVariable]):
    """An Inductor-generated Helion kernel.

    SIMDScheduling picks the tiling and runs the node bodies as for a Triton
    kernel. The pointwise range trees become the tile loop,
    for tile_z, tile_y, tile_x in hl.tile([Z, Y, X]), and the reduction range
    tree an inner for tile_r in hl.tile(R) loop. A range tree's index (yindex)
    stands for its tile's index tensor (tile_y.index), so loads and stores
    subscript tensors by tile, e.g. in0[tile_y, tile_r], instead of offsetting
    a pointer. Helion picks the block sizes and masks partial tiles itself.

    Values in the tile loop keep a shape that broadcasts against the tile, e.g.
    a row vector is [tile_y, 1], so pointwise ops need no reshapes.
    """

    overrides = HelionOverrides  # type: ignore[assignment]

    def __init__(self, tiling: dict[str, sympy.Expr], **kwargs: Any) -> None:
        super().__init__(tiling, **kwargs)
        if unsupported := OrderedSet(self.numels) - OrderedSet(TILE_NAMES):
            raise NotImplementedError(f"Helion kernels do not tile {unsupported}")
        self.axes = tuple(axis for axis in AXIS_ORDER if axis in self.numels)
        self.outer_axes = tuple(axis for axis in self.axes if axis != "r0_")
        roots = {tree.prefix: tree for tree in self.range_trees}
        self.symbols = {axis: roots[axis].index_sym() for axis in self.axes}
        self.tile_ranges = {self.symbols[axis]: self.numels[axis] for axis in self.axes}
        # Buffers stored to memory; the others are only read back by fused
        # nodes, through the CSE store cache.
        nodes = tuple(self.features.scheduler_nodes())
        fused_names = OrderedSet(node.get_name() for node in nodes)
        self.outputs = OrderedSet(
            buf.get_name()
            for node in nodes
            for buf in node.get_outputs()
            if not isinstance(buf.node.get_layout(), MutationLayoutSHOULDREMOVE)
            and not V.graph.scheduler.can_buffer_be_removed_through_fusion(
                buf.get_name(), fused_names
            )
        )
        self.arguments: dict[tuple[Any, ...], HelionTensorArg] = {}
        self.argument_counts: collections.Counter[str] = collections.Counter()
        # Loads and compute share one buffer, so a masked load can follow the
        # computation of its mask.
        self.loads = self.compute
        # Masks of the enclosing ops.masked() bodies, outermost first.
        self.load_masks: list[CSEVariable] = []
        # Dot node name -> its FP32 accumulator.
        self.accumulators: dict[str, CSEVariable] = {}

    def should_use_persistent_reduction(self) -> bool:
        # The reduction is always the K loop.
        return False

    def dtype_to_str(self, dtype: torch.dtype) -> str:
        return str(dtype)

    def outer_shape(self, axes: Sequence[str] | None = None) -> TileShape:
        """Shape of a value that depends on `axes`, broadcastable against the tile.

        With outer axes (z, y, x): {y, x} -> (tile_y, tile_x), {x} -> (tile_x,),
        {y} -> (tile_y, 1), {z, x} -> (tile_z, 1, tile_x), {} -> ().
        """
        axes = self.outer_axes if axes is None else axes
        present = [axis for axis in self.outer_axes if axis in axes]
        if not present:
            return ()
        start = self.outer_axes.index(present[0])
        return tuple(
            TILE_NAMES[axis] if axis in present else 1
            for axis in self.outer_axes[start:]
        )

    def index_tensor(self, axis: str, axes: Sequence[str] | None = None) -> str:
        """Tile index tensor of `axis`.

        By default it broadcasts against the tile, as index arithmetic needs:
        tile_y.index[:, None] and tile_x.index for outer axes (y, x). With
        `axes`, it has one dim per axis, as a tensor subscript needs:
        tile_y.index[:, None] and tile_x.index[None, :] for axes (y, x).
        """
        name = f"{TILE_NAMES[axis]}.index"
        if axes is None:
            trailing = len(self.outer_axes) - 1 - self.outer_axes.index(axis)
            return f"{name}[:{', None' * trailing}]" if trailing else name
        if len(axes) == 1:
            return name
        return f"{name}[{', '.join(':' if a == axis else 'None' for a in axes)}]"

    def align(
        self, value: CSEVariable, target: TileShape, *, full_rank: bool = False
    ) -> str:
        """View `value` so that it broadcasts against a tensor of shape `target`.

        For target (tile_y, tile_x):
            tmp0 of shape (tile_y,) -> tmp0.reshape([tile_y, 1])
            tmp0 of shape (tile_x,) -> tmp0, which broadcasts as is
        A tile axis that `target` lacks must have size 1 and is dropped, e.g.
        acc stored to out0[tile_x] when M == 1 is acc.reshape([tile_x]).
        With `full_rank` the view has every dim of `target`, as hl.load's
        extra_mask requires: tmp0 of shape (tile_x,) -> tmp0.reshape([1, tile_x]).
        """
        shape = tuple(value.shape or ())
        present = [TILE_AXES[str(dim)] for dim in shape if dim != 1]
        target_axes = [TILE_AXES[str(dim)] for dim in target if dim != 1]
        kept = [axis for axis in present if axis in target_axes]
        if kept != sorted(kept, key=target_axes.index) or any(
            axis not in kept and self.numels[axis] != 1 for axis in present
        ):
            raise NotImplementedError(f"Cannot broadcast Helion {shape} to {target}")
        if not kept:
            return str(value)
        names = [TILE_NAMES[axis] for axis in kept]
        dims = list(target) if full_rank else list(target[target.index(names[0]) :])
        aligned = tuple(dim if dim in names else 1 for dim in dims)
        if aligned == shape:
            return str(value)
        return f"{value}.reshape([{', '.join(map(str, aligned))}])"

    def tile_index(self, index: sympy.Expr) -> sympy.Expr:
        """Rewrite `index` over the range tree indices, i.e. the tile indices.

        A node's index uses the range tree entries that split a tile, e.g.
        y1 = yindex // 8 and y0 = yindex % 8 for a [5, 8, 136] view of a
        [40, 136] tile. Rewriting them in yindex lets the kernel read a tile
        directly instead of computing y1 and y0.
        """
        index = sympy.sympify(index)
        entries = {
            symbol: self.range_tree_nodes[symbol].expr
            for symbol in index.free_symbols
            if symbol in self.range_tree_nodes
        }
        index = sympy_subs(index, entries)
        # A size-1 tile always has index 0, so drop its symbol. align() then
        # drops such tiles from values.
        index = index.xreplace(
            {self.symbols[axis]: 0 for axis in self.axes if self.numels[axis] == 1}
        )
        # Expose quotient/remainder cancellation before inferring tensor axes:
        # x % B + B * (x // B) becomes x, so it no longer needs two indices.
        expanded = index.replace(
            ModularIndexing,
            lambda base, divisor, modulus: FloorDiv(base, divisor)
            - modulus * FloorDiv(base, divisor * modulus),
        )
        simplified = V.graph.sizevars.simplify_with_ranges(
            sympy.expand(expanded), self.tile_ranges
        )
        # Keep genuine nonlinear maps compact for flat_subscript().
        return index if simplified.has(FloorDiv) else simplified

    def tensor_arg(self, operand: IRNode, *, output: bool = False) -> HelionTensorArg:
        layout = operand.get_layout()
        key = (
            operand.get_name(),
            tuple(layout.size),
            tuple(layout.stride),
            layout.offset,
        )
        if key not in self.arguments:
            prefix = "out" if output else "in"
            name = f"{prefix}{self.argument_counts[prefix]}"
            self.argument_counts[prefix] += 1
            self.arguments[key] = HelionTensorArg(name, operand, output)
        return self.arguments[key]

    def tiled_view(
        self, buffer: IRNode, index: sympy.Expr, axes: Sequence[str]
    ) -> IRNode | None:
        """A view of `buffer` with one dim per tile axis in `axes`, read at `index`.

        A [B, M, N] buffer read at 136*yindex + xindex by a [B*M, 136] tile is
        passed as reinterpret_tensor(buf, (B*M, 136), (136, 1), 0) and read as
        in0[tile_y, tile_x], with no // or % in the kernel. None if `index` is
        not affine in those tiles or the view would leave the buffer.
        """
        layout = buffer.get_layout()
        expanded = sympy.expand(index)
        sizes = [self.numels[axis] for axis in axes]
        strides = [expanded.coeff(self.symbols[axis]) for axis in axes]
        offset = expanded.xreplace({self.symbols[axis]: 0 for axis in self.axes})
        values = (*sizes, *strides, offset, layout.storage_size())
        if not all(isinstance(value, (int, sympy.Integer)) for value in values):
            return None
        expected = offset + sum(self.symbols[a] * s for a, s in zip(axes, strides))
        # The view reads exactly what `index` reads at every tile position, so
        # it only needs to stay within the storage (masked loads may not).
        end = offset + sum((size - 1) * stride for size, stride in zip(sizes, strides))
        if (
            offset < 0
            or any(stride < 0 for stride in strides)
            or end >= layout.storage_size()
            or not V.graph.sizevars.statically_known_equals(index, expected)
        ):
            return None
        if (sizes, strides, offset) == (
            list(layout.size),
            list(layout.stride),
            layout.offset,
        ):
            return buffer
        layout = FixedLayout(
            layout.device, layout.dtype, sizes, strides, offset, layout.is_pinned
        )
        return ReinterpretView(data=buffer, layout=layout)

    def subscript(
        self, buffer: IRNode, index: sympy.Expr, *, output: bool = False
    ) -> tuple[HelionTensorArg, list[str], TileShape]:
        """The parameter and subscript that read `buffer` at `index`, and its shape.

        Inductor indices are flat offsets into the storage, so the subscript is
        recovered from them, trying in turn:
        1. the buffer's own dims, e.g. [M, N] with strides (N, 1) at
           N*yindex + xindex is in0[tile_y, tile_x], a bias [N] at xindex is
           in0[tile_x] and a 0-d tensor is in0[()];
        2. a view whose dims are tiles, if `index` is affine in them (see
           tiled_view());
        3. a flat view of the storage, read at `index` itself like a Triton
           pointer offset (see flat_subscript()).
        """
        axes = [
            axis for axis in self.outer_axes if self.symbols[axis] in index.free_symbols
        ]
        layout = buffer.get_layout()
        candidates = [
            [
                axis
                for axis in axes
                if V.graph.sizevars.statically_known_equals(size, self.numels[axis])
            ]
            + ([None] if size == 1 else [])
            for size in layout.size
        ]
        for dims in itertools.product(*candidates):
            # Tile dims in AXIS_ORDER, so that values broadcast without permutes.
            if [dim for dim in dims if dim is not None] != axes:
                continue
            expected = layout.offset + sum(
                self.symbols[dim] * stride
                for dim, stride in zip(dims, layout.stride)
                if dim is not None
            )
            if V.graph.sizevars.statically_known_equals(index, expected):
                arg = self.tensor_arg(buffer, output=output)
                coordinates = [TILE_NAMES[dim] if dim else "0" for dim in dims]
                shape = tuple(TILE_NAMES[dim] for dim in dims if dim is not None)
                return arg, coordinates or ["()"], shape
        if (view := self.tiled_view(buffer, index, axes)) is not None:
            tiles = tuple(TILE_NAMES[axis] for axis in axes)
            return self.tensor_arg(view, output=output), list(tiles) or ["()"], tiles
        return self.flat_subscript(buffer, index, output=output)

    def flat_subscript(
        self, buffer: IRNode, index: sympy.Expr, *, output: bool = False
    ) -> tuple[HelionTensorArg, list[str], TileShape]:
        """Read a flat view of the storage at `index`, like a Triton pointer offset.

        A [5, 8, 136] input with strides (1088, 1, 8), read by a [40, 136] tile, is
        passed as reinterpret_tensor(arg2_1, (5440, ), (1, ), 0) and read as
        in2[8*tile_x.index + 1088*(tile_y.index[:, None] // 8) + ((tile_y.index[:, None] % 8))].
        This handles any index, but as a gather: Helion cannot use tensor
        descriptors for it.
        """
        layout = buffer.get_layout()
        size = layout.storage_size()
        operand = buffer
        if (list(layout.size), list(layout.stride), layout.offset) != ([size], [1], 0):
            flat = FixedLayout(layout.device, layout.dtype, [size], [1], 0)
            operand = ReinterpretView(data=buffer, layout=flat)
        arg = self.tensor_arg(operand, output=output)
        axes = [
            axis for axis in self.outer_axes if self.symbols[axis] in index.free_symbols
        ]
        return arg, [self.index_tensor_expr(index, axes)], self.outer_shape(axes)

    def codegen_body(self) -> None:
        """Move the pending code into self.body; reduction code goes in the K loop.

        SIMDKernel.disable_reduction() calls this when the reduction ends, so
        the dots end up in for tile_r in hl.tile(K): and the epilogue after it.
        """
        if not (self.compute or self.stores):
            return
        if self.inside_reduction and self.features.is_reduction():
            self.body.writeline(f"for tile_r in hl.tile({self.numels['r0_']}):")
            with self.body.indent():
                self.body.splice(self.compute)
            # Reduction results are stored after the loop.
            self.body.splice(self.stores)
            # Values from the loop body are out of scope after it.
            self.cse.invalidate(OrderedSet(self.accumulators.values()))
        else:
            self.body.splice(self.compute)
            self.body.splice(self.stores)
        self.compute.clear()
        self.stores.clear()

    def codegen_kernel(self) -> str:
        """Return the kernel module, which defines `kernel`.

        The module holds no buffer names and no kernel name, so identical
        schedules, e.g. repeated layers or the same matmul in another graph,
        share one module and one Helion compile.
        """
        self.codegen_body()
        code = IndentedBuffer()
        code.writelines(
            [
                "import torch",
                "import helion",
                "import helion.language as hl",
                "from torch._inductor.runtime.helion import HelionKernelWrapper",
                "",
            ]
        )
        # FP32 dots follow torch.backends.cuda.matmul.fp32_precision, like
        # Inductor's Triton dots. ref_mode=OFF keeps HELION_INTERPRET from
        # swapping in eager reference mode.
        precision = (
            "tf32" if torch.backends.cuda.matmul.fp32_precision == "tf32" else "ieee"
        )
        code.writeline(
            f'@helion.kernel(backend="{config.helion.backend}", static_shapes=True, '
            f'autotune_effort="{config.helion.autotune_effort}", '
            f'dot_precision="{precision}", ref_mode=helion.RefMode.OFF)'
        )
        signature = ", ".join(f"{arg.name}: torch.Tensor" for arg in self.kernel_args())
        code.writeline(f"def kernel({signature}) -> None:")
        with code.indent():
            tiles = ", ".join(TILE_NAMES[axis] for axis in self.outer_axes)
            sizes = ", ".join(str(self.numels[axis]) for axis in self.outer_axes)
            # hl.tile([40]) would yield 1-tuples.
            sizes = sizes if len(self.outer_axes) == 1 else f"[{sizes}]"
            code.writeline(f"for {tiles} in hl.tile({sizes}):")
            with code.indent():
                code.splice(self.body)
        code.writeline("kernel = HelionKernelWrapper(kernel)")
        return code.getvalue()

    def kernel_args(self) -> list[HelionTensorArg]:
        return sorted(self.arguments.values(), key=lambda arg: arg.output)

    def call_kernel(
        self, name: str, node: IRNode | None = None, deallocate_ws: bool = True
    ) -> None:
        # helion_fused_add_mm_relu_0(arg0_1, reinterpret_tensor(arg1_1, ...), buf1)
        # An output that reuses an input's storage replaces that input in the
        # wrapper (buf1 = buf0; del buf0  # reuse), so the input is passed as a
        # view of the output: reinterpret_tensor(buf1, ...).
        reused = {inp: out for out, inp in self.inplace_update_buffers.items()}
        args = []
        for arg in self.kernel_args():
            operand = arg.operand
            if operand.get_name() in reused:
                target = V.graph.get_buffer(reused[operand.get_name()])
                operand = ReinterpretView(data=target, layout=operand.get_layout())
            args.append(
                operand.codegen_reference()
                if isinstance(operand, ReinterpretView)
                else operand.get_name()
            )
        wrapper = V.graph.wrapper_code
        wrapper.writeline(wrapper.wrap_kernel_call(name, args))

    def load(self, name: str, index: sympy.Expr) -> CSEVariable:
        buffer = V.graph.get_buffer(name)
        index = self.tile_index(index)
        # In the K loop of a native matmul, the dot node loads its operands.
        node = self.current_node
        if self.inside_reduction and node is not None and node.is_native_matmul():
            return self.load_dot_operand(buffer, index)
        arg, coordinates, shape = self.subscript(buffer, index)
        dtype = upcast_compute_type(buffer.get_dtype())
        cast_suffix = "" if dtype == buffer.get_dtype() else f".to({dtype})"
        subscript = ", ".join(coordinates)
        mask = self.load_mask(shape)
        if mask is None:
            line = f"{arg.name}[{subscript}]{cast_suffix}"
        else:
            # hl.load zero-fills masked lanes; masked() then selects `other`.
            mask = self.align(mask, shape, full_rank=True)
            line = f"hl.load({arg.name}, [{subscript}], extra_mask={mask}){cast_suffix}"
        value = self.cse.generate(self.loads, line, dtype=dtype, shape=shape)
        # A row vector continues as tmp1 = tmp0.reshape([tile_y, 1]).
        aligned = self.outer_shape([TILE_AXES[str(dim)] for dim in shape if dim != 1])
        if aligned == shape:
            return value
        return self.cse.generate(
            self.loads,
            self.align(value, self.outer_shape()),
            dtype=dtype,
            shape=aligned,
        )

    def load_dot_operand(self, buffer: IRNode, index: sympy.Expr) -> CSEVariable:
        """Load a dot operand in the K loop: in0[tile_y, tile_r] or in1[tile_r, tile_x].

        The tile keeps its storage dtype: Helion lowers torch.addmm to its
        tensor core GEMM only on such tiles.
        """
        batch = ("z",) if "z" in self.numels else ()
        for axes in [(*batch, "y", "r0_"), (*batch, "r0_", "x")]:
            if (view := self.tiled_view(buffer, index, axes)) is None:
                continue
            tiles = tuple(TILE_NAMES[axis] for axis in axes)
            line = f"{self.tensor_arg(view).name}[{', '.join(tiles)}]"
            return self.cse.generate(
                self.loads, line, dtype=buffer.get_dtype(), shape=tiles
            )
        raise NotImplementedError(f"Cannot read a Helion dot operand at {index}")

    def store(
        self,
        name: str,
        index: sympy.Expr,
        value: CSEVariable,
        mode: str | None = None,
    ) -> None:
        if mode is not None:
            raise NotImplementedError(f"Helion kernels do not support {mode} stores")
        buffer = V.graph.get_buffer(name)
        layout = buffer.get_layout()
        if isinstance(layout, MutationLayoutSHOULDREMOVE):
            # x.add_(a @ b) writes into x: in0[tile_y, tile_x] = tmp3.to(...).
            buffer = layout.get_buffer()
        elif name not in self.outputs:
            return
        arg, coordinates, shape = self.subscript(
            buffer, self.tile_index(index), output=True
        )
        dtype = buffer.get_dtype()
        cast_suffix = "" if value.dtype == dtype else f".to({dtype})"
        # out0[tile_y, tile_x] = tmp4.to(torch.bfloat16)
        self.stores.writeline(
            f"{arg.name}[{', '.join(coordinates)}] = "
            f"{self.align(value, shape)}{cast_suffix}"
        )

    def index_expr(self, expr: sympy.Expr, dtype: torch.dtype) -> CSEVariable:
        # As in Triton, integer indices use the kernel's index dtype.
        if dtype in (torch.int32, torch.int64):
            dtype = self.get_index_dtype_as_torch_dtype()
        return self.tile_expr(expr, dtype)

    def tile_expr(self, expr: sympy.Expr, dtype: torch.dtype) -> CSEVariable:
        """Index arithmetic on tile index tensors.

        torch.arange(40 * 136).reshape(40, 136) in an epilogue reads
        (136*tile_y.index[:, None] + tile_x.index).to(torch.int32).
        """
        index = self.tile_index(expr)
        dtype = upcast_compute_type(dtype)
        axes = [
            axis for axis in self.outer_axes if self.symbols[axis] in index.free_symbols
        ]
        if not index.free_symbols:
            value = int(index) if index.is_Integer else float(index)
            line = HelionOverrides.constant(value, dtype)
            return self.cse.generate(self.compute, line, dtype=dtype, shape=())
        return self.cse.generate(
            self.compute,
            f"({self.index_tensor_expr(index, axes)}).to({dtype})",
            dtype=dtype,
            shape=self.outer_shape(axes),
        )

    def index_tensor_expr(self, index: sympy.Expr, axes: Sequence[str]) -> str:
        """`index` computed on the tile index tensors of `axes`.

        136*yindex + xindex -> 136*tile_y.index[:, None] + tile_x.index.
        """
        if index.free_symbols - OrderedSet(self.symbols[axis] for axis in axes):
            raise NotImplementedError(f"Unsupported Helion index: {index}")
        # tile.index is int32; widen it first if int32 arithmetic could overflow.
        bounds = bound_sympy(
            index,
            {
                self.symbols[axis]: ValueRanges(0, self.numels[axis] - 1)
                for axis in axes
            },
        )
        widen = not (-(2**31) <= bounds.lower and bounds.upper < 2**31)
        suffix = ".to(torch.int64)" if widen else ""
        replacements = {
            self.symbols[axis]: sympy.Symbol(self.index_tensor(axis) + suffix)
            for axis in axes
        }
        return HelionPrinter().doprint(index.xreplace(replacements), simplify=False)

    def load_mask(self, shape: TileShape) -> CSEVariable | None:
        """The extra_mask for a load of `shape` inside ops.masked() bodies.

        A mask over axes the load does not use cannot move its address, e.g.
        cat((bias.expand(10, N), c)) masks rows while bias is read as
        in2[tile_x]: every column is read for some kept row, so that load needs
        no mask and masked() alone selects the rows.
        """
        load_axes = OrderedSet(TILE_AXES[str(dim)] for dim in shape if dim != 1)
        guards = []
        for mask in self.load_masks:
            mask_axes = OrderedSet(
                TILE_AXES[str(dim)] for dim in mask.shape or () if dim != 1
            )
            if mask_axes <= load_axes:
                guards.append(mask)
            elif mask_axes & load_axes:
                raise NotImplementedError("Helion cannot apply a mask over other axes")
        if not guards:
            return None
        result = guards[0]
        for guard in guards[1:]:
            result = V.ops.logical_and(result, guard)
        return result

    def masked(
        self,
        mask: CSEVariable,
        body: Callable[[], CSEVariable],
        other: bool | float | int,
    ) -> CSEVariable:
        # cat((first, second)) reads `second` where its index is in range:
        #   tmp13 = hl.load(in3, [(-64) + tile_x.index], extra_mask=tmp10).to(torch.float32)
        #   tmp14 = torch.where(tmp10, tmp13, tmp8)
        # where tmp8 = hl.full([], 0.0, dtype=torch.float32).
        self.load_masks.append(mask)
        try:
            result = body()
        finally:
            self.load_masks.pop()
        if result.dtype is None:
            raise AssertionError(f"expected a dtype for masked value {result}")
        return V.ops.where(mask, result, V.ops.constant(other, result.dtype))

    def dot(self, lhs: CSEVariable, rhs: CSEVariable) -> CSEVariable:
        """Accumulate one K tile of a dot into its accumulator.

        Each dot gets its own FP32 accumulator, declared before the K loop, so
        horizontally fused dots stay separate sums:
            acc = hl.zeros([tile_y, tile_x], dtype=torch.float32)
            acc1 = hl.zeros([tile_y, tile_x], dtype=torch.float32)
            for tile_r in hl.tile(72):
                tmp0 = in0[tile_y, tile_r]
                tmp1 = in1[tile_r, tile_x]
                acc = torch.addmm(acc, tmp0, tmp1)
                tmp2 = in2[tile_r, tile_x]
                acc1 = torch.addmm(acc1, tmp0, tmp2)
        """
        if self.current_node is None:
            raise AssertionError("expected a dot node")
        node = self.current_node.get_name()
        if node not in self.accumulators:
            name = f"acc{len(self.accumulators) or ''}"
            acc = CSEVariable(
                name, ValueRanges.unknown(), torch.float32, self.outer_shape()
            )
            tiles = ", ".join(TILE_NAMES[axis] for axis in self.outer_axes)
            self.body.writeline(f"{acc} = hl.zeros([{tiles}], dtype=torch.float32)")
            self.accumulators[node] = acc
        acc = self.accumulators[node]
        op = "baddbmm" if "z" in self.numels else "addmm"
        self.compute.writeline(f"{acc} = torch.{op}({acc}, {lhs}, {rhs})")
        return acc

    def reduction(
        self,
        dtype: torch.dtype,
        src_dtype: torch.dtype,
        reduction_type: str,
        value: CSEVariable | tuple[CSEVariable, ...],
    ) -> CSEVariable | tuple[CSEVariable, ...]:
        # Only native matmul dots for now; general reductions over r0_ would
        # accumulate elementwise in the K loop and reduce after it.
        if reduction_type != "dot":
            raise NotImplementedError(f"Helion kernels do not support {reduction_type}")
        # dot() already accumulated into acc.
        return value


class HelionScheduling(TritonScheduling):
    """Emit native matmul schedules as Helion kernels and the rest as Triton."""

    @staticmethod
    def format_kernel_definition(name: str, source: str) -> str:
        # helion_fused_mm_0_source = """\
        # import torch
        # ..."""
        # helion_fused_mm_0 = PyCodeCache.load(helion_fused_mm_0_source).kernel
        escaped = source.replace("\\", "\\\\").replace('"""', '\\"""')
        return (
            f'{name}_source = """\\\n{escaped}"""\n'
            f"{name} = PyCodeCache.load({name}_source).kernel"
        )

    @staticmethod
    def _native_matmul_row_split(nodes: Sequence[BaseSchedulerNode]) -> None:
        # No [M, N] -> [M] row epilogues yet: the row reductions run as Triton
        # kernels after the Helion matmul.
        return None

    def can_fuse(self, node1, node2):
        if not (is_helion_matmul(node1) or is_helion_matmul(node2)):
            return super().can_fuse(node1, node2)
        why = WhyNoFuse(node1, node2)
        nodes = [*node1.get_nodes(), *node2.get_nodes()]
        for node in nodes:
            if not isinstance(node, SchedulerNode) or node.is_template():
                why("Helion matmul kernel fuses only scheduler nodes")
                return False
            if unsupported := OrderedSet(node._body.op_counts) - HELION_OPS:
                why("Helion kernels do not support %s", unsupported)
                return False
        dots = [node for node in nodes if node.is_native_matmul()]
        if len(OrderedSet(tuple(map(tuple, node.get_ranges())) for node in dots)) > 1:
            why("Helion matmul kernel tiles one dot shape")
            return False
        # (a + 1) @ b: the operand producer stays a Triton kernel, and the
        # Helion kernel reads in0[tile_y, tile_r] from its output.
        operands = OrderedSet(
            dep.name for node in dots for dep in node.read_writes.reads
        )
        if operands & (node1.get_buffer_names() | node2.get_buffer_names()):
            why("Helion matmul kernel does not fuse operand producers")
            return False
        # An epilogue node runs once per output element, (B*)M*N, split into
        # the (z,) y, x tiles.
        tiles = list(dots[0].get_ranges()[0])
        numel, groups = sympy_product(tiles), [*tiles, sympy.S.One]
        for node in nodes:
            if node in dots:
                continue
            pointwise = node.group[1] == (numel, 1)
            if not pointwise or not SIMDKernel.is_compatible(groups, node.get_ranges()):
                why("Helion epilogue must iterate over the matmul output tiles")
                return False
            if any(V.graph.is_unspec_arg(dep.name) for dep in node.read_writes.reads):
                why("Helion kernels do not read CPU scalars")
                return False
        return super().can_fuse(node1, node2)

    can_fuse_vertical = can_fuse
    can_fuse_horizontal = can_fuse

    def create_kernel_choices(self, kernel_features, kernel_args, kernel_kwargs):
        if any(is_helion_matmul(node) for node in kernel_features.scheduler_nodes()):
            return [HelionKernel(*kernel_args, **kernel_kwargs)]
        return super().create_kernel_choices(
            kernel_features, kernel_args, kernel_kwargs
        )

    def generate_kernel_code_from_nodes(self, nodes, *args, **kwargs):
        if not any(is_helion_matmul(node) for node in nodes):
            return super().generate_kernel_code_from_nodes(nodes, *args, **kwargs)
        # Fusion benchmarking skips Helion matmuls (see Scheduler), so this
        # only serves debug output such as TORCH_LOGS=ir_post_fusion.
        features = self._kernel_features_from_nodes(nodes)
        tiling, _ = self.get_tiling_and_scores(
            features.node_schedule,
            features.numel,
            features.reduction_numel,
            features.coalesce_analysis,
        )
        kernel = HelionKernel(tiling, features=features)
        self.codegen_node_schedule_with_kernel(features.node_schedule, kernel)
        with V.set_kernel_handler(kernel):
            return kernel.codegen_kernel()

    def define_kernel(self, src_code, node_schedule, kernel):
        if not isinstance(kernel, HelionKernel):
            return super().define_kernel(src_code, node_schedule, kernel)
        wrapper = V.graph.wrapper_code
        if src_code in wrapper.src_to_kernel:
            return wrapper.src_to_kernel[src_code]
        fused_name = (
            get_fused_kernel_name(node_schedule, config.triton.descriptive_names)
            if config.triton.descriptive_names
            else ""
        )
        name = "_".join(
            part
            for part in ("helion", fused_name, wrapper.next_kernel_suffix())
            if part
        )
        wrapper.src_to_kernel[src_code] = name
        wrapper.add_import_once("from torch._inductor.codecache import PyCodeCache")
        # gpu=False keeps the module out of the Triton compile-time autotune block.
        wrapper.define_kernel(
            name,
            self.format_kernel_definition(name, src_code),
            gpu=False,
            standalone=True,
        )
        return name
