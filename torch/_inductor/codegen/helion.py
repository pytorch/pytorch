"""Inductor codegen for Helion kernels.

With ``config.helion.enabled``, ``HelionScheduling`` emits each schedule that
``HelionKernel`` supports, e.g. a pointwise kernel or an ``ir.Reduction``, as
a Helion kernel, and the others as Triton. For ``(x * 2).sum(-1) + b`` the
body of the generated ``@helion.kernel`` is::

    for tile_x in hl.tile(64):
        acc = hl.zeros([tile_x], dtype=torch.float32)
        for tile_r in hl.tile(100):
            tmp0 = in0[tile_x, tile_r]
            tmp1 = hl.full([], 2.0, dtype=torch.float32)
            tmp2 = tmp0 * tmp1
            acc = acc + torch.sum(tmp2, -1)
        tmp3 = in1[tile_x]
        tmp4 = acc + tmp3
        out0[tile_x] = tmp4

``config.helion.native_matmul`` also lowers CUDA mm/bmm/addmm/baddbmm to
native matmul IR (``ops.dot`` plus ``reduction("dot")``), so that a matmul and
its fused pointwise epilogue is one Helion kernel. For bf16
``torch.relu(a @ b + bias)``::

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
import contextlib
import itertools
import logging
import math
from dataclasses import dataclass
from typing import Any, cast, TYPE_CHECKING

import sympy

import torch
from torch.utils._ordered_set import OrderedSet
from torch.utils._sympy.value_ranges import bound_sympy, ValueRanges

from .. import config
from ..ir import (
    ComputedBuffer,
    FixedLayout,
    IRNode,
    MutationLayoutSHOULDREMOVE,
    ReinterpretView,
)
from ..loop_body import MemoryUsageType
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
from .simd_kernel_features import DisableReduction, EnableReduction
from .triton import TritonScheduling


if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence

    from ..scheduler import BaseSchedulerNode

# Shape of a kernel value: a tile name per dim, or 1 for a broadcast dim.
TileShape = tuple[str | int, ...]
TILE_NAMES = {"z": "tile_z", "y": "tile_y", "x": "tile_x", "r0_": "tile_r"}
TILE_AXES = {name: axis for axis, name in TILE_NAMES.items()}

log = logging.getLogger(__name__)


def literal(value: bool | float | int) -> str:
    # repr(-inf) is not Python: hl.full([], -inf, dtype=torch.float32) fails
    # with UndefinedVariable: inf is not defined.
    if isinstance(value, float) and not math.isfinite(value):
        return f"float({repr(value)!r})"
    return repr(value)


class HelionPrinter(PythonPrinter):
    """Print index expressions over tile index tensors.

    PythonPrinter prints ``Min(y, 5)`` as ``min(tile_y.index[:, None], 5)``,
    which fails on a tensor: ``RuntimeError: Boolean value of Tensor with more
    than one value is ambiguous``. This prints
    ``torch.minimum(tile_y.index[:, None], hl.full([], 5, dtype=torch.int64))``;
    the constant is a 0-d tensor because torch.minimum takes no Python scalar.
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
        # A 0-d tensor, as torch.minimum/maximum take no Python scalar. With the
        # bare literal, torch.clamp(a @ b, min=0.0) would emit
        #   tmp2 = 0.0
        #   tmp3 = torch.maximum(acc, tmp2)
        #   TypeError: maximum(): argument 'other' (position 2) must be Tensor, not float
        return f"hl.full([], {literal(value)}, dtype={upcast_compute_type(dtype)})"

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
        # No cast for a value that already has the dtype. The native matmul
        # body casts each dot operand to its own dtype, which would emit
        #   tmp0 = in0[tile_y, tile_r]
        #   tmp1 = tmp0.to(torch.bfloat16)
        #   ...
        #   acc = torch.addmm(acc, tmp1, tmp3)
        if x.dtype == dtype:
            return x
        return f"{x}.to({dtype})"

    @staticmethod
    def index_expr(expr: sympy.Expr, dtype: torch.dtype) -> CSEVariable:
        return cast(HelionKernel, V.kernel).index_expr(expr, dtype)

    @staticmethod
    def value_expr(expr: sympy.Expr, dtype: torch.dtype) -> CSEVariable:
        # SIMDScheduling turns an index_expr that computes a value into
        # value_expr, e.g. torch.arange(40 * 136).reshape(40, 136) in an
        # epilogue, which the OpsHandler default rejects with NotImplementedError:
        #   tmp2 = (tile_x.index + 136*tile_y.index[:, None]).to(torch.float32)
        return cast(HelionKernel, V.kernel).index_expr(expr, dtype)

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
        # Without it, torch.addcmul(c, a @ b, d) is not fused:
        #   helion_fused_mm_0(arg0_1, arg1_1, buf0)
        #   triton_poi_fused_addcmul_1.run(buf1, arg2_1, arg3_1, 5440, ...)
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
# E.g. a @ b + c[i] reads c through indirect_indexing, which HelionKernel does
# not implement (NotImplementedError from check_bounds), so c[i] stays Triton.
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


def lowest(dtype: torch.dtype) -> bool | float | int:
    if dtype == torch.bool:
        return False
    return -math.inf if dtype.is_floating_point else torch.iinfo(dtype).min


def highest(dtype: torch.dtype) -> bool | float | int:
    if dtype == torch.bool:
        return True
    return math.inf if dtype.is_floating_point else torch.iinfo(dtype).max


# Reduction types of ir.Reduction that a Helion kernel supports, and the initial
# value of their accumulator for an input dtype. The others, e.g. argmax or
# Welford, stay Triton for now.
REDUCTIONS: dict[str, Callable[[torch.dtype], bool | float | int]] = {
    "sum": lambda dtype: 0,
    "prod": lambda dtype: 1,
    "max": lowest,
    "min": highest,
}


def is_helion_matmul(node: BaseSchedulerNode) -> bool:
    # is_native_matmul() asserts on templates, e.g. flex_attention in the graph:
    # AssertionError: type(self.node)=<class 'torch._inductor.ir.TritonTemplateBuffer'>
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
    tree an inner for tile_r in hl.tile(R) loop. A range tree's index (xindex)
    stands for its tile's index tensor (tile_x.index), so loads and stores
    subscript tensors by tile, e.g. in0[tile_x, tile_r], instead of offsetting
    a pointer. Helion picks the block sizes and masks partial tiles itself.

    Subscripts list their tiles in the order of self.axes: (z, y, x, r0_), as
    Triton orders XBLOCK and R0_BLOCK, or (z, y, r0_, x) in a native matmul
    kernel, whose dot operands then read as in0[tile_y, tile_r] and
    in1[tile_r, tile_x]. Values keep a shape that broadcasts against the tiles
    in scope, e.g. a row vector is (tile_y, 1) in the tile loop over
    (tile_y, tile_x), and a value of each row is (tile_x, 1) in a reduction
    loop over (tile_x, tile_r), so pointwise ops need no reshapes.
    """

    overrides = HelionOverrides  # type: ignore[assignment]

    def __init__(self, tiling: dict[str, sympy.Expr], **kwargs: Any) -> None:
        # Values defined outside the reduction loops, which stay in scope.
        self.outer_vars: OrderedSet[CSEVariable] = OrderedSet()
        super().__init__(tiling, **kwargs)
        if unsupported := OrderedSet(self.numels) - OrderedSet(TILE_NAMES):
            raise NotImplementedError(f"Helion kernels do not tile {unsupported}")
        if not all(isinstance(n, (int, sympy.Integer)) for n in self.numels.values()):
            raise NotImplementedError("Helion kernels have static shapes")
        nodes = tuple(self.features.scheduler_nodes())
        self.is_matmul = any(node.is_native_matmul() for node in nodes)
        order = ("z", "y", "r0_", "x") if self.is_matmul else ("z", "y", "x", "r0_")
        self.axes = tuple(axis for axis in order if axis in self.numels)
        self.outer_axes = tuple(axis for axis in self.axes if axis != "r0_")
        roots = {tree.prefix: tree for tree in self.range_trees}
        self.symbols = {axis: roots[axis].index_sym() for axis in self.axes}
        self.tile_ranges = {self.symbols[axis]: self.numels[axis] for axis in self.axes}
        # Buffers stored to memory. A fused intermediate is read back through
        # the CSE store cache instead; storing it too would add
        # out0[tile_y, tile_x] = acc.to(torch.bfloat16) to relu(a @ b + c).
        fused_names = OrderedSet(node.get_name() for node in nodes)
        read_later = self.values_read_in_a_later_loop(nodes)
        self.outputs = OrderedSet(
            buf.get_name()
            for node in nodes
            for buf in node.get_outputs()
            if not isinstance(buf.node.get_layout(), MutationLayoutSHOULDREMOVE)
            and (
                buf.get_name() in read_later
                or not V.graph.scheduler.can_buffer_be_removed_through_fusion(
                    buf.get_name(), fused_names
                )
            )
        )
        self.arguments: dict[tuple[Any, ...], HelionTensorArg] = {}
        self.argument_counts: collections.Counter[str] = collections.Counter()
        # Loads and compute share one buffer, so that a masked load follows the
        # computation of its mask. With loads first, as in Triton, cat((f, s))
        # would emit
        #   tmp7 = hl.load(in2, [tile_x.index], extra_mask=tmp6).to(torch.float32)
        # above tmp6 = tmp2 < tmp5: UndefinedVariable: tmp6 is not defined.
        self.loads = self.compute
        # Masks of the enclosing ops.masked() bodies, outermost first.
        self.load_masks: list[CSEVariable] = []
        self.accumulator_count = 0
        # Buffer -> value stored outside the current reduction loop; see
        # disable_reduction().
        self.outer_stores: dict[str, CSEVariable] = {}

    def values_read_in_a_later_loop(
        self, nodes: Sequence[BaseSchedulerNode]
    ) -> OrderedSet[str]:
        """Buffers that a reduction loop computes and a later loop reads.

        A value of a reduction loop is out of scope in the next loop, so it
        goes through memory even when no other kernel reads it. For y * s with
        s = y.sum(-1, keepdim=True) and y a chain of 20 sin(y) * 0.9 + 0.1 * i,
        Inductor keeps y as a buffer, and the kernel emits
            for tile_r in hl.tile(100):
                ...
                out0[tile_x, tile_r] = tmp78
                acc = acc + torch.sum(tmp78, -1)
            for tile_r in hl.tile(100):
                tmp79 = out0[tile_x, tile_r]
        Without the store, the second loop reads a buffer nothing wrote and the
        result is wrong. Reduction results are not such values; they live on in
        their accumulators.
        """
        region, current = {}, 0
        for item in self.features.node_schedule:
            if item is DisableReduction or item is EnableReduction:
                current += 1
            elif isinstance(item, SchedulerNode):
                region[item.get_name()] = current
        writers = {buf.get_name(): node for node in nodes for buf in node.get_outputs()}
        # The schedule starts in a reduction loop: regions 0, 2, ... are loops.
        return OrderedSet(
            dep.name
            for node in nodes
            for dep in node.read_writes.reads
            if (writer := writers.get(dep.name)) is not None
            and not writer.is_reduction()
            and region[writer.get_name()] % 2 == 0
            and region[node.get_name()] != region[writer.get_name()]
            and self.features.is_reduction()
        )

    @property
    def in_loop(self) -> bool:
        """Whether code goes in a reduction loop, for tile_r in hl.tile(R)."""
        return self.inside_reduction and self.features.is_reduction()

    @property
    def scope_axes(self) -> tuple[str, ...]:
        """The tile axes that values in the code being generated broadcast against.

        A native matmul's K loop only holds dot operands, which do not broadcast.
        """
        return self.axes if self.in_loop and not self.is_matmul else self.outer_axes

    def create_cse_var(self, *args: Any, **kwargs: Any) -> CSEVariable:
        var = super().create_cse_var(*args, **kwargs)
        if not (hasattr(self, "features") and self.in_loop):
            self.outer_vars.add(var)
        return var

    def dtype_to_str(self, dtype: torch.dtype) -> str:
        # Dtype propagation reads SIMDKernel.index_dtype for an index_expr, e.g.
        # the mask of a cat input, which raises NotImplementedError without this.
        return str(dtype)

    def index_axes(self, index: sympy.Expr) -> list[str]:
        return [axis for axis in self.axes if self.symbols[axis] in index.free_symbols]

    def value_shape(self, axes: Sequence[str] | None = None) -> TileShape:
        """Shape of a value that depends on `axes`, broadcastable against the tiles in scope.

        In the tile loop over (z, y, x): {y, x} -> (tile_y, tile_x), {x} ->
        (tile_x,), {y} -> (tile_y, 1), {z, x} -> (tile_z, 1, tile_x), {} -> ().
        In a reduction loop over (x, r0_): {x} -> (tile_x, 1), {r0_} -> (tile_r,).
        """
        scope = self.scope_axes
        axes = scope if axes is None else axes
        present = [axis for axis in scope if axis in axes]
        if not present:
            return ()
        start = scope.index(present[0])
        return tuple(
            TILE_NAMES[axis] if axis in present else 1 for axis in scope[start:]
        )

    def index_tensor(self, axis: str, axes: Sequence[str] | None = None) -> str:
        """Tile index tensor of `axis`.

        By default it broadcasts against the tiles in scope, as index arithmetic
        needs: tile_y.index[:, None] and tile_x.index for (y, x). With `axes`,
        it has one dim per axis, as a tensor subscript needs: tile_y.index[:, None]
        and tile_x.index[None, :] for axes (y, x).
        """
        name = f"{TILE_NAMES[axis]}.index"
        if axes is None:
            scope = self.scope_axes
            trailing = len(scope) - 1 - scope.index(axis)
            return f"{name}[:{', None' * trailing}]" if trailing else name
        if len(axes) == 1:
            return name
        return f"{name}[{', '.join(':' if a == axis else 'None' for a in axes)}]"

    def align(self, value: CSEVariable, target: TileShape) -> str:
        """View `value` so that it broadcasts against a tensor of shape `target`.

        For target (tile_y, tile_x):
            tmp0 of shape (tile_y,) -> tmp0.reshape([tile_y, 1])
            tmp0 of shape (tile_x,) -> tmp0, which broadcasts as is
        """
        shape = tuple(value.shape or ())
        present = [TILE_AXES[str(dim)] for dim in shape if dim != 1]
        target_axes = [TILE_AXES[str(dim)] for dim in target if dim != 1]
        if any(axis not in target_axes for axis in present) or present != sorted(
            present, key=target_axes.index
        ):
            raise NotImplementedError(f"Cannot broadcast Helion {shape} to {target}")
        if not present:
            return str(value)
        names = [TILE_NAMES[axis] for axis in present]
        dims = target[target.index(names[0]) :]
        aligned = tuple(dim if dim in names else 1 for dim in dims)
        if aligned == shape:
            return str(value)
        return f"{value}.reshape([{', '.join(map(str, aligned))}])"

    def in_scope(self, value: CSEVariable) -> CSEVariable:
        """`value`, computed outside the current reduction loop, shaped for it.

        acc of shape (tile_x,) reads as tmp6 = acc.reshape([tile_x, 1]) in a
        later loop over (tile_x, tile_r).
        """
        axes = [TILE_AXES[str(dim)] for dim in value.shape or () if dim != 1]
        target = self.value_shape(axes)
        if target == tuple(value.shape or ()):
            return value
        line = self.align(value, self.value_shape())
        return self.cse.generate(self.compute, line, dtype=value.dtype, shape=target)

    def tile_index(self, index: sympy.Expr) -> sympy.Expr:
        """Rewrite `index` over the range tree indices, i.e. the tile indices.

        A node's index uses the range tree entries of its own ranges, e.g.
        y1 = yindex // 8 and y0 = yindex % 8 for a [5, 8, 136] view of a
        [40, 136] tile, and then y0 = yindex for the dot. Without the rewrite,
        (a @ b).reshape(5, 8, 136) + c fails on the dot operand:
            NotImplementedError: Cannot read a Helion dot operand at r0_2 + 72*y0
        """
        index = sympy.sympify(index)
        entries = {
            symbol: self.range_tree_nodes[symbol].expr
            for symbol in index.free_symbols
            if symbol in self.range_tree_nodes
        }
        index = sympy.expand(sympy_subs(index, entries))
        return V.graph.sizevars.simplify_with_ranges(index, self.tile_ranges)

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
        # Only views inside the storage. F.pad(x, (8, 0)) reads x at
        # 128*yindex + xindex - 8 where xindex >= 8, which as a view would be
        # reinterpret_tensor(arg2_1, (40, 136), (128, 1), -8), starting before
        # x's storage; the masked load reads the flat storage instead:
        #   hl.load(in2, [(-8) + tile_x.index + 128*tile_y.index[:, None]], extra_mask=...)
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
        self,
        buffer: IRNode,
        index: sympy.Expr,
        axes: Sequence[str],
        *,
        output: bool = False,
        flat: bool = True,
    ) -> tuple[HelionTensorArg, list[str], TileShape] | None:
        """The parameter and subscript that read `buffer` at `index`, and its shape.

        The subscript has a dim per tile in `axes`, in the order of self.axes.
        Inductor indices are flat offsets into the storage, so the subscript is
        recovered from them, trying in turn:
        1. the buffer's own dims, e.g. [M, N] with strides (N, 1) at
           N*yindex + xindex is in0[tile_y, tile_x], a bias [N] at xindex is
           in0[tile_x] and a 0-d tensor is in0[()];
        2. a view whose dims are tiles, if `index` is affine in them (see
           tiled_view()), e.g. x.sum(0) for x of shape [100, 64] reads
           reinterpret_tensor(arg0_1, (64, 100), (1, 64), 0) as in0[tile_x, tile_r];
        3. if `flat`, a flat view of the storage, read at `index` itself like a
           Triton pointer offset (see flat_subscript()). Otherwise None.
        """
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
            # Tile dims in the order of self.axes only. c.t() for c of shape
            # [136, 40] would match as in2[tile_x, tile_y], a (tile_x, tile_y)
            # value:
            #   NotImplementedError: Cannot broadcast Helion ('tile_x', 'tile_y') to ('tile_y', 'tile_x')
            # tiled_view() reads it as in2[tile_y, tile_x] instead, passing
            # reinterpret_tensor(arg2_1, (40, 136), (1, 40), 0).
            if [dim for dim in dims if dim is not None] != list(axes):
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
        if not flat:
            return None
        return self.flat_subscript(buffer, index, axes, output=output)

    def flat_subscript(
        self,
        buffer: IRNode,
        index: sympy.Expr,
        axes: Sequence[str],
        *,
        output: bool = False,
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
        return arg, [self.index_tensor_expr(index, axes)], self.value_shape(axes)

    def disable_reduction(self) -> contextlib.AbstractContextManager[None]:
        outside = super().disable_reduction()

        @contextlib.contextmanager
        def ctx() -> Iterator[None]:
            with outside:
                yield
            # Back in a reduction loop: a value stored outside it, such as acc
            # of shape (tile_x,), is read through in_scope(), which emits
            # tmp3 = acc.reshape([tile_x, 1]). The CSE store cache would return
            # acc as is, and x / x.abs().amax(-1, keepdim=True) would fail
            # Inductor's shape propagation, which broadcasts trailing dims:
            #   AssertionError: expected str(d1) == str(d2), got tile_r != tile_x
            for name, value in [*self.cse.store_cache.items()]:
                if value in self.outer_vars:
                    self.outer_stores[name] = value
                    del self.cse.store_cache[name]

        return ctx()

    def codegen_body(self) -> None:
        """Move the pending code into self.body; reduction code goes in its loop.

        SIMDKernel.disable_reduction() calls this when a reduction loop ends, so
        the dots end up in for tile_r in hl.tile(K): and the epilogue after it.
        """
        if not (self.compute or self.stores):
            return
        if self.in_loop:
            self.body.writeline(f"for tile_r in hl.tile({self.numels['r0_']}):")
            with self.body.indent():
                self.body.splice(self.compute)
            # Reduction results are stored after the loop.
            self.body.splice(self.stores)
            # Values from the loop are out of scope after it. Without this, the
            # second loop of x / x.abs().amax(-1, keepdim=True) reuses
            # tmp0 = in0[tile_x, tile_r] from the first one:
            #   tmp3 = tmp0 / tmp2
            #   CrossRootDeviceValue: Device value 'tmp0' cannot be carried between top-level loops.
            self.cse.invalidate(self.outer_vars)
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
                "",
            ]
        )
        # FP32 dots follow torch.backends.cuda.matmul.fp32_precision, like
        # Inductor's Triton dots. With Helion's default, TF32, an FP32
        # [40, 72] @ [72, 136] differs from eager by up to 3e-2 instead of 1e-5.
        precision = (
            "tf32" if torch.backends.cuda.matmul.fp32_precision == "tf32" else "ieee"
        )
        code.writeline(
            f'@helion.kernel(backend="{config.helion.backend}", static_shapes=True, '
            f'autotune_effort="{config.helion.autotune_effort}", '
            f'dot_precision="{precision}")'
        )
        signature = ", ".join(f"{arg.name}: torch.Tensor" for arg in self.kernel_args())
        code.writeline(f"def kernel({signature}) -> None:")
        with code.indent():
            tiles = ", ".join(TILE_NAMES[axis] for axis in self.outer_axes)
            sizes = ", ".join(str(self.numels[axis]) for axis in self.outer_axes)
            # One axis is hl.tile(40): hl.tile([40]) yields 1-tuples, on which
            # tile_x.index fails with TypeInferenceError: Attributes are not
            # supported on SequenceType((TileIndexType(0), )).
            sizes = sizes if len(self.outer_axes) == 1 else f"[{sizes}]"
            code.writeline(f"for {tiles} in hl.tile({sizes}):")
            with code.indent():
                code.splice(self.body)
        return code.getvalue()

    def kernel_args(self) -> list[HelionTensorArg]:
        return sorted(self.arguments.values(), key=lambda arg: arg.output)

    def call_kernel(
        self, name: str, node: IRNode | None = None, deallocate_ws: bool = True
    ) -> None:
        # An output can take over an input's storage, e.g. for a @ b + cumsum(x):
        #   buf2 = buf0; del buf0  # reuse
        # Passing the input by its own name would then fail:
        #   helion_fused_add_1(arg0_1, arg1_1, buf0, buf2)
        #   UnboundLocalError: cannot access local variable 'buf0' where it is not associated with a value
        # so it is passed as the output, or as a view of it:
        #   helion_fused_add_1(arg0_1, arg1_1, buf2, buf2)
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
        if name in self.outer_stores:
            return self.in_scope(self.outer_stores[name])
        buffer = V.graph.get_buffer(name)
        index = self.tile_index(index)
        axes = self.index_axes(index)
        node = self.current_node
        if self.in_loop and node is not None and node.is_native_matmul():
            # A dot operand is a whole matrix tile for torch.addmm/baddbmm,
            # (z, y, r0_) or (z, r0_, x), even along an axis its index skips
            # (a stride-0 broadcast), and it keeps its storage dtype: the dot
            # body casts it to that anyway, so an upcast would emit a round trip
            #   tmp0 = in0[tile_y, tile_r].to(torch.float32)
            #   tmp1 = tmp0.to(torch.bfloat16)
            batch = ["z"] if "z" in self.numels else []
            operand = [*batch, "r0_", "x"] if "x" in axes else [*batch, "y", "r0_"]
            found = OrderedSet(axes) <= OrderedSet(operand) and self.subscript(
                buffer, index, operand, flat=False
            )
            if not found:
                raise NotImplementedError(
                    f"Cannot read a Helion dot operand at {index}"
                )
            arg, coordinates, shape = found
            line = f"{arg.name}[{', '.join(coordinates)}]"
            return self.cse.generate(
                self.loads, line, dtype=buffer.get_dtype(), shape=shape
            )
        found = self.subscript(buffer, index, axes)
        if found is None:
            raise AssertionError("expected a flat subscript")
        arg, coordinates, shape = found
        dtype = upcast_compute_type(buffer.get_dtype())
        cast_suffix = "" if dtype == buffer.get_dtype() else f".to({dtype})"
        subscript = ", ".join(coordinates)
        mask = self.load_mask(shape)
        if mask is None:
            line = f"{arg.name}[{subscript}]{cast_suffix}"
        else:
            # hl.load zero-fills masked lanes; masked() then selects `other`.
            # A 0-d tensor takes an empty index list. F.pad(x * s, (0, 1))[:, 3]
            # reads s under the constant mask 3 < 4, and its subscript in1[()]
            # as an index list would fail:
            #   tmp6 = hl.load(in1, [()], extra_mask=tmp4)
            #   InvalidIndexingType: Expected tile/int/None/tensor/etc in tensor[...], got ().
            mask = self.align(mask, shape)
            index = "" if coordinates == ["()"] else subscript
            line = f"hl.load({arg.name}, [{index}], extra_mask={mask}){cast_suffix}"
        value = self.cse.generate(self.loads, line, dtype=dtype, shape=shape)
        # A row vector continues as tmp3 = tmp2.reshape([tile_y, 1]). Left as
        # (tile_y,), a @ b + c for c of shape [40, 1] fails Inductor's shape
        # propagation, which broadcasts trailing dims:
        #   AssertionError: expected str(d1) == str(d2), got tile_x != tile_y
        aligned = self.value_shape(axes)
        if aligned == shape:
            return value
        return self.cse.generate(
            self.loads,
            self.align(value, self.value_shape()),
            dtype=dtype,
            shape=aligned,
        )

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
            # x.add_(a @ b) writes into x: in2[tile_y, tile_x] = tmp3.to(...).
            # The mutation buffer itself is not an output, so without this x
            # would be left unchanged.
            buffer = layout.get_buffer()
        elif name not in self.outputs:
            return
        index = self.tile_index(index)
        # A size-1 tile that the index skips stays in the subscript, through a
        # stride-0 view. x.sum() has one tile_x of size 1, and its acc of shape
        # (tile_x,) stores as out0[tile_x] = acc into
        # reinterpret_tensor(buf0, (1, ), (0, ), 0); out0[()] = acc would fail:
        #   NotImplementedError: Cannot broadcast Helion ('tile_x',) to ()
        value_axes = [TILE_AXES[str(dim)] for dim in value.shape or () if dim != 1]
        axes = [
            axis
            for axis in self.axes
            if self.symbols[axis] in index.free_symbols
            or (axis in value_axes and self.numels[axis] == 1)
        ]
        found = self.subscript(buffer, index, axes, output=True)
        if found is None:
            raise AssertionError("expected a flat subscript")
        arg, coordinates, shape = found
        dtype = buffer.get_dtype()
        cast_suffix = "" if value.dtype == dtype else f".to({dtype})"
        # out0[tile_y, tile_x] = tmp4.to(torch.bfloat16). A store in a reduction
        # loop, out0[tile_x, tile_r] = tmp4, stays in it; after the loop it would
        # fail with CrossRootDeviceValue: Device value 'tmp4' cannot be carried
        # between top-level loops. Reduction results (SIMDKernel.store_reduction)
        # are stored after the loop.
        buffer_code = self.compute if self.in_loop else self.stores
        buffer_code.writeline(
            f"{arg.name}[{', '.join(coordinates)}] = "
            f"{self.align(value, shape)}{cast_suffix}"
        )

    def index_expr(self, expr: sympy.Expr, dtype: torch.dtype) -> CSEVariable:
        """Index arithmetic on tile index tensors.

        torch.arange(40 * 136).reshape(40, 136) in an epilogue reads
        (tile_x.index + 136*tile_y.index[:, None]).to(torch.float32).
        """
        index = self.tile_index(expr)
        dtype = upcast_compute_type(dtype)
        if not index.free_symbols:
            value = int(index) if index.is_Integer else float(index)
            line = HelionOverrides.constant(value, dtype)
            return self.cse.generate(self.compute, line, dtype=dtype, shape=())
        axes = self.index_axes(index)
        return self.cse.generate(
            self.compute,
            f"({self.index_tensor_expr(index, axes)}).to({dtype})",
            dtype=dtype,
            shape=self.value_shape(axes),
        )

    def index_tensor_expr(self, index: sympy.Expr, axes: Sequence[str]) -> str:
        """`index` computed on the tile index tensors of `axes`.

        136*yindex + xindex -> 136*tile_y.index[:, None] + tile_x.index.
        """
        if not OrderedSet(axes) <= OrderedSet(self.scope_axes) or (
            index.free_symbols - OrderedSet(self.symbols[axis] for axis in axes)
        ):
            raise NotImplementedError(f"Unsupported Helion index: {index}")
        # tile.index is int32 (int64 only for tensors past 2**31 elements), so
        # widen it if any subexpression could overflow. Otherwise the mask of
        # torch.where((torch.arange(40 * 136).reshape(40, 136) * 10**8) % 7 == 0, a @ b, 0)
        # is ((100000000*tile_x.index + ...) % 7) in int32 and picks the wrong
        # elements.
        ranges = {
            self.symbols[axis]: ValueRanges(0, self.numels[axis] - 1) for axis in axes
        }
        bounds = [bound_sympy(term, ranges) for term in sympy.preorder_traversal(index)]
        widen = any(not (-(2**31) <= b.lower and b.upper < 2**31) for b in bounds)
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
        no mask and masked() alone selects the rows. Passing the (tile_y, 1)
        mask to that load would fail:
            NotImplementedError: Cannot broadcast Helion ('tile_y', 1) to ('tile_x',)
        """
        # A broadcast dim is 1, not a tile name: torch.bmm(a, b).view(2, 3, 40,
        # 136) + c for c of shape [2, 1, 1, 136] would otherwise fail with
        # KeyError: '1'.
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

    def accumulator(self, init: bool | float | int, dtype: torch.dtype) -> CSEVariable:
        """A new accumulator over the outer tiles, declared before its loop.

        acc = hl.zeros([tile_y, tile_x], dtype=torch.float32), or
        acc = hl.full([tile_x], float('-inf'), dtype=torch.float32) for a max.
        """
        tiles = tuple(TILE_NAMES[axis] for axis in self.outer_axes)
        name = f"acc{self.accumulator_count or ''}"
        self.accumulator_count += 1
        acc = CSEVariable(name, ValueRanges.unknown(), dtype, tiles)
        self.outer_vars.add(acc)
        shape = ", ".join(tiles)
        if init == 0 and dtype != torch.bool:
            self.body.writeline(f"{acc} = hl.zeros([{shape}], dtype={dtype})")
        else:
            value = literal(init)
            self.body.writeline(f"{acc} = hl.full([{shape}], {value}, dtype={dtype})")
        return acc

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
        acc = self.accumulator(0, torch.float32)
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
        """Reduce `value` over the reduction loop into an accumulator.

        Each iteration reduces its tile_r block, the last dim of `value`, and
        combines that with the accumulator. x.amax(-1) for x of shape [64, 100]:
            acc = hl.full([tile_x], float('-inf'), dtype=torch.float32)
            for tile_r in hl.tile(100):
                tmp0 = in0[tile_x, tile_r]
                acc = torch.maximum(acc, torch.amax(tmp0, -1))
            out0[tile_x] = acc
        Helion masks the lanes of a partial tile_r block in torch.amax, as in
        torch.sum, so no lane past R joins the result.
        """
        if reduction_type == "dot":
            # dot() already accumulated into acc.
            return value
        if self.is_matmul or reduction_type not in REDUCTIONS:
            raise NotImplementedError(f"Helion kernels do not support {reduction_type}")
        if not isinstance(value, CSEVariable) or "tile_r" not in map(
            str, value.shape or ()
        ):
            raise NotImplementedError("Helion reductions need a tile_r value")
        init = REDUCTIONS[reduction_type](upcast_compute_type(src_dtype))
        acc = self.accumulator(init, upcast_compute_type(dtype))
        block = {
            "sum": f"{acc} + torch.sum({value}, -1)",
            "prod": f"{acc} * torch.prod({value}, -1)",
            "max": f"torch.maximum({acc}, torch.amax({value}, -1))",
            "min": f"torch.minimum({acc}, torch.amin({value}, -1))",
        }[reduction_type]
        self.compute.writeline(f"{acc} = {block}")
        return acc


class HelionScheduling(TritonScheduling):
    """Emit schedules as Helion kernels where HelionKernel supports them.

    Native matmul schedules are always Helion kernels. Another schedule is a
    Helion kernel if HelionKernel generates it without NotImplementedError, e.g.
    a pointwise or reduction kernel, and a Triton kernel otherwise.
    """

    def __init__(self, scheduler: Any) -> None:
        super().__init__(scheduler)
        # Set while codegen_node_schedule() tries a HelionKernel.
        self.use_helion = False

    @staticmethod
    def format_kernel_definition(name: str, source: str) -> str:
        # helion_fused_mm_0_source = """\
        # import torch
        # ..."""
        # helion_fused_mm_0 = PyCodeCache.load(helion_fused_mm_0_source).kernel
        # Escaped so that e.g. a source line value = """x""" cannot end the
        # string early: SyntaxError when the wrapper module is loaded.
        escaped = source.replace("\\", "\\\\").replace('"""', '\\"""')
        return (
            f'{name}_source = """\\\n{escaped}"""\n'
            f"{name} = PyCodeCache.load({name}_source).kernel"
        )

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
        # An epilogue node runs once per output element, (B*)M*N, split into
        # the (z,) y, x tiles. A node that also iterates over K, e.g.
        # (a.unsqueeze(-1) * b.unsqueeze(0)).sum(1) next to a @ b, has the
        # dot's group (5440, 72) and would land in the K loop:
        #   NotImplementedError: Unsupported Helion index: r0_index + 72*yindex
        # Row reductions of the output, [M, N] -> [M], are rejected here too
        # and run as Triton kernels after the Helion matmul.
        tiles = list(dots[0].get_ranges()[0])
        numel, groups = sympy_product(tiles), [*tiles, sympy.S.One]
        for node in nodes:
            if node in dots:
                continue
            pointwise = node.group[1] == (numel, 1)
            if not pointwise or not SIMDKernel.is_compatible(groups, node.get_ranges()):
                why("Helion epilogue must iterate over the matmul output tiles")
                return False
            # a @ b + torch.tensor(2.0) reads a CPU scalar, which the kernel
            # would get as a CPU tensor:
            #   ValueError: Pointer argument cannot be accessed from Triton (cpu tensor?)
            if any(V.graph.is_unspec_arg(dep.name) for dep in node.read_writes.reads):
                why("Helion kernels do not read CPU scalars")
                return False
        return super().can_fuse(node1, node2)

    can_fuse_vertical = can_fuse
    can_fuse_horizontal = can_fuse

    def generate_node_schedule(
        self, nodes, numel, rnumel, *, required_post_reduction_index=None
    ):
        # Dots first, so that they share one K loop. ((a @ w).float(),
        # (b @ w).float()) fuses as [dot1, cast1, dot2, cast2]; in that order
        # the K loop closes at cast1, which needs dot1's result, and dot2 opens
        # a second K loop that loads w again:
        #   for tile_r in hl.tile(16):
        #       tmp0 = in0[tile_y, tile_r]
        #       tmp1 = in1[tile_r, tile_x]
        #       acc = torch.addmm(acc, tmp0, tmp1)
        #   out0[tile_y, tile_x] = acc
        #   acc1 = hl.zeros([tile_y, tile_x], dtype=torch.float32)
        #   for tile_r in hl.tile(16):
        #       tmp2 = in2[tile_y, tile_r]
        #       tmp3 = in1[tile_r, tile_x]
        #       acc1 = torch.addmm(acc1, tmp2, tmp3)
        # No dot reads another node of its kernel (a fused read must use the
        # index of the write, and a dot reads its operands at a K-dependent
        # index), so moving the dots first keeps every producer before its use.
        if required_post_reduction_index is None and any(map(is_helion_matmul, nodes)):
            nodes = sorted(nodes, key=lambda node: not is_helion_matmul(node))
        return super().generate_node_schedule(
            nodes,
            numel,
            rnumel,
            required_post_reduction_index=required_post_reduction_index,
        )

    @staticmethod
    def helion_candidate(kernel_features: Any) -> bool:
        """Whether a schedule without a dot uses only what HelionKernel supports.

        Checked before codegen, as SchedulerNode.codegen() logs any error at
        critical level, e.g. "Error in codegen for ComputedBuffer(...
        reduction_type=welford_reduce ...)" for a layer norm.
        """
        return all(
            isinstance(node, SchedulerNode)
            and not node.is_template()
            and not is_helion_matmul(node)
            and not OrderedSet(node._body.op_counts) - HELION_OPS
            and (
                not isinstance(node.node, ComputedBuffer)
                or node.node.get_reduction_type() in (None, *REDUCTIONS)
            )
            and not any(
                entry.mode for entry in node._body.memory_usage[MemoryUsageType.STORE]
            )
            and not any(
                V.graph.is_unspec_arg(dep.name) for dep in node.read_writes.reads
            )
            for node in kernel_features.scheduler_nodes()
        )

    def codegen_node_schedule(self, kernel_features):
        if not self.helion_candidate(kernel_features):
            return super().codegen_node_schedule(kernel_features)
        # Codegen fails before anything reaches the wrapper (define_kernel), so
        # the schedule can start over as a Triton kernel.
        self.use_helion = True
        try:
            return super().codegen_node_schedule(kernel_features)
        except NotImplementedError as e:
            log.debug("Emitting a Triton kernel instead of a Helion kernel: %s", e)
        finally:
            self.use_helion = False
        return super().codegen_node_schedule(kernel_features)

    def create_kernel_choices(self, kernel_features, kernel_args, kernel_kwargs):
        if self.use_helion or any(
            is_helion_matmul(node) for node in kernel_features.scheduler_nodes()
        ):
            return [HelionKernel(*kernel_args, **kernel_kwargs)]
        return super().create_kernel_choices(
            kernel_features, kernel_args, kernel_kwargs
        )

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
        # gpu=False keeps the module out of the Triton compile-time autotune
        # block, which has no PyCodeCache import: with
        # triton.autotune_at_compile_time=True it would fail with
        #   RuntimeError: Failed to run autotuning code block: name 'PyCodeCache' is not defined
        wrapper.define_kernel(
            name,
            self.format_kernel_definition(name, src_code),
            gpu=False,
            standalone=True,
        )
        return name
