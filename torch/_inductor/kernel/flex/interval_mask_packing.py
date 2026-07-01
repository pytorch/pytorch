# mypy: allow-untyped-defs
"""Packed interval mask analysis for flex flash attention.

Flex attention can evaluate arbitrary ``mask_mod`` functions lane by lane, but
SM100 packed masks can use a cheaper path when the kept KV lanes in each
32-lane chunk are window-like. This pass recognizes mask graphs where, after
substituting ``kv_idx + mask_lane`` for the per-lane KV index, the predicate can
be represented as a small union of half-open intervals over ``mask_lane``.

This works for causal, sliding-window, block, and document-bound masks whose
bounds depend only on lane-uniform values such as ``q_idx``, ``b_idx``, or aux
loads indexed by them. It intentionally rejects masks that need arbitrary
per-lane inclusion, such as ``doc_ids[b, kv_idx] == doc_ids[b, q_idx]`` or
``kv_idx % 2 == 0``; those would degenerate into many singleton intervals and
are better handled by the generic mask path.
"""

import dataclasses
from collections.abc import Mapping, Sequence
from itertools import product

import sympy

import torch
from torch.fx import GraphModule
from torch.utils._sympy.functions import FloorDiv, Max, Min

from ...codegen.cutedsl.lane_analysis import decompose_affine_lane_expr
from ...virtualized import NullHandler, V
from .aux_vectorization import fx_aux_index_to_sympy


# Fall back to scalar mask lowering before interval expansion makes generated CuTe too large.
MAX_PACKED_MASK_INTERVALS_FOR_CODE_SIZE = 8
LANE_UNIFORM_BINARY_OPS: Mapping[object, str] = {
    torch.ops.aten.add.Tensor: "+",
    torch.ops.aten.add.Scalar: "+",
    torch.ops.aten.sub.Tensor: "-",
    torch.ops.aten.sub.Scalar: "-",
    torch.ops.aten.mul.Tensor: "*",
    torch.ops.aten.mul.Scalar: "*",
    torch.ops.aten.remainder.Tensor: "%",
    torch.ops.aten.remainder.Scalar: "%",
}


@dataclasses.dataclass(frozen=True)
class PackedMaskInterval:
    """Half-open keep interval [lower_lane, upper_lane_exclusive) for a packed mask.

    Packed interval lowering replaces per-lane boolean mask evaluation with one
    32-bit keep mask when the mask predicate can be expressed as lane bounds.
    For example, ``q_idx >= kv_idx`` over a 32-lane KV window starting at
    ``kv_idx`` keeps lanes ``[0, q_idx - kv_idx + 1)``.
    """

    lower_lane: sympy.Expr
    upper_lane_exclusive: sympy.Expr
    # CuTe code for temporary symbols that stand in for lane-uniform aux loads.
    symbol_codes: Mapping[sympy.Symbol, str] = dataclasses.field(
        default_factory=dict, compare=False, repr=False
    )

    @classmethod
    def full(cls) -> "PackedMaskInterval":
        return cls(sympy.Integer(0), sympy.Integer(32))

    def is_full(self) -> bool:
        return self.lower_lane == 0 and self.upper_lane_exclusive == 32

    def with_symbol_codes(
        self, symbol_codes: Mapping[sympy.Symbol, str]
    ) -> "PackedMaskInterval":
        return dataclasses.replace(self, symbol_codes=dict(symbol_codes))

    def render_lower(self) -> str:
        return self._render(self.lower_lane)

    def render_upper(self) -> str:
        return self._render(self.upper_lane_exclusive)

    def keep_mask_expr(self) -> str:
        """Render the 32-bit CuTe keep-mask expression for this interval."""
        lower = self.render_lower()
        upper = self.render_upper()
        return (
            "(utils.shr_u32(cutlass.Uint32(0xFFFFFFFF), "
            "cutlass.Uint32(min(max(cutlass.Int32(32) - "
            f"{upper}, cutlass.Int32(0)), cutlass.Int32(32)))) & "
            "utils.shl_u32(cutlass.Uint32(0xFFFFFFFF), "
            "cutlass.Uint32(min(max("
            f"{lower}, cutlass.Int32(0)), cutlass.Int32(32)))))"
        )

    def _render(self, expr: sympy.Expr) -> str:
        rendered = sympy_to_cute_index(expr, self.symbol_codes)
        if rendered is None:
            raise AssertionError(f"failed to render expr to cute index: {expr}")
        return rendered


IntervalSet = tuple[PackedMaskInterval, ...]
MaybeIntervalSet = IntervalSet | None


def is_bool_full_node(node: torch.fx.Node, value: bool) -> bool:
    return (
        node.op == "call_function"
        and node.target is torch.ops.aten.full.default
        and len(node.args) >= 2
        and node.args[0] == []
        and node.args[1] is value
    )


def _simplify_expr(expr: sympy.Expr) -> sympy.Expr:
    """Simplify with Inductor sizevars when available, otherwise use SymPy."""
    if isinstance(V.graph, NullHandler):
        return sympy.simplify(expr)
    return V.graph.sizevars.simplify(expr)


def sympy_to_cute_index(
    expr: sympy.Expr, symbol_codes: Mapping[sympy.Symbol, str] | None = None
) -> str | None:
    """Render a supported SymPy integer expression as CuTe index code."""
    expr = _simplify_expr(expr)
    match expr:
        case sympy.Integer():
            return f"cutlass.Int32({int(expr)})"
        case sympy.Symbol():
            if symbol_codes is not None and expr in symbol_codes:
                return symbol_codes[expr]
            if expr.name == "q_idx":
                return "q_idx[0]"
            if expr.name == "kv_idx":
                return "kv_idx[0]"
        case FloorDiv():
            lhs, rhs = (sympy_to_cute_index(arg, symbol_codes) for arg in expr.args)
            if lhs is not None and rhs is not None:
                return f"({lhs} // {rhs})"
        case sympy.Add():
            args = []
            for arg in expr.args:
                cute_arg = sympy_to_cute_index(arg, symbol_codes)
                if cute_arg is None:
                    return None
                args.append(cute_arg)
            return "(" + " + ".join(args) + ")"
        case sympy.Mul():
            args = []
            for arg in expr.args:
                cute_arg = sympy_to_cute_index(arg, symbol_codes)
                if cute_arg is None:
                    return None
                args.append(cute_arg)
            return "(" + " * ".join(args) + ")"
        case Min() | Max():
            args = []
            for arg in expr.args:
                cute_arg = sympy_to_cute_index(arg, symbol_codes)
                if cute_arg is None:
                    return None
                args.append(cute_arg)
            op = "min" if isinstance(expr, Min) else "max"
            return f"{op}(" + ", ".join(args) + ")"
    if isinstance(expr, (sympy.Min, sympy.Max)):
        args = []
        for arg in expr.args:
            cute_arg = sympy_to_cute_index(arg, symbol_codes)
            if cute_arg is None:
                return None
            args.append(cute_arg)
        op = "min" if isinstance(expr, sympy.Min) else "max"
        return f"{op}(" + ", ".join(args) + ")"
    return None


def is_aten_index_node(node: torch.fx.Node) -> bool:
    return node.op == "call_function" and node.target is torch.ops.aten.index.Tensor


def fx_node_dtype(expr: torch.fx.Node) -> torch.dtype | None:
    """Read an FX node dtype from metadata, following aten.index bases if needed."""
    tensor_meta = expr.meta.get("tensor_meta")
    if tensor_meta is not None:
        return tensor_meta.dtype
    val = expr.meta.get("val")
    if isinstance(val, torch.Tensor):
        return val.dtype
    if is_aten_index_node(expr):
        base = expr.args[0]
        if isinstance(base, torch.fx.Node):
            return fx_node_dtype(base)
    return None


def fx_node_shape(expr: torch.fx.Node) -> tuple[int, ...] | None:
    """Read an FX node shape from tensor metadata or fake tensor value."""
    tensor_meta = expr.meta.get("tensor_meta")
    if tensor_meta is not None:
        return tuple(tensor_meta.shape)
    val = expr.meta.get("val")
    if isinstance(val, torch.Tensor):
        return tuple(val.shape)
    return None


@dataclasses.dataclass
class PackedMaskAnalyzer:
    """Lower one Flex mask_mod FX graph to packed 32-lane mask intervals.

    The analyzer treats each packed mask as a 32-lane window starting at kv_idx,
    rewrites supported mask predicates into bounds over mask_lane, and records
    renderable code for lane-uniform aux loads used by those bounds. For example,
    a document mask like ``doc_offsets[b_idx] <= kv_idx and kv_idx < q_idx + 128``
    is analyzed over the packed lane expression ``kv_idx + mask_lane`` and
    becomes the interval ``[doc_offsets[b_idx] - kv_idx, q_idx + 128 - kv_idx)``.

    placeholders: FX placeholders for the mask_mod ABI and captured inputs.
    q_idx: FX placeholder for the scalar query index.
    kv_idx: FX placeholder for the first KV index in the packed 32-lane window.
    q_symbol: SymPy symbol used for q_idx during interval analysis.
    kv_symbol: SymPy symbol used for the packed window base KV index.
    lane_symbol: SymPy symbol for the lane offset within the packed window.
    symbol_codes: CuTe renderings for temporary symbols introduced for aux loads.
    placeholder_codes: CuTe renderings for captured scalar and tensor placeholders.
    placeholder_exprs: SymPy expressions for captured scalar placeholders.
    aux_load_symbols: Stable symbols assigned to supported lane-uniform aux loads.
    next_symbol_id: Counter for naming temporary aux-load symbols.
    """

    placeholders: Sequence[torch.fx.Node]
    q_idx: torch.fx.Node
    kv_idx: torch.fx.Node
    q_symbol: sympy.Symbol = dataclasses.field(
        default_factory=lambda: sympy.Symbol("q_idx", integer=True, nonnegative=True)
    )
    kv_symbol: sympy.Symbol = dataclasses.field(
        default_factory=lambda: sympy.Symbol("kv_idx", integer=True, nonnegative=True)
    )
    lane_symbol: sympy.Symbol = dataclasses.field(
        default_factory=lambda: sympy.Symbol(
            "mask_lane", integer=True, nonnegative=True
        )
    )
    symbol_codes: dict[sympy.Symbol, str] = dataclasses.field(default_factory=dict)
    placeholder_codes: Mapping[torch.fx.Node, str] = dataclasses.field(
        default_factory=dict
    )
    placeholder_exprs: Mapping[torch.fx.Node, sympy.Expr] = dataclasses.field(
        default_factory=dict
    )
    aux_load_symbols: dict[torch.fx.Node, sympy.Symbol] = dataclasses.field(
        default_factory=dict
    )
    next_symbol_id: int = 0

    def node_to_intervals(self, node: torch.fx.Node) -> MaybeIntervalSet:
        """Lower one supported boolean FX node into packed mask intervals."""
        if is_bool_full_node(node, True):
            return (PackedMaskInterval.full(),)
        if is_bool_full_node(node, False):
            return ()
        if node.op != "call_function":
            return None
        match node.target:
            case torch.ops.aten.bitwise_and.Tensor | torch.ops.aten.logical_and.default:
                return self.combine_binary_intervals(node, intersect=True)
            case torch.ops.aten.bitwise_or.Tensor | torch.ops.aten.logical_or.default:
                return self.combine_binary_intervals(node, intersect=False)
            case torch.ops.aten.le.Tensor | torch.ops.aten.le.Scalar:
                return self.comparison_to_intervals(
                    node.args[0], node.args[1], strict=False
                )
            case torch.ops.aten.lt.Tensor | torch.ops.aten.lt.Scalar:
                return self.comparison_to_intervals(
                    node.args[0], node.args[1], strict=True
                )
            case torch.ops.aten.ge.Tensor | torch.ops.aten.ge.Scalar:
                return self.comparison_to_intervals(
                    node.args[1], node.args[0], strict=False
                )
            case torch.ops.aten.gt.Tensor | torch.ops.aten.gt.Scalar:
                return self.comparison_to_intervals(
                    node.args[1], node.args[0], strict=True
                )
            case torch.ops.aten.eq.Tensor | torch.ops.aten.eq.Scalar:
                return self.equality_to_intervals(node.args[0], node.args[1])
            case _:
                return None

    def combine_binary_intervals(
        self, node: torch.fx.Node, *, intersect: bool
    ) -> MaybeIntervalSet:
        lhs, rhs = node.args
        if not isinstance(lhs, torch.fx.Node) or not isinstance(rhs, torch.fx.Node):
            return None
        lhs_intervals = self.node_to_intervals(lhs)
        rhs_intervals = self.node_to_intervals(rhs)
        if lhs_intervals is None or rhs_intervals is None:
            return None
        if intersect:
            return self.merge_intersection(lhs_intervals, rhs_intervals)
        if (
            len(lhs_intervals) + len(rhs_intervals)
            > MAX_PACKED_MASK_INTERVALS_FOR_CODE_SIZE
        ):
            return None
        return lhs_intervals + rhs_intervals

    def comparison_to_intervals(
        self,
        lhs: object,
        rhs: object,
        *,
        strict: bool,
    ) -> MaybeIntervalSet:
        exprs = self.mask_operands_to_sympy(lhs, rhs)
        if exprs is None:
            return None
        return self.lane_comparison_to_intervals(*exprs, strict=strict)

    def equality_to_intervals(self, lhs: object, rhs: object) -> MaybeIntervalSet:
        exprs = self.mask_operands_to_sympy(lhs, rhs)
        if exprs is None:
            return None
        return self.lane_equality_to_intervals(*exprs)

    def mask_operands_to_sympy(
        self, lhs: object, rhs: object
    ) -> tuple[sympy.Expr, sympy.Expr] | None:
        lhs_expr = self.fx_mask_expr_to_sympy(lhs)
        rhs_expr = self.fx_mask_expr_to_sympy(rhs)
        if lhs_expr is None or rhs_expr is None:
            return None
        return lhs_expr, rhs_expr

    def lane_comparison_to_intervals(
        self,
        lhs_expr: sympy.Expr,
        rhs_expr: sympy.Expr,
        *,
        strict: bool,
    ) -> MaybeIntervalSet:
        """Convert a lane-affine comparison into one packed keep interval."""
        diff = V.graph.sizevars.simplify(lhs_expr - rhs_expr)
        affine = decompose_affine_lane_expr(diff, self.lane_symbol)
        if affine is None:
            return None
        lane_coeff, rest = affine
        if lane_coeff == 1:
            upper = V.graph.sizevars.simplify(-rest if strict else -rest + 1)
            return self.interval_if_renderable(sympy.Integer(0), upper)
        if lane_coeff == -1:
            lower = V.graph.sizevars.simplify(rest + 1 if strict else rest)
            return self.interval_if_renderable(lower, sympy.Integer(32))
        return None

    def lane_equality_to_intervals(
        self, lhs_expr: sympy.Expr, rhs_expr: sympy.Expr
    ) -> MaybeIntervalSet:
        """Convert a lane-affine equality into singleton keep intervals."""
        if isinstance(lhs_expr, FloorDiv) and isinstance(rhs_expr, FloorDiv):
            intervals = self._floor_div_equality_to_intervals(lhs_expr, rhs_expr)
            if intervals is not None:
                return intervals
        diff = V.graph.sizevars.simplify(lhs_expr - rhs_expr)
        affine = decompose_affine_lane_expr(diff, self.lane_symbol)
        if affine is None:
            return None
        lane_coeff, rest = affine
        if lane_coeff in (1, -1):
            lane_value = -rest if lane_coeff == 1 else rest
            lower = V.graph.sizevars.simplify(lane_value)
            upper = V.graph.sizevars.simplify(lane_value + 1)
            return self.interval_if_renderable(lower, upper)
        return None

    def _floor_div_equality_to_intervals(
        self, lhs_expr: FloorDiv, rhs_expr: FloorDiv
    ) -> MaybeIntervalSet:
        """Convert matching block-id equality into a contiguous lane interval."""
        lhs_base, lhs_divisor = lhs_expr.args
        rhs_base, rhs_divisor = rhs_expr.args
        if lhs_divisor != rhs_divisor:
            return None
        block_start = None
        if lhs_base == self.q_symbol and rhs_base == self.kv_symbol + self.lane_symbol:
            block_start = V.graph.sizevars.simplify(
                lhs_expr * lhs_divisor - self.kv_symbol
            )
        elif (
            rhs_base == self.q_symbol and lhs_base == self.kv_symbol + self.lane_symbol
        ):
            block_start = V.graph.sizevars.simplify(
                rhs_expr * rhs_divisor - self.kv_symbol
            )
        if block_start is None:
            return None
        lower = V.graph.sizevars.simplify(block_start)
        upper = V.graph.sizevars.simplify(block_start + lhs_divisor)
        return self.interval_if_renderable(lower, upper)

    def interval_if_renderable(
        self, lower: sympy.Expr, upper: sympy.Expr
    ) -> MaybeIntervalSet:
        if (
            sympy_to_cute_index(lower, self.symbol_codes) is not None
            and sympy_to_cute_index(upper, self.symbol_codes) is not None
        ):
            return (PackedMaskInterval(lower, upper),)
        return None

    def merge_intersection(
        self,
        intervals: IntervalSet,
        new_intervals: IntervalSet,
    ) -> MaybeIntervalSet:
        """Intersect two interval sets, falling back if code size would grow too much."""
        if (
            len(intervals) * len(new_intervals)
            > MAX_PACKED_MASK_INTERVALS_FOR_CODE_SIZE
        ):
            return None
        return tuple(
            PackedMaskInterval(
                V.graph.sizevars.simplify(Max(lhs.lower_lane, rhs.lower_lane)),
                V.graph.sizevars.simplify(
                    Min(lhs.upper_lane_exclusive, rhs.upper_lane_exclusive)
                ),
            )
            for lhs, rhs in product(intervals, new_intervals)
        )

    def fx_mask_expr_to_sympy(self, expr: object) -> sympy.Expr | None:
        """Translate a mask expression with kv_idx expanded by mask_lane."""
        index_symbols = {
            self.q_idx: self.q_symbol,
            self.kv_idx: self.kv_symbol + self.lane_symbol,
        }
        index_symbols.update(self.placeholder_exprs)
        return fx_aux_index_to_sympy(expr, index_symbols, self.mask_aux_load_to_symbol)

    def mask_aux_load_to_symbol(self, node: torch.fx.Node) -> sympy.Expr | None:
        """Assign a temporary SymPy symbol to a supported aux tensor load."""
        if not is_aten_index_node(node):
            return None
        if node in self.aux_load_symbols:
            return self.aux_load_symbols[node]
        lane_uniform_code = self.render_lane_uniform_scalar_expr(node)
        if lane_uniform_code is None:
            return None
        symbol = sympy.Symbol(f"mask_bound_{self.next_symbol_id}", integer=True)
        self.next_symbol_id += 1
        self.symbol_codes[symbol] = lane_uniform_code
        self.aux_load_symbols[node] = symbol
        return symbol

    def render_lane_uniform_scalar_expr(
        self,
        expr: object,
        *,
        for_index: bool = False,
        index_dim_size: int | sympy.Expr | None = None,
    ) -> str | None:
        """Render an expression that is invariant across the packed KV lanes."""
        if isinstance(expr, int | sympy.Integer):
            index = int(expr)
            if for_index and index < 0:
                if index_dim_size is None:
                    return None
                index = V.graph.sizevars.guard_int(index + index_dim_size)
            return f"cutlass.Int32({index})"
        if not isinstance(expr, torch.fx.Node):
            return None

        if expr is self.kv_idx:
            return None
        if expr is self.q_idx:
            return "q_idx[0]"
        if len(self.placeholders) >= 2:
            if expr is self.placeholders[0]:
                return "b_idx[0]"
            if expr is self.placeholders[1]:
                return "h_idx[0]"
        if expr in self.placeholder_codes:
            return self.placeholder_codes[expr]

        if is_aten_index_node(expr):
            return self._render_lane_uniform_index_expr(expr, for_index=for_index)
        if expr.op != "call_function":
            return None
        match expr.target:
            case torch.ops.aten.div.Tensor_mode:
                if expr.kwargs.get("rounding_mode") != "floor":
                    return None
                op = "//"
            case _:
                op = LANE_UNIFORM_BINARY_OPS.get(expr.target)
                if op is None:
                    return None
        args = expr.args
        if len(args) < 2:
            return None
        lhs = self.render_lane_uniform_scalar_expr(args[0])
        rhs = self.render_lane_uniform_scalar_expr(args[1])
        if lhs is None or rhs is None:
            return None
        return f"({lhs} {op} {rhs})"

    def _render_lane_uniform_index_expr(
        self,
        expr: torch.fx.Node,
        *,
        for_index: bool,
    ) -> str | None:
        """Render a scalar integer aten.index load that is uniform across lanes."""
        if fx_node_dtype(expr) not in (
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        ):
            return None
        result_shape = fx_node_shape(expr)
        if result_shape is not None and len(result_shape) != 0:
            return None
        base, indices = expr.args
        base_code = self.render_lane_uniform_scalar_expr(base)
        base_shape = fx_node_shape(base) if isinstance(base, torch.fx.Node) else None
        if base_code is None or not isinstance(indices, (list, tuple)):
            return None
        if base_shape is None or len(indices) != len(base_shape):
            return None
        index_codes = []
        for dim, index in enumerate(indices):
            dim_size = base_shape[dim]
            index_code = self.render_lane_uniform_scalar_expr(
                index, for_index=True, index_dim_size=dim_size
            )
            if index_code is None:
                return None
            if (
                dim_size is not None
                and isinstance(index, torch.fx.Node)
                and index
                not in (self.q_idx, self.placeholders[0], self.placeholders[1])
            ):
                if not isinstance(dim_size, int | sympy.Integer):
                    return None
                dtype = fx_node_dtype(index)
                integer_type = "Int64" if dtype == torch.int64 else "Int32"
                size_code = f"cutlass.{integer_type}({dim_size})"
                zero_code = f"cutlass.{integer_type}(0)"
                index_code = (
                    f"({index_code} + {size_code} "
                    f"if {index_code} < {zero_code} else {index_code})"
                )
            index_codes.append(index_code)
        load = f"{base_code}[{', '.join(index_codes)}]"
        if for_index and fx_node_dtype(expr) == torch.int64:
            return load
        return f"cutlass.Int32({load})"


@dataclasses.dataclass(frozen=True)
class PackedMaskAuxPlaceholderMap:
    """Maps mask_mod aux placeholders to runtime scalar expressions or tensors."""

    codes: Mapping[torch.fx.Node, str]
    exprs: Mapping[torch.fx.Node, sympy.Expr]

    @classmethod
    def from_placeholders(
        cls,
        placeholders: Sequence[torch.fx.Node],
        other_buffers: Sequence[object],
        symbol_codes: Mapping[sympy.Symbol, str],
    ) -> "PackedMaskAuxPlaceholderMap | None":
        """Render lane-uniform mask_mod captures for packed interval analysis."""
        codes: dict[torch.fx.Node, str] = {}
        exprs: dict[torch.fx.Node, sympy.Expr] = {}
        tensor_idx = 0
        for placeholder, buffer in zip(placeholders[4:], other_buffers):
            if isinstance(buffer, sympy.Expr):
                rendered = sympy_to_cute_index(buffer, symbol_codes)
                if rendered is None:
                    return None
                codes[placeholder] = rendered
                exprs[placeholder] = buffer
            else:
                codes[placeholder] = f"aux_tensors[{tensor_idx}]"
                tensor_idx += 1
        for placeholder in placeholders[4 + len(other_buffers) :]:
            codes[placeholder] = f"aux_tensors[{tensor_idx}]"
            tensor_idx += 1
        return cls(codes, exprs)


def select_packed_mask_intervals(
    graph_module: GraphModule,
    other_buffers: Sequence[object] = (),
    aux_scalar_symbol_codes: Mapping[sympy.Symbol, str] | None = None,
) -> MaybeIntervalSet:
    """Entry point for selecting packed mask intervals from a mask_mod graph.

    Returns ``None`` when the graph is unsupported, when interval packing would
    increase code size too much, or when analysis reduces the whole mask to the
    unconditional ``[0, 32)`` interval.
    """
    graph = graph_module.graph
    nodes = list(graph.nodes)
    placeholders = [node for node in nodes if node.op == "placeholder"]
    output = [node for node in nodes if node.op == "output"]
    if len(placeholders) < 4 or len(output) != 1:
        return None

    q_idx, kv_idx = placeholders[2], placeholders[3]
    output_val = output[0].args[0]
    if not isinstance(output_val, torch.fx.Node):
        return None

    symbol_codes = dict(aux_scalar_symbol_codes or {})
    placeholder_map = PackedMaskAuxPlaceholderMap.from_placeholders(
        placeholders, other_buffers, symbol_codes
    )
    if placeholder_map is None:
        return None

    analyzer = PackedMaskAnalyzer(
        placeholders=placeholders,
        q_idx=q_idx,
        kv_idx=kv_idx,
        symbol_codes=symbol_codes,
        placeholder_codes=placeholder_map.codes,
        placeholder_exprs=placeholder_map.exprs,
    )
    intervals = analyzer.node_to_intervals(output_val)
    if intervals is None:
        return None
    if len(intervals) == 1 and intervals[0].is_full():
        return None
    return tuple(
        interval.with_symbol_codes(analyzer.symbol_codes) for interval in intervals
    )
