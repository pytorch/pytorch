# mypy: allow-untyped-defs
"""Packed interval mask analysis for flex flash attention."""

import dataclasses
from collections.abc import Mapping, Sequence

import sympy

import torch
from torch.fx import GraphModule
from torch.utils._sympy.functions import FloorDiv, Max, Min

from ...codegen.cutedsl.lane_analysis import decompose_affine_lane_expr
from ...virtualized import NullHandler, V
from .aux_vectorization import fx_aux_index_to_sympy, make_fx_index_symbols


MAX_PACKED_MASK_INTERVALS = 8
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
    lower: sympy.Expr
    upper: sympy.Expr
    symbol_codes: Mapping[sympy.Symbol, str] = dataclasses.field(
        default_factory=dict, compare=False, repr=False
    )

    @classmethod
    def full(cls) -> "PackedMaskInterval":
        return cls(sympy.Integer(0), sympy.Integer(32))

    def is_full(self) -> bool:
        return self.lower == 0 and self.upper == 32

    def with_symbol_codes(
        self, symbol_codes: Mapping[sympy.Symbol, str]
    ) -> "PackedMaskInterval":
        return dataclasses.replace(self, symbol_codes=dict(symbol_codes))

    def render_lower(self) -> str:
        return self._render(self.lower)

    def render_upper(self) -> str:
        return self._render(self.upper)

    def keep_mask_expr(self) -> str:
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
        assert rendered is not None
        return rendered


@dataclasses.dataclass(frozen=True)
class LaneIndexAnalysis:
    q_idx: sympy.Symbol
    kv_idx: sympy.Symbol
    lane: sympy.Symbol

    @property
    def kv_lane(self) -> sympy.Expr:
        return self.kv_idx + self.lane


def is_bool_full_node(node: torch.fx.Node, value: bool) -> bool:
    return (
        node.op == "call_function"
        and node.target is torch.ops.aten.full.default
        and len(node.args) >= 2
        and node.args[0] == []
        and node.args[1] is value
    )


def _simplify_expr(expr: sympy.Expr) -> sympy.Expr:
    if isinstance(V.graph, NullHandler):
        return sympy.simplify(expr)
    return V.graph.sizevars.simplify(expr)


def sympy_to_cute_index(
    expr: sympy.Expr, symbol_codes: Mapping[sympy.Symbol, str] | None = None
) -> str | None:
    expr = _simplify_expr(expr)
    if isinstance(expr, sympy.Integer):
        return f"cutlass.Int32({int(expr)})"
    if isinstance(expr, sympy.Symbol):
        if symbol_codes is not None and expr in symbol_codes:
            return symbol_codes[expr]
        if expr.name == "q_idx":
            return "q_idx[0]"
        if expr.name == "kv_idx":
            return "kv_idx[0]"
    if isinstance(expr, FloorDiv):
        lhs, rhs = (sympy_to_cute_index(arg, symbol_codes) for arg in expr.args)
        if lhs is not None and rhs is not None:
            return f"({lhs} // {rhs})"
        return None
    if isinstance(expr, sympy.Add):
        args = []
        for arg in expr.args:
            cute_arg = sympy_to_cute_index(arg, symbol_codes)
            if cute_arg is None:
                return None
            args.append(cute_arg)
        return "(" + " + ".join(args) + ")"
    if isinstance(expr, sympy.Mul):
        args = []
        for arg in expr.args:
            cute_arg = sympy_to_cute_index(arg, symbol_codes)
            if cute_arg is None:
                return None
            args.append(cute_arg)
        return "(" + " * ".join(args) + ")"
    if isinstance(expr, (Min, Max, sympy.Min, sympy.Max)):
        args = []
        for arg in expr.args:
            cute_arg = sympy_to_cute_index(arg, symbol_codes)
            if cute_arg is None:
                return None
            args.append(cute_arg)
        op = "min" if isinstance(expr, (Min, sympy.Min)) else "max"
        return f"{op}(" + ", ".join(args) + ")"
    return None


def is_aten_index_node(node: torch.fx.Node) -> bool:
    return node.op == "call_function" and node.target is torch.ops.aten.index.Tensor


def fx_node_dtype(expr: torch.fx.Node) -> torch.dtype | None:
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
    tensor_meta = expr.meta.get("tensor_meta")
    if tensor_meta is not None:
        return tuple(tensor_meta.shape)
    val = expr.meta.get("val")
    if isinstance(val, torch.Tensor):
        return tuple(val.shape)
    return None


@dataclasses.dataclass
class PackedMaskAnalyzer:
    """Lower one Flex mask_mod FX graph to packed 32-lane mask intervals."""

    placeholders: Sequence[torch.fx.Node]
    q_idx: torch.fx.Node
    kv_idx: torch.fx.Node
    lane_analysis: LaneIndexAnalysis
    symbol_codes: dict[sympy.Symbol, str] = dataclasses.field(default_factory=dict)
    aux_load_symbols: dict[torch.fx.Node, sympy.Symbol] = dataclasses.field(
        default_factory=dict
    )
    next_symbol_id: int = 0

    def lane_uniform_cute_expr(
        self,
        expr: object,
        *,
        for_index: bool = False,
        index_dim_size: int | sympy.Expr | None = None,
    ) -> str | None:
        if isinstance(expr, int | sympy.Integer):
            return self._literal_cute_expr(
                int(expr), for_index=for_index, index_dim_size=index_dim_size
            )
        if not isinstance(expr, torch.fx.Node):
            return None

        if expr is self.q_idx:
            return "q_idx[0]"
        if expr is self.kv_idx:
            return None
        if len(self.placeholders) >= 2:
            if expr is self.placeholders[0]:
                return "b_idx[0]"
            if expr is self.placeholders[1]:
                return "h_idx[0]"
        if expr in self.placeholders[4:]:
            return f"aux_tensors[{self.placeholders[4:].index(expr)}]"

        if is_aten_index_node(expr):
            return self._lane_uniform_index_cute_expr(expr, for_index=for_index)
        return self._lane_uniform_binary_cute_expr(expr)

    def _literal_cute_expr(
        self,
        index: int,
        *,
        for_index: bool,
        index_dim_size: int | sympy.Expr | None,
    ) -> str | None:
        if for_index and index < 0:
            if index_dim_size is None:
                return None
            index = V.graph.sizevars.guard_int(index + index_dim_size)
        return f"cutlass.Int32({index})"

    def _lane_uniform_binary_cute_expr(self, expr: torch.fx.Node) -> str | None:
        if expr.op != "call_function":
            return None
        args = expr.args
        if len(args) < 2:
            return None
        lhs = self.lane_uniform_cute_expr(args[0])
        rhs = self.lane_uniform_cute_expr(args[1])
        if lhs is None or rhs is None:
            return None
        if expr.target is torch.ops.aten.div.Tensor_mode:
            if expr.kwargs.get("rounding_mode") == "floor":
                return f"({lhs} // {rhs})"
            return None
        op = LANE_UNIFORM_BINARY_OPS.get(expr.target)
        if op is None:
            return None
        return f"({lhs} {op} {rhs})"

    def _lane_uniform_index_cute_expr(
        self,
        expr: torch.fx.Node,
        *,
        for_index: bool,
    ) -> str | None:
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
        base_code = self.lane_uniform_cute_expr(base)
        base_shape = fx_node_shape(base) if isinstance(base, torch.fx.Node) else None
        if base_code is None or not isinstance(indices, (list, tuple)):
            return None
        if base_shape is None or len(indices) != len(base_shape):
            return None
        index_codes = []
        for dim, index in enumerate(indices):
            dim_size = base_shape[dim]
            index_code = self.lane_uniform_cute_expr(
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

    def fx_mask_expr_to_sympy(self, expr: object) -> sympy.Expr | None:
        _, _, index_symbols = make_fx_index_symbols(
            self.q_idx, self.kv_idx, kv_expr=self.lane_analysis.kv_lane
        )
        return fx_aux_index_to_sympy(expr, index_symbols, self.mask_aux_load_to_symbol)

    def mask_aux_load_to_symbol(self, node: torch.fx.Node) -> sympy.Expr | None:
        if not is_aten_index_node(node):
            return None
        if node in self.aux_load_symbols:
            return self.aux_load_symbols[node]
        lane_uniform_code = self.lane_uniform_cute_expr(node)
        if lane_uniform_code is None:
            return None
        symbol = sympy.Symbol(f"mask_bound_{self.next_symbol_id}", integer=True)
        self.next_symbol_id += 1
        self.symbol_codes[symbol] = lane_uniform_code
        self.aux_load_symbols[node] = symbol
        return symbol

    def merge_intersection(
        self,
        intervals: tuple[PackedMaskInterval, ...],
        new_intervals: tuple[PackedMaskInterval, ...],
    ) -> tuple[PackedMaskInterval, ...] | None:
        if len(intervals) * len(new_intervals) > MAX_PACKED_MASK_INTERVALS:
            return None
        merged = []
        for lhs in intervals:
            for rhs in new_intervals:
                merged.append(
                    PackedMaskInterval(
                        V.graph.sizevars.simplify(Max(lhs.lower, rhs.lower)),
                        V.graph.sizevars.simplify(Min(lhs.upper, rhs.upper)),
                    )
                )
        return tuple(merged)

    def lane_comparison_to_intervals(
        self,
        lhs_expr: sympy.Expr,
        rhs_expr: sympy.Expr,
        *,
        strict: bool,
    ) -> tuple[PackedMaskInterval, ...] | None:
        diff = V.graph.sizevars.simplify(lhs_expr - rhs_expr)
        affine = decompose_affine_lane_expr(diff, self.lane_analysis.lane)
        if affine is None:
            return None
        lane_coeff, rest = affine
        if lane_coeff == 1:
            upper = V.graph.sizevars.simplify(-rest if strict else -rest + 1)
            if sympy_to_cute_index(upper, self.symbol_codes) is not None:
                return (PackedMaskInterval(sympy.Integer(0), upper),)
            return None
        if lane_coeff == -1:
            lower = V.graph.sizevars.simplify(rest + 1 if strict else rest)
            if sympy_to_cute_index(lower, self.symbol_codes) is not None:
                return (PackedMaskInterval(lower, sympy.Integer(32)),)
            return None
        return None

    def lane_equality_to_intervals(
        self, lhs_expr: sympy.Expr, rhs_expr: sympy.Expr
    ) -> tuple[PackedMaskInterval, ...] | None:
        if isinstance(lhs_expr, FloorDiv) and isinstance(rhs_expr, FloorDiv):
            intervals = self._floor_div_equality_to_intervals(lhs_expr, rhs_expr)
            if intervals is not None:
                return intervals
        diff = V.graph.sizevars.simplify(lhs_expr - rhs_expr)
        affine = decompose_affine_lane_expr(diff, self.lane_analysis.lane)
        if affine is None:
            return None
        lane_coeff, rest = affine
        if lane_coeff in (1, -1):
            lane_value = -rest if lane_coeff == 1 else rest
            lower = V.graph.sizevars.simplify(lane_value)
            upper = V.graph.sizevars.simplify(lane_value + 1)
            if (
                sympy_to_cute_index(lower, self.symbol_codes) is not None
                and sympy_to_cute_index(upper, self.symbol_codes) is not None
            ):
                return (PackedMaskInterval(lower, upper),)
            return None
        return None

    def _floor_div_equality_to_intervals(
        self, lhs_expr: FloorDiv, rhs_expr: FloorDiv
    ) -> tuple[PackedMaskInterval, ...] | None:
        lhs_base, lhs_divisor = lhs_expr.args
        rhs_base, rhs_divisor = rhs_expr.args
        if lhs_divisor != rhs_divisor:
            return None
        block_start = None
        if (
            lhs_base == self.lane_analysis.q_idx
            and rhs_base == self.lane_analysis.kv_lane
        ):
            block_start = V.graph.sizevars.simplify(
                lhs_expr * lhs_divisor - self.lane_analysis.kv_idx
            )
        elif (
            rhs_base == self.lane_analysis.q_idx
            and lhs_base == self.lane_analysis.kv_lane
        ):
            block_start = V.graph.sizevars.simplify(
                rhs_expr * rhs_divisor - self.lane_analysis.kv_idx
            )
        if block_start is None:
            return None
        lower = V.graph.sizevars.simplify(block_start)
        upper = V.graph.sizevars.simplify(block_start + lhs_divisor)
        if (
            sympy_to_cute_index(lower, self.symbol_codes) is not None
            and sympy_to_cute_index(upper, self.symbol_codes) is not None
        ):
            return (PackedMaskInterval(lower, upper),)
        return None

    def comparison_to_intervals(
        self,
        lhs: object,
        rhs: object,
        *,
        strict: bool,
    ) -> tuple[PackedMaskInterval, ...] | None:
        lhs_expr = self.fx_mask_expr_to_sympy(lhs)
        rhs_expr = self.fx_mask_expr_to_sympy(rhs)
        if lhs_expr is None or rhs_expr is None:
            return None
        return self.lane_comparison_to_intervals(lhs_expr, rhs_expr, strict=strict)

    def equality_to_intervals(
        self, lhs: object, rhs: object
    ) -> tuple[PackedMaskInterval, ...] | None:
        lhs_expr = self.fx_mask_expr_to_sympy(lhs)
        rhs_expr = self.fx_mask_expr_to_sympy(rhs)
        if lhs_expr is None or rhs_expr is None:
            return None
        return self.lane_equality_to_intervals(lhs_expr, rhs_expr)

    def node_to_intervals(
        self, node: torch.fx.Node
    ) -> tuple[PackedMaskInterval, ...] | None:
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
    ) -> tuple[PackedMaskInterval, ...] | None:
        lhs, rhs = node.args
        if not isinstance(lhs, torch.fx.Node) or not isinstance(rhs, torch.fx.Node):
            return None
        lhs_intervals = self.node_to_intervals(lhs)
        rhs_intervals = self.node_to_intervals(rhs)
        if lhs_intervals is None or rhs_intervals is None:
            return None
        if intersect:
            return self.merge_intersection(lhs_intervals, rhs_intervals)
        if len(lhs_intervals) + len(rhs_intervals) > MAX_PACKED_MASK_INTERVALS:
            return None
        return lhs_intervals + rhs_intervals


def select_packed_mask_intervals(
    graph_module: GraphModule,
) -> tuple[PackedMaskInterval, ...] | None:
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

    lane_analysis = LaneIndexAnalysis(
        q_idx=sympy.Symbol("q_idx", integer=True, nonnegative=True),
        kv_idx=sympy.Symbol("kv_idx", integer=True, nonnegative=True),
        lane=sympy.Symbol("mask_lane", integer=True, nonnegative=True),
    )
    analyzer = PackedMaskAnalyzer(
        placeholders=placeholders,
        q_idx=q_idx,
        kv_idx=kv_idx,
        lane_analysis=lane_analysis,
    )
    intervals = analyzer.node_to_intervals(output_val)
    if intervals is None:
        return None
    if len(intervals) == 1 and intervals[0].is_full():
        return None
    return tuple(
        interval.with_symbol_codes(analyzer.symbol_codes) for interval in intervals
    )
