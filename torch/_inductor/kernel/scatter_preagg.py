"""
Pre-aggregation for accumulating scatters.

Accumulating scatters (index_put(accumulate=True), index_add, scatter_add,
scatter_reduce(sum), embedding backward) normally issue one atomic per element,
and atomics on a shared destination serialize. Here each program takes BLOCK_T
"tokens" (the iteration positions the destination depends on) and

  1. computes each token's destination key from the fused index expression,
  2. estimates its distinct destinations, and issues plain atomics when
     pre-aggregating would save few,
  3. otherwise sorts (key, position) pairs, loads the fused values at the
     sorted positions and sums runs of equal keys with a segmented scan (a 0/1
     matrix product for fp16 on ROCm),
  4. and issues one atomic per (distinct key, inner element).

That leaves a hot destination one atomic per block, so small outputs taking
many writes per element give the blocks copies of the output to take turns
over, summed into it after (see _num_copies).

Dims the destination does not depend on ("inner" dims, e.g. the embedding dim)
form a second tile axis, so blocks sort one key per token, not per element.

Deterministic mode and 8/16-bit integers use a sorted, atomic-free lowering
instead. 16-bit outputs may accumulate in an fp32 copy (see _stage_in_fp32).
"""

import dataclasses
import functools
from collections.abc import Callable, Sequence
from typing import Any, cast

import sympy

import torch
from torch._dynamo.utils import counters
from torch._logging import getArtifactLogger
from torch.fx.experimental.symbolic_shapes import free_unbacked_symbols
from torch.utils._ordered_set import OrderedSet
from torch.utils._sympy.functions import FloorDiv, Min, Mod, ModularIndexing

from .. import config, ir
from ..codegen.common import CSEVariable
from ..codegen.triton import TritonSymbols
from ..heuristics.template.triton import (
    ROCmScatterPreaggConfig,
    ScatterPreaggConfig,
    ScatterPreaggTarget,
)
from ..lowering import copy_, empty_strided, full, sort_fallback, sort_stable, to_dtype
from ..ops_handler import DefaultHandler, WrapperHandler
from ..select_algorithm import (
    AlgorithmSelectorCache,
    autotune_select_algorithm,
    ModificationWrapper,
    SymbolicGridFn,
    TritonTemplate,
    TritonTemplateKernel,
)
from ..utils import (
    ceildiv,
    generate_assert,
    needs_fallback_due_to_atomic_add_limitations,
    sympy_dot,
    sympy_index_symbol,
    sympy_product,
    triton_type,
)
from ..virtualized import ops, OpsValue, V
from .mm_common import load_kernel_template


log = getArtifactLogger(__name__, "scatter_preagg")

# Output dtype -> dtype the segment sums accumulate in.
_ACC_DTYPE: dict[torch.dtype, torch.dtype] = {
    torch.float16: torch.float32,
    torch.bfloat16: torch.float32,
    torch.float32: torch.float32,
    torch.float64: torch.float64,
    torch.int64: torch.int64,
    torch.int32: torch.int32,
    torch.int16: torch.int32,
    torch.int8: torch.int32,
    torch.uint8: torch.int32,
}
# Triton has no atomics for these.
_NO_ATOMICS = (torch.int16, torch.int8, torch.uint8)

# Ops the template's subgraph codegen (ModificationWrapper) cannot express
# faithfully: its loads are unmasked and it has no iteration ranges.
_UNSUPPORTED_OPS = OrderedSet(
    [
        "masked",
        "load_seed",
        "rand",
        "randn",
        "randint64",
        "bucketize",
        "inline_asm_elementwise",
        "store",
        "store_reduction",
        "reduction",
        "scan",
        "sort",
    ]
)

IndexFn = Callable[[list[sympy.Expr]], list[sympy.Expr]]
ValueFn = Callable[[list[sympy.Expr]], Any]


def _record_skip(reason: str) -> None:
    counters["inductor"][f"scatter_preagg_skipped_{reason}"] += 1
    log.debug("skip reason=%s", reason)


def _use_sorted(dtype: torch.dtype) -> bool:
    return torch.are_deterministic_algorithms_enabled() or dtype in _NO_ATOMICS


def _size_decline(
    dtype: torch.dtype, out_sizes: Sequence[sympy.Expr], sizes: Sequence[sympy.Expr]
) -> str | None:
    """The skip reason if the template declines a scatter over ``sizes`` into an
    output of ``out_sizes`` for their sizes alone."""
    # The sorted lowering replaces the eager fallback rather than atomics, so
    # the size gates do not apply to it.
    if _use_sorted(dtype):
        return None
    numel = _hint(sympy_product(sizes))
    if numel < config.scatter_preagg_template_min_numel:
        return "too_small"
    # bf16 always stages in fp32 (see _try_template), and gets output copies
    # only at many writes per element.
    if dtype == torch.bfloat16 and _hint(sympy_product(out_sizes)) > 2 * numel:
        return "fp32_copy_too_large"
    return None


def declines_by_size(
    dtype: torch.dtype, out_sizes: Sequence[sympy.Expr], sizes: Sequence[sympy.Expr]
) -> bool:
    """
    Whether the template declines a scatter for its sizes alone, which lowerings
    may check before setting up for it. A decline records its skip reason, so
    the caller must then not offer the scatter to the template, which would
    record it again.
    """
    reason = _size_decline(dtype, out_sizes, sizes)
    if reason is not None:
        _record_skip(reason)
    return reason is not None


def can_bypass_atomic_fallback(dtype: torch.dtype) -> bool:
    """
    True when an accumulating scatter that Inductor would send to the eager
    fallback, or to atomics Triton lacks, may be offered to this template
    instead. If the template then declines, the caller must emit the eager
    fallback itself.
    """
    return dtype in _ACC_DTYPE and (
        _use_sorted(dtype) or needs_fallback_due_to_atomic_add_limitations(dtype)
    )


@dataclasses.dataclass
class _Loaded:
    """A buffer load, tracked to tell index tensors from other reads."""

    name: str


class _RecordingHandler(DefaultHandler):
    """
    Evaluates an index or value function symbolically to find which iteration
    variables it reads through, which buffers it loads, and whether it uses ops
    the template cannot generate.
    """

    def __init__(self) -> None:
        self.symbols: OrderedSet[sympy.Symbol] = OrderedSet()
        self.read_names: OrderedSet[str] = OrderedSet()
        self.unsupported: OrderedSet[str] = OrderedSet()
        # Buffers loaded straight into indirect_indexing, with the smallest
        # size they index, and buffers loaded for anything else.
        self.index_bounds: dict[str, sympy.Expr] = {}
        self.other_reads: OrderedSet[str] = OrderedSet()
        # Whether synthetic benchmark inputs could trip a bounds assert or read
        # out of bounds: a bounds-checked index, or a load address, computed
        # rather than loaded. Unchecked writes are masked instead.
        self.unsafe_to_benchmark = False
        self._computed_indices: OrderedSet[sympy.Symbol] = OrderedSet()
        self._num_indirect = 0

    def _note(self, value: Any) -> None:
        if isinstance(value, sympy.Basic):
            self.symbols |= OrderedSet(value.free_symbols)
        elif isinstance(value, _Loaded):
            self.other_reads.add(value.name)
        elif isinstance(value, (list, tuple)):
            for v in value:
                self._note(v)

    def _default(self, name: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        if name in _UNSUPPORTED_OPS:
            self.unsupported.add(name)
        self._note(args)
        self._note(tuple(kwargs.values()))
        return None

    def load(self, name: str, index: sympy.Expr) -> _Loaded:
        self.read_names.add(name)
        self._note(index)
        if not self._computed_indices.isdisjoint(sympy.sympify(index).free_symbols):
            self.unsafe_to_benchmark = True
        return _Loaded(name)

    def to_dtype(self, x: Any, dtype: torch.dtype, *args: Any, **kwargs: Any) -> Any:
        # Casting an integer load to int32/int64 keeps its values, so it still
        # indexes as the load; a cast float does not, and is a computed index.
        if (
            isinstance(x, _Loaded)
            and dtype in (torch.int32, torch.int64)
            and not V.graph.get_dtype(x.name).is_floating_point
        ):
            return x
        self._note(x)
        return None

    def frexp(self, x: Any) -> tuple[None, None]:
        self._note(x)
        return None, None

    def scan(self, dtypes: Any, combine_fn: Any, values: tuple[Any, ...]) -> Any:
        self._default("scan", (values,), {})
        return (None,) * len(values)

    def sort(
        self, dtypes: Any, values: tuple[Any, ...], stable: bool, descending: bool
    ) -> Any:
        self._default("sort", (values,), {})
        return (None,) * len(values)

    def indirect_indexing(
        self, x: Any, size: sympy.Expr, check: bool = True, wrap_neg: bool = True
    ) -> sympy.Symbol:
        size = sympy.sympify(size)
        self._note(size)
        self._num_indirect += 1
        sym = sympy_index_symbol(f"preagg_indirect{self._num_indirect}")
        if isinstance(x, _Loaded):
            bound = self.index_bounds.get(x.name)
            self.index_bounds[x.name] = size if bound is None else Min(bound, size)
        elif generate_assert(check):
            self.unsafe_to_benchmark = True
        else:
            self._computed_indices.add(sym)
        return sym


class _ValidityHandler(WrapperHandler):
    """
    Applies the indirect-index semantics of the regular Scatter codegen itself:
    negative-index wrapping, the bounds assert when requested, and (with
    record_validity) a mask of in-range indices, so unchecked out-of-range
    writes are dropped instead of landing out of bounds.
    """

    def __init__(self, inner: Any, *, record_validity: bool) -> None:
        super().__init__(inner)
        self.record_validity = record_validity
        self.valid: list[Any] = []

    def indirect_indexing(
        self, x: Any, size: sympy.Expr, check: bool = True, wrap_neg: bool = True
    ) -> sympy.Expr:
        size = sympy.sympify(size)
        check = generate_assert(check)
        if size.is_number:
            size_v = self.constant(int(size), torch.int64)
        else:
            size_v = self.index_expr(size, torch.int64)
        zero = self.constant(0, torch.int64)
        if wrap_neg:
            x = self.where(self.lt(x, zero), self.add(x, size_v), x)
        if check or self.record_validity:
            in_bounds = self.and_(self.ge(x, zero), self.lt(x, size_v))
            if check:
                self.device_assert_async(
                    in_bounds, f"index out of bounds: 0 <= index < {size}"
                )
            if self.record_validity:
                self.valid.append(in_bounds)
        return self._inner.indirect_indexing(x, size, False, False)


class _TemplateIndexHandler(_ValidityHandler):
    """_ValidityHandler for template subgraphs, which have no range trees."""

    def index_expr(self, expr: sympy.Expr, dtype: torch.dtype) -> Any:
        # The only free symbols are template-local tensors (tokens, inner ids,
        # indices), so render the expression directly.
        kernel = V.kernel
        expr = sympy.sympify(expr)
        text = kernel.index_to_str(expr)
        if expr.free_symbols:
            line = f"({text}).to({triton_type(dtype)})"
        else:
            line = f"tl.full([], {text}, {triton_type(dtype)})"
        return kernel.cse.generate(
            kernel.compute,
            line,
            dtype=dtype,
            shape=TritonSymbols.get_block_shape(expr),
        )


class _TileValueHandler(_TemplateIndexHandler):
    """
    _TemplateIndexHandler for the value tile. A float load indexed by the token
    alone (a per-token weight) comes out [rows, 1], and Triton then gives the
    whole [rows, cols] tile and its scan that load's layout, converting the
    tile both ways on every inner block, so it loads through a pointer
    broadcast to the tile. Integer loads (indices, masks) stay narrow, where
    the arithmetic on them is cheap.
    """

    def __init__(
        self, inner: Any, inner_sym: sympy.Symbol, shape: tuple[str, str]
    ) -> None:
        super().__init__(inner, record_validity=False)
        self.inner_sym = inner_sym
        self.shape = shape

    def load(self, name: str, index: sympy.Expr) -> Any:
        wrapper = cast(ModificationWrapper, self._inner)
        kernel = wrapper.kernel
        if (
            name in wrapper.fixed_inputs
            or self.inner_sym in index.free_symbols
            or not (dtype := V.graph.get_buffer(name).dtype).is_floating_point
        ):
            return wrapper.load(name, index)
        ptr = f"{wrapper._add_kernel_input(name)} + {wrapper._process_indexing(index)}"
        line = f"tl.load(tl.broadcast_to({ptr}, [{', '.join(self.shape)}]))"
        if (
            dtype in (torch.float16, torch.bfloat16)
            and config.triton.codegen_upcast_to_fp32
        ):
            line += ".to(tl.float32)"
            dtype = torch.float32
        return kernel.cse.generate(kernel.compute, line, dtype=dtype, shape=self.shape)


def _unravel(linear: sympy.Expr, sizes: list[sympy.Expr]) -> list[sympy.Expr]:
    """Row-major coordinates of ``linear`` over ``sizes``."""
    coords: list[sympy.Expr] = []
    stride = 1
    for i in reversed(range(len(sizes))):
        if sizes[i] == 1:
            coords.append(sympy.S.Zero)
        elif i == 0:
            coords.append(FloorDiv(linear, stride))
        else:
            coords.append(ModularIndexing(linear, stride, sizes[i]))
        stride *= sizes[i]
    return coords[::-1]


@dataclasses.dataclass
class _ScatterPlan:
    iter_sizes: list[sympy.Expr]
    tok_dims: list[int]
    inner_dims: list[int]
    # Output dim fed by each inner iteration dim.
    inner_out_dims: list[int]
    key_out_dims: list[int]
    out_sizes: list[sympy.Expr]
    out_indexer: Callable[[list[sympy.Expr]], sympy.Expr]
    output_indexer: IndexFn
    value_fn: ValueFn
    acc_dtype: torch.dtype
    # Buffers read only as indices, with the smallest size they index.
    index_bounds: dict[str, sympy.Expr]
    # Whether an index is computed so that synthetic benchmark inputs could trip
    # its bounds assert or read out of bounds (see _RecordingHandler).
    unsafe_to_benchmark: bool
    # Whether the last inner dim is contiguous in the output.
    inner_contiguous: bool

    @property
    def key_sizes(self) -> list[sympy.Expr]:
        return [self.out_sizes[j] for j in self.key_out_dims]

    @property
    def num_tokens(self) -> sympy.Expr:
        return sympy_product([self.iter_sizes[d] for d in self.tok_dims])

    @property
    def num_inner(self) -> sympy.Expr:
        return sympy_product([self.iter_sizes[d] for d in self.inner_dims])

    @property
    def num_keys(self) -> sympy.Expr:
        return sympy_product(self.key_sizes)

    def sizes(self) -> dict[str, sympy.Expr]:
        return {
            "NUM_TOKENS": self.num_tokens,
            "NUM_INNER": self.num_inner,
            "NUM_KEYS": self.num_keys,
        }

    def static_sizes(self) -> dict[str, int]:
        """Template constexprs for the sizes that are not symbolic."""
        return {name: int(s) for name, s in self.sizes().items() if s.is_number}

    def iteration_coords(
        self, token: sympy.Expr, inner: sympy.Expr
    ) -> list[sympy.Expr]:
        coords: list[sympy.Expr] = [sympy.S.Zero] * len(self.iter_sizes)
        for dims, linear in ((self.tok_dims, token), (self.inner_dims, inner)):
            sizes = [self.iter_sizes[d] for d in dims]
            for d, c in zip(dims, _unravel(linear, sizes)):
                coords[d] = c
        return coords

    def key_expr(self, out_coords: list[sympy.Expr]) -> sympy.Expr:
        return sympy_dot(
            [out_coords[j] for j in self.key_out_dims],
            ir.FlexibleLayout.contiguous_strides(self.key_sizes),
        )

    def address(self, key: sympy.Expr, inner: sympy.Expr) -> sympy.Expr:
        coords: list[sympy.Expr] = [sympy.S.Zero] * len(self.out_sizes)
        for j, c in zip(self.key_out_dims, _unravel(key, self.key_sizes)):
            coords[j] = c
        inner_sizes = [self.iter_sizes[d] for d in self.inner_dims]
        for j, c in zip(self.inner_out_dims, _unravel(inner, inner_sizes)):
            coords[j] = c
        return self.out_indexer(coords)


def _hint(expr: sympy.Expr) -> int:
    return V.graph.sizevars.optimization_hint(expr)


def _guarded(cond: sympy.Basic, *, guard: bool = True) -> bool:
    """
    Whether ``cond`` holds, guarding symbolic sizes on it only when the hints
    satisfy it: a guard on the false side would recompile every other size.
    Without ``guard``, or under AOT compilation (which never checks guards at
    run time), ``cond`` must already be known.
    """
    if not guard or V.graph.aot_mode or free_unbacked_symbols(cond):
        return V.graph.sizevars.statically_known_true(cond)
    hint = V.graph.sizevars.guarding_hint_or_throw
    hinted = cond.xreplace({s: hint(s) for s in cond.free_symbols})
    return bool(hinted) and V.graph.sizevars.guard_or_false(cond)


def _plain_max_d(warp_size: int) -> int:
    """Widest row of the plain path: about one warp's worth of a regular
    Inductor scatter's elements."""
    return 4 * warp_size


def _block_width(cfg: ScatterPreaggConfig, warp_size: int) -> int:
    """Upper bound on the inner block width a config's kernel loads: BLOCK_D,
    or a plain row widened by a power of two dividing D_ITERS."""
    widened = cfg.block_d * (cfg.d_iters & -cfg.d_iters)
    return max(cfg.block_d, min(widened, _plain_max_d(warp_size)))


def _inner_shift(num_inner: sympy.Expr, tile: int, width: int, *, guard: bool) -> bool:
    """
    INNER_SHIFT: whether the last inner block shifts back to stay in bounds,
    rather than clamping its indices. That needs a block of ``width`` elements
    and, for aligned vector atomics, a row size that is a multiple of 16 or of
    the ``tile`` (where the shift is a no-op). A symbolic size is guarded on
    being a multiple of 16, where _align_strides makes it pay off, and holding
    a block, so a new size recompiles only when one of these changes.
    """
    if num_inner.is_number:
        n = int(num_inner)
        return n >= width and (n % tile == 0 or n % 16 == 0)
    aligned = sympy.Eq(Mod(num_inner, 16), 0)
    if not _guarded(aligned, guard=guard):
        # Guard the other side too, or a cached unaligned graph would serve
        # aligned sizes.
        _guarded(sympy.Not(aligned), guard=guard)
        return False
    return _guarded(sympy.Ge(num_inner, width), guard=guard)


def _align_strides(plan: _ScatterPlan, nodes: list[ir.IRNode]) -> None:
    """
    Guard the symbolic row size and strides to multiples of 16 so the kernel is
    marked tt.divisibility for them; a symbolic row size otherwise hides the
    alignment vectorized loads and atomics need. Token and key counts are left
    unguarded, or each new count would recompile.
    """
    counts = (plan.num_tokens.free_symbols | plan.num_keys.free_symbols) - (
        plan.num_inner.free_symbols
    )
    symbols = OrderedSet(plan.num_inner.free_symbols)
    for node in nodes:
        for stride in node.get_layout().stride:
            symbols |= OrderedSet(sympy.sympify(stride).free_symbols)
    for s in symbols - OrderedSet(counts):
        _guarded(sympy.Eq(Mod(s, 16), 0))


class ScatterPreaggTemplateKernel(TritonTemplateKernel):
    """Template kernel exposing the scatter plan (sizes, key, value, address) to
    the Jinja template as helper functions."""

    def __init__(self, *args: Any, plan: _ScatterPlan, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.scatter_plan = plan
        self._register_extra_template_env_fns(
            self.preagg_def_kernel,
            self.preagg_sizes,
            self.preagg_key,
            self.preagg_value,
            self.preagg_addr,
        )

    def preagg_def_kernel(self, *names: str) -> str:
        """def_kernel naming the inputs OUT, ``names``, then IN0, IN1, ..."""
        rest = len(self.input_nodes) - 1 - len(names)
        code = self.def_kernel("OUT", *names, *[f"IN{i}" for i in range(rest)])
        # Results go through OUT (a mutated input); the template's own output is
        # a placeholder that autotune benchmarking still passes, so take it as
        # an unused argument unless the scheduler has removed it.
        out_name = self.output_node.get_name()
        if out_name not in V.graph.removed_buffers:
            self.args.output(out_name)
        return code

    def preagg_sizes(self) -> str:
        """Definitions of the sizes that are kernel arguments, not constexprs."""
        return "\n".join(
            f"{name} = {self.index_to_str(size)}"
            for name, size in self.scatter_plan.sizes().items()
            if not size.is_number
        )

    def _codegen_subgraph(
        self,
        prefix: str,
        fixed_inputs: dict[str, str],
        input_shapes: dict[str, tuple[str, ...]],
        emit: Callable[[], list[str]],
    ) -> str:
        num = 0
        while f"{prefix}_{num}" in self.subgraph_bodies:
            num += 1
        with self.create_subgraph_body(f"{prefix}_{num}"):
            handler = ModificationWrapper(
                self,
                0,
                fixed_inputs,
                None,
                input_shapes,
                dict.fromkeys(fixed_inputs, "index"),
            )
            with V.set_ops_handler(handler):
                lines = emit()
            self.codegen_body()
            for line in lines:
                self.body.writeline(line)
            body = self.body.getvalue()
            self.cse.invalidate(OrderedSet())
            return body

    @staticmethod
    def _symbol_of(value: Any) -> sympy.Symbol:
        return sympy_index_symbol(str(value))

    def preagg_key(
        self,
        tok: str,
        key_out: str,
        valid_out: str,
        rows: str = "BLOCK_T",
    ) -> str:
        """Destination key and validity of each token in ``tok`` (shape [rows])."""
        plan = self.scatter_plan

        def emit() -> list[str]:
            handler = V.get_ops_handler()
            tok_sym = self._symbol_of(handler.load("tok", sympy.S.Zero))
            coords = plan.iteration_coords(tok_sym, sympy.S.Zero)
            indirect = _TemplateIndexHandler(handler, record_validity=True)
            with V.set_ops_handler(indirect):
                out_coords = list(plan.output_indexer(coords))
            key = self.index_to_str(plan.key_expr(out_coords))
            valid = " & ".join(
                [f"tl.full([{rows}], 1, tl.int1)"] + [f"({v})" for v in indirect.valid]
            )
            return [
                f"{key_out} = tl.full([{rows}], 0, tl.int64) + ({key})",
                f"{valid_out} = {valid}",
            ]

        return self._codegen_subgraph(
            "preagg_key", {"tok": tok}, {"tok": (rows,)}, emit
        )

    def preagg_value(
        self,
        ptok: str,
        inner: str,
        out: str,
        rows: str = "BLOCK_T",
        cols: str = "BLOCK_D",
    ) -> str:
        """Value at tokens ``ptok`` ([rows, 1]) and inner ids ([1, cols])."""
        plan = self.scatter_plan

        def emit() -> list[str]:
            handler = V.get_ops_handler()
            tok_sym = self._symbol_of(handler.load("ptok", sympy.S.Zero))
            inner_sym = self._symbol_of(handler.load("inner", sympy.S.Zero))
            coords = plan.iteration_coords(tok_sym, inner_sym)
            with V.set_ops_handler(_TileValueHandler(handler, inner_sym, (rows, cols))):
                value = plan.value_fn(coords)
            if isinstance(value, OpsValue):
                value = value.value
            acc = triton_type(plan.acc_dtype)
            converted = f"({value}).to({acc})"
            if isinstance(value, CSEVariable) and len(value.shape or ()) == 2:
                return [f"{out} = tl.broadcast_to({converted}, [{rows}, {cols}])"]
            # Constants come out rank 1, which broadcast_to rejects.
            return [f"{out} = tl.full([{rows}, {cols}], 0, {acc}) + {converted}"]

        return self._codegen_subgraph(
            "preagg_value",
            {"ptok": ptok, "inner": inner},
            {"ptok": (rows, "1"), "inner": ("1", cols)},
            emit,
        )

    def preagg_addr(self, key: str, inner: str) -> str:
        """Offset into OUT of destination ``key`` and inner element ``inner``."""
        address = self.scatter_plan.address(
            sympy_index_symbol(key), sympy_index_symbol(inner)
        )
        return self.index_to_str(address)


@SymbolicGridFn
def scatter_preagg_grid(
    num_tokens: int, num_inner: int, meta: dict[str, Any], *, cdiv: Callable[..., Any]
) -> tuple[Any, Any, int]:
    inner_per_prog = meta["BLOCK_D"] * meta["D_ITERS"]
    return (cdiv(num_tokens, meta["BLOCK_T"]), cdiv(num_inner, inner_per_prog), 1)


class ScatterPreaggTemplate(TritonTemplate):
    kernel_type = ScatterPreaggTemplateKernel

    def __init__(self, name: str, source: str) -> None:
        super().__init__(
            name=name,
            grid=scatter_preagg_grid,
            source=source,
            always_freeze_layout=True,
        )


# Each template includes the shared macros first.
_COMMON_SOURCE = load_kernel_template("triton_scatter_preagg_common")

scatter_preagg_atomic_template = ScatterPreaggTemplate(
    "scatter_preagg", _COMMON_SOURCE + load_kernel_template("triton_scatter_preagg")
)

# The sorted lowering sums each BLOCK_T chunk in a fixed order. Segments that
# cross chunk boundaries go to PART for the merge kernel.
scatter_sorted_chunk_template = ScatterPreaggTemplate(
    "scatter_sorted_chunk",
    _COMMON_SOURCE + load_kernel_template("triton_scatter_sorted_chunk"),
)
scatter_sorted_merge_template = ScatterPreaggTemplate(
    "scatter_sorted_merge",
    _COMMON_SOURCE + load_kernel_template("triton_scatter_sorted_merge"),
)


def _analyze(
    out_layout: ir.Layout,
    iter_sizes: list[sympy.Expr],
    value_fn: ValueFn,
    output_indexer: IndexFn,
    acc_dtype: torch.dtype,
) -> tuple[_ScatterPlan, _RecordingHandler, _RecordingHandler] | str:
    """
    Split the iteration dims into token and inner dims. Returns the plan and the
    reads and ops recorded from the key and value functions, or a skip reason.
    """
    iter_syms = [
        sympy.Symbol(f"preagg_i{d}", integer=True, nonnegative=True)
        for d in range(len(iter_sizes))
    ]
    key_rec = _RecordingHandler()
    with V.set_ops_handler(key_rec):
        out_coords = [sympy.sympify(c) for c in output_indexer(iter_syms)]
    value_rec = _RecordingHandler()
    with V.set_ops_handler(value_rec):
        value_fn(iter_syms)
    out_sizes = [sympy.sympify(s) for s in out_layout.size]
    if len(out_coords) != len(out_sizes):
        return "rank_mismatch"

    tok_dims: list[int] = []
    inner_dims: list[int] = []
    inner_out_dims: list[int] = []
    for d, sym in enumerate(iter_syms):
        uses = [j for j, c in enumerate(out_coords) if sym in c.free_symbols]
        if sym not in key_rec.symbols and len(uses) == 1 and out_coords[uses[0]] == sym:
            inner_dims.append(d)
            inner_out_dims.append(uses[0])
        else:
            tok_dims.append(d)

    index_bounds: dict[str, sympy.Expr] = {}
    other_reads = key_rec.other_reads | value_rec.other_reads
    for rec in (key_rec, value_rec):
        for name, bound in rec.index_bounds.items():
            if name not in other_reads:
                prev = index_bounds.get(name)
                index_bounds[name] = bound if prev is None else Min(prev, bound)
    plan = _ScatterPlan(
        iter_sizes=iter_sizes,
        tok_dims=tok_dims,
        inner_dims=inner_dims,
        inner_out_dims=inner_out_dims,
        key_out_dims=[j for j in range(len(out_sizes)) if j not in inner_out_dims],
        out_sizes=out_sizes,
        out_indexer=out_layout.make_indexer(),
        output_indexer=output_indexer,
        value_fn=value_fn,
        acc_dtype=acc_dtype,
        index_bounds=index_bounds,
        unsafe_to_benchmark=(
            key_rec.unsafe_to_benchmark or value_rec.unsafe_to_benchmark
        ),
        inner_contiguous=not inner_out_dims
        or V.graph.sizevars.statically_known_equals(
            out_layout.stride[inner_out_dims[-1]], 1
        ),
    )
    return plan, key_rec, value_rec


def _realize(node: ir.TensorBox) -> ir.TensorBox:
    node.realize()
    ir.as_storage_and_layout(node, freeze=True)
    return node


def _materialize_values(
    value_fn: ValueFn, sizes: list[sympy.Expr], device: torch.device, dtype: torch.dtype
) -> ValueFn:
    """
    Compute the values into their own buffer, before the scatter mutates the
    output, and return a loader of it.
    """
    values = ir.Pointwise.create(
        device=device,
        dtype=dtype,
        inner_fn=lambda idx: ops.to_dtype(value_fn(idx), dtype),
        ranges=sizes,
    )
    return _realize(values).make_loader()


def _key_value(plan: _ScatterPlan, token: sympy.Expr) -> Any:
    """
    int64 destination key of ``token`` in a regular kernel, NUM_KEYS for a
    dropped write (unchecked and out of range).
    """
    coords = plan.iteration_coords(token, sympy.S.Zero)
    handler = _ValidityHandler(V.get_ops_handler(), record_validity=True)
    with V.set_ops_handler(handler):
        out_coords = plan.output_indexer(coords)
    key = ops.index_expr(plan.key_expr(out_coords), torch.int64)
    if handler.valid:
        valid = functools.reduce(ops.and_, handler.valid)
        key = ops.where(valid, key, ops.index_expr(plan.num_keys, torch.int64))
    return key


def _keys(plan: _ScatterPlan, device: torch.device) -> ir.TensorBox:
    return ir.Pointwise.create(
        device=device,
        dtype=torch.int64,
        inner_fn=lambda idx: _key_value(plan, idx[0]),
        ranges=[plan.num_tokens],
    )


def _materialize_keys(plan: _ScatterPlan, device: torch.device) -> IndexFn:
    """
    Compute the destination keys into their own buffer, before the scatter
    mutates the output, and return an output indexer that reads them.
    """
    load_key = _realize(_keys(plan, device)).make_loader()

    def output_indexer(coords: list[sympy.Expr]) -> list[sympy.Expr]:
        token: sympy.Expr = sympy.S.Zero
        for d in plan.tok_dims:
            token = token * plan.iter_sizes[d] + coords[d]
        key = ops.indirect_indexing(
            load_key([token]), plan.num_keys, check=False, wrap_neg=False
        )
        out: list[sympy.Expr] = [sympy.S.Zero] * len(plan.out_sizes)
        for j, c in zip(plan.key_out_dims, _unravel(key, plan.key_sizes)):
            out[j] = c
        for d, j in zip(plan.inner_dims, plan.inner_out_dims):
            out[j] = coords[d]
        return out

    return output_indexer


def _plan(
    out_layout: ir.Layout,
    out_name: str,
    sizes: list[sympy.Expr],
    value_fn: ValueFn,
    output_indexer: IndexFn,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[_ScatterPlan, OrderedSet[str]] | str:
    """
    The scatter's plan and the buffers the template reads. Index and value
    functions the template cannot inline (unsupported ops, or reads of the
    output it mutates) are computed into buffers first.
    """

    def analyze() -> tuple[_ScatterPlan, _RecordingHandler, _RecordingHandler] | str:
        return _analyze(out_layout, sizes, value_fn, output_indexer, _ACC_DTYPE[dtype])

    analyzed = analyze()
    if isinstance(analyzed, str):
        return analyzed
    plan, key_rec, value_rec = analyzed
    if value_rec.unsupported or out_name in value_rec.read_names:
        value_fn = _materialize_values(value_fn, sizes, device, dtype)
        counters["inductor"]["scatter_preagg_materialized_values"] += 1
    if key_rec.unsupported or out_name in key_rec.read_names:
        output_indexer = _materialize_keys(plan, device)
        counters["inductor"]["scatter_preagg_materialized_keys"] += 1
    if value_fn is not plan.value_fn or output_indexer is not plan.output_indexer:
        analyzed = analyze()
        if isinstance(analyzed, str):
            return analyzed
        plan, key_rec, value_rec = analyzed
    if key_rec.unsupported or value_rec.unsupported:
        return "unsupported_op"
    reads = key_rec.read_names | value_rec.read_names
    if out_name in reads:
        return "reads_output"
    return plan, reads


def maybe_preaggregated_scatter(
    out: ir.TensorBox,
    value_fn: ValueFn,
    iter_sizes: Sequence[sympy.Expr],
    output_indexer: IndexFn,
    inputs: list[Any],
) -> bool:
    """
    Lower ``out[output_indexer(i)] += value_fn(i)`` over ``iter_sizes`` with the
    pre-aggregating template. ``inputs`` are the IR nodes value_fn and
    output_indexer read. Returns False if the template declines, for the caller
    to lower the scatter. Planning freezes the layouts of the buffers read and
    may compute buffers the scheduler removes when unused. Size guards and the
    fp32 copy come only after the declines that need a plan, so only a codegen
    failure leaves guards behind.
    """
    read_names: OrderedSet[str] = OrderedSet()
    for x in inputs:
        if isinstance(x, ir.IRNode):
            read_names |= x.get_read_names()
            # A realized input reports the reads of its producer, not itself.
            if (name := ir.try_get_name(x)) is not None:
                read_names.add(name)
    sizes = [sympy.sympify(s) for s in iter_sizes]
    if _try_template(out, value_fn, sizes, output_indexer, read_names):
        return True
    log.debug(
        "declined scatter into %s, output sizes=%s, sizes=%s",
        ir.try_get_name(out),
        out.get_size(),
        sizes,
    )
    return False


def _output_storage(out: ir.TensorBox) -> tuple[ir.StorageBox, ir.Layout] | None:
    """Storage and layout a (possibly view) output is scattered through."""
    try:
        view = out.data
        if isinstance(view, ir.BaseView):
            view = ir.ExternKernel.convert_to_reinterpret_view(view)
        storage, layout = ir.as_storage_and_layout(view, freeze=True)
    except NotImplementedError:
        return None
    if not isinstance(storage.data, ir.Buffer):
        return None
    return storage, layout


def _scatter_into_copy(
    reason: str,
    out: ir.TensorBox,
    dtype: torch.dtype,
    value_fn: ValueFn,
    sizes: list[sympy.Expr],
    output_indexer: IndexFn,
    read_names: OrderedSet[str],
) -> bool:
    """Scatter into a contiguous ``dtype`` copy of ``out``, then copy it back."""
    tmp = _realize(to_dtype(out, dtype, copy=True))
    if _output_storage(tmp) is None or not _try_template(
        tmp, value_fn, sizes, output_indexer, read_names
    ):
        return False
    copy_(out, tmp)
    counters["inductor"][f"scatter_preagg_{reason}"] += 1
    return True


def _stage_in_fp32(
    dtype: torch.dtype,
    num_keys: sympy.Expr,
    matrix_preagg: bool,
    target: ScatterPreaggTarget,
) -> bool:
    """
    Whether to accumulate a 16-bit output in an fp32 copy rather than with
    16-bit atomics, which round every block's partial sum into the output.
    """
    if dtype == torch.bfloat16:
        # bf16 atomics are compare-and-swap loops: slower and less accurate
        # than the copy at any contention.
        return True
    # Plain blocks would issue twice the atomics in fp32 as packed fp16, so fp16
    # is staged in fp32 only with few destinations, where blocks mostly sort
    # (smaller blocks of small scatters are no slower staged). The matrix
    # product needs fp16 atomics.
    return (
        dtype == torch.float16
        and not matrix_preagg
        and num_keys.is_number
        and int(num_keys) <= target.always_sorted_keys
    )


def _try_template(
    out: ir.TensorBox,
    value_fn: ValueFn,
    sizes: list[sympy.Expr],
    output_indexer: IndexFn,
    read_names: OrderedSet[str],
) -> bool:
    dtype = out.get_dtype()
    if dtype not in _ACC_DTYPE:
        _record_skip("dtype")
        return False
    orig_out = out
    use_sorted = _use_sorted(dtype)
    if (reason := _size_decline(dtype, out.get_size(), sizes)) is not None:
        _record_skip(reason)
        return False
    device = out.get_device_or_error()
    read_bufs = [
        buf for name in read_names if (buf := V.graph.try_get_buffer(name)) is not None
    ]
    # The template loads every input through a pointer, which a CPU scalar is not.
    if any(buf.get_device() != device for buf in read_bufs):
        _record_skip("input_device")
        return False
    storage = _output_storage(out)
    if storage is None:
        return _scatter_into_copy(
            "output_copy", out, dtype, value_fn, sizes, output_indexer, read_names
        )
    out_storage, out_layout = storage
    out_buf = out_storage.data
    if not out_layout.size or 0 in [_hint(s) for s in (*out_layout.size, *sizes)]:
        _record_skip("empty")
        return False
    # Tracing the loaders indexes any realized input directly, which needs a
    # fixed layout.
    for buf in read_bufs:
        buf.freeze_layout()
    planned = _plan(
        out_layout, out_buf.get_name(), sizes, value_fn, output_indexer, device, dtype
    )
    if isinstance(planned, str):
        _record_skip(planned)
        return False
    plan, reads = planned

    num_tokens, num_inner, num_keys = (
        _hint(plan.num_tokens),
        _hint(plan.num_inner),
        _hint(plan.num_keys),
    )
    target = V.choices.get_scatter_preagg_target(device)
    matrix_preagg = target.matrix_preagg(dtype, num_inner, plan.inner_contiguous)
    stage = (
        _stage_in_fp32(dtype, plan.num_keys, matrix_preagg, target) and not use_sorted
    )
    copy_dtype = torch.float32 if stage else dtype
    copies = (
        1
        if use_sorted
        else _num_copies(out_layout, copy_dtype, num_tokens, num_inner, target)
    )
    inputs: list[ir.IRNode] = []
    for name in reads:
        buf = V.graph.try_get_buffer(name)
        if buf is None:
            _record_skip("unknown_input")
            return False
        inputs.append(buf)

    configs = V.choices.get_scatter_preagg_configs(
        num_tokens,
        num_inner,
        copy_dtype,
        device_type=device.type,
        target=target,
        inner_contiguous=plan.inner_contiguous,
        num_keys=num_keys,
        # Deterministic results need the same config on every compile.
        max_autotune=config.max_autotune
        and not use_sorted
        and not plan.unsafe_to_benchmark,
    )
    if not use_sorted:
        # The atomic template packs (key, position) into one int64.
        key_bits = (num_keys + 1).bit_length()
        configs = [c for c in configs if key_bits + c.block_t.bit_length() - 1 <= 62]
        if not configs:
            _record_skip("key_space")
            return False

    if stage and copies == 1:
        # The copy costs passes over (and twice the bytes of) the output, which
        # outweigh the regular lowering on outputs over twice the scatter's
        # elements (declines_by_size predicts this for bf16).
        if _hint(sympy_product(out_layout.size)) > 2 * num_tokens * num_inner:
            _record_skip("fp32_copy_too_large")
            return False
        return _scatter_into_copy(
            "fp32_copy",
            out,
            torch.float32,
            plan.value_fn,
            sizes,
            plan.output_indexer,
            read_names,
        )

    for buf in inputs:
        buf.freeze_layout()

    # The kernel inlines value_fn and output_indexer through the plan; the
    # subgraph wraps the value function so index-width selection sees its reads.
    value_subgraph = ir.ComputedBuffer(
        name=None,
        layout=ir.FlexibleLayout(device=device, dtype=dtype, size=sizes),
        data=ir.Pointwise(
            device=device, dtype=dtype, inner_fn=plan.value_fn, ranges=sizes
        ),
    )
    out = ir.TensorBox(out_storage)
    # Only the default (first) config guards symbolic sizes; the other
    # autotuning candidates use what those guards establish.
    tile = configs[0].block_d * configs[0].d_iters
    # The sorted kernels load BLOCK_D-wide blocks only.
    width = (
        configs[0].block_d if use_sorted else _block_width(configs[0], target.warp_size)
    )
    if not plan.num_inner.is_number and _inner_shift(
        plan.num_inner, tile, width, guard=True
    ):
        _align_strides(plan, [out, *inputs])

    if use_sorted:
        lowered = _lower_sorted(out, plan, value_subgraph, inputs, configs)
    elif copies > 1:
        lowered = _lower_into_copies(
            orig_out,
            out_layout,
            copies,
            copy_dtype,
            plan,
            value_subgraph,
            inputs,
            configs,
            target,
        )
    else:
        lowered = _lower_atomic(out, plan, value_subgraph, inputs, configs, target)
    if not lowered:
        _record_skip("codegen_failed")
        return False
    if stage:
        # The copies accumulate in fp32 instead.
        counters["inductor"]["scatter_preagg_fp32_copy"] += 1
    counters["inductor"]["scatter_preagg_applied"] += 1
    if use_sorted:
        counters["inductor"]["scatter_preagg_sorted"] += 1
    log.debug(
        "applied to %s tokens=%d inner=%d keys=%d sorted=%s",
        out_buf.get_name(),
        num_tokens,
        num_inner,
        num_keys,
        use_sorted,
    )
    return True


def _select(
    name: str,
    choices: list[Any],
    input_nodes: list[ir.IRNode],
    layout: ir.Layout,
    **kwargs: Any,
) -> None:
    # MultiTemplateBuffer (deferred selection) does not track mutated inputs, so
    # the in-place result would be dropped.
    node, _ = autotune_select_algorithm(
        name, choices, input_nodes, layout, return_multi_template=False, **kwargs
    )
    template_buf = node.data.data
    if not isinstance(template_buf, ir.TemplateBuffer):
        raise AssertionError(f"expected a template buffer, got {template_buf}")
    # The template never stores to its own output, a placeholder that a consumer
    # of the same size could otherwise fuse onto, losing its code.
    template_buf.allow_epilogue_fusion = False


def _placeholder_layout(
    out: ir.TensorBox, plan: _ScatterPlan, configs: list[ScatterPreaggConfig]
) -> ir.FixedLayout:
    """
    The template's unused output: one element, sized as the iteration space
    (padded by a token block) so the kernel's index width covers it.
    """
    max_block_t = max(cfg.block_t for cfg in configs)
    return ir.FixedLayout(
        device=out.get_device_or_error(),
        dtype=out.get_dtype(),
        size=[(plan.num_tokens + max_block_t) * plan.num_inner],
        stride=[0],
    )


def _index_gen_fns(
    plan: _ScatterPlan, input_nodes: list[ir.IRNode]
) -> dict[int, Callable[[ir.Buffer], torch.Tensor]]:
    """
    Autotuning benchmarks integer inputs as zeros, which sends every write to
    one destination. Configs that win there lose on spread indices and vice
    versa, so give index tensors both: half spread uniformly over the size
    they index, half on one destination.
    """

    def mixed(bound: int, node: ir.Buffer) -> torch.Tensor:
        t = AlgorithmSelectorCache.benchmark_example_value(node)
        # Fill the storage, not the view: an expanded index overlaps itself.
        storage = t.new_empty(0).set_(t.untyped_storage())
        storage[: storage.numel() // 2].random_(0, bound)
        return t

    gen_fns: dict[int, Callable[[ir.Buffer], torch.Tensor]] = {}
    for i, node in enumerate(input_nodes):
        dtype = node.get_dtype()
        size = plan.index_bounds.get(node.get_name())
        if size is None or dtype.is_floating_point or dtype == torch.bool:
            continue
        bound = min(_hint(size), torch.iinfo(dtype).max + 1)
        gen_fns[i] = functools.partial(mixed, bound)
    return gen_fns


def _num_copies(
    out_layout: ir.Layout,
    dtype: torch.dtype,
    num_tokens: int,
    num_inner: int,
    target: ScatterPreaggTarget,
) -> int:
    """
    Copies of the output for the atomics to take turns over. Pre-aggregation
    leaves a hot destination one atomic per block, which still serializes over
    thousands of blocks; copies split that, for zeroing and summing them.
    """
    storage = sympy.sympify(out_layout.storage_size())
    min_tokens = (
        target.min_copy_tokens_single_column
        if num_inner == 1
        else target.min_copy_tokens
    )
    if not storage.is_number or num_tokens < min_tokens:
        return 1
    storage = int(storage)
    copies = min(
        target.max_copies,
        num_tokens * num_inner // (storage * target.writes_per_copy),
        target.max_copy_bytes // (storage * dtype.itemsize),
    )
    return 1 << (copies.bit_length() - 1) if copies >= 2 else 1


def _lower_into_copies(
    out: ir.TensorBox,
    out_layout: ir.Layout,
    copies: int,
    dtype: torch.dtype,
    plan: _ScatterPlan,
    value_subgraph: ir.ComputedBuffer,
    inputs: list[ir.IRNode],
    configs: list[ScatterPreaggConfig],
    target: ScatterPreaggTarget,
) -> bool:
    """
    Scatter into ``copies`` ``dtype`` copies of the output storage, then add
    their sum to the output.
    """
    device = out.get_device_or_error()
    storage = int(out_layout.storage_size())
    # -0.0 is the additive identity, so untouched elements keep their sign.
    tmp = _realize(full([copies, storage], -0.0, dtype=dtype, device=device))
    tmp_storage = _output_storage(tmp)
    if tmp_storage is None:
        return False
    tmp_buf, tmp_layout = tmp_storage
    # Freezing the layout may pad the copies' stride past the storage size.
    if not _lower_atomic(
        ir.TensorBox(tmp_buf),
        plan,
        value_subgraph,
        inputs,
        configs,
        target,
        copies=copies,
        copy_numel=int(tmp_layout.stride[0]),
    ):
        return False
    load_out, load_tmp = out.make_loader(), tmp.make_loader()
    offset = out_layout.make_indexer()

    def summed(idx: Sequence[sympy.Expr]) -> Any:
        acc = ops.to_dtype(load_out(idx), dtype)
        for c in range(copies):
            acc = ops.add(acc, load_tmp([sympy.Integer(c), offset(idx)]))
        return ops.to_dtype(acc, out.get_dtype())

    total = ir.Pointwise.create(
        device=device, dtype=out.get_dtype(), inner_fn=summed, ranges=out.get_size()
    )
    copy_(out, total)
    counters["inductor"]["scatter_preagg_copies"] += 1
    return True


def _lower_atomic(
    out: ir.TensorBox,
    plan: _ScatterPlan,
    value_subgraph: ir.ComputedBuffer,
    inputs: list[ir.IRNode],
    configs: list[ScatterPreaggConfig],
    target: ScatterPreaggTarget,
    *,
    copies: int = 1,
    copy_numel: int = 0,
) -> bool:
    dtype = out.get_dtype()
    num_inner, num_keys = _hint(plan.num_inner), _hint(plan.num_keys)
    matrix_preagg = target.matrix_preagg(dtype, num_inner, plan.inner_contiguous)
    plain_max_d = _plain_max_d(target.warp_size)
    # Outputs of 2GB or more (64-bit addresses) keep the sort path's atomic
    # width for plain atomics too, to save registers.
    atomic_vec = target.atomic_vec
    plain_vec = atomic_vec
    out_bytes = _hint(out.get_layout().storage_size()) * dtype.itemsize
    if target.plain_atomic_bytes and out_bytes < 2**31:
        plain_vec = max(1, target.plain_atomic_bytes // dtype.itemsize)

    min_saved = config.scatter_preagg_template_min_saved

    def adaptive_kwargs(
        cfg: ScatterPreaggConfig, inner_shift: bool, dot: bool
    ) -> dict[str, Any]:
        block_min_saved = min_saved
        if dot:
            # The product costs little more than plain atomics, so it pays at
            # half the sort's saving, and skips counting when uniform keys
            # would already save enough.
            block_min_saved /= 2
            distinct = num_keys * (1 - (1 - 1 / num_keys) ** cfg.block_t)
        elif plan.num_keys.is_number:
            # With fewer keys than tokens per block, full blocks always save
            # enough atomics to sort.
            distinct = int(plan.num_keys)
        else:
            distinct = cfg.block_t
        if distinct <= cfg.block_t * (1 - block_min_saved):
            return {"ADAPTIVE": False}
        hist_bins = min(2 * cfg.block_t, 2048)
        # Linear counting assumes hashed keys collide at random, but few keys
        # hash almost one to a bin and would be overcounted, so they get a bin
        # each instead.
        hist_exact = plan.num_keys.is_number and int(plan.num_keys) <= hist_bins
        if hist_exact:
            hist_bins = 1 << (int(plan.num_keys) - 1).bit_length()
        adaptive = {
            "ADAPTIVE": True,
            # Single columns' plain atomics cost more against their sort,
            # which then pays at half the saving.
            "SORT_SAVED": block_min_saved / (2 if cfg.block_d == 1 else 1),
            # A destination taking 1/32 of the block is hot enough to
            # pre-aggregate. The sort's floor sits above the bin counts
            # uniform keys reach; the product counts keys exactly.
            "HOT_COUNT": max(2 if dot else 8, cfg.block_t // 32),
        }
        if not dot:
            # The product counts keys exactly from its sort; the others
            # estimate them over a histogram.
            adaptive |= {
                "HIST_BINS": hist_bins,
                "HIST_EXACT": hist_exact,
                # Recurring in 1/64 of every block takes a key thousands of
                # plain atomics, and too few for its bin to stand out from
                # collisions.
                "TOP_COUNT": max(4, cfg.block_t // 64),
                "LOG_HIST_BINS": hist_bins.bit_length() - 1,
            }
        if not plan.inner_contiguous:
            # Strided inner elements take a transaction each, so plain blocks
            # run one column at a time with lanes over tokens.
            return adaptive | {
                "PLAIN_VEC": 0,
                "PLAIN_T": cfg.block_t,
                "PLAIN_D": 1,
                "PLAIN_ITERS": cfg.d_iters * cfg.block_d,
            }
        # Widen plain rows by a power of two dividing D_ITERS, up to
        # plain_max_d. Chunks under 128 tokens pay only on rows 128 or wider
        # (32 tokens for the matrix product's short blocks).
        widen = 1
        if inner_shift:
            widen = cfg.d_iters & -cfg.d_iters
            min_chunk = 32 if dot else 1

            def too_wide(widen: int) -> bool:
                rows, chunk = cfg.block_d * widen, cfg.block_t // widen
                return (
                    rows > plain_max_d
                    or chunk < min_chunk
                    or (not dot and rows < 128 and chunk < 128)
                )

            while widen > 1 and too_wide(widen):
                widen //= 2
        plain_d = cfg.block_d * widen
        return adaptive | {
            "PLAIN_VEC": 0 if plain_d < plain_vec else plain_vec,
            "PLAIN_T": cfg.block_t // widen,
            "PLAIN_D": plain_d,
            "PLAIN_ITERS": cfg.d_iters // widen,
        }

    layout = _placeholder_layout(out, plan, configs)
    choices: list[Any] = []
    for i, cfg in enumerate(configs):
        guard = i == 0
        log_block_t = cfg.block_t.bit_length() - 1
        int32_keys = _guarded(
            sympy.Lt(plan.num_keys + 1, 2 ** (31 - log_block_t)), guard=guard
        )
        # The [BLOCK_T, BLOCK_T] matrix outgrows registers past 128 tokens.
        dot = matrix_preagg and cfg.block_d >= 32 and 16 <= cfg.block_t <= 128
        inner_shift = _inner_shift(
            plan.num_inner,
            cfg.block_d * cfg.d_iters,
            _block_width(cfg, target.warp_size),
            guard=guard,
        )
        adaptive = adaptive_kwargs(cfg, inner_shift, dot)
        waves_per_eu = (
            cfg.waves_per_eu if isinstance(cfg, ROCmScatterPreaggConfig) else 0
        )
        scatter_preagg_atomic_template.maybe_append_choice(
            choices,
            input_nodes=(out, *inputs),
            layout=layout,
            num_stages=cfg.num_stages,
            num_warps=cfg.num_warps,
            **({"waves_per_eu": waves_per_eu} if waves_per_eu else {}),
            subgraphs=[value_subgraph],
            kernel_kwargs={"plan": plan},
            mutated_inputs=[out],
            call_sizes=[plan.num_tokens, plan.num_inner],
            BLOCK_T=cfg.block_t,
            BLOCK_D=cfg.block_d,
            D_ITERS=cfg.d_iters,
            LOG_BLOCK_T=log_block_t,
            KEY_DTYPE="tl.int32" if int32_keys else "tl.int64",
            INNER_SHIFT=inner_shift,
            OUT_DTYPE=triton_type(dtype),
            DOT=dot,
            COPIES=copies,
            COPY_NUMEL=copy_numel,
            ATOMIC_VEC=0 if cfg.block_d < atomic_vec else atomic_vec,
            # The shifted token block needs one full block; the clamped one
            # handles any size, so symbolic token counts take it unguarded.
            TOK_SHIFT=V.graph.sizevars.statically_known_geq(
                plan.num_tokens, cfg.block_t
            ),
            **adaptive,
            **plan.static_sizes(),
        )
    if not choices:
        return False
    input_nodes = [out, *inputs]
    _select(
        "scatter_preagg",
        choices,
        input_nodes,
        layout,
        input_gen_fns=_index_gen_fns(plan, input_nodes),
    )
    return True


def _lower_sorted(
    out: ir.TensorBox,
    plan: _ScatterPlan,
    value_subgraph: ir.ComputedBuffer,
    inputs: list[ir.IRNode],
    configs: list[ScatterPreaggConfig],
) -> bool:
    """
    Atomic-free lowering: sort the tokens by destination, then sum each
    destination's run in a fixed order (see scatter_sorted_chunk_template).
    """
    cfg = configs[0]
    device = out.get_device_or_error()
    # The sort lowering guards a symbolic length on the range of its indices.
    sort = sort_stable if plan.num_tokens.is_number else sort_fallback
    sorted_keys, perm = sort(_keys(plan, device), stable=True)
    num_chunks = ceildiv(plan.num_tokens, cfg.block_t)
    part = empty_strided(
        [2 * num_chunks * plan.num_inner], [1], dtype=plan.acc_dtype, device=device
    )
    sort_nodes = [_realize(x) for x in (part, sorted_keys, perm)]

    layout = _placeholder_layout(out, plan, configs)
    stages = [
        ("scatter_sorted_chunk", scatter_sorted_chunk_template, [out, part], inputs),
        ("scatter_sorted_merge", scatter_sorted_merge_template, [out], []),
    ]
    # Generate both kernels before emitting either.
    selections = []
    for name, template, mutated, extra in stages:
        choices: list[Any] = []
        input_nodes = [out, *sort_nodes, *extra]
        template.maybe_append_choice(
            choices,
            input_nodes=input_nodes,
            layout=layout,
            num_stages=cfg.num_stages,
            num_warps=cfg.num_warps,
            subgraphs=[value_subgraph],
            kernel_kwargs={"plan": plan},
            mutated_inputs=mutated,
            call_sizes=[plan.num_tokens, plan.num_inner],
            BLOCK_T=cfg.block_t,
            BLOCK_D=cfg.block_d,
            D_ITERS=cfg.d_iters,
            INNER_SHIFT=_inner_shift(
                plan.num_inner, cfg.block_d * cfg.d_iters, cfg.block_d, guard=True
            ),
            OUT_DTYPE=triton_type(out.get_dtype()),
            **plan.static_sizes(),
        )
        if not choices:
            return False
        selections.append((name, choices, input_nodes))
    for name, choices, input_nodes in selections:
        _select(name, choices, input_nodes, layout)
    return True
