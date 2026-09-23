"""Renders Inductor IR as text for the ``*_inductor_ir`` trace artifacts.

The output describes iteration spaces and per-point bodies. It can be rendered
at more than one point in the pipeline, and says which one in its header,
because what it shows is not the same at every point: the body of a node is
rewritten as Inductor picks a loop order for it. Reading the dump therefore
means knowing what a loop order is and how far along that choice is, so both
are spelled out below.

How to print it
---------------
Three stages are exposed as separate ``TORCH_LOGS`` artifacts, all off by
default::

    TORCH_LOGS = +post_lowering_inductor_ir  # as lowered, before any scheduling
    TORCH_LOGS = +post_scheduler_inductor_ir  # scheduler nodes built, before fusion
    TORCH_LOGS = +post_fusion_inductor_ir  # after fusion, final kernels

They compose, so to see a graph at all three boundaries at once::

    TORCH_LOGS="+post_lowering_inductor_ir,+post_scheduler_inductor_ir,+post_fusion_inductor_ir" python my_model.py

What each stage shows, and why they differ:

* **post_lowering** -- buffers exist with the loop order lowering happened to
  produce. No scheduler nodes, so no kernels and no fusion.
* **post_scheduler** -- ``simplify_and_reorder`` has run, so bodies may be
  reordered and merged relative to post_lowering. Still one node per kernel;
  fusion has not happened.
* **post_fusion** -- nodes are grouped into the kernels that will actually be
  launched. Only here do ordinary SIMD kernels print a single selected
  iteration domain in kernel coordinates (``x``/``r``); the earlier stages are
  node-local and use ``p0``/``p1``.

Comparing post_lowering against post_scheduler is how you see dimension
reordering and merging; comparing post_scheduler against post_fusion is how you
see what fused and which buffers became kernel-internal.

To render a graph directly instead of through logging, call
``format_post_lowering_ir(graph)``; it detects which stage the graph is in.

What "loop order" means here
----------------------------
Take a contiguous ``[32, 64]`` tensor and double it. Two ways to write the same
work, differing only in which dim is the inner loop::

    foreach p0 in [0,32), p1 in [0,64):     # A: p1 walks dim1
        out[64*p0 + p1] = 2 * x[64*p0 + p1]

    foreach p0 in [0,64), p1 in [0,32):     # B: p1 walks dim0
        out[64*p1 + p0] = 2 * x[64*p1 + p0]

Same elements written, different traversal. Both are correct. A is faster: its
innermost variable advances one element at a time, so consecutive points touch
consecutive memory, which every backend prefers. B strides by 64 and wastes most of
each fetch.

Nothing anywhere records "A" or "B". The choice is visible only as:

* the order of the range lists -- ``_sizes`` is ``([32, 64], [])`` for A and
  ``([64, 32], [])`` for B; position is nesting order, last entry is innermost
* the index expressions in the body -- ``64*p0 + p1`` for A, ``64*p1 + p0`` for B

That is why changing the order means retracing the body: the arithmetic itself has
to be rewritten. There is no permutation stored on the side to flip.

Backends then read the ranges positionally: the last iter range is the one that
varies fastest, and each backend maps that onto whatever its own fastest axis is --
a grid dimension for the tiled backends, the innermost loop for the C++ one. This
artifact stays backend-agnostic and describes only the ranges and the index
expressions, since that is all the IR holds; how they are mapped is a backend's
business.

Why the stages are not "decided" vs "undecided"
-----------------------------------------------
Inductor can codegen from whatever order it currently holds; no order is invalid,
just more or less optimized. ``simplify_and_reorder`` is what optimizes it, and it
does two independent things, each of which can be turned off:

* reorder dims so small strides end up innermost.
* merge adjacent dims when the access pattern lets you: ``[32, 64]`` with index
  ``64*p0 + p1`` collapses to ``[2048]`` with index ``p0``.

So the stages differ in how optimized the order is, not in whether it exists:

* post-lowering -- the as-lowered order; no stride analysis has run.
* post scheduler-node construction -- reordered as above; on GPU the dims are not
  merged yet, so this often looks identical to the previous stage.
* during fusion -- changed less often, but still changed: ``apply_new_loop_order``
  re-derives a common order when two nodes have to share a kernel.
* after ``Scheduler.merge_loops``, which runs once fusion is done -- dims collapsed
  where possible, indices rewritten accordingly.
"""

from __future__ import annotations

import dataclasses
import itertools
import logging
import operator
import re
from typing import Any, TYPE_CHECKING

import sympy
import torch
from torch._inductor import ir
from torch._inductor.virtualized import V
from torch._logging import getArtifactLogger


if TYPE_CHECKING:
    from collections.abc import Sequence

    from torch._inductor.graph import GraphLowering


post_lowering_inductor_ir_log = getArtifactLogger(__name__, "post_lowering_inductor_ir")
post_scheduler_inductor_ir_log = getArtifactLogger(
    __name__, "post_scheduler_inductor_ir"
)
post_fusion_inductor_ir_log = getArtifactLogger(__name__, "post_fusion_inductor_ir")


_REDUCTION_FIELDS = {
    "welford_reduce": ("mean", "m2", "weight"),
    "welford_combine": ("mean", "m2", "weight"),
    "online_softmax_reduce": ("max", "sum"),
}


def _select(rtype: str, sel: int) -> str:
    fields = _REDUCTION_FIELDS.get(rtype)
    if fields and sel < len(fields):
        return f".{fields[sel]}"
    return f"[{sel}]"


LEGEND_POST_LOWERING = """\
# Inductor IR, post-lowering (bodies from get_default_sizes_body).
# Backend-agnostic: ranges and index expressions only.
# Loop order is the as-lowered order -- usable as-is, but no stride-based
# reordering and no dim merging has run.
#"""

LEGEND_POST_SCHEDULER = """\
# Inductor IR, post scheduler-node construction (simplify_and_reorder applied).
# Backend-agnostic: ranges and index expressions only.
# Loop order optimized by pick_loop_order (small strides innermost); fusion can
# still change it, just less often. Dim merging may still be pending -- see
# config.loop_ordering_after_fusion.
#"""

LEGEND_POST_FUSION = """\
# Inductor IR, post-fusion (fuse_nodes and merge_loops applied).
# Kernel blocks are launch boundaries: inputs and outputs cross the boundary;
# internal values stay within it. Ordinary SIMD kernels show the selected iteration
# domain once; unsupported kernel kinds retain node-local operation domains.
#"""

# (legend line, substring that must appear in the body for it to be relevant)
LEGEND_LINES = [
    (
        "#   foreach p0 in [0,N)   independent points; any order, any parallelism.",
        "foreach ",
    ),
    (
        "#                         Listed outermost to innermost; the last one"
        " varies fastest.",
        "foreach ",
    ),
    ("#   reduce  p1 in [0,N)   points combined; order unspecified", "reduce "),
    ("#   buf[expr]             expr is a FLAT element offset, not a subscript", "["),
    (
        "#   T[shape] @ [x:N,r:M] non-obvious access domain; omitted for a direct match",
        " @ [",
    ),
    ("#   -> b offset=N         no storage of its own; writes into b at N", "  ->  "),
    (
        "#   rN = OP(v over p1)    rN is v combined across p1, one per foreach point",
        " over ",
    ),
    (
        "#   wrap_index(t,n)       t<0 ? t+n : t, then asserts 0 <= result < n",
        "wrap_index(",
    ),
    ("#   index(t,n)            asserts 0 <= t < n", " = index("),
    ("#   indentation           the body of the loop above", "    "),
]


def _legend(body: list[str], head: str) -> list[str]:
    """Only explain notation that actually occurs in this dump."""
    text = "\n".join(body)
    used = [line for line, probe in LEGEND_LINES if probe in text]
    if head == LEGEND_POST_FUSION and "foreach p" not in text:
        used = [line.replace("p0", "x").replace("p1", "r") for line in used]
    return [head, *used] if used else []


WRAP_WIDTH = 92


_BINOP = {
    "add": "+",
    "sub": "-",
    "mul": "*",
    "truediv": "/",
    "floordiv": "//",
    "mod": "%",
    "pow": "**",
    "eq": "==",
    "ne": "!=",
    "lt": "<",
    "gt": ">",
    "le": "<=",
    "ge": ">=",
    "bitwise_and": "&",
    "bitwise_or": "|",
    "bitwise_xor": "^",
    "bitwise_left_shift": "<<",
    "bitwise_right_shift": ">>",
    "logical_and": "and",
    "logical_or": "or",
}

_UNOP = {"neg": "-", "bitwise_not": "~", "logical_not": "not "}

_DTYPE_ABBREV = {
    torch.float32: "f32",
    torch.float64: "f64",
    torch.float16: "f16",
    torch.bfloat16: "bf16",
    torch.int64: "i64",
    torch.int32: "i32",
    torch.int16: "i16",
    torch.int8: "i8",
    torch.uint8: "u8",
    torch.bool: "b8",
}


def _dtype(dtype: torch.dtype) -> str:
    return _DTYPE_ABBREV.get(dtype, str(dtype).replace("torch.", ""))


def _vars(body: Any, vars: Sequence[Any]) -> tuple[sympy.Symbol, ...]:
    """Drop collapsed dims: index_vars_squeeze yields Integer(0) for size-1 dims,
    which never appear in var_ranges."""
    return tuple(v for v in vars if v in body.var_ranges)


def _expr(e: Any) -> str:
    if isinstance(e, (sympy.Expr, sympy.logic.boolalg.Boolean)):
        return sympy.printing.sstr(e)
    return str(e)


def _sizes(sizes: Sequence[Any]) -> str:
    return "[" + ",".join(_expr(s) for s in sizes) + "]"


def _tensor_decl(node: ir.IRNode) -> str:
    """`f32[128,64]` — the shape/layout part of a declaration.

    Contiguous strides are omitted so that a printed `stride=` always means the
    layout is worth noticing.
    """
    try:
        dtype = _dtype(node.get_dtype())
    except (AttributeError, NotImplementedError):
        return "?"
    size = node.get_size()
    out = f"{dtype}{_sizes(size)}"
    try:
        stride = node.get_stride()
    except (AttributeError, NotImplementedError, TypeError):
        return out
    # Inductor's own test, which ignores size-1 dims (their stride is
    # unobservable) and tolerates symbolic sizes.
    try:
        contiguous = ir.is_contiguous_strides_for_shape(stride, size)
    except Exception:
        contiguous = False
    if not contiguous:
        out += f" stride={_sizes(stride)}"
    offset = getattr(node.get_layout(), "offset", 0) if _has_layout(node) else 0
    if offset != 0:
        out += f" offset={_expr(offset)}"
    return out


def _has_layout(node: ir.IRNode) -> bool:
    try:
        node.get_layout()
        return True
    except Exception:
        return False


def _arg(a: Any) -> str:
    """Render one argument of an extern kernel call."""
    if isinstance(a, ir.ReinterpretView):
        inner = _arg(a.data)
        layout = a.layout
        parts = [
            inner,
            _sizes(layout.size),
            _sizes(layout.stride),
            _expr(layout.offset),
        ]
        return f"reinterpret_tensor({', '.join(parts)})"
    if isinstance(a, (ir.TensorBox, ir.StorageBox)):
        return _arg(a.data)
    if isinstance(a, ir.BaseView):
        return _arg(a.data)
    if isinstance(a, ir.IRNode):
        try:
            return a.get_name()
        except (AttributeError, NotImplementedError):
            return type(a).__name__
    if isinstance(a, (list, tuple)):
        return "[" + ", ".join(_arg(x) for x in a) + "]"
    if isinstance(a, torch.dtype):
        return _dtype(a)
    return repr(a)


@dataclasses.dataclass(frozen=True)
class _Access:
    """One rewritten flat buffer access."""

    index: sympy.Expr
    indirect: bool


@dataclasses.dataclass
class _Body:
    """Rendered body of one compute node."""

    iter_decl: str
    reduce_decl: str | None
    inner: list[str]
    outer: list[str]
    reads: list[str]
    accesses: dict[str, list[_Access]]


class _BodyRenderer:
    """Walks a LoopBody's FX graph and renders one line per value."""

    def __init__(
        self,
        body: Any,
        rename: dict[sympy.Symbol, sympy.Expr] | None = None,
    ) -> None:
        self.body = body
        self.rename = rename if rename is not None else self._var_renaming(body)
        self.text: dict[torch.fx.Node, str] = {}
        self.folded: dict[torch.fx.Node, str] = {}
        self.counter = itertools.count()
        self.r_counter = itertools.count()
        self.lines: list[str] = []
        self.reductions: dict[torch.fx.Node, tuple[str, str, Any]] = {}
        self.acc: dict[torch.fx.Node, str | None] = {}
        self.index_uses = self._count_index_uses(body)
        self.index_names: dict[str, str] = {}
        self._sink = self.lines
        self.reads: list[str] = []
        self.accesses: dict[str, list[_Access]] = {}
        self.stored: list[tuple[torch.fx.Node, bool]] = []
        self.indirect_meta: dict[str, str] = {}
        self.indirect_values: dict[str, str] = {}

    @staticmethod
    def _count_index_uses(body: Any) -> dict[str, int]:
        """How many times each named index expression is consumed."""
        counts: dict[str, int] = {}
        for node in body.root_block.graph.nodes:
            if node.op != "call_method":
                continue
            for arg in node.args:
                if (
                    isinstance(arg, torch.fx.Node)
                    and arg.op == "call_module"
                    and arg.target == "get_index"
                ):
                    name = arg.args[0]
                    counts[name] = counts.get(name, 0) + 1
        return counts

    @staticmethod
    def _var_renaming(body: Any) -> dict[sympy.Symbol, sympy.Expr]:
        """Map the ``q``-prefixed vars from get_default_sizes_body to ``p0..pN``."""
        names = {}
        for i, v in enumerate(
            _vars(body, body.iter_vars) + _vars(body, body.reduce_vars)
        ):
            names[v] = sympy.Symbol(f"p{i}", integer=True)
        return names

    def var(self, v: sympy.Symbol) -> str:
        return str(self.rename.get(v, v))

    def _index_info(self, node: torch.fx.Node) -> _Access:
        name = node.args[0]
        raw = self.body.indexing_exprs[name].subs(self.rename)
        indirect = False
        # Indirect symbols are artifacts; show the loaded value instead, while
        # remembering that this is not a rectangular coordinate mapping.
        for sym in list(raw.free_symbols):
            val = self.indirect_values.get(str(sym))
            if val:
                indirect = True
                raw = raw.subs(sym, sympy.Symbol(val))
        return _Access(raw, indirect)

    def raw_index_expr(self, node: torch.fx.Node) -> sympy.Expr:
        return self._index_info(node).index

    def raw_index(self, node: torch.fx.Node) -> str:
        return _expr(self.raw_index_expr(node))

    def index(self, node: torch.fx.Node, buffer: str | None = None) -> str:
        name = node.args[0]
        access = self._index_info(node)
        if buffer is not None:
            self.accesses.setdefault(buffer, []).append(access)
        expr = _expr(access.index)
        # A bare variable is not worth naming.
        if self.index_uses.get(name, 0) < 2 or expr.isidentifier():
            return expr
        if name not in self.index_names:
            var = f"ix{len(self.index_names)}"
            self.index_names[name] = var
            # Emitted at first use, so it lands in the domain where it is valid.
            self._sink.append(f"{var} = {expr}")
        return self.index_names[name]

    def _inline(self, node: Any, params: dict[str, str]) -> str:
        """Rebuild a point-free expression from the SSA chain, for the summary."""
        if not isinstance(node, torch.fx.Node):
            return _expr(node)
        if node in self.folded:
            return self.folded[node]
        if node.op == "call_function":
            if node.target is operator.getitem:
                return self._inline(node.args[0], params)
            return "..."
        if node.op != "call_method":
            return "..."

        target, args = node.target, node.args[1:]
        if target == "load":
            # Include the address because views are folded into indexing by this stage.
            # Without it, ``a + b.T`` is misleadingly summarized as ``a + b``.
            load = f"{args[0]}[{self.raw_index(args[1])}]"
            params[load] = args[0]
            return load
        if target == "reduction":
            return self._inline(args[-1], params)
        if target in _BINOP and len(args) == 2:
            a = self._paren(self._inline(args[0], params))
            b = self._paren(self._inline(args[1], params))
            return f"{a} {_BINOP[target]} {b}"
        if target in _UNOP and len(args) == 1:
            return f"{_UNOP[target]}{self._paren(self._inline(args[0], params))}"
        inner = ", ".join(self._inline(a, params) for a in args)
        return f"{target}({inner})"

    @staticmethod
    def _paren(text: str) -> str:
        """Parenthesise only a bare infix expression, not a call like exp(a - b)."""
        depth = 0
        for ch in text:
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            elif ch == " " and depth == 0:
                return f"({text})"
        return text

    def summary(self, body: Any) -> str | None:
        """One-line reading of this node as a single expression, or None."""
        if len(self.stored) != 1:
            return None
        val_node, is_reduction = self.stored[0]
        params: dict[str, str] = {}
        expr = self._inline(val_node, params)
        if "..." in expr or not params:
            return None

        if is_reduction and val_node in self.reductions:
            rtype, _, sel = self.reductions[val_node]
            over = ",".join(self.var(v) for v in _vars(body, body.reduce_vars))
            tail = _select(rtype, sel) if sel is not None else ""
            return f"{rtype}({expr} over {over}){tail}"
        # Neutral wording: an indirect index can be a read or a write, and this
        # summary cannot tell which without inspecting the store index.
        if self.indirect_meta:
            return f"indirect {expr}"
        if expr in params:
            return "copy"
        return expr

    def value(self, a: Any) -> str:
        if isinstance(a, torch.fx.Node):
            if a in self.folded:
                return self.folded[a]
            return self.text[a]
        if isinstance(a, torch.dtype):
            return _dtype(a)
        if isinstance(a, (list, tuple)):
            values = ", ".join(self.value(v) for v in a)
            if isinstance(a, tuple):
                return f"({values}{',' if len(a) == 1 else ''})"
            return f"[{values}]"
        return _expr(a)

    def emit(self, rhs: str) -> str:
        name = f"t{next(self.counter)}"
        self.lines.append(f"{name} = {rhs}")
        return name

    def run(self) -> _Body:
        body = self.body
        store_lines: list[str] = []

        for node in body.root_block.graph.nodes:
            if node.op == "placeholder" or node.op == "output":
                continue
            if node.op == "call_module":
                # get_index is resolved lazily where it is used; set_indirect<N>
                # binds the sympy symbol that a data-dependent index refers to.
                target = str(node.target)
                if target.startswith("set_indirect"):
                    var = target.removeprefix("set_")
                    shim = self.body.submodules.get(target)
                    kw = getattr(getattr(shim, "clone", None), "keywords", None) or {}
                    val = self.value(node.args[0])
                    size = _expr(kw["size"]) if "size" in kw else "?"
                    # Wrapping and checking are their own ops (indirect_indexing,
                    # check_bounds); give them a line instead of hiding them in
                    # the subscript that consumes the result.
                    name = f"i{var.removeprefix('indirect')}"
                    fn = "wrap_index" if kw.get("wrap_neg") else "index"
                    # Semantics are explained once in the legend; only the
                    # uncommon case (bounds check disabled) is worth a note.
                    note = "" if kw.get("check") else "      # unchecked"
                    self.lines.append(f"{name} = {fn}({val}, {size}){note}")
                    self.indirect_meta[var] = ", ".join(
                        x for x in (fn, "checked" if kw.get("check") else "") if x
                    )
                    self.indirect_values[var] = name
                elif target != "get_index":
                    rendered = ", ".join(self.value(a) for a in node.args)
                    self.text[node] = self.emit(f"{target}({rendered})")
                continue
            if node.op == "call_function":
                if node.target is operator.getitem:
                    parent, i = node.args
                    if parent in self.reductions:
                        rtype, val, _ = self.reductions[parent]
                        self.reductions[node] = (rtype, val, i)
                        self.acc[node] = self.acc.get(parent)
                        self.text[node] = f"{self.text[parent]}{_select(rtype, i)}"
                    else:
                        self.text[node] = self.emit(f"{self.value(parent)}[{i}]")
                    continue
                fn = getattr(node.target, "__name__", str(node.target))
                rendered = ", ".join(self.value(a) for a in node.args)
                self.text[node] = self.emit(f"{fn}({rendered})")
                continue
            if node.op != "call_method":
                continue

            target = node.target
            args = node.args[1:]  # args[0] is the ops handler

            if target == "constant":
                # fold into the use site; keep the dtype only when it is not
                # implied by the surrounding computation
                self.folded[node] = _expr(args[0])
                continue

            if target == "load":
                name = args[0]
                idx = self.index(args[1], name)
                if name not in self.reads:
                    self.reads.append(name)
                self.text[node] = self.emit(f"{name}[{idx}]")
                continue

            if target == "index_expr":
                expr = args[0]
                if (
                    isinstance(expr, torch.fx.Node)
                    and expr.op == "call_module"
                    and expr.target == "get_index"
                ):
                    rendered = self.index(expr)
                elif isinstance(expr, torch.fx.Node):
                    rendered = self.value(expr)
                else:
                    rendered = _expr(expr.subs(self.rename))
                self.text[node] = self.emit(f"index({rendered})")
                continue

            if target == "reduction":
                # args: dtype, src_dtype, reduction_type, value
                rtype, val = args[-2], self.value(args[-1])
                over = ",".join(self.var(v) for v in _vars(body, body.reduce_vars))
                acc = _dtype(args[1]) if args[0] != args[1] else None
                rhs = f"{rtype}({val} over {over})"
                if acc:
                    rhs = f"{rhs} acc={acc}"
                name = f"r{next(self.r_counter)}"
                self._sink.append(f"{name} = {rhs}")
                self.reductions[node] = (rtype, val, None)
                self.acc[node] = acc
                self.text[node] = name
                continue

            if target in ("store", "store_reduction"):
                self._sink = store_lines
                name, idx = args[0], self.index(args[1], args[0])
                val_node = args[2]
                self.stored.append((val_node, target == "store_reduction"))
                rhs = self.value(val_node)
                mode = args[3] if len(args) > 3 else None
                if mode:
                    rhs = f"{mode}({rhs})"
                store_lines.append(f"{name}[{idx}] = {rhs}")
                self._sink = self.lines
                continue

            if target in _BINOP and len(args) == 2:
                a, b = self.value(args[0]), self.value(args[1])
                self.text[node] = self.emit(f"{a} {_BINOP[target]} {b}")
                continue

            if target in _UNOP and len(args) == 1:
                self.text[node] = self.emit(f"{_UNOP[target]}{self.value(args[0])}")
                continue

            rendered = ", ".join(self.value(a) for a in args)
            self.text[node] = self.emit(f"{target}({rendered})")

        iter_decl = self._decl(_vars(body, body.iter_vars))
        reduce_vars = _vars(body, body.reduce_vars)
        reduce_decl = self._decl(reduce_vars) if reduce_vars else None
        return _Body(
            iter_decl,
            reduce_decl,
            self.lines,
            store_lines,
            self.reads,
            self.accesses,
        )

    def _decl(self, vars: Sequence[sympy.Symbol]) -> str:
        return ", ".join(
            f"{self.var(v)} in [0,{_expr(self.body.var_ranges[v])})" for v in vars
        )


def _group(sizes_pair) -> str:
    """The scheduler's fusion key: (pointwise numel, reduction numel).

    Two nodes can share a kernel launch only if these are commensurate, so it
    is the number to compare across nodes -- 2048 fuses into 32x64.
    """
    index_size, reduce_size = sizes_pair
    return f"({_expr(sympy.prod(index_size))}, {_expr(sympy.prod(reduce_size))})"


def _scheduler_nodes() -> dict[str, Any] | None:
    """The scheduler's op-name -> node map, once it exists.

    ``V.graph.scheduler`` is assigned at the top of ``Scheduler._init``, before
    ``name_to_node`` is populated, so both have to be probed.
    """
    sched = getattr(V.graph, "scheduler", None)
    return getattr(sched, "name_to_node", None) if sched is not None else None


def _kernel_map(graph: GraphLowering) -> dict[str, int]:
    """op name -> index of the fused group it will be generated into.

    Empty before scheduling: the grouping does not exist until fusion has run,
    so this annotation appears only when the printer is called late enough.
    """
    sched = getattr(graph, "scheduler", None)
    if sched is None:
        return {}
    out: dict[str, int] = {}
    for i, snode in enumerate(getattr(sched, "nodes", ()) or ()):
        try:
            names = snode.get_operation_names()
        except (AttributeError, NotImplementedError):
            continue
        for name in names:
            out[name] = i
    return out


def _sizes_body(op: ir.Operation) -> tuple[Any, Any]:
    """Prefer the scheduler's derived body, else the one lowering produced.

    ``simplify_and_reorder`` does not rewrite the IR -- it returns a derived view
    that the scheduler stores on the node -- so ``get_default_sizes_body`` keeps
    returning the pre-reorder body no matter how late it is called.
    """
    by_name = _scheduler_nodes()
    snode = by_name.get(op.get_operation_name()) if by_name else None
    body = getattr(snode, "_body", None)
    if body is not None:
        return snode._sizes, body
    sizes_pair, default_body, _ = op.get_default_sizes_body()
    return sizes_pair, default_body


def _render_computed(
    sizes_pair: Any,
    body: Any,
    indent: str = "    ",
    rename: dict[sympy.Symbol, sympy.Expr] | None = None,
    kernel_domain: bool = False,
) -> tuple[list[str], list[str], dict[str, set[sympy.Symbol]], str | None, str]:
    renderer = _BodyRenderer(body, rename)
    rendered = renderer.run()
    summary = renderer.summary(body)

    if kernel_domain:
        return (
            [f"    {line}" for line in (*rendered.inner, *rendered.outer)],
            rendered.reads,
            rendered.accesses,
            summary,
            _group(sizes_pair),
        )

    # One body, one scope: the reduction is an op inside it, not a nested loop.
    parts = []
    if rendered.iter_decl:
        parts.append(f"foreach {rendered.iter_decl}")
    if rendered.reduce_decl:
        parts.append(f"reduce {rendered.reduce_decl}")
    out = [f"{indent}{', '.join(parts) if parts else 'foreach (scalar)'}:"]
    out.extend(f"{indent * 2}{line}" for line in rendered.inner)
    out.extend(f"{indent * 2}{line}" for line in rendered.outer)
    return out, rendered.reads, rendered.accesses, summary, _group(sizes_pair)


def _wrap_call(kernel: str, args: list[str], indent: str) -> list[str]:
    """`kernel(a, b, c)` on one line, else filled across continuation lines."""
    oneline = f"{indent}{kernel}({', '.join(args)})"
    if len(oneline) <= WRAP_WIDTH or not args:
        return [oneline]

    cont = indent + "    "
    lines: list[str] = []
    cur = f"{indent}{kernel}("
    for i, arg in enumerate(args):
        tail = "," if i < len(args) - 1 else ""
        at_open = cur.endswith("(")
        sep = "" if at_open else " "
        if not at_open and len(cur) + len(sep) + len(arg) + len(tail) > WRAP_WIDTH:
            lines.append(cur)
            cur = cont + arg + tail
        else:
            cur = cur + sep + arg + tail
    lines.append(cur + ")")
    return lines


def _alias_target(buf: ir.Buffer) -> tuple[str, Any, Any] | None:
    """``(owner, offset, numel)`` for a buffer that lives inside another's storage."""
    layout = getattr(buf, "layout", None)
    view = getattr(layout, "view", None)
    if view is None:
        return None
    try:
        target = view.get_name()
    except (AttributeError, NotImplementedError):
        return None
    vl = getattr(view, "layout", None)
    if vl is None:
        return None
    offset = getattr(vl, "offset", None)
    if offset is None:
        return None
    return target, offset, sympy.prod(vl.size)


def _const_decl(t: Any) -> str:
    """Declaration for a frozen constant, which is a real tensor not an IRNode."""
    try:
        out = f"{_dtype(t.dtype)}{_sizes(tuple(t.shape))}"
        if not ir.is_contiguous_strides_for_shape(t.stride(), tuple(t.shape)):
            out += f" stride={_sizes(t.stride())}"
        return out
    except Exception:
        return "?"


def _fill(parts: Any) -> list[str]:
    """Comma-separated declarations filled to WRAP_WIDTH, not one per line."""
    lines: list[str] = []
    cur = ""
    for part in parts:
        add = part if not cur else f"{cur}, {part}"
        if len(add) > WRAP_WIDTH and cur:
            lines.append(cur + ",")
            cur = part
        else:
            cur = add
    if cur:
        lines.append(cur)
    return lines


def _render_buffers(graph: GraphLowering) -> list[str]:
    """Buffers that share one allocation, grouped under the owner.

    Buffers that own their storage are declared at their own node, so they are
    not repeated here. Storage sharing is the only part a reader cannot recover
    from the compute section, and it is decided (not a later choice).
    """
    decls = {buf.get_name(): _tensor_decl(buf) for buf in graph.buffers}
    slices: dict[str, list[tuple[Any, str, Any]]] = {}
    for buf in graph.buffers:
        info = _alias_target(buf)
        if info is None:
            continue
        owner, offset, numel = info
        slices.setdefault(owner, []).append((offset, buf.get_name(), numel))

    if not slices:
        return []

    lines = ["# shared storage -- these are regions of one buffer, not copies"]
    for owner, parts in slices.items():
        lines.append(f"{owner}: {decls.get(owner, '?')}")
        for offset, name, numel in sorted(parts, key=lambda p: str(p[0])):
            lo = _expr(offset)
            hi = _expr(offset + numel)
            lines.append(f"    {name}  =  {owner}[{lo}:{hi}]")
    return lines


def _tuple_members(graph: GraphLowering) -> dict[str, list[str]]:
    """Parent of a `MultiOutputLayout` -> the buffers that index into it.

    A tuple-returning kernel has no dtype or size of its own, so without this it
    renders as `?`. The members carry the real layouts.
    """
    out: dict[str, list[str]] = {}
    for op in graph.operations:
        for buf in _outputs_of(op):
            node = getattr(buf, "data", buf)
            if type(node).__name__ != "MultiOutput":
                continue
            inputs = getattr(node, "inputs", ())
            if not inputs:
                continue
            try:
                parent = inputs[0].get_name()
            except (AttributeError, NotImplementedError):
                continue
            out.setdefault(parent, []).append(buf.get_name())
    return out


def _members_of(parent: str) -> list[tuple[Any, str, str]]:
    """`[(index, buffer, op)]` for the buffers indexing into a tuple-returning call."""
    graph = getattr(V, "graph", None)
    if graph is None:
        return []
    out: list[tuple[Any, str, str]] = []
    for op in graph.operations:
        node = op if type(op).__name__ == "MultiOutput" else getattr(op, "data", None)
        if type(node).__name__ != "MultiOutput":
            continue
        srcs = getattr(node, "inputs", ())
        if not srcs:
            continue
        try:
            if srcs[0].get_name() != parent:
                continue
        except (AttributeError, NotImplementedError):
            continue
        idx = [i for _kind, i in getattr(node, "indices", ())]
        out.append((idx[0] if idx else 0, op.get_name(), op.get_operation_name()))
    return sorted(out, key=lambda x: str(x[0]))


def _decl_map(graph: GraphLowering) -> dict[str, str]:
    """name -> shape/layout, so a read can carry its shape at the use site."""
    decls: dict[str, str] = {}
    for name, inp in graph.graph_inputs.items():
        node = inp
        while isinstance(node, (ir.TensorBox, ir.StorageBox)):
            node = node.data
        decls[name] = _tensor_decl(node)
    for buf in graph.buffers:
        decls[buf.get_name()] = _tensor_decl(buf)
    for name, tensor in (getattr(graph, "constants", None) or {}).items():
        decls[name] = _const_decl(tensor)
    return decls


def _primary_aten_of(op: ir.Operation) -> str:
    """The primary ATen operation represented by a template, e.g. ``mm``.

    A template can inherit origins from pointwise producers and consumers. Those
    operations are rendered in their own blocks, so including them in the template
    name would duplicate them and incorrectly suggest that they are template calls.
    """
    try:
        origins = op.get_origins() or ()
    except (AttributeError, NotImplementedError):
        return ""
    for origin in origins:
        aten = getattr(origin, "meta", {}).get("original_aten")
        if aten is None:
            continue
        if isinstance(aten, torch._ops.OpOverload):
            return aten._overloadpacket.__name__
        return str(aten)
    return ""


def _template_desc(op: ir.TemplateBuffer) -> tuple[str, str | None]:
    """``(call name, note)`` for a template.

    A ``MultiTemplateBuffer`` still holds every candidate. The winner is picked
    *during* fusion, not before or after it: ``speedup_by_fusion`` benchmarks each
    choice together with whatever epilogue is being fused in, since the best config
    for a bare op is not always the best config once an epilogue is attached.
    ``finalize_multi_template_buffers`` then just commits the result.

    The candidate kinds matter because an extern choice cannot fuse a prologue or
    epilogue, while a Triton one can.
    """
    choices = getattr(op, "_choices", None)
    what = _primary_aten_of(op)
    call = f"template {what}" if what else "template"
    if choices:
        counts: dict[str, int] = {}
        for choice in choices:
            kind = type(choice).__name__.removesuffix("Caller").removesuffix("Base")
            counts[kind] = counts.get(kind, 0) + 1
        breakdown = ", ".join(f"{n} {kind}" for kind, n in sorted(counts.items()))
        return call, f"not chosen yet -- {breakdown}; benchmarked during fusion"
    name = getattr(op, "python_kernel_name", None)
    return (f"{call} {name}" if name else call), None


def _render_operation(
    op: ir.Operation,
    indent: str = "    ",
    decls: dict[str, str] | None = None,
    kernels: dict[str, int] | None = None,
    rename: dict[sympy.Symbol, sympy.Expr] | None = None,
    kernel_domain: bool = False,
    domain: _KernelDomain | None = None,
) -> list[str]:
    bufs = _outputs_of(op)
    mutations = []
    primary = None
    for buf in bufs:
        names = buf.get_mutation_names()
        if names:
            mutations.extend(names)
        elif primary is None:
            primary = buf

    lhs_parts = []
    if primary is not None:
        # Repeat dtype/shape here: the store index (`buf1[p0]`) shows a flat
        # offset, so it cannot tell you the buffer is logically [32,1].
        decl = _tensor_decl(primary)
        if decl == "?" and decls:
            decl = decls.get(primary.get_name(), decl)
        lhs_parts.append(f"{primary.get_name()}: {decl}")
    lhs_parts += [f"mut({name})" for name in mutations]
    lhs = ", ".join(lhs_parts)

    op_name = op.get_operation_name()
    header = lhs or op_name
    # Which nodes share a kernel is the scheduler's main output; it is absent
    # until fusion has run, so the suffix appears only on a late enough dump.
    kid = kernels.get(op_name) if kernels else None
    op_label = op_name if kid is None else f"{op_name}  kernel{kid}"

    if isinstance(op, ir.ComputedBuffer):
        body_lines, reads, accesses, summary, _ = _render_computed(
            *_sizes_body(op),
            indent,
            rename=rename,
            kernel_domain=kernel_domain,
        )
        if kernel_domain and domain is not None:
            lhs_parts = []
            if primary is not None:
                decl = _tensor_decl(primary)
                if decl == "?" and decls:
                    decl = decls.get(primary.get_name(), decl)
                lhs_parts.append(
                    f"{primary.get_name()}: {decl}"
                    f"{_access_domain(primary.get_name(), accesses, domain)}"
                )
            lhs_parts += [
                f"mut({name}){_access_domain(name, accesses, domain)}"
                for name in mutations
            ]
            header = ", ".join(lhs_parts) or op_name
        if reads:
            shown = [
                (
                    f"{name}: {decls[name]}{_access_domain(name, accesses, domain)}"
                    if decls and name in decls and kernel_domain and domain is not None
                    else f"{name}: {decls[name]}"
                    if decls and name in decls
                    else name
                )
                for name in reads
            ]
            header = f"{header}  <-  {', '.join(shown)}"
        tag = f"# {op_label}" + (f"  {summary}" if summary else "")
        return [tag, header] + body_lines
    if isinstance(op, ir.TemplateBuffer):
        args = [_arg(a) for a in getattr(op, "inputs", ())]
        call, note = _template_desc(op)
        out = [f"# {op_label}", header] + _wrap_call(call, args, indent)
        if note:
            out.append(f"{indent}# {note}")
        return out
    if isinstance(op, ir.NopKernel):
        srcs = [_arg(a) for a in getattr(op, "inputs", ())]
        kind = "concat" if isinstance(op, ir.ConcatKernel) else type(op).__name__
        if srcs:
            return [
                f"# {op_label}",
                f"{header} = {kind}({', '.join(srcs)})  # allocation only",
            ]
        return [f"# {op_label}", header, f"{indent}nop {type(op).__name__}"]

    node = op if type(op).__name__ == "MultiOutput" else getattr(op, "data", None)
    if type(node).__name__ == "MultiOutput":
        # Absorbed into the call that produces the tuple; a separate block per
        # element says nothing, and neither the tuple nor the indexing exists at
        # all in the C++ wrapper.
        return []

    kernel = (
        getattr(op, "python_kernel_name", None)
        or getattr(op, "cpp_kernel_name", None)
        or type(op).__name__
    )
    args = [_arg(a) for a in getattr(op, "inputs", ())]
    args += [_arg(a) for a in getattr(op, "constant_args", ())]
    for k, v in (getattr(op, "kwargs", None) or {}).items():
        args.append(f"{k}={_arg(v)}")

    members = (
        _members_of(primary.get_name())
        if primary is not None and isinstance(op.layout, ir.MultiOutputLayout)
        else []
    )
    if members:
        bufs = [b for _i, b in ((m[0], m[1]) for m in members)]
        ops = [m[2] for m in members]
        out = [f"# {op_label}, " + ", ".join(ops)]
        out += _wrap_call(f"{', '.join(bufs)} = {kernel}", args, "")
        out += [
            f"{indent}{b}: {decls[b]}" if decls and b in decls else f"{indent}{b}"
            for b in bufs
        ]
        return out
    return [f"# {op_label}", header] + _wrap_call(kernel, args, indent)


def _devices(graph: GraphLowering) -> list[str]:
    seen = []
    for op in graph.operations:
        for buf in _outputs_of(op):
            try:
                dev = str(buf.get_device())
            except (AttributeError, NotImplementedError):
                continue
            if dev not in seen:
                seen.append(dev)
    return seen


def _outputs_of(op: ir.Operation) -> list[ir.Buffer]:
    try:
        return list(op.get_outputs())
    except (AttributeError, NotImplementedError):
        return []


def _output_name(node: ir.IRNode) -> str:
    return _arg(node)


@dataclasses.dataclass(frozen=True)
class _KernelDomain:
    """A flat SIMD iteration domain and its display coordinates."""

    groups: tuple[sympy.Expr, ...]
    symbols: tuple[sympy.Symbol, ...]
    pointwise_dims: int
    numel: sympy.Expr
    reduction_numel: sympy.Expr

    def declaration(self) -> str:
        parts = []
        sizevars = V.graph.sizevars
        for i, (symbol, extent) in enumerate(zip(self.symbols, self.groups)):
            if sizevars.statically_known_equals(extent, sympy.S.One):
                continue
            kind = "foreach" if i < self.pointwise_dims else "reduce"
            parts.append(f"{kind} {symbol} in [0,{_expr(extent)})")
        return ", ".join(parts) if parts else "foreach (scalar)"


def _logical_size(name: str) -> tuple[sympy.Expr, ...] | None:
    """Structured logical dimensions for a graph buffer, input, or constant."""
    try:
        node = V.graph.try_get_buffer(name)
        return tuple(node.get_size()) if node is not None else None
    except Exception:
        return None


def _expr_equal(left: sympy.Expr, right: sympy.Expr) -> bool:
    try:
        if V.graph.sizevars.statically_known_equals(left, right):
            return True
        return sympy.simplify(left - right) == 0
    except Exception:
        return False


def _direct_row_major(
    accesses: list[_Access],
    coordinates: list[tuple[sympy.Symbol, sympy.Expr]],
    logical_size: tuple[sympy.Expr, ...] | None,
) -> bool:
    """Whether every access is the obvious row-major index for the logical shape."""
    if logical_size is None or len(logical_size) != len(coordinates):
        return False
    if not logical_size:
        return False  # Keep @ [scalar] explicit even though buf[0] is direct.
    extents = tuple(extent for _symbol, extent in coordinates)
    if not all(
        _expr_equal(size, extent) for size, extent in zip(logical_size, extents)
    ):
        return False
    expected = sympy.S.Zero
    for i, (symbol, _extent) in enumerate(coordinates):
        expected += symbol * sympy.prod(extents[i + 1 :])
    return all(
        not access.indirect and _expr_equal(access.index, expected)
        for access in accesses
    )


def _access_domain(
    name: str,
    accesses: dict[str, list[_Access]],
    domain: _KernelDomain,
) -> str:
    """Non-obvious kernel coordinates used to access one logical buffer."""
    buffer_accesses = accesses.get(name)
    if not buffer_accesses or any(access.indirect for access in buffer_accesses):
        return ""
    used = set().union(*(access.index.free_symbols for access in buffer_accesses))
    coordinates = [
        (symbol, extent)
        for symbol, extent in zip(domain.symbols, domain.groups)
        if symbol in used
        and not V.graph.sizevars.statically_known_equals(extent, sympy.S.One)
    ]
    if _direct_row_major(buffer_accesses, coordinates, _logical_size(name)):
        return ""
    parts = [f"{symbol}:{_expr(extent)}" for symbol, extent in coordinates]
    return f" @ [{','.join(parts) if parts else 'scalar'}]"


def _kernel_domain(sn: Any) -> _KernelDomain | None:
    """Return a real flat SIMD domain, excluding CPU and sentinel groups."""
    try:
        if not sn.is_gpu():
            return None
        _device, raw_groups = sn.group
    except Exception:
        return None

    if (
        not isinstance(raw_groups, (tuple, list))
        or len(raw_groups) != 2
        or any(not isinstance(group, sympy.Expr) for group in raw_groups)
    ):
        return None

    numel, reduction_numel = raw_groups
    try:
        tiling = sn.get_tiling(numel, reduction_numel)
    except Exception:
        return None
    if not tiling or any(
        not isinstance(group, sympy.Expr) for group in tiling.values()
    ):
        return None

    keys = tuple(tiling)
    pointwise_dims = next(
        (i for i, key in enumerate(keys) if key.startswith("r")), len(keys)
    )
    if any(key.startswith("r") for key in keys[:pointwise_dims]) or any(
        not key.startswith("r") for key in keys[pointwise_dims:]
    ):
        return None
    pointwise_names = keys[:pointwise_dims]
    reduction_dims = len(keys) - pointwise_dims
    reduction_names = (
        ("r",) if reduction_dims == 1 else tuple(f"r{i}" for i in range(reduction_dims))
    )
    names = (*pointwise_names, *reduction_names)
    symbols = tuple(sympy.Symbol(name, integer=True) for name in names)
    return _KernelDomain(
        tuple(tiling.values()), symbols, pointwise_dims, numel, reduction_numel
    )


def _node_reduction_membership(sn: Any, domain: _KernelDomain) -> dict[Any, bool]:
    """Whether each leaf is emitted while the kernel reduction is enabled."""
    nodes = list(sn.get_nodes())
    if not nodes:
        return {}

    from torch._inductor.codegen.simd_kernel_features import (
        DisableReduction,
        EnableReduction,
    )

    try:
        backend = sn.scheduler.get_backend(sn.get_device())
        schedule = backend.generate_node_schedule(
            nodes, domain.numel, domain.reduction_numel
        )
        inside_reduction = not V.graph.sizevars.statically_known_equals(
            domain.reduction_numel, sympy.S.One
        )
        membership: dict[Any, bool] = {}
        for item in schedule:
            if item is DisableReduction:
                inside_reduction = False
            elif item is EnableReduction:
                inside_reduction = True
            else:
                membership[item] = inside_reduction
        if all(node in membership for node in nodes):
            return membership
    except Exception:
        pass

    numel, rnumel = domain.numel, domain.reduction_numel
    membership = {}
    for node in nodes:
        try:
            _device, (node_numel, node_rnumel) = node.group
            outside = node_numel == numel and node_rnumel == 1 and rnumel != 1
        except Exception:
            outside = False
        membership[node] = not outside
    return membership


def _kernel_rename(
    sn: Any, domain: _KernelDomain, inside_reduction: bool
) -> dict[sympy.Symbol, sympy.Expr] | None:
    """Map one node's local loop variables into the selected kernel domain."""
    from torch._inductor.codegen.simd import CantSplit, SIMDKernel
    from torch._inductor.codegen.simd_kernel_features import MemoryEstimator

    body = getattr(sn, "_body", None)
    if body is None:
        return None

    groups = list(domain.groups)
    if not inside_reduction:
        for i in range(domain.pointwise_dims, len(groups)):
            groups[i] = sympy.S.One

    def set_ranges(*length_groups: list[sympy.Expr]) -> list[list[sympy.Expr]]:
        if len(length_groups) != len(groups):
            raise CantSplit(groups, length_groups)
        return [
            MemoryEstimator.make_flat_range(symbol, extent, list(lengths))
            for symbol, extent, lengths in zip(domain.symbols, groups, length_groups)
        ]

    try:
        lengths = SIMDKernel.prepare_split_iteration_lengths(
            groups,
            sn.get_ranges(),
            domain.reduction_numel if inside_reduction else sympy.S.One,
        )
        mapped = SIMDKernel.map_kernel_groups_to_node_sizes(groups, lengths, set_ranges)
    except Exception:
        return None

    rename: dict[sympy.Symbol, sympy.Expr] = {}
    for body_vars, mapped_vars in zip(
        (_vars(body, body.iter_vars), _vars(body, body.reduce_vars)), mapped
    ):
        if len(body_vars) != len(mapped_vars):
            return None
        rename.update(zip(body_vars, mapped_vars))
    expected = set(_vars(body, body.iter_vars) + _vars(body, body.reduce_vars))
    return rename if set(rename) == expected else None


def _kernel_renames(
    sn: Any, domain: _KernelDomain
) -> dict[Any, dict[sympy.Symbol, sympy.Expr]] | None:
    """Build every member mapping before printing any unified kernel domain."""
    membership = _node_reduction_membership(sn, domain)
    renames = {}
    for node in sn.get_nodes():
        op = getattr(node, "node", None)
        if not isinstance(op, ir.ComputedBuffer):
            return None
        rename = _kernel_rename(node, domain, membership.get(node, True))
        if rename is None:
            return None
        renames[node] = rename
    return renames


_KERNEL_PATTERNS = {
    "FusedNestedReductions": "pattern: nested reduction",
    "FusedStagedReduction": "pattern: staged reduction",
    "FusedMixOrderReductions": "pattern: mixed-order reduction",
    "ForeachKernelSchedulerNode": (
        "pattern: foreach/combo; subkernels are mutually independent by definition, "
        "and dispatch may interleave them"
    ),
    "FusedExternTritonKernelSchedulerNode": (
        "implementation: user Triton + generated epilogue"
    ),
    "ExternKernelSchedulerNode": "implementation: extern call",
    "NopKernelSchedulerNode": "implementation: allocation only, no kernel",
    "GroupedSchedulerNode": "pattern: grouped, must schedule together",
}


def _order_key(name: str) -> tuple[str, int]:
    """buf2 before buf10."""
    head = name.rstrip("0123456789")
    tail = name[len(head) :]
    return head, int(tail) if tail else -1


def _mutations_of(sn: Any) -> list[str]:
    """Buffers a kernel writes in place, from the scheduler rather than inferred."""
    out: list[str] = []
    for node in [sn, *getattr(sn, "get_nodes", lambda: ())()]:
        fn = getattr(node, "get_mutations", None)
        if fn is None:
            continue
        try:
            out.extend(fn() or ())
        except (AttributeError, NotImplementedError):
            continue
    return out


def _node_sets(
    graph: GraphLowering,
    nodes: Sequence[Any],
    externally_visible: set[str] | None = None,
) -> list[tuple[list[str], list[str], list[str], list[str]]]:
    """Per node: buffers read externally, written externally, private, or mutated."""
    read_by: dict[str, set[int]] = {}
    for i, sn in enumerate(nodes):
        for dep in sn.read_writes.reads:
            read_by.setdefault(dep.name, set()).add(i)

    outs = {_output_name(o) for o in graph.graph_outputs}
    outs.update(externally_visible or ())
    mutated_args = set(getattr(graph, "mutated_inputs", ()) or ())
    sets = []
    for i, sn in enumerate(nodes):
        mine = set(sn.get_buffer_names())
        my_reads = {d.name for d in sn.read_writes.reads}
        # A mutation is read and written, so it belongs in neither list. It
        # cannot be spotted as reads & writes: that also matches a value this
        # kernel produces and then consumes.
        mutates = set(_mutations_of(sn)) | (mine & mutated_args)
        internal = {
            n for n in mine - mutates if n not in outs and read_by.get(n, set()) <= {i}
        }
        sets.append(
            (
                sorted(my_reads - mine - mutates, key=_order_key),
                sorted(mine - internal - mutates, key=_order_key),
                sorted(internal, key=_order_key),
                sorted(mutates, key=_order_key),
            )
        )
    return sets


def _kernel_sets(
    graph: GraphLowering, sched: Any
) -> list[tuple[list[str], list[str], list[str], list[str]]]:
    """Per top-level scheduler node, compute its launch-boundary buffer sets."""
    return _node_sets(graph, list(getattr(sched, "nodes", ()) or ()))


def _scheduler_types() -> dict[str, type[Any]]:
    """Scheduler classes used for ordered dispatch, imported lazily."""
    from torch._inductor import scheduler
    from torch._inductor.codegen.cpp import OuterLoopFusedSchedulerNode

    return {
        "extern": scheduler.ExternKernelSchedulerNode,
        "fused": scheduler.FusedSchedulerNode,
        "grouped": scheduler.GroupedSchedulerNode,
        "mix_order": scheduler.FusedMixOrderReductions,
        "nested": scheduler.FusedNestedReductions,
        "nop": scheduler.NopKernelSchedulerNode,
        "ordinary": scheduler.SchedulerNode,
        "outer_loop": OuterLoopFusedSchedulerNode,
        "staged": scheduler.FusedStagedReduction,
        "foreach": scheduler.ForeachKernelSchedulerNode,
        "user_triton": scheduler.FusedExternTritonKernelSchedulerNode,
    }


def _kernel_kind(sn: Any) -> str:
    """Classify scheduler nodes, checking specialized fused types first."""
    types = _scheduler_types()
    dispatch = (
        (types["nested"], "deferred"),
        (types["staged"], "deferred"),
        (types["mix_order"], "deferred"),
        (types["user_triton"], "deferred"),
        (types["outer_loop"], "deferred"),
        (types["foreach"], "foreach"),
        (types["grouped"], "grouped"),
        (types["extern"], "opaque"),
        (types["nop"], "nop"),
    )
    for cls, kind in dispatch:
        if isinstance(sn, cls):
            return kind
    if type(sn) is types["ordinary"]:
        return "opaque" if sn.is_template() else "ordinary"
    if type(sn) is types["fused"]:
        return "ordinary"
    return "unknown"


def _is_nop(sn: Any) -> bool:
    """True for a node that generates no kernel, only an allocation."""
    return _kernel_kind(sn) == "nop"


def _render_nop(sn: Any, decls: dict[str, str]) -> list[str]:
    """A nop launches nothing, so it gets a line rather than a kernel block.

    What it contributes -- that one buffer is the concatenation of others -- is
    already stated in the shared-storage section.
    """
    out = []
    for sub in sn.get_nodes():
        op = getattr(sub, "node", None)
        if op is None:
            continue
        block = _render_operation(op, decls=decls)
        body = " ".join(ln.strip() for ln in block if not ln.startswith("#"))
        body = body.replace("  # allocation only", "")
        out.append(f"# {sub.get_name()}:  {body}   -- no kernel, allocation only")
    return out


def _shared_reduction_key(op: ir.Operation) -> tuple[Any, ...] | None:
    """Identity of the reduction a node selects one output of.

    ``var_mean`` and online softmax lower to one reduction but several nodes,
    each holding the *same* ``inner_fn`` object and differing only in
    ``output_index``. Nodes sharing this key are one computation, so rendering
    them as separate blocks overstates the work by a factor of N.
    """
    data = getattr(op, "data", None)
    if not isinstance(data, ir.MultiOutputReduction):
        return None
    fn = getattr(data, "inner_fn", None)
    if fn is None:
        return None
    return (
        id(fn),
        getattr(data, "reduction_type", None),
        tuple(str(r) for r in getattr(data, "ranges", ())),
        tuple(str(r) for r in getattr(data, "reduction_ranges", ())),
    )


def _render_kernel_interface(
    sn: Any,
    title: str,
    sets: tuple[list[str], list[str], list[str], list[str]],
    indent: str,
    *,
    show_pattern: bool = True,
) -> list[str]:
    inputs, outputs, internal, mutates = sets
    out = [title]
    if show_pattern:
        pattern = _KERNEL_PATTERNS.get(type(sn).__name__)
        if pattern:
            out.append(f"{indent}# {pattern}")
        elif _kernel_kind(sn) == "unknown":
            out.append(f"{indent}# scheduler node: {type(sn).__name__}")
    for label, names in (
        ("inputs", inputs),
        ("outputs", outputs),
        ("internal", internal),
        ("mutates", mutates),
    ):
        if names:
            out.append(f"{indent}{label + ':':10}[{', '.join(names)}]")
    return out


def _render_ordinary_kernel_body(
    sn: Any, decls: dict[str, str], indent: str
) -> list[str]:
    domain = _kernel_domain(sn)
    renames = _kernel_renames(sn, domain) if domain is not None else None
    if domain is None or renames is None:
        return _render_kernel_body(sn, decls, indent)
    out = ["", f"{indent}{domain.declaration()}:"]
    out.extend(
        _render_kernel_body(
            sn,
            decls,
            indent + "    ",
            renames,
            kernel_domain=True,
            domain=domain,
        )
    )
    return out


def _render_foreach_kernel(
    sn: Any, decls: dict[str, str], indent: str, kernel_domains: bool
) -> list[str]:
    out = ["", f"{indent}parallel:"]
    branch_indent = indent + "    "
    body_indent = branch_indent + "    "
    for i, branch in enumerate(getattr(sn, "snodes", ()) or ()):
        body = (
            _render_ordinary_kernel_body(branch, decls, body_indent)
            if kernel_domains and _kernel_kind(branch) == "ordinary"
            else _render_kernel_body(branch, decls, body_indent)
        )
        if not body:
            continue
        out.extend(("", f"{branch_indent}branch {i}:"))
        out.extend(body)
    return out


def _render_kernel(
    sn: Any,
    idx: int,
    sets: tuple[list[str], list[str], list[str], list[str]],
    decls: dict[str, str],
    indent: str = "    ",
    kernel_domains: bool = False,
) -> list[str]:
    """One executable launch: its interface, selected domain, and operations."""
    kind = _kernel_kind(sn)
    out = _render_kernel_interface(
        sn, f"kernel k{idx}", sets, indent, show_pattern=kind != "foreach"
    )
    if kind == "foreach":
        out.extend(_render_foreach_kernel(sn, decls, indent, kernel_domains))
    elif kind == "ordinary" and kernel_domains:
        out.extend(_render_ordinary_kernel_body(sn, decls, indent))
    else:
        out.extend(_render_kernel_body(sn, decls, indent))
    return out


def _render_grouped(
    graph: GraphLowering,
    sn: Any,
    start_idx: int,
    sets: tuple[list[str], list[str], list[str], list[str]],
    decls: dict[str, str],
    indent: str = "    ",
    kernel_domains: bool = False,
) -> tuple[list[str], int]:
    """A scheduling container, whose children remain separate launches."""
    out = _render_kernel_interface(
        sn, "scheduling group", sets, indent, show_pattern=True
    )
    launched = 0
    children = list(getattr(sn, "snodes", ()) or ())
    child_sets = _node_sets(graph, children, externally_visible=set(sets[1]))
    for child, sets in zip(children, child_sets):
        out.append("")
        if _kernel_kind(child) == "nop":
            out.extend(f"{indent}{line}" for line in _render_nop(child, decls))
            continue
        block = _render_kernel(
            child,
            start_idx + launched,
            sets,
            decls,
            kernel_domains=kernel_domains,
        )
        out.extend(f"{indent}{line}" if line else line for line in block)
        launched += 1
    return out, launched


def _render_kernel_body(
    sn: Any,
    decls: dict[str, str],
    indent: str,
    renames: dict[Any, dict[sympy.Symbol, sympy.Expr]] | None = None,
    kernel_domain: bool = False,
    domain: _KernelDomain | None = None,
) -> list[str]:
    """Render the op sequence inside a (sub)kernel."""
    # Nodes selecting outputs of one reduction are grouped, so the shared body
    # appears once instead of once per output.
    groups: list[list[Any]] = []
    for sub in sn.get_nodes():
        op = getattr(sub, "node", None)
        if op is None:
            continue
        key = _shared_reduction_key(op)
        if key is not None and groups and groups[-1][0] == key:
            groups[-1][1].append(sub)
        else:
            groups.append([key, [sub]])

    out: list[str] = []
    for _key, subs in groups:
        blocks = []
        for sub in subs:
            block = _render_operation(
                sub.node,
                indent=indent,
                decls=decls,
                rename=renames.get(sub) if renames is not None else None,
                kernel_domain=kernel_domain,
                domain=domain,
            )
            name = sub.get_name()
            summary = next((ln.lstrip("# ") for ln in block if ln.startswith("#")), "")
            if summary.startswith(name):
                summary = summary[len(name) :].strip()
            rest = [ln for ln in block if not ln.startswith("#")]
            blocks.append((name, summary, rest[0] if rest else "", rest[1:]))

        names = ", ".join(n for n, _, _, _ in blocks)
        # Merge the written buffers onto one line; the reads are shared, so they
        # are listed once rather than repeated per output.
        lhs: list[str] = []
        reads = ""
        for _, _, h, _ in blocks:
            if not h:
                continue
            written, _, read = h.partition("  <-  ")
            lhs.append(written)
            # Grouped nodes share one inner_fn, so their read lists are equal;
            # taking the first verbatim avoids parsing shapes out of the text.
            reads = reads or read
        headers = ", ".join(lhs)
        if reads:
            headers += "  <-  " + reads
        summary = blocks[0][1]
        if len(blocks) > 1:
            # The selector differs per output; the reduction is shared.
            summary = re.sub(r"\.\w+$", "", summary)
        line = f"{indent}{names}:  {headers}".rstrip()
        if summary:
            line += f"      # {summary}"
        out.append("")
        out.append(line)

        if len(blocks) == 1:
            out.extend(f"{indent}{ln}" for ln in blocks[0][3])
            continue
        # Shared body once, then one store per output.
        shared = blocks[0][3][:-1]
        out.extend(f"{indent}{ln}" for ln in shared)
        for name, _, _, body in blocks:
            if body:
                out.append(f"{indent}{body[-1]}      # {name}")
    return out


def log_post_lowering_inductor_ir(graph: GraphLowering) -> None:
    if post_lowering_inductor_ir_log.isEnabledFor(logging.INFO):
        post_lowering_inductor_ir_log.info("\n%s", format_post_lowering_ir(graph))


def log_post_scheduler_inductor_ir(graph: GraphLowering) -> None:
    if post_scheduler_inductor_ir_log.isEnabledFor(logging.INFO):
        post_scheduler_inductor_ir_log.info("\n%s", format_post_lowering_ir(graph))


def log_post_fusion_inductor_ir(graph: GraphLowering) -> None:
    if post_fusion_inductor_ir_log.isEnabledFor(logging.INFO):
        post_fusion_inductor_ir_log.info("\n%s", format_post_lowering_ir(graph))


def format_post_lowering_ir(graph: GraphLowering) -> str:
    """Render the graph, either post-lowering or post scheduler-node construction.

    Which one depends on whether the scheduler has built its nodes yet; the
    header says which.
    """
    devices = _devices(graph)

    lines: list[str] = []
    if len(devices) == 1:
        lines.append(f"# device: {devices[0]}")
        lines.append("")

    syms, decl_parts, tagged = [], [], []
    for name, inp in graph.graph_inputs.items():
        node = inp
        while isinstance(node, (ir.TensorBox, ir.StorageBox)):
            node = node.data
        # A SymInt input is a bare sympy symbol rather than an IRNode, so it has
        # no dtype or layout to render. It binds the dynamic size that the tensor
        # declarations below refer to, so name the binding instead.
        if isinstance(node, sympy.Expr):
            syms.append(f"{name}: SymInt = {_expr(node)}")
            continue
        kind = type(node).__name__
        if kind != "InputBuffer":
            tagged.append(f"{name}: {_tensor_decl(node)}  # {kind}")
        else:
            decl_parts.append(f"{name}: {_tensor_decl(node)}")
    # Filled rather than one per line: a real model has dozens of weights, and
    # they would otherwise push the compute off the screen.
    lines.append("# inputs")
    # Symbols first: the tensor shapes that follow are written in terms of them.
    lines.extend(_fill(syms))
    lines.extend(_fill(decl_parts))
    lines.extend(tagged)

    consts = getattr(graph, "constants", None) or {}
    if consts:
        lines.append("")
        lines.append("# constants (frozen, known at compile time)")
        lines.extend(_fill(f"{n}: {_const_decl(t)}" for n, t in consts.items()))

    shared = _render_buffers(graph)
    if shared:
        lines.append("")
        lines.extend(shared)

    compute_lines: list[str] = []
    decls = _decl_map(graph)
    for parent, members in _tuple_members(graph).items():
        decls[parent] = f"tuple({', '.join(members)})"
    sched = getattr(graph, "scheduler", None)
    snodes = list(getattr(sched, "nodes", ()) or ()) if sched is not None else []
    if snodes:
        sets = _kernel_sets(graph, sched)
        kernel_domains = hasattr(sched, "num_orig_nodes")
        launched = 0
        for i, sn in enumerate(snodes):
            kind = _kernel_kind(sn)
            if kind == "nop":
                compute_lines.extend(_render_nop(sn, decls))
                compute_lines.append("")
                continue
            if kind == "grouped":
                grouped, child_launches = _render_grouped(
                    graph,
                    sn,
                    launched,
                    sets[i],
                    decls,
                    kernel_domains=kernel_domains,
                )
                compute_lines.extend(grouped)
                compute_lines.append("")
                launched += child_launches
                continue
            compute_lines.extend(
                _render_kernel(
                    sn,
                    launched,
                    sets[i],
                    decls,
                    kernel_domains=kernel_domains,
                )
            )
            compute_lines.append("")
            launched += 1
    else:
        for op in graph.operations:
            compute_lines.extend(_render_operation(op, decls=decls))
            compute_lines.append("")

    if not snodes:
        legend_head = LEGEND_POST_LOWERING
    elif hasattr(sched, "num_orig_nodes"):
        legend_head = LEGEND_POST_FUSION
    else:
        legend_head = LEGEND_POST_SCHEDULER
    legend = _legend(compute_lines, legend_head)
    if legend:
        insertion = 2 if len(devices) == 1 else 0
        lines[insertion:insertion] = [*legend, ""]

    lines.append("")
    lines.append("# compute")
    lines.append("")
    lines.extend(compute_lines)
    lines.append("# outputs")
    lines.append(
        "return (" + ", ".join(_output_name(o) for o in graph.graph_outputs) + ")"
    )
    return "\n".join(lines)
