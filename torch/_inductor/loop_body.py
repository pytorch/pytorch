# mypy: allow-untyped-defs
from __future__ import annotations

import collections
import functools
import itertools
import re
from enum import auto, Enum
from typing import Any, Callable, NamedTuple, Optional, TYPE_CHECKING, TypeVar

import sympy

import torch.fx
from torch._dynamo.utils import identity
from torch.fx.proxy import Scope, TracerBase
from torch.utils._sympy.symbol import SymT

from . import config, dependencies
from .codegen.common import index_prevent_reordering
from .ops_handler import DefaultHandler, OpsHandler, WrapperHandler
from .utils import (
    cache_on_self,
    reduction_num_outputs,
    sympy_index_symbol_with_prefix,
    sympy_subs,
)
from .virtualized import ops, V


if TYPE_CHECKING:
    from collections.abc import Sequence


T = TypeVar("T")


class InterpreterShim(torch.fx.Interpreter):
    @staticmethod
    @functools.lru_cache(None)
    def _dummy_gm():
        return torch.fx.symbolic_trace(identity)

    def __init__(self, graph, submodules):
        # call super() with a placeholder to avoid constructing a
        # GraphModule which is very expensive (it does codegen).
        super().__init__(self._dummy_gm(), garbage_collect_values=False)
        self.module = self  # type: ignore[assignment]
        self.graph = graph
        self.submodules = submodules
        self.extra_traceback = False
        self.fetch_attr = submodules.__getitem__  # type: ignore[method-assign]
        self.current_node = None

    def run_node(self, n: torch.fx.Node) -> Any:
        self.current_node = n
        return super().run_node(n)

    def run(self, *args, **kwargs):
        with V.set_interpreter_handler(self):
            return super().run(*args, **kwargs)


# We don't need the nn.Module and constant handling in Tracer
class LightTracer(TracerBase):
    def __init__(self):
        super().__init__()
        self.graph = torch.fx.Graph(tracer_cls=self.__class__)  # type: ignore[arg-type]
        self.scope = Scope("", None)
        self.module_stack = {}  # type: ignore[assignment]
        self.node_name_to_scope = {}


class MemoryEntry(NamedTuple):
    index_name: str  # LoopBody.indexing_exprs[index_name]
    buffer_name: Optional[str]
    mode: Optional[str]  # V.ops.store(..., mode=mode)


class MemoryUsageType(Enum):
    # These are 1:1 with the opcode generating the usage
    LOAD = auto()
    LOAD_SEED = auto()
    STORE = auto()
    STORE_REDUCTION = auto()
    INDEX_EXPR = auto()
    CHECK_BOUNDS = auto()
    BUCKETIZE = auto()


class LoopBody:
    """
    Captures the body of a Loops subclass into an FX graph.  Persists any
    indexing simplifications and makes it easier to analyze loop bodies.
    """

    indexing_exprs: dict[str, sympy.Expr]
    indexing_exprs_name: dict[sympy.Expr, str]
    submodules: dict[str, Any]
    subblocks: dict[str, LoopBodyBlock]
    indirect_vars: list[sympy.Symbol]
    indirect_var_ranges: dict[sympy.Symbol, sympy.Expr]
    root_block: LoopBodyBlock
    memory_usage: dict[MemoryUsageType, list[MemoryEntry]]
    op_counts: collections.Counter[str]

    def __init__(self, fn, args, var_ranges, iter_vars, reduce_vars):
        super().__init__()

        _flat_sizes = tuple(var_ranges.values())
        self.sizes = (
            _flat_sizes[: len(iter_vars)],
            _flat_sizes[len(iter_vars) :],
        )

        self.iter_vars = iter_vars
        self.reduce_vars = reduce_vars
        self.var_ranges = var_ranges

        if isinstance(fn, LoopBody):
            self._init_with_copy(fn, args)
        else:
            self._init_with_tracing(fn, args)

        self.indexing = None

    def _init_with_tracing(self, fn, args):
        """Do an FX trace of an arbitrary callable to construct self"""
        self.indexing_exprs = {}
        self.indexing_exprs_name = {}
        self.submodules = {"get_index": self.get_index}
        self.subblocks = {}
        self.indirect_vars = []
        self.indirect_var_ranges: dict[sympy.Symbol, sympy.Expr] = {}
        self.memory_usage = {t: [] for t in MemoryUsageType}
        self.op_counts = collections.Counter()
        self.root_block = LoopBodyBlock(self, fn, args)  # traces
        del self.indexing_exprs_name  # not used after _init_with_tracing

    def _init_with_copy(self, other: LoopBody, args):
        """
        _init_with_tracing() is slow, so this is a fast path in the case
        where we are just reordering/merging/splitting the args of an
        existing LoopBody.
        """
        indexing_exprs = other.indexing_from_args(args)
        self.indexing_exprs = {
            name: V.graph.sizevars.simplify_with_ranges(expr, self.var_ranges)
            for name, expr in indexing_exprs.items()
        }
        self.subblocks = {k: v.clone(self) for k, v in other.subblocks.items()}
        self.indirect_vars = other.indirect_vars
        self.indirect_var_ranges = other.indirect_var_ranges
        self.memory_usage = other.memory_usage
        self.op_counts = other.op_counts
        self.root_block = other.root_block.clone(self)

        submodules = {**other.submodules}
        submodules.pop("get_index")
        self.submodules = {
            "get_index": self.get_index,
            **{k: v.clone(self) for k, v in submodules.items()},  # type: ignore[attr-defined]
        }

    def has_op(self, name: str):
        return self.op_counts.get(name, 0) > 0

    def merge_loops(self) -> LoopBody:
        """
        Merge both iteration and reduction loops and return a new LoopBody.
        """
        old_body = self
        old_sizes = self.sizes
        old_iter_vars, old_reduce_vars = old_body.vars
        old_iter_sizes, old_reduce_sizes = old_sizes

        index_exprs = [*old_body.indexing_exprs.values()]

        iter_sizes, iter_reindex, _ = V.graph.sizevars._simplify_loops(
            old_iter_vars,
            old_iter_sizes,
            index_prevent_reordering(index_exprs, old_iter_vars, old_iter_sizes),
        )

        reduce_sizes, reduce_reindex, _ = V.graph.sizevars._simplify_loops(
            old_reduce_vars,
            old_reduce_sizes,
            index_prevent_reordering(index_exprs, old_reduce_vars, old_reduce_sizes),
        )

        # if iter_sizes == old_iter_sizes:
        #     # no dimensions get merged.
        #     return old_sizes, old_body

        # Note: if no dimension get merges, the symbol prefix will
        # remain 'y'. But if we merge dimensions, we change prefix to
        # 'z'. If this is an issue, we can always retrace the LoopBody
        # to change symbol prefix to 'z'.
        #
        # There is indeed an issue due to symbol name conflicting.
        # y0 maybe reused for the y dimension later.
        (
            (
                iter_vars,
                reduce_vars,
            ),
            var_ranges,
        ) = dependencies.index_vars_no_squeeze(iter_sizes, reduce_sizes, prefix="t")
        new_body = LoopBody(
            old_body,
            [iter_reindex(iter_vars), reduce_reindex(reduce_vars)],
            var_ranges,
            iter_vars,
            reduce_vars,
        )

        # use the original symbol prefix
        # Can try to optimize if this is a bottleneck for compilation time
        (iter_vars2, reduce_vars2), var_ranges2 = dependencies.index_vars_no_squeeze(
            iter_sizes, reduce_sizes, prefix="p"
        )
        new_body2 = LoopBody(
            new_body, (iter_vars2, reduce_vars2), var_ranges2, iter_vars2, reduce_vars2
        )
        return new_body2

    def reorder_iter_loops(self, new_order) -> LoopBody:
        """
        Reorder iteration loops and return a new LoopBody.
        """
        from .ir import same_reorder

        old_body = self
        old_sizes = self.sizes
        assert len(old_sizes[0]) == len(new_order)
        reorder_fn = same_reorder(new_order)

        iter_size, reduce_size = old_sizes
        new_iter_size = reorder_fn(iter_size)

        new_sizes = (new_iter_size, reduce_size)

        (iter_vars, reduce_vars), var_ranges = dependencies.index_vars_no_squeeze(
            *new_sizes,
            prefix="t",  # type: ignore[arg-type]
        )

        inverse_order = {b: a for a, b in enumerate(new_order)}
        inverse_order = [inverse_order[i] for i in range(len(new_order))]

        def new_body(*indices: Sequence[sympy.Expr]) -> Any:
            index = [*itertools.chain.from_iterable(indices)]
            assert len(index) == len(iter_size) + len(reduce_size)
            iter_idx = index[: len(iter_size)]
            reduce_idx = index[len(iter_size) :]
            iter_idx = [iter_idx[i] for i in inverse_order]
            return old_body(iter_idx, reduce_idx)

        loop_body = LoopBody(
            new_body, (iter_vars, reduce_vars), var_ranges, iter_vars, reduce_vars
        )

        # use the original symbol prefix so we can do multiple round of reordering
        (iter_vars2, reduce_vars2), var_ranges2 = dependencies.index_vars_no_squeeze(
            *new_sizes,
            prefix="p",  # type: ignore[arg-type]
        )
        new_body = LoopBody(
            loop_body, (iter_vars2, reduce_vars2), var_ranges2, iter_vars2, reduce_vars2
        )
        return new_body

    @property
    def vars(self):
        assert self.iter_vars is not None
        assert self.reduce_vars is not None
        return self.iter_vars, self.reduce_vars

    @cache_on_self
    def get_nodes(self):
        all_graphs = itertools.chain(
            (self.root_block.graph,),
            (block.graph for block in self.subblocks.values()),
        )
        return [node for graph in all_graphs for node in graph.nodes]

    @cache_on_self
    def bounds(self):
        # Doing a local import to avoid dumping all the code here
        from .bounds import BoundVars

        return BoundVars(self)

    def get_read_expr(self, buffer_name):
        # reversed to match old behavior
        for entry in reversed(self.memory_usage[MemoryUsageType.LOAD]):
            if entry.buffer_name == buffer_name:
                return self.indexing_exprs[entry.index_name]
        raise KeyError(buffer_name)

    def get_write_expr(self, buffer_name):
        for entry in itertools.chain(
            self.memory_usage[MemoryUsageType.STORE],
            self.memory_usage[MemoryUsageType.STORE_REDUCTION],
        ):
            if entry.buffer_name == buffer_name:
                return self.indexing_exprs[entry.index_name]
        raise KeyError(buffer_name)

    def get_read_exprs(self):
        return [
            self.indexing_exprs[entry.index_name]
            for entry in self.memory_usage[MemoryUsageType.LOAD]
        ]

    def get_write_exprs(self):
        return [
            self.indexing_exprs[entry.index_name]
            for entry in itertools.chain(
                self.memory_usage[MemoryUsageType.STORE],
                self.memory_usage[MemoryUsageType.STORE_REDUCTION],
            )
        ]

    def debug_str(self):
        lines = [f"var_ranges = {dict(self.var_ranges)}"]
        lines.extend([f"{name} = {val}" for name, val in self.indexing_exprs.items()])
        lines.extend(
            [
                block.debug_str(name)
                for name, block in itertools.chain(
                    [("body", self.root_block)], self.subblocks.items()
                )
            ]
        )
        return "\n".join(lines)

    def is_memory_copy(self) -> bool:
        """
        True of this contains only a single loads and store.
        Note, this could involve a layout change.
        """
        return (
            len(self.memory_usage[MemoryUsageType.LOAD]) == 1
            and len(self.memory_usage[MemoryUsageType.STORE]) == 1
            and len(self.submodules) == 1  # get_index
            and self.root_block.contains_only_ops(("load", "store"))
        )

    __repr__ = debug_str

    def add_index_expr(
        self,
        expr: sympy.Expr,
        mtype: MemoryUsageType,
        buffer_name: Optional[str] = None,
        mode: Optional[str] = None,
    ):
        name = self.indexing_exprs_name.get(expr)
        if not name:
            name = f"index{len(self.indexing_exprs)}"
            self.indexing_exprs_name[expr] = name
            self.indexing_exprs[name] = expr
        self.memory_usage[mtype].append(MemoryEntry(name, buffer_name, mode))
        return name

    def add_submodule(self, block, prefix):
        """Not actually for nn.Modules, but subblocks in generated code are mapped to FX call_module opcodes"""
        if prefix[-1].isnumeric() and prefix not in self.submodules:
            name = prefix
        else:
            name = f"{prefix}{len(self.submodules)}"
        self.submodules[name] = block
        return name

    def add_indirect(self, size):
        var = sympy_index_symbol_with_prefix(SymT.INDIRECT, len(self.indirect_vars))
        assert var not in self.indirect_var_ranges
        self.indirect_vars.append(var)
        self.indirect_var_ranges[var] = size
        return var

    def replace_indirect(self, old, new):
        """Swap in a variable used in indirect indexing"""
        if str(old) == str(new):
            return
        assert self.indexing is not None
        self.indexing = {k: sympy_subs(v, {old: new}) for k, v in self.indexing.items()}

    def get_index(self, name):
        assert self.indexing is not None
        return self.indexing[name]

    def indexing_from_args(self, indices):
        index = [*itertools.chain.from_iterable(indices)]
        assert len(index) == len(self.var_ranges), (index, self.var_ranges)
        assert all(v not in self.var_ranges for v in index), (
            f"{self.var_ranges=}, {indices=}"
        )
        replacements = dict(zip(self.var_ranges.keys(), index))
        return {
            name: sympy_subs(expr, replacements)
            for name, expr in self.indexing_exprs.items()
        }

    def __call__(self, *indices):
        self.indexing = self.indexing_from_args(indices)
        result = self.root_block()
        self.indexing = None
        return result

    def bind_set_indirect_shim(self, var, size, check, wrap_neg):
        def set_indirect(new_var):
            self.replace_indirect(
                var, V.ops.indirect_indexing(new_var, size, check, wrap_neg)
            )

        set_indirect.clone = functools.partial(  # type: ignore[attr-defined]
            LoopBody.bind_set_indirect_shim,
            var=var,
            size=size,
            check=check,
            wrap_neg=wrap_neg,
        )
        return set_indirect

    def bind_scan_shim(self, combine_fn):
        def shim(dtypes, values):
            return V.ops.scan(dtypes, combine_fn, values)

        shim.clone = functools.partial(LoopBody.bind_scan_shim, combine_fn=combine_fn)  # type: ignore[attr-defined]
        return shim

    def bind_masked_shim(self, name):
        def shim(mask, other):
            return V.ops.masked(mask, self.subblocks[name], other)

        shim.clone = functools.partial(LoopBody.bind_masked_shim, name=name)  # type: ignore[attr-defined]
        return shim


class LoopBodyBlock:
    """
    Captures the body of a Loops subclass into an FX graph.
    In normal cases there will be a 1:1 mapping between LoopBody and
    LoopBodyBlock, hower in the case of ops.masked() the masked out
    operations will manifest as an extra LoopBodyBlock.
    """

    def __init__(self, body: LoopBody, fn: Callable[..., Any], args: list[Any]):
        self.body = body

        tracer = LightTracer()
        proxy_ops = tracer.create_proxy("placeholder", "ops", (), {})

        from .index_propagation import IndexPropagation

        handler: Any = CountOps(
            CaptureIndexing(proxy_ops, body, tracer),
            body.op_counts,
        )
        if config.constant_and_index_propagation:
            handler = IndexPropagation(
                handler, self.body.var_ranges, self.body.indirect_var_ranges
            )

        with V.set_ops_handler(handler):
            # This indirection is just a cute way to get IndexPropagation to
            # unwrap the return value.
            ops.output(fn(*args))
        self.graph = tracer.graph

    def __call__(self):
        graph = self.graph
        submodules = self.body.submodules

        return InterpreterShim(graph, submodules).run(V.get_ops_handler())

    def debug_str(self, name="block"):
        code = torch.fx.GraphModule(self.body.submodules, self.graph).code
        return re.sub(
            # strip `; del var0` suffixes to make output prettier
            r";[^\n]*",
            "",
            code.strip().replace("def forward(", f"def {name}("),
        )

    def contains_only_ops(self, allowed_ops) -> bool:
        return all(
            node.target in allowed_ops
            for node in self.graph.find_nodes(op="call_method")
        )

    def clone(self, body: LoopBody):
        """Shallow copy with a new parent LoopBody"""
        copy = LoopBodyBlock.__new__(LoopBodyBlock)
        copy.__dict__.update({**self.__dict__, "body": body})
        return copy


class CountOps(DefaultHandler):
    def __init__(self, inner: OpsHandler[Any], counts: collections.Counter[str]):
        self._inner = inner
        self._counts = counts

    def _default(self, name: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        self._counts[name] += 1
        return getattr(self._inner, name)(*args, **kwargs)


class CaptureIndexing(WrapperHandler):
    name = "CaptureIndexing"

    def __init__(
        self,
        inner: OpsHandler[Any],
        body: LoopBody,
        tracer: LightTracer,
    ):
        super().__init__(inner)
        self.body = body
        self.tracer = tracer

    def _add_index(self, expr: sympy.Expr, mtype: MemoryUsageType, **kwargs: Any):
        return self.tracer.create_proxy(
            "call_module",
            "get_index",
            (self.body.add_index_expr(expr, mtype, **kwargs),),
            {},
        )

    def _simplify(self, expr: sympy.Expr) -> sympy.Expr:
        return V.graph.sizevars.simplify_with_ranges(expr, self.body.var_ranges)

    def load(self, name: str, index: sympy.Expr):
        index = self._simplify(index)
        index = self._add_index(index, MemoryUsageType.LOAD, buffer_name=name)
        return self._inner.load(name, index)

    def load_seed(self, name: str, index: int):
        assert isinstance(index, int)
        self.body.add_index_expr(
            sympy.Integer(index), MemoryUsageType.LOAD_SEED, buffer_name=name
        )
        return self._inner.load_seed(name, index)

    def store(self, name, index, value, mode=None):
        index = self._simplify(index)
        index = self._add_index(
            index, MemoryUsageType.STORE, buffer_name=name, mode=mode
        )
        return self._inner.store(name, index, value, mode)

    def store_reduction(self, name, index, value):
        index = self._simplify(index)
        index = self._add_index(
            index, MemoryUsageType.STORE_REDUCTION, buffer_name=name
        )
        return self._inner.store_reduction(name, index, value)

    def reduction(self, dtype, src_dtype, reduction_type, value):
        result = self._inner.reduction(dtype, src_dtype, reduction_type, value)
        num_outputs = reduction_num_outputs(reduction_type)
        if num_outputs > 1:
            return tuple(result[i] for i in range(num_outputs))
        return result

    def index_expr(self, index, dtype):
        index = self._simplify(index)
        if isinstance(index, (int, sympy.Integer)):
            return self._inner.constant(int(index), dtype)
        index = self._add_index(index, MemoryUsageType.INDEX_EXPR)
        return self._inner.index_expr(index, dtype)

    def check_bounds(self, index, size, lower, upper):
        index = self._simplify(index)
        index = self._add_index(index, MemoryUsageType.CHECK_BOUNDS)
        size = self._add_index(size, MemoryUsageType.CHECK_BOUNDS)
        return self._inner.check_bounds(index, size, lower, upper)

    def bucketize(
        self,
        values: T,
        boundaries: tuple[str, sympy.Expr, sympy.Expr, sympy.Expr],
        boundary_indices: T,
        indexing_dtype: torch.dtype,
        right: bool,
        sorter: Optional[tuple[str, sympy.Expr]] = None,
        sorter_indices: Optional[T] = None,
    ) -> T:
        """
        See [Note: Inductor bucketize op]
        """
        boundaries = (
            boundaries[0],
            self._add_index(
                boundaries[1],
                MemoryUsageType.BUCKETIZE,
                buffer_name=boundaries[0],
            ),
            self._add_index(
                boundaries[2],
                MemoryUsageType.BUCKETIZE,
                buffer_name=boundaries[0],
            ),
            self._add_index(
                boundaries[3],
                MemoryUsageType.BUCKETIZE,
                buffer_name=boundaries[0],
            ),
        )
        if sorter is not None:
            sorter = (
                sorter[0],
                self._add_index(
                    sorter[1], MemoryUsageType.BUCKETIZE, buffer_name=sorter[0]
                ),
            )

        return self._inner.bucketize(
            values,
            boundaries,
            boundary_indices,
            indexing_dtype,
            right,
            sorter,
            sorter_indices,
        )

    def masked(self, mask_proxy, masked_body: Callable[..., Any], other_proxy):
        """
        Recursively capture the masked out body in another LoopBodyBlock
        """
        name = self.body.add_submodule(None, "masked_subblock")
        self.body.submodules[name] = self.body.bind_masked_shim(name)
        self.body.subblocks[name] = LoopBodyBlock(self.body, masked_body, [])
        return self.tracer.create_proxy(
            "call_module", name, (mask_proxy, other_proxy), {}
        )

    def scan(
        self,
        dtype_proxy,
        combine_fn: Callable[[tuple[Any, ...], tuple[Any, ...]], tuple[Any, ...]],
        value_proxy,
    ):
        shim = self.body.bind_scan_shim(combine_fn)
        name = self.body.add_submodule(shim, "scan")
        result = self.tracer.create_proxy(
            "call_module",
            name,
            (dtype_proxy, value_proxy),
            {},
        )
        # Proxies are iterable, but some methods expect tuples/lists
        return tuple(result[i] for i in range(len(value_proxy)))

    def sort(self, dtypes, values, stable, descending):
        result = self._inner.sort(dtypes, values, stable, descending)
        # Proxies are iterable, but some methods expect tuples/lists
        return tuple(result[i] for i in range(len(values)))

    def frexp(self, value_proxy):
        result = self._inner.frexp(value_proxy)
        # Proxies are iterable, but some methods expect tuples/lists
        return (result[0], result[1])

    def indirect_indexing(self, index_proxy, size, check=True, wrap_neg=True):
        """
        Flow data from tensors into indexing formulas.
        Introduce a call_module to update the indexing.
        """

        var = self.body.add_indirect(size)
        set_indirect = self.body.bind_set_indirect_shim(var, size, check, wrap_neg)
        self.tracer.create_proxy(
            "call_module",
            self.body.add_submodule(set_indirect, f"set_{var}"),
            (index_proxy,),
            {},
        )
        return var

    def output(self, *result):
        self.tracer.create_proxy("output", "output", result, {})
