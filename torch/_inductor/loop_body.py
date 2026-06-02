# mypy: allow-untyped-defs
from __future__ import annotations

import collections
import functools
import itertools
import re
from enum import auto, Enum
from typing import Any, NamedTuple, TYPE_CHECKING, TypeVar

import sympy

import torch.fx
from torch._dynamo.utils import identity
from torch.fx.proxy import Scope, TracerBase
from torch.utils._ordered_set import OrderedSet
from torch.utils._sympy.functions import Mod
from torch.utils._sympy.symbol import SymT

from . import config, dependencies
from .codegen.common import index_prevent_reordering
from .ops_handler import DefaultHandler, OpsHandler, WrapperHandler
from .utils import (
    cache_on_self,
    decompose_index,
    flatten_index,
    reduction_num_outputs,
    sympy_index_symbol_with_prefix,
    sympy_subs,
)
from .virtualized import ops, V


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


T = TypeVar("T")


class InterpreterShim(torch.fx.Interpreter):
    @staticmethod
    @functools.cache
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
        # pyrefly: ignore [bad-assignment]
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
    buffer_name: str | None
    mode: str | None  # V.ops.store(..., mode=mode)


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
    submodules: dict[str, Any]
    subblocks: dict[str, LoopBodyBlock]
    indirect_vars: list[sympy.Symbol]
    indirect_var_ranges: dict[sympy.Symbol, sympy.Expr]
    root_block: LoopBodyBlock
    memory_usage: dict[MemoryUsageType, list[MemoryEntry]]
    op_counts: collections.Counter[str]

    # defined only temporarily
    indexing_exprs_name: dict[sympy.Expr, str]

    @staticmethod
    def _wrap_int_to_sympy_integer(expr):
        # Static sizes can enter indexing expressions as Python ints.
        if type(expr) is int:
            return sympy.Integer(expr)
        return expr

    def __init__(
        self,
        fn,
        args,
        var_ranges,
        iter_vars,
        reduce_vars,
        allow_same_symbol_in_index=False,
    ):
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
            self._init_with_copy(fn, args, allow_same_symbol_in_index)
        else:
            self._init_with_tracing(fn, args)
            self.index_expr_usage: dict[torch.fx.Node, tuple[bool, bool]] | None = None

        self.indexing = None

    def get_original_num_rdims(self) -> int:
        assert self.has_partial_accumulate
        node = self.root_block.graph.find_nodes(
            op="call_method", target="partial_accumulate"
        )[0]
        meta = node.args[-1]
        return meta["num_reduction_dims"]

    def extract_pw_from_reduction(self):
        self.root_block = self.root_block.extract_pw_from_reduction()
        self.has_partial_accumulate = True
        self.iter_vars = self.iter_vars + self.reduce_vars
        self.reduce_vars = []
        self.sizes = (self.sizes[0] + self.sizes[1], tuple())
        return self

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
        self.has_partial_accumulate = bool(
            self.root_block.graph.find_nodes(
                op="call_method", target="partial_accumulate"
            )
        )
        del self.indexing_exprs_name  # not used after _init_with_tracing

    def _init_with_copy(self, other: LoopBody, args, allow_same_symbol_in_index):
        """
        _init_with_tracing() is slow, so this is a fast path in the case
        where we are just reordering/merging/splitting the args of an
        existing LoopBody.
        """
        indexing_exprs = other.indexing_from_args(args, allow_same_symbol_in_index)
        self.indexing_exprs = {
            name: self._wrap_int_to_sympy_integer(
                V.graph.sizevars.simplify_with_ranges(expr, self.var_ranges)
            )
            for name, expr in indexing_exprs.items()
        }
        self.subblocks = {k: v.clone(self) for k, v in other.subblocks.items()}
        self.indirect_vars = other.indirect_vars
        self.indirect_var_ranges = other.indirect_var_ranges
        self.memory_usage = other.memory_usage
        self.op_counts = other.op_counts
        self.root_block = other.root_block.clone(self)
        self.has_partial_accumulate = other.has_partial_accumulate
        self.index_expr_usage = other.index_expr_usage

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

        if iter_sizes == old_iter_sizes and reduce_sizes == old_reduce_sizes:
            return old_body

        (
            (
                iter_vars,
                reduce_vars,
            ),
            var_ranges,
        ) = dependencies.index_vars_no_squeeze(iter_sizes, reduce_sizes, prefix="p")
        new_body = LoopBody(
            old_body,
            [iter_reindex(iter_vars), reduce_reindex(reduce_vars)],
            var_ranges,
            iter_vars,
            reduce_vars,
            allow_same_symbol_in_index=True,
        )

        return new_body

    def expand_dimension_for_pointwise_node(
        self, dimension: int, new_range: int
    ) -> LoopBody:
        """
        Expand node on `dimension` to `new_range` and rely on index modular to avoid
        out-of-boundary access.
        """

        old_body = self
        old_sizes = self.sizes

        iter_size, reduce_size = old_sizes
        original_range = iter_size[dimension]
        new_iter_size = list(iter_size)
        new_iter_size[dimension] = new_range
        new_sizes = (new_iter_size, reduce_size)

        (iter_vars, reduce_vars), var_ranges = dependencies.index_vars_no_squeeze(
            *new_sizes,
            prefix="t",  # type: ignore[arg-type]
        )

        def new_body(*indices: Sequence[sympy.Expr]) -> Any:
            index = [*itertools.chain.from_iterable(indices)]
            assert len(index) == len(iter_size) + len(reduce_size)
            iter_idx = index[: len(iter_size)]
            reduce_idx = index[len(iter_size) :]

            new_iter_idx = list(iter_idx)
            new_iter_idx[dimension] = Mod(iter_idx[dimension], original_range)

            return old_body(new_iter_idx, reduce_idx)

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

    def reindex_iter_loops(self, new_iter_sizes: Sequence[sympy.Expr]) -> LoopBody:
        """
        Reindex iteration loops into a different factorization of the same
        total numel. For example, [1024, 8192] -> [65536, 128].

        The old iteration vars are expressed as functions of the new vars via
        FloorDiv and ModularIndexing on the flat index.
        """
        old_body = self
        old_iter_sizes = self.sizes[0]
        reduce_sizes = self.sizes[1]

        new_sizes = (list(new_iter_sizes), list(reduce_sizes))

        (iter_vars, reduce_vars), var_ranges = dependencies.index_vars_no_squeeze(
            *new_sizes,
            prefix="t",  # type: ignore[arg-type]
        )

        def new_body(*indices: Sequence[sympy.Expr]) -> Any:
            index = [*itertools.chain.from_iterable(indices)]
            new_iter_idx = index[: len(new_iter_sizes)]
            reduce_idx = index[len(new_iter_sizes) :]
            flat = flatten_index(new_iter_idx, new_iter_sizes)
            old_iter_idx = decompose_index(flat, old_iter_sizes)
            return old_body(old_iter_idx, list(reduce_idx))

        loop_body = LoopBody(
            new_body, (iter_vars, reduce_vars), var_ranges, iter_vars, reduce_vars
        )

        (iter_vars2, reduce_vars2), var_ranges2 = dependencies.index_vars_no_squeeze(
            *new_sizes,
            prefix="p",  # type: ignore[arg-type]
        )
        return LoopBody(
            loop_body,
            (iter_vars2, reduce_vars2),
            var_ranges2,
            iter_vars2,
            reduce_vars2,
        )

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
            prefix="p",  # type: ignore[arg-type]
        )

        inverse_order = {b: a for a, b in enumerate(new_order)}
        inverse_order = [inverse_order[i] for i in range(len(new_order))]

        def new_body(*indices: Sequence[sympy.Expr]) -> Any:
            index = [*itertools.chain.from_iterable(indices)]
            assert len(index) == len(iter_size) + len(reduce_size)
            iter_idx = index[: len(iter_size)]
            reduce_idx = index[len(iter_size) :]
            iter_idx = [iter_idx[i] for i in inverse_order]
            return old_body(iter_idx, reduce_idx, allow_same_symbol_in_index=True)

        return LoopBody(
            new_body,
            (iter_vars, reduce_vars),
            var_ranges,
            iter_vars,
            reduce_vars,
        )

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

    def get_all_read_expr(self, buffer_name):
        # reversed to match old behavior
        out = []
        for entry in reversed(self.memory_usage[MemoryUsageType.LOAD]):
            if entry.buffer_name == buffer_name:
                out.append(self.indexing_exprs[entry.index_name])
        return out

    def get_write_exprs(self):
        return [
            self.indexing_exprs[entry.index_name]
            for entry in itertools.chain(
                self.memory_usage[MemoryUsageType.STORE],
                self.memory_usage[MemoryUsageType.STORE_REDUCTION],
            )
        ]

    def get_all_write_expr(self, buffer_name):
        out = []
        for entry in itertools.chain(
            self.memory_usage[MemoryUsageType.STORE],
            self.memory_usage[MemoryUsageType.STORE_REDUCTION],
        ):
            if entry.buffer_name == buffer_name:
                out.append(self.indexing_exprs[entry.index_name])
        return out

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
        buffer_name: str | None = None,
        mode: str | None = None,
    ):
        expr = self._wrap_int_to_sympy_integer(expr)
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
        # pyrefly: ignore [bad-assignment]
        self.indexing = {k: sympy_subs(v, {old: new}) for k, v in self.indexing.items()}

    def get_index(self, name):
        assert self.indexing is not None
        return self.indexing[name]

    def indexing_from_args(self, indices, allow_same_symbol_in_index=False):
        index = [*itertools.chain.from_iterable(indices)]
        assert len(index) == len(self.var_ranges), (index, self.var_ranges)
        assert allow_same_symbol_in_index or all(
            v not in self.var_ranges for v in index
        ), f"{self.var_ranges=}, {indices=}"

        replacements = dict(zip(self.var_ranges.keys(), index))
        return {
            name: sympy_subs(expr, replacements)
            for name, expr in self.indexing_exprs.items()
        }

    def __call__(self, *indices, allow_same_symbol_in_index=False):
        self.indexing = self.indexing_from_args(indices, allow_same_symbol_in_index)
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
    LoopBodyBlock, however in the case of ops.masked() the masked out
    operations will manifest as an extra LoopBodyBlock.
    """

    def __init__(self, body: LoopBody, fn: Callable[..., Any], args: list[Any]):
        self.body = body

        tracer = LightTracer()
        proxy_ops = tracer.create_proxy("placeholder", "ops", (), {})

        from .index_propagation import IndexPropagation

        handler: Any = CountOps(
            CaptureIndexing(
                # pyrefly: ignore[bad-argument-type]
                proxy_ops,
                body,
                tracer,
            ),
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

    def extract_pw_from_reduction(self):
        red = None
        store = None
        for node in self.graph.nodes:
            if node.target == "reduction":
                assert not red
                red = node
            if node.target == "store_reduction":
                assert not store
                store = node
        assert red
        assert store
        reduction_type = red.args[-2]
        red_arg = red.args[-1]
        buf = store.args[1]
        ops = store.args[0]

        extra_meta = {
            "num_reduction_dims": len(self.body.reduce_vars),
        }
        with self.graph.inserting_after(store):
            self.graph.call_method(
                "partial_accumulate", (ops, buf, reduction_type, red_arg, extra_meta)
            )
        self.graph.erase_node(store)
        self.graph.erase_node(red)
        return self

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
        size = self.body._wrap_int_to_sympy_integer(size)
        size = self._add_index(size, MemoryUsageType.CHECK_BOUNDS)
        return self._inner.check_bounds(index, size, lower, upper)

    def bucketize(
        self,
        values: T,
        boundaries: tuple[str, sympy.Expr, sympy.Expr, sympy.Expr],
        boundary_indices: T,
        indexing_dtype: torch.dtype,
        right: bool,
        sorter: tuple[str, sympy.Expr] | None = None,
        sorter_indices: T | None = None,
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


def _has_int64_iota_ancestor(dynamo_fx_node: torch.fx.Node | None) -> bool:
    """
    Trace backwards through Dynamo FX graph to find int64 iota ancestors.

    Args:
        dynamo_fx_node: Starting node in Dynamo FX graph (origin_node from IR)

    Returns:
        True if this node has an int64 iota ancestor that's used for int64)
    """
    if not dynamo_fx_node:
        return False

    visited: OrderedSet[torch.fx.Node] = OrderedSet()

    def trace_backwards(node: torch.fx.Node) -> bool:
        if node in visited or node.op == "placeholder":
            return False
        visited.add(node)

        # Check if this is int64 iota used for int64 computation
        if node.target == torch.ops.prims.iota.default:
            val = node.meta.get("val")
            if val is not None and hasattr(val, "dtype") and val.dtype == torch.int64:
                # Check if any user is NOT convert_element_type (int64 usage)
                for user in node.users:
                    if user.target != torch.ops.prims.convert_element_type.default:
                        return True
            return False

        # Recursively check inputs
        for arg in node.args:
            if isinstance(arg, torch.fx.Node):
                if trace_backwards(arg):
                    return True
        for kwarg in node.kwargs.values():
            if isinstance(kwarg, torch.fx.Node):
                if trace_backwards(kwarg):
                    return True

        return False

    return trace_backwards(dynamo_fx_node)


def analyze_index_expr_usage(
    loop_body: LoopBody,
    ir_node: Any = None,
) -> dict[torch.fx.Node, tuple[bool, bool]]:
    """
    Analyze the LoopBody's FX graph to determine which index_expr nodes
    are used ONLY for values and which have int64 origins.

    Algorithm:
    1. Scan for terminal operations (load, store, store_reduction)
    2. Recursively parse search ancestors for indexing or values usage
    3. Stop at barriers
    4. Trace back through IR origin_node to find int64 iota ancestors

    Args:
        loop_body: The LoopBody to analyze
        ir_node: Optional IR node (e.g., Pointwise) with origin_node to Dynamo FX graph

    Returns:
        dict mapping FX node -> (is_used_only_for_values, has_int64_iota_ancestor)
    """
    # Track what each FX node is used for
    used_for_indexing: OrderedSet[torch.fx.Node] = OrderedSet()
    used_for_values: OrderedSet[torch.fx.Node] = OrderedSet()
    used_by_comparison: OrderedSet[torch.fx.Node] = OrderedSet()

    def mark_ancestors(
        node: torch.fx.Node,
        target_set: OrderedSet[torch.fx.Node],
        visited: OrderedSet[torch.fx.Node] | None = None,
    ) -> None:
        """
        Mark all ancestors by adding them to target_set, STOP at load barriers.

        Load operations act as barriers: they break both indexing and value chains
        because the loaded value is independent from the index used to load it.
        """
        if visited is None:
            visited = OrderedSet()
        if node in visited or node.op == "placeholder":
            return
        visited.add(node)

        if node.target == "load":
            return  # Don't propagate backward through load

        target_set.add(node)

        # Trace backward through inputs
        for arg in node.args:
            if isinstance(arg, torch.fx.Node):
                mark_ancestors(arg, target_set, visited)
        for kwarg in node.kwargs.values():
            if isinstance(kwarg, torch.fx.Node):
                mark_ancestors(kwarg, target_set, visited)

    for node in loop_body.root_block.graph.nodes:
        # These sproduce booleans not values.
        if node.target in ("lt", "le", "gt", "ge", "eq", "ne"):
            for arg in node.args:
                if isinstance(arg, torch.fx.Node):
                    used_by_comparison.add(arg)

        # Trace terminal operations
        if node.target in ("store", "store_reduction"):
            if len(node.args) > 2 and isinstance(node.args[2], torch.fx.Node):
                mark_ancestors(node.args[2], used_for_indexing)
            if len(node.args) > 3 and isinstance(node.args[3], torch.fx.Node):
                mark_ancestors(node.args[3], used_for_values)

        elif node.target == "load":
            if len(node.args) > 2 and isinstance(node.args[2], torch.fx.Node):
                mark_ancestors(node.args[2], used_for_indexing)

    # Build result: for each index_expr node, determine usage and int64 iota ancestry
    result: dict[torch.fx.Node, tuple[bool, bool]] = {}

    origin_fx_node = None
    if ir_node and hasattr(ir_node, "get_origin_node"):
        origin_fx_node = ir_node.get_origin_node()

    for node in loop_body.root_block.graph.nodes:
        if node.target == "index_expr":
            in_values = node in used_for_values
            in_indexing = node in used_for_indexing
            in_comparison = node in used_by_comparison
            is_for_values_only = in_values and not in_indexing and not in_comparison

            has_int64_iota = False
            if is_for_values_only and origin_fx_node:
                # The origin_node is for the entire buffer/operation
                # Trace backwards from it through the Dynamo graph
                has_int64_iota = _has_int64_iota_ancestor(origin_fx_node)

            result[node] = (is_for_values_only, has_int64_iota)

    return result


def get_index_expr_int64_usage() -> tuple[bool, bool]:
    """
    Determine if index_expr should use int64 based on usage analysis.

    Returns:
        (is_for_values_only, has_int64_iota_ancestor)
        - is_for_values_only: True if this index_expr is only used for tensor values
        - has_int64_iota_ancestor: True if traced back to int64 iota operation
    """
    from .virtualized import V

    if hasattr(V.interpreter, "current_node") and hasattr(
        V.kernel.current_node, "_body"
    ):
        current_fx_node = V.interpreter.current_node
        loop_body = V.kernel.current_node._body
        scheduler_node = getattr(V.kernel.current_node, "node", None)
        ir_node = getattr(scheduler_node, "data", None)

        # Lazy evaluation: run analysis once per LoopBody and cache result
        if loop_body.index_expr_usage is None:
            loop_body.index_expr_usage = analyze_index_expr_usage(loop_body, ir_node)

        # Look up usage info for this specific FX node
        return loop_body.index_expr_usage.get(current_fx_node, (False, False))

    return (False, False)
