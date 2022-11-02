import collections
import contextlib
import dataclasses
import functools
import itertools
import logging
import math
import operator
from typing import Dict, List

import sympy

import torch

from .. import config, ir, scheduler
from ..ir import ReductionHint
from ..utils import (
    free_symbol_startswith,
    instance_descriptor,
    sympy_product,
    sympy_subs,
    sympy_symbol,
)
from ..virtualized import ops, V
from .common import (
    DeferredLine,
    ExprPrinter,
    IndentedBuffer,
    index_prevent_reordering,
    Kernel,
    OpOverrides,
    SizeArg,
    TensorArg,
)

log = logging.getLogger(__name__)


def signature_of(arg):
    from triton.runtime.jit import JITFunction

    if isinstance(arg, TensorArg):
        tye = JITFunction._type_of(arg.dtype)
        if V.graph.is_unspec_arg(arg.buffer):
            # had unwrapped 0d tensor as scalar
            new_tye = tye.lstrip("*")
            if new_tye in ["fp16", "bf16"]:
                return "fp32"
            else:
                return new_tye
        else:
            return tye
    if isinstance(arg, SizeArg):
        return JITFunction._key_of(V.graph.sizevars.size_hint(arg.expr))
    raise NotImplementedError(f"unhandled {type(arg)}: {arg}")


def config_of(args):
    from ..compile_fx import ALIGNMENT

    def is_aligned(x):
        if isinstance(x, TensorArg):
            return x.buffer not in V.graph.unaligned_buffers
        assert isinstance(x, SizeArg)
        return V.graph.sizevars.maybe_guard_multiple_of(x.expr, ALIGNMENT)

    divisible_by_16 = [i for i, arg in enumerate(args) if is_aligned(arg)]
    return instance_descriptor(tuple(divisible_by_16), ())


class TritonPrinter(ExprPrinter):
    def _print_ModularIndexing(self, expr):
        x, div, mod = expr.args
        x = self.paren(self.doprint(x))
        div = self.paren(self.doprint(div))
        mod = self.paren(self.doprint(mod))
        if div != "1":
            x = f"({x} // {div})"
        return f"{x} % {mod}"

    def _print_IndexingDiv(self, expr):
        x, div = expr.args
        x = self.paren(self.doprint(x))
        div = self.paren(self.doprint(div))
        return f"({x} // {div})"


texpr = TritonPrinter().doprint


def triton_compute_type(dtype):
    triton_type_name = str(dtype).split(".")[-1]
    if triton_type_name == "bool":
        triton_type_name = "int1"
    if triton_type_name in ("float16", "bfloat16"):
        # float16 math is done in float32 inside the kernel
        triton_type_name = "float32"
    return f"tl.{triton_type_name}"


def triton_constant(value, dtype):
    if value == float("inf"):
        return 'float("inf")'
    elif value == float("-inf"):
        return 'float("-inf")'
    elif math.isnan(value):
        return 'float("nan")'
    # 0 -> 0.0, otherwise triton will treat 0 as an int
    if dtype.is_floating_point and isinstance(value, int):
        value = float(value)
    return repr(value)


class TritonOverrides(OpOverrides):
    """Map element-wise ops to Triton"""

    @staticmethod
    def to_dtype(x, dtype: torch.dtype):
        if dtype == torch.bool:
            return f"({x} != 0)"
        return f"{x}.to({triton_compute_type(dtype)})"

    @staticmethod
    def constant(value, dtype):
        return triton_constant(value, dtype)

    @staticmethod
    def abs(x):
        return f"tl.abs({x})"

    @staticmethod
    def libdevice_abs(x):
        return f"tl.libdevice.abs({x})"

    @staticmethod
    def exp(x):
        return f"tl.exp({x})"

    @staticmethod
    def libdevice_exp(x):
        return f"tl.libdevice.exp({x})"

    @staticmethod
    def sqrt(x):
        return f"tl.sqrt({x})"

    @staticmethod
    def libdevice_sqrt(x):
        return f"tl.libdevice.sqrt({x})"

    @staticmethod
    def relu(x):
        return ops.maximum("0", x)

    @staticmethod
    def minimum(a, b):
        return f"tl.minimum({a}, {b})"

    @staticmethod
    def maximum(a, b):
        return f"tl.maximum({a}, {b})"

    @staticmethod
    def where(a, b, c):
        if not config.triton.simple_where:
            # wonkyness to work around https://github.com/openai/triton/issues/532
            # identity calls to force new triton variables (and get access to .shape/.dtype/.numel
            a = ops.identity(a)
            b = ops.identity(b)
            c = ops.identity(c)
            a = ops.identity(
                f"{a} | tl.zeros({b}.shape, {a}.dtype) if {b}.numel > 1 else {a}"
            )
            a = ops.identity(
                f"{a} | tl.zeros({c}.shape, {a}.dtype) if {c}.numel > 1 else {a}"
            )
        return f"tl.where({a}, {b}, {c})"

    @staticmethod
    def cos(x):
        return f"tl.cos({x})"

    @staticmethod
    def libdevice_cos(x):
        return f"tl.libdevice.cos({x})"

    @staticmethod
    def sin(x):
        return f"tl.sin({x})"

    @staticmethod
    def libdevice_sin(x):
        return f"tl.libdevice.sin({x})"

    @staticmethod
    def index_expr(expr, dtype):
        return V.kernel.indexing(expr)[0]

    @staticmethod
    def masked(mask, body, other):
        with V.kernel.mask_loads(mask) as new_mask:
            result = body()
        return ops.where(
            new_mask, result, TritonOverrides.constant(other, torch.float32)
        )

    @staticmethod
    def lgamma(x):
        return f"tl.libdevice.lgamma({x})"

    @staticmethod
    def logical_and(a, b):
        return f"{a} & {b}"

    @staticmethod
    def logical_or(a, b):
        return f"{a} | {b}"

    @staticmethod
    def rand(seed, offset, _):  # _ here to keep the contract identical to CPU rand op
        return f"tl.rand({seed}, {offset})"

    @staticmethod
    def randn(seed, offset, _):  # _ here to keep the contract identical to CPU randn op
        return f"tl.randn({seed}, {offset})"

    @staticmethod
    def rsqrt(x):
        return f"tl.libdevice.rsqrt({x})"

    @staticmethod
    def sigmoid(x):
        return f"tl.sigmoid({x})"

    @staticmethod
    def libdevice_sigmoid(x):
        return f"1/(1 + tl.libdevice.exp(-({x})))"

    @staticmethod
    def signbit(x):
        # XX: This is wrong for the value -0.0 in floating point
        return f"tl.libdevice.signbitf({x}) if ({x}).dtype is tl.float32 else {x} < 0"

    @staticmethod
    def fmod(a, b):
        return f"tl.libdevice.fmod({a}, ({b}).to(tl.float32))"

    @staticmethod
    def pow(a, b):
        return f"tl.libdevice.pow({a}, {b})"

    @staticmethod
    def log(x):
        return f"tl.log({x})"

    @staticmethod
    def libdevice_log(x):
        return f"tl.libdevice.log({x})"

    @staticmethod
    def isinf(x):
        return f"tl.libdevice.isinfd({x}) if ({x}).dtype is tl.float64 else tl.libdevice.isinff({x})"

    @staticmethod
    def isnan(x):
        return f"tl.libdevice.isnand({x}) if ({x}).dtype is tl.float64 else tl.libdevice.isnanf({x})"

    @staticmethod
    def round(x):
        return f"tl.libdevice.nearbyint({x})"

    @staticmethod
    def floor(x):
        return f"tl.libdevice.floor({x})"

    @staticmethod
    def floordiv(a, b):
        # See the comment in lowering.div_mode. a and b are integer type.
        # Similar to div_floor_kernel_cuda in pytorch core.
        # Notice that // in triton behaves as truncdiv instead of floordiv
        quot = f"{a} // {b}"
        rem = f"{a} % {b}"
        return f"tl.where(({a} < 0) != ({b} < 0), tl.where({rem} != 0, {quot} - 1, {quot}), {quot})"

    @staticmethod
    def trunc(x):
        return f"tl.libdevice.trunc({x})"

    @staticmethod
    def truncdiv(a, b):
        # See the comment in lowering.div_mode. a and b are integer type.
        # Notice that // in triton behaves as truncdiv instead of floordiv
        return f"{a} // {b}"

    @staticmethod
    def ceil(x):
        return f"tl.libdevice.ceil({x})"


@dataclasses.dataclass
class IterationRanges:
    """
    Each range tree represents multiple sets of iteration indexing
    in a single tiled dimension in the output kernel.

    If you have two loops ranges one (4, 3, 2) and another (4, 6),
    then the range tree will be:
            4 (i0)
        3 (i1)  6 (i3)
        2 (i2)
    Where i0 is shared between both loops, but then the split into
    different indexing vars.  All loop ranges must iterate over
    the same number of elements.
    """

    def __init__(
        self,
        name: str,
        var_list: List[sympy.Symbol],
        var_ranges: Dict[sympy.Symbol, sympy.Expr],
        numel: sympy.Expr,
        prefix: str,
        divisor=sympy.Integer(1),
        length=sympy.Integer(1),
    ):
        super(IterationRanges, self).__init__()
        self.name = name
        self.var_list = var_list
        self.var_ranges = var_ranges
        self.numel = numel
        self.prefix = prefix
        self.divisor = divisor
        self.length = length

    def is_loop(self):
        return self.prefix == "r"


class IterationRangesRoot(IterationRanges):
    def __init__(
        self,
        name: str,
        numel: sympy.Expr,
        prefix: str,
        index: int,
        kernel: "Kernel",
        pid_cache=None,
    ):
        if pid_cache is None:
            pid_cache = {}
        super(IterationRangesRoot, self).__init__(
            name=name,
            var_list=[],
            var_ranges={},
            numel=numel,
            prefix=prefix,
        )
        self.index = index
        self.kernel = kernel
        # Store all the nodes in one flat list
        self.nodes: Dict[sympy.Expr, IterationRangesEntry] = {}
        # This is for re-ordering program ID in triton mm template
        # pid_cache["tl.program_id(0)"] = pid_m
        self.pid_cache: Dict[str, str] = pid_cache

    def cache_clear(self):
        for node in self.nodes.values():
            node.cache_clear()

    def lookup(self, divisor, length):
        """
        Lookup a given RangeTreeEntry, creating it if needed
        """
        if V.graph.sizevars.maybe_guard_equals(divisor * length, self.numel):
            expr = ir.IndexingDiv(sympy_symbol(f"{self.prefix}index"), divisor)
        else:
            expr = ir.ModularIndexing(
                sympy_symbol(f"{self.prefix}index"), divisor, length
            )

        if expr not in self.nodes:
            node = IterationRangesEntry(
                f"{self.prefix}{next(V.kernel.iter_vars_count)}",
                divisor,
                length,
                expr,
                self,
            )
            V.kernel.range_tree_nodes[node.symbol()] = node
            self.var_list.append(node.symbol())
            self.var_ranges[node.symbol()] = length
            self.nodes[expr] = node
        return self.nodes[expr]

    def construct(self, lengths: List[sympy.Expr]):
        divisor = sympy.Integer(1)
        itervars = []
        for length in reversed(lengths):
            itervars.append(self.lookup(divisor, length).symbol())
            divisor = divisor * length
        return list(reversed(itervars))

    def vars_and_sizes(self, index: sympy.Expr):
        """Figure out vars from this tree used in index"""
        nodes = [V.kernel.range_tree_nodes.get(s) for s in index.free_symbols]
        nodes = [n for n in nodes if n and n.prefix == self.prefix]
        nodes.sort(key=lambda x: V.graph.sizevars.size_hint(x.divisor))
        divisor = sympy.Integer(1)
        index_vars = []
        sizes = []

        def add(node):
            nonlocal divisor
            index_vars.append(node.symbol())
            sizes.append(node.length)
            divisor = divisor * node.length

        for node in nodes:
            if not V.graph.sizevars.maybe_guard_equals(node.divisor, divisor):
                # fill in unused index var
                add(self.lookup(divisor, ir.IndexingDiv(node.divisor, divisor)))
                divisor = node.divisor
            add(node)
        if not V.graph.sizevars.maybe_guard_equals(self.numel, divisor):
            # fill in unused index var
            add(self.lookup(divisor, ir.IndexingDiv(self.numel, divisor)))

        return list(reversed(index_vars)), list(reversed(sizes))

    def ranges_code(self):
        size = self.kernel.reshape_size_str(self.index, self.prefix)
        return f"tl.reshape(tl.arange(0, {self.prefix.upper()}BLOCK), {size})"

    def pid_cache_lookup(self, key):
        if key in self.pid_cache:
            return self.pid_cache[key]
        return key

    def codegen_header(self, code):
        x = self.prefix
        if self.is_loop():
            code.writeline(f"{self.name} = {x}offset + {x}base")
        else:
            pid = self.pid_cache_lookup(f"tl.program_id({self.index})")
            code.writelines(
                [
                    f"{x}offset = {pid} * {x.upper()}BLOCK",
                    f"{self.name} = {x}offset + {self.ranges_code()}",
                ]
            )
        code.writeline(f"{x}mask = {self.name} < {x}numel")


class IterationRangesEntry(IterationRanges):
    def __init__(
        self,
        name: str,
        divisor: sympy.Expr,
        length: sympy.Expr,
        expr: sympy.Expr,
        parent: IterationRanges,
    ):
        super(IterationRangesEntry, self).__init__(
            name=name,
            numel=parent.numel / length,
            var_list=parent.var_list,
            var_ranges=parent.var_ranges,
            prefix=parent.prefix,
            divisor=divisor,
            length=length,
        )
        self.parent = parent
        self.codegen = functools.lru_cache(None)(self._codegen)
        self.expr = expr

    def cache_clear(self):
        self.codegen.cache_clear()

    def writeline(self, line):
        if self.is_loop():
            V.kernel.indexing_code.writeline(line)
        else:
            # lift non-reduction stores outside loop
            V.kernel.body.writeline(line)

    def _codegen(self):
        self.writeline(f"{self.name} = " + texpr(V.kernel.rename_indexing(self.expr)))
        return self.name

    def symbol(self):
        return sympy_symbol(self.name)

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return self.name == other.name


class TritonKernel(Kernel):
    overrides = TritonOverrides
    sexpr = texpr

    def __init__(self, *groups, pid_cache=None, reduction_hint=ReductionHint.DEFAULT):
        if pid_cache is None:
            pid_cache = {}
        super(TritonKernel, self).__init__()
        self.numels = [V.graph.sizevars.simplify(s) for s in groups]
        self.range_trees = []
        self.range_tree_nodes = {}
        self.iter_vars_count = itertools.count()
        self.inside_reduction = self.numels[-1] != 1
        self._load_mask = None
        self.body = IndentedBuffer()
        self.indexing_code = IndentedBuffer()
        self.suffix = IndentedBuffer()
        self.outside_loop_vars = set()
        self.initialize_range_tree(pid_cache)
        self.reduction_hint = reduction_hint

        # define this in a closure to make cache local to object
        @functools.lru_cache(None)
        def simplify_indexing(index: sympy.Expr):
            index = V.graph.sizevars.simplify_with_ranges(index, self.var_ranges())
            for tree in self.range_trees:
                index = self.combine_contiguous_dims(index, tree)
            return index

        self.simplify_indexing = simplify_indexing

    def initialize_range_tree(self, pid_cache):
        names = ["xindex", "yindex", "zindex"][: len(self.numels) - 1] + ["rindex"]
        for i in range(len(self.numels)):
            self.range_trees.append(
                IterationRangesRoot(
                    names[i], self.numels[i], names[i][0], i, self, pid_cache
                )
            )
        for tree in self.range_trees:
            # reduction indexing goes inside a loop
            if tree.prefix != "r":
                tree.codegen_header(self.body)
        if self.inside_reduction and self.range_trees[-1].is_loop():
            # workaround for this issue:
            # https://gist.github.com/jansel/6527126f781559095c5531f98a4235a7
            self.body.writeline(f"rbase = {self.range_trees[-1].ranges_code()}")

    def disable_reduction(self):
        @contextlib.contextmanager
        def ctx():
            if self.numels[-1] == 1:
                assert not self.inside_reduction
                yield
                return
            # calling codegen_body() will flush all the pending buffers
            # and write out a reduction loop
            self.codegen_body()
            self.inside_reduction = False
            yield
            # flush out any code before opening the next loop
            self.codegen_body()
            self.inside_reduction = True

        return ctx()

    def set_ranges(self, *lengths):
        assert len(lengths) == len(self.range_trees)
        return [
            ranges.construct(length)
            for length, ranges in zip(lengths, self.range_trees)
        ]

    @staticmethod
    def _split_iteration_ranges(
        groups: List[sympy.Expr], lengths: List[List[sympy.Expr]]
    ):
        sv = V.graph.sizevars
        new_ranges = [[] for _ in groups]
        remaining = [sv.simplify(g) for g in groups]
        var_count = itertools.count()

        def add_range(i, expr):
            expr = sv.simplify(expr)
            if not sv.maybe_guard_multiple_of(remaining[i], expr):
                raise CantSplit()
            # guard on the last item out
            sv.maybe_guard_equals(remaining[i], expr)
            remaining[i] = ir.IndexingDiv(remaining[i], expr)
            new_ranges[i].append(expr)
            return next(var_count)

        def make_combined(size, idx1, idx2):
            def getter(flat_vars):
                return size * flat_vars[idx1] + flat_vars[idx2]

            return getter

        return_getters_groups = []
        current_group = 0
        for length_group in lengths:
            return_getters = []
            for size in length_group:
                if sv.maybe_guard_equals(size, 1):
                    return_getters.append(lambda _: sympy.Integer(0))
                    continue

                while (
                    current_group < len(remaining)
                    and sv.size_hint(remaining[current_group]) == 1
                ):
                    # scroll to next group with remaining elements
                    current_group += 1

                if sv.size_hint(size) > sv.size_hint(remaining[current_group]):
                    # need to break size in two
                    if not sv.maybe_guard_multiple_of(size, remaining[current_group]):
                        raise CantSplit()
                    size1 = remaining[current_group]
                    size2 = ir.IndexingDiv(size, remaining[current_group])
                    return_getters.append(
                        make_combined(
                            size2,
                            add_range(current_group, size1),
                            add_range(current_group + 1, size2),
                        )
                    )
                else:
                    return_getters.append(
                        operator.itemgetter(add_range(current_group, size))
                    )
            return_getters_groups.append(return_getters)

        assert all(
            V.graph.sizevars.size_hint(s) == 1 for s in remaining
        ), f"failed to set ranges {remaining} {lengths}"

        return new_ranges, return_getters_groups

    @classmethod
    def is_compatible(cls, groups: List[sympy.Expr], lengths: List[List[sympy.Expr]]):
        try:
            cls._split_iteration_ranges(groups, lengths)
            return True
        except CantSplit:
            return False

    def split_and_set_ranges(self, lengths: List[List[sympy.Expr]]):
        """
        We may want to fuse `for i0 in s0*s1` into a tiled kernel with groups (s0, s1).

        To do this we need to split up the iteration space of i0 into something like:
            for i1 in s0:
              for i2 in s1:
                i0 = i1*s1 + i2
                ....

        This function matches and resplits lengths to the groups of
        this kernel to enable tiled + non-tiled fusions.
        """
        groups = [rt.numel for rt in self.range_trees]
        if not self.inside_reduction:
            groups[-1] = sympy.Integer(1)

        if len(lengths) == len(self.range_trees) and all(
            V.graph.sizevars.simplify(sympy_product(x) - g) == 0
            for x, g in zip(lengths, groups)
        ):
            return self.set_ranges(*lengths)

        new_ranges, return_getters_groups = self._split_iteration_ranges(
            groups, lengths
        )
        itervars = list(itertools.chain(*self.set_ranges(*new_ranges)))
        return [[fn(itervars) for fn in fns] for fns in return_getters_groups]

    def is_indirect_indexing(self, index: sympy.Expr):
        # tmpX  means indirect indexing
        return free_symbol_startswith(index, "tmp")

    def combine_contiguous_dims(self, index: sympy.Expr, tree: IterationRangesRoot):
        """
        More aggressive simplification to merge contiguous dims
        """
        if isinstance(index, (sympy.Integer, sympy.Symbol)):
            return index
        index_vars, sizes = tree.vars_and_sizes(index)
        if len(sizes) <= 1:
            return index
        new_sizes, reindex, prune = V.graph.sizevars._simplify_loops(
            index_vars, sizes, index_prevent_reordering([index], index_vars, sizes)
        )
        if new_sizes == sizes:
            return index
        new_index_vars = tree.construct(new_sizes)
        new_index = sympy_subs(index, dict(zip(index_vars, reindex(new_index_vars))))
        return new_index

    def indexing(
        self,
        index: sympy.Expr,
        *,
        copy_shape=None,
        dense_indexing=False,
    ):
        """
        Compute the index and mask to pass to tl.load() or tl.store()
        """
        index = self.simplify_indexing(index)
        index_vars = index.free_symbols
        index_str = texpr(self.rename_indexing(self.codegen_indexing(index)))
        indirect_indexing = self.is_indirect_indexing(index)

        need_dense = (
            config.triton.dense_indexing
            or dense_indexing
            or indirect_indexing
            or self._load_mask is not None
        ) and index != 0

        have_dense = True
        have_loop_vars = False
        mask = []
        dense_mask = []

        for tree in self.range_trees:
            if tree.prefix == "r" and not self.inside_reduction:
                continue
            if index_vars.intersection(tree.var_list):
                have_loop_vars = True
                have_dense = False
                mask.append(f"{tree.prefix}mask")
            dense_mask.append(f"{tree.prefix}mask")

        if (need_dense and not have_dense) or isinstance(index, sympy.Integer):
            index_str = f"{index_str} + tl.zeros({self.dense_size_str()}, tl.int32)"
            if isinstance(index, sympy.Integer):
                return index_str, "None"
            else:
                mask = dense_mask

        elif not have_loop_vars and copy_shape:
            mask = dense_mask
            index_str = f"{index_str} + tl.zeros({copy_shape}.shape, tl.int32)"
        elif indirect_indexing:
            mask = dense_mask

        if self._load_mask:
            mask.append(self._load_mask)
        elif not mask:
            mask = ["None"]

        if mask == ["xmask"] and index == 0 and self.range_trees[0].numel == 1:
            # This causes a triton error:
            # https://github.com/openai/triton/issues/633
            mask = ["None"]

        return index_str, " & ".join(mask)

    def var_ranges(self):
        return dict(
            itertools.chain.from_iterable(
                tree.var_ranges.items() for tree in self.range_trees
            )
        )

    def codegen_indexing(self, expr: sympy.Expr):
        expr = V.graph.sizevars.simplify_with_ranges(expr, self.var_ranges())
        for sym in sorted(expr.free_symbols, key=str):
            if sym in self.range_tree_nodes:
                self.range_tree_nodes[sym].codegen()
        return expr

    @contextlib.contextmanager
    def mask_loads(self, mask):
        """Context manager to add an additional mask to tl.load/store"""
        prior = self._load_mask
        if prior:
            mask = self.cse.generate(self.compute, f"{mask} & {prior}")

        self._load_mask = mask
        with self.swap_buffers(self.compute, self.compute):
            # TODO(jansel): do we need a reshape here?
            yield mask
        self._load_mask = prior

    def load(self, name: str, index: sympy.Expr):
        var = self.args.input(name)
        indirect_indexing = self.is_indirect_indexing(index)
        index, mask = self.indexing(index)

        if "rmask" in mask:
            # This eviction policy heuristic is untested.
            # ptillet suggested we should try only doing this for
            # the first N-1 loops and not for the final loop.
            ep = ", eviction_policy='evict_last'"
        else:
            ep = ""
        # "other" below is a workaround for https://github.com/openai/triton/issues/737
        # for bool, even though it's likely subject to the same bug, setting `other` leads
        # to LLVM errors so we are skipping it for now
        if "tmp" in mask and V.graph.get_dtype(name) != torch.bool:
            other = ", other=0"
        else:
            other = ""

        if V.graph.is_unspec_arg(name):
            line = var
        else:
            line = f"tl.load({var} + ({index}), {mask}{ep}{other})"
            if V.graph.get_dtype(name) in (torch.float16, torch.bfloat16):
                line += ".to(tl.float32)"

        if (
            self.inside_reduction
            and "rmask" not in mask
            and "tmp" not in mask
            and not indirect_indexing
        ):
            # can lift a common load outside of reduction loop
            # One exception is when this is an indirect_load.
            tmp = self.cse.generate(self.body, line)
        else:
            tmp = self.cse.generate(self.loads, line)

        if not self.inside_reduction or "rmask" not in mask:
            self.outside_loop_vars.add(tmp)
        return tmp

    def store(self, name, index, value, mode=None):
        var = self.args.output(name)
        index, mask = self.indexing(index, dense_indexing=True)
        if mode is None:
            line = f"tl.store({var} + ({index}), {value}, {mask})"
        elif mode == "atomic_add":
            line = f"tl.atomic_add({var} + ({index}), {value}, {mask})"
        else:
            raise NotImplementedError(f"store mode={mode}")
        self.stores.writeline(name, line)
        if not self.inside_reduction:
            self.outside_loop_vars.add(value)

    def reduction(self, name, dtype, src_dtype, reduction_type, index, value):
        assert self.inside_reduction
        default = triton_constant(ir.Reduction.default_value(reduction_type, src_dtype), src_dtype)
        masks = [f"{tree.prefix}mask" for tree in self.range_trees]
        if self._load_mask:
            masks.append(self._load_mask)
        sizes = [f"{tree.prefix.upper()}BLOCK" for tree in self.range_trees]
        sizes[-1] = "1"
        reduction_range_prefix = self.range_trees[-1].prefix
        reduction_sizes = ["1" for _ in self.range_trees]
        reduction_sizes[-1] = f"{reduction_range_prefix.upper()}BLOCK"

        if reduction_type == "any":
            reduction_type = "max"

        dim = len(self.range_trees) - 1
        result_var = self.cse.newvar()
        if (src_dtype, reduction_type, value) not in self.cse.reduction_cache:
            self.cse.reduction_cache[(src_dtype, reduction_type, value)] = result_var
            accumulator = f"_{result_var}"
            self.body.writeline(
                f"{accumulator} = tl.zeros({self.dense_size_str()}, {triton_compute_type(src_dtype)}) + {default}"
            )
            accumulator_index = None
            if reduction_type in {"argmax", "argmin"}:
                accumulator_index = f"_{result_var}_index"
                self.body.writeline(
                    f"{accumulator_index} = tl.zeros({self.dense_size_str()}, tl.int64)"
                )

            updated = value
            if reduction_type in {"min", "argmin"}:
                masks.append(f"({accumulator} > {value})")
            elif reduction_type in {"max", "argmax"}:
                masks.append(f"({accumulator} < {value})")
            elif reduction_type == "sum":
                updated = f"{accumulator} + {value}"
            else:
                raise NotImplementedError(f"reduction_type {reduction_type}")

            cond = " & ".join(masks)

            if accumulator_index:
                # argmax or argmin
                self.compute.writeline(
                    f"{accumulator_index} = tl.where({cond},  {reduction_range_prefix}index, {accumulator_index})",
                )
            self.compute.writeline(
                f"{accumulator} = tl.where({cond}, {updated}, {accumulator})"
            )

            if accumulator_index:
                # argmax, argmin
                self.suffix.writelines(
                    [
                        f"{accumulator_index}_reduce = tl.reshape(",
                        f"\ttl.{reduction_type}({accumulator}, {dim}), [{', '.join(sizes)}]).to(tl.int32)",
                        f"{accumulator_index}_mask = (tl.reshape(tl.arange(0, {reduction_range_prefix.upper()}BLOCK),",
                        f"\t[{', '.join(reduction_sizes)}]) == {accumulator_index}_reduce)",
                        f"{result_var} = tl.reshape(tl.sum(",
                        f"\ttl.where({accumulator_index}_mask, {accumulator_index}, 0), {dim}), [{', '.join(sizes)}])",
                    ]
                )
            else:
                self.suffix.writeline(
                    f"{result_var} = tl.reshape(tl.{reduction_type}({accumulator}, {dim}), [{', '.join(sizes)}])"
                )
        else:
            var_name = self.cse.reduction_cache[(src_dtype, reduction_type, value)]
            self.suffix.writeline(f"{result_var} = {var_name}")
        self.inside_reduction = False
        index, mask = self.indexing(index)
        assert "rmask" not in index
        self.inside_reduction = True
        self.outside_loop_vars.add(result_var)
        self.cse.store_cache[name] = result_var
        if name not in V.graph.removed_buffers:
            var = self.args.output(name)
            self.suffix.writeline(
                DeferredLine(name, f"tl.store({var} + {index}, {result_var}, {mask})")
            )

    def codegen_body(self):
        """
        Concat output code from index_code, loads, compute, stores,
        suffix into self.body.

        For pointwise kernels, this is called just once at the end.

        For reduction kernels, this generates a loop over the reduction
        axis.
        """
        if not (
            self.indexing_code
            or self.loads
            or self.stores
            or self.compute
            or self.suffix
        ):
            return

        if self.inside_reduction:
            self.body.writeline("for roffset in range(0, rnumel, RBLOCK):")
            with self.body.indent():
                # last range tree is always reduction
                self.range_trees[-1].codegen_header(self.body)
                self.body.splice(self.indexing_code)
                self.body.splice(self.loads)
                self.body.splice(self.compute)
                self.body.splice(self.stores)

            # invalidate any caches that came from inside the reduction loop
            self.cse.invalidate(self.outside_loop_vars)
            self.range_trees[-1].cache_clear()
        else:
            self.body.splice(self.indexing_code)
            self.body.splice(self.loads)
            self.body.splice(self.compute)
            self.body.splice(self.stores)
        self.body.splice(self.suffix)
        self.indexing_code.clear()
        self.loads.clear()
        self.compute.clear()
        self.stores.clear()
        self.suffix.clear()

    def codegen_kernel(self, name=None):
        from triton import next_power_of_2

        code = IndentedBuffer()
        size_hints = [
            next_power_of_2(V.graph.sizevars.size_hint(numel)) for numel in self.numels
        ]
        if not self.inside_reduction:
            size_hints.pop()
            heuristics = "pointwise"
        else:
            heuristics = "reduction"

        if name is None:
            code.splice(
                f"""
                    import triton
                    import triton.language as tl
                    from {config.inductor_import}.ir import ReductionHint
                    from {config.inductor_import}.triton_ops.autotune import {heuristics}
                    from {config.inductor_import}.utils import instance_descriptor
                """
            )

        argdefs, _, signature = self.args.python_argdefs()
        triton_meta = {
            "signature": dict(enumerate(map(signature_of, signature))),
            "device": V.graph.scheduler.current_device.index,
            "constants": {},
        }

        for tree in self.range_trees:
            if tree.prefix != "r" or self.inside_reduction:
                sizearg = SizeArg(f"{tree.prefix}numel", tree.numel)
                signature.append(sizearg)
                triton_meta["signature"][len(argdefs)] = signature_of(sizearg)
                argdefs.append(f"{tree.prefix}numel")
                # constexpr version causes issues, see
                # https://github.com/pytorch/torchdynamo/pull/1362
                # triton_meta["constants"][len(argdefs)] = V.graph.sizevars.size_hint(
                #     tree.numel
                # )
                # argdefs.append(f"{tree.prefix}numel: tl.constexpr")
        triton_meta["configs"] = [config_of(signature)]

        for tree in self.range_trees:
            if tree.prefix != "r" or self.inside_reduction:
                argdefs.append(f"{tree.prefix.upper()}BLOCK : tl.constexpr")

        if self.inside_reduction:
            reduction_hint = self.reduction_hint
            heuristics_line = f"""
                @{heuristics}(size_hints={size_hints!r},
                              reduction_hint={reduction_hint},
                              filename=__file__,
                              meta={triton_meta!r})
                @triton.jit
            """
        else:
            heuristics_line = f"""
                @{heuristics}(size_hints={size_hints!r}, filename=__file__, meta={triton_meta!r})
                @triton.jit
            """
        code.splice(heuristics_line)
        code.writeline(f"def {name or 'KERNEL_NAME'}({', '.join(argdefs)}):")
        self.codegen_body()
        with code.indent():
            if not config.dynamic_shapes:
                self.codegen_static_numels(code)
            for old, new in self.args.aliases():
                code.writeline(f"{old} = {new}")
            code.splice(self.body)

        if name is not None:
            return code.getvalue()

        wrapper = IndentedBuffer()
        wrapper.writeline("async_compile.triton('''")
        wrapper.splice(code.getvalue(), strip=True)
        wrapper.writeline("''')")
        return wrapper.getvalue()

    def codegen_static_numels(self, code):
        """
        We get a small speedup from hard coding numels if they are static.
        """
        for tree in self.range_trees:
            if tree.prefix != "r" or self.inside_reduction:
                if isinstance(V.graph.sizevars.simplify(tree.numel), sympy.Integer):
                    code.writeline(
                        f"{tree.prefix}numel = {V.graph.sizevars.size_hint(tree.numel)}"
                    )
                elif not config.dynamic_shapes:
                    code.writeline(
                        f"{tree.prefix}numel = {V.graph.sizevars.size_hint(tree.numel)}  # dynamic_shapes=False"
                    )

    def reshape_size_str(self, i=None, x=None):
        sizes = ["1"] * (len(self.range_trees) - int(self.numels[-1] == 1))
        if i is not None:
            sizes[i] = f"{x.upper()}BLOCK"
        return f"[{', '.join(sizes)}]"

    def dense_size_str(self):
        sizes = []
        for tree in self.range_trees:
            if tree.prefix != "r" or self.inside_reduction:
                sizes.append(f"{tree.prefix.upper()}BLOCK")
            elif tree.prefix == "r" and tree.numel != 1:
                sizes.append("1")
        return f"[{', '.join(sizes)}]"

    def call_kernel(self, code, name: str):
        _, call_args, _ = self.args.python_argdefs()
        # dynamo wraps unspec variable as 0d CPU tensor, need convert to scalar
        for i in range(len(call_args)):
            if V.graph.is_unspec_arg(call_args[i]):
                call_args[i] = call_args[i] + ".item()"
        grid = []
        # TODO(jansel): if there are constants, we shouldn't bother passing them as args
        for tree in self.range_trees:
            if isinstance(tree.numel, (sympy.Integer, sympy.Symbol)):
                expr = texpr(tree.numel)
            else:
                expr = f"{name}_{tree.prefix}numel"
                code.writeline(f"{expr} = {texpr(tree.numel)}")
            if tree.prefix != "r" or self.inside_reduction:
                call_args.append(expr)
            if tree.prefix != "r":
                grid.append(expr)
        call_args = ", ".join(call_args)
        stream_name = code.write_get_cuda_stream(V.graph.scheduler.current_device.index)
        code.writeline(
            f"{name}.run({call_args}, grid=grid({', '.join(grid)}), stream={stream_name})"
        )


class TritonScheduling:
    def __init__(self, scheduler):
        self.scheduler = scheduler

    def group_fn(self, sizes):
        return tuple(V.graph.sizevars.simplify(sympy_product(s)) for s in sizes)

    def can_fuse(self, node1, node2):
        """
        Hook called by Scheduler to determine if the Triton backend
        can fuse node1 and node2.  These nodes might already be
        FusedSchedulerNodes.
        """
        _, (numel1, rnumel1) = node1.group
        _, (numel2, rnumel2) = node2.group

        if node1.is_reduction() and node2.is_reduction():
            return numel1 == numel2 and rnumel1 == rnumel2

        if not node1.is_reduction() and not node2.is_reduction():
            if not (numel1 == numel2 and rnumel1 == rnumel2):
                return False

            # check for a bad combined tiling
            tiling1 = self.select_tiling(node1.get_nodes(), numel1, rnumel1)
            tiling2 = self.select_tiling(node2.get_nodes(), numel1, rnumel1)
            tiling3 = self.select_tiling(
                node1.get_nodes() + node2.get_nodes(), numel1, rnumel1
            )
            if config.triton.tiling_prevents_pointwise_fusion:
                if len(tiling1) > 2:
                    if len(tiling2) > 2:
                        return tiling1 == tiling2 == tiling3
                    else:
                        return tiling1 == tiling3
                elif len(tiling2) > 2:
                    return tiling2 == tiling3

            return True

        if not node1.is_reduction() and node2.is_reduction():
            assert rnumel1 == 1 and rnumel2 != 1
            if numel1 == numel2 * rnumel2:
                if not all(
                    TritonKernel.is_compatible((numel2, rnumel2), n.get_ranges())
                    for n in node1.get_nodes()
                ):
                    return False
                if config.triton.tiling_prevents_reduction_fusion:
                    return self.select_tiling(node1.get_nodes(), numel1) in (
                        (numel1, 1),
                        (numel2, rnumel2, 1),
                    )
                return True

            return numel1 == numel2

        assert node1.is_reduction() and not node2.is_reduction()
        # swap args to hit the case above
        return self.can_fuse_horizontal(node2, node1)

    can_fuse_vertical = can_fuse
    can_fuse_horizontal = can_fuse

    def codegen_nodes(self, nodes):
        """
        Given a set of pre-fused nodes, generate a Triton kernel.
        """
        _, (numel, rnumel) = max(nodes, key=lambda x: int(x.is_reduction())).group
        node_schedule = []
        current_loop_writes = set()
        done = set()

        def fits_in_main_body(n):
            _, (node_numel, node_rnumel) = n.group
            return (node_numel == numel and node_rnumel == rnumel) or (
                node_numel == numel * rnumel and node_rnumel == 1
            )

        def fits_outside_reduction(n):
            _, (node_numel, node_rnumel) = n.group
            return node_numel == numel and node_rnumel == 1 and rnumel != 1

        @contextlib.contextmanager
        def end_current_reduction_loop():
            if current_loop_writes:
                # flush out any other runnable nodes to reduce number of loops
                for other_node in nodes[index + 1 :]:
                    if (
                        node not in done
                        and fits_in_main_body(other_node)
                        and not (
                            current_loop_writes & other_node.recursive_predecessors
                        )
                    ):
                        done.add(node)
                        current_loop_writes.add(node.get_name())
                        node_schedule.append(node)

            if node_schedule and node_schedule[-1] is EnableReduction:
                node_schedule.pop()
            else:
                node_schedule.append(DisableReduction)
            yield
            node_schedule.append(EnableReduction)
            current_loop_writes.clear()

        for index, node in enumerate(nodes):
            if node in done:
                continue
            done.add(node)

            if fits_in_main_body(node):
                if current_loop_writes & node.recursive_predecessors and rnumel != 1:
                    with end_current_reduction_loop():
                        pass  # need to start a new reduction loop
                current_loop_writes.add(node.get_name())
                node_schedule.append(node)
            elif fits_outside_reduction(node):
                with end_current_reduction_loop():
                    node_schedule.append(node)
            else:
                raise NotImplementedError(
                    f"unexpected group: ({numel}, {rnumel}) != {node.group[1]}"
                )

        log.log(logging.CODE, "schedule: %s", node_schedule)
        return self.codegen_node_schedule(node_schedule, numel, rnumel)

    @staticmethod
    def reduction_hint(node):
        assert node.is_reduction()
        if all(
            dep.is_contiguous()
            for dep in itertools.chain(node.read_writes.reads, node.read_writes.writes)
        ):
            return ReductionHint.INNER
        else:
            return node.node.data.reduction_hint

    def codegen_node_schedule(self, node_schedule, numel, reduction_numel):
        tiled_groups = self.select_tiling(node_schedule, numel, reduction_numel)
        reductions = list(
            filter(
                lambda n: n not in (EnableReduction, DisableReduction)
                and n.is_reduction(),
                node_schedule,
            )
        )
        if len(reductions) > 0:
            hints = [self.reduction_hint(n) for n in reductions]
            if hints.count(hints[0]) == len(hints):
                reduction_hint_val = hints[0]
            else:
                reduction_hint_val = ReductionHint.DEFAULT
        else:
            reduction_hint_val = ReductionHint.DEFAULT
        with TritonKernel(*tiled_groups, reduction_hint=reduction_hint_val) as kernel:
            stack = contextlib.ExitStack()
            for node in node_schedule:
                if node not in (EnableReduction, DisableReduction):
                    node.mark_run()
            for node in node_schedule:
                if node is DisableReduction:
                    stack.enter_context(kernel.disable_reduction())
                elif node is EnableReduction:
                    stack.close()
                else:
                    node.codegen(kernel.split_and_set_ranges(node.get_ranges()))

        wrapper = V.graph.wrapper_code
        src_code = kernel.codegen_kernel()
        if src_code in wrapper.kernels:
            kernel_name = wrapper.kernels[src_code]
        else:
            kernel_name = wrapper.next_kernel_name()
            wrapper.kernels[src_code] = kernel_name
            subs_name = kernel_name if config.triton.ordered_kernel_names else "kernel"
            src_code = src_code.replace("KERNEL_NAME", subs_name)
            # TODO(voz): Ostensibly, we should not need this. But there are cases where C++ codegen does
            # not use BracesBuffer, so we have no good indicator of a C++ buffer atm.
            src_code = src_code.replace("#pragma CMT", "#")
            wrapper.define_kernel(kernel_name, src_code)
        kernel.call_kernel(wrapper, kernel_name)
        self.scheduler.free_buffers()

    @staticmethod
    @functools.lru_cache(32)
    def candidate_tilings(node):
        ranges, reduction_ranges = node.get_ranges()
        if len(ranges) <= 1:
            return ()

        rw = node.pointwise_read_writes()
        assert len(rw.range_vars) == len(ranges)

        deps = [
            dep
            for dep in itertools.chain(rw.reads, rw.writes)
            if dep.name not in V.graph.removed_buffers
        ]
        write_names = {dep.name for dep in rw.writes}

        tilings = []

        for dep in deps:
            strides = V.graph.sizevars.stride_hints(dep.index, rw.range_vars)
            assert len(strides) == len(ranges)
            try:
                split = strides.index(1) + 1
                if split == len(ranges):
                    continue
                if all(s == 0 for s in strides[split:]):
                    # if this is a broadcasted tensor and all dimensions after split are broadcast,
                    # this is not a real split
                    continue

            except ValueError:
                continue
            tiled_groups = (
                V.graph.sizevars.simplify(sympy_product(ranges[:split])),
                V.graph.sizevars.simplify(sympy_product(ranges[split:])),
            )
            # score by number of elements
            score = V.graph.sizevars.size_hint(
                sympy_product(
                    size for size, stride in zip(ranges, strides) if stride != 0
                )
            )
            if dep.name in write_names:
                # ngimel said contiguous writes is more important than reads
                score *= 2
            if CandidateTiling.is_good_size(tiled_groups[0]):
                score *= 2
            if CandidateTiling.is_good_size(tiled_groups[1]):
                score *= 2

            if (
                V.graph.sizevars.size_hint(
                    score - sympy_product(itertools.chain(ranges, reduction_ranges))
                )
                >= 0
            ):
                tilings.append(CandidateTiling(tiled_groups, score, dep.name))
        return tilings

    @classmethod
    def select_tiling(cls, node_schedule, numel, reduction_numel=sympy.Integer(1)):
        """
        Heuristics to decide how to tile kernels.
        Currently, we tile based on stride-1 dimensions.

        Returns:
            `(tile1, tile2, reduction_numel)` s.t. `tile1 * tile2 == numel`

        """
        if reduction_numel != 1 or config.triton.max_tiles <= 1:
            # TODO(jansel): should we tile reductions?
            return (numel, reduction_numel)

        seen_names = set()
        candidate_tiles = collections.Counter()
        for node in EnableReduction.filter(node_schedule):
            for tiling in cls.candidate_tilings(node):
                if tiling.name in seen_names:
                    continue
                seen_names.add(tiling.name)
                candidate_tiles[tiling.tiling] += tiling.score

        ranked_tilings = [tiling for tiling, score in candidate_tiles.most_common()]

        if config.triton.max_tiles >= 3:
            # Add one 3D tiling choice
            for i in range(1, len(ranked_tilings)):
                a0, a1 = ranked_tilings[0]
                b0, b1 = ranked_tilings[i]
                if V.graph.sizevars.size_hint(a1 - b1) == 0:
                    continue
                if V.graph.sizevars.size_hint(a1 - b1) < 0:
                    # swap so a0 is bigger
                    a0, a1 = ranked_tilings[i]
                    b0, b1 = ranked_tilings[0]
                assert V.graph.sizevars.size_hint(a1 - b1) > 0
                if V.graph.sizevars.maybe_guard_multiple_of(a1, b1):
                    tiling = (a0, ir.IndexingDiv(a1, b1), b1)
                    ranked_tilings = [tiling] + ranked_tilings
                    break  # only 1 choice for now

        for tiled_groups in ranked_tilings:
            new_groups = (*tiled_groups, reduction_numel)
            if all(
                TritonKernel.is_compatible(new_groups, node.get_ranges())
                for node in node_schedule
                if isinstance(node, scheduler.SchedulerNode)
            ):
                return new_groups

        return (numel, reduction_numel)

    def flush(self):
        pass


@dataclasses.dataclass
class CandidateTiling:
    tiling: List[sympy.Expr]
    score: int  # higher is better
    name: str = None

    @staticmethod
    def is_good_size(s):
        """Somewhat arbitrary heuristic used to boost scores for some sizes"""
        s = V.graph.sizevars.size_hint(s)
        return s >= 32 and (s % 32 == 0)


class DisableReduction:
    """
    Marker to invoke `kernel.disable_reduction()`.  This closes a
    reduction loop and allows for pointwise ops to occur on the output
    of a reduction.
    """


class EnableReduction:
    """
    Marker to end a DisableReduction block.
    """

    @staticmethod
    def filter(node_schedule):
        """
        Get the nodes from node_schedule skipping those in a
        DisableReduction block.
        """
        disabled = False
        for node in node_schedule:
            if node in (EnableReduction, DisableReduction):
                # Don't tile stuff outside the main reduction loop
                disabled = node is DisableReduction
            elif disabled:
                pass
            else:
                yield node


class CantSplit(Exception):
    pass
