import dataclasses
import functools
import itertools

import operator
from typing import Dict, List, Set

import sympy
from triton.runtime.jit import JITFunction

from .. import config, ir
from ..utils import instance_descriptor, sympy_symbol
from ..virtualized import V
from .common import CSEVariable, Kernel, PythonPrinter, SizeArg, TensorArg


class TritonCSEVariable(CSEVariable):
    def __init__(self, name):
        super().__init__(name)
        # We'll use this to track which masks the variable needs when used for indirect indexing
        self.mask_vars: Set[str] = set()

    def update_on_args(self, name, args, kwargs):
        # When making a variable that is going to be used in indirect indexing
        # if a where clause is used it should mean that the result is always a
        # valid index, so you shouldn't include any of the dependent variables
        # in the resulting load mask
        if name == "where":
            return
        for arg in args:
            if isinstance(arg, TritonCSEVariable):
                self.mask_vars.update(arg.mask_vars)


def signature_of(arg):
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
        if isinstance(x, SizeArg):
            if isinstance(x.expr, (int, sympy.Integer)):
                # TODO(voz): These are kinda redundant, if we can solve out statically_known_multiple_of with
                # _maybe_evaluate_static...
                return V.graph.sizevars.statically_known_multiple_of(x.expr, ALIGNMENT)
            else:
                return False
        raise NotImplementedError(f"unhandled {type(x)}: {x}")

    if config.triton.divisible_by_16:
        divisible_by_16 = [i for i, arg in enumerate(args) if is_aligned(arg)]
    else:
        divisible_by_16 = []
    return instance_descriptor(tuple(divisible_by_16), ())


class TritonPrinter(PythonPrinter):
    def _print_floor(self, expr):
        assert len(expr.args) == 1
        return f"tl.math.floor({self.paren(self._print(expr.args[0]))})"


texpr = TritonPrinter().doprint


def split_iteration_ranges(groups: List[sympy.Expr], lengths: List[List[sympy.Expr]]):
    sv = V.graph.sizevars
    new_ranges = [[] for _ in groups]
    remaining = [sv.simplify(g) for g in groups]
    var_count = itertools.count()

    def add_range(i, expr):
        expr = sv.simplify(expr)
        if not sv.statically_known_multiple_of(remaining[i], expr):
            raise CantSplit()
        # guard on the last item out
        remaining[i] = ir.FloorDiv(remaining[i], expr)
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
            if sv.statically_known_equals(size, 1):
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
                if not sv.statically_known_multiple_of(size, remaining[current_group]):
                    raise CantSplit()
                size1 = remaining[current_group]
                size2 = ir.FloorDiv(size, remaining[current_group])
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
        *,
        kernel: "Kernel",
        divisor=sympy.Integer(1),
        length=sympy.Integer(1),
    ):
        super().__init__()
        self.name = name
        self.var_list = var_list
        self.var_ranges = var_ranges
        self.numel = numel
        self.prefix = prefix
        self.divisor = divisor
        self.length = length
        self.kernel = kernel

    def is_loop(self):
        return self.prefix == "r" and not self.kernel.persistent_reduction


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
        super().__init__(
            name=name,
            var_list=[],
            var_ranges={},
            numel=numel,
            prefix=prefix,
            kernel=kernel,
        )
        self.index = index
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
        if V.graph.sizevars.statically_known_equals(divisor * length, self.numel):
            expr = ir.FloorDiv(sympy_symbol(f"{self.prefix}index"), divisor)
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

    def construct_entries(self, lengths: List[sympy.Expr]):
        divisor = sympy.Integer(1)
        itervars = []
        for length in reversed(lengths):
            itervars.append(self.lookup(divisor, length))
            divisor = divisor * length
        return list(reversed(itervars))

    def construct(self, lengths: List[sympy.Expr]):
        return [e.symbol() for e in self.construct_entries(lengths)]

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
            if not V.graph.sizevars.statically_known_equals(node.divisor, divisor):
                # fill in unused index var
                add(self.lookup(divisor, ir.FloorDiv(node.divisor, divisor)))
                divisor = node.divisor
            add(node)
        if not V.graph.sizevars.statically_known_equals(self.numel, divisor):
            # fill in unused index var
            add(self.lookup(divisor, ir.FloorDiv(self.numel, divisor)))

        return list(reversed(index_vars)), list(reversed(sizes))

    def ranges_code(self):
        size = self.kernel.indexing_size_str(self.index, self.prefix)
        index_dtype = self.kernel.index_dtype
        convert = f".to({index_dtype})" if index_dtype != "tl.int32" else ""
        return f"tl.arange(0, {self.prefix.upper()}BLOCK){size}{convert}"

    def get_pid(self):
        key = f"tl.program_id({self.index})"
        pid = self.pid_cache.get(key, key)
        if self.kernel.index_dtype != "tl.int32":
            return f"{pid}.to({self.kernel.index_dtype})"
        return pid

    def codegen_header(self, code):
        x = self.prefix
        if self.is_loop():
            code.writeline(f"{self.name} = {x}offset + {x}base")
        elif x == "r" and self.kernel.persistent_reduction:
            # no need to "roffset = "
            code.writeline(
                f"{self.name} = {self.ranges_code()}",
            )
        else:
            code.writelines(
                [
                    f"{x}offset = {self.get_pid()} * {x.upper()}BLOCK",
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
        super().__init__(
            name=name,
            numel=parent.numel / length,
            var_list=parent.var_list,
            var_ranges=parent.var_ranges,
            prefix=parent.prefix,
            divisor=divisor,
            length=length,
            kernel=parent.kernel,
        )
        self.parent = parent
        self.expr = expr
        self.codegen = functools.lru_cache(None)(self._codegen)
        self.code = functools.lru_cache(None)(self._code)

    def set_name(self, name):
        self.codegen = lambda: name
        self.codegen.cache_clear = lambda: None
        self.name = name

    def cache_clear(self):
        self.codegen.cache_clear()

    def _code(self):
        return f"{self.name} = " + texpr(V.kernel.rename_indexing(self.expr))

    def codegen_into(self, buffer):
        buffer.writeline(self._code())

    def _codegen(self):
        if self.is_loop():
            self.codegen_into(V.kernel.indexing_code)
        else:
            # lift non-reduction stores outside loop
            self.codegen_into(V.kernel.body)

        return self.name

    def precomputed_args(self):
        # for dynamic shapes, find parts of indexing expressions that have to be precomputed
        precomputed_args = []
        if isinstance(self.expr, sympy.Symbol):
            return precomputed_args
        assert isinstance(self.expr, (ir.FloorDiv, ir.ModularIndexing)), type(self.expr)
        for arg in self.expr.args[1:]:
            if not isinstance(arg, (sympy.Integer, sympy.Symbol)):
                symbols = arg.free_symbols
                if len(symbols) > 0 and all(s.name.startswith("s") for s in symbols):
                    precomputed_args.append(arg)
        return precomputed_args

    def symbol(self):
        return sympy_symbol(self.name)

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return self.name == other.name
