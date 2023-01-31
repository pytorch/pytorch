import dataclasses
import functools
import itertools
import logging
import math
import operator
from typing import Dict, Iterable, Union

import sympy

import torch
from .ir import IndexingDiv, InterpreterShim, LoopBody, ModularIndexing
from .utils import sympy_subs
from .virtualized import V

log = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class ValueRanges(object):
    lower: Union[sympy.Expr, sympy.Number, int, float, bool]
    upper: Union[sympy.Expr, sympy.Number, int, float, bool]

    def __contains__(self, x):
        # TODO This needs to be generalised if lower/upper are sympy.Expr
        assert not isinstance(x, sympy.Expr)
        return self.lower <= x <= self.upper

    @classmethod
    def wrap(cls, arg):
        if isinstance(arg, ValueRanges):
            return arg
        assert isinstance(arg, (int, float, bool))
        return ValueRanges(arg, arg)

    @classmethod
    def increasing_map(cls, x, fn):
        """map lower and upper bound with fn"""
        x = cls.wrap(x)
        return ValueRanges(fn(x.lower), fn(x.upper))

    @classmethod
    def decreasing_map(cls, x, fn):
        """map lower bound to upper bound and upper bound to lower bound"""
        x = cls.wrap(x)
        return ValueRanges(fn(x.upper), fn(x.lower))

    @classmethod
    def monotone_map(cls, x, fn):
        """check the max and min of computed upper and lower bound for the output"""
        x = cls.wrap(x)
        l = fn(x.lower)
        u = fn(x.upper)
        return ValueRanges(min(l, u), max(l, u))

    @classmethod
    def convex_min_zero_map(cls, x, fn):
        """the max is at one of the ends"""
        x = ValueRanges.wrap(x)
        if 0 in x:
            return ValueRanges(0, max(fn(x.lower), fn(x.upper)))
        else:
            return cls.monotone_map(x, fn)

    @classmethod
    def coordinatewise_increasing_map(cls, x, y, fn):
        """map upper and lower bounds accessing corresponding values of inputs"""
        x, y = cls.wrap(x), cls.wrap(y)
        return ValueRanges(
            fn(x.lower, y.lower),
            fn(x.upper, y.upper),
        )

    @classmethod
    def coordinatewise_monotone_map(cls, x, y, fn):
        """compute the product of all lower and upper bounds and take min and max"""
        x, y = cls.wrap(x), cls.wrap(y)
        products = [
            fn(a, b)
            for a, b in itertools.product([x.lower, x.upper], [y.lower, y.upper])
        ]
        return ValueRanges(min(products), max(products))


class ValueRangeAnalysis(object):
    def __init__(self):
        self.name = "ValueRangeAnalysis"
        boolean_operators = (
            "eq",
            "ne",
            "lt",
            "gt",
            "le",
            "ge",
            "and_",
            "or_",
            "xor",
            "logical_and",
            "logical_or",
            "logical_not",
        )
        for op in boolean_operators:
            setattr(self, op, self.bool_handler)

    @staticmethod
    def bool_handler(*args, **kwargs):
        # just assuming bools can have both values
        return ValueRanges(
            sympy.logic.boolalg.BooleanFalse, sympy.logic.boolalg.BooleanTrue
        )

    @staticmethod
    def default_handler(*args, **kwargs):
        # many ops are unlikely to show up in optimizable indexing compute,
        # so we dont have full coverage
        return ValueRanges(-math.inf, math.inf)

    def load(self, name: str, index: sympy.Expr):
        return ValueRanges(-math.inf, math.inf)

    def store(self, name, index, value, mode=None):
        return

    def reduction(self, name, dtype, src_dtype, reduction_type, index, value):
        return ValueRanges(-math.inf, math.inf)

    def index_expr(self, index, dtype):
        assert isinstance(index, ValueRanges)
        return index

    @staticmethod
    def to_dtype(x, dtype: torch.dtype):
        def is_bool(val):
            return isinstance(val, bool) or (
                hasattr(val, "is_Boolean") and val.is_Boolean
            )

        x = ValueRanges.wrap(x)
        low, up = x.lower, x.upper
        if is_bool(low):
            assert is_bool(up)
            if dtype.is_floating_point:
                return ValueRanges(sympy.Float(0.0), sympy.Float(1.0))
            else:
                return ValueRanges(sympy.Integer(0), sympy.Integer(1))
        return ValueRanges.wrap(x)

    @staticmethod
    def constant(value, dtype):
        # using nan makes subsequent computation throw, and for the purposes of optimization
        # returning -math.inf - math.inf is equivalent to giving up
        if math.isnan(value):
            return ValueRanges(-math.inf, math.inf)
        if isinstance(value, int):
            return ValueRanges(sympy.Integer(value), sympy.Integer(value))
        else:
            return ValueRanges(sympy.Float(value), sympy.Float(value))

    @staticmethod
    def reciprocal(x):
        x = ValueRanges.wrap(x)
        if 0 in x:
            return ValueRanges(-math.inf, math.inf)
        else:
            return ValueRanges.decreasing_map(x, lambda y: 1 / y)

    @staticmethod
    def square(x):
        return ValueRanges.convex_min_zero_map(x, lambda y: y * y)

    @staticmethod
    def abs(x):
        return ValueRanges.convex_min_zero_map(x, abs)

    @staticmethod
    def neg(x):
        return ValueRanges.decreasing_map(x, operator.neg)

    @staticmethod
    def truediv(a, b):
        b = ValueRanges.wrap(b)
        if 0 in b:
            return ValueRanges(-math.inf, math.inf)
        else:
            return ValueRangeAnalysis.mul(a, ValueRanges(1 / b.upper, 1 / b.lower))

    @staticmethod
    def div(a, b):
        # We think of this as floor(a / b)
        out = ValueRangeAnalysis.truediv(a, b)
        return ValueRangeAnalysis.floor(out)

    @staticmethod
    def add(a, b):
        return ValueRanges.coordinatewise_increasing_map(a, b, operator.add)

    @staticmethod
    def mul(a, b):
        return ValueRanges.coordinatewise_monotone_map(a, b, operator.mul)

    @staticmethod
    def sub(a, b):
        b = ValueRanges.wrap(b)
        return ValueRangeAnalysis.add(a, ValueRanges(-b.upper, -b.lower))

    @staticmethod
    def exp(x):
        return ValueRanges.increasing_map(x, sympy.functions.elementary.exponential.exp)

    @staticmethod
    def log(x):
        return ValueRanges.increasing_map(
            x, lambda y: -math.inf if y <= 0 else sympy.log(y)
        )

    @staticmethod
    def sqrt(x):
        return ValueRanges.increasing_map(x, sympy.sqrt)

    @staticmethod
    def pow(a, b):
        def is_integer(val):
            return (
                isinstance(val, int)
                or (isinstance(val, float) and val == int(val))
                or (hasattr(val, "is_integer") and val.is_integer)
            )

        a = ValueRanges.wrap(a)
        b = ValueRanges.wrap(b)
        if a.lower < 0 and not is_integer(b.lower):
            # The function is not defined
            return ValueRanges(-math.inf, math.inf)
        elif 0 in a and b.lower <= 0:
            return ValueRanges(-math.inf, math.inf)
        return ValueRanges.coordinatewise_monotone_map(a, b, operator.pow)

    @staticmethod
    def minimum(a, b):
        return ValueRanges.coordinatewise_increasing_map(a, b, min)

    @staticmethod
    def maximum(a, b):
        return ValueRanges.coordinatewise_increasing_map(a, b, max)

    @staticmethod
    def where(a, b, c):
        b = ValueRanges.wrap(b)
        c = ValueRanges.wrap(c)
        return ValueRanges(min(b.lower, c.lower), max(b.upper, c.upper))

    @staticmethod
    def floor(x):
        return ValueRangeAnalysis.floor_ceil(
            x, sympy.functions.elementary.integers.floor
        )

    @staticmethod
    def ceil(x):
        return ValueRangeAnalysis.floor_ceil(
            x, sympy.functions.elementary.integers.ceiling
        )

    @staticmethod
    def floor_ceil(x, fn_int):
        def is_integer(val):
            return isinstance(val, int) or (
                hasattr(val, "is_integer") and val.is_integer
            )

        if is_integer(x):
            fn = fn_int
        else:

            def fn(x):
                return sympy.core.numbers.Float(fn_int(x))

        return ValueRanges.increasing_map(x, fn)

    def __getattr__(self, name):
        log.warning(f"unhandled ValueRange op {name}")
        return self.default_handler


def dominated_nodes(
    initial_queue: Union[torch.fx.Node, Iterable[torch.fx.Node]], skip_filter=None
):
    """Returns the set of nodes whose values depend on those within initial_queue"""
    if isinstance(initial_queue, torch.fx.Node):
        initial_queue = [initial_queue]

    dominated_set = set(initial_queue)

    while initial_queue:
        node = initial_queue.pop()
        for user in node.users:
            if skip_filter and skip_filter(user):
                continue
            if user not in dominated_set:
                dominated_set.add(user)
                initial_queue.append(user)

    return dominated_set


def val_expressable_in_32_bits(val):
    if hasattr(val, "is_Boolean") and val.is_Boolean:
        return True

    if isinstance(val, sympy.Expr):
        assert val.is_constant()
        if val.is_Integer or val.is_Boolean:
            val = int(val)
        else:
            val = float(val)

    # bound within mantissa
    if isinstance(val, float):
        return val <= (2**24) and val >= -(2**24)

    if isinstance(val, int):
        iinfo = torch.iinfo(torch.int32)
        return val <= iinfo.max and val >= iinfo.min

    raise Exception(f"Unexpected value {val}")


def range_expressable_in_32_bits(range):
    return val_expressable_in_32_bits(range.lower) and val_expressable_in_32_bits(
        range.upper
    )


class OptimizeIndexing(object):
    """
    Performs Value Range Analysis on LoopBody's fx graph to reduce precision of
    intermediaries from int64 to int32. This is an important optimization for indexing
    kernels such as Upsample and Interpolate.
    """

    def __init__(
        self,
        loop_body: LoopBody,
        indices_ranges: Dict[sympy.Symbol, int],
        indexing_exprs: Dict[str, sympy.Expr],
    ):
        self.loop_body = loop_body
        self.indices_range = indices_ranges
        self.indexing_exprs = indexing_exprs
        self.replacement_vals = {}
        self.interp_env = {}
        self.submodules = self.swap_submodules(dict(loop_body.submodules))

        indirect_var_set = set(loop_body.indirect_vars)
        self.index_indirect_dependecies = {
            index: expr.free_symbols & indirect_var_set
            for index, expr in indexing_exprs.items()
        }
        self.all_graphs = [loop_body.root_block.graph] + [
            block.graph for block in loop_body.subblocks.values()
        ]

        for k, v in indices_ranges.items():
            self.replace_indirect(k, ValueRanges(0, v))

        # avoid computing these values, pessimistically assume that they are unbounded
        self.tensor_values_set = dominated_nodes(
            [
                node
                for node in self.all_nodes
                if node.target in ["load", "reduction"]
                or "masked_subblock" in node.target
            ]
        )

    def run(self):
        """Compute Value Ranges and try reduce precision of 'to_dtype' nodes to int32 where possible"""

        int64_dtype_nodes = [
            node
            for node in self.all_nodes
            if (
                node.target == "to_dtype"
                and node.args[2] == torch.int64
                and node not in self.tensor_values_set
            )
        ]
        if not int64_dtype_nodes:
            return

        for node in self.tensor_values_set:
            # we need to evaluate masked_subblock to recurse, and we need to set indirect values
            if (
                "masked_subblock" not in node.target
                and "set_indirect" not in node.target
            ):
                self.interp_env[node] = torch._inductor.optimize_indexing.ValueRanges(
                    -math.inf, math.inf
                )

        interpreter = InterpreterShim(self.loop_body.root_block.graph, self.submodules)
        interpreter.run(V.get_ops_handler(), initial_env=self.interp_env)

        # TODO - if dominated node of one to_dtype is not expressible in int32,
        # we should short circuit another to_dtype node if that node also dominates
        for node in int64_dtype_nodes:
            self.try_to_reduce_precision(node)

    def try_to_reduce_precision(self, node):
        # if a downstream use of a node explicitly converts to int32, or float16/float32/float64,
        # then it's precision is set for that chain of uses, and we don't need to consider those
        # dominated values
        def skip_filter(node):
            return node.target == "to_dtype" and node.args[2] in (
                torch.int32,
                torch.float32,
                torch.float64,
            )

        # TODO - there are dominated uses whose dtype does not depend on whether
        # we reduce the precision here, e.g. add(int64, int64) one of the args can be reduced to
        # int32 without changing the output precision of the node. this case hasn't shown up
        for dominated in dominated_nodes(node, skip_filter):
            if dominated.target in ["store", "output"]:
                continue

            if "set_indirect" in dominated.target:
                idx = int(dominated.target[len("set_indirect") :])
                indirect_var = self.loop_body.indirect_vars[idx]

                for index, indirect_vals in self.index_indirect_dependecies.items():
                    if indirect_var in indirect_vals:
                        index_val = self.replacement_vals[index]

                        if math.isinf(index_val.lower) or math.isinf(index_val.upper):
                            return

                        # all indices are integers, so make sure that we
                        # use the bounds of integers instead of floats.
                        # TODO - not sure if we should be doing int/float casts while tracing,
                        # might interfere with sympy.

                        index_val_int = ValueRanges(
                            int(index_val.lower), int(index_val.upper)
                        )
                        if not range_expressable_in_32_bits(index_val_int):
                            return

            if not range_expressable_in_32_bits(self.interp_env[dominated]):
                return

        args = list(node.args)
        args[2] = torch.int32
        node.args = tuple(args)

    @property
    def all_nodes(self):
        for graph in self.all_graphs:
            for node in graph.nodes:
                yield node

    def swap_submodules(self, submodules):
        keys = list(submodules.keys())
        for key in keys:
            if key == "get_index":
                submodules[key] = self.get_index
            elif "masked_subblock" in key:
                subblock = self.loop_body.subblocks[key]
                submodules[key] = functools.partial(
                    self.masked_subblock, subblock, self.interp_env
                )
            else:
                assert "set_indirect" in key
                idx = int(key[len("set_indirect") :])
                var = self.loop_body.indirect_vars[idx]
                indirect = functools.partial(self.set_indirect, var)
                submodules[key] = indirect

        return submodules

    def masked_subblock(self, subblock, env, mask, value):
        interp = InterpreterShim(subblock.graph, self.submodules)
        interp.run(V.get_ops_handler(), initial_env=env)
        output = [node for node in subblock.graph.nodes if node.target == "output"]
        assert len(output) == 1
        # dont bother unioning with value since the load from buffer will be
        # pessimistically assumed to be inf anyway
        return interp.env[output[0]]

    def set_indirect(self, var, new_var):
        self.replace_indirect(var, new_var)
        return new_var

    def replace_indirect(self, old, new):
        """Swap in a variable used in indirect indexing"""
        assert isinstance(new, ValueRanges)
        self.replacement_vals[old] = new

    def get_index(self, name):
        if name in self.replacement_vals:
            return self.replacement_vals[name]

        out = self._get_index_impl(name)
        self.replacement_vals[name] = out
        return out

    def _get_index_impl(self, name):
        expr = self.indexing_exprs[name]

        free_symbols = list(expr.free_symbols)

        if len(free_symbols) == 0:
            return ValueRanges(expr, expr)

        if expr in self.replacement_vals:
            return self.replacement_vals[expr]

        def replace_symbols_for_deriv(expr, ignore_mod=False):
            # for the purposes of finding local, minimum, maximum, assume smoothness
            def mod_indexing_rep(x, y, z):
                if z.is_constant():
                    return x / y

                # never really happens, we'll bail on optimizing
                return (x / y) % z

            def indexing_div_rep(x, y):
                return x / y

            return expr.replace(ModularIndexing, mod_indexing_rep).replace(
                IndexingDiv, indexing_div_rep
            )

        symbols = expr.free_symbols
        monotonic_increasing = []
        monotonic_decreasing = []
        other_symbols = []

        expr_for_deriv = replace_symbols_for_deriv(expr, True)
        for symbol in symbols:
            diff = sympy.diff(expr_for_deriv, symbol)
            if diff.is_positive:
                monotonic_increasing.append(symbol)
            elif diff.is_positive is False:  # can return None
                monotonic_decreasing.append(symbol)
            else:
                other_symbols.append(symbol)

        if not other_symbols:
            max_val = sympy_subs(
                expr,
                {
                    k: (v.upper if k in monotonic_increasing else v.lower)
                    for k, v in self.replacement_vals.items()
                },
            )
            min_val = sympy_subs(
                expr,
                {
                    k: (v.lower if k in monotonic_increasing else v.upper)
                    for k, v in self.replacement_vals.items()
                },
            )
            return ValueRanges(min_val, max_val)
        else:
            # bail on optimizing, have not run into this yet
            return ValueRanges(-math.inf, math.inf)


def indexing_dtype_strength_reduction(loop_body: LoopBody):
    """
    Performs Value Range Analysis on LoopBody's fx graph to reduce precision of
    intermediaries from int64 to int32
    """
    indices = dict(loop_body.var_ranges)
    indexing = dict(loop_body.indexing_exprs)
    with V.set_ops_handler(ValueRangeAnalysis()):
        OptimizeIndexing(loop_body, indices, indexing).run()
