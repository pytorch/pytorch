import torch
import torch.utils._pytree as pytree
from typing import Set, Dict, List, Type, Optional, cast, Union
import sys
import operator
import itertools
import builtins
import math
import functools
import threading
from contextlib import contextmanager
from functools import lru_cache, partial
import traceback
import collections
import textwrap
from torch._subclasses.meta_utils import MetaConverter
from torch import SymInt, SymFloat

try:
    import sympy  # type: ignore[import]
    from sympy.printing.precedence import precedence  # type: ignore[import]
    HAS_SYMPY = True
except ImportError:
    HAS_SYMPY = False

aten = torch.ops.aten  # type: ignore[has-type]

__all__ = [
    "has_symbolic_sizes_strides", "create_contiguous", "ShapeEnv",
    "SymDispatchMode", "sym_int", "sym_float", "FloorDiv", "guard_int", "wrap_node",
    "sym_sqrt",
]

SYM_FUNCTION_MODE = None

# We don't bother with the metaclass as all of the dispatching logic happens
# entirely from Python
#
# Didn't bother with ancestors for now, unlikely to have multiple modes for
# symints right now


# SymDispatchMode gets invoked whenever an operation is processed on
# a PySymInt.  When this occurs, you get called at __sym_dispatch__
# with the operation in question.  This is symmetric to TorchDispatchMode
# but with some caveats:
#
#   - In TorchDispatchMode, you get the same arguments as what a user
#     invoked your API with; e.g., if you call torch.ops.aten.foo(a, b),
#     you get (a, b) as args to your call.  In SymDispatchMode, if
#     you call a + b (where a and b are SymInts), you will get
#     (a.get_pyobj(), b.get_pyobj()) as your args (these are PySymInts)
#
#   - SymInt/PySymInt don't have FX proxy support (unlike, e.g., Tensor).
#     So you have to manually call Tracer/create_node to write into
#     the graph.  See ProxySymDispatchMode for an example
#
class SymDispatchMode:
    def __sym_dispatch__(self, func, types, args, kwargs):
        raise NotImplementedError()

    def __enter__(self):
        global SYM_FUNCTION_MODE
        old = SYM_FUNCTION_MODE
        if hasattr(self, "inner"):
            raise RuntimeError(f"{self} has already been used as a mode. Please use a fresh version")
        else:
            self.inner = old
        SYM_FUNCTION_MODE = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        global SYM_FUNCTION_MODE
        SYM_FUNCTION_MODE = self.inner

def has_symbolic_sizes_strides(elem):
    return elem._has_symbolic_sizes_strides

def create_contiguous(shape):
    strides = [1]
    for dim in reversed(shape[:-1]):
        strides.append(dim * strides[-1])
    return list(reversed(strides))

def _handle_sym_dispatch(func, args, kwargs):
    global SYM_FUNCTION_MODE
    mode = SYM_FUNCTION_MODE
    assert mode
    SYM_FUNCTION_MODE = mode.inner
    try:
        # TODO: properly compute types
        types: List[Type] = []
        return mode.__sym_dispatch__(func, types, args, kwargs)
    finally:
        SYM_FUNCTION_MODE = mode

def guard_int(a):
    if isinstance(a, SymInt):
        return a.node.guard_int("", 0)  # NB: uses Python backtrace
    assert isinstance(a, int)
    return a

def sym_float(a):
    if isinstance(a, SymFloat):
        return a
    elif hasattr(a, '__sym_float__'):
        return a.__sym_float__()
    return float(a)

# Drop in replacement for math.sqrt
def sym_sqrt(a):
    if hasattr(a, '__sym_sqrt__'):
        return a.__sym_sqrt__()
    return math.sqrt(a)

# Drop in replacement for math.floor/ceil.  Actually, math.floor/ceil
# directly usable, but this has a more relaxed type signature for mypy
# (mypy requires SupportFloat which is too strict)
def sym_floor(a):
    return math.floor(a)  # type: ignore[type]

def sym_ceil(a):
    return math.ceil(a)  # type: ignore[type]

def sym_int(a):
    if isinstance(a, SymInt):
        return a
    elif isinstance(a, SymFloat):
        return sym_floor(a) if a > 0 else sym_ceil(a)
    return int(a)

def to_node(self, num):
    if isinstance(num, (SymInt, SymFloat)):
        return num.node
    elif isinstance(num, int):
        return self.wrap_int(num)
    elif isinstance(num, float):
        return self.wrap_float(num)
    else:
        # NotImplemented is important so that Python tries the
        # other magic method
        return NotImplemented

# TODO: An incomplete list
# 1. Set variables to be equal when we do equality
# 2. Specialize on 0/1 when we do subtraction
class SymNode:
    """
    This is a type erased SymInt/SymFloat which we use to do actual operations.
    End users don't touch this.  Magic methods are NOT defined on this object.
    """
    def __init__(self, expr, symbol, shape_env, pytype, constant=None):
        self._expr = expr
        # Unlike expr, sympy.Symbol is guaranteed to be a symbol,
        # and it never gets simplified into a constant or another symbol.
        # This only exists for fresh create_symint; intermediate values
        # don't have this set
        self.symbol: sympy.Symbol = symbol
        self.shape_env = shape_env
        self.pytype = pytype
        self.constant = constant

    @property
    def expr(self):
        self._update_expr()
        return self._expr

    def _update_expr(self):
        self._expr = self.shape_env.replace(self._expr)

    def is_int(self):
        return self.pytype is int

    def is_float(self):
        return self.pytype is float

    def wrap_int(self, num):
        assert isinstance(num, int)
        return SymNode(sympy.Integer(num), None, self.shape_env, int, constant=num)

    def wrap_float(self, num):
        assert isinstance(num, float)
        return SymNode(sympy.Float(num), None, self.shape_env, float, constant=num)

    def clone(self):
        return self

    def str(self):
        return f"{self.expr}"

    def __str__(self):
        return self.str()

    def __repr__(self):
        return self.str()

    # These methods are metaprogrammed in below
    def sym_int(self) -> "SymNode":
        ...

    def sym_float(self) -> "SymNode":
        ...

    # Today we error on calling int on a symbolic shape, as this is a very accessible footgun.
    def int_(self):
        raise RuntimeError("Trying to extract a concrete int out of a symbolic int")

    # You can manually trigger a guard with this function
    def guard_int(self, file, line):
        # TODO: use the file/line for some useful diagnostic on why a
        # guard occurred
        return int(self.shape_env.evaluate_expr(self.expr))

    def guard_float(self, file, line):
        # TODO: use the file/line for some useful diagnostic on why a
        # guard occurred
        return float(self.shape_env.evaluate_expr(self.expr))

    def bool_(self):
        return bool(self.shape_env.evaluate_expr(self.shape_env.replace(self.expr)))


if HAS_SYMPY:
    class FloorDiv(sympy.Function):
        """
        We maintain this so that:
        1. We can use divisibility guards to simplify FloorDiv(a, b) to a / b.
        2. Printing out the expression is nicer (compared to say, representing a//b as (a - a % b) / b)
        """
        nargs = (2,)

        def _sympystr(self, printer):
            lhs = self.args[0]
            rhs = self.args[1]
            lhs_str = printer._print(lhs)
            rhs_str = printer._print(rhs)
            if precedence(lhs) < precedence(sympy.div):
                lhs_str = f"({lhs_str})"
            if precedence(rhs) < precedence(sympy.div):
                rhs_str = f"({rhs_str})"

            return f"{lhs_str}//{rhs_str}"

        @classmethod
        def eval(cls, base, divisor):
            if base == 0:
                return sympy.Integer(0)
            if divisor == 1:
                return base
            if isinstance(base, sympy.Integer) and isinstance(divisor, sympy.Integer):
                return base // divisor
            if isinstance(base, FloorDiv):
                return FloorDiv(base.args[0], base.args[1] * divisor)

            gcd = sympy.gcd(base, divisor)
            if gcd != 1:
                return FloorDiv(
                    sympy.simplify(base / gcd), sympy.simplify(divisor / gcd)
                )

# Methods that have a `__foo__` as well as `__rfoo__`
reflectable_magic_methods = {
    'add': lambda a, b: a + b,
    'sub': lambda a, b: a - b,
    'mul': lambda a, b: a * b,
    'mod': lambda a, b: a % b,
    'pow': lambda a, b: a ** b,
    'truediv': lambda a, b: a / b,
    'floordiv': lambda a, b: FloorDiv(a, b),
}

magic_methods = {
    **reflectable_magic_methods,
    'eq': lambda a, b: sympy.Eq(a, b),
    'gt': lambda a, b: sympy.Gt(a, b),
    'lt': lambda a, b: sympy.Lt(a, b),
    'le': lambda a, b: sympy.Le(a, b),
    'ge': lambda a, b: sympy.Ge(a, b),
    'floor': lambda a: sympy.floor(a),
    'sym_float': lambda a: a,  # Cannot use sympy.Float(a) here, coz it expects python literals
    'ceil': lambda a: sympy.ceiling(a),
    'neg': lambda a: -a,
    'min': lambda a, b: sympy.Min(a, b),
    'max': lambda a, b: sympy.Max(a, b),
    'sym_sqrt': lambda a: sympy.sqrt(a),
}

unary_magic_methods = {
    'sym_float',
    'ceil',
    'floor',
    'neg',
    'sym_sqrt',
}

magic_methods_on_builtins = {"min", "max"}
magic_methods_on_math = {"ceil", "floor"}
magic_methods_on_submodule = {"sym_float", "sym_sqrt"}

always_float_magic_methods = {"truediv", "sym_float", "sym_sqrt"}
always_int_magic_methods = {"ceil", "floor"}
always_bool_magic_methods = {"eq", "gt", "lt", "le", "ge"}

def wrap_node(x):
    # TODO: let C++ also take advantage of this
    if isinstance(x, SymNode) and x.constant is not None:
        return x.constant
    if x.is_int():
        return SymInt(x)
    elif x.is_float():
        return SymFloat(x)
    else:
        raise AssertionError(f"unrecognized return type {x}")

def _make_node_magic(method, func):
    func = lru_cache(256)(func)

    def binary_magic_impl(self, other):
        if method in magic_methods_on_builtins:
            op = getattr(builtins, method)
        else:
            op = getattr(operator, method)
        if SYM_FUNCTION_MODE:
            r = _handle_sym_dispatch(op, (wrap_node(self), wrap_node(other)), {})
            assert isinstance(r, (SymInt, SymFloat)), type(r)
            return r.node
        assert isinstance(other, SymNode)
        other_expr = other.expr
        # TODO: consider constant prop here
        expr = self.shape_env.replace(self.expr)
        other_expr = self.shape_env.replace(other_expr)
        out = func(expr, other_expr)
        out = sympy.expand(out)
        pytype: Type
        if method in always_float_magic_methods:
            pytype = float
        else:
            pytype = self.pytype

        # TODO: relational operators actually technically return a
        # PySymBool, this is a type error
        return SymNode(out, None, self.shape_env, pytype)

    def unary_magic_impl(self):
        if SYM_FUNCTION_MODE:
            if method in magic_methods_on_math:
                op = getattr(math, method)
            elif method in magic_methods_on_submodule:
                op = getattr(sys.modules[__name__], method)
            else:
                op = getattr(operator, method)
            r = _handle_sym_dispatch(op, (wrap_node(self),), {})
            assert isinstance(r, (SymInt, SymFloat)), type(r)
            return r.node
        # TODO: consider constant prop here
        expr = self.shape_env.replace(self.expr)
        out = func(expr)
        out = sympy.expand(out)
        pytype: Type
        if method in always_int_magic_methods:
            pytype = int
        elif method in always_float_magic_methods:
            pytype = float
        else:
            pytype = self.pytype

        return SymNode(out, None, self.shape_env, pytype)

    if method in unary_magic_methods:
        setattr(SymNode, method, unary_magic_impl)
    else:
        setattr(SymNode, method, binary_magic_impl)

for method, func in magic_methods.items():
    _make_node_magic(method, func)

def _make_user_magic(method, user_type):
    # User magic takes care of wrapping the other operand into a node,
    # so that our internal logic can assume everything is nodes

    def unary_magic_impl(self):
        return wrap_node(getattr(self.node, method)())

    def binary_magic_impl(self, other):
        other_node = to_node(self.node, other)
        if other_node is NotImplemented:
            return NotImplemented
        return wrap_node(getattr(self.node, method)(other_node))

    def rbinary_magic_impl(self, other):
        other_node = to_node(self.node, other)
        if other_node is NotImplemented:
            return NotImplemented
        return wrap_node(getattr(other_node, method)(self.node))

    if method in unary_magic_methods:
        setattr(user_type, f"__{method}__", unary_magic_impl)
    else:
        setattr(user_type, f"__{method}__", binary_magic_impl)
        if method in reflectable_magic_methods:
            setattr(user_type, f"__r{method}__", rbinary_magic_impl)

for method, func in magic_methods.items():
    _make_user_magic(method, SymInt)
    _make_user_magic(method, SymFloat)

del method
del func

def _lru_cache(fn, maxsize=None):
    """
    Wrapper around lru_cache that clears when new info about shapes has been
    updated.

    Use lru_cache if the output is always the same, regardless of the
    constraints we know now (i.e. evaluate_expr)

    Use _lru_cache otherwise.
    """
    fn_cache = lru_cache(maxsize)(fn)
    prior_key = None

    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        nonlocal prior_key
        if prior_key != self._get_key():
            prior_key = self._get_key()
            fn_cache.cache_clear()
        return fn_cache(self, *args, **kwargs)

    wrapper.cache_info = fn_cache.cache_info  # type: ignore[attr-defined]
    return wrapper


class IVarSymbol(sympy.Symbol):
    pass


class ShapeEnv(object):
    def __init__(self):
        self.guards = []
        # We have two types of symbols

        # Input symbols aka ivar (t0.size(0), t1.stride(2), t2.storage_offset(), etc)
        # are bound for input tensors and show up solely in guards, where
        # they encode things like 0/1 specialization (t0.size(0) == 1)
        # duck sizing (t1.size(0) == t2.size(1)) or duck striding
        # (t1.stride(0) == t1.size(0) * t1.stride(1)).  They NEVER show
        # up in intermediate size expressions; instead...

        # Compute symbols aka var (s0, s1, etc) are guaranteed to be greater than one,
        # and are used for our actual size expressions.  After allocating
        # an input symbol, we immediately either guard on it being zero
        # or one (eliminating the symbol entirely), or we introduce a companion
        # compute symbol t0.size(0) == s0.  Multiple input symbols may map
        # to the same compute symbol.  There is always at least one input symbol
        # equal to any given compute symbol (so that given inputs only, you can
        # evaluate expressions involving compute symbols).
        #
        # TODO: As a future performance optimization, we intend for compute
        # symbols to be guaranteed to be greater than zero, giving us the
        # representation t0.size(0) + 1 == s0.  This will prevent us from
        # needing to create a new shape_env in _maybe_evaluate_static, at the
        # cost of uglier SymPy expressions (but you should be able to simplify
        # for printing)

        # Original concrete values for ivars.  Probably technically not
        # necessary
        self.ivar_to_val: Dict["sympy.Symbol", "sympy.Integer"] = {}
        # The ivar chosen to represent a given var.  There may be multiple
        # ivars that are valid but only one is necessary; the rest have
        # guards asserting they're equal
        self.var_to_ivar: Dict["sympy.Symbol", "sympy.Symbol"] = {}
        # We want to give ivars more meaningful names for debugging
        # purposes, so we keep track of tensors/symints separately
        self.next_ivar_tensor = itertools.count()
        self.next_ivar_symint = itertools.count()

        # Maps symbolic ints to their original concrete values
        # Currently populated from tensors
        self.var_to_val: Dict["sympy.Symbol", "sympy.Integer"] = {}
        # Maps from sympy ints to expressions representing them
        # Populated from equality guards (i.e. a.shape[0] == b.shape[0])
        self.replacements: Dict["sympy.Symbol", "sympy.Expr"] = {}  #
        # Set holds a % b expressions that evaluate to 0.
        self.divisible: Set["sympy.Expr"] = set()
        # Duck-shaping says that if two input tensors have the same size,
        # they get assigned the same symbolic variable
        self.val_to_var: Dict[int, "sympy.Expr"] = {0: sympy.Integer(0), 1: sympy.Integer(1)}
        self.tls = threading.local()

    def _suppress_guards_tls(self):
        return getattr(self.tls, "suppress_guards", False)

    @contextmanager
    def suppress_guards(self):
        self.tls.suppress_guards = True
        try:
            yield
        finally:
            self.tls.suppress_guards = False

    def _get_key(self):
        """
        Defines the current "state" of the guards we've accumulated in this ShapeEnv.
        Determines when we need to invalidate our cache
        """
        return (len(self.replacements), len(self.divisible))

    def _build_symint(self, ivar, expr):
        if not self._suppress_guards_tls():
            stack = ''.join(traceback.format_list(traceback.extract_stack()[:-1]))
            if isinstance(expr, sympy.Symbol):
                self.var_to_ivar[expr] = ivar
            # This works because sympy will simplify double negative
            elif isinstance(-expr, sympy.Symbol):
                self.var_to_ivar[expr] = -ivar
            self._add_guard(
                sympy.Eq(ivar, expr),
                stack
            )
        return SymInt(SymNode(expr, ivar, self, int))

    def create_symbolic_sizes_strides_storage_offset(self, ex: torch.Tensor):
        """
        Returns a list of symbolic sizes and strides for the given tensor.
        After allocating ivars, we try to allocate a minimal amount of compute
        variables, e.g., by expressing stride in terms of sizes.
        """
        tid = next(self.next_ivar_tensor)

        dim = ex.dim()
        size = [self._create_symint(f"t{tid}.size({i})", v) for i, v in enumerate(ex.size())]
        storage_offset = self._create_symint(f"t{tid}.storage_offset()", ex.storage_offset())

        # Don't create full symints (which would allocate compute vars) for
        # these immediately, we may be able to avoid allocating a size var for
        # them
        stride_ivars = [self.create_ivar(f"t{tid}.stride({i})", v) for i, v in enumerate(ex.stride())]

        size_exprs: List[sympy.Expr] = [s.node.expr for s in size]
        stride_exprs: List[Optional[sympy.Expr]] = [None] * len(size)

        def duck_stride(i, expr):
            stride_exprs[i] = expr

        for i, val in enumerate(ex.stride()):
            if val in (0, 1):
                duck_stride(i, sympy.Integer(val))

        while any(x is None for x in stride_exprs):
            candidates = {
                ex.size(i) * ex.stride(i): size_exprs[i] * stride_exprs[i]
                for i in range(dim)
                if stride_exprs[i] is not None and ex.stride(i) >= 0
            }
            # iterate over unbound strides in sorted order
            val_list = sorted(
                [(ex.stride(i), i) for i in range(dim) if stride_exprs[i] is None]
            )
            for _, i in val_list:
                if stride_exprs[i] is None and ex.stride(i) in candidates:
                    r = candidates[ex.stride(i)]
                    duck_stride(i, r)
                    candidates[ex.size(i) * ex.stride(i)] = size_exprs[i] * r
            if any(x is None for x in stride_exprs):
                # bind the smallest unbound stride to a new variable
                val, i = min(
                    [
                        (ex.stride(i), i)
                        for i in range(dim)
                        if stride_exprs[i] is None
                    ]
                )
                fresh_var = self.create_var(ex.stride(i))
                duck_stride(i, fresh_var)
        assert all(x is not None for x in stride_exprs)
        stride = [self._build_symint(ivar, expr) for ivar, expr in zip(stride_ivars, stride_exprs)]
        return size, stride, storage_offset

    def create_symint(self, val: int) -> SymInt:
        sid = next(self.next_ivar_symint)
        return self._create_symint(f"v{sid}", val)

    def _create_symint(self, ivar_name: str, val: int) -> SymInt:
        ivar = self.create_ivar(ivar_name, val)
        symbol = self.create_var(val)
        return self._build_symint(ivar, symbol)

    def create_ivar(self, ivar_name: str, val: int) -> "sympy.Symbol":
        # NB: not guaranteed to be positive, could be zero
        r = IVarSymbol(ivar_name, integer=True)
        import traceback
        r.tb = traceback.extract_stack()
        self.ivar_to_val[r] = val
        return r

    def create_var(self, val: int) -> "sympy.Symbol":
        if not HAS_SYMPY:
            raise RuntimeError("Need sympy installed to create symbolic shapes")
        if val < 0:
            # all sympy base variables must be positive and > 1
            return -self.create_var(-val)
        # This implements duck-shaping: input sizes that match are assigned
        # the same symint
        # Note: val_to_var is also initialized with 0/1 mapping to constants, so
        # this also ensures that all symbols are > 1
        if val in self.val_to_var:
            return self.val_to_var[val]
        sympy_expr = sympy.Symbol(f"s{len(self.var_to_val)}", positive=True, integer=True)
        self.var_to_val[sympy_expr] = sympy.Integer(val)
        self.val_to_var[val] = sympy_expr
        return sympy_expr

    def evaluate_guards_for_args(self, *args):
        new_env = ShapeEnv()
        # NB: This must be kept in sync with create_aot_dispatcher_function
        # and wrap_fake_symbolic
        meta_converter = MetaConverter()
        pytree.tree_map_only(torch.Tensor, partial(meta_converter, shape_env=new_env), args)
        subst = new_env.var_to_val.copy()
        subst.update(new_env.ivar_to_val)
        return all(guard.xreplace(subst) for guard, _ in self.guards)

    def get_guard_expr(self):
        """
        Returns a sympy expression representing all of the shape env guards.

        NOTE: Does not include implicit 0/1 or duck-shaping guards
        """
        return sympy.And(*[guard for guard, _ in self.guards])

    def format_guards(self, verbose=False, *, exclude_ivar=False):
        def format_tb(tb):
            if not verbose:
                return ""
            return f"\n   Guarded at:\n{textwrap.indent(tb, '   ')}"

        def pred(expr):
            if not exclude_ivar:
                return True
            if isinstance(expr, sympy.Eq):
                if expr.lhs in self.ivar_to_val:
                    return False
            return True

        return '\n'.join(f" - {guard}{format_tb(tb)}" for guard, tb in self.guards if pred(guard))

    def get_shape_groups(self):
        shape_groups = collections.defaultdict(list)
        for k, v in self.replacements.items():
            shape_groups[v].append(k)
        return shape_groups

    @_lru_cache
    def _maybe_evaluate_static(self, expr: "sympy.Expr") -> "Optional[sympy.Expr]":
        """
        Tries to evaluate expr without introducing guards
        """
        expr = self.simplify(expr)
        # Simplifies assuming that shape vars > 1 (since we cache on 0/1 shape values)
        symbols = list(expr.free_symbols)
        for s in symbols:
            assert self.var_to_val[s] > 1, \
                f"{s} which is {self.var_to_val[s]} in nontrivial expression {expr}, " \
                "but as a 0/1 constant it should already have been eliminated"
        new_shape_env = {
            k: sympy.Symbol(f"shape_{idx}", positive=True, integer=True) + 1
            for idx, k in enumerate(symbols)
        }
        new_expr = expr.xreplace(new_shape_env)
        floor_div_replace = {}
        for atom in new_expr.atoms(FloorDiv):
            floor_div_replace[atom] = sympy.floor(atom.args[0] / atom.args[1])
        new_expr = sympy.expand(new_expr.xreplace(floor_div_replace))
        if len(list(new_expr.free_symbols)) == 0:
            return new_expr
        return None

    @_lru_cache
    def replace(self, expr: "sympy.Expr") -> "sympy.Expr":
        replacements = {s: self._find(cast(sympy.Symbol, s)) for s in expr.free_symbols}
        return sympy.expand(expr.xreplace(replacements))

    @_lru_cache
    def _update_divisible(self):
        new_divisible = set()
        for k in self.divisible:
            res = self.replace(k)
            if len(res.free_symbols) > 0:
                new_divisible.add(k)

        self.divisible = new_divisible

    @_lru_cache
    def simplify(self, expr: "sympy.Expr") -> "sympy.Expr":
        expr = self.replace(expr)
        if expr.has(FloorDiv):
            self._update_divisible()
            div_replacements = {}
            for atom in expr.atoms(FloorDiv):
                base, divisor = atom.args
                if self.replace(base % divisor) in self.divisible:
                    div_replacements[atom] = base / divisor
            expr = expr.xreplace(div_replacements)
            expr = sympy.expand(expr)
        return expr

    @lru_cache(256)
    def size_hint(self, expr: "sympy.Expr"):
        """
        Gets a size hint for a given expression from the underlying shapes we had.
        Does not introduce a guard, so only use this when you can guarantee that
        your code is still valid for arbitrary shapes (such as optimization decisions)
        """
        result_expr = sympy.expand(expr).xreplace(self.var_to_val)
        assert len(result_expr.free_symbols) == 0, "Size hint has variables we don't have underlying values for"
        return result_expr

    @_lru_cache
    def _find(self, a: "sympy.Symbol") -> "sympy.Expr":
        """
        Implements a DSU-like algorithm to find the variable that represents a
        Also handles transitive non-identity replacements.

        a: b + c
        c: d
        """
        if a not in self.replacements:
            return a
        res = self.replacements[a]
        cur_replace = {s: self._find(s) for s in res.free_symbols}
        self.replacements[a] = self.replacements[a].xreplace(cur_replace)
        return self.replacements[a]

    @lru_cache(256)
    def _maybe_guard_eq(self, expr: "sympy.Eq") -> None:
        """
        Evaluates the result of an eq call. If true, uses information to
        simplify shapes (i.e. a == b or a % 5 == 0)
        """
        concrete_bool = bool(self.size_hint(expr))
        if not concrete_bool:
            return
        free = list(expr.free_symbols)

        assert len(free) > 0, "The expression should not be static by this point"
        # In case of really gnarly expression, we don't blow up
        if len(free) > 5:
            return
        free = sorted(free, key=lambda x: (self.size_hint(x), x.name), reverse=True)  # type: ignore[attr-defined]
        lhs = expr.lhs
        rhs = expr.rhs
        try:
            solutions = sympy.solve(lhs - rhs, free[0], dict=True)
            if len(solutions) != 1:
                return
            solution = solutions[0][free[0]]
            if all(t.is_integer for t in sympy.preorder_traversal(solution)):
                new_var = self._find(solution)
                self.replacements[cast(sympy.Symbol, free[0])] = new_var
        except NotImplementedError:
            if expr.has(sympy.Mod):
                mod_expr = tuple(expr.atoms(sympy.Mod))[0]
                try:
                    solutions = sympy.solve(lhs - rhs, mod_expr, dict=True)
                    if len(solutions) == 1 and solutions[0][mod_expr] == 0:
                        self.divisible.add(mod_expr)
                except NotImplementedError:
                    pass
            return

    def _add_guard(self, expr, tb):
        if not self._suppress_guards_tls():
            assert expr is not sympy.false
            self.guards.append((expr, tb))

    @lru_cache(256)
    def evaluate_expr(self, expr: "sympy.Expr"):
        """
        Given an expression, evaluates it, adding guards if necessary
        """
        if len(expr.free_symbols) == 0:
            return expr
        expr = self.simplify(expr)
        static_expr = self._maybe_evaluate_static(expr)
        if static_expr is not None:
            return static_expr

        if isinstance(expr, sympy.Eq):
            self._maybe_guard_eq(expr)
        concrete_val = self.size_hint(expr)

        # TODO: optimize this; avoid formatting traces until we need them
        # NB: drop two frames; evaluate_expr and the Sym* function that
        # actually called us
        if not self._suppress_guards_tls():
            stack = ''.join(traceback.format_list(traceback.extract_stack()[:-2]))
            if concrete_val is sympy.true:
                self.guards.append((expr, stack))
            elif concrete_val is sympy.false:
                self.guards.append((sympy.Not(expr), stack))
            else:
                self.guards.append((sympy.Eq(expr, concrete_val), stack))
        return concrete_val
