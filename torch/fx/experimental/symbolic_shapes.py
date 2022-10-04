import torch
import torch.utils._pytree as pytree
from typing import Set, Dict, List, Type, Optional, cast
import operator
import math
import functools
from functools import lru_cache, partial
import traceback
import collections
import textwrap
from torch._subclasses.meta_utils import MetaConverter

try:
    import sympy  # type: ignore[import]
    HAS_SYMPY = True
except ImportError:
    HAS_SYMPY = False

aten = torch.ops.aten  # type: ignore[has-type]

__all__ = [
    "has_symbolic_sizes_strides", "create_contiguous", "PySymInt", "ShapeEnv",
    "SymDispatchMode", "PySymFloat", "sym_float", "FloorDiv"
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

def sym_float(a):
    if hasattr(a, '__sym_float__'):
        return a.__sym_float__()
    elif isinstance(a, torch._C.SymFloatNode):
        return a
    return float(a)

# TODO: An incomplete list
# 1. Set variables to be equal when we do equality
# 2. Specialize on 0/1 when we do subtraction
class PySymInt(object):
    """
    PySymInt objects are the primary "symbolic shape" objects that flow through
    our program. They're what sit under FakeTensor, and contains our primary
    implementation of symbolic shapes.
    """
    def __init__(self, expr, shape_env, constant=None):
        self.expr = expr
        self.shape_env = shape_env
        self.constant = constant

    def wrap(self, num):
        return PySymInt(sympy.Integer(num), self.shape_env, constant=num)

    def clone(self):
        return PySymInt(self.expr, self.shape_env, constant=self.constant)

    def __str__(self):
        return f"{self.expr}"

    def __repr__(self):
        return f"{self.expr}"

    # Today we error on calling int on a symbolic shape, as this is a very accessible footgun.
    def __int__(self):
        raise RuntimeError("Trying to extract a concrete int out of a symbolic int")

    # You can manually trigger a guard with this function
    def guard_int(self, file, line):
        # TODO: use the file/line for some useful diagnostic on why a
        # guard occurred
        return int(self.shape_env.evaluate_expr(self.expr))

    def __sym_float__(self):
        if SYM_FUNCTION_MODE:
            return _handle_sym_dispatch(sym_float, (self,), {})
        # TODO: consider constant prop here
        # TODO: wrapping the expr with sympy.Float doesn't seem to work, why
        # not?
        return PySymFloat(self.expr, self.shape_env)

    def __bool__(self):
        return bool(self.shape_env.evaluate_expr(self.shape_env.replace(self.expr)))

class PySymFloat:
    def __init__(self, expr, shape_env, constant=None):
        self.expr = expr
        self.shape_env = shape_env
        self.constant = constant

    def wrap(self, num):
        return PySymFloat(sympy.Float(num), self.shape_env, constant=num)

    def __str__(self):
        return f"{self.expr}"

if HAS_SYMPY:
    class FloorDiv(sympy.Function):
        """
        We maintain this so that:
        1. We can use divisibility guards to simplify FloorDiv(a, b) to a / b.
        2. Printing out the expression is nicer (compared to say, representing a//b as (a - a % b) / b)
        """
        nargs = (2,)

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

    class Ceil(sympy.Function):
        """
        sympy doesn't have its own ceil(), so rolling one here.
        We maintain this so that we can simplify a sympy.Rational into a sympy.Float.
        sympy.Float isn't supported.
        """
        nargs = (1,)

        @classmethod
        def eval(cls, a):
            if isinstance(a, sympy.Integer):
                return a
            elif isinstance(a, sympy.core.symbol.Symbol) and a.is_scalar:
                # TODO: do we need to simplify expr's first? (e.g. if we have 3/3), is is_scalar() true?
                return a
            elif isinstance(a, sympy.Rational):
                return a.floor() + 1
            else:
                raise NotImplementedError("math.ceil() not supported for type: " + str(type(a)))

# Methods that have a `__foo__` as well as `__rfoo__`
reflectable_magic_methods = {
    'add': lambda a, b: a + b,
    'sub': lambda a, b: a - b,
    'mul': lambda a, b: a * b,
    'mod': lambda a, b: a % b,
    'truediv': lambda a, b: a / b,
    'floordiv': lambda a, b: FloorDiv(a, b)
}

magic_methods = {
    **reflectable_magic_methods,
    'eq': lambda a, b: sympy.Eq(a, b),
    'gt': lambda a, b: sympy.Gt(a, b),
    'lt': lambda a, b: sympy.Lt(a, b),
    'le': lambda a, b: sympy.Le(a, b),
    'ge': lambda a, b: sympy.Ge(a, b),
    'ceil': lambda a: Ceil(a)
}

unary_magic_methods = {
    'ceil'
}

float_magic_methods = {"add", "sub", "mul", "truediv", "ceil"}

def _make_magic(method, func, py_type):
    func = lru_cache(256)(func)

    def magic_impl(self, other):
        if SYM_FUNCTION_MODE:
            return _handle_sym_dispatch(getattr(operator, method), (self, other), {})
        if isinstance(other, py_type):
            other = other.expr
        # TODO: consider constant prop here
        expr = self.shape_env.replace(self.expr)
        other = self.shape_env.replace(other)
        out = func(expr, other)
        out = sympy.expand(out)
        if method in ["truediv"]:
            return PySymFloat(out, self.shape_env)
        else:
            # TODO: relational operators actually technically return a
            # PySymBool, this is a type error
            return py_type(out, self.shape_env)

    def unary_magic_impl(self):
        if SYM_FUNCTION_MODE:
            if method in ["ceil"]:
                op = getattr(math, method)
            else:
                op = getattr(operator, method)
            return _handle_sym_dispatch(op, (self,), {})
        # TODO: consider constant prop here
        expr = self.shape_env.replace(self.expr)
        out = func(expr)
        out = sympy.expand(out)
        if method in ["ceil"]:
            return PySymInt(out, self.shape_env)
        else:
            return py_type(out, self.shape_env)

    # this should be wrapped transparently into torch.SymIntNode
    if method in unary_magic_methods:
        setattr(py_type, method, unary_magic_impl)
        setattr(py_type, f"__{method}__", unary_magic_impl)
    else:
        setattr(py_type, method, magic_impl)
        setattr(py_type, f"__{method}__", magic_impl)
        if method in reflectable_magic_methods:
            setattr(py_type, f"__r{method}__", magic_impl)

for method, func in magic_methods.items():
    _make_magic(method, func, PySymInt)

for method, func in magic_methods.items():
    if method not in float_magic_methods:
        continue
    _make_magic(method, func, PySymFloat)

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



class ShapeEnv(object):
    def __init__(self):
        self.guards = []
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
        self.val_to_symint: Dict[int, torch.SymIntNode] = {}

    def _get_key(self):
        """
        Defines the current "state" of the guards we've accumulated in this ShapeEnv.
        Determines when we need to invalidate our cache
        """
        return (len(self.replacements), len(self.divisible))

    # NB: This is only called for input symbolic sizes; intermediate symbolic
    # sizes are allocated via a different mechanism
    def create_symint(self, name, val):
        assert val >= 0
        if not HAS_SYMPY:
            raise RuntimeError("Need sympy installed to create symbolic shapes")

        # TODO: Put 0/1 specialization in guards
        if val == 0 or val == 1:
            return val
        # This implements duck-shaping: input sizes that match are assigned
        # the same symint
        # TODO: Create a guard whenever this happens
        # TODO: But how do I represent the guard in this case?
        if val in self.val_to_symint:
            return self.val_to_symint[val]
        sympy_expr = sympy.Symbol(name, positive=True, integer=True)
        py_sym_int = PySymInt(sympy_expr, self)
        cpp_sym_int = torch.SymIntNode.new_symint(py_sym_int)  # type: ignore[attr-defined]
        self.var_to_val[sympy_expr] = sympy.Integer(val)
        self.val_to_symint[val] = cpp_sym_int
        return cpp_sym_int

    def evaluate_guards_for_args(self, *args):
        new_env = ShapeEnv()
        # NB: This must be kept in sync with create_aot_dispatcher_function
        # and wrap_fake_symbolic
        meta_converter = MetaConverter()
        pytree.tree_map_only(torch.Tensor, partial(meta_converter, shape_env=new_env), args)
        return all(guard.xreplace(new_env.var_to_val) == value for guard, value, _ in self.guards)

    def get_nontrivial_guards(self):
        return [(self.simplify(guard), val) for guard, val, _ in self.guards if self._maybe_evaluate_static(guard) is None]

    def format_guards(self, verbose=False):
        def format_val(guard, val):
            if val is sympy.true:
                return str(guard)
            elif val is sympy.false:
                return f"Not({guard})"
            else:
                return f"Eq({guard}, {val})"

        def format_tb(tb):
            if not verbose:
                return ""
            return f"\n   Guarded at:\n{textwrap.indent(tb, '   ')}"

        return '\n'.join(f" - {format_val(guard, val)}{format_tb(tb)}" for guard, val, tb in self.guards)

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
        stack = ''.join(traceback.format_list(traceback.extract_stack()[:-2]))
        self.guards.append((expr, concrete_val, stack))
        return concrete_val
