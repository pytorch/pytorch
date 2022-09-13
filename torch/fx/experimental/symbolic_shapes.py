import torch
import torch.utils._pytree as pytree
from typing import Dict, Any, List, Type
import operator
import functools
from functools import lru_cache

try:
    import sympy  # type: ignore[import]
    HAS_SYMPY = True
except ImportError:
    HAS_SYMPY = False

aten = torch.ops.aten

__all__ = [
    "has_symbolic_sizes_strides", "create_contiguous", "is_symbolic_op", "handle_symbolic_op", "PySymInt", "ShapeEnv",
    "SymDispatchMode", "PySymFloat", "sym_float"
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
    return (
        any([isinstance(i, torch.SymIntNode) for i in elem.shape])
        or any([isinstance(i, torch.SymIntNode) for i in elem.stride()])
        or isinstance(elem.numel(), torch.SymIntNode)
    )

def create_contiguous(shape):
    strides = [1]
    for dim in reversed(shape[:-1]):
        strides.append(dim * strides[-1])
    return list(reversed(strides))

def is_symbolic_op(func):
    return func in [aten.sym_size.default, aten.dim.default,
                    aten.is_contiguous.default, aten.sym_stride.default, aten.sym_numel.default,
                    ]

def handle_symbolic_op(func, args, kwargs):
    assert is_symbolic_op(func)
    if func == torch.ops.aten.sym_size.default:
        return None
    if func == torch.ops.aten.sym_stride.default:
        return None
    if func == torch.ops.aten.dim.default:
        return len(args[0].shape)
    if func == torch.ops.aten.sym_numel.default:
        res = 1
        for s in args[0].shape:
            res = res * s
        return res
    # TODO: hack, need to make is_contiguous calls symbolic (probably through computing on symbolic strides)
    if func == torch.ops.aten.is_contiguous.default:
        return True

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

    def __str__(self):
        return f"{self.expr}"

    def __repr__(self):
        return f"{self.expr}"

    # Today we error on calling int on a symbolic shape, as this is a very accessible footgun.
    # In the future we'll probably need some explicit way of allowing this
    def __int__(self):
        return int(self.shape_env.evaluate_expr(self.expr))
        # raise RuntimeError("Trying to extract a concrete int out of a symbolic int")

    def __sym_float__(self):
        if SYM_FUNCTION_MODE:
            return _handle_sym_dispatch(sym_float, (self,), {})
        # TODO: consider constant prop here
        # TODO: wrapping the expr with sympy.Float doesn't seem to work, why
        # not?
        return PySymFloat(self.expr, self.shape_env)

    def __bool__(self):
        return bool(self.shape_env.evaluate_expr(self.expr))

class PySymFloat:
    def __init__(self, expr, shape_env, constant=None):
        self.expr = expr
        self.shape_env = shape_env
        self.constant = constant

    def wrap(self, num):
        return PySymFloat(sympy.Float(num), self.shape_env, constant=num)

    def __str__(self):
        return f"{self.expr}"

# Methods that have a `__foo__` as well as `__rfoo__`
reflectable_magic_methods = {
    'add': lambda a, b: a + b,
    'sub': lambda a, b: a - b,
    'mul': lambda a, b: a * b,
    'mod': lambda a, b: a % b,
    'floordiv': lambda a, b: (a - (a % b)) / b
}

magic_methods = {
    **reflectable_magic_methods,
    'eq': lambda a, b: sympy.Eq(a, b),
    'gt': lambda a, b: sympy.Gt(a, b),
    'lt': lambda a, b: sympy.Lt(a, b),
    'le': lambda a, b: sympy.Le(a, b),
    'ge': lambda a, b: sympy.Ge(a, b),
}

for method, _func in magic_methods.items():
    def _create_magic_impl(func):
        method_name = method

        def magic_impl(self, other):
            if SYM_FUNCTION_MODE:
                return _handle_sym_dispatch(getattr(operator, method_name), (self, other), {})
            if isinstance(other, PySymInt):
                other = other.expr
            # TODO: consider constant prop here
            return PySymInt(func(self.expr, other), self.shape_env)
        return magic_impl

    # this should be wrapped transparently into torch._C.SymIntNode
    setattr(PySymInt, method, _create_magic_impl(_func))
    setattr(PySymInt, f"__{method}__", _create_magic_impl(_func))
    if method in reflectable_magic_methods:
        setattr(PySymInt, f"__r{method}__", _create_magic_impl(_func))

def _lru_cache(fn, maxsize=None):
    fn_cache = lru_cache(maxsize)(fn)
    prior_key = None

    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        nonlocal prior_key
        if prior_key != self._get_key():
            prior_key = self._get_key()
            fn_cache.cache_clear()
        return fn_cache(self, *args, **kwargs)

    return wrapper


class ShapeEnv(object):
    def __init__(self):
        self.guards = []
        self.var_to_val = {}
        self.replacements = {}
        self.divisible = {}

    def _get_key(self):
        """
        Defines the current "state" of the guards we've accumulated in this ShapeEnv.
        Determines when we need to invalidate our cache
        """
        return (len(self.replacements), len(self.divisible))

    def create_symint(self, name, val):
        if not HAS_SYMPY:
            raise RuntimeError("Need sympy installed to create symbolic shapes")

        # Currently we don't put 0/1 specialization in guards but perhaps we should
        if val == 0 or val == 1:
            return val
        sympy_expr = sympy.Symbol(name, positive=True, integer=True)
        py_sym_int = PySymInt(sympy_expr, self)
        cpp_sym_int = torch._C.SymIntNode.new_symint(py_sym_int)  # type: ignore[attr-defined]
        self.var_to_val[sympy_expr] = val
        return cpp_sym_int

    def create_shapes_for_args(self, args):
        arg_cnt = 0

        def create_shape(x):
            nonlocal arg_cnt
            if not isinstance(x, torch.Tensor):
                return x

            out_shape = [self.create_symint(f"s_{arg_cnt}[{idx}]", sz) for idx, sz in enumerate(x.shape)]
            arg_cnt += 1
            return out_shape
        return list(map(create_shape, pytree.tree_flatten(args)[0]))

    def evaluate_guards_for_args(self, *args):
        new_env = ShapeEnv()
        _ = new_env.create_shapes_for_args(args)
        return all(guard.xreplace(new_env.var_to_val) == value for guard, value in self.guards)

    @_lru_cache
    def maybe_evaluate_static(self, expr):
        """
        Tries to evaluate expr without introducing guards
        """
        # Simplifies assuming that shape vars > 1 (since we cache on 0/1 shape values)
        symbols = list(expr.free_symbols)
        new_shape_env = {
            k: sympy.Symbol(f"shape_{idx}", positive=True, integer=True) + 1
            for idx, k in enumerate(symbols)
        }
        new_expr = expr.xreplace(new_shape_env)
        new_expr = sympy.expand(new_expr)
        if len(list(new_expr.free_symbols)) == 0:
            return new_expr
        return None

    def replace(self, expr):
        replacements = {s: self.find(s) for s in expr.free_symbols}
        return sympy.expand(expr.xreplace(replacements))

    def update_divisible(self):
        new_divisible = {}
        for k in self.divisible:
            res = self.replace(k)
            if len(res.free_symbols) > 0:
                new_divisible[k] = 0

        self.divisible = new_divisible

    @_lru_cache
    def simplify(self, expr):
        expr = self.replace(expr)
        if len(expr.atoms(sympy.Mod)) > 0:
            self.update_divisible()
            expr = expr.xreplace(self.divisible)
            expr = sympy.expand(expr)
        return expr

    @lru_cache
    def size_hint(self, expr: sympy.Expr):
        result_expr = sympy.expand(expr).xreplace(self.var_to_val)
        assert (not isinstance(result_expr, sympy.Expr)) or len(result_expr.free_symbols) == 0, "Size hint has variables we don't have underlying values for"
        return result_expr

    def find(self, a):
        acopy = a
        while a in self.replacements:
            a = self.replacements[a]
        while acopy in self.replacements:
            self.replacements[acopy], acopy = a, self.replacements[acopy]
        return a

    @_lru_cache
    def evaluate_eq(self, expr: sympy.Eq):
        concrete_val = self.size_hint(expr)
        self.guards.append((expr, concrete_val))
        if not concrete_val:
            return concrete_val
        free = list(expr.free_symbols)

        assert len(free) > 0, "The expression should not be static by this point"
        if len(free) in (1, 2, 3):
            free = sorted(free, key=lambda x: (-self.size_hint(x), x.name))
            try:
                lhs = expr.lhs
                rhs = expr.rhs
                solutions = sympy.solve(lhs - rhs, free[0])
                if len(solutions) == 1 and solutions[0] and "/" not in str(solutions[0]):
                    new_var = solutions[0]
                    new_var = self.find(solutions[0])
                    self.replacements[free[0]] = new_var
            except NotImplementedError as e:
                if len(expr.atoms(sympy.Mod)) == 1:
                    mod_expr = tuple(expr.atoms(sympy.Mod))[0]
                    try:
                        solutions = sympy.solve(lhs - rhs, mod_expr)
                        if len(solutions) == 1 and solutions[0] == 0:
                            self.divisible[mod_expr] = 0
                    except NotImplementedError as e:
                        pass
                pass

        return concrete_val


    def evaluate_expr(self, expr):
        try:
            if len(list(expr.free_symbols)) == 0:
                return expr
            expr = self.simplify(expr)

            static_expr = self.maybe_evaluate_static(expr)
            if static_expr is not None:
                return static_expr

            if isinstance(expr, sympy.Eq):
                return self.evaluate_eq(expr)
            concrete_val = self.size_hint(expr)

            # Uncomment this to see what code triggered this guard.
            # TODO: Save this to the guard representation so you can look
            # at it later
            # import traceback
            # traceback.print_stack()

            self.guards.append((expr, concrete_val))
            return concrete_val
        except Exception as e:
            print(e)
            breakpoint()
            print()
            raise e