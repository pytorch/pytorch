import torch
import torch.utils._pytree as pytree
from typing import Dict, Any

try:
    import sympy  # type: ignore[import]
    HAS_SYMPY = True
except ImportError:
    HAS_SYMPY = False

aten = torch.ops.aten

__all__ = ["has_symbolic_sizes_strides", "create_contiguous", "is_symbolic_op", "handle_symbolic_op", "PySymInt", "ShapeEnv"]

def has_symbolic_sizes_strides(elem):
    return any([isinstance(i, torch._C.SymIntNode) for i in elem.shape])

def create_contiguous(shape):
    strides = [1]
    for dim in reversed(shape[:-1]):
        strides.append(dim * strides[-1])
    return list(reversed(strides))


def is_symbolic_op(func):
    return func in [aten.sym_size.default, aten.dim.default, aten.is_contiguous.default, aten.stride.default]


def handle_symbolic_op(func, args, kwargs):
    assert is_symbolic_op(func)
    if func == torch.ops.aten.sym_size.default:
        return None
    if func == torch.ops.aten.dim.default:
        return len(args[0].shape)
    # TODO: hack, need to make is_contiguous calls symbolic (probably through computing on symbolic strides)
    if func == torch.ops.aten.is_contiguous.default:
        return True
    # TODO: hack, we don't currently support symbolic strides properly
    if func == torch.ops.aten.stride.default:
        return create_contiguous(args[0].shape)

# TODO: An incomplete list
# 1. Set variables to be equal when we do equality
# 2. Specialize on 0/1 when we do subtraction
class PySymInt(object):
    """
    PySymInt objects are the primary "symbolic shape" objects that flow through
    our program. They're what sit under FakeTensor, and contains our primary
    implementation of symbolic shapes.
    """
    def __init__(self, expr, shape_env):
        self.expr = expr
        self.shape_env = shape_env

    def wrap(self, num):
        return PySymInt(sympy.Integer(num), self.shape_env)

    def __str__(self):
        return f"PySymInt({self.expr})"

    # Today we error on calling int on a symbolic shape, as this is a very accessible footgun.
    # In the future we'll probably need some explicit way of allowing this
    def __int__(self):
        raise RuntimeError("Trying to extract a concrete int out of a symbolic int")

    def __bool__(self):
        return bool(self.shape_env.evaluate_expr(self.expr))

# Methods that have a `__foo__` as well as `__rfoo__`
reflectable_magic_methods = {
    'add': lambda a, b: a + b,
    'sub': lambda a, b: a - b,
    'mul': lambda a, b: a * b,
    'mod': lambda a, b: a % b,
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
    method_name = f'{method}'

    def _create_magic_impl(func):
        def magic_impl(self, other):
            if isinstance(other, PySymInt):
                other = other.expr
            return PySymInt(func(self.expr, other), self.shape_env)
        return magic_impl

    # this should be wrapped transparently into torch._C.SymIntNode
    setattr(PySymInt, method_name, _create_magic_impl(_func))

class ShapeEnv(object):
    def __init__(self):
        self.guards = []
        self.shape_env = {}

    def create_symint(self, name, val, shape_env=None):
        if not HAS_SYMPY:
            raise RuntimeError("Need sympy installed to create symbolic shapes")
        if shape_env is None:
            shape_env = self.shape_env
        # Currently we don't put 0/1 specialization in guards but perhaps we should
        if val == 0 or val == 1:
            return val
        sympy_expr = sympy.Symbol(name, positive=True)
        py_sym_int = PySymInt(sympy_expr, self)
        cpp_sym_int = torch._C.SymIntNode.new_symint(py_sym_int)  # type: ignore[attr-defined]
        shape_env[sympy_expr] = val
        return cpp_sym_int

    def try_constantify(self, expr):
        # Simplifies assuming that shape vars > 1 (since we cache on 0/1 shape values)
        new_shape_env = {k: sympy.Symbol(f'shape_{idx}', positive=True) + 1 for idx, k in enumerate(self.shape_env.keys())}
        new_expr = expr.subs(new_shape_env)
        new_expr = new_expr.simplify()
        if len(list(new_expr.free_symbols)) == 0:
            return new_expr
        return None

    def create_shapes_for_args(self, args, shape_env=None):
        # Takes pytrees and returns a flat list
        arg_cnt = 0

        def create_shape(x):
            nonlocal arg_cnt
            if not isinstance(x, torch.Tensor):
                return x

            out_shape = [self.create_symint(f"s_{arg_cnt}[{idx}]", sz, shape_env) for idx, sz in enumerate(x.shape)]
            arg_cnt += 1
            return out_shape
        return list(map(create_shape, pytree.tree_flatten(args)[0]))

    def evaluate_guards_for_args(self, *args):
        env: Dict[Any, Any] = {}
        _ = self.create_shapes_for_args(args, shape_env=env)
        return all(guard.subs(env) == value for guard, value in self.guards)


    def evaluate_expr(self, expr):
        const_expr = self.try_constantify(expr)
        if const_expr is not None:
            return const_expr

        expr = expr.simplify()
        concrete_val = expr.subs(self.shape_env)
        self.guards.append((expr, concrete_val))
        return concrete_val
