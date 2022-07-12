import sympy
import torch

def create_contiguous(shape):
    if len(shape) == 0:
        return []

    strides = [1]
    for dim in reversed(shape[:-1]):
        strides.append(dim * strides[-1])
    return list(reversed(strides))

class PySymInt(object):
    def __init__(self, expr, shape_env):
        self.expr = expr
        self.shape_env = shape_env

    def wrap(self, num):
        return PySymInt(sympy.Integer(num), self.shape_env)

    def __str__(self):
        return f"PySymInt({self.expr})"

    def __int__(self):
        raise RuntimeError("Trying to extract a concrete int out of a symbolic int")

        return self.shape_env.evaluate_expr(self.expr)

    def __bool__(self):
        return bool(self.shape_env.evaluate_expr(self.expr))


magic_methods = {
    'add': lambda a, b: a + b,
    'radd': lambda a, b: a + b,
    'sub': lambda a, b: a - b,
    'mul': lambda a, b: a * b,
    'div': lambda a, b: a / b,
    'mod': lambda a, b: a % b,
    'eq': lambda a, b: sympy.Eq(a, b),
    'gt': lambda a, b: sympy.Gt(a, b),
    'lt': lambda a, b: sympy.Lt(a, b),
}

for method, func in magic_methods.items():
    method_name = f'{method}'

    def create_magic_impl(func):
        def magic_impl(self, other):
            if isinstance(other, PySymInt):
                other = other.expr
            return PySymInt(func(self.expr, other), self.shape_env)
        return magic_impl

    # this should be wrapped transparently into torch._C.SymbolicIntNode
    setattr(PySymInt, method_name, create_magic_impl(func))

class ShapeEnv(object):
    def __init__(self):
        self.guards = []
        self.shape_env = {}

    def create_symint(self, name, val):
        if val == 0 or val == 1:
            return val
        sympy_expr = sympy.Symbol(name, positive=True)
        py_sym_int = PySymInt(sympy_expr, self)
        cpp_sym_int = torch._C.SymbolicIntNode.new_symint(py_sym_int)
        self.shape_env[sympy_expr] = val
        return cpp_sym_int

    def try_constantify(self, expr):
        # Simplifies assuming that shape vars > 0
        new_shape_env = {k: sympy.Symbol(f'shape_{idx}', positive=True) + 1 for idx, k in enumerate(self.shape_env.keys())}
        new_expr = expr.subs(new_shape_env)
        new_expr = new_expr.simplify()
        if len(list(new_expr.free_symbols)) == 0:
            return new_expr
        return None

    def evaluate_expr(self, expr):
        if (const_expr := self.try_constantify(expr)) is not None:
            return const_expr

        expr = expr.simplify()
        concrete_val = expr.subs(self.shape_env)
        self.guards.append((expr, concrete_val))
        return concrete_val