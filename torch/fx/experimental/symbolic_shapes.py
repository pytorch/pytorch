import torch

try:
    import sympy  # type: ignore[import]
    HAS_SYMPY = True
except ImportError:
    HAS_SYMPY = False

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
        if not HAS_SYMPY:
            raise RuntimeError("Need sympy installed to create symbolic shapes")
        if val == 0 or val == 1:
            return val
        sympy_expr = sympy.Symbol(name, positive=True)
        py_sym_int = PySymInt(sympy_expr, self)
        cpp_sym_int = torch._C.SymbolicIntNode.new_symint(py_sym_int)  # type: ignore[attr-defined]
        self.shape_env[sympy_expr] = val
        return cpp_sym_int

    def try_constantify(self, expr):
        # Simplifies assuming that shape vars > 1 (since we cache on 0/1 shape values)
        new_shape_env = {k: sympy.Symbol(f'shape_{idx}', positive=True) + 1 for idx, k in enumerate(self.shape_env.keys())}
        new_expr = expr.subs(new_shape_env)
        new_expr = new_expr.simplify()
        if len(list(new_expr.free_symbols)) == 0:
            return new_expr
        return None

    def evaluate_expr(self, expr):
        const_expr = self.try_constantify(expr)
        if const_expr is not None:
            return const_expr

        expr = expr.simplify()
        concrete_val = expr.subs(self.shape_env)
        self.guards.append((expr, concrete_val))
        return concrete_val
