import sympy
import torch
import torch.fx as fx
from torch.utils._pytree import tree_map
import operator
from contextlib import contextmanager
aten = torch.ops.aten


SYM_INT_CLASS = type(torch._C.SymbolicIntNode.new_symint(sympy.Integer(1)))

def check_sym_int(a):
    assert SYM_INT_CLASS == type(a)
    return a

# Brief explanation of the scheme below
sympy.init_printing()

meta_funcs = {}
def register_meta(op):
    def decorator(f):
        def add_func(op):
            meta_funcs[op] = f
        tree_map(add_func, op)
        return f
    return decorator

@register_meta([aten.add.Tensor])
def binary_func_meta(a, b, **kwargs):
    assert a.dim() == b.dim()
    for a_dim, b_dim in zip(a.shape, b.shape):
        if a_dim != 1:
            assert (a_dim == b_dim or b_dim == 1)
    return a

# Copied from prims
# todo: figure out how to unify
@register_meta(aten.cat.default)
def cat_meta(tensors, dim=0):
    concat_length = 0
    shape = tensors[0].shape
    for tensor in tensors:
        for idx, (common_length, length) in enumerate(zip(shape, tensor.shape)):
            if idx == dim:
                concat_length = concat_length + length
            else:
                assert length == common_length
    new_shape = list(shape)
    new_shape[dim] = concat_length
    return tensors[0].new_empty(new_shape)

# PLACEHOLDER: Need real faketensor to propagate through actual meta functions/tensors
#############
# Thoughts:
# Note that for any operator that's *decomposed*, we don't actually need to explicitly propagate shapes for it! The shape guards should already be implicitly asserted by the decomposition itself.
# So, we're only left with operators that we don't have decompositions for (i.e. prims). For those, we can write meta functions. But, in fact, meta-tensors can *also* be thought of as decompositions - they're just decompositions which *only* match in terms of metadata (but not actual values).
# So, FakeTensor (which is what I'm using to propagate dynamic shapes without tracing) should call into decomps when possible, and call into meta funcs otherwise.
class FakeTensor:
    def __init__(self, shape):
        self.shape = shape

    def dim(self):
        return len(self.shape)

    def sizes(self):
        return self.shape

    def new_empty(self, shape):
        return FakeTensor(shape)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if func in meta_funcs:
            return meta_funcs[func](*args, **kwargs)
        raise RuntimeError(f'Unknown function {func}')


def propagate_meta(func, *args, **kwargs):
    def get_fake(x):
        return FakeTensor(x.size()) if isinstance(x, torch.Tensor) or isinstance(x, ProxyTensor) else x

    if func in meta_funcs:
        return meta_funcs[func](*tree_map(get_fake, args), **tree_map(get_fake, kwargs)).shape
    raise RuntimeError(f'Unknown function {func}')

from torch._C import _disabled_torch_function_impl

@contextmanager
def no_dispatch():
    guard = torch._C._DisableTorchDispatch()  # type: ignore[attr-defined]
    try:
        yield
    finally:
        del guard


class ProxyTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, fake_elem, proxy, sym_shape, sym_strides, tracer):
        # TODO: this is wrong in general
        offset = 0
        r = torch.Tensor._make_sym_wrapper_subclass(
            cls, sym_shape, 
            ProxyTensor.sym_strides(tracer, sym_shape), offset,
            dtype=fake_elem.dtype, layout=fake_elem.layout, requires_grad=fake_elem.requires_grad,
            device=fake_elem.device
        )

        r.proxy = proxy
        r.tracer = tracer
        # we would use it to propagate tensor metadata
        r.fake_elem = fake_elem.detach().clone()
        return r

    __torch_function__ = _disabled_torch_function_impl

    @staticmethod
    def sym_strides(tracer, sym_shapes):
        # TODO: quadratic
        def pp(i):
            return check_sym_int(tracer.get_one()) if i == len(sym_shapes) - 1 else check_sym_int(pp(i + 1)) * check_sym_int(sym_shapes[i + 1])
        return [pp(x)  for x in range(len(sym_shapes))]

    @classmethod
    def __torch_dispatch__(cls, func_overload, types, args=(), kwargs=None):
        print(func_overload.__name__)
        try:
            args = args if args else ()
            def get_proxy(x): return x.proxy if isinstance(x, ProxyTensor) else x
            kwargs = kwargs if kwargs else {}
            tracer = []
            def get_tracer(x):
                if isinstance(x, ProxyTensor):
                    tracer.append(x.tracer)

            def get_fake(x): return x.fake_elem if isinstance(x, ProxyTensor) else x

            tree_map(get_tracer, (args, kwargs))
            assert len(tracer) > 0
            tracer = tracer[0]
            output_proxy = tracer.graph.call_function(func_overload, tree_map(get_proxy, args), tree_map(get_proxy, kwargs))
            with no_dispatch():
                # TODO: we will use a FakeTensor or MetaTensor to compute metadata 
                out = func_overload(*tree_map(get_fake, args), **tree_map(get_fake, kwargs))
                assert not isinstance(out, ProxyTensor)


            # this should be producing SymInts
            output_shape = propagate_meta(func_overload, *args, **kwargs)
            output_shape = [check_sym_int(x) for x in output_shape]
            sym_strides = ProxyTensor.sym_strides(tracer, output_shape)
            sym_strides = [check_sym_int(x) for x in sym_strides]
            return ProxyTensor(out, output_proxy, output_shape, sym_strides, tracer)
        except Exception as e:
            print(f"exception {e}")
            exit(0)

# EXTENDABLE
class PySymInt(object):
    def __init__(self, expr, tracer):
        self.expr = expr
        self.tracer = tracer

    def __getattr__(self, name):
        print(f"Missing attribute {name}")
        exit(0)

    def __bool__(self):
        return bool(self.tracer.evaluate_expr(self.expr))

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
    method_name = f'__{method}__'
    def create_magic_impl(func):
        def magic_impl(self, other):
            if isinstance(other, PySymInt):
                other = other.expr
            return PySymInt(func(self.expr, other), self.tracer)
        return magic_impl

    # this should be wrapped transparently into torch._C.SymbolicIntNode
    setattr(PySymInt, method_name, create_magic_impl(func))

# EXTENDABLE
class Tracer(object):
    def __init__(self):
        self.graph = fx.Graph()
        self.guards = []
        self.shape_env = {}

    def get_one(self, shape_env = None):
        #return self.create_symbol('CONST1', 1, shape_env)
        return torch._C.SymbolicIntNode.new_symint(PySymInt(sympy.Integer(1), self))

    def create_symbol(self, name, val, shape_env=None):
        if shape_env is None:
            shape_env = self.shape_env
        sympy_symint = sympy.symbols(name, integer=True)
        sym_int = torch._C.SymbolicIntNode.new_symint(PySymInt(sympy_symint, self))
        shape_env[sympy_symint] = val
        return sym_int

    def evaluate_expr(self, expr):
        concrete_val = expr.subs(self.shape_env)
        self.guards.append((expr, concrete_val))
        return concrete_val

    def create_args(self, *args):
        proxy_args = []
        for idx, arg in enumerate(args):
            name = chr(idx+65)
            proxy = self.graph.placeholder(name)
            sym_shapes = [self.create_symbol(f'{name}_{idx}', i) for idx, i in enumerate(arg.shape)]
            sym_shapes = [check_sym_int(x) for x in sym_shapes]
            sym_strides =[check_sym_int(x) for x in ProxyTensor.sym_strides(self, sym_shapes)]
            proxy_args.append(ProxyTensor(arg, proxy, sym_shapes, sym_strides, self))
        return proxy_args

    def evaluate_guards(self, *args):
        env = {}
        # args = [create_proxy(chr(idx + 65), i.shape) for idx, i in enumerate(args)]
        for idx, arg in enumerate(args):
            name = chr(idx+65)
            sym_shapes = [self.create_symbol(f'{name}_{idx}', i, env) for idx, i in enumerate(arg.shape)]
        return all(guard.subs(env) == value for guard, value in self.guards)



def dynamic_trace(f, args):
    tracer = Tracer()
    args = tracer.create_args(*args)
    f(*args)
    return tracer

## Start reading from here!
# Basically, we create symbolic variables for the shapes of each of our variables
# Then, we simply trace like we would normally. However, our proxies now also contain "symbolic shapes" on them.
# So, when we perform any operation, the symbolic shape must be propagated to the output tensor.
# At any point, we have an expression for the shape of any tensor in terms of the input tensors
# When we come to control flow (including asserts), we do 2 things:
# 1. Evaluate the control flow expression to a boolean. Remember that any tensor's shape can be evaluated in terms of the input tensors!
# 2. Store a guard for our cache. This guard allows us to check if the trace is valid at the input!

# __new__(cls, fake_elem, proxy, sym_shape, sym_strides, tracer):
#pt = ProxyTensor(torch.rand(6), None, )

def f(a, b):
    return a + b
tracer = dynamic_trace(f, [torch.rand(4), torch.rand(4)])
print(tracer.graph)

exit(0)

def f(a, b):
    c = torch.add(a, b)
    if c.shape[0] > 3:
        return torch.add(c, c)
    else:
        return torch.sub(c, c)

tracer = dynamic_trace(f, [torch.rand(4), torch.rand(4)])
print(tracer.graph)
print(tracer.guards)
assert tracer.evaluate_guards(torch.rand(6), torch.rand(6))
assert not tracer.evaluate_guards(torch.rand(3), torch.rand(3))
