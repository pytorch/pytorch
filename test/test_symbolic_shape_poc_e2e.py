from torch._C import _disabled_torch_function_impl
from torch.testing._internal.common_utils import run_tests, TestCase
import unittest
import torch
from torch.utils._pytree import tree_map
import torch._decomp
from torch._meta_registrations import register_meta, meta_funcs
aten = torch.ops.aten

try:
    import sympy
    HAS_SYMPY = True
except ImportError:
    HAS_SYMPY = False
skipIfNoSympy = unittest.skipIf(not HAS_SYMPY, "no sympy")


@register_meta([aten.narrow_copy.SymInt])
def narrow_copy_symint_meta(a, dim, start, length, **kwargs):
    shape = []
    for i, x in enumerate(a.size()):
        if i == dim:
            shape.append(length)
        else:
            shape.append(x)
    return a.new_empty(tuple(shape))


@register_meta([aten.expand.SymInt])
def expand_symint_meta(a, size, implicit=False):
    return a.new_empty(size)


from torch._subclasses import FakeTensor
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import make_fx

def f(x, y):
    val = torch.mul(x, y)
    out = torch.cat([val, val])
    if out.shape[0] * out.shape[1] > 20:
        out = out.cos()
    return out.expand(out.shape)

fx_g = make_fx(f)(torch.randn(5, 1), torch.randn(1, 5))
fx_g.graph.eliminate_dead_code()
fx_g.recompile()
print(fx_g.code)
print(fx_g.shape_env.guards)
exit(0)

foo = torch.empty(shape_env.create_symint("foo", 3), device='meta')
fake_tensor_mode = FakeTensorMode()
test = FakeTensor(fake_tensor_mode, foo, 'cuda')
with fake_tensor_mode:
    print(torch.ops.aten.expand.SymInt(test, [test.shape[0], test.shape[0]]))
    # print(torch.empty(test.shape, device='meta'))
    # print(torch.cat([test, test]).shape)
print(shape_env.guards)