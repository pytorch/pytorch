import torch
from torch.overrides import (
    handle_torch_function,
    has_torch_function,
    get_overridable_functions,
    get_testing_overrides,
    is_tensor_method_or_property,
    TorchFunctionMode
)
import copy

class ModeTensor(torch.Tensor):
    def __new__(cls, elem, mode):
        r = torch.Tensor._make_subclass(cls, elem, elem.requires_grad)
        r.elem = elem
        r.mode = mode
        return r

    def __torch_function__(self, func, types, args=(), kwargs=None):
        with self.mode:
            return func(*args, **kwargs)

class Mode(TorchFunctionMode):
    def __torch_function__(self, func, types, args=(), kwargs=None):
        def unwrap(e):
            if isinstance(e, ModeTensor):
                return e.elem
            else:
                return e

        def wrap(t):
            if isinstance(t, torch.Tensor):
                return ModeTensor(t, self)
            else:
                return t

        kwargs = kwargs if kwargs else {}
        return wrap(func(*tuple(unwrap(a) for a in args), **kwargs))

x = torch.rand([4])
mod = torch.nn.Linear(10, 10)
with Mode():
    mod2 = copy.deepcopy(mod)

print(mod2)
print(mod2.weight)
# print(type(out_func), out_inplace)
# <class '__main__.ModeTensor'> tensor([1.2925, 1.5918, 1.2033, 1.9141])
