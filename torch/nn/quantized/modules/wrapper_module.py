
import torch
from torch._ops import ops
from torch.nn import Module

# Forward declaration. Module mappings are added with every class below
_FLOAT_MODULES = {}

r"""Instantiates a wrapper for the operation.

Args:
    operation: One of the supported operations. See note below.
    quantized: Flag indicating that quantized wrapper is expected.

Returns:
    Instantiated module.
"""
def make_wrapper(operation, quantized=False):
    wrappers = {
        torch.add: (Add, QuantizedAdd),
        torch.cat: (Cat, QuantizedCat),
    }
    if quantized:
        return wrappers[operation][1]()
    else:
        return wrappers[operation][0]()


class _BaseQuantizedWrapper(Module):
    def __init__(self):
        super(_BaseQuantizedWrapper, self).__init__()
        self.register_buffer('scale', torch.tensor([1.0], dtype=torch.double))
        self.register_buffer('zero_point', torch.tensor([0], dtype=torch.long))

    @classmethod
    def from_float(cls, mod, qconfig=None):
        assert(_FLOAT_MODULES.get(type(mod), None) is not None), \
            "I don't know what to do with " + mod.__class___.__name__
        assert (hasattr(mod, 'observer')), \
            "Input float module must have observer attached"
        assert (cls == _FLOAT_MODULES.get(type(mod))), \
            str(cls) + ".from_float only works for its non-quantized module " \
            + "got " + str(type(mod)) + " instead."

        scale, zero_point = mod.observer.calculate_qparams()[:2]
        new_mod = cls()
        new_mod.scale = torch.tensor(scale, dtype=torch.double)
        new_mod.zero_point = torch.tensor(zero_point, dtype=torch.long)
        return new_mod


class Add(Module):
    def __init__(self):
        super(Add, self).__init__()

    def forward(self, x, y):
        # type: (Tensor, Tensor) -> Tensor
        return torch.add(x, y)


class Cat(Module):
    def __init__(self):
        super(Cat, self).__init__()

    def forward(self, x, axis=None):
        # type: (List[Tensor], Optional[int]) -> Tensor
        if axis is None:
            axis = 0
        return torch.cat(x, axis)


class QuantizedAdd(_BaseQuantizedWrapper):
    def __init__(self):
        super(QuantizedAdd, self).__init__()

    def forward(self, x, y):
        # type (Tensor, Tensor) -> Tensor
        scale = float(self.scale)
        zero_point = int(self.zero_point)
        return ops.quantized.add(x, y, scale=scale, zero_point=zero_point)

_FLOAT_MODULES[Add] = QuantizedAdd


class QuantizedCat(_BaseQuantizedWrapper):
    def __init__(self):
        super(QuantizedCat, self).__init__()

    def forward(self, x, axis=None):
        # type: (List[Tensor], Optional[int]) -> Tensor
        if axis is None:
            axis = 0
        scale = float(self.scale.item())
        zero_point = int(self.zero_point.item())
        return ops.quantized.cat(x, axis, scale=scale, zero_point=zero_point)

_FLOAT_MODULES[Cat] = QuantizedCat
