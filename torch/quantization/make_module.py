
import torch

_func_from_float = {
    torch.add: torch.ops.quantized.add
}

"""Wrapper class for functions.

Args:
    func: The function to wrap the module around
    name: Name of the module class. If None, the class name will be capitalized
          op name.
"""
class _WrapModule(torch.nn.Module):
    def __init__(self, func, name=None):
        super(_WrapModule, self).__init__()
        self.func = func
        self.__name__ = self.func.__name__.capitalize() if name is None else name

    def __repr__(self):
        return self.__name__ + '()'

    def forward(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    def from_float(self, *args, **kwargs):
        func = func_from_float[self.func]
        return _WrapModule(func, name=self.__name__ + "_quantized")

"""Factory method to create a module wrapper.

Args:
    func: The function to wrap the module around
    name: Name of the module class. If None, the class name will be capitalized op name.

Returns:
    Module class

Example::
    >>> a = torch.tensor(1, dtype=torch.float)
    >>> qa = torch.quantize_linear(a, 1.0, 127, torch.quint8)
    >>> b = torch.tensor(2, dtype=torch.float)
    >>> qb = torch.quantize_linear(b, 1.0, 127, torch.quint8)
    >>>
    >>> # Non-quantized example
    >>> mod = make_module(torch.add)
    >>> print(type(mod))
    # <class '__main__._WrapModule'>
    >>> print(mod)
    # Add()
    >>> print(mod(a, b))
    # tensor(3)
    >>>
    >>> # Quantized version
    >>> mod_quant = mod.from_float()
    >>> print(mod_quant(qa, qb, 1.0, 0))
    # tensor(3., size=(), dtype=torch.quint8, scale=1.0, zero_point=0)
"""
def make_module(op, name=None):
    return _WrapModule(op, name)
