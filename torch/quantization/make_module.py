import torch

_operation_from_float = {
    torch.add: torch.ops.quantized.add,
    torch.cat: torch.ops.quantized.cat
}

"""Factory method to create a module wrapper.

Args:
    operation: The operation to wrap the module around

Returns:
    Module class
"""
def make_module(op=None, quantized=False):
    name = op.__name__.capitalize()
    name = "_ops_" + name

    if quantized:
        name = name = name + "_quantized"
        op = _operation_from_float.get(op, op)

    def _init(self, operation):
        super(type(self), self).__init__()
        self.operation = operation
        setattr(torch, name, type(self))  # Add the type to the torch namespace

    def _forward(self, *args, **kwargs):
        return self.operation(*args, **kwargs)

    def _from_float(self, mod, *args, **kwargs):
        return make_module(mod.op, name=name, quantized=True)

    _methods = {
        "__init__": _init,
        "forward": _forward
    }
    if quantized:
        _methods["from_float"] = _from_float

    module = type(name, (torch.nn.Module,), _methods)
    return module(op)
