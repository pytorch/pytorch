import torch
from torch._six import with_metaclass


class VariableMeta(type):
    def __instancecheck__(cls, other):
        return isinstance(other, torch.Tensor)


# mypy doesn't understand torch._six.with_metaclass
class Variable(with_metaclass(VariableMeta, torch._C._LegacyVariableBase)):  # type: ignore[misc]
    pass

def register_py_tensor_class_for_device(device, clazz):
    if not isinstance(clazz, type):
        raise RuntimeError("clazz isn't a typeinfo object")
    # TODO maybe we should have a check for CPU, CUDA here
    # rather than buried deep in `RegisterPythonTensorClass`
    torch._C._autograd._register_py_class_for_device

from torch._C import _ImperativeEngine as ImperativeEngine
Variable._execution_engine = ImperativeEngine()
