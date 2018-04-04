import torch
from torch._six import with_metaclass


class VariableMeta(type):
    def __instancecheck__(self, other):
        return isinstance(other, torch.Tensor)


class Variable(with_metaclass(VariableMeta, torch._C._LegacyVariableBase)):
    pass


from torch._C import _ImperativeEngine as ImperativeEngine
Variable._execution_engine = ImperativeEngine()
