import types
from torch._ops import ops

class _Functional(types.ModuleType):
    def __init__(self):
        super(_Functional, self).__init__('torch.nn.quantized.functional')

    def __getattr__(self, op_name):
        op = ops.quantized.__getattr__(op_name)
        setattr(self, op_name, op)
        return op

functional = _Functional()
