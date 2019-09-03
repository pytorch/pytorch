from .. import Tensor
import builtins

class Parameter(Tensor):
    @overload
    def __init__(self, data, requires_grad: builtins.bool): ...

    ...
