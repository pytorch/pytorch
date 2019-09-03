from .. import Tensor
import builtins

class Parameter(Tensor):
    def __init__(self, data: Tensor, requires_grad: builtins.bool): ...

    ...
