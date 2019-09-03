from .. import Tensor
from typing import overload
import builtins

class Parameter(Tensor):
    @overload
    def __init__(self, data, requires_grad: builtins.bool): ...

    ...
