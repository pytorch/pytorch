from .optimizer import Optimizer
from ..nn.modules import Module
from .lr_scheduler import _LRScheduler
from .. import device, Tensor
from typing import Iterable, Any, Optional, Callable, Union, List

class AveragedModel(Module):
    def __init__(self, model: Module, device: Union[int, device]=..., 
                 avg_fun: Callable[[Tensor, Tensor, int], Tensor]=...) -> None:...

    def update_parameters(self, model: Module) -> None:...

def update_bn(loader: Iterable, model: Module, device: Union[int, device]=...) -> None:...

class SWALR(_LRScheduler):
    def __init__(self, optimizer: Optimizer, swa_lr: float, anneal_epochs: int, 
                 anneal_strategy: str, last_epoch: int=...) -> None:...
