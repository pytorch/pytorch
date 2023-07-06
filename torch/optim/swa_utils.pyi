from typing import Any, Callable, Iterable, Union

from torch import device, Tensor
from torch.nn.modules import Module
from .lr_scheduler import _LRScheduler
from .optimizer import Optimizer

class AveragedModel(Module):
    def __init__(
        self,
        model: Module,
        device: Union[int, device] = ...,
        avg_fn: Callable[[Tensor, Tensor, int], Tensor] = ...,
    ) -> None: ...
    def update_parameters(self, model: Module) -> None: ...

def update_bn(
    loader: Iterable[Any],
    model: Module,
    device: Union[int, device] = ...,
) -> None: ...

class SWALR(_LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        swa_lr: float,
        anneal_epochs: int,
        anneal_strategy: str,
        last_epoch: int = ...,
    ) -> None: ...
