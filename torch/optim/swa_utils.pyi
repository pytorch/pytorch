from typing import Any, Callable, Iterable, List, Tuple, Union

from torch import device, Tensor
from torch.nn.modules import Module
from .lr_scheduler import _LRScheduler
from .optimizer import Optimizer

PARAM_LIST = Tuple[Tensor, ...] | List[Tensor]

def get_ema_multi_avg_fn(decay: float) -> Callable[[PARAM_LIST, PARAM_LIST, Any], None]: ...

def get_swa_multi_avg_fn() -> Callable[[PARAM_LIST, PARAM_LIST, int], None]: ...

def get_ema_avg_fn(decay: float) -> Callable[[Tensor, Tensor, Any], Tensor]: ...

def get_swa_avg_fn() -> Callable[[Tensor, Tensor, int], Tensor]: ...

class AveragedModel(Module):
    def __init__(
        self,
        model: Module,
        device: Union[int, device] = ...,
        avg_fn: Callable[[Tensor, Tensor, int], Tensor] = ...,
        multi_avg_fn: Callable[[PARAM_LIST, PARAM_LIST, int], None] = ...,
        use_buffers: bool = ...,
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
    def get_lr(self) -> List[float]: ...
