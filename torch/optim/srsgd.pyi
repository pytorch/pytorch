from typing import Callable, Optional, List
from torch.cuda.amp import GradScaler
from .sgd import SGD


class SRSGD(SGD):
    def step(self, closure: Optional[Callable[[], float]]=..., grad_scaler: GradScaler=...) -> Optional[float]: ...
