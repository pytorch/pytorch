from typing import Callable, Optional, List
from torch.cuda.amp import GradScaler
from .adam import Adam


class RSAdam(Adam):
    def step(self, closure: Optional[Callable[[], float]]=..., grad_scaler: GradScaler=...) -> Optional[float]: ...
