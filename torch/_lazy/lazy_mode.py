from contextlib import contextmanager
from typing import Iterator
import torch._C._lazy

@contextmanager
def lazy_mode(device=torch.device("lazy:0")) -> Iterator[None]:
    """Makes eager tensors behave lazily within this mode scope."""
    torch._C._lazy._lazy_mode_enter(device)
    try:
        yield
    finally:
        torch._C._lazy._lazy_mode_exit(device)
