import torch
from torch.amp.grad_scaler import OptState

__all__ = ["GradScaler", "OptState"]


class GradScaler(torch.amp.GradScaler):
    r"""
    See :class:`torch.amp.GradScaler`.
    ``torch.cuda.amp.GradScaler(args...)`` is equivalent to ``torch.amp.GradScaler("cuda", args...)``
    """

    def __init__(
        self,
        init_scale: float = 2.0**16,
        growth_factor: float = 2.0,
        backoff_factor: float = 0.5,
        growth_interval: int = 2000,
        enabled: bool = True,
    ) -> None:
        super().__init__(
            "cuda",
            init_scale=init_scale,
            growth_factor=growth_factor,
            backoff_factor=backoff_factor,
            growth_interval=growth_interval,
            enabled=enabled,
        )
