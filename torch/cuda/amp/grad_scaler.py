from typing_extensions import deprecated

import torch

# We need to keep this unused import for BC reasons
from torch.amp.grad_scaler import OptState  # noqa: F401


__all__ = ["GradScaler"]


class GradScaler(torch.amp.GradScaler):
    r"""
    See :class:`torch.amp.GradScaler`.
    ``torch.cuda.amp.GradScaler(args...)`` is deprecated. Please use ``torch.amp.GradScaler("cuda", args...)`` instead.
    """

    @deprecated(
        "`torch.cuda.amp.GradScaler(args...)` is deprecated. "
        "Please use `torch.amp.GradScaler('cuda', args...)` instead.",
        category=FutureWarning,
    )
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
