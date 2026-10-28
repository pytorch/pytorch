from typing import NamedTuple

import torch


class Scale(NamedTuple):
    factor: float


def scaled(x: torch.Tensor, s: Scale) -> torch.Tensor:
    return x * s.factor


torch.fx.wrap("scaled")


def double(x: torch.Tensor) -> torch.Tensor:
    return scaled(x, Scale(2.0))
