from .module import Module
from ... import Tensor


class PixelShuffle(Module):
    upscale_factor: int = ...

    def __init__(self, upscale_factor: int) -> None: ...

    def forward(self, input: Tensor) -> Tensor: ...  # type: ignore

    def __call__(self, input: Tensor) -> Tensor: ...  # type: ignore
