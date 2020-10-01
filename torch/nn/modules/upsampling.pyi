from ... import Tensor
from .module import Module
from typing import Optional
from ..common_types import _size_2_t, _ratio_2_t, _size_any_t, _ratio_any_t


class Upsample(Module):
    name: str = ...
    size: _size_any_t = ...
    scale_factor: _ratio_any_t = ...
    mode: str = ...
    align_corners: bool = ...

    def __init__(self, size: Optional[_size_any_t] = ..., scale_factor: Optional[_ratio_any_t] = ..., mode: str = ...,
                 align_corners: Optional[bool] = ...) -> None: ...

    def forward(self, input: Tensor) -> Tensor: ...  # type: ignore

    def __call__(self, input: Tensor) -> Tensor: ...  # type: ignore


class UpsamplingNearest2d(Upsample):
    def __init__(self, size: Optional[_size_2_t] = ..., scale_factor: Optional[_ratio_2_t] = ...) -> None: ...


class UpsamplingBilinear2d(Upsample):
    def __init__(self, size: Optional[_size_2_t] = ..., scale_factor: Optional[_ratio_2_t] = ...) -> None: ...
