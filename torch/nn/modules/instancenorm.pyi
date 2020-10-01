from ... import Tensor
from .batchnorm import _BatchNorm


class _InstanceNorm(_BatchNorm):
    def __init__(self, num_features: int, eps: float = ..., momentum: float = ..., affine: bool = ...,
                 track_running_stats: bool = ...) -> None: ...

    def forward(self, input: Tensor) -> Tensor: ...  # type: ignore

    def __call__(self, input: Tensor) -> Tensor: ...  # type: ignore


class InstanceNorm1d(_InstanceNorm): ...


class InstanceNorm2d(_InstanceNorm): ...


class InstanceNorm3d(_InstanceNorm): ...
