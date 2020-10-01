from ... import Tensor
from .module import Module


class PairwiseDistance(Module):
    norm: float
    eps: float
    keepdim: bool

    def __init__(self, p: float = ..., eps: float = ..., keepdim: bool = ...) -> None: ...

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor: ...  # type: ignore

    def __call__(self, x1: Tensor, x2: Tensor) -> Tensor: ...  # type: ignore


class CosineSimilarity(Module):
    dim: int
    eps: float

    def __init__(self, dim: int = ..., eps: float = ...) -> None: ...

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor: ...  # type: ignore

    def __call__(self, x1: Tensor, x2: Tensor) -> Tensor: ...  # type: ignore
