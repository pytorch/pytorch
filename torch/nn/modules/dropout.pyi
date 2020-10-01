from ... import Tensor
from .module import Module


class _DropoutNd(Module):
    p: float
    inplace: bool

    def __init__(self, p: float = ..., inplace: bool = ...) -> None: ...

    def extra_repr(self): ...


class Dropout(_DropoutNd):
    def forward(self, input: Tensor) -> Tensor: ...  # type: ignore
    def __call__(self, input: Tensor) -> Tensor: ...  # type: ignore


class Dropout2d(_DropoutNd):
    def forward(self, input: Tensor) -> Tensor: ...  # type: ignore
    def __call__(self, input: Tensor) -> Tensor: ...  # type: ignore


class Dropout3d(_DropoutNd):
    def forward(self, input: Tensor) -> Tensor: ...  # type: ignore
    def __call__(self, input: Tensor) -> Tensor: ...  # type: ignore


class AlphaDropout(_DropoutNd):
    def forward(self, input: Tensor) -> Tensor: ...  # type: ignore
    def __call__(self, input: Tensor) -> Tensor: ...  # type: ignore


class FeatureAlphaDropout(_DropoutNd):
    def forward(self, input: Tensor) -> Tensor: ...  # type: ignore
    def __call__(self, input: Tensor) -> Tensor: ...  # type: ignore
