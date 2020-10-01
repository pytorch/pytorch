from typing import Any, Optional
from .module import Module
from ... import Tensor


# The deprecated `size_average` and `reduce` arguments are not included in the stubs
class _Loss(Module):
    reduction: str = ...

    def __init__(self, reduction: str = ...) -> None: ...


class _WeightedLoss(_Loss):
    def __init__(self, weight: Optional[Any] = ..., reduction: str = ...) -> None: ...


class L1Loss(_Loss):
    def __init__(self, reduction: str = ...) -> None: ...

    def forward(self, input: Tensor, target: Tensor) -> Tensor: ...  # type: ignore

    def __call__(self, input: Tensor, target: Tensor) -> Tensor: ...  # type: ignore


class NLLLoss(_WeightedLoss):
    ignore_index: int = ...

    def __init__(self, weight: Optional[Any] = ..., ignore_index: int = ..., reduction: str = ...) -> None: ...

    def forward(self, input: Tensor, target: Tensor) -> Tensor: ...  # type: ignore

    def __call__(self, input: Tensor, target: Tensor) -> Tensor: ...  # type: ignore


class NLLLoss2d(NLLLoss):
    def __init__(self, weight: Optional[Any] = ..., ignore_index: int = ..., reduction: str = ...) -> None: ...


class PoissonNLLLoss(_Loss):
    log_input: bool = ...
    full: bool = ...
    eps: float = ...

    def __init__(self, log_input: bool = ..., full: bool = ..., eps: float = ..., reduction: str = ...) -> None: ...

    def forward(self, log_input: Tensor, target: Tensor) -> Tensor: ...  # type: ignore

    def __call__(self, log_input: Tensor, target: Tensor) -> Tensor: ...  # type: ignore


class KLDivLoss(_Loss):
    def __init__(self, reduction: str = ...) -> None: ...

    def forward(self, input: Tensor, target: Tensor) -> Tensor: ...  # type: ignore

    def __call__(self, input: Tensor, target: Tensor) -> Tensor: ...  # type: ignore


class MSELoss(_Loss):
    def __init__(self, reduction: str = ...) -> None: ...

    def forward(self, input: Tensor, target: Tensor) -> Tensor: ...  # type: ignore

    def __call__(self, input: Tensor, target: Tensor) -> Tensor: ...  # type: ignore


class BCELoss(_WeightedLoss):
    def __init__(self, weight: Optional[Any] = ..., reduction: str = ...) -> None: ...

    def forward(self, input: Tensor, target: Tensor) -> Tensor: ...  # type: ignore

    def __call__(self, input: Tensor, target: Tensor) -> Tensor: ...  # type: ignore


class BCEWithLogitsLoss(_Loss):
    def __init__(self, weight: Optional[Any] = ..., reduction: str = ..., pos_weight: Optional[Any] = ...) -> None: ...

    def forward(self, input: Tensor, target: Tensor) -> Tensor: ...  # type: ignore

    def __call__(self, input: Tensor, target: Tensor) -> Tensor: ...  # type: ignore


class HingeEmbeddingLoss(_Loss):
    margin: Any = ...

    def __init__(self, margin: float = ..., reduction: str = ...) -> None: ...

    def forward(self, input: Tensor, target: Tensor) -> Tensor: ...  # type: ignore

    def __call__(self, input: Tensor, target: Tensor) -> Tensor: ...  # type: ignore


class MultiLabelMarginLoss(_Loss):
    def __init__(self, reduction: str = ...) -> None: ...

    def forward(self, input: Tensor, target: Tensor) -> Tensor: ...  # type: ignore

    def __call__(self, input: Tensor, target: Tensor) -> Tensor: ...  # type: ignore


class SmoothL1Loss(_Loss):
    def __init__(self, reduction: str = ...) -> None: ...

    def forward(self, input: Tensor, target: Tensor) -> Tensor: ...  # type: ignore

    def __call__(self, input: Tensor, target: Tensor) -> Tensor: ...  # type: ignore


class SoftMarginLoss(_Loss):
    def __init__(self, reduction: str = ...) -> None: ...

    def forward(self, input: Tensor, target: Tensor) -> Tensor: ...  # type: ignore

    def __call__(self, input: Tensor, target: Tensor) -> Tensor: ...  # type: ignore


class CrossEntropyLoss(_WeightedLoss):
    ignore_index: int = ...

    def __init__(self, weight: Optional[Any] = ..., ignore_index: int = ..., reduction: str = ...) -> None: ...

    def forward(self, input: Tensor, target: Tensor) -> Tensor: ...  # type: ignore

    def __call__(self, input: Tensor, target: Tensor) -> Tensor: ...  # type: ignore


class MultiLabelSoftMarginLoss(_WeightedLoss):
    def __init__(self, weight: Optional[Any] = ..., reduction: str = ...) -> None: ...

    def forward(self, input: Tensor, target: Tensor) -> Tensor: ...  # type: ignore

    def __call__(self, input: Tensor, target: Tensor) -> Tensor: ...  # type: ignore


class CosineEmbeddingLoss(_Loss):
    margin: float = ...

    def __init__(self, margin: float = ..., reduction: str = ...) -> None: ...

    def forward(self, input1: Tensor, input2: Tensor, target: Tensor) -> Tensor: ...  # type: ignore

    def __call__(self, input1: Tensor, input2: Tensor, target: Tensor) -> Tensor: ...  # type: ignore


class MarginRankingLoss(_Loss):
    margin: float = ...

    def __init__(self, margin: float = ..., reduction: str = ...) -> None: ...

    def forward(self, input1: Tensor, input2: Tensor, target: Tensor) -> Tensor: ...  # type: ignore

    def __call__(self, input1: Tensor, input2: Tensor, target: Tensor) -> Tensor: ...  # type: ignore


class MultiMarginLoss(_WeightedLoss):
    p: int = ...
    margin: float = ...

    def __init__(self, p: int = ..., margin: float = ..., weight: Optional[Any] = ...,
                 reduction: str = ...) -> None: ...

    def forward(self, input: Tensor, target: Tensor) -> Tensor: ...  # type: ignore

    def __call__(self, input: Tensor, target: Tensor) -> Tensor: ...  # type: ignore


class TripletMarginLoss(_Loss):
    margin: float = ...
    p: int = ...
    eps: float = ...
    swap: bool = ...

    def __init__(self, margin: float = ..., p: int = ..., eps: float = ..., swap: bool = ...,
                 reduction: str = ...) -> None: ...

    def forward(self, anchor: Tensor, positive: Tensor, negative: Tensor) -> Tensor: ...  # type: ignore

    def __call__(self, anchor: Tensor, positive: Tensor, negative: Tensor) -> Tensor: ...  # type: ignore


class CTCLoss(_Loss):
    blank: int = ...
    zero_infinity: bool = ...

    def __init__(self, blank: int = ..., reduction: str = ..., zero_infinity: bool = ...) -> None: ...

    def forward(self, log_probs: Tensor, targets: Tensor, input_lengths: Tensor, target_lengths: Tensor) -> Tensor: ...  # type: ignore

    def __call__(self, log_probs: Tensor, targets: Tensor, input_lengths: Tensor, target_lengths: Tensor) -> Tensor: ...  # type: ignore
