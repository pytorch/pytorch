from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, List, NamedTuple, Optional

import torch

__all__ = [
    "Quantizer",
]

# TODO: maybe remove torch.float32
SUPPORTED_DTYPES = [torch.uint8, torch.int8, torch.int32, torch.float16, torch.float32]
SUPPORTED_QSCHEMES = [
    torch.per_tensor_affine,
    torch.per_tensor_symmetric,
    torch.per_channel_affine,
    torch.per_channel_symmetric,
    torch.per_channel_affine_float_qparams,
]


@dataclass(eq=True, frozen=True)
class QuantizationSpec:
    dtype: torch.dtype
    is_dynamic: bool = False
    quant_min: Optional[int] = None
    quant_max: Optional[int] = None
    qscheme: Optional[torch.qscheme] = None
    ch_axis: Optional[int] = None

    def __post_init__(self):
        # check dtype is one of the supported types
        if self.dtype not in SUPPORTED_DTYPES:
            raise TypeError(f"Unsupported dtype {self.dtype}.")

        # quant_min must be less than quant_max
        if (
            self.quant_min is not None
            and self.quant_max is not None
            and self.quant_min > self.quant_max
        ):
            raise ValueError(
                f"quant_min {self.quant_min} must be <= quant_max {self.quant_max}."
            )

        # check qscheme is on of the supported ones
        if self.qscheme is not None and self.qscheme not in SUPPORTED_QSCHEMES:
            raise ValueError(f"Unsupported qscheme {self.qscheme}.")

        # ch_axis must be less than the number of channels
        # but no way to check here. Just check that it is not < 0.
        if self.ch_axis is not None and self.ch_axis < 0:
            raise ValueError("Ch_axis is < 0.")


# In the absence of better name, just winging it with QuantizationConfig
@dataclass(eq=True, frozen=True)
class QuantizationConfig:
    activation: QuantizationSpec
    weight: QuantizationSpec
    bias: QuantizationSpec
    is_qat: bool = False

OperatorPatternType = List[Callable]

OperatorConfig = NamedTuple(
    "OperatorConfig",
    # fix List[str] with List[List[Union[nn.Module, FunctionType, BuiltinFunctionType]]]
    # Basically we are mapping a quantization config to some list of patterns.
    # a pattern is defined as a list of nn module, function or builtin function names
    # e.g. [nn.Conv2d, torch.relu, torch.add]
    # We have not resolved whether fusion can be considered internal details of the
    # quantizer hence it does not need communication to user.
    # Note this pattern is not really informative since it does not really
    # tell us the graph structure resulting from the list of ops.
    [
        ("config", QuantizationConfig),
        (
            "operators",
            List[OperatorPatternType],
        ),
    ],
)


class Quantizer(ABC):

    # annotate nodes in the graph with observer or fake quant constructors
    # to convey the desired way of quantization
    @abstractmethod
    def annotate(self, model: torch.fx.GraphModule) -> torch.fx.GraphModule:
        pass

    # validate the annotated graph is supported by the backend
    @abstractmethod
    def validate(self, model: torch.fx.GraphModule) -> None:
        pass

    # annotate nodes in the graph with observer or fake quant constructors
    # to convey the desired way of quantization
    @classmethod
    @abstractmethod
    def get_supported_operators(cls) -> List[OperatorConfig]:
        pass
