from abc import ABC
from abc import abstractmethod
import torch

__all__ = [
    "Quantizer",
]

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
