from abc import ABC
from abc import abstractmethod
from torch.ao.quantization import BackendConfig
from typing import Optional

class Quantizer(ABC):

    def __init__(self, backend_config: Optional[BackendConfig] = None):
        super().__init__()
        self.backend_config = backend_config

    @abstractmethod
    def annotate(model: torch.fx.GraphModule) -> torch.fx.GraphModule:
        pass

    def validate(model: torch.fx.GraphModule) -> None:
        # TODO: validate the annotated graph with BackendConfig
        return True
