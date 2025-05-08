from torch._C import FileCheck as FileCheck

from . import _utils
from ._comparison import assert_allclose, assert_close as assert_close
from ._creation import make_tensor as make_tensor


class CompileCounter:
    def __init__(self) -> None:
        self.frame_count = 0
        self.op_count = 0

    def __call__(
        self, gm: torch.fx.GraphModule, example_inputs: list[torch.Tensor]
    ) -> Callable[..., Any]:
        self.frame_count += 1
        for node in gm.graph.nodes:
            if "call" in node.op:
                self.op_count += 1
        return gm.forward

    def clear(self) -> None:
        self.frame_count = 0
        self.op_count = 0


class CompileCounterWithBackend:
    def __init__(self, backend: str) -> None:
        self.frame_count = 0
        self.op_count = 0
        self.backend = backend
        self.graphs: list[torch.fx.GraphModule] = []

    def __call__(
        self, gm: torch.fx.GraphModule, example_inputs: list[torch.Tensor]
    ) -> Callable[..., Any]:
        from .backends.registry import lookup_backend

        self.frame_count += 1
        for node in gm.graph.nodes:
            if "call" in node.op:
                self.op_count += 1
        self.graphs.append(gm)
        return lookup_backend(self.backend)(gm, example_inputs)

    def clear(self) -> None:
        self.frame_count = 0
        self.op_count = 0
        self.graphs = []
