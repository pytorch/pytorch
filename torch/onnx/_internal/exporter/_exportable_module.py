"""Abstract interface for ONNX exportable modules."""

from __future__ import annotations

import abc
from typing import Any, TYPE_CHECKING
import torch

if TYPE_CHECKING:
    from collections.abc import Sequence


class ExportableModule(torch.nn.Module, abc.ABC):
    """Abstract interface for ONNX exportable modules.

    Inherit from this class and implement the defined abstract methods
    to create a module that can be exported to ONNX format.

    Example::
        import torch


        class Model(torch.nn.Module):
            def forward(self, x):
                return x * 2


        class MyExportableModule(ExportableModule):
            def __init__(self):
                super().__init__()
                self.model = Model()

            def forward(self, x):
                return self.model(x)

            def example_arguments(self):
                return (torch.randn(2, 3, 224, 224),), None

            def input_names(self):
                return ("input",)

            def output_names(self):
                return ("output",)

            def dynamic_shapes(self):
                return ({0: "batch_size"},)


        exportable_module = MyExportableModule()
        onnx_program = exportable_module.to_onnx()
        # The model can also be supplied directly to torch.onnx.export
        exportable_module = torch.onnx.export(exportable_module)
    """

    @abc.abstractmethod
    def example_arguments(self) -> tuple[tuple[Any], dict[str, Any] | None]:
        raise NotImplementedError

    def dynamic_shapes(self) -> Any:
        return None

    def input_names(self) -> Sequence[str] | None:
        return None

    def output_names(self) -> Sequence[str] | None:
        return None

    def to_onnx(self, **kwargs: Any) -> torch.onnx.ONNXProgram:
        result = torch.onnx.export(self, **kwargs)
        assert result is not None
        return result
