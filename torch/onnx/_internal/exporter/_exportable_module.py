"""Abstract interface for ONNX exportable modules."""

from __future__ import annotations

import abc
from typing import Any, TYPE_CHECKING
import torch

if TYPE_CHECKING:
    from collections.abc import Sequence


class ExportableModule(torch.nn.Module, abc.ABC):
    """Abstract interface for ONNX exportable modules."""

    @abc.abstractmethod
    def example_arguments(self) -> tuple[tuple[Any], dict[str, Any] | None]:
        raise NotImplementedError

    def dynamic_shapes(self):
        return None

    def input_names(self) -> Sequence[str] | None:
        return None

    def output_names(self) -> Sequence[str] | None:
        return None

    def to_onnx(self, **kwargs: Any) -> torch.onnx.ONNXProgram:
        example_args, example_kwargs = self.example_arguments()
        result = torch.onnx.export(
            self,
            example_args,
            kwargs=example_kwargs,
            input_names=self.input_names(),
            output_names=self.output_names(),
            dynamic_shapes=self.dynamic_shapes(),
            **kwargs,
        )
        assert result is not None
        return result
