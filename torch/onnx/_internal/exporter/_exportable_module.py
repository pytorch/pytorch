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

        class Model(torch.nn.Module):
            def forward(self, x):
                return x * 2


        class MyExportableModule(torch.onnx.ExportableModule):
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
        onnx_program = torch.onnx.export(exportable_module)
    """

    @abc.abstractmethod
    def example_arguments(self) -> tuple[tuple[Any], dict[str, Any] | None]:
        """Return example arguments for the model's forward method.

        This method must be implemented by subclasses to provide sample inputs
        that can be used for tracing, testing, and ONNX export. The returned
        arguments should be representative of the expected input shapes and types
        during inference.

        Example::

            def example_arguments(self):
                # For a model expecting a single tensor input
                return (torch.randn(1, 3, 224, 224),), None


            def example_arguments(self):
                # For a model with multiple inputs and keyword arguments
                return (torch.randn(1, 3, 224, 224), torch.randn(1, 512)), {
                    "temperature": 1.0
                }

        Returns:
            A tuple containing:

            - A tuple of positional arguments to pass to the forward method
            - A dictionary of keyword arguments (or None if no kwargs are needed)
        """
        raise NotImplementedError

    def dynamic_shapes(self) -> Any:
        """Return dynamic shape specifications for the model's inputs.

        Override this method to specify which dimensions of the input tensors
        should be treated as dynamic during ONNX export. This allows the exported
        model to accept inputs with varying sizes along the specified dimensions.

        Example::

            def dynamic_shapes(self):
                # Specify batch dimension as dynamic for input named 'x'
                return {"x": {0: "batch_size"}}


            def dynamic_shapes(self):
                # Multiple dynamic dimensions
                return {
                    "input": {0: "batch_size", 2: "height", 3: "width"},
                    "mask": {0: "batch_size"},
                }

        Note:
            The default implementation returns None, indicating all dimensions are static.

        Returns:
            Dynamic shape specification compatible with ``torch.export.export``.
            Return None if all input dimensions should be static. The format can be:

            - A dictionary mapping input names to dimension specifications
            - A tuple/list of dimension specifications corresponding to inputs
            - Any format accepted by the ``dynamic_shapes`` parameter of ``torch.export.export``
        """
        return None

    def input_names(self) -> Sequence[str] | None:
        """Return names for the model's input tensors.

        Override this method to provide custom names for the input tensors in the
        exported ONNX model. These names will be used as identifiers in the ONNX
        graph and can be useful for debugging and model inspection.

        Example::

            def input_names(self):
                return ["image", "mask"]


            def input_names(self):
                # For a single input
                return ["input_tensor"]

        Note:
            The default implementation returns None, which results in auto-generated names.

        Returns:
            A sequence of strings representing input names, or None to use default names.
            The number of names should match the number of positional arguments in the
            forward method.
        """
        return None

    def output_names(self) -> Sequence[str] | None:
        """Return names for the model's output tensors.

        Override this method to provide custom names for the output tensors in the
        exported ONNX model. These names will be used as identifiers in the ONNX
        graph and can be useful for debugging and model inspection.

        Example::

            def output_names(self):
                return ["logits", "probabilities"]


            def output_names(self):
                # For a single output
                return ["prediction"]

        Note:
            The default implementation returns None, which results in auto-generated names.

        Returns:
            A sequence of strings representing output names, or None to use default names.
            The number of names should match the number of outputs from the forward method.
            For models returning multiple outputs, provide a name for each output.
        """
        return None

    def to_onnx(self, **kwargs: Any) -> torch.onnx.ONNXProgram:
        """Export the module to ONNX format.

        This method provides a convenient wrapper around ``torch.onnx.export`` that
        automatically uses the example arguments, dynamic shapes, and input/output
        names defined by the module. Additional export options can be specified via
        keyword arguments.

        See Also: ``torch.onnx.export`` for complete documentation of export options.

        Args:
            **kwargs: Additional keyword arguments to pass to ``torch.onnx.export``.
                Common options include:

                - ``opset_version`` (int): The ONNX opset version to target
                - ``optimize`` (bool): Whether to apply optimizations to the exported model

        Returns:
            An ONNXProgram object containing the exported model and metadata.
        """
        result = torch.onnx.export(self, **kwargs)
        assert result is not None
        return result
