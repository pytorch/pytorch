# necessary to surface onnx.ModelProto through ExportOutput:
from __future__ import annotations

import abc
import inspect
import io
import logging
from typing import (
    Any,
    Callable,
    Final,
    Mapping,
    Optional,
    Protocol,
    runtime_checkable,
    Sequence,
    TYPE_CHECKING,
    TypeVar,
    Union,
)

import torch
import torch._ops

from torch.onnx._internal import _beartype
from torch.utils import _pytree

# We can only import onnx from this module in a type-checking context to ensure that
# 'import torch.onnx' continues to work without having 'onnx' installed. We fully
# 'import onnx' inside of dynamo_export (by way of _assert_dependencies).
if TYPE_CHECKING:
    import onnx


_DEFAULT_OPSET_VERSION: Final[int] = 18
"""The default ONNX opset version the exporter will use if one is not specified explicitly
through ``ExportOptions``. This should NEVER be accessed outside of this module! Users
should reference ``ExportOptions.opset_version``."""


class ExportOptions:
    """Options to influence the TorchDynamo ONNX exporter."""

    opset_version: Optional[int] = None
    """The ONNX opset version the exporter should target. Defaults to the latest
    supported ONNX opset version. The default version will increment over time as
    ONNX continues to evolve."""

    dynamic_shapes: Optional[bool] = None
    """Shape information hint for input/output tensors.

    - ``None``: the exporter determines the most compatible setting.
    - ``True``: all input shapes are considered dynamic.
    - ``False``: all input shapes are considered static."""

    op_level_debug: Optional[bool] = None
    """Whether to export the model with op-level debug information by evaluating
    ops through ONNX Runtime. Note: ``op_level_debug`` is not supported when
    ``dynamic_shapes`` is ``True``."""

    logger: Optional[logging.Logger] = None
    """The logger for the ONNX exporter to use. Defaults to creating a child
    logger named "torch.onnx" under the current logger (as returned by
    :py:meth:`logging.getLogger`)."""

    @_beartype.beartype
    def __init__(
        self,
        *,
        opset_version: Optional[int] = None,
        dynamic_shapes: Optional[bool] = None,
        op_level_debug: Optional[bool] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.opset_version = opset_version
        self.dynamic_shapes = dynamic_shapes
        self.op_level_debug = op_level_debug
        self.logger = logger


class ResolvedExportOptions:
    @_beartype.beartype
    def __init__(self, options: Optional[ExportOptions]):
        if options is None:
            options = ExportOptions()

        T = TypeVar("T")

        @_beartype.beartype
        def resolve(value: Optional[T], fallback: Union[T, Callable[[], T]]) -> T:
            if value is not None:
                return value
            if callable(fallback):
                return fallback()
            return fallback

        self.opset_version = resolve(options.opset_version, _DEFAULT_OPSET_VERSION)
        self.dynamic_shapes = resolve(options.dynamic_shapes, False)
        self.op_level_debug = resolve(options.op_level_debug, False)
        self.logger = resolve(
            options.logger, lambda: logging.getLogger().getChild("torch.onnx")
        )

        if self.dynamic_shapes and self.op_level_debug:
            raise RuntimeError(
                "Both ExportOptions.op_level_debug and ExportOptions.dynamic_shapes "
                + "are True but these options are mutually exclusive. Please set only "
                + "one of them to True.",
            )

        for key in dir(options):
            if not key.startswith("_"):  # skip private attributes
                assert hasattr(self, key), f"Unresolved option '{key}'"


@runtime_checkable
class ExportOutputSerializer(Protocol):
    """Protocol for serializing an ONNX graph into a specific format (e.g. Protobuf).
    Note that this is an advanced usage scenario."""

    def serialize(
        self, export_output: ExportOutput, destination: io.BufferedIOBase
    ) -> None:
        """Protocol method that must be implemented for serialization.

        Args:
            export_output: Represents the in-memory exported ONNX model
            destination: A binary IO stream or pre-allocated buffer into which
                the serialized model should be written.

        Example:

            A simple serializer that writes the exported ``onnx.ModelProto`` in Protobuf
            format to ``destination``:

            ::

                class ProtobufExportOutputSerializer:
                    def serialize(
                        self, export_output: ExportOutput, destination: io.BufferedIOBase
                    ) -> None:
                        destination.write(export_output.model_proto.SerializeToString())

                torch.onnx.dynamo_export(...).save(
                    destination="exported_model.onnx",
                    serializer=ProtobufExportOutputSerializer(),
                )
        """
        ...


class ProtobufExportOutputSerializer:
    """Serializes ONNX graph as Protobuf."""

    @_beartype.beartype
    def serialize(
        self, export_output: ExportOutput, destination: io.BufferedIOBase
    ) -> None:
        import onnx

        if not isinstance(export_output.model_proto, onnx.ModelProto):
            raise ValueError("export_output.ModelProto is not an onnx.ModelProto")
        destination.write(export_output.model_proto.SerializeToString())


class ExportOutput:
    """An in-memory representation of a PyTorch model that has been exported to ONNX."""

    _model_proto: Final[onnx.ModelProto]

    @_beartype.beartype
    def __init__(self, model_proto: onnx.ModelProto):
        self._model_proto = model_proto

    @property
    def model_proto(self) -> onnx.ModelProto:
        """The exported ONNX model as an ``onnx.ModelProto``."""

        return self._model_proto

    @_beartype.beartype
    def save(
        self,
        destination: Union[str, io.BufferedIOBase],
        *,
        serializer: Optional[ExportOutputSerializer] = None,
    ) -> None:
        """Saves the in-memory ONNX model to ``destination`` using specified ``serializer``.
        If no ``serializer`` is specified, the model will be serialized as Protobuf."""

        if serializer is None:
            serializer = ProtobufExportOutputSerializer()
        if isinstance(destination, str):
            with open(destination, "wb") as f:
                serializer.serialize(self, f)
        else:
            serializer.serialize(self, destination)


class Exporter(abc.ABC):
    class WrappedFuncModule(torch.nn.Module):
        def __init__(self, forward: Callable):
            super().__init__()
            self.actual_forward = forward

        def forward(self, *args, **kwargs):
            result, _ = _pytree.tree_flatten(self.actual_forward(*args, **kwargs))
            return result

    @_beartype.beartype
    def __init__(
        self,
        options: Union[ExportOptions, ResolvedExportOptions],
        model: Union[torch.nn.Module, Callable],
        model_args: Sequence[Any],
        model_kwargs: Mapping[str, Any],
    ):
        if isinstance(options, ExportOptions):
            self.options = ResolvedExportOptions(options)
        elif isinstance(options, ResolvedExportOptions):
            self.options = options
        assert self.options is not None

        self.model: torch.nn.Module = (
            model
            if isinstance(model, torch.nn.Module)
            else Exporter.WrappedFuncModule(model)
        )
        self.model_args = model_args
        self.model_kwargs = model_kwargs

    @abc.abstractmethod
    def export(self) -> ExportOutput:
        pass

    @property
    def logger(self) -> logging.Logger:
        # options.logger will always be resolved to an instance when constructing
        assert isinstance(self.options.logger, logging.Logger)
        return self.options.logger

    @property
    def model_signature(self) -> inspect.Signature:
        return inspect.signature(
            self.model.actual_forward
            if isinstance(self.model, Exporter.WrappedFuncModule)
            else self.model.forward
        )


class UnsatisfiedDependencyError(RuntimeError):
    """Raised when an ONNX exporter dependency cannot be satisfied."""

    def __init__(self, package_name: str, message: str):
        super().__init__(message)
        self.package_name = package_name


def _assert_dependencies(export_options: ExportOptions):
    T = TypeVar("T")

    def assert_not_optional(value: Optional[T]) -> T:
        assert value is not None
        return value

    logger = assert_not_optional(export_options.logger)
    opset_version = assert_not_optional(export_options.opset_version)

    def missing_package(package_name: str, exc_info: logging._ExcInfoType):
        message = (
            f"Please install the `{package_name}` package "
            f"(e.g. `python -m pip install {package_name}`)."
        )
        logger.fatal(message, exc_info=exc_info)
        return UnsatisfiedDependencyError(package_name, message)

    def missing_opset(package_name: str):
        message = (
            f"The installed `{package_name}` does not support the specified ONNX opset "
            f"version {opset_version}. Install a newer `{package_name}` package or "
            f"specify an older opset version."
        )
        logger.fatal(message)
        return UnsatisfiedDependencyError(package_name, message)

    try:
        import onnx
    except ImportError as e:
        raise missing_package("onnx", e) from e

    if onnx.defs.onnx_opset_version() < opset_version:
        raise missing_opset("onnx")

    try:
        # PyTorch runs lintrunner in CI without onnxscript installed
        import onnxscript  # type: ignore[import]
    except ImportError as e:
        raise missing_package("onnxscript", e) from e

    if not isinstance(
        onnxscript.onnx_opset.all_opsets[("", opset_version)],
        onnxscript.values.Opset,
    ):
        raise missing_opset("onnxscript")


@_beartype.beartype
def dynamo_export(
    model: Union[torch.nn.Module, Callable],
    /,
    *model_args,
    export_options: Optional[ExportOptions] = None,
    **model_kwargs,
) -> ExportOutput:
    """Export a torch.nn.Module to an ONNX graph.

    Args:
        model: The PyTorch model to be exported to ONNX.
        model_args: Positional inputs to ``model``.
        model_kwargs: Keyword inputs to ``model``.
        export_options: Options to influence the export to ONNX.

    Returns:
        An in-memory representation of the exported ONNX model.

    Example:
    ::

        import torch.onnx
        torch.onnx.dynamo_export(
            my_nn_module,
            torch.randn(2, 2, 2), # positional input 1
            torch.randn(2, 2, 2), # positional input 2
            my_nn_module_attribute="hello", # keyword input
            export_options=ExportOptions(
                opset_version=17,
            )
        ).save("my_model.onnx")
    """

    resolved_export_options = ResolvedExportOptions(export_options)

    _assert_dependencies(resolved_export_options)

    from torch.onnx._internal.exporters.dynamo_optimize import DynamoOptimizeExporter

    return DynamoOptimizeExporter(
        options=resolved_export_options,
        model=model,
        model_args=model_args,
        model_kwargs=model_kwargs,
    ).export()


__all__ = [
    "ExportOptions",
    "ExportOutput",
    "ExportOutputSerializer",
    "UnsatisfiedDependencyError",
    "dynamo_export",
]
