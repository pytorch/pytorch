# necessary to surface onnx.ModelProto through ExportOutput:
from __future__ import annotations

import abc

import contextlib
import dataclasses
import io
import logging
import os

import warnings
from collections import defaultdict
from typing import (
    Any,
    Callable,
    Dict,
    Final,
    List,
    Mapping,
    Optional,
    Protocol,
    runtime_checkable,
    Sequence,
    Set,
    Tuple,
    TYPE_CHECKING,
    TypeVar,
    Union,
)

from typing_extensions import Self

import torch
import torch._ops
import torch.utils._pytree as pytree
from torch._subclasses import fake_tensor

from torch.onnx._internal import _beartype, io_adapter
from torch.onnx._internal.diagnostics import infra

from torch.onnx._internal.fx import (
    decomposition_table,
    patcher as patcher,
    registration,
    serialization as fx_serialization,
)

# We can only import onnx from this module in a type-checking context to ensure that
# 'import torch.onnx' continues to work without having 'onnx' installed. We fully
# 'import onnx' inside of dynamo_export (by way of _assert_dependencies).
if TYPE_CHECKING:
    import onnx
    import onnxscript  # type: ignore[import]
    from onnxscript.function_libs.torch_lib import (  # type: ignore[import]
        registration as torchlib_registry,
    )

    from torch.onnx._internal.fx import diagnostics
else:
    try:
        # beartype needs this import due to runtime type checking.
        # This cannot be normally imported at top level due to
        # https://github.com/pytorch/pytorch/issues/103764
        from torch.onnx._internal.fx import diagnostics
    except ImportError:
        # The error will be handled elsewhere when the exporter is used.
        pass

_DEFAULT_OPSET_VERSION: Final[int] = 18
"""The default ONNX opset version the exporter will use if one is not specified explicitly
through ``ExportOptions``. This should NEVER be accessed outside of this module! Users
should reference ``ExportOptions.opset_version``."""

_PYTORCH_GITHUB_ISSUES_URL = "https://github.com/pytorch/pytorch/issues"
"""The URL to the PyTorch GitHub issues page."""

_DEFAULT_FAILED_EXPORT_SARIF_LOG_PATH = "report_dynamo_export.sarif"
"""The default path to write the SARIF log to if the export fails."""

log = logging.getLogger(__name__)


DiagnosticOptions = infra.DiagnosticOptions


@dataclasses.dataclass
class ONNXFakeContext:
    """A dataclass used to store context for model export using FakeTensor.

    This dataclass stores the FakeTensorMode instance used to convert
    real tensors and model parameters into fake tensors. This ``fake_mode`` is
    reused internally during tracing of a ``torch.nn.Module`` into a FX ``GraphModule``.
    """

    fake_mode: fake_tensor.FakeTensorMode
    """The fake tensor mode used for tracing model using fake tensors and parameters."""

    state_dict_paths: Optional[Tuple[Union[str, io.BytesIO]]] = None
    """List of paths of files that contain the model `state_dict`"""


class OnnxRegistry:
    """Registry for ONNX functions.

    The registry maintains a mapping from qualified names to symbolic functions under a
    fixed opset version. It supports registering custom onnx-script functions and for
    dispatcher to dispatch calls to the appropriate function.

    """

    def __init__(self) -> None:
        """Initializes the registry"""

        # NOTE: _registry is the registry maps OpNameto a list of ONNXFunctions. It is important
        # not to directly modify this variable. Instead, access to it should be done through
        # the public methods: register_custom_op, get_ops, and is_registered_op.
        self._registry: Dict[
            registration.OpName, List[registration.ONNXFunction]
        ] = defaultdict(list)
        # FIXME: Avoid importing onnxscript into torch
        from onnxscript.function_libs.torch_lib import (  # type: ignore[import]  # noqa: F401
            ops,  # TODO(titaiwang): get rid of this import
            registration,
        )

        # opset_version is unused for now, since torchlib only supports opset18.
        # TODO: get opset version from torchlib
        self._opset_version = _DEFAULT_OPSET_VERSION
        warnings.warn(
            f"torch.onnx.dynamo_export only implements opset version {self._opset_version} for now. If you need to use a "
            "different opset version, please register them with register_custom_op."
        )

        # Initialize registry from torchlib
        self._initiate_registry_from_torchlib(registration.default_registry)

    @property
    def opset_version(self) -> int:
        """The ONNX opset version the exporter should target. Defaults to the latest
        supported ONNX opset version: 18. The default version will increment over time as
        ONNX continues to evolve."""

        return self._opset_version

    # TODO(titaiwang): subject to change if multiple opset_version is supported in torchlib
    def _initiate_registry_from_torchlib(
        self, torchlib_registry: torchlib_registry.Registry
    ):
        """Populates the registry with ATen functions from torchlib.

        Args:
            torchlib_registry: The torchlib registry to use for populating the registry.
        """
        for aten_name, aten_overloads_func in torchlib_registry.items():
            internal_name_instance = registration.OpName.from_qualified_name(aten_name)
            for overload_func in aten_overloads_func.overloads:
                symbolic_function = registration.ONNXFunction(
                    onnx_function=overload_func,
                    op_full_name=internal_name_instance.qualified_name(),
                    is_custom=False,
                    is_complex=False,
                )
                self._register(internal_name_instance, symbolic_function)

            for complex_func in aten_overloads_func.complex:
                symbolic_function = registration.ONNXFunction(
                    onnx_function=complex_func,
                    op_full_name=internal_name_instance.qualified_name(),
                    is_custom=False,
                    is_complex=True,
                )
                self._register(internal_name_instance, symbolic_function)

    @_beartype.beartype
    def _register(
        self,
        internal_qualified_name: registration.OpName,
        symbolic_function: registration.ONNXFunction,
    ) -> None:
        """Registers a ONNXFunction to an operator.

        Args:
            internal_qualified_name: The qualified name of the operator to register: OpName.
            symbolic_function: The ONNXFunction to register.
        """
        self._registry[internal_qualified_name].append(symbolic_function)

    @_beartype.beartype
    def register_op(
        self,
        function: Union["onnxscript.OnnxFunction", "onnxscript.TracedOnnxFunction"],
        namespace: str,
        op_name: str,
        overload: Optional[str] = None,
        is_complex: bool = False,
    ) -> None:
        """Registers a custom operator: torch.ops.<namespace>.<op_name>.<overload>.

        Args:
            function: The onnx-sctip function to register.
            namespace: The namespace of the operator to register.
            op_name: The name of the operator to register.
            overload: The overload of the operator to register. If it's default overload,
                leave it to None.
            is_complex: Whether the function is a function that handles complex valued inputs.

        Raises:
            ValueError: If the name is not in the form of 'namespace::op'.
        """
        internal_name_instance = registration.OpName.from_name_parts(
            namespace=namespace, op_name=op_name, overload=overload
        )
        symbolic_function = registration.ONNXFunction(
            onnx_function=function,
            op_full_name=internal_name_instance.qualified_name(),
            is_custom=True,
            is_complex=is_complex,
        )
        self._register(internal_name_instance, symbolic_function)

    @_beartype.beartype
    def get_op_functions(
        self, namespace: str, op_name: str, overload: Optional[str] = None
    ) -> Optional[List[registration.ONNXFunction]]:
        """Returns a list of ONNXFunctions for the given op: torch.ops.<namespace>.<op_name>.<overload>.

        The list is ordered by the time of registration. The custom operators should be
        in the second half of the list.

        Args:
            namespace: The namespace of the operator to get.
            op_name: The name of the operator to get.
            overload: The overload of the operator to get. If it's default overload,
                leave it to None.
        Returns:
            A list of ONNXFunctions corresponding to the given name, or None if
            the name is not in the registry.
        """
        internal_name_instance = registration.OpName.from_name_parts(
            namespace=namespace, op_name=op_name, overload=overload
        )
        return self._registry.get(internal_name_instance)

    @_beartype.beartype
    def is_registered_op(
        self, namespace: str, op_name: str, overload: Optional[str] = None
    ) -> bool:
        """Returns whether the given op is registered: torch.ops.<namespace>.<op_name>.<overload>.

        Args:
            namespace: The namespace of the operator to check.
            op_name: The name of the operator to check.
            overload: The overload of the operator to check. If it's default overload,
                leave it to None.

        Returns:
            True if the given op is registered, otherwise False.
        """
        functions = self.get_op_functions(
            namespace=namespace, op_name=op_name, overload=overload
        )
        return functions is not None

    @_beartype.beartype
    def _all_registered_ops(self) -> Set[str]:
        """Returns the set of all registered function names."""
        return {
            op_name_class.qualified_name() for op_name_class in self._registry.keys()
        }


class ExportOptions:
    """Options to influence the TorchDynamo ONNX exporter."""

    dynamic_shapes: Optional[bool] = None
    """Shape information hint for input/output tensors.

    - ``None``: the exporter determines the most compatible setting.
    - ``True``: all input shapes are considered dynamic.
    - ``False``: all input shapes are considered static."""

    op_level_debug: Optional[bool] = None
    """Whether to export the model with op-level debug information by evaluating
    ops through ONNX Runtime."""

    diagnostic_options: DiagnosticOptions
    """The diagnostic options for the exporter."""

    fake_context: Optional[ONNXFakeContext] = None
    """The fake context used for symbolic tracing."""

    onnx_registry: Optional[OnnxRegistry] = None
    """The ONNX registry used to register ATen operators to ONNX functions. Defaults to
    opset18."""

    @_beartype.beartype
    def __init__(
        self,
        *,
        dynamic_shapes: Optional[bool] = None,
        op_level_debug: Optional[bool] = None,
        fake_context: Optional[ONNXFakeContext] = None,
        onnx_registry: Optional[OnnxRegistry] = None,
        diagnostic_options: Optional[DiagnosticOptions] = None,
    ):
        self.dynamic_shapes = dynamic_shapes
        self.op_level_debug = op_level_debug
        self.fake_context = fake_context
        self.onnx_registry = onnx_registry
        self.diagnostic_options = diagnostic_options or DiagnosticOptions()


class ResolvedExportOptions(ExportOptions):
    """Consolidates `ExportOptions` with default values.
    All unspecified options from `ExportOptions` are assigned a default value.
    This is an internal class and its API may be changed at any time without notice.
    """

    # Public attributes MUST be redefined below without ``Optional[]`` from ``ExportOptions``
    dynamic_shapes: bool
    op_level_debug: bool
    diagnostic_options: DiagnosticOptions
    fake_context: ONNXFakeContext
    onnx_registry: OnnxRegistry

    # Private only attributes
    decomposition_table: Dict[torch._ops.OpOverload, Callable]
    """A dictionary that maps operators to their decomposition functions."""

    onnxfunction_dispatcher: torch.onnx._internal.fx.onnxfunction_dispatcher.OnnxFunctionDispatcher
    """The ONNX dispatcher used to dispatch ATen operators to ONNX functions."""

    fx_tracer: FXGraphExtractor
    """The FXGraphExtractor instance used to extract the FX graph from the model."""

    diagnostic_context: diagnostics.DiagnosticContext
    """The diagnostics context for the export. Responsible for recording diagnostics,
    logging diagnostics, and generating the SARIF log."""

    @_beartype.beartype
    def __init__(
        self, options: Optional[Union[ExportOptions, "ResolvedExportOptions"]]
    ):
        if options is None:
            options = ExportOptions()
        if isinstance(options, ResolvedExportOptions):
            self.dynamic_shapes = options.dynamic_shapes
            self.op_level_debug = options.op_level_debug
            self.diagnostic_options = options.diagnostic_options
            self.fake_context = options.fake_context
            # private
            self.fx_tracer = options.fx_tracer
            self.onnx_registry = options.onnx_registry
            self.onnxfunction_dispatcher = options.onnxfunction_dispatcher
            self.decomposition_table = options.decomposition_table
            self.diagnostic_context = options.diagnostic_context
        else:
            T = TypeVar("T")

            @_beartype.beartype
            def resolve(value: Optional[T], fallback: Union[T, Callable[[], T]]) -> T:
                if value is not None:
                    return value
                if callable(fallback):
                    return fallback()
                return fallback

            self.dynamic_shapes = resolve(options.dynamic_shapes, False)
            from torch.onnx._internal.fx import (  # TODO: Prevent circular dep
                diagnostics,
                dynamo_graph_extractor,
            )

            self.diagnostic_options = resolve(
                options.diagnostic_options, DiagnosticOptions()
            )

            self.fx_tracer = dynamo_graph_extractor.DynamoExport()

            self.fake_context = resolve(options.fake_context, None)
            self.diagnostic_context = diagnostics.DiagnosticContext(
                "torch.onnx.dynamo_export",
                torch.__version__,
                self.diagnostic_options,
            )

            self.onnx_registry = resolve(options.onnx_registry, OnnxRegistry())
            self.decomposition_table = (
                decomposition_table.create_onnx_friendly_decomposition_table(
                    self.onnx_registry
                )
            )

            # TODO(titaiwang, bowbao): Better way to annotate `onnxscript` types in diagnostics.
            from torch.onnx._internal.fx import onnxfunction_dispatcher

            self.op_level_debug = resolve(options.op_level_debug, False)
            self.onnxfunction_dispatcher = (
                onnxfunction_dispatcher.OnnxFunctionDispatcher(
                    self.onnx_registry,
                    self.diagnostic_context,
                )
            )

            for key in dir(options):
                if not key.startswith("_"):  # skip private attributes
                    assert hasattr(self, key), f"Unresolved option '{key}'"


@contextlib.contextmanager
def enable_fake_mode():
    """Enable fake mode for the duration of the context.

    Internally it instantiates a `FakeTensorMode` context manager that converts
    user input and model parameters into `FakeTensor`.

    A [FakeTensor](https://github.com/pytorch/pytorch/blob/main/torch/_subclasses/fake_tensor.py#L870)
    is a `torch.Tensor` with the ability to run PyTorch code without having to
    actually do computation through tensors allocated on a `meta` device. Because
    there is no actual data being allocated on the device, this API allows for
    exporting large models without the actual memory footprint needed for executing it.

    It is highly recommended to enable fake mode when exporting models that
    are too large to fit into memory.

    Returns:
        A `ONNXFakeContext` object that must be passed to `torch.onnx.dynamo_export`
        through the `ExportOptions.fake_context` argument.

    Example::

        # xdoctest: +REQUIRES(env:TORCH_DOCTEST_ONNX)
        >>> import torch
        >>> import torch.onnx
        >>> class MyModel(torch.nn.Module):  # Dummy model
        ...     def __init__(self) -> None:
        ...         super().__init__()
        ...         self.linear = torch.nn.Linear(2, 2)
        ...     def forward(self, x):
        ...         out = self.linear(x)
        ...         return out
        >>> with torch.onnx.enable_fake_mode() as fake_context:
        ...     my_nn_module = MyModel()
        ...     arg1 = torch.randn(2, 2, 2)  # positional input 1
        >>> export_options = torch.onnx.ExportOptions(fake_context=fake_context)
        >>> export_output = torch.onnx.dynamo_export(
        ...     my_nn_module,
        ...     arg1,
        ...     export_options=export_options
        ... )
        >>> # Saving model WITHOUT initializers
        >>> export_output.save("my_model_without_initializers.onnx")
        >>> # Saving model WITH initializers
        >>> export_output.save("my_model_with_initializers.onnx", model_state_dict=MyModel().state_dict())

    .. warning::
        This API is experimental and is *NOT* backward-compatible.

    """
    from torch._subclasses import fake_tensor
    from torch.fx.experimental.symbolic_shapes import ShapeEnv

    # This overrides the internal `FakeTensorMode` instance created by `torch._dynamo.export`[1].
    # It is a good idea to keep them in sync (constructor args) to maintain the same default behavior
    # [1] `torch/_dynamo/output_graph.py::InstructionTranslator::OutputGraph.__init__`
    # Mixed fake/real tensors are only allowed when `torch.onnx.dynamo_export` is not called within `FakeTensorMode`
    # This is needed because models can create new parameters during `forward(self, *args, **kwargs)` run
    fake_mode = fake_tensor.FakeTensorMode(
        allow_non_fake_inputs=not torch._guards.detect_fake_mode(),
        shape_env=ShapeEnv(
            allow_scalar_outputs=False, allow_dynamic_output_shape_ops=False
        ),
    )
    # The patcher is needed for when user calls `fake_model.load_state_dict(...)` within fake mode
    patcher_context = patcher.ONNXTorchPatcher()
    fake_context = ONNXFakeContext(fake_mode=fake_mode)
    with fake_mode, patcher_context:
        yield fake_context
    fake_context.state_dict_paths = tuple(
        patcher_context.paths,
    )  # type: ignore[assignment]


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

                # xdoctest: +REQUIRES(env:TORCH_DOCTEST_ONNX)
                >>> import io
                >>> import torch
                >>> import torch.onnx
                >>> class MyModel(torch.nn.Module):  # Dummy model
                ...     def __init__(self) -> None:
                ...         super().__init__()
                ...         self.linear = torch.nn.Linear(2, 2)
                ...     def forward(self, x):
                ...         out = self.linear(x)
                ...         return out
                >>> class ProtobufExportOutputSerializer:
                ...     def serialize(
                ...         self, export_output: torch.onnx.ExportOutput, destination: io.BufferedIOBase
                ...     ) -> None:
                ...         destination.write(export_output.model_proto.SerializeToString())
                >>> model = MyModel()
                >>> arg1 = torch.randn(2, 2, 2)  # positional input 1
                >>> torch.onnx.dynamo_export(model, arg1).save(
                ...     destination="exported_model.onnx",
                ...     serializer=ProtobufExportOutputSerializer(),
                ... )
        """
        ...


class ProtobufExportOutputSerializer:
    """Serializes ONNX graph as Protobuf."""

    @_beartype.beartype
    def serialize(
        self, export_output: ExportOutput, destination: io.BufferedIOBase
    ) -> None:
        import onnx

        if not isinstance(export_output.model_proto, onnx.ModelProto):  # type: ignore[attr-defined]
            raise ValueError("export_output.ModelProto is not an onnx.ModelProto")
        destination.write(export_output.model_proto.SerializeToString())


class LargeProtobufExportOutputSerializer:
    """Serializes ONNX graph as Protobuf.

    Fallback to serializing as Protobuf with external data for models larger than 2GB.
    """

    _destination_path: Final[str]

    def __init__(self, destination_path: str):
        self._destination_path = destination_path

    @_beartype.beartype
    def serialize(
        self, export_output: ExportOutput, destination: io.BufferedIOBase
    ) -> None:
        """`destination` is ignored. The model is saved to `self._destination_path` instead."""
        import onnx

        try:
            onnx.save_model(export_output.model_proto, self._destination_path)  # type: ignore[attr-defined]
        except ValueError:
            # ValueError: Message onnx.ModelProto exceeds maximum protobuf size of 2GB
            # Fallback to serializing the model with external data.
            onnx.save_model(  # type: ignore[attr-defined]
                export_output.model_proto,
                self._destination_path,
                save_as_external_data=True,
            )


class ExportOutput:
    """An in-memory representation of a PyTorch model that has been exported to ONNX."""

    _model_proto: Final[onnx.ModelProto]  # type: ignore[name-defined]
    _input_adapter: Final[io_adapter.InputAdapter]
    _output_adapter: Final[io_adapter.OutputAdapter]
    _diagnostic_context: Final[diagnostics.DiagnosticContext]
    _fake_context: Final[Optional[ONNXFakeContext]]
    _export_exception: Final[Optional[Exception]]

    @_beartype.beartype
    def __init__(
        self,
        model_proto: onnx.ModelProto,  # type: ignore[name-defined]
        input_adapter: io_adapter.InputAdapter,
        output_adapter: io_adapter.OutputAdapter,
        diagnostic_context: diagnostics.DiagnosticContext,
        *,
        fake_context: Optional[ONNXFakeContext] = None,
        export_exception: Optional[Exception] = None,
    ):
        self._model_proto = model_proto
        self._input_adapter = input_adapter
        self._output_adapter = output_adapter
        self._diagnostic_context = diagnostic_context
        self._fake_context = fake_context
        self._export_exception = export_exception

    @property
    def model_proto(self) -> onnx.ModelProto:  # type: ignore[name-defined]
        """The exported ONNX model as an ``onnx.ModelProto``."""

        if self._export_exception is not None:
            raise self._export_exception
        return self._model_proto

    @property
    def diagnostic_context(self) -> diagnostics.DiagnosticContext:
        """The diagnostic context associated with the export."""

        return self._diagnostic_context

    @property
    def fake_context(self) -> Optional[ONNXFakeContext]:
        """The fake context associated with the export."""

        return self._fake_context

    @_beartype.beartype
    def adapt_torch_inputs_to_onnx(
        self, *model_args, **model_kwargs
    ) -> Sequence[Union[torch.Tensor, int, float, bool]]:
        """Converts the PyTorch model inputs to exported ONNX model inputs format.

        Due to design differences, input/output format between PyTorch model and exported
        ONNX model are often not the same. E.g., None is allowed for PyTorch model, but are
        not supported by ONNX. Nested constructs of tensors are allowed for PyTorch model,
        but only flattened tensors are supported by ONNX, etc.

        The actual adapting steps are associated with each individual export. It
        depends on the PyTorch model, the particular set of model_args and model_kwargs
        used for the export, and export options.

        This method replays the adapting steps recorded during export.

        Args:
            model_args: The PyTorch model inputs.
            model_kwargs: The PyTorch model keyword inputs.

        Returns:
            A sequence of tensors converted from PyTorch model inputs.

        Example::

            # xdoctest: +REQUIRES(env:TORCH_DOCTEST_ONNX)
            >>> import torch
            >>> import torch.onnx
            >>> from typing import Dict, Tuple
            >>> def func_with_nested_input_structure(
            ...     x_dict: Dict[str, torch.Tensor],
            ...     y_tuple: Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
            ... ):
            ...     if "a" in x_dict:
            ...         x = x_dict["a"]
            ...     elif "b" in x_dict:
            ...         x = x_dict["b"]
            ...     else:
            ...         x = torch.randn(3)
            ...
            ...     y1, (y2, y3) = y_tuple
            ...
            ...     return x + y1 + y2 + y3
            >>> x_dict = {"a": torch.tensor(1.)}
            >>> y_tuple = (torch.tensor(2.), (torch.tensor(3.), torch.tensor(4.)))
            >>> export_output = torch.onnx.dynamo_export(func_with_nested_input_structure, x_dict, y_tuple)
            >>> print(x_dict, y_tuple)
            {'a': tensor(1.)} (tensor(2.), (tensor(3.), tensor(4.)))
            >>> print(export_output.adapt_torch_inputs_to_onnx(x_dict, y_tuple))
            (tensor(1.), tensor(2.), tensor(3.), tensor(4.))

        .. warning::
            This API is experimental and is *NOT* backward-compatible.

        """
        return self._input_adapter.apply(*model_args, **model_kwargs)

    @_beartype.beartype
    def adapt_torch_outputs_to_onnx(
        self, model_outputs: Any
    ) -> Sequence[Union[torch.Tensor, int, float, bool]]:
        """Converts the PyTorch model outputs to exported ONNX model outputs format.

        Due to design differences, input/output format between PyTorch model and exported
        ONNX model are often not the same. E.g., None is allowed for PyTorch model, but are
        not supported by ONNX. Nested constructs of tensors are allowed for PyTorch model,
        but only flattened tensors are supported by ONNX, etc.

        The actual adapting steps are associated with each individual export. It
        depends on the PyTorch model, the particular set of model_args and model_kwargs
        used for the export, and export options.

        This method replays the adapting steps recorded during export.

        Args:
            model_outputs: The PyTorch model outputs.

        Returns:
            PyTorch model outputs in exported ONNX model outputs format.

        Example::

            # xdoctest: +REQUIRES(env:TORCH_DOCTEST_ONNX)
            >>> import torch
            >>> import torch.onnx
            >>> def func_returning_tuples(x, y, z):
            ...     x = x + y
            ...     y = y + z
            ...     z = x + y
            ...     return (x, (y, z))
            >>> x = torch.tensor(1.)
            >>> y = torch.tensor(2.)
            >>> z = torch.tensor(3.)
            >>> export_output = torch.onnx.dynamo_export(func_returning_tuples, x, y, z)
            >>> pt_output = func_returning_tuples(x, y, z)
            >>> print(pt_output)
            (tensor(3.), (tensor(5.), tensor(8.)))
            >>> print(export_output.adapt_torch_outputs_to_onnx(pt_output))
            [tensor(3.), tensor(5.), tensor(8.)]

        .. warning::
            This API is experimental and is *NOT* backward-compatible.

        """
        return self._output_adapter.apply(model_outputs)

    @_beartype.beartype
    def save(
        self,
        destination: Union[str, io.BufferedIOBase],
        *,
        model_state_dict: Optional[Union[Dict[str, Any], str]] = None,
        serializer: Optional[ExportOutputSerializer] = None,
    ) -> None:
        """Saves the in-memory ONNX model to ``destination`` using specified ``serializer``.

        Args:
            destination: The destination to save the ONNX model. It can be either a string or a file-like object.
                When used with ``model_state_dict``, it must be a string with a full path to the destination.
                In that case, besides saving the ONNX model, a folder with "_initializers" suffix (without extension)
                will be created to store the each initializer of the ONNX model in a separate file. For example, if the
                destination is "/path/model.onnx", the initializers will be saved in "/path/model_initializers/" folder.
            model_state_dict: The state_dict of the PyTorch model containing all weights on it.
                It can be either a dict as returned by `model.state_dict()`, or a string with a file name.
                Required when ``enable_fake_mode`` is used but real initializers are needed on the ONNX graph.
                It can be either a string with the path to a checkpoint or a dictionary with the actual model state.

            serializer: The serializer to use. If not specified, the model will be serialized as Protobuf.
        """

        if serializer is None:
            if isinstance(destination, str):
                serializer = LargeProtobufExportOutputSerializer(destination)
            else:
                serializer = ProtobufExportOutputSerializer()

        # Add initializers when symbolic tracing is enabled
        _model_state_dict_files: List[Union[str, io.BytesIO]] = []
        if model_state_dict is not None:
            if isinstance(model_state_dict, dict):
                model_state_dict_file = io.BytesIO()
                torch.save(model_state_dict, model_state_dict_file)
                model_state_dict_file.seek(0)
                _model_state_dict_files.append(model_state_dict_file)
            else:
                isinstance(
                    model_state_dict, str
                ), "model_state_dict must be a path to the model's state_dict or the actual state_dict"
                _model_state_dict_files.append(model_state_dict)

        # Load state from previous model.load_state_dict() call within enable_fake_mode() context
        if self._fake_context and self._fake_context.state_dict_paths:
            for path in self._fake_context.state_dict_paths:
                if path in _model_state_dict_files:
                    # ignore duplicate
                    continue
                try:
                    extra_state_dict = torch.load(path)
                    extra_state_dict_file = io.BytesIO()
                    torch.save(extra_state_dict, extra_state_dict_file)
                    extra_state_dict_file.seek(0)
                    _model_state_dict_files.append(extra_state_dict_file)
                except FileNotFoundError:
                    # It is ok to ignore transient state_dict file created within context manager
                    pass

        if _model_state_dict_files:
            if not isinstance(destination, str):
                raise RuntimeError(
                    "`destination` must be a string with a path when model_state_dict is specified."
                )
            destination_path, destination_filename = os.path.split(destination)
            onnx_model_location = destination_filename
            onnx_initializer_location = (
                destination_filename.split(".")[0] + "_initializers"
            )
            # TODO: Should this be part of the serializer?
            fx_serialization.save_model_with_external_data(
                destination_path,
                onnx_model_location,
                onnx_initializer_location,
                tuple(_model_state_dict_files),
                self.model_proto,
            )
        else:
            if isinstance(destination, str):
                with open(destination, "wb") as f:
                    serializer.serialize(self, f)
            else:
                try:
                    serializer.serialize(self, destination)
                except ValueError:
                    raise ValueError(
                        "'destination' should be provided as a path-like string when saving a model larger than 2GB. "
                        "External tensor data will be saved alongside the model on disk."
                    )

    @_beartype.beartype
    def save_diagnostics(self, destination: str) -> None:
        """Saves the export diagnostics as a SARIF log to the specified destination path.

        Args:
            destination: The destination to save the diagnostics SARIF log.
                It must have a `.sarif` extension.

        Raises:
            ValueError: If the destination path does not end with `.sarif` extension.
        """
        if not destination.endswith(".sarif"):
            message = f"'destination' must have a .sarif extension, got {destination}"
            log.fatal(message)
            raise ValueError(message)

        self.diagnostic_context.dump(destination)

    @classmethod
    def _from_failure(
        cls,
        export_exception: Exception,
        diagnostic_context: diagnostics.DiagnosticContext,
    ) -> Self:
        """
        Creates an instance of ``ExportOutput`` when the export process encounters a failure.

        In case of a failed export, this method is used to encapsulate the exception
        and associated diagnostic context within an ``ExportOutput`` instance for
        easier handling and debugging.

        Args:
            export_exception: The exception raised during the export process.
            diagnostic_context: The context associated with diagnostics during export.

        Returns:
            An instance of ``ExportOutput`` representing the failed export output.
        """
        # Defer `import onnx` out of `import torch` path
        # https://github.com/pytorch/pytorch/issues/103764
        import onnx

        return ExportOutput(
            onnx.ModelProto(),  # type: ignore[attr-defined]
            io_adapter.InputAdapter(),
            io_adapter.OutputAdapter(),
            diagnostic_context,
            export_exception=export_exception,
        )


class FXGraphExtractor(abc.ABC):
    """Abstract interface for FX graph extractor engines.
    This class isolates FX extraction logic from the rest of the export logic.
    That allows a single ONNX exporter that can leverage different FX graphs."""

    def __init__(self) -> None:
        super().__init__()
        self.input_adapter: io_adapter.InputAdapter = io_adapter.InputAdapter()
        self.output_adapter: io_adapter.OutputAdapter = io_adapter.OutputAdapter()

    @abc.abstractmethod
    def generate_fx(
        self,
        options: ResolvedExportOptions,
        model: Union[torch.nn.Module, Callable],
        model_args: Sequence[Any],
        model_kwargs: Mapping[str, Any],
    ) -> torch.fx.GraphModule:
        """Analyzes user ``model`` and generates a FX graph.
        Args:
            options: The export options.
            model: The user model.
            model_args: The model's positional input arguments.
            model_kwargs: The model's keyword input arguments.
        Returns:
            The generated FX Graph.
        """
        ...


class Exporter:
    @_beartype.beartype
    def __init__(
        self,
        options: Union[ExportOptions, ResolvedExportOptions],
        model: Union[torch.nn.Module, Callable],
        model_args: Sequence[Any],
        model_kwargs: Mapping[str, Any],
    ):
        self.options = ResolvedExportOptions(options)
        assert self.options is not None

        self.model = model
        self.model_args = model_args
        self.model_kwargs = model_kwargs

        # TODO:Retire FXSymbolicTracer
        # NOTE: FXSymbolicTracer would fail in this assert, as it does not use `enable_fake_mode`
        from torch.onnx._internal.fx import fx_symbolic_graph_extractor

        if not isinstance(
            self.options.fx_tracer, fx_symbolic_graph_extractor.FXSymbolicTracer
        ):
            self._assert_fake_tensor_mode()

    def export(self) -> ExportOutput:
        with self.options.diagnostic_context:
            graph_module = self.options.fx_tracer.generate_fx(
                self.options, self.model, self.model_args, self.model_kwargs
            )

            updated_model_args = self.options.fx_tracer.input_adapter.apply(
                *self.model_args, **self.model_kwargs
            )

            # TODO: Design the passes API
            graph_module = pre_export_passes(
                self.options, self.model, graph_module, updated_model_args
            )

            # TODO: Defer `import onnxscript` out of `import torch` path
            # https://github.com/pytorch/pytorch/issues/103764
            from torch.onnx._internal.fx import fx_onnx_interpreter

            fx_interpreter = fx_onnx_interpreter.FxOnnxInterpreter(
                diagnostic_context=self.options.diagnostic_context
            )
            onnxscript_graph = fx_interpreter.run(
                fx_graph_module=graph_module,
                onnxfunction_dispatcher=self.options.onnxfunction_dispatcher,
                op_level_debug=self.options.op_level_debug,
            )

            # Export TorchScript graph to ONNX ModelProto.
            onnx_model = onnxscript_graph.to_model_proto(
                self.options.onnx_registry.opset_version,
                include_initializers=self.options.fake_context is None,
            )

            return torch.onnx.ExportOutput(
                onnx_model,
                self.options.fx_tracer.input_adapter,
                self.options.fx_tracer.output_adapter,
                self.options.diagnostic_context,
                fake_context=self.options.fake_context,
            )

    def _assert_fake_tensor_mode(self):
        """Asserts that the model and its input do not contain fake tensors."""

        has_any_fake_tensor = pytree.tree_any(
            lambda x: isinstance(x, torch._subclasses.FakeTensor),
            (self.model_args, self.model_kwargs),
        )
        has_any_fake_param_or_buffer = False
        if isinstance(self.model, torch.nn.Module):
            has_any_fake_param_or_buffer = pytree.tree_any(
                lambda x: isinstance(x, torch._subclasses.FakeTensor),
                (self.model.parameters(), self.model.buffers()),
            )
        if (
            has_any_fake_tensor or has_any_fake_param_or_buffer
        ) and not self.options.fake_context:
            raise RuntimeError(
                "Cannot export a model with fake inputs/weights without enabling fake mode.",
            )
        has_any_non_fake_tensors = pytree.tree_any(
            lambda x: isinstance(x, torch.Tensor)
            and not isinstance(x, torch._subclasses.FakeTensor),
            (self.model_args, self.model_kwargs),
        )
        has_any_non_fake_param_or_buffer = False
        if isinstance(self.model, torch.nn.Module):
            has_any_non_fake_param_or_buffer = pytree.tree_any(
                lambda x: isinstance(x, torch.Tensor)
                and not isinstance(x, torch._subclasses.FakeTensor),
                (self.model.parameters(), self.model.buffers()),
            )
        if (
            has_any_non_fake_tensors or has_any_non_fake_param_or_buffer
        ) and self.options.fake_context:
            raise RuntimeError(
                "Cannot export a model with non fake inputs/weights and enabled fake mode.",
            )


class UnsatisfiedDependencyError(RuntimeError):
    """Raised when an ONNX exporter dependency cannot be satisfied."""

    def __init__(self, package_name: str, message: str):
        super().__init__(message)
        self.package_name = package_name


class OnnxExporterError(RuntimeError):
    """Raised when an ONNX exporter error occurs.

    This exception is thrown when there's an error during the ONNX export process.
    It encapsulates the `ExportOutput` object generated until the failure, allowing
    access to the partial export results and associated metadata.
    """

    export_output: Final[ExportOutput]

    def __init__(self, export_output: ExportOutput, message: str):
        """
        Initializes the OnnxExporterError with the given export output and message.

        Args:
            export_output (ExportOutput): The partial results of the ONNX export.
            message (str): The error message to be displayed.
        """
        super().__init__(message)
        self.export_output = export_output


@_beartype.beartype
def _assert_dependencies(export_options: ResolvedExportOptions):
    opset_version = export_options.onnx_registry.opset_version

    def missing_package(package_name: str, exc_info: logging._ExcInfoType):
        message = (
            f"Please install the `{package_name}` package "
            f"(e.g. `python -m pip install {package_name}`)."
        )
        log.fatal(message, exc_info=exc_info)
        return UnsatisfiedDependencyError(package_name, message)

    def missing_opset(package_name: str):
        message = (
            f"The installed `{package_name}` does not support the specified ONNX opset "
            f"version {opset_version}. Install a newer `{package_name}` package or "
            f"specify an older opset version."
        )
        log.fatal(message)
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
                dynamic_shapes=True,
            )
        ).save("my_model.onnx")
    """

    resolved_export_options = (
        export_options
        if isinstance(export_options, ResolvedExportOptions)
        else ResolvedExportOptions(export_options)
    )

    _assert_dependencies(resolved_export_options)

    try:
        return Exporter(
            options=resolved_export_options,
            model=model,
            model_args=model_args,
            model_kwargs=model_kwargs,
        ).export()
    except Exception as e:
        sarif_report_path = _DEFAULT_FAILED_EXPORT_SARIF_LOG_PATH
        resolved_export_options.diagnostic_context.dump(sarif_report_path)
        message = (
            "Failed to export the model to ONNX. Generating SARIF report at {sarif_report_path}. "
            "SARIF is a standard format for the output of static analysis tools. "
            "SARIF log can be loaded in VS Code SARIF viewer extension, "
            "or SARIF web viewer(https://microsoft.github.io/sarif-web-component/)."
            f"Please report a bug on PyTorch Github: {_PYTORCH_GITHUB_ISSUES_URL}"
        )
        raise OnnxExporterError(
            ExportOutput._from_failure(e, resolved_export_options.diagnostic_context),
            message,
        ) from e


@_beartype.beartype
def pre_export_passes(
    options: ResolvedExportOptions,
    original_model: Union[torch.nn.Module, Callable],
    fx_module: torch.fx.GraphModule,
    fx_module_args: Sequence[Any],
):
    # TODO: Import here to prevent circular dependency
    from torch.onnx._internal.fx import analysis, passes

    diagnostic_context = options.diagnostic_context

    # Apply decomposition table to the input graph.
    module = passes.Decompose(
        diagnostic_context,
        fx_module,
        options.decomposition_table,
        enable_dynamic_axes=options.dynamic_shapes,
        allow_fake_constant=options.fake_context is not None,
    ).run(*fx_module_args)

    # ONNX does not support views and mutations.
    # Functionalize to get a semantically equivalent graph without mutations.
    module = passes.Functionalize(
        diagnostic_context,
        module,
        enable_dynamic_axes=options.dynamic_shapes,
        allow_fake_constant=options.fake_context is not None,
    ).run(*fx_module_args)

    # Input mutations are detected and distilled after `Functionalize` pass.
    # Remove them since ONNX inference does not need them.
    module = passes.RemoveInputMutation(diagnostic_context, module).run(*fx_module_args)

    # ONNX does not support concept of (implicit) type promotion.
    # Insert type casts explicitly where needed.
    module = passes.InsertTypePromotion(diagnostic_context, module).run()

    analysis.UnsupportedFxNodesAnalysis(
        diagnostic_context, module, options.onnxfunction_dispatcher
    ).analyze(infra.levels.ERROR)

    if isinstance(original_model, torch.nn.Module):
        module = passes.RestoreParameterAndBufferNames(
            diagnostic_context, module, original_model
        ).run()

    # This operation should be invoked as the last pre export pass.
    # See [NOTE: Modularize pass ordering]
    module = passes.Modularize(diagnostic_context, module).run()

    # ONNX does not support None inputs. During graph building, all None inputs
    # are removed. Here we register this step to input adapter.
    options.fx_tracer.input_adapter.append_step(io_adapter.RemoveNoneInputStep())

    # NOTE: temp workaround for https://github.com/pytorch/pytorch/issues/99534
    # Dynamo doesn't support non-tensor inputs.
    options.fx_tracer.input_adapter.append_step(io_adapter.RemoveNonTensorInputStep())

    # ONNX does not support complex inputs. During graph building, all complex inputs
    # are converted to real representation inputs. Here we register this step to
    # input/output adapter.
    options.fx_tracer.input_adapter.append_step(
        io_adapter.ConvertComplexToRealRepresentationInputStep()
    )

    # ONNX can't represent collection types (e.g., dictionary, tuple of tuple of
    # tensor, etc), we flatten the collection and register each element as output.
    options.fx_tracer.output_adapter.append_step(io_adapter.FlattenOutputStep())

    # Output post-processing steps should happen after `FlattenOutputStep`.
    options.fx_tracer.output_adapter.append_step(
        io_adapter.ConvertComplexToRealRepresentationOutputStep()
    )

    return module


__all__ = [
    "ExportOptions",
    "ExportOutput",
    "ExportOutputSerializer",
    "UnsatisfiedDependencyError",
    "dynamo_export",
    "OnnxExporterError",
    "enable_fake_mode",
    "OnnxRegistry",
    "DiagnosticOptions",
]
