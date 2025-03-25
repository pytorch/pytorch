# mypy: allow-untyped-defs
from __future__ import annotations


__all__ = [
    "ExportOptions",
    "ONNXRuntimeOptions",
    "OnnxRegistry",
    "enable_fake_mode",
]


import abc
import contextlib
import dataclasses
import logging
import warnings
from collections import defaultdict
from typing import Any, Callable, TYPE_CHECKING
from typing_extensions import deprecated

import torch
import torch._ops
from torch.onnx._internal import io_adapter
from torch.onnx._internal._lazy_import import onnxscript_apis
from torch.onnx._internal.exporter import _constants
from torch.onnx._internal.fx import (
    decomposition_table,
    patcher as patcher,
    registration,
)


# We can only import onnx from this module in a type-checking context to ensure that
# 'import torch.onnx' continues to work without having 'onnx' installed. We fully
# 'import onnx' inside of dynamo_export (by way of _assert_dependencies).
if TYPE_CHECKING:
    import io
    from collections.abc import Mapping, Sequence

    import onnxruntime
    import onnxscript

    from torch._subclasses import fake_tensor

log = logging.getLogger(__name__)


@dataclasses.dataclass
class ONNXFakeContext:
    """A dataclass used to store context for model export using FakeTensor.

    This dataclass stores the FakeTensorMode instance used to convert
    real tensors and model parameters into fake tensors. This :attr:`ONNXFakeContext.fake_mode` is
    reused internally during tracing of a :class:`torch.nn.Module` into a FX :class:`GraphModule`.
    """

    fake_mode: fake_tensor.FakeTensorMode
    """The fake tensor mode used for tracing model using fake tensors and parameters."""

    state_dict_paths: tuple[str | io.BytesIO | dict[str, Any]] | None = None
    """List of paths of files that contain the model :meth:`state_dict`"""


@deprecated(
    "torch.onnx.dynamo_export is deprecated since 2.7.0. Please use torch.onnx.export(..., dynamo=True) instead.",
)
class OnnxRegistry:
    """Registry for ONNX functions.

    .. deprecated:: 2.7
        Please use ``torch.onnx.export(..., dynamo=True)`` instead.

    The registry maintains a mapping from qualified names to symbolic functions under a
    fixed opset version. It supports registering custom onnx-script functions and for
    dispatcher to dispatch calls to the appropriate function.

    """

    def __init__(self) -> None:
        """Initializes the registry"""

        # NOTE: _registry is the registry maps OpNameto a list of ONNXFunctions. It is important
        # not to directly modify this variable. Instead, access to it should be done through
        # the public methods: register_custom_op, get_ops, and is_registered_op.
        self._registry: dict[registration.OpName, list[registration.ONNXFunction]] = (
            defaultdict(list)
        )

        self._opset_version = _constants.TORCHLIB_OPSET
        warnings.warn(
            f"torch.onnx.dynamo_export only implements opset version {self._opset_version} for now. If you need to use a "
            "different opset version, please register them with register_custom_op."
        )

        self._initiate_registry_from_torchlib()

    @property
    def opset_version(self) -> int:
        """The ONNX opset version the exporter should target."""

        return self._opset_version

    def _initiate_registry_from_torchlib(self) -> None:
        """Populates the registry with ATen functions from torchlib.

        Args:
            torchlib_registry: The torchlib registry to use for populating the registry.
        """
        for meta in onnxscript_apis.get_torchlib_ops():
            internal_name_instance = registration.OpName.from_qualified_name(
                meta.qualified_name
            )
            symbolic_function = registration.ONNXFunction(
                onnx_function=meta.function,  # type: ignore[arg-type]
                op_full_name=internal_name_instance.qualified_name(),
                is_custom=False,
                is_complex=meta.is_complex,
            )
            self._register(internal_name_instance, symbolic_function)

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

    def register_op(
        self,
        function: onnxscript.OnnxFunction | onnxscript.TracedOnnxFunction,
        namespace: str,
        op_name: str,
        overload: str | None = None,
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

    def get_op_functions(
        self, namespace: str, op_name: str, overload: str | None = None
    ) -> list[registration.ONNXFunction] | None:
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

    def is_registered_op(
        self, namespace: str, op_name: str, overload: str | None = None
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

    def _all_registered_ops(self) -> set[str]:
        """Returns the set of all registered function names."""
        return {
            op_name_class.qualified_name() for op_name_class in self._registry.keys()
        }


@deprecated(
    "torch.onnx.dynamo_export is deprecated since 2.7.0. Please use torch.onnx.export(..., dynamo=True) instead.",
    category=None,
)
class ExportOptions:
    """Options to influence the TorchDynamo ONNX exporter.

    .. deprecated:: 2.7
        Please use ``torch.onnx.export(..., dynamo=True)`` instead.

    Attributes:
        dynamic_shapes: Shape information hint for input/output tensors.
            When ``None``, the exporter determines the most compatible setting.
            When ``True``, all input shapes are considered dynamic.
            When ``False``, all input shapes are considered static.
        fake_context: The fake context used for symbolic tracing.
        onnx_registry: The ONNX registry used to register ATen operators to ONNX functions.
    """

    def __init__(
        self,
        *,
        dynamic_shapes: bool | None = True,
        fake_context: ONNXFakeContext | None = None,
        onnx_registry: OnnxRegistry | None = None,
    ):
        self.dynamic_shapes = dynamic_shapes
        self.fake_context = fake_context
        self.onnx_registry = onnx_registry


@deprecated(
    "torch.onnx.dynamo_export is deprecated since 2.7.0. Please use torch.onnx.export(..., dynamo=True) instead.",
    category=None,
)
class ResolvedExportOptions(ExportOptions):
    """Consolidates :class:`ExportOptions` with default values.
    All unspecified options from :class:`ExportOptions` are assigned a default value.
    This is an internal class and its API may be changed at any time without notice.
    """

    def __init__(self):
        from torch.onnx._internal.fx import (
            dynamo_graph_extractor,
            onnxfunction_dispatcher,
        )

        self.dynamic_shapes: bool = True
        self.fx_tracer: dynamo_graph_extractor.DynamoExport = (
            dynamo_graph_extractor.DynamoExport()
        )
        self.fake_context = None
        self.onnx_registry: OnnxRegistry = OnnxRegistry()
        self.decomposition_table = (
            decomposition_table.create_onnx_friendly_decomposition_table(  # type: ignore[assignment]
                self.onnx_registry
            )
        )
        self.onnxfunction_dispatcher = onnxfunction_dispatcher.OnnxFunctionDispatcher(
            self.onnx_registry,
        )


@contextlib.contextmanager
def enable_fake_mode():
    """Enable fake mode for the duration of the context.

    Internally it instantiates a :class:`torch._subclasses.fake_tensor.FakeTensorMode` context manager
    that converts user input and model parameters into :class:`torch._subclasses.fake_tensor.FakeTensor`.

    A :class:`torch._subclasses.fake_tensor.FakeTensor`
    is a :class:`torch.Tensor` with the ability to run PyTorch code without having to
    actually do computation through tensors allocated on a ``meta`` device. Because
    there is no actual data being allocated on the device, this API allows for
    initializing and exporting large models without the actual memory footprint needed for executing it.

    It is highly recommended to initialize the model in fake mode when exporting models that
    are too large to fit into memory.

    .. note::
        This function does not support torch.onnx.export(..., dynamo=True, optimize=True).
        Please call ONNXProgram.optimize() outside of the function after the model is exported.

    Example::

        # xdoctest: +REQUIRES(env:TORCH_DOCTEST_ONNX)
        >>> import torch
        >>> class MyModel(torch.nn.Module):  # Model with a parameter
        ...     def __init__(self) -> None:
        ...         super().__init__()
        ...         self.weight = torch.nn.Parameter(torch.tensor(42.0))
        ...     def forward(self, x):
        ...         return self.weight + x
        >>> with torch.onnx.enable_fake_mode():
        ...     # When initialized in fake mode, the model's parameters are fake tensors
        ...     # They do not take up memory so we can initialize large models
        ...     my_nn_module = MyModel()
        ...     arg1 = torch.randn(2, 2, 2)
        >>> onnx_program = torch.onnx.export(my_nn_module, (arg1,), dynamo=True, optimize=False)
        >>> # Saving model WITHOUT initializers (only the architecture)
        >>> onnx_program.save(
        ...     "my_model_without_initializers.onnx",
        ...     include_initializers=False,
        ...     keep_initializers_as_inputs=True,
        ... )
        >>> # Saving model WITH initializers after applying concrete weights
        >>> onnx_program.apply_weights({"weight": torch.tensor(42.0)})
        >>> onnx_program.save("my_model_with_initializers.onnx")

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


@deprecated(
    "torch.onnx.dynamo_export is deprecated since 2.7.0. Please use torch.onnx.export(..., dynamo=True) instead.",
)
class ONNXRuntimeOptions:
    """Options to influence the execution of the ONNX model through ONNX Runtime.

    .. deprecated:: 2.7
        Please use ``torch.onnx.export(..., dynamo=True)`` instead.

    Attributes:
        session_options: ONNX Runtime session options.
        execution_providers: ONNX Runtime execution providers to use during model execution.
        execution_provider_options: ONNX Runtime execution provider options.
    """

    session_options: Sequence[onnxruntime.SessionOptions] | None = None
    """ONNX Runtime session options."""

    execution_providers: Sequence[str | tuple[str, dict[Any, Any]]] | None = None
    """ONNX Runtime execution providers to use during model execution."""

    execution_provider_options: Sequence[dict[Any, Any]] | None = None
    """ONNX Runtime execution provider options."""

    def __init__(
        self,
        *,
        session_options: Sequence[onnxruntime.SessionOptions] | None = None,
        execution_providers: Sequence[str | tuple[str, dict[Any, Any]]] | None = None,
        execution_provider_options: Sequence[dict[Any, Any]] | None = None,
    ):
        self.session_options = session_options
        self.execution_providers = execution_providers
        self.execution_provider_options = execution_provider_options


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
        model: torch.nn.Module | Callable,
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

    # TODO: Design the passes API
    @abc.abstractmethod
    def pre_export_passes(
        self,
        options: ResolvedExportOptions,
        original_model: torch.nn.Module | Callable,
        fx_module: torch.fx.GraphModule,
        fx_module_args: Sequence[Any],
    ):
        """Applies pre-export passes to the FX graph.

        Pre-export passes are FX-to-FX graph transformations that make the graph
        more palatable for the FX-to-ONNX conversion.
        For example, it can be used to flatten model input/output, add explicit
        casts to the graph, replace/decompose operators, functionalize the graph, etc.
        """
        ...


def common_pre_export_passes(
    options: ResolvedExportOptions,
    original_model: torch.nn.Module | Callable,
    fx_module: torch.fx.GraphModule,
    fx_module_args: Sequence[Any],
):
    # TODO: Import here to prevent circular dependency
    from torch.onnx._internal.fx import passes

    # Apply decomposition table to the input graph.
    module = passes.Decompose(
        fx_module,
        options.decomposition_table,  # type: ignore[arg-type]
        enable_dynamic_axes=options.dynamic_shapes,
        allow_fake_constant=options.fake_context is not None,
    ).run(*fx_module_args)

    # ONNX does not support views and mutations.
    # Functionalize to get a semantically equivalent graph without mutations.
    module = passes.Functionalize(
        module,
        enable_dynamic_axes=options.dynamic_shapes,
        allow_fake_constant=options.fake_context is not None,
    ).run(*fx_module_args)

    # Input mutations are detected and distilled after `Functionalize` pass.
    # Remove them since ONNX inference does not need them.
    module = passes.RemoveInputMutation(module).run(*fx_module_args)

    # ONNX does not support concept of (implicit) type promotion.
    # Insert type casts explicitly where needed.
    module = passes.InsertTypePromotion(module).run()

    if isinstance(original_model, torch.nn.Module):
        module = passes.RestoreParameterAndBufferNames(module, original_model).run()

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
