# mypy: allow-untyped-defs
# mypy: disable-error-code="attr-defined,name-defined"
from __future__ import annotations


__all__ = ["ONNXProgram"]

import contextlib
import copy
import gc
import logging
import os
import tempfile
import textwrap
import warnings
from typing import Any, Callable, TYPE_CHECKING

import torch
from torch.onnx._internal._lazy_import import onnx, onnxscript_apis, onnxscript_ir as ir
from torch.onnx._internal.exporter import _dynamic_shapes, _ir_passes
from torch.utils import _pytree


# NOTE: DO NOT import module from torch.onnx._internal to this module in the global scope
# because ONNXProgram is exposed to the public API

if TYPE_CHECKING:
    from collections.abc import Sequence

    import onnxruntime as ort

_LARGE_MODEL_THRESHOLD = 1536 * 1024 * 1024  # 1536MB
_NP_UNSUPPORTED_DTYPES_8BIT = frozenset(
    {
        torch.float8_e4m3fn,
        torch.float8_e4m3fnuz,
        torch.float8_e5m2,
        torch.float8_e5m2fnuz,
    }
)

logger = logging.getLogger(__name__)


def _ort_session_initializer(model: str | bytes) -> ort.InferenceSession:
    """Initialize an ONNX Runtime inference session with the specified model."""
    import onnxruntime as ort

    session_options = ort.SessionOptions()
    session_options.log_severity_level = 3  # 3: Error
    possible_providers = (
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    )
    available_providers = set(ort.get_available_providers())
    providers = [
        provider for provider in possible_providers if provider in available_providers
    ]
    return ort.InferenceSession(
        model, providers=providers, sess_options=session_options
    )


def _count_initializer_size(graph: ir.Graph) -> int:
    """Count the total size of the initializers in bytes."""
    return sum(
        v.const_value.nbytes
        for v in graph.initializers.values()
        if v.const_value is not None
    )


@contextlib.contextmanager
def _set_graph_outputs(
    graph: ir.Graph,
    outputs: list[ir.Value],
):
    """Temporarily set the outputs of the graph.

    Args:
        graph: The graph to set the outputs for.
        outputs: The outputs to set.
    """
    original_outputs = graph.outputs.copy()
    graph.outputs.clear()
    graph.outputs.extend(outputs)
    try:
        yield
    finally:
        graph.outputs.clear()
        graph.outputs.extend(original_outputs)


def _create_value_mapping(graph: ir.Graph) -> dict[str, ir.Value]:
    """Return a dictionary mapping names to values in the graph.

    The mapping does not include values from subgraphs.

    Args:
        graph: The graph to extract the mapping from.

    Returns:
        A dictionary mapping names to values.
    """
    values = {}
    values.update(graph.initializers)
    # The names of the values can be None or "", which we need to exclude
    for input in graph.inputs:
        if not input.name:
            continue
        values[input.name] = input
    for node in graph:
        for value in node.outputs:
            if not value.name:
                continue
            values[value.name] = value
    return values


def _to_ort_value(tensor: torch.Tensor) -> ort.OrtValue:
    """Convert a PyTorch tensor to an ONNX Runtime OrtValue."""
    import onnxruntime as ort

    from torch.onnx._internal.exporter import _core

    if tensor.dtype == torch.bfloat16 or tensor.dtype in _NP_UNSUPPORTED_DTYPES_8BIT:
        if hasattr(ort.OrtValue, "ortvalue_from_numpy_with_onnx_type"):
            # This requires ONNX Runtime 1.21 or newer
            if tensor.dtype == torch.bfloat16:
                uint_type = torch.uint16
            else:
                uint_type = torch.uint8
            onnx_type = _core.torch_dtype_to_onnx_dtype(tensor.dtype)
            # Make tensor contiguous to ensure view() works
            tensor = tensor.contiguous()
            return ort.OrtValue.ortvalue_from_numpy_with_onnx_type(
                tensor.view(uint_type).numpy(force=True), onnx_element_type=onnx_type
            )
        raise RuntimeError(
            f"Failed to convert tensor of type '{tensor.dtype}' to OrtValue. "
            "Please ensure that ONNX Runtime is built with DLPack support or is the latest version"
        )
    # TODO(#151064): Use dlpack when ORT properly supports it
    return ort.OrtValue.ortvalue_from_numpy(tensor.numpy(force=True))


def _from_ort_value(value: ort.OrtValue) -> torch.Tensor:
    if value.element_type() in (
        ir.DataType.BFLOAT16,
        ir.DataType.FLOAT8E4M3FN,
        ir.DataType.FLOAT8E4M3FNUZ,
        ir.DataType.FLOAT8E5M2,
        ir.DataType.FLOAT8E5M2FNUZ,
    ):
        # This requires ONNX Runtime 1.21 or newer
        try:
            return torch.from_dlpack(value._get_c_value())
        except Exception as e:
            raise RuntimeError(
                "Failed to convert OrtValue to torch.Tensor. "
                "Please ensure that ONNX Runtime is built with DLPack support or is the latest version"
            ) from e
    return torch.from_numpy(value.numpy())


class ONNXProgram:
    """A class to represent an ONNX program that is callable with torch tensors.

    Attributes:
        model: The ONNX model as an ONNX IR model object.
        exported_program: The exported program that produced the ONNX model.
    """

    def __init__(
        self, model: ir.Model, exported_program: torch.export.ExportedProgram | None
    ):
        """Initialize the ONNX program with the specified model and exported program.
        Args:
            model: The ONNX model.
            exported_program: The exported program that produced the ONNX model. Optional.
        """
        self.model: ir.Model = model
        self.exported_program = exported_program
        self._inference_session: ort.InferenceSession | None = None
        self._tempdir: tempfile.TemporaryDirectory | None = None
        # Strategy used to capture the exported program
        self._capture_strategy: str | None = None

    def __repr__(self) -> str:
        return f"""\
ONNXProgram(
    model=
{textwrap.indent(str(self.model), " " * 8)}
    ,
    exported_program=
{textwrap.indent(str(self.exported_program), " " * 8)}
)
"""

    def __call__(self, *args, **kwargs) -> Sequence[torch.Tensor]:
        """Run the ONNX model with the same arguments you would provide to the GraphModule."""
        import onnxruntime as ort

        flatten_args = _process_args(args, kwargs)

        if self._inference_session is None:
            self.initialize_inference_session()

        assert self._inference_session is not None

        # We don't expect non-tensor as inputs
        ort_input = {
            k.name: _to_ort_value(v)
            for k, v in zip(self.model.graph.inputs, flatten_args)
        }
        run_options = ort.RunOptions()
        run_options.log_severity_level = 3  # 3: Error
        logger.debug("Running the inference session with %s arguments.", len(ort_input))
        outputs = self._inference_session.run_with_ort_values(
            None, ort_input, run_options=run_options
        )
        logger.debug("Inference session run completed.")
        return tuple(_from_ort_value(output) for output in outputs)

    def compute_values(
        self, value_names: Sequence[str], args=(), kwargs=None
    ) -> Sequence[torch.Tensor]:
        """Compute the values of the specified names in the ONNX model.

        This method is used to compute the values of the specified names in the ONNX model.
        The values are returned as a dictionary mapping names to tensors.

        Args:
            value_names: The names of the values to compute.

        Returns:
            A dictionary mapping names to tensors.
        """
        if kwargs is None:
            kwargs = {}
        self.release()
        values = _create_value_mapping(self.model.graph)
        for name in value_names:
            if name not in values:
                raise ValueError(
                    f"Value '{name}' not found in the model. "
                    "Please provide a valid value name."
                )
        temporary_outputs = [values[name] for name in value_names]
        with _set_graph_outputs(self.model.graph, temporary_outputs):
            try:
                result = self(*args, **kwargs)
            finally:
                self.release()
        return result

    @property
    def model_proto(self) -> onnx.ModelProto:
        """Return the ONNX ``ModelProto`` object."""
        return ir.serde.serialize_model(self.model)

    def optimize(self) -> None:
        """Optimize the ONNX model.

        This method optimizes the ONNX model by performing constant folding and
        eliminating redundancies in the graph. The optimization is done in-place.
        """
        self.model = onnxscript_apis.optimize(self.model)

    def save(
        self,
        destination: str | os.PathLike,
        *,
        include_initializers: bool = True,
        keep_initializers_as_inputs: bool = False,
        external_data: bool | None = None,
    ):
        """Save the ONNX model to the specified destination.

        When ``external_data`` is ``True`` or the model is larger than 2GB,
        the weights are saved as external data in a separate file.

        Initializer (model weights) serialization behaviors:

        * ``include_initializers=True``, ``keep_initializers_as_inputs=False`` (default):
          The initializers are included in the saved model.
        * ``include_initializers=True``, ``keep_initializers_as_inputs=True``:
          The initializers are included in the saved model and kept as model inputs.
          Choose this option if you want the ability to override the model weights
          during inference.
        * ``include_initializers=False``, ``keep_initializers_as_inputs=False``:
          The initializers are not included in the saved model and are not listed
          as model inputs. Choose this option if you want to attach the initializers
          to the ONNX model in a separate, post-processing, step.
        * ``include_initializers=False``, ``keep_initializers_as_inputs=True``:
          The initializers are not included in the saved model but are listed as model
          inputs. Choose this option if you want to supply the initializers during
          inference and want to minimize the size of the saved model.

        Args:
            destination: The path to save the ONNX model to.
            include_initializers: Whether to include the initializers in the saved model.
            keep_initializers_as_inputs: Whether to keep the initializers as inputs in the saved model.
                If `True`, the initializers are added as inputs to the model which means they can be overwritten.
                by providing the initializers as model inputs.
            external_data: Whether to save the weights as external data in a separate file.

        Raises:
            TypeError: If ``external_data`` is ``True`` and ``destination`` is not a file path.
        """
        original_initializers = copy.copy(self.model.graph.initializers)
        original_inputs = copy.copy(self.model.graph.inputs)

        # Adjust the model based on options
        if not include_initializers:
            self.model.graph.initializers.clear()
        if keep_initializers_as_inputs:
            self.model.graph.inputs.extend(original_initializers.values())  # type: ignore[arg-type]

        try:
            # Save the model to disk
            if (
                external_data
                or _count_initializer_size(self.model.graph) > _LARGE_MODEL_THRESHOLD
            ):
                onnxscript_apis.save_model_with_external_data(self.model, destination)
            else:
                ir.save(self.model, destination)
        finally:
            # Revert the changes to the model
            if not include_initializers:
                self.model.graph.initializers.update(original_initializers)
            if keep_initializers_as_inputs:
                self.model.graph.inputs.clear()
                self.model.graph.inputs.extend(original_inputs)

    def apply_weights(self, state_dict: dict[str, torch.Tensor]) -> None:
        """Apply the weights from the specified state dict to the ONNX model.

        Use this method to replace FakeTensors or other weights.

        Args:
            state_dict: The state dict containing the weights to apply to the ONNX model.
        """
        from torch.onnx._internal.exporter import _core

        for name, tensor in state_dict.items():
            if name in self.model.graph.initializers:
                self.model.graph.initializers[name].const_value = _core.TorchTensor(
                    tensor, name
                )
            else:
                warnings.warn(
                    f"Weight '{name}' not found in the model. Skipped applying.",
                    category=torch.onnx.errors.OnnxExporterWarning,
                    stacklevel=1,
                )

    def initialize_inference_session(
        self,
        initializer: Callable[
            [str | bytes], ort.InferenceSession
        ] = _ort_session_initializer,
    ) -> None:
        """Initialize the ONNX Runtime inference session.

        Args:
            initializer: The function to initialize the ONNX Runtime inference
                session with the specified model. By default, it uses the
                :func:`_ort_session_initializer` function.
        """
        # TODO(justinchuby): Allow different inference options
        logger.debug("Initializing the inference session.")
        if (
            byte_size := _count_initializer_size(self.model.graph)
        ) > _LARGE_MODEL_THRESHOLD:
            logger.debug("The model initializers is larger than 1.5GB (%s).", byte_size)
            # Save the model to a temporary file if too large
            self._tempdir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
            model_path = os.path.join(self._tempdir.name, "model.onnx")
            self.save(model_path, external_data=True)
            model = model_path
        else:
            model = self.model_proto.SerializeToString()  # type: ignore[assignment]

        self._inference_session = initializer(model)
        logger.debug("Inference session initialized.")

    def release(self) -> None:
        """Release the inference session.

        You may call this method to release the resources used by the inference session.
        """
        # Release the inference session first so that the model file can be deleted
        if self._inference_session is not None:
            self._inference_session = None
        gc.collect()
        if self._tempdir is not None:
            self._tempdir.cleanup()
            self._tempdir = None

    def _rename_dynamic_axes(
        self,
        dynamic_shapes: dict[str, Any] | tuple[Any, ...] | list[Any],
    ) -> None:
        """Rename dynamic axes in a model according to the specified dynamic_axes names."""
        rename_mapping = _dynamic_shapes.create_rename_mapping(
            self.model.graph.inputs, dynamic_shapes
        )
        _ir_passes.rename_axis(self.model, rename_mapping)


def _process_args(args, kwargs) -> tuple[torch.Tensor, ...]:
    """Process input arguments for the ONNX model."""
    args = _flatten_inputs(args, kwargs)
    args = _remove_none_from_inputs(args)
    args = _remove_non_tensor(args)
    args = _convert_complex_to_real_representation(args)
    return args


def _flatten_inputs(model_args, model_kwargs):
    flattened_args, _ = _pytree.tree_flatten((model_args, model_kwargs))
    return flattened_args


def _remove_none_from_inputs(model_args):
    return tuple(arg for arg in model_args if arg is not None)


def _remove_non_tensor(model_args):
    """Remove the non-tensor input arguments.

    Dynamo does not support non-tensor input arguments (https://github.com/pytorch/pytorch/issues/99534).

    Specifically, it does put the input into graph with an empty node, but consumed by no ones.
    The concrete value is embedded into the graph as a constant arg of a target node. Meta
    suggests in this case that one should rewrite the model code to make it tensor if the
    input value is supposed to change at runtime. We might need to further investigate
    the feasibility of that suggestion.

    For example,

        def func(x, b=1.0):
            y = x + b
            z = y.relu()
            return (y, z)

        x = torch.randn(1, 1, 2, dtype=torch.float32)
        gm_fun, _ = dynamo.export(func, x, b=8.0, aten_graph=True, tracing_mode="real")

        # class GraphModule(torch.nn.Module):
        #     def forward(self, x, b):
        #         arg0: f32[1, 1, 2], arg1, = fx_pytree.tree_flatten_spec(([x, b], {}), self._in_spec)
        #         # File: path/to/pytorch/test_constant_input.py:5, code: y = x + b
        #         add_tensor: f32[1, 1, 2] = torch.ops.aten.add.Tensor(arg0, 8.0);  arg0 = None

        #         # File: path/to/pytorch/test_constant_input.py:6, code: z = y.relu()
        #         relu_default: f32[1, 1, 2] = torch.ops.aten.relu.default(add_tensor)
        #         return pytree.tree_unflatten([add_tensor, relu_default], self._out_spec)

    Empty torch.fx.Node input leading to a mismatched number of input with PyTorch, as
    it's ignored in ONNX graph. Thus, we delete the useless input here.

    """

    return tuple(
        arg for arg in model_args if not isinstance(arg, (int, float, bool, str))
    )


def _convert_complex_to_real_representation(model_args):
    """Convert complex dtype tensors to real representation tensors.

    ONNX does not support complex dtype tensors. Thus, we convert complex dtype tensors
    to real representation tensors (i.e., float dtype tensors with an extra dimension
    representing the real and imaginary parts of the complex number).
    """
    return tuple(
        torch.view_as_real(arg.resolve_conj())
        if isinstance(arg, torch.Tensor) and arg.is_complex()
        else arg
        for arg in model_args
    )
