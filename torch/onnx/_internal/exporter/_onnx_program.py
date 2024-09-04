# mypy: allow-untyped-defs
# mypy: disable-error-code="attr-defined,name-defined"
from __future__ import annotations


__all__ = ["ONNXProgram"]

import gc
import logging
import os
import pathlib
import tempfile
import textwrap
from typing import Callable, IO, Sequence, TYPE_CHECKING

import torch
from torch.onnx._internal import _lazy_import
from torch.utils import _pytree as pytree


onnx = _lazy_import.onnx
ir = _lazy_import.onnxscript_ir


if TYPE_CHECKING:
    import onnxruntime as ort

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


class ONNXProgram:
    """A class to represent an ONNX program that is callable with torch tensors."""

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

    def __repr__(self) -> str:
        return f"""\
ONNXProgram(
    model=
{textwrap.indent(str(self.model), ' ' * 8)}
    ,
    exported_program=
{textwrap.indent(str(self.exported_program), ' ' * 8)}
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
            k.name: v.numpy(force=True)
            for k, v in zip(self.model.graph.inputs, flatten_args)
        }
        run_options = ort.RunOptions()
        run_options.log_severity_level = 3  # 3: Error
        logger.debug("Running the inference session with %s arguments.", len(ort_input))
        outputs = self._inference_session.run(None, ort_input, run_options=run_options)
        logger.debug("Inference session run completed.")
        # TODO(justinchuby): Maybe output complex tensors as needed
        return tuple(torch.from_numpy(output) for output in outputs)

    @property
    def model_proto(self) -> onnx.ModelProto:
        """Compatibility property for `torch.onnx.ONNXProgram.model_proto`."""
        return ir.serde.serialize_model(self.model)

    def save(
        self,
        destination: str | os.PathLike | IO[bytes],
        *,
        include_initializers: bool = True,
        keep_initializers_as_inputs: bool = False,
        external_data: bool | None = None,
        **_,
    ):
        """Save the ONNX model to the specified destination.

        When `external_data` is `True` or the model is larger than 2GB,
        the weights are saved as external data in a separate file.

        Args:
            destination: The path to save the ONNX model to.
            include_initializers: Whether to include the initializers in the saved model.
            keep_initializers_as_inputs: Whether to keep the initializers as inputs in the saved model.
                If `True`, the initializers are added as inputs to the model which means they can be overwritten.
                by providing the initializers as model inputs.
            external_data: Whether to save the weights as external data in a separate file.

        Raises:
            TypeError: If `external_data` is `True` and `destination` is not a file path.
        """
        if not include_initializers:
            self.model.graph.initializers.clear()
            logger.warning(
                "The initializers have been removed from the model. This is destructive. "
                "Developers: Please implement ir.Model copy() and remove initializers on the copied model."
            )
        if keep_initializers_as_inputs:
            self.model.graph.inputs.extend(self.model.graph.initializers.values())  # type: ignore[arg-type]
            logger.warning(
                "The initializers have been added as inputs to the model. This is destructive. "
                "Developers: Please implement ir.Model copy() and remove initializers on the copied model."
            )
        proto = ir.serde.serialize_model(self.model)
        byte_size = proto.ByteSize()
        model_too_large = (byte_size) >= 1 << 31
        if external_data or model_too_large:
            # TODO: Create an IR pass to handle external tensors conversion
            if model_too_large:
                logger.warning(
                    "The serialized ONNX model is larger than 2GB (%s). "
                    "Saving the weights as external data in a separate file.",
                    byte_size,
                )
            if not isinstance(destination, (str, os.PathLike)):
                raise TypeError(
                    "Saving the weights as external data is only supported when destination is a file path"
                )
            destination_path = pathlib.Path(destination)
            # Create the directory if it does not exist
            data_path = f"{destination_path.name}.data"
            onnx.save_model(
                proto,
                destination,
                save_as_external_data=True,
                location=data_path,
            )
        else:
            onnx.save_model(proto, destination)

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
        proto = ir.serde.serialize_model(self.model)
        byte_size = proto.ByteSize()
        model_too_large = (byte_size) >= 1 << 31

        if model_too_large:
            logger.debug(
                "The serialized ONNX model is larger than 2GB (%s).", byte_size
            )
            # Save the model to a temporary file if too large
            self._tempdir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
            model_path = os.path.join(self._tempdir.name, "model.onnx")
            data_path = "model.onnx.data"
            onnx.save_model(
                proto,
                model_path,
                save_as_external_data=True,
                location=data_path,
            )
            model = model_path
        else:
            model = proto.SerializeToString()  # type: ignore[assignment]

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


def _process_args(args, kwargs) -> tuple[torch.Tensor, ...]:
    """Process input arguments for the ONNX model."""
    args = _flatten_inputs(args, kwargs)
    args = _remove_none_from_inputs(args)
    args = _remove_non_tensor(args)
    args = _convert_complex_to_real_representation(args)
    return args


def _flatten_inputs(model_args, model_kwargs):
    flattened_args, _ = pytree.tree_flatten((model_args, model_kwargs))
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
