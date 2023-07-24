# NOTE: This file is referenced by name at
#       /opt/pytorch/torch/_dynamo/eval_frame.py::DONT_WRAP_FILES.
#       introduced by https://github.com/pytorch/pytorch/pull/98894.
#       If this file is renamed, moved, etc please update the reference there!

from __future__ import annotations

import functools
import inspect
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)

import torch._dynamo
import torch.fx
import torch.onnx
from torch.onnx._internal import _beartype, exporter, io_adapter
from torch.utils import _pytree as pytree


class _PyTreeExtensionContext:
    """Context manager to register PyTree extension."""

    _extensions: Dict[Type, Tuple[pytree.FlattenFunc, pytree.UnflattenFunc]]

    def __init__(self):
        self._extensions = {}
        # Register PyTree extension for HuggingFace model output.
        self._register_huggingface_model_output_extension()

    def __enter__(self):
        for class_type, (flatten_func, unflatten_func) in self._extensions.items():
            pytree._register_pytree_node(class_type, flatten_func, unflatten_func)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for class_type in self._extensions:
            pytree.SUPPORTED_NODES.pop(class_type)

    @_beartype.beartype
    def register_pytree_node(
        self,
        class_type: Type,
        flatten_func: pytree.FlattenFunc,
        unflatten_func: pytree.UnflattenFunc,
    ):
        """Register PyTree extension for a custom python type.

        Args:
            class_type: The custom python type.
            flatten_func: The flatten function.
            unflatten_func: The unflatten function.

        Raises:
            AssertionError: If the custom python type is already registered.
        """
        assert (
            class_type not in pytree.SUPPORTED_NODES
            and class_type not in self._extensions
        ), "PyTree node already registered"
        self._extensions[class_type] = (flatten_func, unflatten_func)

    def _register_huggingface_model_output_extension(self):
        try:
            from transformers import modeling_outputs  # type: ignore[import]
        except ImportError as e:
            return

        @_beartype.beartype
        def model_output_flatten(
            output: modeling_outputs.ModelOutput,
        ) -> Tuple[List[Any], pytree.Context]:
            return list(output.values()), (type(output), list(output.keys()))

        @_beartype.beartype
        def model_output_unflatten(
            values: List[Any], context: pytree.Context
        ) -> modeling_outputs.ModelOutput:
            output_type, keys = context
            return output_type(**dict(zip(keys, values)))

        # All 'ModelOutput' subclasses are defined under module 'modeling_outputs'.
        named_model_output_classes = inspect.getmembers(
            modeling_outputs,
            lambda x: inspect.isclass(x)
            and issubclass(x, modeling_outputs.ModelOutput),
        )

        for _, class_type in named_model_output_classes:
            self.register_pytree_node(
                class_type, model_output_flatten, model_output_unflatten
            )


class DynamoFlattenOutputStep(io_adapter.FlattenOutputStep):
    """Flatten nested collection and custom python types and return a flat list of elements.

    Extended from :class:`io_adapter.FlattenOutputStep` to support flattening arbitrary
    types via pytree extension. By default this supports many common user defined python
    types such as :class:`ModelOutput` from HuggingFace transformers.

    The pytree extension can be customized by passing in a ``_PyTreeExtensionContext``
    object. See :meth:`_PyTreeExtensionContext.register_pytree_node`.
    """

    def __init__(
        self, pytree_extension_context: Optional[_PyTreeExtensionContext] = None
    ):
        super().__init__()
        self._pytree_extension_context = (
            pytree_extension_context or _PyTreeExtensionContext()
        )

    def apply(self, model_outputs: Any) -> Sequence[Any]:
        """Flatten the model outputs, under the context of pytree extension."""
        with self._pytree_extension_context:
            return super().apply(model_outputs)


def _wrap_model_with_output_adapter(
    model: Union[torch.nn.Module, Callable],
    output_adapter: DynamoFlattenOutputStep,
) -> Callable:
    """Wrap model with output adapter.

    This is a helper function to enable :func:`dynamo.export` on models that produce
    custom user defined types outputs. It wraps the model with an output adapter to
    convert the outputs to :func:`dynamo.export` compatible types, i.e. :class:`torch.Tensor`.

    The adapting logic is controlled by ``output_adapter``.

    Args:
        model: PyTorch model or function.
        output_adapter: Output adapter to apply to model output.
    Returns:
        Wrapped model.
    """
    model_func = model.forward if isinstance(model, torch.nn.Module) else model

    # Preserve original function signature.
    @functools.wraps(model_func)
    def wrapped(*args, **kwargs):
        return output_adapter.apply(model_func(*args, **kwargs))

    return wrapped


class DynamoExport(exporter.FXGraphExtractor):
    """Generates a FX GraphModule using torch.dynamo.export API
    Args:
        aten_graph: If True, exports a graph with ATen operators.
                    If False, exports a graph with Python operators.
    """

    def __init__(
        self,
        aten_graph: Optional[bool] = None,
    ):
        super().__init__()
        self.aten_graph = aten_graph or True

    def generate_fx(
        self,
        options: exporter.ResolvedExportOptions,
        model: Union[torch.nn.Module, Callable],
        model_args: Sequence[Any],
        model_kwargs: Mapping[str, Any],
    ) -> torch.fx.GraphModule:
        # `dynamo.export` does not recognize custom user defined classes as output type.
        # Apply wrapper to adapt the outputs back to `dynamo.export` compatible types,
        # i.e. :class:`torch.Tensor`.
        dynamo_flatten_output_step = DynamoFlattenOutputStep()
        wrapped_model = _wrap_model_with_output_adapter(
            model, dynamo_flatten_output_step
        )
        # Record the output adapter step.
        self.output_adapter.append_step(dynamo_flatten_output_step)

        # Translate callable to FX graph.
        #
        fake_mode = options.fake_context.fake_mode if options.fake_context else None
        fx_mode = "symbolic" if options.dynamic_shapes else "fake"
        graph_module, graph_guard = torch._dynamo.export(
            wrapped_model,
            *model_args,
            tracing_mode=fx_mode,
            fake_mode=fake_mode,  # type: ignore[arg-type]
            aten_graph=fake_mode
            is not None,  # TODO: Tracked by https://github.com/pytorch/pytorch/issues/105467
            **model_kwargs,
        )
        del graph_guard  # Unused
        torch._dynamo.reset()

        # Export FX graph to ONNX ModelProto.
        self.input_adapter.append_step(
            io_adapter.FlattenInputWithTreeSpecValidationStep()
        )
        return graph_module  # type: ignore[return-value]
