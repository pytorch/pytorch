from __future__ import annotations

import copy
import functools
import inspect
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Type, Union

import torch._dynamo
import torch.fx
import torch.onnx
from torch.onnx._internal import _beartype
from torch.onnx._internal.fx import fx_exporter
from torch.utils import _pytree as pytree


class DynamoOptimizeExporter(fx_exporter.FXGraphModuleExporter):
    def export(self) -> torch.onnx.ExportOutput:
        # Fill in default values for optional args and kwargs. The goal is to preserve
        # them as inputs in `dynamo.optimize` produced FX graph. Otherwise, they will
        # be traced as constants.
        _, named_args = self._apply_input_format_step(
            fx_exporter.BindInputStep,
            self.model_args,
            self.model_kwargs,
            step_init_args=(self.model_signature,),
        )
        model_args, _ = self._apply_input_format_step(
            fx_exporter.MergeKwargsIntoArgsStep,
            [],
            named_args,
        )

        # args will be converted to symbolic tensor. Let's copy to avoid side effects.
        model_args = copy.deepcopy(model_args)
        # Translate callable to FX graph.
        #
        # TODO(wechi): There are several symbolic tracing mechanisms to convert
        # nn.Module to FX graph. We should choose the right one after they are
        # matured.

        class GraphCaptureCompiler:
            def __init__(self):
                self.captured_graph: Optional[torch.fx.GraphModule] = None
                self.captured_graph_count = 0

            def compile(self, graph_module: torch.fx.GraphModule, _):
                assert self.captured_graph_count == 0
                self.captured_graph = graph_module
                self.captured_graph_count += 1
                return graph_module

        compiler = GraphCaptureCompiler()
        torch._dynamo.reset()
        # TODO(titaiwang): Set `dynamic` according to `self.options.dynamic_shapes`
        torch._dynamo.optimize(compiler.compile, nopython=True)(self.model)(*model_args)
        torch._dynamo.reset()
        assert compiler.captured_graph

        # Outputs are flattened by `dynamo.optimize`.
        # Apply and record this output format step.
        self._output_formatter.append_step(DynamoFlattenOutputStep())
        # TODO: `dynamo.optimize` does not capture non computation part of the graph.
        # Hence any tensor that is returned multiple times will only appear once
        # in the return statement of `dynamo.optimize` traced graph. A specialized
        # output formatter is required to map this gap between PyTorch model.

        # Export FX graph to ONNX ModelProto.
        #
        # Apply and record the following input format steps.
        # `args` and `kwargs` are merged and flattened by `dynamo.optimize`.
        model_args, _ = self._apply_input_format_step(
            fx_exporter.FlattenInputWithTreeSpecValidationStep, model_args, {}
        )
        # `None` inputs are removed by `dynamo.optimize`.
        model_args, _ = self._apply_input_format_step(
            fx_exporter.RemoveNoneInputStep, model_args, {}
        )
        # TODO: needs additional input format step to remove unused arguments.
        # `dynamo.optimize` drops unused arguments.
        return self.export_fx_to_onnx(compiler.captured_graph, model_args)


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


class DynamoFlattenOutputStep(fx_exporter.FlattenOutputStep):
    """Flatten nested collection and custom python types and return a flat list of elements.

    Extended from :class:`fx_exporter.FlattenOutputStep` to support flattening arbitrary
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

    def format(self, model_outputs: Any) -> Sequence[Any]:
        """Flatten the model outputs, under the context of pytree extension."""
        with self._pytree_extension_context:
            return super().format(model_outputs)


def _wrap_model_with_output_formatter(
    model: Union[torch.nn.Module, Callable],
    output_formatter: DynamoFlattenOutputStep,
) -> Callable:
    """Wrap model with output formatter.

    This is a helper function to enable :func:`dynamo.export` on models that produce
    custom user defined types outputs. It wraps the model with an output formatter to
    convert the outputs to :func:`dynamo.export` compatible types, i.e. :class:`torch.Tensor`.

    The formatting logic is controlled by ``output_formatter``.

    Args:
        model: PyTorch model or function.
        output_formatter: Output formatter to apply to model output.
    Returns:
        Wrapped model.
    """
    model_func = model.forward if isinstance(model, torch.nn.Module) else model

    # Preserve original function signature.
    @functools.wraps(model_func)
    def wrapped(*args, **kwargs):
        return output_formatter.format(model(*args, **kwargs))

    return wrapped


class DynamoExporter(fx_exporter.FXGraphModuleExporter):
    def export(self) -> torch.onnx.ExportOutput:
        # args will be converted to symbolic tensor. Let's copy to avoid side effects.
        args = copy.deepcopy(self.model_args)
        kwargs = copy.deepcopy(self.model_kwargs)

        # `dynamo.export` does not recognize custom user defined classes as output type.
        # Apply wrapper to format the outputs back to `dynamo.export` compatible types,
        # i.e. :class:`torch.Tensor`.
        dynamo_flatten_output_step = DynamoFlattenOutputStep()
        wrapped_model = _wrap_model_with_output_formatter(
            self.model, dynamo_flatten_output_step
        )
        # Record the output formatter step.
        self._output_formatter.append_step(dynamo_flatten_output_step)

        # Translate callable to FX graph.
        #
        # TODO(wechi): There are several symbolic tracing mechanisms to convert
        # nn.Module to FX graph. We should choose the right one after they are
        # matured.
        # TODO(titaiwang): Set `tracing_mode` according to `self.options.dynamic_shapes`
        graph_module, graph_guard = torch._dynamo.export(
            wrapped_model, *args, aten_graph=True, **kwargs
        )
        del graph_guard  # Unused
        torch._dynamo.reset()

        # Export FX graph to ONNX ModelProto.
        #
        # `args` and `kwargs` are merged and flattened by `dynamo.export`.
        # Apply and record this input format step.
        merged_args, _ = self._apply_input_format_step(
            fx_exporter.FlattenInputWithTreeSpecValidationStep, args, kwargs
        )
        return self.export_fx_to_onnx(graph_module, merged_args)
