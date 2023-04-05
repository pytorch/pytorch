from __future__ import annotations

import copy
import inspect
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Type

import torch._dynamo
import torch.fx
import torch.onnx
from torch.onnx._internal import _beartype
from torch.onnx._internal.fx import fx_exporter
from torch.utils import _pytree as pytree


class DynamoOptimizeExporter(fx_exporter.FXGraphModuleExporter):
    def export(self) -> torch.onnx.ExportOutput:
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
        traced_output = torch._dynamo.optimize(compiler.compile, nopython=True)(
            self.model
        )(*model_args)
        torch._dynamo.reset()

        self._apply_output_format_step(
            DynamoFlattenOutputWithTreeSpecValidationStep, traced_output
        )

        assert compiler.captured_graph
        # Export FX graph to ONNX ModelProto.
        model_args, _ = self._apply_input_format_step(
            fx_exporter.RemoveNoneInputStep, model_args, {}
        )

        return self.export_fx_to_onnx(
            compiler.captured_graph,
            model_args,
        )


class _PyTreeExtensionContext:
    _extensions: Dict[Type, Tuple[pytree.FlattenFunc, pytree.UnflattenFunc]]

    def __init__(self):
        self._extensions = {}
        # Register PyTree extension for HuggingFace model output.
        self._register_huggingface_model_output_extension()

    def __enter__(self):
        for class_type, (flatten_func, unflatten_func) in self._extensions.items():
            assert (
                class_type not in pytree.SUPPORTED_NODES
            ), "PyTree node already registered"
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


class DynamoFlattenOutputWithTreeSpecValidationStep(
    fx_exporter.FlattenOutputWithTreeSpecValidationStep
):
    def __init__(self):
        super().__init__()
        self._pytree_extension_context = _PyTreeExtensionContext()

    def format(self, model_outputs: Any) -> Sequence[Any]:
        with self._pytree_extension_context:
            return super().format(model_outputs)


class _WrapModelWithOutputFormatter(torch.nn.Module):
    _output_formatter: DynamoFlattenOutputWithTreeSpecValidationStep

    def __init__(
        self,
        model: torch.nn.Module,
        output_formatter: DynamoFlattenOutputWithTreeSpecValidationStep,
    ):
        super().__init__()
        self._model = model
        self._output_formatter = output_formatter

    def forward(self, *args, **kwargs):
        return self._output_formatter.format(self._model(*args, **kwargs))


class DynamoExporter(fx_exporter.FXGraphModuleExporter):
    def export(self) -> torch.onnx.ExportOutput:
        # args will be converted to symbolic tensor. Let's copy to avoid side effects.
        args = copy.deepcopy(self.model_args)
        kwargs = copy.deepcopy(self.model_kwargs)

        dynamo_flatten_output_step = DynamoFlattenOutputWithTreeSpecValidationStep()
        wrapped_model = _WrapModelWithOutputFormatter(
            self.model, dynamo_flatten_output_step
        )
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
        # Export FX graph to ONNX ModelProto.
        #
        # Note that ALL kwargs are folded into constants in graph_module, so we don't pass kwargs
        # to _export.
        flattened_args, _ = self._apply_input_format_step(
            fx_exporter.FlattenInputWithTreeSpecValidationStep, args, kwargs
        )

        return self.export_fx_to_onnx(graph_module, flattened_args)
