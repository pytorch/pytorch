from __future__ import annotations

import copy
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple, Union

import torch.onnx
from torch.onnx import utils as onnx_utils
from torch.onnx._internal import exporter
from torch.onnx._internal.fx import io_adapter as io_adapter


class DynamoOptimizeGraphCaptureCompiler:
    def __init__(self):
        self.captured_graph: Optional["torch.fx.GraphModule"] = None
        self.captured_graph_count = 0

    def __call__(self, graph_module: "torch.fx.GraphModule", _):
        assert self.captured_graph_count == 0
        self.captured_graph = graph_module
        self.captured_graph_count += 1
        return graph_module


class DynamoOptimize(exporter.FXGraphExtractor):
    """Generates a FX GraphModule using torch.dynamo.optimize API
    Args:
        backend: Specify which backend compiler must be used by Torch Dynamo.
            One of the two things:
                - Either, a function/callable taking a torch.fx.GraphModule and
                    example_inputs and returning a python callable that runs the
                    graph faster.
                    One can also provide additional context for the backend, like
                    torch.jit.fuser("fuser2"), by setting the backend_ctx_ctor attribute.
                    See AOTAutogradMemoryEfficientFusionWithContext for the usage.
                - Or, a string backend name in `torch._dynamo.list_backends()`
        nopython:  If True, graph breaks will be errors and there will be a single whole-program graph.
        disable: If True, turn this engine into a no-op
        dynamic: If True, turn on dynamic shapes support
    """

    def __init__(
        self,
        backend: Union[Callable, str] = None,
        nopython: bool = True,
        disable: bool = False,
        dynamic: bool = False,
    ):
        self.backend = backend or DynamoOptimizeGraphCaptureCompiler()
        self.nopython = nopython
        self.disable = disable
        self.dynamic = dynamic

    def generate_fx(
        self,
        options: exporter._ResolvedExportOptions,
        model: Union[torch.nn.Module, Callable],
        model_args: Sequence[Any],
        model_kwargs: Mapping[str, Any],
    ) -> Tuple[torch.fx.GraphModule, Tuple[Any]]:
        self._input_adapter = io_adapter.InputAdapter()
        self._output_adapter = io_adapter.OutputAdapter()

        # Fill in default values for optional args and kwargs. The goal is to preserve
        # them as inputs in `dynamo.optimize` produced FX graph. Otherwise, they will
        # be traced as constants.
        _, named_args = self.adapt_input(
            io_adapter.BindInputStep,
            model_args,
            model_kwargs,
            step_init_args=(onnx_utils.model_signature(model),),
        )
        model_args, _ = self.adapt_input(
            io_adapter.MergeKwargsIntoArgsStep,
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

        # NOTE: Do not import at top level.
        # Even torch/__init__.py does it internally, only
        # Causes circular when torch._dynamo.* surfaces public facing API during `import torch`
        import torch._dynamo

        torch._dynamo.reset()
        # TODO(titaiwang): Set `dynamic` according to `self.options.dynamic_shapes`
        torch._dynamo.optimize(
            self.backend,
            nopython=self.nopython,
            disable=self.disable,
            dynamic=self.dynamic,
        )(model)(*model_args)
        torch._dynamo.reset()
        # TODO: `self.backend.captured_graph` breaks the PyTorch Dynamo contract for compiler
        # The contract is backend is ``Callable[gm: GraphModule, example_inputs: Any] -> Callable``
        assert self.backend.captured_graph  # type: ignore[union-attr]

        # Outputs are flattened by `dynamo.optimize`.
        # Apply and record this output adapt step.
        self._output_adapter.append_step(io_adapter.DynamoFlattenOutputStep())
        # TODO: `dynamo.optimize` does not capture non computation part of the graph.
        # Hence any tensor that is returned multiple times will only appear once
        # in the return statement of `dynamo.optimize` traced graph. A specialized
        # output adapter is required to map this gap between PyTorch model.

        # Export FX graph to ONNX ModelProto.
        #
        # Apply and record the following input adapt steps.
        # `args` and `kwargs` are merged and flattened by `dynamo.optimize`.
        model_args, _ = self.adapt_input(
            io_adapter.FlattenInputWithTreeSpecValidationStep, model_args, {}
        )
        # `None` inputs are removed by `dynamo.optimize`.
        model_args, _ = self.adapt_input(io_adapter.RemoveNoneInputStep, model_args, {})
        # TODO: needs additional input adapt step to remove unused arguments.
        # `dynamo.optimize` drops unused arguments.
        # TODO: `self.backend.captured_graph` breaks the PyTorch Dynamo contract for compiler
        # The contract is backend is ``Callable[gm: GraphModule, example_inputs: Any] -> Callable``
        return self.backend.captured_graph, list(model_args)  # type: ignore[return-value,union-attr]


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
        self.aten_graph = aten_graph or True

    def generate_fx(
        self,
        options: exporter._ResolvedExportOptions,
        model: Union[torch.nn.Module, Callable],
        model_args: Sequence[Any],
        model_kwargs: Mapping[str, Any],
    ) -> Tuple[torch.fx.GraphModule, Tuple[Any]]:
        self._input_adapter = io_adapter.InputAdapter()
        self._output_adapter = io_adapter.OutputAdapter()

        # args will be converted to symbolic tensor. Let's copy to avoid side effects.
        args = copy.deepcopy(model_args)
        kwargs = copy.deepcopy(model_kwargs)

        # `dynamo.export` does not recognize custom user defined classes as output type.
        # Apply wrapper to adapt the outputs back to `dynamo.export` compatible types,
        # i.e. :class:`torch.Tensor`.
        dynamo_flatten_output_step = io_adapter.DynamoFlattenOutputStep()
        wrapped_model = io_adapter.wrap_model_with_output_adapter(
            model, dynamo_flatten_output_step
        )
        # Record the output adapter step.
        self._output_adapter.append_step(dynamo_flatten_output_step)

        # Translate callable to FX graph.
        #
        # TODO(wechi): There are several symbolic tracing mechanisms to convert
        # nn.Module to FX graph. We should choose the right one after they are
        # matured.

        import torch._dynamo  # TODO: Not even torch/__init__.py imports it globally

        graph_module, graph_guard = torch._dynamo.export(
            wrapped_model,
            *args,
            aten_graph=self.aten_graph,
            decomposition_table=options.decomposition_table,
            **kwargs,
        )
        del graph_guard  # Unused
        # NOTE: Do not import at top level.
        # Even torch/__init__.py does it internally, only
        # Causes circular when torch._dynamo.* surfaces public facing API during `import torch`
        import torch._dynamo

        torch._dynamo.reset()

        # Export FX graph to ONNX ModelProto.
        #
        # `args` and `kwargs` are merged and flattened by `dynamo.export`.
        # Apply and record this input adapt step.
        merged_args, _ = self.adapt_input(
            io_adapter.FlattenInputWithTreeSpecValidationStep, args, kwargs
        )
        return graph_module, merged_args  # type: ignore[return-value]
