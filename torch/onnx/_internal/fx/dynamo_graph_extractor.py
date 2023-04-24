from __future__ import annotations

import copy
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple, Union

import torch.onnx
from torch.onnx._internal import exporter
from torch.onnx._internal.fx import io_adapter as io_adapter


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
    ) -> Tuple[torch.fx.GraphModule, Tuple[Any]]:
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
        self.output_adapter.append_step(dynamo_flatten_output_step)

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
