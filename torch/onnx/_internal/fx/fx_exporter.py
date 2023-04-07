import abc
from typing import Any, Callable, Mapping, Sequence

import torch._ops

import torch.onnx
import torch.onnx._internal.exporter
import torch.onnx._internal.fx.function_dispatcher as function_dispatcher
import torch.onnx._internal.fx.passes as passes
from torch.onnx._internal import _beartype

# TODO: make_fx lose stack info https://github.com/pytorch/pytorch/issues/90276


class FXGraphModuleExporter(torch.onnx._internal.exporter.Exporter, abc.ABC):
    @property
    def decomposition_table(self) -> Mapping[torch._ops.OpOverload, Callable]:
        return function_dispatcher._ONNX_FRIENDLY_DECOMPOSITION_TABLE

    @_beartype.beartype
    def export_fx_to_onnx(
        self,
        fx_module: torch.fx.GraphModule,
        fx_module_args: Sequence[Any],
    ) -> torch.onnx.ExportOutput:
        # Apply decomposition table to the input graph.
        decomposed_module = passes.Decompose(
            fx_module,
            self.decomposition_table,
            enable_dynamic_axes=self.options.dynamic_shapes,
        ).run(*fx_module_args)

        # We want to pass list of ints and floats to TorchScript graph correctly
        # in _export_fx_to_ts, so we must disable FakeTensorMode. Otherwise, graph may
        # receive FakeTensor and results runtime error. In addition, TorchScript-based
        # ONNX exporter used in _ts_graph_to_onnx_model_in_protobuf is not compatible
        # with FakeTensorMode.
        with torch.utils._mode_utils.no_dispatch():
            onnxscript_graph = passes.export_fx_to_onnxscript(
                decomposed_module, self.options
            )
        # Export TorchScript graph to ONNX ModelProto.
        onnx_model = onnxscript_graph.to_model_proto(self.options.opset_version)
        return torch.onnx.ExportOutput(onnx_model)
