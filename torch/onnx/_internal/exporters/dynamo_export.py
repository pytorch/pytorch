import copy

import torch._dynamo
import torch.onnx
import torch.onnx._internal.exporters.fx_base


class DynamoExportExporter(
    torch.onnx._internal.exporters.fx_base.FXGraphModuleExporter
):
    def export(self) -> torch.onnx.ExportOutput:
        # args will be converted to symbolic tensor. Let's copy to avoid side effects.
        args = copy.deepcopy(self.model_args)
        # Translate callable to FX graph.
        #
        # TODO(wechi): There are several symbolic tracing mechanisms to convert
        # nn.Module to FX graph. We should choose the right one after they are
        # matured.
        graph_module, graph_guard = torch._dynamo.export(
            self.model, *args, aten_graph=True
        )
        del graph_guard  # Unused
        # Export FX graph to ONNX ModelProto.
        #
        # Note that ALL kwargs are folded into constants in graph_module, so we don't pass kwargs
        # to _export.
        return self.export_fx_to_onnx(graph_module, args)
