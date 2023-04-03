import copy
from typing import Optional

import torch._dynamo
import torch.onnx
import torch.onnx._internal.exporters.fx_base


class DynamoOptimizeExporter(
    torch.onnx._internal.exporters.fx_base.FXGraphModuleExporter
):
    def export(self) -> torch.onnx.ExportOutput:
        # We hope the input kwargs will be mapped to bound.args after binding.
        # If not, we will raise an error.
        bound = self.model_signature.bind(*self.model_args, **self.model_kwargs)
        bound.apply_defaults()

        # keyword-only arguments are not handled.
        # bound.kwargs only contains keyword-only arguments after calling
        # bind & apply_defaults, so we throw if it's not empty.
        assert not bound.kwargs

        # args will be converted to symbolic tensor. Let's copy to avoid side effects.
        bound_args = copy.deepcopy(bound.args)
        # Translate callable to FX graph.
        #
        # TODO(wechi): There are several symbolic tracing mechanisms to convert
        # nn.Module to FX graph. We should choose the right one after they are
        # matured.

        class GraphCaptureCompiler:
            def __init__(self):
                self.captured_graph: Optional["torch.fx.GraphModule"] = None
                self.captured_graph_count = 0

            def compile(self, graph_module: "torch.fx.GraphModule", _):
                assert self.captured_graph_count == 0
                self.captured_graph = graph_module
                self.captured_graph_count += 1
                return graph_module

        compiler = GraphCaptureCompiler()
        torch._dynamo.reset()
        torch._dynamo.optimize(compiler.compile, nopython=True)(self.model)(*bound_args)
        torch._dynamo.reset()
        assert compiler.captured_graph
        # Export FX graph to ONNX ModelProto.
        return self.export_fx_to_onnx(
            compiler.captured_graph, tuple(arg for arg in bound_args if arg is not None)
        )
