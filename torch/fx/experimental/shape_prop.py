import torch
from torch.fx.experimental.tracing_analysis import TracingAnalysis


class ShapeProp(TracingAnalysis):
    def store_result(self, node, result):
        if isinstance(result, torch.Tensor):
            node.shape = result.shape
            node.dtype = result.dtype
        super(ShapeProp, self).store_result(node, result)

    propagate = TracingAnalysis.run  # backwards compatibility
