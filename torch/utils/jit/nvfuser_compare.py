import torch

@dataclass
class ComparisonResult():
    graph: str
    rel_diff: float
    abs_diff: float
    ops_count: float

def count_nonconstant_ops(graph):
    count = 0
    for n in graph.nodes():
        if n.kind() != 'prim::Constant':
            count += 1
    return count

class CompareLogger():
    def __init__(self, print_all=True):
        self.print_all = print_all

    def callback(self, fused_outputs, unfused_outputs, graph):
        result = ComparisonResult()
        result.graph = str(graph)
        result.ops_count = count_nonconstant_ops(graph)

        abs_diff = torch.max(torch.abs(fused_outputs - unfused_outputs)).item()
        matches = (fused_outputs == unfused_outputs)
        rel_diff_tensor = torch.abs(fused_outputs - unfused_outputs) / torch.abs(unfused_outputs)
        rel_diff_tensor[matches] = 0
        rel_diff = torch.max(rel_diff_tensor).item()

def register_compare_log(options):
    logger = CompareLogger()
    torch._C._jit_nvfuser_set_comparison_callback(True, logger)
    return logger
