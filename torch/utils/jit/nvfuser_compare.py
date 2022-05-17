import torch
from dataclasses import dataclass

class ComparisonResult():
    graph: str
    rel_diff: float
    abs_diff: float
    expected_dtype = torch.int
    actual_dtype = torch.int
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
        self.results = []

    def callback(self, fused_outputs, unfused_outputs, graph):
        for fused, unfused in zip(fused_outputs, unfused_outputs):
            result = ComparisonResult()
            result.graph = str(graph)
            result.ops_count = count_nonconstant_ops(graph)
            result.abs_diff = torch.max(torch.abs(fused - unfused)).item()
            matches = (fused == unfused)
            rel_diff_tensor = torch.abs(fused - unfused) / torch.abs(unfused)
            rel_diff_tensor[matches] = 0
            result.rel_diff = torch.max(rel_diff_tensor).item()

            result.unfused_dtype = unfused.dtype
            result.fused_dtype = fused.dtype
            self.results.append(result)

    def dump(self):
        for result in self.results:
            print(result.graph, result.rel_diff, result.abs_diff, result.expected_dtype, result.actual_dtype, result.ops_count)

def register_compare_log(options):
    logger = CompareLogger()
    torch._C._jit_nvfuser_set_comparison_callback(True, logger)
    return logger
