import torch
from torch.fx import symbolic_trace

class M(torch.nn.Module):
    def __init__(self):
        super(M, self).__init__()
        self.W = torch.nn.Parameter(torch.randn(5))

    def forward(self, x):
        return torch.dot(self.W, x)

traced = torch.fx.symbolic_trace(M())

out = [n for n in traced.graph.nodes if n.op == "output"][-1]
with traced.graph.inserting_before(out):
    relu_out = traced.graph.call_method(method_name='relu', args=(out.args[0],))
out.args = (relu_out,)

traced.recompile()

traced(5)
