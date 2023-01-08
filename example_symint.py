import torch
import torch._dynamo
from torch.export.serialization import ExportInterpreter
from typing import List
from torch.fx.experimental.proxy_tensor import make_fx

import prettyprinter as pp

class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # self.a = torch.nn.Parameter(torch.tensor([1.0]))
        # self.b = torch.nn.Parameter(torch.tensor([2.0]))

    def forward(self, x: torch.Tensor):
        a =  torch.empty(x.shape[0] * 2)
        # b = a * x.shape[1]
        return a

device = "cuda"
batch_size = 2
model = MyModule().cuda()

inp = (torch.empty(4, 4), )
gm, guard = torch._dynamo.export(model, *inp, aten_graph=True, tracing_mode="symbolic")
# gm = make_fx(model, tracing_mode="symbolic")(torch.empty(4))

gm.graph.print_tabular()

gm.print_readable()

exporter = ExportInterpreter(gm)
exporter.run(inp)

pp.install_extras()
pp.pprint(exporter.ex_gm)

