import torch
import torch._dynamo
from torch.export.serialization import ExportInterpreter
from typing import List
from torch.fx.experimental.proxy_tensor import make_fx

import prettyprinter as pp
pp.install_extras()
from prettyprinter.prettyprinter import IMPLICIT_MODULES
IMPLICIT_MODULES.add(
    'torch.export.export_schema'
)

class SymIntAsArg(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        new_shape = x.shape[0] * 2
        a =  torch.empty(new_shape)  # SymInt as arg
        return a

    inp = (torch.empty(4, 4), )

class SymIntsAsArg(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        new_shape = [x.shape[0] * x.shape[1], x.shape[-1]]
        a =  x.view(new_shape)      # SymInt[] as arg
        return a

    inp = (torch.empty(4, 3, 2), )

class TensorsAsArg(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        y = x.relu()
        out = torch.cat([x, y], dim=1)  # Tensor[] as arg
        return out

    inp = (torch.empty(4, 4), )


model = SymIntsAsArg().cuda()
gm, guard = torch._dynamo.export(model, *model.inp, aten_graph=True, tracing_mode="symbolic")

gm.graph.print_tabular()
gm.print_readable()

exporter = ExportInterpreter(gm)
exporter.run(model.inp)

pp.pprint(exporter.ex_gm)

