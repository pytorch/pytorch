import torch
import torch._dynamo
from torch.export.serialization import export_graphmodule
from typing import List
from torch.fx.experimental.proxy_tensor import make_fx
from functorch.experimental.control_flow import cond, map

import prettyprinter as pp
pp.install_extras()
from prettyprinter.prettyprinter import IMPLICIT_MODULES
IMPLICIT_MODULES.add(
    'torch.export.export_schema'
)

class ConditionOp(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def true_fn(self, x, y):
        return x * y

    def false_fn(self, x, y):
        return x + y

    def forward(self, pred, x, y):
        return cond(pred, self.true_fn, self.false_fn, [x, y])

    inp = (torch.tensor(False), torch.empty(4, 4), torch.empty(4, 4),)


model = ConditionOp().cuda()
gm, guard = torch._dynamo.export(model, *model.inp, aten_graph=True, tracing_mode="symbolic")

# gm.graph.print_tabular()
gm.print_readable()

ex_gm = export_graphmodule(gm, model.inp)
pp.pprint(ex_gm)
