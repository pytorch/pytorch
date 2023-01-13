import torch
import torch._dynamo
from serialization import ExportInterpreter
from torchvision.models import resnet18

import prettyprinter as pp
pp.install_extras()
from prettyprinter.prettyprinter import IMPLICIT_MODULES
IMPLICIT_MODULES.add(
    'torch.export.export_schema'
)

device = "cuda"
batch_size = 2
model = resnet18().cuda().eval()
x = torch.rand(batch_size, 3, 224, 224, device=device, dtype=torch.float)
gm, guard = torch._dynamo.export(model, x, aten_graph=True, tracing_mode="real")
# gm.graph.print_tabular()

exporter = ExportInterpreter(gm)
exporter.run(x)
pp.pprint(exporter.ex_gm)