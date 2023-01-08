import torch
import torch._dynamo
from serialization import ExportInterpreter
from torchvision.models import resnet18

device = "cuda"
batch_size = 2
model = resnet18().cuda().eval()
x = torch.rand(batch_size, 3, 224, 224, device=device, dtype=torch.float)
gm, guard = torch._dynamo.export(model, x, aten_graph=True)

# gm.graph.print_tabular()

exporter = ExportInterpreter(gm)
exporter.run(x)

import prettyprinter as pp
pp.install_extras()
pp.pprint(exporter.ex_gm)