import torch
import torch._dynamo
from serialization import ExportInterpreter
from torch._decomp import get_decompositions

import prettyprinter as pp
pp.install_extras()
from prettyprinter.prettyprinter import IMPLICIT_MODULES
IMPLICIT_MODULES.add(
    'torch._export.logical_schema'
)

class MutateBuffer(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.register_buffer(
            "_bin_num_examples",
            torch.empty([4], dtype=torch.float64).fill_(0.0),
        )  # ConstantFill

    def forward(self, supervision_weight: torch.Tensor, index: torch.Tensor, dummy: torch.Tensor):
        # Read from buffer
        read = self._bin_num_examples.index_select(0, index)

        # Buffer update
        _weighted_slices = torch.index_select(self._bin_num_examples, 0, index) * 1.0

        self._bin_num_examples.index_copy_(
            dim=0, index=index.long(), source=_weighted_slices.squeeze()
        )
        self._bin_num_examples.index_add_(
            dim=0, index=index, source=supervision_weight.squeeze()
        )  # ScatterWeightedSum

        return read

    inp = (torch.tensor(0.5, dtype=torch.double), torch.tensor(2, dtype=torch.int), torch.rand(2,2, requires_grad=True))

aten = torch.ops.aten
my_decomp = get_decompositions(
    [
        aten.index_add_,
        aten.index_copy_,
    ]
)

model = MutateBuffer()
gm, guard = torch._dynamo.export(model, *model.inp, aten_graph=True, decomposition_table=my_decomp, tracing_mode="real")
gm.print_readable()

# gm = torch.compile(model, backend="aot_eager")
# out = gm(*model.inp)

# out.sum().backward()

# gm.graph.print_tabular()


# exporter = ExportInterpreter(gm)
# exporter.run(model.inp)
# pp.pprint(exporter.ex_gm)

