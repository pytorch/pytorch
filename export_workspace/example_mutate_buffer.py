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


'''
class GraphModule(torch.nn.Module):
    def forward(self, orig_arg_0, orig_arg_1, orig_arg_2):
        arg0: f64[], arg1: i32[], arg2, = fx_pytree.tree_flatten_spec(([orig_arg_0, orig_arg_1, orig_arg_2], {}), self._in_spec)
        # No stacktrace found for following nodes
        _tensor_constant0 = self._tensor_constant0

        # File: /scratch/bahuang/work/repos/pytorch/export_workspace/example_mutate_buffer.py:24, code: read = self._bin_num_examples.index_select(0, index)
        index_select_default = torch.ops.aten.index_select.default(_tensor_constant0, 0, arg1);  _tensor_constant0 = None

        # No stacktrace found for following nodes
        _tensor_constant0_1 = self._tensor_constant0

        # File: /scratch/bahuang/work/repos/pytorch/export_workspace/example_mutate_buffer.py:27, code: _weighted_slices = torch.index_select(self._bin_num_examples, 0, index) * 1.0
        index_select_default_1 = torch.ops.aten.index_select.default(_tensor_constant0_1, 0, arg1);  _tensor_constant0_1 = None
        mul_tensor = torch.ops.aten.mul.Tensor(index_select_default_1, 1.0);  index_select_default_1 = None

        # File: /scratch/bahuang/work/repos/pytorch/export_workspace/example_mutate_buffer.py:30, code: dim=0, index=index.long(), source=_weighted_slices.squeeze()
        _to_copy_default = torch.ops.aten._to_copy.default(arg1, dtype = torch.int64)
        squeeze_default = torch.ops.aten.squeeze.default(mul_tensor);  mul_tensor = None

        # No stacktrace found for following nodes
        _tensor_constant0_2 = self._tensor_constant0

        # File: /scratch/bahuang/work/repos/pytorch/export_workspace/example_mutate_buffer.py:29, code: self._bin_num_examples.index_copy_(
        index_put__default = torch.ops.aten.index_put_.default(_tensor_constant0_2, [_to_copy_default], squeeze_default);  _tensor_constant0_2 = _to_copy_default = squeeze_default = None

        # File: /scratch/bahuang/work/repos/pytorch/export_workspace/example_mutate_buffer.py:33, code: dim=0, index=index, source=supervision_weight.squeeze()
        squeeze_default_1 = torch.ops.aten.squeeze.default(arg0);  arg0 = None

        # File: /scratch/bahuang/work/repos/pytorch/export_workspace/example_mutate_buffer.py:32, code: self._bin_num_examples.index_add_(
        index_put__default_1 = torch.ops.aten.index_put_.default(index_put__default, [arg1], squeeze_default_1, True);  index_put__default = arg1 = squeeze_default_1 = None
        return pytree.tree_unflatten([index_select_default], self._out_spec)

'''

# gm = torch.compile(model, backend="aot_eager")
# out = gm(*model.inp)

# out.sum().backward()

# gm.graph.print_tabular()


# exporter = ExportInterpreter(gm)
# exporter.run(model.inp)
# pp.pprint(exporter.ex_gm)

