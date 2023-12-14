# Owner(s): ["module: dynamo"]
# flake8: noqa
import torch
import torch._dynamo

from torch._functorch.aot_autograd import aot_export_module
from torch._higher_order_ops.strict_mode import strict_mode

from torch.testing import FileCheck
from torch.testing._internal.common_utils import run_tests, TestCase


def _mark_strict_DO_NOT_USE(cls):
    def call(self, *args):
        return strict_mode(self, args)

    cls.__call__ = call
    return cls

class TestExperiment(TestCase):

    def test_with_buffer_as_submodule(self):
        @_mark_strict_DO_NOT_USE
        class B(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("buffer1", torch.ones(3))

            def forward(self, x):
                y = x + 2
                y.add_(4)
                # this doesnt' work today with HOO
                #self.buffer1.add_(6)
                buffer_updated = self.buffer1 + 6
                return x.sum() + y.sum() + buffer_updated.sum()

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.submodule = B()

            def forward(self, x):
                x_v2 = x.sin()
                return (self.submodule(x_v2), x + 3)

        inp = torch.randn(3)
        gm, _ = aot_export_module(M(), (inp,), trace_joint=False)
        self.assertExpectedInline(str(gm.code.strip()), """\
def forward(self, arg0_1, arg1_1):
    sin = torch.ops.aten.sin.default(arg1_1)
    strict_graph_0 = self.strict_graph_0
    strict_mode = torch.ops.higher_order.strict_mode(strict_graph_0, (sin, arg0_1));  strict_graph_0 = sin = arg0_1 = None
    getitem = strict_mode[0];  strict_mode = None
    add = torch.ops.aten.add.Tensor(arg1_1, 3);  arg1_1 = None
    return (getitem, add)""")

        self.assertExpectedInline(str(gm.strict_graph_0.code.strip()), """\
def forward(self, arg0_1, arg1_1):
    add = torch.ops.aten.add.Tensor(arg0_1, 2)
    add_1 = torch.ops.aten.add.Tensor(add, 4);  add = None
    add_2 = torch.ops.aten.add.Tensor(arg1_1, 6);  arg1_1 = None
    sum_1 = torch.ops.aten.sum.default(arg0_1);  arg0_1 = None
    sum_2 = torch.ops.aten.sum.default(add_1);  add_1 = None
    add_3 = torch.ops.aten.add.Tensor(sum_1, sum_2);  sum_1 = sum_2 = None
    sum_3 = torch.ops.aten.sum.default(add_2);  add_2 = None
    add_4 = torch.ops.aten.add.Tensor(add_3, sum_3);  add_3 = sum_3 = None
    return (add_4,)""")

        eager_mod = M()
        ep = torch.export.export(eager_mod, (inp,), strict=True)

        graph_res_1, graph_res_2 = ep(inp)
        eager_res_1, eager_res_2 = eager_mod(inp)

        self.assertTrue(torch.allclose(graph_res_2, eager_res_2))
        self.assertTrue(torch.allclose(graph_res_1, eager_res_1))

        graph_res_1, graph_res_2 = ep(inp)
        eager_res_1, eager_res_2 = eager_mod(inp)

        self.assertTrue(torch.allclose(graph_res_2, eager_res_2))
        self.assertTrue(torch.allclose(graph_res_1, eager_res_1))



if __name__ == '__main__':
    run_tests()
