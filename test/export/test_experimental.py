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
                self.buffer1.add_(6)
                return x.sum() + y.sum() + self.buffer1.sum()

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

        eager_mod = M()

        graph_res_1, graph_res_2 = gm(torch.ones(3), inp)
        eager_res_1, eager_res_2 = eager_mod(inp)

        self.assertTrue(torch.allclose(graph_res_2, eager_res_1))
        self.assertTrue(torch.allclose(graph_res_3, eager_res_2))

        graph_res_1, graph_res_2 = gm(graph_res_1, inp)
        eager_res_1, eager_res_2 = eager_mod(inp)

        self.assertTrue(torch.allclose(graph_res_2, eager_res_1))

    def test_cond(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                def true_fn(x):
                    return x.cos()

                def false_fn(x):
                    return x.sin()

                a = torch.cond(x.shape[0] > 4, true_fn, false_fn, [x])
                return (a + 3, a + 4)

        inp = torch.randn(3, 4)
        from torch.fx.experimental.proxy_tensor import make_fx
        gm, _ = aot_export_module(M(), (inp,), trace_joint=False)
        print(gm)



if __name__ == '__main__':
    run_tests()
