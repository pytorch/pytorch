import torch
from torch.fx import GraphModule

from torch.testing._internal.common_utils import TestCase, run_tests

class TestFX(TestCase):
    def test_graph_module(self):
        class MySub(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.w = torch.nn.Parameter(torch.rand(4, 3))
            def forward(self, x):
                return self.w + x

        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = torch.nn.Linear(4, 3)
                self.sub_mod = MySub()
                self.w = torch.nn.Parameter(torch.rand(3))
            def forward(self, A, B, c):
                t = torch.sigmoid(A) + self.lin(c)
                return self.sub_mod(t.data + self.w + t + 1 - A + B // A + -A + A.add(B, alpha=3))

        m = GraphModule(MyModule())
        print(m(torch.rand(3), torch.rand(4, 3), torch.rand(4)))

        ms = torch.jit.script(m)
        print(ms.code)

        class M2(torch.nn.Module):
            def forward(self, A):
                m, idx = torch.max(A, 0)
                return m + 1, idx + 1

        m2 = GraphModule(M2())
        print(m2(torch.rand(3, 4)))

        class T(torch.nn.Module):
            def forward(self, A, b=4,  *args, c=5, **kwargs):
                x = A + 1 + args[0] + kwargs['3']
                return x

        GraphModule(T())
        print('foo')

if __name__ == '__main__':
    run_tests()
