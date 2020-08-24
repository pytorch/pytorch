import torch
import unittest
from torch.fx import symbolic_trace, Proxy, Node, GraphModule, DefaultDelegate

from fx.quantization import Quantizer

from typing import Any, Callable, Dict, Optional, Tuple, Union
from torch.testing._internal.common_utils import TestCase, run_tests

try:
    from torchvision.models import resnet18
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False
skipIfNoTorchVision = unittest.skipIf(not HAS_TORCHVISION, "no torchvision")

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

        m = MyModule()
        gm = symbolic_trace(m)

        ms = torch.jit.script(gm)

        class M2(torch.nn.Module):
            def forward(self, A):
                m, idx = torch.max(A, 0)
                return m + 1, idx + 1

        m2 = M2()
        gm2 = symbolic_trace(m2)

        class T(torch.nn.Module):

            def forward(self, A, b=4, *args, c=5, **kwargs):
                x = A + 1 + args[0] + kwargs['3']
                return x

        t = T()
        symbolic_trace(t)

    def test_fx_shifts(self):
        class MyModule(torch.nn.Module):
            def forward(self, x):
                return x << 3, x >> 3

        input = torch.LongTensor(10).random_(0, 1024)

        m = MyModule()
        ref_outs = m(input)
        gm = symbolic_trace(m)
        test_outs = gm(input)

        self.assertEqual(ref_outs, test_outs)

    def test_dict(self):
        class MyDictMod(torch.nn.Module):
            def forward(self, d):
                return d['3'].relu(), {'4' : d['3'].neg()}

        input_dict = {'3': torch.rand(3, 4)}
        m = MyDictMod()
        ref_out = m(input_dict)
        gm = symbolic_trace(m)
        out = gm(input_dict)

        self.assertEqual(out, ref_out)

    def test_disallow_override(self):
        # Custom delegate to disallow in-place tensor operations
        class NoMutableCallDelegate(DefaultDelegate):
            def create_node(self, kind : str, target : Union[str, Callable],
                            args : Tuple[Any], kwargs : Dict[str, Any], name : Optional[str] = None) -> Node:
                name = target if isinstance(target, str) else torch.typename(target)
                if name[-1] == '_':
                    raise RuntimeError('In-place operations are not supported')
                return super().create_node(kind, target, args, kwargs, name)

        # Test method
        class MyInplaceMod(torch.nn.Module):
            def forward(self, x):
                x.add_(3.0)
                return x

        m = MyInplaceMod()

        with self.assertRaisesRegex(RuntimeError, 'In-place operations'):
            symbolic_trace(m, delegate_class=NoMutableCallDelegate)

        # Test free function
        class MyInplaceMod2(torch.nn.Module):
            def forward(self, x):
                torch.log_(x)
                return x
        m2 = MyInplaceMod2()
        with self.assertRaisesRegex(RuntimeError, 'In-place operations'):
            symbolic_trace(m2, delegate_class=NoMutableCallDelegate)

        # Test symbolic node as an arg
        class MyInplaceMod3(torch.nn.Module):
            def forward(self, x):
                y = torch.ones(3, 4)
                y.add_(x)
                return x
        m3 = MyInplaceMod3()
        with self.assertRaisesRegex(RuntimeError, 'In-place operations'):
            symbolic_trace(m3, delegate_class=NoMutableCallDelegate)

    def test_leaf_module(self):
        # Custom delegate to make it so that there are no leaf modules, everything
        # should get traced through
        class NoLeafModulesDelegate(DefaultDelegate):
            def is_leaf_module(self, m):
                return False

        class MyReluMod(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                return self.relu(x)

        mrm = MyReluMod()
        sym = symbolic_trace(mrm, delegate_class=NoLeafModulesDelegate)
        for node in sym.graph.nodes:
            self.assertNotEqual(node.op, 'call_module')

    def test_graph_edit_with_proxy(self):
        class M(torch.nn.Module):
            def forward(self, a, b):
                return a + b
        m = M()
        g = symbolic_trace(m).graph
        t = Proxy(g.result)
        # test that we can use proxy objects to generate more graph code later for things that do not need to work with modules.
        g.output((t + t).node)
        gm = GraphModule(m, g)
        self.assertEqual(gm(3, 4), 14)

    @skipIfNoTorchVision
    def test_resnet(self):
        resnet = resnet18()
        resnet.train()

        res_graph = symbolic_trace(resnet)
        res_script = torch.jit.script(res_graph)

        ip = torch.rand(1, 3, 224, 224)

        a = resnet(ip)
        b = res_graph(ip)
        c = res_script(ip)
        assert torch.allclose(a, b)
        assert torch.allclose(a, c)

        quantizer = Quantizer(res_graph)

        for i in range(10):
            quantizer.observe((torch.rand(1, 3, 224, 224),))

        qgraph = quantizer.quantize()
        qgraph_script = torch.jit.script(qgraph)

        d = qgraph(ip)
        e = qgraph_script(ip)

        assert (a - d).abs().max() < 2
        assert torch.allclose(d, e)

    def test_unpack(self):
        class M(torch.nn.Module):
            def forward(self, a, b):
                c, d = a
                return c + d + b
        m = M()
        m_g = symbolic_trace(m)
        a = (torch.rand(1), torch.rand(1))
        b = torch.rand(1)
        self.assertEqual(m(a, b), m_g(a, b))

if __name__ == '__main__':
    run_tests()
