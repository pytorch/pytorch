# Owner(s): ["oncall: fx"]

from torch.testing._internal.common_utils import TestCase, run_tests
from torch import fx
import torch

class TestLazyRecompile(TestCase):
    def test_replace_sin_with_cos(self):
        def f(x):
            return x.sin()

        x = torch.randn(2, 3)
        gm = fx.symbolic_trace(f)

        for n in gm.graph.nodes:
            if n.target == "sin":
                n.target = "cos"

        gm.recompile()
        expected = x.cos()
        actual = gm(x)

        self.assertTrue(torch.allclose(expected, actual))
        code = gm.print_readable(False)
        self.assertTrue("cos()" in code)

    def test_call_forward_directly(self):
        def f(x):
            return x.sin()

        x = torch.randn(2, 3)
        gm = fx.symbolic_trace(f)

        for n in gm.graph.nodes:
            if n.target == "sin":
                n.target = "cos"

        gm.recompile()
        expected = x.cos()
        actual = gm.forward(x)

        print(f"sin {x.sin()}, cos {x.cos()}, expected {expected}, actual {actual}")
        self.assertTrue(torch.allclose(expected, actual))


if __name__ == "__main__":
    run_tests()
