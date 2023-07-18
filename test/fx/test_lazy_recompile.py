# Owner(s): ["oncall: fx"]

from torch.testing._internal.common_utils import TestCase, run_tests
from torch import fx
from torch.fx.experimental.proxy_tensor import make_fx
import torch
import torch._export

class TestLazyRecompile(TestCase):
    @staticmethod
    def replace_sin_with_cos(gm):
        for n in gm.graph.nodes:
            if n.target == "sin":
                n.target = "cos"

    def test_replace_sin_with_cos(self):
        def f(x):
            return x.sin()

        x = torch.randn(2, 3)
        gm = fx.symbolic_trace(f)
        self.replace_sin_with_cos(gm)

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
        self.replace_sin_with_cos(gm)
        gm.recompile()
        expected = x.cos()
        actual = gm.forward(x)

        self.assertTrue(torch.allclose(expected, actual))

    def test_export(self):
        """
        torch.export will access GraphModule._out_spec. Make sure we generate them
        if we have not done that yet.
        """
        def f(x):
            return x.sin()
        gm = torch._export.export(f, (torch.randn(2, 3),))
        self.assertTrue(isinstance(gm, torch._export.ExportedProgram))

    def test_needs_recompile(self):
        """
        Make sure needs_recompile() return the corrent state.
        """
        def f(x):
            return x.sin()

        gm = fx.symbolic_trace(f)
        self.assertTrue(gm._needs_recompile())
        gm(torch.randn(2, 3))
        self.assertFalse(gm._needs_recompile())

    def test_multi_recompile(self):
        """
        Cover the case that multiple recompilation happens.
        """
        def f(x):
            return x.sin()

        gm = fx.symbolic_trace(f)
        self.assertTrue(gm._needs_recompile())
        x = torch.randn(2, 3)
        # trigger the first recompilation
        self.assertTrue(torch.allclose(x.sin(), gm(x)))
        self.assertFalse(gm._needs_recompile())

        self.replace_sin_with_cos(gm)
        self.assertFalse(gm._needs_recompile())
        gm.recompile()
        self.assertTrue(gm._needs_recompile())
        # trigger the second recompilation
        self.assertTrue(torch.allclose(x.cos(), gm(x)))
        self.assertFalse(gm._needs_recompile())


    def test_accessing_code_cause_recompiling(self):
        """
        Make sure we recompile if we have not done that yet when we access the code
        property of a GraphModule.
        """
        def f(x):
            return x.sin()

        gm = fx.symbolic_trace(f)
        self.assertTrue(gm._needs_recompile())
        # should trigger a recompilation
        _ = gm.code
        self.assertFalse(gm._needs_recompile())

    def test_graph_module_str(self):
        def f(x):
            return x.sin()

        gm = fx.symbolic_trace(f)
        self.assertTrue("sin" in str(gm))

    def test_recapture_with_make_fx(self):
        def f(x):
            return x.sin()

        gm = fx.symbolic_trace(f)
        self.assertTrue(gm._needs_recompile())
        gm2 = make_fx(gm, (torch.randn(2, 3),))

        # gm still has pending recompilation make_fx can smoothly handle
        # lazye recompilation since its implemented thru the dispatcher.
        self.assertTrue(gm._needs_recompile())

    def test_recapture_with_symbolic_trace(self):
        def f(x):
            return x.sin()

        gm = fx.symbolic_trace(f)
        self.assertTrue(gm._needs_recompile())
        gm2 = fx.symbolic_trace(gm)

        # the lazy recompilcation is already realized. We realize the
        # recompilation in the beginning of symbolic_trace since symbolic_trace can not
        # handle the tracing of lazy recompilation.
        self.assertFalse(gm._needs_recompile())

    def test_recapture_with_dynamo(self):
        def f(x):
            return x.sin()

        gm = fx.symbolic_trace(f)
        self.assertTrue(gm._needs_recompile())
        gm2 = torch.compile(gm)(torch.rand(2, 3))

        # the lazy recompilcation is already realized. We realize the
        # recompilation in the beginning of dynamo since dynamo can not
        # handle the tracing of lazy recompilation.
        self.assertFalse(gm._needs_recompile())


    def test_recapture_with_torchscript(self):
        def f(x):
            return x.sin()

        gm = fx.symbolic_trace(f)
        self.assertTrue(gm._needs_recompile())
        gm2 = torch.jit.script(gm)

        # the lazy recompilcation is already realized. We realize the
        # recompilation in the beginning of torchscript since torchscript can not
        # handle the tracing of lazy recompilation.
        #
        # The real recompilation is triggered for torchscript automatically
        # when the get_overload_annotations API in torch/jit/_recursive.py is called.
        # This API will access the perperties like graph_in_spec etc which force
        # recompilation.
        self.assertFalse(gm._needs_recompile())


if __name__ == "__main__":
    run_tests()
