# Owner(s): ["oncall: fx"]

import contextlib
import importlib.util
import pickle
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch
import torch._export
from torch import fx
from torch.fx._lazy_graph_module import (
    _LazyGraphModule,
    _make_graph_module,
    _use_lazy_graph_module,
)
from torch.fx.experimental.proxy_tensor import make_fx
from torch.package import PackageExporter, PackageImporter
from torch.testing._internal.common_utils import TestCase


class TestLazyGraphModule(TestCase):
    exit_stack = None

    @classmethod
    def setUpClass(cls):
        cls.exit_stack = contextlib.ExitStack()
        cls.exit_stack.enter_context(_use_lazy_graph_module(True))

    @classmethod
    def tearDownClass(cls):
        cls.exit_stack.close()

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
        self.assertTrue(isinstance(gm, _LazyGraphModule))

    def test_call_forward_directly(self):
        def f(x):
            return x.sin()

        x = torch.randn(2, 3)
        gm = fx.symbolic_trace(f)
        self.assertTrue(isinstance(gm, _LazyGraphModule))
        self.replace_sin_with_cos(gm)
        gm.recompile()
        expected = x.cos()
        actual = gm.forward(x)

        self.assertTrue(torch.allclose(expected, actual))

    def test_needs_recompile(self):
        """
        Make sure needs_recompile() return the correct state.
        """

        def f(x):
            return x.sin()

        gm = fx.symbolic_trace(f)
        self.assertTrue(isinstance(gm, _LazyGraphModule))
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
        self.assertTrue(isinstance(gm, _LazyGraphModule))
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
        self.assertTrue(isinstance(gm, _LazyGraphModule))
        self.assertTrue(gm._needs_recompile())
        # should trigger a recompilation
        code = gm.code
        self.assertTrue("sin" in code)
        self.assertFalse(gm._needs_recompile())

    def test_graph_module_str(self):
        def f(x):
            return x.sin()

        gm = fx.symbolic_trace(f)
        self.assertTrue(isinstance(gm, _LazyGraphModule))
        self.assertTrue("sin" in str(gm))

    def test_recapture_with_make_fx(self):
        def f(x):
            return x.sin()

        gm = fx.symbolic_trace(f)
        self.assertTrue(isinstance(gm, _LazyGraphModule))
        self.assertTrue(gm._needs_recompile())
        gm2 = make_fx(gm)(torch.randn(2, 3))
        self.assertTrue(isinstance(gm2, _LazyGraphModule))
        self.assertTrue(gm2._needs_recompile())

        # make_fx will cal forward method of gm. That clears the _needs_recompile()
        # flag.
        self.assertFalse(gm._needs_recompile())

    def test_recapture_with_symbolic_trace(self):
        def f(x):
            return x.sin()

        gm = fx.symbolic_trace(f)
        self.assertTrue(isinstance(gm, _LazyGraphModule))
        self.assertTrue(gm._needs_recompile())
        gm2 = fx.symbolic_trace(gm)

        # the lazy recompilcation is already realized. We realize the
        # recompilation in the beginning of symbolic_trace since symbolic_trace can not
        # handle the tracing of lazy recompilation.
        self.assertFalse(gm._needs_recompile())
        self.assertTrue(gm2._needs_recompile())

    def test_recapture_with_dynamo(self):
        def f(x):
            return x.sin()

        gm = fx.symbolic_trace(f)
        self.assertTrue(isinstance(gm, _LazyGraphModule))
        self.assertTrue(gm._needs_recompile())
        torch.compile(gm)(torch.rand(2, 3))

        # dynamo calls gm.forward with eval hook installed. That will trigger
        # the real recompilation.
        self.assertFalse(gm._needs_recompile())

    def test_save_lazy_foward(self):
        """
        Save the lazy forward method and call it repeatedly. Make sure we
        don't recompile for each such call.
        """

        def f(x):
            return x.sin()

        orig_gm_recompile = fx.GraphModule.recompile
        recompile_count = 0

        def mock_gm_recompile(self):
            nonlocal recompile_count
            recompile_count += 1
            return orig_gm_recompile(self)

        with patch.object(fx.GraphModule, "recompile", mock_gm_recompile):
            gm = fx.symbolic_trace(f)
            self.assertTrue(isinstance(gm, _LazyGraphModule))
            saved_fwd = gm.forward

            x = torch.rand(2, 3)
            for _ in range(10):
                saved_fwd(x)

        self.assertEqual(recompile_count, 1)

    def test_pickle(self):
        """
        Fx graph cache need the ability to pickle GraphModule/_LazyGraphModule.
        """

        def f(x):
            return x.sin()

        gm = fx.symbolic_trace(f)
        self.assertTrue(isinstance(gm, _LazyGraphModule))
        serialized = pickle.dumps(gm)
        gm2 = pickle.loads(serialized)
        self.assertTrue(isinstance(gm2, _LazyGraphModule))
        self.assertTrue("sin" in gm2.code)

    def test_make_graph_module(self):
        gm = fx.symbolic_trace(lambda x: x.sin())
        self.assertTrue(isinstance(gm, _LazyGraphModule))

        gm1 = _make_graph_module(
            gm, gm.graph, class_name="MyGraphModule", graph_module_cls=fx.GraphModule
        )
        self.assertFalse(isinstance(gm1, _LazyGraphModule))
        self.assertTrue(gm1.__class__.__name__ == "MyGraphModule")

        gm2 = _make_graph_module(gm, gm.graph)
        self.assertTrue(isinstance(gm2, _LazyGraphModule))
        self.assertTrue(gm2.__class__.__name__ == "GraphModule")

    def test_package_fx_simple(self):
        """
        Copied from test/package/test_package_fx.py to make sure LazyGraphModule
        works with torch.package.
        """

        class SimpleTest(torch.nn.Module):
            def forward(self, x):
                return torch.relu(x + 3.0)

        st = SimpleTest()
        traced = fx.symbolic_trace(st)

        f = BytesIO()
        with PackageExporter(f) as pe:
            pe.save_pickle("model", "model.pkl", traced)

        f.seek(0)
        pi = PackageImporter(f)
        loaded_traced = pi.load_pickle("model", "model.pkl")
        input = torch.rand(2, 3)
        self.assertEqual(loaded_traced(input), traced(input))

    def test_dynamo_innermost_fn(self):
        """
        Repro for https://github.com/pytorch/pytorch/issues/121198 .
        """

        def f(x):
            return x * 2

        gm = torch.fx.symbolic_trace(f)
        lazy_gm = torch.fx._lazy_graph_module._LazyGraphModule.from_graphmodule(gm)

        wrapped_forward = torch._dynamo.disable(gm.forward)
        got_inner_forward = torch._dynamo.eval_frame.innermost_fn(wrapped_forward)
        assert hasattr(got_inner_forward, "__self__")

        wrapped_lazy_forward = torch._dynamo.disable(lazy_gm.forward)
        got_lazy_inner_forward = torch._dynamo.eval_frame.innermost_fn(
            wrapped_lazy_forward
        )
        assert hasattr(got_lazy_inner_forward, "__self__")

    def test_to_folder_class(self):
        class TestModule(torch.nn.Module):
            def __init__(self, features_in=10):
                super().__init__()

                self.seq = torch.nn.Sequential(
                    torch.nn.Linear(features_in, features_in), torch.nn.ReLU()
                )
                self.a = torch.nn.Parameter(torch.randn(features_in))
                setattr(self, "0", torch.nn.Parameter(torch.randn(features_in)))

            def forward(self, x):
                return self.seq.forward(x) + self.a + getattr(self, "0")

        in_features = 10

        f = TestModule(in_features)

        self.validate_module(f, in_features=in_features)

    def test_to_folder_sequential(self):
        in_features = 10

        f = torch.nn.Sequential(
            torch.nn.Linear(in_features=in_features, out_features=in_features)
        )

        self.validate_module(f, in_features=in_features)

    def validate_module(
        self, module: torch.nn.Module, in_features: int, n_samples: int = 100
    ):
        gm = torch.fx.symbolic_trace(module)

        with TemporaryDirectory() as tmp:
            gm.to_folder(tmp)

            path = Path(tmp) / "module.py"

            spec = importlib.util.spec_from_file_location(
                "module.FxModule", path.absolute()
            )
            foo = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(foo)

            loaded_module = foo.FxModule()

            del foo

        val = torch.randn(n_samples, in_features)

        self.assertEqual(module(val), loaded_module(val))


if __name__ == "__main__":
    raise RuntimeError(
        "This test is not currently used and should be "
        "enabled in discover_tests.py if required."
    )
