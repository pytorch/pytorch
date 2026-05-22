# Owner(s): ["module: inductor"]

import torch
from torch._inductor.test_case import TestCase
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.torchbind_impls import init_torchbind_implementations


HAS_CUDA = torch.cuda.is_available()


class TestTorchbindAOTI(TestCase):
    """Verify that torchbind constants embedded in an AOTI .pt2 are reachable
    via AOTIModelPackageLoader.get_custom_objs() after load.

    Backends (e.g. torch-tensorrt) need post-load access to torchbind
    constants to mutate runtime state (stream binding, comm binding, profiling
    toggles). Before this accessor existed, the IValues lived in
    OSSProxyExecutor::custom_objs_ with no public way out.
    """

    @classmethod
    def setUpClass(cls):
        # Loads the test torchbind library AND registers fake classes for
        # _Foo / _TensorQueue / etc. needed by torch.export tracing.
        init_torchbind_implementations()
        super().setUpClass()

    def _make_model(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.attr = torch.classes._TorchScriptTesting._Foo(10, 20)

            def forward(self, x):
                return self.attr.add_tensor(x) + x

        return M()

    def test_custom_objs_exposed_through_loader(self):
        device = "cuda" if HAS_CUDA else "cpu"
        m = self._make_model().to(device)
        x = torch.randn(2, 3, device=device)
        ep = torch.export.export(m, (x,), strict=False)

        pt2_path = torch._inductor.aoti_compile_and_package(ep)

        loader = torch._C._aoti.AOTIModelPackageLoader(pt2_path, "model", False, 1, -1)
        custom_objs = loader.get_custom_objs()

        self.assertGreater(
            len(custom_objs),
            0,
            msg="Expected at least one torchbind constant, got none",
        )
        any_torchbind = next(iter(custom_objs.values()))
        # The IValue payload must be a real custom class instance.
        self.assertTrue(
            isinstance(any_torchbind, torch.ScriptObject)
            or hasattr(any_torchbind, "_type"),
            msg=f"Expected a torchbind ScriptObject, got {type(any_torchbind)}",
        )

    def test_mutating_custom_obj_after_load_affects_run(self):
        # The central contract: IValues returned by get_custom_objs() share
        # intrusive_ptr ownership with the live entries inside
        # OSSProxyExecutor::custom_objs_, so mutating state on the returned
        # custom-class instance affects subsequent run() invocations.
        # Forward computes self.attr.add_tensor(x) + x, where Foo.add_tensor
        # returns (x + y) * z. Foo.increment(k) does x += k, y += k.
        device = "cuda" if HAS_CUDA else "cpu"
        m = self._make_model().to(device)  # Foo(10, 20), so x+y == 30
        x = torch.randn(2, 3, device=device)
        ep = torch.export.export(m, (x,), strict=False)

        pt2_path = torch._inductor.aoti_compile_and_package(ep)
        loader = torch._C._aoti.AOTIModelPackageLoader(pt2_path, "model", False, 1, -1)

        # Baseline: (x + y == 30) * x + x
        before = loader.run([x])[0]
        torch.testing.assert_close(before, 30 * x + x)

        # Mutate the live torchbind object via the snapshot returned by
        # get_custom_objs(). After Foo.increment(5) we have (15, 25), so
        # x + y == 40.
        custom_objs = loader.get_custom_objs()
        foo = next(
            obj
            for obj in custom_objs.values()
            if isinstance(obj, torch.ScriptObject)
            and obj._type().qualified_name()
            == "__torch__.torch.classes._TorchScriptTesting._Foo"
        )
        foo.increment(5)

        # Re-run: the executor must observe the mutated state.
        after = loader.run([x])[0]
        torch.testing.assert_close(after, 40 * x + x)

    def test_custom_objs_empty_when_no_torchbind(self):
        # A plain model with no torchbind attrs should yield an empty map.
        class Plain(torch.nn.Module):
            def forward(self, x):
                return x + 1

        device = "cuda" if HAS_CUDA else "cpu"
        m = Plain().to(device)
        x = torch.randn(2, 3, device=device)
        ep = torch.export.export(m, (x,), strict=False)

        pt2_path = torch._inductor.aoti_compile_and_package(ep)

        loader = torch._C._aoti.AOTIModelPackageLoader(pt2_path, "model", False, 1, -1)
        self.assertEqual(loader.get_custom_objs(), {})


if __name__ == "__main__":
    run_tests()
