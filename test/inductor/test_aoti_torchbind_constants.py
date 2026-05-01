# Owner(s): ["module: inductor"]

import torch
from torch._inductor.test_case import TestCase
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.torchbind_impls import load_torchbind_test_lib


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
        # Canonical helper that handles FBCODE / Linux / macOS / Windows.
        load_torchbind_test_lib()
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

        loader = torch._C._aoti.AOTIModelPackageLoader(pt2_path, "model")
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

        loader = torch._C._aoti.AOTIModelPackageLoader(pt2_path, "model")
        self.assertEqual(loader.get_custom_objs(), {})


if __name__ == "__main__":
    run_tests()
