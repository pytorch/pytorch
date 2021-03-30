from io import BytesIO

import torch
from torch import Tensor
from torch.fx import Graph, GraphModule, symbolic_trace
from torch.package import (
    ObjMismatchError,
    PackageExporter,
    PackageImporter,
    sys_importer,
)
from torch.testing._internal.common_utils import run_tests

try:
    from .common import PackageTestCase
except ImportError:
    # Support the case where we run this file directly.
    from common import PackageTestCase  # type: ignore


class TestPackageScript(PackageTestCase):
    """Tests for compatibility with TorchScript."""

    def test_package_interface(self):
        """Packaging an interface class should work correctly."""
        import package_a.interface_fake as fake
        uses_interface = fake.UsesInterface()
        scripted = torch.jit.script(uses_interface)
        scripted.proxy_mod = torch.jit.script(fake.NewModule())

        buffer = BytesIO()
        with PackageExporter(buffer, verbose=False) as pe:
            pe.save_pickle("model", "model.pkl", uses_interface)
        buffer.seek(0)

        package_importer = PackageImporter(buffer)
        loaded = package_importer.load_pickle("model", "model.pkl")

        scripted_loaded = torch.jit.script(loaded)
        scripted_loaded.proxy_mod = torch.jit.script(fake.NewModule())

        input = torch.tensor(1)

        self.assertTrue(torch.allclose(scripted(input), scripted_loaded(input)))

    def test_package_script_class(self):
        import package_a.script_class_fake as fake

        scripted = torch.jit.script(fake.uses_script_class)

        buffer = BytesIO()
        with PackageExporter(buffer, verbose=False) as pe:
            pe.save_module(fake.__name__)
        buffer.seek(0)



        package_importer = PackageImporter(buffer)
        loaded = package_importer.import_module(fake.__name__).uses_script_class
        scripted_loaded = torch.jit.script(loaded)

        buffer.seek(0)
        package_importer = PackageImporter(buffer)
        loaded = package_importer.import_module(fake.__name__).uses_script_class
        scripted_loaded = torch.jit.script(loaded)
        torch.jit.save(scripted_loaded, "/tmp/foo.pt")

        input = torch.tensor(1)
        self.assertTrue(torch.allclose((scripted(input)), scripted_loaded(input)))



if __name__ == "__main__":
    run_tests()
