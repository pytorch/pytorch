from io import BytesIO
from textwrap import dedent

import torch
from torch.package import (
    PackageExporter,
    PackageImporter,
)
from torch.testing._internal.common_utils import run_tests

try:
    from .common import PackageTestCase
except ImportError:
    # Support the case where we run this file directly.
    from common import PackageTestCase


class TestPackageScript(PackageTestCase):
    """Tests for compatibility with TorchScript."""

    def test_package_interface(self):
        """Packaging an interface class should work correctly."""

        import package_a.fake_interface as fake

        uses_interface = fake.UsesInterface()
        scripted = torch.jit.script(uses_interface)
        scripted.proxy_mod = torch.jit.script(fake.NewModule())

        buffer = BytesIO()
        with PackageExporter(buffer) as pe:
            pe.intern("**")
            pe.save_pickle("model", "model.pkl", uses_interface)
        buffer.seek(0)

        package_importer = PackageImporter(buffer)
        loaded = package_importer.load_pickle("model", "model.pkl")

        scripted_loaded = torch.jit.script(loaded)
        scripted_loaded.proxy_mod = torch.jit.script(fake.NewModule())

        input = torch.tensor(1)

        self.assertTrue(torch.allclose(scripted(input), scripted_loaded(input)))

    def test_different_package_interface(self):
        """Test a case where the interface defined in the package is
        different than the one defined in the loading environment, to make
        sure TorchScript can distinguish between the two.
        """
        # Import one version of the interface
        import package_a.fake_interface as fake

        # Simulate a package that contains a different version of the
        # interface, with the exact same name.
        buffer = BytesIO()
        with PackageExporter(buffer) as pe:
            pe.save_source_string(
                fake.__name__,
                dedent(
                    """\
                    import torch
                    from torch import Tensor

                    @torch.jit.interface
                    class ModuleInterface(torch.nn.Module):
                        def one(self, inp1: Tensor) -> Tensor:
                            pass

                    class ImplementsInterface(torch.nn.Module):
                        def one(self, inp1: Tensor) -> Tensor:
                            return inp1 + 1

                    class UsesInterface(torch.nn.Module):
                        proxy_mod: ModuleInterface

                        def __init__(self):
                            super().__init__()
                            self.proxy_mod = ImplementsInterface()

                        def forward(self, input: Tensor) -> Tensor:
                            return self.proxy_mod.one(input)
                    """
                ),
            )
        buffer.seek(0)

        package_importer = PackageImporter(buffer)
        diff_fake = package_importer.import_module(fake.__name__)
        # We should be able to script successfully.
        torch.jit.script(diff_fake.UsesInterface())

    def test_package_script_class(self):
        import package_a.fake_script_class as fake

        buffer = BytesIO()
        with PackageExporter(buffer) as pe:
            pe.save_module(fake.__name__)
        buffer.seek(0)

        package_importer = PackageImporter(buffer)
        loaded = package_importer.import_module(fake.__name__)

        input = torch.tensor(1)
        self.assertTrue(
            torch.allclose(
                fake.uses_script_class(input), loaded.uses_script_class(input)
            )
        )

    def test_different_package_script_class(self):
        """Test a case where the script class defined in the package is
        different than the one defined in the loading environment, to make
        sure TorchScript can distinguish between the two.
        """
        import package_a.fake_script_class as fake

        # Simulate a package that contains a different version of the
        # script class ,with the attribute `bar` instead of `foo`
        buffer = BytesIO()
        with PackageExporter(buffer) as pe2:
            pe2.save_source_string(
                fake.__name__,
                dedent(
                    """\
                    import torch

                    @torch.jit.script
                    class MyScriptClass:
                        def __init__(self, x):
                            self.bar = x
                    """
                ),
            )
        buffer.seek(0)

        package_importer = PackageImporter(buffer)
        diff_fake = package_importer.import_module(fake.__name__)
        input = torch.rand(2, 3)
        loaded_script_class = diff_fake.MyScriptClass(input)
        orig_script_class = fake.MyScriptClass(input)
        self.assertTrue(torch.allclose(loaded_script_class.bar, orig_script_class.foo))


if __name__ == "__main__":
    run_tests()
