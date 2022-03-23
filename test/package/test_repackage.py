# Owner(s): ["oncall: package/deploy"]

from io import BytesIO

from torch.package import (
    PackageExporter,
    PackageImporter,
    sys_importer,
)
from torch.testing._internal.common_utils import run_tests
from torch.package.package_importer_no_torch import PackageImporter as PackageImporterNoTorch
from torch.package.package_exporter_no_torch import PackageExporter as PackageExporterNoTorch

try:
    from .common import PackageTestCase
except ImportError:
    # Support the case where we run this file directly.
    from common import PackageTestCase


class TestRepackage(PackageTestCase):
    """Tests for repackaging."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.PackageImporter = PackageImporter
        self.PackageExporter = PackageExporter

    def test_repackage_import_indirectly_via_parent_module(self):
        from package_d.imports_directly import ImportsDirectlyFromSubSubPackage
        from package_d.imports_indirectly import ImportsIndirectlyFromSubPackage

        model_a = ImportsDirectlyFromSubSubPackage()
        buffer = BytesIO()
        with self.PackageExporter(buffer) as pe:
            pe.intern("**")
            pe.save_pickle("default", "model.py", model_a)

        buffer.seek(0)
        pi = self.PackageImporter(buffer)
        loaded_model = pi.load_pickle("default", "model.py")

        model_b = ImportsIndirectlyFromSubPackage()
        buffer = BytesIO()
        with self.PackageExporter(
            buffer,
            importer=(
                pi,
                sys_importer,
            ),
        ) as pe:
            pe.intern("**")
            pe.save_pickle("default", "model_b.py", model_b)

class TestRepackageNoTorch(TestRepackage):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.PackageImporter = PackageImporterNoTorch
        self.PackageExporter = PackageExporterNoTorch

if __name__ == "__main__":
    run_tests()
