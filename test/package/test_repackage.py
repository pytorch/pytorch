# Owner(s): ["oncall: package/deploy"]

from io import BytesIO

from torch.package import PackageExporter, PackageImporter, sys_importer
from torch.testing._internal.common_utils import run_tests


try:
    from .common import PackageTestCase
except ImportError:
    # Support the case where we run this file directly.
    from common import PackageTestCase


class TestRepackage(PackageTestCase):
    """Tests for repackaging."""

    def test_repackage_import_indirectly_via_parent_module(self):
        from package_d.imports_directly import ImportsDirectlyFromSubSubPackage
        from package_d.imports_indirectly import ImportsIndirectlyFromSubPackage

        model_a = ImportsDirectlyFromSubSubPackage()
        buffer = BytesIO()
        with PackageExporter(buffer) as pe:
            pe.intern("**")
            pe.save_pickle("default", "model.py", model_a)

        buffer.seek(0)
        pi = PackageImporter(buffer)
        loaded_model = pi.load_pickle("default", "model.py")

        model_b = ImportsIndirectlyFromSubPackage()
        buffer = BytesIO()
        with PackageExporter(
            buffer,
            importer=(
                pi,
                sys_importer,
            ),
        ) as pe:
            pe.intern("**")
            pe.save_pickle("default", "model_b.py", model_b)


if __name__ == "__main__":
    run_tests()
