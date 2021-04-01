from io import BytesIO

from torch.package import (
    OrderedImporter,
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


class TestImporter(PackageTestCase):
    """Tests for Importer and derived classes."""

    def test_sys_importer(self):
        import package_a
        import package_a.subpackage

        self.assertIs(sys_importer.import_module("package_a"), package_a)
        self.assertIs(
            sys_importer.import_module("package_a.subpackage"), package_a.subpackage
        )

    def test_sys_importer_roundtrip(self):
        import package_a
        import package_a.subpackage

        importer = sys_importer
        type_ = package_a.subpackage.PackageASubpackageObject
        module_name, type_name = importer.get_name(type_)

        module = importer.import_module(module_name)
        self.assertIs(getattr(module, type_name), type_)

    def test_single_ordered_importer(self):
        import module_a  # noqa: F401
        import package_a

        buffer = BytesIO()
        with PackageExporter(buffer, verbose=False) as pe:
            pe.save_module(package_a.__name__)

        buffer.seek(0)
        importer = PackageImporter(buffer)

        # Construct an importer-only environment.
        ordered_importer = OrderedImporter(importer)

        # The module returned by this environment should be the same one that's
        # in the importer.
        self.assertIs(
            ordered_importer.import_module("package_a"),
            importer.import_module("package_a"),
        )
        # It should not be the one available in the outer Python environment.
        self.assertIsNot(ordered_importer.import_module("package_a"), package_a)

        # We didn't package this module, so it should not be available.
        with self.assertRaises(ModuleNotFoundError):
            ordered_importer.import_module("module_a")

    def test_ordered_importer_basic(self):
        import package_a

        buffer = BytesIO()
        with PackageExporter(buffer, verbose=False) as pe:
            pe.save_module(package_a.__name__)

        buffer.seek(0)
        importer = PackageImporter(buffer)

        ordered_importer_sys_first = OrderedImporter(sys_importer, importer)
        self.assertIs(ordered_importer_sys_first.import_module("package_a"), package_a)

        ordered_importer_package_first = OrderedImporter(importer, sys_importer)
        self.assertIs(
            ordered_importer_package_first.import_module("package_a"),
            importer.import_module("package_a"),
        )


if __name__ == "__main__":
    run_tests()
