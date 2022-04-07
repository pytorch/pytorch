# -*- coding: utf-8 -*-
# Owner(s): ["oncall: package/deploy"]

from textwrap import dedent
from unittest import skipIf

from torch.package import PackageExporter, PackageImporter
from torch.package.package_importer_no_torch import PackageImporter as PackageImporterNoTorch
from torch.package.package_exporter_no_torch import PackageExporter as PackageExporterNoTorch
from torch.package._zip_file import DefaultPackageZipFileReader, DefaultPackageZipFileWriter
from torch.package._zip_file_torchscript import TorchScriptPackageZipFileReader, TorchScriptPackageZipFileWriter
from torch.testing._internal.common_utils import (
    run_tests,
    IS_FBCODE,
    IS_SANDCASTLE,
    IS_WINDOWS,
)

try:
    from .common import PackageTestCase
except ImportError:
    # Support the case where we run this file directly.
    from common import PackageTestCase

from pathlib import Path

packaging_directory = Path(__file__).parent


@skipIf(
    IS_FBCODE or IS_SANDCASTLE or IS_WINDOWS,
    "Tests that use temporary files are disabled in fbcode",
)
class ZipfileTest(PackageTestCase):
    """Tests use of DirectoryReader as accessor for opened packages."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ZipfileReader = DefaultPackageZipFileReader
        self.ZipfileWriter = DefaultPackageZipFileWriter
        self.PackageExporter = PackageExporter
        self.PackageImporter = PackageImporter

    # def test_save_and_read(self):
    #     zip_file =

    def test_has_record(self):
        """
        Test DirectoryReader's has_record().
        """
        import package_a  # noqa: F401

        filename = self.temp()
        with self.PackageExporter(filename) as e:
            e.save_module("package_a")

        zip_file = self.ZipfileReader(filename)
        self.assertTrue(zip_file.has_record("package_a/__init__.py"))
        self.assertFalse(zip_file.has_record("package_a"))

    def test_read_record(self):
        """Packaged modules should be able to use the importlib.resources API to access
        resources saved in the package.
        """
        mod_src = dedent(
            """\
            import importlib.resources
            import my_cool_resources

            def secret_message():
                return importlib.resources.read_text(my_cool_resources, 'sekrit.txt')
            """
        )
        filename = self.temp()
        with self.PackageExporter(filename) as pe:
            pe.save_source_string("foo.bar", mod_src)
            pe.save_text("my_cool_resources", "sekrit.txt", "my sekrit plays")

        zip_file = self.ZipfileReader(filename)
        self.assertEqual(
            zip_file.get_record("my_cool_resources/sekrit.txt"),
            b"my sekrit plays",
        )

    def test_readall(self):
        """Packaged modules should be able to use the importlib.resources API to access
        resources saved in the package.
        """
        mod_src = dedent(
            """\
            import importlib.resources
            import my_cool_resources

            def secret_message():
                return importlib.resources.read_text(my_cool_resources, 'sekrit.txt')
            """
        )
        filename = self.temp()
        with self.PackageExporter(filename) as pe:
            pe.save_source_string("foo.bar", mod_src)
            pe.save_text("my_cool_resources", "sekrit.txt", "my sekrit plays")
            pe.save_text("my_cool_resources", "bar.txt", "foo")
            pe.save_text("my_less_cool_resources", "another_one.txt", "foo bar")
            pe.save_text("my_less_cool_resources", "foo.txt", "bar")

        zip_file = self.ZipfileReader(filename)
        zip_file_contents = zip_file.get_all_records()
        self.assertCountEqual(
            zip_file_contents, ['.data/version',
                                '.data/python_version',
                                'my_cool_resources/sekrit.txt',
                                'my_cool_resources/bar.txt',
                                'my_less_cool_resources/another_one.txt',
                                'my_less_cool_resources/foo.txt',
                                'foo/bar.py',
                                '.data/extern_modules']
        )

class ZipfileTestNoTorch(ZipfileTest):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ZipfileWriter = TorchScriptPackageZipFileWriter
        self.ZipfileReader = TorchScriptPackageZipFileReader
        self.PackageExporter = PackageExporterNoTorch
        self.PackageImporter = PackageImporterNoTorch

if __name__ == "__main__":
    run_tests()
