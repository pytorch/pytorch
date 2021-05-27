from tempfile import TemporaryDirectory
from unittest import skipIf

import torch
from torch.package import PackageExporter, PackageImporter
from torch.testing._internal.common_utils import (
    run_tests,
    IS_FBCODE,
    IS_SANDCASTLE,
    IS_WINDOWS,
)

try:
    from torchvision.models import resnet18

    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False
skipIfNoTorchVision = skipIf(not HAS_TORCHVISION, "no torchvision")


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
class DirectoryReaderTest(PackageTestCase):
    """Tests use of DirectoryReader as accessor for opened packages."""

    @skipIfNoTorchVision
    def test_loading_pickle(self):
        """
        Test basic saving and loading of modules and pickles from a DirectoryReader.
        """
        resnet = resnet18()

        filename = self.temp()
        with PackageExporter(filename, verbose=False) as e:
            e.intern("**")
            e.save_pickle("model", "model.pkl", resnet)

        import zipfile

        zip_file = zipfile.ZipFile(filename, "r")

        with TemporaryDirectory() as temp_dir:
            zip_file.extractall(path=temp_dir)
            importer = PackageImporter(str(Path(temp_dir) / Path(filename).name))
            dir_mod = importer.load_pickle("model", "model.pkl")
            input = torch.rand(1, 3, 224, 224)
            self.assertTrue(torch.allclose(dir_mod(input), resnet(input)))

    def test_loading_module(self):
        """
        Test basic saving and loading of a packages from a DirectoryReader.
        """
        import package_a

        filename = self.temp()
        with PackageExporter(filename, verbose=False) as e:
            e.save_module("package_a")

        import zipfile

        zip_file = zipfile.ZipFile(filename, "r")

        with TemporaryDirectory() as temp_dir:
            zip_file.extractall(path=temp_dir)
            dir_importer = PackageImporter(str(Path(temp_dir) / Path(filename).name))
            dir_mod = dir_importer.import_module("package_a")
            self.assertEqual(dir_mod.result, package_a.result)

    def test_resource_reader(self):
        """Tests DirectoryReader as the base for get_resource_reader."""
        filename = self.temp()
        with PackageExporter(filename, verbose=False) as pe:
            # Layout looks like:
            #    package
            #    ├── one/
            #    │   ├── a.txt
            #    │   ├── b.txt
            #    │   ├── c.txt
            #    │   └── three/
            #    │       ├── d.txt
            #    │       └── e.txt
            #    └── two/
            #       ├── f.txt
            #       └── g.txt
            pe.save_text("one", "a.txt", "hello, a!")
            pe.save_text("one", "b.txt", "hello, b!")
            pe.save_text("one", "c.txt", "hello, c!")

            pe.save_text("one.three", "d.txt", "hello, d!")
            pe.save_text("one.three", "e.txt", "hello, e!")

            pe.save_text("two", "f.txt", "hello, f!")
            pe.save_text("two", "g.txt", "hello, g!")

        import zipfile

        zip_file = zipfile.ZipFile(filename, "r")

        with TemporaryDirectory() as temp_dir:
            zip_file.extractall(path=temp_dir)
            importer = PackageImporter(str(Path(temp_dir) / Path(filename).name))
            reader_one = importer.get_resource_reader("one")
            with self.assertRaises(FileNotFoundError):
                reader_one.resource_path("a.txt")

            self.assertTrue(reader_one.is_resource("a.txt"))
            self.assertEqual(
                reader_one.open_resource("a.txt").getbuffer(), b"hello, a!"
            )
            self.assertFalse(reader_one.is_resource("three"))
            reader_one_contents = list(reader_one.contents())
            self.assertSequenceEqual(
                reader_one_contents, ["a.txt", "b.txt", "c.txt", "three"]
            )

            reader_two = importer.get_resource_reader("two")
            self.assertTrue(reader_two.is_resource("f.txt"))
            self.assertEqual(
                reader_two.open_resource("f.txt").getbuffer(), b"hello, f!"
            )
            reader_two_contents = list(reader_two.contents())
            self.assertSequenceEqual(reader_two_contents, ["f.txt", "g.txt"])

            reader_one_three = importer.get_resource_reader("one.three")
            self.assertTrue(reader_one_three.is_resource("d.txt"))
            self.assertEqual(
                reader_one_three.open_resource("d.txt").getbuffer(), b"hello, d!"
            )
            reader_one_three_contenst = list(reader_one_three.contents())
            self.assertSequenceEqual(reader_one_three_contenst, ["d.txt", "e.txt"])

            self.assertIsNone(importer.get_resource_reader("nonexistent_package"))

    def test_scriptobject_failure_message(self):
        """
        Test basic saving and loading of a ScriptModule in a directory.
        Currently not supported.
        """
        from package_a.test_module import ModWithTensor

        scripted_mod = torch.jit.script(ModWithTensor(torch.rand(1, 2, 3)))

        filename = self.temp()
        with PackageExporter(filename, verbose=False) as e:
            e.save_pickle("res", "mod.pkl", scripted_mod)

        import zipfile

        zip_file = zipfile.ZipFile(filename, "r")

        with self.assertRaisesRegex(
            RuntimeError,
            "Loading ScriptObjects from a PackageImporter created from a "
            "directory is not supported. Use a package archive file instead.",
        ):
            with TemporaryDirectory() as temp_dir:
                zip_file.extractall(path=temp_dir)
                dir_importer = PackageImporter(
                    str(Path(temp_dir) / Path(filename).name)
                )
                dir_mod = dir_importer.load_pickle("res", "mod.pkl")


if __name__ == "__main__":
    run_tests()
