# Owner(s): ["oncall: package/deploy"]

from io import BytesIO
from sys import version_info
from textwrap import dedent
from unittest import skipIf

from torch.package import PackageExporter, PackageImporter
from torch.testing._internal.common_utils import run_tests

try:
    from .common import PackageTestCase
except ImportError:
    # Support the case where we run this file directly.
    from common import PackageTestCase


@skipIf(version_info < (3, 7), "ResourceReader API introduced in Python 3.7")
class TestResources(PackageTestCase):
    """Tests for access APIs for packaged resources."""

    def test_resource_reader(self):
        """Test compliance with the get_resource_reader importlib API."""
        buffer = BytesIO()
        with PackageExporter(buffer) as pe:
            # Layout looks like:
            #    package
            #    |-- one/
            #    |   |-- a.txt
            #    |   |-- b.txt
            #    |   |-- c.txt
            #    |   +-- three/
            #    |       |-- d.txt
            #    |       +-- e.txt
            #    +-- two/
            #       |-- f.txt
            #       +-- g.txt
            pe.save_text("one", "a.txt", "hello, a!")
            pe.save_text("one", "b.txt", "hello, b!")
            pe.save_text("one", "c.txt", "hello, c!")

            pe.save_text("one.three", "d.txt", "hello, d!")
            pe.save_text("one.three", "e.txt", "hello, e!")

            pe.save_text("two", "f.txt", "hello, f!")
            pe.save_text("two", "g.txt", "hello, g!")

        buffer.seek(0)
        importer = PackageImporter(buffer)

        reader_one = importer.get_resource_reader("one")
        with self.assertRaises(FileNotFoundError):
            reader_one.resource_path("a.txt")

        self.assertTrue(reader_one.is_resource("a.txt"))
        self.assertEqual(reader_one.open_resource("a.txt").getbuffer(), b"hello, a!")
        self.assertFalse(reader_one.is_resource("three"))
        reader_one_contents = list(reader_one.contents())
        self.assertSequenceEqual(
            reader_one_contents, ["a.txt", "b.txt", "c.txt", "three"]
        )

        reader_two = importer.get_resource_reader("two")
        self.assertTrue(reader_two.is_resource("f.txt"))
        self.assertEqual(reader_two.open_resource("f.txt").getbuffer(), b"hello, f!")
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

    def test_package_resource_access(self):
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
        buffer = BytesIO()
        with PackageExporter(buffer) as pe:
            pe.save_source_string("foo.bar", mod_src)
            pe.save_text("my_cool_resources", "sekrit.txt", "my sekrit plays")

        buffer.seek(0)
        importer = PackageImporter(buffer)
        self.assertEqual(
            importer.import_module("foo.bar").secret_message(), "my sekrit plays"
        )

    def test_importer_access(self):
        buffer = BytesIO()
        with PackageExporter(buffer) as he:
            he.save_text("main", "main", "my string")
            he.save_binary("main", "main_binary", b"my string")
            src = dedent(
                """\
                import importlib
                import torch_package_importer as resources

                t = resources.load_text('main', 'main')
                b = resources.load_binary('main', 'main_binary')
                """
            )
            he.save_source_string("main", src, is_package=True)
        buffer.seek(0)
        hi = PackageImporter(buffer)
        m = hi.import_module("main")
        self.assertEqual(m.t, "my string")
        self.assertEqual(m.b, b"my string")

    def test_resource_access_by_path(self):
        """
        Tests that packaged code can used importlib.resources.path.
        """
        buffer = BytesIO()
        with PackageExporter(buffer) as he:
            he.save_binary("string_module", "my_string", b"my string")
            src = dedent(
                """\
                import importlib.resources
                import string_module

                with importlib.resources.path(string_module, 'my_string') as path:
                    with open(path, mode='r', encoding='utf-8') as f:
                        s = f.read()
                """
            )
            he.save_source_string("main", src, is_package=True)
        buffer.seek(0)
        hi = PackageImporter(buffer)
        m = hi.import_module("main")
        self.assertEqual(m.s, "my string")


if __name__ == "__main__":
    run_tests()
