import inspect
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
    from common import PackageTestCase  # type: ignore


class TestMisc(PackageTestCase):
    """Tests for one-off or random functionality. Try not to add to this!"""

    def test_file_structure(self):
        filename = self.temp()

        export_plain = dedent(
            """\
                ├── main
                │   └── main
                ├── obj
                │   └── obj.pkl
                ├── package_a
                │   ├── __init__.py
                │   └── subpackage.py
                └── module_a.py
            """
        )
        export_include = dedent(
            """\
                ├── obj
                │   └── obj.pkl
                └── package_a
                    └── subpackage.py
            """
        )
        import_exclude = dedent(
            """\
                ├── .data
                │   ├── extern_modules
                │   └── version
                ├── main
                │   └── main
                ├── obj
                │   └── obj.pkl
                ├── package_a
                │   ├── __init__.py
                │   └── subpackage.py
                └── module_a.py
            """
        )

        with PackageExporter(filename, verbose=False) as he:
            import module_a
            import package_a
            import package_a.subpackage

            obj = package_a.subpackage.PackageASubpackageObject()
            he.save_module(module_a.__name__)
            he.save_module(package_a.__name__)
            he.save_pickle("obj", "obj.pkl", obj)
            he.save_text("main", "main", "my string")

            export_file_structure = he.file_structure()
            # remove first line from testing because WINDOW/iOS/Unix treat the filename differently
            self.assertEqual(
                dedent("\n".join(str(export_file_structure).split("\n")[1:])),
                export_plain,
            )
            export_file_structure = he.file_structure(
                include=["**/subpackage.py", "**/*.pkl"]
            )
            self.assertEqual(
                dedent("\n".join(str(export_file_structure).split("\n")[1:])),
                export_include,
            )

        hi = PackageImporter(filename)
        import_file_structure = hi.file_structure(exclude="**/*.storage")
        self.assertEqual(
            dedent("\n".join(str(import_file_structure).split("\n")[1:])),
            import_exclude,
        )

    @skipIf(version_info < (3, 7), "mock uses __getattr__ a 3.7 feature")
    def test_custom_requires(self):
        filename = self.temp()

        class Custom(PackageExporter):
            def require_module(self, name, dependencies):
                if name == "module_a":
                    self.save_mock_module("module_a")
                elif name == "package_a":
                    self.save_source_string(
                        "package_a", "import module_a\nresult = 5\n"
                    )
                else:
                    raise NotImplementedError("wat")

        with Custom(filename, verbose=False) as he:
            he.save_source_string("main", "import package_a\n")

        hi = PackageImporter(filename)
        hi.import_module("module_a").should_be_mocked
        bar = hi.import_module("package_a")
        self.assertEqual(bar.result, 5)

    def test_inspect_class(self):
        """Should be able to retrieve source for a packaged class."""
        import package_a.subpackage

        buffer = BytesIO()
        obj = package_a.subpackage.PackageASubpackageObject()

        with PackageExporter(buffer, verbose=False) as pe:
            pe.save_pickle("obj", "obj.pkl", obj)

        buffer.seek(0)
        pi = PackageImporter(buffer)
        packaged_class = pi.import_module(
            "package_a.subpackage"
        ).PackageASubpackageObject
        regular_class = package_a.subpackage.PackageASubpackageObject

        packaged_src = inspect.getsourcelines(packaged_class)
        regular_src = inspect.getsourcelines(regular_class)
        self.assertEqual(packaged_src, regular_src)


if __name__ == "__main__":
    run_tests()
