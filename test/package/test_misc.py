# -*- coding: utf-8 -*-
# Owner(s): ["oncall: package/deploy"]

import inspect
import os
import platform
import sys
from io import BytesIO
from pathlib import Path
from textwrap import dedent
from unittest import skipIf

from torch.package import is_from_package, PackageExporter, PackageImporter
from torch.package.package_exporter import PackagingError
from torch.testing._internal.common_utils import IS_FBCODE, IS_SANDCASTLE, run_tests, skipIfTorchDynamo

try:
    from .common import PackageTestCase
except ImportError:
    # Support the case where we run this file directly.
    from common import PackageTestCase


class TestMisc(PackageTestCase):
    """Tests for one-off or random functionality. Try not to add to this!"""

    def test_file_structure(self):
        """
        Tests package's Directory structure representation of a zip file. Ensures
        that the returned Directory prints what is expected and filters
        inputs/outputs correctly.
        """
        buffer = BytesIO()

        export_plain = dedent(
            """\
                ├── .data
                │   ├── extern_modules
                │   ├── python_version
                │   ├── serialization_id
                │   └── version
                ├── main
                │   └── main
                ├── obj
                │   └── obj.pkl
                ├── package_a
                │   ├── __init__.py
                │   └── subpackage.py
                ├── byteorder
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
                │   ├── python_version
                │   ├── serialization_id
                │   └── version
                ├── main
                │   └── main
                ├── obj
                │   └── obj.pkl
                ├── package_a
                │   ├── __init__.py
                │   └── subpackage.py
                ├── byteorder
                └── module_a.py
            """
        )

        with PackageExporter(buffer) as he:
            import module_a
            import package_a
            import package_a.subpackage

            obj = package_a.subpackage.PackageASubpackageObject()
            he.intern("**")
            he.save_module(module_a.__name__)
            he.save_module(package_a.__name__)
            he.save_pickle("obj", "obj.pkl", obj)
            he.save_text("main", "main", "my string")

        buffer.seek(0)
        hi = PackageImporter(buffer)

        file_structure = hi.file_structure()
        # remove first line from testing because WINDOW/iOS/Unix treat the buffer differently
        self.assertEqual(
            dedent("\n".join(str(file_structure).split("\n")[1:])),
            export_plain,
        )
        file_structure = hi.file_structure(include=["**/subpackage.py", "**/*.pkl"])
        self.assertEqual(
            dedent("\n".join(str(file_structure).split("\n")[1:])),
            export_include,
        )

        file_structure = hi.file_structure(exclude="**/*.storage")
        self.assertEqual(
            dedent("\n".join(str(file_structure).split("\n")[1:])),
            import_exclude,
        )

    def test_loaders_that_remap_files_work_ok(self):
        from importlib.abc import MetaPathFinder
        from importlib.machinery import SourceFileLoader
        from importlib.util import spec_from_loader

        class LoaderThatRemapsModuleA(SourceFileLoader):
            def get_filename(self, name):
                result = super().get_filename(name)
                if name == "module_a":
                    return os.path.join(os.path.dirname(result), "module_a_remapped_path.py")
                else:
                    return result

        class FinderThatRemapsModuleA(MetaPathFinder):
            def find_spec(self, fullname, path, target):
                """Try to find the original spec for module_a using all the
                remaining meta_path finders."""
                if fullname != "module_a":
                    return None
                spec = None
                for finder in sys.meta_path:
                    if finder is self:
                        continue
                    if hasattr(finder, "find_spec"):
                        spec = finder.find_spec(fullname, path, target=target)
                    elif hasattr(finder, "load_module"):
                        spec = spec_from_loader(fullname, finder)
                    if spec is not None:
                        break
                assert spec is not None and isinstance(spec.loader, SourceFileLoader)
                spec.loader = LoaderThatRemapsModuleA(spec.loader.name, spec.loader.path)
                return spec

        sys.meta_path.insert(0, FinderThatRemapsModuleA())
        # clear it from sys.modules so that we use the custom finder next time
        # it gets imported
        sys.modules.pop("module_a", None)
        try:
            buffer = BytesIO()
            with PackageExporter(buffer) as he:
                import module_a

                he.intern("**")
                he.save_module(module_a.__name__)


            buffer.seek(0)
            hi = PackageImporter(buffer)
            self.assertTrue("remapped_path" in hi.get_source("module_a"))
        finally:
            # pop it again to ensure it does not mess up other tests
            sys.modules.pop("module_a", None)
            sys.meta_path.pop(0)

    def test_python_version(self):
        """
        Tests that the current python version is stored in the package and is available
        via PackageImporter's python_version() method.
        """
        buffer = BytesIO()

        with PackageExporter(buffer) as he:
            from package_a.test_module import SimpleTest

            he.intern("**")
            obj = SimpleTest()
            he.save_pickle("obj", "obj.pkl", obj)

        buffer.seek(0)
        hi = PackageImporter(buffer)

        self.assertEqual(hi.python_version(), platform.python_version())

    @skipIf(
        IS_FBCODE or IS_SANDCASTLE,
        "Tests that use temporary files are disabled in fbcode",
    )
    def test_load_python_version_from_package(self):
        """Tests loading a package with a python version embdded"""
        importer1 = PackageImporter(
            f"{Path(__file__).parent}/package_e/test_nn_module.pt"
        )
        self.assertEqual(importer1.python_version(), "3.9.7")

    def test_file_structure_has_file(self):
        """
        Test Directory's has_file() method.
        """
        buffer = BytesIO()
        with PackageExporter(buffer) as he:
            import package_a.subpackage

            he.intern("**")
            obj = package_a.subpackage.PackageASubpackageObject()
            he.save_pickle("obj", "obj.pkl", obj)

        buffer.seek(0)

        importer = PackageImporter(buffer)
        file_structure = importer.file_structure()
        self.assertTrue(file_structure.has_file("package_a/subpackage.py"))
        self.assertFalse(file_structure.has_file("package_a/subpackage"))

    def test_exporter_content_lists(self):
        """
        Test content list API for PackageExporter's contained modules.
        """

        with PackageExporter(BytesIO()) as he:
            import package_b

            he.extern("package_b.subpackage_1")
            he.mock("package_b.subpackage_2")
            he.intern("**")
            he.save_pickle("obj", "obj.pkl", package_b.PackageBObject(["a"]))
            self.assertEqual(he.externed_modules(), ["package_b.subpackage_1"])
            self.assertEqual(he.mocked_modules(), ["package_b.subpackage_2"])
            self.assertEqual(
                he.interned_modules(),
                ["package_b", "package_b.subpackage_0.subsubpackage_0"],
            )
            self.assertEqual(he.get_rdeps("package_b.subpackage_2"), ["package_b"])

        with self.assertRaises(PackagingError) as e:
            with PackageExporter(BytesIO()) as he:
                import package_b

                he.deny("package_b")
                he.save_pickle("obj", "obj.pkl", package_b.PackageBObject(["a"]))
                self.assertEqual(he.denied_modules(), ["package_b"])

    def test_is_from_package(self):
        """is_from_package should work for objects and modules"""
        import package_a.subpackage

        buffer = BytesIO()
        obj = package_a.subpackage.PackageASubpackageObject()

        with PackageExporter(buffer) as pe:
            pe.intern("**")
            pe.save_pickle("obj", "obj.pkl", obj)

        buffer.seek(0)
        pi = PackageImporter(buffer)
        mod = pi.import_module("package_a.subpackage")
        loaded_obj = pi.load_pickle("obj", "obj.pkl")

        self.assertFalse(is_from_package(package_a.subpackage))
        self.assertTrue(is_from_package(mod))

        self.assertFalse(is_from_package(obj))
        self.assertTrue(is_from_package(loaded_obj))

    def test_inspect_class(self):
        """Should be able to retrieve source for a packaged class."""
        import package_a.subpackage

        buffer = BytesIO()
        obj = package_a.subpackage.PackageASubpackageObject()

        with PackageExporter(buffer) as pe:
            pe.intern("**")
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

    def test_dunder_package_present(self):
        """
        The attribute '__torch_package__' should be populated on imported modules.
        """
        import package_a.subpackage

        buffer = BytesIO()
        obj = package_a.subpackage.PackageASubpackageObject()

        with PackageExporter(buffer) as pe:
            pe.intern("**")
            pe.save_pickle("obj", "obj.pkl", obj)

        buffer.seek(0)
        pi = PackageImporter(buffer)
        mod = pi.import_module("package_a.subpackage")
        self.assertTrue(hasattr(mod, "__torch_package__"))

    def test_dunder_package_works_from_package(self):
        """
        The attribute '__torch_package__' should be accessible from within
        the module itself, so that packaged code can detect whether it's
        being used in a packaged context or not.
        """
        import package_a.use_dunder_package as mod

        buffer = BytesIO()

        with PackageExporter(buffer) as pe:
            pe.intern("**")
            pe.save_module(mod.__name__)

        buffer.seek(0)
        pi = PackageImporter(buffer)
        imported_mod = pi.import_module(mod.__name__)
        self.assertTrue(imported_mod.is_from_package())
        self.assertFalse(mod.is_from_package())

    @skipIfTorchDynamo("Not a suitable test for TorchDynamo")
    def test_std_lib_sys_hackery_checks(self):
        """
        The standard library performs sys.module assignment hackery which
        causes modules who do this hackery to fail on import. See
        https://github.com/pytorch/pytorch/issues/57490 for more information.
        """
        import package_a.std_sys_module_hacks

        buffer = BytesIO()
        mod = package_a.std_sys_module_hacks.Module()

        with PackageExporter(buffer) as pe:
            pe.intern("**")
            pe.save_pickle("obj", "obj.pkl", mod)

        buffer.seek(0)
        pi = PackageImporter(buffer)
        mod = pi.load_pickle("obj", "obj.pkl")
        mod()


if __name__ == "__main__":
    run_tests()
