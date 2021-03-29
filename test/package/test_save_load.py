import pickle
from io import BytesIO
from textwrap import dedent
from unittest import skipIf

from torch.package import PackageExporter, PackageImporter, sys_importer
from torch.testing._internal.common_utils import run_tests, IS_FBCODE, IS_SANDCASTLE

try:
    from .common import PackageTestCase
except ImportError:
    # Support the case where we run this file directly.
    from common import PackageTestCase  # type: ignore

from pathlib import Path

packaging_directory = Path(__file__).parent


class TestSaveLoad(PackageTestCase):
    """Core save_* and loading API tests."""

    @skipIf(IS_FBCODE or IS_SANDCASTLE, "Tests that use temporary files are disabled in fbcode")
    def test_saving_source(self):
        filename = self.temp()
        with PackageExporter(filename, verbose=False) as he:
            he.save_source_file("foo", str(packaging_directory / "module_a.py"))
            he.save_source_file("foodir", str(packaging_directory / "package_a"))
        hi = PackageImporter(filename)
        foo = hi.import_module("foo")
        s = hi.import_module("foodir.subpackage")
        self.assertEqual(foo.result, "module_a")
        self.assertEqual(s.result, "package_a.subpackage")

    @skipIf(IS_FBCODE or IS_SANDCASTLE, "Tests that use temporary files are disabled in fbcode")
    def test_saving_string(self):
        filename = self.temp()
        with PackageExporter(filename, verbose=False) as he:
            src = dedent(
                """\
                import math
                the_math = math
                """
            )
            he.save_source_string("my_mod", src)
        hi = PackageImporter(filename)
        m = hi.import_module("math")
        import math

        self.assertIs(m, math)
        my_mod = hi.import_module("my_mod")
        self.assertIs(my_mod.math, math)

    @skipIf(IS_FBCODE or IS_SANDCASTLE, "Tests that use temporary files are disabled in fbcode")
    def test_save_module(self):
        filename = self.temp()
        with PackageExporter(filename, verbose=False) as he:
            import module_a
            import package_a

            he.save_module(module_a.__name__)
            he.save_module(package_a.__name__)
        hi = PackageImporter(filename)
        module_a_i = hi.import_module("module_a")
        self.assertEqual(module_a_i.result, "module_a")
        self.assertIsNot(module_a, module_a_i)
        package_a_i = hi.import_module("package_a")
        self.assertEqual(package_a_i.result, "package_a")
        self.assertIsNot(package_a_i, package_a)

    def test_save_module_binary(self):
        f = BytesIO()
        with PackageExporter(f, verbose=False) as he:
            import module_a
            import package_a

            he.save_module(module_a.__name__)
            he.save_module(package_a.__name__)
        f.seek(0)
        hi = PackageImporter(f)
        module_a_i = hi.import_module("module_a")
        self.assertEqual(module_a_i.result, "module_a")
        self.assertIsNot(module_a, module_a_i)
        package_a_i = hi.import_module("package_a")
        self.assertEqual(package_a_i.result, "package_a")
        self.assertIsNot(package_a_i, package_a)

    @skipIf(IS_FBCODE or IS_SANDCASTLE, "Tests that use temporary files are disabled in fbcode")
    def test_pickle(self):
        import package_a.subpackage

        obj = package_a.subpackage.PackageASubpackageObject()
        obj2 = package_a.PackageAObject(obj)

        filename = self.temp()
        with PackageExporter(filename, verbose=False) as he:
            he.save_pickle("obj", "obj.pkl", obj2)
        hi = PackageImporter(filename)

        # check we got dependencies
        sp = hi.import_module("package_a.subpackage")
        # check we didn't get other stuff
        with self.assertRaises(ImportError):
            hi.import_module("module_a")

        obj_loaded = hi.load_pickle("obj", "obj.pkl")
        self.assertIsNot(obj2, obj_loaded)
        self.assertIsInstance(obj_loaded.obj, sp.PackageASubpackageObject)
        self.assertIsNot(
            package_a.subpackage.PackageASubpackageObject, sp.PackageASubpackageObject
        )

    @skipIf(IS_FBCODE or IS_SANDCASTLE, "Tests that use temporary files are disabled in fbcode")
    def test_save_imported_module_fails(self):
        """
        Directly saving/requiring an PackageImported module should raise a specific error message.
        """
        import package_a.subpackage

        obj = package_a.subpackage.PackageASubpackageObject()
        obj2 = package_a.PackageAObject(obj)
        f1 = self.temp()
        with PackageExporter(f1, verbose=False) as pe:
            pe.save_pickle("obj", "obj.pkl", obj)

        importer1 = PackageImporter(f1)
        loaded1 = importer1.load_pickle("obj", "obj.pkl")

        f2 = self.temp()
        pe = PackageExporter(f2, verbose=False, importer=(importer1, sys_importer))
        with self.assertRaisesRegex(ModuleNotFoundError, "torch.package"):
            pe.require_module(loaded1.__module__)
        with self.assertRaisesRegex(ModuleNotFoundError, "torch.package"):
            pe.save_module(loaded1.__module__)

    @skipIf(IS_FBCODE or IS_SANDCASTLE, "Tests that use temporary files are disabled in fbcode")
    def test_exporting_mismatched_code(self):
        """
        If an object with the same qualified name is loaded from different
        packages, the user should get an error if they try to re-save the
        object with the wrong package's source code.
        """
        import package_a.subpackage

        obj = package_a.subpackage.PackageASubpackageObject()
        obj2 = package_a.PackageAObject(obj)
        f1 = self.temp()
        with PackageExporter(f1, verbose=False) as pe:
            pe.save_pickle("obj", "obj.pkl", obj2)

        importer1 = PackageImporter(f1)
        loaded1 = importer1.load_pickle("obj", "obj.pkl")
        importer2 = PackageImporter(f1)
        loaded2 = importer2.load_pickle("obj", "obj.pkl")

        f2 = self.temp()

        def make_exporter():
            pe = PackageExporter(f2, verbose=False, importer=[importer1, sys_importer])
            # Ensure that the importer finds the 'PackageAObject' defined in 'importer1' first.
            return pe

        # This should fail. The 'PackageAObject' type defined from 'importer1'
        # is not necessarily the same 'obj2's version of 'PackageAObject'.
        pe = make_exporter()
        with self.assertRaises(pickle.PicklingError):
            pe.save_pickle("obj", "obj.pkl", obj2)

        # This should also fail. The 'PackageAObject' type defined from 'importer1'
        # is not necessarily the same as the one defined from 'importer2'
        pe = make_exporter()
        with self.assertRaises(pickle.PicklingError):
            pe.save_pickle("obj", "obj.pkl", loaded2)

        # This should succeed. The 'PackageAObject' type defined from
        # 'importer1' is a match for the one used by loaded1.
        pe = make_exporter()
        pe.save_pickle("obj", "obj.pkl", loaded1)


if __name__ == "__main__":
    run_tests()
