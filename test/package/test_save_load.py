# Owner(s): ["oncall: package/deploy"]

import pickle
from io import BytesIO
from textwrap import dedent
from unittest import skipIf

from torch.package import PackageExporter, PackageImporter, sys_importer
from torch.testing._internal.common_utils import run_tests, IS_FBCODE, IS_SANDCASTLE
from torch.package.package_importer_no_torch import PackageImporter as PackageImporterNoTorch
from torch.package.package_exporter_no_torch import PackageExporter as PackageExporterNoTorch

try:
    from .common import PackageTestCase
except ImportError:
    # Support the case where we run this file directly.
    from common import PackageTestCase

from pathlib import Path

packaging_directory = Path(__file__).parent


class TestSaveLoad(PackageTestCase):
    """Core save_* and loading API tests."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.PackageImporter = PackageImporter
        self.PackageExporter = PackageExporter

    @skipIf(
        IS_FBCODE or IS_SANDCASTLE,
        "Tests that use temporary files are disabled in fbcode",
    )
    def test_saving_source(self):
        filename = self.temp()
        with self.PackageExporter(filename) as he:
            he.save_source_file("foo", str(packaging_directory / "module_a.py"))
            he.save_source_file("foodir", str(packaging_directory / "package_a"))
        hi = self.PackageImporter(filename)
        foo = hi.import_module("foo")
        s = hi.import_module("foodir.subpackage")
        self.assertEqual(foo.result, "module_a")
        self.assertEqual(s.result, "package_a.subpackage")

    @skipIf(
        IS_FBCODE or IS_SANDCASTLE,
        "Tests that use temporary files are disabled in fbcode",
    )
    def test_saving_string(self):
        filename = self.temp()
        with self.PackageExporter(filename) as he:
            src = dedent(
                """\
                import math
                the_math = math
                """
            )
            he.save_source_string("my_mod", src)
        hi = self.PackageImporter(filename)
        m = hi.import_module("math")
        import math

        self.assertIs(m, math)
        my_mod = hi.import_module("my_mod")
        self.assertIs(my_mod.math, math)

    @skipIf(
        IS_FBCODE or IS_SANDCASTLE,
        "Tests that use temporary files are disabled in fbcode",
    )
    def test_save_module(self):
        filename = self.temp()
        with self.PackageExporter(filename) as he:
            import module_a
            import package_a

            he.save_module(module_a.__name__)
            he.save_module(package_a.__name__)
        hi = self.PackageImporter(filename)
        module_a_i = hi.import_module("module_a")
        self.assertEqual(module_a_i.result, "module_a")
        self.assertIsNot(module_a, module_a_i)
        package_a_i = hi.import_module("package_a")
        self.assertEqual(package_a_i.result, "package_a")
        self.assertIsNot(package_a_i, package_a)

    def test_dunder_imports(self):
        buffer = BytesIO()
        with self.PackageExporter(buffer) as he:
            import package_b

            obj = package_b.PackageBObject
            he.intern("**")
            he.save_pickle("res", "obj.pkl", obj)

        buffer.seek(0)
        hi = self.PackageImporter(buffer)
        loaded_obj = hi.load_pickle("res", "obj.pkl")

        package_b = hi.import_module("package_b")
        self.assertEqual(package_b.result, "package_b")

        math = hi.import_module("math")
        self.assertEqual(math.__name__, "math")

        xml_sub_sub_package = hi.import_module("xml.sax.xmlreader")
        self.assertEqual(xml_sub_sub_package.__name__, "xml.sax.xmlreader")

        subpackage_1 = hi.import_module("package_b.subpackage_1")
        self.assertEqual(subpackage_1.result, "subpackage_1")

        subpackage_2 = hi.import_module("package_b.subpackage_2")
        self.assertEqual(subpackage_2.result, "subpackage_2")

        subsubpackage_0 = hi.import_module("package_b.subpackage_0.subsubpackage_0")
        self.assertEqual(subsubpackage_0.result, "subsubpackage_0")

    def test_bad_dunder_imports(self):
        """Test to ensure bad __imports__ don't cause PackageExporter to fail."""
        buffer = BytesIO()
        with self.PackageExporter(buffer) as e:
            e.save_source_string(
                "m", '__import__(these, unresolvable, "things", wont, crash, me)'
            )

    def test_save_module_binary(self):
        f = BytesIO()
        with self.PackageExporter(f) as he:
            import module_a
            import package_a

            he.save_module(module_a.__name__)
            he.save_module(package_a.__name__)
        f.seek(0)
        hi = self.PackageImporter(f)
        module_a_i = hi.import_module("module_a")
        self.assertEqual(module_a_i.result, "module_a")
        self.assertIsNot(module_a, module_a_i)
        package_a_i = hi.import_module("package_a")
        self.assertEqual(package_a_i.result, "package_a")
        self.assertIsNot(package_a_i, package_a)

    @skipIf(
        IS_FBCODE or IS_SANDCASTLE,
        "Tests that use temporary files are disabled in fbcode",
    )
    def test_pickle(self):
        import package_a.subpackage

        obj = package_a.subpackage.PackageASubpackageObject()
        obj2 = package_a.PackageAObject(obj)

        filename = self.temp()
        with self.PackageExporter(filename) as he:
            he.intern("**")
            he.save_pickle("obj", "obj.pkl", obj2)
        hi = self.PackageImporter(filename)

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

    @skipIf(
        IS_FBCODE or IS_SANDCASTLE,
        "Tests that use temporary files are disabled in fbcode",
    )
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
        with self.PackageExporter(f1) as pe:
            pe.intern("**")
            pe.save_pickle("obj", "obj.pkl", obj2)

        importer1 = self.PackageImporter(f1)
        loaded1 = importer1.load_pickle("obj", "obj.pkl")
        importer2 = self.PackageImporter(f1)
        loaded2 = importer2.load_pickle("obj", "obj.pkl")

        f2 = self.temp()

        def make_exporter():
            pe = self.PackageExporter(f2, importer=[importer1, sys_importer])
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

    def test_save_imported_module(self):
        """Saving a module that came from another PackageImporter should work."""
        import package_a.subpackage

        obj = package_a.subpackage.PackageASubpackageObject()
        obj2 = package_a.PackageAObject(obj)

        buffer = BytesIO()
        with self.PackageExporter(buffer) as exporter:
            exporter.intern("**")
            exporter.save_pickle("model", "model.pkl", obj2)

        buffer.seek(0)

        importer = self.PackageImporter(buffer)
        imported_obj2 = importer.load_pickle("model", "model.pkl")
        imported_obj2_module = imported_obj2.__class__.__module__

        # Should export without error.
        buffer2 = BytesIO()
        with self.PackageExporter(buffer2, importer=(importer, sys_importer)) as exporter:
            exporter.intern("**")
            exporter.save_module(imported_obj2_module)

    def test_save_imported_module_using_package_importer(self):
        """Exercise a corner case: re-packaging a module that uses `torch_package_importer`"""
        import package_a.use_torch_package_importer  # noqa: F401

        buffer = BytesIO()
        with self.PackageExporter(buffer) as exporter:
            exporter.intern("**")
            exporter.save_module("package_a.use_torch_package_importer")

        buffer.seek(0)

        importer = self.PackageImporter(buffer)

        # Should export without error.
        buffer2 = BytesIO()
        with self.PackageExporter(buffer2, importer=(importer, sys_importer)) as exporter:
            exporter.intern("**")
            exporter.save_module("package_a.use_torch_package_importer")

class TestSaveLoadNoTorch(TestSaveLoad):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.PackageImporter = PackageImporterNoTorch
        self.PackageExporter = PackageExporterNoTorch

if __name__ == "__main__":
    run_tests()
