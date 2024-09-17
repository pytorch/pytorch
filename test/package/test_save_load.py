# Owner(s): ["oncall: package/deploy"]

import pickle
from io import BytesIO
from textwrap import dedent

from torch.package import PackageExporter, PackageImporter, sys_importer
from torch.testing._internal.common_utils import run_tests


try:
    from .common import PackageTestCase
except ImportError:
    # Support the case where we run this file directly.
    from common import PackageTestCase

from pathlib import Path


packaging_directory = Path(__file__).parent


class TestSaveLoad(PackageTestCase):
    """Core save_* and loading API tests."""

    def test_saving_source(self):
        buffer = BytesIO()
        with PackageExporter(buffer) as he:
            he.save_source_file("foo", str(packaging_directory / "module_a.py"))
            he.save_source_file("foodir", str(packaging_directory / "package_a"))
        buffer.seek(0)
        hi = PackageImporter(buffer)
        foo = hi.import_module("foo")
        s = hi.import_module("foodir.subpackage")
        self.assertEqual(foo.result, "module_a")
        self.assertEqual(s.result, "package_a.subpackage")

    def test_saving_string(self):
        buffer = BytesIO()
        with PackageExporter(buffer) as he:
            src = dedent(
                """\
                import math
                the_math = math
                """
            )
            he.save_source_string("my_mod", src)
        buffer.seek(0)
        hi = PackageImporter(buffer)
        m = hi.import_module("math")
        import math

        self.assertIs(m, math)
        my_mod = hi.import_module("my_mod")
        self.assertIs(my_mod.math, math)

    def test_save_module(self):
        buffer = BytesIO()
        with PackageExporter(buffer) as he:
            import module_a
            import package_a

            he.save_module(module_a.__name__)
            he.save_module(package_a.__name__)
        buffer.seek(0)
        hi = PackageImporter(buffer)
        module_a_i = hi.import_module("module_a")
        self.assertEqual(module_a_i.result, "module_a")
        self.assertIsNot(module_a, module_a_i)
        package_a_i = hi.import_module("package_a")
        self.assertEqual(package_a_i.result, "package_a")
        self.assertIsNot(package_a_i, package_a)

    def test_dunder_imports(self):
        buffer = BytesIO()
        with PackageExporter(buffer) as he:
            import package_b

            obj = package_b.PackageBObject
            he.intern("**")
            he.save_pickle("res", "obj.pkl", obj)

        buffer.seek(0)
        hi = PackageImporter(buffer)
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
        with PackageExporter(buffer) as e:
            e.save_source_string(
                "m", '__import__(these, unresolvable, "things", wont, crash, me)'
            )

    def test_save_module_binary(self):
        f = BytesIO()
        with PackageExporter(f) as he:
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

    def test_pickle(self):
        import package_a.subpackage

        obj = package_a.subpackage.PackageASubpackageObject()
        obj2 = package_a.PackageAObject(obj)

        buffer = BytesIO()
        with PackageExporter(buffer) as he:
            he.intern("**")
            he.save_pickle("obj", "obj.pkl", obj2)
        buffer.seek(0)
        hi = PackageImporter(buffer)

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

    def test_pickle_long_name_with_protocol_4(self):
        import package_a.long_name

        container = []

        # Indirectly grab the function to avoid pasting a 256 character
        # function into the test
        package_a.long_name.add_function(container)

        buffer = BytesIO()
        with PackageExporter(buffer) as exporter:
            exporter.intern("**")
            exporter.save_pickle(
                "container", "container.pkl", container, pickle_protocol=4
            )

        buffer.seek(0)
        importer = PackageImporter(buffer)
        unpickled_container = importer.load_pickle("container", "container.pkl")
        self.assertIsNot(container, unpickled_container)
        self.assertEqual(len(unpickled_container), 1)
        self.assertEqual(container[0](), unpickled_container[0]())

    def test_exporting_mismatched_code(self):
        """
        If an object with the same qualified name is loaded from different
        packages, the user should get an error if they try to re-save the
        object with the wrong package's source code.
        """
        import package_a.subpackage

        obj = package_a.subpackage.PackageASubpackageObject()
        obj2 = package_a.PackageAObject(obj)

        b1 = BytesIO()
        with PackageExporter(b1) as pe:
            pe.intern("**")
            pe.save_pickle("obj", "obj.pkl", obj2)

        b1.seek(0)
        importer1 = PackageImporter(b1)
        loaded1 = importer1.load_pickle("obj", "obj.pkl")

        b1.seek(0)
        importer2 = PackageImporter(b1)
        loaded2 = importer2.load_pickle("obj", "obj.pkl")

        def make_exporter():
            pe = PackageExporter(BytesIO(), importer=[importer1, sys_importer])
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
        with PackageExporter(buffer) as exporter:
            exporter.intern("**")
            exporter.save_pickle("model", "model.pkl", obj2)

        buffer.seek(0)

        importer = PackageImporter(buffer)
        imported_obj2 = importer.load_pickle("model", "model.pkl")
        imported_obj2_module = imported_obj2.__class__.__module__

        # Should export without error.
        buffer2 = BytesIO()
        with PackageExporter(buffer2, importer=(importer, sys_importer)) as exporter:
            exporter.intern("**")
            exporter.save_module(imported_obj2_module)

    def test_save_imported_module_using_package_importer(self):
        """Exercise a corner case: re-packaging a module that uses `torch_package_importer`"""
        import package_a.use_torch_package_importer  # noqa: F401

        buffer = BytesIO()
        with PackageExporter(buffer) as exporter:
            exporter.intern("**")
            exporter.save_module("package_a.use_torch_package_importer")

        buffer.seek(0)

        importer = PackageImporter(buffer)

        # Should export without error.
        buffer2 = BytesIO()
        with PackageExporter(buffer2, importer=(importer, sys_importer)) as exporter:
            exporter.intern("**")
            exporter.save_module("package_a.use_torch_package_importer")


if __name__ == "__main__":
    run_tests()
