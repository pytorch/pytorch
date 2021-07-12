from io import BytesIO

import torch
from torch.package import (
    Importer,
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
    from common import PackageTestCase


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
        with PackageExporter(buffer) as pe:
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
        with PackageExporter(buffer) as pe:
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

    def test_ordered_importer_whichmodule(self):
        """OrderedImporter's implementation of whichmodule should try each
        underlying importer's whichmodule in order.
        """

        class DummyImporter(Importer):
            def __init__(self, whichmodule_return):
                self._whichmodule_return = whichmodule_return

            def import_module(self, module_name):
                raise NotImplementedError()

            def whichmodule(self, obj, name):
                return self._whichmodule_return

        class DummyClass:
            pass

        dummy_importer_foo = DummyImporter("foo")
        dummy_importer_bar = DummyImporter("bar")
        dummy_importer_not_found = DummyImporter(
            "__main__"
        )  # __main__ is used as a proxy for "not found" by CPython

        foo_then_bar = OrderedImporter(dummy_importer_foo, dummy_importer_bar)
        self.assertEqual(foo_then_bar.whichmodule(DummyClass(), ""), "foo")

        bar_then_foo = OrderedImporter(dummy_importer_bar, dummy_importer_foo)
        self.assertEqual(bar_then_foo.whichmodule(DummyClass(), ""), "bar")

        notfound_then_foo = OrderedImporter(
            dummy_importer_not_found, dummy_importer_foo
        )
        self.assertEqual(notfound_then_foo.whichmodule(DummyClass(), ""), "foo")

    def test_package_importer_whichmodule_no_dunder_module(self):
        """Exercise corner case where we try to pickle an object whose
        __module__ doesn't exist because it's from a C extension.
        """
        # torch.float16 is an example of such an object: it is a C extension
        # type for which there is no __module__ defined. The default pickler
        # finds it using special logic to traverse sys.modules and look up
        # `float16` on each module (see pickle.py:whichmodule).
        #
        # We must ensure that we emulate the same behavior from PackageImporter.
        my_dtype = torch.float16

        # Set up a PackageImporter which has a torch.float16 object pickled:
        buffer = BytesIO()
        with PackageExporter(buffer) as exporter:
            exporter.save_pickle("foo", "foo.pkl", my_dtype)
        buffer.seek(0)

        importer = PackageImporter(buffer)
        my_loaded_dtype = importer.load_pickle("foo", "foo.pkl")

        # Re-save a package with only our PackageImporter as the importer
        buffer2 = BytesIO()
        with PackageExporter(buffer2, importer=importer) as exporter:
            exporter.save_pickle("foo", "foo.pkl", my_loaded_dtype)

        buffer2.seek(0)

        importer2 = PackageImporter(buffer2)
        my_loaded_dtype2 = importer2.load_pickle("foo", "foo.pkl")
        self.assertIs(my_dtype, my_loaded_dtype)
        self.assertIs(my_dtype, my_loaded_dtype2)


if __name__ == "__main__":
    run_tests()
