from io import BytesIO
from sys import version_info
from textwrap import dedent
from unittest import skipIf

from torch.package import (
    DeniedModuleError,
    EmptyMatchError,
    PackageExporter,
    PackageImporter,
)
from torch.testing._internal.common_utils import run_tests

try:
    from .common import PackageTestCase
except ImportError:
    # Support the case where we run this file directly.
    from common import PackageTestCase  # type: ignore


class TestDependencyAPI(PackageTestCase):
    """Dependency management API tests.
    - mock()
    - extern()
    - deny()
    """

    def test_extern(self):
        buffer = BytesIO()
        with PackageExporter(buffer, verbose=False) as he:
            he.extern(["package_a.subpackage", "module_a"])
            he.require_module("package_a.subpackage")
            he.require_module("module_a")
            he.save_module("package_a")
        buffer.seek(0)
        hi = PackageImporter(buffer)
        import module_a
        import package_a.subpackage

        module_a_im = hi.import_module("module_a")
        hi.import_module("package_a.subpackage")
        package_a_im = hi.import_module("package_a")

        self.assertIs(module_a, module_a_im)
        self.assertIsNot(package_a, package_a_im)
        self.assertIs(package_a.subpackage, package_a_im.subpackage)

    def test_extern_glob(self):
        buffer = BytesIO()
        with PackageExporter(buffer, verbose=False) as he:
            he.extern(["package_a.*", "module_*"])
            he.save_module("package_a")
            he.save_source_string(
                "test_module",
                dedent(
                    """\
                    import package_a.subpackage
                    import module_a
                    """
                ),
            )
        buffer.seek(0)
        hi = PackageImporter(buffer)
        import module_a
        import package_a.subpackage

        module_a_im = hi.import_module("module_a")
        hi.import_module("package_a.subpackage")
        package_a_im = hi.import_module("package_a")

        self.assertIs(module_a, module_a_im)
        self.assertIsNot(package_a, package_a_im)
        self.assertIs(package_a.subpackage, package_a_im.subpackage)

    def test_extern_glob_allow_empty(self):
        """
        Test that an error is thrown when a extern glob is specified with allow_empty=True
        and no matching module is required during packaging.
        """
        buffer = BytesIO()
        with self.assertRaisesRegex(EmptyMatchError, r"did not match any modules"):
            with PackageExporter(buffer, verbose=False) as exporter:
                exporter.extern(include=["package_a.*"], allow_empty=False)
                exporter.save_module("package_b.subpackage")

    def test_deny(self):
        """
        Test marking packages as "deny" during export.
        """
        buffer = BytesIO()

        with self.assertRaisesRegex(
            DeniedModuleError,
            "required during packaging but has been explicitly blocklisted",
        ):
            with PackageExporter(buffer, verbose=False) as exporter:
                exporter.deny(["package_a.subpackage", "module_a"])
                exporter.require_module("package_a.subpackage")

    def test_deny_glob(self):
        """
        Test marking packages as "deny" using globs instead of package names.
        """
        buffer = BytesIO()
        with self.assertRaisesRegex(
            DeniedModuleError,
            "required during packaging but has been explicitly blocklisted",
        ):
            with PackageExporter(buffer, verbose=False) as exporter:
                exporter.deny(["package_a.*", "module_*"])
                exporter.save_source_string(
                    "test_module",
                    dedent(
                        """\
                        import package_a.subpackage
                        import module_a
                        """
                    ),
                )

    @skipIf(version_info < (3, 7), "mock uses __getattr__ a 3.7 feature")
    def test_mock(self):
        buffer = BytesIO()
        with PackageExporter(buffer, verbose=False) as he:
            he.mock(["package_a.subpackage", "module_a"])
            he.save_module("package_a")
            he.require_module("package_a.subpackage")
            he.require_module("module_a")
        buffer.seek(0)
        hi = PackageImporter(buffer)
        import package_a.subpackage

        _ = package_a.subpackage
        import module_a

        _ = module_a

        m = hi.import_module("package_a.subpackage")
        r = m.result
        with self.assertRaisesRegex(NotImplementedError, "was mocked out"):
            r()

    @skipIf(version_info < (3, 7), "mock uses __getattr__ a 3.7 feature")
    def test_mock_glob(self):
        buffer = BytesIO()
        with PackageExporter(buffer, verbose=False) as he:
            he.mock(["package_a.*", "module*"])
            he.save_module("package_a")
            he.save_source_string(
                "test_module",
                dedent(
                    """\
                    import package_a.subpackage
                    import module_a
                    """
                ),
            )
        buffer.seek(0)
        hi = PackageImporter(buffer)
        import package_a.subpackage

        _ = package_a.subpackage
        import module_a

        _ = module_a

        m = hi.import_module("package_a.subpackage")
        r = m.result
        with self.assertRaisesRegex(NotImplementedError, "was mocked out"):
            r()

    def test_mock_glob_allow_empty(self):
        """
        Test that an error is thrown when a mock glob is specified with allow_empty=True
        and no matching module is required during packaging.
        """
        buffer = BytesIO()
        with self.assertRaisesRegex(EmptyMatchError, r"did not match any modules"):
            with PackageExporter(buffer, verbose=False) as exporter:
                exporter.mock(include=["package_a.*"], allow_empty=False)
                exporter.save_module("package_b.subpackage")

    def test_module_glob(self):
        from torch.package.package_exporter import _GlobGroup

        def check(include, exclude, should_match, should_not_match):
            x = _GlobGroup(include, exclude)
            for e in should_match:
                self.assertTrue(x.matches(e))
            for e in should_not_match:
                self.assertFalse(x.matches(e))

        check(
            "torch.*",
            [],
            ["torch.foo", "torch.bar"],
            ["tor.foo", "torch.foo.bar", "torch"],
        )
        check(
            "torch.**",
            [],
            ["torch.foo", "torch.bar", "torch.foo.bar", "torch"],
            ["what.torch", "torchvision"],
        )
        check("torch.*.foo", [], ["torch.w.foo"], ["torch.hi.bar.baz"])
        check(
            "torch.**.foo", [], ["torch.w.foo", "torch.hi.bar.foo"], ["torch.f.foo.z"]
        )
        check("torch*", [], ["torch", "torchvision"], ["torch.f"])
        check(
            "torch.**",
            ["torch.**.foo"],
            ["torch", "torch.bar", "torch.barfoo"],
            ["torch.foo", "torch.some.foo"],
        )
        check("**.torch", [], ["torch", "bar.torch"], ["visiontorch"])

    @skipIf(version_info < (3, 7), "mock uses __getattr__ a 3.7 feature")
    def test_pickle_mocked(self):
        import package_a.subpackage

        obj = package_a.subpackage.PackageASubpackageObject()
        obj2 = package_a.PackageAObject(obj)

        buffer = BytesIO()
        with PackageExporter(buffer, verbose=False) as he:
            he.mock(include="package_a.subpackage")
            he.save_pickle("obj", "obj.pkl", obj2)

        buffer.seek(0)

        hi = PackageImporter(buffer)
        with self.assertRaises(NotImplementedError):
            hi.load_pickle("obj", "obj.pkl")


if __name__ == "__main__":
    run_tests()
