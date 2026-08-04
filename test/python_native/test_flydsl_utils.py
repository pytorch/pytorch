# Owner(s): ["module: dsl-native-ops"]

from types import SimpleNamespace
from unittest.mock import patch

from torch._native import flydsl_utils
from torch._vendor.packaging.version import Version
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


class TestFlyDSLRuntimeProbe(TestCase):
    """The import-free probe the eager gate and Inductor both rely on.

    It must never import flydsl -- test_no_dsl_imports_after_import_torch in
    test_native_dsl_ops.py enforces that end to end; these cases pin the two
    reasons it can decline.
    """

    # The module binds these at import time (``from ... import x as _x``), so
    # they have to be patched on the module, not on importlib.
    def _with_specs(self, package_spec, mlir_spec=None):
        return (
            patch.object(flydsl_utils, "_find_spec", return_value=package_spec),
            patch.object(flydsl_utils._PathFinder, "find_spec", return_value=mlir_spec),
        )

    def test_missing_package(self):
        package, mlir = self._with_specs(None)
        with package, mlir:
            self.assertIn(
                "missing optional dependency `flydsl`",
                flydsl_utils._flydsl_runtime_unavailable_reason(),
            )

    def test_missing_mlir_submodule(self):
        spec = SimpleNamespace(submodule_search_locations=["/nonexistent"])
        package, mlir = self._with_specs(spec, mlir_spec=None)
        with package, mlir:
            self.assertIn(
                "flydsl._mlir",
                flydsl_utils._flydsl_runtime_unavailable_reason(),
            )

    def test_available_returns_no_reason(self):
        spec = SimpleNamespace(submodule_search_locations=["/nonexistent"])
        package, mlir = self._with_specs(spec, mlir_spec=object())
        with package, mlir:
            self.assertIsNone(flydsl_utils._flydsl_runtime_unavailable_reason())


class TestFlyDSLVersionGate(TestCase):
    """The gate that decides whether FlyDSL overrides register at all.

    Every case has to clear the cache twice: ``_version_is_ok`` is
    ``functools.cache``d, so a stale verdict would leak into the next case and
    a real verdict from this machine's install would leak into the first.
    """

    def setUp(self):
        super().setUp()
        flydsl_utils._version_is_ok.cache_clear()
        self.addCleanup(flydsl_utils._version_is_ok.cache_clear)

    def _with_version(self, version):
        return patch.object(
            flydsl_utils,
            "_check_runtime_available",
            return_value=(True, version),
        )

    @parametrize("version", ("0.3.0", "0.3.0.dev765", "0.3.5", "0.3.0rc1"))
    def test_supported_release_is_accepted(self, version):
        with self._with_version(Version(version)):
            self.assertTrue(flydsl_utils._version_is_ok())

    @parametrize("version", ("0.2.9", "0.4.0", "1.3.0"))
    def test_other_releases_are_rejected(self, version):
        # Exact 0.3.x match, so a future 0.4 falls back to ATen rather than
        # loading kernels written against an API it may no longer provide.
        with self._with_version(Version(version)):
            self.assertFalse(flydsl_utils._version_is_ok())

    def test_missing_version_metadata_is_rejected(self):
        # _available_version returns None when the distribution metadata is
        # absent -- e.g. flydsl imported from a source checkout on PYTHONPATH.
        # The runtime probe passes but the gate still declines, and the reason
        # has to name the missing metadata rather than read as a bad version.
        with (
            self._with_version(None),
            self.assertLogs("torch._native.flydsl_utils", level="INFO") as logs,
        ):
            self.assertFalse(flydsl_utils._version_is_ok())

        self.assertIn("metadata is missing", "\n".join(logs.output))

    def test_unsupported_version_reports_the_version(self):
        with (
            self._with_version(Version("0.4.0")),
            self.assertLogs("torch._native.flydsl_utils", level="INFO") as logs,
        ):
            self.assertFalse(flydsl_utils._version_is_ok())

        self.assertIn("0.4.0 is not supported", "\n".join(logs.output))

    def test_skip_check_overrides_unsupported_version(self):
        with (
            self._with_version(Version("0.4.0")),
            patch.object(flydsl_utils, "check_native_version_skip", return_value=True),
        ):
            self.assertTrue(flydsl_utils._version_is_ok())

    def test_skip_check_overrides_missing_version(self):
        with (
            self._with_version(None),
            patch.object(flydsl_utils, "check_native_version_skip", return_value=True),
        ):
            self.assertTrue(flydsl_utils._version_is_ok())


instantiate_parametrized_tests(TestFlyDSLVersionGate)


if __name__ == "__main__":
    run_tests()
