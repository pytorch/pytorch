# Owner(s): ["module: dsl-native-ops"]

import importlib.util
import os
import subprocess
import sys
import textwrap
import uuid
from unittest.mock import patch

from torch.testing._internal.common_utils import run_tests, TestCase


def _subprocess_lastline(script, env=None):
    """Run script in a fresh interpreter and return the last line of stdout."""
    result = subprocess.check_output(
        [sys.executable, "-c", script],
        cwd=os.path.dirname(os.path.realpath(__file__)),
        text=True,
    ).strip()
    return result.rsplit("\n", 1)[-1]


def _import_module_directly(module_name, file_name):
    """Import a module directly without triggering package imports."""
    test_dir = os.path.dirname(os.path.abspath(__file__))
    pytorch_root = os.path.dirname(os.path.dirname(test_dir))
    module_path = os.path.join(pytorch_root, "torch", "_native", file_name)

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class TestNativeDSLOps(TestCase):
    """Tests for the torch._native DSL ops framework."""

    def setUp(self):
        """Clear all caches before each test to ensure test isolation."""
        try:
            # Clear function caches that might affect test behavior
            from torch._native.common_utils import (
                check_native_jit_disabled,
                check_native_version_skip,
            )

            check_native_jit_disabled.cache_clear()
            check_native_version_skip.cache_clear()

            from torch._native import cutedsl_utils, triton_utils

            triton_utils._version_is_sufficient.cache_clear()
            cutedsl_utils._version_is_ok.cache_clear()
            triton_utils.check_native_jit_disabled.cache_clear()
            cutedsl_utils.check_native_jit_disabled.cache_clear()
            triton_utils.check_native_version_skip.cache_clear()
            cutedsl_utils.check_native_version_skip.cache_clear()
        except (AttributeError, ImportError):
            # Some functions might not exist or be cached, ignore errors
            pass

    def test_consistent_helper_interface(self):
        """triton_utils and cutedsl_utils expose the same public API."""
        # Import modules directly to avoid dependency issues
        triton_utils = _import_module_directly(
            "torch._native.triton_utils", "triton_utils.py"
        )
        cutedsl_utils = _import_module_directly(
            "torch._native.cutedsl_utils", "cutedsl_utils.py"
        )

        REQUIRED_METHODS = {
            "runtime_available",
            "runtime_version",
            "register_op_override",
            "deregister_op_overrides",
        }

        for mod in (cutedsl_utils, triton_utils):
            public = {name for name in dir(mod) if not name.startswith("_")}
            self.assertTrue(
                REQUIRED_METHODS <= public,
                f"{mod.__name__} missing: {REQUIRED_METHODS - public}",
            )
            for name in REQUIRED_METHODS:
                self.assertTrue(callable(getattr(mod, name)))

        triton_public = {n for n in dir(triton_utils) if not n.startswith("_")}
        cute_public = {n for n in dir(cutedsl_utils) if not n.startswith("_")}

        self.assertEqual(triton_public, cute_public)

        for mod in (cutedsl_utils, triton_utils):
            self.assertIsInstance(mod.runtime_available(), bool)
            ver = mod.runtime_version()
            if ver is not None:
                from packaging.version import Version

                self.assertIsInstance(ver, Version)

    def test_no_dsl_imports_after_import_torch(self):
        """import torch must not transitively import DSL runtimes.

        Note: cuda.bindings may appear because importlib.util.find_spec on
        nested modules (e.g. cuda.bindings.driver) imports parent packages
        as a side-effect.  We check only the primary DSL runtimes here.
        """
        script = textwrap.dedent("""\
            import sys
            import torch
            dsl_modules = ["triton", "cutlass", "tvm_ffi"]
            leaked = [m for m in dsl_modules if m in sys.modules]
            print(repr(leaked))
        """)
        result = _subprocess_lastline(script)
        self.assertEqual(result, "[]", f"DSL modules leaked on import torch: {result}")

    def test_check_native_jit_disabled_default(self):
        """TORCH_DISABLE_NATIVE_JIT unset -> check returns False."""
        from torch._native.common_utils import check_native_jit_disabled

        with patch.dict(os.environ, {}, clear=False):
            # Ensure TORCH_DISABLE_NATIVE_JIT is not set
            os.environ.pop("TORCH_DISABLE_NATIVE_JIT", None)
            # Clear the cache so the function re-reads the environment variable
            check_native_jit_disabled.cache_clear()
            self.assertFalse(check_native_jit_disabled())

    def test_check_native_jit_disabled_set(self):
        """TORCH_DISABLE_NATIVE_JIT=1 -> check returns True."""
        from torch._native.common_utils import check_native_jit_disabled

        with patch.dict(os.environ, {"TORCH_DISABLE_NATIVE_JIT": "1"}):
            # Clear the cache so the function re-reads the environment variable
            check_native_jit_disabled.cache_clear()
            self.assertTrue(check_native_jit_disabled())

    def test_unavailable_reason_missing(self):
        """Nonexistent package -> _unavailable_reason returns a string."""
        common_utils = _import_module_directly(
            "torch._native.common_utils", "common_utils.py"
        )
        reason = common_utils._unavailable_reason(
            [("nonexistent_pkg_xyz", "nonexistent_pkg_xyz")]
        )
        self.assertIsNotNone(reason)
        self.assertIn("nonexistent_pkg_xyz", reason)

    def test_available_version(self):
        """_available_version returns a packaging.version.Version"""
        from packaging.version import Version

        common_utils = _import_module_directly(
            "torch._native.common_utils", "common_utils.py"
        )

        # Use typing_extensions which always has a clean major.minor.patch version,
        # unlike torch which may have pre-release suffixes in dev builds.
        ver = common_utils._available_version("typing_extensions")
        self.assertIsInstance(ver, Version)

    def test_registry_mechanics(self):
        """_get_or_create_library caches Library instances per (lib, dispatch_key)."""
        import torch.library

        registry = _import_module_directly("torch._native.registry", "registry.py")

        key = ("_test_native_dsl_registry", "CPU")
        registry._libs.pop(key, None)

        lib1 = registry._get_or_create_library(*key)
        self.assertIsInstance(lib1, torch.library.Library)
        lib2 = registry._get_or_create_library(*key)
        self.assertIs(lib1, lib2, "should return cached instance")

        # Different dispatch key -> different Library
        key2 = ("_test_native_dsl_registry", "CUDA")
        registry._libs.pop(key2, None)
        lib3 = registry._get_or_create_library(*key2)
        self.assertIsNot(lib1, lib3)

        # cleanup
        registry._libs.pop(key, None)
        registry._libs.pop(key2, None)

    def test_deregister_op_overrides_functionality(self):
        """deregister_op_overrides methods are callable and exist."""
        # Import modules directly to avoid dependency issues
        triton_utils = _import_module_directly(
            "torch._native.triton_utils", "triton_utils.py"
        )
        cutedsl_utils = _import_module_directly(
            "torch._native.cutedsl_utils", "cutedsl_utils.py"
        )

        # Test that deregister_op_overrides methods exist and are callable
        for mod in (triton_utils, cutedsl_utils):
            self.assertTrue(hasattr(mod, "deregister_op_overrides"))
            self.assertTrue(callable(mod.deregister_op_overrides))

        # Test that the methods can be called without error (they should be no-ops
        # when no overrides are registered)
        try:
            triton_utils.deregister_op_overrides()
            cutedsl_utils.deregister_op_overrides()
        except Exception as e:
            self.fail(f"deregister_op_overrides raised an exception: {e}")

    def test_register_op_skips_when_jit_disabled(self):
        """register_op_override does not call through when TORCH_DISABLE_NATIVE_JIT=1."""
        from torch._native import cutedsl_utils, triton_utils

        # Test the actual environment variable behavior to ensure it works
        # Set TORCH_DISABLE_NATIVE_JIT=1 and clear caches
        with patch.dict(os.environ, {"TORCH_DISABLE_NATIVE_JIT": "1"}):
            # Import and clear caches for both modules
            from torch._native.common_utils import check_native_jit_disabled

            check_native_jit_disabled.cache_clear()

            # Import functions from each module and clear their caches too
            triton_utils.check_native_jit_disabled.cache_clear()
            cutedsl_utils.check_native_jit_disabled.cache_clear()

            # Verify the function returns True
            self.assertTrue(check_native_jit_disabled())

            # Mock the registry calls to count how many times they would be called
            with patch("torch._native.registry.register_op_override") as registry_mock:
                # Use a unique operation name
                unique_op = f"test_jit_disabled_{uuid.uuid4().hex[:8]}.Tensor"
                triton_utils.register_op_override(
                    "aten", unique_op, "CPU", lambda: None
                )
                cutedsl_utils.register_op_override(
                    "aten", unique_op, "CPU", lambda: None
                )
                # Should not call the registry function at all since JIT is disabled
                self.assertEqual(registry_mock.call_count, 0)

    def test_version_skip_env_var_overrides(self):
        """TORCH_NATIVE_SKIP_VERSION_CHECK=1 allows non-blessed versions."""
        from packaging.version import Version

        from torch._native import cutedsl_utils, triton_utils

        fake_version = Version("99.99.99")

        # Set the environment variable and clear caches
        with patch.dict(os.environ, {"TORCH_NATIVE_SKIP_VERSION_CHECK": "1"}):
            # Clear caches for version check functions
            from torch._native.common_utils import check_native_version_skip

            check_native_version_skip.cache_clear()
            triton_utils._version_is_sufficient.cache_clear()
            cutedsl_utils._version_is_ok.cache_clear()
            triton_utils.check_native_version_skip.cache_clear()
            cutedsl_utils.check_native_version_skip.cache_clear()

            with (
                patch.object(
                    triton_utils,
                    "_check_runtime_available",
                    return_value=(True, fake_version),
                ),
                patch.object(
                    cutedsl_utils,
                    "_check_runtime_available",
                    return_value=(True, fake_version),
                ),
                patch.object(triton_utils, "_register_op_override_impl") as triton_mock,
                patch.object(cutedsl_utils, "_register_op_override_impl") as cute_mock,
            ):
                # Use unique operation names to avoid conflicts
                op_name = f"test_version_skip_{uuid.uuid4().hex[:8]}.Tensor"
                triton_utils.register_op_override("aten", op_name, "CPU", lambda: None)
                cutedsl_utils.register_op_override("aten", op_name, "CPU", lambda: None)
                self.assertEqual(triton_mock.call_count + cute_mock.call_count, 2)

    def test_check_native_version_skip_default(self):
        """TORCH_NATIVE_SKIP_VERSION_CHECK unset -> returns False."""
        from torch._native.common_utils import check_native_version_skip

        with patch.dict(os.environ, {}, clear=False):
            # Ensure TORCH_NATIVE_SKIP_VERSION_CHECK is not set
            os.environ.pop("TORCH_NATIVE_SKIP_VERSION_CHECK", None)
            # Clear the cache so the function re-reads the environment variable
            check_native_version_skip.cache_clear()
            self.assertFalse(check_native_version_skip())

    def test_check_native_version_skip_set(self):
        """TORCH_NATIVE_SKIP_VERSION_CHECK=1 -> returns True."""
        from torch._native.common_utils import check_native_version_skip

        with patch.dict(os.environ, {"TORCH_NATIVE_SKIP_VERSION_CHECK": "1"}):
            # Clear the cache so the function re-reads the environment variable
            check_native_version_skip.cache_clear()
            self.assertTrue(check_native_version_skip())

    def test_available_version_prerelease(self):
        """_available_version parses valid versions and rejects unparsable ones."""
        from unittest.mock import patch

        from packaging.version import Version

        common_utils = _import_module_directly(
            "torch._native.common_utils", "common_utils.py"
        )

        valid_versions = ["0.7.0rc1", "3.1.0.post1", "2.4.0a1", "1.2.3"]
        for version_str in valid_versions:
            with patch("importlib.metadata.version", return_value=version_str):
                result = common_utils._available_version("fake_package")
                self.assertEqual(
                    result,
                    Version(version_str),
                    f"_available_version({version_str!r}) = {result}",
                )

        # Completely unparsable -> None
        with patch("importlib.metadata.version", return_value="abc"):
            result = common_utils._available_version("fake_package")
            self.assertIsNone(result)


if __name__ == "__main__":
    run_tests()
