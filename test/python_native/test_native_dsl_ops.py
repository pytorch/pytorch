# Owner(s): ["module: dsl-native-ops"]

import contextlib
import importlib.util
import os
import subprocess
import sys
import textwrap
import uuid
from importlib.metadata import PackageNotFoundError
from unittest.mock import patch

from torch._native import triton_utils
from torch._vendor.packaging.version import Version
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    skipIfNoCuteDSL,
    TestCase,
)


def _subprocess_lastline(script, env=None):
    """Run script in a fresh interpreter and return the last line of stdout."""
    result = subprocess.check_output(
        [sys.executable, "-c", script],
        cwd=os.path.dirname(os.path.realpath(__file__)),
        text=True,
    ).strip()
    return result.rsplit("\n", 1)[-1]


# Read from the lookup rather than repeated here, so that a name added there is
# covered rather than silently untested. `triton` is excluded: it is the one
# name the lookup resolved before the rest were handled.
_WHEEL_DISTRIBUTIONS = tuple(
    name for name in triton_utils._TRITON_DISTRIBUTIONS if name != "triton"
)

# Where the importable `triton` resolves to, for the tests that care which
# distribution installed it.
_MODULE_ORIGIN = "/site-packages/triton/__init__.py"


def _triton_installed(versions):
    """Patch the per-distribution version lookup with `versions`.

    A distribution absent from the mapping is not installed.
    """
    return patch.object(
        triton_utils,
        "_available_version",
        side_effect=lambda distribution: (
            Version(versions[distribution]) if distribution in versions else None
        ),
    )


def _triton_provided_by(*distributions, raises=False):
    """Patch the sys.path scan to report `distributions` for the module.

    With no distributions the module is absent from the mapping entirely, as it
    is from the real scan: that only ever creates a key by appending to it, so
    it cannot report a module with an empty provider list.
    """
    kwargs = (
        {"side_effect": RuntimeError("unreadable metadata")}
        if raises
        else {"return_value": {"triton": list(distributions)} if distributions else {}}
    )
    return patch.object(triton_utils, "_packages_distributions", **kwargs)


class _InstalledFile:
    def __init__(self, path):
        self._path = path

    def locate(self):
        return self._path


class _InstalledDistribution:
    def __init__(self, paths):
        # None is a distribution with no RECORD, which reports no files at all.
        self.files = None if paths is None else [_InstalledFile(p) for p in paths]


def _triton_module_at(origin):
    """Patch the resolved location of the importable module."""
    return patch.object(triton_utils, "_module_origin", return_value=origin)


def _triton_records(files_by_distribution):
    """Patch distribution metadata with the files each one installed.

    A distribution absent from the mapping is not installed; one mapped to None
    installed no RECORD, so what it owns cannot be decided.
    """

    def lookup(name):
        if name not in files_by_distribution:
            raise PackageNotFoundError(name)
        return _InstalledDistribution(files_by_distribution[name])

    return patch.object(triton_utils, "_distribution", side_effect=lookup)


def _rejects_a_nameless_distribution(distribution):
    """Stand in for the version lookup, which rejects a `None` name.

    A dist-info whose METADATA carries no `Name` is reported by the scan as a
    `None` provider, and `importlib.metadata.version(None)` raises.
    """
    if distribution is None:
        raise ValueError("A distribution name is required.")
    return None


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
        super().setUp()
        self._cache_functions_to_clear = [
            (
                "torch._native.common_utils",
                ["check_native_jit_disabled", "check_native_version_skip"],
            ),
            (
                "torch._native.triton_utils",
                [
                    "_check_runtime_available",
                    "_version_is_sufficient",
                    "check_native_jit_disabled",
                    "check_native_version_skip",
                ],
            ),
            (
                "torch._native.cutedsl_utils",
                [
                    "_version_is_ok",
                    "check_native_jit_disabled",
                    "check_native_version_skip",
                ],
            ),
            (
                "torch._native.helion_utils",
                [
                    "_version_is_sufficient",
                    "check_native_jit_disabled",
                    "check_native_version_skip",
                ],
            ),
            (
                "torch._native.flydsl_utils",
                [
                    "_check_runtime_available",
                    "_resolve_rocm_arch",
                    "_version_is_ok",
                    "check_native_jit_disabled",
                    "check_native_version_skip",
                ],
            ),
        ]
        self._clear_function_caches()

    def _clear_function_caches(self):
        """Helper method to clear function caches with error handling."""
        for module_name, function_names in self._cache_functions_to_clear:
            try:
                module = __import__(module_name, fromlist=function_names)
                for func_name in function_names:
                    if hasattr(module, func_name):
                        getattr(module, func_name).cache_clear()
            except (AttributeError, ImportError):
                # Some functions might not exist or be cached, ignore errors
                pass

    def test_consistent_helper_interface(self):
        """Test all registered DSL utils expose consistent public APIs."""
        from torch.testing._internal.common_utils import get_all_dsls

        # Automatically discover all registered DSLs
        dsl_names = get_all_dsls()
        if not dsl_names:
            # Fallback to hardcoded list if registry not available
            dsl_names = ["triton", "cutedsl", "helion"]

        modules_info = [
            (f"{dsl}_utils.py", f"torch._native.{dsl}_utils") for dsl in dsl_names
        ]

        # Import modules directly to avoid dependency issues
        modules = {}
        for file_name, module_name in modules_info:
            modules[module_name] = _import_module_directly(module_name, file_name)

        required_methods = {
            "runtime_available",
            "runtime_version",
            "register_op_override",
            "deregister_op_overrides",
        }

        # Test each module has required methods and they're callable
        public_apis = {}
        for module_name, mod in modules.items():
            with self.subTest(module=module_name, test="required_methods"):
                public = {name for name in dir(mod) if not name.startswith("_")}
                public_apis[module_name] = public

                self.assertTrue(
                    required_methods <= public,
                    lambda msg: f"{msg}\n{module_name} missing: {required_methods - public}",
                )

                for method_name in required_methods:
                    with self.subTest(module=module_name, method=method_name):
                        self.assertTrue(callable(getattr(mod, method_name)))

        # Test modules expose identical public APIs
        api_sets = list(public_apis.values())
        if len(api_sets) > 1:
            for i, api_set in enumerate(api_sets[1:], 1):
                self.assertEqual(
                    api_sets[0],
                    api_set,
                    lambda msg: f"{msg}\nModule {i} should have identical public API to module 0",
                )

        # Test runtime functions return expected types
        for module_name, mod in modules.items():
            with self.subTest(module=module_name, test="runtime_functions"):
                # runtime_available should return bool
                self.assertIsInstance(mod.runtime_available(), bool)

                # runtime_version should return Version or None
                ver = mod.runtime_version()
                if ver is not None:
                    from torch._vendor.packaging.version import Version

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
            dsl_modules = ["triton", "cutlass", "tvm_ffi", "helion", "flydsl"]
            leaked = [m for m in dsl_modules if m in sys.modules]
            print(repr(leaked))
        """)
        result = _subprocess_lastline(script)
        self.assertEqual(
            result,
            "[]",
            lambda msg: f"{msg}\nDSL modules leaked on import torch: {result}",
        )

    def test_no_external_packaging_dependency(self):
        """torch._native must not import the external `packaging` package.

        It should use the vendored copy at torch._vendor.packaging instead.
        This guards against ModuleNotFoundError in environments where the
        external `packaging` is not installed (e.g. torchvision Windows CI).
        """
        script = textwrap.dedent("""\
            import sys
            # Remove external packaging from sys.modules if already loaded
            for mod_name in list(sys.modules):
                if mod_name == "packaging" or mod_name.startswith("packaging."):
                    del sys.modules[mod_name]
            # Block external packaging from being imported
            import importlib.abc
            import importlib.machinery
            class BlockPackaging(importlib.abc.MetaPathFinder):
                def find_module(self, fullname, path=None):
                    if fullname == "packaging" or fullname.startswith("packaging."):
                        return self
                def load_module(self, fullname):
                    raise ImportError(f"External {fullname} is blocked")
            sys.meta_path.insert(0, BlockPackaging())
            import torch
            print("OK")
        """)
        result = _subprocess_lastline(script)
        self.assertEqual(result, "OK")

    @parametrize("env_value, expected", [(None, False), ("1", True)])
    def test_check_native_jit_disabled_environment_variable(self, env_value, expected):
        """Test TORCH_DISABLE_NATIVE_JIT environment variable behavior."""
        from torch._native.common_utils import check_native_jit_disabled

        if env_value is None:
            os.environ.pop("TORCH_DISABLE_NATIVE_JIT", None)
        else:
            os.environ["TORCH_DISABLE_NATIVE_JIT"] = env_value

        try:
            # Clear cache so function re-reads environment variable
            check_native_jit_disabled.cache_clear()
            self.assertEqual(check_native_jit_disabled(), expected)
        finally:
            # Clean up environment variable
            os.environ.pop("TORCH_DISABLE_NATIVE_JIT", None)

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

    def test_available_version_parsing(self):
        """Test _available_version parses various version formats and handles invalid ones."""
        from torch._vendor.packaging.version import Version

        common_utils = _import_module_directly(
            "torch._native.common_utils", "common_utils.py"
        )

        # Test with real package that has clean version
        ver = common_utils._available_version("typing_extensions")
        self.assertIsInstance(ver, Version)

        # Test various version format scenarios
        version_scenarios = [
            ("0.7.0rc1", Version("0.7.0rc1"), "pre-release version"),
            ("3.1.0.post1", Version("3.1.0.post1"), "post-release version"),
            ("2.4.0a1", Version("2.4.0a1"), "alpha version"),
            ("1.2.3", Version("1.2.3"), "standard version"),
            ("abc", None, "invalid version string"),
        ]

        for version_str, expected_result, description in version_scenarios:
            with self.subTest(version=version_str, scenario=description):
                with patch("importlib.metadata.version", return_value=version_str):
                    result = common_utils._available_version("fake_package")
                    self.assertEqual(
                        result,
                        expected_result,
                        lambda msg: f"{msg}\n_available_version({version_str!r}) = {result}",
                    )

    def test_registry_mechanics(self):
        """_get_or_create_library caches Library instances per dispatch_key."""
        import torch._native.registry as registry
        import torch.library

        # Save original state for restoration
        original_libs = dict(registry._libs)
        original_filter_state = (
            set(registry._filter_state._dsl_names),
            set(registry._filter_state._op_symbols),
            set(registry._filter_state._dispatch_keys),
        )

        try:
            cpu_key = ("_native", "CPU")
            cuda_key = ("_native", "CUDA")
            registry._libs.pop(cpu_key, None)
            registry._libs.pop(cuda_key, None)

            lib1 = registry._get_or_create_library("CPU")
            self.assertIsInstance(lib1, torch.library.Library)
            lib2 = registry._get_or_create_library("CPU")
            self.assertIs(lib1, lib2, "should return cached instance")

            # Different dispatch key -> different Library
            lib3 = registry._get_or_create_library("CUDA")
            self.assertIsNot(lib1, lib3)

            # cleanup
            registry._libs.pop(cpu_key, None)
            registry._libs.pop(cuda_key, None)
        finally:
            # Restore original registry state
            registry._libs.clear()
            registry._libs.update(original_libs)

            # Restore filter state
            filter_state = registry._filter_state
            filter_state._dsl_names.clear()
            filter_state._op_symbols.clear()
            filter_state._dispatch_keys.clear()
            filter_state._dsl_names.update(original_filter_state[0])
            filter_state._op_symbols.update(original_filter_state[1])
            filter_state._dispatch_keys.update(original_filter_state[2])

    def test_deregister_op_overrides_functionality(self):
        """Test deregister_op_overrides methods exist, are callable, and work correctly."""
        modules_to_test = [
            ("triton_utils.py", "torch._native.triton_utils"),
            ("cutedsl_utils.py", "torch._native.cutedsl_utils"),
            ("helion_utils.py", "torch._native.helion_utils"),
        ]

        # Use the preserve_filter_state context manager pattern
        from torch._native.registry import _filter_state

        original_filter_state = (
            set(_filter_state._dsl_names),
            set(_filter_state._op_symbols),
            set(_filter_state._dispatch_keys),
        )

        try:
            for file_name, module_name in modules_to_test:
                with self.subTest(module=module_name):
                    mod = _import_module_directly(module_name, file_name)

                    # Test method exists and is callable
                    self.assertTrue(hasattr(mod, "deregister_op_overrides"))
                    self.assertTrue(callable(mod.deregister_op_overrides))

                    # Test method can be called without error (should be no-op when no overrides registered)
                    try:
                        mod.deregister_op_overrides()
                    except Exception as e:
                        self.fail(
                            f"deregister_op_overrides on {module_name} raised exception: {e}"
                        )
        finally:
            # Restore original filter state
            _filter_state._dsl_names.clear()
            _filter_state._op_symbols.clear()
            _filter_state._dispatch_keys.clear()
            _filter_state._dsl_names.update(original_filter_state[0])
            _filter_state._op_symbols.update(original_filter_state[1])
            _filter_state._dispatch_keys.update(original_filter_state[2])

    def test_register_op_skips_when_jit_disabled(self):
        """register_op_override does not call through when TORCH_DISABLE_NATIVE_JIT=1."""
        from torch._native import cutedsl_utils, helion_utils, triton_utils

        # Test the actual environment variable behavior to ensure it works
        # Set TORCH_DISABLE_NATIVE_JIT=1 and clear caches
        with patch.dict(os.environ, {"TORCH_DISABLE_NATIVE_JIT": "1"}):
            # Import and clear caches for both modules
            from torch._native.common_utils import check_native_jit_disabled

            check_native_jit_disabled.cache_clear()

            # Import functions from each module and clear their caches too
            triton_utils.check_native_jit_disabled.cache_clear()
            cutedsl_utils.check_native_jit_disabled.cache_clear()
            helion_utils.check_native_jit_disabled.cache_clear()

            # Verify the function returns True
            self.assertTrue(check_native_jit_disabled())

            with (
                patch.object(triton_utils, "_register_op_override_impl") as triton_mock,
                patch.object(
                    cutedsl_utils, "_register_op_override_impl"
                ) as cutedsl_mock,
                patch.object(helion_utils, "_register_op_override_impl") as helion_mock,
            ):
                # Use a unique operation name
                unique_op = f"test_jit_disabled_{uuid.uuid4().hex[:8]}.Tensor"
                triton_utils.register_op_override(
                    "aten", unique_op, "CPU", lambda *a, **k: True, lambda: None
                )
                cutedsl_utils.register_op_override(
                    "aten", unique_op, "CPU", lambda *a, **k: True, lambda: None
                )
                helion_utils.register_op_override(
                    "aten", unique_op, "CPU", lambda *a, **k: True, lambda: None
                )
                self.assertEqual(triton_mock.call_count, 0)
                self.assertEqual(cutedsl_mock.call_count, 0)
                self.assertEqual(helion_mock.call_count, 0)

    def test_helion_availability_requires_supported_backend_and_version(self):
        from torch._native import helion_utils
        from torch._vendor.packaging.version import Version

        with patch.dict(os.environ, {"HELION_BACKEND": "metal"}):
            helion_utils._check_runtime_available.cache_clear()
            helion_utils._version_is_sufficient.cache_clear()
            self.assertFalse(helion_utils.runtime_available())

        helion_utils._check_runtime_available.cache_clear()
        helion_utils._version_is_sufficient.cache_clear()
        with patch.object(
            helion_utils,
            "_check_runtime_available",
            return_value=(True, Version("1.2.0")),
        ):
            self.assertTrue(helion_utils.runtime_available())
            self.assertTrue(helion_utils._version_is_sufficient())

        helion_utils._check_runtime_available.cache_clear()
        helion_utils._version_is_sufficient.cache_clear()
        with patch.object(
            helion_utils,
            "_check_runtime_available",
            return_value=(True, Version("1.0.0")),
        ):
            self.assertTrue(helion_utils.runtime_available())
            self.assertFalse(helion_utils._version_is_sufficient())

    def test_version_skip_env_var_overrides(self):
        """TORCH_NATIVE_SKIP_VERSION_CHECK=1 allows non-blessed versions."""
        from torch._vendor.packaging.version import Version

        fake_version = Version("1.0.0")

        # Set the environment variable and clear caches
        with patch.dict(os.environ, {"TORCH_NATIVE_SKIP_VERSION_CHECK": "1"}):
            # Import fresh modules to avoid cached state
            from torch._native import cutedsl_utils, helion_utils, triton_utils
            from torch._native.common_utils import check_native_version_skip

            # Clear all relevant caches to ensure clean state
            check_native_version_skip.cache_clear()

            utils = (triton_utils, cutedsl_utils, helion_utils)
            op_name = f"test_version_skip_{uuid.uuid4().hex[:8]}.Tensor"

            for module in utils:
                # Clear cached lookups so the patched runtime takes effect.
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if hasattr(attr, "cache_clear"):
                        attr.cache_clear()

                with (
                    patch.object(
                        module,
                        "_check_runtime_available",
                        return_value=(True, fake_version),
                    ),
                    patch.object(module, "_register_op_override_impl") as mock,
                ):
                    module.register_op_override(
                        "aten", op_name, "CPU", lambda *a, **k: True, lambda: None
                    )
                    self.assertEqual(
                        mock.call_count,
                        1,
                        f"{module.__name__}: impl not called under skip flag",
                    )

    @parametrize("env_value, expected", [(None, False), ("1", True)])
    def test_check_native_version_skip_environment_variable(self, env_value, expected):
        """Test TORCH_NATIVE_SKIP_VERSION_CHECK environment variable behavior."""
        from torch._native.common_utils import check_native_version_skip

        if env_value is None:
            os.environ.pop("TORCH_NATIVE_SKIP_VERSION_CHECK", None)
        else:
            os.environ["TORCH_NATIVE_SKIP_VERSION_CHECK"] = env_value

        try:
            # Clear cache so function re-reads environment variable
            check_native_version_skip.cache_clear()
            self.assertEqual(check_native_version_skip(), expected)
        finally:
            # Clean up environment variable
            os.environ.pop("TORCH_NATIVE_SKIP_VERSION_CHECK", None)

    def test_dsl_registry_functionality(self):
        """Test that DSL registry works correctly"""
        from torch.testing._internal.common_utils import (
            get_all_dsls,
            get_available_dsls,
            is_dsl_available,
        )

        # Test registry returns expected DSLs
        all_dsls = get_all_dsls()
        self.assertIsInstance(all_dsls, list)
        self.assertIn("triton", all_dsls)
        self.assertIn("cutedsl", all_dsls)
        self.assertIn("helion", all_dsls)

        # Test available DSLs are subset of all DSLs
        available_dsls = get_available_dsls()
        self.assertIsInstance(available_dsls, list)
        for dsl in available_dsls:
            self.assertIn(dsl, all_dsls)

        # Test availability check function
        for dsl in all_dsls:
            availability = is_dsl_available(dsl)
            self.assertIsInstance(availability, bool)
            # If DSL is in available list, it should return True
            if dsl in available_dsls:
                self.assertTrue(availability)

    def test_dsl_test_helpers(self):
        """Test that DSL test helper decorators work"""
        from torch.testing._internal.common_utils import (
            skipIfDSLUnavailable,
            skipIfNoHelionDSL,
            skipIfNoTritonDSL,
            skipUnlessDSLAvailable,
        )

        # Test that decorators are callable
        self.assertTrue(callable(skipIfNoTritonDSL))
        self.assertTrue(callable(skipIfNoCuteDSL))
        self.assertTrue(callable(skipIfNoHelionDSL))
        self.assertTrue(callable(skipIfDSLUnavailable))
        self.assertTrue(callable(skipUnlessDSLAvailable))

        # Test dynamic decorators can be called
        try:
            decorator1 = skipIfDSLUnavailable("nonexistent_dsl")
            decorator2 = skipUnlessDSLAvailable("triton")
            self.assertTrue(callable(decorator1))
            self.assertTrue(callable(decorator2))
        except Exception as e:
            self.fail(f"Dynamic DSL decorators failed: {e}")

    def test_cache_invalidation_after_re_registration(self):
        """Test that caches are properly invalidated when DSLs are re-registered"""
        from unittest.mock import Mock

        from torch._native.dsl_registry import DSLRegistry

        # Create a fresh registry for this test
        registry = DSLRegistry()

        # Create mock DSL modules
        mock_dsl_1 = Mock()
        mock_dsl_1.runtime_available.return_value = False  # Initially unavailable
        mock_dsl_1.runtime_version.return_value = None

        mock_dsl_2 = Mock()
        mock_dsl_2.runtime_available.return_value = True  # Available
        mock_dsl_2.runtime_version.return_value = None

        # Register first DSL and cache results
        registry.register_dsl("test_cache_dsl", mock_dsl_1)
        initial_available = registry.is_dsl_available("test_cache_dsl")
        initial_list = registry.list_available_dsls()

        self.assertFalse(initial_available)
        self.assertNotIn("test_cache_dsl", initial_list)

        # Re-register with different module that is available
        registry.register_dsl("test_cache_dsl", mock_dsl_2)

        # Verify cache was invalidated and new results are returned
        new_available = registry.is_dsl_available("test_cache_dsl")
        new_list = registry.list_available_dsls()

        self.assertTrue(
            new_available, "Cache should be invalidated and return new result"
        )
        self.assertIn(
            "test_cache_dsl",
            new_list,
            "Available DSLs list should reflect new registration",
        )

    def test_incomplete_protocol_implementation(self):
        """Test that registration fails when module doesn't implement required protocol methods"""
        from torch._native.dsl_registry import DSLRegistry

        # Create a fresh registry for this test
        registry = DSLRegistry()

        # Create an object missing required protocol methods (not using Mock)
        class IncompleteModule:
            def runtime_available(self):
                return True

            # Missing: runtime_version, register_op_override, deregister_op_overrides

        incomplete_module = IncompleteModule()

        # Attempt to register should raise TypeError due to missing methods
        with self.assertRaises(TypeError) as cm:
            registry.register_dsl("incomplete_dsl", incomplete_module)

        self.assertIn("missing required methods", str(cm.exception))
        self.assertIn("runtime_version", str(cm.exception))
        self.assertIn("register_op_override", str(cm.exception))

        # Verify DSL was not registered
        self.assertNotIn("incomplete_dsl", registry.list_all_dsls())


class TestTritonDistributionDiscovery(TestCase):
    """Which distribution answers for the importable `triton` module.

    A name this lookup cannot resolve reports no version, and no version fails
    the gate below -- so a working Triton install stops registering ops, with
    nothing in the output naming the wheel as the reason.
    """

    def setUp(self):
        super().setUp()
        # Answer the ownership question the same way everywhere by default, so
        # that only the cases below decide it, rather than this machine's own
        # Triton install.
        for default in (_triton_module_at(_MODULE_ORIGIN), _triton_records({})):
            default.start()
            self.addCleanup(default.stop)

    def test_distribution_named_after_the_module_answers_directly(self):
        with _triton_installed({"triton": "3.7.1"}), _triton_provided_by() as scan:
            self.assertEqual(triton_utils._available_triton_version(), Version("3.7.1"))
        # Not merely an optimization: the scan reads the metadata of every
        # distribution on sys.path, on the import path of every torch process.
        scan.assert_not_called()

    @parametrize("distribution", _WHEEL_DISTRIBUTIONS)
    def test_wheel_distribution_names_are_resolved(self, distribution):
        with (
            _triton_installed({distribution: "3.7.1"}),
            _triton_provided_by(distribution),
        ):
            self.assertEqual(triton_utils._available_triton_version(), Version("3.7.1"))

    def test_provider_without_a_version_falls_through_to_the_next(self):
        with (
            _triton_installed({"pytorch-triton-rocm": "3.7.1"}),
            _triton_provided_by("fbtriton", "pytorch-triton-rocm"),
        ):
            self.assertEqual(triton_utils._available_triton_version(), Version("3.7.1"))

    def test_module_owned_by_no_distribution_reports_no_version(self):
        # A source or editable install: importable, with no metadata to read a
        # version from.
        with (
            _triton_installed({}),
            _triton_provided_by(),
            self.assertLogs("torch._native.triton_utils", level="INFO") as logs,
        ):
            self.assertIsNone(triton_utils._available_triton_version())

        self.assertIn("no installed distribution", "\n".join(logs.output))

    def test_provider_that_reports_no_version_exhausts_the_candidates(self):
        # A wheel owns the module but its metadata carries no parseable
        # version, so every candidate the scan named is tried and rejected.
        with (
            _triton_installed({}),
            _triton_provided_by("pytorch-triton-rocm"),
            self.assertLogs("torch._native.triton_utils", level="INFO") as logs,
        ):
            self.assertIsNone(triton_utils._available_triton_version())

        self.assertIn("no installed distribution", "\n".join(logs.output))

    def test_unreadable_metadata_declines_instead_of_raising(self):
        # This runs while the overrides register, during `import torch`, where
        # an exception would take the process down.
        with (
            _triton_installed({}),
            _triton_provided_by(raises=True) as scan,
            self.assertLogs("torch._native.triton_utils", level="WARNING") as logs,
        ):
            self.assertIsNone(triton_utils._available_triton_version())

        scan.assert_called_once()
        self.assertIn("will not register", "\n".join(logs.output))

    def test_stale_distribution_loses_to_the_one_that_owns_the_module(self):
        # `triton` was installed, then overwritten by another provider whose
        # uninstall left the dist-info behind. It still reports a version, and
        # answering with it disables the ops against a working 3.7.1 install.
        with (
            _triton_installed({"triton": "3.2.0", "pytorch-triton-rocm": "3.7.1"}),
            _triton_records(
                {
                    "triton": ["/site-packages/triton/removed.py"],
                    "pytorch-triton-rocm": [_MODULE_ORIGIN],
                }
            ),
            _triton_provided_by("triton", "pytorch-triton-rocm"),
        ):
            self.assertEqual(triton_utils._available_triton_version(), Version("3.7.1"))

    def test_distribution_published_under_an_unlisted_name_is_scanned_for(self):
        # The list is an optimization, not the set of answers: a name it does
        # not carry still resolves, through the scan.
        with (
            _triton_installed({"triton-nightly": "3.7.1"}),
            _triton_records({"triton-nightly": [_MODULE_ORIGIN]}),
            _triton_provided_by("triton-nightly") as scan,
        ):
            self.assertEqual(triton_utils._available_triton_version(), Version("3.7.1"))

        scan.assert_called_once()

    def test_distribution_without_a_record_is_taken_at_its_word(self):
        # Nothing was learned about what it owns, so it keeps the answer it
        # would have given before the question was asked.
        with (
            _triton_installed({"triton": "3.7.1"}),
            _triton_records({"triton": None}),
            _triton_provided_by() as scan,
        ):
            self.assertEqual(triton_utils._available_triton_version(), Version("3.7.1"))

        scan.assert_not_called()

    def test_unresolvable_module_is_not_held_against_a_distribution(self):
        with (
            _triton_installed({"triton": "3.7.1"}),
            _triton_module_at(None),
            _triton_records({"triton": ["/site-packages/somewhere/else.py"]}),
            _triton_provided_by(),
        ):
            self.assertEqual(triton_utils._available_triton_version(), Version("3.7.1"))

    def test_nameless_distribution_declines_instead_of_raising(self):
        # The scan reports a dist-info with no `Name` as a `None` provider, and
        # the version lookup raises on it -- during `import torch`.
        with (
            patch.object(
                triton_utils,
                "_available_version",
                side_effect=_rejects_a_nameless_distribution,
            ),
            _triton_provided_by(None),
            self.assertLogs("torch._native.triton_utils", level="WARNING"),
        ):
            self.assertIsNone(triton_utils._available_triton_version())


class TestTritonModuleOrigin(TestCase):
    """Locating a module without importing it."""

    def test_absent_module_has_no_origin(self):
        self.assertIsNone(triton_utils._module_origin("_no_such_native_dsl_module"))

    def test_broken_parent_package_declines_instead_of_raising(self):
        # find_spec imports the parent package to search it, so anything the
        # parent raises arrives here, during `import torch`.
        with patch.object(triton_utils, "_find_spec", side_effect=ImportError("boom")):
            self.assertIsNone(triton_utils._module_origin("triton"))


class TestTritonVersionGate(TestCase):
    """The gate that decides whether the Triton overrides register at all.

    Both verdicts are `functools.cache`d, so every case has to clear them twice:
    otherwise this machine's own install decides the first case and a stale
    verdict decides the next one.
    """

    def setUp(self):
        super().setUp()
        # The escape hatch is cached too, and the tests above leave it set: a
        # verdict of "skip the version check" would pass every case here.
        for verdict in (
            triton_utils._check_runtime_available,
            triton_utils._version_is_sufficient,
            triton_utils.check_native_version_skip,
            triton_utils.check_native_jit_disabled,
        ):
            verdict.cache_clear()
            self.addCleanup(verdict.cache_clear)

    @contextlib.contextmanager
    def _installed_triton(self, versions, *distributions):
        """Run the gate against an importable Triton described by `versions`."""
        with (
            _triton_installed(versions),
            _triton_provided_by(*distributions),
            _triton_module_at(_MODULE_ORIGIN),
            _triton_records({}),
            patch.object(triton_utils._cuda, "is_built", return_value=True),
            patch.object(triton_utils, "_unavailable_reason", return_value=None),
        ):
            yield

    def test_wheel_distribution_name_passes_the_gate(self):
        with self._installed_triton({"triton-rocm": "3.7.1"}, "triton-rocm"):
            self.assertTrue(triton_utils.runtime_available())
            self.assertEqual(triton_utils.runtime_version(), Version("3.7.1"))
            self.assertTrue(triton_utils._version_is_sufficient())

    def test_versionless_install_still_fails_the_gate(self):
        with self._installed_triton({}):
            self.assertTrue(triton_utils.runtime_available())
            self.assertIsNone(triton_utils.runtime_version())
            self.assertFalse(triton_utils._version_is_sufficient())

    # 3.6.0 is the boundary; 3.42.0 is a minor far past it, since the check
    # compares the minor rather than the release.
    @parametrize("version", ("3.6.0", "3.42.0"))
    def test_supported_versions_pass_the_gate(self, version):
        with self._installed_triton({"triton-rocm": version}, "triton-rocm"):
            self.assertTrue(triton_utils._version_is_sufficient())

    # One case per way of failing: 3.5.9 is the minor below the boundary, and
    # 2.9.0 and 4.6.0 are rejected on the major alone -- both clear the minor,
    # so nothing else here would notice the major check being loosened to an
    # inequality, which would accept a 4.x Triton the overrides are not built
    # against.
    @parametrize("version", ("3.5.9", "2.9.0", "4.6.0"))
    def test_unsupported_versions_fail_the_gate(self, version):
        with self._installed_triton({"triton-rocm": version}, "triton-rocm"):
            self.assertFalse(triton_utils._version_is_sufficient())

    def test_version_skip_overrides_an_unsupported_version(self):
        with (
            self._installed_triton({"triton-rocm": "3.5.0"}, "triton-rocm"),
            patch.object(triton_utils, "check_native_version_skip", return_value=True),
        ):
            self.assertTrue(triton_utils._version_is_sufficient())

    def test_version_skip_does_not_rescue_an_unreported_version(self):
        # The escape hatch overrides a version that is too old, not the absence
        # of one: there is nothing to run the ops against.
        with (
            self._installed_triton({}),
            patch.object(triton_utils, "check_native_version_skip", return_value=True),
        ):
            self.assertFalse(triton_utils._version_is_sufficient())


instantiate_parametrized_tests(TestNativeDSLOps)
instantiate_parametrized_tests(TestTritonDistributionDiscovery)
instantiate_parametrized_tests(TestTritonVersionGate)


if __name__ == "__main__":
    run_tests()
