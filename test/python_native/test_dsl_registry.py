# Owner(s): ["module: dsl-native-ops"]

from unittest.mock import Mock

from torch._vendor.packaging.version import Version
from torch.testing._internal.common_utils import run_tests, TestCase


class TestDSLRegistry(TestCase):
    """Comprehensive tests for DSL registry functionality."""

    def setUp(self):
        """Set up clean registry state for each test"""
        # Import registry here to avoid import-time side effects
        from torch._native.dsl_registry import DSLRegistry

        # Save original registry state
        self.original_registry = None
        try:
            from torch._native.dsl_registry import dsl_registry as original

            self.original_modules = original._dsl_modules.copy()
            self.original_registry = original
        except ImportError:
            pass

        # Create isolated registry for testing
        self.test_registry = DSLRegistry()

    def tearDown(self):
        """Restore original registry state"""
        if self.original_registry is not None:
            # Restore original DSL modules
            self.original_registry._dsl_modules.clear()
            self.original_registry._dsl_modules.update(self.original_modules)

            # Clear any caches
            if hasattr(self.original_registry.is_dsl_available, "cache_clear"):
                self.original_registry.is_dsl_available.cache_clear()
            if hasattr(self.original_registry.get_dsl_version, "cache_clear"):
                self.original_registry.get_dsl_version.cache_clear()
            if hasattr(self.original_registry.list_available_dsls, "cache_clear"):
                self.original_registry.list_available_dsls.cache_clear()

    def create_valid_mock_dsl(self, name="test_dsl", available=True, version="1.0.0"):
        """Helper to create valid mock DSL for testing"""
        mock = Mock()
        mock.runtime_available.return_value = available
        mock.runtime_version.return_value = Version(version) if version else None
        mock.deregister_op_overrides = Mock()
        mock.register_op_override = Mock()
        return mock

    def create_broken_mock_dsl(
        self, break_method="runtime_available", error=ImportError
    ):
        """Helper to create mock DSL with broken methods for error testing"""
        mock = self.create_valid_mock_dsl()
        getattr(mock, break_method).side_effect = error("Simulated error")
        return mock

    # Phase 1: Basic Registry Operations (4 methods)

    def test_register_dsl_basic(self):
        """Test basic DSL registration functionality"""
        mock_dsl = self.create_valid_mock_dsl()
        self.test_registry.register_dsl("test_dsl", mock_dsl)

        # Verify registration
        self.assertIn("test_dsl", self.test_registry.list_all_dsls())
        self.assertTrue(self.test_registry.is_dsl_available("test_dsl"))

        # Verify the DSL module is stored correctly
        self.assertEqual(len(self.test_registry.list_all_dsls()), 1)

    def test_register_dsl_duplicates(self):
        """Test duplicate DSL registration scenarios"""
        scenarios = [
            ("same_module", "re-registered with same module"),
            ("different_module", "re-registered with different module"),
        ]

        for scenario, expected_log_contains in scenarios:
            with self.subTest(scenario=scenario):
                # Register initial DSL
                mock_dsl1 = self.create_valid_mock_dsl()
                dsl_name = f"duplicate_test_{scenario}"
                self.test_registry.register_dsl(dsl_name, mock_dsl1)

                # Register duplicate based on scenario
                if scenario == "same_module":
                    mock_dsl2 = mock_dsl1  # Same module object
                else:
                    mock_dsl2 = self.create_valid_mock_dsl()  # Different module object

                # Capture logging
                with self.assertLogs(
                    "torch._native.dsl_registry", level="DEBUG"
                ) as log_capture:
                    self.test_registry.register_dsl(dsl_name, mock_dsl2)

                # Verify appropriate logging
                log_messages = " ".join(log_capture.output)
                self.assertIn(expected_log_contains, log_messages)

                # Verify DSL is still registered
                self.assertIn(dsl_name, self.test_registry.list_all_dsls())

    def test_get_dsl_version(self):
        """Test DSL version querying with various conditions"""
        test_cases = [("1.2.3", Version), (None, type(None)), ("error", type(None))]

        for version_setup, expected_result_type in test_cases:
            with self.subTest(version_setup=version_setup):
                if version_setup == "error":
                    mock_dsl = self.create_broken_mock_dsl(
                        "runtime_version", RuntimeError
                    )
                else:
                    mock_dsl = self.create_valid_mock_dsl(version=version_setup)

                dsl_name = f"version_test_{version_setup}"
                self.test_registry.register_dsl(dsl_name, mock_dsl)

                result = self.test_registry.get_dsl_version(dsl_name)
                self.assertIsInstance(result, expected_result_type)

                if version_setup == "1.2.3":
                    self.assertEqual(result, Version("1.2.3"))

    def test_list_dsls_operations(self):
        """Test list_all_dsls and list_available_dsls"""
        # Register mix of available/unavailable DSLs
        available_dsl = self.create_valid_mock_dsl("available", available=True)
        unavailable_dsl = self.create_valid_mock_dsl("unavailable", available=False)

        self.test_registry.register_dsl("available_dsl", available_dsl)
        self.test_registry.register_dsl("unavailable_dsl", unavailable_dsl)

        # Test list operations
        all_dsls = self.test_registry.list_all_dsls()
        available_dsls = self.test_registry.list_available_dsls()

        # Verify results
        self.assertEqual(set(all_dsls), {"available_dsl", "unavailable_dsl"})
        self.assertEqual(set(available_dsls), {"available_dsl"})

        # Verify available is subset of all
        self.assertTrue(set(available_dsls).issubset(set(all_dsls)))

    # Phase 2: Input Validation (2 methods)

    def test_register_dsl_invalid_names(self):
        """Test registration with invalid name inputs"""
        test_cases = [
            (None, TypeError),
            (123, TypeError),
            ([], TypeError),
            ({}, TypeError),
            ("", ValueError),
            ("   ", ValueError),
            ("\t\n", ValueError),
        ]

        mock_dsl = self.create_valid_mock_dsl()

        for invalid_name, expected_exception in test_cases:
            with self.subTest(invalid_name=repr(invalid_name)):
                with self.assertRaises(expected_exception):
                    self.test_registry.register_dsl(invalid_name, mock_dsl)

    def test_register_dsl_valid_names(self):
        """Test registration with valid name formats"""
        valid_names = ["triton", "cutedsl", "my_dsl", "dsl_v2", "a", "test_dsl_123"]

        for valid_name in valid_names:
            with self.subTest(valid_name=valid_name):
                mock_dsl = self.create_valid_mock_dsl()
                self.test_registry.register_dsl(valid_name, mock_dsl)
                self.assertIn(valid_name, self.test_registry.list_all_dsls())

    # Phase 3: Error Handling (3 methods)

    def test_is_dsl_available_errors(self):
        """Test is_dsl_available when runtime_available() raises errors"""
        error_types = [ImportError, ModuleNotFoundError, RuntimeError, AttributeError]

        for error_type in error_types:
            with self.subTest(error_type=error_type.__name__):
                mock_dsl = self.create_broken_mock_dsl("runtime_available", error_type)
                dsl_name = f"broken_dsl_{error_type.__name__}"
                self.test_registry.register_dsl(dsl_name, mock_dsl)

                result = self.test_registry.is_dsl_available(dsl_name)
                self.assertEqual(result, False)  # All errors should result in False

    def test_get_dsl_version_errors(self):
        """Test get_dsl_version when runtime_version() raises errors"""
        error_types = [ImportError, RuntimeError, AttributeError, TypeError]

        for error_type in error_types:
            with self.subTest(error_type=error_type.__name__):
                mock_dsl = self.create_broken_mock_dsl("runtime_version", error_type)
                dsl_name = f"broken_version_dsl_{error_type.__name__}"
                self.test_registry.register_dsl(dsl_name, mock_dsl)

                result = self.test_registry.get_dsl_version(dsl_name)
                self.assertIsNone(result)

    def test_nonexistent_dsl_queries(self):
        """Test querying non-existent DSLs returns appropriate defaults"""
        # Test with empty registry
        self.assertFalse(self.test_registry.is_dsl_available("nonexistent"))
        self.assertIsNone(self.test_registry.get_dsl_version("nonexistent"))
        self.assertEqual(self.test_registry.list_all_dsls(), [])
        self.assertEqual(self.test_registry.list_available_dsls(), [])

        # Test with some DSLs registered but querying non-existent
        mock_dsl = self.create_valid_mock_dsl()
        self.test_registry.register_dsl("existing", mock_dsl)

        self.assertFalse(self.test_registry.is_dsl_available("still_nonexistent"))
        self.assertIsNone(self.test_registry.get_dsl_version("still_nonexistent"))

    # Phase 4: Protocol & Integration (4 methods)

    def test_dsl_protocol_interface(self):
        """Test DSL modules implement complete protocol"""
        # Test with actual registered DSLs from the global registry
        if self.original_registry:
            all_dsls = self.original_registry.list_all_dsls()

            for dsl_name in all_dsls:
                # Get the actual DSL module from registry
                dsl_module = self.original_registry._dsl_modules.get(dsl_name)
                if dsl_module:
                    # Verify all required methods exist and are callable
                    required_methods = [
                        "runtime_available",
                        "runtime_version",
                        "deregister_op_overrides",
                        "register_op_override",
                    ]

                    for method_name in required_methods:
                        self.assertTrue(
                            hasattr(dsl_module, method_name),
                            f"DSL '{dsl_name}' missing method '{method_name}'",
                        )
                        method = getattr(dsl_module, method_name)
                        self.assertTrue(
                            callable(method),
                            f"DSL '{dsl_name}' method '{method_name}' is not callable",
                        )

    def test_real_dsl_integration(self):
        """Test integration with actual DSL modules"""
        if not self.original_registry:
            self.skipTest("Original registry not available")

        dsl_names = ["triton", "cutedsl"]

        for dsl_name in dsl_names:
            with self.subTest(dsl_name=dsl_name):
                # Verify DSL is registered in original registry
                all_dsls = self.original_registry.list_all_dsls()
                if dsl_name not in all_dsls:
                    self.skipTest(
                        f"DSL '{dsl_name}' not registered in original registry"
                    )

                # Test all registry operations work
                availability = self.original_registry.is_dsl_available(dsl_name)
                self.assertIsInstance(availability, bool)

                version = self.original_registry.get_dsl_version(dsl_name)
                self.assertTrue(version is None or isinstance(version, Version))

                # Verify DSL appears in appropriate lists
                if availability:
                    self.assertIn(
                        dsl_name, self.original_registry.list_available_dsls()
                    )
                self.assertIn(dsl_name, self.original_registry.list_all_dsls())

    def test_common_utils_wrappers(self):
        """Test common_utils wrapper functions work correctly"""
        from torch.testing._internal.common_utils import (
            get_all_dsls,
            get_available_dsls,
        )

        # Compare wrapper results with direct registry calls
        if self.original_registry:
            self.assertEqual(get_all_dsls(), self.original_registry.list_all_dsls())
            self.assertEqual(
                get_available_dsls(), self.original_registry.list_available_dsls()
            )

    def test_skip_decorators(self):
        """Test DSL skip decorators work with registry"""
        from torch.testing._internal.common_utils import (
            skipIfDSLUnavailable,
            skipUnlessDSLAvailable,
        )

        # Test decorators are callable
        self.assertTrue(callable(skipIfDSLUnavailable))
        self.assertTrue(callable(skipUnlessDSLAvailable))

        # Test decorator creation works
        decorator1 = skipIfDSLUnavailable("nonexistent_dsl")
        decorator2 = skipUnlessDSLAvailable("triton")

        self.assertTrue(callable(decorator1))
        self.assertTrue(callable(decorator2))

    # Phase 5: Test Infrastructure (2 methods)

    def test_registry_isolation(self):
        """Test registry state can be saved and restored"""
        # Verify we start with clean test registry
        self.assertEqual(self.test_registry.list_all_dsls(), [])

        # Register test DSL
        mock_dsl = self.create_valid_mock_dsl()
        self.test_registry.register_dsl("isolation_test", mock_dsl)

        # Verify registration
        self.assertIn("isolation_test", self.test_registry.list_all_dsls())

        # Verify original registry is unaffected
        if self.original_registry:
            original_dsls = self.original_registry.list_all_dsls()
            self.assertNotIn("isolation_test", original_dsls)

    def test_mock_dsl_helpers(self):
        """Test mock DSL creation utilities work correctly"""
        # Test create_valid_mock_dsl
        valid_mock = self.create_valid_mock_dsl("test", available=True, version="2.1.0")

        self.assertTrue(callable(valid_mock.runtime_available))
        self.assertTrue(callable(valid_mock.runtime_version))
        self.assertTrue(callable(valid_mock.deregister_op_overrides))
        self.assertTrue(callable(valid_mock.register_op_override))

        # Test behavior
        self.assertTrue(valid_mock.runtime_available())
        self.assertEqual(valid_mock.runtime_version(), Version("2.1.0"))

        # Test create_broken_mock_dsl
        broken_mock = self.create_broken_mock_dsl("runtime_available", ImportError)

        with self.assertRaises(ImportError):
            broken_mock.runtime_available()

        # Other methods should still work
        self.assertIsInstance(broken_mock.runtime_version(), Version)


if __name__ == "__main__":
    run_tests()
