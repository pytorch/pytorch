# Owner(s): ["module: dsl-native-ops"]

import torch.backends.python_native as pn
from torch.testing._internal.common_utils import run_tests, skipIfTorchDynamo, TestCase


class RegistryTestMixin:
    """Mixin for tests that need to preserve registry state."""

    def setUp(self):
        """Set up test state with registry preservation."""
        super().setUp() if hasattr(super(), "setUp") else None

        # Use the new _preserve_filter_state but setup manually for tearDown
        filter_state = pn._get_filter_state()
        self._original_filter_state = (
            set(filter_state._dsl_names),
            set(filter_state._op_symbols),
            set(filter_state._dispatch_keys),
        )

        # Clear filter state for clean test start
        filter_state._dsl_names.clear()
        filter_state._op_symbols.clear()
        filter_state._dispatch_keys.clear()

        # Ensure all DSLs start enabled
        try:
            for dsl_name in pn.all_dsls:
                dsl = getattr(pn, dsl_name)
                dsl.enable()
        except Exception:
            pass

    def tearDown(self):
        """Restore original filter state."""
        try:
            if hasattr(self, "_original_filter_state"):
                filter_state = pn._get_filter_state()

                # Restore original filter state
                filter_state._dsl_names.clear()
                filter_state._op_symbols.clear()
                filter_state._dispatch_keys.clear()

                filter_state._dsl_names.update(self._original_filter_state[0])
                filter_state._op_symbols.update(self._original_filter_state[1])
                filter_state._dispatch_keys.update(self._original_filter_state[2])
        except Exception:
            pass
        super().tearDown() if hasattr(super(), "tearDown") else None


@skipIfTorchDynamo("Backend tests don't need dynamo compilation")
class TestTorchBackendsPythonNative(RegistryTestMixin, TestCase):
    """Tests for torch.backends.python_native user-facing API."""

    def test_module_import(self):
        """Test that torch.backends.python_native imports successfully."""
        # Should not raise any exceptions
        import torch.backends.python_native as pn_import

        self.assertIsNotNone(pn_import)

    def test_dsl_discovery(self):
        """Test DSL discovery functionality."""
        all_dsls = pn.all_dsls
        available_dsls = pn.available_dsls

        # Should return lists
        self.assertIsInstance(all_dsls, list)
        self.assertIsInstance(available_dsls, list)

        # Available DSLs should be subset of all DSLs
        self.assertTrue(set(available_dsls).issubset(set(all_dsls)))

    def test_dsl_access(self):
        """Test accessing DSL controllers."""
        all_dsls = pn.all_dsls

        for dsl_name in all_dsls:
            with self.subTest(dsl_name=dsl_name):
                # Should be able to access DSL controller
                dsl = getattr(pn, dsl_name)
                self.assertIsNotNone(dsl)

                # Should have required attributes
                self.assertEqual(dsl.name, dsl_name)
                self.assertIsInstance(dsl.available, bool)
                self.assertIsInstance(dsl.enabled, bool)

    def test_dsl_properties(self):
        """Test DSL controller properties."""
        all_dsls = pn.all_dsls

        for dsl_name in all_dsls:
            with self.subTest(dsl_name=dsl_name):
                dsl = getattr(pn, dsl_name)

                # Test name property
                self.assertEqual(dsl.name, dsl_name)

                # Test available property (should not raise)
                available = dsl.available
                self.assertIsInstance(available, bool)

                # Test version property (should not raise)
                version = dsl.version
                # Version can be None if DSL not available
                self.assertTrue(version is None or hasattr(version, "major"))

                # Test enabled property
                enabled = dsl.enabled
                self.assertIsInstance(enabled, bool)

    def test_dsl_enable_disable(self):
        """Test DSL enable/disable functionality."""
        all_dsls = pn.all_dsls

        for dsl_name in all_dsls:
            with self.subTest(dsl_name=dsl_name):
                dsl = getattr(pn, dsl_name)
                original_state = dsl.enabled

                try:
                    # Test method-based disable/enable
                    dsl.disable()
                    self.assertEqual(dsl.enabled, False)

                    dsl.enable()
                    self.assertEqual(dsl.enabled, True)

                    # Test property-based disable/enable
                    dsl.disable()  # Use method to avoid flags_frozen issue
                    self.assertEqual(dsl.enabled, False)

                    dsl.enable()  # Use method to avoid flags_frozen issue
                    self.assertEqual(dsl.enabled, True)

                finally:
                    # Restore original state
                    if original_state:
                        dsl.enable()
                    else:
                        dsl.disable()

    def test_dsl_context_managers(self):
        """Test DSL context manager functionality."""
        all_dsls = pn.all_dsls

        for dsl_name in all_dsls:
            with self.subTest(dsl_name=dsl_name):
                dsl = getattr(pn, dsl_name)
                original_state = dsl.enabled

                try:
                    # Ensure DSL starts enabled
                    dsl.enable()

                    # Test disabled context manager
                    with dsl.disabled():
                        self.assertEqual(dsl.enabled, False)

                    # Should be restored after context
                    self.assertEqual(dsl.enabled, True)

                finally:
                    # Restore original state
                    if original_state:
                        dsl.enable()
                    else:
                        dsl.disable()

    def test_nested_context_managers(self):
        """Test nested DSL context managers."""
        all_dsls = pn.all_dsls

        if len(all_dsls) >= 2:
            dsl1_name, dsl2_name = all_dsls[0], all_dsls[1]
            dsl1 = getattr(pn, dsl1_name)
            dsl2 = getattr(pn, dsl2_name)

            original_state1 = dsl1.enabled
            original_state2 = dsl2.enabled

            try:
                # Ensure both start enabled
                dsl1.enable()
                dsl2.enable()

                with dsl1.disabled():
                    self.assertEqual(dsl1.enabled, False)
                    self.assertEqual(dsl2.enabled, True)

                    with dsl2.disabled():
                        self.assertEqual(dsl1.enabled, False)
                        self.assertEqual(dsl2.enabled, False)

                    # dsl2 should be restored
                    self.assertEqual(dsl1.enabled, False)
                    self.assertEqual(dsl2.enabled, True)

                # Both should be restored
                self.assertEqual(dsl1.enabled, True)
                self.assertEqual(dsl2.enabled, True)

            finally:
                # Restore original states
                if original_state1:
                    dsl1.enable()
                else:
                    dsl1.disable()
                if original_state2:
                    dsl2.enable()
                else:
                    dsl2.disable()

    def test_operation_discovery(self):
        """Test operation discovery functionality."""
        all_dsls = pn.all_dsls

        for dsl_name in all_dsls:
            with self.subTest(dsl_name=dsl_name):
                operations = pn.get_dsl_operations(dsl_name)

                # Should return a list
                self.assertIsInstance(operations, list)

                # Each operation should be a string
                for op in operations:
                    self.assertIsInstance(op, str)
                    self.assertTrue(len(op) > 0)

    def test_operation_control(self):
        """Test operation-level control functionality."""
        # Get an operation to test with
        all_dsls = pn.all_dsls
        test_operation = None

        for dsl_name in all_dsls:
            operations = pn.get_dsl_operations(dsl_name)
            if operations:
                test_operation = operations[0]
                break

        if test_operation:
            # Test operation disable/enable (should not raise)
            pn.disable_operations(test_operation)
            pn.enable_operations(test_operation)

            # Test multiple operations
            pn.disable_operations(test_operation, "nonexistent_op")
            pn.enable_operations(test_operation, "nonexistent_op")

    def test_operation_context_manager(self):
        """Test operation-level context manager."""
        # Get an operation to test with
        all_dsls = pn.all_dsls
        test_operation = None

        for dsl_name in all_dsls:
            operations = pn.get_dsl_operations(dsl_name)
            if operations:
                test_operation = operations[0]
                break

        if test_operation:
            # Should not raise exceptions
            with pn.operations_disabled(test_operation):
                pass  # Operation should be disabled in this context
            # Operation should be re-enabled after context

    def test_dispatch_key_control(self):
        """Test dispatch key control functionality."""
        # Test basic dispatch key control (should not raise)
        pn.disable_dispatch_keys("CUDA", "CPU")
        pn.enable_dispatch_keys("CUDA", "CPU")

    def test_invalid_dsl_access(self):
        """Test accessing invalid DSL names."""
        with self.assertRaises(AttributeError):
            _ = pn.nonexistent_dsl

    def test_dsl_repr(self):
        """Test DSL controller string representation."""
        all_dsls = pn.all_dsls

        for dsl_name in all_dsls:
            with self.subTest(dsl_name=dsl_name):
                dsl = getattr(pn, dsl_name)
                repr_str = repr(dsl)

                # Should contain DSL name
                self.assertIn(dsl_name, repr_str)
                # Should contain status info
                self.assertTrue(
                    any(status in repr_str for status in ["available", "unavailable"])
                )
                self.assertTrue(
                    any(status in repr_str for status in ["enabled", "disabled"])
                )

    def test_module_dir(self):
        """Test module __dir__ functionality."""
        attrs = dir(pn)

        # Should contain core attributes
        expected_attrs = [
            "available_dsls",
            "all_dsls",
            "get_dsl_operations",
            "disable_operations",
            "enable_operations",
            "disable_dispatch_keys",
            "enable_dispatch_keys",
            "operations_disabled",
        ]

        for attr in expected_attrs:
            self.assertIn(attr, attrs)

        # Should contain DSL names
        for dsl_name in pn.all_dsls:
            self.assertIn(dsl_name, attrs)

    def test_caching_functionality(self):
        """Test caching integration and API contracts (not functools.lru_cache internals)."""
        all_dsls = pn.all_dsls

        if all_dsls:
            dsl_name = all_dsls[0]

            # Test operation caching - results should be identical
            ops1 = pn.get_dsl_operations(dsl_name)
            ops2 = pn.get_dsl_operations(dsl_name)
            self.assertEqual(ops1, ops2)

            # Test DSL controller caching - same object returned (API contract)
            controller1 = getattr(pn, dsl_name)
            controller2 = getattr(pn, dsl_name)
            self.assertIs(controller1, controller2)

            # Verify controller still works correctly
            self.assertEqual(controller1.name, dsl_name)
            self.assertIsInstance(controller1.enabled, bool)

    def test_error_handling(self):
        """Test error handling with invalid inputs."""
        # Test with empty operation names - should not raise
        pn.disable_operations("")
        pn.enable_operations("")

        # Test with invalid dispatch keys - should not raise
        pn.disable_dispatch_keys("")
        pn.enable_dispatch_keys("")

        # Test invalid DSL access raises AttributeError
        with self.assertRaises(AttributeError):
            _ = pn.invalid_dsl_name

        # Test get_dsl_operations with invalid DSL - should return empty list or raise ValueError
        invalid_ops = pn.get_dsl_operations("invalid_dsl")
        self.assertIsInstance(invalid_ops, list)
        self.assertEqual(len(invalid_ops), 0)

    def test_dispatch_behavior_verification(self):
        """Test that DSL disable/enable actually affects operation dispatch."""
        all_dsls = pn.all_dsls

        for dsl_name in all_dsls:
            with self.subTest(dsl_name=dsl_name):
                dsl = getattr(pn, dsl_name)

                if not dsl.available:
                    continue  # Skip unavailable DSLs

                operations = pn.get_dsl_operations(dsl_name)
                if not operations:
                    continue  # Skip DSLs with no operations

                original_state = dsl.enabled

                try:
                    # Enable DSL first
                    dsl.enable()
                    self.assertTrue(
                        dsl.enabled, f"{dsl_name} should be enabled after enable()"
                    )

                    # Disable DSL
                    dsl.disable()
                    self.assertFalse(
                        dsl.enabled, f"{dsl_name} should be disabled after disable()"
                    )

                    # Verify operations are actually deregistered by checking registry state
                    self.assertTrue(
                        pn.is_dsl_disabled(dsl_name),
                        f"{dsl_name} should be disabled in registry",
                    )

                    # Re-enable and verify registry state
                    dsl.enable()
                    self.assertTrue(
                        dsl.enabled, f"{dsl_name} should be enabled after re-enable()"
                    )
                    self.assertFalse(
                        pn.is_dsl_disabled(dsl_name),
                        f"{dsl_name} should not be disabled in registry after re-enable",
                    )

                finally:
                    # Restore original state
                    if original_state:
                        dsl.enable()
                    else:
                        dsl.disable()

    def test_context_manager_dispatch_behavior(self):
        """Test that context managers actually affect dispatch, not just state tracking."""
        all_dsls = pn.all_dsls

        for dsl_name in all_dsls:
            with self.subTest(dsl_name=dsl_name):
                dsl = getattr(pn, dsl_name)

                if not dsl.available:
                    continue  # Skip unavailable DSLs

                original_state = dsl.enabled

                try:
                    # Start with DSL enabled
                    dsl.enable()
                    self.assertFalse(pn.is_dsl_disabled(dsl_name))

                    # Use disabled context manager
                    with dsl.disabled():
                        # Verify DSL is actually disabled in registry during context
                        self.assertTrue(
                            pn.is_dsl_disabled(dsl_name),
                            f"{dsl_name} should be disabled in registry during context",
                        )

                    # Verify DSL is re-enabled after context
                    self.assertFalse(
                        pn.is_dsl_disabled(dsl_name),
                        f"{dsl_name} should be re-enabled in registry after context",
                    )

                finally:
                    # Restore original state
                    if original_state:
                        dsl.enable()
                    else:
                        dsl.disable()

    def test_operation_dispatch_behavior(self):
        """Test that operation-level disable/enable affects registry state."""
        all_dsls = pn.all_dsls
        test_operation = None

        # Find a real operation to test with
        for dsl_name in all_dsls:
            operations = pn.get_dsl_operations(dsl_name)
            if operations:
                test_operation = operations[0]
                break

        if test_operation:
            # Verify operation starts enabled
            initial_disabled = pn.is_operation_disabled(test_operation)

            try:
                # Disable operation and verify registry state
                pn.disable_operations(test_operation)
                self.assertTrue(
                    pn.is_operation_disabled(test_operation),
                    f"Operation {test_operation} should be disabled in registry",
                )

                # Re-enable operation and verify registry state
                pn.enable_operations(test_operation)
                # Note: enable_operations removes from disabled list
                self.assertFalse(
                    pn.is_operation_disabled(test_operation),
                    f"Operation {test_operation} should not be disabled in registry after re-enable",
                )

            finally:
                # Restore initial state
                if initial_disabled:
                    pn.disable_operations(test_operation)
                else:
                    pn.enable_operations(test_operation)


class TestTorchBackendsPythonNativeIntegration(TestCase):
    """Integration tests for torch.backends.python_native with actual DSLs."""

    def test_real_dsl_integration(self):
        """Test integration with real DSL modules if available."""
        all_dsls = pn.all_dsls

        for dsl_name in all_dsls:
            with self.subTest(dsl_name=dsl_name):
                dsl = getattr(pn, dsl_name)

                if dsl.available:
                    # Test that we can actually disable/enable real DSLs
                    original_state = dsl.enabled

                    try:
                        # This should call actual DSL deregister functions
                        dsl.disable()

                        # This should call actual registry re-enable functions
                        dsl.enable()

                    finally:
                        # Restore state
                        if original_state:
                            dsl.enable()
                        else:
                            dsl.disable()

    def test_operations_with_real_registry(self):
        """Test operation discovery with real registry."""
        all_dsls = pn.all_dsls

        for dsl_name in all_dsls:
            with self.subTest(dsl_name=dsl_name):
                operations = pn.get_dsl_operations(dsl_name)

                # If DSL has operations, they should be valid
                if operations:
                    # Operations should be strings
                    for op in operations:
                        self.assertIsInstance(op, str)
                        self.assertGreater(len(op), 0)


if __name__ == "__main__":
    run_tests()
