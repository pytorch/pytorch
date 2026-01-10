# Owner(s): ["module: inductor"]
from torch._inductor.template_heuristics.base import TemplateConfigHeuristics
from torch._inductor.template_heuristics.registry import (
    _TEMPLATE_HEURISTIC_REGISTRY,
    clear_registry,
    get_template_heuristic,
    register_template_heuristic,
)
from torch._inductor.template_heuristics.triton import BlackwellGPUGemmConfig
from torch._inductor.test_case import run_tests, TestCase


class TestBlackwellGPUGemmConfig(TestCase):
    """Tests for BlackwellGPUGemmConfig class."""

    def test_default_values(self):
        """Test that BlackwellGPUGemmConfig has correct default values."""
        config = BlackwellGPUGemmConfig(
            block_m=128,
            block_n=256,
            block_k=64,
            num_stages=3,
            num_warps=8,
        )
        # Verify inherited GemmConfig fields
        self.assertEqual(config.block_m, 128)
        self.assertEqual(config.block_n, 256)
        self.assertEqual(config.block_k, 64)
        self.assertEqual(config.num_stages, 3)
        self.assertEqual(config.num_warps, 8)

        # Verify new BlackwellGPUGemmConfig-specific fields with default values
        self.assertFalse(config.epilogue_subtile)  # default=False
        self.assertTrue(config.warp_specialize)  # default=True
        self.assertTrue(config.flatten)  # default=True

    def test_custom_values(self):
        """Test that BlackwellGPUGemmConfig accepts custom values for new fields."""
        config = BlackwellGPUGemmConfig(
            block_m=64,
            block_n=128,
            block_k=32,
            num_stages=2,
            num_warps=4,
            epilogue_subtile=True,
            warp_specialize=False,
            flatten=False,
        )
        # Verify custom values are set correctly
        self.assertTrue(config.epilogue_subtile)
        self.assertFalse(config.warp_specialize)
        self.assertFalse(config.flatten)


class TestTemplateHeuristicsRegistry(TestCase):
    def setUp(self):
        super().setUp()
        # Save original registry state
        self.original_registry = _TEMPLATE_HEURISTIC_REGISTRY.copy()
        clear_registry()  # Test heuristic classes using the decorator registration

    def tearDown(self):
        # Restore original registry
        clear_registry()
        _TEMPLATE_HEURISTIC_REGISTRY.update(self.original_registry)
        super().tearDown()

    def test_register_class(self):
        """Test basic registration of a heuristic class."""
        # Clear registry for this isolated test
        clear_registry()

        @register_template_heuristic("test_mm", "cuda")
        class TestHeuristic(TemplateConfigHeuristics):
            pass

        # Verify registration
        key = ("test_mm", "cuda", None)
        self.assertIn(key, _TEMPLATE_HEURISTIC_REGISTRY)
        self.assertEqual(_TEMPLATE_HEURISTIC_REGISTRY[key], TestHeuristic)

    def test_assertion_existing_class(self):
        @register_template_heuristic("triton::mm", "cuda")
        class _CrossOpHeuristic(TemplateConfigHeuristics):
            """(template, device, None) - Cross-op for specific device"""

        """Test that registered class can be retrieved."""
        # The _CrossOpHeuristic is registered at module level for ("mm", "cuda", None)
        # Test retrieval - it should match for any op on cuda device
        heuristic = get_template_heuristic("triton::mm", "cuda", "bmm")
        self.assertIsInstance(heuristic, _CrossOpHeuristic)

    def test_hierarchy_lookup(self):
        """Test complete hierarchy: (template, device, op) -> (template, None, None)"""

        @register_template_heuristic("triton::mm", "cuda", op_name="scaled_mm")
        class _MostSpecificHeuristic(TemplateConfigHeuristics):
            """(template, device, op) - Most specific"""

        @register_template_heuristic("triton::mm", None, op_name="scaled_mm")
        class _CrossDeviceHeuristic(TemplateConfigHeuristics):
            """(template, None, op) - Cross-device for specific op"""

        @register_template_heuristic("triton::mm", "cuda")
        class _CrossOpHeuristic(TemplateConfigHeuristics):
            """(template, device, None) - Cross-op for specific device"""

        @register_template_heuristic("triton::mm", None)
        class _MostGeneralHeuristic(TemplateConfigHeuristics):
            """(template, None, None) - Most general"""

        # All classes are already registered via decorators:
        # _MostSpecificHeuristic: ("mm", "cuda", "scaled_mm") - Most specific
        # _CrossDeviceHeuristic: ("mm", None, "scaled_mm") - Cross-device for specific op
        # _CrossOpHeuristic: ("mm", "cuda", None) - Cross-op for specific device
        # _MostGeneralHeuristic: ("mm", None, None) - Most general

        # Test 1: Exact match - should get most specific
        heuristic = get_template_heuristic("triton::mm", "cuda", "scaled_mm")
        self.assertIsInstance(heuristic, _MostSpecificHeuristic)

        # Test 2: Different device, same op - should get cross-device
        heuristic = get_template_heuristic("triton::mm", "xpu", "scaled_mm")
        self.assertIsInstance(heuristic, _CrossDeviceHeuristic)

        # Test 3: Same device, different op - should get cross-op
        heuristic = get_template_heuristic("triton::mm", "cuda", "bmm")
        self.assertIsInstance(heuristic, _CrossOpHeuristic)

        # Test 4: Different device and op - should get most general
        heuristic = get_template_heuristic("triton::mm", "xpu", "bmm")
        self.assertIsInstance(heuristic, _MostGeneralHeuristic)

    def test_partial_hierarchy_scenarios(self):
        """Test hierarchy behavior with partial registrations"""

        # Scenario 1: Register partial hierarchy using decorators
        @register_template_heuristic("triton::tma", None, op_name="scaled_tma")
        class _TestCrossDeviceHeuristic(TemplateConfigHeuristics):
            pass

        @register_template_heuristic("triton::tma", None)
        class _TestGeneralHeuristic(TemplateConfigHeuristics):
            pass

        # Should get cross-device for matching op, regardless of device
        heuristic = get_template_heuristic("triton::tma", "cuda", "scaled_tma")
        self.assertIsInstance(heuristic, _TestCrossDeviceHeuristic)

        # Should fallback to general for different op
        heuristic = get_template_heuristic("triton::tma", "cuda", "scaled_mm")
        self.assertIsInstance(heuristic, _TestGeneralHeuristic)

        # Scenario 2: Only specific device exists
        @register_template_heuristic("triton::bmm", "cuda")
        class _TestDeviceSpecificHeuristic(TemplateConfigHeuristics):
            pass

        # Should get device-specific for cuda
        heuristic = get_template_heuristic("triton::bmm", "cuda", "any_op")
        self.assertIsInstance(heuristic, _TestDeviceSpecificHeuristic)

        # Should return fallback instance for other devices (no specific heuristic registered)
        heuristic = get_template_heuristic("triton::bmm", "xpu", "any_op")
        self.assertIsInstance(heuristic, TemplateConfigHeuristics)
        # Make sure it's not the registered specific heuristic
        self.assertNotIsInstance(heuristic, _TestDeviceSpecificHeuristic)

        # Scenario 3: Only most general exists
        @register_template_heuristic("triton::mm", None)
        class _TestMostGeneralHeuristic(TemplateConfigHeuristics):
            pass

        # Should always get general regardless of device/op
        heuristic = get_template_heuristic("triton::mm", "cuda", "scaled_addmm")
        self.assertIsInstance(heuristic, _TestMostGeneralHeuristic)

        heuristic = get_template_heuristic("triton::mm", "xpu", "regular_addmm")
        self.assertIsInstance(heuristic, _TestMostGeneralHeuristic)

    def test_fallback_behavior(self):
        """Test that fallback TemplateConfigHeuristics is returned when no heuristic is found"""

        # Test 1: Get fallback for unregistered template
        heuristic = get_template_heuristic("unknown_template", "cuda", "unknown_op")
        self.assertIsInstance(heuristic, TemplateConfigHeuristics)
        # Make sure it's the base class and not a subclass
        self.assertEqual(type(heuristic), TemplateConfigHeuristics)

        # Test 2: Verify fallback instances are NOT cached (new instance each time)
        heuristic2 = get_template_heuristic("unknown_template", "cuda", "unknown_op")
        self.assertIsInstance(heuristic2, TemplateConfigHeuristics)
        self.assertEqual(type(heuristic2), TemplateConfigHeuristics)
        # Should be different instances (not cached)
        self.assertIsNot(heuristic, heuristic2)

        # Test 3: After registering a heuristic, should get the registered one instead
        @register_template_heuristic("unknown_template", "cuda", op_name="unknown_op")
        class _NewlyRegisteredHeuristic(TemplateConfigHeuristics):
            pass

        # Now should get the registered heuristic, not the fallback
        heuristic3 = get_template_heuristic("unknown_template", "cuda", "unknown_op")
        self.assertIsInstance(heuristic3, _NewlyRegisteredHeuristic)
        self.assertNotEqual(type(heuristic3), TemplateConfigHeuristics)

        # Test 4: Verify registered instances ARE cached (same instance each time)
        heuristic4 = get_template_heuristic("unknown_template", "cuda", "unknown_op")
        self.assertIsInstance(heuristic4, _NewlyRegisteredHeuristic)
        self.assertIs(heuristic3, heuristic4)  # Should be same cached instance


if __name__ == "__main__":
    run_tests()
