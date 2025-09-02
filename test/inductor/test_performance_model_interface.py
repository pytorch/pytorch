# Owner(s): ["module: inductor"]
import unittest
from unittest.mock import MagicMock

from torch._inductor.performance_model.core import filter_and_sort_choices
from torch._inductor.performance_model.registry import (
    clear_registry,
    get_functions_for_templates,
    get_model_function_for_key,
    list_registered_models,
    register_performance_model,
    register_performance_model_fn,
)
from torch._inductor.test_case import run_tests


# Mock performance model functions for testing
def mock_model_a(choices, op_name, multiplier=1.0):
    """Mock implementation for testing."""
    for choice in choices:
        choice.performance_prediction = 1.0 * multiplier
    return choices


def mock_model_b(choices, op_name, offset=0.5):
    """Mock implementation for testing."""
    for choice in choices:
        choice.performance_prediction = 2.0 + offset
    return choices


def mock_model_c(choices, op_name, factor=3.0):
    """Mock implementation for testing."""
    for choice in choices:
        choice.performance_prediction = factor
    return choices


class TestPerformanceModelInterface(unittest.TestCase):
    def setUp(self):
        """Clear registry before each test."""
        clear_registry()

    def tearDown(self):
        """Clear registry after each test."""
        clear_registry()

    def test_decorator_registration(self):
        """Test that decorator registration works correctly."""

        @register_performance_model("mm", "mm", "cuda")
        def decorated_model(choices, op_name):
            for choice in choices:
                choice.performance_prediction = 2.0
            return choices

        # Check that the function was registered
        result_func = get_model_function_for_key("mm", "mm", "cuda")
        self.assertIsNotNone(result_func)
        self.assertEqual(result_func, decorated_model)

    def test_function_registration(self):
        """Test that function registration works correctly."""

        def function_registered_model(choices, op_name):
            for choice in choices:
                choice.performance_prediction = 5.0
            return choices

        # Register using function
        register_performance_model_fn(function_registered_model, "bmm", "bmm", "cuda")

        # Check that the function was registered
        result_func = get_model_function_for_key("bmm", "bmm", "cuda")
        self.assertIsNotNone(result_func)
        self.assertEqual(result_func, function_registered_model)

    def test_registration_override_specific_subpart(self):
        """Test that we can override registration for one specific (template, op, hardware) combo."""

        # Register mock_model_a for multiple combinations
        register_performance_model_fn(mock_model_a, "mm", "mm", "cuda")
        register_performance_model_fn(mock_model_a, "addmm", "mm", "cuda")
        register_performance_model_fn(mock_model_a, "bmm", "mm", "cuda")

        # Verify initial registrations
        self.assertEqual(get_model_function_for_key("mm", "mm", "cuda"), mock_model_a)
        self.assertEqual(
            get_model_function_for_key("addmm", "mm", "cuda"), mock_model_a
        )
        self.assertEqual(get_model_function_for_key("bmm", "mm", "cuda"), mock_model_a)

        # Override only one specific combination
        register_performance_model_fn(mock_model_b, "addmm", "mm", "cuda")

        # Check that only the overridden key changed
        self.assertEqual(get_model_function_for_key("mm", "mm", "cuda"), mock_model_a)
        self.assertEqual(
            get_model_function_for_key("addmm", "mm", "cuda"), mock_model_b
        )  # Changed
        self.assertEqual(get_model_function_for_key("bmm", "mm", "cuda"), mock_model_a)

    def test_get_functions_for_templates_multiple_functions(self):
        """Test get_functions_for_templates with multiple functions and missing templates."""

        # Register mock_model_a for 3 templates
        register_performance_model_fn(mock_model_a, "mm", "mm", "cuda")
        register_performance_model_fn(mock_model_a, "addmm", "mm", "cuda")
        register_performance_model_fn(mock_model_a, "baddbmm", "mm", "cuda")

        # Register mock_model_b for 1 template
        register_performance_model_fn(mock_model_b, "bmm", "mm", "cuda")

        # Note: "scaled_mm" has no registered function

        # Test with all templates including one without a function
        template_ids = ["mm", "addmm", "baddbmm", "bmm", "scaled_mm"]
        result = get_functions_for_templates(template_ids, "mm", "cuda")

        # Should get 2 functions (A and B), scaled_mm should be excluded
        self.assertEqual(len(result), 2)

        # Find which function corresponds to which template list
        func_a_templates = None
        func_b_templates = None

        for func, templates in result.items():
            if func == mock_model_a:
                func_a_templates = templates
                # Should handle 3 templates
                self.assertEqual(set(templates), {"mm", "addmm", "baddbmm"})
            elif func == mock_model_b:
                func_b_templates = templates
                # Should handle 1 template
                self.assertEqual(templates, ["bmm"])

        # Ensure we found both functions
        self.assertIsNotNone(func_a_templates)
        self.assertIsNotNone(func_b_templates)

    def test_get_functions_for_templates_with_override(self):
        """Test get_functions_for_templates after overriding one of the registrations."""

        # Initially register mock_model_a for 3 templates
        register_performance_model_fn(mock_model_a, "mm", "mm", "cuda")
        register_performance_model_fn(mock_model_a, "addmm", "mm", "cuda")
        register_performance_model_fn(mock_model_a, "baddbmm", "mm", "cuda")

        # Verify initial state
        template_ids = ["mm", "addmm", "baddbmm"]
        result = get_functions_for_templates(template_ids, "mm", "cuda")

        self.assertEqual(len(result), 1)  # Only mock_model_a
        func_a = next(iter(result.keys()))
        self.assertEqual(func_a, mock_model_a)
        self.assertEqual(set(result[func_a]), {"mm", "addmm", "baddbmm"})

        # Override one registration (addmm) with mock_model_c
        register_performance_model_fn(mock_model_c, "addmm", "mm", "cuda")

        # Test again after override
        result = get_functions_for_templates(template_ids, "mm", "cuda")

        # Now should get 2 functions
        self.assertEqual(len(result), 2)

        func_a_templates = None
        func_c_templates = None

        for func, templates in result.items():
            if func == mock_model_a:
                func_a_templates = templates
                # Should handle 2 templates now (addmm was taken away)
                self.assertEqual(set(templates), {"mm", "baddbmm"})
            elif func == mock_model_c:
                func_c_templates = templates
                # Should handle 1 template (the overridden one)
                self.assertEqual(templates, ["addmm"])

        # Ensure we found both functions
        self.assertIsNotNone(func_a_templates)
        self.assertIsNotNone(func_c_templates)

    def test_clear_registry(self):
        """Test that clear_registry removes all registrations."""

        # Register some functions
        register_performance_model_fn(mock_model_a, "mm", "mm", "cuda")
        register_performance_model_fn(mock_model_b, "bmm", "mm", "cuda")

        # Verify they exist
        self.assertIsNotNone(get_model_function_for_key("mm", "mm", "cuda"))
        self.assertIsNotNone(get_model_function_for_key("bmm", "mm", "cuda"))
        self.assertEqual(len(list_registered_models()), 2)

        # Clear registry
        clear_registry()

        # Verify they're gone
        self.assertIsNone(get_model_function_for_key("mm", "mm", "cuda"))
        self.assertIsNone(get_model_function_for_key("bmm", "mm", "cuda"))
        self.assertEqual(len(list_registered_models()), 0)

    def test_list_registered_models(self):
        """Test that list_registered_models returns correct keys."""

        # Register some functions
        register_performance_model_fn(mock_model_a, "mm", "mm", "cuda")
        register_performance_model_fn(mock_model_b, "bmm", "mm", "cpu")
        register_performance_model_fn(mock_model_c, "addmm", "addmm", "cuda")

        registered = list_registered_models()
        expected_keys = {
            ("cuda", "mm", "mm"),
            ("cpu", "bmm", "mm"),
            ("cuda", "addmm", "addmm"),
        }

        self.assertEqual(set(registered), expected_keys)

    def test_empty_template_list(self):
        """Test get_functions_for_templates with empty template list."""

        register_performance_model_fn(mock_model_a, "mm", "mm", "cuda")

        result = get_functions_for_templates([], "mm", "cuda")
        self.assertEqual(result, {})

    def test_no_matching_functions(self):
        """Test get_functions_for_templates when no templates have registered functions."""

        register_performance_model_fn(mock_model_a, "mm", "mm", "cuda")

        # Request templates that don't have registered functions
        result = get_functions_for_templates(["bmm", "addmm"], "mm", "cuda")
        self.assertEqual(result, {})

    def test_function_call_with_parameters(self):
        """Test that registered functions can be called with custom parameters."""

        # Create wrapper functions with different parameters
        def model_with_multiplier_2(choices, op_name):
            return mock_model_a(choices, op_name, multiplier=2.0)

        def model_with_offset_1_5(choices, op_name):
            return mock_model_b(choices, op_name, offset=1.5)

        # Register the wrapper functions
        register_performance_model_fn(model_with_multiplier_2, "mm", "mm", "cuda")
        register_performance_model_fn(model_with_offset_1_5, "addmm", "mm", "cuda")

        # Verify they can be retrieved and called
        func1 = get_model_function_for_key("mm", "mm", "cuda")
        func2 = get_model_function_for_key("addmm", "mm", "cuda")

        self.assertEqual(func1, model_with_multiplier_2)
        self.assertEqual(func2, model_with_offset_1_5)

        # Test calling the functions
        mock_choices = [MagicMock(), MagicMock()]

        func1(mock_choices, "mm")
        # Should set prediction to 1.0 * 2.0 = 2.0
        for choice in mock_choices:
            self.assertEqual(choice.performance_prediction, 2.0)

        mock_choices = [MagicMock(), MagicMock()]
        func2(mock_choices, "addmm")
        # Should set prediction to 2.0 + 1.5 = 3.5
        for choice in mock_choices:
            self.assertEqual(choice.performance_prediction, 3.5)


class TestFilterAndSortChoices(unittest.TestCase):
    """Test cases for filter_and_sort_choices function."""

    def _create_mock_ktc(self, performance_prediction=None):
        """Helper to create a mock KernelTemplateChoice object."""
        ktc = MagicMock()
        ktc.performance_prediction = performance_prediction
        return ktc

    def test_basic_filtering_and_sorting(self):
        """Test basic filtering and sorting functionality."""
        # Create choices with predictions
        ktc1 = self._create_mock_ktc(3.0)
        ktc2 = self._create_mock_ktc(1.0)  # best
        ktc3 = self._create_mock_ktc(2.0)
        ktc4 = self._create_mock_ktc(None)  # unranked

        ktc_stack = [ktc1, ktc2, ktc3, ktc4]

        # Test topk=-1 (all ranked) with discard_unranked=False
        result = filter_and_sort_choices(ktc_stack, topk=-1, discard_unranked=False)

        # Should have 3 ranked + 1 unranked = 4 total
        self.assertEqual(len(result), 4)
        # First three should be ranked in order
        self.assertEqual(result[0].performance_prediction, 1.0)
        self.assertEqual(result[1].performance_prediction, 2.0)
        self.assertEqual(result[2].performance_prediction, 3.0)
        # Last should be unranked
        self.assertIsNone(result[3].performance_prediction)

    def test_topk_filtering(self):
        """Test topk filtering works correctly."""
        # Create 5 ranked choices
        ktcs = [
            self._create_mock_ktc(5.0),
            self._create_mock_ktc(2.0),  # 2nd best
            self._create_mock_ktc(1.0),  # best
            self._create_mock_ktc(4.0),
            self._create_mock_ktc(3.0),  # 3rd best
        ]

        # Test topk=3
        result = filter_and_sort_choices(ktcs, topk=3, discard_unranked=False)

        self.assertEqual(len(result), 3)
        self.assertEqual(result[0].performance_prediction, 1.0)  # best
        self.assertEqual(result[1].performance_prediction, 2.0)  # 2nd
        self.assertEqual(result[2].performance_prediction, 3.0)  # 3rd

    def test_topk_larger_than_available(self):
        """Test topk larger than number of ranked choices."""
        ktcs = [
            self._create_mock_ktc(2.5),
            self._create_mock_ktc(1.2),  # best
            self._create_mock_ktc(None),  # unranked
        ]

        result = filter_and_sort_choices(ktcs, topk=5, discard_unranked=False)

        # Should get all 2 ranked + 1 unranked = 3 total
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0].performance_prediction, 1.2)
        self.assertEqual(result[1].performance_prediction, 2.5)
        self.assertIsNone(result[2].performance_prediction)

    def test_discard_unranked_true(self):
        """Test discard_unranked=True filters out unranked choices."""
        ktcs = [
            self._create_mock_ktc(2.0),
            self._create_mock_ktc(1.0),  # best
            self._create_mock_ktc(None),  # unranked - should be discarded
            self._create_mock_ktc(None),  # unranked - should be discarded
        ]

        result = filter_and_sort_choices(ktcs, topk=-1, discard_unranked=True)

        # Should only have the 2 ranked choices
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].performance_prediction, 1.0)
        self.assertEqual(result[1].performance_prediction, 2.0)

    def test_discard_unranked_false(self):
        """Test discard_unranked=False keeps unranked choices."""
        ktcs = [
            self._create_mock_ktc(2.0),
            self._create_mock_ktc(1.0),  # best
            self._create_mock_ktc(None),  # unranked - should be kept
        ]

        result = filter_and_sort_choices(ktcs, topk=-1, discard_unranked=False)

        # Should have 2 ranked + 1 unranked = 3 total
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0].performance_prediction, 1.0)
        self.assertEqual(result[1].performance_prediction, 2.0)
        self.assertIsNone(result[2].performance_prediction)

    def test_all_unranked_discard_true(self):
        """Test all unranked choices with discard_unranked=True returns empty list."""
        ktcs = [
            self._create_mock_ktc(None),
            self._create_mock_ktc(None),
            self._create_mock_ktc(None),
        ]

        result = filter_and_sort_choices(ktcs, topk=2, discard_unranked=True)

        # Should return empty list
        self.assertEqual(len(result), 0)

    def test_all_unranked_discard_false(self):
        """Test all unranked choices with discard_unranked=False keeps all."""
        ktcs = [
            self._create_mock_ktc(None),
            self._create_mock_ktc(None),
            self._create_mock_ktc(None),
        ]

        result = filter_and_sort_choices(ktcs, topk=2, discard_unranked=False)

        # Should keep all unranked choices
        self.assertEqual(len(result), 3)
        for ktc in result:
            self.assertIsNone(ktc.performance_prediction)

    def test_empty_stack(self):
        """Test empty stack returns empty list."""
        result = filter_and_sort_choices([], topk=5, discard_unranked=False)
        self.assertEqual(result, [])

    def test_topk_with_mixed_choices(self):
        """Test topk with both ranked and unranked choices."""
        ktcs = [
            self._create_mock_ktc(3.0),
            self._create_mock_ktc(1.0),  # best ranked
            self._create_mock_ktc(None),  # unranked
            self._create_mock_ktc(2.0),  # 2nd best ranked
            self._create_mock_ktc(None),  # unranked
        ]

        # topk=1 should give 1 ranked + 2 unranked (when discard_unranked=False)
        result = filter_and_sort_choices(ktcs, topk=1, discard_unranked=False)

        self.assertEqual(len(result), 3)  # 1 ranked + 2 unranked
        self.assertEqual(result[0].performance_prediction, 1.0)  # best ranked first

        # Remaining should be unranked
        unranked_count = sum(
            1 for ktc in result[1:] if ktc.performance_prediction is None
        )
        self.assertEqual(unranked_count, 2)


if __name__ == "__main__":
    run_tests()
