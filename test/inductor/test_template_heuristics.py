# Owner(s): ["module: inductor"]

from torch._inductor.template_heuristics import _config_from_lookup, GemmConfig
from torch._inductor.test_case import run_tests, TestCase
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)


@instantiate_parametrized_tests
class TestTemplateHeuristics(TestCase):
    def test_config_from_lookup_basic_valid(self):
        """Test _config_from_lookup with basic valid list inputs."""
        # Valid 5-parameter list
        result = _config_from_lookup([64, 64, 32, 2, 4])
        expected = [GemmConfig(64, 64, 32, 2, 4)]
        self.assertEqual(result, expected)

        # Valid 6-parameter list
        result = _config_from_lookup([32, 32, 16, 1, 2, 8])
        expected = [GemmConfig(32, 32, 16, 1, 2, 8)]
        self.assertEqual(result, expected)

        # Valid list with different values
        result = _config_from_lookup([128, 128, 64, 3, 8])
        expected = [GemmConfig(128, 128, 64, 3, 8)]
        self.assertEqual(result, expected)

    def test_config_from_lookup_empty_inputs(self):
        """Test _config_from_lookup with empty inputs."""
        # None input
        result = _config_from_lookup(None)
        self.assertEqual(result, [])

        # Empty list
        result = _config_from_lookup([])
        self.assertEqual(result, [])

    @parametrize(
        "invalid_input",
        [
            "not_a_list",
            (64, 64, 32, 2, 4),  # tuple instead of list
            {"block_m": 64, "block_n": 64},  # dict instead of list
            [64, 64, 32, 2],  # too few elements (4 instead of 5 or 6)
            [64, 64, 32, 2, 4, 8, 16],  # too many elements (7 instead of 5 or 6)
            [64, 64, 32],  # too few elements (3 instead of 5 or 6)
        ],
    )
    def test_config_from_lookup_invalid_input_types(self, invalid_input):
        """Test _config_from_lookup with invalid input types or sizes."""
        with self.assertRaises(ValueError):
            _config_from_lookup(invalid_input)

    @parametrize(
        "non_int_input",
        [
            [64.5, 64, 32, 2, 4],  # float in first position
            [64, 64.0, 32, 2, 4],  # float in second position
            [64, 64, 32.5, 2, 4],  # float in third position
            [64, 64, 32, 2.5, 4],  # float in fourth position
            [64, 64, 32, 2, 4.0],  # float in fifth position
            [64, 64, 32, 2, 4, 8.5],  # float in sixth position
            ["64", 64, 32, 2, 4],  # string instead of int
            [64, "64", 32, 2, 4],  # string instead of int
            [64, 64, None, 2, 4],  # None instead of int
        ],
    )
    def test_config_from_lookup_non_int_elements(self, non_int_input):
        """Test _config_from_lookup with non-integer elements (should fail)."""
        with self.assertRaises(ValueError):
            _config_from_lookup(non_int_input)

    def test_config_from_lookup_result_validation(self):
        """Test that _config_from_lookup returns properly structured results."""
        result = _config_from_lookup([64, 64, 32, 2, 4])

        # Check result structure
        self.assertEqual(len(result), 1)
        config = result[0]
        self.assertIsInstance(config, GemmConfig)

        # Verify all attributes are integers
        self.assertIsInstance(config.block_m, int)
        self.assertIsInstance(config.block_n, int)
        self.assertIsInstance(config.block_k, int)
        self.assertIsInstance(config.num_stages, int)
        self.assertIsInstance(config.num_warps, int)

        # Verify values match input
        self.assertEqual(config.block_m, 64)
        self.assertEqual(config.block_n, 64)
        self.assertEqual(config.block_k, 32)
        self.assertEqual(config.num_stages, 2)
        self.assertEqual(config.num_warps, 4)

    def test_config_from_lookup_six_parameter_validation(self):
        """Test that _config_from_lookup handles 6-parameter configs correctly."""
        result = _config_from_lookup([32, 32, 16, 1, 2, 8])

        # Check result structure
        self.assertEqual(len(result), 1)
        config = result[0]
        self.assertIsInstance(config, GemmConfig)

        # Verify all attributes including group_m
        self.assertEqual(config.block_m, 32)
        self.assertEqual(config.block_n, 32)
        self.assertEqual(config.block_k, 16)
        self.assertEqual(config.num_stages, 1)
        self.assertEqual(config.num_warps, 2)
        self.assertEqual(config.group_m, 8)

    def test_config_from_lookup_large_numbers(self):
        """Test _config_from_lookup with large numbers."""
        result = _config_from_lookup([1024, 2048, 512, 10, 16])
        self.assertEqual(len(result), 1)
        config = result[0]
        self.assertEqual(config.block_m, 1024)
        self.assertEqual(config.block_n, 2048)
        self.assertEqual(config.block_k, 512)
        self.assertEqual(config.num_stages, 10)
        self.assertEqual(config.num_warps, 16)

    def test_config_from_lookup_zero_values(self):
        """Test _config_from_lookup with zero values (edge case)."""
        result = _config_from_lookup([0, 0, 0, 0, 0])
        self.assertEqual(len(result), 1)
        config = result[0]
        self.assertEqual(config.block_m, 0)
        self.assertEqual(config.block_n, 0)
        self.assertEqual(config.block_k, 0)
        self.assertEqual(config.num_stages, 0)
        self.assertEqual(config.num_warps, 0)

    def test_config_from_lookup_negative_values(self):
        """Test _config_from_lookup with negative values (edge case)."""
        # This should work as the function doesn't validate value ranges
        result = _config_from_lookup([-1, -2, -3, -4, -5])
        self.assertEqual(len(result), 1)
        config = result[0]
        self.assertEqual(config.block_m, -1)
        self.assertEqual(config.block_n, -2)
        self.assertEqual(config.block_k, -3)
        self.assertEqual(config.num_stages, -4)
        self.assertEqual(config.num_warps, -5)

    @parametrize(
        "invalid_size_input,expected_message",
        [
            ([64], "Expected list of 5 or 6 ints"),
            ([64, 64], "Expected list of 5 or 6 ints"),
            ([64, 64, 32], "Expected list of 5 or 6 ints"),
            ([64, 64, 32, 2], "Expected list of 5 or 6 ints"),
            ([64, 64, 32, 2, 4, 8, 16], "Expected list of 5 or 6 ints"),
        ],
    )
    def test_config_from_lookup_invalid_sizes(
        self, invalid_size_input, expected_message
    ):
        """Test _config_from_lookup with invalid list sizes."""
        with self.assertRaises(ValueError) as cm:
            _config_from_lookup(invalid_size_input)
        self.assertIn(expected_message, str(cm.exception))

    def test_config_from_lookup_error_messages(self):
        """Test that _config_from_lookup provides helpful error messages."""
        # Test non-list input
        with self.assertRaises(ValueError) as cm:
            _config_from_lookup("not_a_list")
        self.assertIn("Expected list format", str(cm.exception))

        # Test non-integer elements
        with self.assertRaises(ValueError) as cm:
            _config_from_lookup([64, "not_int", 32, 2, 4])
        self.assertIn("Expected list of ints", str(cm.exception))

        # Test wrong size
        with self.assertRaises(ValueError) as cm:
            _config_from_lookup([64, 64, 32])
        self.assertIn("Expected list of 5 or 6 ints", str(cm.exception))


if __name__ == "__main__":
    run_tests()
