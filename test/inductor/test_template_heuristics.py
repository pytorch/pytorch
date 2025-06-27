# Owner(s): ["module: inductor"]

from torch._inductor.template_heuristics import _parse_valid_config, GemmConfig
from torch._inductor.test_case import run_tests, TestCase
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)


@instantiate_parametrized_tests
class TestTemplateHeuristics(TestCase):
    def test_parse_valid_config_basic_valid(self):
        """Test _parse_valid_config with basic valid tuple representations."""
        # Valid 5-parameter tuple
        result = _parse_valid_config("(64, 64, 32, 2, 4)")
        expected = [GemmConfig(64, 64, 32, 2, 4)]
        self.assertEqual(result, expected)

        # Valid 6-parameter tuple
        result = _parse_valid_config("(32, 32, 16, 1, 2, 8)")
        expected = [GemmConfig(32, 32, 16, 1, 2, 8)]
        self.assertEqual(result, expected)

        # Valid tuple with different values
        result = _parse_valid_config("(128, 128, 64, 3, 8)")
        expected = [GemmConfig(128, 128, 64, 3, 8)]
        self.assertEqual(result, expected)

    def test_parse_valid_config_with_spaces(self):
        """Test _parse_valid_config with spaces in input."""
        result = _parse_valid_config("( 64 , 64 , 32 , 2 , 4 )")
        expected = [GemmConfig(64, 64, 32, 2, 4)]
        self.assertEqual(result, expected)

    def test_parse_valid_config_empty_inputs(self):
        """Test _parse_valid_config with empty inputs."""
        # Empty string
        result = _parse_valid_config("")
        self.assertEqual(result, [])

        # Empty parentheses
        result = _parse_valid_config("()")
        self.assertEqual(result, [])

    @parametrize(
        "invalid_input",
        [
            "garbage",
            "(not, a, tuple)",
            "(1, 2, three, 4, 5)",
            "invalid_format",
            "(1,2,3,)",  # trailing comma with missing value
        ],
    )
    def test_parse_valid_config_garbage_input(self, invalid_input):
        """Test _parse_valid_config with invalid garbage input."""
        with self.assertRaises(ValueError):
            _parse_valid_config(invalid_input)

    @parametrize(
        "float_input",
        [
            "(64.5, 64, 32, 2, 4)",
            "(64, 64.0, 32, 2, 4)",
            "(64, 64, 32.5, 2, 4)",
            "(64, 64, 32, 2.5, 4)",
        ],
    )
    def test_parse_valid_config_floats_instead_of_ints(self, float_input):
        """Test _parse_valid_config with floats instead of ints (should fail)."""
        with self.assertRaises(ValueError):
            _parse_valid_config(float_input)

    @parametrize(
        "mixed_input",
        [
            "(64, 64, 32, two, 4)",
            "(64, 64, 32, 2.5, 4)",
        ],
    )
    def test_parse_valid_config_mixed_invalid(self, mixed_input):
        """Test _parse_valid_config with mixed valid/invalid inputs."""
        with self.assertRaises(ValueError):
            _parse_valid_config(mixed_input)

    def test_parse_valid_config_result_validation(self):
        """Test that _parse_valid_config returns properly structured results."""
        result = _parse_valid_config("(64, 64, 32, 2, 4)")

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

    @parametrize(
        "edge_case_input,expected_exception",
        [
            ("(64, 64, 32, 2, 4", ValueError),  # missing closing paren
            ("64, 64, 32, 2, 4)", ValueError),  # missing opening paren
            (
                "(64, 64)",
                ValueError,
            ),  # too few values - GemmConfig requires at least 5 args
        ],
    )
    def test_parse_valid_config_edge_cases(self, edge_case_input, expected_exception):
        """Test edge cases for _parse_valid_config function."""
        with self.assertRaises(expected_exception):
            _parse_valid_config(edge_case_input)

    def test_parse_valid_config_large_numbers(self):
        """Test _parse_valid_config with large numbers."""
        result = _parse_valid_config("(1024, 2048, 512, 10, 16)")
        self.assertEqual(len(result), 1)
        config = result[0]
        self.assertEqual(config.block_m, 1024)
        self.assertEqual(config.block_n, 2048)
        self.assertEqual(config.block_k, 512)
        self.assertEqual(config.num_stages, 10)
        self.assertEqual(config.num_warps, 16)


if __name__ == "__main__":
    run_tests()
