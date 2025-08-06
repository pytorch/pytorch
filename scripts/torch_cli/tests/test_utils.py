import os
import unittest
from unittest import mock
from unittest.mock import patch

from cli.lib.utils import read_yaml_file

from utils import create_temp_yaml


class TestReadYamlFile(unittest.TestCase):
    def setUp(self):
        os.environ.pop("EXISTING_VAR", None)
        os.environ.pop("MISSING_VAR", None)
        os.environ.pop("ANOTHER_MISSING", None)

    def test_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            read_yaml_file("/nonexistent/file.yaml")

    def test_valid_yaml_with_existing_env_var(self):
        os.environ["EXISTING_VAR"] = "yes"
        content = {
            "key1": "$EXISTING_VAR",
            "key2": "literal",
        }
        path = create_temp_yaml(content)
        result = read_yaml_file(path)
        self.assertEqual(result["key1"], True)
        self.assertEqual(result["key2"], "literal")

    def test_missing_env_vars_warned_and_removed(self):
        content = {
            "key1": "$MISSING_VAR",
            "key2": "${ANOTHER_MISSING}",
        }
        path = create_temp_yaml(content)

        with patch("cli.lib.utils.logger") as mock_logger:
            result = read_yaml_file(path)

            self.assertIsNone(result["key1"])
            self.assertIsNone(result["key2"])

            mock_logger.warning.assert_called_once()
            warning_msg = mock_logger.warning.call_args[0][0]
            self.assertIn("Missing environment variables", warning_msg)

    def test_yaml_content_not_dict(self):
        path = create_temp_yaml('["item1", "item2"]')

        with self.assertRaises(ValueError):
            read_yaml_file(path)

    def test_empty_yaml_returns_empty_dict(self):
        path = create_temp_yaml({})
        result = read_yaml_file(path)
        self.assertEqual(result, {})

    def test_invalid_yaml_syntax(self):
        content = "{ invalid: yaml: ["
        path = create_temp_yaml(content)
        with self.assertRaises(ValueError) as cm:
            read_yaml_file(path)
        self.assertIn("Failed to parse YAML file", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
