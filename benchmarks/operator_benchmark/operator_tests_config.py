"""
Configuration file for operator microbenchmark tests.

This file reads the configuration from operator_tests_config.yml
and provides a Python interface for accessing the configuration.
"""

import os
import yaml
from pathlib import Path

def load_config():
    """Load configuration from YAML file."""
    config_path = Path(__file__).parent / "operator_tests_config.yml"
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        # Fallback to hardcoded values if YAML file doesn't exist
        return {
            "operator_benchmark_tests": ["matmul", "mm", "addmm", "bmm"],
            "operator_benchmark_config": {
                "tag_filter": "long",
                "benchmark_name": "PyTorch operator microbenchmark",
                "use_compile": True
            }
        }

# Load configuration
_config = load_config()

# Export configuration variables
OPERATOR_BENCHMARK_TESTS = _config["operator_benchmark_tests"]
OPERATOR_BENCHMARK_CONFIG = _config["operator_benchmark_config"]

# For backward compatibility, also export the test matrix if available
if "test_matrix" in _config:
    TEST_MATRIX = _config["test_matrix"]
else:
    TEST_MATRIX = None
