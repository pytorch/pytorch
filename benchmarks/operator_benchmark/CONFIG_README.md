# Operator Benchmark Configuration

This directory contains configuration files for the operator microbenchmark tests.

## Files

- `operator_tests_config.yml` - Main configuration file in YAML format
- `operator_tests_config.py` - Python module that reads the YAML configuration
- `get_config.py` - Helper script to extract configuration in various formats

## Configuration

The operator benchmark tests list and configuration are defined in `operator_tests_config.yml`:

```yaml
operator_benchmark_tests:
  - matmul
  - mm
  - addmm
  - bmm

operator_benchmark_config:
  tag_filter: "long"
  benchmark_name: "PyTorch operator microbenchmark"
  use_compile: true
```

## Usage

### In Python scripts

```python
from operator_tests_config import OPERATOR_BENCHMARK_TESTS, OPERATOR_BENCHMARK_CONFIG

# Get the list of tests
tests = OPERATOR_BENCHMARK_TESTS  # ['matmul', 'mm', 'addmm', 'bmm']

# Get configuration
config = OPERATOR_BENCHMARK_CONFIG
```

### In shell scripts

```bash
# Load configuration as environment variables
eval $(python get_config.py --format shell)

# Now you can use:
# $OP_BENCHMARK_TESTS
# $OP_BENCHMARK_TAG_FILTER
# $OP_BENCHMARK_NAME
# $OP_BENCHMARK_USE_COMPILE
```

### Command line usage

```bash
# List all tests
python get_config.py --tests-only

# Get configuration in JSON format
python get_config.py --format json

# Get configuration in YAML format
python get_config.py --format yaml

# Get shell variables
python get_config.py --format shell
```

## Adding new tests

To add new operator benchmark tests:

1. Edit `operator_tests_config.yml`
2. Add the new test name to the `operator_benchmark_tests` list
3. Ensure the corresponding test file exists in the `pt/` directory (e.g., `new_test_test.py`)

## Modifying configuration

To change test parameters:

1. Edit `operator_tests_config.yml`
2. Modify the `operator_benchmark_config` section as needed
3. The changes will be automatically picked up by the test scripts
