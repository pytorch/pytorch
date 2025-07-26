# Build CLI Test Suite

This directory contains unit tests for the PyTorch Build CLI tool, which is a Cement-based CLI application for building external components like vLLM.

## Test Structure

The test suite is organized into the following files:

- `conftest.py`: Common fixtures used across all test files
- `test_utils.py`: Tests for utility functions in `utils.py`
- `test_external_vllm_build.py`: Tests for vLLM build functionality in `controllers/external_vllm_build.py`
- `test_external.py`: Tests for the external build controller in `controllers/external.py`
- `test_cli.py`: Tests for the main CLI application in `cli.py`

## Running Tests

To run the tests, you need to have pytest installed. You can install it with pip:

```bash
pip install pytest
```

Then, from the `build_cli` directory, run:

```bash
pytest tests/
```

To run a specific test file:

```bash
pytest tests/test_utils.py
```

To run a specific test:

```bash
pytest tests/test_utils.py::TestRun::test_run_basic
```

To run tests with verbose output:

```bash
pytest -v tests/
```

## Test Coverage

The tests cover the following functionality:

### Utils Tests
- `run`: Command execution with various parameters
- `get_post_build_pinned_commit`: Retrieving pinned commit hashes
- `get_env`: Environment variable handling
- `create_directory`: Directory creation with cleanup
- `delete_directory`: Directory deletion
- `Timer`: Performance timing context manager

### External vLLM Build Tests
- `clone_vllm`: Git clone and checkout functionality
- `build_vllm`: Docker build command generation with various environment variables

### External Controller Tests
- Controller registration
- `run` command with vLLM target
- `help` command with vLLM target
- Error handling for unknown targets

### CLI Tests
- App creation and metadata
- Controller registration
- Main function execution

## Mocking Strategy

The tests use pytest fixtures to mock external dependencies:

- `subprocess.run`: To avoid executing actual commands
- `os.environ`: To control environment variables
- `Path.exists` and `Path.read_text`: To control file operations
- `os.makedirs` and `shutil.rmtree`: To avoid actual filesystem changes

This allows the tests to run in isolation without affecting the actual system.
