"""
PyTorch test environment configuration module.

This subpackage provides centralized environment variable management and build
verification for PyTorch CI tests.

Structure:
    - torch_test_env_config.py: Main PytorchTestEnvironment class
    - properties.py: Property mixins for build/test config checks
    - env_setup.py: Environment setup methods (_setup_*)
    - env_verification.py: Build verification methods (_verify_*)

Usage:
    from cli.lib.core.pytorch.env_config import PytorchTestEnvironment

    env = PytorchTestEnvironment(
        build_environment="linux-focal-cuda12.1-py3.10",
        test_config="default",
    )
    env.apply()
    env.verify_build_configuration()
"""

from .torch_test_env_config import PytorchTestEnvironment

__all__ = ["PytorchTestEnvironment"]
