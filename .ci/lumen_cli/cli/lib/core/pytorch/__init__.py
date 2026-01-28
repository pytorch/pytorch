"""
PyTorch test environment configuration module.

This module provides centralized environment variable management and build
verification for PyTorch CI tests.

Structure:
    - env_config/: Subpackage with environment configuration
        - torch_test_env_config.py: Main PytorchTestEnvironment class
        - properties.py: Property mixins for build/test config checks
        - env_setup.py: Environment setup methods (_setup_*)
        - env_verification.py: Build verification methods (_verify_*)

Usage:
    from cli.lib.core.pytorch import PytorchTestEnvironment

    env = PytorchTestEnvironment(
        build_environment="linux-focal-cuda12.1-py3.10",
        test_config="default",
    )
    env.apply()
    env.verify_build_configuration()
"""

from .env_config import PytorchTestEnvironment

__all__ = ["PytorchTestEnvironment"]
