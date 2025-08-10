"""
Python test suite implementations.

This module contains test suites for Python-based PyTorch testing,
including standard Python tests, sharded tests, and smoke tests.
"""

import logging
import os
from typing import List, Dict, Any

try:
    from test_config.base import TestSuite
    from test_config.environment import EnvironmentConfig
    from utils.shell_utils import run_command, source_and_run, get_ci_dir
    from utils.install_utils import install_torchvision, install_monkeytype, install_numpy_2_compatibility
except ImportError:
    from ..test_config.base import TestSuite
    from ..test_config.environment import EnvironmentConfig
    from ..utils.shell_utils import run_command, source_and_run, get_ci_dir
    from ..utils.install_utils import install_torchvision, install_monkeytype, install_numpy_2_compatibility


class PythonTestSuite(TestSuite):
    """Standard Python test suite."""
    
    def __init__(self):
        super().__init__("python", "Standard Python tests")
    
    def matches(self, env_config: EnvironmentConfig) -> bool:
        """This suite is used by the default test configuration."""
        return False  # Handled by registry logic
    
    def get_test_names(self) -> List[str]:
        return ["test_python"]
    
    def run(self, env_config: EnvironmentConfig) -> bool:
        """Run standard Python tests."""
        self.logger.info("Running standard Python tests")
        
        # For transitional period, call the original shell function
        ci_dir = get_ci_dir()
        return source_and_run(
            str(ci_dir / "test.sh"),
            "test_python",
            cwd=str(ci_dir.parent.parent)
        )


class PythonShardTestSuite(TestSuite):
    """Sharded Python test suite."""
    
    def __init__(self):
        super().__init__("python_shard", "Sharded Python tests")
    
    def matches(self, env_config: EnvironmentConfig) -> bool:
        """Check if this should run sharded tests."""
        # This would be the default case in the original script
        return True
    
    def get_test_names(self) -> List[str]:
        return [f"test_python_shard_{self._get_shard_number()}"]
    
    def _get_shard_number(self) -> int:
        """Get the current shard number."""
        return int(os.environ.get('SHARD_NUMBER', '1'))
    
    def run(self, env_config: EnvironmentConfig) -> bool:
        """Run sharded Python tests."""
        shard_number = env_config.shard_number
        self.logger.info(f"Running Python tests for shard {shard_number}")
        
        # Install dependencies
        if not install_torchvision():
            self.logger.error("Failed to install torchvision")
            return False
        
        if not install_monkeytype():
            self.logger.error("Failed to install monkeytype")
            return False
        
        # Run the sharded test
        ci_dir = get_ci_dir()
        return source_and_run(
            str(ci_dir / "test.sh"),
            f"test_python_shard {shard_number}",
            cwd=str(ci_dir.parent.parent)
        )


class PythonSmokeTestSuite(TestSuite):
    """Python smoke test suite."""
    
    def __init__(self):
        super().__init__("python_smoke", "Python smoke tests")
    
    def matches(self, env_config: EnvironmentConfig) -> bool:
        """Check if this is a smoke test configuration."""
        return env_config.test_config == "smoke"
    
    def get_test_names(self) -> List[str]:
        return ["test_python_smoke"]
    
    def run(self, env_config: EnvironmentConfig) -> bool:
        """Run Python smoke tests."""
        self.logger.info("Running Python smoke tests")
        
        ci_dir = get_ci_dir()
        return source_and_run(
            str(ci_dir / "test.sh"),
            "test_python_smoke",
            cwd=str(ci_dir.parent.parent)
        )


class PythonLegacyJITTestSuite(TestSuite):
    """Legacy JIT Python test suite."""
    
    def __init__(self):
        super().__init__("python_legacy_jit", "Legacy JIT Python tests")
    
    def matches(self, env_config: EnvironmentConfig) -> bool:
        """Check if this is a legacy JIT test configuration."""
        return env_config.test_config == "jit_legacy"
    
    def get_test_names(self) -> List[str]:
        return ["test_python_legacy_jit"]
    
    def run(self, env_config: EnvironmentConfig) -> bool:
        """Run legacy JIT Python tests."""
        self.logger.info("Running legacy JIT Python tests")
        
        ci_dir = get_ci_dir()
        return source_and_run(
            str(ci_dir / "test.sh"),
            "test_python_legacy_jit",
            cwd=str(ci_dir.parent.parent)
        )


class NumPy2CompatibilityTestSuite(TestSuite):
    """NumPy 2.0 compatibility test suite."""
    
    def __init__(self):
        super().__init__("numpy_2_compatibility", "NumPy 2.0 compatibility tests")
    
    def matches(self, env_config: EnvironmentConfig) -> bool:
        """Check if this is a NumPy 2.0 test configuration."""
        return "numpy_2" in env_config.test_config
    
    def get_test_names(self) -> List[str]:
        return ["numpy_2_compatibility_tests"]
    
    def run(self, env_config: EnvironmentConfig) -> bool:
        """Run NumPy 2.0 compatibility tests."""
        self.logger.info("Running NumPy 2.0 compatibility tests")
        
        # Install NumPy 2.0 and compatible packages
        from ..utils.install_utils import install_numpy_2_compatibility
        if not install_numpy_2_compatibility():
            self.logger.error("Failed to install NumPy 2.0 compatibility packages")
            return False
        
        # Run specific tests for NumPy 2.0 compatibility
        test_files = [
            "dynamo/test_functions.py",
            "dynamo/test_unspec.py", 
            "test_binary_ufuncs.py",
            "test_fake_tensor.py",
            "test_linalg.py",
            "test_numpy_interop.py",
            "test_tensor_creation_ops.py",
            "test_torch.py",
            "torch_np/test_basic.py"
        ]
        
        test_args = " ".join([f"--include {test}" for test in test_files])
        command = f"python test/run_test.py {test_args}"
        
        return run_command(command, check=False)
