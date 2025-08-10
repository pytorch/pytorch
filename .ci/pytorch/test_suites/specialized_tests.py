"""
Specialized test suite implementations.

This module contains test suites for specialized PyTorch tests, including
backward compatibility, XLA, ONNX, JIT, and other specialized test configurations.
"""

import logging
import os
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from test_config.base import TestSuite
    from test_config.environment import EnvironmentConfig
    from utils.shell_utils import run_command, source_and_run, get_ci_dir
    from utils.install_utils import install_torchvision, install_torchaudio
except ImportError:
    from ..test_config.base import TestSuite
    from ..test_config.environment import EnvironmentConfig
    from ..utils.shell_utils import run_command, source_and_run, get_ci_dir
    from ..utils.install_utils import install_torchvision, install_torchaudio


class BackwardTestSuite(TestSuite):
    """Backward compatibility test suite."""
    
    def __init__(self):
        super().__init__("backward", "Backward compatibility tests")
    
    def matches(self, env_config: EnvironmentConfig) -> bool:
        """Check if this is a backward compatibility test configuration."""
        return env_config.test_config == "backward"
    
    def get_test_names(self) -> List[str]:
        return ["test_backward_compatibility"]
    
    def run(self, env_config: EnvironmentConfig) -> bool:
        """Run backward compatibility tests."""
        self.logger.info("Running backward compatibility tests")
        
        # Run the backward compatibility tests
        ci_dir = get_ci_dir()
        return source_and_run(
            str(ci_dir / "test.sh"),
            "test_backward_compatibility",
            cwd=str(ci_dir.parent.parent)
        )


class XLATestSuite(TestSuite):
    """XLA test suite."""
    
    def __init__(self):
        super().__init__("xla", "XLA tests")
    
    def matches(self, env_config: EnvironmentConfig) -> bool:
        """Check if this is an XLA test configuration."""
        return "xla" in env_config.test_config
    
    def get_test_names(self) -> List[str]:
        return ["test_xla"]
    
    def run(self, env_config: EnvironmentConfig) -> bool:
        """Run XLA tests."""
        self.logger.info("Running XLA tests")
        
        # Determine which XLA test to run
        test_func = "test_xla"
        if "xla_full" in env_config.test_config:
            test_func = "test_xla_full"
        elif "xla_nightly" in env_config.test_config:
            test_func = "test_xla_nightly"
        
        # Run the XLA tests
        ci_dir = get_ci_dir()
        return source_and_run(
            str(ci_dir / "test.sh"),
            test_func,
            cwd=str(ci_dir.parent.parent)
        )


class ONNXTestSuite(TestSuite):
    """ONNX test suite."""
    
    def __init__(self):
        super().__init__("onnx", "ONNX tests")
    
    def matches(self, env_config: EnvironmentConfig) -> bool:
        """Check if this is an ONNX test configuration."""
        return "onnx" in env_config.test_config
    
    def get_test_names(self) -> List[str]:
        return ["test_onnx"]
    
    def setup(self, env_config: EnvironmentConfig) -> bool:
        """Install dependencies for ONNX tests."""
        self.logger.info("Installing ONNX dependencies")
        
        # Install ONNX dependencies
        dependencies = ["onnx", "onnxruntime"]
        if not install_pip_dependencies(dependencies):
            self.logger.error(f"Failed to install dependencies: {dependencies}")
            return False
        
        return True
    
    def run(self, env_config: EnvironmentConfig) -> bool:
        """Run ONNX tests."""
        self.logger.info("Running ONNX tests")
        
        # Set up environment
        if not self.setup(env_config):
            return False
        
        # Run the ONNX tests
        ci_dir = get_ci_dir()
        return source_and_run(
            str(ci_dir / "test.sh"),
            "test_onnx",
            cwd=str(ci_dir.parent.parent)
        )


class JITLegacyTestSuite(TestSuite):
    """JIT legacy test suite."""
    
    def __init__(self):
        super().__init__("jit_legacy", "JIT legacy tests")
    
    def matches(self, env_config: EnvironmentConfig) -> bool:
        """Check if this is a JIT legacy test configuration."""
        return env_config.test_config == "jit_legacy"
    
    def get_test_names(self) -> List[str]:
        return ["test_jit_legacy"]
    
    def run(self, env_config: EnvironmentConfig) -> bool:
        """Run JIT legacy tests."""
        self.logger.info("Running JIT legacy tests")
        
        # Run the JIT legacy tests
        ci_dir = get_ci_dir()
        return source_and_run(
            str(ci_dir / "test.sh"),
            "test_jit_legacy",
            cwd=str(ci_dir.parent.parent)
        )


class Aarch64TestSuite(TestSuite):
    """Aarch64 test suite."""
    
    def __init__(self):
        super().__init__("aarch64", "Aarch64 tests")
    
    def matches(self, env_config: EnvironmentConfig) -> bool:
        """Check if this is an Aarch64 test configuration."""
        return env_config.test_config == "aarch64"
    
    def get_test_names(self) -> List[str]:
        return ["test_aarch64"]
    
    def run(self, env_config: EnvironmentConfig) -> bool:
        """Run Aarch64 tests."""
        self.logger.info("Running Aarch64 tests")
        
        # Run the Aarch64 tests
        ci_dir = get_ci_dir()
        return source_and_run(
            str(ci_dir / "test.sh"),
            "test_aarch64",
            cwd=str(ci_dir.parent.parent)
        )


class CrossCompileTestSuite(TestSuite):
    """Cross compile test suite."""
    
    def __init__(self):
        super().__init__("crosscompile", "Cross compile tests")
    
    def matches(self, env_config: EnvironmentConfig) -> bool:
        """Check if this is a cross compile test configuration."""
        return env_config.test_config == "crosscompile"
    
    def get_test_names(self) -> List[str]:
        return ["test_crosscompile"]
    
    def run(self, env_config: EnvironmentConfig) -> bool:
        """Run cross compile tests."""
        self.logger.info("Running cross compile tests")
        
        # Run the cross compile tests
        ci_dir = get_ci_dir()
        return source_and_run(
            str(ci_dir / "test.sh"),
            "test_crosscompile",
            cwd=str(ci_dir.parent.parent)
        )


class ROCmTestSuite(TestSuite):
    """ROCm test suite."""
    
    def __init__(self):
        super().__init__("rocm", "ROCm tests")
    
    def matches(self, env_config: EnvironmentConfig) -> bool:
        """Check if this is a ROCm test configuration."""
        return env_config.is_rocm and "rocm" in env_config.test_config
    
    def get_test_names(self) -> List[str]:
        return ["test_rocm"]
    
    def run(self, env_config: EnvironmentConfig) -> bool:
        """Run ROCm tests."""
        self.logger.info("Running ROCm tests")
        
        # Run the ROCm tests
        ci_dir = get_ci_dir()
        return source_and_run(
            str(ci_dir / "test.sh"),
            "test_rocm",
            cwd=str(ci_dir.parent.parent)
        )


class ASANTestSuite(TestSuite):
    """ASAN test suite."""
    
    def __init__(self):
        super().__init__("asan", "ASAN tests")
    
    def matches(self, env_config: EnvironmentConfig) -> bool:
        """Check if this is an ASAN test configuration."""
        return env_config.is_asan
    
    def get_test_names(self) -> List[str]:
        return ["test_asan"]
    
    def run(self, env_config: EnvironmentConfig) -> bool:
        """Run ASAN tests."""
        self.logger.info("Running ASAN tests")
        
        # Run the ASAN tests
        ci_dir = get_ci_dir()
        return source_and_run(
            str(ci_dir / "test.sh"),
            "test_asan",
            cwd=str(ci_dir.parent.parent)
        )
