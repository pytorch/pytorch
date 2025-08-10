"""
Distributed test suite implementations.

This module contains test suites for PyTorch distributed training and RPC testing.
"""

import logging
import os
from typing import List, Dict, Any

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


class DistributedTestSuite(TestSuite):
    """Distributed training test suite."""
    
    def __init__(self):
        super().__init__("distributed", "Distributed training tests")
    
    def matches(self, env_config: EnvironmentConfig) -> bool:
        """Check if this is a distributed test configuration."""
        return env_config.test_config == "distributed"
    
    def get_test_names(self) -> List[str]:
        return ["test_distributed", "test_rpc"]
    
    def run(self, env_config: EnvironmentConfig) -> bool:
        """Run distributed tests."""
        self.logger.info("Running distributed training tests")
        
        # Run distributed tests
        ci_dir = get_ci_dir()
        success = source_and_run(
            str(ci_dir / "test.sh"),
            "test_distributed",
            cwd=str(ci_dir.parent.parent)
        )
        
        # Only run RPC C++ tests on the first shard
        if env_config.shard_number == 1:
            self.logger.info("Running RPC tests (shard 1 only)")
            rpc_success = source_and_run(
                str(ci_dir / "test.sh"),
                "test_rpc",
                cwd=str(ci_dir.parent.parent)
            )
            success = success and rpc_success
        
        return success


class H100DistributedTestSuite(TestSuite):
    """H100 distributed test suite."""
    
    def __init__(self):
        super().__init__("h100_distributed", "H100 distributed tests")
    
    def matches(self, env_config: EnvironmentConfig) -> bool:
        """Check if this is an H100 distributed test configuration."""
        return env_config.test_config == "h100_distributed"
    
    def get_test_names(self) -> List[str]:
        return ["test_h100_distributed"]
    
    def run(self, env_config: EnvironmentConfig) -> bool:
        """Run H100 distributed tests."""
        self.logger.info("Running H100 distributed tests")
        
        ci_dir = get_ci_dir()
        return source_and_run(
            str(ci_dir / "test.sh"),
            "test_h100_distributed",
            cwd=str(ci_dir.parent.parent)
        )


class InductorDistributedTestSuite(TestSuite):
    """Inductor distributed test suite."""
    
    def __init__(self):
        super().__init__("inductor_distributed", "Inductor distributed tests")
    
    def matches(self, env_config: EnvironmentConfig) -> bool:
        """Check if this is an Inductor distributed test configuration."""
        return "inductor_distributed" in env_config.test_config
    
    def get_test_names(self) -> List[str]:
        return ["test_inductor_distributed"]
    
    def run(self, env_config: EnvironmentConfig) -> bool:
        """Run Inductor distributed tests."""
        self.logger.info("Running Inductor distributed tests")
        
        ci_dir = get_ci_dir()
        return source_and_run(
            str(ci_dir / "test.sh"),
            "test_inductor_distributed",
            cwd=str(ci_dir.parent.parent)
        )
