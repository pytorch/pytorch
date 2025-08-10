"""
Benchmark test suite implementations.

This module contains test suites for PyTorch benchmarking, including
operator benchmarks, TorchBench, and Dynamo benchmarks.
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
    from ..utils.install_utils import install_torchvision, install_torchaudio, install_torchao, install_opencv


class OperatorBenchmarkTestSuite(TestSuite):
    """Operator benchmark test suite."""
    
    def __init__(self):
        super().__init__("operator_benchmark", "Operator benchmarking tests")
    
    def matches(self, env_config: EnvironmentConfig) -> bool:
        """Check if this is an operator benchmark test configuration."""
        return "operator_benchmark" in env_config.test_config
    
    def get_test_names(self) -> List[str]:
        return ["test_operator_benchmark"]
    
    def run(self, env_config: EnvironmentConfig) -> bool:
        """Run operator benchmark tests."""
        self.logger.info("Running operator benchmark tests")
        
        # Determine test mode
        test_mode = "short"  # Default
        if "long" in env_config.test_config:
            test_mode = "long"
        elif "all" in env_config.test_config:
            test_mode = "all"
        
        # Determine device
        device = "cpu"
        if "cuda" in env_config.test_config:
            device = "cuda"
        
        self.logger.info(f"Benchmark mode: {test_mode}, device: {device}")
        
        # Run the benchmark
        ci_dir = get_ci_dir()
        return source_and_run(
            str(ci_dir / "test.sh"),
            f"test_operator_benchmark {device} {test_mode}",
            cwd=str(ci_dir.parent.parent)
        )


class TorchBenchTestSuite(TestSuite):
    """TorchBench test suite."""
    
    def __init__(self):
        super().__init__("torchbench", "TorchBench performance tests")
    
    def matches(self, env_config: EnvironmentConfig) -> bool:
        """Check if this is a TorchBench test configuration."""
        return "torchbench" in env_config.test_config
    
    def get_test_names(self) -> List[str]:
        return ["test_torchbench"]
    
    def setup(self, env_config: EnvironmentConfig) -> bool:
        """Install dependencies for TorchBench."""
        self.logger.info("Installing TorchBench dependencies")
        
        # Install dependencies
        if not install_torchaudio():
            self.logger.error("Failed to install torchaudio")
            return False
            
        if not install_torchvision():
            self.logger.error("Failed to install torchvision")
            return False
            
        if not install_torchao():
            self.logger.error("Failed to install torchao")
            return False
            
        if not install_opencv():
            self.logger.error("Failed to install opencv-python")
            return False
        
        # Install torchrec and fbgemm for non-CPU configs
        if "cpu" not in env_config.test_config:
            from ..utils.install_utils import install_torchrec_and_fbgemm
            if not install_torchrec_and_fbgemm():
                self.logger.error("Failed to install torchrec and fbgemm")
                return False
        
        return True
    
    def run(self, env_config: EnvironmentConfig) -> bool:
        """Run TorchBench tests."""
        self.logger.info("Running TorchBench tests")
        
        # Set up environment
        if not self.setup(env_config):
            return False
        
        # Determine which specific TorchBench test to run
        if "inductor_torchbench_smoketest_perf" in env_config.test_config:
            test_func = "test_inductor_torchbench_smoketest_perf"
        elif "inductor_torchbench_cpu_smoketest_perf" in env_config.test_config:
            test_func = "test_inductor_torchbench_cpu_smoketest_perf"
        elif "torchbench_gcp_smoketest" in env_config.test_config:
            test_func = "test_torchbench_gcp_smoketest"
        else:
            # Default TorchBench test
            shard_id = env_config.shard_number - 1
            test_func = f"test_torchbench {shard_id}"
        
        # Run the benchmark
        ci_dir = get_ci_dir()
        env = {"PYTHONPATH": "/torchbench"}
        
        return source_and_run(
            str(ci_dir / "test.sh"),
            test_func,
            cwd=str(ci_dir.parent.parent),
            env=env
        )


class DynamoBenchmarkTestSuite(TestSuite):
    """Dynamo benchmark test suite."""
    
    def __init__(self):
        super().__init__("dynamo_benchmark", "Dynamo benchmarking tests")
    
    def matches(self, env_config: EnvironmentConfig) -> bool:
        """Check if this is a Dynamo benchmark test configuration."""
        return ("huggingface" in env_config.test_config or 
                "timm" in env_config.test_config)
    
    def get_test_names(self) -> List[str]:
        return ["test_dynamo_benchmark"]
    
    def setup(self, env_config: EnvironmentConfig) -> bool:
        """Install dependencies for Dynamo benchmarks."""
        self.logger.info("Installing Dynamo benchmark dependencies")
        
        if not install_torchvision():
            self.logger.error("Failed to install torchvision")
            return False
        
        return True
    
    def run(self, env_config: EnvironmentConfig) -> bool:
        """Run Dynamo benchmark tests."""
        self.logger.info("Running Dynamo benchmark tests")
        
        # Set up environment
        if not self.setup(env_config):
            return False
        
        # Determine benchmark type
        benchmark_type = "huggingface" if "huggingface" in env_config.test_config else "timm_models"
        shard_id = env_config.shard_number - 1
        
        # Run the benchmark
        ci_dir = get_ci_dir()
        return source_and_run(
            str(ci_dir / "test.sh"),
            f"test_dynamo_benchmark {benchmark_type} {shard_id}",
            cwd=str(ci_dir.parent.parent)
        )


class CacheBenchTestSuite(TestSuite):
    """CacheBench test suite."""
    
    def __init__(self):
        super().__init__("cachebench", "CacheBench tests")
    
    def matches(self, env_config: EnvironmentConfig) -> bool:
        """Check if this is a CacheBench test configuration."""
        return env_config.test_config == "cachebench"
    
    def get_test_names(self) -> List[str]:
        return ["test_cachebench"]
    
    def setup(self, env_config: EnvironmentConfig) -> bool:
        """Install dependencies for CacheBench."""
        self.logger.info("Installing CacheBench dependencies")
        
        if not install_torchaudio():
            self.logger.error("Failed to install torchaudio")
            return False
            
        if not install_torchvision():
            self.logger.error("Failed to install torchvision")
            return False
        
        return True
    
    def run(self, env_config: EnvironmentConfig) -> bool:
        """Run CacheBench tests."""
        self.logger.info("Running CacheBench tests")
        
        # Set up environment
        if not self.setup(env_config):
            return False
        
        # Run the benchmark
        ci_dir = get_ci_dir()
        env = {"PYTHONPATH": "/torchbench"}
        
        return source_and_run(
            str(ci_dir / "test.sh"),
            "test_cachebench",
            cwd=str(ci_dir.parent.parent),
            env=env
        )


class VerifyCacheBenchTestSuite(TestSuite):
    """Verify CacheBench test suite."""
    
    def __init__(self):
        super().__init__("verify_cachebench", "Verify CacheBench tests")
    
    def matches(self, env_config: EnvironmentConfig) -> bool:
        """Check if this is a verify CacheBench test configuration."""
        return env_config.test_config == "verify_cachebench"
    
    def get_test_names(self) -> List[str]:
        return ["test_verify_cachebench"]
    
    def setup(self, env_config: EnvironmentConfig) -> bool:
        """Install dependencies for verify CacheBench."""
        self.logger.info("Installing verify CacheBench dependencies")
        
        if not install_torchaudio():
            self.logger.error("Failed to install torchaudio")
            return False
            
        if not install_torchvision():
            self.logger.error("Failed to install torchvision")
            return False
        
        return True
    
    def run(self, env_config: EnvironmentConfig) -> bool:
        """Run verify CacheBench tests."""
        self.logger.info("Running verify CacheBench tests")
        
        # Set up environment
        if not self.setup(env_config):
            return False
        
        # Run the benchmark
        ci_dir = get_ci_dir()
        env = {"PYTHONPATH": "/torchbench"}
        
        return source_and_run(
            str(ci_dir / "test.sh"),
            "test_verify_cachebench",
            cwd=str(ci_dir.parent.parent),
            env=env
        )
