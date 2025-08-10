"""
Base classes for test configuration and execution.

This module provides the foundation classes for organizing and running tests.
"""

import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from .environment import EnvironmentConfig


class TestSuite(ABC):
    """Abstract base class for test suites."""
    
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self.logger = logging.getLogger(f"{__name__}.{name}")
    
    @abstractmethod
    def matches(self, env_config: EnvironmentConfig) -> bool:
        """Check if this test suite should run for the given environment."""
        pass
    
    @abstractmethod
    def get_test_names(self) -> List[str]:
        """Get list of test names that would be executed."""
        pass
    
    @abstractmethod
    def run(self, env_config: EnvironmentConfig) -> bool:
        """Execute the test suite. Returns True if all tests pass."""
        pass
    
    def setup(self, env_config: EnvironmentConfig) -> bool:
        """Setup phase before running tests. Override if needed."""
        return True
    
    def teardown(self, env_config: EnvironmentConfig) -> bool:
        """Cleanup phase after running tests. Override if needed."""
        return True


class ConditionalTestSuite(TestSuite):
    """Test suite that runs based on environment conditions."""
    
    def __init__(self, name: str, description: str = "", 
                 build_env_patterns: Optional[List[str]] = None,
                 test_config_patterns: Optional[List[str]] = None,
                 exclude_patterns: Optional[List[str]] = None):
        super().__init__(name, description)
        self.build_env_patterns = build_env_patterns or []
        self.test_config_patterns = test_config_patterns or []
        self.exclude_patterns = exclude_patterns or []
        self.test_functions: List[str] = []
    
    def matches(self, env_config: EnvironmentConfig) -> bool:
        """Check if this test suite matches the environment configuration."""
        # Check exclusion patterns first
        for pattern in self.exclude_patterns:
            if pattern in env_config.build_environment or pattern in env_config.test_config:
                return False
        
        # Check build environment patterns
        build_match = not self.build_env_patterns  # Default to True if no patterns
        for pattern in self.build_env_patterns:
            if pattern in env_config.build_environment:
                build_match = True
                break
        
        # Check test config patterns
        test_match = not self.test_config_patterns  # Default to True if no patterns
        for pattern in self.test_config_patterns:
            if pattern in env_config.test_config:
                test_match = True
                break
        
        return build_match and test_match
    
    def add_test_function(self, func_name: str) -> None:
        """Add a test function to this suite."""
        self.test_functions.append(func_name)
    
    def get_test_names(self) -> List[str]:
        """Get list of test function names."""
        return self.test_functions.copy()
    
    def run(self, env_config: EnvironmentConfig) -> bool:
        """Run the test suite by calling shell functions."""
        from ..utils.shell_utils import source_and_run, get_ci_dir
        
        success = True
        ci_dir = get_ci_dir()
        
        for test_func in self.test_functions:
            self.logger.info(f"Running {test_func}")
            
            # For transitional period, call the original shell functions
            result = source_and_run(
                str(ci_dir / "test.sh"),
                test_func,
                cwd=str(ci_dir.parent.parent)
            )
            
            if not result:
                self.logger.error(f"Test function {test_func} failed")
                success = False
        
        return success


class DefaultTestSuite(TestSuite):
    """Default test suite that runs when no other suite matches."""
    
    def __init__(self):
        super().__init__("default", "Default test suite for standard PyTorch testing")
        self.test_functions = [
            "install_torchvision",
            "install_monkeytype", 
            "test_python",
            "test_aten",
            "test_vec256",
            "test_libtorch",
            "test_aot_compilation",
            "test_custom_script_ops",
            "test_custom_backend",
            "test_torch_function_benchmark",
            "test_benchmarks"
        ]
    
    def matches(self, env_config: EnvironmentConfig) -> bool:
        """Default suite matches when no other suite matches."""
        return True  # This is handled by the registry as fallback
    
    def get_test_names(self) -> List[str]:
        """Get list of default test names."""
        return self.test_functions.copy()
    
    def run(self, env_config: EnvironmentConfig) -> bool:
        """Run the default test suite."""
        from ..utils.shell_utils import run_command
        
        success = True
        
        for test_func in self.test_functions:
            self.logger.info(f"Running {test_func}")
            
            # For now, we'll call the original shell functions
            # In a full migration, these would be converted to Python
            result = run_command(f"source .ci/pytorch/test.sh && {test_func}")
            
            if not result:
                self.logger.error(f"Test function {test_func} failed")
                success = False
        
        return success


@dataclass
class TestResult:
    """Result of a test execution."""
    name: str
    success: bool
    duration: float
    output: str = ""
    error: str = ""
