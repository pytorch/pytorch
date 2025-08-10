"""
Mobile test suite implementations.

This module contains test suites for PyTorch mobile testing, including
lite interpreter, mobile optimizer, and other mobile-specific tests.
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
    from utils.install_utils import install_pip_dependencies
except ImportError:
    from ..test_config.base import TestSuite
    from ..test_config.environment import EnvironmentConfig
    from ..utils.shell_utils import run_command, source_and_run, get_ci_dir
    from ..utils.install_utils import install_pip_dependencies


class MobileOptimizerTestSuite(TestSuite):
    """Mobile optimizer test suite."""
    
    def __init__(self):
        super().__init__("mobile_optimizer", "Mobile optimizer tests")
    
    def matches(self, env_config: EnvironmentConfig) -> bool:
        """Check if this is a mobile optimizer test configuration."""
        return env_config.test_config == "mobile_optimizer"
    
    def get_test_names(self) -> List[str]:
        return ["test_mobile_optimizer"]
    
    def run(self, env_config: EnvironmentConfig) -> bool:
        """Run mobile optimizer tests."""
        self.logger.info("Running mobile optimizer tests")
        
        # Run the mobile optimizer tests
        ci_dir = get_ci_dir()
        return source_and_run(
            str(ci_dir / "test.sh"),
            "test_mobile_optimizer",
            cwd=str(ci_dir.parent.parent)
        )


class LiteInterpreterTestSuite(TestSuite):
    """Lite interpreter test suite."""
    
    def __init__(self):
        super().__init__("lite_interpreter", "Lite interpreter tests")
    
    def matches(self, env_config: EnvironmentConfig) -> bool:
        """Check if this is a lite interpreter test configuration."""
        return env_config.test_config == "mobile_lite_interpreter"
    
    def get_test_names(self) -> List[str]:
        return ["test_lite_interpreter"]
    
    def run(self, env_config: EnvironmentConfig) -> bool:
        """Run lite interpreter tests."""
        self.logger.info("Running lite interpreter tests")
        
        # Run the lite interpreter tests
        ci_dir = get_ci_dir()
        return source_and_run(
            str(ci_dir / "test.sh"),
            "test_lite_interpreter",
            cwd=str(ci_dir.parent.parent)
        )


class MobileCodegenTestSuite(TestSuite):
    """Mobile codegen test suite."""
    
    def __init__(self):
        super().__init__("mobile_codegen", "Mobile codegen tests")
    
    def matches(self, env_config: EnvironmentConfig) -> bool:
        """Check if this is a mobile codegen test configuration."""
        return env_config.test_config == "mobile_codegen"
    
    def get_test_names(self) -> List[str]:
        return ["test_mobile_codegen"]
    
    def run(self, env_config: EnvironmentConfig) -> bool:
        """Run mobile codegen tests."""
        self.logger.info("Running mobile codegen tests")
        
        # Run the mobile codegen tests
        ci_dir = get_ci_dir()
        return source_and_run(
            str(ci_dir / "test.sh"),
            "test_mobile_codegen",
            cwd=str(ci_dir.parent.parent)
        )


class AndroidTestSuite(TestSuite):
    """Android test suite."""
    
    def __init__(self):
        super().__init__("android", "Android tests")
    
    def matches(self, env_config: EnvironmentConfig) -> bool:
        """Check if this is an Android test configuration."""
        return env_config.test_config == "android"
    
    def get_test_names(self) -> List[str]:
        return ["test_android"]
    
    def run(self, env_config: EnvironmentConfig) -> bool:
        """Run Android tests."""
        self.logger.info("Running Android tests")
        
        # Run the Android tests
        ci_dir = get_ci_dir()
        return source_and_run(
            str(ci_dir / "test.sh"),
            "test_android",
            cwd=str(ci_dir.parent.parent)
        )


class AndroidNDKTestSuite(TestSuite):
    """Android NDK test suite."""
    
    def __init__(self):
        super().__init__("android_ndk", "Android NDK tests")
    
    def matches(self, env_config: EnvironmentConfig) -> bool:
        """Check if this is an Android NDK test configuration."""
        return env_config.test_config == "android-ndk"
    
    def get_test_names(self) -> List[str]:
        return ["test_android_ndk"]
    
    def run(self, env_config: EnvironmentConfig) -> bool:
        """Run Android NDK tests."""
        self.logger.info("Running Android NDK tests")
        
        # Run the Android NDK tests
        ci_dir = get_ci_dir()
        return source_and_run(
            str(ci_dir / "test.sh"),
            "test_android_ndk",
            cwd=str(ci_dir.parent.parent)
        )


class IOSTestSuite(TestSuite):
    """iOS test suite."""
    
    def __init__(self):
        super().__init__("ios", "iOS tests")
    
    def matches(self, env_config: EnvironmentConfig) -> bool:
        """Check if this is an iOS test configuration."""
        return env_config.test_config == "ios"
    
    def get_test_names(self) -> List[str]:
        return ["test_ios"]
    
    def run(self, env_config: EnvironmentConfig) -> bool:
        """Run iOS tests."""
        self.logger.info("Running iOS tests")
        
        # Run the iOS tests
        ci_dir = get_ci_dir()
        return source_and_run(
            str(ci_dir / "test.sh"),
            "test_ios",
            cwd=str(ci_dir.parent.parent)
        )


class MobileCustomBuildTestSuite(TestSuite):
    """Mobile custom build test suite."""
    
    def __init__(self):
        super().__init__("mobile_custom_build", "Mobile custom build tests")
    
    def matches(self, env_config: EnvironmentConfig) -> bool:
        """Check if this is a mobile custom build test configuration."""
        return env_config.test_config == "mobile_custom_build"
    
    def get_test_names(self) -> List[str]:
        return ["test_mobile_custom_build"]
    
    def run(self, env_config: EnvironmentConfig) -> bool:
        """Run mobile custom build tests."""
        self.logger.info("Running mobile custom build tests")
        
        # Run the mobile custom build tests
        ci_dir = get_ci_dir()
        return source_and_run(
            str(ci_dir / "test.sh"),
            "test_mobile_custom_build",
            cwd=str(ci_dir.parent.parent)
        )
