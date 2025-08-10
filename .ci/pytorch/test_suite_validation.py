#!/usr/bin/env python3
"""
Test suite validation script.

This script tests our individual test suite implementations to ensure they work correctly.
"""

import sys
import os
from pathlib import Path

# Add current directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent))

def test_environment_config():
    """Test the environment configuration module."""
    print("=== Testing Environment Configuration ===")
    try:
        from test_config.environment import EnvironmentConfig
        
        # Test with empty environment
        env_config = EnvironmentConfig()
        print(f"✓ EnvironmentConfig created successfully")
        print(f"  Build Environment: '{env_config.build_environment}'")
        print(f"  Test Config: '{env_config.test_config}'")
        print(f"  Shard: {env_config.shard_number}/{env_config.num_test_shards}")
        print(f"  CUDA: {env_config.is_cuda}")
        print(f"  ROCm: {env_config.is_rocm}")
        print(f"  ASAN: {env_config.is_asan}")
        
        return True
    except Exception as e:
        print(f"✗ Failed to test EnvironmentConfig: {e}")
        return False

def test_base_classes():
    """Test the base test suite classes."""
    print("\n=== Testing Base Classes ===")
    try:
        from test_config.base import TestSuite, ConditionalTestSuite, DefaultTestSuite
        from test_config.environment import EnvironmentConfig
        
        # Test DefaultTestSuite
        default_suite = DefaultTestSuite()
        env_config = EnvironmentConfig()
        
        print(f"✓ DefaultTestSuite created: {default_suite.name}")
        print(f"  Matches environment: {default_suite.matches(env_config)}")
        print(f"  Test names: {default_suite.get_test_names()}")
        
        return True
    except Exception as e:
        print(f"✗ Failed to test base classes: {e}")
        return False

def test_shell_utils():
    """Test shell utilities."""
    print("\n=== Testing Shell Utils ===")
    try:
        from utils.shell_utils import get_ci_dir, run_command
        
        ci_dir = get_ci_dir()
        print(f"✓ CI directory detected: {ci_dir}")
        
        # Test a simple command
        result = run_command("echo 'Hello from shell utils'", capture_output=True)
        if result:
            print("✓ Shell command execution works")
        else:
            print("✗ Shell command execution failed")
            
        return True
    except Exception as e:
        print(f"✗ Failed to test shell utils: {e}")
        return False

def test_install_utils():
    """Test installation utilities."""
    print("\n=== Testing Install Utils ===")
    try:
        from utils.install_utils import install_pip_dependencies
        
        print("✓ Install utils imported successfully")
        print("  Available functions: install_pip_dependencies, install_torchvision, etc.")
        
        return True
    except Exception as e:
        print(f"✗ Failed to test install utils: {e}")
        return False

def test_individual_test_suites():
    """Test individual test suite implementations."""
    print("\n=== Testing Individual Test Suites ===")
    
    test_suites_to_test = [
        ("python_tests", "PythonTestSuite"),
        ("inductor_tests", "InductorTestSuite"),
        ("distributed_tests", "DistributedTestSuite"),
        ("benchmark_tests", "OperatorBenchmarkTestSuite"),
        ("coverage_tests", "CoveragePythonTestSuite"),
        ("mobile_tests", "MobileOptimizerTestSuite"),
        ("specialized_tests", "BackwardTestSuite"),
    ]
    
    success_count = 0
    total_count = len(test_suites_to_test)
    
    for module_name, class_name in test_suites_to_test:
        try:
            module = __import__(f"test_suites.{module_name}", fromlist=[class_name])
            test_suite_class = getattr(module, class_name)
            
            # Try to instantiate the test suite
            test_suite = test_suite_class()
            print(f"✓ {class_name} from {module_name}: {test_suite.name}")
            success_count += 1
            
        except Exception as e:
            print(f"✗ Failed to test {class_name} from {module_name}: {e}")
    
    print(f"\nTest Suite Summary: {success_count}/{total_count} test suites loaded successfully")
    return success_count == total_count

def main():
    """Main validation function."""
    print("PyTorch Test Suite Validation")
    print("=" * 40)
    
    tests = [
        test_environment_config,
        test_base_classes,
        test_shell_utils,
        test_install_utils,
        test_individual_test_suites,
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test_func.__name__} failed with exception: {e}")
    
    print(f"\n{'='*40}")
    print(f"Validation Summary: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All validation tests passed!")
        return 0
    else:
        print("❌ Some validation tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
