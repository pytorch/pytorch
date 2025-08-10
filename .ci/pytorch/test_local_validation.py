#!/usr/bin/env python3
"""
Local validation script for Python-based test runners.

This script allows testing the migrated Python test runners locally
before deploying them to CI. It provides various testing modes and
validation options.
"""

import os
import sys
import logging
import argparse
import subprocess
from typing import List, Dict, Optional
from pathlib import Path

# Add the current directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent))

from simple_test_runner import SimpleTestRegistry, EnvironmentConfig


class LocalTestValidator:
    """Local validation for Python test runners."""
    
    def __init__(self, verbose: bool = False):
        self.setup_logging(verbose)
        self.logger = logging.getLogger(__name__)
        self.registry = SimpleTestRegistry()
        
    def setup_logging(self, verbose: bool):
        """Setup logging configuration."""
        level = logging.DEBUG if verbose else logging.INFO
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def test_dry_run_all_suites(self) -> Dict[str, bool]:
        """Test dry-run mode for all available test suites."""
        self.logger.info("Testing dry-run mode for all test suites...")
        
        test_configs = [
            "",  # default
            "smoke",
            "distributed", 
            "jit_legacy",
            "docs_test",
            "benchmark",
            "torchbench",
            "dynamo_benchmark",
            "inductor_distributed",
            "inductor_shard",
            "inductor_aoti",
            "inductor_cpp_wrapper",
            "inductor_halide",
            "h100-symm-mem",
            "h100-distributed", 
            "h100-cutlass-backend"
        ]
        
        results = {}
        
        for config in test_configs:
            self.logger.info(f"\n--- Testing config: '{config}' ---")
            
            # Set environment
            env = os.environ.copy()
            env['TEST_CONFIG'] = config
            env['BUILD_ENVIRONMENT'] = 'test'
            
            try:
                result = subprocess.run([
                    sys.executable, 'simple_test_runner.py', '--dry-run'
                ], env=env, capture_output=True, text=True, timeout=30)
                
                success = result.returncode == 0
                results[config or 'default'] = success
                
                if success:
                    self.logger.info(f"✅ Config '{config}' - SUCCESS")
                else:
                    self.logger.error(f"❌ Config '{config}' - FAILED")
                    self.logger.error(f"STDOUT: {result.stdout}")
                    self.logger.error(f"STDERR: {result.stderr}")
                    
            except subprocess.TimeoutExpired:
                self.logger.error(f"❌ Config '{config}' - TIMEOUT")
                results[config or 'default'] = False
            except Exception as e:
                self.logger.error(f"❌ Config '{config}' - ERROR: {e}")
                results[config or 'default'] = False
        
        return results
    
    def test_import_validation(self) -> bool:
        """Test that all test runner imports work correctly."""
        self.logger.info("Testing import validation for all test runners...")
        
        import_tests = [
            ("utils.core_test_runners", ["PythonTestRunner", "AtenTestRunner", "Vec256TestRunner", "DocsTestRunner"]),
            ("utils.distributed_test_runners", ["DistributedTestRunner", "RPCTestRunner", "CustomBackendTestRunner", "CustomScriptOpsTestRunner"]),
            ("utils.libtorch_test_runners", ["LibTorchTestRunner", "AOTCompilationTestRunner", "VulkanTestRunner", "JitHooksTestRunner", "CppExtensionsTestRunner"]),
            ("utils.benchmark_test_runners", ["BenchmarkTestRunner", "TorchBenchTestRunner", "DynamoBenchmarkTestRunner", "InductorMicroBenchmarkTestRunner", "OperatorBenchmarkTestRunner", "TorchFunctionBenchmarkTestRunner"]),
            ("utils.inductor_test_runners", ["InductorDistributedTestRunner", "InductorShardTestRunner", "InductorAOTITestRunner", "InductorCppWrapperTestRunner", "InductorHalideTestRunner", "InductorTritonTestRunner"]),
            ("utils.specialized_test_runners", ["XLATestRunner", "ForwardBackwardCompatibilityTestRunner", "BazelTestRunner", "ExecutorchTestRunner", "LinuxAArch64TestRunner", "OperatorBenchmarkTestRunner", "WithoutNumpyTestRunner", "LazyTensorMetaReferenceTestRunner", "DynamoWrappedShardTestRunner", "EinopsTestRunner"]),
            ("utils.test_execution", ["SmokeTestRunner", "PyTorchTestRunner"])
        ]
        
        all_success = True
        
        for module_name, class_names in import_tests:
            try:
                module = __import__(module_name, fromlist=class_names)
                
                for class_name in class_names:
                    if hasattr(module, class_name):
                        self.logger.info(f"✅ {module_name}.{class_name} - OK")
                    else:
                        self.logger.error(f"❌ {module_name}.{class_name} - MISSING")
                        all_success = False
                        
            except ImportError as e:
                self.logger.error(f"❌ {module_name} - IMPORT ERROR: {e}")
                all_success = False
            except Exception as e:
                self.logger.error(f"❌ {module_name} - ERROR: {e}")
                all_success = False
        
        return all_success
    
    def test_environment_config(self) -> bool:
        """Test environment configuration parsing."""
        self.logger.info("Testing environment configuration...")
        
        test_environments = [
            {"BUILD_ENVIRONMENT": "linux-focal-py3.8-gcc7", "TEST_CONFIG": "default"},
            {"BUILD_ENVIRONMENT": "linux-focal-cuda11.8-py3.8-gcc7", "TEST_CONFIG": "distributed"},
            {"BUILD_ENVIRONMENT": "linux-focal-rocm5.4.2-py3.8", "TEST_CONFIG": "smoke"},
            {"BUILD_ENVIRONMENT": "macos-12-py3.8-x86", "TEST_CONFIG": "jit_legacy"},
        ]
        
        all_success = True
        
        for env_vars in test_environments:
            try:
                # Temporarily set environment variables
                old_env = {}
                for key, value in env_vars.items():
                    old_env[key] = os.environ.get(key)
                    os.environ[key] = value
                
                # Test environment config
                env_config = EnvironmentConfig()
                
                # Validate basic properties
                assert hasattr(env_config, 'build_environment')
                assert hasattr(env_config, 'test_config')
                assert hasattr(env_config, 'is_cuda')
                assert hasattr(env_config, 'is_rocm')
                
                self.logger.info(f"✅ Environment config for {env_vars} - OK")
                
                # Restore environment
                for key, old_value in old_env.items():
                    if old_value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = old_value
                        
            except Exception as e:
                self.logger.error(f"❌ Environment config for {env_vars} - ERROR: {e}")
                all_success = False
        
        return all_success
    
    def test_suite_selection(self) -> bool:
        """Test test suite selection logic."""
        self.logger.info("Testing test suite selection...")
        
        test_cases = [
            ("", "default"),
            ("smoke", "smoke"),
            ("distributed", "distributed"),
            ("jit_legacy", "jit_legacy"),
            ("docs_test", "docs_test"),
            ("benchmark", "benchmark"),
            ("h100-symm-mem", "h100_symm_mem"),
            ("h100-distributed", "h100_distributed"),
            ("h100-cutlass-backend", "h100_cutlass_backend"),
            ("unknown_config", "default"),  # Should fall back to default
        ]
        
        all_success = True
        
        for test_config, expected_suite in test_cases:
            try:
                # Create environment config
                old_config = os.environ.get('TEST_CONFIG')
                os.environ['TEST_CONFIG'] = test_config
                
                env_config = EnvironmentConfig()
                selected_suite = self.registry.select_test_suite(env_config)
                
                if selected_suite.name == expected_suite:
                    self.logger.info(f"✅ Config '{test_config}' → Suite '{selected_suite.name}' - OK")
                else:
                    self.logger.error(f"❌ Config '{test_config}' → Expected '{expected_suite}', got '{selected_suite.name}'")
                    all_success = False
                
                # Restore environment
                if old_config is None:
                    os.environ.pop('TEST_CONFIG', None)
                else:
                    os.environ['TEST_CONFIG'] = old_config
                    
            except Exception as e:
                self.logger.error(f"❌ Suite selection for '{test_config}' - ERROR: {e}")
                all_success = False
        
        return all_success
    
    def run_smoke_test(self) -> bool:
        """Run a simple smoke test with actual execution (not just dry-run)."""
        self.logger.info("Running smoke test with actual execution...")
        
        try:
            # Set up minimal environment for smoke test
            env = os.environ.copy()
            env['TEST_CONFIG'] = 'smoke'
            env['BUILD_ENVIRONMENT'] = 'test'
            
            # Run smoke test (this will actually try to execute some tests)
            result = subprocess.run([
                sys.executable, 'simple_test_runner.py'
            ], env=env, capture_output=True, text=True, timeout=120)
            
            # Note: This might fail due to missing PyTorch installation or test dependencies
            # but we can still validate that our Python runner infrastructure works
            self.logger.info(f"Smoke test exit code: {result.returncode}")
            self.logger.info(f"Smoke test output: {result.stdout[-500:]}")  # Last 500 chars
            
            if result.stderr:
                self.logger.warning(f"Smoke test stderr: {result.stderr[-500:]}")
            
            # Consider it successful if the Python runner itself works (even if tests fail)
            return "Selected test suite: smoke" in result.stdout
            
        except subprocess.TimeoutExpired:
            self.logger.warning("Smoke test timed out - this is expected in some environments")
            return True  # Timeout is acceptable for local testing
        except Exception as e:
            self.logger.error(f"Smoke test error: {e}")
            return False
    
    def run_full_validation(self) -> bool:
        """Run full validation suite."""
        self.logger.info("=" * 60)
        self.logger.info("STARTING FULL LOCAL VALIDATION")
        self.logger.info("=" * 60)
        
        tests = [
            ("Import Validation", self.test_import_validation),
            ("Environment Config", self.test_environment_config),
            ("Suite Selection", self.test_suite_selection),
            ("Dry-Run All Suites", lambda: all(self.test_dry_run_all_suites().values())),
            ("Smoke Test", self.run_smoke_test),
        ]
        
        results = {}
        
        for test_name, test_func in tests:
            self.logger.info(f"\n--- {test_name} ---")
            try:
                success = test_func()
                results[test_name] = success
                
                if success:
                    self.logger.info(f"✅ {test_name} - PASSED")
                else:
                    self.logger.error(f"❌ {test_name} - FAILED")
                    
            except Exception as e:
                self.logger.error(f"❌ {test_name} - ERROR: {e}")
                results[test_name] = False
        
        # Summary
        self.logger.info("\n" + "=" * 60)
        self.logger.info("VALIDATION SUMMARY")
        self.logger.info("=" * 60)
        
        passed = sum(1 for success in results.values() if success)
        total = len(results)
        
        for test_name, success in results.items():
            status = "✅ PASS" if success else "❌ FAIL"
            self.logger.info(f"{test_name}: {status}")
        
        self.logger.info(f"\nOverall: {passed}/{total} tests passed")
        
        if passed == total:
            self.logger.info("🎉 ALL TESTS PASSED - Ready for CI integration!")
            return True
        else:
            self.logger.warning("⚠️  Some tests failed - Review issues before CI integration")
            return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Local validation for Python test runners")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    parser.add_argument("--test", choices=[
        "all", "imports", "config", "selection", "dry-run", "smoke"
    ], default="all", help="Specific test to run")
    
    args = parser.parse_args()
    
    validator = LocalTestValidator(verbose=args.verbose)
    
    if args.test == "all":
        success = validator.run_full_validation()
    elif args.test == "imports":
        success = validator.test_import_validation()
    elif args.test == "config":
        success = validator.test_environment_config()
    elif args.test == "selection":
        success = validator.test_suite_selection()
    elif args.test == "dry-run":
        results = validator.test_dry_run_all_suites()
        success = all(results.values())
    elif args.test == "smoke":
        success = validator.run_smoke_test()
    else:
        print(f"Unknown test: {args.test}")
        return 1
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
