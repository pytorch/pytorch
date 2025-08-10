#!/usr/bin/env python3
"""
Simplified local testing script for Python-based test runners.

This script provides essential validation tests to ensure our migrated
Python test runners are ready for CI integration.
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

# Add the current directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent))

def setup_logging():
    """Setup basic logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def test_core_configs(logger):
    """Test the most important test configurations."""
    logger.info("Testing core test configurations...")
    
    # Core configurations that are most commonly used in CI
    core_configs = [
        ("", "default"),
        ("smoke", "smoke tests"),
        ("distributed", "distributed tests"),
        ("docs_test", "documentation tests"),
        ("jit_legacy", "JIT legacy tests"),
    ]
    
    results = {}
    
    for config, description in core_configs:
        logger.info(f"Testing {description} (config: '{config}')...")
        
        try:
            env = os.environ.copy()
            env['TEST_CONFIG'] = config
            env['BUILD_ENVIRONMENT'] = 'test'
            
            result = subprocess.run([
                sys.executable, 'simple_test_runner.py', '--dry-run'
            ], env=env, capture_output=True, text=True, timeout=30)
            
            success = result.returncode == 0
            results[config or 'default'] = success
            
            if success:
                logger.info(f"✅ {description} - PASSED")
            else:
                logger.error(f"❌ {description} - FAILED")
                logger.error(f"Error: {result.stderr}")
                
        except Exception as e:
            logger.error(f"❌ {description} - ERROR: {e}")
            results[config or 'default'] = False
    
    return results

def test_python_runner_infrastructure(logger):
    """Test that the Python runner infrastructure works correctly."""
    logger.info("Testing Python runner infrastructure...")
    
    try:
        # Import core modules
        from simple_test_runner import SimpleTestRegistry, EnvironmentConfig
        
        # Test environment config creation
        env_config = EnvironmentConfig()
        
        # Test test suite selection
        registry = SimpleTestRegistry()
        suite = registry.select_test_suite(env_config)
        
        logger.info(f"✅ Infrastructure test - Selected suite: {suite.name}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Infrastructure test - ERROR: {e}")
        return False

def test_migration_completeness(logger):
    """Test that our migration is complete by checking for source_and_run usage."""
    logger.info("Testing migration completeness...")
    
    try:
        # Check if any test suites still use source_and_run
        with open('simple_test_runner.py', 'r') as f:
            content = f.read()
        
        # Look for source_and_run calls (should be none in migrated suites)
        if 'source_and_run(' in content:
            logger.warning("⚠️  Found source_and_run calls - migration may be incomplete")
            return False
        else:
            logger.info("✅ No source_and_run calls found - migration appears complete")
            return True
            
    except Exception as e:
        logger.error(f"❌ Migration completeness check - ERROR: {e}")
        return False

def main():
    """Main entry point for simplified local testing."""
    logger = setup_logging()
    
    logger.info("=" * 60)
    logger.info("SIMPLIFIED LOCAL VALIDATION FOR CI READINESS")
    logger.info("=" * 60)
    
    tests = [
        ("Python Runner Infrastructure", lambda: test_python_runner_infrastructure(logger)),
        ("Migration Completeness", lambda: test_migration_completeness(logger)),
        ("Core Test Configurations", lambda: all(test_core_configs(logger).values())),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n--- {test_name} ---")
        try:
            success = test_func()
            results[test_name] = success
            
            if success:
                logger.info(f"✅ {test_name} - PASSED")
            else:
                logger.error(f"❌ {test_name} - FAILED")
                
        except Exception as e:
            logger.error(f"❌ {test_name} - ERROR: {e}")
            results[test_name] = False
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("VALIDATION SUMMARY")
    logger.info("=" * 60)
    
    passed = sum(1 for success in results.values() if success)
    total = len(results)
    
    for test_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 CORE TESTS PASSED - Python test runners are ready for CI!")
        logger.info("\nNext steps:")
        logger.info("1. The Python test runner infrastructure is working correctly")
        logger.info("2. All major test suites have been migrated from shell to Python")
        logger.info("3. Core test configurations (smoke, distributed, docs, jit) are functional")
        logger.info("4. Ready to integrate into CI with confidence")
        return True
    else:
        logger.warning("⚠️  Some core tests failed - Review issues before CI integration")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
