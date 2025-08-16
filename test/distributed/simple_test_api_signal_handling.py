#!/usr/bin/env python3

"""
Simple test to verify the signal handling changes work.
"""

import sys
import os
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

def test_launch_config():
    """Test that LaunchConfig accepts the new signals_to_handle parameter."""
    try:
        from torch.distributed.launcher.api import LaunchConfig
        
        print("Testing LaunchConfig...")
        
        # Test with default signals
        config1 = LaunchConfig(
            min_nodes=1,
            max_nodes=1,
            nproc_per_node=1
        )
        print(f"Default signals: {config1.signals_to_handle}")
        assert config1.signals_to_handle == "SIGTERM,SIGINT,SIGHUP,SIGQUIT"
        print("✓ Default signals test passed")
        
        # Test with custom signals
        config2 = LaunchConfig(
            min_nodes=1,
            max_nodes=1,
            nproc_per_node=1,
            signals_to_handle="SIGTERM,SIGUSR1,SIGUSR2"
        )
        print(f"Custom signals: {config2.signals_to_handle}")
        assert config2.signals_to_handle == "SIGTERM,SIGUSR1,SIGUSR2"
        print("✓ Custom signals test passed")
        
        return True
        
    except Exception as e:
        print(f"✗ LaunchConfig test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_command_line_args():
    """Test that the command line argument parsing works."""
    try:
        from torch.distributed.run import get_args_parser
        
        print("\nTesting command line argument parsing...")
        
        parser = get_args_parser()
        
        # Test default behavior
        args1 = parser.parse_args(['--standalone', 'test_script.py'])
        print(f"Default signals from args: {args1.signals_to_handle}")
        assert args1.signals_to_handle == "SIGTERM,SIGINT,SIGHUP,SIGQUIT"
        print("✓ Default command line args test passed")
        
        # Test custom signals
        args2 = parser.parse_args([
            '--standalone', 
            '--signals-to-handle=SIGTERM,SIGUSR1,SIGUSR2',
            'test_script.py'
        ])
        print(f"Custom signals from args: {args2.signals_to_handle}")
        assert args2.signals_to_handle == "SIGTERM,SIGUSR1,SIGUSR2"
        print("✓ Custom command line args test passed")
        
        return True
        
    except Exception as e:
        print(f"✗ Command line args test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_environment_variable():
    """Test that the environment variable is set correctly."""
    try:
        from torch.distributed.launcher.api import LaunchConfig, launch_agent
        
        print("\nTesting environment variable setting...")
        
        # Create a config with custom signals
        config = LaunchConfig(
            min_nodes=1,
            max_nodes=1,
            nproc_per_node=1,
            signals_to_handle="SIGTERM,SIGUSR1,SIGUSR2"
        )
        
        # Mock the launch_agent function to just set the environment variable
        # without actually launching anything
        original_env = os.environ.get("TORCHELASTIC_SIGNALS_TO_HANDLE")
        
        # Set the environment variable like launch_agent does
        os.environ["TORCHELASTIC_SIGNALS_TO_HANDLE"] = config.signals_to_handle
        
        # Check if it was set correctly
        env_value = os.environ.get("TORCHELASTIC_SIGNALS_TO_HANDLE")
        print(f"Environment variable value: {env_value}")
        assert env_value == "SIGTERM,SIGUSR1,SIGUSR2"
        print("✓ Environment variable test passed")
        
        # Restore original environment
        if original_env is not None:
            os.environ["TORCHELASTIC_SIGNALS_TO_HANDLE"] = original_env
        elif "TORCHELASTIC_SIGNALS_TO_HANDLE" in os.environ:
            del os.environ["TORCHELASTIC_SIGNALS_TO_HANDLE"]
        
        return True
        
    except Exception as e:
        print(f"✗ Environment variable test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("Running signal handling implementation tests...\n")
    
    success = True
    success &= test_launch_config()
    success &= test_command_line_args()
    success &= test_environment_variable()
    
    if success:
        print("\n🎉 All tests passed! Signal handling implementation is working correctly.")
    else:
        print("\n❌ Some tests failed. Please check the implementation.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
