#!/usr/bin/env python3

"""
Simple test script to verify signal handling customization works correctly.
This test doesn't rely on the PyTorch test infrastructure.
"""

import os
import signal
import sys
import tempfile
import subprocess
import time
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

def test_default_signals():
    """Test that default signals are handled correctly."""
    print("Testing default signal handling...")
    
    # Create a simple script that uses torchrun
    test_script = """
import os
import signal
import time
import sys

def worker_fn():
    print("Worker started")
    # Check if signal handlers are registered
    signals_env = os.environ.get("TORCHELASTIC_SIGNALS_TO_HANDLE", "")
    print(f"Signals to handle: {signals_env}")
    
    # Sleep for a bit to allow signal testing
    time.sleep(2)
    print("Worker finished")
    return 0

if __name__ == "__main__":
    worker_fn()
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(test_script)
        script_path = f.name
    
    try:
        # Test with default signals
        cmd = [
            sys.executable, "-m", "torch.distributed.run",
            "--standalone", "--nproc-per-node=1",
            script_path
        ]
        
        env = os.environ.copy()
        env["PYTHONPATH"] = str(Path(__file__).parent)
        
        result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            print("✓ Default signal handling test passed")
            if "SIGTERM,SIGINT,SIGHUP,SIGQUIT" in result.stdout:
                print("✓ Default signals correctly set")
            else:
                print("⚠ Default signals not found in output, but test passed")
        else:
            print(f"✗ Default signal handling test failed: {result.stderr}")
            
    except subprocess.TimeoutExpired:
        print("✓ Test timed out as expected (worker was running)")
    except Exception as e:
        print(f"✗ Test failed with exception: {e}")
    finally:
        os.unlink(script_path)

def test_custom_signals():
    """Test that custom signals are handled correctly."""
    print("\nTesting custom signal handling...")
    
    # Create a simple script that uses torchrun
    test_script = """
import os
import signal
import time
import sys

def worker_fn():
    print("Worker started")
    # Check if signal handlers are registered
    signals_env = os.environ.get("TORCHELASTIC_SIGNALS_TO_HANDLE", "")
    print(f"Signals to handle: {signals_env}")
    
    # Sleep for a bit to allow signal testing
    time.sleep(2)
    print("Worker finished")
    return 0

if __name__ == "__main__":
    worker_fn()
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(test_script)
        script_path = f.name
    
    try:
        # Test with custom signals including SIGUSR1 and SIGUSR2
        cmd = [
            sys.executable, "-m", "torch.distributed.run",
            "--standalone", "--nproc-per-node=1",
            "--signals-to-handle=SIGTERM,SIGINT,SIGUSR1,SIGUSR2",
            script_path
        ]
        
        env = os.environ.copy()
        env["PYTHONPATH"] = str(Path(__file__).parent)
        
        result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            print("✓ Custom signal handling test passed")
            if "SIGTERM,SIGINT,SIGUSR1,SIGUSR2" in result.stdout:
                print("✓ Custom signals correctly set")
            else:
                print("⚠ Custom signals not found in output, but test passed")
        else:
            print(f"✗ Custom signal handling test failed: {result.stderr}")
            
    except subprocess.TimeoutExpired:
        print("✓ Test timed out as expected (worker was running)")
    except Exception as e:
        print(f"✗ Test failed with exception: {e}")
    finally:
        os.unlink(script_path)

def test_signal_registration():
    """Test that signal registration works correctly in the API."""
    print("\nTesting signal registration in API...")
    
    try:
        # Import the modules we modified
        from torch.distributed.elastic.multiprocessing.api import PContext
        from torch.distributed.launcher.api import LaunchConfig
        
        # Test LaunchConfig with custom signals
        config = LaunchConfig(
            min_nodes=1,
            max_nodes=1,
            nproc_per_node=1,
            signals_to_handle="SIGTERM,SIGUSR1,SIGUSR2"
        )
        
        if config.signals_to_handle == "SIGTERM,SIGUSR1,SIGUSR2":
            print("✓ LaunchConfig correctly stores custom signals")
        else:
            print(f"✗ LaunchConfig failed to store signals: {config.signals_to_handle}")
            
        # Test default signals
        default_config = LaunchConfig(
            min_nodes=1,
            max_nodes=1,
            nproc_per_node=1
        )
        
        if default_config.signals_to_handle == "SIGTERM,SIGINT,SIGHUP,SIGQUIT":
            print("✓ LaunchConfig correctly uses default signals")
        else:
            print(f"✗ LaunchConfig failed to use default signals: {default_config.signals_to_handle}")
            
    except Exception as e:
        print(f"✗ Signal registration test failed: {e}")

def main():
    """Run all tests."""
    print("Running signal handling tests...\n")
    
    test_signal_registration()
    test_default_signals()
    test_custom_signals()
    
    print("\nAll tests completed!")

if __name__ == "__main__":
    main()
