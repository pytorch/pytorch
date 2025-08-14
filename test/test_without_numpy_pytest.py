"""
Tests to verify PyTorch's ability to function without NumPy.
These tests ensure that basic PyTorch functionality works even when NumPy is not available.
"""
import sys
import os
import subprocess
import pytest
from unittest import TestCase

# Add fake_numpy to sys.path to simulate NumPy not being available
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
FAKE_NUMPY_DIR = os.path.join(REPO_ROOT, "fake_numpy")


class TestWithoutNumpy(TestCase):
    """Test PyTorch functionality when NumPy is not available."""

    def test_numpy_import_fails(self):
        """Test that importing numpy raises ImportError."""
        # Create a Python script that tries to import numpy
        test_script = '''
import sys
import os

# Remove numpy from sys.modules if it's there
if 'numpy' in sys.modules:
    del sys.modules['numpy']

# Add fake_numpy to the path
fake_numpy_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fake_numpy")
sys.path.insert(0, fake_numpy_dir)

# Try to import numpy
import numpy
'''
        
        # Write the script to a temporary file
        script_path = os.path.join(REPO_ROOT, "test_numpy_import.py")
        with open(script_path, "w") as f:
            f.write(test_script)
        
        try:
            # Run the script in a subprocess with -S flag to disable site packages
            result = subprocess.run(
                [sys.executable, "-S", script_path],
                capture_output=True,
                text=True,
                env={"PYTHONPATH": FAKE_NUMPY_DIR}
            )
            
            print("Script output:")
            print("stdout:", result.stdout)
            print("stderr:", result.stderr)
            print("returncode:", result.returncode)
            
            # Check that the script failed with ImportError
            assert result.returncode != 0
            assert any(msg in result.stderr for msg in [
                "ImportError: numpy is not available",
                "ModuleNotFoundError: No module named 'numpy'"
            ])
            
        finally:
            # Clean up
            if os.path.exists(script_path):
                os.remove(script_path)