"""
Reproduction script for issue #163033:
ONNX Legacy Exporter segfault with invalid dynamic_axes keys
"""

import torch
import tempfile
import os
import subprocess
import sys


class SimpleModel(torch.nn.Module):
    """Simple model with multiple inputs for testing dynamic_axes validation."""

    def forward(self, x, y):
        return x + y


def run_export_in_subprocess():
    """
    Run the ONNX export in a subprocess to catch segfaults.
    Returns (exit_code, stdout, stderr)
    """
    code = """
import torch
import tempfile
import os

class SimpleModel(torch.nn.Module):
    def forward(self, x, y):
        return x + y

model = SimpleModel()
x = torch.randn(2, 3)
y = torch.randn(2, 3)

with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
    temp_file = f.name

try:
    # Force legacy exporter with dynamo=False
    # Actual input names are "x" and "y", output name is "z"
    # But dynamic_axes uses "input" and "output" - deliberate mismatch
    torch.onnx.export(
        model,
        (x, y),
        temp_file,
        opset_version=17,
        input_names=["x", "y"],
        output_names=["z"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        dynamo=False
    )
    print("Export succeeded")
except Exception as e:
    print(f"Caught exception: {type(e).__name__}: {e}")
    import sys
    sys.exit(1)
finally:
    if os.path.exists(temp_file):
        os.remove(temp_file)
"""

    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=30
    )
    return result.returncode, result.stdout, result.stderr


def test_invalid_dynamic_axes_segfault():
    """
    Test that invalid dynamic_axes keys raise a proper Python exception
    instead of causing a segfault.
    """
    exit_code, stdout, stderr = run_export_in_subprocess()
    combined = (stdout + stderr).lower()

    print(f"Exit code: {exit_code}")
    print(f"Stdout:\n{stdout}")
    print(f"Stderr:\n{stderr}")

    # After the fix:
    # - export should fail with a clean Python exception (non-zero exit)
    # - stderr should NOT contain "segmentation fault"
    # - the message should mention invalid dynamic_axes keys
    assert exit_code != 0, "Expected export to fail for invalid dynamic_axes keys"
    assert "segmentation fault" not in combined, "Should not segfault - must raise Python exception"
    assert "dynamic_axes" in combined or "invalid" in combined, "Error message should mention dynamic_axes or invalid keys"

