import os
import pytest
import torch


def is_debug_build():
    """Check if this is a debug build of PyTorch."""
    build_env = os.getenv("BUILD_ENVIRONMENT", "")
    return "-debug" in build_env


def is_bazel_build():
    """Check if this is a bazel build of PyTorch."""
    build_env = os.getenv("BUILD_ENVIRONMENT", "")
    return "-bazel-" in build_env


@pytest.mark.skipif(
    is_bazel_build(),
    reason="Skip bazel jobs because torch isn't available there yet"
)
def test_debug_asserts():
    """
    Test that debug assertions work correctly.
    
    The torch._C._crash_if_debug_asserts_fail() function should only fail if both:
    1. The build is in debug mode
    2. The value 424242 is passed in
    """
    if is_debug_build():
        # In debug mode, expect the assertion to fail with 424242
        with pytest.raises(RuntimeError):
            torch._C._crash_if_debug_asserts_fail(424242)
            
        # But not with other values
        torch._C._crash_if_debug_asserts_fail(0)
        torch._C._crash_if_debug_asserts_fail(123)
    else:
        # In non-debug mode, no assertions should fail
        torch._C._crash_if_debug_asserts_fail(424242)
        torch._C._crash_if_debug_asserts_fail(0)
        torch._C._crash_if_debug_asserts_fail(123)
