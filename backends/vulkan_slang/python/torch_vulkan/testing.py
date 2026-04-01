"""Test utilities for torch_vulkan."""

import functools
import pytest
import torch


def skip_if_no_vulkan(fn):
    """Decorator to skip test if no Vulkan device is available."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            import torch_vulkan
            if not torch_vulkan.is_available():
                pytest.skip("No Vulkan device available")
        except ImportError:
            pytest.skip("torch_vulkan not installed")
        return fn(*args, **kwargs)
    return wrapper


@pytest.fixture
def vulkan_device():
    """Pytest fixture that provides a Vulkan device, skipping if unavailable."""
    try:
        import torch_vulkan
        if not torch_vulkan.is_available():
            pytest.skip("No Vulkan device available")
    except ImportError:
        pytest.skip("torch_vulkan not installed")
    return torch.device("vulkan:0")


# Tolerance presets for SwiftShader (slightly less precise than GPU)
SWIFTSHADER_TOLERANCES = {
    "rtol": 1e-4,
    "atol": 1e-4,
}

GPU_TOLERANCES = {
    "rtol": 1e-5,
    "atol": 1e-5,
}
