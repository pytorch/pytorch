import pytest
import torch


def pytest_configure(config):
    config.addinivalue_line("markers", "vulkan: test requires Vulkan device")


@pytest.fixture
def vulkan_device():
    """Provides a Vulkan device, skipping if unavailable."""
    try:
        import torch_vulkan
        if not torch_vulkan.is_available():
            pytest.skip("No Vulkan device (install SwiftShader for CPU testing)")
    except ImportError:
        pytest.skip("torch_vulkan not installed")
    return torch.device("vulkan:0")
