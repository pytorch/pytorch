"""Basic operation tests for the Vulkan backend."""

import torch
import pytest


@pytest.fixture(autouse=True)
def setup():
    try:
        import torch_vulkan
        if not torch_vulkan.is_available():
            pytest.skip("No Vulkan device")
    except ImportError:
        pytest.skip("torch_vulkan not installed")


class TestDeviceBasics:
    def test_device_available(self):
        import torch_vulkan
        assert torch_vulkan.is_available()

    def test_device_count(self):
        import torch_vulkan
        assert torch_vulkan.device_count() >= 1

    def test_device_name(self):
        import torch_vulkan
        name = torch_vulkan.get_device_name(0)
        assert isinstance(name, str)
        assert len(name) > 0

    def test_device_creation(self):
        device = torch.device("vulkan:0")
        assert device.type == "vulkan"
        assert device.index == 0


class TestTensorCreation:
    def test_empty(self):
        t = torch.empty(4, 4, device="vulkan:0")
        assert t.device.type == "vulkan"
        assert t.shape == (4, 4)

    def test_zeros_roundtrip(self):
        t = torch.zeros(4, 4, device="vulkan:0")
        cpu_t = t.cpu()
        assert cpu_t.device.type == "cpu"
        torch.testing.assert_close(cpu_t, torch.zeros(4, 4))

    def test_ones_roundtrip(self):
        t = torch.ones(3, 5, device="vulkan:0")
        cpu_t = t.cpu()
        torch.testing.assert_close(cpu_t, torch.ones(3, 5))

    def test_cpu_to_vulkan_roundtrip(self):
        original = torch.randn(8, 8)
        vulkan_t = original.to("vulkan:0")
        back = vulkan_t.cpu()
        torch.testing.assert_close(back, original)

    def test_fill(self):
        t = torch.empty(4, 4, device="vulkan:0")
        t.fill_(3.14)
        cpu_t = t.cpu()
        torch.testing.assert_close(cpu_t, torch.full((4, 4), 3.14))

    def test_various_shapes(self):
        for shape in [(1,), (100,), (3, 4), (2, 3, 4), (1, 1, 1, 1)]:
            t = torch.zeros(*shape, device="vulkan:0")
            cpu_t = t.cpu()
            assert cpu_t.shape == shape
            torch.testing.assert_close(cpu_t, torch.zeros(*shape))
