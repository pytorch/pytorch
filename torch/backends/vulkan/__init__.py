# mypy: allow-untyped-defs
import torch


def is_available():
    r"""Return whether PyTorch is built with Vulkan support."""
    return torch._C._is_vulkan_available()


__all__ = ["is_available"]