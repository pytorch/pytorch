"""
Unified device-scoped heuristic module for PyTorch Inductor.

This module provides a single entry point for all device-specific heuristic
dispatch, covering both:

- triton_template: pre-written triton template kernels (mm, conv, flex_attention...)
  used at compile time.
- triton_codegen: inductor-generated triton kernels (pointwise, reduction...)
  used at runtime.
"""

from .triton_template import get_config_heuristic_for_device


__all__ = ["get_config_heuristic_for_device"]
