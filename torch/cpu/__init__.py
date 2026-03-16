# mypy: allow-untyped-defs
r"""
This package implements abstractions found in ``torch.cuda``
to facilitate writing device-agnostic code.
"""

from collections.abc import Mapping
from contextlib import AbstractContextManager
from functools import lru_cache
from types import MappingProxyType
from typing import Any

import torch

from .. import device as _device
from . import amp


__all__ = [
    "is_available",
    "is_initialized",
    "synchronize",
    "current_device",
    "current_stream",
    "stream",
    "set_device",
    "device_count",
    "Stream",
    "StreamContext",
    "Event",
    "get_capabilities",
]


@lru_cache(None)
def get_capabilities() -> Mapping[str, Any]:
    """
    Returns an immutable mapping of CPU capabilities detected at runtime.

    This function queries the CPU for supported instruction sets and features
    using cpuinfo. The result is cached after the first call for efficiency.

    The returned mapping contains architecture-specific capabilities:

    For x86/x86_64:
        - SSE family: sse, sse2, sse3, ssse3, sse4_1, sse4_2, sse4a
        - AVX family: avx, avx2, avx_vnni
        - AVX-512 family: avx512_f, avx512_cd, avx512_dq, avx512_bw, avx512_vl,
          avx512_ifma, avx512_vbmi, avx512_vbmi2, avx512_bitalg, avx512_vpopcntdq,
          avx512_vnni, avx512_bf16, avx512_fp16, avx512_vp2intersect,
          avx512_4vnniw, avx512_4fmaps
        - AVX10 family: avx10_1, avx10_2
        - AVX-VNNI-INT: avx_vnni_int8, avx_vnni_int16, avx_ne_convert
        - AMX: amx_bf16, amx_tile, amx_int8, amx_fp16
        - FMA: fma3, fma4
        - Other: f16c, bmi, bmi2, popcnt, lzcnt, aes, sha, clflush, clflushopt, clwb

    For ARM64:
        - SIMD: neon, fp16_arith, bf16, i8mm, dot
        - SVE: sve, sve2, sve_bf16, sve_max_length (when supported)
        - SME: sme, sme2, sme_max_length (when supported)
        - Other: atomics, fhm, rdm, crc32, aes, sha1, sha2, pmull

    Common to all architectures:
        - architecture: string identifying the CPU architecture

    Returns:
        MappingProxyType: An immutable mapping where keys are capability names
        (e.g., 'avx2', 'sve') and values are booleans indicating
        support, or integers for properties like vector lengths.

    Example:
        >>> caps = torch.cpu.get_capabilities()
        >>> if caps.get("avx2", False):
        ...     print("AVX2 is supported")
        >>> print(f"Architecture: {caps['architecture']}")
    """
    return MappingProxyType(torch._C._cpu._get_cpu_capability())


def _is_avx2_supported() -> bool:
    r"""Returns a bool indicating if CPU supports AVX2."""
    return get_capabilities().get("avx2", False)


def _is_avx512_supported() -> bool:
    r"""Returns a bool indicating if CPU supports AVX512."""
    return get_capabilities().get("avx512_f", False)


def _is_avx512_bf16_supported() -> bool:
    r"""Returns a bool indicating if CPU supports AVX512_BF16."""
    return get_capabilities().get("avx512_bf16", False)


def _is_vnni_supported() -> bool:
    r"""Returns a bool indicating if CPU supports VNNI."""
    # Note: Currently, it only checks avx512_vnni, will add the support of avx2_vnni later.
    return get_capabilities().get("avx512_vnni", False)


def _is_amx_tile_supported() -> bool:
    r"""Returns a bool indicating if CPU supports AMX_TILE."""
    return get_capabilities().get("amx_tile", False)


def _is_amx_fp16_supported() -> bool:
    r"""Returns a bool indicating if CPU supports AMX FP16."""
    return get_capabilities().get("amx_fp16", False)


def _init_amx() -> bool:
    r"""Initializes AMX instructions."""
    return torch._C._cpu._init_amx()


def is_available() -> bool:
    r"""Returns a bool indicating if CPU is currently available.

    N.B. This function only exists to facilitate device-agnostic code

    """
    return True


def synchronize(device: torch.types.Device = None) -> None:
    r"""Waits for all kernels in all streams on the CPU device to complete.

    Args:
        device (torch.device or int, optional): ignored, there's only one CPU device.

    N.B. This function only exists to facilitate device-agnostic code.
    """


class Stream:
    """
    N.B. This class only exists to facilitate device-agnostic code
    """

    def __init__(self, priority: int = -1) -> None:
        pass

    def wait_stream(self, stream) -> None:
        pass

    def record_event(self) -> None:
        pass

    def wait_event(self, event) -> None:
        pass


class Event:
    def query(self) -> bool:
        return True

    def record(self, stream=None) -> None:
        pass

    def synchronize(self) -> None:
        pass

    def wait(self, stream=None) -> None:
        pass


_default_cpu_stream = Stream()
_current_stream = _default_cpu_stream


def current_stream(device: torch.types.Device = None) -> Stream:
    r"""Returns the currently selected :class:`Stream` for a given device.

    Args:
        device (torch.device or int, optional): Ignored.

    N.B. This function only exists to facilitate device-agnostic code

    """
    return _current_stream


class StreamContext(AbstractContextManager):
    r"""Context-manager that selects a given stream.

    N.B. This class only exists to facilitate device-agnostic code

    """

    cur_stream: Stream | None

    def __init__(self, stream):
        self.stream = stream
        self.prev_stream = _default_cpu_stream

    def __enter__(self):
        cur_stream = self.stream
        if cur_stream is None:
            return

        global _current_stream
        self.prev_stream = _current_stream
        _current_stream = cur_stream

    def __exit__(self, type: Any, value: Any, traceback: Any) -> None:
        cur_stream = self.stream
        if cur_stream is None:
            return

        global _current_stream
        _current_stream = self.prev_stream


def stream(stream: Stream) -> AbstractContextManager:
    r"""Wrapper around the Context-manager StreamContext that
    selects a given stream.

    N.B. This function only exists to facilitate device-agnostic code
    """
    return StreamContext(stream)


def device_count() -> int:
    r"""Returns number of CPU devices (not cores). Always 1.

    N.B. This function only exists to facilitate device-agnostic code
    """
    return 1


def set_device(device: torch.types.Device) -> None:
    r"""Sets the current device, in CPU we do nothing.

    N.B. This function only exists to facilitate device-agnostic code
    """


def current_device() -> str:
    r"""Returns current device for cpu. Always 'cpu'.

    N.B. This function only exists to facilitate device-agnostic code
    """
    return "cpu"


def is_initialized() -> bool:
    r"""Returns True if the CPU is initialized. Always True.

    N.B. This function only exists to facilitate device-agnostic code
    """
    return True
