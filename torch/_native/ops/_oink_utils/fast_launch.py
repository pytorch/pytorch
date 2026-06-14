# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Host-side fast-launch helpers for CuTeDSL pointer entrypoints.

CuTeDSL's Python runtime typically marshals each kernel call by allocating
`Int32` / `Float32` wrappers and runtime `Pointer` descriptors per invocation.
For latency-sensitive cases (small/medium M), this overhead can dominate.

These helpers provide:
- Stable scalar argument wrappers (`StableI32Arg`, `StableF32Arg`) that avoid
  per-call ctypes allocations.
- In-place mutation of runtime pointer descriptors (`set_runtime_ptr`) so a
  compiled kernel can be launched repeatedly with different raw device pointers
  without rebuilding argument objects.
- A small thread-local cache to store packed args objects (when supported by the
  installed CuTeDSL version).

All of this relies on a few private-ish CuTeDSL internals. Callers must treat
fast-launch as an optional optimization and fall back to the normal launch
path if those internals are unavailable.
"""

from __future__ import annotations

import ctypes
import os
import threading
from typing import Any

_FAST_LAUNCH_TLS = threading.local()


def _env_flag(name: str, default: bool) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() not in {"0", "false", "no", "off", ""}


# Fast-launch uses internal CuTeDSL plumbing (packed args + pointer descriptors).
# Keep it enabled by default in our pinned environment, but allow disabling it
# via env var and auto-disable it if CuTeDSL internals change.
_ENABLE_FAST_LAUNCH = _env_flag("OINK_CUTEDSL_FAST_LAUNCH", default=True)
_FAST_LAUNCH_SUPPORTED = True


def fast_launch_enabled() -> bool:
    return _ENABLE_FAST_LAUNCH and _FAST_LAUNCH_SUPPORTED


def disable_fast_launch() -> None:
    """Globally disable fast-launch for all consumers (rmsnorm, layernorm, etc.).

    This is intentionally global: if CuTeDSL internals are missing or changed,
    the failure applies to every kernel module, not just the one that hit it
    first.  Disabling once avoids repeated slow AttributeError fallbacks.
    """
    global _FAST_LAUNCH_SUPPORTED
    _FAST_LAUNCH_SUPPORTED = False


def tls_cache() -> dict[tuple[Any, ...], Any]:
    cache = getattr(_FAST_LAUNCH_TLS, "cache", None)
    if cache is None:
        cache = {}
        _FAST_LAUNCH_TLS.cache = cache
    return cache


class StableI32Arg:
    """A stable Int32 runtime arg (avoids per-call Int32().__c_pointers__ allocations)."""

    def __init__(self, value: int):
        self._c_value = ctypes.c_int32(int(value))
        self._c_pointer = ctypes.cast(ctypes.pointer(self._c_value), ctypes.c_void_p)

    def set(self, value: int) -> None:
        self._c_value.value = int(value)

    def __c_pointers__(self):
        return [self._c_pointer]


class StableF32Arg:
    """A stable Float32 runtime arg (avoids per-call Float32().__c_pointers__ allocations)."""

    def __init__(self, value: float):
        self._c_value = ctypes.c_float(float(value))
        self._c_pointer = ctypes.cast(ctypes.pointer(self._c_value), ctypes.c_void_p)

    def set(self, value: float) -> None:
        self._c_value.value = float(value)

    def __c_pointers__(self):
        return [self._c_pointer]


def set_runtime_ptr(ptr: Any, device_ptr: int) -> None:
    """Update a CuTeDSL runtime Pointer descriptor in-place.

    This relies on internal runtime pointer fields (`_desc`, `_pointer`, etc.).
    If these internals change in a future CuTeDSL upgrade, this function may
    raise AttributeError; callers should catch it and fall back.
    """
    device_ptr = int(device_ptr)
    ptr._pointer = device_ptr  # type: ignore[attr-defined]
    if getattr(ptr, "_c_pointer", None) is None:
        ptr.__c_pointers__()  # type: ignore[attr-defined]
    ptr._desc.value = device_ptr  # type: ignore[attr-defined]
