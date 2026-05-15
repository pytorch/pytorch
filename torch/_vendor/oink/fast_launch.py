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

import cuda.bindings.driver as cuda  # type: ignore

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


class GenericFastLaunch:
    def __init__(
        self,
        *,
        capi_func: object,
        executor: object,
        packed_args: object,
        keepalive: tuple[object, ...],
        ptr_slots: tuple[tuple[object | None, str], ...],
        scalar_slots: tuple[tuple[object, str, object], ...],
        fallback_launch,
    ):
        self._capi_func = capi_func
        self._packed_args = packed_args
        self._keepalive = keepalive
        self._cuda_result = getattr(executor, "cuda_result", None)
        self._use_fast_launch = True
        self._ptr_slots = ptr_slots
        self._ptr_last = [-1] * len(ptr_slots)
        self._scalar_slots = scalar_slots
        self._scalar_last = [initial for _, _, initial in scalar_slots]
        self._fallback_launch = fallback_launch

    def launch(self, **kwargs) -> None:
        if not fast_launch_enabled() or not self._use_fast_launch:
            self._fallback_launch(**kwargs)
            return
        try:
            for idx, (runtime_ptr, tensor_name) in enumerate(self._ptr_slots):
                if runtime_ptr is None:
                    continue
                device_ptr = kwargs[tensor_name].data_ptr()
                if device_ptr != self._ptr_last[idx]:
                    set_runtime_ptr(runtime_ptr, device_ptr)
                    self._ptr_last[idx] = device_ptr
            for idx, (stable_arg, arg_name, _) in enumerate(self._scalar_slots):
                value = kwargs[arg_name]
                if value != self._scalar_last[idx]:
                    stable_arg.set(value)
                    self._scalar_last[idx] = value
        except AttributeError:
            self._use_fast_launch = False
            disable_fast_launch()
            self._fallback_launch(**kwargs)
            return

        if self._cuda_result is not None:
            self._cuda_result.value = 0
        ret = self._capi_func(self._packed_args)  # type: ignore[misc]
        if ret != 0:
            raise RuntimeError(f"CuTeDSL capi_func returned non-zero: {ret}")
        if self._cuda_result is not None:
            err = int(self._cuda_result.value)
            if err != 0:
                raise RuntimeError(f"CuTeDSL kernel launch failed (cuda_result={err})")


def build_fast_launcher(
    *,
    key: tuple[object, ...],
    compiled: object,
    device_index: int,
    stream_handle: int,
    execution_args_builder,
    keepalive_items: tuple[object, ...],
    ptr_slots: tuple[tuple[object | None, str], ...],
    scalar_slots: tuple[tuple[object, str, object], ...],
    fallback_launch_builder,
):
    if not fast_launch_enabled():
        return None
    cache = tls_cache()
    cached = cache.get(key)
    if cached is not None:
        return cached

    stream = cuda.CUstream(int(stream_handle))
    executor = compiled.to(device_index)  # type: ignore[attr-defined]
    try:
        exe_args, adapted_args = executor.generate_execution_args(
            *execution_args_builder(stream)
        )
        packed_args = executor._get_invoke_packed_args(list(exe_args))  # type: ignore[attr-defined]
        capi_func = compiled.capi_func  # type: ignore[attr-defined]
    except AttributeError:
        disable_fast_launch()
        return None

    launcher = GenericFastLaunch(
        capi_func=capi_func,
        executor=executor,
        packed_args=packed_args,
        keepalive=(executor, *keepalive_items, stream, *adapted_args),
        ptr_slots=ptr_slots,
        scalar_slots=scalar_slots,
        fallback_launch=fallback_launch_builder(stream),
    )
    cache[key] = launcher
    return launcher
