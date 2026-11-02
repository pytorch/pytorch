r"""ctypes NVML binding backing the NVML-based ``torch.cuda`` APIs.

``torch.cuda`` only uses a small slice of NVML (the device query APIs behind
``utilization``, ``memory_usage``, ``temperature``, ``power_draw``,
``clock_rate`` and ``list_gpu_processes``). NVML ships with the NVIDIA driver
(``nvml.dll`` on Windows, ``libnvidia-ml.so.1`` on Linux), so binding that slice
directly lets those APIs work without the optional ``nvidia-ml-py`` wheel, which
is frequently absent on Windows and conda installs (issue #118173). This mirrors
the C++ loader in ``c10/cuda/driver_api.cpp``.

The public names here intentionally match the ``pynvml`` names previously used
in ``torch.cuda`` so the call sites did not have to change.
"""

import ctypes
import sys
import threading


# nvmlReturn_t (nvml.h)
NVML_SUCCESS = 0
NVML_ERROR_INSUFFICIENT_SIZE = 7
NVML_ERROR_DRIVER_NOT_LOADED = 9
NVML_ERROR_FUNCTION_NOT_FOUND = 13

# nvmlClockType_t.NVML_CLOCK_SM
NVML_CLOCK_SM = 1


class NVMLError(Exception):
    def __init__(self, value: int) -> None:
        self.value = value
        super().__init__(f"NVML call failed with error {value}")


class NVMLError_DriverNotLoaded(NVMLError):
    pass


class _nvmlMemory_t(ctypes.Structure):
    _fields_ = [
        ("total", ctypes.c_ulonglong),
        ("free", ctypes.c_ulonglong),
        ("used", ctypes.c_ulonglong),
    ]


class _nvmlUtilization_t(ctypes.Structure):
    _fields_ = [
        ("gpu", ctypes.c_uint),
        ("memory", ctypes.c_uint),
    ]


# Matches the nvmlProcessInfo_t layout expected by
# nvmlDeviceGetComputeRunningProcesses_v3.
class _nvmlProcessInfo_t(ctypes.Structure):
    _fields_ = [
        ("pid", ctypes.c_uint),
        ("usedGpuMemory", ctypes.c_ulonglong),
        ("gpuInstanceId", ctypes.c_uint),
        ("computeInstanceId", ctypes.c_uint),
    ]


_lib: ctypes.CDLL | None = None
_lib_lock = threading.Lock()


def _load() -> ctypes.CDLL:
    global _lib
    if _lib is not None:
        return _lib
    with _lib_lock:
        if _lib is None:
            name = "nvml.dll" if sys.platform == "win32" else "libnvidia-ml.so.1"
            try:
                _lib = ctypes.CDLL(name)
            except OSError as err:
                raise NVMLError_DriverNotLoaded(NVML_ERROR_DRIVER_NOT_LOADED) from err
    return _lib


def _check(result: int) -> None:
    if result == NVML_SUCCESS:
        return
    if result == NVML_ERROR_DRIVER_NOT_LOADED:
        raise NVMLError_DriverNotLoaded(result)
    raise NVMLError(result)


def nvmlInit() -> None:
    _check(_load().nvmlInit_v2())


def nvmlShutdown() -> None:
    _check(_load().nvmlShutdown())


def nvmlDeviceGetHandleByIndex(index: int) -> ctypes.c_void_p:
    handle = ctypes.c_void_p()
    get_handle = _load().nvmlDeviceGetHandleByIndex_v2
    _check(get_handle(ctypes.c_uint(index), ctypes.byref(handle)))
    return handle


def nvmlDeviceGetMemoryInfo(handle: ctypes.c_void_p) -> _nvmlMemory_t:
    memory = _nvmlMemory_t()
    _check(_load().nvmlDeviceGetMemoryInfo(handle, ctypes.byref(memory)))
    return memory


def nvmlDeviceGetUtilizationRates(handle: ctypes.c_void_p) -> _nvmlUtilization_t:
    util = _nvmlUtilization_t()
    _check(_load().nvmlDeviceGetUtilizationRates(handle, ctypes.byref(util)))
    return util


# Call an NVML getter whose last parameter is an `unsigned int*` out-param.
def _uint_out(name: str, *args: object) -> int:
    out = ctypes.c_uint()
    _check(getattr(_load(), name)(*args, ctypes.byref(out)))
    return out.value


def nvmlDeviceGetTemperature(handle: ctypes.c_void_p, sensor: int) -> int:
    return _uint_out("nvmlDeviceGetTemperature", handle, ctypes.c_uint(sensor))


def nvmlDeviceGetPowerUsage(handle: ctypes.c_void_p) -> int:
    return _uint_out("nvmlDeviceGetPowerUsage", handle)


def nvmlDeviceGetClockInfo(handle: ctypes.c_void_p, clock_type: int) -> int:
    return _uint_out("nvmlDeviceGetClockInfo", handle, ctypes.c_uint(clock_type))


def nvmlDeviceGetMaxClockInfo(handle: ctypes.c_void_p, clock_type: int) -> int:
    return _uint_out("nvmlDeviceGetMaxClockInfo", handle, ctypes.c_uint(clock_type))


def nvmlDeviceGetComputeRunningProcesses(
    handle: ctypes.c_void_p,
) -> list[_nvmlProcessInfo_t]:
    lib = _load()
    # _v3 and _v2 share the 4-field nvmlProcessInfo_t layout; prefer _v3 and
    # fall back to _v2 on drivers that predate it.
    fn = None
    for suffix in ("_v3", "_v2"):
        try:
            fn = getattr(lib, "nvmlDeviceGetComputeRunningProcesses" + suffix)
            break
        except AttributeError:
            continue
    if fn is None:
        raise NVMLError(NVML_ERROR_FUNCTION_NOT_FOUND)
    # Query the count first (NVML_ERROR_INSUFFICIENT_SIZE), then the entries.
    count = ctypes.c_uint(0)
    result = fn(handle, ctypes.byref(count), None)
    if result == NVML_SUCCESS:
        return []
    if result != NVML_ERROR_INSUFFICIENT_SIZE:
        _check(result)
    procs = (_nvmlProcessInfo_t * count.value)()
    _check(fn(handle, ctypes.byref(count), procs))
    return [procs[i] for i in range(count.value)]
