"""Machine identity, TestEnvironment flags and properties for the report line.
The only PyTorch-aware module of the writer. capture() runs in the test process
before its tests, so it creates no device context (no torch.cuda lazy init) and
doesn't call torch.accelerator, either of which tests can observe."""

from __future__ import annotations

import contextlib
import json
import os
import platform
import re
import subprocess
import sys
import sysconfig
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass
class Environment:
    os: str
    os_version: str
    cpu_architecture: str
    cpu_capability: str
    python_version: str
    cc_compiler: str
    cc_compiler_version: str
    accelerator: str
    accelerator_version: str
    device_count: int
    device_name: str
    flags: dict[str, str]
    properties: dict[str, str]

    def identity(self) -> dict[str, Any]:
        names = (
            "os", "os_version", "cpu_architecture", "cpu_capability", "python_version",
            "cc_compiler", "cc_compiler_version", "accelerator", "accelerator_version",
            "device_count", "device_name",
        )  # fmt: skip
        return {name: getattr(self, name) for name in names}


def capture() -> Environment:
    from torch.testing._internal.common_utils import TestEnvironment

    os_name, os_version, os_release = _os()
    cc_compiler, cc_version = _compiler(torch.__config__.show())
    accelerator, acc_version = _accelerator()
    device_count = _device_count(accelerator)
    device_name, memory, driver = _device(accelerator) if device_count else ("", "", "")
    cpu_capability = torch.backends.cpu.get_cpu_capability().lower().replace(" ", "")
    if cpu_capability == "avx512" and torch.cpu._is_amx_tile_supported():
        cpu_capability = "amx"
    properties = {
        "torch_version": torch.__version__,
        "os_release": os_release,
        "device_memory_mib": memory,
        "driver_version": driver,
        "host_memory_mib": _host_memory_mib(),
        "build_environment": os.environ.get("BUILD_ENVIRONMENT", ""),
        "test_config": os.environ.get("TEST_CONFIG", ""),
        "runner_name": os.environ.get("RUNNER_NAME", ""),
    }
    return Environment(
        os=os_name,
        os_version=os_version,
        cpu_architecture=_cpu_architecture(),
        cpu_capability=cpu_capability,
        python_version=_python_version(),
        cc_compiler=cc_compiler,
        cc_compiler_version=cc_version,
        accelerator=accelerator,
        accelerator_version=acc_version,
        device_count=device_count,
        device_name=device_name,
        flags=dict(TestEnvironment.env_var_values),
        properties={k: v for k, v in properties.items() if v},
    )


def _os() -> tuple[str, str, str]:
    system = platform.system()
    if system == "Linux":
        try:
            release = platform.freedesktop_os_release()
        except OSError:
            release = {}
        return "linux", release.get("VERSION_ID", ""), release.get("PRETTY_NAME", "")
    if system == "Darwin":
        version = platform.mac_ver()[0]
        return "macos", ".".join(version.split(".")[:2]), version
    if system == "Windows":
        # Every Windows since 10 and Server 2016 is NT 10.0, so keep the build: it
        # changes only with the release (17763 is Server 2019, 20348 Server 2022).
        version = platform.version()
        return "windows", version, version
    return system.lower(), "", ""


def _cpu_architecture() -> str:
    machine = platform.machine().lower()
    return {"amd64": "x86_64", "arm64": "aarch64"}.get(machine, machine)


def _python_version() -> str:
    free_threaded = "t" if sysconfig.get_config_var("Py_GIL_DISABLED") else ""
    return f"{sys.version_info.major}.{sys.version_info.minor}{free_threaded}"


def _compiler(config: str) -> tuple[str, str]:
    # clang builds also print a GCC line (clang defines __GNUC__), so check clang
    # and MSVC first.
    for name, pattern in (("clang", r"clang"), ("msvc", r"MSVC"), ("gcc", r"GCC")):
        match = re.search(rf"^  - {pattern} (\S+)", config, re.MULTILINE)
        if match:
            version = match.group(1)
            # MSVC prints _MSC_FULL_VER, e.g. 194134120 for 19.41.
            major = version[:2] if name == "msvc" else version.split(".")[0]
            return name, major
    return "", ""


def _accelerator() -> tuple[str, str]:
    if torch.version.cuda:
        return "cuda", torch.version.cuda
    if torch.version.hip:
        # The ROCm release (10.1), not HIP's own version (7.16).
        return "rocm", ".".join((torch.version.rocm or "").split(".")[:2])
    xpu = str(getattr(torch.version, "xpu", "") or "")
    if xpu:
        # The SYCL version packed as major * 10000 + minor * 100 + patch.
        version = xpu
        if xpu.isdigit() and len(xpu) == 8:
            version = f"{xpu[:4]}.{int(xpu[4:6])}"
        return "xpu", version
    if torch.backends.mps.is_built():
        return "mps", ""
    return "cpu", ""


def _device_count(accelerator: str) -> int:
    if accelerator == "cpu":
        return 0
    # ROCm builds expose HIP devices as torch.cuda.
    backend = "cuda" if accelerator == "rocm" else accelerator
    try:
        # An explicit backend, not torch.accelerator: that latches MTIA's hooks
        # before a test can register its own (test_cpp_extensions_mtia_backend) and
        # counts PrivateUse1 backends that tests register at import.
        return torch.get_device_module(backend).device_count()
    except Exception:
        return 0


def _device(accelerator: str) -> tuple[str, str, str]:
    """Name, memory (MiB) and driver of the first visible device, read without a
    device context, which would poison fork."""
    try:
        if accelerator == "cuda":
            return _nvml_device()
        if accelerator == "rocm":
            return _amdsmi_device()
        if accelerator == "xpu":
            return _xpu_smi_device()
        if accelerator == "mps":
            return torch.backends.mps.get_name(), "", ""
    except Exception:
        pass
    return "", "", ""


def _nvml_device() -> tuple[str, str, str]:
    import ctypes

    class Memory(ctypes.Structure):
        _fields_ = [(name, ctypes.c_ulonglong) for name in ("total", "free", "used")]

    nvml = ctypes.CDLL("nvml.dll" if sys.platform == "win32" else "libnvidia-ml.so.1")
    if nvml.nvmlInit() != 0:
        return "", "", ""
    try:
        handle = ctypes.c_void_p()
        # An int device, so CUDA_VISIBLE_DEVICES is resolved without initializing CUDA.
        index = ctypes.c_uint(torch.cuda._get_nvml_device_index(0))
        if nvml.nvmlDeviceGetHandleByIndex_v2(index, ctypes.byref(handle)) != 0:
            return "", "", ""
        name = ctypes.create_string_buffer(96)
        driver = ctypes.create_string_buffer(80)
        memory = Memory()
        name_value = (
            name.value.decode() if nvml.nvmlDeviceGetName(handle, name, 96) == 0 else ""
        )
        memory_value = (
            str(memory.total >> 20)
            if nvml.nvmlDeviceGetMemoryInfo(handle, ctypes.byref(memory)) == 0
            else ""
        )
        driver_value = (
            driver.value.decode()
            if nvml.nvmlSystemGetDriverVersion(driver, 80) == 0
            else ""
        )
        return name_value, memory_value, driver_value
    finally:
        nvml.nvmlShutdown()


def _amdsmi_device() -> tuple[str, str, str]:
    # Maps HIP device 0 to its amdsmi index. Like torch.cuda's own amdsmi calls, it
    # leaves amdsmi initialized.
    handle = torch.cuda._get_amdsmi_handler(0)
    amdsmi = torch.cuda.amdsmi
    name = amdsmi.amdsmi_get_gpu_asic_info(handle)["market_name"]
    memory = driver = ""
    with contextlib.suppress(Exception):
        vram = amdsmi.amdsmi_get_gpu_memory_total(handle, amdsmi.AmdSmiMemoryType.VRAM)
        memory = str(int(vram) >> 20)
    with contextlib.suppress(Exception):
        driver = amdsmi.amdsmi_get_gpu_driver_info(handle)["driver_version"]
    return name, memory, driver


def _xpu_smi_device() -> tuple[str, str, str]:
    output = subprocess.check_output(
        ["xpu-smi", "discovery", "-j"], text=True, timeout=30
    )
    discovery = json.loads(output)
    return discovery["device_list"][0]["device_name"], "", ""


def _host_memory_mib() -> str:
    try:
        limit = Path("/sys/fs/cgroup/memory.max").read_text().strip()
        if limit.isdigit():
            return str(int(limit) >> 20)
    except OSError:
        pass
    try:
        return str((os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")) >> 20)
    except (AttributeError, OSError, ValueError):
        return ""
