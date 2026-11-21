import functools
import json
import re

import torch
import torch.xpu
from torch.testing._internal.common_utils import (
    IS_WINDOWS,
    LazyVal,
    TemporaryFileName,
    TEST_XPU,
)
from torch.xpu._utils import get_xpu_arch, XPUArch


XPU_ALREADY_INITIALIZED_ON_IMPORT = torch.xpu.is_initialized()


@functools.lru_cache(1)
def _pti_version():
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.XPU,
        ],
    ) as p:
        pass
    with TemporaryFileName() as trace_path:
        p.export_chrome_trace(trace_path)
        with open(trace_path) as f:
            data = json.load(f)
    match = re.match(r"(\d+)\.(\d+)\.(\d+)", str(data.get("xpupti_version", "")))
    return tuple(int(component) for component in match.groups()) if match else None


def _pti_version_at_least(major, minor, patch):
    version = _pti_version()
    return version is not None and version >= (major, minor, patch)


Xe2_Or_Later = LazyVal(
    lambda: torch.xpu.is_available() and get_xpu_arch() >= XPUArch.Xe2
)


def evaluate_platform_supports_flash_attention():
    if TEST_XPU:
        return not IS_WINDOWS and Xe2_Or_Later
    return False


PLATFORM_SUPPORTS_FLASH_ATTENTION_XPU: bool = LazyVal(
    lambda: evaluate_platform_supports_flash_attention()
)

# Importing this module should NOT eagerly initialize XPU
if not XPU_ALREADY_INITIALIZED_ON_IMPORT:
    if torch.xpu.is_initialized():
        raise AssertionError("XPU should not be initialized on import")
