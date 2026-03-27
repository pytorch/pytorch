import enum
import functools

import torch
import torch.xpu
from torch.testing._internal.common_utils import IS_WINDOWS, LazyVal, TEST_XPU


XPU_ALREADY_INITIALIZED_ON_IMPORT = torch.xpu.is_initialized()


class XPUCodename(enum.Enum):
    PVC = "PVC"  # Intel® Data Center GPU Max Series
    BMG = "BMG"  # Intel® Arc™ Pro Battlemage Graphics


class XPUArch(enum.Enum):
    Xe = "Xe"  # Xe HPC
    Xe2 = "Xe2"


# device_id -> GPU codename
_DEVICE_ID_TO_CODENAME = {
    0x0BD0: XPUCodename.PVC,
    0x0BD5: XPUCodename.PVC,
    0x0BD6: XPUCodename.PVC,
    0x0BD7: XPUCodename.PVC,
    0x0BD8: XPUCodename.PVC,
    0x0BD9: XPUCodename.PVC,
    0x0BDA: XPUCodename.PVC,
    0x0BDB: XPUCodename.PVC,
    0x0B69: XPUCodename.PVC,
    0x0B6E: XPUCodename.PVC,
    0xE20B: XPUCodename.BMG,
    0xE20C: XPUCodename.BMG,
    0xE211: XPUCodename.BMG,
    0xE212: XPUCodename.BMG,
    0xE223: XPUCodename.BMG,
    0xE222: XPUCodename.BMG,
}

# GPU codename -> architecture
_CODENAME_TO_ARCH = {
    XPUCodename.PVC: XPUArch.Xe,
    XPUCodename.BMG: XPUArch.Xe2,
}


@functools.lru_cache(1)
def get_xpu_device_id() -> int | None:
    try:
        return torch.xpu.get_device_capability()["device_id"]
    except Exception:
        log.exception("Error in getting xpu device_id.")
        return None


@functools.lru_cache(1)
def get_xpu_codename() -> XPUCodename | None:
    device_id = get_xpu_device_id()
    if device_id is None:
        return None
    return _DEVICE_ID_TO_CODENAME.get(device_id)


@functools.lru_cache(1)
def get_xpu_arch() -> XPUArch | None:
    codename = get_xpu_codename()
    if codename is None:
        return None
    return _CODENAME_TO_ARCH.get(codename)


Xe2_Or_Later = LazyVal(
    lambda: torch.xpu.is_available() and get_xpu_arch() in (XPUArch.Xe2,)
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
