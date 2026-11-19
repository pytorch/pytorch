import torch
import torch.xpu
from torch.testing._internal.common_utils import IS_WINDOWS, LazyVal, TEST_XPU
from torch.xpu._utils import get_xpu_arch, XPUArch


XPU_ALREADY_INITIALIZED_ON_IMPORT = torch.xpu.is_initialized()


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
