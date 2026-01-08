import functools
import logging
from typing import Optional

import torch
from torch._inductor.utils import clear_on_fresh_cache


log = logging.getLogger(__name__)


@clear_on_fresh_cache
@functools.lru_cache(1)
def get_xpu_arch() -> Optional[str]:
    arch_code2name = {
        13136561920: "Xe12",
        21479031808: "Xe20",
    }
    try:
        arch_code = torch.xpu.get_device_capability()["architecture"]
        return arch_code2name[arch_code]
    except Exception:
        log.exception("Error in getting xpu arch.")
        return None


@clear_on_fresh_cache
@functools.lru_cache(1)
def get_xpu_version() -> Optional[str]:
    # string of version, like 20250101
    try:
        xpu_version = torch.version.xpu
        return xpu_version
    except Exception:
        log.exception("Error getting xpu version")
        return None
