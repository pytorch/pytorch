import functools
import logging

import torch
from torch._inductor.utils import clear_on_fresh_cache


log = logging.getLogger(__name__)


@clear_on_fresh_cache
@functools.lru_cache(1)
def get_xpu_arch() -> str | None:
    from torch.testing._internal.common_xpu import get_xpu_codename, XPUCodename

    name2arch = {
        XPUCodename.PVC: "Xe12",
        XPUCodename.BMG: "Xe20",
    }

    codename = get_xpu_codename()
    if not codename or codename not in name2arch:
        log.warning("Unknown XPU codename, cannot determine architecture")
        return None

    return name2arch[codename]


@clear_on_fresh_cache
@functools.lru_cache(1)
def get_xpu_version() -> str | None:
    # string of version, like 20250101
    try:
        xpu_version = torch.version.xpu or ""
        return xpu_version
    except Exception:
        log.exception("Error getting xpu version")
        return None
