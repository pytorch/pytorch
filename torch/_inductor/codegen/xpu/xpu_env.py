import functools
import logging
from typing import Optional

import torch
from torch._inductor.utils import clear_on_fresh_cache


log = logging.getLogger(__name__)


@clear_on_fresh_cache
@functools.lru_cache(1)
def get_xpu_arch() -> Optional[str]:
    arch_name2code = {"pvc": "11"}
    try:
        assert len(torch.xpu.get_arch_list()) == 1
        arch_name = torch.xpu.get_arch_list()[0]
        return arch_name2code[arch_name]
    except Exception as e:
        log.error("Error getting xpu arch: %s", e)
        return None


@clear_on_fresh_cache
@functools.lru_cache(1)
def get_xpu_version() -> Optional[str]:
    # string of version, like 20250101
    try:
        xpu_version = torch.version.xpu
        return xpu_version
    except Exception as e:
        log.error("Error getting xpu version: %s", e)
        return None
