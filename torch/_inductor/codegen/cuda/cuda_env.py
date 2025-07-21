import functools
import logging
import shutil
from typing import Optional

import torch
from torch._inductor.utils import clear_on_fresh_cache

from ... import config


log = logging.getLogger(__name__)


@clear_on_fresh_cache
@functools.lru_cache(1)
def get_cuda_arch() -> Optional[str]:
    try:
        cuda_arch = config.cuda.arch
        if cuda_arch is None:
            # Get Compute Capability of the first Visible device
            major, minor = torch.cuda.get_device_capability(0)
            return str(major * 10 + minor)
        return str(cuda_arch)
    except Exception as e:
        log.error("Error getting cuda arch: %s", e)
        return None


@clear_on_fresh_cache
@functools.lru_cache(1)
def get_cuda_version() -> Optional[str]:
    try:
        cuda_version = config.cuda.version
        if cuda_version is None:
            cuda_version = torch.version.cuda
        return cuda_version
    except Exception as e:
        log.error("Error getting cuda version: %s", e)
        return None


@functools.cache
def nvcc_exist(nvcc_path: Optional[str] = "nvcc") -> bool:
    return nvcc_path is not None and shutil.which(nvcc_path) is not None
