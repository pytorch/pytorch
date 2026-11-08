import functools
import logging
import shutil

import torch
from torch._inductor.utils import clear_on_fresh_cache

from ... import config


log = logging.getLogger(__name__)


@clear_on_fresh_cache
@functools.lru_cache(1)
def get_cuda_arch() -> str | None:
    try:
        cuda_arch = config.cuda.arch
        if cuda_arch is None:
            # Get Compute Capability of the first Visible device
            major, minor = torch.cuda.get_device_capability(0)
            return str(major * 10 + minor)
        return str(cuda_arch)
    except Exception:
        log.exception("Error getting cuda arch")
        return None


@clear_on_fresh_cache
@functools.lru_cache(1)
def is_datacenter_blackwell_arch() -> bool:
    """True for NVIDIA GPUs with tcgen05: SM 10.x and SM 11.x (Jetson Thor)."""
    # ROCm reports the gfx version as the capability, e.g. (11, 0) for RDNA3.
    if torch.version.hip:
        return False
    arch = get_cuda_arch()
    if arch is None:
        return False
    arch_number = int(arch)
    # CUDA 13 renumbered Jetson Thor from SM 10.1 to SM 11.0. Like SM 10.x it has
    # tcgen05; SM 12.x (consumer Blackwell) does not.
    return arch_number >= 100 and arch_number < 120


@clear_on_fresh_cache
@functools.lru_cache(1)
def get_cuda_version() -> str | None:
    try:
        cuda_version = config.cuda.version
        if cuda_version is None:
            cuda_version = torch.version.cuda
        return cuda_version
    except Exception:
        log.exception("Error getting cuda version")
        return None


@functools.cache
def nvcc_exist(nvcc_path: str | None = "nvcc") -> bool:
    return nvcc_path is not None and shutil.which(nvcc_path) is not None
