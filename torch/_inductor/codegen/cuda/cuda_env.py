import functools
import logging
import torch
from subprocess import PIPE, Popen

from ... import config


def _detect_cuda_arch() -> str:
    # Get Compute Capability of the first Visible device
    major, minor = torch.cuda.get_device_capability(0)
    comp_cap = major * 10 + minor
    if comp_cap >= 90:
        return "90"
    elif comp_cap >= 80:
        return "80"
    elif comp_cap >= 75:
        return "75"
    elif comp_cap >= 70:
        return "70"
    else:
        return None


@functools.lru_cache(maxsize=1)
def get_cuda_arch() -> str:
    try:
        cuda_arch = config.cuda.arch
        if cuda_arch is None:
            cuda_arch = _detect_cuda_arch()
        return cuda_arch
    except Exception as e:
        logging.error(f"Error getting cuda arch: {e=}")
        return None


@functools.lru_cache(maxsize=1)
def get_cuda_version() -> str:
    try:
        cuda_version = config.cuda.version
        if cuda_version is None:
            cuda_version = torch.version.cuda
        return cuda_version
    except Exception as e:
        logging.error(f"Error getting cuda version: {e=}")
        return None
