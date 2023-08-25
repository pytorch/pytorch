import functools
import logging
from subprocess import PIPE, Popen

import torch

from ... import config


def get_cuda_arch() -> str:
    try:
        cuda_arch = config.cuda.arch
        if cuda_arch is None:
            # Get Compute Capability of the first Visible device
            major, minor = torch.cuda.get_device_capability(0)
            cuda_arch = major * 10 + minor
        return str(cuda_arch)
    except Exception as e:
        logging.error(f"Error getting cuda arch: {e=}")
        return None


def get_cuda_version() -> str:
    try:
        cuda_version = config.cuda.version
        if cuda_version is None:
            cuda_version = torch.version.cuda
        return cuda_version
    except Exception as e:
        logging.error(f"Error getting cuda version: {e=}")
        return None


def nvcc_exist() -> bool:
    import subprocess
    res = subprocess.call(["which", "nvcc"])
    return res == 0
