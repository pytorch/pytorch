import warnings

import torch

import torch.utils.device_limits.nvidia_gpu as nv
from torch._C import dtype


def get_tflops_per_second_per_dtype(
    target_device: torch.device,
    data_type: dtype,
) -> float:
    if torch.cuda.is_available():
        return nv.get_tflops_per_second(target_device, data_type)
    else:
        warnings.warn(
            "Tflops per second is only supported for CUDA devices. Returning default value of 1.0.",
            UserWarning,
        )
    return 1.0


def get_memory_bandwidth_GBps(target_device: torch.device) -> float:
    if torch.cuda.is_available():
        bytes_per_second = nv.get_memory_bandwidth_Bps(target_device)
        return bytes_per_second / 1e9
    else:
        warnings.warn(
            "Memory bandwidth is only supported for CUDA devices. Returning default value of 1.0.",
            UserWarning,
        )
    return 1.0


def get_memory_bandwidth_GiBps(target_device: torch.device) -> float:
    if torch.cuda.is_available():
        bytes_per_second = nv.get_memory_bandwidth_Bps(target_device)
        return bytes_per_second / (2**30)
    else:
        warnings.warn(
            "Memory bandwidth is only supported for CUDA devices. Returning default value of 1.0.",
            UserWarning,
        )
    return 1.0


def get_l1_cache_bandwidth_GBps(target_device: torch.device) -> float:
    if torch.cuda.is_available():
        bytes_per_second = nv.get_shared_memory_bandwidth_Bps(target_device)
        return bytes_per_second / 1e9
    else:
        warnings.warn(
            "Local cache bandwidth is only supported for CUDA devices. Returning default value of 1.0.",
            UserWarning,
        )
    return 1.0
