import os
import torch


def cuda_or_xla_gpu_available():
    # use `XRT_DEVICE_MAP` to check whether GPU is enabled in torch/xla
    return torch.cuda.is_available() or "GPU" in os.getenv("XRT_DEVICE_MAP", "")
