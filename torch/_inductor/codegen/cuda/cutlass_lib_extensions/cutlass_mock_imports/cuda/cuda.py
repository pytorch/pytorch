# mypy: disable-error-code="no-untyped-def"
# flake8: noqa
import torch


class CUdeviceptr:
    pass


class CUstream:
    def __init__(self, v):
        pass


class CUresult:
    CUDA_SUCCESS = True


class nvrtc:
    pass


def cuDeviceGetCount():
    return (CUresult.CUDA_SUCCESS, torch.cuda.device_count())
