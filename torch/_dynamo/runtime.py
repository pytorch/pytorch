from typing import Dict, Union

import torch

_device_t = Union[torch.device, str, int, None]


class GPURuntimeInterface:
    """
    This is a simple device runtime abstraction for Inductor. It enables custom
    backends to be integrated with Inductor in a device-agnostic semantic.
    """

    class Event:
        def __new__(cls, *args, **kwargs):
            raise NotImplementedError()

    class device:
        def __new__(cls, device: _device_t):
            raise NotImplementedError()

    @staticmethod
    def current_device():
        raise NotImplementedError()

    @staticmethod
    def set_device(device: _device_t):
        raise NotImplementedError()

    @staticmethod
    def device_count():
        raise NotImplementedError()

    @staticmethod
    def is_available():
        raise NotImplementedError()

    @staticmethod
    def current_stream():
        raise NotImplementedError()

    @staticmethod
    def set_stream(stream: torch.Stream):
        raise NotImplementedError()

    @staticmethod
    def get_raw_stream():
        raise NotImplementedError()

    @staticmethod
    def synchronize(device: _device_t = None):
        raise NotImplementedError()

    @staticmethod
    def get_device_properties(device: _device_t = None):
        raise NotImplementedError()

    @staticmethod
    def get_compute_capability(device: _device_t = None):
        raise NotImplementedError()


class CudaRuntime(GPURuntimeInterface):
    class Event:
        def __new__(cls, *args, **kwargs):
            return torch.cuda.Event(*args, **kwargs)

    class device:
        def __new__(cls, device: _device_t):
            return torch.cuda.device(device)

    @staticmethod
    def current_device() -> int:
        return torch.cuda.current_device()

    @staticmethod
    def set_device(device: _device_t):
        torch.cuda.set_device(device)

    @staticmethod
    def device_count() -> int:
        return torch.cuda.device_count()

    @staticmethod
    def is_available() -> bool:
        return torch.cuda.is_available()

    @staticmethod
    def current_stream():
        return torch.cuda.current_stream()

    @staticmethod
    def set_stream(stream: torch.Stream):
        torch.cuda.set_stream(stream)

    @staticmethod
    def get_raw_stream(device: int):
        from torch._C import _cuda_getCurrentRawStream as get_cuda_stream

        return get_cuda_stream(device)

    @staticmethod
    def synchronize(device: _device_t = None):
        return torch.cuda.synchronize(device)

    @staticmethod
    def get_device_properties(device: _device_t = None):
        return torch.cuda.get_device_properties(device)

    @staticmethod
    def get_compute_capability(device: _device_t = None):
        major, min = torch.cuda.get_device_capability(device)
        return major * 10 + min


device_runtimes: Dict[str, GPURuntimeInterface] = {}


def register_runtime_for_device(device: str, device_runtime: GPURuntimeInterface):
    device_runtimes[device] = device_runtime


def get_runtime_for_device(device: str):
    return device_runtimes[device] if device in device_runtimes else None


def get_registered_device_runtimes():
    return device_runtimes.items()


if torch.cuda.is_available():
    register_runtime_for_device("cuda", CudaRuntime)
