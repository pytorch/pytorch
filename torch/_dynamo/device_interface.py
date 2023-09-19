from typing import Dict, Union

import torch

if torch.cuda._is_compiled():
    from torch._C import _cuda_getCurrentRawStream as get_cuda_stream
else:
    get_cuda_stream = None

_device_t = Union[torch.device, str, int, None]


class DeviceInterface:
    """
    This is a simple device runtime interface for Inductor. It enables custom
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
    def is_available() -> bool:
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


class CudaInterface(DeviceInterface):
    Event = torch.cuda.Event
    device = torch.cuda.device
    current_device = staticmethod(torch.cuda.current_device)
    set_device = staticmethod(torch.cuda.set_device)
    device_count = staticmethod(torch.cuda.device_count)
    current_stream = staticmethod(torch.cuda.current_stream)
    set_stream = staticmethod(torch.cuda.set_stream)
    synchronize = staticmethod(torch.cuda.synchronize)
    get_device_properties = staticmethod(torch.cuda.get_device_properties)
    get_raw_stream = staticmethod(get_cuda_stream)

    @staticmethod
    def is_available() -> bool:
        return torch.cuda.is_available()

    @staticmethod
    def get_compute_capability(device: _device_t = None):
        major, min = torch.cuda.get_device_capability(device)
        return major * 10 + min


device_interfaces: Dict[str, DeviceInterface] = {}


def register_interface_for_device(device: str, device_interface: DeviceInterface):
    device_interfaces[device] = device_interface


def get_interface_for_device(device: str):
    return device_interfaces[device] if device in device_interfaces else None


def get_registered_device_interfaces():
    return device_interfaces.items()


register_interface_for_device("cuda", CudaInterface)
