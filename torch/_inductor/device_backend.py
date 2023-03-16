from abc import ABC, abstractmethod

import torch


class DeviceBackend(ABC):
    @abstractmethod
    def create_stream(self):
        pass

    @abstractmethod
    def get_device_capability(self, device=None):
        pass

    @abstractmethod
    def get_device_properties(self, device):
        pass

    @abstractmethod
    def current_stream(self):
        pass

    @abstractmethod
    def wait_stream(self):
        pass

    @abstractmethod
    def device_count(self):
        pass

    @abstractmethod
    def synchronize(self):
        pass

    @abstractmethod
    def current_device(self):
        pass

    @abstractmethod
    def set_device(self):
        pass

    @abstractmethod
    def vinfo(self):
        pass

    @abstractmethod
    def allow_tf32(self):
        pass

    @abstractmethod
    def tf32_min_version(self):
        pass

    @abstractmethod
    def processor_count(self, index):
        pass

    @abstractmethod
    def is_available(self) -> bool:
        pass

    @abstractmethod
    def _DeviceGuard(self):
        pass

    # Utility function
    @abstractmethod
    def nms(self):
        pass

    @abstractmethod
    def backend_name(self):
        pass

    def __str__(self):
        return self.backend_name()

    def __hash__(self):
        return hash(self.backend_name())

    def __bool__(self):
        return self.is_available()


class _CUDABackend(DeviceBackend):
    def create_stream(self):
        return torch.cuda.Stream()

    def get_device_capability(self, device=None):
        p = self.get_device_properties(device)
        return p.major, p.minor

    def get_device_properties(self, device):
        return torch.cuda.get_device_properties(device)

    def current_stream(self):
        return torch.cuda.current_stream()

    def wait_stream(self, stream):
        return torch.cuda.current_stream().wait_stream(stream)

    def device_count(self):
        return torch.cuda.device_count()

    def synchronize(self):
        return torch.cuda.synchronize()

    def current_device(self):
        return torch.cuda.current_device()

    def set_device(self, index):
        return torch.cuda.set_device(index)

    def vinfo(self):
        return torch.version.cuda

    def allow_tf32(self):
        return torch.backends.cuda.matmul.allow_tf32

    def tf32_min_version(self):
        return (8, 0)

    def processor_count(self, index):
        return torch.cuda.get_device_properties(index).multi_processor_count

    def is_available(self) -> bool:
        return torch.cuda.is_available()

    def nms(self):
        return "torch.cuda"

    def _DeviceGuard(self, index):
        return torch.cuda._DeviceGuard(index)

    def backend_name(self):
        return "cuda"


CUDA_BACKEND = _CUDABackend()

ALL_DEVICE_BACKENDS = [CUDA_BACKEND]


def register_device_backend(device_backend: DeviceBackend):
    assert isinstance(device_backend, DeviceBackend)
    all_backends = [(hash(backend)) for backend in ALL_DEVICE_BACKENDS]
    if hash(device_backend) not in all_backends:
        ALL_DEVICE_BACKENDS.append(device_backend)
        return True
    else:
        return device_backend in ALL_DEVICE_BACKENDS
