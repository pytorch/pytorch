import functools
import threading
from abc import ABC, abstractmethod
from typing import Tuple, Union

import torch


class TritonBackend(ABC):
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

    # Utility function
    @abstractmethod
    def namespace(self) -> str:
        pass

    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def device_name(self, device: torch.device) -> str:
        pass

    @abstractmethod
    def compatible_with_triton(self, device):
        pass

    @abstractmethod
    def target_version(self):
        pass

    @abstractmethod
    def mem_alignment(self):
        pass

    @abstractmethod
    def codegen_check(self):
        # This function is used to check whether the Triton backend
        # can generate code for Triton.
        pass

    @abstractmethod
    def gen_codegen_string(self, attr: str) -> Tuple[str, str]:
        pass

    def __str__(self):
        return self.name()

    def __hash__(self):
        return hash(self.name())

    def __bool__(self):
        return self.is_available() and self.codegen_check()


class _CUDABackend(TritonBackend):
    def __init__(self):
        super().__init__()
        self.properties = {
            idx: torch.cuda.get_device_properties(torch.device(idx))
            for idx in range(self.device_count())
        }
        self.codegen_check_properties = {
            "_DeviceGuard": torch.cuda,
            "set_device": torch.cuda,
            "synchronize": torch.cuda,
            "_cuda_getCurrentRawStream": torch._C,
        }

    def create_stream(self):
        return torch.cuda.Stream()

    def get_device_capability(self, device: torch.device):
        p = self.get_device_properties(device)
        return p.major, p.minor

    def get_device_properties(self, device: Union[torch.device, int]):
        idx = device.index if isinstance(device, torch.device) else device
        return self.properties[idx]

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
        return self.get_device_properties(index).multi_processor_count

    def is_available(self) -> bool:
        return torch.cuda.is_available()

    def namespace(self) -> str:
        return "torch.cuda"

    def device_name(self, device: torch.device) -> str:
        return self.get_device_capability(device).name

    def name(self) -> str:
        return "cuda"

    def target_version(self):
        major, minor = torch.cuda.get_device_capability()
        cc = major * 10 + minor
        return cc

    def mem_alignment(self):
        return 16

    def compatible_with_triton(self, device=0):
        device_props = self.get_device_properties(device)
        return device_props.major >= 7

    @functools.lru_cache(None)
    def codegen_check(self):
        return all(hasattr(v, k) for k, v in self.codegen_check_properties.items())

    def gen_codegen_string(self, attr: str) -> Tuple[str, str]:
        if attr in self.codegen_check_properties:
            return self.namespace(), attr
        elif attr == "getCurrentRawStream":
            return "torch._C", f"_{self.name()}_getCurrentRawStream"
        else:
            raise f"The Triton backend {self.name()} has not supported {attr} the for Triton code generation"


_triton_cuda_backend = _CUDABackend()

triton_backends = []

_register_lock = threading.Lock()


def register_triton_backend(_triton_backend: TritonBackend):
    assert isinstance(_triton_backend, TritonBackend)
    with _register_lock:
        all_backends = [(hash(backend)) for backend in triton_backends]
        if hash(_triton_backend) not in all_backends:
            triton_backends.append(_triton_backend)
            return True
        else:
            return _triton_backend in triton_backends


if _triton_cuda_backend:
    register_triton_backend(_triton_cuda_backend)


@functools.lru_cache(None)
def get_triton_backend(device_type: str) -> TritonBackend:
    for _triton_backend in triton_backends:
        if _triton_backend.name() == device_type:
            return _triton_backend

    return None


def all_triton_backend_name() -> str:
    return [str(_triton_backend) for _triton_backend in triton_backends]
