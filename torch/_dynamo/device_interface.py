# mypy: allow-untyped-defs
import inspect
from typing import Any, Callable, Dict, Iterable, Optional, Tuple, Type, Union

import torch
from torch._streambase import _EventBase, _StreamBase


get_cuda_stream: Optional[Callable[[int], int]]
if torch.cuda._is_compiled():
    from torch._C import _cuda_getCurrentRawStream as get_cuda_stream
else:
    get_cuda_stream = None

_device_t = Union[torch.device, str, int, None]

# Recording the device properties in the main process but used in worker process.
caching_worker_device_properties: Dict[str, Any] = {}
caching_worker_current_devices: Dict[str, int] = {}


class DeviceInterfaceMeta(type):
    def __new__(metacls, *args, **kwargs):
        class_member = args[2]
        if "Event" in class_member:
            assert inspect.isclass(class_member["Event"]) and issubclass(
                class_member["Event"], _EventBase
            ), "DeviceInterface member Event should be inherit from _EventBase"
        if "Stream" in class_member:
            assert inspect.isclass(class_member["Stream"]) and issubclass(
                class_member["Stream"], _StreamBase
            ), "DeviceInterface member Stream should be inherit from _StreamBase"
        return super().__new__(metacls, *args, **kwargs)


class DeviceInterface(metaclass=DeviceInterfaceMeta):
    """
    This is a simple device runtime interface for Inductor. It enables custom
    backends to be integrated with Inductor in a device-agnostic semantic.
    """

    class device:
        def __new__(cls, device: _device_t):
            raise NotImplementedError

    class Worker:
        """
        Worker API to query device properties that will work in multi processing
        workers that cannot use the GPU APIs (due to processing fork() and
        initialization time issues). Properties are recorded in the main process
        before we fork the workers.
        """

        @staticmethod
        def set_device(device: int):
            raise NotImplementedError

        @staticmethod
        def current_device() -> int:
            raise NotImplementedError

        @staticmethod
        def get_device_properties(device: _device_t = None):
            raise NotImplementedError

    @staticmethod
    def current_device():
        raise NotImplementedError

    @staticmethod
    def set_device(device: _device_t):
        raise NotImplementedError

    @staticmethod
    def maybe_exchange_device(device: int) -> int:
        raise NotImplementedError

    @staticmethod
    def exchange_device(device: int) -> int:
        raise NotImplementedError

    @staticmethod
    def device_count():
        raise NotImplementedError

    @staticmethod
    def is_available() -> bool:
        raise NotImplementedError

    @staticmethod
    def stream(stream: torch.Stream):
        raise NotImplementedError

    @staticmethod
    def current_stream():
        raise NotImplementedError

    @staticmethod
    def set_stream(stream: torch.Stream):
        raise NotImplementedError

    @staticmethod
    def _set_stream_by_id(stream_id: int, device_index: int, device_type: int):
        raise NotImplementedError

    @staticmethod
    def get_raw_stream():
        raise NotImplementedError

    @staticmethod
    def synchronize(device: _device_t = None):
        raise NotImplementedError

    @staticmethod
    def get_device_properties(device: _device_t = None):
        raise NotImplementedError

    @staticmethod
    def get_compute_capability(device: _device_t = None):
        raise NotImplementedError


class DeviceGuard:
    """
    This class provides a context manager for device switching. This is a stripped
    down version of torch.{device_name}.device.

    The context manager changes the current device to the given device index
    on entering the context and restores the original device on exiting.
    The device is switched using the provided device interface.
    """

    def __init__(self, device_interface: Type[DeviceInterface], index: Optional[int]):
        self.device_interface = device_interface
        self.idx = index
        self.prev_idx = -1

    def __enter__(self):
        if self.idx is not None:
            self.prev_idx = self.device_interface.exchange_device(self.idx)

    def __exit__(self, type: Any, value: Any, traceback: Any):
        if self.idx is not None:
            self.idx = self.device_interface.maybe_exchange_device(self.prev_idx)
        return False


class CudaInterface(DeviceInterface):
    device = torch.cuda.device

    # register Event and Stream class into the backend interface
    # make sure Event and Stream are implemented and inherited from the _EventBase and _StreamBase
    Event = torch.cuda.Event
    Stream = torch.cuda.Stream

    class Worker:
        @staticmethod
        def set_device(device: int):
            caching_worker_current_devices["cuda"] = device

        @staticmethod
        def current_device() -> int:
            if "cuda" in caching_worker_current_devices:
                return caching_worker_current_devices["cuda"]
            return torch.cuda.current_device()

        @staticmethod
        def get_device_properties(device: _device_t = None):
            if device is not None:
                if isinstance(device, str):
                    device = torch.device(device)
                    assert device.type == "cuda"
                if isinstance(device, torch.device):
                    device = device.index
            if device is None:
                device = CudaInterface.Worker.current_device()

            if "cuda" not in caching_worker_device_properties:
                device_prop = [
                    torch.cuda.get_device_properties(i)
                    for i in range(torch.cuda.device_count())
                ]
                caching_worker_device_properties["cuda"] = device_prop

            return caching_worker_device_properties["cuda"][device]

    current_device = staticmethod(torch.cuda.current_device)
    set_device = staticmethod(torch.cuda.set_device)
    device_count = staticmethod(torch.cuda.device_count)
    stream = staticmethod(torch.cuda.stream)  # type: ignore[assignment]
    current_stream = staticmethod(torch.cuda.current_stream)
    set_stream = staticmethod(torch.cuda.set_stream)  # type: ignore[assignment]
    _set_stream_by_id = staticmethod(torch.cuda._set_stream_by_id)  # type: ignore[assignment]
    synchronize = staticmethod(torch.cuda.synchronize)
    get_device_properties = staticmethod(torch.cuda.get_device_properties)  # type: ignore[assignment]
    get_raw_stream = staticmethod(get_cuda_stream)  # type: ignore[arg-type]
    exchange_device = staticmethod(torch.cuda._exchange_device)  # type: ignore[arg-type]
    maybe_exchange_device = staticmethod(torch.cuda._maybe_exchange_device)  # type: ignore[arg-type]

    # Can be mock patched by @patch decorator.
    @staticmethod
    def is_available() -> bool:
        return torch.cuda.is_available()

    @staticmethod
    def get_compute_capability(device: _device_t = None):
        if torch.version.hip is None:
            major, min = torch.cuda.get_device_capability(device)
            return major * 10 + min
        else:
            return torch.cuda.get_device_properties(device).gcnArchName.split(":", 1)[0]


get_xpu_stream: Optional[Callable[[int], int]]
if torch.xpu._is_compiled():
    from torch._C import _xpu_getCurrentRawStream as get_xpu_stream
else:
    get_xpu_stream = None


class XpuInterface(DeviceInterface):
    device = torch.xpu.device
    Event = torch.xpu.Event
    Stream = torch.xpu.Stream

    class Worker:
        @staticmethod
        def set_device(device: int):
            caching_worker_current_devices["xpu"] = device

        @staticmethod
        def current_device() -> int:
            if "xpu" in caching_worker_current_devices:
                return caching_worker_current_devices["xpu"]
            return torch.xpu.current_device()

        @staticmethod
        def get_device_properties(device: _device_t = None):
            if device is not None:
                if isinstance(device, str):
                    device = torch.device(device)
                    assert device.type == "xpu"
                if isinstance(device, torch.device):
                    device = device.index
            if device is None:
                device = XpuInterface.Worker.current_device()

            if "xpu" not in caching_worker_device_properties:
                device_prop = [
                    torch.xpu.get_device_properties(i)
                    for i in range(torch.xpu.device_count())
                ]
                caching_worker_device_properties["xpu"] = device_prop

            return caching_worker_device_properties["xpu"][device]

    current_device = staticmethod(torch.xpu.current_device)
    set_device = staticmethod(torch.xpu.set_device)
    device_count = staticmethod(torch.xpu.device_count)
    stream = staticmethod(torch.xpu.stream)  # type: ignore[assignment]
    current_stream = staticmethod(torch.xpu.current_stream)
    set_stream = staticmethod(torch.xpu.set_stream)  # type: ignore[assignment]
    _set_stream_by_id = staticmethod(torch.xpu._set_stream_by_id)  # type: ignore[assignment]
    synchronize = staticmethod(torch.xpu.synchronize)
    get_device_properties = staticmethod(torch.xpu.get_device_properties)  # type: ignore[assignment]
    get_raw_stream = staticmethod(get_xpu_stream)  # type: ignore[arg-type]
    exchange_device = staticmethod(torch.xpu._exchange_device)  # type: ignore[arg-type]
    maybe_exchange_device = staticmethod(torch.xpu._maybe_exchange_device)  # type: ignore[arg-type]

    # Can be mock patched by @patch decorator.
    @staticmethod
    def is_available() -> bool:
        return torch.xpu.is_available()

    @staticmethod
    def get_compute_capability(device: _device_t = None):
        cc = torch.xpu.get_device_capability(device)
        return cc


device_interfaces: Dict[str, Type[DeviceInterface]] = {}
_device_initialized = False


def register_interface_for_device(
    device: Union[str, torch.device], device_interface: Type[DeviceInterface]
):
    if isinstance(device, torch.device):
        device = str(device)
    device_interfaces[device] = device_interface


def get_interface_for_device(device: Union[str, torch.device]) -> Type[DeviceInterface]:
    if isinstance(device, torch.device):
        device = str(device)
    if not _device_initialized:
        init_device_reg()
    if device in device_interfaces:
        return device_interfaces[device]
    raise NotImplementedError(f"No interface for device {device}")


def get_registered_device_interfaces() -> Iterable[Tuple[str, Type[DeviceInterface]]]:
    if not _device_initialized:
        init_device_reg()
    return device_interfaces.items()


def init_device_reg():
    global _device_initialized
    register_interface_for_device("cuda", CudaInterface)
    for i in range(torch.cuda.device_count()):
        register_interface_for_device(f"cuda:{i}", CudaInterface)

    register_interface_for_device("xpu", XpuInterface)
    for i in range(torch.xpu.device_count()):
        register_interface_for_device(f"xpu:{i}", XpuInterface)

    _device_initialized = True
