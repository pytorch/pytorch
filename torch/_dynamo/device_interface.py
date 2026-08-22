"""
Device abstraction layer for TorchDynamo and Inductor backends.

This module provides a unified interface for different hardware backends (CUDA, XPU,
CPU, MPS, MTIA) through a common device interface. Key components include:

- DeviceInterface: Base class defining the common API for all device types
- Device-specific implementations: CudaInterface, XpuInterface, CpuInterface, MpsInterface, MtiaInterface
- Device registration system for managing available backends
- Worker APIs for multi-processing scenarios
- Stream and event management across different devices
- Device property caching for worker processes

The abstraction layer enables device-agnostic code in TorchDynamo while allowing
specialized implementations for each hardware backend's unique features.
"""

import functools
import inspect
import time
from collections import namedtuple
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from typing import Any, Literal

import torch
from torch.utils._pallas import has_torch_tpu


get_cuda_stream: Callable[[int], int] | None
if torch.cuda._is_compiled():
    from torch._C import _cuda_getCurrentRawStream as get_cuda_stream
else:
    get_cuda_stream = None

# Recording the device properties in the main process but used in worker process.
caching_worker_device_properties: dict[str, Any] = {}
caching_worker_current_devices: dict[str, int] = {}


class DeviceInterface:
    """
    This is a simple device runtime interface for Dynamo and Inductor. It
    enables custom backends to be integrated with them in a device-agnostic
    semantic.
    """

    # Autocast classes this device provides, e.g. ``torch.foo.amp.autocast``.
    # Dynamo uses these to route an out-of-tree autocast class to the
    # device_type it was registered under; see
    # ``device_type_for_autocast_class``.  Entries must be strict subclasses of
    # ``torch.amp.autocast_mode.autocast``.
    autocast_classes: frozenset[type] = frozenset()

    class device:
        def __new__(cls, device: torch.types.Device) -> Any:
            raise NotImplementedError

    class Event:
        def __new__(cls, *args: Any, **kwargs: Any) -> Any:
            raise NotImplementedError(
                "Event should be inherited from torch.Event, otherwise, it couldn't be captured by dynamo."
            )

    class Stream:
        def __new__(cls, *args: Any, **kwargs: Any) -> Any:
            raise NotImplementedError(
                "Stream should be inherited from torch.Stream, otherwise, it couldn't be captured by dynamo."
            )

    class Worker:
        """
        Worker API to query device properties that will work in multi processing
        workers that cannot use the GPU APIs (due to processing fork() and
        initialization time issues). Properties are recorded in the main process
        before we fork the workers.
        """

        @staticmethod
        def set_device(device: int) -> None:
            raise NotImplementedError

        @staticmethod
        def current_device() -> int:
            raise NotImplementedError

        @staticmethod
        def get_device_properties(device: torch.types.Device = None) -> Any:
            raise NotImplementedError

    @staticmethod
    def current_device() -> int:
        raise NotImplementedError

    @staticmethod
    def set_device(device: torch.types.Device) -> None:
        raise NotImplementedError

    @staticmethod
    def maybe_exchange_device(device: int) -> int:
        raise NotImplementedError

    @staticmethod
    def exchange_device(device: int) -> int:
        raise NotImplementedError

    @staticmethod
    def device_count() -> int:
        raise NotImplementedError

    @staticmethod
    def is_available() -> bool:
        raise NotImplementedError

    @staticmethod
    def stream(stream: torch.Stream) -> Any:
        raise NotImplementedError

    @staticmethod
    def current_stream() -> torch.Stream:
        raise NotImplementedError

    @staticmethod
    def set_stream(stream: torch.Stream) -> None:
        raise NotImplementedError

    @staticmethod
    def _set_stream_by_id(stream_id: int, device_index: int, device_type: int) -> None:
        raise NotImplementedError

    @staticmethod
    def get_raw_stream(device_idx: int) -> int:
        raise NotImplementedError

    @staticmethod
    def synchronize(device: torch.types.Device = None) -> None:
        raise NotImplementedError

    @classmethod
    def get_device_properties(cls, device: torch.types.Device = None) -> Any:
        return cls.Worker.get_device_properties(device)

    @staticmethod
    def get_compute_capability(device: torch.types.Device = None) -> Any:
        raise NotImplementedError

    @staticmethod
    def is_bf16_supported(including_emulation: bool = False) -> bool:
        raise NotImplementedError

    @classmethod
    def is_dtype_supported(
        cls, dtype: torch.dtype, including_emulation: bool = False
    ) -> bool:
        return dtype != torch.bfloat16 or cls.is_bf16_supported(including_emulation)

    @staticmethod
    def memory_allocated(device: torch.types.Device = None) -> int:
        raise NotImplementedError

    @staticmethod
    def is_triton_capable(device: torch.types.Device = None) -> bool:
        """
        Returns True if the device has Triton support, False otherwise, even if
        the appropriate Triton backend is not available.
        """
        return False

    @classmethod
    def raise_if_triton_unavailable(cls, device: torch.types.Device = None) -> None:
        """
        Raises a `TritonUnavailableError` with human-readable instructions if
        the Triton backend for the given device (or the default device if
        `device` is `None`) is not built. Implementations may raise a
        different, more specific error when the device itself is not
        Triton-capable (e.g. CUDA raises `GPUTooOldForTriton`), so callers
        that only want an availability answer should check
        `is_triton_capable()` first.

        The caller should ensure the presence of the 'triton' package before
        calling this method.
        """
        from torch._dynamo.exc import TritonUnavailableError

        if not cls.is_triton_capable():
            raise TritonUnavailableError(
                "This device is not capable of supporting Triton"
            )


class DeviceGuard:
    """
    This class provides a context manager for device switching. This is a stripped
    down version of torch.{device_name}.device.

    The context manager changes the current device to the given device index
    on entering the context and restores the original device on exiting.
    The device is switched using the provided device interface.
    """

    def __init__(
        self, device_interface: type[DeviceInterface], index: int | None
    ) -> None:
        self.device_interface = device_interface
        self.idx = index
        self.prev_idx = -1

    def __enter__(self) -> None:
        if self.idx is not None:
            self.prev_idx = self.device_interface.exchange_device(self.idx)

    def __exit__(self, type: Any, value: Any, traceback: Any) -> Literal[False]:
        if self.idx is not None:
            self.idx = self.device_interface.maybe_exchange_device(self.prev_idx)
        return False


class CudaInterface(DeviceInterface):
    device = torch.cuda.device  # type: ignore[assignment]

    # register Event and Stream class into the backend interface
    # make sure Event and Stream are implemented and inherited from the torch.Event and torch.Stream
    Event = torch.cuda.Event  # type: ignore[assignment]
    Stream = torch.cuda.Stream  # type: ignore[assignment]

    # pyrefly: ignore [bad-override]
    class Worker:
        @staticmethod
        def set_device(device: int) -> None:
            caching_worker_current_devices["cuda"] = device

        @staticmethod
        def current_device() -> int:
            if "cuda" in caching_worker_current_devices:
                return caching_worker_current_devices["cuda"]
            return torch.cuda.current_device()

        @staticmethod
        def get_device_properties(device: torch.types.Device = None) -> Any:
            if device is not None:
                if isinstance(device, str):
                    device = torch.device(device)
                    if device.type != "cuda":
                        raise AssertionError(
                            f"Expected device type 'cuda', got '{device.type}'"
                        )
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
    get_raw_stream = staticmethod(get_cuda_stream)  # type: ignore[assignment, arg-type]
    exchange_device = staticmethod(torch.cuda._exchange_device)  # type: ignore[arg-type, has-type]
    maybe_exchange_device = staticmethod(torch.cuda._maybe_exchange_device)  # type: ignore[arg-type, has-type]
    memory_allocated = staticmethod(torch.cuda.memory_allocated)
    is_bf16_supported = staticmethod(torch.cuda.is_bf16_supported)  # type: ignore[arg-type]

    # Can be mock patched by @patch decorator.
    @staticmethod
    def is_available() -> bool:
        return torch.cuda.is_available()

    @staticmethod
    def get_compute_capability(device: torch.types.Device = None) -> int | str:
        if torch.version.hip is None:
            major, min = torch.cuda.get_device_capability(device)
            return major * 10 + min
        else:
            return torch.cuda.get_device_properties(device).gcnArchName.split(":", 1)[0]

    @staticmethod
    def is_triton_capable(device: torch.types.Device = None) -> bool:
        # Use the Worker API (device properties cached in the main process
        # before fork) instead of torch.cuda.get_device_properties directly, so
        # the capability check stays safe when called from spawn-based compile
        # workers.
        return (
            torch.version.hip is not None
            or CudaInterface.Worker.get_device_properties(device).major >= 7
        )

    @staticmethod
    def raise_if_triton_unavailable(device: torch.types.Device = None) -> None:
        from torch._dynamo.exc import TritonUnavailableError
        from torch._inductor.exc import GPUTooOldForTriton

        if not CudaInterface.is_triton_capable(device):
            device_props = torch.cuda.get_device_properties(device)
            raise GPUTooOldForTriton(device_props, inspect.currentframe())

        import triton.backends

        if torch.version.hip is not None:
            if "amd" not in triton.backends.backends:
                raise TritonUnavailableError("triton not built with the 'amd' backend")
        elif "nvidia" not in triton.backends.backends:
            raise TritonUnavailableError("triton not built with the 'nvidia' backend")


get_mtia_stream: Callable[[int], int] | None
if torch.mtia._is_compiled():
    from torch._C import _mtia_getCurrentRawStream as get_mtia_stream
else:
    get_mtia_stream = None


class MtiaInterface(DeviceInterface):
    device = torch.mtia.device  # type: ignore[assignment]
    Event = torch.mtia.Event  # type: ignore[assignment]
    Stream = torch.mtia.Stream  # type: ignore[assignment]

    # pyrefly: ignore [bad-override]
    class Worker:
        @staticmethod
        def set_device(device: int) -> None:
            caching_worker_current_devices["mtia"] = device

        @staticmethod
        def current_device() -> int:
            if "mtia" in caching_worker_current_devices:
                return caching_worker_current_devices["mtia"]
            return torch.mtia.current_device()

        @staticmethod
        def get_device_properties(device: torch.types.Device = None) -> Any:
            if device is not None:
                if isinstance(device, str):
                    device = torch.device(device)
                    if device.type != "mtia":
                        raise AssertionError(
                            f"Expected device type 'mtia', got '{device.type}'"
                        )
                if isinstance(device, torch.device):
                    device = device.index
            if device is None:
                device = MtiaInterface.Worker.current_device()

            if "mtia" not in caching_worker_device_properties:
                device_prop = [
                    torch.mtia.get_device_properties(i)
                    for i in range(torch.mtia.device_count())
                ]
                caching_worker_device_properties["mtia"] = device_prop

            return caching_worker_device_properties["mtia"][device]

    current_device = staticmethod(torch.mtia.current_device)
    set_device = staticmethod(torch.mtia.set_device)  # type: ignore[assignment]
    device_count = staticmethod(torch.mtia.device_count)
    stream = staticmethod(torch.mtia.stream)  # type: ignore[assignment]
    current_stream = staticmethod(torch.mtia.current_stream)
    set_stream = staticmethod(torch.mtia.set_stream)  # type: ignore[assignment]
    _set_stream_by_id = staticmethod(torch.mtia._set_stream_by_id)  # type: ignore[assignment]
    synchronize = staticmethod(torch.mtia.synchronize)
    get_device_properties = staticmethod(torch.mtia.get_device_properties)  # type: ignore[assignment]
    get_raw_stream = staticmethod(get_mtia_stream)  # type: ignore[assignment, arg-type]
    exchange_device = staticmethod(torch.mtia._exchange_device)  # type: ignore[arg-type, has-type]
    maybe_exchange_device = staticmethod(torch.mtia._maybe_exchange_device)  # type: ignore[arg-type, has-type]
    memory_allocated = staticmethod(torch.mtia.memory_allocated)  # type: ignore[assignment]
    is_bf16_supported = staticmethod(torch.mtia.is_bf16_supported)  # type: ignore[arg-type]

    # Can be mock patched by @patch decorator.
    @staticmethod
    def is_available() -> bool:
        ret = torch.mtia.is_available()
        return ret

    @staticmethod
    def get_compute_capability(device: torch.types.Device = None) -> Any:
        cc = torch.mtia.get_device_capability(device)
        return cc

    @staticmethod
    def is_triton_capable(device: torch.types.Device = None) -> bool:
        return True

    @staticmethod
    def raise_if_triton_unavailable(device: torch.types.Device = None) -> None:
        import triton.backends

        from torch._dynamo.exc import TritonUnavailableError

        if "mtia" not in triton.backends.backends:
            raise TritonUnavailableError("triton not built with the 'mtia' backend")


get_xpu_stream: Callable[[int], int] | None
if torch.xpu._is_compiled():
    from torch._C import _xpu_getCurrentRawStream as get_xpu_stream
else:
    get_xpu_stream = None


class XpuInterface(DeviceInterface):
    device = torch.xpu.device  # type: ignore[assignment]
    Event = torch.xpu.Event  # type: ignore[assignment]
    Stream = torch.xpu.Stream  # type: ignore[assignment]

    # pyrefly: ignore [bad-override]
    class Worker:
        @staticmethod
        def set_device(device: int) -> None:
            caching_worker_current_devices["xpu"] = device

        @staticmethod
        def current_device() -> int:
            if "xpu" in caching_worker_current_devices:
                return caching_worker_current_devices["xpu"]
            return torch.xpu.current_device()

        @staticmethod
        def get_device_properties(device: torch.types.Device = None) -> Any:
            if device is not None:
                if isinstance(device, str):
                    device = torch.device(device)
                    if device.type != "xpu":
                        raise AssertionError(
                            f"Expected device type 'xpu', got '{device.type}'"
                        )
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
    device_count = staticmethod(torch.xpu.device_count)  # type: ignore[has-type]
    stream = staticmethod(torch.xpu.stream)  # type: ignore[assignment]
    current_stream = staticmethod(torch.xpu.current_stream)
    set_stream = staticmethod(torch.xpu.set_stream)  # type: ignore[assignment]
    _set_stream_by_id = staticmethod(torch.xpu._set_stream_by_id)  # type: ignore[assignment]
    synchronize = staticmethod(torch.xpu.synchronize)
    get_device_properties = staticmethod(torch.xpu.get_device_properties)  # type: ignore[assignment]
    get_raw_stream = staticmethod(get_xpu_stream)  # type: ignore[assignment, arg-type]
    exchange_device = staticmethod(torch.xpu._exchange_device)  # type: ignore[arg-type, has-type]
    maybe_exchange_device = staticmethod(torch.xpu._maybe_exchange_device)  # type: ignore[arg-type, has-type]
    memory_allocated = staticmethod(torch.xpu.memory_allocated)

    # Can be mock patched by @patch decorator.
    @staticmethod
    def is_available() -> bool:
        return torch.xpu.is_available()

    @staticmethod
    def get_compute_capability(device: torch.types.Device = None) -> Any:
        cc = torch.xpu.get_device_capability(device)
        return cc

    @staticmethod
    def is_bf16_supported(including_emulation: bool = False) -> bool:
        return torch.xpu.is_bf16_supported()

    @staticmethod
    def is_triton_capable(device: torch.types.Device = None) -> bool:
        return True

    @staticmethod
    def raise_if_triton_unavailable(device: torch.types.Device = None) -> None:
        import triton.backends

        from torch._dynamo.exc import TritonUnavailableError

        if "intel" not in triton.backends.backends:
            raise TritonUnavailableError("triton not built with the 'intel' backend")


@dataclass
class CpuDeviceProperties:
    multi_processor_count: int


class CpuInterface(DeviceInterface):
    # pyrefly: ignore [bad-override]
    class Event(torch.Event):
        def __init__(self, enable_timing: bool = True) -> None:
            self.time = 0.0

        def elapsed_time(self, other: Any) -> float:
            return (other.time - self.time) * 1000

        def record(self, stream: Any = None) -> None:
            self.time = time.perf_counter()

    # pyrefly: ignore [bad-override]
    class Worker:
        @staticmethod
        def get_device_properties(
            device: torch.types.Device = None,
        ) -> CpuDeviceProperties:
            import multiprocessing

            cpu_count = multiprocessing.cpu_count()
            return CpuDeviceProperties(cpu_count)

    @staticmethod
    def is_available() -> bool:
        return True

    @staticmethod
    def is_bf16_supported(including_emulation: bool = False) -> bool:
        return True

    @staticmethod
    def get_compute_capability(device: torch.types.Device = None) -> str:
        return ""

    @staticmethod
    def get_raw_stream(device_idx: Any) -> int:
        return 0

    @staticmethod
    def current_device() -> int:
        return 0

    @staticmethod
    def synchronize(device: torch.types.Device = None) -> None:
        pass

    @staticmethod
    def is_triton_capable(device: torch.types.Device = None) -> bool:
        return True

    @staticmethod
    def raise_if_triton_unavailable(device: torch.types.Device = None) -> None:
        import triton.backends

        from torch._dynamo.exc import TritonUnavailableError

        if "cpu" not in triton.backends.backends:
            raise TritonUnavailableError("triton not built with the 'cpu' backend")


class MpsInterface(DeviceInterface):
    @staticmethod
    def is_bf16_supported(including_emulation: bool = False) -> bool:
        return True

    @classmethod
    def is_dtype_supported(
        cls, dtype: torch.dtype, including_emulation: bool = False
    ) -> bool:
        if dtype in [torch.float64, torch.complex128]:
            return False
        return True

    @staticmethod
    def is_available() -> bool:
        return torch.backends.mps.is_available()

    @staticmethod
    def current_device() -> int:
        return 0

    @staticmethod
    def get_compute_capability(device: torch.types.Device = None) -> str:
        return ""

    @staticmethod
    def synchronize(device: torch.types.Device = None) -> None:
        torch.mps.synchronize()

    # pyrefly: ignore [bad-override]
    class Worker:
        @staticmethod
        def get_device_properties(device: torch.types.Device = None) -> Any:
            return namedtuple("MPSProperties", ["multi_processor_count"])(
                torch.backends.mps.get_core_count()  # type: ignore[arg-type]
            )

        @staticmethod
        def current_device() -> int:
            return 0


class TpuInterface(DeviceInterface):
    @staticmethod
    def is_bf16_supported(including_emulation: bool = False) -> bool:
        return True

    @classmethod
    def is_dtype_supported(
        cls, dtype: torch.dtype, including_emulation: bool = False
    ) -> bool:
        return dtype not in (
            torch.float64,
            torch.complex32,
            torch.complex64,
            torch.complex128,
            torch.half,
        )

    @staticmethod
    def is_available() -> bool:
        return has_torch_tpu()

    @staticmethod
    def current_device() -> int:
        return 0

    @staticmethod
    def get_compute_capability(device: torch.types.Device = None) -> str:
        return ""

    # pyrefly: ignore [bad-override]
    class Worker:
        @staticmethod
        def get_device_properties(device: torch.types.Device = None) -> Any:
            return namedtuple("TPUProperties", ["multi_processor_count"])(
                1  # type: ignore[arg-type]
            )

        @staticmethod
        def current_device() -> int:
            return 0


device_interfaces: dict[str, type[DeviceInterface]] = {}
_device_initialized = False


def _iter_registered_autocast_classes() -> Iterator[tuple[type, str]]:
    for device, device_interface in get_registered_device_interfaces():
        device_type = device.split(":")[0]
        for autocast_class in device_interface.autocast_classes:
            yield autocast_class, device_type


def _autocast_class_location(autocast_class: type) -> tuple[str, str] | None:
    """A class identity that survives the class being imported twice.

    An out-of-tree backend usually installs itself as a ``torch`` submodule, so
    the same file is reachable under two dotted names, e.g.
    ``torch_npu.npu.amp.autocast_mode`` and ``torch.npu.amp.autocast_mode``.
    Importing both names loads two module objects, runs the class body twice and
    yields two class objects; which one traced code reaches depends on import
    order, so ``is`` against the registered one is not reliable.  The defining
    file plus the qualified name is the same for both.

    This is deliberately narrower than matching on the file alone: a *different*
    class defined in that same file has a different ``__qualname__`` and so
    still does not match.

    ``None`` means the class has no source file to key on (C extension, exec'd
    code), in which case only identity matching applies to it.
    """
    try:
        file = inspect.getfile(autocast_class)
    except (TypeError, OSError):
        return None
    return file, autocast_class.__qualname__


def _conflict(key: object, previous: str, device_type: str) -> RuntimeError:
    # Iteration order over device_interfaces is not something a caller
    # controls, so first-wins would make the device_type non-deterministic.
    return RuntimeError(
        f"Autocast class {key} is registered by both device_type "
        f"{previous!r} and {device_type!r}; each autocast class must belong "
        f"to exactly one device_type."
    )


@functools.cache
def get_device_autocast_classes() -> dict[type, str]:
    """Map every registered autocast class to the device_type providing it."""
    result: dict[type, str] = {}
    for autocast_class, device_type in _iter_registered_autocast_classes():
        previous = result.setdefault(autocast_class, device_type)
        if previous != device_type:
            raise _conflict(autocast_class, previous, device_type)
    return result


@functools.cache
def get_device_autocast_class_locations() -> dict[tuple[str, str], str]:
    """``get_device_autocast_classes`` keyed by ``_autocast_class_location``."""
    result: dict[tuple[str, str], str] = {}
    for autocast_class, device_type in _iter_registered_autocast_classes():
        location = _autocast_class_location(autocast_class)
        if location is None:
            continue
        previous = result.setdefault(location, device_type)
        if previous != device_type:
            raise _conflict(location, previous, device_type)
    return result


def device_type_for_autocast_class(autocast_class: Any) -> str | None:
    """Return the device_type that registered ``autocast_class``, else ``None``.

    A device opts in by listing the class in ``DeviceInterface.autocast_classes``;
    nothing it did not list can match.  Registered classes are recognised by
    identity, or failing that by ``_autocast_class_location`` so that a second
    import of the defining module still resolves.
    """
    if not isinstance(autocast_class, type):
        return None
    device_type = get_device_autocast_classes().get(autocast_class)
    if device_type is not None:
        return device_type
    location = _autocast_class_location(autocast_class)
    if location is None:
        return None
    return get_device_autocast_class_locations().get(location)


def _validate_autocast_classes(
    device: str, device_interface: type[DeviceInterface]
) -> None:
    base = torch.amp.autocast_mode.autocast
    for autocast_class in device_interface.autocast_classes:
        if not (isinstance(autocast_class, type) and issubclass(autocast_class, base)):
            raise TypeError(
                f"{device_interface.__name__}.autocast_classes entry "
                f"{autocast_class!r} registered for device {device!r} is not a "
                f"subclass of torch.amp.autocast_mode.autocast."
            )
        if autocast_class is base:
            raise ValueError(
                f"{device_interface.__name__}.autocast_classes registered for "
                f"device {device!r} must not contain "
                f"torch.amp.autocast_mode.autocast itself; list the "
                f"device-specific subclass instead."
            )


def register_interface_for_device(
    device: str | torch.device, device_interface: type[DeviceInterface]
) -> None:
    if isinstance(device, torch.device):
        device = device.type
    _validate_autocast_classes(device, device_interface)
    device_interfaces[device] = device_interface
    get_device_autocast_classes.cache_clear()
    get_device_autocast_class_locations.cache_clear()


def get_interface_for_device(device: str | torch.device) -> type[DeviceInterface]:
    if isinstance(device, torch.device):
        device = device.type
    if not _device_initialized:
        init_device_reg()
    if device in device_interfaces:
        return device_interfaces[device]
    raise NotImplementedError(f"No interface for device {device}")


def get_registered_device_interfaces() -> Iterable[tuple[str, type[DeviceInterface]]]:
    if not _device_initialized:
        init_device_reg()
    return device_interfaces.items()


def init_device_reg() -> None:
    global _device_initialized
    register_interface_for_device("cuda", CudaInterface)
    for i in range(torch.cuda.device_count()):
        register_interface_for_device(f"cuda:{i}", CudaInterface)

    register_interface_for_device("xpu", XpuInterface)
    for i in range(torch.xpu.device_count()):
        register_interface_for_device(f"xpu:{i}", XpuInterface)

    register_interface_for_device("mtia", MtiaInterface)
    for i in range(torch.mtia.device_count()):
        register_interface_for_device(f"mtia:{i}", MtiaInterface)

    register_interface_for_device("cpu", CpuInterface)
    register_interface_for_device("mps", MpsInterface)
    register_interface_for_device("tpu", TpuInterface)

    _device_initialized = True
