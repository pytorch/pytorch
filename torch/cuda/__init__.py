# mypy: allow-untyped-defs
r"""
This package adds support for CUDA tensor types.

It implements the same function as CPU tensors, but they utilize
GPUs for computation.

It is lazily initialized, so you can always import it, and use
:func:`is_available()` to determine if your system supports CUDA.

:ref:`cuda-semantics` has more details about working with CUDA.
"""

import importlib
import os
import threading
import traceback
import warnings
from collections.abc import Callable
from functools import lru_cache
from typing import Any, cast, NewType, Optional, TYPE_CHECKING, Union

import torch
import torch._C
from torch._utils import _dummy_type, _LazySeedTracker, classproperty
from torch.types import Device

from . import _device_limits, gds
from ._utils import _get_device_index
from .graphs import (
    CUDAGraph,
    graph,
    graph_pool_handle,
    is_current_stream_capturing,
    make_graphed_callables,
)
from .green_contexts import GreenContext
from .streams import Event, ExternalStream, Stream


try:
    from torch._C import _cudart  # type: ignore[attr-defined]
except ImportError:
    _cudart = None

_initialized = False
_tls = threading.local()
_initialization_lock = threading.Lock()
_queued_calls: list[
    tuple[Callable[[], None], list[str]]
] = []  # don't invoke these until initialization occurs
_is_in_bad_fork = getattr(torch._C, "_cuda_isInBadFork", lambda: False)

_HAS_PYNVML = False
_PYNVML_ERR = None
try:
    from torch import version as _version

    try:
        if not _version.hip:
            import pynvml  # type: ignore[import]
        else:
            import ctypes
            from pathlib import Path

            # In ROCm (at least up through 6.3.2) there're 2 copies of libamd_smi.so:
            # - One at lib/libamd_smi.so
            # - One at share/amd_smi/amdsmi/libamd_smi.so
            #
            # The amdsmi python module hardcodes loading the second one in share-
            # https://github.com/ROCm/amdsmi/blob/1d305dc9708e87080f64f668402887794cd46584/py-interface/amdsmi_wrapper.py#L174
            #
            # See also https://github.com/ROCm/amdsmi/issues/72.
            #
            # This creates an ODR violation if the copy of libamd_smi.so from lib
            # is also loaded (via `ld` linking, `LD_LIBRARY_PATH` or `rpath`).
            #
            # In order to avoid the violation we hook CDLL and try using the
            # already loaded version of amdsmi, or any version in the processes
            # rpath/LD_LIBRARY_PATH first, so that we only load a single copy
            # of the .so.
            class _amdsmi_cdll_hook:
                def __init__(self) -> None:
                    self.original_CDLL = ctypes.CDLL  # type: ignore[misc,assignment]
                    paths = ["libamd_smi.so"]
                    if rocm_home := os.getenv("ROCM_HOME", os.getenv("ROCM_PATH")):
                        paths = [os.path.join(rocm_home, "lib/libamd_smi.so")] + paths
                    self.paths: list[str] = paths

                def hooked_CDLL(
                    self, name: str | Path | None, *args: Any, **kwargs: Any
                ) -> ctypes.CDLL:
                    if name and Path(name).name == "libamd_smi.so":
                        for path in self.paths:
                            try:
                                return self.original_CDLL(path, *args, **kwargs)
                            except OSError:
                                pass
                    return self.original_CDLL(name, *args, **kwargs)  # type: ignore[arg-type]

                def __enter__(self) -> None:
                    ctypes.CDLL = self.hooked_CDLL  # type: ignore[misc,assignment]

                def __exit__(self, type: Any, value: Any, traceback: Any) -> None:
                    ctypes.CDLL = self.original_CDLL  # type: ignore[misc]

            with _amdsmi_cdll_hook():
                import amdsmi  # type: ignore[import]

        _HAS_PYNVML = True
    except ModuleNotFoundError:
        pass
    finally:
        del _version
except ImportError as err:
    _PYNVML_ERR = err  # sometimes a lib is installed but the import fails for some other reason, so we log the error for later

_lazy_seed_tracker = _LazySeedTracker()

# Define dummy _CudaDeviceProperties type if PyTorch was compiled without CUDA
if hasattr(torch._C, "_CudaDeviceProperties"):
    _CudaDeviceProperties = torch._C._CudaDeviceProperties
else:
    _CudaDeviceProperties = _dummy_type("_CudaDeviceProperties")  # type: ignore[assignment, misc]

if hasattr(torch._C, "_cuda_exchangeDevice"):
    _exchange_device = torch._C._cuda_exchangeDevice
else:

    def _exchange_device(device: int) -> int:
        if device < 0:
            return -1
        raise RuntimeError("PyTorch was compiled without CUDA support")


if hasattr(torch._C, "_cuda_maybeExchangeDevice"):
    _maybe_exchange_device = torch._C._cuda_maybeExchangeDevice
else:

    def _maybe_exchange_device(device: int) -> int:
        if device < 0:
            return -1
        raise RuntimeError("PyTorch was compiled without CUDA support")


has_half: bool = True
has_magma: bool = torch._C._has_magma

default_generators: tuple[torch._C.Generator] = ()  # type: ignore[assignment]


def _is_compiled() -> bool:
    r"""Return true if compile with CUDA support."""
    return hasattr(torch._C, "_cuda_getDeviceCount")


def _nvml_based_avail() -> bool:
    return os.getenv("PYTORCH_NVML_BASED_CUDA_CHECK") == "1"


def is_available() -> bool:
    r"""
    Return a bool indicating if CUDA is currently available.

    .. note:: This function will NOT poison fork if the environment variable
        ``PYTORCH_NVML_BASED_CUDA_CHECK=1`` is set. For more details, see
        :ref:`multiprocessing-poison-fork-note`.
    """
    if not _is_compiled():
        return False
    if _nvml_based_avail():
        # The user has set an env variable to request this availability check that attempts to avoid fork poisoning by
        # using NVML at the cost of a weaker CUDA availability assessment. Note that if NVML discovery/initialization
        # fails, this assessment falls back to the default CUDA Runtime API assessment (`cudaGetDeviceCount`)
        return device_count() > 0
    else:
        # The default availability inspection never throws and returns 0 if the driver is missing or can't
        # be initialized. This uses the CUDA Runtime API `cudaGetDeviceCount` which in turn initializes the CUDA Driver
        # API via `cuInit`
        return torch._C._cuda_getDeviceCount() > 0


def is_bf16_supported(including_emulation: bool = True):
    r"""Return a bool indicating if the current CUDA/ROCm device supports dtype bfloat16."""
    # Check for ROCm, if true return true, no ROCM_VERSION check required,
    # since it is supported on AMD GPU archs.
    if torch.version.hip:
        return True

    # If CUDA is not available, than it does not support bf16 either
    if not is_available():
        return False

    device = torch.cuda.current_device()

    # Check for CUDA version and device compute capability.
    # This is a fast way to check for it.
    cuda_version = torch.version.cuda
    if cuda_version is not None and torch.cuda.get_device_properties(device).major >= 8:
        return True

    if not including_emulation:
        return False

    # Finally try to create a bfloat16 device.
    return _check_bf16_tensor_supported(device)


@lru_cache(maxsize=16)
def _check_bf16_tensor_supported(device: Device):
    try:
        torch.tensor([1.0], dtype=torch.bfloat16, device=device)
        return True
    except Exception:
        return False


def is_tf32_supported() -> bool:
    r"""Return a bool indicating if the current CUDA/ROCm device supports dtype tf32."""
    if torch.version.hip:
        prop_name = torch.cuda.get_device_properties().gcnArchName
        archs = ("gfx94", "gfx95")
        for arch in archs:
            if arch in prop_name:
                return True
        return False

    # Otherwise, tf32 is supported on CUDA platforms that natively (i.e. no emulation)
    # support bfloat16.
    return is_bf16_supported(including_emulation=False)


def _sleep(cycles):
    torch._C._cuda_sleep(cycles)


def _extract_arch_version(arch_string: str) -> int:
    """Extracts the architecture string from a CUDA version"""
    base = arch_string.split("_", maxsplit=2)[1]
    base = base.removesuffix("a").removesuffix("f")
    return int(base)


class _CompatInterval:
    """
    Defines a range of compute capabilities starting at a given
    version and going up to the end of that major version. This
    also allows excluding specific versions from the range.
    """

    def __init__(self, start, exclude: Optional[set[int]] = None):
        self.major, self.minor = start // 10, start % 10
        self.exclude = set() if exclude is None else exclude

    def __contains__(self, x):
        if x in self.exclude:
            return False
        x_major, x_minor = x // 10, x % 10
        return x_major == self.major and x_minor >= self.minor

    def __str__(self):
        result = f">={self.major}.{self.minor},<{self.major + 1}.0"
        if len(self.exclude) > 0:
            exceptions = ", ".join(f"{x // 10}.{x % 10}" for x in self.exclude)
            result += f" except {{{exceptions}}}"
        return result


class _CompatSet:
    """
    A set of compute capabilities. It exists primarily to support custom
    printing logic and is otherwise equivalent to a plain python set().
    """

    def __init__(self, values: set[int]):
        self.values = values

    def __contains__(self, x):
        return x in self.values

    def __str__(self):
        return "{" + ", ".join(f"{v // 10}.{v % 10}" for v in self.values) + "}"


# (code SM)->(device SM required to execute the code)
#
# Developer Notes:
# - This dict should be kept up to date with keys corresponding
#   to SM versions that PyTorch can be built for. An out of date
#   mapping will lead to false warnings.
# - The keys in dict correspond to known sm versions but the values
#   are merely rules based on sm compatibility guarantees for NVIDIA
#   devices while accounting for incompatibility of iGPU and dGPU.
DEVICE_REQUIREMENT: dict[int, Union[_CompatSet, _CompatInterval]] = {
    50: _CompatInterval(start=50, exclude={53}),
    52: _CompatInterval(start=52, exclude={53}),
    53: _CompatSet({53}),
    60: _CompatInterval(start=60, exclude={62}),
    61: _CompatInterval(start=61, exclude={62}),
    62: _CompatSet({62}),
    70: _CompatInterval(start=70, exclude={72}),
    72: _CompatSet({72}),
    75: _CompatInterval(start=75),
    80: _CompatInterval(start=80, exclude={87}),
    86: _CompatInterval(start=86, exclude={87}),
    87: _CompatSet({87}),
    89: _CompatInterval(start=89),
    90: _CompatInterval(start=90),
    100: _CompatInterval(start=100, exclude={101}),
    101: _CompatSet({101, 110}),
    103: _CompatInterval(start=103),
    110: _CompatInterval(start=110),
    120: _CompatInterval(start=120),
    121: _CompatInterval(start=121),
}


# TORCH_CUDA_ARCH_LIST for PyTorch releases
PYTORCH_RELEASES_CODE_CC: dict[str, set[int]] = {
    "12.6": {50, 60, 70, 80, 86, 90},
    "12.8": {70, 80, 86, 90, 100, 120},
    "13.0": {75, 80, 86, 90, 100, 110, 120},
}


def _code_compatible_with_device(device_cc: int, code_cc: int):
    if code_cc not in DEVICE_REQUIREMENT:
        warnings.warn(
            f"PyTorch was compiled with an unknown compute capability {code_cc // 10}.{code_cc % 10}. "
            + " Please create an issue on Github if this is a valid compute capability.",
            stacklevel=2,
        )
        return device_cc in _CompatInterval(start=code_cc)
    return device_cc in DEVICE_REQUIREMENT[code_cc]


def _warn_unsupported_code(device_index: int, device_cc: int, code_ccs: list[int]):
    name = get_device_name(device_index)

    compatible_releases: list[str] = []
    for cuda, build_ccs in PYTORCH_RELEASES_CODE_CC.items():
        if any(_code_compatible_with_device(device_cc, cc) for cc in build_ccs):
            compatible_releases.append(cuda)

    lines = [
        f"Found GPU{device_index} {name} which is of compute capability (CC) {device_cc // 10}.{device_cc % 10}.",
        "The following list shows the CCs this version of PyTorch was built for and the hardware CCs it supports:",
    ] + [
        f"- {cc // 10}.{cc % 10} which supports hardware CC {DEVICE_REQUIREMENT[cc]}"
        for cc in code_ccs
    ]

    if len(compatible_releases) > 0:
        releases_str = ", ".join(compatible_releases)
        lines.append(
            "Please follow the instructions at https://pytorch.org/get-started/locally/ to "
            + f"install a PyTorch release that supports one of these CUDA versions: {releases_str}"
        )

    warnings.warn("\n".join(lines), stacklevel=2)


def _check_capability():
    if torch.version.cuda is None:  # on ROCm we don't want this check
        return

    code_ccs = [_extract_arch_version(cc) for cc in get_arch_list()]
    for d in range(device_count()):
        major, minor = get_device_capability(d)
        device_cc = 10 * major + minor
        if not any(
            _code_compatible_with_device(device_cc, code_cc) for code_cc in code_ccs
        ):
            _warn_unsupported_code(d, device_cc, code_ccs)


def _check_cubins():
    incompatible_device_warn = """
{} with CUDA capability sm_{} is not compatible with the current PyTorch installation.
The current PyTorch install supports CUDA capabilities {}.
If you want to use the {} GPU with PyTorch, please check the instructions at https://pytorch.org/get-started/locally/
"""
    if torch.version.cuda is None:  # on ROCm we don't want this check
        return
    arch_list = get_arch_list()
    if len(arch_list) == 0:
        return
    supported_sm = [_extract_arch_version(arch) for arch in arch_list if "sm_" in arch]
    for idx in range(device_count()):
        cap_major, cap_minor = get_device_capability(idx)
        # NVIDIA GPU compute architectures are backward compatible within major version
        supported = any(sm // 10 == cap_major for sm in supported_sm)
        if not supported:
            device_name = get_device_name(idx)
            capability = cap_major * 10 + cap_minor
            warnings.warn(
                incompatible_device_warn.format(
                    device_name, capability, " ".join(arch_list), device_name
                ),
                stacklevel=2,
            )


def is_initialized():
    r"""Return whether PyTorch's CUDA state has been initialized."""
    return _initialized and not _is_in_bad_fork()


def _lazy_call(callable, **kwargs):
    with _initialization_lock:
        if is_initialized():
            callable()
        else:
            # TODO(torch_deploy): this accesses linecache, which attempts to read the
            # file system to get traceback info. Patch linecache or do something
            # else here if this ends up being important.
            global _lazy_seed_tracker
            if kwargs.get("seed_all", False):
                _lazy_seed_tracker.queue_seed_all(callable, traceback.format_stack())
            elif kwargs.get("seed", False):
                _lazy_seed_tracker.queue_seed(callable, traceback.format_stack())
            else:
                # Don't store the actual traceback to avoid memory cycle
                _queued_calls.append((callable, traceback.format_stack()))


_lazy_call(_check_capability)
_lazy_call(_check_cubins)


class DeferredCudaCallError(Exception):
    pass


AcceleratorError = torch._C.AcceleratorError
OutOfMemoryError = torch._C.OutOfMemoryError


def init():
    r"""Initialize PyTorch's CUDA state.

    You may need to call this explicitly if you are interacting with
    PyTorch via its C API, as Python bindings for CUDA functionality
    will not be available until this initialization takes place.
    Ordinary users should not need this, as all of PyTorch's CUDA methods
    automatically initialize CUDA state on-demand.

    Does nothing if the CUDA state is already initialized.
    """
    _lazy_init()


def _lazy_init():
    global _initialized, _queued_calls
    if is_initialized() or hasattr(_tls, "is_initializing"):
        return
    with _initialization_lock:
        # We be double-checked locking, boys!  This is OK because
        # the above test was GIL protected anyway.  The inner test
        # is for when a thread blocked on some other thread which was
        # doing the initialization; when they get the lock, they will
        # find there is nothing left to do.
        if is_initialized():
            return
        # It is important to prevent other threads from entering _lazy_init
        # immediately, while we are still guaranteed to have the GIL, because some
        # of the C calls we make below will release the GIL
        if _is_in_bad_fork():
            raise RuntimeError(
                "Cannot re-initialize CUDA in forked subprocess. To use CUDA with "
                "multiprocessing, you must use the 'spawn' start method"
            )
        if not hasattr(torch._C, "_cuda_getDeviceCount"):
            raise AssertionError("Torch not compiled with CUDA enabled")
        if _cudart is None:
            raise AssertionError(
                "libcudart functions unavailable. It looks like you have a broken build?"
            )
        # This function throws if there's a driver initialization error, no GPUs
        # are found or any other error occurs
        torch._C._cuda_init()
        # Some of the queued calls may reentrantly call _lazy_init();
        # we need to just return without initializing in that case.
        # However, we must not let any *other* threads in!
        _tls.is_initializing = True

        _queued_calls.extend(calls for calls in _lazy_seed_tracker.get_calls() if calls)

        try:
            for queued_call, orig_traceback in _queued_calls:
                try:
                    queued_call()
                except Exception as e:
                    msg = (
                        f"CUDA call failed lazily at initialization with error: {str(e)}\n\n"
                        f"CUDA call was originally invoked at:\n\n{''.join(orig_traceback)}"
                    )
                    raise DeferredCudaCallError(msg) from e
        finally:
            delattr(_tls, "is_initializing")
        _initialized = True


def cudart():
    r"""Retrieves the CUDA runtime API module.


    This function initializes the CUDA runtime environment if it is not already
    initialized and returns the CUDA runtime API module (_cudart). The CUDA
    runtime API module provides access to various CUDA runtime functions.

    Args:
        ``None``

    Returns:
        module: The CUDA runtime API module (_cudart).

    Raises:
        RuntimeError: If CUDA cannot be re-initialized in a forked subprocess.
        AssertionError: If PyTorch is not compiled with CUDA support or if libcudart functions are unavailable.

    Example of CUDA operations with profiling:
        >>> import torch
        >>> from torch.cuda import cudart, check_error
        >>> import os
        >>>
        >>> os.environ["CUDA_PROFILE"] = "1"
        >>>
        >>> def perform_cuda_operations_with_streams():
        >>>     stream = torch.cuda.Stream()
        >>>     with torch.cuda.stream(stream):
        >>>         x = torch.randn(100, 100, device='cuda')
        >>>         y = torch.randn(100, 100, device='cuda')
        >>>         z = torch.mul(x, y)
        >>>     return z
        >>>
        >>> torch.cuda.synchronize()
        >>> print("====== Start nsys profiling ======")
        >>> check_error(cudart().cudaProfilerStart())
        >>> with torch.autograd.profiler.emit_nvtx():
        >>>     result = perform_cuda_operations_with_streams()
        >>>     print("CUDA operations completed.")
        >>> check_error(torch.cuda.cudart().cudaProfilerStop())
        >>> print("====== End nsys profiling ======")

    To run this example and save the profiling information, execute:
        >>> $ nvprof --profile-from-start off --csv --print-summary -o trace_name.prof -f -- python cudart_test.py

    This command profiles the CUDA operations in the provided script and saves
    the profiling information to a file named `trace_name.prof`.
    The `--profile-from-start off` option ensures that profiling starts only
    after the `cudaProfilerStart` call in the script.
    The `--csv` and `--print-summary` options format the profiling output as a
    CSV file and print a summary, respectively.
    The `-o` option specifies the output file name, and the `-f` option forces the
    overwrite of the output file if it already exists.
    """
    _lazy_init()
    return _cudart


class cudaStatus:
    SUCCESS: int = 0
    ERROR_NOT_READY: int = 34


class CudaError(RuntimeError):
    def __init__(self, code: int) -> None:
        # pyrefly: ignore [missing-attribute]
        msg = _cudart.cudaGetErrorString(_cudart.cudaError(code))
        super().__init__(f"{msg} ({code})")


def check_error(res: int) -> None:
    r"""Raise an error if the result of a CUDA runtime API call is not success."""
    # pyrefly: ignore [missing-attribute]
    if res != _cudart.cudaError.success:
        raise CudaError(res)


class _DeviceGuard:
    def __init__(self, index: int):
        self.idx = index
        self.prev_idx = -1

    def __enter__(self):
        self.prev_idx = torch.cuda._exchange_device(self.idx)

    def __exit__(self, type: Any, value: Any, traceback: Any):
        self.idx = torch.cuda._maybe_exchange_device(self.prev_idx)
        return False


class device:
    r"""Context-manager that changes the selected device.

    Args:
        device (torch.device or int): device index to select. It's a no-op if
            this argument is a negative integer or ``None``.
    """

    def __init__(self, device: Any):
        self.idx = _get_device_index(device, optional=True)
        self.prev_idx = -1

    def __enter__(self):
        self.prev_idx = torch.cuda._exchange_device(self.idx)

    def __exit__(self, type: Any, value: Any, traceback: Any):
        self.idx = torch.cuda._maybe_exchange_device(self.prev_idx)
        return False


class device_of(device):
    r"""Context-manager that changes the current device to that of given object.

    You can use both tensors and storages as arguments. If a given object is
    not allocated on a GPU, this is a no-op.

    Args:
        obj (Tensor or Storage): object allocated on the selected device.
    """

    def __init__(self, obj):
        idx = obj.get_device() if obj.is_cuda else -1
        super().__init__(idx)


def set_device(device: Device) -> None:
    r"""Set the current device.

    Usage of this function is discouraged in favor of :any:`device`. In most
    cases it's better to use ``CUDA_VISIBLE_DEVICES`` environmental variable.

    Args:
        device (torch.device or int): selected device. This function is a no-op
            if this argument is negative.
    """
    device = _get_device_index(device)
    if device >= 0:
        torch._C._cuda_setDevice(device)


def get_device_name(device: Device = None) -> str:
    r"""Get the name of a device.

    Args:
        device (torch.device or int or str, optional): device for which to return the
            name. This function is a no-op if this argument is a negative
            integer. It uses the current device, given by :func:`~torch.cuda.current_device`,
            if :attr:`device` is ``None`` (default).

    Returns:
        str: the name of the device
    """
    return get_device_properties(device).name


def get_device_capability(device: Device = None) -> tuple[int, int]:
    r"""Get the cuda capability of a device.

    Args:
        device (torch.device or int or str, optional): device for which to return the
            device capability. This function is a no-op if this argument is
            a negative integer. It uses the current device, given by
            :func:`~torch.cuda.current_device`, if :attr:`device` is ``None``
            (default).

    Returns:
        tuple(int, int): the major and minor cuda capability of the device
    """
    prop = get_device_properties(device)
    return prop.major, prop.minor


# pyrefly: ignore [not-a-type]
def get_device_properties(device: Device = None) -> _CudaDeviceProperties:
    r"""Get the properties of a device.

    Args:
        device (torch.device or int or str, optional): device for which to return the
            properties of the device.  It uses the current device, given by
            :func:`~torch.cuda.current_device`, if :attr:`device` is ``None``
            (default).

    Returns:
        _CudaDeviceProperties: the properties of the device
    """
    _lazy_init()  # will define _get_device_properties
    device = _get_device_index(device, optional=True)
    if device < 0 or device >= device_count():
        raise AssertionError("Invalid device id")
    return _get_device_properties(device)  # type: ignore[name-defined]


def can_device_access_peer(device: Device, peer_device: Device) -> bool:
    r"""Check if peer access between two devices is possible."""
    _lazy_init()
    device = _get_device_index(device, optional=True)
    peer_device = _get_device_index(peer_device)
    if device < 0 or device >= device_count():
        raise AssertionError("Invalid device id")
    if peer_device < 0 or peer_device >= device_count():
        raise AssertionError("Invalid peer device id")
    return torch._C._cuda_canDeviceAccessPeer(device, peer_device)


class StreamContext:
    r"""Context-manager that selects a given stream.

    All CUDA kernels queued within its context will be enqueued on a selected
    stream.

    Args:
        Stream (Stream): selected stream. This manager is a no-op if it's
            ``None``.
    .. note:: Streams are per-device.
    """

    cur_stream: Optional["torch.cuda.Stream"]

    def __init__(self, stream: Optional["torch.cuda.Stream"]):
        self.stream = stream
        self.idx = _get_device_index(None, True)
        if not torch.jit.is_scripting():
            if self.idx is None:
                # pyrefly: ignore [bad-assignment]
                self.idx = -1

        self.src_prev_stream = (
            None if not torch.jit.is_scripting() else torch.cuda.default_stream(None)
        )
        self.dst_prev_stream = (
            None if not torch.jit.is_scripting() else torch.cuda.default_stream(None)
        )

    def __enter__(self):
        # Local cur_stream variable for type refinement
        cur_stream = self.stream
        # Return if stream is None or CUDA device not available
        if cur_stream is None or self.idx == -1:
            return
        self.src_prev_stream = torch.cuda.current_stream(None)

        # If the stream is not on the current device, then
        # set the current stream on the device
        if self.src_prev_stream.device != cur_stream.device:
            with device(cur_stream.device):
                self.dst_prev_stream = torch.cuda.current_stream(cur_stream.device)
        torch.cuda.set_stream(cur_stream)

    def __exit__(self, type: Any, value: Any, traceback: Any):
        # Local cur_stream variable for type refinement
        cur_stream = self.stream
        # If stream is None or no CUDA device available, return
        if cur_stream is None or self.idx == -1:
            return

        # Reset the stream on the original device
        # and destination device
        if self.src_prev_stream.device != cur_stream.device:  # type: ignore[union-attr]
            torch.cuda.set_stream(self.dst_prev_stream)  # type: ignore[arg-type]
        torch.cuda.set_stream(self.src_prev_stream)  # type: ignore[arg-type]


def stream(stream: Optional["torch.cuda.Stream"]) -> StreamContext:
    r"""Wrap around the Context-manager StreamContext that selects a given stream.

    Arguments:
        stream (Stream): selected stream. This manager is a no-op if it's
            ``None``.
    .. note::
        In eager mode stream is of type Stream class while in JIT it is
        an object of the custom class ``torch.classes.cuda.Stream``.
    """
    return StreamContext(stream)


def _set_stream_by_id(stream_id, device_index, device_type):
    r"""set stream specified by the stream id, device index and
        device type

    Args: stream_id (int): stream id in stream pool
          device_index (int): device index in topo
          device_type (int): enum device type
    """
    torch._C._cuda_setStream(
        stream_id=stream_id,
        device_index=device_index,
        device_type=device_type,
    )


def set_stream(stream: Stream):
    r"""Set the current stream. This is a wrapper API to set the stream.
        Usage of this function is discouraged in favor of the ``stream``
        context manager.

    Args:
        stream (Stream): selected stream. This function is a no-op
            if this argument is ``None``.
    """
    if stream is None:
        return
    _set_stream_by_id(
        stream_id=stream.stream_id,
        device_index=stream.device_index,
        device_type=stream.device_type,
    )


def _parse_visible_devices() -> list[int] | list[str]:
    r"""Parse CUDA_VISIBLE_DEVICES environment variable."""
    var = os.getenv("CUDA_VISIBLE_DEVICES")

    if torch.version.hip:
        hip_devices = os.getenv("HIP_VISIBLE_DEVICES")
        rocr_devices = os.getenv("ROCR_VISIBLE_DEVICES")

        # You must take care if both HIP and ROCR env vars are set as they have
        # different meanings. Both env vars accept either a list of ints or a
        # list of UUIDs. The ROCR env var is processed first which then reduces
        # the number of GPUs that HIP can select from.
        if rocr_devices is not None:
            rocr_count = len(rocr_devices.split(","))
            if hip_devices is not None:
                # sanity check if both env vars are set
                if len(hip_devices.split(",")) > rocr_count:
                    raise RuntimeError(
                        "HIP_VISIBLE_DEVICES contains more devices than ROCR_VISIBLE_DEVICES"
                    )
                # HIP_VISIBLE_DEVICES is preferred over ROCR_VISIBLE_DEVICES
                var = hip_devices
            else:
                return list(range(rocr_count))
        elif hip_devices is not None:
            var = hip_devices

    if var is None:
        return list(range(64))

    def _strtoul(s: str) -> int:
        """Return -1 or positive integer sequence string starts with."""
        if not s:
            return -1
        for idx, c in enumerate(s):
            if not (c.isdigit() or (idx == 0 and c in "+-")):
                break
            if idx + 1 == len(s):
                idx += 1
        return int(s[:idx]) if idx > 0 else -1

    def parse_list_with_prefix(lst: str, prefix: str) -> list[str]:
        rcs: list[str] = []
        for elem in lst.split(","):
            # Repeated id results in empty set
            if elem in rcs:
                return cast(list[str], [])
            # Anything other but prefix is ignored
            if not elem.startswith(prefix):
                break
            rcs.append(elem)
        return rcs

    if var.startswith("GPU-"):
        return parse_list_with_prefix(var, "GPU-")
    if var.startswith("MIG-"):
        return parse_list_with_prefix(var, "MIG-")
    # CUDA_VISIBLE_DEVICES uses something like strtoul
    # which makes `1gpu2,2ampere` is equivalent to `1,2`
    rc: list[int] = []
    for elem in var.split(","):
        x = _strtoul(elem.strip())
        # Repeated ordinal results in empty set
        if x in rc:
            return cast(list[int], [])
        # Negative value aborts the sequence
        if x < 0:
            break
        rc.append(x)
    return rc


def _raw_device_count_amdsmi() -> int:
    if not _HAS_PYNVML:  # If amdsmi is not available
        return -1
    try:
        amdsmi.amdsmi_init()
    except amdsmi.AmdSmiException as e:
        warnings.warn(
            f"Can't initialize amdsmi - Error code: {e.err_code}", stacklevel=2
        )
        return -1
    socket_handles = amdsmi.amdsmi_get_processor_handles()
    return len(socket_handles)


def _raw_device_count_nvml() -> int:
    r"""Return number of devices as reported by NVML or negative value if NVML discovery/initialization failed."""
    from ctypes import byref, c_int, CDLL

    nvml_h = CDLL("libnvidia-ml.so.1")
    rc = nvml_h.nvmlInit()
    if rc != 0:
        warnings.warn("Can't initialize NVML", stacklevel=2)
        return -1
    dev_count = c_int(-1)
    rc = nvml_h.nvmlDeviceGetCount_v2(byref(dev_count))
    if rc != 0:
        warnings.warn("Can't get nvml device count", stacklevel=2)
        return -1
    del nvml_h
    return dev_count.value


def _raw_device_uuid_amdsmi() -> list[str] | None:
    from ctypes import byref, c_int, c_void_p, CDLL, create_string_buffer

    if not _HAS_PYNVML:  # If amdsmi is not available
        return None
    try:
        amdsmi.amdsmi_init()
    except amdsmi.AmdSmiException:
        warnings.warn("Can't initialize amdsmi", stacklevel=2)
        return None
    try:
        socket_handles = amdsmi.amdsmi_get_processor_handles()
        dev_count = len(socket_handles)
    except amdsmi.AmdSmiException:
        warnings.warn("Can't get amdsmi device count", stacklevel=2)
        return None
    uuids: list[str] = []
    for idx in range(dev_count):
        try:
            handler = amdsmi.amdsmi_get_processor_handles()[idx]
        except amdsmi.AmdSmiException:
            warnings.warn("Cannot get amd device handler", stacklevel=2)
            return None
        try:
            uuid = amdsmi.amdsmi_get_gpu_asic_info(handler)["asic_serial"][
                2:
            ]  # Removes 0x prefix from serial
        except amdsmi.AmdSmiException:
            warnings.warn("Cannot get uuid for amd device", stacklevel=2)
            return None
        uuids.append(
            str(uuid).lower()
        )  # Lower-case to match expected HIP_VISIBLE_DEVICES uuid input
    return uuids


def _raw_device_uuid_nvml() -> list[str] | None:
    r"""Return list of device UUID as reported by NVML or None if NVM discovery/initialization failed."""
    from ctypes import byref, c_int, c_void_p, CDLL, create_string_buffer

    nvml_h = CDLL("libnvidia-ml.so.1")
    rc = nvml_h.nvmlInit()
    if rc != 0:
        warnings.warn("Can't initialize NVML", stacklevel=2)
        return None
    dev_count = c_int(-1)
    rc = nvml_h.nvmlDeviceGetCount_v2(byref(dev_count))
    if rc != 0:
        warnings.warn("Can't get nvml device count", stacklevel=2)
        return None
    uuids: list[str] = []
    for idx in range(dev_count.value):
        dev_id = c_void_p()
        rc = nvml_h.nvmlDeviceGetHandleByIndex_v2(idx, byref(dev_id))
        if rc != 0:
            warnings.warn("Can't get device handle", stacklevel=2)
            return None
        buf_len = 96
        buf = create_string_buffer(buf_len)
        rc = nvml_h.nvmlDeviceGetUUID(dev_id, buf, buf_len)
        if rc != 0:
            warnings.warn("Can't get device UUID", stacklevel=2)
            return None
        uuids.append(buf.raw.decode("ascii").strip("\0"))
    del nvml_h
    return uuids


def _transform_uuid_to_ordinals(candidates: list[str], uuids: list[str]) -> list[int]:
    r"""Given the set of partial uuids and list of known uuids builds a set of ordinals excluding ambiguous partials IDs."""

    def uuid_to_ordinal(candidate: str, uuids: list[str]) -> int:
        best_match = -1
        for idx, uuid in enumerate(uuids):
            if not uuid.startswith(candidate):
                continue
            # Ambiguous candidate
            if best_match != -1:
                return -1
            best_match = idx
        return best_match

    rc: list[int] = []
    for candidate in candidates:
        if torch.version.hip:
            candidate = candidate.replace(
                "GPU-", "", 1
            )  # Remove GPU-prefix to match amdsmi asic serial
        idx = uuid_to_ordinal(candidate, uuids)
        # First invalid ordinal stops parsing
        if idx < 0:
            break
        # Duplicates result in empty set
        if idx in rc:
            return cast(list[int], [])
        rc.append(idx)
    return rc


def _device_count_amdsmi() -> int:
    visible_devices = _parse_visible_devices()
    if not visible_devices:
        return 0
    try:
        if type(visible_devices[0]) is str:
            uuids = _raw_device_uuid_amdsmi()
            if uuids is None:
                return -1
            # Create string version of visible devices to avoid mypy warnings
            visible_device_str = cast(list[str], visible_devices)
            visible_devices = _transform_uuid_to_ordinals(visible_device_str, uuids)
        else:
            raw_cnt = _raw_device_count_amdsmi()
            if raw_cnt <= 0:
                return raw_cnt
            # Trim the list up to a maximum available device
            # pyrefly: ignore [bad-argument-type]
            for idx, val in enumerate(visible_devices):
                # pyrefly: ignore [redundant-cast]
                if cast(int, val) >= raw_cnt:
                    return idx
    except OSError:
        return -1
    except AttributeError:
        return -1
    return len(visible_devices)


def _device_count_nvml() -> int:
    r"""Return number of devices as reported by NVML taking CUDA_VISIBLE_DEVICES into account.

    Negative value is returned if NVML discovery or initialization has failed.
    """
    visible_devices = _parse_visible_devices()
    if not visible_devices:
        return 0
    try:
        if type(visible_devices[0]) is str:
            # Skip MIG parsing
            if visible_devices[0].startswith("MIG-"):
                return -1
            uuids = _raw_device_uuid_nvml()
            if uuids is None:
                return -1
            visible_devices = _transform_uuid_to_ordinals(
                cast(list[str], visible_devices), uuids
            )
        else:
            raw_cnt = _raw_device_count_nvml()
            if raw_cnt <= 0:
                return raw_cnt
            # Trim the list up to a maximum available device
            # pyrefly: ignore [bad-argument-type]
            for idx, val in enumerate(visible_devices):
                # pyrefly: ignore [redundant-cast]
                if cast(int, val) >= raw_cnt:
                    return idx
    except OSError:
        return -1
    except AttributeError:
        return -1
    return len(visible_devices)


def _get_nvml_device_index(device: Device) -> int:
    r"""Return the NVML index of the device, taking CUDA_VISIBLE_DEVICES into account."""
    idx = _get_device_index(device, optional=True)
    visible_devices = _parse_visible_devices()
    if type(visible_devices[0]) is str:
        uuids = _raw_device_uuid_nvml()
        if uuids is None:
            raise RuntimeError("Can't get device UUIDs")
        visible_devices = _transform_uuid_to_ordinals(
            cast(list[str], visible_devices), uuids
        )
    visible_devices = cast(list[int], visible_devices)
    if idx < 0 or idx >= len(visible_devices):
        raise RuntimeError(
            f"device {idx} is not visible (CUDA_VISIBLE_DEVICES={visible_devices})"
        )
    return visible_devices[idx]


_cached_device_count: int | None = None


def device_count() -> int:
    r"""
    Return the number of GPUs available.

    .. note:: This API will NOT poison fork if NVML discovery succeeds.
        See :ref:`multiprocessing-poison-fork-note` for more details.
    """
    global _cached_device_count
    if not _is_compiled():
        return 0
    if _cached_device_count is not None:
        return _cached_device_count
    # bypass _device_count_nvml() if rocm (not supported)
    nvml_count = _device_count_amdsmi() if torch.version.hip else _device_count_nvml()
    r = torch._C._cuda_getDeviceCount() if nvml_count < 0 else nvml_count
    # NB: Do not cache the device count prior to CUDA initialization, because
    # the number of devices can change due to changes to CUDA_VISIBLE_DEVICES
    # setting prior to CUDA initialization.
    if _initialized:
        _cached_device_count = r
    return r


def get_arch_list() -> list[str]:
    r"""Return list CUDA architectures this library was compiled for."""
    if not is_available():
        return []
    arch_flags = torch._C._cuda_getArchFlags()
    if arch_flags is None:
        return []
    return arch_flags.split()


def get_gencode_flags() -> str:
    r"""Return NVCC gencode flags this library was compiled with."""
    arch_list = get_arch_list()
    if len(arch_list) == 0:
        return ""
    arch_list_ = [arch.split("_") for arch in arch_list]
    return " ".join(
        [
            f"-gencode compute=compute_{arch},code={kind}_{arch}"
            for (kind, arch) in arch_list_
        ]
    )


def current_device() -> int:
    r"""Return the index of a currently selected device."""
    _lazy_init()
    return torch._C._cuda_getDevice()


def synchronize(device: Device = None) -> None:
    r"""Wait for all kernels in all streams on a CUDA device to complete.

    Args:
        device (torch.device or int, optional): device for which to synchronize.
            It uses the current device, given by :func:`~torch.cuda.current_device`,
            if :attr:`device` is ``None`` (default).
    """
    _lazy_init()
    with torch.cuda.device(device):
        return torch._C._cuda_synchronize()


def ipc_collect():
    r"""Force collects GPU memory after it has been released by CUDA IPC.

    .. note::
        Checks if any sent CUDA tensors could be cleaned from the memory. Force
        closes shared memory file used for reference counting if there is no
        active counters. Useful when the producer process stopped actively sending
        tensors and want to release unused memory.
    """
    _lazy_init()
    return torch._C._cuda_ipc_collect()


def current_stream(device: Device = None) -> Stream:
    r"""Return the currently selected :class:`Stream` for a given device.

    Args:
        device (torch.device or int, optional): selected device. Returns
            the currently selected :class:`Stream` for the current device, given
            by :func:`~torch.cuda.current_device`, if :attr:`device` is ``None``
            (default).
    """
    _lazy_init()
    streamdata = torch._C._cuda_getCurrentStream(
        _get_device_index(device, optional=True)
    )
    return Stream(
        stream_id=streamdata[0], device_index=streamdata[1], device_type=streamdata[2]
    )


def default_stream(device: Device = None) -> Stream:
    r"""Return the default :class:`Stream` for a given device.

    Args:
        device (torch.device or int, optional): selected device. Returns
            the default :class:`Stream` for the current device, given by
            :func:`~torch.cuda.current_device`, if :attr:`device` is ``None``
            (default).
    """
    _lazy_init()
    streamdata = torch._C._cuda_getDefaultStream(
        _get_device_index(device, optional=True)
    )
    return Stream(
        stream_id=streamdata[0], device_index=streamdata[1], device_type=streamdata[2]
    )


def get_stream_from_external(data_ptr: int, device: Device = None) -> Stream:
    r"""Return a :class:`Stream` from an externally allocated CUDA stream.

    This function is used to wrap streams allocated in other libraries in order
    to facilitate data exchange and multi-library interactions.

    .. note:: This function doesn't manage the stream life-cycle, it is the user
       responsibility to keep the referenced stream alive while this returned
       stream is being used.

    Args:
        data_ptr(int): Integer representation of the `cudaStream_t` value that
            is allocated externally.
        device(torch.device or int, optional): the device where the stream
            was originally allocated. If device is specified incorrectly,
            subsequent launches using this stream may fail.
    """
    _lazy_init()
    streamdata = torch._C._cuda_getStreamFromExternal(
        data_ptr, _get_device_index(device, optional=True)
    )
    return Stream(
        stream_id=streamdata[0], device_index=streamdata[1], device_type=streamdata[2]
    )


def current_blas_handle():
    r"""Return cublasHandle_t pointer to current cuBLAS handle"""
    _lazy_init()
    return torch._C._cuda_getCurrentBlasHandle()


def set_sync_debug_mode(debug_mode: int | str) -> None:
    r"""Set the debug mode for cuda synchronizing operations.

    Args:
        debug_mode(str or int): if "default" or 0, don't error or warn on synchronizing operations,
            if "warn" or 1, warn on synchronizing operations, if "error" or 2, error out synchronizing operations.

    Warning:
        This is an experimental feature, and not all synchronizing operations will trigger warning or error. In
        particular, operations in torch.distributed and torch.sparse namespaces are not covered yet.
    """
    _lazy_init()
    if isinstance(debug_mode, str):
        if debug_mode == "default":
            debug_mode = 0
        elif debug_mode == "warn":
            debug_mode = 1
        elif debug_mode == "error":
            debug_mode = 2
        else:
            raise RuntimeError(
                "invalid value of debug_mode, expected one of `default`, `warn`, `error`"
            )

    torch._C._cuda_set_sync_debug_mode(debug_mode)


def get_sync_debug_mode() -> int:
    r"""Return current value of debug mode for cuda synchronizing operations."""
    _lazy_init()
    return torch._C._cuda_get_sync_debug_mode()


def _get_pynvml_handler(device: Device = None):
    if not _HAS_PYNVML:
        raise ModuleNotFoundError(
            "nvidia-ml-py does not seem to be installed or it can't be imported."
            # pyrefly: ignore [invalid-inheritance]
        ) from _PYNVML_ERR
    # pyrefly: ignore [import-error, missing-import, missing-module-attribute]
    from pynvml import NVMLError_DriverNotLoaded

    try:
        pynvml.nvmlInit()
    except NVMLError_DriverNotLoaded as e:
        raise RuntimeError("cuda driver can't be loaded, is cuda enabled?") from e

    device = _get_nvml_device_index(device)
    handle = pynvml.nvmlDeviceGetHandleByIndex(device)
    return handle


def _get_amdsmi_handler(device: Device = None):
    if not _HAS_PYNVML:
        raise ModuleNotFoundError(
            "amdsmi does not seem to be installed or it can't be imported."
            # pyrefly: ignore [invalid-inheritance]
        ) from _PYNVML_ERR
    try:
        amdsmi.amdsmi_init()
    except amdsmi.AmdSmiException as e:
        raise RuntimeError(
            "amdsmi driver can't be loaded, requires >=ROCm6.0 installation"
        ) from e
    device = _get_amdsmi_device_index(device)
    handle = amdsmi.amdsmi_get_processor_handles()[device]
    return handle


def _get_amdsmi_device_index(device: Device) -> int:
    r"""Return the amdsmi index of the device, taking visible_devices into account."""
    idx = _get_device_index(device, optional=True)
    visible_devices = _parse_visible_devices()
    if type(visible_devices[0]) is str:
        uuids = _raw_device_uuid_amdsmi()
        if uuids is None:
            raise RuntimeError("Can't get device UUIDs")
        visible_devices_str = cast(
            list[str], visible_devices
        )  # Create str variable for mypy
        visible_devices = _transform_uuid_to_ordinals(visible_devices_str, uuids)
    idx_map = dict(enumerate(cast(list[int], visible_devices)))
    if idx not in idx_map:
        raise RuntimeError(
            f"device {idx} is not visible (HIP_VISIBLE_DEVICES={visible_devices})"
        )
    return idx_map[idx]


def _get_amdsmi_device_memory_used(device: Device = None) -> int:
    handle = _get_amdsmi_handler(device)
    # amdsmi_get_gpu_vram_usage returns mem usage in megabytes
    mem_mega_bytes = amdsmi.amdsmi_get_gpu_vram_usage(handle)["vram_used"]
    mem_bytes = mem_mega_bytes * 1024 * 1024
    return mem_bytes


def _get_amdsmi_memory_usage(device: Device = None) -> int:
    handle = _get_amdsmi_handler(device)
    return amdsmi.amdsmi_get_gpu_activity(handle)["umc_activity"]


def _get_amdsmi_utilization(device: Device = None) -> int:
    handle = _get_amdsmi_handler(device)
    return amdsmi.amdsmi_get_gpu_activity(handle)["gfx_activity"]


def _get_amdsmi_temperature(device: Device = None) -> int:
    handle = _get_amdsmi_handler(device)
    return amdsmi.amdsmi_get_temp_metric(
        handle,
        amdsmi.AmdSmiTemperatureType.JUNCTION,
        amdsmi.AmdSmiTemperatureMetric.CURRENT,
    )


def _get_amdsmi_power_draw(device: Device = None) -> int:
    handle = _get_amdsmi_handler(device)
    socket_power = amdsmi.amdsmi_get_power_info(handle)["average_socket_power"]
    if socket_power != "N/A":
        return socket_power
    else:
        socket_power = amdsmi.amdsmi_get_power_info(handle)["current_socket_power"]
        if socket_power != "N/A":
            return socket_power
        else:
            return 0


def _get_amdsmi_clock_rate(device: Device = None) -> int:
    handle = _get_amdsmi_handler(device)
    clock_info = amdsmi.amdsmi_get_clock_info(handle, amdsmi.AmdSmiClkType.GFX)
    if "cur_clk" in clock_info:  # ROCm 6.2 deprecation
        clock_rate = clock_info["cur_clk"]
    else:
        clock_rate = clock_info["clk"]
    if clock_rate != "N/A":
        return clock_rate
    else:
        return 0


def device_memory_used(device: Device = None) -> int:
    r"""Return used global (device) memory in bytes as given by `nvidia-smi` or `amd-smi`.

    Args:
        device (torch.device or int, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch.cuda.current_device`,
            if :attr:`device` is ``None`` (default).

    """
    if not torch.version.hip:
        handle = _get_pynvml_handler()
        device = _get_nvml_device_index(device)
        handle = pynvml.nvmlDeviceGetHandleByIndex(device)
        return pynvml.nvmlDeviceGetMemoryInfo(handle).used
    else:
        return _get_amdsmi_device_memory_used(device)


def memory_usage(device: Device = None) -> int:
    r"""Return the percent of time over the past sample period during which global (device)
    memory was being read or written as given by `nvidia-smi`.

    Args:
        device (torch.device or int, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch.cuda.current_device`,
            if :attr:`device` is ``None`` (default).

    Warning: Each sample period may be between 1 second and 1/6 second,
    depending on the product being queried.
    """
    if not torch.version.hip:
        handle = _get_pynvml_handler()
        device = _get_nvml_device_index(device)
        handle = pynvml.nvmlDeviceGetHandleByIndex(device)
        return pynvml.nvmlDeviceGetUtilizationRates(handle).memory
    else:
        return _get_amdsmi_memory_usage(device)


def utilization(device: Device = None) -> int:
    r"""Return the percent of time over the past sample period during which one or
    more kernels was executing on the GPU as given by `nvidia-smi`.

    Args:
        device (torch.device or int, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch.cuda.current_device`,
            if :attr:`device` is ``None`` (default).

    Warning: Each sample period may be between 1 second and 1/6 second,
    depending on the product being queried.
    """
    if not torch.version.hip:
        handle = _get_pynvml_handler(device)
        device = _get_nvml_device_index(device)
        handle = pynvml.nvmlDeviceGetHandleByIndex(device)
        return pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
    else:
        return _get_amdsmi_utilization(device)


def temperature(device: Device = None) -> int:
    r"""Return the average temperature of the GPU sensor in Degrees C (Centigrades).

    The average temperature is computed based on past sample period as given by `nvidia-smi`.

    Args:
        device (torch.device or int, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch.cuda.current_device`,
            if :attr:`device` is ``None`` (default).

    Warning: Each sample period may be between 1 second and 1/6 second,
    depending on the product being queried.
    """
    if not torch.version.hip:
        handle = _get_pynvml_handler(device)
        # 0 refers to the temperature sensor for the GPU die.
        return pynvml.nvmlDeviceGetTemperature(handle, 0)
    else:
        return _get_amdsmi_temperature(device)


def power_draw(device: Device = None) -> int:
    r"""Return the average power draw of the GPU sensor in mW (MilliWatts)
        over the past sample period as given by `nvidia-smi` for Fermi or newer fully supported devices.

    Args:
        device (torch.device or int, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch.cuda.current_device`,
            if :attr:`device` is ``None`` (default).

    Warning: Each sample period may be between 1 second and 1/6 second,
    depending on the product being queried.
    """
    if not torch.version.hip:
        handle = _get_pynvml_handler(device)
        return pynvml.nvmlDeviceGetPowerUsage(handle)
    else:
        return _get_amdsmi_power_draw(device)


def clock_rate(device: Device = None) -> int:
    r"""Return the clock speed of the GPU SM in MHz (megahertz) over the past sample period as given by `nvidia-smi`.

    Args:
        device (torch.device or int, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch.cuda.current_device`,
            if :attr:`device` is ``None`` (default).

    Warning: Each sample period may be between 1 second and 1/6 second,
    depending on the product being queried.
    """
    if not torch.version.hip:
        handle = _get_pynvml_handler(device)
        return pynvml.nvmlDeviceGetClockInfo(handle, 1)
    else:
        return _get_amdsmi_clock_rate(device)


def _get_device(device: int | str | torch.device) -> torch.device:
    r"""Return the torch.device type object from the passed in device.

    Args:
        device (torch.device or int): selected device.
    """
    if isinstance(device, str):
        device = torch.device(device)
    elif isinstance(device, int):
        device = torch.device("cuda", device)
    return device


def _get_generator(device: torch.device) -> torch._C.Generator:
    r"""Return the CUDA Generator object for the given device.

    Args:
        device (torch.device): selected device.
    """
    idx = device.index
    if idx is None:
        idx = current_device()
    return torch.cuda.default_generators[idx]


def _set_rng_state_offset(
    offset: int, device: int | str | torch.device = "cuda"
) -> None:
    r"""Set the random number generator state offset of the specified GPU.

    Args:
        offset (int): The desired offset
        device (torch.device or int, optional): The device to set the RNG state.
            Default: ``'cuda'`` (i.e., ``torch.device('cuda')``, the current CUDA device).
    """
    final_device = _get_device(device)

    def cb():
        default_generator = _get_generator(final_device)
        default_generator.set_offset(offset)

    _lazy_call(cb)


def _get_rng_state_offset(device: int | str | torch.device = "cuda") -> int:
    r"""Return the random number generator state offset of the specified GPU.

    Args:
        device (torch.device or int, optional): The device to return the RNG state offset of.
            Default: ``'cuda'`` (i.e., ``torch.device('cuda')``, the current CUDA device).

    .. warning::
        This function eagerly initializes CUDA.
    """
    _lazy_init()
    final_device = _get_device(device)
    default_generator = _get_generator(final_device)
    return default_generator.get_offset()


# pyrefly: ignore [deprecated]
from .memory import *  # noqa: F403
from .random import *  # noqa: F403


################################################################################
# Define Storage and Tensor classes
################################################################################


@staticmethod  # type: ignore[misc]
def _lazy_new(cls, *args, **kwargs):
    _lazy_init()
    # We may need to call lazy init again if we are a forked child
    # del _CudaBase.__new__
    return super(_CudaBase, cls).__new__(cls, *args, **kwargs)


class _CudaBase:
    is_cuda = True
    is_sparse = False

    def type(self, *args, **kwargs):
        # We could use a Protocol here to tell mypy that self has `get_device` method
        # but it is only available in the typing module on Python >= 3.8
        # or on typing_extensions module on Python >= 3.6
        with device(self.get_device()):  # type: ignore[attr-defined]
            return super().type(*args, **kwargs)  # type: ignore[misc]

    __new__ = _lazy_new


from torch.storage import _LegacyStorage, _warn_typed_storage_removal


class _CudaLegacyStorage(_LegacyStorage):
    @classmethod
    def from_buffer(cls, *args, **kwargs):
        _warn_typed_storage_removal()
        raise RuntimeError("from_buffer: Not available for CUDA storage")

    @classmethod
    def _new_with_weak_ptr(cls, *args, **kwargs):
        raise RuntimeError("_new_with_weak_ptr: Not available for CUDA storage")

    @classmethod
    def _new_shared_filename(cls, manager, obj, size, *, device=None, dtype=None):
        raise RuntimeError("_new_shared_filename: Not available for CUDA storage")


class ByteStorage(_CudaLegacyStorage):
    @classproperty
    def dtype(self):
        _warn_typed_storage_removal()
        return self._dtype

    @classproperty
    def _dtype(self):
        return torch.uint8


class DoubleStorage(_CudaLegacyStorage):
    @classproperty
    def dtype(self):
        _warn_typed_storage_removal()
        return self._dtype

    @classproperty
    def _dtype(self):
        return torch.double


class FloatStorage(_CudaLegacyStorage):
    @classproperty
    def dtype(self):
        _warn_typed_storage_removal()
        return self._dtype

    @classproperty
    def _dtype(self):
        return torch.float


class HalfStorage(_CudaLegacyStorage):
    @classproperty
    def dtype(self):
        _warn_typed_storage_removal()
        return self._dtype

    @classproperty
    def _dtype(self):
        return torch.half


class LongStorage(_CudaLegacyStorage):
    @classproperty
    def dtype(self):
        _warn_typed_storage_removal()
        return self._dtype

    @classproperty
    def _dtype(self):
        return torch.long


class IntStorage(_CudaLegacyStorage):
    @classproperty
    def dtype(self):
        _warn_typed_storage_removal()
        return self._dtype

    @classproperty
    def _dtype(self):
        return torch.int


class ShortStorage(_CudaLegacyStorage):
    @classproperty
    def dtype(self):
        _warn_typed_storage_removal()
        return self._dtype

    @classproperty
    def _dtype(self):
        return torch.short


class CharStorage(_CudaLegacyStorage):
    @classproperty
    def dtype(self):
        _warn_typed_storage_removal()
        return self._dtype

    @classproperty
    def _dtype(self):
        return torch.int8


class BoolStorage(_CudaLegacyStorage):
    @classproperty
    def dtype(self):
        _warn_typed_storage_removal()
        return self._dtype

    @classproperty
    def _dtype(self):
        return torch.bool


class BFloat16Storage(_CudaLegacyStorage):
    @classproperty
    def dtype(self):
        _warn_typed_storage_removal()
        return self._dtype

    @classproperty
    def _dtype(self):
        return torch.bfloat16


class ComplexDoubleStorage(_CudaLegacyStorage):
    @classproperty
    def dtype(self):
        _warn_typed_storage_removal()
        return self._dtype

    @classproperty
    def _dtype(self):
        return torch.cdouble


class ComplexFloatStorage(_CudaLegacyStorage):
    @classproperty
    def dtype(self):
        _warn_typed_storage_removal()
        return self._dtype

    @classproperty
    def _dtype(self):
        return torch.cfloat


del _LegacyStorage
del _CudaLegacyStorage

torch._storage_classes.add(DoubleStorage)
torch._storage_classes.add(FloatStorage)
torch._storage_classes.add(LongStorage)
torch._storage_classes.add(IntStorage)
torch._storage_classes.add(ShortStorage)
torch._storage_classes.add(CharStorage)
torch._storage_classes.add(ByteStorage)
torch._storage_classes.add(HalfStorage)
torch._storage_classes.add(BoolStorage)
torch._storage_classes.add(BFloat16Storage)
torch._storage_classes.add(ComplexDoubleStorage)
torch._storage_classes.add(ComplexFloatStorage)


class _WrappedTritonKernel:
    """Just a simple wrapper to store some metadata for testing purposes."""

    def __init__(self, kernel):
        self.kernel = kernel
        self.kernel_invoked = False

    def __call__(self, *args, **kwargs):
        res = self.kernel(*args, **kwargs)
        self.kernel_invoked = True
        return res


def _register_triton_kernels():
    @_WrappedTritonKernel
    def kernel_impl(*args, **kwargs):
        from torch.sparse._triton_ops import bsr_dense_mm

        # pyrefly: ignore [not-callable]
        return bsr_dense_mm(*args, skip_checks=True, **kwargs)

    @_WrappedTritonKernel
    def addmm_kernel_impl(*args, **kwargs):
        from torch.sparse._triton_ops import bsr_dense_addmm

        return bsr_dense_addmm(*args, skip_checks=True, **kwargs)

    has_triton = importlib.util.find_spec("triton") is not None
    if has_triton:
        torch._TritonLibrary.registerOp(
            "_triton_bsr_dense_mm_out",
            "_triton_bsr_dense_mm_out(Tensor bsr, Tensor dense, *, Tensor(a!) out) -> Tensor(a!)",
            kernel_impl,
            "SparseCsrCUDA",
        )

        torch._TritonLibrary.registerOp(
            "_triton_bsr_dense_addmm_out",
            (
                "_triton_bsr_dense_addmm_out(Tensor input, Tensor bsr, Tensor dense,"
                " *, Scalar beta, Scalar alpha, Tensor(a!) out) -> Tensor(a!)"
            ),
            addmm_kernel_impl,
            "SparseCsrCUDA",
        )


_lazy_call(_register_triton_kernels)


def _compile_kernel(
    kernel_source: str,
    kernel_name: str,
    compute_capability: str | None = None,
    cuda_include_dirs: list | None = None,
    nvcc_options: list | None = None,
):
    """
    Compiles a CUDA kernel using NVRTC and returns a callable function.

    This function is a wrapper for NVRTC that enables runtime compilation of CUDA kernels.
    Note that this returns a raw CUDA kernel that operates on raw memory pointers.
    To use this kernel as a proper PyTorch operator, you should wrap it following the guide at:
    pytorch.org/tutorials/advanced/python_custom_ops.html

    Args:
        kernel_source (str): The CUDA kernel source code as a string
        kernel_name (str): The name of the kernel function to compile
        compute_capability (str, optional): The compute capability to target (e.g., "86").
                                           If None, will detect from current device.
        cuda_include_dirs (list, optional): List of directories containing CUDA headers
        nvcc_options (list, optional): Additional options to pass to NVRTC

    Returns:
        callable: A Python function that can be called with PyTorch tensor arguments to execute the kernel

    Example:
        >>> # xdoctest: +SKIP
        >>> kernel_code = '''
        extern "C"
        __global__ void add_tensors(const float* a, const float* b, float* c, int n) {
            int i = threadIdx.x + blockIdx.x * blockDim.x;
            if (i < n)
                c[i] = a[i] + b[i];
        }
        '''
        >>> add_kernel = torch.cuda.compile_kernel(kernel_code, "add_tensors")
        >>> a = torch.randn(1024, device="cuda")
        >>> b = torch.randn(1024, device="cuda")
        >>> c = torch.empty_like(a)
        >>> add_kernel(grid=(4, 1, 1), block=(256, 1, 1), args=[a, b, c, a.numel()])
    """
    from torch.cuda._utils import _cuda_load_module, _nvrtc_compile

    # Compile the kernel to PTX
    ptx, mangled_name = _nvrtc_compile(
        kernel_source,
        kernel_name,
        compute_capability,
        cuda_include_dirs,
        nvcc_options,
    )

    # Load the module and get the kernel
    result = _cuda_load_module(ptx, [mangled_name])

    if isinstance(result, dict):
        return result[mangled_name]
    else:
        # This branch shouldn't be executed if kernel_names is provided,
        # but MyPy needs this to understand type narrowing
        return getattr(result, mangled_name)


from . import amp, jiterator, nvtx, profiler, sparse, tunable


_POOL_HANDLE = NewType("_POOL_HANDLE", tuple[int, int])


__all__ = [
    # Typed storage and tensors
    "BFloat16Storage",
    "BFloat16Tensor",
    "BoolStorage",
    "BoolTensor",
    "ByteStorage",
    "ByteTensor",
    "CharStorage",
    "CharTensor",
    "ComplexDoubleStorage",
    "ComplexFloatStorage",
    "DoubleStorage",
    "DoubleTensor",
    "FloatStorage",
    "FloatTensor",
    "HalfStorage",
    "HalfTensor",
    "IntStorage",
    "IntTensor",
    "LongStorage",
    "LongTensor",
    "ShortStorage",
    "ShortTensor",
    "CUDAGraph",
    "CudaError",
    "DeferredCudaCallError",
    "Event",
    "ExternalStream",
    "Stream",
    "StreamContext",
    "GreenContext",
    "amp",
    "caching_allocator_alloc",
    "caching_allocator_delete",
    "caching_allocator_enable",
    "can_device_access_peer",
    "check_error",
    "cudaStatus",
    "cudart",
    "current_blas_handle",
    "current_device",
    "current_stream",
    "default_generators",
    "default_stream",
    "device",
    "device_count",
    "device_memory_used",
    "device_of",
    "empty_cache",
    "get_allocator_backend",
    "CUDAPluggableAllocator",
    "change_current_allocator",
    "get_arch_list",
    "get_device_capability",
    "get_device_name",
    "get_device_properties",
    "get_gencode_flags",
    "get_per_process_memory_fraction",
    "get_rng_state",
    "get_rng_state_all",
    "get_stream_from_external",
    "get_sync_debug_mode",
    "graph",
    "graph_pool_handle",
    "graphs",
    "has_half",
    "has_magma",
    "host_memory_stats",
    "host_memory_stats_as_nested_dict",
    "init",
    "initial_seed",
    "ipc_collect",
    "is_available",
    "is_bf16_supported",
    "is_current_stream_capturing",
    "is_initialized",
    "is_tf32_supported",
    "jiterator",
    "list_gpu_processes",
    "make_graphed_callables",
    "manual_seed",
    "manual_seed_all",
    "max_memory_allocated",
    "max_memory_cached",
    "max_memory_reserved",
    "mem_get_info",
    "memory",
    "memory_allocated",
    "memory_cached",
    "memory_reserved",
    "memory_snapshot",
    "memory_stats",
    "memory_stats_as_nested_dict",
    "memory_summary",
    "memory_usage",
    "MemPool",
    "use_mem_pool",
    "temperature",
    "power_draw",
    "clock_rate",
    "nccl",
    "nvtx",
    "profiler",
    "random",
    "reset_accumulated_host_memory_stats",
    "reset_accumulated_memory_stats",
    "reset_max_memory_allocated",
    "reset_max_memory_cached",
    "reset_peak_host_memory_stats",
    "reset_peak_memory_stats",
    "seed",
    "seed_all",
    "set_device",
    "set_per_process_memory_fraction",
    "set_rng_state",
    "set_rng_state_all",
    "set_stream",
    "set_sync_debug_mode",
    "sparse",
    "stream",
    "streams",
    "synchronize",
    "tunable",
    "utilization",
]
