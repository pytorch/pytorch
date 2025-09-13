import logging
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Optional, Union

import torch


log = logging.getLogger(__name__)


def _get_pynvml() -> Optional[Any]:
    """Get pynvml from torch.cuda if available."""
    return getattr(torch.cuda, "pynvml", None) if torch.cuda._HAS_PYNVML else None


def _get_amd_smi() -> Optional[Any]:
    """Get AMD SMI from torch.cuda if available."""
    return getattr(torch.cuda, "amdsmi", None) if torch.cuda._HAS_PYNVML else None


@contextmanager
def _device_library_context(
    library_getter: Callable[[], Optional[Any]],
    library_name: str,
    init_method: str,
    shutdown_method: str,
) -> Generator[Any, None, None]:
    """
    Generic context manager for device library operations.
    Handles initialization, exception catching, and cleanup.

    Args:
        library_getter: Function that returns the library module or None
        library_name: Name of the library for error messages
        init_method: Name of the initialization method to call
        shutdown_method: Name of the shutdown method to call
    """
    library = library_getter()
    if library is None:
        raise RuntimeError(f"{library_name} not available")

    try:
        getattr(library, init_method)()
        yield library
    finally:
        try:
            getattr(library, shutdown_method)()
        except Exception:
            pass


@contextmanager
def _nvml_context() -> Generator[Any, None, None]:
    """Context manager for NVML operations."""
    with _device_library_context(
        _get_pynvml, "pynvml", "nvmlInit", "nvmlShutdown"
    ) as library:
        yield library


@contextmanager
def _amd_smi_context() -> Generator[Any, None, None]:
    """Context manager for AMD SMI operations."""
    with _device_library_context(
        _get_amd_smi, "amdsmi", "amdsmi_init", "amdsmi_shut_down"
    ) as library:
        yield library


@dataclass(frozen=True)
class DeviceSpec:
    """
    Theoretical Numbers from data sheet. If two numbers are given, Tensor/Matrix Core vs not,
    then the higher number is reported. Sparsity is not considered.

    Bandwidth numbers are tricky, because there are platform differences that may not show up in the profiler trace.
    """

    tops: dict[Union[torch.dtype, str], float]
    dram_bw_gbs: float
    dram_gb: float
    sm_count: Optional[int]
    clock_hz: Optional[float]
    memory_clock_hz: Optional[float]


class DeviceInfo:
    """
    Device information lookup utility for GPU hardware introspection.

    This class provides methods to retrieve various hardware specifications
    and performance characteristics of GPU devices. It supports both NVIDIA
    and AMD GPUs through hardware lookup methods and falls back to datasheet
    values when hardware information is not available.

    The class can provide information about:
    - Streaming multiprocessor (SM) count
    - Clock frequencies (core and memory)
    - DRAM capacity and bandwidth
    - Peak FLOPS/TOPS performance

    Methods use a two-tier lookup strategy:
    1. Hardware introspection via pynvml (NVIDIA) or AMD SMI libraries
    2. Fallback to predefined datasheet values for known device models

    Example usage:
        device_name = torch.cuda.get_device_name()
        peak_tops = DeviceInfo.lookup_tops(device_name, torch.float32)
    """

    @staticmethod
    def _hardware_lookup_sm_count() -> Optional[int]:
        """Get the number of streaming multiprocessors from the hardware."""
        try:
            # rely on device_properties api
            device_props = torch.cuda.get_device_properties(0)
            return device_props.multi_processor_count
        except Exception:
            return None

    @staticmethod
    def _hardware_lookup_clock_hz() -> Optional[float]:
        """Get the clock speed in Hz from the hardware."""
        if torch.version.hip is not None:
            amd_clock = DeviceInfo._amd_hardware_lookup_clock_hz()
            return amd_clock

        try:
            with _nvml_context() as pynvml:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                clock_mhz = pynvml.nvmlDeviceGetMaxClockInfo(
                    handle, pynvml.NVML_CLOCK_SM
                )
                return clock_mhz * 1e6
        except Exception:
            return None

    @staticmethod
    def _amd_hardware_lookup_clock_hz() -> Optional[float]:
        """Get the clock speed in Hz from AMD hardware."""
        try:
            with _amd_smi_context() as amd_smi:
                device_handle = amd_smi.amdsmi_get_processor_handles()[0]
                clock_info = amd_smi.amdsmi_get_clock_info(
                    device_handle, amd_smi.AmdSmiClkType.SYS
                )
                return clock_info["max_clk"] * 1e6 if "max_clk" in clock_info else None
        except Exception as e:
            log.info("Failed to get AMD clock frequency: %s", e)
            return None

    @staticmethod
    def _hardware_lookup_memory_clock_hz() -> Optional[float]:
        """Get the memory clock speed in Hz from the hardware."""
        if torch.version.hip is not None:
            amd_memory_clock = DeviceInfo._amd_hardware_lookup_memory_clock_hz()
            return amd_memory_clock

        try:
            with _nvml_context() as pynvml:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                mem_clock_mhz = pynvml.nvmlDeviceGetMaxClockInfo(
                    handle, pynvml.NVML_CLOCK_MEM
                )
                return mem_clock_mhz * 1e6
        except Exception:
            return None

    @staticmethod
    def _amd_hardware_lookup_memory_clock_hz() -> Optional[float]:
        """Get the memory clock speed in Hz from AMD hardware."""
        try:
            with _amd_smi_context() as amd_smi:
                device_handle = amd_smi.amdsmi_get_processor_handles()[0]
                mem_clock_info = amd_smi.amdsmi_get_clock_info(
                    device_handle, amd_smi.AmdSmiClkType.MEM
                )
                return (
                    mem_clock_info["max_clk"] * 1e6
                    if "max_clk" in mem_clock_info
                    else None
                )
        except Exception as e:
            log.info("Failed to get AMD memory clock frequency: %s", e)
            return None

    @staticmethod
    def _hardware_dram_gb() -> Optional[float]:
        """Get the DRAM memory size in GB from the hardware."""
        try:
            device_props = torch.cuda.get_device_properties(0)
            # Convert from bytes to GB
            return device_props.total_memory / (1024**3)
        except Exception:
            return None

    @staticmethod
    def _generic_lookup(
        device_name: str, element_name: str
    ) -> Optional[Union[int, float]]:
        """
        Generic lookup method for device elements.
        First attempts hardware lookup, then falls back to device mapping.

        Args:
            element_name: Name of the element to lookup (e.g., 'sm_count', 'clock_hz')

        Returns:
            The value from hardware lookup or device mapping, or None if not available.
        """
        hardware_lookup_methods = {
            "sm_count": DeviceInfo._hardware_lookup_sm_count,
            "clock_hz": DeviceInfo._hardware_lookup_clock_hz,
            "memory_clock_hz": DeviceInfo._hardware_lookup_memory_clock_hz,
            "dram_gb": DeviceInfo._hardware_dram_gb,
        }

        if torch.cuda.is_available() and torch.cuda.get_device_name() == device_name:
            # we're on the device that we're testing, so try to look up values via hardware libraries.
            hardware_method = hardware_lookup_methods.get(element_name)
            if hardware_method:
                hardware_value = hardware_method()
                if hardware_value is not None:
                    return hardware_value

        # Attempt to lookup from device mapping
        device_info = lookup_device_info(device_name)
        if device_info is not None:
            return getattr(device_info, element_name, None)

        return None

    @staticmethod
    def lookup_sm_count(device_name: str) -> Optional[int]:
        """Get the number of streaming multiprocessors for the current device."""
        result = DeviceInfo._generic_lookup(device_name, "sm_count")
        return result if isinstance(result, int) or result is None else None

    @staticmethod
    def lookup_clock_hz(device_name: str) -> Optional[float]:
        """Get the clock speed in Hz for the current device."""
        return DeviceInfo._generic_lookup(device_name, "clock_hz")

    @staticmethod
    def lookup_memory_clock_hz(device_name: str) -> Optional[float]:
        """Get the memory clock speed in Hz for the current device."""
        return DeviceInfo._generic_lookup(device_name, "memory_clock_hz")

    @staticmethod
    def lookup_dram_gb(device_name: str) -> Optional[float]:
        """Get the DRAM memory size in GB for the current device."""
        return DeviceInfo._generic_lookup(device_name, "dram_gb")

    @staticmethod
    def lookup_dram_bw_gbs(device_name: str) -> Optional[float]:
        """
        Get the DRAM bandwidth in GB/s for the current device.

        Uses hardware lookup first, then falls back to datasheet value
        scaled by memory clock ratio if available.
        """
        lookupable = torch.cuda.is_available() and (
            torch.cuda.get_device_name() == device_name
        )

        # Fall back to datasheet value with memory clock scaling
        device_info = lookup_device_info(device_name)
        if device_info is None:
            return None

        datasheet_bw = device_info.dram_bw_gbs
        if datasheet_bw is None:
            return None

        # Apply memory clock adjustment if current memory clock is available
        if lookupable:
            current_memory_clock_hz = DeviceInfo.lookup_memory_clock_hz(device_name)
            if (
                current_memory_clock_hz is not None
                and device_info.memory_clock_hz is not None
            ):
                # Scale bandwidth by memory clock ratio
                expected_memory_clock_hz = device_info.memory_clock_hz
                memory_clock_ratio = current_memory_clock_hz / expected_memory_clock_hz
                datasheet_bw *= memory_clock_ratio

        return datasheet_bw

    @staticmethod
    def lookup_tops(
        device_name: str,
        dtype: torch.dtype,
        is_tf32: bool = False,
    ) -> Optional[float]:
        """
        Our best attempt to calculate the current tops. Adjust by the ratio of current clock speed to theoretical.

        Returns:
            Peak FLOPS as a float, or None if calculation fails
        """
        lookupable = torch.cuda.is_available() and (
            torch.cuda.get_device_name() == device_name
        )

        # Use datasheet values adjusted by clock ratio
        peak_ops = datasheet_tops(dtype, is_tf32)
        if peak_ops is None:
            return None
        peak_ops *= 1e12  # Convert TOPS to FLOPS

        # Apply clock adjustment for datasheet fallback calculations

        if not torch.cuda.is_available():
            return peak_ops

        device_name = torch.cuda.get_device_name()
        if device_name is None:
            return peak_ops

        device_info = lookup_device_info(device_name)
        if device_info is None:
            return peak_ops

        if lookupable:
            current_clock_hz = DeviceInfo.lookup_clock_hz(device_name)
            if current_clock_hz is not None and device_info.clock_hz is not None:
                # Use the expected clock speed from the device mapping for scaling
                expected_clock_hz = device_info.clock_hz
                clock_ratio = current_clock_hz / expected_clock_hz
                peak_ops *= clock_ratio

        return peak_ops

    @staticmethod
    def lookup_tops_current_device(
        dtype: torch.dtype,
        is_tf32: bool = False,
    ) -> Optional[float]:
        """
        Our best attempt to calculate the current tops. Adjust by the ratio of current clock speed to theoretical.

        Returns:
            Peak FLOPS as a float, or None if calculation fails
        """
        if not torch.cuda.is_available():
            return None
        name: Optional[str] = torch.cuda.get_device_name()
        if name is None:
            return None
        return DeviceInfo.lookup_tops(name, dtype, is_tf32)


# Indexing is based on `torch.cuda.get_device_name()`
# TODO investigate profiler support for tf32 and allow device to report correct number when it's turned on.
_device_mapping: dict[str, DeviceSpec] = {
    # Source:
    # @lint-ignore https://www.nvidia.com/en-us/data-center/h100/
    # These are from H100 SXM.
    #
    "NVIDIA H100": DeviceSpec(
        tops={
            torch.float64: 34.0,
            torch.float32: 67.0,
            "torch.tf32": 989.0,
            torch.bfloat16: 1979.0,
            torch.float16: 1979.0,
            torch.float8_e8m0fnu: 3958.0,
            torch.float8_e8m0fnu: 3958.0,
            torch.float8_e4m3fnuz: 3958.0,
            torch.float8_e5m2: 3958.0,
            torch.float8_e5m2fnuz: 3958.0,
            torch.float8_e8m0fnu: 3958.0,
            torch.int8: 3958.0,
        },
        dram_bw_gbs=3350,
        dram_gb=80,
        sm_count=132,
        # boost clock
        clock_hz=1.98e9,
        memory_clock_hz=1.4e10,
        # bus: 5120 bit
    ),
    # Source:
    # @lint-ignore https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/
    # nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf
    # Tensor cores enabled + SXM
    "NVIDIA A100": DeviceSpec(
        tops={
            torch.float64: 19.5,
            torch.float32: 19.5,
            torch.bfloat16: 312.5,
            torch.float16: 312.5,
            # Not in datasheet: float8
            torch.int8: 624.0,
            "torch.tf32": 312.0,
        },
        dram_bw_gbs=2039.0,
        dram_gb=80.0,
        sm_count=108,
        # boost clock
        clock_hz=1410 * 1e6,
        memory_clock_hz=1593 * 1e6,
    ),
    # Source:
    # @lint-ignore https://resources.nvidia.com/en-us-gpu-resources/l4-tensor-datasheet
    # @lint-ignore https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/l4/PB-11316-001_v01.pdf
    "NVIDIA L4": DeviceSpec(
        tops={
            # This is a guess, not in datasheet
            torch.float64: 15.1,
            torch.float32: 30.3,
            "torch.tf32": 120.0,
            torch.bfloat16: 242.0,
            torch.float16: 242.0,
            torch.float8_e8m0fnu: 485.0,
            torch.float8_e8m0fnu: 485.0,
            torch.float8_e4m3fnuz: 485.0,
            torch.float8_e5m2: 485.0,
            torch.float8_e5m2fnuz: 485.0,
            torch.float8_e8m0fnu: 485.0,
            torch.int8: 485.0,
        },
        dram_bw_gbs=3350,
        dram_gb=24,
        sm_count=58,
        clock_hz=2040 * 1e6,
        # bus: 192 bit
        memory_clock_hz=6251 * 1e6,
    ),
    # Source:
    # @lint-ignore https://www.amd.com/content/dam/amd/en/documents\
    # /instinct-tech-docs/data-sheets/amd-instinct-mi300a-data-sheet.pdf
    "AMD MI300A": DeviceSpec(
        tops={
            torch.float64: 122.6,
            torch.float32: 122.6,
            "torch.tf32": 490.3,
            torch.bfloat16: 980.6,
            torch.float16: 980.6,
            torch.float8_e8m0fnu: 1961.2,
            torch.float8_e8m0fnu: 1961.2,
            torch.float8_e4m3fnuz: 1961.2,
            torch.float8_e5m2: 1961.2,
            torch.float8_e5m2fnuz: 1961.2,
            torch.float8_e8m0fnu: 1961.2,
            torch.int8: 1961.2,
        },
        dram_bw_gbs=5300.0,
        dram_gb=128.0,
        sm_count=228,
        # bus: 8192 bit
        clock_hz=2100 * 1e6,
        memory_clock_hz=2600 * 1e6,
    ),
    # Source:
    # @lint-ignore https://www.amd.com/content/dam/amd/en/documents/\
    # instinct-tech-docs/data-sheets/amd-instinct-mi300x-data-sheet.pdf
    "AMD MI300X": DeviceSpec(
        tops={
            torch.float64: 163.4,
            torch.float32: 163.4,
            "torch.tf32": 653.7,
            torch.bfloat16: 1307.4,
            torch.float16: 1307.4,
            torch.float8_e8m0fnu: 2614.9,
            torch.float8_e8m0fnu: 2614.9,
            torch.float8_e4m3fnuz: 2614.9,
            torch.float8_e5m2: 2614.9,
            torch.float8_e5m2fnuz: 2614.9,
            torch.float8_e8m0fnu: 2614.9,
            torch.int8: 2614.9,
        },
        dram_bw_gbs=5300.0,
        dram_gb=192.0,
        sm_count=304,
        clock_hz=2100 * 1e6,
        memory_clock_hz=5200 * 1e6,
    ),
    # Source:
    # @lint-ignore https://www.amd.com/content/dam/amd/\
    # en/documents/instinct-business-docs/product-briefs/instinct-mi210-brochure.pdf
    "AMD MI210X": DeviceSpec(
        tops={
            torch.float64: 45.3,
            torch.float32: 45.3,
            # not specified, fall back to float32 numbers
            "torch.tf32": 45.3,
            torch.bfloat16: 181.0,
            torch.float16: 181.0,
            # not specified, fall back to float16 numbers
            torch.float8_e8m0fnu: 181.0,
            torch.float8_e8m0fnu: 181.0,
            torch.float8_e4m3fnuz: 181.0,
            torch.float8_e5m2: 181.0,
            torch.float8_e5m2fnuz: 181.0,
            torch.float8_e8m0fnu: 181.0,
            torch.int8: 181.0,
        },
        # pcie4.0x16
        dram_bw_gbs=1600.0,
        dram_gb=64.0,
        sm_count=104,
        clock_hz=1700 * 1e6,
        memory_clock_hz=1600 * 1e6,
    ),
}
_device_mapping["AMD INSTINCT MI300X"] = _device_mapping["AMD MI300X"]
_device_mapping["AMD INSTINCT MI210X"] = _device_mapping["AMD MI210X"]


def lookup_device_info(name: str) -> Optional[DeviceSpec]:
    """
    Problem: when diffing profiles between amd and nvidia, we don't have access to the device information
    of the other one. Also, since the analysis is static, we should be able to do it on another device unrelated
    to the recorded device. Therefore, _device_mapping statically contains the information for lots of devices.
    If one is missing, please run DeviceSpec.get_device_info() and add it to _device_mapping.
      name (str): name of the device to lookup. Should map onto torch.cuda.get_device_name().
    """
    return _device_mapping.get(name, None)


def datasheet_tops(dtype: torch.dtype, is_tf32: bool = False) -> Optional[float]:
    """
    Get the theoretical TFLOPS of the device for a given dtype. This can throw an exception if the device
    is not in the datasheet list above.
    """
    name: Optional[str] = torch.cuda.get_device_name()
    if name is None:
        log.info("No device found, returning None")
        return None
    device_info = lookup_device_info(name)
    if device_info is None:
        log_str = f"Device {name} not in datasheet, returning None"
        log.info(log_str)
        return None
    if dtype not in device_info.tops:
        log.info(
            "Device %s does not have a datasheet entry for %s, returning None",
            name,
            dtype,
        )
        return None

    return device_info.tops[
        "torch.tf32" if dtype == torch.float32 and is_tf32 else dtype
    ]
