import logging
from dataclasses import dataclass
from typing import Any, Optional, Union

import torch


log = logging.getLogger(__name__)


# Lazy import cache for pynvml
_pynvml_cache: Optional[Any] = None
_pynvml_initialized: bool = False

# Lazy import cache for AMD ROCm SMI library
_amd_smi_cache: Optional[Any] = None
_amd_smi_name: Optional[str] = None


def _get_pynvml() -> Optional[Any]:
    """
    Lazy import handler for pynvml.
    Imports pynvml once on first call and caches the result.
    Returns None if pynvml is not available.
    """
    global _pynvml_cache, _pynvml_initialized

    if not _pynvml_initialized:
        try:
            import pynvml  # type: ignore[import]

            _pynvml_cache = pynvml
        except ImportError:
            _pynvml_cache = None
        _pynvml_initialized = True

    return _pynvml_cache


def _get_amd_smi() -> Optional[Any]:
    """
    Lazy import handler for AMD SMI library.
    Attempts to import various possible AMD GPU monitoring libraries.
    Returns None if no AMD SMI library is available.
    """
    global _amd_smi_cache, _amd_smi_name

    if _amd_smi_name is None:
        _amd_smi_cache = None

        # Try various possible AMD SMI library names
        possible_libraries = ["amdsmi", "rocm_smi"]

        for lib_name in possible_libraries:
            try:
                _amd_smi_cache = __import__(lib_name)
                log.info("Successfully imported AMD SMI library: %s", lib_name)
                _amd_smi_name = lib_name
                break
            except ImportError:
                continue

        if _amd_smi_name is None:
            log.info(
                "No AMD SMI library found. AMD hardware introspection unavailable."
            )
            _amd_smi_name = "none"

    return _amd_smi_cache


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
    cores_per_sm: Optional[int]
    clock_hz: Optional[float]
    memory_clock_hz: Optional[float]
    ops_per_core_per_cycle: Optional[int]


class DeviceInfo:
    """
    Device information lookup utility for GPU hardware introspection.

    This class provides methods to retrieve various hardware specifications
    and performance characteristics of GPU devices. It supports both NVIDIA
    and AMD GPUs through hardware lookup methods and falls back to datasheet
    values when hardware information is not available.

    The class can provide information about:
    - Streaming multiprocessor (SM) count
    - Cores per streaming multiprocessor
    - Clock frequencies (core and memory)
    - Operations per core per cycle
    - DRAM capacity and bandwidth
    - Peak FLOPS/TOPS performance

    Methods use a two-tier lookup strategy:
    1. Hardware introspection via pynvml (NVIDIA) or AMD SMI libraries
    2. Fallback to predefined datasheet values for known device models

    Example usage:
        device_name = torch.cuda.get_device_name()
        sm_count = DeviceInfo.lookup_sm_count(device_name)
        peak_tops = DeviceInfo.lookup_tops(device_name, torch.float32)
    """

    @staticmethod
    def _hardware_lookup_sm_count() -> Optional[int]:
        """Get the number of streaming multiprocessors from the hardware."""

        # Fall back to NVIDIA
        try:
            device_props = torch.cuda.get_device_properties(0)
            return device_props.multi_processor_count
        except Exception:
            return None

    @staticmethod
    def _hardware_lookup_cores_per_sm() -> Optional[int]:
        """Get the number of cores per streaming multiprocessor from the hardware."""
        # This information is not directly available via NVML currently
        return None

    @staticmethod
    def _hardware_lookup_clock_hz() -> Optional[float]:
        """Get the clock speed in Hz from the hardware."""
        if torch.version.hip is not None:
            amd_clock = DeviceInfo._amd_hardware_lookup_clock_hz()
            return amd_clock

        pynvml = _get_pynvml()
        if pynvml is None:
            return None

        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            clock_mhz = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)
            clock_hz = clock_mhz * 1e6
            pynvml.nvmlShutdown()
            return clock_hz
        except Exception:
            return None

    @staticmethod
    def _amd_hardware_lookup_clock_hz() -> Optional[float]:
        """Get the clock speed in Hz from AMD hardware."""
        amd_smi = _get_amd_smi()
        if amd_smi is None:
            return None

        try:
            # TODO
            if hasattr(amd_smi, "rsmi_init"):
                amd_smi.rsmi_init()
                clock_mhz = amd_smi.rsmi_dev_gpu_clk_freq_get(
                    0, amd_smi.RSMI_CLK_TYPE_SYS
                )
                amd_smi.rsmi_shut_down()
                return clock_mhz * 1e6
            elif hasattr(amd_smi, "amdsmi_init"):
                amd_smi.amdsmi_init()
                device_handle = amd_smi.amdsmi_get_processor_handle(0)
                clock_info = amd_smi.amdsmi_get_gpu_clock_info(device_handle)
                amd_smi.amdsmi_shut_down()
                return (
                    clock_info.current_clk * 1e6
                    if hasattr(clock_info, "current_clk")
                    else None
                )
            else:
                log.info(
                    "Unknown AMD SMI library API. Cannot determine clock frequency."
                )
                return None
        except Exception as e:
            log.info("Failed to get AMD clock frequency: %s", e)
            return None

    @staticmethod
    def _hardware_lookup_memory_clock_hz() -> Optional[float]:
        """Get the memory clock speed in Hz from the hardware."""
        if torch.version.hip is not None:
            amd_memory_clock = DeviceInfo._amd_hardware_lookup_memory_clock_hz()
            return amd_memory_clock

        pynvml = _get_pynvml()
        if pynvml is None:
            return None

        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            mem_clock_mhz = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)
            mem_clock_hz = mem_clock_mhz * 1e6
            pynvml.nvmlShutdown()
            return mem_clock_hz
        except Exception:
            return None

    @staticmethod
    def _amd_hardware_lookup_memory_clock_hz() -> Optional[float]:
        """Get the memory clock speed in Hz from AMD hardware."""
        amd_smi = _get_amd_smi()
        if amd_smi is None:
            return None

        try:
            # TODO
            if hasattr(amd_smi, "rsmi_init"):
                amd_smi.rsmi_init()
                mem_clock_mhz = amd_smi.rsmi_dev_gpu_clk_freq_get(
                    0, amd_smi.RSMI_CLK_TYPE_MEM
                )
                amd_smi.rsmi_shut_down()
                return mem_clock_mhz * 1e6
            elif hasattr(amd_smi, "amdsmi_init"):
                amd_smi.amdsmi_init()
                device_handle = amd_smi.amdsmi_get_processor_handle(0)
                mem_clock_info = amd_smi.amdsmi_get_gpu_memory_clock_info(device_handle)
                amd_smi.amdsmi_shut_down()
                return (
                    mem_clock_info.current_clk * 1e6
                    if hasattr(mem_clock_info, "current_clk")
                    else None
                )
            else:
                log.info(
                    "Unknown AMD SMI library API. Cannot determine memory clock frequency."
                )
                return None
        except Exception as e:
            log.info("Failed to get AMD memory clock frequency: %s", e)
            return None

    @staticmethod
    def _hardware_lookup_ops_per_core_per_cycle() -> Optional[int]:
        """Get the operations per core per cycle from the hardware."""
        # This information is not directly available via NVML
        # and requires architecture-specific lookup
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
    def _hardware_dram_bw_gbs() -> Optional[float]:
        """Get the DRAM bandwidth in GB/s from the hardware."""
        if torch.version.hip is not None:
            amd_bw = DeviceInfo._amd_hardware_dram_bw_gbs()
            return amd_bw

        pynvml = _get_pynvml()
        if pynvml is None:
            return None

        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)

            mem_clock_mhz = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)
            bus_width_bits = pynvml.nvmlDeviceGetMemoryBusWidth(handle)

            mem_clock_hz = mem_clock_mhz * 1e6
            # Effective rate (GDDR uses DDR so *2)
            effective_rate = mem_clock_hz * 2
            # Theoretical peak bandwidth in bytes/sec
            peak_bw = (effective_rate * bus_width_bits) / 8
            # convert to GB/s
            peak_bw = peak_bw / (1024**3)
            pynvml.nvmlShutdown()
            return peak_bw

        except Exception:
            return None

    @staticmethod
    def _amd_hardware_dram_bw_gbs() -> Optional[float]:
        """Get the DRAM bandwidth in GB/s from AMD hardware."""
        amd_smi = _get_amd_smi()
        if amd_smi is None:
            return None

        try:
            # TODO
            if hasattr(amd_smi, "rsmi_init"):
                # ROCm SMI pattern
                amd_smi.rsmi_init()
                # Memory bandwidth is typically not directly available
                # Would need memory clock and bus width to calculate
                # For now, return None as this requires device-specific calculations
                amd_smi.rsmi_shut_down()
                log.info(
                    "AMD memory bandwidth calculation not implemented. Requires memory clock and bus width."
                )
                return None
            elif hasattr(amd_smi, "amdsmi_init"):
                # AMD SMI pattern
                amd_smi.amdsmi_init()
                # Similar issue - bandwidth calculation requires multiple parameters
                amd_smi.amdsmi_shut_down()
                log.info(
                    "AMD memory bandwidth calculation not implemented. Requires memory clock and bus width."
                )
                return None
            else:
                log.info(
                    "Unknown AMD SMI library API. Cannot determine memory bandwidth."
                )
                return None
        except Exception as e:
            log.info("Failed to get AMD memory bandwidth: %s", e)
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
            "cores_per_sm": DeviceInfo._hardware_lookup_cores_per_sm,
            "clock_hz": DeviceInfo._hardware_lookup_clock_hz,
            "memory_clock_hz": DeviceInfo._hardware_lookup_memory_clock_hz,
            "ops_per_core_per_cycle": DeviceInfo._hardware_lookup_ops_per_core_per_cycle,
            "dram_gb": DeviceInfo._hardware_dram_gb,
            "dram_bw_gbs": DeviceInfo._hardware_dram_bw_gbs,
            # tops missing from here because of custom implementation
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
    def lookup_cores_per_sm(device_name: str) -> Optional[int]:
        """Get the number of cores per streaming multiprocessor for the current device."""
        result = DeviceInfo._generic_lookup(device_name, "cores_per_sm")
        return result if isinstance(result, int) or result is None else None

    @staticmethod
    def lookup_clock_hz(device_name: str) -> Optional[float]:
        """Get the clock speed in Hz for the current device."""
        result = DeviceInfo._generic_lookup(device_name, "clock_hz")
        return result if isinstance(result, (int, float)) or result is None else None

    @staticmethod
    def lookup_memory_clock_hz(device_name: str) -> Optional[float]:
        """Get the memory clock speed in Hz for the current device."""
        result = DeviceInfo._generic_lookup(device_name, "memory_clock_hz")
        return result if isinstance(result, (int, float)) or result is None else None

    @staticmethod
    def lookup_ops_per_core_per_cycle(device_name: str) -> Optional[int]:
        """Get the operations per core per cycle for the current device."""
        result = DeviceInfo._generic_lookup(device_name, "ops_per_core_per_cycle")
        return result if isinstance(result, int) or result is None else None

    @staticmethod
    def lookup_dram_gb(device_name: str) -> Optional[float]:
        """Get the DRAM memory size in GB for the current device."""
        result = DeviceInfo._generic_lookup(device_name, "dram_gb")
        return result if isinstance(result, (int, float)) or result is None else None

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
        if lookupable:
            hardware_bw = DeviceInfo._hardware_dram_bw_gbs()
            if hardware_bw is not None:
                return hardware_bw

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
        force_datasheet: bool = False,
    ) -> Optional[float]:
        """
        Calculate peak FLOPS for the current device.

        Uses the formula: sm_count * cores_per_sm * clock_hz * ops_per_core_per_cycle

        Returns:
            Peak FLOPS as a float, or None if calculation fails
        """
        peak_ops = None
        lookupable = torch.cuda.is_available() and (
            torch.cuda.get_device_name() == device_name
        )

        if not force_datasheet and lookupable:
            # We're on the device that we're testing, so try to look up values via hardware libraries.
            try:
                sm_count = DeviceInfo.lookup_sm_count(device_name)
                cores_per_sm = DeviceInfo.lookup_cores_per_sm(device_name)
                clock_hz = DeviceInfo.lookup_clock_hz(device_name)
                ops_per_core_per_cycle = DeviceInfo.lookup_ops_per_core_per_cycle(
                    device_name
                )

                if all(
                    x is not None
                    for x in [sm_count, cores_per_sm, clock_hz, ops_per_core_per_cycle]
                ):
                    return sm_count * cores_per_sm * clock_hz * ops_per_core_per_cycle  # type: ignore[operator]
            except Exception:
                pass

        # Fallback to datasheet if hardware calculation failed
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


# Indexing is based on `torch.cuda.get_device_name()`
# TODO investigate profiler support for tf32 and allow device to report correct number when it's turned on.
_device_mapping: dict[str, DeviceSpec] = {
    # Source:
    # @lint-ignore https://www.nvidia.com/en-us/data-center/h100/
    # These are from H100 SXM.
    #
    "NVIDIA H100": DeviceSpec(
        tops={
            torch.float64: 67.0,
            torch.float32: 67.5,
            "torch.tf32": 156.0,
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
        sm_count=None,  # TODO
        cores_per_sm=None,  # TODO
        clock_hz=None,  # TODO
        memory_clock_hz=None,  # TODO
        ops_per_core_per_cycle=None,  # TODO
    ),
    # Source:
    # @lint-ignore https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/
    # nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf
    "NVIDIA A100": DeviceSpec(
        tops={
            torch.float64: 19.5,
            torch.float32: 19.5,
            torch.bfloat16: 312.5,
            torch.float16: 312.5,
            # Not in datasheet: float8
            torch.int8: 624.0,
            "torch.tf32": 156.0,
        },
        dram_bw_gbs=2039.0,
        dram_gb=80.0,
        sm_count=None,  # TODO
        cores_per_sm=None,  # TODO
        clock_hz=None,  # TODO
        memory_clock_hz=None,  # TODO
        ops_per_core_per_cycle=None,  # TODO
    ),
    # Source:
    # @lint-ignore https://resources.nvidia.com/en-us-gpu-resources/l4-tensor-datasheet
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
        sm_count=None,  # TODO
        cores_per_sm=None,  # TODO
        clock_hz=None,  # TODO
        memory_clock_hz=None,  # TODO
        ops_per_core_per_cycle=None,  # TODO
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
        sm_count=None,  # TODO
        cores_per_sm=None,  # TODO
        clock_hz=None,  # TODO
        memory_clock_hz=None,  # TODO
        ops_per_core_per_cycle=None,  # TODO
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
        sm_count=None,  # TODO
        cores_per_sm=None,  # TODO
        clock_hz=None,  # TODO
        memory_clock_hz=None,  # TODO
        ops_per_core_per_cycle=None,  # TODO
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
        sm_count=None,  # TODO
        cores_per_sm=None,  # TODO
        clock_hz=None,  # TODO
        memory_clock_hz=None,  # TODO
        ops_per_core_per_cycle=None,  # TODO
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
    if not torch.cuda.is_available():
        return None
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
