import logging
from dataclasses import dataclass
from typing import Any, Optional, Union

import torch


log = logging.getLogger(__name__)


# Lazy import cache for pynvml to avoid importing it multiple times
_pynvml_cache: Optional[Any] = None
_pynvml_initialized = False

# Lazy import cache for AMD ROCm SMI library
_amd_smi_cache: Optional[Any] = None
_amd_smi_initialized = False


def _get_pynvml() -> Optional[Any]:
    """
    Lazy import handler for pynvml.
    Imports pynvml once on first call and caches the result.
    Returns None if pynvml is not available.
    """
    global _pynvml_cache, _pynvml_initialized

    if not _pynvml_initialized:
        try:
            import pynvml

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

    NOTE: The exact library name and API for AMD GPU monitoring is uncertain.
    Common possibilities include:
    - rocm_smi or rocmsmi (ROCm System Management Interface)
    - amdsmi (AMD System Management Interface)
    - py-rocm-smi

    This implementation attempts several common options but may need adjustment
    based on the actual available AMD monitoring library and its API.
    """
    global _amd_smi_cache, _amd_smi_initialized

    if not _amd_smi_initialized:
        _amd_smi_cache = None

        # Try various possible AMD SMI library names
        possible_libraries = ["amdsmi", "rocm_smi", "rocmsmi", "py_rocm_smi"]

        for lib_name in possible_libraries:
            try:
                _amd_smi_cache = __import__(lib_name)
                log.info("Successfully imported AMD SMI library: %s", lib_name)
                break
            except ImportError:
                continue

        if _amd_smi_cache is None:
            log.info(
                "No AMD SMI library found. AMD hardware introspection unavailable."
            )

        _amd_smi_initialized = True

    return _amd_smi_cache


@dataclass(frozen=True)
class DeviceInfo:
    """
    Theoretical Numbers from data sheet. If two numbers are given, Tensor/Matrix Core vs not,
    then the higher number is reported. Sparsity is not considered.


    Bandwidth numbers are tricky, because there are platform differences that may not show up in the profiler trace.
    For example,
    """

    tops: dict[Union[torch.dtype, str], float]
    dram_bw_gbs: float
    dram_gb: float
    sm_count: Optional[int]
    cores_per_sm: Optional[int]
    clock_hz: Optional[float]
    ops_per_core_per_cycle: Optional[int]

    def _hardware_lookup_sm_count(self) -> Optional[int]:
        """Get the number of streaming multiprocessors/compute units from the hardware."""
        # Try AMD first
        if torch.version.hip is not None:
            amd_count = self._amd_hardware_lookup_sm_count()
            if amd_count is not None:
                return amd_count

        # Fall back to NVIDIA
        pynvml = _get_pynvml()
        if pynvml is None:
            return None

        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            sm_count = pynvml.nvmlDeviceGetAttributes(handle)[
                pynvml.NVML_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT
            ]
            pynvml.nvmlShutdown()
            return sm_count
        except Exception:
            return None

    def _amd_hardware_lookup_sm_count(self) -> Optional[int]:
        """Get the number of compute units from AMD hardware."""
        amd_smi = _get_amd_smi()
        if amd_smi is None:
            return None

        try:
            # NOTE: This is speculative implementation based on common AMD SMI patterns.
            # The actual API may differ depending on the specific AMD library available.
            # Common patterns include:
            # - rocm_smi.rsmi_compute_unit_count_get(device_id)
            # - amdsmi.amdsmi_get_gpu_compute_unit_count(device_handle)

            # Attempt a generic approach - this will likely need adjustment
            # based on the actual available AMD SMI library
            if hasattr(amd_smi, "rsmi_init"):
                # ROCm SMI pattern
                amd_smi.rsmi_init()
                compute_units = amd_smi.rsmi_compute_unit_count_get(0)
                amd_smi.rsmi_shut_down()
                return compute_units
            elif hasattr(amd_smi, "amdsmi_init"):
                # AMD SMI pattern
                amd_smi.amdsmi_init()
                device_handle = amd_smi.amdsmi_get_processor_handle(0)
                compute_units = amd_smi.amdsmi_get_gpu_compute_unit_count(device_handle)
                amd_smi.amdsmi_shut_down()
                return compute_units
            else:
                log.info(
                    "Unknown AMD SMI library API. Cannot determine compute unit count."
                )
                return None
        except Exception as e:
            log.info("Failed to get AMD compute unit count: %s", e)
            return None

    def _hardware_lookup_cores_per_sm(self) -> Optional[int]:
        """Get the number of cores per streaming multiprocessor from the hardware."""
        # This information is not directly available via NVML
        # and requires architecture-specific lookup
        return None

    def _hardware_lookup_clock_hz(self) -> Optional[float]:
        """Get the clock speed in Hz from the hardware."""
        # Try AMD first
        if torch.version.hip is not None:
            amd_clock = self._amd_hardware_lookup_clock_hz()
            if amd_clock is not None:
                return amd_clock

        # Fall back to NVIDIA
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

    def _amd_hardware_lookup_clock_hz(self) -> Optional[float]:
        """Get the clock speed in Hz from AMD hardware."""
        amd_smi = _get_amd_smi()
        if amd_smi is None:
            return None

        try:
            # NOTE: This is speculative implementation based on common AMD SMI patterns.
            # The actual API may differ depending on the specific AMD library available.

            if hasattr(amd_smi, "rsmi_init"):
                # ROCm SMI pattern
                amd_smi.rsmi_init()
                # Get current GPU clock frequency (typically in MHz)
                clock_mhz = amd_smi.rsmi_dev_gpu_clk_freq_get(
                    0, amd_smi.RSMI_CLK_TYPE_SYS
                )
                amd_smi.rsmi_shut_down()
                return clock_mhz * 1e6  # Convert MHz to Hz
            elif hasattr(amd_smi, "amdsmi_init"):
                # AMD SMI pattern
                amd_smi.amdsmi_init()
                device_handle = amd_smi.amdsmi_get_processor_handle(0)
                clock_info = amd_smi.amdsmi_get_gpu_clock_info(device_handle)
                amd_smi.amdsmi_shut_down()
                # Assuming clock_info contains frequency in MHz
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

    def _hardware_lookup_ops_per_core_per_cycle(self) -> Optional[int]:
        """Get the operations per core per cycle from the hardware."""
        # This information is not directly available via NVML
        # and requires architecture-specific lookup
        return None

    def _hardware_dram_gb(self) -> Optional[float]:
        """Get the DRAM memory size in GB from the hardware."""
        # Try AMD first
        if torch.version.hip is not None:
            amd_memory = self._amd_hardware_dram_gb()
            if amd_memory is not None:
                return amd_memory

        # Fall back to NVIDIA
        pynvml = _get_pynvml()
        if pynvml is None:
            return None

        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            # Convert bytes to GB
            dram_gb = memory_info.total / (1024**3)
            pynvml.nvmlShutdown()
            return dram_gb
        except Exception:
            return None

    def _amd_hardware_dram_gb(self) -> Optional[float]:
        """Get the DRAM memory size in GB from AMD hardware."""
        amd_smi = _get_amd_smi()
        if amd_smi is None:
            return None

        try:
            # NOTE: This is speculative implementation based on common AMD SMI patterns.
            # The actual API may differ depending on the specific AMD library available.

            if hasattr(amd_smi, "rsmi_init"):
                # ROCm SMI pattern
                amd_smi.rsmi_init()
                # Get memory size (typically in bytes)
                memory_bytes = amd_smi.rsmi_dev_memory_total_get(0)
                amd_smi.rsmi_shut_down()
                return memory_bytes / (1024**3)  # Convert bytes to GB
            elif hasattr(amd_smi, "amdsmi_init"):
                # AMD SMI pattern
                amd_smi.amdsmi_init()
                device_handle = amd_smi.amdsmi_get_processor_handle(0)
                memory_info = amd_smi.amdsmi_get_gpu_memory_info(device_handle)
                amd_smi.amdsmi_shut_down()
                # Assuming memory_info contains total memory in bytes
                return (
                    memory_info.total / (1024**3)
                    if hasattr(memory_info, "total")
                    else None
                )
            else:
                log.info("Unknown AMD SMI library API. Cannot determine memory size.")
                return None
        except Exception as e:
            log.info("Failed to get AMD memory size: %s", e)
            return None

    def _hardware_dram_bw_gbs(self) -> Optional[float]:
        """Get the DRAM bandwidth in GB/s from the hardware."""
        # Try AMD first
        if torch.version.hip is not None:
            amd_bw = self._amd_hardware_dram_bw_gbs()
            if amd_bw is not None:
                return amd_bw

        # Fall back to NVIDIA
        pynvml = _get_pynvml()
        if pynvml is None:
            return None

        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)

            # Memory bandwidth calculation requires memory clock speed, bus width,
            # and memory type information which are not always available via NVML
            # Get memory clock speed in MHz
            _mem_clock_mhz = pynvml.nvmlDeviceGetClockInfo(
                handle, pynvml.NVML_CLOCK_MEM
            )

            # Get memory bus width - this is architecture dependent and not always available
            # We'll try to get it via device attributes, but it may not be available
            try:
                # Try to get memory bus width (in bits)
                _attributes = pynvml.nvmlDeviceGetAttributes(handle)
                # Memory bus width is not always available via NVML attributes
                # This is a simplified calculation and may not be accurate for all devices

                # For now, we'll return None as bandwidth calculation requires
                # architecture-specific information about memory bus width,
                # memory type (GDDR6, HBM2, etc.), and other factors
                pynvml.nvmlShutdown()
                return None
            except Exception:
                pynvml.nvmlShutdown()
                return None
        except Exception:
            return None

    def _amd_hardware_dram_bw_gbs(self) -> Optional[float]:
        """Get the DRAM bandwidth in GB/s from AMD hardware."""
        amd_smi = _get_amd_smi()
        if amd_smi is None:
            return None

        try:
            # NOTE: This is speculative implementation based on common AMD SMI patterns.
            # The actual API may differ depending on the specific AMD library available.
            # Memory bandwidth calculation is complex and may not be directly available
            # from hardware APIs. It often requires memory clock speed, bus width, and
            # memory type information to calculate properly.

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
                _device_handle = amd_smi.amdsmi_get_processor_handle(0)
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

    def _generic_lookup(self, element_name: str) -> Optional[Union[int, float]]:
        """
        Generic lookup method for device elements.
        First attempts hardware lookup, then falls back to device mapping.

        Args:
            element_name: Name of the element to lookup (e.g., 'sm_count', 'clock_hz')

        Returns:
            The value from hardware lookup or device mapping, or None if not available.
        """
        hardware_lookup_methods = {
            "sm_count": self._hardware_lookup_sm_count,
            "cores_per_sm": self._hardware_lookup_cores_per_sm,
            "clock_hz": self._hardware_lookup_clock_hz,
            "ops_per_core_per_cycle": self._hardware_lookup_ops_per_core_per_cycle,
            "dram_gb": self._hardware_dram_gb,
            "dram_bw_gbs": self._hardware_dram_bw_gbs,
        }

        hardware_method = hardware_lookup_methods.get(element_name)
        if hardware_method:
            hardware_value = hardware_method()
            if hardware_value is not None:
                return hardware_value

        return getattr(self, element_name, None)

    def lookup_sm_count(self) -> Optional[int]:
        """Get the number of streaming multiprocessors for the current device."""
        result = self._generic_lookup("sm_count")
        return result if isinstance(result, int) or result is None else None

    def lookup_cores_per_sm(self) -> Optional[int]:
        """Get the number of cores per streaming multiprocessor for the current device."""
        result = self._generic_lookup("cores_per_sm")
        return result if isinstance(result, int) or result is None else None

    def lookup_clock_hz(self) -> Optional[float]:
        """Get the clock speed in Hz for the current device."""
        result = self._generic_lookup("clock_hz")
        return result if isinstance(result, (int, float)) or result is None else None

    def lookup_ops_per_core_per_cycle(self) -> Optional[int]:
        """Get the operations per core per cycle for the current device."""
        result = self._generic_lookup("ops_per_core_per_cycle")
        return result if isinstance(result, int) or result is None else None

    def lookup_dram_gb(self) -> Optional[float]:
        """Get the DRAM memory size in GB for the current device."""
        result = self._generic_lookup("dram_gb")
        return result if isinstance(result, (int, float)) or result is None else None

    def lookup_dram_bw_gbs(self) -> Optional[float]:
        """Get the DRAM bandwidth in GB/s for the current device."""
        result = self._generic_lookup("dram_bw_gbs")
        return result if isinstance(result, (int, float)) or result is None else None

    def flops(self, datasheet_tops: bool = False) -> Optional[float]:
        """
        Calculate peak FLOPS for the current device.

        Uses the formula: sm_count * cores_per_sm * clock_hz * ops_per_core_per_cycle

        Args:
            datasheet_tops: If True, returns the estimated FLOPS from the datasheet
                          instead of hardware lookup calculation

        Returns:
            Peak FLOPS as a float, or None if calculation fails
        """
        if datasheet_tops:
            # Get TOPS value from datasheet for float32 operations
            name = torch.cuda.get_device_name()
            if name is None:
                return None

            device_info = lookup_device_info(name)
            if device_info is None or torch.float32 not in device_info.tops:
                return None

            # Convert TOPS to FLOPS (TOPS * 10^12)
            return device_info.tops[torch.float32] * 1e12

        # Try to calculate using hardware lookup
        try:
            sm_count = self.lookup_sm_count()
            cores_per_sm = self.lookup_cores_per_sm()
            clock_hz = self.lookup_clock_hz()
            ops_per_core_per_cycle = self.lookup_ops_per_core_per_cycle()

            if all(
                x is not None
                for x in [sm_count, cores_per_sm, clock_hz, ops_per_core_per_cycle]
            ):
                # Type assertion to ensure values are not None for multiplication
                assert sm_count is not None
                assert cores_per_sm is not None
                assert clock_hz is not None
                assert ops_per_core_per_cycle is not None
                peak_ops = sm_count * cores_per_sm * clock_hz * ops_per_core_per_cycle
                return peak_ops
        except Exception:
            pass

        # Fall back to datasheet estimate if hardware calculation fails
        name = torch.cuda.get_device_name()
        if name is None:
            return None

        device_info = lookup_device_info(name)
        if device_info is None or torch.float32 not in device_info.tops:
            return None

        # Convert TOPS to FLOPS (TOPS * 10^12)
        datasheet_flops = device_info.tops[torch.float32] * 1e12

        # Adjust by clock ratio if current clock is available
        current_clock_hz = self.lookup_clock_hz()
        if current_clock_hz is not None and device_info.clock_hz is not None:
            # Use the expected clock speed from the device mapping for scaling
            expected_clock_hz = device_info.clock_hz
            clock_ratio = current_clock_hz / expected_clock_hz
            datasheet_flops *= clock_ratio

        return datasheet_flops


# Indexing is based on `torch.cuda.get_device_name()`
# TODO investigate profiler support for tf32 and allow device to report correct number when it's turned on.
_device_mapping: dict[str, DeviceInfo] = {
    # Source:
    # @lint-ignore https://www.nvidia.com/en-us/data-center/h100/
    "NVIDIA H100": DeviceInfo(
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
        ops_per_core_per_cycle=None,  # TODO
    ),
    # Source:
    # @lint-ignore https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/
    # nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf
    "NVIDIA A100": DeviceInfo(
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
        ops_per_core_per_cycle=None,  # TODO
    ),
    # Source:
    # @lint-ignore https://resources.nvidia.com/en-us-gpu-resources/l4-tensor-datasheet
    "NVIDIA L4": DeviceInfo(
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
        ops_per_core_per_cycle=None,  # TODO
    ),
    # Source:
    # @lint-ignore https://www.amd.com/content/dam/amd/en/documents\
    # /instinct-tech-docs/data-sheets/amd-instinct-mi300a-data-sheet.pdf
    "AMD MI300A": DeviceInfo(
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
        ops_per_core_per_cycle=None,  # TODO
    ),
    # Source:
    # @lint-ignore https://www.amd.com/content/dam/amd/en/documents/\
    # instinct-tech-docs/data-sheets/amd-instinct-mi300x-data-sheet.pdf
    "AMD MI300X": DeviceInfo(
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
        ops_per_core_per_cycle=None,  # TODO
    ),
    # Source:
    # @lint-ignore https://www.amd.com/content/dam/amd/\
    # en/documents/instinct-business-docs/product-briefs/instinct-mi210-brochure.pdf
    "AMD MI210X": DeviceInfo(
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
        ops_per_core_per_cycle=None,  # TODO
    ),
}
_device_mapping["AMD INSTINCT MI300X"] = _device_mapping["AMD MI300X"]
_device_mapping["AMD INSTINCT MI210X"] = _device_mapping["AMD MI210X"]


def lookup_device_info(name: str) -> Optional[DeviceInfo]:
    """
    Problem: when diffing profiles between amd and nvidia, we don't have access to the device information
    of the other one. Also, since the analysis is static, we should be able to do it on another device unrelated
    to the recorded device. Therefore, _device_mapping statically contains the information for lots of devices.
    If one is missing, please run DeviceInfo.get_device_info() and add it to _device_mapping.
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
