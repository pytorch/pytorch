import torch
from torch._C import dtype


__all__ = ["GPULimits"]


class GPULimits:
    r"""Utility class that provides the theoretical limits of Nvidia GPU devices. The
    limits don't take into account thermal throttling (assume that the GPU run at its
    peak rated frequency). This is because user hardware configuration may influence
    power behavior.
    """

    def __init__(self, target_device: torch.device):
        # The device properties object is obtained by calling 'cudaGetDeviceProperties' CUDA
        # runtime function. We need the total memory bus width and the memory clock rate to
        # calculate the memory bandwidth.
        self.device_properties = torch.cuda.get_device_properties(target_device)

        # The compute capability is needed to determine the number of FLOPs per cycle per SM
        self.compute_capability = int(
            f"{self.device_properties.major}{self.device_properties.minor}"
        )

    # FLOPs per cycle information derived from Table 2 in:
    # https://resources.nvidia.com/en-us-hopper-architecture/nvidia-h100-tensor-c

    # Returns the number of FMA instructions retired per cycle per SM for a given
    # data type, when tensor cores are NOT used
    def get_fma_per_cycle_per_sm_cuda_cores(self, data_type: dtype) -> int:
        hardcoded_device_values = {
            # Ampere Architecture
            "fp16_80": 256,
            "fp32_80": 64,
            "fp64_80": 32,
            # Hopper Architecture
            "fp16_90": 64,
            "fp32_90": 128,
            "fp64_90": 64,
            # Blackwell Architecture
            "fp16_100": 256,
            "fp32_100": 128,
            "fp64_100": 64,
        }
        dict_key = ""
        if data_type is torch.float16:
            dict_key = f"fp16_{self.compute_capability}"
        elif data_type is torch.float32:
            dict_key = f"fp32_{self.compute_capability}"
        elif data_type is torch.float64:
            dict_key = f"fp64_{self.compute_capability}"
        else:
            dict_key = "unknown"

        if dict_key not in hardcoded_device_values:
            raise RuntimeError(
                f"No data for sm_{self.compute_capability} and {data_type}."
            )

        return hardcoded_device_values[dict_key]

    # Returns the number of FMA instructions retired per cycle per SM for a given
    # data type, when tensor cores ARE used
    def get_fma_per_cycle_per_sm_tensor_cores(self, data_type: dtype) -> int:
        hardcoded_device_values = {
            # Ampere Architecture
            "int8_80": 2048,
            "fp16_80": 1024,
            "fp32_80": 512,
            "fp64_80": 64,
            # Hopper Architecture
            "int8_90": 4096,
            "fp8_90": 4096,
            "fp16_90": 2048,
            "fp32_90": 1024,
            "fp64_90": 128,
            # Blackwell Architecture
            "int8_100": 8192,
            "fp8_100": 8192,
            "fp16_100": 4096,
            "fp32_100": 2048,
        }
        dict_key = ""
        if data_type is torch.float16:
            dict_key = f"fp16_{self.compute_capability}"
        elif data_type is torch.bfloat16:
            # FP16 and BF16 are equivalent in terms of FLOPs per cycle per SM
            dict_key = f"fp16_{self.compute_capability}"
        elif data_type is torch.float32:
            dict_key = f"fp32_{self.compute_capability}"
        elif data_type is torch.int8:
            dict_key = f"int8_{self.compute_capability}"
        elif data_type is torch.float64:
            dict_key = f"fp64_{self.compute_capability}"
        else:
            dict_key = "unknown"

        if dict_key not in hardcoded_device_values:
            raise RuntimeError(
                f"No data for sm_{self.compute_capability} and {data_type}."
            )

        return hardcoded_device_values[dict_key]

    def get_tflops_per_second(
        self, data_type: dtype, use_tensor_cores: bool = True
    ) -> float:
        num_sms = self.device_properties.multi_processor_count
        clock_rate = self.device_properties.clock_rate  # KHz

        fma_per_cycle = 0
        if use_tensor_cores:
            fma_per_cycle = self.get_fma_per_cycle_per_sm_tensor_cores(data_type)
        else:
            fma_per_cycle = self.get_fma_per_cycle_per_sm_cuda_cores(data_type)

        # 1 FMA counts as 2 floating point operations
        # Clock rate is in KHz
        tflops_per_second = num_sms * fma_per_cycle * 2 * clock_rate / 1e9
        return tflops_per_second

    def get_memory_bandwidth_Bps(self) -> int:
        # DRAM devices are Double-Data which means they provide an output at both fronts of
        # a clock beat
        bus_bytes_per_cycle = int(2 * self.device_properties.memory_bus_width / 8)
        mem_clock_rate_Hz = self.device_properties.memory_clock_rate * 1000
        bytes_per_second = bus_bytes_per_cycle * mem_clock_rate_Hz * 2
        return bytes_per_second

    def get_shared_memory_bandwidth_Bps(self) -> int:
        # Each warp can LD or ST 32 x 4 bytes per cycle. To calculate the
        # device's throughput we need to multiply with frequency and number of SMs.
        num_sms = self.device_properties.multi_processor_count
        bytes_per_cycle_per_sm = 128
        bytes_per_cycle_per_device = num_sms * bytes_per_cycle_per_sm
        bytes_per_second = (
            bytes_per_cycle_per_device * self.device_properties.clock_rate * 1000
        )
        return bytes_per_second
