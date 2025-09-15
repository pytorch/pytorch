import warnings

import torch
from torch._C import dtype

# FLOPs per cycle information derived from Table 2 in:
# https://resources.nvidia.com/en-us-hopper-architecture/nvidia-h100-tensor-c


# Returns the number of FMA instructions retired per cycle per SM for a given
# data type, when tensor cores are NOT used
def get_fma_per_cycle_per_sm_cuda_cores(
    compute_capability: int, data_type: dtype
) -> int:
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
        dict_key = f"fp16_{compute_capability}"
    elif data_type is torch.float32:
        dict_key = f"fp32_{compute_capability}"
    elif data_type is torch.float64:
        dict_key = f"fp64_{compute_capability}"
    else:
        dict_key = "unknown"

    if dict_key not in hardcoded_device_values.keys():
        # We return 1 because typically the peak FLOPs/sec value is used to compute
        # MFU. We want to avoid a division by zero.
        warnings.warn(
            f"No data for sm_{compute_capability} and {data_type}. Returning default value of 1.",
            UserWarning,
        )
        return 1

    return hardcoded_device_values[dict_key]


# Returns the number of FMA instructions retired per cycle per SM for a given
# data type, when tensor cores ARE used
def get_fma_per_cycle_per_sm_tensor_cores(
    compute_capability: int, data_type: dtype
) -> int:
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
        "int8_90": 4096,
        "fp8_90": 4096,
        "fp16_90": 2048,
        "fp32_90": 1024,
        "fp64_90": 128,
    }
    dict_key = ""
    if data_type is torch.float16:
        dict_key = f"fp16_{compute_capability}"
    elif data_type is torch.bfloat16:
        # FP16 and BF16 are equivalent in terms of FLOPs per cycle per SM
        dict_key = f"fp16_{compute_capability}"
    elif data_type is torch.float32:
        dict_key = f"fp32_{compute_capability}"
    elif data_type is torch.int8:
        dict_key = f"int8_{compute_capability}"
    elif data_type is torch.float64:
        dict_key = f"fp64_{compute_capability}"
    else:
        dict_key = "unkown"

    if dict_key not in hardcoded_device_values.keys():
        # We return 1 because typically the peak FLOPs/sec value is used to compute
        # MFU. We want to avoid a division by zero.
        warnings.warn(
            f"No data for sm_{compute_capability} and {data_type}. Returning default value of 1.",
            UserWarning,
        )
        return 1

    return hardcoded_device_values[dict_key]


def get_tflops_per_second(
    target_device: torch.device, data_type: dtype, use_tensor_cores: bool = True
) -> int:
    device_properties = torch.cuda.get_device_properties(target_device)
    comp_capability = int(f"{device_properties.major}{device_properties.minor}")
    num_sms = device_properties.multi_processor_count
    clock_rate = device_properties.clock_rate  # KHz

    fma_per_cycle = 0
    if use_tensor_cores:
        fma_per_cycle = get_fma_per_cycle_per_sm_tensor_cores(
            comp_capability, data_type
        )
    else:
        fma_per_cycle = get_fma_per_cycle_per_sm_cuda_cores(comp_capability, data_type)

    # 1 FMA counts as 2 floating point operations
    # Clock rate is in KHz
    tflops_per_second = num_sms * fma_per_cycle * 2 * clock_rate / 1e9
    return tflops_per_second


def get_memory_bandwidth_Bps(target_device: torch.device) -> int:
    # The device properties object is obtained by calling 'cudaGetDeviceProperties' CUDA
    # runtime function. We need the total memory bus width and the memory clock rate to
    # calculate the memory bandwidth.
    device_properties = torch.cuda.get_device_properties(target_device)

    # DRAM devices are Double-Data which means they provide an output at both fronts of
    # a clock beat
    bus_bytes_per_cycle = 2 * device_properties.memory_bus_width / 8
    mem_clock_rate_Hz = device_properties.memory_clock_rate * 1e3
    bytes_per_second = bus_bytes_per_cycle * mem_clock_rate_Hz * 2
    return bytes_per_second


def get_shared_memory_bandwidth_Bps(target_device: torch.device) -> int:
    # The device properties object is obtained by calling 'cudaGetDeviceProperties' CUDA
    # runtime function. Each warp can LD or ST 32 x 4 bytes per cycle. To calculate the
    # device's throughput we need to multiply with frequency and number of SMs.
    device_properties = torch.cuda.get_device_properties(target_device)
    num_sms = device_properties.multi_processor_count
    bytes_per_cycle_per_sm = 128
    bytes_per_cycle_per_device = num_sms * bytes_per_cycle_per_sm
    bytes_per_second = bytes_per_cycle_per_device * device_properties.clock_rate * 1e3
    return bytes_per_second
