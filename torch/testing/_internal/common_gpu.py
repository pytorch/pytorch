import os

import torch
from torch.testing._internal.common_cuda import (
    evaluate_gfx_arch_within,
    IS_SM100,
    PLATFORM_SUPPORTS_CUDNN_ATTENTION,
    ROCM_VERSION,
    SM100OrLater,
    SM80OrLater,
    SM90OrLater,
)
from torch.testing._internal.common_utils import (
    IS_WINDOWS,
    LazyVal,
    TEST_CUDA,
    TEST_WITH_ROCM,
    TEST_XPU,
)


def evaluate_platform_supports_flash_attention():
    if TEST_WITH_ROCM:
        arch_list = ["gfx90a", "gfx942", "gfx1100", "gfx1201", "gfx950"]
        if os.environ.get("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "0") != "0":
            arch_list += ["gfx1101", "gfx1102", "gfx1150", "gfx1151", "gfx1200"]
        return evaluate_gfx_arch_within(arch_list)
    if TEST_CUDA:
        return not IS_WINDOWS and SM80OrLater
    if TEST_XPU:
        return True
    return False


def evaluate_platform_supports_efficient_attention():
    if TEST_WITH_ROCM:
        arch_list = ["gfx90a", "gfx942", "gfx1100", "gfx1201", "gfx950"]
        if os.environ.get("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "0") != "0":
            arch_list += ["gfx1101", "gfx1102", "gfx1150", "gfx1151", "gfx1200"]
        return evaluate_gfx_arch_within(arch_list)
    if TEST_CUDA:
        return True
    if TEST_XPU:
        return True
    return False


PLATFORM_SUPPORTS_FLASH_ATTENTION: bool = LazyVal(
    lambda: evaluate_platform_supports_flash_attention()
)
PLATFORM_SUPPORTS_MEM_EFF_ATTENTION: bool = LazyVal(
    lambda: evaluate_platform_supports_efficient_attention()
)
# This condition always evaluates to PLATFORM_SUPPORTS_MEM_EFF_ATTENTION but for logical clarity we keep it separate
PLATFORM_SUPPORTS_FUSED_ATTENTION: bool = LazyVal(
    lambda: PLATFORM_SUPPORTS_FLASH_ATTENTION
    or PLATFORM_SUPPORTS_CUDNN_ATTENTION
    or PLATFORM_SUPPORTS_MEM_EFF_ATTENTION
)
PLATFORM_SUPPORTS_FUSED_SDPA: bool = TEST_CUDA and not TEST_WITH_ROCM


def evaluate_platform_supports_fp8():
    if torch.cuda.is_available():
        if torch.version.hip:
            archs = ["gfx94"]
            if ROCM_VERSION >= (6, 3):
                archs.extend(["gfx120"])
            if ROCM_VERSION >= (6, 5):
                archs.append("gfx95")
            for arch in archs:
                if arch in torch.cuda.get_device_properties(0).gcnArchName:
                    return True
        else:
            return SM90OrLater or torch.cuda.get_device_capability() == (8, 9)
    if TEST_XPU:
        return True
    # As CPU supports FP8 and is always available, return True.
    return True


def evaluate_platform_supports_bf16():
    if torch.version.cuda:
        return SM80OrLater
    elif torch.version.hip:
        return True
    elif TEST_XPU:
        return True
    return False


def evaluate_platform_supports_bf16_atomics():
    if torch.version.cuda:
        return SM80OrLater
    elif torch.version.hip:
        return ROCM_VERSION >= (8, 0)
    return False


def evaluate_platform_supports_half_atomics():
    if torch.version.hip:
        return ROCM_VERSION >= (8, 0)
    return True


PLATFORM_SUPPORTS_FP8: bool = LazyVal(lambda: evaluate_platform_supports_fp8())
PLATFORM_SUPPORTS_BF16: bool = LazyVal(lambda: evaluate_platform_supports_bf16())
PLATFORM_SUPPORTS_BF16_ATOMICS: bool = LazyVal(
    lambda: evaluate_platform_supports_bf16_atomics()
)
PLATFORM_SUPPORTS_HALF_ATOMICS: bool = LazyVal(
    lambda: evaluate_platform_supports_half_atomics()
)


def evaluate_platform_supports_fp8_grouped_gemm():
    if torch.cuda.is_available():
        if torch.version.hip:
            if "USE_MSLK" not in torch.__config__.show():
                return False
            archs = ["gfx942", "gfx950"]
            for arch in archs:
                if arch in torch.cuda.get_device_properties(0).gcnArchName:
                    return True
        else:
            return SM90OrLater and not SM100OrLater
    return False


def evaluate_platform_supports_mx_gemm():
    if torch.cuda.is_available():
        if torch.version.hip:
            if ROCM_VERSION >= (7, 0):
                return "gfx950" in torch.cuda.get_device_properties(0).gcnArchName
        else:
            return SM100OrLater
    return False


def evaluate_platform_supports_mxfp8_grouped_gemm():
    if torch.cuda.is_available() and not torch.version.hip:
        built_with_mslk = "USE_MSLK" in torch.__config__.show()
        return built_with_mslk and IS_SM100
    return False


PLATFORM_SUPPORTS_MX_GEMM: bool = LazyVal(lambda: evaluate_platform_supports_mx_gemm())
PLATFORM_SUPPORTS_FP8_GROUPED_GEMM: bool = LazyVal(
    lambda: evaluate_platform_supports_fp8_grouped_gemm()
)
PLATFORM_SUPPORTS_MXFP8_GROUPED_GEMM: bool = LazyVal(
    lambda: evaluate_platform_supports_mxfp8_grouped_gemm()
)
