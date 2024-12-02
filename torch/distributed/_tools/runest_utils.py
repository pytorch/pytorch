import logging
import subprocess
import warnings
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch._inductor.utils import get_device_tflops, get_gpu_dram_gbps
from torch.utils._python_dispatch import is_traceable_wrapper_subclass
from torch.utils.checkpoint import SAC_IGNORED_OPS


aten = torch.ops.aten

_ADDITIONAL_IGNORED_OPS = {
    aten.lift_fresh.default,  # type: ignore[attr-defined]
    torch.ops.profiler._record_function_exit._RecordFunction,  # type: ignore[attr-defined]
    aten.clone.default,  # type: ignore[attr-defined] # seems needed for torch.compile
}
OPS_TO_ALWAYS_SKIP = SAC_IGNORED_OPS | _ADDITIONAL_IGNORED_OPS

# No fall-back kernel needed/exists for view ops
VIEW_OPS = {
    aten.lift_fresh,
    aten.t,
    aten.transpose,
    aten.view,
    aten.detach,
    aten._unsafe_view,
    aten.split,
    aten.adjoint,
    aten.as_strided,
    aten.diagonal,
    aten.expand,
    aten.expand_as,
    aten.movedim,
    aten.permute,
    aten.select,
    aten.squeeze,
    aten.mT,
    aten.mH,
    aten.real,
    aten.imag,
    aten.view_as,
    aten.unflatten,
    aten.unfold,
    aten.unbind,
    aten.unsqueeze,
    aten.vsplit,
    aten.hsplit,
    aten.split_with_sizes,
    aten.swapaxes,
    aten.swapdims,
    aten.chunk,
}
# We can ignore benchmarking tensor create ops
CREATE_OPS = {
    aten.randint,
    aten.randn,
    aten.rand,
    aten.randn_like,
    aten.rand_like,
    aten.randint_like,
    aten.arange,
    aten.ones_like,
    aten.zeros_like,
}

REDUCTION_OPS = {
    aten.native_group_norm,
    aten._softmax.default,
    aten._log_softmax,
    aten.native_layer_norm,
}

# Create a logger object
logger = logging.getLogger(__name__)

# Set the logging level to INFO
logger.setLevel(logging.INFO)

_PRESET_GPU_FAMILY = {"H100", "A100"}

_PRESET_GPU_TYPES = {
    "A100_PCIe_40GB",
    "A100_PCIe_80GB",
    "A100_SXM_40GB",
    "A100_SXM_80GB",
    "H100_NVL_94GB",
    "H100_SXM_80GB",
}

_A100_peak_flops: Dict[torch.dtype, float] = {
    torch.float64: 9.7e3,
    torch.float32: 19.5e3,
    torch.bfloat16: 312e3,
    torch.float16: 312e3,
    torch.int8: 624e3,
}

_A100_peak_factors: Dict[torch.dtype, float] = {
    torch.float16: 0.65,
    torch.bfloat16: 0.65,
    torch.float32: 0.55,
    torch.float64: 0.55,
}

# Sources:
# https://www.nvidia.com/en-us/data-center/a100/
# https://www.nvidia.com/en-us/data-center/h100/

_peak_flops_factors: Dict[str, Dict[torch.dtype, float]] = {
    "H100_SXM_80GB": {
        torch.float16: 0.75,
        torch.bfloat16: 0.75,
        torch.float32: 0.5,
        torch.float64: 0.3,
    },
    "H100_NVL_94GB": {
        torch.float16: 0.75,
        torch.bfloat16: 0.75,
        torch.float32: 0.5,
        torch.float64: 0.5,
    },
    "A100_PCIe_40GB": _A100_peak_factors,
    "A100_PCIe_80GB": _A100_peak_factors,
    "A100_SXM_40GB": _A100_peak_factors,
    "A100_SXM_80GB": _A100_peak_factors,
}

_peak_bandwidth_registry: Dict[str, float] = {
    "A100_PCIe_40GB": 1.555e3,
    "A100_PCIe_80GB": 1.935e3,
    "A100_SXM_40GB": 1.555e3,
    "A100_SXM_80GB": 2.039e3,
    "H100_NVL_94GB": 3.9e3,
    "H100_SXM_80GB": 3.35e3,
}

_peak_flops_registry: Dict[str, Dict[torch.dtype, float]] = {
    "A100_PCIe_40GB": _A100_peak_flops,
    "A100_PCIe_80GB": _A100_peak_flops,
    "A100_SXM_40GB": _A100_peak_flops,
    "A100_SXM_80GB": _A100_peak_flops,
    "H100_SXM_80GB": {
        torch.float64: 34e3,
        torch.float32: 67e3,
        torch.bfloat16: 1979e3,
        torch.float16: 1979e3,
        torch.int8: 3958e3,
    },
    "H100_NVL_94GB": {
        torch.float64: 30e3,
        torch.float32: 60e3,
        torch.bfloat16: 1671e3,
        torch.float16: 1671e3,
        torch.int8: 3341e3,
    },
}


def _display_float_dict(d: Dict[Any, float]) -> str:
    result = []
    for dtype, value in d.items():
        result.append(f"{dtype}: {value:.2e}")
    return "\n".join(result)


def _get_device_lines() -> List[str]:
    try:
        result = subprocess.run(
            ["lspci"], stdout=subprocess.PIPE, text=True, check=True
        )
        return [
            line
            for line in result.stdout.splitlines()
            if any(family in line for family in _PRESET_GPU_FAMILY)
        ]
    except FileNotFoundError:
        logger.warning("`lspci` command not found. Returning empty device lines.")
        return []
    except subprocess.CalledProcessError as e:
        logger.error("Error running lspci: %s. Returning empty device lines.", e)
        return []


def _parse_device_line(line: str) -> Tuple[str, str, str]:
    if "H100" in line:
        if "NVL" in line:
            return "H100", "NVL", "94GB"
        elif "SXM" in line:
            return "H100", "SXM", "80GB"
    elif "A100" in line:
        if "PCIe" in line:
            return "A100", "PCIe", "80GB" if "80GB" in line else "40GB"
        elif "SXM" in line:
            return "A100", "SXM", "80GB" if "80GB" in line else "40GB"
    return "", "", ""


def resolve_gpu_type(gpu_type: str = "") -> str:
    """
    Resolve GPU type from lspci output or provided string.

    Args:
        gpu_type (str, optional): GPU type string. Defaults to "".

    Returns:
        str: Resolved GPU type (e.g., "A100_PCIe_80GB") or "unknown" if unresolved.
    """
    device_lines = [gpu_type] if gpu_type else _get_device_lines()
    for line in device_lines:
        gpu_family, gpu_model_type, gpu_capacity = _parse_device_line(line)
        if gpu_family:
            gpu_type = f"{gpu_family}_{gpu_model_type}_{gpu_capacity}"
            return gpu_type if gpu_type in _PRESET_GPU_TYPES else "unknown"
    return "unknown"


def get_estimation_configs(
    gpu_type: str = "",
) -> Tuple[Dict[torch.dtype, float], Dict[torch.dtype, float], float]:
    """
    Get estimation configurations for a given GPU type.

    Args:
        gpu_type (str, optional): GPU type string. Defaults to "".

    Returns:
        Tuple[Dict[torch.dtype, float], Dict[torch.dtype, float], float]:
            Peak FLOPS (in GFLOPS/s), peak FLOPS factors, and peak bandwidth (in GB/s) for the GPU type.
    """
    if gpu_type in _PRESET_GPU_TYPES:
        peak_flops = _peak_flops_registry[gpu_type]
        peak_flops_fac = _peak_flops_factors[gpu_type]
        peak_bw = _peak_bandwidth_registry[gpu_type]
        logger.debug(
            "Found pre-existing configs for GPU\n"
            "With peak flops (GF/s): %s\n"
            "And peak bandwidth (GiB/s): %.2g",
            _display_float_dict(peak_flops),
            peak_bw,
        )
        return peak_flops, peak_flops_fac, peak_bw

    peak_flops = {}
    for dtype in [torch.float16, torch.bfloat16, torch.float32]:
        # This actually gives Peta-FLOPS/sec, multiply by 1e6 to get GFLOPS/sec
        peak_flops[dtype] = get_device_tflops(dtype) * 1e6
    peak_flops_fac = _A100_peak_factors
    peak_bw = get_gpu_dram_gbps()
    logger.debug(
        "Automatically derived configs for GPU\n"
        "With peak flops (GF/s): %s\n"
        "And peak bandwidth (GiB/s): %.2g",
        _display_float_dict(peak_flops),
        peak_bw,
    )
    return peak_flops, peak_flops_fac, peak_bw


def get_flattened_tensor(t: torch.Tensor) -> Optional[torch.Tensor]:
    """
    Recursively extracts flattened tensor from a traceable wrapper-subclass of tensor.

    Args:
        t (torch.Tensor): The tensor to extract from.

    Returns:
        torch.Tensor: A flattened tensor.
    """
    unflattened_tensors = [t]
    flattened_tensor = None
    while len(unflattened_tensors) > 0:
        obj = unflattened_tensors.pop()
        if is_traceable_wrapper_subclass(obj):
            attrs, _ = obj.__tensor_flatten__()  # type: ignore[attr-defined]
            unflattened_tensors.extend([getattr(obj, attr) for attr in attrs])
        else:
            if not hasattr(obj, "untyped_storage"):
                warnings.warn(
                    f"Expected a tensor or a traceable wrapper-subclass of tensor, but got {type(obj)}",
                    category=UserWarning,
                    stacklevel=2,
                )
            flattened_tensor = obj
            assert len(unflattened_tensors) == 0, "More than one flattened tensors"
            break
    return flattened_tensor
