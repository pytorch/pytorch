from __future__ import annotations

import dataclasses
from abc import ABC
from dataclasses import asdict
from typing import Any


@dataclasses.dataclass
class KernelTemplateParams(ABC):  # noqa: B024 # Will add abstract methods as we develop this system
    """
    Abstract base class for kernel template parameters.
    This serves as a common interface for all template parameter classes.
    """

    def kwargs(self) -> dict[str, Any]:
        """
        Returns a dictionary of all fields in the class.
        This can be used to pass the parameters as keyword arguments to functions.
        """
        return asdict(self)


@dataclasses.dataclass
class TritonTemplateMMParams(KernelTemplateParams):
    """
    Parameters for Triton MM template.
    Contains all parameters that would be passed to mm_template.maybe_append_choice
    via mm_options.
    """

    # Parameters from mm_options
    BLOCK_M: int
    BLOCK_N: int
    BLOCK_K: int
    num_stages: int
    num_warps: int
    GROUP_M: int = 8
    EVEN_K: bool = False
    ALLOW_TF32: bool = False
    USE_FAST_ACCUM: bool = False
    ACC_TYPE: str = "tl.float32"


@dataclasses.dataclass
class PersistentTMATritonTemplateMMParams(TritonTemplateMMParams):
    """
    Parameters for Persistent TMA MM template.
    Contains all parameters that would be passed to persistent_tma_mm_template.maybe_append_choice
    via mm_options and persistent_mm_options.
    """

    # Additional parameters from persistent_mm_options
    A_ROW_MAJOR: bool = True
    B_ROW_MAJOR: bool = True
    NUM_SMS: int = 0
    TMA_SIZE: int = 0
    TMA_EXPERIMENTAL_API: bool = True


@dataclasses.dataclass
class ROCmTritonTemplateMMParams(TritonTemplateMMParams):
    """
    Parameters for ROCm Triton MM template.
    Contains all parameters that would be passed to mm_template.maybe_append_choice
    via mm_options with ROCm-specific additions.
    """

    # Additional parameters for ROCm
    matrix_instr_nonkdim: int = 16
    waves_per_eu: int = 0
    kpack: int = 2


@dataclasses.dataclass
class CPUTritonTemplateKernelParams(TritonTemplateMMParams):
    """
    Parameters for CPU Triton MM template.
    Contains all parameters that would be passed to mm_template.maybe_append_choice
    via mm_options with CPU-specific additions.
    """

    # CPU-specific parameters
    scale: float = 0.5
    exclude: bool = False
