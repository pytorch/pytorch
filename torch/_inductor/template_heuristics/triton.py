from __future__ import annotations

import dataclasses
import itertools
import math
import os
from functools import partial
from threading import Lock
from typing import Any, Optional, TYPE_CHECKING

import sympy

import torch
from torch._inductor.template_heuristics.triton_addmm import AddMMConfigMixin
from torch.utils._ordered_set import OrderedSet
from torch.utils._triton import has_triton_stable_tma_api

from .. import config, config as inductor_config
from ..kernel.bmm import bmm_template
from ..kernel.mm import (
    blackwell_ws_persistent_device_tma_mm_template,
    get_scaling_options,
    get_tile_size,
    mm_template,
    persistent_tma_mm_template,
    scaled_mm_device_tma_epilogue_scaling_template,
    scaled_mm_device_tma_main_loop_scaling_template,
)
from ..kernel.mm_plus_mm import mm_plus_mm_template
from ..kernel_inputs import KernelInputs, MMKernelInputs
from ..utils import (
    get_backend_num_stages,
    get_num_sms,
    get_tma_workspace_arg,
    TMA_DESCRIPTOR_SIZE,
    using_b200,
)
from ..virtualized import V
from .gemm import GemmMaxAutotuneTemplateConfigHeuristics
from .registry import register_template_heuristic


if TYPE_CHECKING:
    from collections.abc import Callable, Generator

    from triton import Config as TritonConfig

else:
    from torch._inductor.runtime.triton_compat import Config as TritonConfig


# Gemm Configs
@dataclasses.dataclass
class BaseConfig:
    """
    Base Gemm configuration used for most backends (CPU, CUDA)
    """

    block_m: int
    block_n: int
    block_k: int
    num_stages: int
    num_warps: int
    hint_override: Optional[int] = dataclasses.field(kw_only=True, default=None)


@dataclasses.dataclass
class GemmConfig(BaseConfig):
    """
    Gemm configuration used for most backends (CPU, CUDA)
    """

    group_m: int = dataclasses.field(kw_only=True, default=8)


ConvConfig = BaseConfig


@dataclasses.dataclass
class DepthwiseConvConfig:
    """
    Configuration for depthwise conv1d Triton template.
    Uses BLOCK_N x BLOCK_L x BLOCK_C tiling (channels-last NLC layout).
    Matches the hand-written NLC kernel from depthwise_conv1d_benchmark.py.
    """

    block_n: int
    block_l: int
    block_c: int
    num_stages: int
    num_warps: int


@dataclasses.dataclass
class BlackwellGPUGemmConfig(GemmConfig):
    """
    Gemm configuration used for templates with features explicitly
    targeting Nvidia Blackwell GPUs
    """

    # epilogue_subtile must be a power of 2 (1 means no subtiling)
    epilogue_subtile: int = dataclasses.field(kw_only=True, default=1)
    warp_specialize: bool = dataclasses.field(kw_only=True, default=True)
    flatten: bool = dataclasses.field(kw_only=True, default=True)


# FlexAttention Configs
@dataclasses.dataclass
class FlexConfig:
    """
    Base Config class for flex attention
    - FlexAttn forward and backward will use this. For flex decoding,
      please use FlexDecodingConfig.

    NOTE:
    For flex_attn bwd block_m and block_n are reused for block_m1, block_m2, block_n1, block_n2

    """

    block_m: int
    block_n: int
    num_stages: int
    num_warps: int


@dataclasses.dataclass
class FlexBwDConfig:
    """
    Base Config class for flex attention backward
    - FlexAttn backward will use this.

    Note: flex bwd configs

    Kernel Constraints:
      * BLOCK_N1 % BLOCK_M1 == 0
      * BLOCK_M2 % BLOCK_N2 == 0

    Pattern 1 - Symmetric Pairing (M, N, N, M):
    - Used in autotune configs
    - block_m1=M, block_n1=N, block_m2=N, block_n2=M
    - Only requires checking BLOCK_N % BLOCK_M == 0
    - Second constraint (BLOCK_M2 % BLOCK_N2) automatically satisfied

    Pattern 2 - Independent Parameters (M1, N1, M2, N2):
    - Used in exhaustive search for maximum flexibility
    - All four parameters can be set independently
    - Requires checking both constraints

    """

    block_m1: int
    block_n1: int
    block_m2: int
    block_n2: int
    num_stages: int
    num_warps: int


@dataclasses.dataclass
class FlexDecodeConfig:
    """
    Config class for flex decoding
    """

    block_n: int
    num_stages: int
    num_warps: int


# ROCm classes
@dataclasses.dataclass
class ROCmGemmConfig(GemmConfig):
    """
    ROCm subclass for GEMMs, with AMD backend specific tuneable kernargs
    """

    matrix_instr_nonkdim: int = 16
    waves_per_eu: int = 0
    kpack: int = 2


@dataclasses.dataclass
class ROCmConvConfig(ConvConfig):
    """
    ROCm subclass for Conv, with AMD backend specific tuneable kernargs
    """

    matrix_instr_nonkdim: int = 16
    waves_per_eu: int = 0
    kpack: int = 2


@dataclasses.dataclass
class ROCmFlexConfig(FlexConfig):
    """
    ROCm subclass for FlexAttn, with AMD backend specific tuneable kernargs
    """

    matrix_instr_nonkdim: int = 0
    waves_per_eu: int = 0
    kpack: int = 2


@dataclasses.dataclass
class ROCmFlexBwDConfig(FlexBwDConfig):
    """
    ROCm subclass for FlexAttn backward, with AMD backend specific tuneable kernargs
    """

    matrix_instr_nonkdim: int = 0
    waves_per_eu: int = 0
    kpack: int = 2


@dataclasses.dataclass
class ROCmFlexDecodeConfig(FlexDecodeConfig):
    """
    ROCm subclass for FlexDecode, with AMD backend specific tuneable kernargs
    """

    matrix_instr_nonkdim: int = 0
    waves_per_eu: int = 0
    kpack: int = 2


class BaseHeuristicSingleton(type):
    """
    Thread-safe implementation of single to be used in the config heuristic subclasses
    to ensure heavy __init__ calls are not repeatedly run
    """

    _instances: dict[type[Any], Any] = {}
    _lock: Lock = Lock()

    def __call__(
        cls: BaseHeuristicSingleton, *args: Any, **kwargs: Any
    ) -> BaseConfigHeuristic:
        with cls._lock:
            if cls not in cls._instances:
                instance = super().__call__()
                cls._instances[cls] = instance
            return cls._instances[cls]


class BaseConfigHeuristic(metaclass=BaseHeuristicSingleton):
    """
    Base class for mm_configs, device specific triton kernels config inherit from here
    """

    def __init__(self) -> None:
        # Whether the heuristic is used for int8. Use this when the heuristic is int8 exclusive
        # but prefer the preprocess_mm_configs argument when it's used for both
        self.has_int8_tensor: bool = False
        # Whether to scale configs at all
        # TODO(coconutruben): remove this once mm_plus_mm and tests support scaling
        self.should_scale_configs: bool = True
        # List of dictionaries to store the kernel configs. Configs that evaluate to true
        # will be utilised on the target platform. The configs are as follows:
        # (BLOCK_M, BLOCK_N, BLOCK_K, num_stages, num_warps)
        self.mm_configs: list[BaseConfig] = [
            GemmConfig(32, 32, 16, 1, 2),
            GemmConfig(32, 32, 128, 2, 4),
            GemmConfig(32, 64, 32, 5, 8),
            GemmConfig(64, 32, 32, 5, 8),
            GemmConfig(64, 32, 128, 5, 4),
            GemmConfig(64, 64, 16, 2, 4),
            GemmConfig(64, 64, 32, 2, 4),
            GemmConfig(64, 64, 64, 3, 8),
            GemmConfig(64, 64, 128, 5, 4),
            GemmConfig(64, 128, 32, 3, 4),
            GemmConfig(64, 128, 32, 4, 8),
            GemmConfig(64, 128, 64, 3, 4),
            GemmConfig(64, 128, 128, 4, 4),
            GemmConfig(128, 64, 32, 3, 4),
            GemmConfig(128, 64, 32, 4, 8),
            GemmConfig(128, 128, 32, 2, 8),
            GemmConfig(128, 128, 32, 3, 4),
            GemmConfig(128, 128, 64, 3, 4),
            GemmConfig(128, 128, 64, 5, 8),
            GemmConfig(128, 128, 128, 4, 8),
        ]

        # Exhaustive search for mm configs
        self.exhaustive_configs: list[BaseConfig] = [
            GemmConfig(
                BLOCK_M, BLOCK_N, BLOCK_K, num_stages, num_warps, group_m=group_m
            )
            for BLOCK_M, BLOCK_N, BLOCK_K in itertools.product(
                [16, 32, 64, 128, 256], repeat=3
            )
            for num_stages in [1, 2, 3, 4, 5]
            for num_warps in [2, 4, 8]
            for group_m in [8]
        ]

        # these are only used in tuned_mm when AutoHeuristic is enabled
        # the idea is that when AutoHeuristic collects data to learn a heuristic, more configs are autotuned
        # when the learned heuristic is used, the learned heuristic reduces the number of configs down to 10
        # which saves compilation time (since less configs are autotuned) and potentially increase performance
        # because the learned heuristic might predict a config that is not part mm_configs
        self.extra_mm_configs: list[BaseConfig] = [
            GemmConfig(16, 32, 16, 3, 2),
            GemmConfig(16, 32, 32, 4, 2),
            GemmConfig(16, 32, 32, 5, 2),
            GemmConfig(64, 64, 128, 3, 4),
            GemmConfig(128, 64, 32, 2, 2),
            GemmConfig(128, 64, 64, 3, 8),
            GemmConfig(128, 64, 128, 4, 8),
            GemmConfig(128, 128, 32, 4, 4),
            GemmConfig(128, 128, 64, 3, 8),
            GemmConfig(128, 128, 64, 5, 4),
        ]

        self.int8_mm_configs: list[BaseConfig] = [
            GemmConfig(64, 64, 32, 2, 4),
            GemmConfig(64, 128, 32, 3, 4),
            GemmConfig(128, 64, 32, 3, 4),
            GemmConfig(64, 128, 32, 4, 8),
            GemmConfig(128, 64, 32, 4, 8),
            GemmConfig(64, 32, 32, 5, 8),
            GemmConfig(32, 64, 32, 5, 8),
            GemmConfig(128, 128, 32, 2, 8),
            GemmConfig(64, 64, 64, 3, 8),
            GemmConfig(128, 256, 128, 3, 8),
            GemmConfig(256, 128, 128, 3, 8),
        ]

        self.mixed_mm_configs: list[BaseConfig] = [
            GemmConfig(16, 128, 256, 3, 4),
            GemmConfig(16, 128, 256, 5, 8),
        ]

        self.persistent_mm_configs: list[BaseConfig] = [
            GemmConfig(128, 256, 64, 3, 8),
            GemmConfig(128, 128, 64, 3, 8),
            GemmConfig(128, 128, 128, 3, 8),
            GemmConfig(128, 128, 128, 3, 4),
            GemmConfig(128, 128, 64, 4, 8),
            GemmConfig(128, 128, 64, 5, 8),
            GemmConfig(256, 128, 64, 4, 8),
            GemmConfig(128, 128, 64, 5, 4),
        ]

        self.blackwell_persistent_mm_configs: list[BaseConfig] = [
            BlackwellGPUGemmConfig(
                128,
                256,
                64,
                4,
                8,
                epilogue_subtile=2,
                warp_specialize=True,
                flatten=True,
            ),
            BlackwellGPUGemmConfig(
                256,
                128,
                64,
                3,
                8,
                epilogue_subtile=2,
                warp_specialize=True,
                flatten=True,
            ),
            BlackwellGPUGemmConfig(
                128,
                256,
                128,
                2,
                8,
                epilogue_subtile=2,
                warp_specialize=True,
                flatten=True,
            ),
            BlackwellGPUGemmConfig(
                128,
                256,
                64,
                3,
                8,
                epilogue_subtile=2,
                warp_specialize=True,
                flatten=True,
            ),
            BlackwellGPUGemmConfig(
                128,
                128,
                128,
                3,
                4,
                epilogue_subtile=2,
                warp_specialize=True,
                flatten=True,
            ),
            BlackwellGPUGemmConfig(
                256,
                128,
                64,
                3,
                8,
                epilogue_subtile=2,
                warp_specialize=True,
                flatten=True,
            ),
            BlackwellGPUGemmConfig(
                128,
                128,
                128,
                3,
                8,
                epilogue_subtile=2,
                warp_specialize=True,
                flatten=True,
            ),
            # Include no-subtiling. Always required for testing.
            BlackwellGPUGemmConfig(
                256,
                128,
                64,
                3,
                8,
                epilogue_subtile=1,
                warp_specialize=True,
                flatten=True,
            ),
            BlackwellGPUGemmConfig(
                128,
                128,
                128,
                3,
                8,
                epilogue_subtile=1,
                warp_specialize=True,
                flatten=True,
            ),
            # Include subtile=4. Always required for testing.
            BlackwellGPUGemmConfig(
                256,
                128,
                64,
                4,
                8,
                epilogue_subtile=4,
                warp_specialize=True,
                flatten=True,
            ),
            BlackwellGPUGemmConfig(
                128,
                128,
                128,
                4,
                8,
                epilogue_subtile=4,
                warp_specialize=True,
                flatten=True,
            ),
        ]

        self.blackwell_persistent_addmm_configs: list[BaseConfig] = [
            # Include each subtiling factor for testing.
            BlackwellGPUGemmConfig(
                256,
                128,
                64,
                2,
                4,
                epilogue_subtile=2,
                warp_specialize=True,
                flatten=True,
            ),
            BlackwellGPUGemmConfig(
                256,
                128,
                64,
                2,
                4,
                epilogue_subtile=1,
                warp_specialize=True,
                flatten=True,
            ),
            BlackwellGPUGemmConfig(
                256,
                128,
                64,
                2,
                4,
                epilogue_subtile=4,
                warp_specialize=True,
                flatten=True,
            ),
        ]

        self.scaled_mm_configs: list[BaseConfig] = [
            GemmConfig(128, 256, 32, 3, 8),
            GemmConfig(256, 128, 32, 3, 8),
            GemmConfig(256, 64, 32, 4, 4),
            GemmConfig(64, 256, 32, 4, 4),
            GemmConfig(128, 128, 32, 4, 4),
            GemmConfig(128, 64, 32, 4, 4),
            GemmConfig(64, 128, 32, 4, 4),
            GemmConfig(128, 32, 32, 4, 4),
            GemmConfig(64, 32, 32, 5, 2),
            GemmConfig(256, 128, 128, 3, 8),
            GemmConfig(256, 64, 128, 4, 4),
            GemmConfig(64, 256, 128, 4, 4),
            GemmConfig(128, 128, 128, 4, 4),
            GemmConfig(128, 64, 64, 4, 4),
            GemmConfig(64, 128, 64, 4, 4),
            GemmConfig(128, 32, 64, 4, 4),
            GemmConfig(64, 32, 64, 5, 2),
            GemmConfig(16, 32, 32, 2, 2),
            GemmConfig(16, 64, 32, 2, 2),
            GemmConfig(16, 128, 32, 2, 4),
            GemmConfig(16, 256, 32, 2, 4),
            GemmConfig(16, 32, 64, 2, 2),
            GemmConfig(16, 64, 64, 2, 2),
            GemmConfig(16, 128, 64, 2, 4),
            GemmConfig(16, 256, 64, 2, 4),
            GemmConfig(32, 32, 32, 2, 2),
            GemmConfig(32, 64, 32, 2, 2),
            GemmConfig(32, 128, 32, 2, 4),
            GemmConfig(32, 256, 32, 2, 4),
            GemmConfig(32, 32, 64, 2, 2),
            GemmConfig(32, 64, 64, 2, 2),
            GemmConfig(32, 128, 64, 2, 4),
            GemmConfig(32, 256, 64, 2, 4),
            GemmConfig(16, 32, 32, 3, 2),
            GemmConfig(16, 64, 32, 3, 2),
            GemmConfig(16, 128, 32, 3, 4),
            GemmConfig(16, 256, 32, 3, 4),
            GemmConfig(16, 32, 64, 3, 2),
            GemmConfig(16, 64, 64, 3, 2),
            GemmConfig(16, 128, 64, 3, 4),
            GemmConfig(16, 256, 64, 3, 4),
            GemmConfig(32, 32, 32, 3, 2),
            GemmConfig(32, 64, 32, 3, 2),
            GemmConfig(32, 128, 32, 3, 4),
            GemmConfig(32, 256, 32, 3, 4),
            GemmConfig(32, 32, 64, 3, 2),
            GemmConfig(32, 64, 64, 3, 2),
            GemmConfig(32, 128, 64, 3, 4),
            GemmConfig(32, 256, 64, 3, 4),
            GemmConfig(16, 32, 32, 4, 2),
            GemmConfig(16, 64, 32, 4, 2),
            GemmConfig(16, 128, 32, 4, 4),
            GemmConfig(16, 256, 32, 4, 4),
            GemmConfig(16, 32, 64, 4, 2),
            GemmConfig(16, 64, 64, 4, 2),
            GemmConfig(16, 128, 64, 4, 4),
            GemmConfig(16, 256, 64, 4, 4),
            GemmConfig(32, 32, 32, 4, 2),
            GemmConfig(32, 64, 32, 4, 2),
            GemmConfig(32, 128, 32, 4, 4),
            GemmConfig(32, 256, 32, 4, 4),
            GemmConfig(32, 32, 64, 4, 2),
            GemmConfig(32, 64, 64, 4, 2),
            GemmConfig(32, 128, 64, 4, 4),
            GemmConfig(32, 256, 64, 4, 4),
            GemmConfig(16, 32, 32, 5, 2),
            GemmConfig(16, 64, 32, 5, 2),
            GemmConfig(16, 128, 32, 5, 4),
            GemmConfig(16, 256, 32, 5, 4),
            GemmConfig(16, 32, 64, 5, 2),
            GemmConfig(16, 64, 64, 5, 2),
            GemmConfig(16, 128, 64, 5, 4),
            GemmConfig(16, 256, 64, 5, 4),
            GemmConfig(32, 32, 32, 5, 2),
            GemmConfig(32, 64, 32, 5, 2),
            GemmConfig(32, 128, 32, 5, 4),
            GemmConfig(32, 256, 32, 5, 4),
            GemmConfig(32, 32, 64, 5, 2),
            GemmConfig(32, 64, 64, 5, 2),
            GemmConfig(32, 128, 64, 5, 4),
            GemmConfig(32, 256, 64, 5, 4),
            GemmConfig(16, 32, 32, 6, 2),
            GemmConfig(16, 64, 32, 6, 2),
            GemmConfig(16, 128, 32, 6, 4),
            GemmConfig(16, 256, 32, 6, 4),
            GemmConfig(16, 32, 64, 6, 2),
            GemmConfig(16, 64, 64, 6, 2),
            GemmConfig(16, 128, 64, 6, 4),
            GemmConfig(16, 256, 64, 6, 4),
            GemmConfig(32, 32, 32, 6, 2),
            GemmConfig(32, 64, 32, 6, 2),
            GemmConfig(32, 128, 32, 6, 4),
            GemmConfig(32, 256, 32, 6, 4),
            GemmConfig(32, 32, 64, 6, 2),
            GemmConfig(32, 64, 64, 6, 2),
            GemmConfig(32, 128, 64, 6, 4),
            GemmConfig(32, 256, 64, 6, 4),
            GemmConfig(64, 16, 256, 5, 4),
            GemmConfig(64, 32, 256, 5, 4),
            GemmConfig(64, 128, 128, 2, 4),
            GemmConfig(64, 128, 128, 3, 4),
            GemmConfig(128, 128, 128, 2, 4),
            GemmConfig(128, 256, 128, 4, 8),
            GemmConfig(256, 128, 128, 2, 4),
            GemmConfig(256, 128, 128, 2, 8),
        ]

        self.scaled_persistent_mm_configs: list[BaseConfig] = [
            GemmConfig(128, 128, 64, 3, 8),
            GemmConfig(128, 128, 128, 3, 8),
            GemmConfig(128, 128, 128, 4, 8),
            GemmConfig(128, 128, 128, 4, 4),
            GemmConfig(128, 128, 128, 3, 4),
            GemmConfig(128, 128, 128, 5, 4),
            GemmConfig(128, 128, 128, 5, 8),
            GemmConfig(128, 128, 128, 6, 8),
            GemmConfig(128, 128, 64, 4, 8),
            GemmConfig(64, 32, 256, 5, 4),
            GemmConfig(128, 256, 128, 3, 8),
            GemmConfig(64, 128, 256, 4, 4),
            GemmConfig(64, 256, 128, 4, 4),
        ]

        self.blackwell_scaled_persistent_mm_configs = [
            BlackwellGPUGemmConfig(
                block_m=c.block_m,
                block_n=c.block_n,
                block_k=c.block_k,
                num_stages=c.num_stages,
                num_warps=c.num_warps,
                hint_override=c.hint_override,
                group_m=8,
                epilogue_subtile=2,
                warp_specialize=True,
                flatten=True,
            )
            for c in self.scaled_persistent_mm_configs
        ]

        # TODO: Unify with other gemm patterns, mm_plus_mm currently follows
        # slightly different pattern than rest
        self.mm_plus_mm_configs: list[BaseConfig] = [
            GemmConfig(64, 64, 32, 2, 4),
            GemmConfig(64, 64, 32, 3, 8),
            GemmConfig(64, 64, 32, 4, 16),
            GemmConfig(64, 32, 32, 4, 8),
            GemmConfig(32, 64, 32, 4, 8),
            GemmConfig(128, 128, 32, 1, 8),
            GemmConfig(64, 64, 64, 1, 8),
            GemmConfig(32, 32, 128, 1, 8),
            GemmConfig(64, 64, 16, 2, 4),
            GemmConfig(32, 32, 16, 1, 2),
        ]

        self.conv_configs: list[BaseConfig] = [
            # BLOCK_K=16 configs
            ConvConfig(64, 256, 16, 2, 4),
            ConvConfig(256, 64, 16, 2, 4),
            ConvConfig(1024, 16, 16, 1, 8),
            # BLOCK_K=32 configs
            ConvConfig(128, 128, 32, 2, 8),
            ConvConfig(64, 64, 32, 2, 4),
            ConvConfig(64, 256, 32, 2, 8),
            ConvConfig(256, 64, 32, 2, 8),
            # BLOCK_K=64 configs
            ConvConfig(128, 128, 64, 3, 8),
            ConvConfig(64, 128, 64, 4, 4),
            ConvConfig(128, 64, 64, 4, 4),
            ConvConfig(256, 128, 64, 2, 8),
            ConvConfig(128, 256, 64, 2, 8),
            # BLOCK_K=128 configs - optimal when IN_C=128 (single iteration over channels)
            ConvConfig(128, 128, 128, 2, 8),
            ConvConfig(128, 128, 128, 3, 8),
            ConvConfig(64, 128, 128, 4, 4),
            ConvConfig(256, 128, 128, 2, 8),
            ConvConfig(128, 256, 128, 2, 8),
        ]

        # Depthwise conv1d configs: BLOCK_N x BLOCK_L x BLOCK_C tiling
        # Derived from autotuning results on H100 for depthwise conv1d
        # channels-last (NLC) layout with shape x=[3072, 128, 202]
        # Matches _nlc_autotune_configs from depthwise_conv1d_benchmark.py
        self.depthwise_conv_configs: list[DepthwiseConvConfig] = [
            # BLOCK_C=32, BLOCK_L=32
            DepthwiseConvConfig(
                block_n=16, block_l=32, block_c=32, num_stages=4, num_warps=8
            ),
            DepthwiseConvConfig(
                block_n=16, block_l=32, block_c=32, num_stages=4, num_warps=4
            ),
            DepthwiseConvConfig(
                block_n=32, block_l=32, block_c=32, num_stages=5, num_warps=8
            ),
            DepthwiseConvConfig(
                block_n=32, block_l=32, block_c=32, num_stages=4, num_warps=4
            ),
            # BLOCK_C=32, BLOCK_L=64
            DepthwiseConvConfig(
                block_n=16, block_l=64, block_c=32, num_stages=4, num_warps=8
            ),
            DepthwiseConvConfig(
                block_n=16, block_l=64, block_c=32, num_stages=4, num_warps=4
            ),
            DepthwiseConvConfig(
                block_n=32, block_l=64, block_c=32, num_stages=3, num_warps=8
            ),
            # BLOCK_C=32, BLOCK_L=256
            DepthwiseConvConfig(
                block_n=16, block_l=256, block_c=32, num_stages=5, num_warps=8
            ),
            DepthwiseConvConfig(
                block_n=16, block_l=256, block_c=32, num_stages=4, num_warps=4
            ),
            DepthwiseConvConfig(
                block_n=32, block_l=256, block_c=32, num_stages=3, num_warps=8
            ),
            # BLOCK_C=64
            DepthwiseConvConfig(
                block_n=16, block_l=32, block_c=64, num_stages=4, num_warps=8
            ),
            DepthwiseConvConfig(
                block_n=16, block_l=32, block_c=64, num_stages=4, num_warps=4
            ),
            DepthwiseConvConfig(
                block_n=16, block_l=64, block_c=64, num_stages=3, num_warps=8
            ),
            # BLOCK_C=128
            DepthwiseConvConfig(
                block_n=16, block_l=32, block_c=128, num_stages=3, num_warps=8
            ),
            DepthwiseConvConfig(
                block_n=16, block_l=32, block_c=128, num_stages=3, num_warps=4
            ),
        ]

        self.flex_attn_fwd_autotune_configs: list[FlexConfig] = [
            FlexConfig(128, 64, 3, 4),
            FlexConfig(128, 128, 3, 4),
            FlexConfig(128, 128, 2, 8),
            FlexConfig(128, 128, 1, 8),
            FlexConfig(64, 128, 3, 4),
            FlexConfig(64, 64, 3, 4),
        ]

        self.flex_attn_bwd_autotune_configs: list[FlexBwDConfig] = [
            # See Note: flex bwd configs
            FlexBwDConfig(BLOCK_M, BLOCK_N, BLOCK_N, BLOCK_M, s, w)
            for BLOCK_M in [32, 64]
            for BLOCK_N in [32, 64, 128]
            for s in [1, 3, 4, 5]  # num_stages
            for w in ([4, 8] if BLOCK_M >= 128 or BLOCK_N >= 128 else [4])
            if BLOCK_N % BLOCK_M == 0
        ]

        self.flex_decode_autotune_configs: list[FlexDecodeConfig] = [
            FlexDecodeConfig(64, 3, 2),
            FlexDecodeConfig(32, 3, 2),
            FlexDecodeConfig(128, 3, 2),
        ]

        self.exhaustive_flex_attn_fwd_configs: list[FlexConfig] = [
            FlexConfig(BLOCK_M, BLOCK_N, num_stages, num_warps)
            for BLOCK_M in [16, 32, 64, 128]
            for BLOCK_N in [32, 64, 128]
            for num_stages in [1, 3, 4, 5]
            for num_warps in [2, 4, 8]
        ]

        self.exhaustive_flex_attn_bwd_configs: list[FlexBwDConfig] = [
            # See Note: flex bwd configs
            FlexBwDConfig(BLOCK_M1, BLOCK_N1, BLOCK_M2, BLOCK_N2, num_stages, num_warps)
            for BLOCK_M1 in [16, 32, 64, 128]
            for BLOCK_N1 in [16, 32, 64, 128]
            for BLOCK_M2 in [16, 32, 64, 128]
            for BLOCK_N2 in [16, 32, 64, 128]
            for num_stages in [1, 3, 4]
            for num_warps in [2, 4, 8]
            if BLOCK_N1 % BLOCK_M1 == 0
            and BLOCK_M2 % BLOCK_N2 == 0  # kernel static assertions
        ]

        self.exhaustive_flex_decode_configs: list[FlexDecodeConfig] = [
            FlexDecodeConfig(block_n, num_stages, num_warps)
            for block_n in [16, 32, 64, 128]
            for num_stages in [1, 3, 4, 5]
            for num_warps in [2, 4, 8]
        ]

    def _finalize_mm_configs(
        self,
        configs: list[BaseConfig],
    ) -> Generator[TritonConfig, None, None]:
        """
        Finalizes configs after scaling, applying additional constraints.
        """
        used: OrderedSet[tuple[Optional[int], ...]] = OrderedSet()

        max_mm_configs = config.test_configs.max_mm_configs

        for conf in configs:
            # Each warp computes a 16x16 tile = 256 elements
            num_warps = min(conf.num_warps, conf.block_m * conf.block_n // 256)

            # Construct key for finding duplicate configs
            key: tuple[Optional[int], ...] = (
                conf.block_m,
                conf.block_n,
                conf.block_k,
                conf.num_stages,
                conf.hint_override,
                num_warps,
            )

            # Check if gemm specific arg exists - add to key if does
            group_m = getattr(conf, "group_m", None)
            if group_m is not None:
                key += (group_m,)

            # Add BlackwellGPUGemmConfig specific fields to key if present
            if isinstance(conf, BlackwellGPUGemmConfig):
                key += (conf.epilogue_subtile, conf.warp_specialize, conf.flatten)

            # Add TlxGemmConfig specific fields to key if present
            if config.is_fbcode() and config.triton.enable_tlx_templates:
                from torch._inductor.fb.tlx_templates.registry import (
                    get_tlx_config_key_and_kwargs,
                )

                tlx_key_fields, tlx_kwargs = get_tlx_config_key_and_kwargs(conf)
                key += tlx_key_fields
            else:
                tlx_kwargs = {}

            if key not in used and (
                max_mm_configs is None or len(used) < max_mm_configs
            ):
                used.add(key)
                kwargs: dict[str, Any] = {
                    "BLOCK_M": conf.block_m,
                    "BLOCK_N": conf.block_n,
                    "BLOCK_K": conf.block_k,
                    "hint_override": conf.hint_override,
                }
                if group_m is not None:
                    kwargs["GROUP_M"] = group_m

                # Add BlackwellGPUGemmConfig specific fields if present
                if isinstance(conf, BlackwellGPUGemmConfig):
                    kwargs["EPILOGUE_SUBTILE"] = conf.epilogue_subtile
                    kwargs["WARP_SPECIALIZE"] = conf.warp_specialize
                    kwargs["FLATTEN"] = conf.flatten

                # Add TlxGemmConfig specific fields if present
                kwargs.update(tlx_kwargs)

                yield self.triton_config(conf.num_stages, num_warps, **kwargs)

    def _scale_mm_configs(
        self,
        m: int,
        n: int,
        k: int,
        configs: list[BaseConfig],
        scale: float,
        has_int8_tensor: bool,
        exclude: Callable[[sympy.Integer, sympy.Integer, sympy.Integer], bool],
        hint_override: Optional[int] = None,
    ) -> list[BaseConfig]:
        """
        Scales and filters matrix multiplication configs based on input size.
        """
        if not self.should_scale_configs:
            return configs
        from ..runtime.runtime_utils import next_power_of_2

        min_block_size = 16
        min_block_size_k = 32 if (has_int8_tensor or self.has_int8_tensor) else 16

        scaled_configs = []
        for hint_override in [None] + config.multi_kernel_hints:
            m_hint = max(
                next_power_of_2(
                    V.graph.sizevars.optimization_hint_with_override(
                        m,
                        hint_override=hint_override,
                    )
                ),
                min_block_size,
            )
            n_hint = max(
                next_power_of_2(
                    V.graph.sizevars.optimization_hint_with_override(
                        n,
                        hint_override=hint_override,
                    )
                ),
                min_block_size,
            )
            k_hint = max(
                next_power_of_2(
                    V.graph.sizevars.optimization_hint_with_override(
                        k,
                        hint_override=hint_override,
                    )
                ),
                min_block_size_k,
            )

            for c in configs:
                block_m = max(min(int(c.block_m * scale), m_hint), min_block_size)
                block_n = max(min(int(c.block_n * scale), n_hint), min_block_size)
                block_k = max(min(int(c.block_k * scale), k_hint), min_block_size_k)
                if not exclude(block_m, block_n, block_k):
                    # This copy is expensive, so avoid it if we can.
                    if (block_m, block_n, block_k, hint_override) != (
                        c.block_m,
                        c.block_n,
                        c.block_k,
                        c.hint_override,
                    ):
                        c = dataclasses.replace(
                            c,
                            block_m=block_m,
                            block_n=block_n,
                            block_k=block_k,
                            hint_override=hint_override,
                        )

                    scaled_configs.append(c)

        return scaled_configs

    # Estimate theoretical maximum shared memory
    def get_shared_memory_estimation(
        self,
        gemm_config: BaseConfig,
        dtype_size: int,
        has_sm_layout_conversion: bool,
        layout_conversion_byte_size: int,
    ):
        shared_mem_loads = dtype_size * (
            gemm_config.block_m * gemm_config.block_k
            + gemm_config.block_n * gemm_config.block_k
        )

        # Extra bytes to account for barriers in boundary conditions
        extra_bytes = 128

        # In persistent tma case, the layout conversion from mma -> blocked layout
        # is not free and takes additional shared memory, while next loads are prefetched
        # For addmm, the conversion is in the acc dtype, as it is needed before the bias addition
        # For mm, the conversion is in the output dtype, as it happens before the store
        if has_sm_layout_conversion:
            element_bits = layout_conversion_byte_size * 8
            # 8 bytes of padding for fp16/bf16
            max_padding = 128 // element_bits
            block_n = max_padding + gemm_config.block_n
            shared_mem_epilogue = (
                layout_conversion_byte_size * gemm_config.block_m * block_n
            )
        else:
            shared_mem_epilogue = 0

        return (
            shared_mem_loads * gemm_config.num_stages
            + shared_mem_epilogue
            + extra_bytes
        )

    def _get_exceeding_shared_memory_checker(
        self,
        has_sm_layout_conversion: bool,
        layout_conversion_byte_size: int,
    ) -> Optional[Callable[[BaseConfig, int], bool]]:
        """
        Returns a function that checks whether a given configuration exceeds the available shared memory for the device.
        based on the config's theoretical maximum shared memory used.
        If the device does not report available shared memory, returns None.
        """

        try:
            device = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(device)
            if hasattr(props, "shared_memory_per_block_optin"):  # for NVidia GPUs
                sm_available = int(props.shared_memory_per_block_optin)
            elif hasattr(props, "shared_memory_per_block"):  # for ROCm
                sm_available = int(props.shared_memory_per_block)
            else:
                return None

        except Exception:
            # If CUDA is not available or properties cannot be queried, return None
            return None

        # TODO make a BaseDeviceConfigHeuristics to handle different device configuration in its own implementation.
        def exceeds(gemm_config: BaseConfig, dtype_size: int) -> bool:
            estimation = self.get_shared_memory_estimation(
                gemm_config,
                dtype_size,
                has_sm_layout_conversion,
                layout_conversion_byte_size,
            )
            return estimation > sm_available

        return exceeds

    def _prune_exceeding_max_shared_mem_configs(
        self,
        configs: list[BaseConfig],
        dtype_size: int,
        has_sm_layout_conversion: bool = False,
        layout_conversion_byte_size: int = 0,
    ) -> list[BaseConfig]:
        if dtype_size <= 0:
            return configs

        is_exceeding_shared_memory = self._get_exceeding_shared_memory_checker(
            has_sm_layout_conversion, layout_conversion_byte_size
        )
        if is_exceeding_shared_memory is None:
            return configs

        return [c for c in configs if not is_exceeding_shared_memory(c, dtype_size)]

    def _prune_reg_spill_configs(
        self,
        configs: list[BaseConfig],
    ) -> list[BaseConfig]:
        pruned_configs = []
        for gemm_config in configs:
            NUM_REG = 255
            acc_regs = math.ceil(
                gemm_config.block_m * gemm_config.block_n / (gemm_config.num_warps * 32)
            )
            # Lower bound for register spillage, if exceeds the kernel will certainly spill
            if acc_regs > NUM_REG:
                continue

            pruned_configs.append(gemm_config)

        return pruned_configs

    def _filter_configs(self, configs: list[BaseConfig]) -> list[BaseConfig]:
        """
        Filter configs based on specific requirements.
        Subclasses can override this to implement custom filtering logic.
        """
        return configs

    def preprocess_mm_configs(
        self,
        m: int,
        n: int,
        k: int,
        configs: list[BaseConfig],
        has_int8_tensor: bool = False,
        scale: float = 1.0,
        exclude: Callable[
            [sympy.Integer, sympy.Integer, sympy.Integer], bool
        ] = lambda m, n, k: False,
        dtype_size: int = 0,
        op_name: str = "mm",  # For preprocessing overrides e.g. on CPU
        **kwargs,
    ) -> Generator[TritonConfig, None, None]:
        configs = self._filter_configs(configs)
        scaled_configs = self._scale_mm_configs(
            m, n, k, configs, scale, has_int8_tensor, exclude
        )

        # Filter out configs that require more shared memory than is available.
        # Theoretical upper bound, will over-prune configs. Off by default for maximum
        # performance
        if config.max_autotune_prune_choices_based_on_shared_mem:
            scaled_configs = self._prune_exceeding_max_shared_mem_configs(
                scaled_configs,
                dtype_size,
                kwargs.get("has_sm_layout_conversion", False),
                kwargs.get("layout_conversion_byte_size", 0),
            )

        if config.max_autotune_gemm_search_space == "EXHAUSTIVE":
            scaled_configs = self._prune_reg_spill_configs(scaled_configs)
        return self._finalize_mm_configs(scaled_configs)

    def triton_config(
        self, num_stages: int, num_warps: int, **kwargs: Any
    ) -> TritonConfig:
        return TritonConfig(kwargs, num_stages=num_stages, num_warps=num_warps)

    def get_mm_configs(self) -> partial[Generator[TritonConfig, None, None]]:
        return partial(self.preprocess_mm_configs, configs=self.mm_configs)

    def get_exhaustive_mm_configs(self) -> partial[Generator[TritonConfig, None, None]]:
        return partial(self.preprocess_mm_configs, configs=self.exhaustive_configs)

    def get_conv_configs(self) -> partial[Generator[TritonConfig, None, None]]:
        return partial(
            self.preprocess_mm_configs, configs=self.conv_configs, op_name="conv"
        )

    def get_depthwise_conv_configs(self) -> list[TritonConfig]:
        """Return TritonConfig list for depthwise conv1d autotuning."""
        return [
            TritonConfig(
                {
                    "BLOCK_N": cfg.block_n,
                    "BLOCK_L": cfg.block_l,
                    "BLOCK_C": cfg.block_c,
                },
                num_stages=cfg.num_stages,
                num_warps=cfg.num_warps,
            )
            for cfg in self.depthwise_conv_configs
        ]

    # Flex attn helpers
    def get_flex_attn_fwd_configs(self, head_dim: int, dtype: Any) -> list[FlexConfig]:
        flex_attn_fwd_configs: list[FlexConfig] = []

        if config.max_autotune:
            if config.max_autotune_flex_search_space == "EXHAUSTIVE":
                return self.exhaustive_flex_attn_fwd_configs
            flex_attn_fwd_configs += self.flex_attn_fwd_autotune_configs

        if head_dim <= 256:
            if dtype == torch.float32:
                default_config = FlexConfig(64, 64, 3, 4)
            else:
                default_config = FlexConfig(128, 64, 3, 4)
        else:
            if dtype == torch.float32:
                default_config = FlexConfig(32, 16, 3, 4)
            else:
                default_config = FlexConfig(64, 32, 3, 4)

        if default_config not in flex_attn_fwd_configs:
            flex_attn_fwd_configs.append(default_config)

        return flex_attn_fwd_configs

    def get_flex_attn_bwd_configs(
        self, head_dim: int, dtype: Any
    ) -> list[FlexBwDConfig]:
        flex_attn_bwd_configs: list[FlexBwDConfig] = []

        if config.max_autotune:
            if config.max_autotune_flex_search_space == "EXHAUSTIVE":
                return self.exhaustive_flex_attn_bwd_configs
            flex_attn_bwd_configs += self.flex_attn_bwd_autotune_configs

        default_config = FlexBwDConfig(16, 16, 16, 16, 1, 4)

        if default_config not in flex_attn_bwd_configs:
            flex_attn_bwd_configs.append(default_config)

        return flex_attn_bwd_configs

    def get_flex_decode_configs(
        self, head_dim: int, dtype: Any
    ) -> list[FlexDecodeConfig]:
        flex_decode_configs: list[FlexDecodeConfig] = []

        if config.max_autotune:
            if config.max_autotune_flex_search_space == "EXHAUSTIVE":
                return self.exhaustive_flex_decode_configs
            flex_decode_configs += self.flex_decode_autotune_configs

        default_config = FlexDecodeConfig(block_n=64, num_stages=1, num_warps=2)

        if default_config not in flex_decode_configs:
            flex_decode_configs.append(default_config)

        return flex_decode_configs


class CPUConfigHeuristic(BaseConfigHeuristic):
    """
    CPU-specific config heuristic with CPU-specific optimizations.
    """

    def _get_cpu_exclude_function(
        self, method: str = "bmm"
    ) -> Callable[[sympy.Integer, sympy.Integer, sympy.Integer], bool]:
        """
        Get CPU-specific exclude function based on method type.
        Returns a function that can be used as exclude condition.
        Moved from mm_common._is_large_block_for_cpu and refactored to return a function.
        """
        if method in ("conv"):

            def exclude_conv(
                m: sympy.Integer, n: sympy.Integer, k: sympy.Integer
            ) -> bool:
                # Thresholds are experimentally determined to reduce Triton CPU compile times
                if m > 256 or n > 256 or k > 256:
                    return True
                return m * n * k > 2**17

            return exclude_conv
        elif method in ("mm", "addmm", "int_mm"):

            def exclude_mm(
                m: sympy.Integer, n: sympy.Integer, k: sympy.Integer
            ) -> bool:
                return m * n > 2**13

            return exclude_mm
        else:  # Default to bmm implementation for unknown methods

            def exclude_bmm(
                m: sympy.Integer, n: sympy.Integer, k: sympy.Integer
            ) -> bool:
                if m > 128 or n > 128 or k > 128:
                    return True
                return m * n > 2**12

            return exclude_bmm

    def preprocess_mm_configs(
        self,
        m: int,
        n: int,
        k: int,
        configs: list[BaseConfig],
        has_int8_tensor: bool = False,
        scale: float = 1.0,
        exclude: Callable[
            [sympy.Integer, sympy.Integer, sympy.Integer], bool
        ] = lambda m, n, k: False,
        dtype_size: int = 0,
        op_name: str = "mm",  # For preprocessing overrides e.g. on CPU
        **kwargs,
    ) -> Generator[TritonConfig, None, None]:
        """
        CPU-specific preprocessing that applies CPU-specific scaling (0.5) and exclusion logic.
        """
        # Get CPU-specific exclude function based on operation type
        cpu_exclude_fn = self._get_cpu_exclude_function(op_name)

        # Apply CPU-specific scaling (0.5) and exclusion logic
        return super().preprocess_mm_configs(
            m,
            n,
            k,
            configs=configs,
            has_int8_tensor=has_int8_tensor,
            scale=0.5,
            exclude=cpu_exclude_fn,
            dtype_size=dtype_size,
            op_name=op_name,
            **kwargs,
        )


class CUDAConfigHeuristic(BaseConfigHeuristic):
    """
    Child class for CUDA device specific gemm/flex attention/conv/ configs.
    """

    def __init__(self) -> None:
        super().__init__()
        self.sm_120_default_flex_config = {
            (torch.float32, 64): FlexConfig(128, 32, 2, 4),
            (torch.float32, 128): FlexConfig(128, 32, 2, 4),
            (torch.float32, 256): FlexConfig(64, 16, 2, 4),
            (torch.bfloat16, 64): FlexConfig(128, 64, 2, 4),
            (torch.bfloat16, 128): FlexConfig(128, 64, 2, 8),
            (torch.bfloat16, 256): FlexConfig(32, 64, 2, 4),
            (torch.float16, 64): FlexConfig(128, 64, 2, 4),
            (torch.float16, 128): FlexConfig(128, 64, 2, 8),
            (torch.float16, 256): FlexConfig(32, 64, 2, 4),
        }

        self.sm_100_default_flex_config = {
            (torch.float32, 64): FlexConfig(128, 32, 3, 4),
            (torch.float32, 128): FlexConfig(32, 64, 3, 4),
            (torch.float32, 192): FlexConfig(32, 64, 2, 4),
            (torch.float32, 256): FlexConfig(32, 32, 3, 4),
            (torch.bfloat16, 64): FlexConfig(128, 128, 3, 4),
            (torch.bfloat16, 128): FlexConfig(128, 64, 3, 8),
            (torch.bfloat16, 192): FlexConfig(128, 128, 1, 8),
            (torch.bfloat16, 256): FlexConfig(64, 32, 3, 4),
            (torch.float16, 64): FlexConfig(128, 128, 3, 4),
            (torch.float16, 128): FlexConfig(128, 64, 3, 8),
            (torch.float16, 192): FlexConfig(128, 128, 1, 8),
            (torch.float16, 256): FlexConfig(64, 32, 3, 4),
        }

        self.h100_default_flex_config = {
            (torch.float32, 64): FlexConfig(128, 32, 3, 4),
            (torch.float32, 128): FlexConfig(32, 64, 3, 4),
            (torch.float32, 256): FlexConfig(32, 32, 3, 4),
            (torch.bfloat16, 64): FlexConfig(128, 128, 3, 4),
            (torch.bfloat16, 128): FlexConfig(128, 64, 3, 8),
            (torch.bfloat16, 256): FlexConfig(64, 32, 3, 4),
            (torch.float16, 64): FlexConfig(128, 128, 3, 4),
            (torch.float16, 128): FlexConfig(128, 64, 3, 8),
            (torch.float16, 256): FlexConfig(64, 32, 3, 4),
        }

        self.a100_default_flex_config = {
            (torch.float32, 64): FlexConfig(128, 32, 3, 4),
            (torch.float32, 128): FlexConfig(128, 32, 3, 4),
            (torch.float32, 256): FlexConfig(64, 16, 3, 4),
            (torch.bfloat16, 64): FlexConfig(128, 64, 3, 4),
            (torch.bfloat16, 128): FlexConfig(128, 64, 3, 8),
            (torch.bfloat16, 256): FlexConfig(32, 64, 3, 4),
            (torch.float16, 64): FlexConfig(128, 64, 3, 4),
            (torch.float16, 128): FlexConfig(128, 64, 3, 8),
            (torch.float16, 256): FlexConfig(32, 64, 3, 4),
        }

        # Overwriting the configs omitting BLOCK_N of size 128 that cause ULFs
        self.flex_attn_bwd_autotune_configs: list[FlexBwDConfig] = [
            # See Note: flex bwd configs
            FlexBwDConfig(BLOCK_M, BLOCK_N, BLOCK_N, BLOCK_M, s, 4)
            for BLOCK_M in [32, 64]
            for BLOCK_N in [32, 64]
            for s in [1, 3, 4, 5]  # num_stages
            if BLOCK_N % BLOCK_M == 0
        ]

    def get_flex_attn_fwd_configs(self, head_dim: int, dtype: Any) -> list[FlexConfig]:
        capability = torch.cuda.get_device_capability()
        flex_attn_fwd_configs: list[FlexConfig] = []

        if config.max_autotune:
            if config.max_autotune_flex_search_space == "EXHAUSTIVE":
                return self.exhaustive_flex_attn_fwd_configs
            flex_attn_fwd_configs += self.flex_attn_fwd_autotune_configs

        if head_dim <= 256:
            if dtype == torch.float32:
                default_config = FlexConfig(64, 64, 3, 4)
            else:
                default_config = FlexConfig(64, 64, 3, 4)
            # here we are using sm_120_default_flex_config on THOR as well
            if capability >= (11, 0):
                default_config = self.sm_120_default_flex_config.get(
                    (dtype, head_dim), default_config
                )
            elif capability >= (10, 0):
                default_config = self.sm_100_default_flex_config.get(
                    (dtype, head_dim), default_config
                )
            elif capability == (9, 0):
                default_config = self.h100_default_flex_config.get(
                    (dtype, head_dim), default_config
                )
            elif capability >= (8, 0):
                default_config = self.a100_default_flex_config.get(
                    (dtype, head_dim), default_config
                )
        else:
            if dtype == torch.float32:
                default_config = FlexConfig(32, 16, 3, 4)
            else:
                default_config = FlexConfig(64, 32, 3, 4)

        if default_config not in flex_attn_fwd_configs:
            flex_attn_fwd_configs.append(default_config)

        return flex_attn_fwd_configs

    def get_flex_attn_bwd_configs(
        self, head_dim: int, dtype: Any
    ) -> list[FlexBwDConfig]:
        capability = torch.cuda.get_device_capability()
        flex_attn_bwd_configs: list[FlexBwDConfig] = []
        if config.max_autotune:
            if config.max_autotune_flex_search_space == "EXHAUSTIVE":
                return self.exhaustive_flex_attn_bwd_configs
            flex_attn_bwd_configs += self.flex_attn_bwd_autotune_configs

        major, minor = capability
        if dtype == torch.float32:
            capability_class = "float32"
        elif major == 12:
            capability_class = "sm12x"
        elif major == 11:
            capability_class = "sm11x"
        elif major >= 10:
            capability_class = "sm10x"
        elif capability == (9, 0):
            capability_class = "sm90"
        elif major >= 8:
            capability_class = "sm8x"
        else:
            capability_class = "baseline"

        # fmt: off
        config_map = {
            "float32": lambda h: FlexBwDConfig(16, 16, 16, 16, 1, 4),
            "baseline": lambda h: FlexBwDConfig(16, 16, 16, 16, 1, 4),
            "sm90": lambda h: (
                FlexBwDConfig(64, 64, 64, 64, 3, 4) if h < 64 else
                FlexBwDConfig(64, 128, 128, 64, 3, 8) if h <= 128 else
                FlexBwDConfig(64, 64, 64, 64, 2, 4)
            ),
            "sm10x": lambda h: (
                FlexBwDConfig(64, 128, 128, 64, 3, 4) if h <= 128 else
                FlexBwDConfig(64, 64, 64, 64, 1, 8) if h <= 192 else
                FlexBwDConfig(64, 64, 64, 64, 1, 4)
            ),
            "sm8x": lambda h: (
                FlexBwDConfig(32, 128, 128, 32, 3, 4)
                if h < 64
                else FlexBwDConfig(
                    64, 64, 64, 64, 3 if minor == 6 and h == 128 else 2, 4
                )
            ),
            "sm11x": lambda h: (
                FlexBwDConfig(32, 128, 128, 32, 3, 4)
                if h < 64
                else FlexBwDConfig(
                    64, 64, 64, 64, 1 if h >= 128 else 2, 4
                )
            ),
            "sm12x": lambda h: (
                FlexBwDConfig(32, 128, 128, 32, 3, 4)
                if h < 64
                else FlexBwDConfig(
                    64, 64, 64, 64, 1 if h >= 128 else 2, 4
                )
            ),
        }
        # fmt: on

        if head_dim <= 256:
            default_config = config_map[capability_class](head_dim)
        else:
            default_config = FlexBwDConfig(16, 16, 16, 16, 1, 4)

        if default_config not in flex_attn_bwd_configs:
            flex_attn_bwd_configs.append(default_config)

        return flex_attn_bwd_configs

    def get_flex_decode_configs(
        self, head_dim: int, dtype: Any
    ) -> list[FlexDecodeConfig]:
        capability = torch.cuda.get_device_capability()

        default_config = FlexDecodeConfig(64, 1, 2)

        flex_decode_configs: list[FlexDecodeConfig] = []

        if config.max_autotune:
            if config.max_autotune_flex_search_space == "EXHAUSTIVE":
                return self.exhaustive_flex_decode_configs
            flex_decode_configs += self.flex_decode_autotune_configs

        if capability in [(9, 0), (10, 0), (10, 3)]:  # sm_90, sm_100, sm_103
            if head_dim > 128 and dtype == torch.float32:
                default_config = FlexDecodeConfig(64, 1, 2)
            else:
                default_config = FlexDecodeConfig(64, 3, 2)
        if capability == (11, 0):
            default_config = FlexDecodeConfig(16, 1, 2)
        else:
            default_config = FlexDecodeConfig(64, 1, 2)

        if default_config not in flex_decode_configs:
            flex_decode_configs.append(default_config)

        return flex_decode_configs


class ROCmConfigHeuristic(BaseConfigHeuristic):
    """
    Child class for ROCm specific gemm/flex attention/conv/ configs.
    """

    def __init__(self) -> None:
        super().__init__()

        self.default_num_stages = get_backend_num_stages()

        self.mm_configs: list[BaseConfig] = [
            ROCmGemmConfig(
                16, 16, 256, self.default_num_stages, 4, group_m=4, waves_per_eu=2
            ),
            ROCmGemmConfig(32, 16, 256, self.default_num_stages, 4, group_m=4),
            ROCmGemmConfig(
                32, 32, 16, self.default_num_stages, 4, group_m=8, waves_per_eu=2
            ),
            ROCmGemmConfig(32, 32, 128, self.default_num_stages, 4, group_m=8),
            ROCmGemmConfig(32, 64, 64, self.default_num_stages, 4, group_m=8),
            ROCmGemmConfig(
                64, 16, 128, self.default_num_stages, 4, group_m=8, waves_per_eu=2
            ),
            ROCmGemmConfig(64, 32, 32, self.default_num_stages, 4, group_m=8),
            ROCmGemmConfig(64, 32, 64, self.default_num_stages, 4, group_m=8),
            ROCmGemmConfig(64, 32, 64, self.default_num_stages, 8, group_m=8),
            ROCmGemmConfig(64, 32, 128, self.default_num_stages, 4, group_m=8),
            ROCmGemmConfig(64, 64, 16, self.default_num_stages, 4, group_m=8),
            ROCmGemmConfig(64, 64, 64, self.default_num_stages, 4, group_m=4),
            ROCmGemmConfig(64, 64, 128, self.default_num_stages, 8, group_m=16),
            ROCmGemmConfig(64, 64, 256, self.default_num_stages, 8, group_m=4),
            ROCmGemmConfig(
                64, 128, 32, self.default_num_stages, 4, group_m=4, waves_per_eu=2
            ),
            ROCmGemmConfig(64, 128, 32, self.default_num_stages, 8, group_m=8),
            ROCmGemmConfig(64, 128, 64, self.default_num_stages, 8, group_m=4),
            ROCmGemmConfig(64, 128, 128, self.default_num_stages, 8, group_m=4),
            ROCmGemmConfig(128, 32, 32, self.default_num_stages, 4, group_m=8),
            ROCmGemmConfig(128, 32, 64, self.default_num_stages, 4, group_m=8),
            ROCmGemmConfig(
                128, 64, 32, self.default_num_stages, 4, group_m=8, waves_per_eu=2
            ),
            ROCmGemmConfig(128, 64, 64, self.default_num_stages, 4, group_m=16),
            ROCmGemmConfig(128, 64, 128, self.default_num_stages, 8, group_m=4),
            ROCmGemmConfig(
                128, 128, 32, self.default_num_stages, 4, group_m=16, waves_per_eu=2
            ),
            ROCmGemmConfig(128, 128, 32, self.default_num_stages, 8, group_m=16),
            ROCmGemmConfig(
                128, 128, 32, self.default_num_stages, 8, group_m=16, waves_per_eu=2
            ),
            ROCmGemmConfig(128, 128, 64, self.default_num_stages, 4, group_m=16),
            ROCmGemmConfig(128, 128, 64, self.default_num_stages, 8, group_m=8),
            ROCmGemmConfig(128, 128, 128, self.default_num_stages, 8, group_m=16),
            ROCmGemmConfig(
                128, 256, 32, self.default_num_stages, 4, group_m=16, waves_per_eu=2
            ),
            ROCmGemmConfig(128, 256, 64, self.default_num_stages, 8, group_m=4),
            ROCmGemmConfig(256, 64, 64, self.default_num_stages, 8, group_m=4),
            ROCmGemmConfig(
                256, 128, 32, self.default_num_stages, 4, group_m=4, waves_per_eu=2
            ),
            ROCmGemmConfig(256, 128, 32, self.default_num_stages, 8, group_m=16),
            ROCmGemmConfig(256, 128, 64, self.default_num_stages, 8, group_m=4),
            ROCmGemmConfig(256, 256, 64, self.default_num_stages, 8, group_m=4),
        ]

        # Exhaustive search for mm configs
        self.exhaustive_configs: list[BaseConfig] = [
            ROCmGemmConfig(
                BLOCK_M,
                BLOCK_N,
                BLOCK_K,
                num_stages,
                num_warps,
                group_m=group_m,
                matrix_instr_nonkdim=matrix_instr_nonkdim,
                waves_per_eu=waves_per_eu,
                kpack=kpack,
            )
            for BLOCK_M, BLOCK_N, BLOCK_K in itertools.product(
                [16, 32, 64, 128, 256], repeat=3
            )
            for num_stages in [1, self.default_num_stages]
            for num_warps in [4, 8]
            for group_m in [4, 8, 16]
            for matrix_instr_nonkdim in [0, 16]
            for waves_per_eu in [0, 2]
            for kpack in [2]
        ]

        self.default_flex_config = {
            (torch.float32, 64): ROCmFlexConfig(128, 32, 1, 4),
            (torch.float32, 128): ROCmFlexConfig(128, 32, 1, 4),
            (torch.float32, 256): ROCmFlexConfig(64, 16, 1, 4),
            (torch.bfloat16, 64): ROCmFlexConfig(128, 64, 1, 8),
            (torch.bfloat16, 128): ROCmFlexConfig(128, 64, 1, 8),
            (torch.bfloat16, 256): ROCmFlexConfig(32, 64, 1, 8),
            (torch.float16, 64): ROCmFlexConfig(128, 64, 1, 8),
            (torch.float16, 128): ROCmFlexConfig(128, 64, 1, 8),
            (torch.float16, 256): ROCmFlexConfig(32, 64, 1, 4),
        }

        self.flex_attn_fwd_autotune_configs: list[FlexConfig] = [
            ROCmFlexConfig(BLOCK1, BLOCK2, 1, w)
            for BLOCK1 in [16, 64, 128]
            for BLOCK2 in [16, 32, 64, 128]
            for w in [4, 8]
        ]

        self.flex_attn_bwd_autotune_configs: list[FlexBwDConfig] = [
            # See Note: flex bwd configs
            ROCmFlexBwDConfig(BLOCK1, BLOCK2, BLOCK2, BLOCK1, 1, w, mfma)
            for BLOCK1 in [16, 32, 64]
            for BLOCK2 in [32, 64, 128]
            for w in ([4, 8] if BLOCK1 >= 128 or BLOCK2 >= 128 else [4])
            for mfma in [0, 16]
            if BLOCK2 % BLOCK1 == 0
        ]

        self.flex_decode_autotune_configs: list[FlexDecodeConfig] = [
            ROCmFlexDecodeConfig(32, 1, 4),
            ROCmFlexDecodeConfig(64, 1, 4),
            ROCmFlexDecodeConfig(128, 1, 4),
            ROCmFlexDecodeConfig(32, 1, 8),
            ROCmFlexDecodeConfig(64, 1, 8),
            ROCmFlexDecodeConfig(128, 1, 8),
        ]

        self.exhaustive_flex_attn_fwd_configs: list[FlexConfig] = [
            ROCmFlexConfig(BLOCK_M, BLOCK_N, num_stages, num_warps, mfma, wpeu)
            for BLOCK_M in [16, 32, 64, 128]
            for BLOCK_N in [32, 64, 128]
            for num_stages in [1, 2]
            for num_warps in [2, 4, 8]
            for mfma in [0, 16]
            for wpeu in [0, int(8 // num_warps)]
        ]

        self.exhaustive_flex_attn_bwd_configs: list[FlexBwDConfig] = [
            # See Note: flex bwd configs
            ROCmFlexBwDConfig(
                BLOCK_M1,
                BLOCK_N1,
                BLOCK_M2,
                BLOCK_N2,
                num_stages,
                num_warps,
                mfma,
                wpeu,
            )
            for BLOCK_M1 in [16, 32, 64, 128]
            for BLOCK_N1 in [16, 32, 64, 128]
            for BLOCK_M2 in [16, 32, 64, 128]
            for BLOCK_N2 in [16, 32, 64, 128]
            for num_stages in [1, 2]
            for num_warps in [2, 4, 8]
            for mfma in [0, 16]
            for wpeu in [0, int(8 // num_warps)]
            if BLOCK_N1 % BLOCK_M1 == 0
            and BLOCK_M2 % BLOCK_N2 == 0  # kernel static assertions
        ]

        self.exhaustive_flex_decode_configs: list[FlexDecodeConfig] = [
            ROCmFlexDecodeConfig(block_n, num_stages, num_warps, mfma, wpeu, kpack=2)
            for block_n in [16, 32, 64, 128]
            for num_stages in [1, 2]
            for num_warps in [2, 4, 8]
            for mfma in [0, 16]
            for wpeu in [0, int(8 // num_warps)]
        ]

    def _prune_exhaustive_configs(
        self,
        configs: list[BaseConfig],
        dtype_size: int,
    ) -> list[BaseConfig]:
        # these cause AMD compile to crash
        pruned_configs = [
            c
            for c in configs
            if not (
                getattr(c, "matrix_instr_nonkdim", 0) == 2
                and getattr(c, "kpack", 0) == 2
            )
        ]
        return pruned_configs

    def _filter_configs(self, configs: list[BaseConfig]) -> list[BaseConfig]:
        """
        ROCm specific filtering
        """
        for c in configs:
            c.num_stages = self.default_num_stages
        return super()._filter_configs(configs)

    def _finalize_mm_configs(
        self,
        configs: list[BaseConfig],
    ) -> Generator[TritonConfig, None, None]:
        """
        Finalizes configs after scaling, applying additional constraints.
        """
        used: OrderedSet[tuple[int, ...]] = OrderedSet()

        max_mm_configs = config.test_configs.max_mm_configs

        for conf in configs:
            # Each warp computes a 16x16 tile = 256 elements
            conf.num_warps = min(conf.num_warps, conf.block_m * conf.block_n // 256)

            # Defaults for AMD triton backend kern args if not set
            matrix_instr_nonkdim = getattr(conf, "matrix_instr_nonkdim", 16)
            waves_per_eu = getattr(conf, "waves_per_eu", 0)
            kpack = getattr(conf, "kpack", 2)

            if matrix_instr_nonkdim != 0 and (
                conf.block_m % matrix_instr_nonkdim != 0
                or conf.block_n % matrix_instr_nonkdim != 0
            ):
                #  block_m and block_n must be a multiple of matrix_instr_nonkdim
                continue

            # Construct key for finding duplicate configs
            key: tuple[int, ...] = (
                conf.block_m,
                conf.block_n,
                conf.block_k,
                conf.num_stages,
                conf.num_warps,
                waves_per_eu,
                matrix_instr_nonkdim,
                kpack,
            )

            # Check if gemm specific arg exists - add to key if does
            group_m = getattr(conf, "group_m", None)
            # AMD GPU crashes if group_m = 0
            if group_m is not None and group_m <= 0:
                group_m = 8
            if group_m is not None:
                key += (group_m,)

            if waves_per_eu != 0:
                waves_per_eu = int(8 // conf.num_warps)

            if key not in used and (
                max_mm_configs is None or len(used) < max_mm_configs
            ):
                used.add(key)
                kwargs = {
                    "BLOCK_M": conf.block_m,
                    "BLOCK_N": conf.block_n,
                    "BLOCK_K": conf.block_k,
                    "num_stages": conf.num_stages,
                    "num_warps": conf.num_warps,
                    "matrix_instr_nonkdim": matrix_instr_nonkdim,
                    "waves_per_eu": waves_per_eu,
                    "kpack": kpack,
                }
                if group_m is not None:
                    kwargs["GROUP_M"] = group_m
                yield self.triton_config(**kwargs)

    def get_flex_attn_fwd_configs(self, head_dim: int, dtype: Any) -> list[FlexConfig]:
        flex_attn_fwd_configs: list[FlexConfig] = []

        if config.max_autotune:
            if config.max_autotune_flex_search_space == "EXHAUSTIVE":
                return self.exhaustive_flex_attn_fwd_configs
            flex_attn_fwd_configs += self.flex_attn_fwd_autotune_configs

        if head_dim <= 256:
            if dtype == torch.float32:
                default_config = ROCmFlexConfig(64, 64, 1, 4)
            else:
                default_config = ROCmFlexConfig(128, 64, 1, 8)
            default_config = self.default_flex_config.get(
                (dtype, head_dim), default_config
            )
        else:
            if dtype == torch.float32:
                default_config = ROCmFlexConfig(32, 16, 1, 4)
            else:
                default_config = ROCmFlexConfig(64, 32, 1, 4)

        if default_config not in flex_attn_fwd_configs:
            flex_attn_fwd_configs.append(default_config)

        return flex_attn_fwd_configs

    def get_flex_attn_bwd_configs(
        self, head_dim: int, dtype: Any
    ) -> list[FlexBwDConfig]:
        flex_attn_bwd_configs: list[FlexBwDConfig] = []

        if config.max_autotune:
            if config.max_autotune_flex_search_space == "EXHAUSTIVE":
                return self.exhaustive_flex_attn_bwd_configs
            flex_attn_bwd_configs += self.flex_attn_bwd_autotune_configs

        if dtype == torch.float32:
            default_config = ROCmFlexBwDConfig(16, 16, 16, 16, 1, 4)
        elif head_dim <= 256:
            if head_dim == 64:
                default_config = ROCmFlexBwDConfig(64, 64, 64, 64, 1, 4)
            elif head_dim == 128:
                default_config = ROCmFlexBwDConfig(64, 128, 128, 64, 1, 8)
            else:
                default_config = ROCmFlexBwDConfig(64, 64, 64, 64, 1, 4)
        else:
            default_config = ROCmFlexBwDConfig(16, 16, 16, 16, 1, 4)

        if default_config not in flex_attn_bwd_configs:
            flex_attn_bwd_configs.append(default_config)

        return flex_attn_bwd_configs

    def get_flex_decode_configs(
        self, head_dim: int, dtype: Any
    ) -> list[FlexDecodeConfig]:
        flex_decode_configs: list[FlexDecodeConfig] = []

        if config.max_autotune:
            if config.max_autotune_flex_search_space == "EXHAUSTIVE":
                return self.exhaustive_flex_decode_configs
            flex_decode_configs += self.flex_decode_autotune_configs

        default_config = ROCmFlexDecodeConfig(64, 1, 4)

        if default_config not in flex_decode_configs:
            flex_decode_configs.append(default_config)

        return flex_decode_configs


class XPUConfigHeuristic(BaseConfigHeuristic):
    """
    Placeholder child class for Intel GPU specific overrides.
    """

    def __init__(self) -> None:
        super().__init__()
        self.xpu_default_flex_config = {
            (torch.float32, 64): FlexConfig(128, 32, 1, 16),
            (torch.float32, 128): FlexConfig(128, 32, 1, 16),
            (torch.float32, 256): FlexConfig(64, 16, 1, 8),
            (torch.bfloat16, 64): FlexConfig(128, 64, 1, 16),
            (torch.bfloat16, 128): FlexConfig(128, 64, 1, 16),
            (torch.bfloat16, 256): FlexConfig(32, 64, 1, 4),
            (torch.float16, 64): FlexConfig(128, 64, 1, 16),
            (torch.float16, 128): FlexConfig(128, 64, 1, 16),
            (torch.float16, 256): FlexConfig(32, 64, 1, 4),
        }
        self.flex_attn_fwd_autotune_configs: list[FlexConfig] = [
            FlexConfig(32, 16, 2, 4),
            FlexConfig(128, 64, 2, 16),
            FlexConfig(128, 64, 2, 8),
            FlexConfig(128, 32, 2, 16),
            FlexConfig(128, 32, 2, 8),
        ]
        self.flex_attn_bwd_autotune_configs: list[FlexBwDConfig] = [
            FlexBwDConfig(32, 32, 32, 32, 2, 4),
            FlexBwDConfig(64, 64, 64, 64, 2, 4),
        ]
        self.flex_decode_autotune_configs: list[FlexDecodeConfig] = []

        if not bool(os.getenv("CI")):
            self.flex_attn_bwd_autotune_configs += [
                # See Note: flex bwd configs
                FlexBwDConfig(BLOCK1, BLOCK2, BLOCK2, BLOCK1, s, w)
                for BLOCK1 in [32, 64]
                for BLOCK2 in [32, 64, 128]
                for s in [1, 3, 4, 5]  # num_stages
                for w in ([4, 8] if BLOCK1 >= 128 or BLOCK2 >= 128 else [4])
                if BLOCK2 % BLOCK1 == 0
            ]
            self.flex_decode_autotune_configs += [
                FlexDecodeConfig(32, 1, 2),
                FlexDecodeConfig(32, 1, 1),
                FlexDecodeConfig(32, 2, 2),
                FlexDecodeConfig(32, 2, 1),
                FlexDecodeConfig(64, 1, 2),
                FlexDecodeConfig(64, 1, 1),
                FlexDecodeConfig(64, 2, 2),
                FlexDecodeConfig(64, 2, 1),
            ]

    def get_flex_attn_fwd_configs(self, head_dim: int, dtype: Any) -> list[FlexConfig]:
        flex_attn_fwd_configs: list[FlexConfig] = []

        if config.max_autotune:
            if config.max_autotune_flex_search_space == "EXHAUSTIVE":
                return self.exhaustive_flex_attn_fwd_configs
            flex_attn_fwd_configs += self.flex_attn_fwd_autotune_configs

        if head_dim <= 256:
            if dtype == torch.float32:
                default_config = FlexConfig(64, 64, 1, 8)
            else:
                default_config = FlexConfig(128, 64, 1, 16)
            default_config = self.xpu_default_flex_config.get(
                (dtype, head_dim), default_config
            )
        else:
            if dtype == torch.float32:
                default_config = FlexConfig(32, 16, 1, 4)
            else:
                default_config = FlexConfig(64, 32, 1, 8)

        if default_config not in flex_attn_fwd_configs:
            flex_attn_fwd_configs.append(default_config)

        return flex_attn_fwd_configs

    def get_flex_attn_bwd_configs(
        self, head_dim: int, dtype: Any
    ) -> list[FlexBwDConfig]:
        flex_attn_bwd_configs: list[FlexBwDConfig] = []

        if config.max_autotune:
            if config.max_autotune_flex_search_space == "EXHAUSTIVE":
                return self.exhaustive_flex_attn_bwd_configs
            flex_attn_bwd_configs += self.flex_attn_bwd_autotune_configs

        if dtype == torch.float32:
            default_config = FlexBwDConfig(16, 16, 16, 16, 1, 4)
        elif head_dim <= 256:
            if head_dim == 64:
                default_config = FlexBwDConfig(64, 64, 64, 64, 1, 8)
            elif head_dim == 128:
                default_config = FlexBwDConfig(64, 64, 64, 64, 1, 8)
            else:
                default_config = FlexBwDConfig(64, 64, 64, 64, 1, 8)
        else:  # modest hardware or extremely large head_dim
            default_config = FlexBwDConfig(16, 16, 16, 16, 1, 4)

        if default_config not in flex_attn_bwd_configs:
            flex_attn_bwd_configs.append(default_config)

        return flex_attn_bwd_configs

    def get_flex_decode_configs(
        self, head_dim: int, dtype: Any
    ) -> list[FlexDecodeConfig]:
        flex_decode_configs: list[FlexDecodeConfig] = []

        if config.max_autotune:
            if config.max_autotune_flex_search_space == "EXHAUSTIVE":
                return self.exhaustive_flex_decode_configs
            flex_decode_configs += self.flex_decode_autotune_configs

        default_config = FlexDecodeConfig(64, 1, 2)

        if default_config not in flex_decode_configs:
            flex_decode_configs.append(default_config)

        return flex_decode_configs

    def _prune_exhaustive_configs(
        self,
        configs: list[BaseConfig],
        dtype_size: int,
    ) -> list[BaseConfig]:
        return configs


class MTIAConfigHeuristic(BaseConfigHeuristic):
    """
    Placeholder child class for MTIA specific overrides.
    """


# Template-specific mixin classes
class MMTemplateConfigMixin(GemmMaxAutotuneTemplateConfigHeuristics):
    """
    Mixin class that converts config lists to template kwargs.
    This handles the logic that was previously in choices.get_mm_configs.

    This mixin expects to be used with BaseConfigHeuristic or its subclasses.
    """

    # Type annotations to ensure the mixin works with BaseConfigHeuristic
    get_mm_configs: Callable[[], partial[Generator[TritonConfig, None, None]]]
    get_exhaustive_mm_configs: Callable[
        [], partial[Generator[TritonConfig, None, None]]
    ]
    _filter_configs: Callable[[list[BaseConfig]], list[BaseConfig]]

    def get_extra_kwargs(
        self,
        kernel_inputs: KernelInputs,
        op_name: str,
    ) -> dict[str, Any]:
        assert isinstance(kernel_inputs, MMKernelInputs)
        m, n, k = kernel_inputs.mnk_symbolic()
        # Calculate allow_tf32
        allow_tf32 = torch.backends.cuda.matmul.fp32_precision == "tf32" and (
            not inductor_config.force_same_precision
            or ((m % 16) == 0 and (n % 16) == 0 and (k % 8) == 0)
        )

        return {
            "ALLOW_TF32": allow_tf32,
        }

    def _valid(self, kernel_inputs: KernelInputs) -> bool:
        return True

    def _get_config_generator(
        self,
    ) -> partial[Generator[TritonConfig, None, None]]:
        """
        Get the appropriate config generator based on search space.
        Can be overridden by subclasses for template-specific behavior.
        """
        # Handle exhaustive search case
        if config.max_autotune_gemm_search_space == "EXHAUSTIVE":
            return self.get_exhaustive_mm_configs()
        else:
            return self.get_mm_configs()

    def _get_template_configs_impl(
        self,
        kernel_inputs: KernelInputs,
        op_name: str,
        **kwargs,
    ) -> Generator[dict[str, Any], None, None]:
        """
        Convert config lists to template kwargs.
        This replaces the logic from choices.get_mm_configs and inlines mm_options.
        """
        assert isinstance(kernel_inputs, MMKernelInputs), (
            f"{self.__class__.__name__} requires MMKernelInputs"
        )
        input_nodes = kernel_inputs.nodes()
        if len(input_nodes) < 2:
            raise ValueError(f"Need at least 2 input tensors, got {len(input_nodes)}")
        if not self._valid(kernel_inputs):
            return

        # Extract M, N, K from kernel_inputs
        m, n, k = kernel_inputs.mnk_symbolic()

        # Extract dtype and device_type from kernel_inputs
        dtype = kernel_inputs.dtype()

        # Get the appropriate config generator
        configs = self._get_config_generator()

        # Generate and process configs
        for c in configs(
            m,
            n,
            k,
            dtype_size=dtype.itemsize,
            op_name=op_name,
            **kwargs,
        ):
            template_kwargs = self._convert_config_to_template_kwargs(
                c,
                m,
                n,
                k,
                kernel_inputs.out_dtype(),
            )
            yield template_kwargs

    def _convert_config_to_template_kwargs(
        self,
        triton_config: TritonConfig,
        m: sympy.Integer | sympy.Symbol,
        n: sympy.Integer | sympy.Symbol,
        k: sympy.Integer | sympy.Symbol,
        out_dtype: torch.dtype,
    ) -> dict[str, Any]:
        """
        Convert triton config to template kwargs.
        Moved from mm_common.mm_options.
        """
        # Calculate EVEN_K symbolic. (It isn't worth guarding on this)
        even_k_symbolic = (k % triton_config.kwargs["BLOCK_K"]) == 0
        even_k_symbolic = V.graph.sizevars.statically_known_true(even_k_symbolic)

        # Build options dict

        options_dict = dict(
            EVEN_K=even_k_symbolic,
            USE_FAST_ACCUM=False,  # Option for _scaled_mm
            ACC_TYPE=self._get_acc_type(out_dtype),
            num_stages=triton_config.num_stages,
            num_warps=triton_config.num_warps,
            **triton_config.kwargs,
        )

        # If GROUP_M not specified then default to 8
        if "GROUP_M" not in triton_config.kwargs:
            group_m = triton_config.kwargs.get("GROUP_M", 8)
            options_dict["GROUP_M"] = group_m

        return options_dict

    def _get_acc_type(self, dtype: torch.dtype) -> str:
        """
        Get accumulator type for the given dtype.
        Moved from mm_common.acc_type.
        """
        if dtype in (torch.float16, torch.bfloat16):
            return "tl.float32"
        return f"tl.{dtype}".replace("torch.", "")


# INT8 specific mixin to filter correctly
class INT8MMTemplateConfigMixin(MMTemplateConfigMixin):
    """
    Ensure that we feed in has_int8_tensor=True
    """

    def __init__(self) -> None:
        super().__init__()
        self.has_int8_tensor = True


# MMPlusMM specific mixin to avoid running _scale_mm_configs
class MMPlusMMTemplateConfigMixin(MMTemplateConfigMixin):
    """
    Ensure that _should_scale_configs is False
    """

    # TODO(coconutruben): remove this once all tests work
    # with proper scaling on mm_plus_mm
    def __init__(self) -> None:
        super().__init__()
        self.should_scale_configs = False

    def _get_template_configs_impl(
        self,
        kernel_inputs: KernelInputs,
        op_name: str,
        **kwargs,
    ) -> Generator[dict[str, Any], None, None]:
        assert isinstance(kernel_inputs, MMKernelInputs), "Expect MMKernelInputs"
        m, n, k = kernel_inputs.mnk_symbolic()
        for template_kwargs in super()._get_template_configs_impl(
            kernel_inputs, op_name, **kwargs
        ):
            # Apply BLOCK_K constraint specific to mm_plus_mm
            # see https://github.com/triton-lang/triton/issues/1298
            # BLOCK_K = K causes llvm error
            if V.graph.sizevars.statically_known_lt(
                template_kwargs.get("BLOCK_K", k), k
            ):
                yield template_kwargs


class TMAWorkspaceMixin(MMTemplateConfigMixin):
    """
    Small mixin to ensure that the workspace arg is correct for TMA
    and TMA specific filtering can happen.
    """

    def get_extra_kwargs(
        self,
        kernel_inputs: KernelInputs,
        op_name: str,
    ) -> dict[str, Any]:
        kwargs = super().get_extra_kwargs(kernel_inputs, op_name)
        kwargs["workspace_arg"] = get_tma_workspace_arg(
            num_tma_descriptors=2,
            device=kernel_inputs.device(),
        )
        return kwargs

    # pyrefly: ignore [bad-override]
    def _filter_configs(self, configs: list[BaseConfig]) -> list[BaseConfig]:
        """
        TMA specific filtering, as num_warps=2 not safe for TMA
        """
        configs = [c for c in configs if c.num_warps != 2]
        return super()._filter_configs(configs)


def get_shared_memory_checker_opts(op_name: str, dtype_size: int):
    return {
        "has_sm_layout_conversion": True,
        # addmm requires the acc dtype for layout conversion due to adding bias
        # mm just input dtype
        "layout_conversion_byte_size": 4 if op_name == "addmm" else dtype_size,
    }


# TMA-specific mixin for TMA templates
class TMATemplateConfigMixin(TMAWorkspaceMixin, MMTemplateConfigMixin):
    """
    TMA-specific mixin that uses persistent configs and adds TMA options.
    This inherits from MMTemplateConfigMixin and overrides config generation.
    """

    def _get_template_configs_impl(
        self,
        kernel_inputs: KernelInputs,
        op_name: str,
        **kwargs,
    ) -> Generator[dict[str, Any], None, None]:
        """
        Generate TMA template configs by calling super and adding TMA-specific options.
        """
        assert isinstance(kernel_inputs, MMKernelInputs), (
            "TMATemplateConfigMixin requires MMKernelInputs"
        )
        mat1, mat2 = kernel_inputs.mat1mat2()
        tma_opts = {
            "A_ROW_MAJOR": not mat1.layout.is_transposed(),
            "B_ROW_MAJOR": not mat2.layout.is_transposed(),
            "NUM_SMS": get_num_sms(),
            "TMA_SIZE": TMA_DESCRIPTOR_SIZE,
            "TMA_EXPERIMENTAL_API": not has_triton_stable_tma_api(),
            "tma_store": config.triton.enable_template_tma_store,
            "transpose_discontiguous_tensor_descriptors_override": True,
        }

        # Get base template configs from superclass
        for template_kwargs in super()._get_template_configs_impl(
            kernel_inputs,
            op_name,
            **get_shared_memory_checker_opts(
                op_name, dtype_size=kernel_inputs.dtype().itemsize
            ),
        ):
            yield {**template_kwargs, **tma_opts}


# TMA mixins for Blackwell templates
class BlackwellTMATemplateConfigMixin(TMATemplateConfigMixin):
    def _get_template_configs_impl(
        self,
        kernel_inputs: KernelInputs,
        op_name: str,
        **kwargs,
    ) -> Generator[dict[str, Any], None, None]:
        """
        Generate TMA template configs by calling super and adding TMA-specific options.
        """
        # Get base template configs from superclass
        for template_kwargs in super()._get_template_configs_impl(
            kernel_inputs,
            op_name,
            **kwargs,
        ):
            # Some Triton versions requires num_warps >= 4 for WS
            # to avoid compilation issues. Triton disables WS if num_warps < 4
            # or num_stages < 2. Similar issues have been seen with num_stages=1
            constraints_violated = (
                template_kwargs["num_warps"] < 4 or template_kwargs["num_stages"] < 2
            )
            ws = (
                template_kwargs.get("WARP_SPECIALIZE", True)
                and not constraints_violated
            )
            flatten = template_kwargs.get("FLATTEN", True) and not constraints_violated
            yield {
                **template_kwargs,
                "NUM_SMS": get_num_sms(),
                "WARP_SPECIALIZE": ws,
                "FLATTEN": flatten,
            }

    @staticmethod
    def _generate_exhaustive_configs() -> list[BaseConfig]:
        configs: list[BaseConfig] = []
        for BLOCK_M, BLOCK_N, BLOCK_K in itertools.product(
            [32, 64, 128, 256],
            repeat=3,
        ):
            for num_stages in [2, 3, 4, 5, 6]:
                # AutoWS doesn't work with num_warps < 4
                for num_warps in [4, 8]:
                    for EPILOGUE_SUBTILE in [1, 2, 4]:
                        configs.append(
                            BlackwellGPUGemmConfig(
                                block_m=BLOCK_M,
                                block_n=BLOCK_N,
                                block_k=BLOCK_K,
                                num_stages=num_stages,
                                num_warps=num_warps,
                                group_m=8,
                                epilogue_subtile=EPILOGUE_SUBTILE,
                                warp_specialize=True,
                                flatten=True,
                            )
                        )
        return configs


# Scaled MM-specific mixin for scaled MM templates
class BaseScaledMMConfigMixin(MMTemplateConfigMixin):
    """
    This is a base that handles the common case for ScaledMM

    The TMA and non-TMA should build on top of this
    """

    def adjust_kernel_inputs(
        self, kernel_inputs: KernelInputs, op_name: str
    ) -> KernelInputs:
        """
        for scaled_mm, we need to unsqueeze scale tensors, and bias
        """
        assert isinstance(kernel_inputs, MMKernelInputs), (
            "Expect MMKernelInputs for scaled MM"
        )
        inputs = super().adjust_kernel_inputs(kernel_inputs, op_name)
        nodes = inputs.nodes()
        mat_a, mat_b, scale_a, scale_b, *bias = nodes
        bias = bias[0] if bias else None
        # Prepare triton input nodes and create kernel_inputs at the top
        from ..lowering import lowerings as L

        aten = torch.ops.aten
        if bias and len(mat_b.get_size()) == len(bias.get_size()) + 1:
            # Need to unsqueeze bias from [N] -> [1, N]
            bias = L[aten.unsqueeze](bias, 0)

        if len(scale_a.get_size()) == 0 or len(scale_b.get_size()) == 0:
            assert len(scale_a.get_size()) == len(scale_b.get_size())
            # Need to unsqueeze scale from [] -> [1, 1]
            scale_a = L[aten.unsqueeze](L[aten.unsqueeze](scale_a, 0), 1)
            scale_b = L[aten.unsqueeze](L[aten.unsqueeze](scale_b, 0), 1)
        nodes = [mat_a, mat_b, scale_a, scale_b]
        if bias:
            nodes.append(bias)
        return MMKernelInputs(
            nodes, mat1_idx=kernel_inputs._mat1_idx, mat2_idx=kernel_inputs._mat2_idx
        )

    def _get_template_configs_impl(
        self,
        kernel_inputs: KernelInputs,
        op_name: str,
        **kwargs,
    ) -> Generator[dict[str, Any], None, None]:
        """
        Generate scaled MM template configs with scaled MM-specific options.
        Handles the remaining logic from mm_common, including assertions.
        """
        kernel_inputs = self.adjust_kernel_inputs(kernel_inputs, op_name)
        input_nodes = kernel_inputs.nodes()
        # Initial assertion from mm_common.scaled_mm_options
        assert len(input_nodes) >= 4, (
            f"scaled_mm requires at least 4 inputs, got {len(input_nodes)}"
        )

        # Extract scale tensors (typically scale_a and scale_b are input_nodes[2] and input_nodes[3])
        scale_a = input_nodes[2]
        scale_b = input_nodes[3]

        # Scale compatibility assertion from mm_common.scaled_mm_options
        def are_compatible_scales(size_a: Any, size_b: Any) -> bool:
            # Same sized scales are compatible
            if len(size_a) == len(size_b):
                return True

            # Both need to be scalars or len(1) tensors
            if len(size_a) <= 1 and len(size_b) <= 1:
                return True

            return False

        size_a, size_b = scale_a.get_size(), scale_b.get_size()
        assert are_compatible_scales(size_a, size_b), (
            "Expect scale_a and scale_b to be either both scalars (including single-element tensors) "
            f"or 1-dimensional tensors with the same size. Got scale_a: {len(size_a)} and scale_b: {len(size_b)}."
        )

        assert isinstance(kernel_inputs, MMKernelInputs), (
            f"{self.__class__.__name__} requires MMKernelInputs"
        )

        if not self._valid(kernel_inputs):
            return

        # Get base template configs from superclass
        for template_kwargs in super()._get_template_configs_impl(
            kernel_inputs, op_name, **kwargs
        ):
            # Add scaled MM-specific options (moved from mm_common.scaled_mm_options)
            # Override accumulator type for scaled MM
            template_kwargs["ACC_TYPE"] = "tl.float32"

            yield template_kwargs


class ScaledMMConfigMixin(BaseScaledMMConfigMixin):
    """Mixing for scaled mm with the regular mm template"""

    def get_extra_kwargs(
        self,
        kernel_inputs: KernelInputs,
        op_name: str,
    ) -> dict[str, Any]:
        kwargs = super().get_extra_kwargs(kernel_inputs, op_name)
        from ..kernel.mm_common import scale_mm_epilogue

        return {
            **kwargs,
            "suffix_args": kernel_inputs.count - 2,
            "epilogue_fn": scale_mm_epilogue(),
            "epilogue_fn_hash": "scale_mm_epilogue",
        }

    def _valid(self, kernel_inputs: KernelInputs) -> bool:
        assert isinstance(kernel_inputs, MMKernelInputs), (
            "Expect MMKernelInputs for ScaledMMConfigMixin"
        )
        _, _, k = kernel_inputs.mnk_symbolic()
        if V.graph.sizevars.guard_or_false(sympy.Le(k, 16)):
            # Triton crashes however uncommon for real workloads
            return False

        # On NVIDIA B200 GPUs, K dim must be >= 32 for tcgen05.mma.kind::f8f6f4.* PTX instruction to be valid
        # source: https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-matrix-shape
        if using_b200() and V.graph.sizevars.guard_or_false(sympy.Lt(k, 32)):
            return False
        return True

    # pyrefly: ignore [bad-override]
    def _filter_configs(self, configs: list[BaseConfig]) -> list[BaseConfig]:
        """
        Filter out bad configs for specific hardware.
        On AMD MI350X (GFX 9.5+), skip configs with BLOCK_K<=64 due to lack of corresponding MFMA instructions.
        """

        def should_skip_mi350x_config(config: BaseConfig) -> bool:
            """Skip config if BLOCK_K<=64 on MI350X (GFX 9.5+)"""
            try:
                return (
                    config.block_k <= 64
                    and torch.version.hip is not None
                    and torch.cuda.get_device_capability() >= (9, 5)
                )
            except RuntimeError:
                # If no HIP GPUs are available, we can't check device capability
                # so we don't skip any configs
                return False

        filtered_configs = [c for c in configs if not should_skip_mi350x_config(c)]
        return super()._filter_configs(filtered_configs)


# Scaled TMA-specific mixin for scaled MM templates with TMA
class ScaledTMAConfigMixin(TMAWorkspaceMixin, BaseScaledMMConfigMixin):
    """
    Scaled TMA-specific mixin that extends BaseScaledMMConfigMixin with TMA functionality.
    This is for scaled MM templates that use device TMA.
    This inherits from BaseScaledMMConfigMixin and adds TMA-specific options.
    """

    # pyrefly: ignore [bad-override]
    def _filter_configs(self, configs: list[BaseConfig]) -> list[BaseConfig]:
        """
        TMA specific filtering:
        - num_warps=2 not safe for TMA
        - block_k >= 32 required for TMA (requires inner-most dimension >= 32)
        """
        configs = [c for c in configs if c.num_warps != 2 and c.block_k >= 32]
        return super()._filter_configs(configs)

    def _get_template_configs_impl(
        self,
        kernel_inputs: KernelInputs,
        op_name: str,
        **kwargs,
    ) -> Generator[dict[str, Any], None, None]:
        """
        Generate scaled TMA template configs with both scaled MM and TMA-specific options.
        """
        # Get base scaled MM template configs from superclass
        for template_kwargs in super()._get_template_configs_impl(
            kernel_inputs,
            op_name,
            **kwargs,
        ):
            # Add TMA-specific options for device TMA scaled MM
            template_kwargs["TMA_SIZE"] = TMA_DESCRIPTOR_SIZE
            template_kwargs["NUM_SMS"] = get_num_sms()
            template_kwargs["TMA_EXPERIMENTAL_API"] = not has_triton_stable_tma_api()

            yield template_kwargs


# Scaled Blackwell TMA-specific mixin for scaled MM templates with TMA
class ScaledBlackwellTMAConfigMixin(
    BlackwellTMATemplateConfigMixin, ScaledMMConfigMixin
):
    """
    Scaled Blackwell TMA-specific mixin that extends ScaledMMConfigMixin with TMA functionality.
    This is for scaled MM templates that use device TMA on Blackwell.
    This inherits from ScaledMMConfigMixin, which inherits the scale_mm_epilogue, and adds TMA-specific options.
    """

    def _filter_configs(self, configs: list[BaseConfig]) -> list[BaseConfig]:
        """
        Warp specialization-specific filtering (BlackwellTMATemplateConfigMixin)
        (compilation issues occur in some versions of Triton)
        - num_warps < 4 unsafe for warpspec
        - num_stages < 2 unsafe for warpspec

        TMA-specific filtering:
        - block_k >= 32 required for TMA (requires inner-most dimension >= 32)
        """
        configs = [c for c in configs if c.block_k >= 32]
        return super()._filter_configs(configs)


# Template-specific heuristic classes using multiple inheritance


@register_template_heuristic(
    mm_template.uid,
    "cuda",
    register=torch.version.hip is None,
)
@register_template_heuristic(
    bmm_template.uid,
    "cuda",
    register=torch.version.hip is None,
)
class CUDAMMTemplateConfigHeuristic(MMTemplateConfigMixin, CUDAConfigHeuristic):
    """Standard MM template heuristic for CUDA"""


@register_template_heuristic(
    mm_template.uid, "cuda", register=torch.version.hip is None, op_name="addmm"
)
@register_template_heuristic(
    bmm_template.uid, "cuda", register=torch.version.hip is None, op_name="baddbmm"
)
class CUDAAddMMTemplateConfigHeuristic(AddMMConfigMixin, CUDAMMTemplateConfigHeuristic):
    """Addmm specific mixin for CUDA"""


# TODO(coconutruben): deprecate once autoheuristic is deprecated
@register_template_heuristic(
    mm_template.uid,
    "cuda",
    register=torch.version.hip is None,
    op_name="mm-ah",
)
class CUDAMMAHTemplateConfigHeuristic(MMTemplateConfigMixin, CUDAConfigHeuristic):
    """Standard MM template heuristic for CUDA using the extra mm configs only (for autoheuristic)"""

    def __init__(self) -> None:
        super().__init__()
        # Override mm_configs to use scaled_mm_configs
        self.mm_configs = self.extra_mm_configs
        self.exhaustive_configs = self.extra_mm_configs


@register_template_heuristic(
    persistent_tma_mm_template.uid,
    "cuda",
    register=torch.version.hip is None,
)
class CUDAPersistentTMATemplateConfigHeuristic(
    TMATemplateConfigMixin, CUDAConfigHeuristic
):
    """Persistent TMA template heuristic for CUDA"""

    def __init__(self) -> None:
        super().__init__()
        # Override mm_configs to use persistent_mm_configs
        self.mm_configs = self.persistent_mm_configs


@register_template_heuristic(
    blackwell_ws_persistent_device_tma_mm_template.uid,
    "cuda",
    register=torch.version.hip is None,
)
class CUDABlackwellPersistentTMATemplateConfigHeuristic(
    BlackwellTMATemplateConfigMixin, CUDAConfigHeuristic
):
    """Blackwell Persistent TMA template"""

    def __init__(self) -> None:
        super().__init__()
        self.mm_configs = self.blackwell_persistent_mm_configs
        self.exhaustive_configs = self._generate_exhaustive_configs()


@register_template_heuristic(
    persistent_tma_mm_template.uid,
    "cuda",
    register=torch.version.hip is None,
    op_name="addmm",
)
class CUDAAddmmPersistentTMATemplateConfigHeuristic(
    AddMMConfigMixin, CUDAPersistentTMATemplateConfigHeuristic
):
    """Addmm specific mixin for CUDA"""


@register_template_heuristic(
    blackwell_ws_persistent_device_tma_mm_template.uid,
    "cuda",
    register=torch.version.hip is None,
    op_name="addmm",
)
class CUDABlackwellAddmmPersistentTMATemplateConfigHeuristic(
    AddMMConfigMixin, CUDABlackwellPersistentTMATemplateConfigHeuristic
):
    """Addmm extension for DataCenter Blackwell Templates"""

    def __init__(self) -> None:
        super().__init__()
        # NOTE: to ensure that we pass tests, addmm needs a small config
        self.mm_configs = (
            self.blackwell_persistent_mm_configs
            + self.blackwell_persistent_addmm_configs
        )
        self.exhaustive_configs = self._generate_exhaustive_configs()


@register_template_heuristic(
    mm_template.uid, "cuda", register=torch.version.hip is None, op_name="scaled_mm"
)
class CUDAScaledMMTemplateConfigHeuristic(ScaledMMConfigMixin, CUDAConfigHeuristic):
    """Scaled MM template heuristic for CUDA"""

    def __init__(self) -> None:
        super().__init__()
        # Override mm_configs to use scaled_mm_configs
        self.mm_configs = self.scaled_mm_configs

    def _filter_configs(self, configs: list[BaseConfig]) -> list[BaseConfig]:
        configs = [c for c in configs if c.block_k >= 32]
        return super()._filter_configs(configs)


@register_template_heuristic(
    scaled_mm_device_tma_epilogue_scaling_template.uid,
    "cuda",
    register=torch.version.hip is None,
    op_name="scaled_mm",
)
class CUDAScaledTMAEpilogueScalingTemplateConfigHeuristic(
    ScaledTMAConfigMixin, CUDAConfigHeuristic
):
    """Scaled TMA template heuristic for CUDA: epilogue scaling variants (TensorWise, RowWise)"""

    def __init__(self) -> None:
        super().__init__()
        # Override mm_configs to use scaled_persistent_mm_configs for TMA
        self.mm_configs = self.scaled_persistent_mm_configs


@register_template_heuristic(
    scaled_mm_device_tma_main_loop_scaling_template.uid,
    "cuda",
    register=torch.version.hip is None,
    op_name="scaled_mm",
)
class CUDAScaledTMAMainLoopScalingTemplateConfigHeuristic(
    ScaledTMAConfigMixin, CUDAConfigHeuristic
):
    """
    Scaled TMA template heuristic for CUDA:
        main loop scaling variants (BlockWise1x128, BlockWise1x32, BlockWise1x16, BlockWise128x128)
    """

    def __init__(self) -> None:
        super().__init__()
        # Override mm_configs to use scaled_persistent_mm_configs for TMA
        self.mm_configs = self.scaled_persistent_mm_configs

    def _get_template_configs_impl(
        self,
        kernel_inputs: KernelInputs,
        op_name: str,
        **kwargs,
    ) -> Generator[dict[str, Any], None, None]:
        """
        Generate main loop scaling kernel inputs.
        """
        mat_a, mat_b, scale_a, scale_b = kernel_inputs._input_nodes
        scale_a_size, scale_b_size = scale_a.get_size(), scale_b.get_size()

        scale_option_a, scale_option_b = get_scaling_options(
            mat_a, mat_b, scale_a_size, scale_b_size
        )
        tile_size_a = get_tile_size(scale_option_a)
        tile_size_b = get_tile_size(scale_option_b)

        # Get base scaled MM template configs from superclass
        for template_kwargs in super()._get_template_configs_impl(
            kernel_inputs,
            op_name,
            **kwargs,
        ):
            # Add scaling-specific options for main loop scaling variants

            # Inductor templates require compile-time constants passed in as tl.constexpr values.
            # In cases in which the block size (BLOCK_*) is smaller than the tile size (128, 32, 16),
            # scales must be broadcasted to BLOCK_* (rather than to a tile_sizextile_size chunk).

            template_kwargs["TILE_SIZE_A"] = tile_size_a
            template_kwargs["TILE_SIZE_B"] = tile_size_b

            template_kwargs["MIN_BLOCK_TILE_AM"] = min(
                template_kwargs["BLOCK_M"], tile_size_a
            )
            template_kwargs["MIN_BLOCK_TILE_AK"] = min(
                template_kwargs["BLOCK_K"], tile_size_a
            )
            template_kwargs["MIN_BLOCK_TILE_BK"] = min(
                template_kwargs["BLOCK_K"], tile_size_b
            )
            template_kwargs["MIN_BLOCK_TILE_BN"] = min(
                template_kwargs["BLOCK_N"], tile_size_b
            )

            yield template_kwargs


@register_template_heuristic(
    blackwell_ws_persistent_device_tma_mm_template.uid,  # regular Blackwell MM template + scaling epilogue from ScaledMMConfigMixin
    "cuda",
    register=torch.version.hip is None,
    op_name="scaled_mm",
)
class CUDAScaledBlackwellTMATemplateConfigHeuristic(
    ScaledBlackwellTMAConfigMixin, CUDAConfigHeuristic
):
    """Scaled Blackwell TMA template heuristic for CUDA"""

    def __init__(self) -> None:
        super().__init__()
        # TODO: Tune scaled_persistent_mm_configs for Blackwell
        self.mm_configs = self.blackwell_persistent_addmm_configs


@register_template_heuristic(
    mm_plus_mm_template.uid,
    "cuda",
    register=torch.version.hip is None,
)
class CUDAMMPlusMMTemplateConfigHeuristic(
    MMPlusMMTemplateConfigMixin, CUDAConfigHeuristic
):
    """MM Plus MM template heuristic for CUDA"""

    def __init__(self) -> None:
        super().__init__()
        # Override mm_configs to use mm_plus_mm_configs
        self.mm_configs = self.mm_plus_mm_configs
        # NOTE: overriding exhaustive configs here to be the same as mm_configs
        # as we haven't validated exhaustive support here yet
        # TODO(coconutruben): remove this once we have validated exhaustive support
        # for scaled_mm
        self.exhaustive_configs = self.mm_plus_mm_configs


@register_template_heuristic(
    mm_template.uid,
    "cuda",
    register=torch.version.hip is None,
    op_name="int_mm",
)
class CUDAInt8MMTemplateConfigHeuristic(INT8MMTemplateConfigMixin, CUDAConfigHeuristic):
    """Int8 MM template heuristic for CUDA"""

    def __init__(self) -> None:
        super().__init__()
        # Override mm_configs to use int8_mm_configs
        self.mm_configs = self.int8_mm_configs
        # NOTE: overriding exhaustive configs here to be the same as mm_configs
        # as we haven't validated exhaustive support here yet
        # TODO(coconutruben): remove this once we have validated exhaustive support
        # for scaled_mm
        self.exhaustive_configs = self.int8_mm_configs


# ROCm template-specific classes


@register_template_heuristic(
    mm_template.uid,
    "cuda",
    register=torch.version.hip is not None,
)
@register_template_heuristic(
    bmm_template.uid,
    "cuda",
    register=torch.version.hip is not None,
)
class ROCmMMTemplateConfigHeuristic(MMTemplateConfigMixin, ROCmConfigHeuristic):
    """Standard MM template heuristic for ROCm"""


# TODO(coconutruben): replace with template.name once templates are importable
@register_template_heuristic(
    mm_template.uid, "cuda", register=torch.version.hip is not None, op_name="addmm"
)
# TODO(coconutruben): replace with template.name once templates are importable
@register_template_heuristic(
    bmm_template.uid, "cuda", register=torch.version.hip is not None, op_name="baddbmm"
)
class ROCmAddMMTemplateConfigHeuristic(AddMMConfigMixin, ROCmMMTemplateConfigHeuristic):
    """Addmm specific mixin for ROCm"""


# TODO(coconutruben): deprecate once autoheuristic is deprecated
@register_template_heuristic("mm-ah", "cuda", register=torch.version.hip is not None)
class ROCmMMAHTemplateConfigHeuristic(MMTemplateConfigMixin, ROCmConfigHeuristic):
    """Standard MM template heuristic for ROCm using the extra mm configs only (for autoheuristic)"""

    def __init__(self) -> None:
        super().__init__()
        # Override mm_configs to use scaled_mm_configs
        self.mm_configs = self.extra_mm_configs
        self.exhaustive_configs = self.extra_mm_configs


@register_template_heuristic(
    mm_template.uid,
    "cuda",
    register=torch.version.hip is not None,
    op_name="scaled_mm",
)
class ROCmScaledMMTemplateConfigHeuristic(ScaledMMConfigMixin, ROCmConfigHeuristic):
    """Scaled MM template heuristic for ROCm (non-TMA)"""

    def __init__(self) -> None:
        super().__init__()
        # Override mm_configs to use scaled_mm_configs
        self.mm_configs = self.scaled_mm_configs
        # NOTE: overriding exhaustive configs here to be the same as mm_configs
        # as we haven't validated exhaustive support here yet
        # TODO(coconutruben): remove this once we have validated exhaustive support
        # for scaled_mm
        self.exhaustive_configs = self.scaled_mm_configs


@register_template_heuristic(
    mm_template.uid,
    "cuda",
    register=torch.version.hip is not None,
    op_name="int_mm",
)
class ROCmInt8MMTemplateConfigHeuristic(INT8MMTemplateConfigMixin, ROCmConfigHeuristic):
    """Int8 MM template heuristic for ROCm"""

    def __init__(self) -> None:
        super().__init__()
        # Override mm_configs to use int8_mm_configs
        self.mm_configs = self.int8_mm_configs
        # NOTE: overriding exhaustive configs here to be the same as mm_configs
        # as we haven't validated exhaustive support here yet
        # TODO(coconutruben): remove this once we have validated exhaustive support
        # for scaled_mm
        self.exhaustive_configs = self.int8_mm_configs


@register_template_heuristic(
    mm_plus_mm_template.uid,
    "cuda",
    register=torch.version.hip is not None,
)
class ROCmMMPlusMMTemplateConfigHeuristic(
    MMPlusMMTemplateConfigMixin, ROCmConfigHeuristic
):
    """MM Plus MM template heuristic for ROCm"""

    def __init__(self) -> None:
        super().__init__()
        # self.default_num_stages is used to make sure all configs have that in ROCm land
        # for mm_plus_mm, we actually just want stages = 1, as pipelining brings no benefits
        self.default_num_stages = 1
        # Override mm_configs to use mm_plus_mm_configs
        self.mm_configs = self.mm_plus_mm_configs
        # NOTE: overriding exhaustive configs here to be the same as mm_configs
        # as we haven't validated exhaustive support here yet
        # TODO(coconutruben): remove this once we have validated exhaustive support
        # for scaled_mm
        self.exhaustive_configs = self.mm_plus_mm_configs


# CPU template-specific classes


@register_template_heuristic(mm_template.uid, "cpu")
@register_template_heuristic(bmm_template.uid, "cpu")
class CPUMMTemplateConfigHeuristic(MMTemplateConfigMixin, CPUConfigHeuristic):
    """Standard MM template heuristic for CPU"""


@register_template_heuristic(mm_template.uid, "cpu", op_name="addmm")
@register_template_heuristic(bmm_template.uid, "cpu", op_name="baddbmm")
class CPUAddmmTemplateConfigHeuristic(AddMMConfigMixin, CPUMMTemplateConfigHeuristic):
    """Addmm specific mixin for CPU"""


@register_template_heuristic(mm_template.uid, "cpu", op_name="scaled_mm")
class CPUScaledMMTemplateConfigHeuristic(ScaledMMConfigMixin, CPUConfigHeuristic):
    """Scaled MM template heuristic for CPU (non-TMA)"""

    def __init__(self) -> None:
        super().__init__()
        # Override mm_configs to use scaled_mm_configs
        self.mm_configs = self.scaled_mm_configs
        # NOTE: overriding exhaustive configs here to be the same as mm_configs
        # as we haven't validated exhaustive support here yet
        # TODO(coconutruben): remove this once we have validated exhaustive support
        # for scaled_mm
        self.exhaustive_configs = self.scaled_mm_configs


@register_template_heuristic(mm_template.uid, "cpu", op_name="int_mm")
class CPUInt8MMTemplateConfigHeuristic(INT8MMTemplateConfigMixin, CPUConfigHeuristic):
    """Int8 MM template heuristic for CPU"""

    def __init__(self) -> None:
        super().__init__()
        # Override mm_configs to use int8_mm_configs
        self.mm_configs = self.int8_mm_configs
        # NOTE: overriding exhaustive configs here to be the same as mm_configs
        # as we haven't validated exhaustive support here yet
        # TODO(coconutruben): remove this once we have validated exhaustive support
        # for scaled_mm
        self.exhaustive_configs = self.int8_mm_configs


@register_template_heuristic(mm_plus_mm_template.uid, "cpu")
class CPUMMPlusMMTemplateConfigHeuristic(
    MMPlusMMTemplateConfigMixin, CPUConfigHeuristic
):
    """MM Plus MM template heuristic for CPU"""

    def __init__(self) -> None:
        super().__init__()
        # Override mm_configs to use mm_plus_mm_configs
        self.mm_configs = self.mm_plus_mm_configs
        # NOTE: overriding exhaustive configs here to be the same as mm_configs
        # as we haven't validated exhaustive support here yet
        # TODO(coconutruben): remove this once we have validated exhaustive support
        # for scaled_mm
        self.exhaustive_configs = self.mm_plus_mm_configs


# XPU template-specific classes


@register_template_heuristic(mm_template.uid, "xpu")
@register_template_heuristic(bmm_template.uid, "xpu")
class XPUMMTemplateConfigHeuristic(MMTemplateConfigMixin, XPUConfigHeuristic):
    """Standard MM template heuristic for XPU"""

    def __init__(self) -> None:
        super().__init__()

        # TODO(etaf): Design proper exhaustive search space for XPU.
        self.exhaustive_configs = self.mm_configs


@register_template_heuristic(mm_template.uid, "xpu", op_name="addmm")
@register_template_heuristic(bmm_template.uid, "xpu", op_name="baddbmm")
class XPUAddmmTemplateConfigHeuristic(AddMMConfigMixin, XPUMMTemplateConfigHeuristic):
    """Addmm specific mixin for XPU"""


@register_template_heuristic(
    persistent_tma_mm_template.uid,
    "xpu",
)
class XPUPersistentTMATemplateConfigHeuristic(
    TMATemplateConfigMixin, XPUConfigHeuristic
):
    """Persistent TMA template heuristic for XPU"""

    def __init__(self) -> None:
        super().__init__()
        # Override mm_configs to use persistent_mm_configs
        self.mm_configs = self.persistent_mm_configs


@register_template_heuristic(persistent_tma_mm_template.uid, "xpu", op_name="addmm")
class XPUAddmmPersistentTMATemplateConfigHeuristic(
    AddMMConfigMixin, XPUPersistentTMATemplateConfigHeuristic
):
    """Addmm specific mixin for XPU"""


@register_template_heuristic(mm_template.uid, "xpu", op_name="scaled_mm")
class XPUScaledMMTemplateConfigHeuristic(ScaledMMConfigMixin, XPUConfigHeuristic):
    """Scaled MM template heuristic for XPU (non-TMA)"""

    def __init__(self) -> None:
        super().__init__()
        # Override mm_configs to use scaled_mm_configs
        self.mm_configs = self.scaled_mm_configs
        # NOTE: overriding exhaustive configs here to be the same as mm_configs
        # as we haven't validated exhaustive support here yet
        # TODO(coconutruben): remove this once we have validated exhaustive support
        # for scaled_mm
        self.exhaustive_configs = self.scaled_mm_configs


@register_template_heuristic(mm_template.uid, "xpu", op_name="int_mm")
class XPUInt8MMTemplateConfigHeuristic(INT8MMTemplateConfigMixin, XPUConfigHeuristic):
    """Int8 MM template heuristic for XPU"""

    def __init__(self) -> None:
        super().__init__()
        # Override mm_configs to use int8_mm_configs
        self.mm_configs = self.int8_mm_configs
        # NOTE: overriding exhaustive configs here to be the same as mm_configs
        # as we haven't validated exhaustive support here yet
        # TODO(coconutruben): remove this once we have validated exhaustive support
        # for scaled_mm
        self.exhaustive_configs = self.int8_mm_configs


@register_template_heuristic(mm_plus_mm_template.uid, "xpu")
class XPUMMPlusMMTemplateConfigHeuristic(
    MMPlusMMTemplateConfigMixin, XPUConfigHeuristic
):
    """MM Plus MM template heuristic for XPU"""

    def __init__(self) -> None:
        super().__init__()
        # Override mm_configs to use mm_plus_mm_configs
        self.mm_configs = self.mm_plus_mm_configs
        # NOTE: overriding exhaustive configs here to be the same as mm_configs
        # as we haven't validated exhaustive support here yet
        # TODO(coconutruben): remove this once we have validated exhaustive support
        # for scaled_mm
        self.exhaustive_configs = self.mm_plus_mm_configs


# MTIA template-specific classes


@register_template_heuristic(mm_template.uid, "mtia")
@register_template_heuristic(bmm_template.uid, "mtia")
class MTIAMMTemplateConfigHeuristic(MMTemplateConfigMixin, MTIAConfigHeuristic):
    """Standard MM template heuristic for MTIA"""


@register_template_heuristic(mm_template.uid, "mtia", op_name="addmm")
@register_template_heuristic(bmm_template.uid, "mtia", op_name="baddbmm")
class MTIAAddMMTemplateConfigHeuristic(AddMMConfigMixin, MTIAMMTemplateConfigHeuristic):
    """Addmm specific mixin for MTIA"""


@register_template_heuristic(mm_template.uid, "mtia", op_name="scaled_mm")
class MTIAScaledMMTemplateConfigHeuristic(ScaledMMConfigMixin, MTIAConfigHeuristic):
    """Scaled MM template heuristic for MTIA (non-TMA)"""

    def __init__(self) -> None:
        super().__init__()
        # Override mm_configs to use scaled_mm_configs
        self.mm_configs = self.scaled_mm_configs
        # NOTE: overriding exhaustive configs here to be the same as mm_configs
        # as we haven't validated exhaustive support here yet
        # TODO(coconutruben): remove this once we have validated exhaustive support
        # for scaled_mm
        self.exhaustive_configs = self.scaled_mm_configs


@register_template_heuristic(mm_template.uid, "mtia", op_name="int_mm")
class MTIAInt8MMTemplateConfigHeuristic(INT8MMTemplateConfigMixin, MTIAConfigHeuristic):
    """Int8 MM template heuristic for MTIA"""

    def __init__(self) -> None:
        super().__init__()
        # Override mm_configs to use int8_mm_configs
        self.mm_configs = self.int8_mm_configs
        # NOTE: overriding exhaustive configs here to be the same as mm_configs
        # as we haven't validated exhaustive support here yet
        # TODO(coconutruben): remove this once we have validated exhaustive support
        # for scaled_mm
        self.exhaustive_configs = self.int8_mm_configs


@register_template_heuristic(mm_plus_mm_template.uid, "mtia")
class MTIAMMPlusMMTemplateConfigHeuristic(
    MMPlusMMTemplateConfigMixin, MTIAConfigHeuristic
):
    """MM Plus MM template heuristic for MTIA"""

    def __init__(self) -> None:
        super().__init__()
        # Override mm_configs to use mm_plus_mm_configs
        self.mm_configs = self.mm_plus_mm_configs
        # NOTE: overriding exhaustive configs here to be the same as mm_configs
        # as we haven't validated exhaustive support here yet
        # TODO(coconutruben): remove this once we have validated exhaustive support
        # for scaled_mm
        self.exhaustive_configs = self.mm_plus_mm_configs
