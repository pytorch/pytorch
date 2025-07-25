from __future__ import annotations

import dataclasses
import itertools
import math
from functools import partial
from threading import Lock
from typing import Any, Callable, Optional, TYPE_CHECKING

import sympy

import torch
from torch.utils._ordered_set import OrderedSet
from torch.utils._triton import has_triton_stable_tma_api

from . import config, config as inductor_config
from .kernel_inputs import KernelInputs, MMKernelInputs
from .template_registry import register_template_heuristic
from .utils import get_backend_num_stages, get_num_sms, TMA_DESCRIPTOR_SIZE
from .virtualized import V


if TYPE_CHECKING:
    from collections.abc import Generator

    from triton import Config as TritonConfig


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
    hint_override: Optional[int] = None


@dataclasses.dataclass
class GemmConfig(BaseConfig):
    """
    Gemm configuration used for most backends (CPU, CUDA)
    """

    group_m: int = 8


ConvConfig = BaseConfig


# FlexAttention Configs
@dataclasses.dataclass
class FlexConfig:
    """
    Base Config class for flex attention
    - FlexAttn forward, backward and flex decode will use this

    NOTE:
    For flex_attn bwd block_m and block_n are reused for block_m1, block_m2, block_n1, block_n2

    """

    block_m: int
    block_n: int
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
        ]

        # Exhaustive search for mm configs
        self.exhaustive_configs: list[BaseConfig] = [
            GemmConfig(BLOCK_M, BLOCK_N, BLOCK_K, num_stages, num_warps, group_m)
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
            ConvConfig(64, 256, 16, 2, 4),
            ConvConfig(256, 64, 16, 2, 4),
            ConvConfig(1024, 16, 16, 1, 8),
            ConvConfig(128, 128, 32, 2, 8),
            ConvConfig(64, 64, 32, 2, 4),
            ConvConfig(64, 256, 32, 2, 8),
            ConvConfig(256, 64, 32, 2, 8),
        ]

        self.flex_attn_fwd_autotune_configs: list[FlexConfig] = [
            FlexConfig(128, 64, 3, 4),
            FlexConfig(128, 128, 3, 4),
            FlexConfig(128, 128, 2, 8),
            FlexConfig(64, 128, 3, 4),
            FlexConfig(64, 64, 3, 4),
        ]

        self.flex_attn_bwd_autotune_configs: list[FlexConfig] = [
            FlexConfig(BLOCK1, BLOCK2, s, w)
            for BLOCK1 in [32, 64]
            for BLOCK2 in [32, 64, 128]
            for s in [1, 3, 4, 5]  # num_stages
            for w in ([4, 8] if BLOCK1 >= 128 or BLOCK2 >= 128 else [4])
            if BLOCK2 % BLOCK1 == 0
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

        self.exhaustive_flex_attn_bwd_configs: list[FlexConfig] = [
            FlexConfig(BLOCK1, BLOCK2, num_stages, num_warps)
            for BLOCK1 in [16, 32, 64, 128]
            for BLOCK2 in [16, 32, 64, 128]
            for num_stages in [1, 3, 4, 5]
            for num_warps in [2, 4, 8]
            if BLOCK2 % BLOCK1 == 0
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

            if key not in used and (
                max_mm_configs is None or len(used) < max_mm_configs
            ):
                used.add(key)
                kwargs = {
                    "BLOCK_M": conf.block_m,
                    "BLOCK_N": conf.block_n,
                    "BLOCK_K": conf.block_k,
                    "hint_override": conf.hint_override,
                }
                if group_m is not None:
                    kwargs["GROUP_M"] = group_m
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
        from .runtime.runtime_utils import next_power_of_2

        min_block_size = 16
        min_block_size_k = 32 if (has_int8_tensor or self.has_int8_tensor) else 16

        scaled_configs = []
        for hint_override in [None] + config.multi_kernel_hints:
            m_hint = max(
                next_power_of_2(
                    V.graph.sizevars.size_hint(
                        m,
                        fallback=config.unbacked_symint_fallback,  # type: ignore[arg-type]
                        hint_override=hint_override,
                    )
                ),
                min_block_size,
            )
            n_hint = max(
                next_power_of_2(
                    V.graph.sizevars.size_hint(
                        n,
                        fallback=config.unbacked_symint_fallback,  # type: ignore[arg-type]
                        hint_override=hint_override,
                    )
                ),
                min_block_size,
            )
            k_hint = max(
                next_power_of_2(
                    V.graph.sizevars.size_hint(
                        k,
                        fallback=config.unbacked_symint_fallback,  # type: ignore[arg-type]
                        hint_override=hint_override,
                    )
                ),
                min_block_size_k,
            )

            for c in configs:
                scaled_config = dataclasses.replace(
                    c,
                    block_m=max(min(int(c.block_m * scale), m_hint), min_block_size),
                    block_n=max(min(int(c.block_n * scale), n_hint), min_block_size),
                    block_k=max(min(int(c.block_k * scale), k_hint), min_block_size_k),
                    hint_override=hint_override,
                )

                if not exclude(
                    scaled_config.block_m, scaled_config.block_n, scaled_config.block_k
                ):
                    scaled_configs.append(scaled_config)

        return scaled_configs

    def _prune_exhaustive_configs(
        self,
        configs: list[BaseConfig],
        dtype_size: int,
    ) -> list[BaseConfig]:
        import torch

        pruned_configs = []
        for gemm_config in configs:
            device = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(device)
            sm_available = props.shared_memory_per_block_optin  # type: ignore[attr-defined]
            NUM_REG = 255

            acc_regs = math.ceil(
                gemm_config.block_m * gemm_config.block_n / (gemm_config.num_warps * 32)
            )

            shared_mem_accum = dtype_size * (
                gemm_config.block_m * gemm_config.block_k
                + gemm_config.block_n * gemm_config.block_k
            )

            # Will use more shared memory than available
            if shared_mem_accum * gemm_config.num_stages > sm_available:
                continue
            # Lower bound for register spillage, if exceeds the kernel will certainly spill
            elif acc_regs > NUM_REG:
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
    ) -> Generator[TritonConfig, None, None]:
        configs = self._filter_configs(configs)
        scaled_configs = self._scale_mm_configs(
            m, n, k, configs, scale, has_int8_tensor, exclude
        )
        if config.max_autotune_gemm_search_space == "EXHAUSTIVE":
            assert dtype_size > 0, "dtype_size must be provided for exhaustive search"
            scaled_configs = self._prune_exhaustive_configs(scaled_configs, dtype_size)
        return self._finalize_mm_configs(scaled_configs)

    def triton_config(
        self, num_stages: int, num_warps: int, **kwargs: Any
    ) -> TritonConfig:
        from triton import Config as TritonConfig  # type: ignore[attr-defined]

        return TritonConfig(kwargs, num_stages=num_stages, num_warps=num_warps)

    def get_mm_configs(self) -> partial[Generator[TritonConfig, None, None]]:
        return partial(self.preprocess_mm_configs, configs=self.mm_configs)

    def get_exhaustive_mm_configs(self) -> partial[Generator[TritonConfig, None, None]]:
        return partial(self.preprocess_mm_configs, configs=self.exhaustive_configs)

    def get_conv_configs(self) -> partial[Generator[TritonConfig, None, None]]:
        return partial(
            self.preprocess_mm_configs, configs=self.conv_configs, op_name="conv"
        )

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

    def get_flex_attn_bwd_configs(self, head_dim: int, dtype: Any) -> list[FlexConfig]:
        flex_attn_bwd_configs: list[FlexConfig] = []

        if config.max_autotune:
            if config.max_autotune_flex_search_space == "EXHAUSTIVE":
                return self.exhaustive_flex_attn_bwd_configs
            flex_attn_bwd_configs += self.flex_attn_bwd_autotune_configs

        default_config = FlexConfig(16, 16, 1, 4)

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
        )


class CUDAConfigHeuristic(BaseConfigHeuristic):
    """
    Child class for CUDA device specific gemm/flex attention/conv/ configs.
    """

    def __init__(self) -> None:
        super().__init__()

        self.b200_default_flex_config = {
            (torch.float32, 64): FlexConfig(128, 32, 3, 4),
            (torch.float32, 128): FlexConfig(32, 64, 3, 4),
            (torch.float32, 256): FlexConfig(32, 32, 3, 4),
            (torch.bfloat16, 64): FlexConfig(128, 128, 3, 4),
            (torch.bfloat16, 128): FlexConfig(128, 64, 2, 8),
            (torch.bfloat16, 256): FlexConfig(64, 32, 3, 4),
            (torch.float16, 64): FlexConfig(128, 128, 3, 4),
            (torch.float16, 128): FlexConfig(128, 128, 3, 8),
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
            (torch.float16, 128): FlexConfig(128, 128, 3, 8),
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
                default_config = FlexConfig(128, 64, 3, 4)
            if capability >= (10, 0):
                default_config = self.b200_default_flex_config.get(
                    (dtype, head_dim), default_config
                )
            elif capability >= (9, 0):
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

    def get_flex_attn_bwd_configs(self, head_dim: int, dtype: Any) -> list[FlexConfig]:
        capability = torch.cuda.get_device_capability()

        flex_attn_bwd_configs: list[FlexConfig] = []

        if config.max_autotune:
            if config.max_autotune_flex_search_space == "EXHAUSTIVE":
                return self.exhaustive_flex_attn_bwd_configs
            flex_attn_bwd_configs += self.flex_attn_bwd_autotune_configs

        if dtype == torch.float32:
            default_config = FlexConfig(16, 16, 1, 4)
        elif head_dim <= 256 and capability >= (9, 0):  # H100
            if head_dim == 64:
                default_config = FlexConfig(64, 64, 3, 4)
            elif head_dim == 128:
                default_config = FlexConfig(64, 128, 3, 8)
            else:
                default_config = FlexConfig(64, 64, 2, 4)
        elif capability >= (8, 0):  # A100
            if head_dim == 64:
                default_config = FlexConfig(32, 128, 3, 4)
            elif head_dim == 128:
                # SM86/89 have smaller shared memory sizes
                num_stages = 3 if capability[1] == 0 else 2
                default_config = FlexConfig(64, 64, num_stages, 4)
            else:
                default_config = FlexConfig(64, 64, 2, 4)
        else:  # modest hardware or extremely large head_dim
            default_config = FlexConfig(16, 16, 1, 4)

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

        if capability >= (9, 0):  # sm_90+
            if head_dim > 128 and dtype == torch.float32:
                default_config = FlexDecodeConfig(64, 1, 2)
            else:
                default_config = FlexDecodeConfig(64, 3, 2)
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
                group_m,
                matrix_instr_nonkdim,
                waves_per_eu,
                kpack,
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

        self.flex_attn_bwd_autotune_configs: list[FlexConfig] = [
            ROCmFlexConfig(BLOCK1, BLOCK2, 1, w, mfma)
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

        self.exhaustive_flex_attn_bwd_configs: list[FlexConfig] = [
            ROCmFlexConfig(BLOCK1, BLOCK2, num_stages, num_warps, mfma, wpeu)
            for BLOCK1 in [16, 32, 64, 128]
            for BLOCK2 in [16, 32, 64, 128]
            for num_stages in [1, 2]
            for num_warps in [2, 4, 8]
            for mfma in [0, 16]
            for wpeu in [0, int(8 // num_warps)]
            if BLOCK2 % BLOCK1 == 0
        ]

        self.exhaustive_flex_decode_configs: list[FlexDecodeConfig] = [
            ROCmFlexDecodeConfig(block_n, num_stages, num_warps, mfma, wpeu, kpack=2)
            for block_n in [16, 32, 64, 128]
            for num_stages in [1, 2]
            for num_warps in [2, 4, 8]
            for mfma in [0, 16]
            for wpeu in [0, int(8 // num_warps)]
        ]

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

    def get_flex_attn_bwd_configs(self, head_dim: int, dtype: Any) -> list[FlexConfig]:
        flex_attn_bwd_configs: list[FlexConfig] = []

        if config.max_autotune:
            if config.max_autotune_flex_search_space == "EXHAUSTIVE":
                return self.exhaustive_flex_attn_bwd_configs
            flex_attn_bwd_configs += self.flex_attn_bwd_autotune_configs

        if dtype == torch.float32:
            default_config = ROCmFlexConfig(16, 16, 1, 4)
        elif head_dim <= 256:
            if head_dim == 64:
                default_config = ROCmFlexConfig(64, 64, 1, 4)
            elif head_dim == 128:
                default_config = ROCmFlexConfig(64, 128, 1, 8)
            else:
                default_config = ROCmFlexConfig(64, 64, 1, 4)
        else:
            default_config = ROCmFlexConfig(16, 16, 1, 4)

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
    Placeholder child class for XPU specific overrides.
    """


# Template-specific mixin classes


class TemplateConfigHeuristics:
    def get_template_configs(
        self,
        kernel_inputs: KernelInputs,
        layout: Any,
        op_name: str,
    ) -> Generator[dict[str, Any], None, None]:
        """
        Get template configs for the given inputs.
        This is the main entry point for template-specific logic.
        """
        # NOTE: not an abstract class, because that clashed below for the mixin
        # functionality. Can be adjusted, but not a high priority
        yield from {}


class MMTemplateConfigMixin(TemplateConfigHeuristics):
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

    def get_template_configs(
        self,
        kernel_inputs: KernelInputs,
        layout: Any,
        op_name: str,
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

        # Extract M, N, K from kernel_inputs
        m, n, k = kernel_inputs.mnk_symbolic()

        # Extract dtype and device_type from kernel_inputs
        dtype = kernel_inputs.dtype()

        # Get the appropriate config generator
        configs = self._get_config_generator()

        # Generate and process configs
        for c in configs(m, n, k, dtype_size=dtype.itemsize, op_name=op_name):
            template_kwargs = self._convert_config_to_template_kwargs(
                c, m, n, k, layout
            )
            yield template_kwargs

    def _convert_config_to_template_kwargs(
        self,
        triton_config: TritonConfig,
        m: sympy.Integer,
        n: sympy.Integer,
        k: sympy.Integer,
        layout: Any,
    ) -> dict[str, Any]:
        """
        Convert triton config to template kwargs.
        Moved from mm_common.mm_options.
        """
        # Calculate EVEN_K symbolic
        even_k_symbolic = (
            # it isn't worth guarding on this
            sympy.gcd(k, triton_config.kwargs["BLOCK_K"])
            == triton_config.kwargs["BLOCK_K"]
        )

        # Calculate allow_tf32
        allow_tf32 = torch.backends.cuda.matmul.allow_tf32 and (
            not inductor_config.force_same_precision
            or ((m % 16) == 0 and (n % 16) == 0 and (k % 8) == 0)
        )

        # Build options dict
        options_dict = dict(
            EVEN_K=even_k_symbolic,
            ALLOW_TF32=allow_tf32,
            USE_FAST_ACCUM=False,  # Option for _scaled_mm
            ACC_TYPE=self._get_acc_type(layout.dtype),
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


# TMA-specific mixin for TMA templates
class TMAConfigMixin(MMTemplateConfigMixin):
    """
    TMA-specific mixin that uses persistent configs and adds TMA options.
    This inherits from MMTemplateConfigMixin and overrides config generation.
    """

    def _filter_configs(self, configs: list[BaseConfig]) -> list[BaseConfig]:
        """
        TMA specific filtering, as num_warps=2 not safe for TMA
        """
        configs = [c for c in configs if c.num_warps != 2]
        return super()._filter_configs(configs)

    def get_template_configs(
        self,
        kernel_inputs: KernelInputs,
        layout: Any,
        op_name: str,
    ) -> Generator[dict[str, Any], None, None]:
        """
        Generate TMA template configs by calling super and adding TMA-specific options.
        """
        # Get base template configs from superclass
        for template_kwargs in super().get_template_configs(
            kernel_inputs, layout, op_name
        ):
            # Add TMA-specific options (moved from mm_common.persistent_mm_options)
            input_nodes = kernel_inputs.nodes()
            self._add_tma_options(template_kwargs, input_nodes)
            yield template_kwargs

    def _add_tma_options(
        self, template_kwargs: dict[str, Any], input_nodes: list[Any]
    ) -> None:
        """
        Add TMA-specific options to template kwargs.
        Moved from mm_common.persistent_mm_options and mm_common.tma_options.
        """
        # For TMA templates, we need the actual matrix tensors
        mat1 = input_nodes[-2]
        mat2 = input_nodes[-1]

        tma_opts = {
            "A_ROW_MAJOR": not mat1.layout.is_transposed(),
            "B_ROW_MAJOR": not mat2.layout.is_transposed(),
            "NUM_SMS": get_num_sms(),
            "TMA_SIZE": TMA_DESCRIPTOR_SIZE,
            "TMA_EXPERIMENTAL_API": not has_triton_stable_tma_api(),
        }
        template_kwargs.update(tma_opts)


# Scaled MM-specific mixin for scaled MM templates (non-TMA)
class ScaledMMConfigMixin(MMTemplateConfigMixin):
    """
    Scaled MM-specific mixin that uses scaled configs and adds scaled MM options.
    This is for non-TMA scaled MM templates only.
    This inherits from MMTemplateConfigMixin and overrides config generation.
    """

    def get_template_configs(
        self,
        kernel_inputs: KernelInputs,
        layout: Any,
        op_name: str,
    ) -> Generator[dict[str, Any], None, None]:
        """
        Generate scaled MM template configs with scaled MM-specific options.
        Handles the remaining logic from mm_common including assertions and SCALING_ROWWISE.
        """
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

        # Get base template configs from superclass
        for template_kwargs in super().get_template_configs(
            kernel_inputs, layout, op_name
        ):
            # Add scaled MM-specific options (moved from mm_common.scaled_mm_options)
            # Override accumulator type for scaled MM
            template_kwargs["ACC_TYPE"] = "tl.float32"
            # Add SCALING_ROWWISE attribute based on scale_a tensor shape
            template_kwargs["SCALING_ROWWISE"] = len(size_a) == 2

            yield template_kwargs


# Scaled TMA-specific mixin for scaled MM templates with TMA
class ScaledTMAConfigMixin(ScaledMMConfigMixin):
    """
    Scaled TMA-specific mixin that extends ScaledMMConfigMixin with TMA functionality.
    This is for scaled MM templates that use device TMA.
    This inherits from ScaledMMConfigMixin and adds TMA-specific options.
    """

    def _filter_configs(self, configs: list[BaseConfig]) -> list[BaseConfig]:
        """
        TMA specific filtering, as num_warps=2 not safe for TMA
        """
        configs = [c for c in configs if c.num_warps != 2]
        return super()._filter_configs(configs)

    def get_template_configs(
        self,
        kernel_inputs: KernelInputs,
        layout: Any,
        op_name: str,
    ) -> Generator[dict[str, Any], None, None]:
        """
        Generate scaled TMA template configs with both scaled MM and TMA-specific options.
        """
        # Get base scaled MM template configs from superclass
        for template_kwargs in super().get_template_configs(
            kernel_inputs, layout, op_name
        ):
            # Add TMA-specific options for device TMA scaled MM
            template_kwargs["TMA_SIZE"] = TMA_DESCRIPTOR_SIZE
            template_kwargs["NUM_SMS"] = get_num_sms()
            template_kwargs["TMA_EXPERIMENTAL_API"] = not has_triton_stable_tma_api()

            yield template_kwargs


# Template-specific heuristic classes using multiple inheritance


# TODO(coconutruben): replace with template.name once templates are importable
@register_template_heuristic("mm", "cuda", register=torch.version.hip is None)
# TODO(coconutruben): replace with template.name once templates are importable
@register_template_heuristic("bmm", "cuda", register=torch.version.hip is None)
class CUDAMMTemplateConfigHeuristic(MMTemplateConfigMixin, CUDAConfigHeuristic):
    """Standard MM template heuristic for CUDA"""


# TODO(coconutruben): deprecate once autoheuristic is deprecated
# TODO(coconutruben): replace with template.name once templates are importable
@register_template_heuristic("mm-ah", "cuda", register=torch.version.hip is None)
class CUDAMMAHTemplateConfigHeuristic(MMTemplateConfigMixin, CUDAConfigHeuristic):
    """Standard MM template heuristic for CUDA using the extra mm configs only (for autoheuristic)"""

    def __init__(self) -> None:
        super().__init__()
        # Override mm_configs to use scaled_mm_configs
        self.mm_configs = self.extra_mm_configs
        self.exhaustive_configs = self.extra_mm_configs


# TODO(coconutruben): replace with template.name once templates are importable
@register_template_heuristic(
    "mm_persistent_tma", "cuda", register=torch.version.hip is None
)
class CUDAPersistentTMATemplateConfigHeuristic(TMAConfigMixin, CUDAConfigHeuristic):
    """Persistent TMA template heuristic for CUDA"""

    def __init__(self) -> None:
        super().__init__()
        # Override mm_configs to use persistent_mm_configs
        self.mm_configs = self.persistent_mm_configs


# TODO(coconutruben): replace with template.name once templates are importable
@register_template_heuristic(
    "mm", "cuda", register=torch.version.hip is None, op_name="scaled_mm"
)
class CUDAScaledMMTemplateConfigHeuristic(ScaledMMConfigMixin, CUDAConfigHeuristic):
    """Scaled MM template heuristic for CUDA"""

    def __init__(self) -> None:
        super().__init__()
        # Override mm_configs to use scaled_mm_configs
        self.mm_configs = self.scaled_mm_configs
        # NOTE: overriding exhaustive configs here to be the same as mm_configs
        # as we haven't validated exhaustive support here yet
        # TODO(coconutruben): remove this once we have validated exhaustive support
        # for scaled_mm
        self.exhaustive_configs = self.scaled_mm_configs


# TODO(coconutruben): replace with template.name once templates are importable
@register_template_heuristic(
    "scaled_mm_device_tma", "cuda", register=torch.version.hip is None
)
class CUDAScaledTMATemplateConfigHeuristic(ScaledTMAConfigMixin, CUDAConfigHeuristic):
    """Scaled TMA template heuristic for CUDA"""

    def __init__(self) -> None:
        super().__init__()
        # Override mm_configs to use scaled_persistent_mm_configs for TMA
        self.mm_configs = self.scaled_persistent_mm_configs
        # NOTE: overriding exhaustive configs here to be the same as mm_configs
        # as we haven't validated exhaustive support here yet
        # TODO(coconutruben): remove this once we have validated exhaustive support
        # for scaled_mm
        self.exhaustive_configs = self.scaled_persistent_mm_configs


# TODO(coconutruben): replace with template.name once templates are importable
@register_template_heuristic("mm_plus_mm", "cuda", register=torch.version.hip is None)
class CUDAMMPlusMMTemplateConfigHeuristic(MMTemplateConfigMixin, CUDAConfigHeuristic):
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


# TODO(coconutruben): replace with template.name once templates are importable
@register_template_heuristic(
    "mm", "cuda", register=torch.version.hip is None, op_name="int_mm"
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


# TODO(coconutruben): replace with template.name once templates are importable
@register_template_heuristic("mm", "cuda", register=torch.version.hip is not None)
# TODO(coconutruben): replace with template.name once templates are importable
@register_template_heuristic("bmm", "cuda", register=torch.version.hip is not None)
class ROCmMMTemplateConfigHeuristic(MMTemplateConfigMixin, ROCmConfigHeuristic):
    """Standard MM template heuristic for ROCm"""


# TODO(coconutruben): deprecate once autoheuristic is deprecated
# TODO(coconutruben): replace with template.name once templates are importable
@register_template_heuristic("mm-ah", "cuda", register=torch.version.hip is not None)
class ROCmMMAHTemplateConfigHeuristic(MMTemplateConfigMixin, ROCmConfigHeuristic):
    """Standard MM template heuristic for ROCm using the extra mm configs only (for autoheuristic)"""

    def __init__(self) -> None:
        super().__init__()
        # Override mm_configs to use scaled_mm_configs
        self.mm_configs = self.extra_mm_configs
        self.exhaustive_configs = self.extra_mm_configs


# TODO(coconutruben): replace with template.name once templates are importable
@register_template_heuristic(
    "mm", "cuda", register=torch.version.hip is not None, op_name="scaled_mm"
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


# TODO(coconutruben): replace with template.name once templates are importable
@register_template_heuristic(
    "mm", "cuda", register=torch.version.hip is not None, op_name="int_mm"
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


# TODO(coconutruben): replace with template.name once templates are importable
@register_template_heuristic(
    "mm_plus_mm", "cuda", register=torch.version.hip is not None
)
class ROCmMMPlusMMTemplateConfigHeuristic(MMTemplateConfigMixin, ROCmConfigHeuristic):
    """MM Plus MM template heuristic for ROCm"""

    def __init__(self) -> None:
        super().__init__()
        # Override mm_configs to use mm_plus_mm_configs
        self.mm_configs = self.mm_plus_mm_configs
        # NOTE: overriding exhaustive configs here to be the same as mm_configs
        # as we haven't validated exhaustive support here yet
        # TODO(coconutruben): remove this once we have validated exhaustive support
        # for scaled_mm
        self.exhaustive_configs = self.mm_plus_mm_configs


# CPU template-specific classes


@register_template_heuristic("mm", "cpu")
@register_template_heuristic("bmm", "cpu")
class CPUMMTemplateConfigHeuristic(MMTemplateConfigMixin, CPUConfigHeuristic):
    """Standard MM template heuristic for CPU"""


@register_template_heuristic("mm", "cpu", op_name="scaled_mm")
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


@register_template_heuristic("mm", "cpu", op_name="int_mm")
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


@register_template_heuristic("mm_plus_mm", "cpu")
class CPUMMPlusMMTemplateConfigHeuristic(MMTemplateConfigMixin, CPUConfigHeuristic):
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


@register_template_heuristic("mm", "xpu")
@register_template_heuristic("bmm", "xpu")
class XPUMMTemplateConfigHeuristic(MMTemplateConfigMixin, XPUConfigHeuristic):
    """Standard MM template heuristic for XPU"""


@register_template_heuristic("mm", "xpu", op_name="scaled_mm")
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


@register_template_heuristic("mm", "xpu", op_name="int_mm")
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


@register_template_heuristic("mm_plus_mm", "xpu")
class XPUMMPlusMMTemplateConfigHeuristic(MMTemplateConfigMixin, XPUConfigHeuristic):
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
