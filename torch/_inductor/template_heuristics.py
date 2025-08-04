from __future__ import annotations

import dataclasses
import itertools
import math
from functools import partial
from threading import Lock
from typing import Any, Callable, Optional, TYPE_CHECKING

import torch
from torch.utils._ordered_set import OrderedSet

from . import config
from .utils import get_backend_num_stages
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
        exclude: Callable[[int, int, int], bool],
        hint_override: Optional[int] = None,
    ) -> list[BaseConfig]:
        """
        Scales and filters matrix multiplication configs based on input size.
        """
        from .runtime.runtime_utils import next_power_of_2

        min_block_size = 16
        min_block_size_k = 32 if has_int8_tensor else 16

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

    def preprocess_mm_configs(
        self,
        m: int,
        n: int,
        k: int,
        configs: list[BaseConfig],
        has_int8_tensor: bool = False,
        scale: int = 1,
        exclude: Callable[[int, int, int], bool] = lambda m, n, k: False,
        dtype_size: int = 0,
    ) -> Generator[TritonConfig, None, None]:
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

    def get_extra_mm_configs(self) -> partial[Generator[TritonConfig, None, None]]:
        return partial(self.preprocess_mm_configs, configs=self.extra_mm_configs)

    def get_int8_mm_configs(self) -> partial[Generator[TritonConfig, None, None]]:
        return partial(self.preprocess_mm_configs, configs=self.int8_mm_configs)

    def get_mixed_mm_configs(self) -> partial[Generator[TritonConfig, None, None]]:
        mm_configs = (
            self.mm_configs + self.mixed_mm_configs
            if config.max_autotune_gemm_search_space == "EXHAUSTIVE"
            else self.mm_configs
        )
        return partial(self.preprocess_mm_configs, configs=mm_configs)

    def get_persistent_mm_configs(self) -> partial[Generator[TritonConfig, None, None]]:
        persistent_mm_configs = (
            self.exhaustive_configs
            if config.max_autotune_gemm_search_space == "EXHAUSTIVE"
            else self.persistent_mm_configs
        )

        # num_warps=2 not safe for TMA
        persistent_mm_configs = [
            config for config in persistent_mm_configs if config.num_warps != 2
        ]
        return partial(self.preprocess_mm_configs, configs=persistent_mm_configs)

    def get_scaled_mm_configs(self) -> partial[Generator[TritonConfig, None, None]]:
        return partial(self.preprocess_mm_configs, configs=self.scaled_mm_configs)

    def get_scaled_persistent_mm_configs(
        self,
    ) -> partial[Generator[TritonConfig, None, None]]:
        return partial(
            self.preprocess_mm_configs, configs=self.scaled_persistent_mm_configs
        )

    def get_mm_plus_mm_configs(self) -> partial[Generator[TritonConfig, None, None]]:
        return partial(self._finalize_mm_configs, configs=self.mm_plus_mm_configs)

    def get_conv_configs(self) -> partial[Generator[TritonConfig, None, None]]:
        return partial(self.preprocess_mm_configs, configs=self.conv_configs)

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
    pass


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

    def _filter_configs(
        self, configs: list[BaseConfig], new_num_stages: int
    ) -> list[BaseConfig]:
        # TODO: _filter_configs can be removed once backend specific configs are added
        # for all methods
        for c in configs:
            c.num_stages = self.default_num_stages
        return configs

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

    def get_extra_mm_configs(self) -> partial[Generator[TritonConfig, None, None]]:
        filtered_configs = self._filter_configs(
            self.extra_mm_configs, self.default_num_stages
        )
        return partial(self.preprocess_mm_configs, configs=filtered_configs)

    def get_int8_mm_configs(self) -> partial[Generator[TritonConfig, None, None]]:
        filtered_configs = self._filter_configs(
            self.int8_mm_configs, self.default_num_stages
        )
        return partial(self.preprocess_mm_configs, configs=filtered_configs)

    def get_mixed_mm_configs(self) -> partial[Generator[TritonConfig, None, None]]:
        mm_configs = (
            self.mm_configs + self.mixed_mm_configs
            if config.max_autotune_gemm_search_space == "EXHAUSTIVE"
            else self.mm_configs
        )
        filtered_configs = self._filter_configs(mm_configs, self.default_num_stages)
        return partial(self.preprocess_mm_configs, configs=filtered_configs)

    def get_persistent_mm_configs(self) -> partial[Generator[TritonConfig, None, None]]:
        filtered_configs = self._filter_configs(
            self.persistent_mm_configs, self.default_num_stages
        )
        return partial(self.preprocess_mm_configs, configs=filtered_configs)

    def get_scaled_mm_configs(self) -> partial[Generator[TritonConfig, None, None]]:
        filtered_configs = self._filter_configs(
            self.scaled_mm_configs, self.default_num_stages
        )
        return partial(self.preprocess_mm_configs, configs=filtered_configs)

    def get_scaled_persistent_mm_configs(
        self,
    ) -> partial[Generator[TritonConfig, None, None]]:
        filtered_configs = self._filter_configs(
            self.scaled_persistent_mm_configs, self.default_num_stages
        )
        return partial(self.preprocess_mm_configs, configs=filtered_configs)

    def get_mm_plus_mm_configs(self) -> partial[Generator[TritonConfig, None, None]]:
        filtered_configs = self._filter_configs(self.mm_plus_mm_configs, 1)
        return partial(self._finalize_mm_configs, configs=filtered_configs)

    def get_conv_configs(self) -> partial[Generator[TritonConfig, None, None]]:
        filtered_configs = self._filter_configs(
            self.conv_configs, self.default_num_stages
        )
        return partial(self.preprocess_mm_configs, configs=filtered_configs)

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


class MTIAConfigHeuristic(BaseConfigHeuristic):
    """
    Placeholder child class for MTIA specific overrides.
    """
