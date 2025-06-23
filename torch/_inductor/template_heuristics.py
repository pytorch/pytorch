from __future__ import annotations

import dataclasses
import itertools
from functools import partial
from threading import Lock
from typing import Any, Callable, TYPE_CHECKING

from torch.utils._ordered_set import OrderedSet

from . import config
from .utils import get_backend_num_stages
from .virtualized import V


if TYPE_CHECKING:
    from collections.abc import Generator

    from triton import Config as TritonConfig


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


@dataclasses.dataclass
class GemmConfig(BaseConfig):
    """
    Gemm configuration used for most backends (CPU, CUDA)
    """

    group_m: int = 8


ConvConfig = BaseConfig


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
            num_warps = min(conf.num_warps, conf.block_m * conf.block_n // 256)

            # Construct key for finding duplicate configs
            key: tuple[int, ...] = (
                conf.block_m,
                conf.block_n,
                conf.block_k,
                conf.num_stages,
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
                    "num_stages": conf.num_stages,
                    "num_warps": num_warps,
                }
                if group_m is not None:
                    kwargs["GROUP_M"] = group_m
                yield self.triton_config(**kwargs)

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
                )

                if not exclude(
                    scaled_config.block_m, scaled_config.block_n, scaled_config.block_k
                ):
                    scaled_configs.append(scaled_config)

        return scaled_configs

    def preprocess_mm_configs(
        self,
        m: int,
        n: int,
        k: int,
        configs: list[BaseConfig],
        has_int8_tensor: bool = False,
        scale: int = 1,
        exclude: Callable[[int, int, int], bool] = lambda m, n, k: False,
    ) -> Generator[TritonConfig, None, None]:
        scaled_configs = self._scale_mm_configs(
            m, n, k, configs, scale, has_int8_tensor, exclude
        )
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
        return partial(self.preprocess_mm_configs, configs=self.persistent_mm_configs)

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


class CPUConfigHeuristic(BaseConfigHeuristic):
    pass


class CUDAConfigHeuristic(BaseConfigHeuristic):
    pass


class ROCmConfigHeuristic(BaseConfigHeuristic):
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


class XPUConfigHeuristic(BaseConfigHeuristic):
    pass
