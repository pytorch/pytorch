from __future__ import annotations

import itertools
from collections import namedtuple
from functools import partial
from threading import Lock
from typing import Any, Callable, TYPE_CHECKING

from torch.utils._ordered_set import OrderedSet

from . import config
from .utils import get_backend_num_stages
from .virtualized import V


if TYPE_CHECKING:
    from triton import Config as TritonConfig
    from collections.abc import Generator, Sequence

class BaseConfigSingleton(type):
    """
    Thread-safe implementation of single to be used in the config heuristic subclasses
    to ensure heavy __init__ calls are not repeatedly run
    """

    _instances: dict[type[Any], Any] = {}
    _lock: Lock = Lock()

    def __call__(
        cls: BaseConfigSingleton, *args: Any, **kwargs: Any
    ) -> BaseConfigHeuristic:
        with cls._lock:
            if cls not in cls._instances:
                instance = super().__call__()
                cls._instances[cls] = instance
            return cls._instances[cls]


Config = namedtuple(
    "Config", ["block_m", "block_n", "block_k", "num_stages", "num_warps"]
)


class BaseConfigHeuristic(metaclass=BaseConfigSingleton):
    """
    Base class for mm_configs, device specific triton kernels config inherit from here
    """

    def __init__(self) -> None:
        # List of dictionaries to store the kernel configs. Configs that evaluate to true
        # will be utilised on the target platform. The configs are as follows:
        # (BLOCK_M, BLOCK_N, BLOCK_K, num_stages, num_warps)
        self.mm_configs = [
            Config(32, 32, 16, 1, 2),
            Config(32, 32, 128, 2, 4),
            Config(32, 64, 32, 5, 8),
            Config(64, 32, 32, 5, 8),
            Config(64, 32, 128, 5, 4),
            Config(64, 64, 16, 2, 4),
            Config(64, 64, 32, 2, 4),
            Config(64, 64, 64, 3, 8),
            Config(64, 64, 128, 5, 4),
            Config(64, 128, 32, 3, 4),
            Config(64, 128, 32, 4, 8),
            Config(64, 128, 64, 3, 4),
            Config(64, 128, 128, 4, 4),
            Config(128, 64, 32, 3, 4),
            Config(128, 64, 32, 4, 8),
            Config(128, 128, 32, 2, 8),
            Config(128, 128, 32, 3, 4),
            Config(128, 128, 64, 3, 4),
            Config(128, 128, 64, 5, 8),
        ]

        # Exhaustive search for mm configs
        self.exhaustive_configs = [
            Config(BLOCK_M, BLOCK_N, BLOCK_K, num_stages, num_warps)
            for BLOCK_M, BLOCK_N, BLOCK_K in itertools.product(
                [16, 32, 64, 128, 256], repeat=3
            )
            for num_stages in [1, 2, 3, 4, 5]
            for num_warps in [2, 4, 8]
        ]

        # these are only used in tuned_mm when AutoHeuristic is enabled
        # the idea is that when AutoHeuristic collects data to learn a heuristic, more configs are autotuned
        # when the learned heuristic is used, the learned heuristic reduces the number of configs down to 10
        # which saves compilation time (since less configs are autotuned) and potentially increase performance
        # because the learned heuristic might predict a config that is not part mm_configs
        self.extra_mm_configs = [
            Config(16, 32, 16, 3, 2),
            Config(16, 32, 32, 4, 2),
            Config(16, 32, 32, 5, 2),
            Config(64, 64, 128, 3, 4),
            Config(128, 64, 32, 2, 2),
            Config(128, 64, 64, 3, 8),
            Config(128, 64, 128, 4, 8),
            Config(128, 128, 32, 4, 4),
            Config(128, 128, 64, 3, 8),
            Config(128, 128, 64, 5, 4),
        ]

        self.int8_mm_configs = [
            Config(64, 64, 32, 2, 4),
            Config(64, 128, 32, 3, 4),
            Config(128, 64, 32, 3, 4),
            Config(64, 128, 32, 4, 8),
            Config(128, 64, 32, 4, 8),
            Config(64, 32, 32, 5, 8),
            Config(32, 64, 32, 5, 8),
            Config(128, 128, 32, 2, 8),
            Config(64, 64, 64, 3, 8),
            Config(128, 256, 128, 3, 8),
            Config(256, 128, 128, 3, 8),
        ]

        self.mixed_mm_configs = [
            Config(16, 128, 256, 3, 4),
            Config(16, 128, 256, 5, 8),
        ]

        self.persistent_mm_configs = [
            Config(128, 256, 64, 3, 8),
            Config(128, 128, 64, 3, 8),
            Config(128, 128, 128, 3, 8),
            Config(128, 128, 128, 3, 4),
            Config(128, 128, 64, 4, 8),
        ]

        self.scaled_mm_configs = [
            Config(128, 256, 32, 3, 8),
            Config(256, 128, 32, 3, 8),
            Config(256, 64, 32, 4, 4),
            Config(64, 256, 32, 4, 4),
            Config(128, 128, 32, 4, 4),
            Config(128, 64, 32, 4, 4),
            Config(64, 128, 32, 4, 4),
            Config(128, 32, 32, 4, 4),
            Config(64, 32, 32, 5, 2),
            Config(256, 128, 128, 3, 8),
            Config(256, 64, 128, 4, 4),
            Config(64, 256, 128, 4, 4),
            Config(128, 128, 128, 4, 4),
            Config(128, 64, 64, 4, 4),
            Config(64, 128, 64, 4, 4),
            Config(128, 32, 64, 4, 4),
            Config(64, 32, 64, 5, 2),
            Config(16, 32, 32, 2, 2),
            Config(16, 64, 32, 2, 2),
            Config(16, 128, 32, 2, 4),
            Config(16, 256, 32, 2, 4),
            Config(16, 32, 64, 2, 2),
            Config(16, 64, 64, 2, 2),
            Config(16, 128, 64, 2, 4),
            Config(16, 256, 64, 2, 4),
            Config(32, 32, 32, 2, 2),
            Config(32, 64, 32, 2, 2),
            Config(32, 128, 32, 2, 4),
            Config(32, 256, 32, 2, 4),
            Config(32, 32, 64, 2, 2),
            Config(32, 64, 64, 2, 2),
            Config(32, 128, 64, 2, 4),
            Config(32, 256, 64, 2, 4),
            Config(16, 32, 32, 3, 2),
            Config(16, 64, 32, 3, 2),
            Config(16, 128, 32, 3, 4),
            Config(16, 256, 32, 3, 4),
            Config(16, 32, 64, 3, 2),
            Config(16, 64, 64, 3, 2),
            Config(16, 128, 64, 3, 4),
            Config(16, 256, 64, 3, 4),
            Config(32, 32, 32, 3, 2),
            Config(32, 64, 32, 3, 2),
            Config(32, 128, 32, 3, 4),
            Config(32, 256, 32, 3, 4),
            Config(32, 32, 64, 3, 2),
            Config(32, 64, 64, 3, 2),
            Config(32, 128, 64, 3, 4),
            Config(32, 256, 64, 3, 4),
            Config(16, 32, 32, 4, 2),
            Config(16, 64, 32, 4, 2),
            Config(16, 128, 32, 4, 4),
            Config(16, 256, 32, 4, 4),
            Config(16, 32, 64, 4, 2),
            Config(16, 64, 64, 4, 2),
            Config(16, 128, 64, 4, 4),
            Config(16, 256, 64, 4, 4),
            Config(32, 32, 32, 4, 2),
            Config(32, 64, 32, 4, 2),
            Config(32, 128, 32, 4, 4),
            Config(32, 256, 32, 4, 4),
            Config(32, 32, 64, 4, 2),
            Config(32, 64, 64, 4, 2),
            Config(32, 128, 64, 4, 4),
            Config(32, 256, 64, 4, 4),
            Config(16, 32, 32, 5, 2),
            Config(16, 64, 32, 5, 2),
            Config(16, 128, 32, 5, 4),
            Config(16, 256, 32, 5, 4),
            Config(16, 32, 64, 5, 2),
            Config(16, 64, 64, 5, 2),
            Config(16, 128, 64, 5, 4),
            Config(16, 256, 64, 5, 4),
            Config(32, 32, 32, 5, 2),
            Config(32, 64, 32, 5, 2),
            Config(32, 128, 32, 5, 4),
            Config(32, 256, 32, 5, 4),
            Config(32, 32, 64, 5, 2),
            Config(32, 64, 64, 5, 2),
            Config(32, 128, 64, 5, 4),
            Config(32, 256, 64, 5, 4),
            Config(16, 32, 32, 6, 2),
            Config(16, 64, 32, 6, 2),
            Config(16, 128, 32, 6, 4),
            Config(16, 256, 32, 6, 4),
            Config(16, 32, 64, 6, 2),
            Config(16, 64, 64, 6, 2),
            Config(16, 128, 64, 6, 4),
            Config(16, 256, 64, 6, 4),
            Config(32, 32, 32, 6, 2),
            Config(32, 64, 32, 6, 2),
            Config(32, 128, 32, 6, 4),
            Config(32, 256, 32, 6, 4),
            Config(32, 32, 64, 6, 2),
            Config(32, 64, 64, 6, 2),
            Config(32, 128, 64, 6, 4),
            Config(32, 256, 64, 6, 4),
        ]

        self.scaled_persistent_mm_configs = [
            Config(128, 128, 64, 3, 8),
            Config(128, 128, 128, 3, 8),
            Config(128, 128, 128, 4, 8),
            Config(128, 128, 128, 4, 4),
            Config(128, 128, 128, 3, 4),
            Config(128, 128, 128, 5, 4),
            Config(128, 128, 128, 5, 8),
            Config(128, 128, 128, 6, 8),
            Config(128, 128, 64, 4, 8),
        ]

        # TODO: Unify with other gemm patterns, mm_plus_mm currently follows
        # slightly different pattern than rest
        self.mm_plus_mm_configs = [
            Config(64, 64, 32, 2, 4),
            Config(64, 64, 32, 3, 8),
            Config(64, 64, 32, 4, 16),
            Config(64, 32, 32, 4, 8),
            Config(32, 64, 32, 4, 8),
            Config(128, 128, 32, 1, 8),
            Config(64, 64, 64, 1, 8),
            Config(32, 32, 128, 1, 8),
            Config(64, 64, 16, 2, 4),
            Config(32, 32, 16, 1, 2),
        ]

        self.conv_configs = [
            Config(64, 256, 16, 2, 4),
            Config(256, 64, 16, 2, 4),
            Config(1024, 16, 16, 1, 8),
            Config(128, 128, 32, 2, 8),
            Config(64, 64, 32, 2, 4),
            Config(64, 256, 32, 2, 8),
            Config(256, 64, 32, 2, 8),
        ]

    def _finalize_mm_configs(
        self,
        configs: list[Config],
    ) -> Generator[TritonConfig, None, None]:
        """
        Finalizes configs after scaling, applying additional constraints.
        """
        used = OrderedSet[Config]()

        max_mm_configs = config.test_configs.max_mm_configs

        for block_m, block_n, block_k, num_stages, num_warps in configs:
            # Each warp computes a 16x16 tile = 256 elements
            num_warps = min(num_warps, block_m * block_n // 256)

            if (
                Config(block_m, block_n, block_k, num_stages, num_warps)
            ) not in used and (max_mm_configs is None or len(used) < max_mm_configs):
                used.add(Config(block_m, block_n, block_k, num_stages, num_warps))
                yield self.triton_config(
                    BLOCK_M=block_m,
                    BLOCK_N=block_n,
                    BLOCK_K=block_k,
                    num_stages=num_stages,
                    num_warps=num_warps,
                )

    def _scale_mm_configs(
        self,
        m: int,
        n: int,
        k: int,
        configs: Sequence[Config],
        scale: float,
        has_int8_tensor: bool,
        exclude: Callable[[int, int, int], bool],
    ) -> list[Config]:
        """
        Scales and filters matrix multiplication configs based on input size.
        """
        from .runtime.runtime_utils import next_power_of_2

        min_block_size = 16
        min_block_size_k = 32 if has_int8_tensor else 16

        m = max(
            next_power_of_2(
                V.graph.sizevars.size_hint(
                    m,
                    fallback=config.unbacked_symint_fallback,  # type: ignore[arg-type]
                )
            ),
            min_block_size,
        )
        n = max(
            next_power_of_2(
                V.graph.sizevars.size_hint(
                    n,
                    fallback=config.unbacked_symint_fallback,  # type: ignore[arg-type]
                )
            ),
            min_block_size,
        )
        k = max(
            next_power_of_2(
                V.graph.sizevars.size_hint(
                    k,
                    fallback=config.unbacked_symint_fallback,  # type: ignore[arg-type]
                )
            ),
            min_block_size_k,
        )

        scaled_configs = []
        for c in configs:
            scaled_config = c._replace(
                block_m=max(min(int(c.block_m * scale), m), min_block_size),
                block_n=max(min(int(c.block_n * scale), n), min_block_size),
                block_k=max(min(int(c.block_k * scale), k), min_block_size_k),
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
        configs: Sequence[Config],
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

        # Exhaustive search for mm configs
        self.exhaustive_configs = [
            Config(BLOCK_M, BLOCK_N, BLOCK_K, num_stages, num_warps)
            for BLOCK_M, BLOCK_N, BLOCK_K in itertools.product(
                [16, 32, 64, 128, 256], repeat=3
            )
            for num_stages in [1, self.default_num_stages]
            for num_warps in [4, 8]
        ]

    def _filter_configs(
        self, configs: list[Config], new_num_stages: int
    ) -> list[Config]:
        filtered_configs = [
            c._replace(num_stages=self.default_num_stages) for c in configs
        ]
        return filtered_configs

    def _finalize_mm_configs(
        self,
        configs: list[Config],
    ) -> Generator[TritonConfig, None, None]:
        used = OrderedSet[tuple[Config, int]]()

        max_mm_configs = config.test_configs.max_mm_configs
        for block_m, block_n, block_k, num_stages, num_warps in configs:
            # each warp computes 16x16 tile = 256
            num_warps = min(num_warps, block_m * block_n // 256)

            for matrix_instr_nonkdim in [0, 16]:
                if matrix_instr_nonkdim != 0 and (
                    block_m % matrix_instr_nonkdim != 0
                    or block_n % matrix_instr_nonkdim != 0
                ):
                    #  block_m and block_n must be a multiple of matrix_instr_nonkdim
                    continue
                if (
                    Config(
                        block_m,
                        block_n,
                        block_k,
                        num_stages,
                        num_warps,
                    ),
                    matrix_instr_nonkdim,
                ) not in used and (
                    max_mm_configs is None or len(used) < max_mm_configs
                ):
                    used.add(
                        (
                            Config(
                                block_m,
                                block_n,
                                block_k,
                                num_stages,
                                num_warps,
                            ),
                            matrix_instr_nonkdim,
                        )
                    )

                    yield self.triton_config(
                        BLOCK_M=block_m,
                        BLOCK_N=block_n,
                        BLOCK_K=block_k,
                        num_stages=num_stages,
                        num_warps=num_warps,
                        matrix_instr_nonkdim=matrix_instr_nonkdim,
                    )

    def get_mm_configs(self) -> partial[Generator[TritonConfig, None, None]]:
        filtered_configs = self._filter_configs(
            self.mm_configs, self.default_num_stages
        )
        return partial(self.preprocess_mm_configs, configs=filtered_configs)

    def get_exhaustive_mm_configs(self) -> partial[Generator[TritonConfig, None, None]]:
        filtered_configs = self._filter_configs(
            self.exhaustive_configs, self.default_num_stages
        )
        return partial(self.preprocess_mm_configs, configs=filtered_configs)

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
