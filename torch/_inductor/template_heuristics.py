from __future__ import annotations

import dataclasses
import itertools
from functools import partial
from threading import Lock
from typing import Any, Callable, TYPE_CHECKING

import sympy

import torch
from torch.utils._ordered_set import OrderedSet

from . import config
from .kernel_params.params import (
    CPUTritonTemplateKernelParams,
    PersistentTMATritonTemplateMMParams,
    ROCmTritonTemplateMMParams,
    TritonTemplateMMParams,
)
from .utils import get_backend_num_stages, get_num_sms, TMA_DESCRIPTOR_SIZE
from .virtualized import V


if TYPE_CHECKING:
    from collections.abc import Generator

    from triton import Config as TritonConfig

    from .kernel_inputs import MMKernelInputs


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

    def _mm_options(
        self, tconfig: TritonConfig, sym_m: int, sym_n: int, sym_k: int, layout: Any
    ) -> dict[str, Any]:
        """
        Common options to matmul triton templates.
        Inlined from mm_common.mm_options.
        """
        even_k_symbolic = (
            # it isn't worth guarding on this
            sympy.gcd(sym_k, tconfig.kwargs["BLOCK_K"]) == tconfig.kwargs["BLOCK_K"]
        )
        allow_tf32 = torch.backends.cuda.matmul.allow_tf32 and (
            not config.force_same_precision
            or ((sym_m % 16) == 0 and (sym_n % 16) == 0 and (sym_k % 8) == 0)
        )

        # acc_type function inlined
        if layout.dtype in (torch.float16, torch.bfloat16):
            acc_type_val = "tl.float32"
        else:
            acc_type_val = f"tl.{layout.dtype}".replace("torch.", "")

        options_dict = dict(
            EVEN_K=even_k_symbolic,
            ALLOW_TF32=allow_tf32,
            USE_FAST_ACCUM=False,  # Option for _scaled_mm
            ACC_TYPE=acc_type_val,
            num_stages=tconfig.num_stages,
            num_warps=tconfig.num_warps,
            **tconfig.kwargs,
        )

        # If GROUP_M not specified then default to 8
        if "GROUP_M" not in tconfig.kwargs:
            group_m = tconfig.kwargs.get("GROUP_M", 8)
            options_dict["GROUP_M"] = group_m

        return options_dict

    def _persistent_mm_options(self, mat1: Any, mat2: Any) -> dict[str, Any]:
        """
        Options for persistent matrix multiplication templates.
        Inlined from mm_common.persistent_mm_options and mm_common.tma_options.
        """
        res = dict(
            A_ROW_MAJOR=not mat1.layout.is_transposed(),
            B_ROW_MAJOR=not mat2.layout.is_transposed(),
            NUM_SMS=get_num_sms(),
            TMA_SIZE=TMA_DESCRIPTOR_SIZE,
        )

        # Inline tma_options logic
        from torch.utils._triton import has_triton_stable_tma_api

        tma_options_dict = {"TMA_EXPERIMENTAL_API": not has_triton_stable_tma_api()}
        res.update(tma_options_dict)

        return res

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
    ) -> list[BaseConfig]:
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
            scaled_config = dataclasses.replace(
                c,
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

    def _to_params(
        self,
        kernel_inputs: MMKernelInputs,
        config_gen_fn: Callable[[int, int, int], Generator[TritonConfig, None, None]],
    ) -> Generator[TritonTemplateMMParams, None, None]:
        """
        Generate TritonTemplateMMParams for matrix multiplication.
        Uses the provided config generating function to generate TritonConfig objects,
        then converts them to TritonTemplateMMParams.

        Args:
            kernel_inputs: MMKernelInputs object containing input nodes for the problem
            config_gen_fn: Function that generates TritonConfig objects
        """
        input_nodes = kernel_inputs.nodes()
        m, n, k = kernel_inputs.mnk_hinted()
        mm_configs_gen = config_gen_fn(m, n, k)
        for triton_config in mm_configs_gen:
            # Generate the options dictionary using _mm_options
            options = self._mm_options(triton_config, m, n, k, input_nodes[0].layout)

            # Create and yield a TritonTemplateMMParams object
            yield TritonTemplateMMParams(**options)

    def _to_persistent_params(
        self,
        kernel_inputs: MMKernelInputs,
        config_gen_fn: Callable[[int, int, int], Generator[TritonConfig, None, None]],
    ) -> Generator[TritonTemplateMMParams, None, None]:
        """
        Generate PersistentTMATritonTemplateMMParams for persistent matrix multiplication.
        First generates TritonTemplateMMParams using _to_params, then converts them
        to PersistentTMATritonTemplateMMParams by adding persistent-specific parameters.

        Args:
            kernel_inputs: MMKernelInputs object containing input nodes for the problem
            config_gen_fn: Function that generates TritonConfig objects
        """
        input_nodes = kernel_inputs.nodes()
        # Get persistent MM options using helper method
        persistent_options = self._persistent_mm_options(input_nodes[0], input_nodes[1])

        # Generate base params first
        for base_params in self._to_params(kernel_inputs, config_gen_fn):
            # Convert to PersistentTMATritonTemplateMMParams by adding persistent-specific parameters
            yield PersistentTMATritonTemplateMMParams(
                **base_params.kwargs(),
                **persistent_options,
            )

    def get_mm_params(
        self, kernel_inputs: MMKernelInputs
    ) -> partial[Generator[TritonTemplateMMParams, None, None]]:
        """
        Return a partial function that generates TritonTemplateMMParams for matrix multiplication.
        """
        return partial(
            self._to_params,
            kernel_inputs=kernel_inputs,
            config_gen_fn=self.get_mm_configs(),
        )

    def get_exhaustive_mm_params(
        self, kernel_inputs: MMKernelInputs
    ) -> partial[Generator[TritonTemplateMMParams, None, None]]:
        """
        Return a partial function that generates TritonTemplateMMParams for exhaustive matrix multiplication.
        """
        return partial(
            self._to_params,
            kernel_inputs=kernel_inputs,
            config_gen_fn=self.get_exhaustive_mm_configs(),
        )

    def get_persistent_mm_params(
        self, kernel_inputs: MMKernelInputs
    ) -> partial[Generator[TritonTemplateMMParams, None, None]]:
        """
        Return a partial function that generates TritonTemplateMMParams for persistent matrix multiplication.
        """
        return partial(
            self._to_persistent_params,
            kernel_inputs=kernel_inputs,
            config_gen_fn=self.get_persistent_mm_configs(),
        )

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
    def _to_params(
        self,
        kernel_inputs: MMKernelInputs,
        config_gen_fn: Callable[[int, int, int], Generator[TritonConfig, None, None]],
    ) -> Generator[TritonTemplateMMParams, None, None]:
        """
        Generate CPUTritonKernelParams for matrix multiplication on CPU.
        Uses the provided config generating function to generate TritonConfig objects,
        then converts them to CPUTritonKernelParams.

        Args:
            kernel_inputs: MMKernelInputs object containing input nodes for the problem
            config_gen_fn: Function that generates TritonConfig objects
        """
        input_nodes = kernel_inputs.nodes()
        m, n, k = kernel_inputs.mnk_hinted()
        mm_configs_gen = config_gen_fn(m, n, k)
        for triton_config in mm_configs_gen:
            # Generate the options dictionary using _mm_options
            options = self._mm_options(triton_config, m, n, k, input_nodes[0].layout)

            # Add CPU-specific parameters
            # Thresholds are experimentally determined to reduce Triton CPU compile times
            exclude = m * n > 2**13  # _is_large_block_for_cpu from mm.py

            # Create and yield a CPUTritonKernelParams object
            yield CPUTritonTemplateKernelParams(**options, exclude=exclude)

    def get_mm_params(
        self, kernel_inputs: MMKernelInputs
    ) -> partial[Generator[TritonTemplateMMParams, None, None]]:
        """
        Return a partial function that generates TritonTemplateMMParams for matrix multiplication.
        """
        return partial(
            self._to_params,
            kernel_inputs=kernel_inputs,
            config_gen_fn=self.get_mm_configs(),
        )

    def get_exhaustive_mm_params(
        self, kernel_inputs: MMKernelInputs
    ) -> partial[Generator[TritonTemplateMMParams, None, None]]:
        """
        Return a partial function that generates TritonTemplateMMParams for exhaustive matrix multiplication.
        """
        return partial(
            self._to_params,
            kernel_inputs=kernel_inputs,
            config_gen_fn=self.get_exhaustive_mm_configs(),
        )

    def get_persistent_mm_params(
        self, kernel_inputs: MMKernelInputs
    ) -> partial[Generator[TritonTemplateMMParams, None, None]]:
        """
        Return a partial function that generates TritonTemplateMMParams for persistent matrix multiplication.
        """
        return partial(
            self._to_params,
            kernel_inputs=kernel_inputs,
            config_gen_fn=self.get_persistent_mm_configs(),
        )


class CUDAConfigHeuristic(BaseConfigHeuristic):
    pass


class ROCmConfigHeuristic(BaseConfigHeuristic):
    """ROCm-specific config heuristic with AMD backend parameters like matrix_instr_nonkdim, waves_per_eu, and kpack."""

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

    def _to_params(
        self,
        kernel_inputs: MMKernelInputs,
        config_gen_fn: Callable[[int, int, int], Generator[TritonConfig, None, None]],
    ) -> Generator[TritonTemplateMMParams, None, None]:
        """
        Generate ROCmTritonTemplateMMParams for matrix multiplication on ROCm.
        Uses the provided config generating function to generate TritonConfig objects,
        then converts them to ROCmTritonTemplateMMParams.

        Args:
            kernel_inputs: MMKernelInputs object containing input nodes for the problem
            config_gen_fn: Function that generates TritonConfig objects
        """
        input_nodes = kernel_inputs.nodes()
        m, n, k = kernel_inputs.mnk_hinted()
        mm_configs_gen = config_gen_fn(m, n, k)
        for triton_config in mm_configs_gen:
            # Generate the options dictionary using _mm_options
            options = self._mm_options(triton_config, m, n, k, input_nodes[0].layout)

            # Create and yield a ROCmTritonTemplateMMParams object
            yield ROCmTritonTemplateMMParams(**options)

    def get_mm_params(
        self, kernel_inputs: MMKernelInputs
    ) -> partial[Generator[TritonTemplateMMParams, None, None]]:
        """
        Return a partial function that generates TritonTemplateMMParams for matrix multiplication.
        """
        return partial(
            self._to_params,
            kernel_inputs=kernel_inputs,
            config_gen_fn=self.get_mm_configs(),
        )

    def get_exhaustive_mm_params(
        self, kernel_inputs: MMKernelInputs
    ) -> partial[Generator[TritonTemplateMMParams, None, None]]:
        """
        Return a partial function that generates TritonTemplateMMParams for exhaustive matrix multiplication.
        """
        return partial(
            self._to_params,
            kernel_inputs=kernel_inputs,
            config_gen_fn=self.get_exhaustive_mm_configs(),
        )

    def get_persistent_mm_params(
        self, kernel_inputs: MMKernelInputs
    ) -> partial[Generator[TritonTemplateMMParams, None, None]]:
        """
        Return a partial function that generates TritonTemplateMMParams for persistent matrix multiplication.
        """
        return partial(
            self._to_params,
            kernel_inputs=kernel_inputs,
            config_gen_fn=self.get_persistent_mm_configs(),
        )


class PersistentTMAConfigHeuristics(BaseConfigHeuristic):
    """
    Config heuristic for Persistent TMA templates that uses persistent_mm_configs
    instead of regular mm_configs.
    """

    def __init__(self) -> None:
        super().__init__()
        # Override mm_configs to use persistent_mm_configs
        self.mm_configs = self.persistent_mm_configs

    def _to_params(
        self,
        kernel_inputs: MMKernelInputs,
        config_gen_fn: Callable[[int, int, int], Generator[TritonConfig, None, None]],
    ) -> Generator[TritonTemplateMMParams, None, None]:
        """
        Generate PersistentTMATritonTemplateMMParams for persistent matrix multiplication.
        First calls the super's _to_params, then adds persistent-specific parameters.

        Args:
            kernel_inputs: MMKernelInputs object containing input nodes for the problem
            config_gen_fn: Function that generates TritonConfig objects
        """
        # Call the super's _to_params to get base parameters
        for base_params in super()._to_params(kernel_inputs, config_gen_fn):
            input_nodes = kernel_inputs.nodes()
            # Get persistent MM options
            persistent_options = self._persistent_mm_options(
                input_nodes[0], input_nodes[1]
            )

            # Convert to PersistentTMATritonTemplateMMParams by adding persistent-specific parameters
            yield PersistentTMATritonTemplateMMParams(
                **base_params.kwargs(),
                **persistent_options,
            )


class XPUConfigHeuristic(BaseConfigHeuristic):
    pass
