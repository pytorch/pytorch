from __future__ import annotations

import logging
import typing
from typing import Any, Literal, Optional, TYPE_CHECKING, Union

import sympy

import torch

from . import config
from .codecache import write_text
from .kernel_lut import MMProblem
from .metrics import get_metric_table, is_metric_table_enabled
from .runtime.hints import DeviceProperties, ReductionHint
from .scheduler import BaseSchedulerNode, Scheduler, WhyNoFuse
from .template_heuristics import (
    BaseConfigHeuristic,
    CPUConfigHeuristic,
    CUDAConfigHeuristic,
    ROCmConfigHeuristic,
    XPUConfigHeuristic,
)
from .virtualized import V


log = logging.getLogger(__name__)


if TYPE_CHECKING:
    from collections.abc import Generator
    from functools import partial

    from triton import Config as TritonConfig

    from torch._inductor.ir import Layout, MutableBox
    from torch.utils._ordered_set import OrderedSet

    from .codegen.simd_kernel_features import SIMDKernelFeatures
    from .codegen.triton import TritonKernel

"""
Neural network model for predicting triton kernel performance.

This module provides functionality to load and use a pre-trained neural network
for predicting the performance of triton kernels.
"""

import os
import time
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from pyre_extensions import assert_is_instance  # type: ignore[import-untyped]

import torch
import torch.nn as nn
from torch._inductor.kernel_lut import TritonGEMMConfig


# Default model path - can be overridden by environment variable
script_dir = os.path.dirname(__file__)
DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "aoti_mm_model.pt2")
MODEL_PATH = os.environ.get("TRITON_KERNEL_SELECTION_MODEL_PATH", DEFAULT_MODEL_PATH)
import logging


log = logging.getLogger(__name__)
# turn on info logging
logging.basicConfig(level=logging.INFO)


class NeuralNetwork(nn.Module):
    """
    Multilayer perceptron with a single output.

    It is designed for modeling runtime when there is a constant overhead of
    `kernel_overhead` and the non-overhead runtime tends to be easier to model
    on a log scale (e.g.  doubling a dimension involved in a matrix
    multiplication results in runtime roughly doubling.)
    """

    def __init__(
        self,
        n_inputs: int,
        hidden_layer_widths: Sequence[int],
        kernel_overhead: float = 0.00541,
    ) -> None:
        """
        Args:
            n_inputs: Number of inputs
            hidden_layer_widths: Hidden layer widths
            kernel_overhead: Overhead of the kernel, assumed to be constant. The
                default of 0.00541 is the lowest runtime seen in Triton H100 data.
        """
        super().__init__()
        self.n_inputs = n_inputs
        self.kernel_overhead = kernel_overhead
        self.log_kernel_overhead: float = torch.log(
            torch.tensor(kernel_overhead, device="cuda")
        ).item()
        all_layer_widths = list(hidden_layer_widths) + [1]
        all_input_widths = [n_inputs] + list(hidden_layer_widths)
        layers: list[nn.Module] = []
        for n_in, n_out in zip(all_input_widths, all_layer_widths, strict=True):
            layers.append(nn.Linear(n_in, n_out))
            layers.append(nn.BatchNorm1d(n_out))
            layers.append(nn.ReLU())

        self.linear_relu_stack = nn.Sequential(*layers[:-2])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict as log(exp(inputs) + self.kernel_overhead).

        Works well for predicting log(runtime) when runtime contains a constant
        overhead of `kernel_overhead`. (The log specification means that this
        wouldn't be trivially modeled with a bias term.)

        Probably could have fit the overhead rather than hard-coding it by
        having `self.kernel_overhead` be a tunable parameter or by having exp
        and log layers.
        """
        log_base_pred = self.linear_relu_stack(x)
        log_overhead_tsr = torch.full_like(
            input=log_base_pred, fill_value=self.log_kernel_overhead, device="cuda"
        )
        return torch.logsumexp(
            torch.stack([log_base_pred, log_overhead_tsr], dim=-1), dim=-1
        )


def get_nn_x(
    df: pd.DataFrame, mean: torch.Tensor | None = None, std: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Standardize the data and convert it to a tensor."""
    x_df = df[
        [
            "dtype_size",
            "dim_m",
            "dim_n",
            "dim_k",
            "total_gb",
            "total_gflop",
            "flops_per_byte",
            "config_block_k",
            "config_block_m",
            "config_block_n",
            "config_num_stages",
            "config_num_warps",
        ]
    ].copy()
    for col in x_df.columns:
        x_df[col] = np.log(x_df[col])

    x_tens = torch.from_numpy(x_df.astype(float).to_numpy()).to(device="cuda")
    if mean is None:
        mean = torch.from_numpy(
            assert_is_instance(x_df.mean(), pd.Series).to_numpy()
        ).to(device="cuda")
    if std is None:
        std = torch.from_numpy(assert_is_instance(x_df.std(), pd.Series).to_numpy()).to(
            device="cuda"
        )
    x_tens -= mean
    x_tens /= std
    return x_tens.to(torch.float32), mean, std


def get_total_gb_feature(df: pd.DataFrame) -> pd.Series:
    """
    Calculate the total gigabytes feature from the dataframe.

    Args:
        df: DataFrame containing the necessary columns for calculation

    Returns:
        Series containing the calculated total gigabytes
    """
    # Calculate memory access in bytes
    m, n, k = df["dim_m"], df["dim_n"], df["dim_k"]
    dtype_size = df["dtype_size"] / 8  # Convert bits to bytes

    # A: m×k, B: k×n, C: m×n
    return ((m * k + k * n + m * n) * dtype_size) / 1e9  # Convert to GB


def get_total_gflop_feature(df: pd.DataFrame) -> pd.Series:
    """
    Calculate the total gigaflops feature from the dataframe.

    Args:
        df: DataFrame containing the necessary columns for calculation

    Returns:
        Series containing the calculated total gigaflops
    """
    # For matrix multiplication, flops = 2 * m * n * k
    m, n, k = df["dim_m"], df["dim_n"], df["dim_k"]
    return (2 * m * n * k) / 1e9  # Convert to GFLOP


class ModelWrapper:
    """
    Wrapper for the neural network model that handles encoding inputs and decoding outputs.

    This class provides methods to prepare inputs for the model and interpret its outputs,
    handling the necessary standardization and feature engineering.
    """

    def __init__(self) -> None:
        """Initialize the model wrapper with the pre-trained model and standardization parameters."""
        start_time = time.time()
        self.model = NeuralNetwork(
            n_inputs=12, hidden_layer_widths=[2**8 for _ in range(6)]
        )
        self.model: NeuralNetwork = torch._inductor.aoti_load_package(MODEL_PATH)
        end_time = time.time()

        log.info("NN Kernel Prediction Model loaded.")
        log.info("Took: %s seconds", end_time - start_time)

        # Mean values for standardizing input features
        self.mean_for_standardization = torch.tensor(
            [
                2.78275084,
                8.23996746,
                7.27791873,
                7.92035942,
                -2.39558163,
                3.40679233,
                5.80237395,
                3.95781827,
                4.19478321,
                4.19098234,
                0.9045909,
                1.28331208,
            ],
            device="cuda",
        )

        # Standard deviation values for standardizing input features
        self.std_for_standardization = torch.tensor(
            [
                0.08322756,
                2.31893439,
                1.65605574,
                2.15447078,
                2.19682881,
                2.99600806,
                1.24328795,
                0.92352521,
                0.93849802,
                0.93872011,
                0.57455891,
                0.5837217,
            ],
            device="cuda",
        )

    def vec(
        self, m: int, n: int, k: int, dsize: int, config: Any
    ) -> tuple[int, int, int, int, int, int, int, int, int]:
        """
        Convert matrix multiplication parameters and config to a feature vector.

        Args:
            m: First dimension of matrix multiplication
            n: Second dimension of matrix multiplication
            k: Third dimension of matrix multiplication
            dsize: Data size in bits (e.g., 16 for float16, 32 for float32)
            config: Configuration object containing kernel parameters

        Returns:
            Tuple containing the extracted features
        """
        kwargs = config.all_kwargs()

        return (
            int(m),
            int(n),
            int(k),
            int(dsize),
            int(kwargs["BLOCK_M"]),
            int(kwargs["BLOCK_N"]),
            int(kwargs["BLOCK_K"]),
            int(kwargs["num_stages"]),
            int(kwargs["num_warps"]),
        )

    @staticmethod
    def vec_params(
        m: int, n: int, k: int, dsize: int, params: TritonGEMMConfig
    ) -> tuple[int, int, int, int, int, int, int, int, int]:
        """
        Convert matrix multiplication parameters and config to a feature vector.

        Args:
            m: First dimension of matrix multiplication
            n: Second dimension of matrix multiplication
            k: Third dimension of matrix multiplication
            dsize: Data size in bits (e.g., 16 for float16, 32 for float32)
            config: Configuration object containing kernel parameters

        Returns:
            Tuple containing the extracted features
        """

        return (
            int(m),
            int(n),
            int(k),
            int(dsize),
            int(params.block_m),
            int(params.block_n),
            int(params.block_k),
            int(params.num_stages),
            int(params.num_warps),
        )

    def encode(
        self, m: int, n: int, k: int, dtype: torch.dtype, configs: list[Any]
    ) -> torch.Tensor:
        """
        Encode the matrix multiplication parameters and configs as input tensors for the model.

        Args:
            m: First dimension of matrix multiplication
            n: Second dimension of matrix multiplication
            k: Third dimension of matrix multiplication
            dtype: Data type of the matrices
            configs: List of configuration objects

        Returns:
            Tensor containing the encoded inputs ready for the model

        Raises:
            ValueError: If the dtype is not supported
        """
        # Determine data size based on dtype
        if dtype == torch.bfloat16 or dtype == torch.float16:
            dsize = 16
        elif dtype == torch.float32:
            dsize = 32
        else:
            raise ValueError(f"Unsupported dtype: {dtype}. Add support for this dtype.")

        # Create feature dataframe
        df = pd.DataFrame(
            columns=[
                "dim_m",
                "dim_n",
                "dim_k",
                "dtype_size",
                "config_block_m",
                "config_block_n",
                "config_block_k",
                "config_num_stages",
                "config_num_warps",
            ],
            data=[self.vec(m, n, k, dsize, config) for config in configs],
        )

        # Calculate derived features
        df["total_gb"] = get_total_gb_feature(df=df).astype(np.float32)
        df["total_gflop"] = get_total_gflop_feature(df=df).astype(np.float32)
        df["flops_per_byte"] = df["total_gflop"] / df["total_gb"]

        # Reorder columns to match expected model input
        df = df[
            [
                "dtype_size",
                "dim_m",
                "dim_n",
                "dim_k",
                "total_gb",
                "total_gflop",
                "flops_per_byte",
                "config_block_k",
                "config_block_m",
                "config_block_n",
                "config_num_stages",
                "config_num_warps",
            ]
        ]

        # Standardize the input
        inp, _, _ = get_nn_x(
            df=df, mean=self.mean_for_standardization, std=self.std_for_standardization
        )
        inp.to(device="cuda")

        return inp

    def inference(self, inp_tensor: torch.Tensor) -> torch.Tensor:
        """
        Run inference on the model with the given input tensor.

        Args:
            inp_tensor: Input tensor for the model

        Returns:
            Output tensor from the model
        """
        with torch.no_grad():
            return self.model(inp_tensor)

    def decode(self, ret_tensor: torch.Tensor) -> torch.Tensor:
        """
        Decode the model output tensor.

        Args:
            ret_tensor: Output tensor from the model

        Returns:
            Decoded tensor representing runtime predictions
        """
        return ret_tensor


# Create a singleton instance of the model wrapper
import functools


@functools.lru_cache
def get_model() -> ModelWrapper:
    return ModelWrapper()

class Sortable(typing.Protocol):
    """Anything that can be used as a list.sort() key (int/tuple/etc)"""

    def __lt__(self, other: typing.Self) -> bool: ...


class InductorChoices:
    """
    This class contains a collection of default heuristics that effect performance of our generated
    code.  We try to not put correctness requirements in this file.

    You can override the choices made here by doing:

            class MyHeuristics(InductorChoices):
                ...

            torch._inductor.virtualized.V.set_choices_handler(MyHeuristics())
    """

    def get_config_heuristics(
        self, device_type: Optional[str] = "cuda"
    ) -> BaseConfigHeuristic:
        if device_type == "cuda":
            if torch.version.hip is None:
                return CUDAConfigHeuristic()
            else:
                return ROCmConfigHeuristic()
        elif device_type == "xpu":
            return XPUConfigHeuristic()
        elif device_type == "cpu":
            return CPUConfigHeuristic()
        else:
            return BaseConfigHeuristic()

    # GEMM configs
    def get_base_mm_configs(
        self, device_type: Optional[str] = "cuda"
    ) -> partial[Generator[TritonConfig, None, None]]:
        mm_heuristics = self.get_config_heuristics(device_type)
        if config.max_autotune_gemm_search_space != "EXHAUSTIVE":
            return mm_heuristics.get_mm_configs()
        else:
            return mm_heuristics.get_exhaustive_mm_configs()

    def get_mm_configs_search_space(
        self, device_type: Optional[str] = "cuda"
    ) -> partial[Generator[TritonConfig, None, None]]:
        mm_heuristics = self.get_config_heuristics(device_type)
        if config.max_autotune_gemm_search_space != "EXHAUSTIVE":
            return mm_heuristics.get_mm_configs()
        else:
            return mm_heuristics.get_exhaustive_mm_configs()

    def get_exhaustive_mm_configs(
        self, device_type: Optional[str] = "cuda"
    ) -> partial[Generator[TritonConfig, None, None]]:
        mm_heuristics = self.get_config_heuristics(device_type)
        return mm_heuristics.get_exhaustive_mm_configs()

    def get_extra_mm_configs(
        self, device_type: Optional[str] = "cuda"
    ) -> partial[Generator[TritonConfig, None, None]]:
        mm_heuristics = self.get_config_heuristics(device_type)
        return mm_heuristics.get_extra_mm_configs()

    def get_int8_mm_configs(
        self, device_type: Optional[str] = "cuda"
    ) -> partial[Generator[TritonConfig, None, None]]:
        mm_heuristics = self.get_config_heuristics(device_type)
        return mm_heuristics.get_int8_mm_configs()

    def get_mixed_mm_configs(
        self, device_type: Optional[str] = "cuda"
    ) -> partial[Generator[TritonConfig, None, None]]:
        mm_heuristics = self.get_config_heuristics(device_type)
        return mm_heuristics.get_mixed_mm_configs()

    def get_persistent_mm_configs(
        self, device_type: Optional[str] = "cuda"
    ) -> partial[Generator[TritonConfig, None, None]]:
        mm_heuristics = self.get_config_heuristics(device_type)
        return mm_heuristics.get_persistent_mm_configs()

    def get_scaled_mm_configs(
        self, device_type: Optional[str] = "cuda"
    ) -> partial[Generator[TritonConfig, None, None]]:
        mm_heuristics = self.get_config_heuristics(device_type)
        return mm_heuristics.get_scaled_mm_configs()

    def get_scaled_persistent_mm_configs(
        self, device_type: Optional[str] = "cuda"
    ) -> partial[Generator[TritonConfig, None, None]]:
        mm_heuristics = self.get_config_heuristics(device_type)
        return mm_heuristics.get_scaled_persistent_mm_configs()

    def get_mm_plus_mm_configs(
        self, device_type: Optional[str] = "cuda"
    ) -> partial[Generator[TritonConfig, None, None]]:
        mm_heuristics = self.get_config_heuristics(device_type)
        return mm_heuristics.get_mm_plus_mm_configs()

    def filter_triton_mm_choices(
        self,
        m: int,
        n: int,
        k: int,
        mat1: MutableBox,
        mat2: MutableBox,
        layout: Layout,
        choices: list[TritonConfig],
        default_topk: int,
    ) -> list[TritonConfig]:
        if len(layout.size) == 2:
            out_size = (1, layout.size[0], layout.size[1])
            out_stride = (1, layout.stride[0], layout.stride[1])
        else:
            out_size = (layout.size[0], layout.size[1], layout.size[2])
            out_stride = (layout.stride[0], layout.stride[1], layout.stride[2])
        problem = MMProblem(
            B=1,
            M=m,
            N=n,
            K=k,
            M_dtype=mat1.dtype,
            K_dtype=mat2.dtype,
            out_dtype=layout.dtype,
            out_size=out_size,
            out_stride=out_stride,
        )
        # if config.kernel_lut_path is not None:
        #     lut = get_table(config.kernel_lut_path)
        #     if lut is None:
        #         log.warning("Failed to load kernel LUT from %s", config.kernel_lut_path)
        #     else:
        #         newconf = convert_triton_configs_to_gemm_configs(choices)
        #         hardware_name = torch.cuda.get_device_name()

        #         new_choices = lut.filter(hardware_name, "mm", problem, newconf)
        #         if new_choices is not None and len(new_choices) > 0:
        #             choices = new_choices

        benchmarking_space: Union[int, Literal["SAME", "DEFAULT"]] = (
            config.matmul_gemm_autotune_benchmark_space
        )
        if benchmarking_space == "DEFAULT":
            # set benchmarking_space to match the number of configs in the default space
            benchmarking_space = default_topk

        if benchmarking_space != "SAME" and len(choices) > benchmarking_space:
            # filter configs
            model = get_model()
            encoded = model.encode(m, n, k, mat1.dtype, choices)
            inference = model.inference(encoded)
            res = torch.exp(inference)
            timings = list(zip(res.flatten().tolist(), choices))
            timings.sort(key=lambda x: x[0])
            def log_timing(timings):
                if config.fast_autotune_feedback_path is not None:
                    with open(f"{config.fast_autotune_feedback_path}_{benchmarking_space}_{mat1.dtype}", "a") as f:
                        for timing, cfg in timings:
                            f.write(
                                f"{m},{k},{n},{cfg.kwargs['BLOCK_M']},{cfg.kwargs['BLOCK_K']},{cfg.kwargs['BLOCK_N']},{cfg.num_stages},{cfg.num_warps},{cfg},{timing}\n"
                            )               
            log_timing(timings)

            top_configs = timings[:benchmarking_space]
            msg = f"Top X predicted configs on M:{m} K:{k} N:{n}: "
            log.info(msg)
            for timing, cfg in top_configs:
                kw = cfg.kwargs
                msg = f"{timing}, Config(M: {kw['BLOCK_M']}, K: {kw['BLOCK_K']}, K: {kw['BLOCK_N']}, \
num_stages: {cfg.num_stages}, num_warps: {cfg.num_warps})"
                log.info(msg)
            choices = [cfg for _, cfg in top_configs]
        return choices

    # Conv configs
    def get_conv_configs(
        self, device_type: Optional[str] = "cuda"
    ) -> partial[Generator[TritonConfig, None, None]]:
        conv_heuristics = self.get_config_heuristics(device_type)
        return conv_heuristics.get_conv_configs()

    # Flex attention configs
    def get_flex_attention_fwd_configs(
        self, head_dim: int, dtype: torch.dtype, device_type: Optional[str] = "cuda"
    ) -> list[Any]:
        flex_heuristics = self.get_config_heuristics(device_type)
        return flex_heuristics.get_flex_attn_fwd_configs(head_dim, dtype)

    def get_flex_attention_bwd_configs(
        self, head_dim: int, dtype: torch.dtype, device_type: Optional[str] = "cuda"
    ) -> list[Any]:
        flex_heuristics = self.get_config_heuristics(device_type)
        return flex_heuristics.get_flex_attn_bwd_configs(head_dim, dtype)

    def get_flex_decode_configs(
        self, head_dim: int, dtype: torch.dtype, device_type: Optional[str] = "cuda"
    ) -> list[Any]:
        flex_heuristics = self.get_config_heuristics(device_type)
        return flex_heuristics.get_flex_decode_configs(head_dim, dtype)

    def triton_kernel_kwargs(
        self,
        kernel_cls: type[TritonKernel],
        features: SIMDKernelFeatures,
        groups: list[sympy.Expr],
        kernel_kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        """Hook to change the kwargs passed to TritonKernel, used to apply fixed configurations"""
        return kernel_kwargs

    @staticmethod
    def should_use_cooperative_reduction(features: SIMDKernelFeatures) -> bool:
        """Heuristic to decide if a cooperative reduction should be used."""
        if config.triton.force_cooperative_reductions:
            return True
        if (
            not config.triton.cooperative_reductions
            or V.graph.get_current_device_or_throw().type == "cpu"
        ):
            return False

        xhint = V.graph.sizevars.size_hint(features.numel, fallback=2)
        if xhint <= 8:
            threshold = 32768 * xhint
        elif xhint <= 16:
            threshold = 2097152
        else:
            return False
        # TODO(jansel): should this default on for dynamic shapes?
        return V.graph.sizevars.statically_known_geq(
            features.reduction_numel, threshold
        )

    @staticmethod
    def should_use_persistent_reduction(
        features: SIMDKernelFeatures, cooperative_reduction: bool
    ) -> bool:
        """
        Heuristic to decide if a persistent reduction should be used.
        """
        if not config.triton.persistent_reductions:
            return False
        threshold = {
            ReductionHint.INNER: 1024,
        }.get(features.get_reduction_hint(), 64)

        if cooperative_reduction:
            # The RSPLIT of cooperative reductions means each thread block is operating on fewer elements
            try:
                threshold *= 32 // min(V.graph.sizevars.size_hint(features.numel), 32)
            except ValueError:
                pass  # unbacked symint

        # If multi_kernel is enabled, we do more aggressive persistent reduction.
        # This may result in some persistent reductions slower than the
        # corresponding non-persistent reductions. MultiKernel will do benchmarking
        # to pick the faster one.
        if config.triton.multi_kernel:
            threshold *= 16
        return V.graph.sizevars.statically_known_leq(
            features.reduction_numel, threshold
        )  # type: ignore[arg-types]

    @staticmethod
    def want_no_x_dim(features: SIMDKernelFeatures) -> bool:
        """
        Heuristic to decide if we should drop the X dimension from a persistent reduction kernel.
        So the [XBLOCK, RBLOCK] block becomes a [RBLOCK] block and XBLOCK is forced to be always 1.
        Strangely this is faster than a [1, RBLOCK] block in some cases.
        """
        return (
            features.get_reduction_hint() == ReductionHint.INNER
            and V.graph.sizevars.statically_known_geq(features.reduction_numel, 256)
        )

    @staticmethod
    def reduction_split_factor(
        device: torch.device,
        reduction_numel_hint: int,
        numel_hint: int,
        inner_reduction: bool,
    ) -> int:
        """Heuristic to decide the RSPLIT used for split reductions.
        When a reduction has a small number of outputs there is not enough parallelism,
        so we will do the reduction in two phases."""
        props = DeviceProperties.create(device)
        num_sm = props.multi_processor_count
        min_elements_per_thread = 32
        max_elements_per_thread = 512
        threads_per_sm = 2048
        min_elements_per_device = min_elements_per_thread * num_sm * threads_per_sm
        max_elements_per_device = max_elements_per_thread * num_sm * threads_per_sm
        num_warps = 8
        num_threads = 32 * num_warps

        if inner_reduction:
            # do heuristics that's close to eager mode for split inner reduction
            # we leak reduction autotune configs here, and will need to refactor to avoid this later
            if numel_hint >= 2 * num_sm:  # don't split if there are enough outputs
                return 1
            if reduction_numel_hint <= 8192:
                return 1
            if reduction_numel_hint * numel_hint <= min_elements_per_device:
                split_size = min_elements_per_thread
            elif reduction_numel_hint * numel_hint < max_elements_per_device:
                target_blocks = num_sm * threads_per_sm // (2 * num_threads)
                blocks_per_output = (target_blocks + numel_hint - 1) // numel_hint
                tmp_split_size = (
                    reduction_numel_hint + num_threads * blocks_per_output - 1
                ) // (num_threads * blocks_per_output)
                divisors = sympy.divisors(reduction_numel_hint)
                closest = min(divisors, key=lambda x: abs(x - tmp_split_size))
                if abs(closest - tmp_split_size) < 30:
                    # prefer even splits, but never smalle than min_elements_per_thread
                    split_size = max(closest, min_elements_per_thread)
                else:
                    split_size = tmp_split_size
            else:
                divisors = sympy.divisors(reduction_numel_hint)
                closest = min(divisors, key=lambda x: abs(x - max_elements_per_thread))
                if abs(closest - max_elements_per_thread) < 50:
                    # prefer even splits
                    split_size = closest
                else:
                    split_size = max_elements_per_thread
            return (reduction_numel_hint + split_size * num_threads - 1) // (
                split_size * num_threads
            )
        else:
            # TODO the best heuristic currently has XBLOCK (corresponding to numel_hint) 128
            # extend to even smaller number of outputs
            rvals_per_thread = 4  # comes from heuristics, refactor to not leak here
            xvals_per_block = 128
            xblocks = (numel_hint + xvals_per_block - 1) // xvals_per_block
            if reduction_numel_hint * numel_hint < min_elements_per_device:
                split_size = min_elements_per_thread
            elif reduction_numel_hint * numel_hint < max_elements_per_device:
                target_blocks = num_sm * threads_per_sm // (num_threads)
                target_blocks = (target_blocks + xblocks - 1) // xblocks
                tmp_split_size = (
                    reduction_numel_hint + rvals_per_thread * target_blocks - 1
                ) // (rvals_per_thread * target_blocks)
                divisors = sympy.divisors(reduction_numel_hint)
                closest = min(divisors, key=lambda x: abs(x - tmp_split_size))
                if abs(tmp_split_size - closest) < 20:
                    split_size = max(closest, min_elements_per_thread)
                else:
                    split_size = tmp_split_size
            else:
                divisors = sympy.divisors(reduction_numel_hint)
                closest = min(divisors, key=lambda x: abs(x - max_elements_per_thread))
                if abs(closest - max_elements_per_thread) < 50:
                    # prefer even splits
                    split_size = closest
                else:
                    split_size = max_elements_per_thread

            return (reduction_numel_hint + rvals_per_thread * split_size - 1) // (
                rvals_per_thread * split_size
            )

    @staticmethod
    def can_fuse(
        scheduler: Scheduler,
        node1: BaseSchedulerNode,
        node2: BaseSchedulerNode,
        shared_data_score: int,
    ) -> bool:
        """
        Heuristics to prevent fusion applied to both horizontal and vertical fusions.  Heuristics here should not
        be needed for correctness and tweaking them may yield additional performance.

        See also some related heuristics that can be changed via config:
            - config.triton.tiling_prevents_pointwise_fusion
            - config.triton.tiling_prevents_reduction_fusion
            - config.aggressive_fusion (will cause this function to be called more times)
        """
        if shared_data_score == 0 and (
            not config.aggressive_fusion or node1.is_reduction() or node2.is_reduction()
        ):
            if is_metric_table_enabled("fusion_failure_due_to_indexing_mismatch"):
                common_buf_names: OrderedSet[str] = (
                    node1.read_writes.buffer_names() & node2.read_writes.buffer_names()
                )
                if len(common_buf_names) > 0:
                    get_metric_table("fusion_failure_due_to_indexing_mismatch").add_row(
                        lambda: {
                            "pre_grad_graph_id": V.graph.graph_id,
                            "post_grad_graph_id": V.graph.post_grad_graph_id,
                            "node1_name": node1.get_name(),
                            "node2_name": node2.get_name(),
                            "node1_debug_str": write_text(node1.debug_str()),
                            "node2_debug_str": write_text(node2.debug_str()),
                            "common_buffer_names": list(common_buf_names),  # type: ignore[dict-item]
                            "failure_reason": scheduler.decide_fusion_fail_reason(
                                node1, node2, common_buf_names
                            ),
                        }
                    )

                    WhyNoFuse(node1, node2)("no shared data due to indexing mismatch")
                    return False
            WhyNoFuse(node1, node2)("no shared data")
            return False  # heuristic not needed for correctness

        if (
            not node1.is_foreach()
            and not node2.is_foreach()
            and len(node1.get_nodes()) + len(node2.get_nodes()) > config.max_fusion_size
        ):
            WhyNoFuse(node1, node2)("exceeds max fusion")
            return False  # heuristic not needed for correctness

        if scheduler.can_fusion_increase_peak_memory(node1, node2):
            WhyNoFuse(node1, node2)("Fusion will increase peak memory")
            return False

        return True

    @staticmethod
    def can_fuse_vertical(
        scheduler: Scheduler,
        node1: BaseSchedulerNode,
        node2: BaseSchedulerNode,
        shared_data_score: int,
    ) -> bool:
        """Hook for heuristics to prevent vertical (producer/consumer) fusions"""
        return True

    @staticmethod
    def can_fuse_horizontal(
        scheduler: Scheduler,
        node1: BaseSchedulerNode,
        node2: BaseSchedulerNode,
        shared_data_score: int,
    ) -> bool:
        """Hook for heuristics to prevent horizontal (consumer/consumer) fusions"""
        if shared_data_score < config.score_fusion_memory_threshold:
            WhyNoFuse(node1, node2)("score_fusion_memory_threshold")
            return False
        if scheduler.are_long_distant_nodes(node1, node2):
            WhyNoFuse(node1, node2)(
                "Nodes are too far away. Fusing them may increase peak memory."
            )
            return False
        return True

    @staticmethod
    def score_fusion(
        scheduler: Scheduler,
        node1: BaseSchedulerNode,
        node2: BaseSchedulerNode,
    ) -> Sortable:
        """
        Assign a score (higher comes first) to the fusion of node1 and node2.
        When different fusions conflict with each other, this is the way we
        decide what order to run them in.

        Our current score is based on:
        - The type of fusion (template/reduction/etc)
        - Estimate of the saved memory operations
        - Fusions closer together in original graph order
        """
        memory_score = scheduler.score_fusion_memory(node1, node2)
        proximity_score = -max(
            abs(node1.min_order - node2.max_order),
            abs(node2.min_order - node1.max_order),
        )

        # prologue fusion always last
        if node2.is_template():
            template_score = 0
        else:
            template_score = 1 + (
                (node1.is_template() == config.epilogue_fusion_first)
                and memory_score > 0
            )

        return (
            template_score,
            node1.is_reduction() == node2.is_reduction() and memory_score > 0,
            memory_score,
            proximity_score,
        )
