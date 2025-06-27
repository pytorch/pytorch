"""
Neural network model for predicting triton kernel performance.

This module provides functionality to load and use a pre-trained neural network
for predicting the performance of triton kernels.
"""

import copy
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
from torch.optim.lr_scheduler import StepLR


# Default model path - can be overridden by environment variable
import os
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
        mean = torch.from_numpy(assert_is_instance(x_df.mean(), pd.Series).to_numpy()).to(device="cuda")
    if std is None:
        std = torch.from_numpy(assert_is_instance(x_df.std(), pd.Series).to_numpy()).to(device="cuda")
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
            ], device="cuda"
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
            ], device="cuda"
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
