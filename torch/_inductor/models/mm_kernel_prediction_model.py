"""
Neural network model for predicting triton kernel performance.

This module provides functionality to load and use a pre-trained neural network
for predicting the performance of triton kernels.
"""

import logging
import os
import re
import time
import math
from collections.abc import Sequence
from typing import Any, Optional

import torch
import torch._inductor.config as config
import torch.nn as nn


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
        self.log_kernel_overhead: float = math.log(kernel_overhead)
        all_layer_widths = list(hidden_layer_widths) + [1]
        all_input_widths = [n_inputs] + list(hidden_layer_widths)
        layers: list[nn.Module] = []
        if len(all_input_widths) != len(all_layer_widths):
            raise ValueError("Input lists must have the same length")
        for n_in, n_out in zip(all_input_widths, all_layer_widths):
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
            input=log_base_pred,
            fill_value=self.log_kernel_overhead,
            device=torch.device("cuda"),
        )
        return torch.logsumexp(
            torch.stack([log_base_pred, log_overhead_tsr], dim=-1), dim=-1
        )


def get_nn_x(
    dict_of_features: dict[str, torch.Tensor],
    mean: Optional[torch.Tensor] = None,
    std: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Take logs, standardize the data, and convert it to a tensor.

    Args:
        dict_of_features: Dict mapping feature names to 1d tensors
        mean: Mean values for standardizing input features. Optional.
        std: Standard deviation values for standardizing input features.
            Optional.
    """
    col_order = [
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
    x_tens = torch.stack(
        [torch.log(dict_of_features[col].to(dtype=torch.float64)) for col in col_order],
        dim=-1,
    ).to(device=torch.device("cuda"))

    if mean is None:
        mean = x_tens.mean(dim=0)
    if std is None:
        std = x_tens.std(dim=0)
    x_tens -= mean
    x_tens /= std
    return x_tens.to(torch.float32), mean, std


def get_total_gb_feature(dict_of_features: dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Calculate the total gigabytes feature.

    Args:
        dict_of_features: Dict mapping feature names to 1d tensors. Must include
            keys "dim_m", "dim_n", "dim_k", and "dtype_size".

    Returns:
        Tensor containing the calculated total gigabytes
    """
    # Calculate memory access in bytes
    m, n, k = (
        dict_of_features["dim_m"],
        dict_of_features["dim_n"],
        dict_of_features["dim_k"],
    )
    # Convert bits to bytes
    dtype_size = dict_of_features["dtype_size"] / 8

    # Convert to GB
    # A: m×k, B: k×n, C: m×n
    return ((m * k + k * n + m * n) * dtype_size) / 1e9


def get_total_gflop_feature(dict_of_features: dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Calculate the total gigaflops feature.

    Args:
        dict_of_features: Dict mapping feature names to 1d tensors. Must include
            keys "dim_m", "dim_n", and "dim_k".

    Returns:
        Tensor containing the calculated total gigaflops
    """
    # For matrix multiplication, flops = 2 * m * n * k
    m, n, k = (
        dict_of_features["dim_m"],
        dict_of_features["dim_n"],
        dict_of_features["dim_k"],
    )
    # Convert to GFLOP
    return (2 * m * n * k) / 1e9


def _sanitize_path(input_string: str) -> str:
    """
    Sanitizes an arbitrary string to create a valid cross-platform file or directory path.
    - Replaces invalid/special characters with underscores.
    - Removes leading/trailing whitespace.
    - Collapses multiple underscores.
    """
    s = input_string.strip().lower()
    s = re.sub(r'[<>:"/\\|?*\t\n\r\f\v]', "_", s)
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"_+", "_", s)
    s = s.strip("_")
    return s


class ModelWrapper:
    """
    Wrapper for the neural network model that handles loading the model, encoding inputs
    and decoding outputs.
    """

    MODEL_BASE_NAME = "triton_mm"

    @classmethod
    def _device_name_projection(cls, device_name: str) -> str:
        """
        Project device name to a common format.

        Args:
            device_name: Device name to project

        Returns:
            Projected device name
        """
        if "H100" in device_name or "h100" in device_name:
            return "NVIDIA H100"
        if "mi300x" in device_name or "MI300X" in device_name:
            return "AMD INSTINCT MI300X"
        return device_name

    @classmethod
    def _get_device_model_path(cls, device_name: str) -> str:
        if config.fast_autotune_model_path is not None:
            return config.fast_autotune_model_path
        else:
            base_directory = os.path.dirname(__file__)
        new_device_name = cls._device_name_projection(device_name)

        file_name = _sanitize_path(new_device_name)
        return os.path.join(
            base_directory, "artifacts", f"{file_name}_{cls.MODEL_BASE_NAME}.pt2"
        )

    def __init__(self, device_name: Optional[str] = None) -> None:
        """Initialize the model wrapper with the pre-trained model and standardization parameters."""
        if not torch.cuda.is_available():
            raise RuntimeError("ModelWrapper created when CUDA is not available.")
        if device_name is None:
            device_name = torch.cuda.get_device_name()
        model_path = self._get_device_model_path(device_name)
        # check to see if model_path exists
        # TODO remove logging
        print("Loading NN Kernel Prediction Model from ", model_path)
        # List files in the model path directory
        model_dir = os.path.dirname(model_path)
        print("Files in model directory:", os.listdir(model_dir))

        # List files in the parent directory of the model path
        parent_dir = os.path.dirname(model_dir)
        print("Files in parent directory:", os.listdir(parent_dir))

        print("Model path exists: ", os.path.exists(model_path))
        if not os.path.exists(model_path):
            print("Model path does not exist: %s", model_path)
            raise RuntimeError(f"Model path not found for device {device_name}. ")

        start_time = time.time()
        self.model = torch._inductor.aoti_load_package(model_path)
        end_time = time.time()

        log.info(
            "NN Kernel Prediction Model loaded in % seconds.", end_time - start_time
        )

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
            device=torch.device("cuda"),
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
            device=torch.device("cuda"),
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
        feature_names = [
            "dim_m",
            "dim_n",
            "dim_k",
            "dtype_size",
            "config_block_m",
            "config_block_n",
            "config_block_k",
            "config_num_stages",
            "config_num_warps",
        ]
        tensor = torch.stack(
            [torch.tensor(self.vec(m, n, k, dsize, config)) for config in configs],
            dim=0,
        )
        dict_of_features = dict(
            zip(feature_names, (tensor[:, i] for i in range(tensor.shape[1])))
        )

        # Calculate derived features
        dict_of_features["total_gb"] = get_total_gb_feature(
            dict_of_features=dict_of_features
        )
        dict_of_features["total_gflop"] = get_total_gflop_feature(
            dict_of_features=dict_of_features
        )
        dict_of_features["flops_per_byte"] = (
            dict_of_features["total_gflop"] / dict_of_features["total_gb"]
        )

        # Standardize the input
        inp, _, _ = get_nn_x(
            dict_of_features=dict_of_features,
            mean=self.mean_for_standardization,
            std=self.std_for_standardization,
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
def get_model(device_name: Optional[str] = None) -> Optional[ModelWrapper]:
    if not torch.cuda.is_available():
        return None
    if device_name is None:
        device_name = torch.cuda.get_device_name()
    model_path = ModelWrapper._get_device_model_path(device_name)
    if model_path is None:
        return None
    return ModelWrapper(device_name)
