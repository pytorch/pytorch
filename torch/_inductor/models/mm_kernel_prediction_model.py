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
DEFAULT_MODEL_PATH = "./triton_h100_from_arm_108.pkl"
MODEL_PATH = os.environ.get("TRITON_KERNEL_SELECTION_MODEL_PATH", DEFAULT_MODEL_PATH)
import logging


log = logging.getLogger(__name__)


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
        # TODO: Make this a `Parameter` and fit it.
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
            torch.tensor(kernel_overhead)
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
        # TODO: test this
        log_base_pred = self.linear_relu_stack(x)
        log_overhead_tsr = torch.full_like(
            input=log_base_pred, fill_value=self.log_kernel_overhead
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

    x_tens = torch.from_numpy(x_df.astype(float).to_numpy())
    if mean is None:
        mean = torch.from_numpy(assert_is_instance(x_df.mean(), pd.Series).to_numpy())
    if std is None:
        std = torch.from_numpy(assert_is_instance(x_df.std(), pd.Series).to_numpy())
    x_tens -= mean
    x_tens /= std
    return x_tens.to(torch.float32), mean, std


def prepare_training_batches(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    total_gflop_col_idx: int,
    ignore_threshold: float,
    batch_size: int,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """
    Prepares training data by filtering to those that do not have small values
    in column `total_gflop_col_idx`, and batching. We ignore small values
    because we are modeling log(runtime) but care about runtime, and tiny
    operations can have a dispoportionate influence on the data. Batches will be
    moved to the GPU one at a time for training in `train_nn`.

    Args:
        train_x: Train features tensor
        train_y: Target values tensor
        total_gflop_col_idx: Index of the "total_gflop" column, which is used to
            filter
        ignore_threshold: Threshold of "total_gflop" below which data is dropped
            for training
        batch_size: Maximum size of each batch

    Returns:
        Tuple of (train_x_by_batch, train_y_by_batch)
    """
    # Drop where x has nulls
    keeps = (~torch.isnan(train_x).any(dim=1)) & (
        train_x[:, total_gflop_col_idx] > ignore_threshold
    )

    train_x = train_x[keeps, :]
    train_y = train_y[keeps, :]

    n_samples = train_x.shape[0]
    indices = torch.randperm(n_samples)

    split_indices = torch.split(indices, batch_size)
    train_x_by_batch = [train_x[batch_idx, :] for batch_idx in split_indices]
    train_y_by_batch = [train_y[batch_idx, :] for batch_idx in split_indices]

    return train_x_by_batch, train_y_by_batch


def _get_loss(
    criterion: nn.MSELoss,
    objective_type: str,
    outputs: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    """
    If objective_type is "log", returns MSE for `outputs` and `y`. In the
    context of this module, these are on a log scale, so this is MSE for
    predicting log(y).

    If objective_type is "both added", returns the sum of MSE for `y` and
    `outputs` and for `exp(y)` and `exp(outputs)`, weighted by the inverse
    variance of each of `y` and `exp(y)`. If `y` is on a log scale, this is a
    weighted sum of MSEs for `log(y)` and for `y`; the variance weights make
    minimizing this loss equivalent to maximizing the sum of R-squareds for
    `log(y)` and `y`.
    """
    if objective_type == "both added":
        exp_y = torch.exp(y)
        return (
            criterion(torch.exp(outputs), exp_y) / exp_y.var()
            + criterion(outputs, y) / y.var()
        ) / 2
    elif objective_type == "log":
        return criterion(outputs, y)
    else:
        raise ValueError(f"Unknown objective type: {objective_type}")


def validate(
    model: NeuralNetwork,
    criterion: nn.MSELoss,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    best_mse_seen: float,
    best_state_dict: dict[str, torch.Tensor],
    epoch: int,
    verbose: bool = False,
) -> tuple[float, float, dict[str, torch.Tensor]]:
    """
    Validates the model on test data and updates best model metrics.

    Returns:
        Tuple of (test_mse, test_log_mse, dict[str, torch.Tensor])
    """
    model.eval()
    with torch.no_grad():
        test_outputs = model(test_x)
        test_log_mse = criterion(test_outputs, test_y).item()
        test_mse = criterion(torch.exp(test_outputs), torch.exp(test_y)).item()

    test_log_r_squared = 1 - test_log_mse / torch.var(test_y).item()
    test_r_squared = 1 - test_mse / torch.var(torch.exp(test_y)).item()

    if verbose:
        print(f"Epoch {epoch}")
        print(f"Test log MSE: {test_log_mse}")
        print(f"Test MSE: {test_mse}")
        print(f"Test log R-squared: {test_log_r_squared}")
        print(f"Test R-squared: {test_r_squared}")

    # Update best model if needed -- this could be done on either test_mse or
    # test_log_mse; it hasn't been studied
    if test_mse < best_mse_seen:
        best_state_dict = copy.copy(model.state_dict())

    return test_mse, test_log_mse, best_state_dict


def train_one_batch(
    model: NeuralNetwork,
    optimizer: torch.optim.Optimizer,
    criterion: nn.MSELoss,
    batch_x: torch.Tensor,
    batch_y: torch.Tensor,
    device: torch.device,
    objective_type: str,
) -> float:
    """
    Trains the model on a single batch.

    Returns:
        The loss value for this batch
    """
    batch_x = batch_x.to(device=device, non_blocking=True)
    batch_y = batch_y.to(device=device, non_blocking=True)

    optimizer.zero_grad()
    outputs = model(batch_x)
    loss = _get_loss(
        criterion=criterion, objective_type=objective_type, outputs=outputs, y=batch_y
    )
    loss.backward()
    optimizer.step()
    return loss.item()


def train_nn(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    total_gflop_col_idx: int,
    # hparams
    ignore_threshold: float,
    n_epochs: int,
    hidden_layer_widths: Sequence[int],
    lr: float = 1e-3,
    lr_schedule_step_size: int = 100,
    adam_weight_decay: float = 1.0,
    steplr_gamma: float = 0.5,
    validation_frequency: int = 100,
    objective_type: str = "log",
    verbose: bool = False,
    # TODO: figure out how large this can be
    batch_size: int = 1048576,
) -> tuple[NeuralNetwork, list[int], dict[str, list[float]]]:
    """
    Train a `NeuralNetwork` on the given data.

    x and y should be in "log" form, and an overhead of 0.00541 is assumed in
    the model; this makes y closer to being a linear function of x, which is
    easier for the model to fit. Observations with small "total_gflop" values
    are ignored in training so that tiny operations that contribute little to
    runtime do not have excessive influence, but they are still used for
    validation.

    Test performance is measured every `print_frequency` epochs, using MSE
    for logged and non-logged test exectime. If MSE for exectime increases twice
    in a row, training terminates.

    The best model is saved and returned.

    Args:
        train_x: Train features tensor, assumed to be on CPU. For fitting MM
            data, this should ideally be in "log" form (e.g. one column should
            be log(dim_m)).
        train_y: Target values tensor, assumed to be on CPU. For fitting MM,
            this should ideally be in "log" form (e.g. one column should be
            `log(exectime)`).
        test_x: Test features tensor, in the same format as `train_x`.
        test_y: Test target values tensor, in the same format as `train_y`.
        total_gflop_col_idx: Index of the (log) "total_gflop" column, which is
            used to identify small operations to ignore in training.
        ignore_threshold: Threshold of (log) "total_gflop" below which data is
            dropped.
        n_epochs: Maximum number of epochs to train for.
        hidden_layer_widths: Hidden layer widths for the `NeuralNetwork`.
        lr: Learning rate for the AdamW optimizer.
        lr_schedule_step_size: Step size for `StepLR` learning rate scheduler.
        adam_weight_decay: Weight decay for the AdamW optimizer.
        steplr_gamma: ``gamma`` for ``StepLR`` learning rate scheduler.
        validation_frequency: How often to evaluate test losses, in epochs.
        verbose: If True, print losses every ``validation_frequency`` epochs.
        batch_size: Maximum size of each batch.

    Returns:
        Tuple of (best_model, epochs, dict of test_mses, test_log_mses, train_mses).
    """
    # Note: The model might train better if data is shuffled at the beginning of each epoch,
    # but that can be a substantial fraction of runtime with a small model, so
    # we shuffle only once.
    train_x_by_batch, train_y_by_batch = prepare_training_batches(
        train_x=train_x,
        train_y=train_y,
        total_gflop_col_idx=total_gflop_col_idx,
        ignore_threshold=ignore_threshold,
        batch_size=batch_size,
    )

    device = torch.device("cuda")
    test_x = test_x.to(device=device)
    test_y = test_y.to(device=device)

    model = NeuralNetwork(
        n_inputs=train_x.shape[1], hidden_layer_widths=hidden_layer_widths
    ).to(device=device, dtype=train_x.dtype)

    # TODO: try model.compile()
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=adam_weight_decay
    )
    scheduler = StepLR(optimizer, step_size=lr_schedule_step_size, gamma=steplr_gamma)

    # TODO: try evaluating loss with `get_test_loss`
    epochs = []
    test_mses: list[float] = []
    test_log_mses = []
    train_mses = []
    best_state_dict = copy.copy(model.state_dict())

    try:
        for epoch in range(1, n_epochs + 1):
            model.train()
            epoch_loss = 0.0

            for i in range(len(train_x_by_batch)):
                batch_loss = train_one_batch(
                    model=model,
                    optimizer=optimizer,
                    criterion=criterion,
                    batch_x=train_x_by_batch[i],
                    batch_y=train_y_by_batch[i],
                    device=device,
                    objective_type=objective_type,
                )

                epoch_loss += batch_loss

            scheduler.step()

            if epoch % validation_frequency == 1:
                train_mse = epoch_loss / len(train_x_by_batch)
                if verbose:
                    print(f"Average epoch loss: {train_mse}")

                # Validate model and update metrics
                test_mse, test_log_mse, best_state_dict = validate(
                    model=model,
                    criterion=criterion,
                    test_x=test_x,
                    test_y=test_y,
                    best_mse_seen=min(test_mses) if len(test_mses) > 0 else np.inf,
                    best_state_dict=best_state_dict,
                    epoch=epoch,
                    verbose=verbose,
                )
                epochs.append(epoch)
                test_mses.append(test_mse)
                test_log_mses.append(test_log_mse)
                train_mses.append(train_mse)
                should_stop = test_mses[-1] > 3 * min(test_mses)

                if should_stop:
                    print(f"Test MSE of {test_mse} > 3 * {min(test_mses)}")
                    break

    except KeyboardInterrupt:
        max_epoch = 0 if len(epochs) == 0 else epochs[-1]
        print(f"KeyboardInterrupt after {max_epoch} epochs")
    mse_traces = {
        "test_mses": test_mses,
        "test_log_mses": test_log_mses,
        "train_mses": train_mses,
    }
    model.load_state_dict(state_dict=best_state_dict)
    return model.to(device="cpu").eval(), epochs, mse_traces


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
        self.model.load_state_dict(torch.load(MODEL_PATH))
        self.model.eval()
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
            ]
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
            ]
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
