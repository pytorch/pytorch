"""
Gaussian Mixture Model testing suite.
"""
# Owner(s): ["module: distributions"]

import time
from dataclasses import dataclass
from typing import Any, Callable

import pytest

import torch
from torch.distributions import (
    Categorical,
    MixtureSameFamily,
    MultivariateNormal,
    multivariate_normal,
)
from torch.testing._internal.common_utils import run_tests


def compute_gmm_log_probability_manual(
    test_points: torch.Tensor,
    gmm_model: MixtureSameFamily,
) -> torch.Tensor:
    """
    Compute GMM log probability with custom _batch_mahalanobis function.

    This function temporarily replaces the standard _batch_mahalanobis with our
    custom implementation that includes debugging output and verification checks.

    Args:
        test_points: Input tensor for probability computation
        gmm_model: Gaussian Mixture Model instance

    Returns:
        torch.Tensor: Log probabilities computed using the custom function
    """
    # Store original function for safe restoration
    original_func = multivariate_normal._batch_mahalanobis

    # Apply monkey patch with our custom function
    multivariate_normal._batch_mahalanobis = _batch_mahalanobis_chunked

    try:
        # Compute log probabilities with patched function
        pytorch_log_probs = gmm_model.log_prob(test_points)

    finally:
        # Always restore original function (even if error occurs)
        multivariate_normal._batch_mahalanobis = original_func

    return pytorch_log_probs


def _batch_mahalanobis_chunked(
    bL: torch.Tensor,
    bx: torch.Tensor,
    batch_chunk_size: int = 1000,
    c_chunk_size: int = 100,
) -> torch.Tensor:
    r"""
    Computes the squared Mahalanobis distance :math:`\mathbf{x}^\top\mathbf{M}^{-1}\mathbf{x}`
    for a factored :math:`\mathbf{M} = \mathbf{L}\mathbf{L}^\top`.

    Accepts batches for both bL and bx. They are not necessarily assumed to have the same batch
    shape, but `bL` one should be able to broadcasted to `bx` one.
    """
    n = bx.size(-1)
    bx_batch_shape = bx.shape[:-1]

    # Assume that bL.shape = (i, 1, n, n), bx.shape = (..., i, j, n),
    # we are going to make bx have shape (..., 1, j,  i, 1, n) to apply batched tri.solve
    bx_batch_dims = len(bx_batch_shape)
    bL_batch_dims = bL.dim() - 2
    outer_batch_dims = bx_batch_dims - bL_batch_dims
    old_batch_dims = outer_batch_dims + bL_batch_dims
    new_batch_dims = outer_batch_dims + 2 * bL_batch_dims

    # Reshape bx with the shape (..., 1, i, j, 1, n)
    bx_new_shape = bx.shape[:outer_batch_dims]
    for sL, sx in zip(bL.shape[:-2], bx.shape[outer_batch_dims:-1]):
        bx_new_shape += (sx // sL, sL)
    bx_new_shape += (n,)
    bx = bx.reshape(bx_new_shape)

    # Permute bx to make it have shape (..., 1, j, i, 1, n)
    permute_dims = (
        list(range(outer_batch_dims))
        + list(range(outer_batch_dims, new_batch_dims, 2))
        + list(range(outer_batch_dims + 1, new_batch_dims, 2))
        + [new_batch_dims]
    )
    bx = bx.permute(permute_dims)

    # Core computation: flatten tensors for triangular solve
    flat_L = bL.reshape(-1, n, n)  # shape = b x n x n
    flat_x = bx.reshape(-1, flat_L.size(0), n)  # shape = c x b x n
    flat_x_swap = flat_x.permute(1, 2, 0)  # shape = b x n x c

    # ═══════════════════════════════════════════════════════════════════════════════
    # DEBUGGING & VERIFICATION SECTION
    # ═══════════════════════════════════════════════════════════════════════════════

    print("Chunked Tensor Shapes:")
    print(f"   • flat_x_swap shape: {flat_x_swap.shape}")
    print(f"   • flat_L shape:      {flat_L.shape}")

    # Current approach: Chunked triangular solve for memory efficiency
    batch_size, n, c = flat_x_swap.shape
    batch_chunk_size = min(
        batch_chunk_size, batch_size
    )  # Don't exceed actual batch size
    c_chunk_size = min(c_chunk_size, c)  # Use actual c size, with reasonable max chunk

    M_swap = solve_triangular_synchronized(
        flat_L, flat_x_swap, batch_chunk_size, c_chunk_size
    )

    # Verify the triangular solve by multiplying back
    reconstructed = flat_L @ M_swap

    # Compute reconstruction error
    difference = reconstructed - flat_x_swap

    # Calculate error metrics
    max_abs_error = torch.max(torch.abs(difference)).item()
    relative_error = (torch.norm(difference) / torch.norm(flat_x_swap)).item()

    print("Chunked Reconstruction Verification:")
    print(f"   • Max absolute error: {max_abs_error:.2e}")
    print(f"   • Relative error:     {relative_error:.2e}")

    # ═══════════════════════════════════════════════════════════════════════════════

    M_swap = M_swap.pow(2).sum(-2)
    M = M_swap.t()  # shape = c x b

    # Now we revert the above reshape and permute operators.
    permuted_M = M.reshape(bx.shape[:-1])  # shape = (..., 1, j, i, 1)
    permute_inv_dims = list(range(outer_batch_dims))
    for i in range(bL_batch_dims):
        permute_inv_dims += [outer_batch_dims + i, old_batch_dims + i]
    reshaped_M = permuted_M.permute(permute_inv_dims)  # shape = (..., 1, i, j, 1)

    return reshaped_M.reshape(bx_batch_shape)


def solve_triangular_synchronized(
    L: torch.Tensor, x: torch.Tensor, batch_chunk_size: int, c_chunk_size: int
) -> torch.Tensor:
    """
    Solve triangular system with synchronized batch chunking for memory efficiency.

    Always chunks L and x together by batch dimension to maintain correspondence
    between the triangular matrices and right-hand sides.

    Args:
        L: Lower triangular matrices [batch, n, n]
        x: Right-hand side vectors [batch, n, c]
        batch_chunk_size: Maximum batch size per chunk
        c_chunk_size: Maximum c dimension size per chunk

    Returns:
        torch.Tensor: Solution to Lx = b for each batch [batch, n, c]
    """
    batch_size, n, c = x.shape
    device = L.device
    dtype = L.dtype

    # Pre-allocate result tensor
    result = torch.zeros(batch_size, n, c, device=device, dtype=dtype)

    # Process in batch chunks to manage memory usage
    for b_start in range(0, batch_size, batch_chunk_size):
        b_end = min(b_start + batch_chunk_size, batch_size)

        # Extract synchronized batch chunks for both L and x
        L_batch = L[b_start:b_end]  # [batch_chunk, n, n]
        x_batch = x[b_start:b_end]  # [batch_chunk, n, c]

        # Process c dimension in chunks for current batch
        for c_start in range(0, c, c_chunk_size):
            c_end = min(c_start + c_chunk_size, c)
            x_chunk = x_batch[:, :, c_start:c_end]  # [batch_chunk, n, c_chunk]

            # Solve triangular system: L @ result = x_chunk
            solved = torch.linalg.solve_triangular(L_batch, x_chunk, upper=False)

            # Store solution in result tensor
            result[b_start:b_end, :, c_start:c_end] = solved

    return result


def _batch_mahalanobis(bL: torch.Tensor, bx: torch.Tensor) -> torch.Tensor:
    r"""
    Computes the squared Mahalanobis distance :math:`\mathbf{x}^\top\mathbf{M}^{-1}\mathbf{x}`
    for a factored :math:`\mathbf{M} = \mathbf{L}\mathbf{L}^\top`.

    Accepts batches for both bL and bx. They are not necessarily assumed to have the same batch
    shape, but `bL` one should be able to broadcasted to `bx` one.
    """
    n = bx.size(-1)
    bx_batch_shape = bx.shape[:-1]

    # Assume that bL.shape = (i, 1, n, n), bx.shape = (..., i, j, n),
    # we are going to make bx have shape (..., 1, j,  i, 1, n) to apply batched tri.solve
    bx_batch_dims = len(bx_batch_shape)
    bL_batch_dims = bL.dim() - 2
    outer_batch_dims = bx_batch_dims - bL_batch_dims
    old_batch_dims = outer_batch_dims + bL_batch_dims
    new_batch_dims = outer_batch_dims + 2 * bL_batch_dims

    # Reshape bx with the shape (..., 1, i, j, 1, n)
    bx_new_shape = bx.shape[:outer_batch_dims]
    for sL, sx in zip(bL.shape[:-2], bx.shape[outer_batch_dims:-1]):
        bx_new_shape += (sx // sL, sL)
    bx_new_shape += (n,)
    bx = bx.reshape(bx_new_shape)

    # Permute bx to make it have shape (..., 1, j, i, 1, n)
    permute_dims = (
        list(range(outer_batch_dims))
        + list(range(outer_batch_dims, new_batch_dims, 2))
        + list(range(outer_batch_dims + 1, new_batch_dims, 2))
        + [new_batch_dims]
    )
    bx = bx.permute(permute_dims)

    # Core computation: flatten tensors for triangular solve
    flat_L = bL.reshape(-1, n, n)  # shape = b x n x n
    flat_x = bx.reshape(-1, flat_L.size(0), n)  # shape = c x b x n
    flat_x_swap = flat_x.permute(1, 2, 0)  # shape = b x n x c
    M_swap = torch.linalg.solve_triangular(
        flat_L, flat_x_swap, upper=False
    )  # shape = b x c

    # ═══════════════════════════════════════════════════════════════════════════════
    # DEBUGGING & VERIFICATION SECTION
    # ═══════════════════════════════════════════════════════════════════════════════

    print("Original Tensor Shapes:")
    print(f"   • flat_x_swap shape: {flat_x_swap.shape}")
    print(f"   • flat_L shape:      {flat_L.shape}")

    # Verify the triangular solve by multiplying back
    reconstructed = flat_L @ M_swap

    # Compute reconstruction error
    difference = reconstructed - flat_x_swap

    # Calculate error metrics
    max_abs_error = torch.max(torch.abs(difference)).item()
    relative_error = (torch.norm(difference) / torch.norm(flat_x_swap)).item()

    print("Original Reconstruction Verification:")
    print(f"   • Max absolute error: {max_abs_error:.2e}")
    print(f"   • Relative error:     {relative_error:.2e}")

    # ═══════════════════════════════════════════════════════════════════════════════

    M_swap = M_swap.pow(2).sum(-2)  # shape = b x c
    M = M_swap.t()  # shape = c x b

    # Now we revert the above reshape and permute operators.
    permuted_M = M.reshape(bx.shape[:-1])  # shape = (..., 1, j, i, 1)
    permute_inv_dims = list(range(outer_batch_dims))
    for i in range(bL_batch_dims):
        permute_inv_dims += [outer_batch_dims + i, old_batch_dims + i]
    reshaped_M = permuted_M.permute(permute_inv_dims)  # shape = (..., 1, i, j, 1)

    return reshaped_M.reshape(bx_batch_shape)


def compute_gmm_log_probability_original(
    test_points: torch.Tensor,
    gmm_model: MixtureSameFamily,
) -> torch.Tensor:
    """
    Compute GMM log probability with custom _batch_mahalanobis function.

    This function temporarily replaces the standard _batch_mahalanobis with our
    custom implementation that includes debugging output and verification checks.

    Args:
        test_points: Input tensor for probability computation
        gmm_model: Gaussian Mixture Model instance

    Returns:
        torch.Tensor: Log probabilities computed using the custom function
    """
    # Store original function for safe restoration
    original_func = multivariate_normal._batch_mahalanobis

    # Apply monkey patch with our custom function
    multivariate_normal._batch_mahalanobis = _batch_mahalanobis

    try:
        # Compute log probabilities with patched function
        pytorch_log_probs = gmm_model.log_prob(test_points)

    finally:
        # Always restore original function (even if error occurs)
        multivariate_normal._batch_mahalanobis = original_func

    return pytorch_log_probs


@dataclass(frozen=True)
class GMMTestConfiguration:
    """
    Configuration for randomized GMM‑input generation.

    Encapsulates all parameters controlling the shape of the randomly
    generated batches, numbers of components and dimensions, and
    carries a human‑readable identifier for logging or pytest ids.
    """

    min_batch_dims: int  # minimum number of batch dimensions (≥ 0)
    max_batch_dims: int  # maximum number of batch dimensions (≥ 0)
    min_batch_size: int  # minimum size of each batch dimension (≥ 1)
    max_batch_size: int  # maximum size of each batch dimension (> 0)
    min_batch_outer_dims: int  # minimum number of outer batch dimensions (≥ 0)
    max_batch_outer_dims: int  # maximum number of outer batch dimensions (≥ 0)
    min_batch_outer_size: int  # minimum size of each outer batch dimension (≥ 1)
    max_batch_outer_size: int  # maximum size of each outer batch dimension (> 0)
    min_components: int  # minimum number of mixture components (≥ 1)
    max_components: int  # maximum number of mixture components (> 0)
    min_dimensions: int  # minimum vector dimensionality (≥ 1)
    max_dimensions: int  # maximum vector dimensionality (> 0)
    test_identifier: str  # arbitrary label for this test configuration

    def __post_init__(self) -> None:
        """Validate all configuration parameters."""
        # Validate individual field constraints
        self._validate_batch_dims()
        self._validate_batch_size()
        self._validate_batch_outer_dims()
        self._validate_batch_outer_size()
        self._validate_components()
        self._validate_dimensions()
        self._validate_test_identifier()

        # Validate min <= max relationships
        self._validate_min_max_relationships()

    def _validate_batch_dims(self) -> None:
        """Validate batch dimension constraints."""
        if self.min_batch_dims < 0:
            raise ValueError(
                f"min_batch_dims must be non‑negative. Got {self.min_batch_dims}."
            )
        if self.max_batch_dims < 0:
            raise ValueError(
                f"max_batch_dims must be non‑negative. Got {self.max_batch_dims}."
            )

    def _validate_batch_size(self) -> None:
        """Validate batch size constraints."""
        if self.min_batch_size < 1:
            raise ValueError(
                f"min_batch_size must be positive (≥ 1). Got {self.min_batch_size}."
            )
        if self.max_batch_size <= 0:
            raise ValueError(
                f"max_batch_size must be positive (> 0). Got {self.max_batch_size}."
            )

    def _validate_batch_outer_dims(self) -> None:
        """Validate outer batch dimension constraints."""
        if self.min_batch_outer_dims < 0:
            raise ValueError(
                f"min_batch_outer_dims must be non‑negative. Got {self.min_batch_outer_dims}."
            )
        if self.max_batch_outer_dims < 0:
            raise ValueError(
                f"max_batch_outer_dims must be non‑negative. Got {self.max_batch_outer_dims}."
            )

    def _validate_batch_outer_size(self) -> None:
        """Validate outer batch size constraints."""
        if self.min_batch_outer_size < 1:
            raise ValueError(
                f"min_batch_outer_size must be positive (≥ 1). Got {self.min_batch_outer_size}."
            )
        if self.max_batch_outer_size <= 0:
            raise ValueError(
                f"max_batch_outer_size must be positive (> 0). Got {self.max_batch_outer_size}."
            )

    def _validate_components(self) -> None:
        """Validate mixture component constraints."""
        if self.min_components < 1:
            raise ValueError(
                f"min_components must be positive (≥ 1). Got {self.min_components}."
            )
        if self.max_components <= 0:
            raise ValueError(
                f"max_components must be positive (> 0). Got {self.max_components}."
            )

    def _validate_dimensions(self) -> None:
        """Validate dimensionality constraints."""
        if self.min_dimensions < 1:
            raise ValueError(
                f"min_dimensions must be positive (≥ 1). Got {self.min_dimensions}."
            )
        if self.max_dimensions <= 0:
            raise ValueError(
                f"max_dimensions must be positive (> 0). Got {self.max_dimensions}."
            )

    def _validate_test_identifier(self) -> None:
        """Validate test identifier constraints."""
        if not isinstance(self.test_identifier, str):
            raise TypeError(
                f"test_identifier must be a string. Got {type(self.test_identifier).__name__}."
            )
        if not self.test_identifier.strip():
            raise ValueError("test_identifier cannot be empty or whitespace-only.")

    def _validate_min_max_relationships(self) -> None:
        """Validate that all min values are <= corresponding max values."""
        relationships = [
            ("batch_dims", self.min_batch_dims, self.max_batch_dims),
            ("batch_size", self.min_batch_size, self.max_batch_size),
            ("batch_outer_dims", self.min_batch_outer_dims, self.max_batch_outer_dims),
            ("batch_outer_size", self.min_batch_outer_size, self.max_batch_outer_size),
            ("components", self.min_components, self.max_components),
            ("dimensions", self.min_dimensions, self.max_dimensions),
        ]

        for field_name, min_val, max_val in relationships:
            if min_val > max_val:
                raise ValueError(
                    f"min_{field_name} ({min_val}) must be <= max_{field_name} ({max_val})."
                )

    def __str__(self) -> str:
        """Human-readable representation of the configuration."""
        return (
            f"GMMConfig({self.test_identifier}: "
            f"batch_dims[{self.min_batch_dims}‑{self.max_batch_dims}], "
            f"batch_size[{self.min_batch_size}‑{self.max_batch_size}], "
            f"outer_dims[{self.min_batch_outer_dims}‑{self.max_batch_outer_dims}], "
            f"outer_size[{self.min_batch_outer_size}‑{self.max_batch_outer_size}], "
            f"components[{self.min_components}‑{self.max_components}], "
            f"dimensions[{self.min_dimensions}‑{self.max_dimensions}])"
        )

    def is_valid_for_testing(self) -> bool:
        """
        Check if configuration defines meaningful ranges for testing.

        Returns:
            bool: True if all ranges allow for meaningful variation in testing
        """
        # Check if ranges allow for meaningful variation
        meaningful_ranges = [
            self.max_batch_dims >= self.min_batch_dims,
            self.max_batch_size >= self.min_batch_size,
            self.max_batch_outer_dims >= self.min_batch_outer_dims,
            self.max_batch_outer_size >= self.min_batch_outer_size,
            self.max_components >= self.min_components,
            self.max_dimensions >= self.min_dimensions,
        ]

        return all(meaningful_ranges)


def random_gmm_inputs(
    config: GMMTestConfiguration,
    device: torch.device,
    dtype: torch.dtype,
    random_seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate random GMM inputs for testing based on configuration parameters.

    Creates randomized Gaussian Mixture Model components and test points with
    batch dimensions, component counts, and feature dimensions sampled from
    the ranges specified in the configuration.

    Args:
        config: GMMTestConfiguration specifying parameter ranges
        device: PyTorch device for tensor allocation
        dtype: PyTorch data type for tensor creation
        random_seed: Optional seed for reproducible random generation

    Returns:
        Tuple containing:
            mixture_weights:    Tensor of shape [*B, K] - component weights
            component_means:    Tensor of shape [*B, K, D] - component centers
            precision_matrices: Tensor of shape [*B, K, D, D] - inverse covariances
            test_points:        Tensor of shape [*N, *B, D] - evaluation points

        Where:
            *B: Randomly chosen inner batch shape of up to max_batch_dims dimensions
            *N: Randomly chosen outer batch shape of up to max_batch_outer_dims dimensions
            K:  Number of mixture components [min_components, max_components]
            D:  Feature dimensionality [min_dimensions, max_dimensions]
    """
    # ═══════════════════════════════════════════════════════════════════════════════
    # INITIALIZATION AND VALIDATION
    # ═══════════════════════════════════════════════════════════════════════════════

    if random_seed is not None:
        torch.manual_seed(random_seed)

    # Validate configuration
    if not config.is_valid_for_testing():
        raise ValueError(f"Invalid configuration for testing: {config}")

    # ═══════════════════════════════════════════════════════════════════════════════
    # DIMENSION SAMPLING
    # ═══════════════════════════════════════════════════════════════════════════════

    # Sample inner batch dimensions: B = [b1, b2, ..., bM]
    M = int(torch.randint(config.min_batch_dims, config.max_batch_dims + 1, ()).item())

    B = [
        int(torch.randint(config.min_batch_size, config.max_batch_size + 1, ()).item())
        for _ in range(M)
    ]

    # Sample outer batch dimensions: N = [n1, n2, ..., nL]
    N_outer = int(
        torch.randint(
            config.min_batch_outer_dims, config.max_batch_outer_dims + 1, ()
        ).item()
    )

    N = [
        int(
            torch.randint(
                config.min_batch_outer_size, config.max_batch_outer_size + 1, ()
            ).item()
        )
        for _ in range(N_outer)
    ]

    # Sample mixture components and feature dimensions
    K = int(torch.randint(config.min_components, config.max_components + 1, ()).item())

    D = int(torch.randint(config.min_dimensions, config.max_dimensions + 1, ()).item())

    # ═══════════════════════════════════════════════════════════════════════════════
    # MIXTURE WEIGHTS GENERATION
    # ═══════════════════════════════════════════════════════════════════════════════

    # Generate normalized mixture weights: shape [*B, K]
    raw_weights = torch.rand(*B, K, dtype=dtype, device=device)
    mixture_weights = raw_weights / raw_weights.sum(dim=-1, keepdim=True)

    # ═══════════════════════════════════════════════════════════════════════════════
    # COMPONENT MEANS GENERATION
    # ═══════════════════════════════════════════════════════════════════════════════

    # Generate component means: shape [*B, K, D]
    component_means = torch.randn(*B, K, D, dtype=dtype, device=device)

    # ═══════════════════════════════════════════════════════════════════════════════
    # PRECISION MATRICES GENERATION
    # ═══════════════════════════════════════════════════════════════════════════════

    # Initialize precision matrices: shape [*B, K, D, D]
    precision_matrices = torch.zeros(*B, K, D, D, dtype=dtype, device=device)
    flat_prec = precision_matrices.view(-1, K, D, D)

    # Adaptive regularization based on dtype and dimensionality
    base_regularization = 0.1 if dtype == torch.float32 else 1e-3
    dimension_regularization = D * 1e-3
    regularization = max(base_regularization, dimension_regularization)

    # Scale random matrix generation for better numerical conditioning
    scale_factor = 1.0 / (D**0.5)

    # Generate positive definite precision matrices
    batch_flat_size = flat_prec.shape[0]
    for batch_idx in range(batch_flat_size):
        for component_idx in range(K):
            # Create random matrix and form Gram matrix for positive definiteness
            random_matrix = torch.randn(D, D, dtype=dtype, device=device) * scale_factor
            gram_matrix = random_matrix @ random_matrix.mT

            # Add regularization to ensure numerical stability
            regularized_matrix = (
                gram_matrix + torch.eye(D, dtype=dtype, device=device) * regularization
            )

            # Assign to precision matrices
            flat_prec[batch_idx, component_idx].copy_(regularized_matrix)

    # ═══════════════════════════════════════════════════════════════════════════════
    # TEST POINTS GENERATION
    # ═══════════════════════════════════════════════════════════════════════════════

    # Generate test points: shape [*N, *B, D]
    test_points = torch.randn(*N, *B, D, dtype=dtype, device=device)

    # ═══════════════════════════════════════════════════════════════════════════════
    # FINAL VALIDATION AND RETURN
    # ═══════════════════════════════════════════════════════════════════════════════

    # Validate tensor shapes before returning
    expected_shapes = {
        "mixture_weights": (*B, K),
        "component_means": (*B, K, D),
        "precision_matrices": (*B, K, D, D),
        "test_points": (*N, *B, D),
    }

    actual_shapes = {
        "mixture_weights": tuple(mixture_weights.shape),
        "component_means": tuple(component_means.shape),
        "precision_matrices": tuple(precision_matrices.shape),
        "test_points": tuple(test_points.shape),
    }

    for tensor_name, expected_shape in expected_shapes.items():
        actual_shape = actual_shapes[tensor_name]
        if actual_shape != expected_shape:
            raise RuntimeError(
                f"Shape mismatch for {tensor_name}: "
                f"expected {expected_shape}, got {actual_shape}"
            )

    return mixture_weights, component_means, precision_matrices, test_points


def create_gmm_distributions(
    mixture_weights: torch.Tensor,
    component_means: torch.Tensor,
    precision_matrices: torch.Tensor,
    target_device: torch.device,
) -> MixtureSameFamily:
    """
    Create GMM distribution from parameters on specified device.

    Args:
        mixture_weights: Component weights tensor
        component_means: Component mean vectors
        precision_matrices: Component precision matrices
        target_device: Device for distribution computation

    Returns:
        Complete GMM distribution ready for evaluation

    Raises:
        RuntimeError: If distribution creation fails
    """
    try:
        # Transfer parameters to target device
        weights_device = mixture_weights.to(target_device)
        means_device = component_means.to(target_device)
        precisions_device = precision_matrices.to(target_device)

        # Create mixture and component distributions
        mixture_categorical = Categorical(weights_device)
        multivariate_components = MultivariateNormal(
            means_device, precision_matrix=precisions_device
        )

        # Combine into complete GMM
        gmm_distribution = MixtureSameFamily(
            mixture_categorical, multivariate_components
        )

        return gmm_distribution

    except Exception as e:
        raise RuntimeError(f"GMM distribution creation failed: {str(e)}") from e


class PerformanceTimer:
    """
    Utility class for accurate performance timing across different devices.
    """

    @staticmethod
    def time_gpu_operation(operation: Callable[[], Any]) -> tuple[Any, float]:
        """Time GPU operation with proper synchronization."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = time.perf_counter()
        result = operation()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_time = time.perf_counter()
        return result, end_time - start_time

    @staticmethod
    def time_cpu_operation(operation: Callable[[], Any]) -> tuple[Any, float]:
        """Time CPU operation with standard timing."""
        start_time = time.perf_counter()
        result = operation()
        end_time = time.perf_counter()
        return result, end_time - start_time


def analyze_performance_results(
    pytorch_gpu_time: float,
    pytorch_cpu_time: float,
    manual_gpu_time: float,
    manual_cpu_time: float,
    config: GMMTestConfiguration,
    dtype_description: str,
) -> None:
    """
    Analyze and display performance characteristics.

    Args:
        pytorch_gpu_time: PyTorch GPU execution time
        pytorch_cpu_time: PyTorch CPU execution time
        manual_gpu_time: Manual GPU implementation time
        manual_cpu_time: Manual CPU implementation time
        total_points: Number of evaluation points processed
        config: Test configuration for context
        dtype_description: Data type description for clarity
    """
    print(f"\nPerformance Analysis: {config.test_identifier} ({dtype_description})")

    print("\nExecution Times:")
    print(f"   PyTorch GPU:      {pytorch_gpu_time * 1000:8.2f} ms")
    print(f"   PyTorch CPU:      {pytorch_cpu_time * 1000:8.2f} ms")
    print(f"   Manual GPU:       {manual_gpu_time * 1000:8.2f} ms")
    print(f"   Manual CPU:       {manual_cpu_time * 1000:8.2f} ms")


# Test configuration definitions for comprehensive coverage

TEST_CONFIGURATIONS = [
    # ═══════════════════════════════════════════════════════════════════════════════
    # MINIMAL CONFIGURATIONS
    # ═══════════════════════════════════════════════════════════════════════════════
    GMMTestConfiguration(
        min_batch_dims=0,
        max_batch_dims=0,
        min_batch_size=1,
        max_batch_size=1,
        min_batch_outer_dims=0,
        max_batch_outer_dims=0,
        min_batch_outer_size=1,
        max_batch_outer_size=1,
        min_components=1,
        max_components=1,
        min_dimensions=1,
        max_dimensions=1,
        test_identifier="minimal_1comp_1dim",
    ),
    # ═══════════════════════════════════════════════════════════════════════════════
    # SMALL SCALE CONFIGURATIONS
    # ═══════════════════════════════════════════════════════════════════════════════
    GMMTestConfiguration(
        min_batch_dims=0,
        max_batch_dims=0,
        min_batch_size=1,
        max_batch_size=1,
        min_batch_outer_dims=1,
        max_batch_outer_dims=2,
        min_batch_outer_size=10,
        max_batch_outer_size=50,
        min_components=2,
        max_components=5,
        min_dimensions=2,
        max_dimensions=2,
        test_identifier="small_no_batch_5comp_2dim",
    ),
    GMMTestConfiguration(
        min_batch_dims=1,
        max_batch_dims=2,
        min_batch_size=2,
        max_batch_size=5,
        min_batch_outer_dims=1,
        max_batch_outer_dims=2,
        min_batch_outer_size=10,
        max_batch_outer_size=50,
        min_components=3,
        max_components=10,
        min_dimensions=2,
        max_dimensions=3,
        test_identifier="medium_batch_10comp_3dim",
    ),
    # ═══════════════════════════════════════════════════════════════════════════════
    # LARGE SCALE CONFIGURATIONS
    # ═══════════════════════════════════════════════════════════════════════════════
    GMMTestConfiguration(
        min_batch_dims=2,
        max_batch_dims=3,
        min_batch_size=3,
        max_batch_size=5,
        min_batch_outer_dims=2,
        max_batch_outer_dims=3,
        min_batch_outer_size=10,
        max_batch_outer_size=20,
        min_components=10,
        max_components=50,
        min_dimensions=3,
        max_dimensions=5,
        test_identifier="large_batch_50comp_5dim",
    ),
    # ═══════════════════════════════════════════════════════════════════════════════
    # HIGH DIMENSIONAL CONFIGURATIONS
    # ═══════════════════════════════════════════════════════════════════════════════
    GMMTestConfiguration(
        min_batch_dims=1,
        max_batch_dims=1,
        min_batch_size=2,
        max_batch_size=2,
        min_batch_outer_dims=1,
        max_batch_outer_dims=1,
        min_batch_outer_size=100,
        max_batch_outer_size=1000,
        min_components=3,
        max_components=7,
        min_dimensions=50,
        max_dimensions=100,
        test_identifier="high_dim_7comp_100dim",
    ),
    GMMTestConfiguration(
        min_batch_dims=1,
        max_batch_dims=1,
        min_batch_size=10,
        max_batch_size=20,
        min_batch_outer_dims=1,
        max_batch_outer_dims=1,
        min_batch_outer_size=10,
        max_batch_outer_size=20,
        min_components=2,
        max_components=5,
        min_dimensions=300,
        max_dimensions=400,
        test_identifier="ultra_high_dim_7comp_400dim",
    ),
    # ═══════════════════════════════════════════════════════════════════════════════
    # MANY COMPONENT CONFIGURATIONS
    # ═══════════════════════════════════════════════════════════════════════════════
    GMMTestConfiguration(
        min_batch_dims=1,
        max_batch_dims=1,
        min_batch_size=2,
        max_batch_size=2,
        min_batch_outer_dims=1,
        max_batch_outer_dims=1,
        min_batch_outer_size=5,
        max_batch_outer_size=10,
        min_components=50,
        max_components=100,
        min_dimensions=2,
        max_dimensions=2,
        test_identifier="many_comp_100comp_2dim",
    ),
    # ═══════════════════════════════════════════════════════════════════════════════
    # COMPLEX BATCH STRUCTURE CONFIGURATIONS
    # ═══════════════════════════════════════════════════════════════════════════════
    GMMTestConfiguration(
        min_batch_dims=3,
        max_batch_dims=4,
        min_batch_size=2,
        max_batch_size=3,
        min_batch_outer_dims=2,
        max_batch_outer_dims=3,
        min_batch_outer_size=3,
        max_batch_outer_size=5,
        min_components=5,
        max_components=10,
        min_dimensions=3,
        max_dimensions=3,
        test_identifier="multi_batch_10comp_3dim",
    ),
    # ═══════════════════════════════════════════════════════════════════════════════
    # EDGE CASE CONFIGURATIONS
    # ═══════════════════════════════════════════════════════════════════════════════
    GMMTestConfiguration(
        min_batch_dims=0,
        max_batch_dims=1,
        min_batch_size=1,
        max_batch_size=3,
        min_batch_outer_dims=0,
        max_batch_outer_dims=1,
        min_batch_outer_size=1,
        max_batch_outer_size=5,
        min_components=1,
        max_components=2,
        min_dimensions=1,
        max_dimensions=10,
        test_identifier="variable_dims_edge_case",
    ),
    GMMTestConfiguration(
        min_batch_dims=0,
        max_batch_dims=0,
        min_batch_size=1,
        max_batch_size=1,
        min_batch_outer_dims=0,
        max_batch_outer_dims=0,
        min_batch_outer_size=1,
        max_batch_outer_size=1,
        min_components=1,
        max_components=20,
        min_dimensions=1,
        max_dimensions=20,
        test_identifier="scalar_with_variation",
    ),
    # ═══════════════════════════════════════════════════════════════════════════════
    # CPU/CUDA MISMATCH CONFIGURATIONS
    # ═══════════════════════════════════════════════════════════════════════════════
    GMMTestConfiguration(
        min_batch_dims=0,
        max_batch_dims=0,
        min_batch_size=1,
        max_batch_size=1,
        min_batch_outer_dims=1,
        max_batch_outer_dims=2,
        min_batch_outer_size=10,
        max_batch_outer_size=1200,
        min_components=2,
        max_components=5,
        min_dimensions=2,
        max_dimensions=2,
        test_identifier="test_cpu_cuda_mismatch_1",
    ),
    GMMTestConfiguration(
        min_batch_dims=1,
        max_batch_dims=3,
        min_batch_size=1,
        max_batch_size=1,
        min_batch_outer_dims=2,
        max_batch_outer_dims=3,
        min_batch_outer_size=1200,
        max_batch_outer_size=1300,
        min_components=2,
        max_components=5,
        min_dimensions=2,
        max_dimensions=2,
        test_identifier="test_cpu_cuda_mismatch_2",
    ),
    GMMTestConfiguration(
        min_batch_dims=2,
        max_batch_dims=3,
        min_batch_size=2,
        max_batch_size=3,
        min_batch_outer_dims=2,
        max_batch_outer_dims=3,
        min_batch_outer_size=1100,
        max_batch_outer_size=1200,
        min_components=1,
        max_components=4,
        min_dimensions=4,
        max_dimensions=7,
        test_identifier="test_cpu_cuda_mismatch_3",
    ),
    GMMTestConfiguration(
        min_batch_dims=1,
        max_batch_dims=2,
        min_batch_size=5,
        max_batch_size=6,
        min_batch_outer_dims=2,
        max_batch_outer_dims=3,
        min_batch_outer_size=900,
        max_batch_outer_size=1000,
        min_components=1,
        max_components=2,
        min_dimensions=4,
        max_dimensions=7,
        test_identifier="test_cpu_cuda_mismatch_4",
    ),
]


PRECISION_CONFIGURATIONS = [
    (torch.float32, "float32_standard"),
    (torch.float64, "float64_double"),
]


def format_test_description(config: GMMTestConfiguration) -> str:
    """Format test configuration for readable pytest output."""
    return config.test_identifier


def format_precision_description(precision_info: tuple[torch.dtype, str]) -> str:
    """Format precision configuration for readable pytest output."""
    return precision_info[1]


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for GPU testing"
)
class TestGMMNumericalConsistency:
    """Test suite for GMM numerical consistency and performance validation."""

    @pytest.mark.parametrize("config", TEST_CONFIGURATIONS, ids=format_test_description)
    def test_parameter_generation_validity(self, config: GMMTestConfiguration) -> None:
        """Test that random_gmm_inputs produces valid shapes and values."""
        # Generate randomized inputs
        mixture_weights, component_means, precision_matrices, test_points = (
            random_gmm_inputs(
                config=config,
                dtype=torch.float64,
                device=torch.device("cpu"),
                random_seed=42,
            )
        )

        # 1) Validate mixture_weights: shape [*B, K]
        assert mixture_weights.ndim >= 1
        *batch_shape, K = mixture_weights.shape
        # K must be positive and ≤ max_components
        assert 1 <= K <= config.max_components
        # each batch dimension must be ≤ max_batch_size
        for b in batch_shape:
            assert 1 <= b <= config.max_batch_size

        # mixture weights must sum to 1 over last axis
        sums = mixture_weights.sum(dim=-1)
        assert torch.allclose(
            sums,
            torch.ones_like(
                sums, dtype=mixture_weights.dtype, device=mixture_weights.device
            ),
        )

        # all weights strictly positive
        assert torch.all(mixture_weights > 0)

        # 2) Validate component_means: shape [*B, K, D]
        assert component_means.shape[: len(batch_shape)] == tuple(batch_shape)
        assert component_means.shape[-2] == K
        D = component_means.shape[-1]
        # D must be positive and ≤ max_dimensions
        assert 1 <= D <= config.max_dimensions
        # finite entries
        assert torch.all(torch.isfinite(component_means))

        # 3) Validate precision_matrices: shape [*B, K, D, D]
        assert precision_matrices.shape[: len(batch_shape)] == tuple(batch_shape)
        assert precision_matrices.shape[-3] == K
        assert precision_matrices.shape[-2:] == (D, D)
        # positive-definiteness check
        flat_prec = precision_matrices.reshape(-1, D, D)
        for P in flat_prec:
            eigvals = torch.linalg.eigvals(P)
            assert torch.all(eigvals.real > 0), "Found non-PD precision matrix"

        # 4) Validate test_points: shape [*N, *B, D]
        pts_shape = test_points.shape
        assert pts_shape[-1] == D, f"Expected last dim D={D}, got {pts_shape[-1]}"
        dims = list(pts_shape[:-1])
        # separate outer dims from batch dims
        outer_count = len(dims) - len(batch_shape)
        assert (
            0 <= outer_count <= config.max_batch_outer_dims
        ), f"Number of outer dims {outer_count} exceeds max {config.max_batch_outer_dims}"
        outer_dims = dims[:outer_count]

        assert test_points.ndim == len(outer_dims) + len(batch_shape) + 1
        assert 0 <= len(outer_dims) <= config.max_batch_outer_dims
        for b in outer_dims:
            assert 1 <= b <= config.max_batch_outer_size

        # finite entries
        assert torch.all(torch.isfinite(test_points))

    @pytest.mark.parametrize("config", TEST_CONFIGURATIONS, ids=format_test_description)
    @pytest.mark.parametrize(
        "precision_info", PRECISION_CONFIGURATIONS, ids=format_precision_description
    )
    def test_cpu_default_cpu_manual_consistency(
        self, config: GMMTestConfiguration, precision_info: tuple[torch.dtype, str]
    ) -> None:
        """Test that manual implementation matches PyTorch implementation mathematically."""
        dtype, _ = precision_info

        # Generate test scenario
        mixture_weights, component_means, precision_matrices, test_points = (
            random_gmm_inputs(
                config=config,
                dtype=dtype,
                device=torch.device("cpu"),
                random_seed=42,
            )
        )

        # Create PyTorch distribution
        gmm = create_gmm_distributions(
            mixture_weights, component_means, precision_matrices, torch.device("cpu")
        )

        # Compute log probabilities using both methods
        pytorch_log_probs = compute_gmm_log_probability_original(test_points, gmm)
        manual_log_probs = compute_gmm_log_probability_manual(test_points, gmm)

        # Verify mathematical consistency
        max_absolute_error = (pytorch_log_probs - manual_log_probs).abs().max()

        assert torch.allclose(
            pytorch_log_probs, manual_log_probs
        ), f"Manual implementation mismatch: max error = {max_absolute_error.item():.2e}"

    @pytest.mark.parametrize("config", TEST_CONFIGURATIONS, ids=format_test_description)
    @pytest.mark.parametrize(
        "precision_info", PRECISION_CONFIGURATIONS, ids=format_precision_description
    )
    def test_cpu_default_gpu_manual_consistency(
        self, config: GMMTestConfiguration, precision_info: tuple[torch.dtype, str]
    ) -> None:
        """Test that manual implementation matches PyTorch implementation mathematically."""
        dtype, _ = precision_info

        # Generate test scenario
        mixture_weights, component_means, precision_matrices, test_points = (
            random_gmm_inputs(
                config=config,
                dtype=dtype,
                device=torch.device("cpu"),
                random_seed=42,
            )
        )

        # Create PyTorch distribution
        cpu_gmm = create_gmm_distributions(
            mixture_weights, component_means, precision_matrices, torch.device("cpu")
        )
        cuda_gmm = create_gmm_distributions(
            mixture_weights, component_means, precision_matrices, torch.device("cuda")
        )

        pytorch_log_probs = compute_gmm_log_probability_original(test_points, cpu_gmm)
        manual_log_probs = compute_gmm_log_probability_manual(
            test_points.to("cuda"), cuda_gmm
        ).to("cpu")

        # Verify mathematical consistency
        max_absolute_error = (pytorch_log_probs - manual_log_probs).abs().max()

        assert torch.allclose(
            pytorch_log_probs, manual_log_probs
        ), f"Manual implementation mismatch: max error = {max_absolute_error.item():.2e}"

    @pytest.mark.parametrize("config", TEST_CONFIGURATIONS, ids=format_test_description)
    @pytest.mark.parametrize(
        "precision_info", PRECISION_CONFIGURATIONS, ids=format_precision_description
    )
    def test_cpu_default_gpu_default_consistency(
        self, config: GMMTestConfiguration, precision_info: tuple[torch.dtype, str]
    ) -> None:
        """Test that GPU and CPU implementations produce identical results."""
        dtype, _ = precision_info

        # Generate test scenario
        mixture_weights, component_means, precision_matrices, test_points = (
            random_gmm_inputs(
                config=config,
                dtype=dtype,
                device=torch.device("cpu"),
                random_seed=42,
            )
        )

        # Create distributions on both devices
        cpu_gmm = create_gmm_distributions(
            mixture_weights, component_means, precision_matrices, torch.device("cpu")
        )
        gpu_gmm = create_gmm_distributions(
            mixture_weights, component_means, precision_matrices, torch.device("cuda")
        )

        # Compute log probabilities on both devices
        cpu_log_probs = compute_gmm_log_probability_original(test_points, cpu_gmm)
        gpu_log_probs = compute_gmm_log_probability_original(
            test_points.to("cuda"), gpu_gmm
        ).cpu()

        # Verify device consistency
        max_device_error = (cpu_log_probs - gpu_log_probs).abs().max()

        assert torch.allclose(
            cpu_log_probs, gpu_log_probs
        ), f"GPU/CPU device mismatch: max error = {max_device_error.item():.2e}"

    @pytest.mark.parametrize("config", TEST_CONFIGURATIONS, ids=format_test_description)
    @pytest.mark.parametrize(
        "precision_info", PRECISION_CONFIGURATIONS, ids=format_precision_description
    )
    def test_performance_benchmarking(
        self, config: GMMTestConfiguration, precision_info: tuple[torch.dtype, str]
    ) -> None:
        """
        Benchmark GMM implementations across devices with proper warmup.

        Warmup iterations ensure we measure steady-state performance.
        """
        dtype, dtype_description = precision_info

        # Configuration for warmup iterations
        WARMUP_ITERATIONS = {
            "cpu": 3,  # Fewer needed for CPU cache warming
            "gpu": 5,  # More needed for CUDA kernel compilation
        }

        # Helper functions for cleaner benchmarking code
        def warmup_and_time(
            operation: Callable, device: str, timer_method: Callable
        ) -> float:
            """Perform warmup iterations then time the operation."""
            # Warmup phase
            for _ in range(WARMUP_ITERATIONS[device]):
                operation()
                if device == "gpu" and torch.cuda.is_available():
                    torch.cuda.synchronize()

            # Timing phase
            _, execution_time = timer_method(operation)
            return execution_time

        # Step 1: Generate test scenario
        mixture_weights, component_means, precision_matrices, test_points_cpu = (
            random_gmm_inputs(
                config=config,
                dtype=dtype,
                device=torch.device("cpu"),
                random_seed=42,
            )
        )

        test_points_gpu = test_points_cpu.to("cuda")

        # Step 2: Create distributions on both devices
        distributions = {
            "cpu": create_gmm_distributions(
                mixture_weights,
                component_means,
                precision_matrices,
                torch.device("cpu"),
            ),
            "gpu": create_gmm_distributions(
                mixture_weights,
                component_means,
                precision_matrices,
                torch.device("cuda"),
            ),
        }

        # Step 3: Define benchmark operations
        benchmark_operations = {
            "pytorch_cpu": lambda: compute_gmm_log_probability_original(
                test_points_cpu, distributions["cpu"]
            ),
            "pytorch_gpu": lambda: compute_gmm_log_probability_original(
                test_points_gpu, distributions["gpu"]
            ).cpu(),
            "manual_cpu": lambda: compute_gmm_log_probability_manual(
                test_points_cpu, distributions["cpu"]
            ),
            "manual_gpu": lambda: compute_gmm_log_probability_manual(
                test_points_gpu, distributions["gpu"]
            ).cpu(),
        }

        # Step 4: Initialize timer
        timer = PerformanceTimer()

        # Step 5: Benchmark each implementation with appropriate warmup
        timing_results = {
            "pytorch_cpu": warmup_and_time(
                benchmark_operations["pytorch_cpu"], "cpu", timer.time_cpu_operation
            ),
            "pytorch_gpu": warmup_and_time(
                benchmark_operations["pytorch_gpu"], "gpu", timer.time_gpu_operation
            ),
            "manual_cpu": warmup_and_time(
                benchmark_operations["manual_cpu"], "cpu", timer.time_cpu_operation
            ),
            "manual_gpu": warmup_and_time(
                benchmark_operations["manual_gpu"], "gpu", timer.time_gpu_operation
            ),
        }

        # Step 6: Analyze and display results
        analyze_performance_results(
            pytorch_gpu_time=timing_results["pytorch_gpu"],
            pytorch_cpu_time=timing_results["pytorch_cpu"],
            manual_gpu_time=timing_results["manual_gpu"],
            manual_cpu_time=timing_results["manual_cpu"],
            config=config,
            dtype_description=dtype_description,
        )

        # Step 7: Validate timing results
        for name, time_value in timing_results.items():
            assert time_value > 0, f"Invalid timing for {name}: {time_value}"


if __name__ == "__main__":
    run_tests()
    # pytest.main([__file__, "-vvv", "-s", "--tb=short"])
