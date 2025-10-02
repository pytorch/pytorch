# Owner(s): ["module: distributions"]

import random

import numpy as np
import pytest

import torch
from torch.testing._internal.common_utils import run_tests


class TriangularSystemGenerator:
    """Generator for triangular systems to test numerical stability."""

    def __init__(self, seed: int | None = None):
        self.seed = seed
        self._rng = random.Random(seed)
        self._torch_generators: dict[torch.device, torch.Generator] = {}

    def _get_generator(self, device: torch.device) -> torch.Generator | None:
        """Get or create a torch generator for the specified device."""
        if self.seed is None:
            return None

        device = torch.device(device) if isinstance(device, str) else device
        if device not in self._torch_generators:
            generator = torch.Generator(device)
            generator.manual_seed(self.seed)
            self._torch_generators[device] = generator
        return self._torch_generators[device]

    def generate_triangular_system(
        self,
        batch_shape: tuple[int, ...],
        n: int,
        k: int,
        cond_min: float = 1.0,
        cond_max: float = 1e6,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate a lower triangular system L @ y = x with controlled conditioning.

        Args:
            batch_shape: Shape of batch dimensions
            n: Matrix size
            k: Number of right-hand sides
            cond_min: Minimum condition number
            cond_max: Maximum condition number
            device: Device for tensors
            dtype: Data type for tensors

        Returns:
            L: Lower triangular matrix (*batch_shape, n, n)
            x: Right-hand side (*batch_shape, n, k)
            y_true: True solution (*batch_shape, n, k)
        """
        device = torch.device(device) if isinstance(device, str) else device
        generator = self._get_generator(device)
        eps = torch.finfo(dtype).eps

        # Generate condition number with log-uniform distribution
        log_cond = self._rng.uniform(np.log(cond_min), np.log(cond_max))
        cond = np.exp(log_cond)

        # Generate random orthogonal matrix via QR decomposition
        Q = torch.randn(
            *batch_shape, n, n, device=device, dtype=dtype, generator=generator
        )
        Q, _ = torch.linalg.qr(Q)

        # Create eigenvalues with log spacing
        eigvals = torch.exp(
            torch.linspace(0, np.log(cond), n, device=device, dtype=dtype)
        ).expand(*batch_shape, n)

        # Construct SPD matrix with known condition number
        diag = torch.diag_embed(eigvals)
        A = Q @ diag @ Q.mT
        A = (A + A.mT) / 2.0  # Ensure symmetry

        # Add stabilization proportional to matrix size and machine epsilon
        A = A + n * eps * torch.eye(n, device=device, dtype=dtype).expand_as(A)

        # Compute Cholesky decomposition
        L = torch.linalg.cholesky(A)

        # Generate true solution and compute RHS
        y_true = torch.randn(
            *batch_shape, n, k, device=device, dtype=dtype, generator=generator
        )
        x = L @ y_true

        return L, x, y_true


def verify_solution_correctness(
    L: torch.Tensor,
    x: torch.Tensor,
    y_computed: torch.Tensor,
    y_true: torch.Tensor,
) -> dict[str, float]:
    """
    Verify solution correctness and compute error metrics.

    Args:
        L: Lower triangular matrix [*, n, n]
        x: Right-hand side vectors [*, n, k]
        y_computed: Computed solutions [*, n, k]
        y_true: True solutions [*, n, k] (optional)

    Returns:
        Dictionary containing computed metrics
    """
    eps = torch.finfo(L.dtype).eps

    # Compute residuals and norms
    residual = x - L @ y_computed
    residual_norm = torch.linalg.vector_norm(residual, dim=-2)
    x_norm = torch.linalg.vector_norm(x, dim=-2)
    relative_residual = residual_norm / (x_norm + eps)

    # Compute backward error
    L_norm = torch.linalg.matrix_norm(L, dim=(-2, -1), keepdim=True)
    y_norm = torch.linalg.vector_norm(y_computed, dim=-2)
    backward_error = residual_norm / (L_norm.squeeze(-1) * y_norm + x_norm + eps)

    # Compute forward error if true solution available
    error = y_computed - y_true
    error_norm = torch.linalg.vector_norm(error, dim=-2)
    y_true_norm = torch.linalg.vector_norm(y_true, dim=-2)
    forward_error = error_norm / (y_true_norm + eps)

    # Estimate condition number from diagonal elements
    diag = torch.diagonal(L, dim1=-2, dim2=-1).abs()
    cond_estimate = diag.max(dim=-1).values / (diag.min(dim=-1).values + eps)

    return {
        "max_relative_residual": relative_residual.max().item(),
        "max_backward_error": backward_error.max().item(),
        "max_forward_error": (
            forward_error.max().item()
        ),
        "max_condition": cond_estimate.max().item(),
    }


def compare_cpu_cuda_solutions(
    y_cpu: torch.Tensor,
    y_cuda: torch.Tensor,
    condition_number: float,
    dtype: torch.dtype,
) -> dict[str, float]:
    """
    Compare CPU and CUDA solutions with conditioning awareness.

    Returns:
        max_relative_diff: Maximum norm-wise relative difference
        max_elementwise_diff: Maximum element-wise relative difference
        expected_diff: Expected difference based on condition number
    """
    eps = torch.finfo(dtype).eps
    y_cuda_cpu = y_cuda.cpu()

    # Compute norm-wise differences
    diff_norm = torch.linalg.vector_norm(y_cpu - y_cuda_cpu, dim=-2)
    y_norm = torch.linalg.vector_norm(y_cpu, dim=-2)
    relative_diff = diff_norm / (y_norm + eps)

    # Compute element-wise differences
    y_abs = torch.maximum(y_cpu.abs(), y_cuda_cpu.abs())
    elementwise_diff = (y_cpu - y_cuda_cpu).abs() / (y_abs + eps)

    return {
        "max_relative_diff": relative_diff.max().item(),
        "max_elementwise_diff": elementwise_diff.max().item(),
        "expected_diff": condition_number * eps,
    }


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    "batch_shape",
    [
        (),
        (5,),
        (50,),
        (100,),
        (10, 20),
        (5, 10, 15),
    ],
)
@pytest.mark.parametrize("n", [1, 2, 5])
@pytest.mark.parametrize("k", [1, 2, 10, 50, 100, 5000, 10000])
@pytest.mark.parametrize(
    "cond_range",
    [
        (1.0, 1.0),  # Perfectly conditioned
        (1.0, 1e3),  # Well-conditioned
        (1e3, 1e6),  # Moderately conditioned
    ],
)
def test_solve_triangular_consistency(
    dtype: torch.dtype,
    batch_shape: tuple[int, ...],
    n: int,
    k: int,
    cond_range: tuple[float, float],
) -> None:
    """Test numerical consistency of triangular solve across devices."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    # Generate test system
    generator = TriangularSystemGenerator(seed=42)
    L_cpu, x_cpu, y_true_cpu = generator.generate_triangular_system(
        batch_shape, n, k, *cond_range, device="cpu", dtype=dtype
    )

    # Move to CUDA and solve
    y_cpu = torch.linalg.solve_triangular(L_cpu, x_cpu, upper=False)

    L_cuda, x_cuda = L_cpu.cuda(), x_cpu.cuda()
    y_cuda = torch.linalg.solve_triangular(L_cuda, x_cuda, upper=False)

    # Verify solutions
    metrics_cpu = verify_solution_correctness(L_cpu, x_cpu, y_cpu, y_true_cpu)
    metrics_cuda = verify_solution_correctness(
        L_cuda, x_cuda, y_cuda, y_true_cpu.cuda()
    )
    comparison = compare_cpu_cuda_solutions(
        y_cpu, y_cuda, metrics_cpu["max_condition"], dtype
    )

    # Set tolerances
    eps = torch.finfo(dtype).eps
    cond = metrics_cpu["max_condition"]
    backward_tol = 100 * eps
    comparison_tol = max(100 * eps, 10 * cond * eps)

    # Check backward stability
    assert (
        metrics_cpu["max_backward_error"] < backward_tol
    ), f"CPU backward error too large: {metrics_cpu['max_backward_error']:.2e} > {backward_tol:.2e}"
    assert (
        metrics_cuda["max_backward_error"] < backward_tol
    ), f"CUDA backward error too large: {metrics_cuda['max_backward_error']:.2e} > {backward_tol:.2e}"

    # # Check solution correctness
    assert (
        metrics_cpu["max_relative_residual"] < backward_tol
    ), f"CPU residual too large: {metrics_cpu['max_relative_residual']:.2e} > {backward_tol:.2e}"
    assert (
        metrics_cuda["max_relative_residual"] < backward_tol
    ), f"CUDA residual too large: {metrics_cuda['max_relative_residual']:.2e} > {backward_tol:.2e}"

    # Check device consistency
    assert comparison["max_relative_diff"] < comparison_tol, (
        f"Device mismatch: {comparison['max_relative_diff']:.2e} > {comparison_tol:.2e} "
        f"(cond={cond:.1e}, expected {comparison['expected_diff']:.1e})"
    )


if __name__ == "__main__":
    run_tests()
    # pytest.main([__file__, "-vvv", "-s"])
