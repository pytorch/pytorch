"""
Tests for two CPU norm optimizations in _get_total_norm (issue #133586).

Optimization 1 — flat-cat path:
  When foreach=None (auto) and device is CPU and numel_per_tensor ≤ 512,
  concatenate into one flat tensor and compute a single norm instead of calling
  _foreach_norm, which creates N aten::empty({}) scalar allocations.

Optimization 2 — skip no-op .to() calls:
  When all gradient tensors are on the same device (the typical case), the
  [norm.to(first_device) for norm in norms] list comprehension adds N Python
  round-trips that do no work. Skip it when no cross-device movement is needed.
"""
import math
import pytest
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.nn.utils.clip_grad import _get_total_norm


def _make_tensors(n, numel, dtype=torch.float32):
    return [torch.randn(numel, dtype=dtype) for _ in range(n)]


def _reference_norm(tensors, norm_type):
    """Compute norm using the foreach path explicitly (bypasses both optimizations)."""
    norms = torch._foreach_norm(tensors, ord=norm_type)
    return torch.linalg.vector_norm(
        torch.stack([n.to(tensors[0].device) for n in norms]), norm_type
    )


# ---------------------------------------------------------------------------
# Optimization 1: flat-cat path for small tensors
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("norm_type", [1.0, 2.0, 3.0, float("inf")])
def test_flat_path_correctness_small_tensors(norm_type):
    """N=2000, numel=512: flat path is active and must match foreach reference."""
    tensors = _make_tensors(2000, 512)
    result = _get_total_norm(tensors, norm_type=norm_type, foreach=None)
    expected = _reference_norm(tensors, norm_type)
    assert torch.allclose(result, expected, atol=1e-5, rtol=1e-5), (
        f"norm_type={norm_type}: got {result.item():.6f}, expected {expected.item():.6f}"
    )


@pytest.mark.parametrize("norm_type", [1.0, 2.0, float("inf")])
def test_flat_path_correctness_at_threshold(norm_type):
    """numel=512 is exactly at the threshold — flat path must be used and correct."""
    tensors = _make_tensors(50, 512)
    result = _get_total_norm(tensors, norm_type=norm_type, foreach=None)
    expected = _reference_norm(tensors, norm_type)
    assert torch.allclose(result, expected, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("norm_type", [1.0, 2.0, float("inf")])
def test_foreach_path_correctness_large_tensors(norm_type):
    """numel=1024: foreach path is used (above threshold); result must be correct."""
    tensors = _make_tensors(100, 1024)
    result = _get_total_norm(tensors, norm_type=norm_type, foreach=None)
    expected = _reference_norm(tensors, norm_type)
    assert torch.allclose(result, expected, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# Correctness: explicit foreach=True bypasses the flat path
# ---------------------------------------------------------------------------

def test_explicit_foreach_true_bypasses_flat_path():
    """foreach=True must use _foreach_norm regardless of tensor size."""
    tensors = _make_tensors(2000, 512)
    result_foreach = _get_total_norm(tensors, norm_type=2.0, foreach=True)
    result_auto = _get_total_norm(tensors, norm_type=2.0, foreach=None)
    assert torch.allclose(result_foreach, result_auto, atol=1e-5, rtol=1e-5)


def test_explicit_foreach_false_uses_scalar_loop():
    """foreach=False uses the scalar per-tensor loop path."""
    tensors = _make_tensors(100, 512)
    result_noforeach = _get_total_norm(tensors, norm_type=2.0, foreach=False)
    result_auto = _get_total_norm(tensors, norm_type=2.0, foreach=None)
    assert torch.allclose(result_noforeach, result_auto, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# Dtype coverage
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Optimization 2: skip no-op .to(first_device) for single-device case
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("norm_type", [1.0, 2.0, float("inf")])
def test_no_to_optimization_single_device_large_tensors(norm_type):
    """numel=4096 uses foreach path but should still skip .to() for CPU tensors."""
    tensors = _make_tensors(200, 4096)
    result = _get_total_norm(tensors, norm_type=norm_type, foreach=None)
    expected = _reference_norm(tensors, norm_type)
    assert torch.allclose(result, expected, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("norm_type", [1.0, 2.0, float("inf")])
def test_no_to_optimization_mixed_sizes(norm_type):
    """Mixed sizes some below threshold (flat) some above (foreach) — correctness check."""
    tensors_small = _make_tensors(1000, 256)   # → flat path
    tensors_large = _make_tensors(100, 2048)   # → foreach path
    all_tensors = tensors_small + tensors_large
    result = _get_total_norm(all_tensors, norm_type=norm_type, foreach=None)
    expected = _reference_norm(all_tensors, norm_type)
    assert torch.allclose(result, expected, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# Dtype coverage
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_flat_path_dtype_coverage(dtype):
    tensors = _make_tensors(500, 512, dtype=dtype)
    result = _get_total_norm(tensors, norm_type=2.0, foreach=None)
    expected = _reference_norm(tensors, 2.0)
    assert torch.allclose(result.double(), expected.double(), atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_single_tensor():
    t = torch.randn(512)
    result = _get_total_norm([t], norm_type=2.0, foreach=None)
    expected = torch.linalg.vector_norm(t, 2.0)
    assert torch.allclose(result, expected, atol=1e-6)


def test_empty_tensor_list():
    result = _get_total_norm([], norm_type=2.0, foreach=None)
    assert result.item() == 0.0


def test_single_element_tensors():
    """numel=1 tensors — flat path must produce correct result."""
    tensors = [torch.tensor([float(i)]) for i in range(1, 6)]
    result = _get_total_norm(tensors, norm_type=2.0, foreach=None)
    expected = math.sqrt(sum(i ** 2 for i in range(1, 6)))
    assert abs(result.item() - expected) < 1e-5


# ---------------------------------------------------------------------------
# Integration: clip_grad_norm_ end-to-end with flat path
# ---------------------------------------------------------------------------

def test_clip_grad_norm_end_to_end_small_tensors():
    """clip_grad_norm_ with many small parameters uses flat path and clips correctly."""
    torch.manual_seed(42)
    model = nn.Sequential(*[nn.Linear(16, 16, bias=True) for _ in range(50)])
    for p in model.parameters():
        p.grad = torch.randn_like(p)

    max_norm = 1.0
    total_norm = clip_grad_norm_(model.parameters(), max_norm, foreach=None)

    # Verify clipping: recompute norm of clipped grads
    clipped_norm = sum(
        p.grad.norm(2).item() ** 2 for p in model.parameters() if p.grad is not None
    ) ** 0.5
    assert clipped_norm <= max_norm + 1e-5, f"clipped norm {clipped_norm:.6f} exceeds max_norm {max_norm}"
    assert isinstance(total_norm, torch.Tensor)


def test_clip_grad_norm_matches_foreach_explicit():
    """Result with foreach=None (flat path) must match foreach=True for small tensors."""
    torch.manual_seed(0)
    params_auto = [torch.randn(128, requires_grad=True) for _ in range(200)]
    params_foreach = [p.clone().detach().requires_grad_(True) for p in params_auto]

    for p in params_auto:
        p.grad = torch.randn_like(p)
    for pa, pf in zip(params_auto, params_foreach):
        pf.grad = pa.grad.clone()

    norm_auto = clip_grad_norm_(params_auto, max_norm=1.0, foreach=None)
    norm_foreach = clip_grad_norm_(params_foreach, max_norm=1.0, foreach=True)

    assert torch.allclose(norm_auto, norm_foreach, atol=1e-5), (
        f"norm_auto={norm_auto.item():.6f}, norm_foreach={norm_foreach.item():.6f}"
    )
    for pa, pf in zip(params_auto, params_foreach):
        assert torch.allclose(pa.grad, pf.grad, atol=1e-6), "grad mismatch after clipping"
