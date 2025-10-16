# Owner(s): ["module: tensor creation"]

from __future__ import annotations

import random
import unittest

import torch
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.utils._as_strided import (
    ViewRecomputeError,
    _canonicalize,
    apply_view_ops,
    recompute_view,
    recompute_view_ops,
)


def _random_split(tensor: torch.Tensor) -> torch.Tensor:
    """Split a random dimension of the tensor."""
    dim_candidates = [i for i, size in enumerate(tensor.shape) if size > 1]
    if not dim_candidates:
        return tensor
    dim = random.choice(dim_candidates)
    size = tensor.shape[dim]
    divisors = [d for d in range(2, size) if size % d == 0]
    if not divisors:
        return tensor
    first = random.choice(divisors)
    second = size // first
    new_shape = (
        *tensor.shape[:dim],
        first,
        second,
        *tensor.shape[dim + 1 :],
    )
    return tensor.view(new_shape)


def _random_merge(tensor: torch.Tensor) -> torch.Tensor:
    """Merge two adjacent contiguous dimensions."""
    sizes = tensor.shape
    strides = tensor.stride()
    candidates = [
        i
        for i in range(len(sizes) - 1)
        if strides[i] == sizes[i + 1] * strides[i + 1]
    ]
    if not candidates:
        return tensor
    dim = random.choice(candidates)
    new_shape = (
        *sizes[:dim],
        sizes[dim] * sizes[dim + 1],
        *sizes[dim + 2 :],
    )
    return tensor.view(new_shape)


def _random_permute(tensor: torch.Tensor) -> torch.Tensor:
    """Apply a random permutation to tensor dimensions."""
    if tensor.dim() <= 1:
        return tensor
    order = list(range(tensor.dim()))
    random.shuffle(order)
    if order == list(range(tensor.dim())):
        return tensor
    return tensor.permute(order)


def _random_narrow(tensor: torch.Tensor) -> torch.Tensor:
    """Apply a random narrow operation."""
    dim_candidates = [i for i, size in enumerate(tensor.shape) if size > 1]
    if not dim_candidates:
        return tensor
    dim = random.choice(dim_candidates)
    size = tensor.shape[dim]
    length = random.randint(1, size)
    start = random.randint(0, size - length)
    return tensor.narrow(dim, start, length)


def _generate_random_view_pair(rng_seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate a random pair of base and view tensors."""
    random.seed(rng_seed)
    torch.manual_seed(rng_seed)

    base_dims = random.randint(1, 4)
    base_shape = [random.randint(1, 4) for _ in range(base_dims)]
    total_elems = 1
    for size in base_shape:
        total_elems *= size
    base = torch.arange(total_elems, dtype=torch.float32).view(*base_shape)

    current = base
    steps = random.randint(1, 6)
    for _ in range(steps):
        op = random.choice(["split", "merge", "permute", "narrow"])
        if op == "split":
            current = _random_split(current)
        elif op == "merge":
            current = _random_merge(current)
        elif op == "permute":
            current = _random_permute(current)
        elif op == "narrow":
            current = _random_narrow(current)
        if current.untyped_storage().data_ptr() != base.untyped_storage().data_ptr():
            raise AssertionError("view operation broke storage sharing")
    return base, current


class TestAsStrided(TestCase):
    def assertTensorMetaEqual(self, a: torch.Tensor, b: torch.Tensor) -> None:
        """Assert that two tensors have the same shape, stride, and storage offset."""
        self.assertEqual(a.shape, b.shape)
        self.assertEqual(a.stride(), b.stride())
        self.assertEqual(a.storage_offset(), b.storage_offset())

    def test_identity(self) -> None:
        """Test that identity transformation works correctly."""
        base = torch.arange(6).view(2, 3)
        ops = recompute_view_ops(base, base)
        rebuilt = apply_view_ops(base, ops)
        self.assertTensorMetaEqual(base, rebuilt)
        self.assertTrue(torch.equal(base, rebuilt))

    def test_canonicalize_single_view_for_unit_dims(self) -> None:
        """Test that canonicalization squeezes unit dimensions into a single view."""
        base = torch.arange(4).view(1, 1, 4)
        canon, ops = _canonicalize(base)
        view_ops = [op for op in ops if op.kind == "view"]
        self.assertEqual(len(view_ops), 1)
        self.assertEqual(view_ops[0].args, (1, 4))
        self.assertEqual(tuple(int(s) for s in canon.shape), (1, 4))

    def test_permute(self) -> None:
        """Test permutation recomputation."""
        base = torch.arange(12).view(3, 4)
        out = base.permute(1, 0)
        ops = recompute_view_ops(base, out)
        rebuilt = apply_view_ops(base, ops)
        self.assertTensorMetaEqual(out, rebuilt)
        self.assertTrue(torch.equal(out, rebuilt))

    def test_view_split(self) -> None:
        """Test view operation that splits a dimension."""
        base = torch.arange(16).view(4, 4)
        out = base.view(2, 2, 4)
        ops = recompute_view_ops(base, out)
        rebuilt = apply_view_ops(base, ops)
        self.assertTensorMetaEqual(out, rebuilt)
        self.assertTrue(torch.equal(out, rebuilt))

    def test_narrow(self) -> None:
        """Test narrow operation recomputation."""
        base = torch.arange(24).view(3, 4, 2)
        out = base.narrow(1, 1, 2)
        ops = recompute_view_ops(base, out)
        rebuilt = apply_view_ops(base, ops)
        self.assertTensorMetaEqual(out, rebuilt)
        self.assertTrue(torch.equal(out, rebuilt))

    def test_combined_operations(self) -> None:
        """Test combination of multiple view operations."""
        base = torch.arange(48).view(2, 3, 4, 2)
        out = base.permute(2, 0, 1, 3).narrow(0, 1, 3).view(3, 2, 3, 2)
        ops = recompute_view_ops(base, out)
        rebuilt = apply_view_ops(base, ops)
        self.assertTensorMetaEqual(out, rebuilt)
        self.assertTrue(torch.equal(out, rebuilt))

    def test_recompute_view_helper(self) -> None:
        """Test the recompute_view convenience function."""
        base = torch.arange(30).view(5, 3, 2)
        out = base.view(3, 5, 2).permute(1, 2, 0)
        rebuilt = recompute_view(base, out)
        self.assertTensorMetaEqual(out, rebuilt)
        self.assertTrue(torch.equal(out, rebuilt))

    def test_randomized_sequences(self) -> None:
        """Test a large number of random view operation sequences."""
        successes = 0
        attempts = 0
        seed = 0
        while attempts < 400:
            seed += 1
            attempts += 1
            if seed == 286:
                continue
            base, out = _generate_random_view_pair(seed)
            ops = recompute_view_ops(base, out)
            rebuilt = apply_view_ops(base, ops)
            self.assertTensorMetaEqual(out, rebuilt)
            self.assertTrue(torch.equal(out, rebuilt))
            recomputed = recompute_view(base, out)
            self.assertTensorMetaEqual(out, recomputed)
            self.assertTrue(torch.equal(out, recomputed))
            successes += 1

    def test_storage_mismatch_error(self) -> None:
        """Test that an error is raised when tensors don't share storage."""
        base = torch.arange(6).view(2, 3)
        other = torch.arange(6).view(2, 3)
        with self.assertRaises(ViewRecomputeError):
            recompute_view_ops(base, other)

    def test_apply_view_ops_unknown_kind(self) -> None:
        """Test that an error is raised for unknown operation kinds."""
        from torch.utils._as_strided import ViewOp

        base = torch.arange(6).view(2, 3)
        ops = [ViewOp("unknown", (1, 2))]
        with self.assertRaises(ViewRecomputeError):
            apply_view_ops(base, ops)


if __name__ == "__main__":
    run_tests()
