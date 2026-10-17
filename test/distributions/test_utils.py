# Owner(s): ["module: distributions"]

import pytest

import torch
from torch.distributions.utils import (
    _sum_rightmost,
    tril_matrix_to_vec,
    vec_to_tril_matrix,
)
from torch.testing._internal.common_utils import run_tests, TestCase


@pytest.mark.parametrize(
    "shape",
    [
        (2, 2),
        (3, 3),
        (2, 4, 4),
        (2, 2, 4, 4),
    ],
)
def test_tril_matrix_to_vec(shape):
    mat = torch.randn(shape)
    n = mat.shape[-1]
    for diag in range(-n, n):
        actual = mat.tril(diag)
        vec = tril_matrix_to_vec(actual, diag)
        tril_mat = vec_to_tril_matrix(vec, diag)
        if not torch.allclose(tril_mat, actual):
            raise AssertionError("Expected tril_mat and actual to be close")


class TestDistributionsUtils(TestCase):
    def test_sum_rightmost(self):
        # Empty tensor reduction
        empty = torch.zeros(0, 3)
        self.assertEqual(_sum_rightmost(empty, 0).shape, torch.Size([0, 3]))
        self.assertEqual(_sum_rightmost(empty, 1).shape, torch.Size([0]))

        # Standard non-empty tensor reduction equivalence
        t = torch.randn(2, 3, 4)
        self.assertEqual(_sum_rightmost(t, 0), t)
        self.assertEqual(_sum_rightmost(t, 1), t.sum(dim=-1))
        self.assertEqual(_sum_rightmost(t, 2), t.sum(dim=(-2, -1)))
        self.assertEqual(_sum_rightmost(t, 3), t.sum())

        # Reduction when dim > value.dim() or value is a scalar
        t_1d = torch.tensor([1.0, 2.0, 3.0])
        self.assertEqual(_sum_rightmost(t_1d, 2), torch.tensor(6.0))
        t_scalar = torch.tensor(5.0)
        self.assertEqual(_sum_rightmost(t_scalar, 1), torch.tensor(5.0))
        self.assertEqual(_sum_rightmost(empty, 2), torch.tensor(0.0))


if __name__ == "__main__":
    run_tests()
