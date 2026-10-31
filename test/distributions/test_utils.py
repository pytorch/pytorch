# Owner(s): ["module: distributions"]

import torch
from torch.distributions.utils import tril_matrix_to_vec, vec_to_tril_matrix
from torch.testing._internal.common_utils import (
    HardwareClassification,
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


class TestTrilMatrixToVec(TestCase):
    hw_classification = HardwareClassification.GENERIC

    @parametrize(
        "shape",
        [
            (2, 2),
            (3, 3),
            (2, 4, 4),
            (2, 2, 4, 4),
        ],
    )
    def test_tril_matrix_to_vec(self, shape):
        mat = torch.randn(shape)
        n = mat.shape[-1]
        for diag in range(-n, n):
            actual = mat.tril(diag)
            vec = tril_matrix_to_vec(actual, diag)
            tril_mat = vec_to_tril_matrix(vec, diag)
            if not torch.allclose(tril_mat, actual):
                raise AssertionError("Expected tril_mat and actual to be close")


instantiate_parametrized_tests(TestTrilMatrixToVec)


if __name__ == "__main__":
    run_tests()
