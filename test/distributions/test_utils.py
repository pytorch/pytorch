import pytest

import torch
from torch.distributions.utils import tril_matrix_to_vec, vec_to_tril_matrix


@pytest.mark.parametrize('shape', [
    (2, 2),
    (3, 3),
    (2, 4, 4),
    (2, 2, 4, 4),
])
def test_tril_matrix_to_vec(shape):
    mat = torch.randn(shape)
    n = mat.shape[-1]
    for diag in range(-n, n):
        actual = mat.tril(diag)
        vec = tril_matrix_to_vec(actual, diag)
        tril_mat = vec_to_tril_matrix(vec, diag)
        assert torch.allclose(tril_mat, actual)


if __name__ == '__main__':
    pytest.main([__file__])
