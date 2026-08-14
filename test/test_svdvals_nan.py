import torch
from torch.testing._internal.common_utils import TestCase, run_tests


class TestLinalgNaNHandling(TestCase):
    '''Tests for correct NaN propagation in linalg functions (gh#187759).'''

    def test_svdvals_nan_input(self):
        '''svdvals must propagate NaN, not swallow it.'''
        A = torch.tensor([[1.0, 2.0, 3.0],
                          [4.0, float('nan'), 6.0],
                          [7.0, 8.0, 9.0]])
        result = torch.linalg.svdvals(A)
        self.assertTrue(
            torch.isnan(result).any(),
            f'svdvals must propagate NaN, got {result}'
        )

    def test_svdvals_nan_consistency_with_svd(self):
        '''svdvals and svd must agree on NaN handling.'''
        A = torch.tensor([[1.0, 2.0],
                          [float('nan'), 4.0]])
        try:
            svdvals_result = torch.linalg.svdvals(A)
            svdvals_has_nan = torch.isnan(svdvals_result).any()
        except RuntimeError:
            svdvals_has_nan = None
        try:
            U, S, Vh = torch.linalg.svd(A)
            svd_has_nan = torch.isnan(S).any()
        except RuntimeError:
            svd_has_nan = None
        if svdvals_has_nan is not None and svd_has_nan is not None:
            self.assertEqual(
                svdvals_has_nan, svd_has_nan,
                'svdvals and svd must agree on NaN handling'
            )


if __name__ == '__main__':
    run_tests()
