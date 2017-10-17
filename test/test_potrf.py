import torch
import numpy as np
from test_autograd import _make_cov
from torch.autograd import Variable
from common import TestCase, run_tests, skipIfNoLapack

from torch.autograd._functions.linalg import Potrf


class TestPotrf(TestCase):

    def _calc_deriv_numeric(self, A, L, upper):
        # numerical forward derivative
        dA = Variable(_make_cov(5))
        eps = 1e-6
        outb = Potrf.apply(A + (eps / 2) * dA, upper)
        outa = Potrf.apply(A - (eps / 2) * dA, upper)
        dL = (outb - outa) / eps

        return dA, dL

    def _calc_deriv_sym(self, A, L, upper):
        # reverse mode
        Lbar = Variable(torch.rand(5, 5).tril())
        if upper:
            Lbar = Lbar.t()
        L.backward(Lbar)
        Abar = A.grad

        return Abar, Lbar

    def _check_total_variation(self, A, L, upper):
        dA, dL = self._calc_deriv_numeric(A, L, upper)
        Abar, Lbar = self._calc_deriv_sym(A, L, upper)

        # compare df = Tr(dA^T Abar) = Tr(dL^T Lbar)
        df1 = (dL * Lbar).sum()
        df2 = (dA * Abar).sum()

        atol = 1e-5
        rtol = 1e-3
        assert (df1 - df2).abs().data[0] <= atol + rtol * df1.abs().data[0]

    @skipIfNoLapack
    def test_potrf(self):
        for upper in [True, False]:
            A = Variable(_make_cov(5), requires_grad=True)
            L = Potrf.apply(A, upper)
            self._check_total_variation(A, L, upper)

if __name__ == '__main__':
    run_tests()
